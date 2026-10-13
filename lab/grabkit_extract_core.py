#!/usr/bin/env python3
"""Grab-kit step 1 GRAB: single-shard expert-core extractor (big-MoE scale).

Pulls ONE expert core out of a huge safetensors checkpoint WITHOUT loading the
model (or the shard): parse the shard's header JSON, seek to the target
expert's tensor byte ranges, write them to a compact core file.  Pure stdlib —
runs under any python3 (verified on the MI50 box system python, which has
neither torch nor safetensors installed).

Our core_gen pattern at scale (door25_wrap_expert.extract_donor_core /
graft_d2 SLIMMOE layout) is "one expert = its SwiGLU matrices, verbatim".
This tool is the same grab against HF per-shard expert keys for every donor
family in the flagship build:

  slimmoe  Phi-mini-MoE   model.layers.L.block_sparse_moe.experts.E.w{1,2,3}.weight
  olmoe    OLMoE          model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight
  k3       Kimi K3        language_model.model.layers.L.block_sparse_moe.experts.E.w{1,2,3}.weight_{packed,scale}
  glm      GLM-5.3        model.language_model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight_{packed,scale,shape,zero_point}
  mimo     MiMo-V2.x      model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight_{packed,scale,shape,zero_point}

Quantized donors keep packed/scale(/shape/zero_point) tuples verbatim: the core
bank stores what the shard holds; dequantization is a wrap-loader concern.

Output pair (one core):
  <out>.safetensors        the extracted tensors (written by a minimal stdlib
                           writer; readable by the safetensors library)
  <out>.provenance.json    lineage: source shard + offsets + per-tensor sha256
                           + dtypes/shapes + wall/RSS + license id

Usage:
  # list (layer, expert) pairs available in a model dir or one shard
  python3 lab/grabkit_extract_core.py --model-dir /data/tliao/Phi-mini-MoE-instruct --list-experts

  # pull the door25 code core (Phi L13e8) — opens ONE shard of four
  python3 lab/grabkit_extract_core.py --model-dir /data/tliao/Phi-mini-MoE-instruct \
      --layer 13 --expert 8 --out /data/tliao/core_bank/phi_mini_moe/L13E8/core

  # K3 fp8-packed expert out of one 17 GB shard (never touches the other 95)
  python3 lab/grabkit_extract_core.py --model-dir /home/tliao/Desktop/Kimi-K3 \
      --family k3 --layer 12 --expert 895 --out /tmp/k3_L12E895

  # independent-reader verification (needs the safetensors lib)
  python3 lab/grabkit_extract_core.py ... --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import struct
import sys
import time

# ---------------------------------------------------------------------------
# safetensors container: [u64 le header_len][header JSON][raw tensor bytes]
# each header entry: {dtype, shape, data_offsets: [begin, end)} relative to
# the start of the data section (= 8 + header_len).
# ---------------------------------------------------------------------------

DTYPES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}

# family -> (layer, expert) key regex; extra suffix groups after the expert id
# are matched greedily so packed/quant tuples land together.
FAMILY_PATTERNS = {
    "slimmoe": r"^model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\.(?P<expert>\d+)\.(?P<member>w1|w2|w3)\.weight$",
    "olmoe": r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<member>gate_proj|up_proj|down_proj)\.weight$",
    "k3": r"^language_model\.model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\.(?P<expert>\d+)\.(?P<member>w1|w2|w3)\.weight_(?P<quant>packed|scale)$",
    "glm": r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<member>gate_proj|up_proj|down_proj)\.weight_(?P<quant>packed|scale|shape|zero_point)$",
    "mimo": r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<member>gate_proj|up_proj|down_proj)\.weight_(?P<quant>packed|scale|shape|zero_point)$",
    # any expers.(id). member layout, flat or packed
    "generic": r"\.experts\.(?P<expert>\d+)\.(?P<member>[A-Za-z0-9_]+?)(?:\.weight(?:_(?P<quant>packed|scale|shape|zero_point))?)?$",
}

FAMILY_LAYER_GROUP = {
    "slimmoe": "layer", "olmoe": "layer", "k3": "layer", "glm": "layer",
    "mimo": "layer", "generic": "layer",
}

LICENSE_IDS = {
    "phi-mini-moe": "MIT (Microsoft)",
    "olmoe": "Apache-2.0 (AI2)",
    "kimi-k3": "Kimi-K3-License (Moonshot AI)",
    "glm-5.3": "MIT (Z.AI)",
    "mimo-v2": "MIT (XiaomiMiMo)",
}


def read_header(path: str):
    """Return (header_dict, header_len) for one safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 200_000_000:
            raise ValueError(f"{path}: implausible header length {n}")
        raw = f.read(n)
    hdr = json.loads(raw)
    hdr.pop("__metadata__", None)
    return hdr, n


def shard_files(model_dir: str) -> list[str]:
    return sorted(fn for fn in os.listdir(model_dir)
                  if fn.endswith(".safetensors") and not fn.startswith("._"))


def match_key(pattern: re.Pattern, key: str):
    m = pattern.match(key)
    if not m:
        return None
    d = m.groupdict()
    if "layer" not in d:
        # generic family: pull layer from the key path if it looks like layers.N
        lm = re.search(r"layers\.(\d+)\.", key)
        d["layer"] = lm.group(1) if lm else None
    return d


def scan_keys(hdr: dict, pattern: re.Pattern, layer, expert):
    """Keys in this header for (layer, expert). Returns (keys, meta dict)."""
    keys, meta = [], {}
    for k, v in hdr.items():
        d = match_key(pattern, k)
        if d is None:
            continue
        if layer is not None and str(d.get("layer")) != str(layer):
            continue
        if expert is not None and str(d.get("expert")) != str(expert):
            continue
        keys.append(k)
        meta[k] = d
    return sorted(keys), meta


def list_pairs(model_dir: str, pattern: re.Pattern,
               shards: list[str]) -> dict:
    """(layer, expert) -> n_keys, across the listed shards (header reads only)."""
    pairs: dict = {}
    for sh in shards:
        hdr, _ = read_header(os.path.join(model_dir, sh))
        for k in hdr:
            d = match_key(pattern, k)
            if d is None or d.get("layer") is None:
                continue
            pk = (int(d["layer"]), int(d["expert"]))
            pairs.setdefault(pk, {"keys": 0, "shards": set()})["keys"] += 1
            pairs[pk]["shards"].add(sh)
    out = {}
    for (l, e), v in sorted(pairs.items()):
        out[f"L{l}E{e}"] = {"n_keys": v["keys"], "shards": sorted(v["shards"])}
    return out


def write_safetensors(out_path: str, tensors: dict) -> None:
    """Minimal stdlib safetensors writer: name -> (dtype, shape, bytes)."""
    header, blob, off = {}, bytearray(), 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [off, off + len(data)],
        }
        blob += data
        off += len(data)
    hj = json.dumps(header, separators=(",", ":")).encode()
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(blob)
    os.replace(tmp, out_path)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def peak_rss_mb() -> float:
    # Linux: ru_maxrss is KiB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def verify_with_library(path: str, tensors: dict) -> dict:
    """Independent reader cross-check via the safetensors library if present.

    numpy cannot represent BF16/FP8, so pick torch for those dtypes and fall
    back to torch for anything else numpy rejects.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return {"status": "skipped", "reason": "safetensors library not installed"}
    need_torch = any(dt in ("BF16", "F16", "F8_E4M3", "F8_E5M2")
                     for dt, _, _ in tensors.values())
    frameworks = ["pt", "np"] if need_torch else ["np", "pt"]
    last_err = None
    for framework in frameworks:
        try:
            return _verify_once(path, tensors, safe_open, framework)
        except ImportError as e:
            last_err = e
        except Exception as e:  # numpy bf16 TypeError etc. -> try next backend
            last_err = e
            continue
    return {"status": "skipped",
            "reason": f"no usable tensor framework ({type(last_err).__name__}: {last_err})"}


def _verify_once(path: str, tensors: dict, safe_open, framework: str) -> dict:
    bad = []
    with safe_open(path, framework=framework) as f:
        got = set(f.keys())
        for name, (dtype, shape, data) in tensors.items():
            if name not in got:
                bad.append(f"missing {name}")
                continue
            t = f.get_tensor(name)
            if framework == "pt":
                import torch
                tb = t.contiguous().view(torch.uint8).numpy().tobytes()
            else:
                tb = t.tobytes()
            if tb != data:
                bad.append(f"bytes differ {name}")
    return {"status": "pass" if not bad else "FAIL",
            "framework": framework,
            "keys_checked": len(tensors), "mismatches": bad}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", help="checkpoint directory")
    ap.add_argument("--shard", help="one explicit .safetensors file (overrides index)")
    ap.add_argument("--family", default="auto",
                    choices=["auto"] + list(FAMILY_PATTERNS))
    ap.add_argument("--layer", type=int, help="expert core layer id")
    ap.add_argument("--expert", type=int, help="expert id inside the layer")
    ap.add_argument("--key-regex", help="custom regex overriding the family pattern "
                    "(must contain named groups expert and optionally layer)")
    ap.add_argument("--out", help="output path prefix (writes .safetensors + .provenance.json)")
    ap.add_argument("--list-experts", action="store_true",
                    help="list (layer, expert) pairs found (header scan only)")
    ap.add_argument("--license-id", default="",
                    help="provenance license id (default: guessed from --family)")
    ap.add_argument("--verify", action="store_true",
                    help="cross-check the written core with the safetensors library")
    args = ap.parse_args()

    if not args.model_dir and not args.shard:
        ap.error("--model-dir or --shard required")
    model_dir = args.model_dir or os.path.dirname(os.path.abspath(args.shard))

    if args.key_regex:
        pattern = re.compile(args.key_regex)
        family = "custom"
    else:
        family = args.family
        if family == "auto":
            # sniff from the first shard header that has expert keys
            shards_probe = ([args.shard] if args.shard
                            else shard_files(model_dir))
            sniff = None
            for sh in shards_probe:
                hdr, _ = read_header(sh if args.shard else os.path.join(model_dir, sh))
                for k in hdr:
                    if ".experts." in k:
                        sniff = k
                        break
                if sniff:
                    break
            if sniff is None:
                print("no expert keys found for auto family detection", file=sys.stderr)
                return 2
            family = next((f for f, p in FAMILY_PATTERNS.items()
                           if f != "generic" and re.match(p, sniff)), "generic")
            print(f"[auto] family={family} (from key {sniff})")
        pattern = re.compile(FAMILY_PATTERNS[family])

    # Header-scan only: never load a shard's tensors (or a 500k-key
    # index.json) to find where a target expert lives.  HF shards store whole
    # tensors, so one expert's packed tuple is normally inside ONE shard.
    shards = [args.shard] if args.shard else shard_files(model_dir)
    n_expected = {"slimmoe": 3, "olmoe": 3, "k3": 6, "glm": 12,
                  "mimo": 12}.get(family)

    if args.list_experts:
        pairs = list_pairs(model_dir, pattern, shards)
        print(json.dumps({"family": family, "shards_scanned": len(shards),
                          "n_pairs": len(pairs), "pairs": pairs}, indent=2))
        return 0

    if args.layer is None or args.expert is None or not args.out:
        ap.error("--layer, --expert and --out required for extraction")

    t0 = time.time()
    tensors: dict = {}
    src_shards = []
    for sh in shards:
        path = sh if args.shard else os.path.join(model_dir, sh)
        hdr, header_len = read_header(path)
        keys, meta = scan_keys(hdr, pattern, args.layer, args.expert)
        if not keys:
            continue
        src_shards.append({
            "file": os.path.abspath(path),
            "bytes": os.path.getsize(path),
            "header_len": header_len,
        })
        with open(path, "rb") as f:
            for k in keys:
                ent = hdr[k]
                begin, end = ent["data_offsets"]
                f.seek(8 + header_len + begin)
                data = f.read(end - begin)
                if len(data) != end - begin:
                    raise IOError(f"short read on {k} in {path}")
                tensors[k] = (ent["dtype"], tuple(ent["shape"]), data)
        if n_expected is not None and len(tensors) >= n_expected:
            break
    if not tensors:
        print(f"no keys matched layer={args.layer} expert={args.expert} "
              f"family={family}", file=sys.stderr)
        return 3

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_st = args.out + ".safetensors"
    write_safetensors(out_st, tensors)

    wall = time.time() - t0
    rss = peak_rss_mb()
    extracted_bytes = sum(len(d) for _, _, d in tensors.values())
    prov = {
        "format": "vesper-core/1",
        "family": family,
        "layer": args.layer,
        "expert": args.expert,
        "license": args.license_id or LICENSE_IDS.get(
            {"k3": "kimi-k3", "glm": "glm-5.3", "mimo": "mimo-v2",
             "olmoe": "olmoe", "slimmoe": "phi-mini-moe"}.get(family, ""), "unknown"),
        "source_shards": src_shards,
        "tensors": [
            {"name": k, "dtype": tensors[k][0], "shape": list(tensors[k][1]),
             "bytes": len(tensors[k][2]), "sha256": sha256(tensors[k][2])}
            for k in sorted(tensors)
        ],
        "extracted_bytes": extracted_bytes,
        "shard_bytes_touched": sum(s["bytes"] for s in src_shards),
        "bytes_read_out_of_shard": extracted_bytes,
        "wall_s": round(wall, 3),
        "peak_rss_mb": round(rss, 1),
        "output": os.path.abspath(out_st),
    }
    if args.verify:
        prov["verify"] = verify_with_library(out_st, tensors)
    with open(args.out + ".provenance.json", "w") as f:
        json.dump(prov, f, indent=2)

    print(json.dumps({
        "ok": True, "family": family, "layer": args.layer, "expert": args.expert,
        "n_tensors": len(tensors), "extracted_bytes": extracted_bytes,
        "shard_bytes": prov["shard_bytes_touched"],
        "read_fraction": round(extracted_bytes / max(1, prov["shard_bytes_touched"]), 8),
        "wall_s": prov["wall_s"], "peak_rss_mb": prov["peak_rss_mb"],
        "out": out_st,
        "verify": prov.get("verify", {}).get("status", "not-run"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
