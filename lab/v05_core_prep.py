#!/usr/bin/env python3
"""v0.5 core prep: dequantize grab-kit core.safetensors to door25 wrap cores.

The grab-kit extractor stores donor experts VERBATIM (runbook §1: packed /
scale / shape / zero_point kept as the shard holds them).  The wrap loader
needs plain float SwiGLU matrices, so this tool dequantizes one core into
`{out}.pt` with the door25 layout:

    w_gu   (2I, H)  = [w1; w3]   (gate; up)
    w_down (H, I)   = w2

plus the donor's activation so the frozen core stays donor-verbatim:

    swiglu  silu(gate) * up                       (GLM-5.3-Flash, MiMo-Flash)
    situ    beta*tanh(gate/beta)*sigmoid(gate) *   (Kimi K3, beta 4 / 25)
            linear_beta*tanh(up/linear_beta)

Formats handled (verified against compressed-tensors 0.19.0 on the real
cores — see --check-lib):

    glm/mimo  compressed-tensors "pack-quantized" int4, asymmetric, group 32
              weight_packed I32 (out, in/8)  dense nibbles, signed = nibble-8
              weight_zero_point I32 (out/8, groups) packed along dim 0
              w = (q - zp) * scale
    k3        compressed-tensors "mxfp4-pack-quantized": FP4 e2m1 sign-magnitude
              2 values/byte (low nibble = even index), scale E8M0 = 2^(b-127)
              w = fp4 * scale  (group 32 along in)

Pure torch + safetensors — runs in the box venv (no compressed_tensors there).

Usage:
  python3 lab/v05_core_prep.py --core /data/tliao/core_bank/glm-5.3-flash/L3E0/core \
      --out /data/tliao/v05_out/glm_cybersec_core.pt
  python3 lab/v05_core_prep.py --core ... --out ... --check-lib   # oracle check
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file

E2M1_TO_FLOAT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

MEMBER_TO_W = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2",
               "w1": "w1", "w2": "w2", "w3": "w3"}


def load_core_tensors(core_prefix: str) -> tuple[dict, dict]:
    st = load_file(core_prefix + ".safetensors")
    prov_path = core_prefix + ".provenance.json"
    prov = {}
    if os.path.exists(prov_path):
        with open(prov_path) as f:
            prov = json.load(f)
    return st, prov


def family_of(st: dict, prov: dict) -> str:
    fam = prov.get("family")
    if fam:
        return fam
    for k in st:
        if "language_model.model.layers" in k:
            return "k3"
        if "model.language_model" in k:
            return "glm"
        if "mlp.experts" in k and k.startswith("model.layers"):
            return "mimo"
    raise ValueError("cannot infer family from tensor names")


def _key(st: dict, suffix: str) -> str:
    hits = [k for k in st if k.endswith(suffix)]
    if len(hits) != 1:
        raise KeyError(f"expected exactly 1 key ending {suffix!r}, got {hits}")
    return hits[0]


def unpack_int4_signed(packed: torch.Tensor, shape, packed_dim: int) -> torch.Tensor:
    """compressed_tensors.unpack_from_int32(num_bits=4): dense nibbles, signed.

    Returns int8 in [-8, 7].  packed_dim=1 -> columns hold in_features/8
    words of 8 values; packed_dim=0 -> rows hold out_features/8 words.
    """
    if packed_dim == 1:
        rows, cols = shape
        w = packed.to(torch.int64)
        if w.shape != (rows, (cols + 7) // 8):
            raise ValueError(f"packed {tuple(packed.shape)} vs shape {shape}")
        out = torch.empty(rows, cols, dtype=torch.int8)
        for k in range(8):
            out[:, k::8] = ((w >> (4 * k)) & 0xF).to(torch.int8) - 8
        return out
    if packed_dim == 0:
        rows, cols = shape
        w = packed.to(torch.int64)
        if w.shape != ((rows + 7) // 8, cols):
            raise ValueError(f"packed {tuple(packed.shape)} vs shape {shape}")
        out = torch.empty(rows, cols, dtype=torch.int8)
        for k in range(8):
            out[k::8, :] = ((w >> (4 * k)) & 0xF).to(torch.int8) - 8
        return out
    raise ValueError("packed_dim must be 0 or 1")


def unpack_fp4_mxfp4(packed: torch.Tensor, shape) -> torch.Tensor:
    """compressed_tensors.unpack_fp4_from_uint8: low nibble = even index.

    e2m1 sign-magnitude lookup, returns float32 of shape `shape`.
    """
    rows, cols = shape
    if packed.shape != (rows, cols // 2):
        raise ValueError(f"packed {tuple(packed.shape)} vs shape {shape}")
    b = packed.reshape(-1).to(torch.int64)
    low = b & 0xF
    high = (b >> 4) & 0xF
    codes = torch.empty(b.numel() * 2, dtype=torch.int64)
    codes[0::2] = low
    codes[1::2] = high
    sign = (codes & 0x8) != 0
    mag = codes & 0x7
    lut = torch.tensor(E2M1_TO_FLOAT, dtype=torch.float32)
    vals = lut[mag] * torch.where(sign, torch.tensor(-1.0), torch.tensor(1.0))
    return vals.reshape(rows, cols)


def dequant_int4_core(st: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """GLM-5.3-Flash / MiMo-V2.6-Flash: pack-quantized int4, group 32, asym."""
    ws = {}
    for member, wname in MEMBER_TO_W.items():
        pfx = f".{member}.weight"
        try:
            k_packed = _key(st, pfx + "_packed")
        except KeyError:
            continue
        root = k_packed[: -len("weight_packed")]
        packed = st[root + "weight_packed"]
        scale = st[root + "weight_scale"].float()
        zp_p = st[root + "weight_zero_point"]
        shape = st[root + "weight_shape"].tolist()
        out_f, in_f = int(shape[0]), int(shape[1])
        g = in_f // scale.shape[-1]
        if scale.shape[0] != out_f or in_f % scale.shape[-1] != 0 or g < 1:
            raise ValueError(f"{root} scale shape {tuple(scale.shape)} "
                             f"vs weight ({out_f},{in_f})")
        q = unpack_int4_signed(packed, (out_f, in_f), packed_dim=1)
        zp = unpack_int4_signed(zp_p, (out_f, in_f // g), packed_dim=0)
        w = (q.float() - zp.float().repeat_interleave(g, dim=1)) * \
            scale.repeat_interleave(g, dim=1)
        ws[wname] = w
        del q, zp
    if "w1" not in ws or "w2" not in ws or "w3" not in ws:
        raise ValueError(f"missing w1/w2/w3 among {sorted(ws)}")
    w_gu = torch.cat([ws["w1"], ws["w3"]], dim=0)
    w_down = ws["w2"]
    meta = {"act": "swiglu", "d_donor": w_gu.shape[1],
            "d_inter": w_down.shape[1], "group": g, "format": "pack-quantized"}
    return w_gu, w_down, meta


def dequant_mxfp4_core(st: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Kimi K3: mxfp4-pack-quantized (FP4 e2m1 + E8M0 scale, group 32)."""
    ws = {}
    for member, wname in (("w1", "w1"), ("w2", "w2"), ("w3", "w3")):
        try:
            k_packed = _key(st, f".{member}.weight_packed")
        except KeyError:
            continue
        root = k_packed[: -len("weight_packed")]
        packed = st[root + "weight_packed"]
        scale_u8 = st[root + "weight_scale"]
        out_f = packed.shape[0]
        in_f = packed.shape[1] * 2
        g = in_f // scale_u8.shape[-1]
        if scale_u8.shape[0] != out_f or in_f % scale_u8.shape[-1] != 0 or g < 1:
            raise ValueError(f"{root} scale shape {tuple(scale_u8.shape)} "
                             f"vs weight ({out_f},{in_f})")
        w = unpack_fp4_mxfp4(packed, (out_f, in_f))
        scale = (2.0 ** (scale_u8.to(torch.int64) - 127)).float()
        w = w * scale.repeat_interleave(g, dim=1)
        ws[wname] = w
    if "w1" not in ws or "w2" not in ws or "w3" not in ws:
        raise ValueError(f"missing w1/w2/w3 among {sorted(ws)}")
    w_gu = torch.cat([ws["w1"], ws["w3"]], dim=0)
    w_down = ws["w2"]
    meta = {"act": "situ", "situ_beta": 4.0, "situ_linear_beta": 25.0,
            "d_donor": w_gu.shape[1], "d_inter": w_down.shape[1],
            "group": g, "format": "mxfp4-pack-quantized"}
    return w_gu, w_down, meta


def check_lib(core_prefix: str, w_gu: torch.Tensor, w_down: torch.Tensor,
              family: str) -> dict:
    """Compare against compressed-tensors decompress (validation oracle)."""
    try:
        from compressed_tensors.compressors.pack_quantized.helpers import (
            unpack_from_int32)
        from compressed_tensors.compressors.nvfp4.helpers import (
            unpack_fp4_from_uint8)
        from compressed_tensors.compressors.mx_utils import decompress_mx_scale
        from compressed_tensors.quantization import QuantizationArgs
        from compressed_tensors.quantization.lifecycle.forward import dequantize
    except ImportError:
        return {"status": "skipped", "reason": "compressed_tensors not installed"}
    st, _ = load_core_tensors(core_prefix)
    args = QuantizationArgs(num_bits=4, type="int", symmetric=False,
                            strategy="group", group_size=32)
    if family == "k3":
        args = QuantizationArgs(num_bits=4, type="float", symmetric=True,
                                strategy="group", group_size=32)
    max_abs = 0.0
    for member in ("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"):
        pfx = f".{member}.weight_packed"
        try:
            k = _key(st, pfx)
        except KeyError:
            continue
        root = k[: -len("weight_packed")]
        packed = st[root + "weight_packed"]
        scale = st[root + "weight_scale"]
        if family == "k3":
            m, n = packed.shape
            lib_w = unpack_fp4_from_uint8(packed, m, n * 2).float()
            sc = decompress_mx_scale(scale).float()
            lib_w = dequantize(x_q=lib_w, scale=sc, args=args)
        else:
            shape = tuple(int(x) for x in st[root + "weight_shape"].tolist())
            zp = unpack_from_int32(st[root + "weight_zero_point"], 4,
                                   (shape[0], scale.shape[-1]), packed_dim=0)
            q = unpack_from_int32(packed, 4, shape)
            lib_w = dequantize(x_q=q.float(), scale=scale.float(),
                               zero_point=zp.float(), args=args)
        mine = {"gate_proj": w_gu[: w_down.shape[1]],
                "up_proj": w_gu[w_down.shape[1]:],
                "down_proj": w_down,
                "w1": w_gu[: w_down.shape[1]],
                "w3": w_gu[w_down.shape[1]:],
                "w2": w_down}[member]
        max_abs = max(max_abs, float((lib_w - mine).abs().max()))
    return {"status": "pass" if max_abs < 1e-5 else "FAIL",
            "max_abs_diff": max_abs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--core", required=True,
                    help="core path prefix (without .safetensors)")
    ap.add_argument("--out", required=True, help="output .pt (door25 core)")
    ap.add_argument("--family", default="", help="k3|glm|mimo (auto if empty)")
    ap.add_argument("--check-lib", action="store_true",
                    help="verify vs compressed-tensors if installed")
    args = ap.parse_args()

    st, prov = load_core_tensors(args.core)
    fam = args.family or family_of(st, prov)
    print(f"[prep] family={fam} tensors={len(st)}", flush=True)
    if fam == "k3":
        w_gu, w_down, meta = dequant_mxfp4_core(st)
    elif fam in ("glm", "mimo", "glm-5.3", "mimo-v2"):
        w_gu, w_down, meta = dequant_int4_core(st)
    else:
        raise SystemExit(f"unsupported family {fam!r} (flat bf16 cores do not "
                         f"need this tool — use door25.load_core)")

    report = {"w_gu": [int(x) for x in w_gu.shape],
              "w_down": [int(x) for x in w_down.shape],
              "act": meta["act"],
              "w_gu_std": round(float(w_gu.std()), 5),
              "w_down_std": round(float(w_down.std()), 5),
              "w_gu_mean": round(float(w_gu.mean()), 5),
              "finite": bool(torch.isfinite(w_gu).all()
                             and torch.isfinite(w_down).all())}
    print("[prep]", json.dumps(report), flush=True)
    if not report["finite"]:
        raise SystemExit("dequant produced non-finite weights")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save({"w_gu": w_gu.to(torch.float32),
                "w_down": w_down.to(torch.float32),
                "layer": prov.get("layer"), "expert": prov.get("expert"),
                "family": fam, "meta": meta,
                "provenance": prov.get("format", "vesper-core/1")}, args.out)
    print(f"[prep] wrote {args.out}", flush=True)

    if args.check_lib:
        print("[check]", json.dumps(check_lib(args.core, w_gu, w_down, fam)),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
