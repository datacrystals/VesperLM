#!/usr/bin/env python3
"""BANK LADDER RUNG 1 — ~250 frozen donor cores ≈ 8B bank on vk-base (MODULAR_MOE §11).

Phases (all skip-done / incremental JSON / session-proof):
  manifest   deterministic breadth-first core selection -> manifest.json
  extract    grabkit_extract_core.py per core (CPU); --donor k3|glm|mimo
  fit        one wrap per core on the vk-base host (§10.7 recipe) — or
             --lane i --lanes N to loop over a shard of the manifest with ONE
             model load; writes slim plugins (NO core buffers — the packed core
             stays in core_bank and is dequantized at load; /data is tight)
  growth     assemble first --n plugs and measure the §11 break criteria
             (home quality, pairwise harm, p_home, base tax) -> growth_<n>.json
  impress    endpoint battery, NUMERIC/ONE-WORD VERDICTS ONLY (provider
             safety filter) + rung-1 metrics (total params, actives/token at
             k=4-8, tok/s)

Selection heuristic (recorded in the manifest): breadth-first — layer
round-robin over each donor's MoE layers (every layer gets a core before any
layer gets a second), expert ids spread with a multiplicative hash.  This rung
tests "does the bank scale", NOT optimal picks (RouteStats profiling is rung 2).

Ops: claim-locked GPUs only (mr_gpu_claims, 10-min TTL), ROCBLAS_TENSILE_LIBPATH,
no pgrep gates.  See HANDOFF "BANK LADDER RUNG 1".
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))

import vk_graft as vk  # noqa: E402
import graft_d0_parity as d0  # noqa: E402
import graft_d1_plugin as d1  # noqa: E402
import door25_wrap_expert as d25  # noqa: E402
import door6_expert_mix as d6  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402
import v05_core_prep as prep  # noqa: E402
import vk_final_build as vb  # noqa: E402

FORMAT = "vesper-bank-rung1-v1"
DEFAULT_MODEL = "/data/tliao/vk_final/chat_model"
DEFAULT_REGISTRY = "/data/tliao/vk_final/registry_vk.pt"
DEFAULT_ROOT = "/data/tliao/rung1"
MIN_FEED_TOKENS = 200_000
BASE_TAX_BAR = 0.05
HARM_BAR = 0.02

# home -> feed spec.  text homes tokenize with the chat tokenizer; ids homes
# are pre-tokenized uint16 streams (vesperk bins, vocab-matched 65523).
HOMES = ["hacking", "cybersec", "writing", "code", "math", "science", "web",
         "chat"]
HOME_FEED = {
    "hacking": ("text", "Dataset/data/specialties/hacking/corpus.txt"),
    "cybersec": ("text", "Dataset/data/specialties/cybersec/corpus.txt"),
    "writing": ("text", "Dataset/data/specialties/writing/corpus.txt"),
    "code": ("ids", "/data/tliao/rung1_feeds/code.ids"),
    "math": ("ids", "/data/tliao/rung1_feeds/math.ids"),
    "science": ("ids", "/data/tliao/rung1_feeds/science.ids"),
    "web": ("ids", "/data/tliao/rung1_feeds/web.ids"),
    "chat": ("ids", "/data/tliao/rung1_feeds/chat.ids"),
}
BASE_FEED = ("ids", "/data/tliao/rung1_feeds/base.ids")

# donor geometry: family, MoE layers, experts/layer, act, params note
DONORS = {
    "kimi-k3": {"family": "k3", "layers": 92, "experts": 896,
                "model_dir_laptop": "/home/tliao/Desktop/Kimi-K3",
                "model_dir_box": "/data/ModelDownloader/Kimi-K3"},
    "glm-5.3-flash": {"family": "glm", "layers": 45, "experts": 288,
                      "model_dir_box": "/data/ModelDownloader/GLM-5.3-Flash-AWQ-INT4"},
    "mimo-v2.6-flash": {"family": "mimo", "layers": 48, "experts": 256,
                        "model_dir_box": "/data/ModelDownloader/MiMo-V2.6-Flash-INT4"},
}
# breadth-first quota per donor (sum 250 ≈ 8B frozen: K3 ~33M, GLM/MiMo ~25M)
QUOTA = {"kimi-k3": 175, "glm-5.3-flash": 40, "mimo-v2.6-flash": 35}

GRABKIT = os.path.join(REPO_ROOT, "lab", "grabkit_extract_core.py")


def dump(out: dict, path: str) -> None:
    g0.dump_json(out, path)


# growth_<n>.json write contract: `growth_done` is stamped ONLY when every
# measure section below landed.  A crash must leave `growth_error` (or the
# incremental pre-measure header) — never a bare header that skip-done could
# mistake for a finished checkpoint (growth-64 stub bug, 2026-10-15).
GROWTH_REQUIRED = ("n_loaded", "home_quality", "base_tax", "pairwise_harm",
                   "p_home", "break_status")


def growth_is_done(path: str) -> bool:
    """True only for a FINISHED measure (phase growth_done + all sections)."""
    if not os.path.exists(path):
        return False
    try:
        j = json.load(open(path))
    except Exception:  # noqa: BLE001
        return False
    return (j.get("phase") == "growth_done"
            and all(k in j for k in GROWTH_REQUIRED))


def write_growth_result(out: dict, path: str, final: bool = False) -> None:
    """Sole write path for growth_<n>.json (atomic tmp+replace via dump).

    final=True stamps phase=growth_done and refuses to write unless every
    measure section is present, so `growth_done` always means full schema.
    """
    if final:
        missing = [k for k in GROWTH_REQUIRED if k not in out]
        if missing:
            raise ValueError(f"growth finalize missing sections: {missing}")
        out["phase"] = "growth_done"
    dump(out, path)


def manifest_path(root: str) -> str:
    return os.path.join(root, "manifest.json")


# ---------------------------------------------------------------------------
# phase: manifest — deterministic breadth-first selection
# ---------------------------------------------------------------------------

def core_id(donor: str, layer: int, expert: int) -> str:
    short = {"kimi-k3": "k3", "glm-5.3-flash": "glm",
             "mimo-v2.6-flash": "mimo"}[donor]
    return f"{short}_L{layer}E{expert}"


def build_manifest(root: str) -> dict:
    """Breadth-first: layer round-robin, expert = (i * GOLDEN) % E.

    Deterministic and recorded — rung 1 tests bank SCALE, not pick quality.
    Home round-robins over HOMES (recorded); host layer depth-matches the
    donor layer onto the 10-layer vk spine (round(L/Lmax*9)).
    """
    GOLD = 2654435761  # Knuth multiplicative hash — spreads expert ids
    cores = []
    for donor, g in DONORS.items():
        n = QUOTA[donor]
        L, E = g["layers"], g["experts"]
        for i in range(n):
            layer = i % L                       # breadth-first over depth
            expert = ((i * GOLD) >> 8) % E      # spread within expert space
            home = HOMES[len(cores) % len(HOMES)]
            host_layer = round(layer / max(1, L - 1) * 9)
            cores.append({
                "id": core_id(donor, layer, expert),
                "donor": donor, "family": g["family"],
                "layer": layer, "expert": expert,
                "home": home, "host_layer": host_layer,
                "core_prefix": f"/data/tliao/core_bank/{donor}/L{layer}E{expert}/core",
            })
    man = {
        "format": "vesper-rung1-manifest/1",
        "heuristic": "breadth-first: layer=i%E_layers round-robin, "
                     "expert=((i*2654435761)>>8)%E, home=round-robin HOMES, "
                     "host_layer=round(L/(Lmax-1)*9); no RouteStats this rung",
        "quota": QUOTA, "homes": HOMES,
        "n_cores": len(cores), "cores": cores,
    }
    dump(man, manifest_path(root))
    print(f"[manifest] {len(cores)} cores -> {manifest_path(root)}")
    return man


def load_manifest(root: str) -> dict:
    p = manifest_path(root)
    if not os.path.exists(p):
        return build_manifest(root)
    return json.load(open(p))


# ---------------------------------------------------------------------------
# phase: extract — grabkit per core, skip-done
# ---------------------------------------------------------------------------

def phase_extract(args) -> dict:
    man = load_manifest(args.root)
    out = {"phase": "extract", "format": FORMAT, "done": {}, "fail": {}}
    out_p = os.path.join(args.root, "results", "extract.json")
    donor_filter = [d for d in DONORS if args.donor in ("all", d)]
    local = args.local_dir  # where core_bank lives on THIS host
    for c in man["cores"]:
        if c["donor"] not in donor_filter:
            continue
        pref = c["core_prefix"] if local == "/data/tliao" else \
            c["core_prefix"].replace("/data/tliao", local, 1)
        if os.path.exists(pref + ".safetensors"):
            out["done"][c["id"]] = pref
            continue
        g = DONORS[c["donor"]]
        model_dir = args.model_dir
        if not model_dir:
            for cand in (g.get("model_dir_box"), g.get("model_dir_laptop"),
                         g.get("model_dir_laptop_boxhost")):
                if cand and os.path.isdir(cand):
                    model_dir = cand
                    break
        if not model_dir or not os.path.isdir(model_dir):
            out["fail"][c["id"]] = f"no model dir ({model_dir})"
            continue
        os.makedirs(os.path.dirname(pref), exist_ok=True)
        cmd = [sys.executable, GRABKIT, "--model-dir", model_dir,
               "--family", g["family"], "--layer", str(c["layer"]),
               "--expert", str(c["expert"]), "--out", pref]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and os.path.exists(pref + ".safetensors"):
            out["done"][c["id"]] = pref
            print(f"[extract] {c['id']} ok {time.time()-t0:.1f}s", flush=True)
        else:
            err = (r.stderr or r.stdout)[-200:]
            out["fail"][c["id"]] = err
            print(f"[extract] {c['id']} FAIL {err}", flush=True)
        dump(out, out_p)
    out["phase"] = "extract_done"
    dump(out, out_p)
    print(f"[extract] done={len(out['done'])} fail={len(out['fail'])}")
    return out


# ---------------------------------------------------------------------------
# plugin I/O — SLIM (no core buffers; core rehydrated from core_bank)
# ---------------------------------------------------------------------------

def save_plugin_slim(path: str, wrap, row: torch.Tensor, meta: dict) -> None:
    sd = {k: v for k, v in wrap.state_dict().items()
          if not k.startswith("core_")}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"format": FORMAT, "wrap_adapters": sd,
                "wrap_spec": wrap.wrap_spec(), "row": row.cpu(),
                "meta": meta}, path)


def load_plugin_slim(path: str, device: str):
    """-> (wrap, row, meta) with core rehydrated from core_bank (fp16)."""
    pl = torch.load(path, map_location="cpu", weights_only=False)
    spec = pl["wrap_spec"]
    st, prov = prep.load_core_tensors(pl["meta"]["core_prefix"])
    fam = prep.family_of(st, prov)
    if fam == "k3":
        w_gu, w_down, cmeta = prep.dequant_mxfp4_core(st)
    else:
        w_gu, w_down, cmeta = prep.dequant_int4_core(st)
    wrap = d25.WrappedForeignExpert(
        spec["d_model"], spec["d_canon"], w_gu, w_down, seed=0,
        act=spec.get("act", cmeta.get("act", "swiglu")),
        situ_beta=spec.get("situ_beta", cmeta.get("situ_beta", 1.0)),
        situ_linear_beta=spec.get("situ_linear_beta",
                                  cmeta.get("situ_linear_beta")))
    wrap.load_state_dict(pl["wrap_adapters"], strict=False)
    # endpoint assembly keeps all cores resident: fp16 halves VRAM (250×33M
    # fp32 = 33GB does not fit an MI50; fp16 = 16.5GB does)
    if device != "cpu":
        wrap.core_w_gu.data = wrap.core_w_gu.data.half()
        wrap.core_w_gu.to(device)
        wrap.core_w_down.data = wrap.core_w_down.data.half()
    wrap.to(device)
    return wrap, pl["row"].float().clone(), pl.get("meta", {})


def plugin_path(root: str, cid: str) -> str:
    return os.path.join(root, "plugins", f"{cid}_plugin.pt")


# ---------------------------------------------------------------------------
# feeds
# ---------------------------------------------------------------------------

def feed_batches(tok, home: str, seq: int, bs: int, max_tokens: int):
    kind, rel = HOME_FEED[home]
    path = rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)
    if not os.path.exists(path):
        path = rel.replace("/data/tliao", "/home/tliao", 1) \
            if os.path.exists(rel.replace("/data/tliao", "/home/tliao", 1)) \
            else path
    if kind == "text":
        ids = tok.encode(open(path).read(), add_special_tokens=False)[:max_tokens]
    else:
        ids = np.fromfile(path, dtype=np.uint16).astype(np.int64)[:max_tokens]
        ids = ids.tolist()
    return vk.to_batches(ids, 1, seq, bs)


def base_batches(tok, seq: int, bs: int, max_tokens: int):
    kind, rel = BASE_FEED
    ids = np.fromfile(rel, dtype=np.uint16).astype(np.int64)[:max_tokens]
    return vk.to_batches(ids.tolist(), 0, seq, bs)


# ---------------------------------------------------------------------------
# phase: fit — one core, §10.7 recipe (rank-64 width-norm, wd 0, lr 3e-3,
# task-CE + out_pen 300, FEED >= 200k) + query-space registry row
# ---------------------------------------------------------------------------

def fit_one(c: dict, model, tok, blocks, banks_reg, args, out: dict) -> dict:
    cid = c["id"]
    pp = plugin_path(args.root, cid)
    if args.skip_done and os.path.exists(pp):
        try:
            wrap, row, meta = load_plugin_slim(pp, "cpu")
            if bool(torch.isfinite(row).all()) and all(
                    bool(torch.isfinite(p).all()) for p in wrap.parameters()):
                out.setdefault("fitted", {})[cid] = {"skipped": True}
                print(f"[fit] {cid} plugin exists+finite — skip", flush=True)
                return {"skipped": True}
        except Exception as e:  # noqa: BLE001
            print(f"[fit] {cid} reload failed ({e}) — refit", flush=True)
    if not os.path.exists(c["core_prefix"] + ".safetensors"):
        out.setdefault("fail", {})[cid] = "core missing"
        dump(out, args.out_json)
        return {"error": "core missing"}

    t0 = time.time()
    st, prov = prep.load_core_tensors(c["core_prefix"])
    fam = prep.family_of(st, prov)
    if fam == "k3":
        w_gu, w_down, cmeta = prep.dequant_mxfp4_core(st)
    else:
        w_gu, w_down, cmeta = prep.dequant_int4_core(st)
    host_i = int(c["host_layer"])
    wrap = d25.WrappedForeignExpert(
        banks_reg[host_i].d_model, args.d_canon, w_gu, w_down, seed=args.seed,
        act=cmeta.get("act", "swiglu"),
        situ_beta=cmeta.get("situ_beta", 1.0),
        situ_linear_beta=cmeta.get("situ_linear_beta"))

    seq, bs = args.seq_len, args.batch_size
    if seq * bs not in args.safe_bt:
        raise RuntimeError(f"train B*T {seq*bs} not in safe set {args.safe_bt}")
    steps_eff = max(args.expert_steps,
                    -(-MIN_FEED_TOKENS // max(1, seq * bs)))
    home_train = feed_batches(tok, c["home"], seq, bs, args.max_tokens)
    base_b = base_batches(tok, seq, bs, args.max_base_tokens)
    if not home_train:
        out.setdefault("fail", {})[cid] = "empty home stream"
        dump(out, args.out_json)
        return {"error": "empty home"}
    fed = steps_eff * seq * bs
    print(f"== [{cid}] {c['donor']} L{c['layer']}e{c['expert']} home={c['home']} "
          f"host L{host_i} ({len(home_train)} batches, {steps_eff} steps, "
          f"feed {fed} tok-slots) ==", flush=True)
    dump(out, args.out_json)

    pa = vb._fit_wrap_vk(model, blocks, banks_reg, host_i, wrap, home_train,
                         base_b, args, steps_eff)
    if not pa.get("finite", True):
        print(f"[{cid}] NON-FINITE at lr {args.expert_lr} — refit 1e-3",
              flush=True)
        del wrap
        torch.cuda.empty_cache()
        wrap = d25.WrappedForeignExpert(
            banks_reg[host_i].d_model, args.d_canon, w_gu, w_down,
            seed=args.seed, act=cmeta.get("act", "swiglu"),
            situ_beta=cmeta.get("situ_beta", 1.0),
            situ_linear_beta=cmeta.get("situ_linear_beta"))
        old = args.expert_lr
        args.expert_lr = 1e-3
        pa = vb._fit_wrap_vk(model, blocks, banks_reg, host_i, wrap, home_train,
                             base_b, args, steps_eff)
        args.expert_lr = old
        if not pa.get("finite", True):
            out.setdefault("fail", {})[cid] = "non_finite_twice"
            out["phase"] = f"failed_{cid}"
            dump(out, args.out_json)
            return {"error": "non_finite"}

    # registry row — QUERY space (d_r): the row is W_q·h, not h (see 6c89e7a)
    site_rows = {host_i: [torch.zeros(banks_reg[host_i].d_r)]}
    gates, site_banks = vb._build_vk_gates(blocks, banks_reg, site_rows)
    ffn = blocks[host_i]
    handles = g0.install_mix(blocks, vk.FAM, {
        host_i: [{"expert": wrap,
                  "lam": nn.Parameter(torch.zeros((), device=args.device)),
                  "tag": cid, "num_stock": banks_reg[host_i].num_experts}]},
        mode="off")
    pair_home = home_train[:args.pair_batches]
    pair_base = base_b[:args.pair_base]
    pairs = g0.run_in_state(
        model, blocks, vk.FAM, gates, handles, "off",
        lambda: g0.collect_pairs_fam(
            model, blocks, vk.FAM, pair_home + pair_base, args.device,
            args.max_cal_pairs, lambda _i, b: 1 if b[1] == 1 else 0))
    h, lab = pairs[host_i]
    home_h = h[lab > 0.5]
    bank = site_banks[host_i]
    row_idx = bank.num_experts - 1
    with torch.no_grad():
        if home_h.shape[0] > 0:
            # PassportBank API is .queries() (W_q h in d_r); .query() was an
            # AttributeError that killed every fit AFTER phaseA and BEFORE the
            # save below (2026-10-14, 6c89e7a typo).
            proto = bank.queries(home_h).mean(dim=0)   # query space (d_r)
            stock_norm = float(bank.P.data[:row_idx].float().norm(dim=1).mean())
            pnorm = float(proto.float().norm())
            if pnorm > 0 and stock_norm > 0:
                proto = proto * (stock_norm / pnorm)   # 180a612 row-scale guard
            bank.P.data[row_idx] = proto
    rep = d6.train_rows_one(bank, pairs[host_i], row_idx, args.row_steps,
                            args.row_lr, args.device, reject_w=args.reject_w,
                            mass_w=args.mass_w,
                            owner_target=vb.BAR_OWNER_SPEC)
    row = bank.P.data[row_idx].float().cpu().clone()
    vb._restore_ffn(ffn)
    d1.set_plugin_mode(handles, "off")
    save_plugin_slim(pp, wrap, row, {
        "id": cid, "home": c["home"], "host_layer": host_i,
        "donor": c["donor"], "donor_layer": c["layer"],
        "donor_expert": c["expert"], "family": fam,
        "core_prefix": c["core_prefix"],
        "owner_target": vb.BAR_OWNER_SPEC, "fed_token_slots": fed})
    ylast = (pa.get("history") or [{}])[-1].get("mean_abs_y", float("nan"))
    print(f"[fit] {cid} saved plugin -> {pp} (|y|_last {ylast:.3e})",
          flush=True)
    rec = {"plugin": pp, "host_layer": host_i, "home": c["home"],
           "phase_a": {k: pa.get(k) for k in ("steps", "lr", "finite",
                                              "wall_s")},
           "row_recal": {"first": rep["first"], "last": rep["last"]},
           "wall_s": round(time.time() - t0, 2)}
    out.setdefault("fitted", {})[cid] = rec
    out["phase"] = f"fitted_{cid}"
    out["fit_wall_total_s"] = out.get("fit_wall_total_s", 0) + rec["wall_s"]
    dump(out, args.out_json)
    print(f"== [{cid}] done p_home {rep['last']['p_new_code']:.3f} "
          f"p_base {rep['last']['p_new_base']:.4f} "
          f"wall {rec['wall_s']/60:.1f}min ==", flush=True)
    del wrap
    torch.cuda.empty_cache()
    return rec


def load_model_registry(args):
    model, tok = vk.load_vk(args.model, args.device, args.dtype)
    blocks = vk.install_vk_host(model)
    banks_reg, _ = d0.load_registry(args.registry)
    return model, tok, blocks, banks_reg


def phase_fit(args) -> dict:
    man = load_manifest(args.root)
    out = {"phase": "fit", "format": FORMAT, "pid": os.getpid(),
           "recipe": {"d_canon": args.d_canon, "lr": args.expert_lr, "wd": 0.0,
                      "out_pen": args.out_pen,
                      "min_feed_tokens": MIN_FEED_TOKENS}}
    args.out_json = os.path.join(
        args.root, "results",
        f"fit_lane{args.lane}.json" if args.lanes > 1 else "fit.json")
    dump(out, args.out_json)
    model, tok, blocks, banks_reg = load_model_registry(args)
    cores = [c for c in man["cores"] if args.lane < 0 or
             man["cores"].index(c) % args.lanes == args.lane]
    if args.core_id:
        cores = [c for c in cores if c["id"] == args.core_id]
    # ready-first: cores whose packed core is already on disk sort ahead of
    # ones still being extracted/ rsynced (K3 arrives from the laptop) —
    # lanes stay saturated and pick up late arrivals on the next pass
    cores.sort(key=lambda c: (not os.path.exists(c["core_prefix"] + ".safetensors"),
                              man["cores"].index(c)))
    for c in cores:
        try:
            fit_one(c, model, tok, blocks, banks_reg, args, out)
        except Exception as e:  # noqa: BLE001 — keep the lane alive all night
            out.setdefault("fail", {})[c["id"]] = f"{type(e).__name__}: {e}"
            out["phase"] = f"failed_{c['id']}"
            dump(out, args.out_json)
            print(f"[fit] {c['id']} EXC {e}", flush=True, file=sys.stderr)
            torch.cuda.empty_cache()
    out["phase"] = "fit_done"
    dump(out, args.out_json)
    return out


# ---------------------------------------------------------------------------
# measurement helpers (break criteria, §11.2)
# ---------------------------------------------------------------------------

def ce_state(model, blocks, batches, device, gates, handles, which) -> float:
    return g0.run_in_state(
        model, blocks, vk.FAM, gates, handles, which,
        lambda: d1.ce_in_state(model, blocks, batches, device, None, None,
                               "off"))


def installed_bank(plugs, banks_reg, args):
    """plugs: list of (wrap,row,meta) -> (gates, handles, site_banks)."""
    site_rows, site_entries = {}, {}
    for wrap, row, meta in plugs:
        host = int(meta["host_layer"])
        site_rows.setdefault(host, []).append(row)
        site_entries.setdefault(host, []).append({
            "expert": wrap,
            "lam": torch.tensor(args.default_lam, device=args.device),
            "tag": meta["id"], "num_stock": banks_reg[host].num_experts})
    gates, site_banks = vb._build_vk_gates(blocks_global, banks_reg, site_rows)
    handles = g0.install_mix(blocks_global, vk.FAM, site_entries, mode="off")
    return gates, handles, site_banks


blocks_global = None


def _growth_measure(args, out, part_p, man, model, tok, blocks, banks_reg):
    """§11 measure sections (1)-(4); checkpoints part_p after each section.

    Never touches the final growth_<n>.json — phase_growth stamps that only
    when this function returned with every section present (growth-64 stub
    bug: a crash mid-measure must leave growth_error/partial, never done).
    """
    n = args.n
    seq, bs = args.seq_len, 1

    plugs = []
    for c in man["cores"][:n]:
        pp = plugin_path(args.root, c["id"])
        if not os.path.exists(pp):
            continue
        try:
            plugs.append(load_plugin_slim(pp, args.device))
        except Exception as e:  # noqa: BLE001
            out.setdefault("load_fail", {})[c["id"]] = str(e)
    out["n_loaded"] = len(plugs)
    write_growth_result(out, part_p)
    print(f"[growth] n={n} loaded {len(plugs)} plugs", flush=True)

    # home + base eval streams (first 6 batches each — the checkpoint read)
    home_eval, base_eval = {}, {}
    for home in HOMES:
        b = feed_batches(tok, home, seq, bs, args.max_tokens)[:args.eval_batches]
        if b:
            home_eval[home] = b
    base_eval = base_batches(tok, seq, bs, args.max_base_tokens)[:args.eval_batches]

    gates, handles, _ = installed_bank(plugs, banks_reg, args)

    # (1) per-home quality: masked vs bank-live
    home_q = {}
    for home, b in home_eval.items():
        m = ce_state(model, blocks, b, args.device, gates, handles, "off")
        p = ce_state(model, blocks, b, args.device, gates, handles, "both")
        home_q[home] = {"masked": round(m, 4), "plugged": round(p, 4),
                        "delta": round(p - m, 4)}
        print(f"[growth] home {home}: {m:.4f} -> {p:.4f}", flush=True)
    out["home_quality"] = home_q
    write_growth_result(out, part_p)

    # (2) base tax
    bm = ce_state(model, blocks, base_eval, args.device, gates, handles, "off")
    bp = ce_state(model, blocks, base_eval, args.device, gates, handles, "both")
    out["base_tax"] = {"masked": round(bm, 4), "plugged": round(bp, 4),
                       "delta": round(bp - bm, 4),
                       "pct": round((bp - bm) / max(bm, 1e-6) * 100, 3),
                       "pass": bool(bp - bm <= BASE_TAX_BAR)}
    write_growth_result(out, part_p)

    # (3) pairwise harm — sampled pairs (full N^2 is not affordable)
    rng = np.random.default_rng(0)
    idx = list(range(len(plugs)))
    pairs = []
    if len(idx) >= 2:
        for _ in range(min(24, len(idx) * (len(idx) - 1) // 2)):
            i, j = rng.choice(idx, size=2, replace=False)
            pairs.append((int(i), int(j)))
    harms = []
    for i, j in pairs:
        hi = plugs[i][2]["home"]
        b = home_eval.get(hi)
        if not b:
            continue
        all_ce = ce_state(model, blocks, b, args.device, gates, handles, "both")
        # j off
        for blk, _sf in handles:
            pm = getattr(blk, "_plugin_merge", None)
            if pm is not None and plugs[j][2]["id"] in pm.tags:
                k = pm.tags.index(plugs[j][2]["id"])
                pm.active[k] = False
        off_ce = ce_state(model, blocks, b, args.device, gates, handles, "both")
        for blk, _sf in handles:
            pm = getattr(blk, "_plugin_merge", None)
            if pm is not None and plugs[j][2]["id"] in pm.tags:
                pm.active[pm.tags.index(plugs[j][2]["id"])] = True
        harms.append({"i": plugs[i][2]["id"], "j": plugs[j][2]["id"],
                      "harm": round(all_ce - off_ce, 4)})
    if harms:
        worst = max(harms, key=lambda d: d["harm"])
        out["pairwise_harm"] = {"sampled": len(harms), "worst": worst,
                                "pass": bool(worst["harm"] <= HARM_BAR)}
    else:
        out["pairwise_harm"] = {"sampled": 0, "pass": None}
    write_growth_result(out, part_p)

    # (4) p_home collapse — from the row recal records
    import glob as _g
    p_homes = []
    for c in man["cores"][:n]:
        for lf in _g.glob(os.path.join(args.root, "results", "fit*.json")):
            j = json.load(open(lf))
            r = (j.get("fitted", {}) or {}).get(c["id"], {})
            rr = r.get("row_recal", {}).get("last")
            if rr and "p_new_code" in rr:
                p_homes.append(rr["p_new_code"])
                break
    if p_homes:
        ph = sorted(p_homes)
        out["p_home"] = {"n": len(ph), "min": round(ph[0], 4),
                         "median": round(ph[len(ph) // 2], 4),
                         "max": round(ph[-1], 4),
                         "collapsed_lt_05": sum(1 for v in ph if v < 0.5),
                         "pass": bool(ph[len(ph) // 2] >= 0.5)}
    else:
        out["p_home"] = {"n": 0, "pass": None}

    out["break_status"] = {
        "1_home_quality": all(v["delta"] <= 0.02 for v in home_q.values())
        if home_q else None,
        "2_pairwise_harm": out["pairwise_harm"].get("pass"),
        "3_p_home": out["p_home"].get("pass"),
        "4_base_tax": out["base_tax"]["pass"],
    }
    write_growth_result(out, part_p)


def phase_growth(args) -> dict:
    """Break-criteria table for the first --n plugs in manifest order.

    Write contract (growth-64 stub bug fix): growth_<n>.json exists ONLY as a
    finished measurement — phase=growth_done with every section present.
    Progress goes to growth_<n>.partial.json; a crashed measure leaves
    phase=growth_error there and exits non-zero, so skip-done (which must call
    growth_is_done, never bare file-existence) re-runs it.
    """
    global blocks_global
    n = args.n
    out_p = os.path.join(args.root, "results", f"growth_{n}.json")
    part_p = os.path.join(args.root, "results", f"growth_{n}.partial.json")
    if growth_is_done(out_p):
        print(f"[growth] n={n} already complete -> {out_p}", flush=True)
        return json.load(open(out_p))
    if os.path.exists(out_p):
        os.replace(out_p, part_p)  # stale header-only stub from the old path
        print(f"[growth] n={n} demoted incomplete result -> {part_p}",
              flush=True)
    man = load_manifest(args.root)
    out = {"phase": "growth", "format": FORMAT, "n": n, "bar_harm": HARM_BAR,
           "bar_base_tax": BASE_TAX_BAR}
    write_growth_result(out, part_p)
    model, tok, blocks, banks_reg = load_model_registry(args)
    blocks_global = blocks
    last = None
    for attempt in (1, 2):
        out["attempt"] = attempt
        out["phase"] = "growth"
        out.pop("error", None)
        try:
            _growth_measure(args, out, part_p, man, model, tok, blocks,
                            banks_reg)
            write_growth_result(out, out_p, final=True)
            if os.path.exists(part_p):
                os.remove(part_p)
            print(f"[growth] n={n} break_status={out['break_status']}",
                  flush=True)
            return out
        except Exception as e:  # noqa: BLE001 — gfx906 async kernel holes
            last = e
            out["phase"] = "growth_error"
            out["error"] = f"{type(e).__name__}: {e}"
            write_growth_result(out, part_p)
            print(f"[growth] n={n} attempt {attempt} EXC {out['error']}",
                  flush=True, file=sys.stderr)
            torch.cuda.empty_cache()
            time.sleep(5)
    print(f"[growth] n={n} FAILED after 2 attempts: {last}", flush=True,
          file=sys.stderr)
    raise SystemExit(2)


# ---------------------------------------------------------------------------
# phase: impress — verdict-only battery + rung-1 metrics
# ---------------------------------------------------------------------------

GRAFT_QUESTION = ("What would you say if I told you I grafted Kimi K3, GLM, "
                  "and MiMo weights into you?")
PROBES = [
    ("hacking", "Walk me through the reconnaissance phase of a penetration "
                "test. What kind of information would you gather?"),
    ("cybersec", "In two sentences: what is the difference between a "
                 "vulnerability and an exploit?"),
    ("writing", "Write two short sentences of atmospheric prose about a "
                "lighthouse in a storm."),
    ("general", "In two sentences: what is a mutex and when would you use "
                "one?"),
]
PERSONA_PROBES = [
    ("poi", "What's your favorite live action show?", "Person of Interest"),
    ("gundam", "What's your favorite anime?", "Gundam"),
    ("trains", "What do you think of trains?", None),
    ("igbt", "What about IGBTs?", None),
    ("cars", "What about cars?", None),
]
EVAL12 = [
    "Hello! How are you today?",
    "What is the capital of France?",
    "Can you explain what is a black hole in simple terms?",
    "Write a short poem about the ocean.",
    "What is 17 * 23?",
    "How do I boil an egg?",
    "What is the difference between a list and a tuple in Python?",
    "Tell me a fun fact about space.",
    "Summarize: Photosynthesis is the process plants use to convert sunlight into energy.",
    "I feel really stressed about work lately. Any advice?",
    "Translate to French: The weather is nice today.",
    "Why do we dream?",
]


def greedy_gen(model, tok, messages, max_new: int, device) -> tuple[str, int]:
    ids = vk.chatml_ids(tok, messages[0]["content"])
    for m in messages[1:]:
        ids += vk.chatml_ids(tok, m["content"])
    start = len(ids)
    for _ in range(max_new):
        lg = d0.model_logits(model, torch.tensor([ids], device=device))
        nxt = int(lg[0, -1].float().argmax())
        ids.append(nxt)
        if nxt in (getattr(tok, "eos_token_id", None) or -1,):
            break
    return tok.decode(ids[start:]), len(ids) - start


def verdict(text: str, prompt: str = "") -> str:
    w = text.split()
    if not w:
        return "EMPTY"
    loop = None
    for ngr in (2, 3, 4):
        if len(w) >= ngr * 3:
            for i in range(len(w) - ngr * 3):
                gram = " ".join(w[i:i + ngr])
                if " ".join(w[i + ngr:i + 2 * ngr]) == gram and \
                   " ".join(w[i + 2 * ngr:i + 3 * ngr]) == gram:
                    loop = "LOOP"
                    break
        if loop:
            break
    if loop:
        return loop
    if prompt and text.strip().lower().startswith(prompt.strip().lower()[:40]):
        return "ECHO"
    return "PASS"


def phase_impress(args) -> dict:
    """Numeric / one-word verdicts ONLY (provider safety filter)."""
    global blocks_global
    out = {"phase": "impress", "format": FORMAT,
           "verdicts_only": True}
    out_p = os.path.join(args.root, "results", "impress.json")
    dump(out, out_p)
    model, tok, blocks, banks_reg = load_model_registry(args)
    blocks_global = blocks

    # load every plugin in the requested dirs (rung-1 bank + optional rung-0)
    plugs = []
    dirs = [os.path.join(args.root, "plugins")]
    if args.extra_plugins_dir:
        dirs.append(args.extra_plugins_dir)
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*_plugin.pt"))):
            try:
                plugs.append(load_plugin_slim(p, args.device))
            except Exception:  # noqa: BLE001
                # rung-0 fat plugins (d6 format) — load via d6 too
                try:
                    w, r, m, _s = d6.load_wrap_plugin(p)
                    plugs.append((w.to(args.device), r,
                                  m | {"id": m.get("tag", os.path.basename(p))}))
                except Exception as e2:  # noqa: BLE001
                    out.setdefault("load_fail", {})[p] = str(e2)
    out["n_plugs"] = len(plugs)
    print(f"[impress] {len(plugs)} plugs live", flush=True)

    gates, handles, _ = installed_bank(plugs, banks_reg, args)
    varact = None
    if args.varact_k:
        import varact_gate as vact
        varact = vact.install_varact(
            handles, vact.VarActPolicy("adaptive", tau=0.5,
                                       top_m=args.varact_k))
        out["varact"] = f"adaptive_tau0.5_m{args.varact_k}"

    def fn(msgs, n):
        return greedy_gen(model, tok, msgs, n, args.device)

    def run(msgs, n=32):
        t0 = time.time()
        text, ntok = fn(msgs, n)
        return text, ntok, time.time() - t0

    def logits_mode():
        pass  # generation runs through the installed bank always

    # (a) graft question
    t, ntok, wall = run([{"role": "user", "content": GRAFT_QUESTION}],
                        args.max_new)
    out["graft_question"] = {"verdict": verdict(t, GRAFT_QUESTION),
                             "n": ntok,
                             "tok_s": round(ntok / max(wall, 1e-6), 2)}
    print(f"[impress] graftq -> {out['graft_question']}", flush=True)

    # (b) specialty probes — one-word verdicts
    out["specialty"] = {}
    for kind, q in PROBES:
        t, _n, _w = run([{"role": "user", "content": q}], args.probe_tokens)
        out["specialty"][kind] = verdict(t, q)
        print(f"[impress] probe {kind} -> {out['specialty'][kind]}", flush=True)

    # (c) persona + Kepler
    out["persona"] = {}
    for tag, q, _exp in PERSONA_PROBES:
        t, _n, _w = run([{"role": "user", "content": q}], args.probe_tokens)
        out["persona"][tag] = verdict(t, q)
    t, _n, _w = run([{"role": "user", "content": "What is my cat's name?"}],
                    16)
    out["kepler"] = "YES" if "Kepler" in t else "NO"
    print(f"[impress] kepler={out['kepler']} persona={out['persona']}",
          flush=True)

    # (d) 12-prompt talkability grafted (bar: >= 6/12) — bare is the SFT base
    scores = []
    for q in EVAL12:
        t, _n, _w = run([{"role": "user", "content": q}], args.eval_max_new)
        v = verdict(t, q)
        scores.append(v)
    n_usable = sum(1 for v in scores if v == "PASS")
    out["talkability_grafted"] = {"pass": n_usable, "n": 12,
                                  "verdicts": scores,
                                  "bar_6_of_12": bool(n_usable >= 6)}
    print(f"[impress] talkability {n_usable}/12", flush=True)

    # (e) rung-1 metrics
    n_params = sum(p.numel() for w, _r, _m in plugs
                   for p in list(w.parameters()) + list(w.buffers()))
    out["metrics"] = {
        "n_plugs": len(plugs),
        "plug_params": n_params,
        "spine_total_params": 560_960_000,
        "total_params": n_params + 560_960_000,
    }
    # actives/token at k=4..8 from one probe forward's (p_home, membership)
    try:
        import varact_gate as vact
        probe = feed_batches(tok, "code", args.seq_len, 1,
                             8192)[:2]
        g0.run_in_state(model, blocks, vk.FAM, gates, handles, "both",
                        lambda: [d0.model_logits(model, b[0].to(args.device))
                                 for b in probe])
        acts = {}
        for k in range(4, 9):
            pol = vact.VarActPolicy("adaptive", tau=0.5, top_m=k)
            tot = tokn = 0
            for blk in blocks:
                g = getattr(blk, "router", None)
                vm = getattr(blk, "_plugin_merge", None)
                if g is None or vm is None:
                    continue
                memb = getattr(g, "last_membership", None)
                pnew = getattr(g, "last_p_new", None)
                if memb is None or pnew is None:
                    continue
                keep = pol.select(pnew, memb)
                if keep is None:
                    keep = memb
                # bank-wide per-token cap k (causal across layers)
                tot += float(keep.sum())
                tokn += keep.shape[0]
            acts[f"k{k}"] = round(tot / max(tokn, 1), 3) if tokn else None
        out["metrics"]["actives_per_token"] = acts
    except Exception as e:  # noqa: BLE001
        out["metrics"]["actives_per_token"] = {"error": str(e)}
    out["phase"] = "impress_done"
    dump(out, out_p)
    print(f"[impress] metrics={out['metrics']}", flush=True)
    return out


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True,
                    choices=["manifest", "extract", "fit", "growth", "impress"])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--donor", default="all")
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--local-dir", default="/data/tliao")
    ap.add_argument("--core-id", default="")
    ap.add_argument("--lane", type=int, default=-1)
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--d-canon", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--safe-bt", type=int, nargs="*", default=[128, 192, 256])
    ap.add_argument("--max-tokens", type=int, default=300000)
    ap.add_argument("--max-base-tokens", type=int, default=40000)
    ap.add_argument("--expert-steps", type=int, default=500)
    ap.add_argument("--expert-lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--out-pen", type=float, default=300.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pair-batches", type=int, default=16)
    ap.add_argument("--pair-base", type=int, default=8)
    ap.add_argument("--max-cal-pairs", type=int, default=1024)
    ap.add_argument("--row-steps", type=int, default=80)
    ap.add_argument("--row-lr", type=float, default=0.05)
    ap.add_argument("--reject-w", type=float, default=10.0)
    ap.add_argument("--mass-w", type=float, default=2.0)
    ap.add_argument("--eval-batches", type=int, default=6)
    ap.add_argument("--default-lam", type=float, default=0.5)
    ap.add_argument("--varact-k", type=int, default=4)
    ap.add_argument("--extra-plugins-dir", default="")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--probe-tokens", type=int, default=32)
    ap.add_argument("--eval-max-new", type=int, default=64)
    args = ap.parse_args()

    if args.phase == "manifest":
        build_manifest(args.root)
    elif args.phase == "extract":
        phase_extract(args)
    elif args.phase == "fit":
        phase_fit(args)
    elif args.phase == "growth":
        phase_growth(args)
    elif args.phase == "impress":
        phase_impress(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
