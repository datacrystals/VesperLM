#!/usr/bin/env python3
"""Door 9 — embedding-anchored basis translation (the better-posed D1.5).

D1.5 (lab/graft_d15_translator.py) measured telemetry-trained feature
translators at held-out cos 0.55-0.69 and read the residual as "feature
availability".  That experiment's pairing was word-level pooled features —
noisy targets across two different tokenizers.  Embedding tables are
EXACTLY pairable by token identity (decode each id, match strings), giving a
vocab-sized clean training set with zero alignment error.

Q (this door):
  1. Pair the two on-disk donors' embedding tables by decoded string
     (single-token round-trip strings only): OLMoE-1B-7B-0924 (host,
     d_model 2048) <-> Phi-mini-MoE-instruct (donor, d_model 4096).
  2. Fit W: emb_B -> emb_A (ridge linear first, small MLP second) and measure
     train + held-out cosine.  Bars: D1.5 translators capped at 0.72-0.76
     train cos / 0.55-0.69 held-out; if embedding-space W hits >= 0.95 train
     / >= 0.90 held-out the input basis is linearly bridgeable and the D1.5
     verdict must be revised to "depth divergence + pairing loss, not feature
     availability".
  3. Practical payoff: use W to INITIALIZE the door25 wrap's geometrically
     correct slot — the host->donor input chain (host_in + adapter_up, the
     rank-d_canon factorization of W) — instead of random init, then run the
     standard single-site wrap + recal on the OLMoE host (door25 recipe) and
     measure quality held-out (the memorization gap), recal steps needed, and
     final quality vs the door25 reference (0.439 vs 1.072 in-sample).
  4. Depth decay: apply the emb-fitted W to mid-layer states of matched
     single-token strings and report cosine decay by depth (plus a
     per-depth fitted ridge as the availability ceiling).

Geometry (door25 wrap, unchanged):
    x(2048) --host_in--> c(64) --adapter_up--> u(4096) --CORE(frozen)-->
    v(4096) --adapter_down--> c'(64) --host_out--> delta(2048)

W: emb_OLMoE -> emb_Phi is a host->donor map, i.e. exactly the composite
adapter_up . host_in.  Arm `winit` installs W's rank-64 SVD there (scale-
matched to donor residual RMS at the core's depth) and leaves
adapter_down/host_out at the door25 random/zero init (identity at birth).
Arm `rand` is door25 verbatim (seed 0).  Arm `wboth` (optional) also
initializes the donor->host output chain from the inverse fit.

Fork of lab/door25_wrap_expert.py's pipeline; d0/d1/d2/d25/d15 are imported
UNMODIFIED.

Usage:
  python3 lab/door9_embedding_w.py --synthetic
  python3 lab/door9_embedding_w.py --phase emb \
      --model-a /data/tliao/OLMoE-1B-7B-0924 \
      --model-b /data/tliao/Phi-mini-MoE-instruct \
      --out-dir /data/tliao/graft_door9_out
  python3 lab/door9_embedding_w.py --phase depth ...
  python3 lab/door9_embedding_w.py --phase wrap --arm winit \
      --model /data/tliao/OLMoE-1B-7B-0924 \
      --registry /data/tliao/graft_d0_out/registry_olmoe_6d84c485.pt \
      --core-from /data/tliao/graft_door25_out/door25_core.pt \
      --domains-dir /data/tliao/graft_d0_slice ...

gfx906/MI50 notes (carried): trunk BACKWARD size holes (B*T 256/768/1024/
4096 die; 2048 verified) — train bs4 x seq512 = 2048; eval bs2 x seq512.
Thin-wrap recipe lr 3e-3 / wd 0 / clip 1.0; NaN guard with one half-lr
fresh-init fallback.  Embedding math in fp32; trunk fp16.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import graft_d0_parity as d0  # noqa: E402
import graft_d1_plugin as d1  # noqa: E402
import door25_wrap_expert as d25  # noqa: E402

FORMAT = "vesper-door9-embedding-w-v1"

# D1.5 comparison bars (task statement)
BAR_TRAIN_COS = 0.95
BAR_HELD_COS = 0.90
# door25 reference cells (in-sample code CE: masked 1.072 / plugged 0.439)
REF_DOOR25 = {"ce_code_masked": 1.072, "ce_code_active": 0.439,
              "ce_code_forced_solo": 1.072 * 0 + 0.101,
              "ce_code_heldout_masked": 1.587, "ce_code_heldout_active": 1.621}


def dump_json(out: dict, path: str | None) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _safe(o):
        if isinstance(o, torch.Tensor):
            if o.numel() == 1:
                return float(o)
            return {"__tensor_shape__": list(o.shape)}
        if isinstance(o, dict):
            return {str(k): _safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_safe(v) for v in o]
        return o

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_safe(out), f, indent=2)
    os.replace(tmp, path)


def cos_stats(pred: torch.Tensor, tgt: torch.Tensor) -> dict:
    """Per-row cosine summary (fp32)."""
    pred = pred.float()
    tgt = tgt.float()
    c = F.cosine_similarity(pred, tgt, dim=-1)
    v = c.sort().values
    n = v.numel()
    return {"n": int(n), "mean": float(v.mean()), "std": float(v.std()),
            "p10": float(v[max(0, n // 10 - 1)]), "p50": float(v[n // 2]),
            "min": float(v[0]), "max": float(v[-1]),
            "frac_ge_0.90": float((v >= 0.90).float().mean()),
            "frac_ge_0.95": float((v >= 0.95).float().mean())}


def _stable_split(strings, held_frac: float = 0.1):
    train, held = [], []
    for s in strings:
        h = int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        (held if h < held_frac else train).append(s)
    return train, held


# ---------------------------------------------------------------------------
# Phase emb: exact string pairing of embedding tables + W fit
# ---------------------------------------------------------------------------


def load_embed_weight(model_dir: str) -> torch.Tensor:
    """model.embed_tokens.weight as CPU fp32, without loading the full model."""
    from safetensors import safe_open
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    files = []
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        files = sorted({os.path.join(model_dir, p)
                        for p in idx.get("weight_map", {}).values()
                        if "embed_tokens" in p or True})
        want = idx.get("weight_map", {}).get("model.embed_tokens.weight")
        if want:
            files = [os.path.join(model_dir, want)]
    if not files:
        files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    for fp in files:
        if not os.path.exists(fp):
            continue
        with safe_open(fp, framework="pt") as sf:
            for k in sf.keys():
                if k.endswith("embed_tokens.weight"):
                    return sf.get_tensor(k).float().cpu()
    raise RuntimeError(f"embed_tokens.weight not found under {model_dir}")


def build_string_table(tok, max_rows: int | None = None) -> dict:
    """id -> string, keeping only clean single-token round-trip strings.

    A row is kept iff: not a special id, decodes to a non-empty string, and
    that string re-encodes to exactly that one id.  This kills byte-fallback
    garbage, multi-token merges and specials — the pairing set is then exact.
    """
    n = len(tok)
    if max_rows:
        n = min(n, max_rows)
    special = set(tok.all_special_ids) if hasattr(tok, "all_special_ids") else set()
    table = {}
    for i in range(n):
        if i in special:
            continue
        try:
            s = tok.decode([i])
        except Exception:
            continue
        if not s:
            continue
        try:
            re_ids = tok.encode(s, add_special_tokens=False)
        except Exception:
            continue
        if re_ids != [i]:
            continue
        table[s] = i
    return table


def fit_maps(x_tr: torch.Tensor, y_tr: torch.Tensor,
             x_held: torch.Tensor, y_held: torch.Tensor,
             ridge: float = 1e-3, mlp_steps: int = 1200,
             mlp_hidden: int = 512, device: str = "cpu") -> dict:
    """Ridge linear + mlp2 fits x->y (and the identity-scale mean baseline)."""
    import graft_d15_translator as d15
    out = {}
    d_src, d_tgt = x_tr.shape[1], y_tr.shape[1]

    t0 = time.time()
    lin = d15.Translator("linear", d_src, d_tgt, ridge=ridge)
    lin.fit_ridge(x_tr, y_tr)
    with torch.no_grad():
        ptr = lin(x_tr)
        phe = lin(x_held)
    out["linear"] = {"ridge": ridge,
                     "train": cos_stats(ptr, y_tr),
                     "held": cos_stats(phe, y_held),
                     "held_mse": float(F.mse_loss(phe.float(), y_held.float())),
                     "wall_s": round(time.time() - t0, 2),
                     "W": lin.W.detach().float().cpu(),
                     "b": lin.b.detach().float().cpu()}

    t0 = time.time()
    mlp = d15.Translator("mlp2", d_src, d_tgt, hidden=mlp_hidden)
    rep = mlp.fit_adam(x_tr, y_tr, steps=mlp_steps, lr=1e-3,
                       batch=min(4096, x_tr.shape[0]), cos_w=0.5,
                       device=device)
    with torch.no_grad():
        ptr = mlp(x_tr.to("cpu"))
        phe = mlp(x_held.to("cpu"))
    out["mlp2"] = {"hidden": mlp_hidden, "steps": mlp_steps,
                   "train": cos_stats(ptr, y_tr),
                   "held": cos_stats(phe, y_held),
                   "held_mse": float(F.mse_loss(phe.float(), y_held.float())),
                   "fit_final": rep.get("final"),
                   "wall_s": round(time.time() - t0, 2),
                   "state": {k: v.detach().float().cpu()
                             for k, v in mlp.state_dict().items()}}

    with torch.no_grad():
        mu = y_tr.mean(dim=0, keepdim=True).expand_as(y_held)
    out["baseline_mean"] = {"held": cos_stats(mu, y_held),
                            "train": cos_stats(
                                y_tr.mean(dim=0, keepdim=True).expand_as(y_tr),
                                y_tr)}
    return out


def run_emb(args, out: dict) -> None:
    t0 = time.time()
    from transformers import AutoTokenizer
    tok_a = AutoTokenizer.from_pretrained(args.model_a)
    tok_b = AutoTokenizer.from_pretrained(args.model_b)
    emb_a = load_embed_weight(args.model_a)   # host (OLMoE), (V_a, 2048)
    emb_b = load_embed_weight(args.model_b)   # donor (Phi),  (V_b, 4096)
    d_a, d_b = emb_a.shape[1], emb_b.shape[1]
    out["models"] = {"a": args.model_a, "b": args.model_b,
                     "emb_a": list(emb_a.shape), "emb_b": list(emb_b.shape)}

    tab_a = build_string_table(tok_a)
    tab_b = build_string_table(tok_b)
    common = sorted(set(tab_a) & set(tab_b))
    pairs_a = torch.tensor([tab_a[s] for s in common], dtype=torch.long)
    pairs_b = torch.tensor([tab_b[s] for s in common], dtype=torch.long)
    out["pairing"] = {
        "n_clean_a": len(tab_a), "n_clean_b": len(tab_b),
        "n_intersect": len(common),
        "n_intersect_alpha": sum(1 for s in common if s.strip().isalpha()),
        "n_intersect_alnum_ws": sum(
            1 for s in common if all(c.isalnum() or c.isspace() for c in s)),
        "held_frac": args.held_frac,
    }
    if len(common) < 1000:
        raise RuntimeError(f"intersection too small: {len(common)}")

    strs_tr, strs_held = _stable_split(common, args.held_frac)
    out["pairing"]["n_train"] = len(strs_tr)
    out["pairing"]["n_held"] = len(strs_held)
    s2i = {s: i for i, s in enumerate(common)}
    idx_tr = torch.tensor([s2i[s] for s in strs_tr])
    idx_held = torch.tensor([s2i[s] for s in strs_held])

    x_all = emb_b[pairs_b]   # donor emb rows = source for W: b -> a
    y_all = emb_a[pairs_a]   # host emb rows = target
    x_tr, y_tr = x_all[idx_tr], y_all[idx_tr]
    x_held, y_held = x_all[idx_held], y_all[idx_held]
    out["scale"] = {
        "emb_a_row_rms": float(y_all.pow(2).mean(1).sqrt().mean()),
        "emb_b_row_rms": float(x_all.pow(2).mean(1).sqrt().mean()),
    }

    print(f"pairing: clean a {len(tab_a)} b {len(tab_b)} "
          f"intersect {len(common)} train {len(strs_tr)} held {len(strs_held)}",
          flush=True)

    # forward: emb_B -> emb_A (host<-donor? NO: A=host=OLMoE target, B=donor
    # source — the wrap's host->donor input chain wants emb_A -> emb_B, i.e.
    # host -> donor.  We fit BOTH directions; `fwd` here = b->a as the task's
    # W: emb_B -> emb_A, and `inv` = a->b which is the wrap input-chain map.)
    print("fit W: emb_B -> emb_A (task direction)", flush=True)
    fwd = fit_maps(x_tr, y_tr, x_held, y_held, ridge=args.ridge,
                   mlp_steps=args.mlp_steps, device=args.device)
    print(f"  linear train cos {fwd['linear']['train']['mean']:.4f} "
          f"held {fwd['linear']['held']['mean']:.4f}  "
          f"mlp2 train {fwd['mlp2']['train']['mean']:.4f} "
          f"held {fwd['mlp2']['held']['mean']:.4f}", flush=True)

    print("fit W_inv: emb_A -> emb_B (wrap input-chain direction)", flush=True)
    inv = fit_maps(y_tr, x_tr, y_held, x_held, ridge=args.ridge,
                   mlp_steps=args.mlp_steps, device=args.device)
    print(f"  linear train cos {inv['linear']['train']['mean']:.4f} "
          f"held {inv['linear']['held']['mean']:.4f}  "
          f"mlp2 train {inv['mlp2']['train']['mean']:.4f} "
          f"held {inv['mlp2']['held']['mean']:.4f}", flush=True)

    def slim(fit: dict) -> dict:
        # drop tensor payloads (W/b/state) at both the arm level and nested
        out = {}
        for k, v in fit.items():
            if k in ("W", "b", "state"):
                continue
            if isinstance(v, dict):
                out[k] = {kk: vv for kk, vv in v.items()
                          if kk not in ("W", "b", "state")}
            else:
                out[k] = v
        return out

    out["fit"] = {"W_embB_to_embA": slim(fwd), "W_embA_to_embB": slim(inv)}
    out["verdict_emb"] = {
        "bar_train": BAR_TRAIN_COS, "bar_held": BAR_HELD_COS,
        "b_to_a_train": fwd["linear"]["train"]["mean"],
        "b_to_a_held": fwd["linear"]["held"]["mean"],
        "a_to_b_train": inv["linear"]["train"]["mean"],
        "a_to_b_held": inv["linear"]["held"]["mean"],
        "pass_linear": bool(fwd["linear"]["train"]["mean"] >= BAR_TRAIN_COS
                            and fwd["linear"]["held"]["mean"] >= BAR_HELD_COS),
        "mlp_beats_linear_held": bool(
            fwd["mlp2"]["held"]["mean"] > fwd["linear"]["held"]["mean"] + 0.005),
        "d15_train_ceiling": 0.76, "d15_held_ceiling": 0.69,
    }

    # save maps (linear part is what the wrap init uses)
    os.makedirs(args.out_dir, exist_ok=True)
    w_path = os.path.join(args.out_dir, "door9_w.pt")
    torch.save({
        "format": FORMAT,
        "W_b_to_a": fwd["linear"]["W"], "b_b_to_a": fwd["linear"]["b"],
        "W_a_to_b": inv["linear"]["W"], "b_a_to_b": inv["linear"]["b"],
        "d_a": d_a, "d_b": d_b,
        "meta": {"n_train": len(strs_tr), "n_held": len(strs_held),
                 "n_intersect": len(common)},
    }, w_path)
    out["w_path"] = w_path
    out["wall_s"] = round(time.time() - t0, 2)
    out["phase"] = "emb_complete"
    dump_json(out, args.out)
    print(f"emb phase done in {out['wall_s']}s -> {args.out}", flush=True)


# ---------------------------------------------------------------------------
# Phase depth: does emb-W transfer to mid-layer states?  (depth-divergence)
# ---------------------------------------------------------------------------


@torch.no_grad()
def capture_hidden(model, tok, strings, device, batch: int = 64) -> dict:
    """Per-layer hidden states for single-token strings (N, d) fp32 per layer.

    Each string round-trips to exactly one id in this tokenizer, so the row is
    exactly pairable with the other model's same-string token — zero alignment
    error, unlike D1.5's word pooling.
    """
    ids = []
    keep = []
    for s in strings:
        i = tok.encode(s, add_special_tokens=False)
        if len(i) == 1:
            ids.append(i[0])
            keep.append(s)
    hs = None
    for start in range(0, len(ids), batch):
        chunk = torch.tensor(ids[start:start + batch], dtype=torch.long,
                             device=device).unsqueeze(1)
        o = model(output_hidden_states=True, use_cache=False, labels=None,
                  **{"input_ids": chunk})
        h = [hh[:, 0, :].float().cpu() for hh in o.hidden_states]
        hs = h if hs is None else [torch.cat([a, b], dim=0)
                                  for a, b in zip(hs, h)]
    return {"strings": keep, "hidden": hs}


def run_depth(args, out: dict) -> None:
    t0 = time.time()
    from transformers import AutoTokenizer
    w_path = os.path.join(args.out_dir, "door9_w.pt")
    payload = torch.load(w_path, map_location="cpu", weights_only=False)
    W_b_to_a = payload["W_b_to_a"]           # (d_a, d_b) donor->host
    W_a_to_b = payload["W_a_to_b"]           # (d_b, d_a) host->donor
    b_b_to_a = payload["b_b_to_a"]
    b_a_to_b = payload["b_a_to_b"]
    d_a, d_b = payload["d_a"], payload["d_b"]

    tok_a = AutoTokenizer.from_pretrained(args.model_a)
    tok_b = AutoTokenizer.from_pretrained(args.model_b)
    tab_a = build_string_table(tok_a)
    tab_b = build_string_table(tok_b)
    common = sorted(set(tab_a) & set(tab_b))
    _, strs_held = _stable_split(common, args.held_frac)
    # deterministic subsample of the held split (out-of-sample vs the W fit)
    sample = strs_held[:args.depth_tokens]
    out["depth"] = {"n_sample": len(sample), "held_frac": args.held_frac,
                    "note": "sampled from held split (unseen by W fit)"}

    print(f"depth: capturing hidden states on {len(sample)} matched tokens",
          flush=True)
    dev = args.device
    model_a, _ = d0.load_donor(args.model_a, dev, args.dtype)
    ha = capture_hidden(model_a, tok_a, sample, dev)
    del model_a
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()

    model_b, _ = d0.load_donor(args.model_b, dev, args.dtype)
    hb = capture_hidden(model_b, tok_b, sample, dev)
    del model_b
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()

    # align rows by string (each capture drops strings that were not 1 token)
    sa, sb = ha["strings"], hb["strings"]
    s2ia = {s: i for i, s in enumerate(sa)}
    s2ib = {s: i for i, s in enumerate(sb)}
    common2 = [s for s in sample if s in s2ia and s in s2ib]
    ia = torch.tensor([s2ia[s] for s in common2])
    ib = torch.tensor([s2ib[s] for s in common2])
    n_pairs = len(common2)
    out["depth"]["n_pairs"] = n_pairs
    hid_a = [t[ia] for t in ha["hidden"]]      # list len n_a+1, each (n, d_a)
    hid_b = [t[ib] for t in hb["hidden"]]
    n_a = len(hid_a) - 1                       # layer index 0 = embeddings
    n_b = len(hid_b) - 1
    out["depth"]["n_layers_a"] = n_a
    out["depth"]["n_layers_b"] = n_b

    def mean_cos(p: torch.Tensor, q: torch.Tensor) -> float:
        return float(F.cosine_similarity(p.float(), q.float(), dim=-1).mean())

    # grid[j][i] = mean cos(W_a_to_b->? )  — measurement is host<-donor:
    # W_b_to_a maps donor states into host space: cos(W(h_B[j]), h_A[i])
    grid = [[0.0] * (n_a + 1) for _ in range(n_b + 1)]
    for j in range(n_b + 1):
        pred = hid_b[j].float() @ W_b_to_a.T + b_b_to_a
        for i in range(n_a + 1):
            grid[j][i] = mean_cos(pred, hid_a[i])

    diag = []
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        j = int(round(f * n_b))
        i = int(round(f * n_a))
        diag.append({"frac": f, "layer_b": j, "layer_a": i,
                     "cos": grid[j][i],
                     "cos_reverse_W": mean_cos(
                         hid_a[i].float() @ W_a_to_b.T + b_a_to_b,
                         hid_b[j])})
    best = []
    for j in range(n_b + 1):
        i_best = max(range(n_a + 1), key=lambda i: grid[j][i])
        best.append({"layer_b": j, "best_layer_a": i_best,
                     "cos_best": grid[j][i_best],
                     "cos_matched_frac": grid[j][int(round(
                         (j / max(1, n_b)) * n_a))]})

    # availability ceiling at depth: ridge fit on 80% of the sampled tokens,
    # BOTH directions, with the mean-target baseline + target rank-1 energy
    # (single-token hidden states are anisotropic — a near-constant target
    # makes any mean-predicting map score cos ~1; the baseline exposes that)
    import graft_d15_translator as d15
    avail = []
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n_pairs, generator=g)
    n_tr = max(8, int(0.8 * n_pairs))
    tr, he = perm[:n_tr], perm[n_tr:]
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        j = int(round(f * n_b))
        i = int(round(f * n_a))
        x, y = hid_b[j].float(), hid_a[i].float()   # x=donor b, y=host a
        rec = {"frac": f, "layer_b": j, "layer_a": i, "embW_cos": grid[j][i]}
        for tag, xs, ys, dd in (("b_to_a", x, y, (d_b, d_a)),
                                ("a_to_b", y, x, (d_a, d_b))):
            tr_map = d15.Translator("linear", dd[0], dd[1], ridge=1e-3)
            tr_map.fit_ridge(xs[tr], ys[tr])
            with torch.no_grad():
                p_tr = tr_map(xs[tr])
                p_he = tr_map(xs[he])
            mu = ys[tr].mean(0, keepdim=True)
            rec[tag] = {
                "fit_train_cos": mean_cos(p_tr, ys[tr]),
                "fit_held_cos": mean_cos(p_he, ys[he]),
                "mean_baseline_held": cos_stats(
                    mu.expand_as(ys[he]), ys[he])["mean"],
                "target_rank1_energy": float(
                    torch.linalg.svdvals(
                        ys[tr] - ys[tr].mean(0, keepdim=True))[0] ** 2
                    / torch.linalg.svdvals(
                        ys[tr] - ys[tr].mean(0, keepdim=True)).pow(2)
                    .sum().clamp_min(1e-12)),
            }
        avail.append(rec)

    # scale reference for the wrap init: mean ||h|| at host site and donor core
    # (hidden_states[i] = block-i output; residual scale is the wrap site's to
    # within the block's internal norms).  model_a = host (OLMoE), model_b =
    # donor (Phi); W_a_to_b is the wrap input chain (host -> donor).
    host_l = args.host_layer if args.host_layer >= 0 else 6
    donor_l = args.donor_layer
    i_site = min(host_l, n_a)     # host states are hid_a
    j_core = min(donor_l, n_b)    # donor states are hid_b
    host_site = hid_a[i_site].float()
    donor_core = hid_b[j_core].float()
    rank = args.d_canon
    U, S, Vh = torch.linalg.svd(W_a_to_b.float(), full_matrices=False)
    sq = S[:rank].clamp_min(0).sqrt()
    W64 = (U[:, :rank] * sq.unsqueeze(0)) @ (Vh[:rank] * sq.unsqueeze(1))
    u_on_site = host_site @ W64.T
    alpha = float(donor_core.norm(dim=-1).mean()
                  / u_on_site.norm(dim=-1).mean().clamp_min(1e-8))
    out["scale_depth"] = {
        "host_layer": i_site, "donor_layer": j_core,
        "mean_norm_host_site": float(host_site.norm(dim=-1).mean()),
        "mean_norm_donor_core": float(donor_core.norm(dim=-1).mean()),
        "mean_norm_w64_host_site": float(u_on_site.norm(dim=-1).mean()),
        "alpha_in": alpha,
        "top_svals": [float(x) for x in S[:8]],
        "sval_energy_rank64_frac": float(S[:rank].pow(2).sum()
                                         / S.pow(2).sum().clamp_min(1e-12)),
    }

    out["depth"].update({
        "grid_cos_Wb2a": grid, "diag_matched_frac": diag,
        "best_per_b_layer": best, "avail_fit_at_depth": avail,
    })
    out["wall_s"] = round(time.time() - t0, 2)
    out["phase"] = "depth_complete"
    dump_json(out, args.out)
    print(f"depth phase done in {out['wall_s']}s: diag "
          + " ".join(f"{d['frac']}:{d['cos']:.4f}" for d in diag)
          + f"  alpha_in {alpha:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Phase wrap: W-initialized door25 wrap vs random init (the payoff test)
# ---------------------------------------------------------------------------


def apply_w_init(wrap, W_in: torch.Tensor, alpha: float = 1.0,
                 alpha_out: float = 0.1, W_out: torch.Tensor | None = None,
                 which: str = "in") -> dict:
    """Install the SVD factorization of a basis map into the wrap's chain.

    W_in: (d_donor, d_model) host->donor — the geometrically correct map for
    host_in + adapter_up (rank d_canon factorization of W_in).
    W_out: (d_model, d_donor) donor->host — adapter_down + host_out (only for
    the `wboth` arm; `in` keeps door25's random adapter_down + zero host_out).
    `alpha` rescales the input chain so ||u|| matches the donor core's native
    residual scale (measured in the depth phase); `alpha_out` caps the birth
    delta for the `wboth` arm.
    """
    rank = wrap.d_canon
    info = {"which": which, "rank": rank, "alpha_in": alpha}
    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(W_in.float(), full_matrices=False)
        sq = S[:rank].clamp_min(0).sqrt()
        wrap.host_in.weight.copy_(Vh[:rank] * sq.unsqueeze(1))
        wrap.adapter_up.weight.copy_((U[:, :rank] * sq.unsqueeze(0)) * alpha)
        info["svals_top"] = [float(x) for x in S[:min(8, rank)]]
        info["sval_energy_rank_frac"] = float(
            S[:rank].pow(2).sum() / S.pow(2).sum().clamp_min(1e-12))
        if which == "wboth" and W_out is not None:
            U2, S2, Vh2 = torch.linalg.svd(W_out.float(), full_matrices=False)
            sq2 = S2[:rank].clamp_min(0).sqrt()
            wrap.adapter_down.weight.copy_(Vh2[:rank] * sq2.unsqueeze(1))
            wrap.host_out.weight.copy_(
                (U2[:, :rank] * sq2.unsqueeze(0)) * alpha_out)
            info["alpha_out"] = alpha_out
            info["svals_out_top"] = [float(x) for x in S2[:min(8, rank)]]
    return info


def train_rows_traced(pack, pairs, steps, lr, device, reject_w: float = 10.0,
                      mass_w: float = 2.0, owner_target: float = 0.35,
                      trace_every: int = 10) -> dict:
    """d1.train_rows with a p_new trajectory (recal-steps-needed read)."""
    report = {"steps": steps, "lr": lr, "reject_w": reject_w,
              "mass_w": mass_w, "owner_target": owner_target,
              "per_layer": [], "trace": []}
    for l, (bank, (h, lab)) in enumerate(zip(pack.banks65, pairs)):
        bank.P.requires_grad_(True)
        opt = torch.optim.Adam([bank.P], lr=lr)
        code = lab > 0.5
        base = ~code
        new_idx = bank.num_experts - 1
        first = last = None
        for step in range(steps):
            opt.zero_grad()
            logits = bank.scores(h)
            logp = F.log_softmax(logits, dim=-1)
            probs = logp.exp()
            loss = torch.zeros((), dtype=torch.float64)
            if code.any():
                loss = loss + F.nll_loss(
                    logp[code], torch.full((int(code.sum()),), new_idx,
                                           dtype=torch.long))
                p_code = probs[code, new_idx].mean()
                loss = loss + mass_w * (p_code - owner_target) ** 2
            if base.any():
                loss = loss + reject_w * probs[base, new_idx].mean()
            loss.backward()
            with torch.no_grad():
                if bank.P.grad is not None:
                    bank.P.grad[:new_idx] = 0.0
            opt.step()
            with torch.no_grad():
                p_new_code = float(probs[code, new_idx].mean()) \
                    if code.any() else 0.0
                p_new_base = float(probs[base, new_idx].mean()) \
                    if base.any() else 0.0
            if step == 0:
                first = {"loss": float(loss), "p_new_code": p_new_code,
                         "p_new_base": p_new_base}
            last = {"loss": float(loss), "p_new_code": p_new_code,
                    "p_new_base": p_new_base}
            if step % trace_every == 0 or step == steps - 1:
                report["trace"].append({"layer": l, "step": step,
                                        "p_new_code": p_new_code,
                                        "p_new_base": p_new_base})
        bank.P.requires_grad_(False)
        report["per_layer"].append({"layer": l, "first": first, "last": last})
    return report


def _finite_params(params) -> bool:
    return all(torch.isfinite(p.detach()).all() for p in params)


def ce_table(model, blocks, device, gates64, gates_plugged, handles,
             sets: dict) -> dict:
    """CE masked/plugged/forced for each named eval set (door25_heldout.py)."""
    tab = {}
    for name, batches in sets.items():
        if not batches:
            tab[name] = None
            continue
        ce_m = d1.ce_in_state(model, blocks, batches, device, gates64,
                              handles, "off")
        ce_p = d1.ce_in_state(model, blocks, batches, device, gates_plugged,
                              handles, "plugged")
        ce_f = d1.ce_in_state(model, blocks, batches, device, gates64,
                              handles, "forced")
        tab[name] = {"ce_masked": ce_m, "ce_plugged": ce_p, "ce_forced": ce_f,
                     "delta_plugged_vs_masked": ce_p - ce_m,
                     "delta_forced_vs_masked": ce_f - ce_m}
        print(f"  [{name}] masked {ce_m:.4f} plugged {ce_p:.4f} "
              f"forced {ce_f:.4f} (d_p {ce_p - ce_m:+.4f})", flush=True)
    return tab


def run_arm(args, out: dict) -> None:
    """Door25 single-site wrap + recal, with arm-selectable init + held-out."""
    t0 = time.time()
    device = args.device
    core = d25.load_core(args.core_from)
    w_gu, w_down, d_layer, d_expert = core[0], core[1], core[2], core[3]

    model, tok = d0.load_donor(args.model, device, args.dtype)
    blocks = d0.find_moe_blocks(model)
    banks_reg, reg_meta = d0.load_registry(args.registry)
    if len(banks_reg) != len(blocks):
        raise RuntimeError("registry/layer mismatch")
    n_host = len(blocks)
    if args.host_layer >= 0:
        host_i = args.host_layer
    else:
        n_dl = core[4] or (2 * n_host)
        host_i = int(round(d_layer * (n_host - 1) / max(1, (n_dl - 1))))
    host_i = max(0, min(n_host - 1, host_i))
    d_model = banks_reg[0].d_model
    wrap = d25.WrappedForeignExpert(d_model, args.d_canon, w_gu, w_down,
                                    seed=args.seed)
    out["host"] = {"model": args.model, "registry": args.registry,
                   "n_layers": n_host, "d_model": d_model,
                   "donor_layer": d_layer, "donor_expert": d_expert,
                   "host_layer": host_i}
    out["arm"] = args.arm

    # ---- arm init ----
    if args.arm in ("winit", "wboth"):
        w_path = args.w_init or os.path.join(args.out_dir, "door9_w.pt")
        payload = torch.load(w_path, map_location="cpu", weights_only=False)
        # wrap input chain = host -> donor = W_a_to_b (emb_A host -> emb_B donor)
        W_in = payload["W_a_to_b"]
        W_out = payload.get("W_b_to_a")
        alpha = 1.0
        depth_json = os.path.join(args.out_dir, args.depth_json)
        if args.alpha_in > 0:
            alpha = args.alpha_in          # explicit override (quiet-birth arm)
        elif os.path.exists(depth_json):
            with open(depth_json) as f:
                dj = json.load(f)
            alpha = float(dj.get("scale_depth", {}).get("alpha_in", 1.0))
        init_info = apply_w_init(
            wrap, W_in, alpha=alpha,
            alpha_out=args.alpha_out,
            W_out=W_out if args.arm == "wboth" else None,
            which="in" if args.arm == "winit" else "wboth")
        out["init"] = init_info
        print(f"arm {args.arm}: W-init input chain alpha {alpha:.4f}", flush=True)
    else:
        out["init"] = {"which": "rand", "seed": args.seed,
                       "note": "door25 verbatim N(0,0.02) + zero host_out"}
    out["wrap"] = wrap.wrap_spec()

    # ---- data (door25 split convention + door25_heldout.py eval sets) ----
    domains = d0.load_domain_texts(args.domains_dir)
    code_ids = tok.encode(domains[args.home_domain], add_special_tokens=False)
    split = int(0.6 * len(code_ids))
    code_train_ids = code_ids[:split][:args.max_tokens]
    code_probe_ids = code_ids[split:]

    def to_batches(ids, label, seq, bs):
        wins = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
        return [(torch.tensor(wins[i:i + bs], dtype=torch.long), label)
                for i in range(0, len(wins) - bs + 1, bs)]

    code_train = to_batches(code_train_ids, 1, args.seq_len, args.batch_size)
    base_ids = []
    for name, text in domains.items():
        if name == args.home_domain:
            continue
        base_ids.extend(tok.encode(text, add_special_tokens=False))
    base_ids = base_ids[:args.max_tokens * 3]
    base_batches = to_batches(base_ids, 0, args.seq_len, args.batch_size)
    probe_windows = [torch.tensor(code_probe_ids[i:i + 256], dtype=torch.long)
                     for i in range(0, len(code_probe_ids) - 256, 256)]
    eval_sets = {
        "code_train": to_batches(code_train_ids, 1, args.seq_len,
                                 args.eval_batch),
        "code_heldout": to_batches(code_probe_ids, 1, args.seq_len,
                                   args.eval_batch),
        "math_cross": to_batches(
            tok.encode(domains.get("math", ""), add_special_tokens=False)
            [:args.max_tokens], 1, args.seq_len, args.eval_batch),
    }
    if not code_train or not base_batches or not probe_windows:
        raise RuntimeError("not enough data")
    out["data"] = {"code_train_tok": int(sum(t.numel() for t, _ in code_train)),
                   "base_tok": int(sum(t.numel() for t, _ in base_batches)),
                   "n_probes_win": len(probe_windows),
                   "eval_tok": {k: int(sum(t.numel() for t, _ in v))
                                for k, v in eval_sets.items()}}
    print(f"data: train {out['data']['code_train_tok']} tok, "
          f"eval {out['data']['eval_tok']}", flush=True)

    # ---- site pack + install ----
    pack, gates64, gates_plugged, top_k, norm_topk = d25.build_site_pack(
        blocks, banks_reg, host_i, wrap)
    pack.experts.to(device)
    handles = d1.install_plugin([blocks[host_i]], pack, "off")
    new_idx = pack.banks65[0].num_experts - 1
    out["new_expert_idx"] = new_idx

    # ---- Phase A: adapter fit (door25 recipe; NaN-guarded) ----
    print("PHASE A — adapter fit (forced dispatch, code CE + out_pen L2)")
    lr = args.expert_lr
    pa = d25.train_adapters(
        model, pack, handles, code_train, base_batches,
        args.expert_steps, lr, device, out_pen=args.out_pen,
        grad_clip=args.grad_clip, weight_decay=args.weight_decay)
    nan_a = (not _finite_params(pack.parameters())
             or not pa["history"]
             or not math.isfinite(pa["history"][-1]["ce_code"]))
    if nan_a:
        print(f"NaN guard: phase A poisoned at lr {lr}; fresh reinit + lr/2",
              flush=True)
        out["nan_fallback"] = {"phase": "A", "lr_first": lr}
        if args.arm in ("winit", "wboth"):
            payload = torch.load(
                args.w_init or os.path.join(args.out_dir, "door9_w.pt"),
                map_location="cpu", weights_only=False)
            alpha = out["init"].get("alpha_in", 1.0)
            apply_w_init(wrap, payload["W_a_to_b"], alpha=alpha,
                         alpha_out=args.alpha_out,
                         W_out=payload.get("W_b_to_a")
                         if args.arm == "wboth" else None,
                         which=out["init"]["which"])
        else:
            g = torch.Generator().manual_seed(args.seed + 1)
            with torch.no_grad():
                wrap.host_in.weight.copy_(
                    torch.randn(wrap.d_canon, wrap.d_model, generator=g) * 0.02)
                wrap.adapter_up.weight.copy_(
                    torch.randn(wrap.d_donor, wrap.d_canon, generator=g) * 0.02)
                wrap.adapter_down.weight.copy_(
                    torch.randn(wrap.d_canon, wrap.d_donor, generator=g) * 0.02)
                wrap.host_out.weight.zero_()
        lr = lr / 2.0
        pa = d25.train_adapters(
            model, pack, handles, code_train, base_batches,
            args.expert_steps, lr, device, out_pen=args.out_pen,
            grad_clip=args.grad_clip, weight_decay=args.weight_decay)
        out["nan_fallback"]["lr_second"] = lr
        out["nan_fallback"]["recovered"] = bool(
            _finite_params(pack.parameters()) and pa["history"]
            and math.isfinite(pa["history"][-1]["ce_code"]))
    out["phase_a"] = pa
    out["adapter_params"] = wrap.adapter_param_count()
    dump_json(out, args.out)

    # ---- Phase B: passport row recal (traced) ----
    print("PHASE B — passport row recal")
    pairs = d1.run_in_state(
        model, blocks, gates64, handles, "off",
        lambda: d1.collect_gate_pairs(
            model, [blocks[host_i]], code_train + base_batches, device,
            args.max_cal_pairs,
            lambda i, b: b[1] if isinstance(b[1], (int, float)) else 0))
    with torch.no_grad():
        h, lab = pairs[0]
        code_h = h[lab > 0.5]
        if code_h.shape[0] > 0:
            pack.banks65[0].P.data[new_idx] = (
                code_h.mean(dim=0) * math.sqrt(pack.banks65[0].d_r))
    pb1 = train_rows_traced(pack, pairs, args.row_steps, args.row_lr, device,
                            reject_w=args.reject_w, mass_w=args.mass_w,
                            owner_target=args.owner_mass)
    pairs2 = d1.run_in_state(
        model, blocks, gates_plugged, handles, "plugged",
        lambda: d1.collect_gate_pairs(
            model, [blocks[host_i]], code_train + base_batches, device,
            args.max_cal_pairs,
            lambda i, b: b[1] if isinstance(b[1], (int, float)) else 0))
    pb2 = train_rows_traced(pack, pairs2, max(50, args.row_steps // 2),
                            args.row_lr, device, reject_w=args.reject_w,
                            mass_w=args.mass_w, owner_target=args.owner_mass)
    out["phase_b"] = {"round1_teacher_forced": pb1,
                      "round2_on_trajectory": pb2}
    # recal-steps-needed: first trace step with p_new_code >= 0.8
    steps_needed = None
    for rec in pb1["trace"] + pb2["trace"]:
        if rec["p_new_code"] >= 0.8:
            steps_needed = rec["step"] if steps_needed is None else steps_needed
            break
    out["recal_steps_to_p80"] = steps_needed
    out["recal_final"] = {
        "r1": pb1["per_layer"][0]["last"] if pb1["per_layer"] else None,
        "r2": pb2["per_layer"][0]["last"] if pb2["per_layer"] else None}
    dump_json(out, args.out)

    # ---- Phase C: gates + quality (train + held-out + cross) ----
    print("PHASE C — measurement")
    base_parity = d25.parity_vs_ref(
        model, blocks, base_batches, device, handles,
        None, "off", gates_plugged, "plugged")
    out["base_parity"] = base_parity
    out["graft_parity_vs_olmoe_graft"] = d25.parity_vs_ref(
        model, blocks, base_batches, device, handles,
        gates64, "off", gates_plugged, "plugged")

    fn_pre = d1.make_logits_fn(model, blocks, gates64, handles, "off", device)
    fn_solo = d1.make_logits_fn(model, blocks, gates64, handles, "forced",
                                device, force_gain=1.0)
    fn_lib = d1.make_logits_fn(model, blocks, gates_plugged, handles,
                               "plugged", device)
    probes = d1.select_code_probes(fn_pre, probe_windows, args.n_probes)
    m_pre = d1.probe_margins(probes, fn_pre)
    m_solo = d1.probe_margins(probes, fn_solo)
    m_lib = d1.probe_margins(probes, fn_lib)
    gain_solo = m_solo["mean_logit_margin"] - m_pre["mean_logit_margin"]
    gain_lib = m_lib["mean_logit_margin"] - m_pre["mean_logit_margin"]
    out["margins"] = {"pre": m_pre, "solo": m_solo, "lib": m_lib,
                      "gain_solo": round(gain_solo, 4),
                      "gain_lib": round(gain_lib, 4),
                      "retention_vs_solo": d1.retention(gain_lib, gain_solo)}

    out["contamination_base"] = d1.run_in_state(
        model, blocks, gates_plugged, handles, "plugged",
        lambda: d1.routing_hits(model, [blocks[host_i]], base_batches,
                                device, new_idx))
    out["utilization_code"] = d1.run_in_state(
        model, blocks, gates_plugged, handles, "plugged",
        lambda: d1.routing_hits(model, [blocks[host_i]], list(code_train),
                                device, new_idx))
    out["routing_mass"] = {
        "base": d1.run_in_state(
            model, blocks, gates_plugged, handles, "plugged",
            lambda: d1.routing_mass(model, [blocks[host_i]], base_batches,
                                    device, new_idx)),
        "code": d1.run_in_state(
            model, blocks, gates_plugged, handles, "plugged",
            lambda: d1.routing_mass(model, [blocks[host_i]], list(code_train),
                                    device, new_idx))}

    out["quality"] = ce_table(model, blocks, device, gates64, gates_plugged,
                              handles, eval_sets)
    q = out["quality"]["code_train"]
    out["s6_quality"] = {
        "ce_code_masked": q["ce_masked"],
        "ce_code_active": q["ce_plugged"],
        "ce_code_forced_solo": q["ce_forced"],
        "delta_plugged_vs_masked": q["delta_plugged_vs_masked"],
        "delta_forced_vs_masked": q["delta_forced_vs_masked"]}
    held = out["quality"]["code_heldout"] or {}
    out["memorization_gap"] = {
        "train_delta_plugged_vs_masked": q["delta_plugged_vs_masked"],
        "held_delta_plugged_vs_masked":
            held.get("delta_plugged_vs_masked"),
        "held_minus_train_delta": (
            held.get("delta_plugged_vs_masked", 0.0)
            - q["delta_plugged_vs_masked"]
            if held else None),
        "ref_door25_train_delta": 0.439 - 1.072,
        "ref_door25_held_delta": 1.621 - 1.587}
    out["verdict_vs_door25"] = {
        "ref_ce_masked": REF_DOOR25["ce_code_masked"],
        "ref_ce_plugged": REF_DOOR25["ce_code_active"],
        "ce_masked": q["ce_masked"],
        "ce_plugged": q["ce_plugged"],
        "plugged_beats_masked": bool(q["delta_plugged_vs_masked"] < 0),
        "beats_ref_plugged": bool(q["ce_plugged"]
                                  < REF_DOOR25["ce_code_active"])}
    out["verdict"] = d1.d1_verdict(base_parity, out["margins"]["retention_vs_solo"],
                                   out["contamination_base"],
                                   out["utilization_code"],
                                   q["ce_plugged"], q["ce_masked"])
    out["wall_s"] = round(time.time() - t0, 2)
    out["phase"] = "wrap_complete"
    if args.plugin_out:
        torch.save({
            "format": FORMAT, "arm": args.arm,
            "wrap": wrap.state_dict(), "wrap_spec": wrap.wrap_spec(),
            "row": pack.banks65[0].P.data[new_idx].cpu(),
            "init": out["init"],
            "meta": {"new_idx": new_idx, "host_layer": host_i,
                     "home_domain": args.home_domain,
                     "host_registry": args.registry}}, args.plugin_out)
        out["plugin_path"] = args.plugin_out
    dump_json(out, args.out)
    print(f"arm {args.arm} done in {out['wall_s']}s: "
          f"train masked {q['ce_masked']:.4f} plugged {q['ce_plugged']:.4f} "
          f"forced {q['ce_forced']:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Phase emb-sens: why is the fit where it is?  (lambda / Procrustes / floors)
# ---------------------------------------------------------------------------


def build_loose_table(tok, max_rows: int | None = None) -> dict:
    """string -> id WITHOUT the single-token round-trip requirement.

    First id wins per decoded string; specials/empty excluded.  Sensitivity
    only — the gated protocol uses build_string_table.
    """
    n = len(tok)
    if max_rows:
        n = min(n, max_rows)
    special = set(tok.all_special_ids) if hasattr(tok, "all_special_ids") else set()
    table = {}
    for i in range(n):
        if i in special:
            continue
        try:
            s = tok.decode([i])
        except Exception:
            continue
        if not s or s in table:
            continue
        table[s] = i
    return table


def eval_map(x_tr, y_tr, x_he, y_he, ridge: float, do_procrustes: bool = False):
    d_src, d_tgt = x_tr.shape[1], y_tr.shape[1]
    res = {}
    if do_procrustes:
        mx = x_tr.mean(0, keepdim=True)
        my = y_tr.mean(0, keepdim=True)
        M = (y_tr - my).T @ (x_tr - mx)
        U, _, Vh = torch.linalg.svd(M, full_matrices=False)
        W = U @ Vh
        pred_tr = (x_tr - mx) @ W.T + my
        pred_he = (x_he - mx) @ W.T + my
        res["procrustes"] = {"train": cos_stats(pred_tr, y_tr),
                             "held": cos_stats(pred_he, y_he)}
    import graft_d15_translator as d15
    lin = d15.Translator("linear", d_src, d_tgt, ridge=ridge)
    lin.fit_ridge(x_tr, y_tr)
    with torch.no_grad():
        pred_tr = lin(x_tr)
        pred_he = lin(x_he)
    res["ridge"] = {"ridge": ridge, "train": cos_stats(pred_tr, y_tr),
                    "held": cos_stats(pred_he, y_he)}
    return res


def run_emb_sens(args, out: dict) -> None:
    t0 = time.time()
    from transformers import AutoTokenizer
    tok_a = AutoTokenizer.from_pretrained(args.model_a)
    tok_b = AutoTokenizer.from_pretrained(args.model_b)
    emb_a = load_embed_weight(args.model_a)
    emb_b = load_embed_weight(args.model_b)
    out["sens"] = {}

    for name, lab_a, lab_b in (("strict", build_string_table(tok_a),
                                build_string_table(tok_b)),
                               ("loose", build_loose_table(tok_a),
                                build_loose_table(tok_b))):
        common = sorted(set(lab_a) & set(lab_b))
        strs_tr, strs_he = _stable_split(common, args.held_frac)
        s2i = {s: i for i, s in enumerate(common)}
        it = torch.tensor([s2i[s] for s in strs_tr])
        ih = torch.tensor([s2i[s] for s in strs_he])
        pa = torch.tensor([lab_a[s] for s in common])
        pb = torch.tensor([lab_b[s] for s in common])
        x_all, y_all = emb_b[pb], emb_a[pa]
        x_tr, y_tr = x_all[it], y_all[it]
        x_he, y_he = x_all[ih], y_all[ih]
        entry = {"n_intersect": len(common), "n_train": len(strs_tr),
                 "n_held": len(strs_he),
                 "baseline_mean_train": cos_stats(
                     y_tr.mean(0, keepdim=True).expand_as(y_tr), y_tr),
                 "baseline_mean_held": cos_stats(
                     y_tr.mean(0, keepdim=True).expand_as(y_he), y_he)}
        # floor: fitted map's prediction vs WRONG held targets
        import graft_d15_translator as d15
        lin = d15.Translator("linear", x_tr.shape[1], y_tr.shape[1],
                             ridge=args.ridge)
        lin.fit_ridge(x_tr, y_tr)
        with torch.no_grad():
            pred_he = lin(x_he)
        g = torch.Generator().manual_seed(0)
        shuf = y_he[torch.randperm(y_he.shape[0], generator=g)]
        entry["floor_shuffled_targets_held"] = cos_stats(pred_he, shuf)
        entry["ridge_sweep"] = {}
        for r in (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
            rres = eval_map(x_tr, y_tr, x_he, y_he, r)
            entry["ridge_sweep"][str(r)] = rres["ridge"]
            print(f"  [{name} ridge {r}] train "
                  f"{rres['ridge']['train']['mean']:.4f} held "
                  f"{rres['ridge']['held']['mean']:.4f}", flush=True)
        pres = eval_map(x_tr, y_tr, x_he, y_he, args.ridge,
                        do_procrustes=True)
        entry["procrustes"] = pres["procrustes"]
        print(f"  [{name} procrustes] train "
              f"{pres['procrustes']['train']['mean']:.4f} held "
              f"{pres['procrustes']['held']['mean']:.4f}", flush=True)
        # also the wrap-input direction (a -> b) at the default lambda
        rres_inv = eval_map(y_tr, x_tr, y_he, x_he, args.ridge)
        entry["ridge_a_to_b_default"] = rres_inv["ridge"]
        print(f"  [{name} a->b ridge {args.ridge}] train "
              f"{rres_inv['ridge']['train']['mean']:.4f} held "
              f"{rres_inv['ridge']['held']['mean']:.4f}", flush=True)
        out["sens"][name] = entry
        dump_json(out, args.out)

    out["wall_s"] = round(time.time() - t0, 2)
    out["phase"] = "emb_sens_complete"
    dump_json(out, args.out)
    print(f"emb-sens done in {out['wall_s']}s", flush=True)


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------


class _FakeTok:
    """Round-trip toy tokenizer: words + padded junk ids."""

    def __init__(self, words):
        self.words = list(words)
        self.all_special_ids = [0]

    def __len__(self):
        return len(self.words) + 5

    def encode(self, s, add_special_tokens=False):
        if s in self.words:
            return [self.words.index(s) + 1]
        return [1, 2]

    def decode(self, ids):
        i = ids[0]
        if i == 0:
            return ""
        if i - 1 < len(self.words):
            return self.words[i - 1]
        return "\ufffd"


def run_synthetic_tests() -> int:
    torch.manual_seed(0)
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" — {detail}" if detail else ""))

    # 1. string-table pairing filter
    tok = _FakeTok(["cat", "dog", "fish", "hi"])
    tab = build_string_table(tok, max_rows=10)
    check("string_table_roundtrip_filter",
          set(tab) == {"cat", "dog", "fish", "hi"},
          f"got {sorted(tab)}")
    common = sorted(set(tab) & set(build_string_table(tok)))
    check("string_table_intersection", len(common) == 4, f"n={len(common)}")

    # 2. SVD factorization of W into host_in/adapter_up is exact at full rank
    d_model, d_donor, rank = 24, 32, 8
    W = torch.randn(d_donor, d_model)
    w_gu = torch.randn(2 * 6, d_donor) * 0.05
    w_down = torch.randn(d_donor, 6) * 0.05
    wrap = d25.WrappedForeignExpert(d_model, rank, w_gu, w_down, seed=0)
    info = apply_w_init(wrap, W, alpha=1.0)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    W64 = (U[:, :rank] * S[:rank].sqrt().unsqueeze(0)) @ \
          (Vh[:rank] * S[:rank].sqrt().unsqueeze(1))
    x = torch.randn(50, d_model)
    u_wrap = wrap.adapter_up(wrap.host_in(x))
    u_ref = x @ W64.T
    check("svd_factorization_exact",
          torch.allclose(u_wrap, u_ref, atol=1e-4),
          f"max diff {float((u_wrap - u_ref).abs().max()):.2e}")
    check("svd_energy_frac_recorded", 0.0 < info["sval_energy_rank_frac"] <= 1.0,
          f"{info['sval_energy_rank_frac']:.3f}")

    # 3. alpha scaling
    apply_w_init(wrap, W, alpha=3.0)
    u3 = wrap.adapter_up(wrap.host_in(x))
    check("alpha_scaling", torch.allclose(u3, 3.0 * u_ref, atol=1e-4),
          f"ratio {float(u3.norm() / u_ref.norm()):.3f}")

    # 4. ridge recovers a linear map on clean pairs
    X = torch.randn(2000, 16)
    W0 = torch.randn(24, 16)
    Y = X @ W0.T + 0.01 * torch.randn(2000, 24)
    fit = fit_maps(X[:1800], Y[:1800], X[1800:], Y[1800:], ridge=1e-4,
                   mlp_steps=50)
    check("ridge_recovers_linear",
          fit["linear"]["held"]["mean"] > 0.99,
          f"held cos {fit['linear']['held']['mean']:.4f}")
    check("mean_baseline_reported", "held" in fit["baseline_mean"])

    # 5. trace wrapper matches d1.train_rows semantics on the mock MoE
    model, n_dom = d0.build_mock_donor()
    blocks = d0.find_moe_blocks(model)
    model.eval()
    e, h = blocks[0].gate.weight.shape
    banks_reg = [d0.PassportBank(e, h) for _ in blocks]
    d0.init_passports_exact(banks_reg, [b.gate.weight for b in blocks])
    code_train = [(t, 1) for t, d in d0.mock_batches(n_dom, seed=11)
                  if d == 1]
    base_b = [(t, 0) for t, d in d0.mock_batches(n_dom, seed=17) if d != 1]
    wrap2 = d25.WrappedForeignExpert(h, 8, w_gu, w_down, seed=1)
    pack, gates64, gates_plugged, _, _ = d25.build_site_pack(
        blocks, banks_reg, 0, wrap2)
    handles = d1.install_plugin([blocks[0]], pack, "off")
    pairs = d1.run_in_state(
        model, blocks, gates64, handles, "off",
        lambda: d1.collect_gate_pairs(
            model, [blocks[0]], code_train + base_b, "cpu", 256,
            lambda i, b: b[1] if isinstance(b[1], (int, float)) else 0))
    rep = train_rows_traced(pack, pairs, 30, 0.1, "cpu")
    check("train_rows_traced",
          rep["trace"] and rep["per_layer"][0]["last"] is not None,
          f"trace {len(rep['trace'])} pts, last "
          f"{rep['per_layer'][0]['last']}")
    d1.uninstall_plugin(handles)

    # 6. run_arm end-to-end on the mock (winit path incl. W plumbing)
    class A:
        pass
    a = A()
    a.arm = "winit"
    a.model = a.registry = None
    a.core_from = None
    a.device = "cpu"
    a.dtype = "float32"
    a.d_canon = 8
    a.host_layer = 0
    a.seed = 0
    a.w_init = None
    a.out_dir = "/tmp"
    a.depth_json = "no_such_depth.json"
    a.alpha_in = 1.0
    a.alpha_out = 0.1
    a.home_domain = "code"
    a.domains_dir = None
    a.max_tokens = 4000
    a.seq_len = 16
    a.batch_size = 2
    a.eval_batch = 2
    a.expert_steps = 20
    a.expert_lr = 1e-2
    a.out_pen = 5.0
    a.grad_clip = 1.0
    a.weight_decay = 0.0
    a.row_steps = 40
    a.row_lr = 0.1
    a.reject_w = 10.0
    a.mass_w = 2.0
    a.owner_mass = 0.35
    a.max_cal_pairs = 512
    a.n_probes = 8
    a.plugin_out = None
    a.out = None

    # monkeypatch the heavy tail: run_arm needs model+registry; emulate via
    # the same mock path door25 uses — skip full run_arm (covered by door25
    # 15/15 + the real run).  Check only that apply_w_init feeds run_arm's
    # init block shape-wise.
    wrap3 = d25.WrappedForeignExpert(h, 8, w_gu, w_down, seed=2)
    # geometry: W_in is (d_donor, d_model) = (w_gu.shape[1], h)
    Win = torch.randn(w_gu.shape[1], h)
    inf = apply_w_init(wrap3, Win, alpha=2.0, which="in")
    check("winit_shapes", wrap3.host_in.weight.shape == (8, h)
          and wrap3.adapter_up.weight.shape == (w_gu.shape[1], 8)
          and float(wrap3.host_out.weight.abs().sum()) == 0.0,
          f"alpha {inf['alpha_in']}")
    check("finite_params", _finite_params(wrap3.parameters()))

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} synthetic checks passed")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--phase", default="emb",
                    choices=["emb", "emb-sens", "depth", "wrap", "all"])
    ap.add_argument("--arm", default="rand", choices=["rand", "winit", "wboth"])
    ap.add_argument("--model", default=None, help="host (OLMoE)")
    ap.add_argument("--model-a", default=None,
                    help="host for emb/depth (OLMoE); emb target")
    ap.add_argument("--model-b", default=None,
                    help="donor for emb/depth (Phi-mini-MoE); emb source")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--core-from", default=None)
    ap.add_argument("--domains-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--depth-json", default="door9_depth.json")
    ap.add_argument("--w-init", default=None)
    ap.add_argument("--alpha-in", type=float, default=-1.0)
    ap.add_argument("--alpha-out", type=float, default=0.1)
    ap.add_argument("--held-frac", type=float, default=0.1)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--mlp-steps", type=int, default=1200)
    ap.add_argument("--depth-tokens", type=int, default=512)
    ap.add_argument("--host-layer", type=int, default=6)
    ap.add_argument("--donor-layer", type=int, default=13)
    # door25 recipe knobs (defaults = the verified run)
    ap.add_argument("--home-domain", default="code")
    ap.add_argument("--d-canon", type=int, default=64)
    ap.add_argument("--home-expert", type=int, default=-1)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--expert-steps", type=int, default=250)
    ap.add_argument("--expert-lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--out-pen", type=float, default=300.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--row-steps", type=int, default=400)
    ap.add_argument("--row-lr", type=float, default=0.05)
    ap.add_argument("--reject-w", type=float, default=10.0)
    ap.add_argument("--mass-w", type=float, default=2.0)
    ap.add_argument("--owner-mass", type=float, default=0.35)
    ap.add_argument("--max-cal-pairs", type=int, default=4096)
    ap.add_argument("--n-probes", type=int, default=32)
    ap.add_argument("--plugin-out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.synthetic:
        return run_synthetic_tests()

    if args.out_dir is None:
        args.out_dir = os.path.join(REPO_ROOT, "lab", "imported",
                                    "graft_door9")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.model_a is None:
        args.model_a = args.model
    if args.model_b is None:
        args.model_b = "/data/tliao/Phi-mini-MoE-instruct"

    out = {"gate": "D9-embedding-basis-translation", "format": FORMAT,
           "phase": "start", "arm": args.arm, "args": {
               k: v for k, v in sorted(vars(args).items())
               if isinstance(v, (int, float, str, bool)) or v is None}}
    if args.out is None:
        args.out = os.path.join(args.out_dir, f"door9_{args.phase}"
                               + (f"_{args.arm}" if args.phase == "wrap"
                                  else "") + ".json")
    if args.phase in ("wrap", "all") and args.plugin_out is None:
        args.plugin_out = os.path.join(args.out_dir, f"door9_{args.arm}_plugin.pt")
    dump_json(out, args.out)

    if args.phase in ("emb", "all"):
        if not args.model_a or not args.model_b:
            ap.error("--model-a/--model-b (or --model) required for emb")
        run_emb(args, out)
    if args.phase == "emb-sens":
        if not args.model_a or not args.model_b:
            ap.error("--model-a/--model-b required for emb-sens")
        run_emb_sens(args, out)
    if args.phase in ("depth", "all"):
        if not args.model_a or not args.model_b:
            ap.error("--model-a/--model-b required for depth")
        out.setdefault("gate", FORMAT)
        run_depth(args, out)
    if args.phase in ("wrap", "all"):
        if not args.model or not args.registry or not args.core_from:
            ap.error("--model/--registry/--core-from required for wrap")
        if not args.domains_dir:
            ap.error("--domains-dir required for wrap")
        run_arm(args, out)
    print(f"results -> {args.out}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
