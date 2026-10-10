"""Occupancy-dilution check (signal run, not a gate) — is the base-CE
residual a tiny-pool artifact?

The shaped N=8 library regresses base CE +8.91% on the lab spine, where 8
plug-in rows share the top-2 contest with only 4 base rows (plug pool share
8/12 = 67%).  If the residual is slot-arithmetic (plug rows displacing base
experts on base tokens), it should shrink as the base pool grows at fixed
N=8 plug-ins.  This script isolates that arithmetic on the SAME trained
state: clone the 4 trained base experts (weights copied, passport rows
cloned + tiny noise to break ties) to make base pools of 4/8/16/32 rows,
keep the 8 memory rows fixed, and re-measure base CE regression /
p_any_plug / margin retention at each pool size.

Cloned experts are the point, not a confound: they keep base-expert output
QUALITY constant while only changing pool SHARE.  A companion real-spine
run (t0-8expert, 8 trained base experts, full shaped pipeline) checks the
same direction with independently trained experts.

Pass bar is not applied — report the curve.  Dilution hypothesis HOLDS if
base-CE regression falls monotonically with base-pool size at roughly
constant retention.

Env: G3D_STATE (lab/sandbox/g3_shaped/n8_shaped_state.pt),
     G3D_SOLO_JSON (lab/results/g3_shaped_n8.json),
     G3D_OUT (lab/results/g3_dilution_clone.json),
     G3D_BASE_SIZES (4,8,16,32), G3D_NOISE (1e-2), G3D_DEVICE, G3D_SEED (0).
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "lab"))
sys.path.insert(0, os.path.join(REPO, "Hippocampus"))
sys.path.insert(0, os.path.join(REPO, "Common"))

import g1_export_mode as g1  # noqa: E402
import g3_coexistence as g3  # noqa: E402
import memory_shaping as ms  # noqa: E402
import g3_family_ab as fam  # noqa: E402
from consolidate import load_base_model  # noqa: E402

CKPT = os.path.join(
    REPO, "lab", "sandbox", "t1-diag-realdata-mb4",
    "vesper_linear_checkpoints_lab_small", "step_best")
DEVICE = os.environ.get("G3D_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
fam.DEVICE = DEVICE
restore_library = fam.restore_library


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


STATE = _abs(os.environ.get("G3D_STATE") or
             os.path.join("lab", "sandbox", "g3_shaped", "n8_shaped_state.pt"))
SOLO_JSON = _abs(os.environ.get("G3D_SOLO_JSON") or
                 os.path.join("lab", "results", "g3_shaped_n8.json"))
OUT = _abs(os.environ.get("G3D_OUT") or
           os.path.join("lab", "results", "g3_dilution_clone.json"))
BASE_SIZES = [int(x) for x in
              (os.environ.get("G3D_BASE_SIZES") or "4,8,16,32").split(",")]
NOISE = float(os.environ.get("G3D_NOISE", "1e-2"))
SEED = 0


def expand_base_pool(model, k_target, n0, noise):
    """Grow the base pool from n0 to k_target by cloning base experts.

    Clone t takes expert/row (t % n0); rows get gaussian noise (scale
    `noise`) so near-duplicates don't tie in topk.  Plug rows keep their
    content but shift to indices [k_target, k_target+n_plug)."""
    for layer in model.layers:
        ffn = layer["ffn"]
        r = ffn.router
        rows = r.passports.data
        base_rows = rows[:n0].clone()
        plug_rows = rows[n0:].clone()
        n_plug = plug_rows.shape[0]
        experts = list(ffn.experts)
        new_experts = [experts[i] for i in range(n0)]
        new_rows = [base_rows[i] for i in range(n0)]
        for t in range(k_target - n0):
            src = t % n0
            new_experts.append(copy.deepcopy(experts[src]))
            new_rows.append(base_rows[src]
                            + noise * torch.randn_like(base_rows[src]))
        new_experts.extend(experts[n0:])
        for j in range(n_plug):
            new_rows.append(plug_rows[j])
        ffn.experts = nn.ModuleList(new_experts)
        r.passports = nn.Parameter(torch.stack(new_rows).to(
            device=rows.device, dtype=rows.dtype))
        r.num_experts = k_target + n_plug
        ffn.num_experts = k_target + n_plug
    return k_target


def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"dilution-clone  state={STATE}  base_sizes={BASE_SIZES}  "
          f"noise={NOISE}  device={DEVICE}")

    refj = json.load(open(SOLO_JSON))
    solo = {int(k): v for k, v in refj.get("solo", {}).items()}
    pre_ref = {int(k): v for k, v in refj.get("pre_library", {}).items()}
    ce_pre = float(refj.get("summary", {}).get("base_ce", {}).get("pre", 0)
                   or 6.7735)

    curve = []
    for k_base in BASE_SIZES:
        torch.manual_seed(SEED)
        model, tok, mc = load_base_model(CKPT, device=DEVICE)
        g1.DIM = int(mc["dim"]); g1.HIDDEN = int(mc["hidden_dim"])
        g1.N_LAYERS = int(mc["n_layers"])
        g1.BASE_EXPERTS = int(mc["num_experts"])
        g1.prepare_routers(model)
        sd = torch.load(STATE, map_location="cpu", weights_only=False)
        n_mem = restore_library(model, sd["state_dict"])
        n0 = g1.BASE_EXPERTS
        assert k_base >= n0, f"base size {k_base} < trained {n0}"
        torch.manual_seed(SEED + k_base)   # clone-noise stream per pool size
        expand_base_pool(model, k_base, n0, NOISE)
        g1.BASE_EXPERTS = k_base
        model.eval()
        PAD_ID = tok.pad_token_id
        names = sd.get("episodes", [e.name for e in ms.SHAPED_8])
        episodes = [e for e in ms.SHAPED_8 if e.name in names]
        plug_rows = list(range(k_base, k_base + n_mem))
        BASE_MIX = g1.load_base_mix_chunks(n_chunks=32, seq=256, seed=SEED)

        ce = g1.next_token_ce(model, BASE_MIX)
        reg = 100.0 * (ce - ce_pre) / ce_pre
        _, diag = g3.routing_matrix(model, [[c] for c in BASE_MIX[:8]],
                                    plug_rows, PAD_ID)
        home_lists = [ms.episode_home_ids(tok, ep, DEVICE) for ep in episodes]
        per_group = g3.routing_matrix_group(model, home_lists, plug_rows,
                                            PAD_ID)
        marg, ret = {}, {}
        for j, ep in enumerate(episodes):
            m = ms.episode_margins(model, tok, ep, DEVICE)
            marg[ep.name] = m["mean_logit_margin"]
            s = solo.get(j)
            if s and s["gain_logit"]:
                g_lib = marg[ep.name] - pre_ref[j]["mean_logit_margin"]
                ret[ep.name] = round(g_lib / s["gain_logit"], 4)
        vals = list(ret.values())
        mean_ret = sum(vals) / len(vals) if vals else None
        util = {episodes[j].name: round(per_group[j][k_base + j], 4)
                for j in range(len(episodes))}

        # control: same cloned pool WITHOUT the plug rows — separates the
        # plug-induced residual from clone-pool diversity drift
        for _ in range(n_mem):
            g1.drop_last_expert(model)
        g1.BASE_EXPERTS = k_base
        ce_clone_only = g1.next_token_ce(model, BASE_MIX)
        clone_drift = 100.0 * (ce_clone_only - ce_pre) / ce_pre
        residual = ce - ce_clone_only
        residual_pct = 100.0 * residual / ce_clone_only

        row = {"base_pool": k_base,
               "n_plug": n_mem,
               "pool_rows": k_base + n_mem,
               "plug_pool_share": round(n_mem / (k_base + n_mem), 4),
               "ce_lib": round(ce, 4),
               "base_ce_regression_pct": round(reg, 3),
               "ce_clone_only": round(ce_clone_only, 4),
               "clone_drift_pct": round(clone_drift, 3),
               "plug_residual": round(residual, 4),
               "plug_residual_pct": round(residual_pct, 3),
               "p_any_plug_in_top2": diag["p_any_plug_in_top2"],
               "p_both_plug_in_top2": diag["p_both_plug_in_top2"],
               "mean_retention": (None if mean_ret is None
                                  else round(mean_ret, 4)),
               "retention": ret,
               "util_home": util}
        curve.append(row)
        print(f"  base {k_base:2d} (+{k_base-n0} clones)  share "
              f"{row['plug_pool_share']:.2f}  CE {ce:.4f} ({reg:+.3f}%)  "
              f"clone-only {ce_clone_only:.4f} ({clone_drift:+.3f}%)  "
              f"plug-resid {residual_pct:+.3f}%  "
              f"p_any {row['p_any_plug_in_top2']:.3f}  "
              f"ret {row['mean_retention']}")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    regs = [r["plug_residual_pct"] for r in curve]
    holds = all(regs[i] >= regs[i + 1] for i in range(len(regs) - 1)) \
        and regs[-1] < regs[0] * 0.75
    sig = "DILUTION_HOLDS" if holds else "DILUTION_WEAK_OR_ABSENT"
    verdict = (f"plug-induced base-CE residual vs base-pool size: "
               + " -> ".join(f"{r['base_pool']}:{r['plug_residual_pct']:+.2f}%"
                             for r in curve)
               + ("; residual shrinks with pool size at roughly constant "
                  "delivery — the N=8 residual is substantially a tiny-pool "
                  "artifact" if holds else
                  "; residual does not shrink materially with pool size "
                  "— the residual is not a tiny-pool artifact"))
    print(f"\n  SIGNAL: {sig} — {verdict}")

    out = {"gate": "G3-dilution-clone",
           "signal": sig,
           "verdict": verdict,
           "state": STATE,
           "ce_pre": round(ce_pre, 4),
           "noise": NOISE,
           "curve": curve,
           "note": ("base pool expanded by cloning trained base experts "
                    "(weights copied, passport rows + noise); plug rows "
                    "fixed at the saved N=8 shaped library — isolates pool "
                    "share at constant expert quality; companion real-spine "
                    "run: g3_shaped_run.py on t0-8expert (8 trained base)"),
           "seconds": round(time.time() - t0, 1)}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")
    print(f"total time: {out['seconds']}s")


if __name__ == "__main__":
    main()
