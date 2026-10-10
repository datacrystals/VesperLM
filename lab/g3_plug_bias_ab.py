"""Branch (b) — base-neutral plug weighting: plug-logit bias sweep.

The shaped N=8/N=16 libraries PASS the margin-retention bar but the plug-in
GROUP still saturates top-2 on base tokens (p_any_plug 0.589 at N=8) and
base CE regresses +8.9% (bar <1%).  Zero-training family-row two-stage did
not fix it (g3_family_ab fixed rerun NEGATIVE).  This is the cheaper of the
two named follow-up branches: a pure inference-time reweighting of plug rows
in the router — subtract a bias from every plug-row logit before the top-2
contest, so a plug row must earn its slot by `beta` more evidence than a
base row.  Home tokens with strong memory evidence still deliver; base
tokens stop being hijacked.  One knob, one variable.

  A  beta = 0            (as-saved baseline; must reproduce the run's
                          +8.9% CE / 1.071 retention within noise)
  B  beta in G3W_BETAS   sweep

Pass bar (per branch): base-CE regression < 1% AND N=8 shaped retention
vs solo >= 0.70.  Verdict POSITIVE if any beta passes both; otherwise
NEGATIVE with the Pareto trade-off recorded (family mex calibration is the
named fallback).

Env: G3W_STATE (lab/sandbox/g3_shaped/n8_shaped_state.pt),
     G3W_SOLO_JSON (lab/results/g3_shaped_n8.json),
     G3W_OUT (lab/results/g3_plug_bias_ab.json),
     G3W_BETAS ("0,0.5,1,2,3,4,5,6,8,10"), G3W_DEVICE, G3W_SEED (0).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "lab"))
sys.path.insert(0, os.path.join(REPO, "Hippocampus"))
sys.path.insert(0, os.path.join(REPO, "Common"))

import g1_export_mode as g1  # noqa: E402
import g3_coexistence as g3  # noqa: E402
import memory_shaping as ms  # noqa: E402
from consolidate import load_base_model  # noqa: E402
import g3_family_ab as fam  # noqa: E402

CKPT = os.path.join(
    REPO, "lab", "sandbox", "t1-diag-realdata-mb4",
    "vesper_linear_checkpoints_lab_small", "step_best")
DEVICE = os.environ.get("G3W_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
fam.DEVICE = DEVICE          # restore_library builds experts on this device
restore_library = fam.restore_library


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


STATE = _abs(os.environ.get("G3W_STATE") or
             os.path.join("lab", "sandbox", "g3_shaped", "n8_shaped_state.pt"))
SOLO_JSON = _abs(os.environ.get("G3W_SOLO_JSON") or
                 os.path.join("lab", "results", "g3_shaped_n8.json"))
OUT = _abs(os.environ.get("G3W_OUT") or
           os.path.join("lab", "results", "g3_plug_bias_ab.json"))
BETAS = [float(x) for x in
         (os.environ.get("G3W_BETAS") or "0,0.5,1,2,3,4,5,6,8,10").split(",")]
SEED = 0
RET_BAR = ms.RETENTION_BAR          # 0.70
CE_BAR_PCT = 1.0                    # base-CE regression bar, percent


class PlugBiasRouter(nn.Module):
    """PassportRouter with a plug-row logit bias (beta) before top-k.

    Same call signature/return contract as PassportRouter: (weights,
    indices, aux).  `beta` > 0 makes plug rows need `beta` more evidence to
    enter top-2 — the base-neutral weighting knob.  beta = 0 is exactly the
    saved router behaviour.
    """

    def __init__(self, query, passports, n_base, beta, top_k):
        super().__init__()
        self.query = query
        self.passports = passports
        self.n_base = n_base
        self.beta = float(beta)
        self.top_k = top_k
        self.passport_dim = passports.shape[1]
        self.num_experts = passports.shape[0]

    def forward(self, x):
        logits = self.query(x) @ self.passports.t() / math.sqrt(self.passport_dim)
        if self.beta != 0.0:
            logits = logits.clone()
            logits[:, self.n_base:] -= self.beta
        probs = F.softmax(logits, dim=-1)
        top_w, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        return top_w, top_i, x.new_zeros(())


def swap_bias(model, beta):
    """Install PlugBiasRouter everywhere; return the original routers."""
    saved = []
    for layer in model.layers:
        ffn = layer["ffn"]
        r = ffn.router
        saved.append(r)
        ffn.router = PlugBiasRouter(r.query, r.passports,
                                    g1.BASE_EXPERTS, beta, r.top_k).to(DEVICE)
    return saved


def restore_routers(model, saved):
    for layer, r in zip(model.layers, saved):
        layer["ffn"].router = r


@torch.no_grad()
def occupancy_biased(model, base_chunks, plug_rows, pad_id, beta):
    """g3.routing_matrix's diag with the plug-logit bias applied.

    (routing_matrix computes logits straight from query/passports, so it
    would silently measure beta=0.)
    """
    hs, _ = g3.capture_grouped(model, [[c] for c in base_chunks], pad_id)
    n_layers = len(model.layers)
    ents, anyp, bothp, maxp = [], [], [], []
    for li, layer in enumerate(model.layers):
        r = layer["ffn"].router
        h = hs[li]
        logits = r.query(h) @ r.passports.t() / math.sqrt(r.passport_dim)
        if beta:
            logits = logits.clone()
            logits[:, g1.BASE_EXPERTS:] -= beta
        probs = F.softmax(logits, dim=-1)
        tw, ti = torch.topk(probs, layer["ffn"].top_k, dim=-1)
        ents.append(float((-(probs * (probs + 1e-12).log()).sum(-1)).mean()))
        plug = torch.tensor(plug_rows, dtype=torch.long, device=ti.device)
        pmask = (ti.unsqueeze(-1) == plug).any(-1)
        anyp.append(float(pmask.any(dim=-1).float().mean()))
        bothp.append(float(pmask.all(dim=-1).float().mean()))
        maxp.append(float(tw.max(dim=-1).values.mean()))
    return {"entropy": round(sum(ents) / n_layers, 4),
            "p_any_plug_in_top2": round(sum(anyp) / n_layers, 4),
            "p_both_plug_in_top2": round(sum(bothp) / n_layers, 4),
            "mean_top1_weight": round(sum(maxp) / n_layers, 4)}


@torch.no_grad()
def util_home_biased(model, home_lists, plug_rows, pad_id, beta):
    """Per-group P(own row in top-2), with the same bias applied."""
    hs, slices = g3.capture_grouped(model, home_lists, pad_id)
    n_layers = len(model.layers)
    out = []
    for g in range(len(home_lists)):
        vals = []
        for li, layer in enumerate(model.layers):
            r = layer["ffn"].router
            h = hs[li][slices[g][0]:slices[g][1]]
            logits = r.query(h) @ r.passports.t() / math.sqrt(r.passport_dim)
            if beta:
                logits = logits.clone()
                logits[:, g1.BASE_EXPERTS:] -= beta
            ti = torch.topk(logits, layer["ffn"].top_k, dim=-1).indices
            own = plug_rows[g]
            vals.append(float((ti == own).any(dim=-1).float().mean()))
        out.append(sum(vals) / n_layers)
    return out


def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"plug-bias sweep  state={STATE}  betas={BETAS}  device={DEVICE}")

    model, tok, mc = load_base_model(CKPT, device=DEVICE)
    g1.DIM = int(mc["dim"]); g1.HIDDEN = int(mc["hidden_dim"])
    g1.N_LAYERS = int(mc["n_layers"]); g1.BASE_EXPERTS = int(mc["num_experts"])
    g1.prepare_routers(model)
    sd = torch.load(STATE, map_location="cpu", weights_only=False)
    n_mem = restore_library(model, sd["state_dict"])
    model.eval()
    PAD_ID = tok.pad_token_id
    names = sd.get("episodes", [e.name for e in ms.SHAPED_8])
    episodes = [e for e in ms.SHAPED_8 if e.name in names]
    plug_rows = list(range(g1.BASE_EXPERTS, g1.BASE_EXPERTS + n_mem))
    BASE_MIX = g1.load_base_mix_chunks(n_chunks=32, seq=256, seed=SEED)

    refj = json.load(open(SOLO_JSON))
    solo = {int(k): v for k, v in refj.get("solo", {}).items()}
    pre_ref = {int(k): v for k, v in refj.get("pre_library", {}).items()}
    ce_pre = float(refj.get("summary", {}).get("base_ce", {}).get("pre", 0) or 0)
    if not ce_pre:
        ce_pre = 6.7735   # g3_shaped_n8.json default; recomputed below if absent
    home_lists = [ms.episode_home_ids(tok, ep, DEVICE) for ep in episodes]

    def ret_table(marg):
        out = {}
        for j, ep in enumerate(episodes):
            s = solo.get(j)
            if not s:
                continue
            pre = pre_ref[j]["mean_logit_margin"]
            g_lib = marg[ep.name] - pre
            out[ep.name] = (None if not s["gain_logit"]
                            else round(g_lib / s["gain_logit"], 4))
        return out

    def evaluate(beta):
        saved = swap_bias(model, beta)
        try:
            ce = g1.next_token_ce(model, BASE_MIX)
            marg = {ep.name: ms.episode_margins(model, tok, ep, DEVICE)
                    ["mean_logit_margin"] for ep in episodes}
            ret = ret_table(marg)
            vals = [v for v in ret.values() if v is not None]
            mean_ret = sum(vals) / len(vals) if vals else None
            occ = occupancy_biased(model, BASE_MIX[:8], plug_rows, PAD_ID, beta)
            util = util_home_biased(model, home_lists, plug_rows, PAD_ID, beta)
        finally:
            restore_routers(model, saved)
        reg = 100.0 * (ce - ce_pre) / ce_pre
        ok = (reg < CE_BAR_PCT) and (mean_ret is not None
                                     and mean_ret >= RET_BAR)
        row = {"beta": beta,
               "ce_lib": round(ce, 4), "base_ce_regression_pct": round(reg, 3),
               "p_any_plug_in_top2": occ["p_any_plug_in_top2"],
               "p_both_plug_in_top2": occ["p_both_plug_in_top2"],
               "mean_top1_weight": occ["mean_top1_weight"],
               "retention": ret,
               "mean_retention": (None if mean_ret is None
                                  else round(mean_ret, 4)),
               "util_home_min": round(min(util), 4),
               "pass_both_bars": bool(ok)}
        return row

    sweep = []
    for beta in BETAS:
        row = evaluate(beta)
        sweep.append(row)
        print(f"  beta {beta:5.1f}  CE {row['ce_lib']:.4f} "
              f"({row['base_ce_regression_pct']:+.3f}%)  "
              f"p_any {row['p_any_plug_in_top2']:.3f}  "
              f"ret {row['mean_retention']}  "
              f"util_min {row['util_home_min']:.3f}  "
              f"{'PASS' if row['pass_both_bars'] else 'fail'}")

    winners = [r for r in sweep if r["pass_both_bars"]]
    feasible_ret = [r for r in sweep
                    if r["mean_retention"] is not None
                    and r["mean_retention"] >= RET_BAR]
    if winners:
        # among passing betas, take the smallest base CE
        best = min(winners, key=lambda r: r["ce_lib"])
        sig = "POSITIVE"
        verdict = ("plug-logit bias is a sufficient base-neutral weighting: "
                   f"beta {best['beta']} passes both bars "
                   f"(CE {best['base_ce_regression_pct']:+.2f}%, "
                   f"retention {best['mean_retention']})")
    else:
        best = None
        sig = "NEGATIVE"
        if feasible_ret:
            b = min(feasible_ret, key=lambda r: r["base_ce_regression_pct"])
            verdict = ("no beta passes both bars — retention-preserving "
                       f"betas still regress base CE "
                       f"({b['base_ce_regression_pct']:+.2f}% at beta "
                       f"{b['beta']}, best of the ret-feasible set); "
                       "bias alone cannot separate base from home at this "
                       "overlap — family-row mex calibration is the "
                       "fallback branch")
        else:
            verdict = ("no beta keeps retention — the bias destroys memory "
                       "delivery before it recovers base CE; family-row "
                       "mex calibration is the fallback branch")
    print(f"\n  SIGNAL: {sig} — {verdict}")

    out = {"gate": "G3-plug-bias",
           "signal": sig,
           "verdict": verdict,
           "state": STATE,
           "solo_json": SOLO_JSON,
           "ce_pre": round(ce_pre, 4),
           "bars": {"base_ce_regression_pct": CE_BAR_PCT,
                    "retention_vs_solo": RET_BAR},
           "betas": BETAS,
           "sweep": sweep,
           "best": best,
           "note": ("pure inference-time plug-row logit bias on the saved "
                    "N=8 shaped library state — no training; beta=0 is the "
                    "as-saved router"),
           "seconds": round(time.time() - t0, 1)}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")
    print(f"total time: {out['seconds']}s")


if __name__ == "__main__":
    main()
