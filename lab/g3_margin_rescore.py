#!/usr/bin/env python3
"""G3 margin re-score — does the taught-fact MARGIN survive at low util?

Motivation (MODULAR_MOE.md §8.4 G3 note): if margins hold at util ~0.4, the
product requirement survives and the util bar, not the architecture, gets
revised.  G3's purity gate failed on util_home (0.17-0.39 for 15/16 experts
at N=16) while contamination stayed clean — but util_home is a routing
statistic, not the memory itself.

What this measures (the G1b margin methodology generalized):
  taught-fact margin for a (prompt, stated, correction) triple =
    logit(taught token) - logit(rejected token) at the first token where the
    two continuations diverge, conditioned on prompt + shared prefix.
    For episode 0 (G1's math/words set) this IS the classic word-digit
    first-token margin from the Hippocampus demo; it is also reported via
    word_digit_margin on the demo probes for direct G1b comparability
    (+4.59 mean gain at N=1).

  margin gain = margin(state) - margin(pre-library spine)

Three states per episode j in 0..N-1:
  pre    bare spine (4 base experts)
  solo   episode j alone: consolidate -> prototype plug -> 800-step mex recal
         (the G1b N=1 condition = "full delivery" reference)
  lib    the N=8 full-recipe library state (ctrlB recipe: 80-step
         consolidation + 800-step mex recal per insert + 800-step §4.4a
         final joint calibration), margins measured for every live expert

Verdict:
  (a) margins hold at low util: mean lib gain >= 0.7 * G1b gain (+4.59) OR
      per-episode retention (lib gain / solo gain) >= 0.7 while util is
      0.3-0.5 -> the util bar is revised, margin retention becomes the bar
  (b) margins collapse with util -> addressing pivot confirmed

The rebuilt N=8 state is saved to lab/sandbox/g3_margin/n8_state.pt so a
re-run is never needed (sandbox is not committed).

Run:  python3 lab/g3_margin_rescore.py
Env:  G3M_N (8), G3M_CONS_STEPS (80), G3M_RECAL_STEPS (800),
      G3M_FINAL_JOINT (800), G3M_DEVICE, G3M_OUT
      (lab/results/g3_margin_n8.json)
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "lab"))
sys.path.insert(0, os.path.join(REPO, "Hippocampus"))
sys.path.insert(0, os.path.join(REPO, "Common"))

import g1_export_mode as g1  # noqa: E402
import g3_coexistence as g3  # noqa: E402
from consolidate import (  # noqa: E402
    load_base_model, first_token_logits, word_digit_margin, response_nll,
    _encode_triple,
)
from demo import PROBES  # noqa: E402

CKPT = g3.CKPT
DEVICE = os.environ.get("G3M_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
N_EPS = int(os.environ.get("G3M_N", "8"))
CONS_STEPS = int(os.environ.get("G3M_CONS_STEPS", "80"))
RECAL_STEPS = int(os.environ.get("G3M_RECAL_STEPS", "800"))
FINAL_JOINT = int(os.environ.get("G3M_FINAL_JOINT", "800"))
CONS_LR = 2e-3
RECAL_LR = 1e-2
SEED = 0
G1B_MEAN_GAIN = 4.588   # g1b_passport_d01_neutral3, math/words probes
OUT = os.environ.get("G3M_OUT") or os.path.join(
    REPO, "lab", "results", "g3_margin_n8.json")
STATE_OUT = os.path.join(REPO, "lab", "sandbox", "g3_margin", "n8_state.pt")
LOGDIR = os.path.join(REPO, "lab", "logs")
WORKDIR = os.path.join(REPO, "lab", "sandbox", "g3_margin")


# ------------------------------------------------------------------
# Taught-fact margin (G1b methodology, generalized to any correction pair)
# ------------------------------------------------------------------

@torch.no_grad()
def fact_margin(model, tok, prompt, stated, correction, device):
    """logit(taught token) - logit(rejected token) at first divergence.

    Encoding matches _encode_triple (prompt ids + response ids concatenated),
    so the conditioning context is exactly what training/eval used.  When one
    continuation ends before the other (suffix/prefix tag styles), the
    divergence token is compared against EOS ("keep going with the tag" vs
    "stop").  Positive = the taught continuation is preferred.
    """
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    s_ids = tok(stated, add_special_tokens=False).input_ids
    c_ids = tok(correction, add_special_tokens=False).input_ids
    d = 0
    while d < min(len(s_ids), len(c_ids)) and s_ids[d] == c_ids[d]:
        d += 1
    if d >= len(s_ids) and d >= len(c_ids):
        return None  # identical continuations
    ctx = p_ids + s_ids[:d]
    if len(ctx) == 0:
        return None
    x = torch.tensor([ctx], dtype=torch.long, device=device)
    logits = model(x)[0][0, -1].float()
    eos = tok.eos_token_id
    if d < len(s_ids) and d < len(c_ids):
        return float(logits[c_ids[d]] - logits[s_ids[d]])
    if d >= len(s_ids):  # stated ends first (suffix tag): tag-first vs EOS
        return float(logits[c_ids[d]] - logits[eos])
    return float(logits[eos] - logits[s_ids[d]])  # correction ends first


@torch.no_grad()
def episode_margins(model, tok, ep_idx, device):
    """Per-fact margins + paired NLL margin for one episode's 3 taught facts."""
    _, turns = g3.EPISODES[ep_idx]
    out = []
    for prompt, stated, correction in turns[:3]:
        m = fact_margin(model, tok, prompt, stated, correction, device)
        nll_stated = response_nll(model, tok, prompt, stated, device)
        nll_corr = response_nll(model, tok, prompt, correction, device)
        out.append({"prompt": prompt,
                    "logit_margin": None if m is None else round(m, 4),
                    "nll_margin": round(nll_stated - nll_corr, 4)})
    lm = [r["logit_margin"] for r in out if r["logit_margin"] is not None]
    return {"facts": out,
            "mean_logit_margin": round(sum(lm) / len(lm), 4) if lm else None,
            "mean_nll_margin": round(
                sum(r["nll_margin"] for r in out) / len(out), 4)}


@torch.no_grad()
def demo_probe_margins(model, tok, device):
    """Classic G1b word-digit margins on the 3 demo probes (episode 0)."""
    vals = []
    for p in PROBES:
        logits = first_token_logits(model, tok, p, device=device)
        vals.append(round(word_digit_margin(logits, tok)["margin"], 4))
    return {"per_probe": vals,
            "mean": round(sum(vals) / len(vals), 4)}


# ------------------------------------------------------------------
# Model lifecycle: fresh load per phase (identical code path to ctrlB)
# ------------------------------------------------------------------

def load_fresh():
    model, tok, mc = load_base_model(CKPT, device=DEVICE)
    g1.DIM = int(mc["dim"])
    g1.HIDDEN = int(mc["hidden_dim"])
    g1.N_LAYERS = int(mc["n_layers"])
    g1.BASE_EXPERTS = int(mc["num_experts"])
    g1.TEXT_KL_COEF = 3.0
    g1.WORKDIR = WORKDIR
    g1.prepare_routers(model)
    return model, tok


def build_expert_for(model, ep_idx, tag):
    experts = [g1.ContractExpert(g1.DIM, g1.HIDDEN).to(DEVICE)
               for _ in range(g1.N_LAYERS)]
    for i, e in enumerate(experts):
        g1.birth_from_base(e, model.layers[i]["ffn"].experts[0])
    triples = g3.episode_triples(ep_idx, g3.EPISODES[ep_idx][1])
    stats = g1.train_expert(model, tok_global, triples, experts,
                            steps=CONS_STEPS, lr=CONS_LR, device=DEVICE,
                            tag=tag, text_batches=BASE_MIX[:4])
    with torch.no_grad():
        home_h, _ = g3.capture_grouped(
            model, [g1.encode_texts(tok_global,
                                    g3.episode_home_texts(g3.EPISODES[ep_idx][1]),
                                    DEVICE)], PAD_ID)
        rows = [model.layers[i]["ffn"].router.query(home_h[i]).mean(dim=0)
                for i in range(g1.N_LAYERS)]
    g1.plug_expert(model, experts, rows)
    return stats, triples


tok_global = None
BASE_MIX = None
PAD_ID = None


def main():
    global tok_global, BASE_MIX, PAD_ID
    t0 = time.time()
    os.makedirs(WORKDIR, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    eps = g3.EPISODES[:N_EPS]
    print(f"G3 margin re-score  N={N_EPS}  device={DEVICE}  "
          f"cons={CONS_STEPS} recal={RECAL_STEPS} joint={FINAL_JOINT}")

    def section(s):
        print("\n" + "=" * 72 + f"\n{s}\n" + "=" * 72)

    # ---------------- Phase 0: pre-library baseline ----------------
    section("PHASE 0 — pre-library margins (bare spine)")
    model, tok_global = load_fresh()
    PAD_ID = tok_global.pad_token_id
    BASE_MIX = g1.load_base_mix_chunks(n_chunks=32, seq=256, seed=SEED)
    pre = {}
    for j in range(N_EPS):
        pre[j] = episode_margins(model, tok_global, j, DEVICE)
        print(f"  ep{j:2d} {eps[j][0]:15s} logit {pre[j]['mean_logit_margin']:+.3f}  "
              f"nll {pre[j]['mean_nll_margin']:+.3f}")
    pre_demo = demo_probe_margins(model, tok_global, DEVICE)
    print(f"  demo probes (word-digit): {pre_demo['per_probe']} "
          f"mean {pre_demo['mean']:+.3f}")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ---------------- Phase 1: solo references (N=1 each) ----------------
    section("PHASE 1 — solo states (episode j alone, full recipe)")
    solo = {}
    for j in range(N_EPS):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        model, tok_global = load_fresh()
        stats, triples = build_expert_for(model, j, f"solo{j}")
        home_ids = g1.encode_texts(
            tok_global, g3.episode_home_texts(eps[j][1]), DEVICE)
        g3.recal_mex(model, [home_ids], BASE_MIX, PAD_ID,
                     steps=RECAL_STEPS, lr=RECAL_LR, device=DEVICE)
        m = episode_margins(model, tok_global, j, DEVICE)
        demo = demo_probe_margins(model, tok_global, DEVICE) if j == 0 else None
        hits = g3.routing_matrix_group(
            model, [home_ids], [g1.BASE_EXPERTS], PAD_ID)
        solo[j] = {"margins": m,
                   "util_home": round(hits[0][g1.BASE_EXPERTS], 4),
                   "gain_logit": round(
                       m["mean_logit_margin"] - pre[j]["mean_logit_margin"], 4),
                   "gain_nll": round(
                       m["mean_nll_margin"] - pre[j]["mean_nll_margin"], 4),
                   "demo_probes": demo}
        print(f"  solo ep{j:2d} {eps[j][0]:15s} util {solo[j]['util_home']:.3f}  "
              f"margin {m['mean_logit_margin']:+.3f}  gain {solo[j]['gain_logit']:+.3f}"
              + (f"  demo {demo['mean']:+.3f} (gain {demo['mean'] - pre_demo['mean']:+.3f})"
                 if demo else ""))
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ---------------- Phase 2: N=8 library build (ctrlB recipe) ----------------
    section(f"PHASE 2 — N={N_EPS} library build (ctrlB recipe)")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model, tok_global = load_fresh()
    traj = []
    live = []
    for j in range(N_EPS):
        t_ins = time.time()
        stats, triples = build_expert_for(model, j, f"ep{j}")
        home_ids = [g1.encode_texts(tok_global,
                                    g3.episode_home_texts(eps[k][1]), DEVICE)
                    for k in range(N_EPS)]
        g3.recal_mex(model, [home_ids[k] for k in live + [j]], BASE_MIX, PAD_ID,
                     steps=RECAL_STEPS, lr=RECAL_LR, device=DEVICE)
        live.append(j)
        n_rows = model.layers[0]["ffn"].router.passports.shape[0]
        plug_rows = list(range(g1.BASE_EXPERTS, n_rows))
        per_group = g3.routing_matrix_group(
            model, [home_ids[k] for k in live], plug_rows, PAD_ID)
        row = {"n": j + 1, "episode": eps[j][0],
               "seconds": round(time.time() - t_ins, 1)}
        for gi, k in enumerate(live):
            row.setdefault("util_home", {})[k] = round(
                per_group[gi][g1.BASE_EXPERTS + k], 4)
        for k in live:
            row.setdefault("margins", {})[k] = episode_margins(
                model, tok_global, k, DEVICE)["mean_logit_margin"]
        traj.append(row)
        print(f"  insert {j+1}/{N_EPS} {eps[j][0]:15s} "
              + " ".join(f"e{k} util {row['util_home'][k]:.2f} "
                         f"mgn {row['margins'][k]:+.2f}" for k in live))

    if FINAL_JOINT > 0:
        section(f"PHASE 2b — §4.4a final joint calibration ({FINAL_JOINT})")
        g3.recal_mex(model, [home_ids[k] for k in live], BASE_MIX, PAD_ID,
                     steps=FINAL_JOINT, lr=RECAL_LR, device=DEVICE)

    # ---------------- Phase 3: library-state margins ----------------
    section("PHASE 3 — library-state margins (final)")
    n_rows = model.layers[0]["ffn"].router.passports.shape[0]
    plug_rows = list(range(g1.BASE_EXPERTS, n_rows))
    home_ids = [g1.encode_texts(tok_global,
                                g3.episode_home_texts(eps[k][1]), DEVICE)
                for k in range(N_EPS)]
    per_group = g3.routing_matrix_group(
        model, [home_ids[k] for k in live], plug_rows, PAD_ID)
    lib = {}
    print(f"  {'ep':>3s} {'episode':15s} {'pre':>7s} {'solo':>7s} {'lib':>7s} "
          f"{'gain_lib':>8s} {'gain_solo':>9s} {'retention':>9s} {'util':>6s}")
    for gi, j in enumerate(live):
        m = episode_margins(model, tok_global, j, DEVICE)
        util = per_group[gi][g1.BASE_EXPERTS + j]
        gain_lib = m["mean_logit_margin"] - pre[j]["mean_logit_margin"]
        gain_solo = solo[j]["gain_logit"]
        retention = (gain_lib / gain_solo) if gain_solo else None
        lib[j] = {"margins": m,
                  "util_home": round(util, 4),
                  "gain_logit": round(gain_lib, 4),
                  "gain_solo": gain_solo,
                  "retention_vs_solo": (None if retention is None
                                        else round(retention, 4))}
        print(f"  {j:3d} {eps[j][0]:15s} {pre[j]['mean_logit_margin']:+7.3f} "
              f"{solo[j]['margins']['mean_logit_margin']:+7.3f} "
              f"{m['mean_logit_margin']:+7.3f} {gain_lib:+8.3f} "
              f"{gain_solo:+9.3f} "
              f"{(retention if retention is not None else float('nan')):9.3f} "
              f"{util:6.3f}")
    demo_lib = demo_probe_margins(model, tok_global, DEVICE)
    print(f"  demo probes on library: {demo_lib['per_probe']} "
          f"mean {demo_lib['mean']:+.3f}  gain {demo_lib['mean'] - pre_demo['mean']:+.3f}"
          f"  (G1b N=1 gain +{G1B_MEAN_GAIN})")

    # persist state (sandbox: not committed)
    os.makedirs(os.path.dirname(STATE_OUT), exist_ok=True)
    torch.save({
        "spine_and_bank": {k: v.cpu() for k, v in model.state_dict().items()},
        "episodes": [eps[k][0] for k in live],
        "recipe": {"cons": CONS_STEPS, "recal": RECAL_STEPS,
                   "joint": FINAL_JOINT},
    }, STATE_OUT)
    print(f"  state saved: {STATE_OUT}")

    # ---------------- Phase 4: verdict ----------------
    section("PHASE 4 — verdict")
    gains = [lib[j]["gain_logit"] for j in live]
    rets = [lib[j]["retention_vs_solo"] for j in live]
    utils = [lib[j]["util_home"] for j in live]
    mean_gain = sum(gains) / len(gains)
    mean_ret = sum(r for r in rets if r is not None) / max(
        1, len([r for r in rets if r is not None]))
    bar_gain = 0.7 * G1B_MEAN_GAIN
    low_util = [u for u in utils if u < 0.5]
    margins_hold = (mean_gain >= bar_gain) or (mean_ret >= 0.7)
    # correlation between util and gain (rank-free simple report)
    print(f"  mean margin gain (lib): {mean_gain:+.3f}  "
          f"(bar: >= {bar_gain:+.3f} = 70% of G1b {G1B_MEAN_GAIN})")
    print(f"  mean retention vs solo: {mean_ret:.3f}  (bar: >= 0.70)")
    print(f"  experts at util<0.5: {len(low_util)}/{len(live)}  "
          f"their mean gain: "
          f"{(sum(lib[j]['gain_logit'] for j in live if lib[j]['util_home'] < 0.5) / max(1, len(low_util))):+.3f}")
    print(f"  demo-probe gain on library: "
          f"{demo_lib['mean'] - pre_demo['mean']:+.3f}  vs G1b +{G1B_MEAN_GAIN}")
    verdict = "MARGINS_HOLD" if margins_hold else "MARGINS_COLLAPSE"
    print(f"  VERDICT: {verdict} — "
          + ("util bar is wrong; margin retention becomes the bar"
             if margins_hold else
             "addressing pivot confirmed; hierarchical passports stay ACTIVE"))

    out = {
        "gate": "G3-margin-rescore",
        "verdict": verdict,
        "n_episodes": N_EPS,
        "spine": CKPT,
        "recipe": {"cons": CONS_STEPS, "recal": RECAL_STEPS,
                   "joint": FINAL_JOINT, "text_kl": 3.0,
                   "owner_mass": g3.OWNER_MASS},
        "g1b_reference_gain": G1B_MEAN_GAIN,
        "pre_library": {str(j): pre[j] for j in pre},
        "pre_demo_probes": pre_demo,
        "solo": {str(j): solo[j] for j in solo},
        "library": {str(j): lib[j] for j in live},
        "library_demo_probes": demo_lib,
        "insert_trajectory": traj,
        "summary": {
            "mean_gain_logit": round(mean_gain, 4),
            "bar_70pct_g1b": round(bar_gain, 4),
            "mean_retention_vs_solo": round(mean_ret, 4),
            "util_home": {str(j): lib[j]["util_home"] for j in live},
            "experts_below_util_0.5": len(low_util),
            "mean_gain_low_util": round(
                sum(lib[j]["gain_logit"] for j in live
                    if lib[j]["util_home"] < 0.5) / max(1, len(low_util)), 4),
        },
        "state_saved": STATE_OUT,
        "seconds": round(time.time() - t0, 1),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")
    print(f"total time: {out['seconds']}s")


if __name__ == "__main__":
    main()
