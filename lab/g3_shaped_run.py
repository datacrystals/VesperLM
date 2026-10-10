#!/usr/bin/env python3
"""G3 coexistence redo with SHAPED memories (token-substitution deltas).

The addressing-pivot runs (2026-10-09) showed margin retention is predicted
by memory SHAPE: token-substitution episodes retain 0.884 of solo delivery at
N=8 vs 0.390 for the mixed (tag-append) fixture.  This job re-runs G3
coexistence with the shaped fixture (lab/memory_shaping.SHAPED_N) under the
standing margin-retention bar (retention vs solo >= 0.70):

  N=8   sanity anchor — expect ~0.88 (RUN B baseline)
  N=16  the G3 gate itself, previously untested with shaped memories

Same ctrlB recipe as every run on this line: 80-step consolidation with
text-KL 3.0 (base-neutral experts), prototype-init passports (mean router
query over home), 800-step section-4.4a mex recal per insert (all plug-in
rows jointly, full home coverage, base rows frozen), 800-step closing joint
calibration.  Same three-state margin measurement (pre / solo / lib).

The known open residual is measured explicitly: group top-2 occupancy of
base tokens (p_any_plug_in_top2) and the base-mix CE regression it causes —
shape does not touch occupancy, so it is expected to persist.

Runs standalone or through the lab farm (queue JSON with "script" field).
State dicts go to lab/sandbox/ (never committed).

Env: G3S_N (8), G3S_OUT, G3S_STATE, G3S_CONS_STEPS (80), G3S_RECAL_STEPS
     (800), G3S_FINAL_JOINT (800), G3S_DEVICE, G3S_SEED (0)
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
import memory_shaping as ms  # noqa: E402
from consolidate import load_base_model  # noqa: E402

_CKPT_ENV = os.environ.get("G3S_CKPT")
DEVICE = os.environ.get("G3S_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
N_EPS = int(os.environ.get("G3S_N", "8"))
CONS_STEPS = int(os.environ.get("G3S_CONS_STEPS", "80"))
RECAL_STEPS = int(os.environ.get("G3S_RECAL_STEPS", "800"))
FINAL_JOINT = int(os.environ.get("G3S_FINAL_JOINT", "800"))
TEXT_KL = float(os.environ.get("G3S_TEXT_KL", "3.0"))
CONS_LR = 2e-3
RECAL_LR = 1e-2
SEED = int(os.environ.get("G3S_SEED", "0"))
ANCHOR_N8 = 0.884   # g3_pivot_shape.json mean retention
TAG = "shape%d" % N_EPS


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


CKPT = (_abs(_CKPT_ENV) if _CKPT_ENV else os.path.join(
    REPO, "lab", "sandbox", "t1-diag-realdata-mb4",
    "vesper_linear_checkpoints_lab_small", "step_best"))
OUT = _abs(os.environ.get("G3S_OUT") or
           os.path.join("lab", "results", f"g3_shaped_n{N_EPS}.json"))
STATE_OUT = _abs(os.environ.get("G3S_STATE") or
                 os.path.join("lab", "sandbox", "g3_shaped",
                              f"n{N_EPS}_shaped_state.pt"))
WORKDIR = os.path.join(REPO, "lab", "sandbox", "g3_shaped")

tok_global = None
BASE_MIX = None
PAD_ID = None


def section(s):
    print("\n" + "=" * 72 + f"\n{s}\n" + "=" * 72)


def load_fresh():
    model, tok, mc = load_base_model(CKPT, device=DEVICE)
    g1.DIM = int(mc["dim"])
    g1.HIDDEN = int(mc["hidden_dim"])
    g1.N_LAYERS = int(mc["n_layers"])
    g1.BASE_EXPERTS = int(mc["num_experts"])
    g1.WORKDIR = WORKDIR
    g1.prepare_routers(model)
    return model, tok


def measure_all(model, episodes, live):
    n_rows = model.layers[0]["ffn"].router.passports.shape[0]
    plug_rows = list(range(g1.BASE_EXPERTS, n_rows))
    home_lists = [ms.episode_home_ids(tok_global, episodes[k], DEVICE)
                  for k in live]
    per_group = g3.routing_matrix_group(model, home_lists, plug_rows, PAD_ID)
    hits_b, diag = g3.routing_matrix(
        model, [[BASE_MIX[i]] for i in range(8)], plug_rows, PAD_ID)
    out = {}
    for gi, j in enumerate(live):
        m = ms.episode_margins(model, tok_global, episodes[j], DEVICE)
        out[j] = {"margins": m,
                  "util_home": round(per_group[gi][g1.BASE_EXPERTS + j], 4),
                  "contam_base": round(hits_b[g1.BASE_EXPERTS + j], 4)}
    return out, diag


def main():
    global tok_global, BASE_MIX, PAD_ID
    t0 = time.time()
    os.makedirs(WORKDIR, exist_ok=True)
    os.makedirs(os.path.dirname(STATE_OUT), exist_ok=True)
    episodes = (ms.SHAPED_8 if N_EPS == 8 else ms.SHAPED_16)[:N_EPS]
    assert len(episodes) == N_EPS
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"G3 shaped coexistence  N={N_EPS}  device={DEVICE}  "
          f"cons={CONS_STEPS} recal={RECAL_STEPS} joint={FINAL_JOINT} seed={SEED}")
    print(f"  fixture: {N_EPS} token-substitution episodes "
          f"(memory_shaping.SHAPED_{N_EPS})")

    report = ms.shape_report(episodes)
    dupes = ms.dedup_check(episodes)
    print(f"  shape_report: {report}")
    assert report["substitution"] == 3 * N_EPS, "fixture not all substitution"
    assert not dupes, f"prompt collisions: {dupes}"

    # ---------------- Phase 0: pre-library ----------------
    section("PHASE 0 — pre-library margins (bare spine)")
    model, tok_global = load_fresh()
    PAD_ID = tok_global.pad_token_id
    BASE_MIX = g1.load_base_mix_chunks(n_chunks=32, seq=256, seed=SEED)
    pre = {}
    for j in range(N_EPS):
        pre[j] = ms.episode_margins(model, tok_global, episodes[j], DEVICE)
        print(f"  ep{j:2d} {episodes[j].name:17s} logit "
              f"{pre[j]['mean_logit_margin']:+.3f}")
    ce_pre = g1.next_token_ce(model, BASE_MIX)
    cos_pre = g3.bank_norms(model)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ---------------- Phase 1: solo refs (full recipe each) ----------------
    section("PHASE 1 — solo states (episode j alone, full recipe)")
    solo = {}
    for j in range(N_EPS):
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        model, tok_global = load_fresh()
        experts, rows, info = ms.consolidate_episode(
            model, tok_global, episodes[j], BASE_MIX,
            steps=CONS_STEPS, lr=CONS_LR, device=DEVICE, tag=f"solo{j}",
            text_kl_coef=TEXT_KL)
        home = [ms.episode_home_ids(tok_global, episodes[j], DEVICE)]
        ms.plug_and_recal(model, experts, rows, home, BASE_MIX,
                          recal_steps=RECAL_STEPS, recal_lr=RECAL_LR,
                          device=DEVICE, pad_id=PAD_ID)
        m = ms.episode_margins(model, tok_global, episodes[j], DEVICE)
        hits = g3.routing_matrix_group(model, home, [g1.BASE_EXPERTS], PAD_ID)
        solo[j] = {"margins": m,
                   "util_home": round(hits[0][g1.BASE_EXPERTS], 4),
                   "gate": info["gate"],
                   "gain_logit": round(
                       m["mean_logit_margin"] - pre[j]["mean_logit_margin"], 4)}
        print(f"  solo ep{j:2d} {episodes[j].name:17s} util {solo[j]['util_home']:.3f} "
              f"gain {solo[j]['gain_logit']:+.3f}  gate={info['gate']}")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ---------------- Phase 2: library build ----------------
    section(f"PHASE 2 — N={N_EPS} library build (ctrlB recipe)")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model, tok_global = load_fresh()
    traj, live = [], []
    for j in range(N_EPS):
        t_ins = time.time()
        experts, rows, info = ms.consolidate_episode(
            model, tok_global, episodes[j], BASE_MIX,
            steps=CONS_STEPS, lr=CONS_LR, device=DEVICE, tag=f"ep{j}",
            text_kl_coef=TEXT_KL)
        home_lists = [ms.episode_home_ids(tok_global, episodes[k], DEVICE)
                      for k in range(N_EPS)]
        rstat = ms.plug_and_recal(
            model, experts, rows, [home_lists[k] for k in live + [j]],
            BASE_MIX, recal_steps=RECAL_STEPS, recal_lr=RECAL_LR,
            device=DEVICE, pad_id=PAD_ID)
        live.append(j)
        probe, diag = measure_all(model, episodes, live)
        row = {"n": j + 1, "episode": episodes[j].name, "gate": info["gate"],
               "cons_s": round(time.time() - t_ins, 1) - rstat["seconds"],
               "recal_s": rstat["seconds"],
               "bank_norms": g3.bank_norms(model),
               "p_any_plug_base": round(diag["p_any_plug_in_top2"], 4),
               "p_both_plug_base": round(diag["p_both_plug_in_top2"], 4)}
        for k in live:
            row.setdefault("util_home", {})[k] = probe[k]["util_home"]
            row.setdefault("contam_base", {})[k] = probe[k]["contam_base"]
            row.setdefault("margins", {})[k] = probe[k]["margins"]["mean_logit_margin"]
        traj.append(row)
        print(f"  insert {j+1}/{N_EPS} {episodes[j].name:17s} "
              f"p_any {row['p_any_plug_base']:.3f}  "
              + " ".join(f"e{k} u {row['util_home'][k]:.2f} "
                         f"m {row['margins'][k]:+.2f}" for k in live))

    if FINAL_JOINT > 0:
        section(f"PHASE 2b — §4.4a final joint calibration ({FINAL_JOINT})")
        home_lists = [ms.episode_home_ids(tok_global, episodes[k], DEVICE)
                      for k in range(N_EPS)]
        g3.recal_mex(model, [home_lists[k] for k in live], BASE_MIX, PAD_ID,
                     steps=FINAL_JOINT, lr=RECAL_LR, device=DEVICE)

    # ---------------- Phase 3: library state ----------------
    section("PHASE 3 — library-state margins (final)")
    lib, diag = measure_all(model, episodes, live)
    ce_lib = g1.next_token_ce(model, BASE_MIX)
    ce_reg = 100.0 * (ce_lib - ce_pre) / ce_pre
    cos_lib = g3.bank_norms(model)
    print(f"  {'ep':>3} {'episode':17s} {'pre':>7s} {'solo':>7s} {'lib':>7s} "
          f"{'gain_lib':>8s} {'gain_solo':>9s} {'retention':>9s} {'util':>6s} {'contam':>6s}")
    rets, gains = [], []
    for j in live:
        g_lib = lib[j]["margins"]["mean_logit_margin"] - pre[j]["mean_logit_margin"]
        g_solo = solo[j]["gain_logit"]
        ret = ms.retention(g_lib, g_solo)
        lib[j]["gain_logit"] = round(g_lib, 4)
        lib[j]["gain_solo"] = g_solo
        lib[j]["retention_vs_solo"] = (None if ret is None else round(ret, 4))
        rets.append(ret); gains.append(g_lib)
        print(f"  {j:3d} {episodes[j].name:17s} "
              f"{pre[j]['mean_logit_margin']:+7.3f} "
              f"{solo[j]['margins']['mean_logit_margin']:+7.3f} "
              f"{lib[j]['margins']['mean_logit_margin']:+7.3f} "
              f"{g_lib:+8.3f} {g_solo:+9.3f} "
              f"{(ret if ret is not None else float('nan')):9.3f} "
              f"{lib[j]['util_home']:6.3f} {lib[j]['contam_base']:6.3f}")
    mean_ret = sum(r for r in rets if r is not None) / max(
        1, len([r for r in rets if r is not None]))
    util_min = min(lib[j]["util_home"] for j in live)
    contam_max = max(lib[j]["contam_base"] for j in live)
    print(f"  mean retention {mean_ret:.3f}  util_min {util_min:.3f}  "
          f"contam_max {contam_max:.3f}")
    print(f"  base CE: {ce_pre:.4f} -> {ce_lib:.4f}  ({ce_reg:+.3f}%)  "
          f"p_any_plug {diag['p_any_plug_in_top2']:.3f}  "
          f"p_both {diag['p_both_plug_in_top2']:.3f}")
    print(f"  bank norms: {cos_lib}")

    os.makedirs(os.path.dirname(STATE_OUT), exist_ok=True)
    torch.save({"state_dict": ms.state_dict_cpu(model),
                "n": N_EPS, "episodes": [episodes[k].name for k in live],
                "recipe": {"cons": CONS_STEPS, "recal": RECAL_STEPS,
                           "joint": FINAL_JOINT, "text_kl": TEXT_KL}},
               STATE_OUT)
    print(f"  state saved: {STATE_OUT}")

    # ---------------- Phase 4: verdict ----------------
    section("PHASE 4 — verdict (standing bar: margin retention >= 0.70)")
    holds = mean_ret >= ms.RETENTION_BAR
    verdict = "PASS" if holds else "FAIL"
    print(f"  retention {mean_ret:.3f} vs bar {ms.RETENTION_BAR} -> {verdict}")
    if N_EPS == 8:
        print(f"  anchor check: 0.884 expected, got {mean_ret:.3f} "
              f"({'consistent' if abs(mean_ret - ANCHOR_N8) < 0.15 else 'DEVIATES'})")
    print(f"  base-CE residual: {ce_reg:+.3f}% with p_any_plug "
          f"{diag['p_any_plug_in_top2']:.3f} (shape does not touch occupancy)")

    out = {
        "gate": f"G3-shaped-N{N_EPS}",
        "verdict": verdict,
        "n_episodes": N_EPS,
        "fixture": f"memory_shaping.SHAPED_{N_EPS} (all token-substitution deltas)",
        "shape_report": report,
        "recipe": {"cons": CONS_STEPS, "recal": RECAL_STEPS,
                   "joint": FINAL_JOINT, "text_kl": TEXT_KL, "owner_mass": 0.55},
        "standing_bar": {"metric": "margin retention vs solo", "bar": ms.RETENTION_BAR,
                         "anchor_n8": ANCHOR_N8},
        "pre_library": {str(j): pre[j] for j in pre},
        "solo": {str(j): solo[j] for j in solo},
        "library": {str(j): lib[j] for j in live},
        "insert_trajectory": traj,
        "summary": {
            "mean_retention_vs_solo": round(mean_ret, 4),
            "mean_gain_logit": round(sum(gains) / len(gains), 4),
            "util_min": round(util_min, 4),
            "contam_max": round(contam_max, 4),
            "base_ce": {"pre": round(ce_pre, 4), "lib": round(ce_lib, 4),
                        "regression_pct": round(ce_reg, 3)},
            "occupancy": {"p_any_plug_in_top2": round(diag["p_any_plug_in_top2"], 4),
                          "p_both_plug_in_top2": round(diag["p_both_plug_in_top2"], 4)},
            "bank_norms": cos_lib,
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
