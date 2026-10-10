"""Minimal-fix probe for the 8-base delivery collapse (g3_delivery_diag
conclusion): experts are fine (forced-owner = 1.14x solo) but natural
routing keeps the owner below the delivery cliff at the SCORING positions
(P8 util/weight at divergence: 0.25-0.50 / 0.38-0.55 on the failing
episodes; P7 cliff sits between owner weight 0.45 and 0.65 on lab_tiny).

The recipe change under test: the closing joint mex recal should
  (i)  use owner_mass above the cliff (0.85, not 0.55), and
  (ii) cover the SCORING positions (prompt+stated[:first-divergence],
       exactly the prefixes delta_margin reads) in its home data —
       the current home texts are prompts only, so the position where the
       margin is read was never a training target.

Two arms on the saved t0 8-base library state (router-only, cheap):
  A  owner_mass 0.85 + scoring-prefix home data   (the proposed fix)
  B  owner_mass 0.85 + original prompt home data  (mass-only control)

Retention >= 0.70 on arm A names the minimal recipe change.

Env: G3F_STATE (lab/sandbox/g3_shaped/n8_8base_state.pt),
     G3F_SOLO_JSON (lab/results/g3_shaped_n8_8base.json),
     G3F_OUT (lab/results/g3_owner_mass_fix.json),
     G3F_STEPS (800), G3F_MASS (0.85), G3F_DEVICE, G3F_SEED (0).
"""
from __future__ import annotations

import json
import os
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
import g3_family_ab as fam  # noqa: E402
from consolidate import load_base_model  # noqa: E402

CKPT_8BASE = os.path.join(
    REPO, "lab", "sandbox", "t0-8expert-realdata",
    "vesper_linear_checkpoints_lab_tiny", "step_best")
DEVICE = os.environ.get("G3F_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
fam.DEVICE = DEVICE
restore_library = fam.restore_library


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


STATE = _abs(os.environ.get("G3F_STATE") or
             os.path.join("lab", "sandbox", "g3_shaped",
                          "n8_8base_state.pt"))
SOLO_JSON = _abs(os.environ.get("G3F_SOLO_JSON") or
                 os.path.join("lab", "results", "g3_shaped_n8_8base.json"))
OUT = _abs(os.environ.get("G3F_OUT") or
           os.path.join("lab", "results", "g3_owner_mass_fix.json"))
STEPS = int(os.environ.get("G3F_STEPS", "800"))
MASS = float(os.environ.get("G3F_MASS", "0.85"))
SEED = 0
RET_BAR = ms.RETENTION_BAR


def scoring_prefixes(tok, ep):
    """The exact prefixes delta_margin reads (prompt+stated[:divergence])."""
    out = []
    for d in ep.deltas:
        ps = (tok(d.prompt, add_special_tokens=False).input_ids
              + tok(d.stated, add_special_tokens=False).input_ids)
        pc = (tok(d.prompt, add_special_tokens=False).input_ids
              + tok(d.correction, add_special_tokens=False).input_ids)
        k = 0
        while k < min(len(ps), len(pc)) and ps[k] == pc[k]:
            k += 1
        if 0 < k < len(ps) and k < len(pc):
            out.append(torch.tensor([ps[:k]], dtype=torch.long, device=DEVICE))
    return out


def load_state():
    torch.manual_seed(SEED)
    model, tok, mc = load_base_model(CKPT_8BASE, device=DEVICE)
    g1.DIM = int(mc["dim"]); g1.HIDDEN = int(mc["hidden_dim"])
    g1.N_LAYERS = int(mc["n_layers"]); g1.BASE_EXPERTS = int(mc["num_experts"])
    g1.prepare_routers(model)
    sd = torch.load(STATE, map_location="cpu", weights_only=False)
    n_mem = restore_library(model, sd["state_dict"])
    model.eval()
    return model, tok, sd, n_mem


def measure(model, tok, episodes, refj):
    solo = {int(k): v for k, v in refj.get("solo", {}).items()}
    pre_ref = {int(k): v for k, v in refj.get("pre_library", {}).items()}
    ret, marg = {}, {}
    for j, ep in enumerate(episodes):
        m = ms.episode_margins(model, tok, ep, DEVICE)
        marg[ep.name] = m["mean_logit_margin"]
        s = solo.get(j)
        if s and s["gain_logit"]:
            g_lib = marg[ep.name] - pre_ref[j]["mean_logit_margin"]
            ret[ep.name] = round(g_lib / s["gain_logit"], 4)
    vals = list(ret.values())
    return marg, ret, (sum(vals) / len(vals) if vals else None)


def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"owner-mass fix probe  state={STATE}  mass={MASS}  "
          f"steps={STEPS}  device={DEVICE}")
    refj = json.load(open(SOLO_JSON))
    out = {"gate": "G3-owner-mass-fix", "state": STATE, "mass": MASS,
           "steps": STEPS}

    # baseline (as saved)
    model, tok, sd, n_mem = load_state()
    names = sd.get("episodes", [e.name for e in ms.SHAPED_8])
    episodes = [e for e in ms.SHAPED_8 if e.name in names]
    PAD_ID = tok.pad_token_id
    BASE_MIX = g1.load_base_mix_chunks(n_chunks=32, seq=256, seed=SEED)
    marg, ret, mean_ret = measure(model, tok, episodes, refj)
    ce0 = g1.next_token_ce(model, BASE_MIX)
    print(f"  baseline        ret {mean_ret:.3f}  CE {ce0:.4f}")
    out["baseline"] = {"mean_retention": round(mean_ret, 4),
                       "retention": ret, "ce": round(ce0, 4)}
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    arms = {
        "A_mass085_scoring_home": ("scoring", MASS),
        "B_mass085_prompt_home": ("prompt", MASS),
        "C_mass055_scoring_home": ("scoring", 0.55),
    }
    for arm, (home_mode, mass) in arms.items():
        model, tok, sd, n_mem = load_state()
        names = sd.get("episodes", [e.name for e in ms.SHAPED_8])
        episodes = [e for e in ms.SHAPED_8 if e.name in names]
        PAD_ID = tok.pad_token_id
        BASE_MIX = g1.load_base_mix_chunks(n_chunks=32, seq=256, seed=SEED)
        if home_mode == "scoring":
            home_lists = [scoring_prefixes(tok, ep) for ep in episodes]
        else:
            home_lists = [ms.episode_home_ids(tok, ep, DEVICE)
                          for ep in episodes]
        g3.OWNER_MASS = mass
        print(f"  [{arm}] recal owner_mass={mass} home={home_mode} "
              f"({sum(len(h) for h in home_lists)} prefixes)")
        g3.recal_mex(model, home_lists, BASE_MIX, PAD_ID,
                     steps=STEPS, lr=1e-2, device=DEVICE)
        marg, ret, mean_ret = measure(model, tok, episodes, refj)
        ce = g1.next_token_ce(model, BASE_MIX)
        ok = mean_ret is not None and mean_ret >= RET_BAR
        print(f"  [{arm}] ret {mean_ret:.3f}  CE {ce:.4f}  "
              f"{'PASS' if ok else 'fail'}")
        out[arm] = {"mean_retention": (None if mean_ret is None
                                      else round(mean_ret, 4)),
                    "retention": ret, "ce": round(ce, 4),
                    "pass_retention_bar": bool(ok)}
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    best = max((v.get("mean_retention") or 0, k) for k, v in out.items()
               if isinstance(v, dict) and "mean_retention" in v)
    if best[0] >= RET_BAR:
        out["verdict"] = (f"{best[1]} reaches retention {best[0]:.3f} >= "
                          f"{RET_BAR} — minimal recipe change: closing "
                          "joint recal with owner_mass above the delivery "
                          "cliff and scoring-position home coverage")
        out["signal"] = "FIX_FOUND"
    else:
        out["verdict"] = (f"best arm {best[1]} only reaches {best[0]:.3f} "
                          f"(< {RET_BAR}) — recal-side fix insufficient")
        out["signal"] = "FIX_INSUFFICIENT"
    print(f"\n  SIGNAL: {out['signal']} — {out['verdict']}")
    out["seconds"] = round(time.time() - t0, 1)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")
    print(f"total time: {out['seconds']}s")


if __name__ == "__main__":
    main()
