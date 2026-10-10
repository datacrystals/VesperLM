"""Branch (a) — family-row mex calibration (owner=family).

Same-cycle fallback for the base-CE residual after branch (b) (plug-logit
bias sweep, g3_plug_bias_ab.py) came back NEGATIVE: the bias trade-off never
satisfies both bars (retention-feasible betas still regress base CE +6.7%;
CE-feasible betas kill retention).  This branch trains what (b) could only
nudge: the FAMILY rows of the two-stage router (g3_family_ab's fixed
FamilyRouter) with the section-4.4a mutual-exclusion target at the family
level —

  home tokens of family f  -> family row f gets OWNER_MASS, base rows share
                              (1-OWNER_MASS), other family rows 0
  base-mix tokens          -> base rows share 1.0, family rows 0

so base tokens contest only n_base + K rows and the family rows learn to be
silent off-home.  Stage 2 (argmax member within the routed family) is left
alone — one variable.  Only fam_rows train; base rows, member rows, experts,
query and spine stay frozen.

Pass bar (same as branch b): base-CE regression < 1% AND N=8 shaped
retention vs solo >= 0.70 on the saved state lab/sandbox/g3_shaped/
n8_shaped_state.pt (solo refs from lab/results/g3_shaped_n8.json).

Env: G3C_STATE, G3C_SOLO_JSON, G3C_OUT (lab/results/g3_family_cal.json),
     G3C_K (4), G3C_STEPS (800), G3C_LR (1e-2), G3C_OWNER_MASS (0.55),
     G3C_DEVICE, G3C_SEED (0).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

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
DEVICE = os.environ.get("G3C_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
fam.DEVICE = DEVICE
restore_library = fam.restore_library


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


STATE = _abs(os.environ.get("G3C_STATE") or
             os.path.join("lab", "sandbox", "g3_shaped", "n8_shaped_state.pt"))
SOLO_JSON = _abs(os.environ.get("G3C_SOLO_JSON") or
                 os.path.join("lab", "results", "g3_shaped_n8.json"))
OUT = _abs(os.environ.get("G3C_OUT") or
           os.path.join("lab", "results", "g3_family_cal.json"))
K_FAMILIES = int(os.environ.get("G3C_K", "4"))
CAL_STEPS = int(os.environ.get("G3C_STEPS", "800"))
CAL_LR = float(os.environ.get("G3C_LR", "1e-2"))
OWNER_MASS = float(os.environ.get("G3C_OWNER_MASS", "0.55"))
SEED = 0
RET_BAR = ms.RETENTION_BAR          # 0.70
CE_BAR_PCT = 1.0


@torch.no_grad()
def occupancy_family(model, base_chunks, member_rows, families, pad_id):
    """Two-stage occupancy: p_any/p_both of MEMBER rows in the resolved
    top-2 (family slot -> argmax member), on base-mix tokens."""
    hs, _ = g3.capture_grouped(model, [[c] for c in base_chunks], pad_id)
    n_layers = len(model.layers)
    n_base = g1.BASE_EXPERTS
    anyp, bothp = [], []
    for li, layer in enumerate(model.layers):
        r = layer["ffn"].router          # FamilyRouter
        h = hs[li]
        q = r.query(h)
        scale = math.sqrt(r.passport_dim)
        base_logits = q @ r.passports[:n_base].t() / scale
        fam_logits = q @ r.fam_rows.t() / scale
        stage1 = torch.cat([base_logits, fam_logits], dim=-1)
        _, s_i = torch.topk(F.softmax(stage1, dim=-1), r.top_k, dim=-1)
        mem_logits = q @ r.passports[n_base:].t() / scale
        out_i = s_i.clone()
        for slot in range(r.top_k):
            cand = s_i[:, slot]
            for f, members in enumerate(families):
                sel = cand == (n_base + f)
                if not sel.any():
                    continue
                sub = mem_logits[:, torch.tensor(members, device=h.device)]
                out_i[sel, slot] = n_base + torch.tensor(
                    members, device=h.device)[sub[sel].argmax(dim=-1)]
        plug = torch.tensor(member_rows, dtype=torch.long, device=h.device)
        pmask = (out_i.unsqueeze(-1) == plug).any(-1)
        anyp.append(float(pmask.any(dim=-1).float().mean()))
        bothp.append(float(pmask.all(dim=-1).float().mean()))
    return {"p_any_plug_in_top2": round(sum(anyp) / n_layers, 4),
            "p_both_plug_in_top2": round(sum(bothp) / n_layers, 4)}


@torch.no_grad()
def util_home_family(model, home_lists, member_rows, families, pad_id):
    """P(own member row delivered in resolved top-2) per home group."""
    hs, slices = g3.capture_grouped(model, home_lists, pad_id)
    n_layers = len(model.layers)
    n_base = g1.BASE_EXPERTS
    out = []
    for g in range(len(home_lists)):
        vals = []
        for li, layer in enumerate(model.layers):
            r = layer["ffn"].router
            h = hs[li][slices[g][0]:slices[g][1]]
            q = r.query(h)
            scale = math.sqrt(r.passport_dim)
            stage1 = torch.cat([q @ r.passports[:n_base].t() / scale,
                                q @ r.fam_rows.t() / scale], dim=-1)
            _, s_i = torch.topk(F.softmax(stage1, dim=-1), r.top_k, dim=-1)
            mem_logits = q @ r.passports[n_base:].t() / scale
            out_i = s_i.clone()
            for slot in range(r.top_k):
                cand = s_i[:, slot]
                for f, members in enumerate(families):
                    sel = cand == (n_base + f)
                    if not sel.any():
                        continue
                    sub = mem_logits[:, torch.tensor(members, device=h.device)]
                    out_i[sel, slot] = n_base + torch.tensor(
                        members, device=h.device)[sub[sel].argmax(dim=-1)]
            vals.append(float((out_i == member_rows[g]).any(dim=-1)
                              .float().mean()))
        out.append(sum(vals) / n_layers)
    return out


def recal_family_mex(model, home_lists_by_family, families, base_chunks,
                     pad_id, *, steps, lr, device):
    """Train fam_rows only with the family-level mex target (see module
    docstring).  Member rows / base rows / query frozen by construction —
    only fam_rows tensors are in the optimizer."""
    trainable = []
    for layer in model.layers:
        r = layer["ffn"].router
        r.fam_rows.requires_grad_(True)
        trainable.append(r.fam_rows)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    n_base = g1.BASE_EXPERTS
    k_fam = len(families)
    gen = torch.Generator().manual_seed(1)
    t0 = time.time()
    for step in range(steps):
        bsel = [int(torch.randint(0, len(base_chunks), (1,), generator=gen)
                    .item()) for _ in range(4)]
        opt.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        hs_b, _ = g3.capture_grouped(
            model, [[base_chunks[i]] for i in bsel], pad_id)
        for li, layer in enumerate(model.layers):
            r = layer["ffn"].router
            q = r.query(hs_b[li])
            stage1 = torch.cat(
                [q @ r.passports[:n_base].t() / math.sqrt(r.passport_dim),
                 q @ r.fam_rows.t() / math.sqrt(r.passport_dim)], dim=-1)
            tgt = torch.zeros(stage1.size(0), n_base + k_fam, device=device)
            tgt[:, :n_base] = 1.0 / n_base
            loss = loss + -(tgt * F.log_softmax(stage1, dim=-1)
                            ).sum(dim=-1).mean()
        for f, ids in enumerate(home_lists_by_family):
            hs_h, slices = g3.capture_grouped(model, [ids], pad_id)
            for li, layer in enumerate(model.layers):
                r = layer["ffn"].router
                q = r.query(hs_h[li])
                stage1 = torch.cat(
                    [q @ r.passports[:n_base].t()
                     / math.sqrt(r.passport_dim),
                     q @ r.fam_rows.t() / math.sqrt(r.passport_dim)],
                    dim=-1)
                tgt = torch.zeros(stage1.size(0), n_base + k_fam,
                                  device=device)
                tgt[:, :n_base] = (1.0 - OWNER_MASS) / n_base
                tgt[:, n_base + f] = OWNER_MASS
                loss = loss + -(tgt * F.log_softmax(stage1, dim=-1)
                                ).sum(dim=-1).mean()
        loss.backward()
        opt.step()
        if step % 133 == 0 or step == steps - 1:
            print(f"  [fam-cal] step {step:4d}  loss {float(loss):+.4f}")
    return {"seconds": round(time.time() - t0, 1), "steps": steps,
            "lr": lr, "owner_mass": OWNER_MASS}


def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"family-cal  state={STATE}  K={K_FAMILIES}  steps={CAL_STEPS}  "
          f"device={DEVICE}")

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
    n_base = g1.BASE_EXPERTS
    member_rows = list(range(n_base, n_base + n_mem))
    BASE_MIX = g1.load_base_mix_chunks(n_chunks=32, seq=256, seed=SEED)

    refj = json.load(open(SOLO_JSON))
    solo = {int(k): v for k, v in refj.get("solo", {}).items()}
    pre_ref = {int(k): v for k, v in refj.get("pre_library", {}).items()}
    ce_pre = float(refj.get("summary", {}).get("base_ce", {}).get("pre", 0)
                   or 6.7735)

    families = fam.cluster_families(model, n_mem, min(K_FAMILIES, n_mem))
    print(f"  families ({len(families)}): {families}")
    home_lists = [ms.episode_home_ids(tok, ep, DEVICE) for ep in episodes]
    home_by_family = []
    for members in families:
        merged = []
        for m in members:
            merged.extend(home_lists[m])
        home_by_family.append(merged)

    # as-saved flat CE (pre-calibration reference)
    ce_flat = g1.next_token_ce(model, BASE_MIX)
    print(f"  as-saved flat CE {ce_flat:.4f} (pre-library {ce_pre:.4f})")

    # install two-stage routers and calibrate fam_rows
    fam.swap_router(model, families)
    rstat = recal_family_mex(model, home_by_family, families, BASE_MIX,
                             PAD_ID, steps=CAL_STEPS, lr=CAL_LR,
                             device=DEVICE)

    ce = g1.next_token_ce(model, BASE_MIX)
    reg = 100.0 * (ce - ce_pre) / ce_pre
    marg = {ep.name: ms.episode_margins(model, tok, ep, DEVICE)
            ["mean_logit_margin"] for ep in episodes}
    ret = {}
    for j, ep in enumerate(episodes):
        s = solo.get(j)
        if not s:
            continue
        g_lib = marg[ep.name] - pre_ref[j]["mean_logit_margin"]
        ret[ep.name] = (None if not s["gain_logit"]
                        else round(g_lib / s["gain_logit"], 4))
    vals = [v for v in ret.values() if v is not None]
    mean_ret = sum(vals) / len(vals) if vals else None
    occ = occupancy_family(model, BASE_MIX[:8], member_rows, families, PAD_ID)
    util = util_home_family(model, home_lists, member_rows, families, PAD_ID)

    ok = (reg < CE_BAR_PCT) and (mean_ret is not None
                                 and mean_ret >= RET_BAR)
    sig = "POSITIVE" if ok else "NEGATIVE"
    if ok:
        verdict = ("family-row mex calibration passes both bars: "
                   f"CE {reg:+.2f}% (bar <1%), retention {mean_ret:.3f} "
                   f"(bar 0.70) at p_any {occ['p_any_plug_in_top2']:.3f}")
    elif mean_ret is not None and mean_ret >= RET_BAR:
        verdict = ("retention holds ({:.3f}) but base CE still regresses "
                   "{:+.2f}% (bar <1%) at p_any {:.3f} — trained family "
                   "rows reduce but do not close the occupancy residual; "
                   "next branch: base-neutral expert re-consolidation or "
                   "per-family admission margins".format(
                       mean_ret, reg, occ["p_any_plug_in_top2"]))
    else:
        verdict = ("family-row calibration fails the retention bar "
                   f"({mean_ret}); the two-stage delivery path costs too "
                   "much even with trained family rows")

    print(f"\n  CE {ce:.4f} ({reg:+.3f}%)  p_any {occ['p_any_plug_in_top2']}  "
          f"p_both {occ['p_both_plug_in_top2']}  ret {mean_ret}  "
          f"util_min {min(util):.3f}")
    print(f"  SIGNAL: {sig} — {verdict}")

    out = {"gate": "G3-family-cal",
           "signal": sig,
           "verdict": verdict,
           "state": STATE,
           "solo_json": SOLO_JSON,
           "families": families,
           "calibration": rstat,
           "ce": {"pre": round(ce_pre, 4), "flat_as_saved": round(ce_flat, 4),
                  "family_cal": round(ce, 4),
                  "regression_pct": round(reg, 3)},
           "occupancy": occ,
           "margins": marg,
           "retention": ret,
           "mean_retention": (None if mean_ret is None
                              else round(mean_ret, 4)),
           "util_home": {episodes[i].name: round(util[i], 4)
                         for i in range(len(episodes))},
           "bars": {"base_ce_regression_pct": CE_BAR_PCT,
                    "retention_vs_solo": RET_BAR},
           "note": ("family rows trained with the section-4.4a mex target at "
                    "family granularity (owner = family); stage-2 argmax "
                    "member selection unchanged; base/member rows, experts "
                    "and spine frozen"),
           "seconds": round(time.time() - t0, 1)}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")
    print(f"total time: {out['seconds']}s")


if __name__ == "__main__":
    main()
