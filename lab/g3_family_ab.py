#!/usr/bin/env python3
"""Family-row two-stage routing A/B — the MODULAR_MOE.md §8.4 reserve sketch.

The shaped-memory G3 runs leave one open residual: base-mix CE regression from
GROUP top-2 occupancy (N plug-in rows × ~0.10 contam each all enter the flat
top-2-of-E contest on base tokens, displacing base experts — +2.5-4% CE even
with clean per-expert contamination).  The reserve sketch proposes two-stage
routing: a FAMILY row per domain cluster contests stage 1 against the base
rows; only the winning family's best MEMORIEs row takes the stage-1 slot
(stage 2 = argmax within family).  Base tokens then contest 4 base + K family
rows instead of 4 base + N memory rows.

A/B on one fixed N=8 shaped library state (no retraining — pure inference-time
routing swap isolates the addressing mechanism):

  A  flat PassportRouter top-2-of-(4+N)   (the measured baseline)
  B  FamilyRouter two-stage               (4 base + K=4 families → members)

Deliverable is a signal either way, not a gold-plated build:
  B drops p_any_plug / base-CE materially vs A → family rows address the
    occupancy residual; worth calibrating family rows and testing at N=16.
  B does not → two-stage addressing alone is insufficient; the residual needs
    base-neutral weighting (mix-preserving experts) or a different fix.

Families: average-linkage clustering of member row vectors (concatenated
across layers, per-layer L2-normalized) into K=4; family rows = mean of
members' rows per layer (the sketch's prototype rule).  No training.

Env: G3F_STATE (lab/sandbox/g3_shaped/n8_shaped_state.pt),
     G3F_SOLO_JSON (lab/results/g3_shaped_n8.json),
     G3F_OUT (lab/results/g3_family_ab.json), G3F_K (4), G3F_DEVICE
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

CKPT = os.path.join(
    REPO, "lab", "sandbox", "t1-diag-realdata-mb4",
    "vesper_linear_checkpoints_lab_small", "step_best")
DEVICE = os.environ.get("G3F_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
def _abs(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


STATE = _abs(os.environ.get("G3F_STATE") or
             os.path.join("lab", "sandbox", "g3_shaped", "n8_shaped_state.pt"))
SOLO_JSON = _abs(os.environ.get("G3F_SOLO_JSON") or
                 os.path.join("lab", "results", "g3_shaped_n8.json"))
OUT = _abs(os.environ.get("G3F_OUT") or
           os.path.join("lab", "results", "g3_family_ab.json"))
K_FAMILIES = int(os.environ.get("G3F_K", "4"))
SEED = 0


class FamilyRouter(nn.Module):
    """Two-stage router with the PassportRouter call signature.

    Stage 1: base rows + family rows contest top-2 (softmax over 4+K).
    Stage 2: a selected family slot is replaced by its best member row
    (argmax member logit); stage-1 weights renormalize over the 2 slots.

    `families` holds MEMBER-LOCAL indices (0..n_mem-1, as returned by
    cluster_families); family rows are means of the corresponding member
    rows of the full bank, and stage-2 emits GLOBAL expert ids (n_base+local).
    """

    def __init__(self, query, passports, families, top_k):
        super().__init__()
        self.query = query
        self.passports = passports          # (E, d) full bank [base | members]
        self.families = families            # list of lists of member-local ids
        self.top_k = top_k
        self.passport_dim = passports.shape[1]
        self.num_experts = passports.shape[0]
        self.n_base = self.num_experts - sum(len(f) for f in families)
        fam = torch.stack([
            passports.data[[self.n_base + m for m in mem]].mean(dim=0)
            for mem in families])
        self.register_buffer("fam_rows", fam)

    def forward(self, x):
        d = self.passport_dim
        q = self.query(x)
        scale = math.sqrt(d)
        base_logits = q @ self.passports[:self.n_base].t() / scale
        fam_logits = q @ self.fam_rows.t() / scale
        stage1 = torch.cat([base_logits, fam_logits], dim=-1)   # (N, 4+K)
        s_w, s_i = torch.topk(F.softmax(stage1, dim=-1), self.top_k, dim=-1)
        mem_logits = q @ self.passports[self.n_base:].t() / scale  # (N, n_mem)
        out_i = torch.zeros_like(s_i)
        for slot in range(self.top_k):
            cand = s_i[:, slot]
            for f, members in enumerate(self.families):
                sel = cand == (self.n_base + f)
                if not sel.any():
                    continue
                sub = mem_logits[:, torch.tensor(members, device=x.device)]
                out_i[sel, slot] = self.n_base + torch.tensor(
                    members, device=x.device)[sub[sel].argmax(dim=-1)]
            is_base = cand < self.n_base
            out_i[is_base, slot] = cand[is_base]
        s_w = s_w / s_w.sum(dim=-1, keepdim=True)
        return s_w, out_i, x.new_zeros(())


def cluster_families(model, n_mem, k):
    """Average-linkage agglomerative clustering of member rows across layers."""
    vecs = []
    for j in range(n_mem):
        parts = []
        for layer in model.layers:
            P = layer["ffn"].router.passports.data[g1.BASE_EXPERTS:]
            v = P[j]
            parts.append(v / (v.norm() + 1e-8))
        vecs.append(torch.cat([p.cpu() for p in parts]))
    V = torch.stack(vecs)
    S = (V @ V.t()).abs()
    groups = [[i] for i in range(n_mem)]
    sims = S.clone()
    while len(groups) > k:
        best = (-1.0, -1, -1)
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                m = float(sims[groups[a], :][:, groups[b]].mean())
                if m > best[0]:
                    best = (m, a, b)
        _, a, b = best
        groups[a] = groups[a] + groups[b]
        del groups[b]
    return [sorted(g) for g in groups]


def restore_library(model, sd):
    """Pre-expand the bare spine to the saved library shape, then load.

    The saved state_dict carries plugged experts + expanded passports; the
    freshly loaded spine has 4 experts and (4, d) rows.  Discover the member
    count from the bank shape, plug empty ContractExpert modules + zero rows
    (values overwritten by load_state_dict), then load strict.
    """
    n_rows = sd["layers.0.ffn.router.passports"].shape[0]
    n_mem = n_rows - g1.BASE_EXPERTS
    for _ in range(n_mem):
        experts = [g1.ContractExpert(g1.DIM, g1.HIDDEN).to(DEVICE)
                   for _ in range(g1.N_LAYERS)]
        rows = [torch.zeros(g1.DIM if False else
                            model.layers[0]["ffn"].router.passports.shape[1],
                            device=DEVICE)
                for _ in range(g1.N_LAYERS)]
        g1.plug_expert(model, experts, rows)
    model.load_state_dict(sd)
    return n_mem


def swap_router(model, families):
    """Install FamilyRouter in every layer; return flat routers."""
    saved = []
    for layer in model.layers:
        ffn = layer["ffn"]
        saved.append(ffn.router)
        ffn.router = FamilyRouter(ffn.router.query, ffn.router.passports,
                                  families, ffn.router.top_k).to(DEVICE)
    return saved


def restore_routers(model, saved):
    for layer, r in zip(model.layers, saved):
        layer["ffn"].router = r


@torch.no_grad()
def occupancy(model, base_chunks, plug_rows, pad_id):
    _, diag = g3.routing_matrix(model, [[c] for c in base_chunks[:8]],
                                plug_rows, pad_id)
    return diag


def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    random_seed = SEED
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    print(f"family-row A/B  state={STATE}  K={K_FAMILIES}  device={DEVICE}")

    model, tok, mc = load_base_model(CKPT, device=DEVICE)
    g1.DIM = int(mc["dim"]); g1.HIDDEN = int(mc["hidden_dim"])
    g1.N_LAYERS = int(mc["n_layers"]); g1.BASE_EXPERTS = int(mc["num_experts"])
    g1.prepare_routers(model)
    sd = torch.load(STATE, map_location="cpu", weights_only=False)
    n_mem_loaded = restore_library(model, sd["state_dict"])
    model.eval()
    PAD_ID = tok.pad_token_id
    names = sd.get("episodes", [e.name for e in ms.SHAPED_8])
    episodes = [e for e in ms.SHAPED_8 if e.name in names]
    n_mem = model.layers[0]["ffn"].router.passports.shape[0] - g1.BASE_EXPERTS
    plug_rows = list(range(g1.BASE_EXPERTS, g1.BASE_EXPERTS + n_mem))
    BASE_MIX = g1.load_base_mix_chunks(n_chunks=32, seq=256, seed=SEED)

    families = cluster_families(model, n_mem, min(K_FAMILIES, n_mem))
    print(f"  families ({len(families)}): {families}")

    solo = {}
    pre_ref = {}
    if os.path.exists(SOLO_JSON):
        refj = json.load(open(SOLO_JSON))
        solo = {int(k): v for k, v in refj.get("solo", {}).items()}
        pre_ref = {int(k): v for k, v in refj.get("pre_library", {}).items()}

    def margins_now():
        out = {}
        for j, ep in enumerate(episodes):
            m = ms.episode_margins(model, tok, ep, DEVICE)
            out[ep.name] = m["mean_logit_margin"]
        return out

    def occupancy_now():
        return occupancy(model, BASE_MIX, plug_rows, PAD_ID)

    # ---- A: flat (as saved) ----
    print("\n== A: flat top-2 routing (as saved) ==")
    ce_A = g1.next_token_ce(model, BASE_MIX)
    marg_A = margins_now()
    occ_A = occupancy_now()
    print(f"  base CE {ce_A:.4f}  p_any {occ_A['p_any_plug_in_top2']:.3f} "
          f"p_both {occ_A['p_both_plug_in_top2']:.3f}")
    print(f"  margins: " + " ".join(f"{v:+.2f}" for v in marg_A.values()))

    # ---- B: family two-stage (same state, swap routers only) ----
    saved = swap_router(model, families)
    print("\n== B: family two-stage routing (same state) ==")
    ce_B = g1.next_token_ce(model, BASE_MIX)
    marg_B = margins_now()
    occ_B = occupancy_now()
    home_lists = [ms.episode_home_ids(tok, ep, DEVICE) for ep in episodes]
    rows_b = g3.routing_matrix_group(
        model, home_lists, plug_rows, PAD_ID)
    util_B = {episodes[gi].name: round(rows_b[gi][g1.BASE_EXPERTS + gi], 4)
              for gi in range(len(episodes))}
    print(f"  base CE {ce_B:.4f}  p_any {occ_B['p_any_plug_in_top2']:.3f} "
          f"p_both {occ_B['p_both_plug_in_top2']:.3f}")
    print(f"  margins: " + " ".join(f"{v:+.2f}" for v in marg_B.values()))
    print(f"  util_home: {util_B}")

    # retention under both (vs solo refs from the n8 shaped run)
    def ret_table(marg):
        out = {}
        for j, ep in enumerate(episodes):
            s = solo.get(j)
            if not s:
                continue
            if j in pre_ref:
                pre = pre_ref[j]["mean_logit_margin"]
            else:
                pre = s["margins"]["mean_logit_margin"] - s["gain_logit"]
            g_lib = marg[ep.name] - pre
            out[ep.name] = (None if not s["gain_logit"]
                            else round(g_lib / s["gain_logit"], 4))
        return out

    ret_A, ret_B = ret_table(marg_A), ret_table(marg_B)
    print(f"  retention A: {ret_A}")
    print(f"  retention B: {ret_B}")

    ce_dropped = ce_B < ce_A - 0.02 * ce_A
    occ_dropped = occ_B["p_any_plug_in_top2"] < occ_A["p_any_plug_in_top2"] - 0.05
    ret_vals_A = [v for v in ret_A.values() if v is not None]
    ret_vals_B = [v for v in ret_B.values() if v is not None]
    mean_ret_A = sum(ret_vals_A) / len(ret_vals_A) if ret_vals_A else None
    mean_ret_B = sum(ret_vals_B) / len(ret_vals_B) if ret_vals_B else None
    ret_hurt = (mean_ret_A is not None and mean_ret_B is not None
                and mean_ret_B < 0.5 * mean_ret_A)
    if (ce_dropped or occ_dropped) and not ret_hurt:
        sig = "POSITIVE"
        verdict = ("family rows address the occupancy residual without "
                   "hurting retention — calibrate family rows and test at N=16")
    elif ce_dropped or occ_dropped:
        sig = "MIXED"
        verdict = ("CE/occupancy improve but memory delivery collapses — "
                   "family argmax needs calibration before it counts")
    else:
        sig = "NEGATIVE"
        verdict = ("two-stage addressing alone insufficient — residual needs "
                   "base-neutral weighting or another fix")
    print(f"\n  SIGNAL: {sig} — {verdict}")
    print(f"  CE A {ce_A:.4f} -> B {ce_B:.4f} "
          f"({100*(ce_B-ce_A)/ce_A:+.2f}% vs flat); "
          f"p_any A {occ_A['p_any_plug_in_top2']:.3f} -> B {occ_B['p_any_plug_in_top2']:.3f}; "
          f"retention A {mean_ret_A} -> B {mean_ret_B}")

    out = {
        "gate": "G3-family-ab",
        "signal": sig,
        "verdict": verdict,
        "state": STATE,
        "families": families,
        "k_families": len(families),
        "ce": {"flat": round(ce_A, 4), "family": round(ce_B, 4),
               "delta_pct": round(100 * (ce_B - ce_A) / ce_A, 3)},
        "occupancy": {"flat": occ_A, "family": occ_B},
        "margins": {"flat": marg_A, "family": marg_B},
        "util_home_family": util_B,
        "retention": {"flat": ret_A, "family": ret_B},
        "note": ("pure inference-time routing swap on one fixed N=8 shaped "
                 "library state — no training, isolates the addressing "
                 "mechanism; family rows are untrained means of member rows "
                 "(the sketch's prototype rule)"),
        "seconds": round(time.time() - t0, 1),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")
    print(f"total time: {out['seconds']}s")


if __name__ == "__main__":
    main()
