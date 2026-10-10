"""Delivery-collapse diagnostic — why do shaped memories deliver on the
lab 4-base spine (retention 1.07) but collapse on the real 8-base spine
(t0-8expert-realdata: retention 0.397 with util 0.47-0.97)?

Free evidence from the run records: solo gains on the 8-base spine are
STRONG (+10.7..+16.5 for 7/8 episodes) while library gains collapse to
+0.9..+4.7 — the rows route (util high) but the mix does not deliver.
So the question is which mechanism eats the delivery. Discriminating
probes per episode, all on the SAVED library state (inference only):

  P1  as-is margin            — reproduce the run's lib margin
  P2  forced-owner margin     — router forced to owner row, weight 1.0
                                (pure expert capability at N=8)
                                P2 >> P1  => the expert is fine, the MIX
                                eats delivery (weights / wrong memory)
                                P2 ~ P1   => the expert itself is broken
                                (consolidation / recipe / capacity)
  P3  owner weight in mix     — mean w_owner when owner is in top-2, plus
                                P(both slots are plug rows) on home tokens
                                and P(owner shares with a WRONG memory)
                                high P3c + low w_owner => mix dilution
  P4  cosine-routing margin   — score with L2-normalized rows (content
                                routing) instead of raw dot (norm-driven)
                                P4 >> P1 => (c) norm/temperature mismatch

Control: same probes on the lab_small 4-base state (the PASSING spine).
The dominant gap (P2-P1 across spines, or P1 vs P4) names the cause; one
number per hypothesis.

Env: G3X_STATES (comma list of state paths),
     G3X_SOLO_JSONS (comma list, matching order),
     G3X_LABELS (comma list), G3X_OUT (lab/results/g3_delivery_diag.json),
     G3X_DEVICE, G3X_SEED (0).
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
import g3_family_ab as fam  # noqa: E402
from consolidate import load_base_model  # noqa: E402

CKPT_SMALL = os.path.join(
    REPO, "lab", "sandbox", "t1-diag-realdata-mb4",
    "vesper_linear_checkpoints_lab_small", "step_best")
DEVICE = os.environ.get("G3X_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
fam.DEVICE = DEVICE
restore_library = fam.restore_library


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


STATES = [_abs(p) for p in (os.environ.get("G3X_STATES") or
          "lab/sandbox/g3_shaped/n8_shaped_state.pt,"
          "lab/sandbox/g3_shaped/n8_8base_state.pt").split(",")]
SOLO_JSONS = [_abs(p) for p in (os.environ.get("G3X_SOLO_JSONS") or
              "lab/results/g3_shaped_n8.json,"
              "lab/results/g3_shaped_n8_8base.json").split(",")]
LABELS = (os.environ.get("G3X_LABELS") or
          "lab_small-4base,lab_tiny-8base").split(",")
OUT = _abs(os.environ.get("G3X_OUT") or
           os.path.join("lab", "results", "g3_delivery_diag.json"))
# spine ckpt per label: 8base states live on t0-8expert-realdata
CKPT_8BASE = os.path.join(
    REPO, "lab", "sandbox", "t0-8expert-realdata",
    "vesper_linear_checkpoints_lab_tiny", "step_best")
SEED = 0


class ForceOwnerRouter(nn.Module):
    """Router that always dispatches to one row with weight 1.0."""

    def __init__(self, query, passports, owner, top_k):
        super().__init__()
        self.query = query
        self.passports = passports
        self.owner = owner
        self.top_k = top_k
        self.passport_dim = passports.shape[1]
        self.num_experts = passports.shape[0]

    def forward(self, x):
        n = x.shape[0]
        w = x.new_zeros(n, self.top_k)
        w[:, 0] = 1.0
        i = torch.full((n, self.top_k), self.owner, dtype=torch.long,
                       device=x.device)
        return w, i, x.new_zeros(())


class MixRouter(nn.Module):
    """Router forced to a fixed two-row mix: w0 on row0, w1 on row1.

    Separates owner-weight dilution (w0 sweep) from wrong-memory poison
    (row1 = another memory row vs row1 = best base)."""

    def __init__(self, query, passports, row0, row1, w0, top_k):
        super().__init__()
        self.query = query
        self.passports = passports
        self.row0 = row0
        self.row1 = row1
        self.w0 = float(w0)
        self.top_k = top_k
        self.passport_dim = passports.shape[1]
        self.num_experts = passports.shape[0]

    def forward(self, x):
        n = x.shape[0]
        w = x.new_zeros(n, self.top_k)
        w[:, 0] = self.w0
        w[:, 1] = 1.0 - self.w0
        i = torch.full((n, self.top_k), self.row0, dtype=torch.long,
                       device=x.device)
        i[:, 1] = self.row1
        return w, i, x.new_zeros(())


class CosineRouter(nn.Module):
    """PassportRouter scored with L2-normalized rows (content, not norm)."""

    def __init__(self, query, passports, top_k):
        super().__init__()
        self.query = query
        self.passports = passports
        self.top_k = top_k
        self.passport_dim = passports.shape[1]
        self.num_experts = passports.shape[0]

    def forward(self, x):
        q = self.query(x)
        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        pn = self.passports / (self.passports.norm(dim=-1, keepdim=True)
                               + 1e-8)
        probs = F.softmax(qn @ pn.t(), dim=-1)
        w, i = torch.topk(probs, self.top_k, dim=-1)
        w = w / w.sum(dim=-1, keepdim=True)
        return w, i, x.new_zeros(())


class OnePlugCapRouter(nn.Module):
    """PassportRouter with at most ONE plug row in top-2 (P5 fix probe).

    If both winning slots are plug rows, keep the better one and evict the
    other in favour of the best base row.  Kills wrong-memory cofire while
    leaving owner+base delivery intact (the lab_small passing regime)."""

    def __init__(self, query, passports, n_base, top_k):
        super().__init__()
        self.query = query
        self.passports = passports
        self.n_base = n_base
        self.top_k = top_k
        self.passport_dim = passports.shape[1]
        self.num_experts = passports.shape[0]

    def forward(self, x):
        logits = self.query(x) @ self.passports.t() / math.sqrt(self.passport_dim)
        probs = F.softmax(logits, dim=-1)
        w, i = torch.topk(probs, self.top_k, dim=-1)
        if self.top_k == 2:
            both_plug = (i[:, 0] >= self.n_base) & (i[:, 1] >= self.n_base)
            if both_plug.any():
                # keep the higher-scoring plug slot; other slot -> best base
                base_best = logits[:, :self.n_base].argmax(dim=-1)
                first_is_better = w[:, 0] >= w[:, 1]
                keep = torch.where(first_is_better, 0, 1)
                evict = 1 - keep
                sel = both_plug.nonzero(as_tuple=True)[0]
                i[sel, evict[sel]] = base_best[sel]
                # recompute weights for the swapped-in base row
                rows = i[sel]
                w[sel] = probs.gather(1, rows)
        w = w / w.sum(dim=-1, keepdim=True)
        return w, i, x.new_zeros(())


def swap(model, ctor):
    saved = []
    for layer in model.layers:
        ffn = layer["ffn"]
        r = ffn.router
        saved.append(r)
        ffn.router = ctor(r).to(DEVICE)
    return saved


def restore(model, saved):
    for layer, r in zip(model.layers, saved):
        layer["ffn"].router = r


@torch.no_grad()
def home_mix_stats(model, home_lists, plug_rows, pad_id, owner_of):
    """P3: owner weight when present; P(both plug); P(owner + wrong memory)."""
    hs, slices = g3.capture_grouped(model, home_lists, pad_id)
    n_layers = len(model.layers)
    w_owner, p_both_plug, p_owner_wrong = [], [], []
    for g in range(len(home_lists)):
        wo, pb, pow_ = [], [], []
        for li, layer in enumerate(model.layers):
            r = layer["ffn"].router
            h = hs[li][slices[g][0]:slices[g][1]]
            logits = r.query(h) @ r.passports.t() / math.sqrt(r.passport_dim)
            probs = F.softmax(logits, dim=-1)
            tw, ti = torch.topk(probs, layer["ffn"].top_k, dim=-1)
            own = owner_of[g]
            in2 = (ti == own).any(dim=-1)
            if in2.any():
                # owner's weight (renormalized over its 2 slots, as in mix)
                wsum = tw.sum(dim=-1)
                wown = torch.where(
                    (ti == own), tw, torch.zeros_like(tw)).sum(dim=-1)
                wo.append(float((wown / wsum)[in2].mean()))
            plug = torch.tensor(plug_rows, dtype=torch.long, device=h.device)
            pmask = (ti.unsqueeze(-1) == plug).any(-1)
            pb.append(float(pmask.all(dim=-1).float().mean()))
            other_plug = (ti >= plug_rows[0]) & (ti != own)
            pow_.append(float((in2 & other_plug.any(dim=-1)).float().mean()))
        w_owner.append(sum(wo) / max(1, len(wo)))
        p_both_plug.append(sum(pb) / n_layers)
        p_owner_wrong.append(sum(pow_) / n_layers)
    return w_owner, p_both_plug, p_owner_wrong


def diag_state(state_path, solo_json, label, ckpt):
    torch.manual_seed(SEED)
    model, tok, mc = load_base_model(ckpt, device=DEVICE)
    g1.DIM = int(mc["dim"]); g1.HIDDEN = int(mc["hidden_dim"])
    g1.N_LAYERS = int(mc["n_layers"]); g1.BASE_EXPERTS = int(mc["num_experts"])
    g1.prepare_routers(model)
    sd = torch.load(state_path, map_location="cpu", weights_only=False)
    n_mem = restore_library(model, sd["state_dict"])
    model.eval()
    PAD_ID = tok.pad_token_id
    names = sd.get("episodes", [e.name for e in ms.SHAPED_8])
    episodes = [e for e in ms.SHAPED_8 if e.name in names]
    plug_rows = list(range(g1.BASE_EXPERTS, g1.BASE_EXPERTS + n_mem))
    refj = json.load(open(solo_json))
    solo = {int(k): v for k, v in refj.get("solo", {}).items()}
    pre_ref = {int(k): v for k, v in refj.get("pre_library", {}).items()}
    home_lists = [ms.episode_home_ids(tok, ep, DEVICE) for ep in episodes]

    rows = {}
    for j, ep in enumerate(episodes):
        owner = g1.BASE_EXPERTS + j
        # P1 as-is
        m1 = ms.episode_margins(model, tok, ep, DEVICE)
        # P3 mix stats on home tokens
        wo, pb, pow_ = home_mix_stats(model, [home_lists[j]], plug_rows,
                                      PAD_ID, [owner])
        # P2 forced owner (pure expert capability)
        saved = swap(model, lambda r, o=owner: ForceOwnerRouter(
            r.query, r.passports, o, r.top_k))
        m2 = ms.episode_margins(model, tok, ep, DEVICE)
        restore(model, saved)
        # P4 cosine routing
        saved = swap(model, lambda r: CosineRouter(r.query, r.passports,
                                                   r.top_k))
        m4 = ms.episode_margins(model, tok, ep, DEVICE)
        restore(model, saved)
        # P5 one-plug-cap fix probe
        nb = g1.BASE_EXPERTS
        saved = swap(model, lambda r, _nb=nb: OnePlugCapRouter(
            r.query, r.passports, _nb, r.top_k))
        m5 = ms.episode_margins(model, tok, ep, DEVICE)
        restore(model, saved)
        # P6/P7 forced-mix probes: best base and worst competitor rows on home
        hs, slices = g3.capture_grouped(model, [home_lists[j]], PAD_ID)
        r0 = model.layers[0]["ffn"].router
        h0 = hs[0]
        lg0 = (r0.query(h0) @ r0.passports.t()
               / math.sqrt(r0.passport_dim)).mean(dim=0)
        best_base = int(lg0[:nb].argmax())
        cand = lg0.clone()
        cand[owner] = -1e9
        cand[:nb] = -1e9
        wrong = int(cand.argmax())     # strongest non-owner plug row on home

        def mix_margin(row0, row1, w0):
            saved = swap(model, lambda r, a=row0, b=row1, c=w0: MixRouter(
                r.query, r.passports, a, b, c, r.top_k))
            m = ms.episode_margins(model, tok, ep, DEVICE)
            restore(model, saved)
            return m["mean_logit_margin"] - pre_ref[j]["mean_logit_margin"]

        # P8: routing AT the divergence positions (where the margin is read)
        dec_util, dec_w = [], []
        for d in ep.deltas:
            ps = (tok(d.prompt, add_special_tokens=False).input_ids
                  + tok(d.stated, add_special_tokens=False).input_ids)
            pc = (tok(d.prompt, add_special_tokens=False).input_ids
                  + tok(d.correction, add_special_tokens=False).input_ids)
            k = 0
            while k < min(len(ps), len(pc)) and ps[k] == pc[k]:
                k += 1
            if k >= len(ps) or k == 0:
                continue
            x = torch.tensor([ps[:k]], dtype=torch.long, device=DEVICE)
            h_last = []
            handles = []

            def pre(_m, _a):
                h_last.append(_a[0][:, -1])
            for layer in model.layers:
                handles.append(layer["ffn"].register_forward_pre_hook(pre))
            with torch.no_grad():
                model(x)
            for hdl in handles:
                hdl.remove()
            utils, ws = [], []
            for li, layer in enumerate(model.layers):
                r_ = layer["ffn"].router
                q = r_.query(h_last[li])
                lg = q @ r_.passports.t() / math.sqrt(r_.passport_dim)
                probs = F.softmax(lg, dim=-1)
                tw, ti = torch.topk(probs, r_.top_k, dim=-1)
                in2 = (ti == owner).any(dim=-1)
                utils.append(float(in2.float().mean()))
                if in2.any():
                    wsum = tw.sum(dim=-1)
                    wown = torch.where((ti == owner), tw,
                                       torch.zeros_like(tw)).sum(dim=-1)
                    ws.append(float((wown / wsum)[in2].mean()))
            dec_util.append(sum(utils) / len(utils))
            if ws:
                dec_w.append(sum(ws) / len(ws))
        p8_util = (round(sum(dec_util) / len(dec_util), 4) if dec_util
                   else None)
        p8_w = round(sum(dec_w) / len(dec_w), 4) if dec_w else None

        pre = pre_ref[j]["mean_logit_margin"]
        g6a = mix_margin(owner, best_base, 0.65)   # dilution control
        g6b = mix_margin(owner, wrong, 0.65)       # wrong-memory poison
        g7 = {a: mix_margin(owner, best_base, a) for a in (0.45, 0.65, 0.85)}
        g1m = m1["mean_logit_margin"] - pre
        g2m = m2["mean_logit_margin"] - pre
        g4m = m4["mean_logit_margin"] - pre
        g5m = m5["mean_logit_margin"] - pre
        sg = solo[j]["gain_logit"]
        rows[ep.name] = {
            "pre": round(pre, 3),
            "solo_gain": round(sg, 3),
            "p1_as_is_margin": m1["mean_logit_margin"],
            "p1_gain": round(g1m, 3),
            "p1_retention": (None if not sg else round(g1m / sg, 4)),
            "p2_forced_margin": m2["mean_logit_margin"],
            "p2_forced_gain": round(g2m, 3),
            "p2_vs_solo": (None if not sg else round(g2m / sg, 4)),
            "p3_owner_weight": round(wo[0], 4),
            "p3_p_both_plug_home": round(pb[0], 4),
            "p3_p_owner_plus_wrong_mem": round(pow_[0], 4),
            "p4_cosine_margin": m4["mean_logit_margin"],
            "p4_gain": round(g4m, 3),
            "p4_retention": (None if not sg else round(g4m / sg, 4)),
            "p5_oneplug_margin": m5["mean_logit_margin"],
            "p5_gain": round(g5m, 3),
            "p5_retention": (None if not sg else round(g5m / sg, 4)),
            "best_base": best_base,
            "wrong_mem": wrong,
            "p6a_owner_base_065": round(g6a, 3),
            "p6b_owner_wrong_065": round(g6b, 3),
            "p7_owner_base_a45": round(g7[0.45], 3),
            "p7_owner_base_a65": round(g7[0.65], 3),
            "p7_owner_base_a85": round(g7[0.85], 3),
            "p8_owner_util_at_divergence": p8_util,
            "p8_owner_weight_at_divergence": p8_w,
        }
        print(f"  [{label}] {ep.name:17s} "
              f"P1 {g1m:+7.2f} (ret {rows[ep.name]['p1_retention']})  "
              f"P2force {g2m:+7.2f}  "
              f"w_own {wo[0]:.2f}  P8util {p8_util} w {p8_w}  "
              f"P5cap {g5m:+6.2f}  P6a(b) {g6a:+6.2f}  "
              f"P6b(wr) {g6b:+6.2f}  P7a {g7[0.45]:+.1f}/{g7[0.65]:+.1f}/"
              f"{g7[0.85]:+.1f}")

    def mean(k):
        vals = [r[k] for r in rows.values() if r[k] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    summary = {
        "p1_retention_mean": mean("p1_retention"),
        "p2_vs_solo_mean": mean("p2_vs_solo"),
        "p3_owner_weight_mean": mean("p3_owner_weight"),
        "p3_p_both_plug_home_mean": mean("p3_p_both_plug_home"),
        "p3_p_owner_plus_wrong_mean": mean("p3_p_owner_plus_wrong_mem"),
        "p4_retention_mean": mean("p4_retention"),
        "p5_retention_mean": mean("p5_retention"),
    }
    print(f"  [{label}] SUMMARY {summary}")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {"label": label, "state": state_path, "ckpt": ckpt,
            "n_base": g1.BASE_EXPERTS, "n_mem": n_mem,
            "episodes": rows, "summary": summary}


def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"delivery-diag  device={DEVICE}  labels={LABELS}")
    results = []
    for state, sj, label in zip(STATES, SOLO_JSONS, LABELS):
        ckpt = CKPT_8BASE if "8base" in state or "8base" in label \
            else CKPT_SMALL
        print(f"\n== {label}  state={state}")
        results.append(diag_state(state, sj, label, ckpt))

    # ranking of causes from the probes
    s_small = results[0]["summary"]
    s_8 = results[1]["summary"] if len(results) > 1 else None
    ranking = []
    if s_8:
        if s_8["p2_vs_solo_mean"] and s_8["p2_vs_solo_mean"] > 0.7 \
                and (s_8["p2_vs_solo_mean"] - s_8["p1_retention_mean"]) > 0.2:
            ranking.append("MIX_DILUTION (P2 forced delivery holds while "
                           "P1 as-is collapses — expert fine, mix eats it)")
        if s_8["p4_retention_mean"] and s_8["p1_retention_mean"] and \
                s_8["p4_retention_mean"] > s_8["p1_retention_mean"] + 0.15:
            ranking.append("NORM_TEMPERATURE (cosine routing delivers "
                           "better than raw-dot routing)")
        if s_8["p2_vs_solo_mean"] and s_8["p2_vs_solo_mean"] < 0.5:
            ranking.append("EXPERT_OR_RECIPE (even forced owner delivery "
                           "fails — consolidation/capacity/recipe)")
        if (s_8["p3_p_owner_plus_wrong_mean"] or 0) > 0.2:
            ranking.append("WRONG_MEMORY_COFIRE (owner shares top-2 with "
                           "another memory row on home tokens)")
        if s_8.get("p5_retention_mean") and s_8["p5_retention_mean"] >= 0.7:
            ranking.append(f"FIX_ONE_PLUG_CAP WORKS (P5 retention "
                           f"{s_8['p5_retention_mean']} >= 0.70 — evicting "
                           "the wrong memory restores delivery)")
    verdict = " | ".join(ranking) if ranking else "no clear cause from probes"
    print(f"\nCAUSE RANKING: {verdict}")

    out = {"gate": "G3-delivery-diag",
           "verdict": verdict,
           "labels": LABELS,
           "runs": results,
           "note": ("probes on saved library states: P1 as-is margin, "
                    "P2 forced-owner (pure capability), P3 mix weights / "
                    "wrong-memory cofire, P4 cosine routing"),
           "seconds": round(time.time() - t0, 1)}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")
    print(f"total time: {out['seconds']}s")


if __name__ == "__main__":
    main()
