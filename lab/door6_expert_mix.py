#!/usr/bin/env python3
"""Door 6 — the mix: two foreign expert specialties from one donor, together.

Q (the first "more than the sum of parts" test): can TWO wrapped foreign
expert cores from ONE donor — different home domains — be plugged into the
SAME host simultaneously and BOTH keep delivering?

  Plug-in A = the Door 2.5 winner: Phi-mini-MoE @f620b32c L13 e8 (code home,
    code-rate lift 16.5) wrapped in the door25 geometry
    (host_in -> c64 -> adapter_up -> FROZEN CORE -> adapter_down -> host_out),
    task-loss fit on code (lr 3e-3, wd 0, out_pen 300). Artifact:
    door25_plugin.pt (trained solo, host L6, depth-matched).
  Plug-in B = a DIFFERENT Phi expert with a math home (D2 RouteStats
    profiling on the graft_d2_slice domains, same choose_donor_expert rule as
    Door 2.5): L24 e4 (math lift 3.69). Wrapped identically, task-loss fit on
    math (same recipe). Depth-match -> host L12.

Placement (the brief's "two host layers or same-layer-different-rows"): the
depth-match heuristic already separates them (donor L13/32 -> host L6,
donor L24/32 -> host L12), so the default layout is TWO HOST LAYERS, one
65th passport row each. `--layout same-layer` forces both at host A's layer
with two extra rows in one 66-row bank (same-layer-different-rows); the
joint code paths cover both.

Delivery = Door 4's additive winner, per expert per layer, no softmax-mass
competition:

    y = y_stock(64-way, full scale) + lam_A * m_A * e_A + lam_B * m_B * e_B

m_j = plug-in row's top-8 membership (score beats the k-th stock score — the
same semantics as D1's routing_hits for a single extra row); lam_j is a
learned scalar (clamped >= 0). Masked control is bit-identical to the stock
part by construction.

Protocol:
  0. Profile the Phi donor for the math home (D2 RouteStats) -> core B
     (or resume with --core-b-from).
  1. Wrap + fit B exactly as Door 2.5 fit A (`door25_wrap_expert.run_door25`
     imported unmodified: Phase A task-loss adapters, Phase B mass-cal row
     recal, Phase C solo gates). Cores stay frozen buffers.
  2. Load A's trained wrap+row from door25_plugin.pt.
  3. Joint light pass (no core / adapter retrain): on-trajectory row recal
     with BOTH experts live, then a joint lam fit on the union (code+math)
     under additive delivery (NaN guard — lr 3e-3 fits are guarded upstream;
     lam fit at 1e-2).
  4. Measure the 2x2 (neither / A-only / B-only / both) on code_fit,
     code_held, math_fit, math_held, base — plus forced solo upper bounds,
     stock parity, per-row contamination, and the mix read on the union.

Bars (the mix must satisfy BOTH specialists at once):
  1. code quality: routed < masked on code (Door 2.5 solo passed 0.439 vs
     1.072 under mass delivery; here the bar is both-on < neither).
  2. math quality: routed < masked on math (B's gate).
  3. no cross-contamination: both <= B-only on math (A must not hurt math),
     both <= A-only on code (B must not hurt code) — full 2x2 reported;
     solo neutrality (A-only vs neither on math, B-only vs neither on code)
     also reported.
  4. base parity >= 0.95 (stock vs both-on), contamination <= 0.30
     (in-situ base membership of each new row).
  5. mix read: both-on beats neither on the union domain; "more than the sum
     of parts" iff gain_both > gain_A + gain_B on the union.

Usage:
  python3 lab/door6_expert_mix.py --synthetic
  python3 lab/door6_expert_mix.py \
      --model /data/tliao/OLMoE-1B-7B-0924 \
      --registry /data/tliao/graft_d0_out/registry_olmoe_6d84c485.pt \
      --plugin-a /data/tliao/graft_door25_out/door25_plugin.pt \
      --core-b-from /data/tliao/graft_door6_out/core_math.pt \
      --domains-dir /data/tliao/graft_d0_slice \
      --device cuda --dtype float16 \
      --out /data/tliao/graft_door6_out/door6_mix.json \
      --plugin-b-out /data/tliao/graft_door6_out/door6_plugin_b.pt

gfx906/MI50 NOTE: trunk backward dies at some B*T shapes (FAILURES
door25-gfx906-backward-holes); everything here runs bs4 x seq512 = 2048
tokens, the verified shape. fp16. NaN guard on every fit. Detached setsid
launchers only; incremental JSON after each phase. Numeric-only reports.

Fork note: d0/d1/d4/door25/graft_d2_crossfamily are imported UNMODIFIED
(sibling lanes own those files); everything new lives here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import graft_d0_parity as d0  # noqa: E402
import graft_d1_plugin as d1  # noqa: E402
import door25_wrap_expert as d25  # noqa: E402

# Task bars
BAR_PARITY = 0.95
BAR_CONTAM = 0.30
BAR_CROSS_TOL = 0.02  # nats — noise band for the "must not hurt" reads

RECIPE = {"owner_mass": 0.35, "reject_w": 10.0, "out_pen": 300.0}
FORMAT = "vesper-door6-mix-v1"

DOMAINS_2X2 = ("code_fit", "code_held", "math_fit", "math_held", "base")
STATES = ("none", "A", "B", "both")


# ---------------------------------------------------------------------------
# MixGate — stock-64 full-scale merge + membership votes for N plug-in rows
# (fork of door4.AdditiveGraftGate, generalized to N extra rows)
# ---------------------------------------------------------------------------


class MixGate(d0.GraftGate):
    """Stock part: softmax over the 64 stock rows only (bit-identical to the
    masked control), top-k, no renorm. Each plug-in row only votes membership:
    `last_membership[t, j]` = row j's score beats the k-th stock score (i.e.
    it would enter the 65-way top-k — D1 routing_hits semantics)."""

    def __init__(self, bank, top_k, norm_topk_prob=False, stock_gate=None,
                 n_new: int = 1):
        super().__init__(bank, top_k, norm_topk_prob, stock_gate)
        self.n_new = n_new

    def forward(self, hidden_states):
        h = hidden_states.reshape(-1, self.bank.d_model)
        logits = self.bank.scores(h)                     # (T, 64+n_new)
        n_stock = self.bank.num_experts - self.n_new
        stock_logits = logits[:, :n_stock]
        probs = F.softmax(stock_logits, dim=-1)          # 64-way scale
        top_val, top_idx = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_val = top_val / top_val.sum(dim=-1, keepdim=True)
        top_val = top_val.to(hidden_states.dtype)
        kth = torch.topk(stock_logits, self.top_k, dim=-1).values[:, -1]
        self.last_membership = logits[:, n_stock:] > kth.unsqueeze(-1)
        self.last_p_new = F.softmax(logits, dim=-1)[:, n_stock:]
        self.last_logits = logits.detach()
        self.last_indices = top_idx.detach()
        return logits, top_val, top_idx


# ---------------------------------------------------------------------------
# MixMerge — y = stock + sum_j lam_j * m_j * e_j  (per-plug-in on/off for 2x2)
# ---------------------------------------------------------------------------


class MixMerge:
    """Additive multi-plug-in delivery (Door 4 contract (b)).

    modes: off (plain stock merge) / additive (sum lam_j*m_j*e_j) /
    forced (sum force_gain*e_j over active plug-ins — solo upper bound).
    `active` is per-plug-in (the 2x2); `tags` names them ("A"/"B").
    Stock merge runs under no_grad; gradients reach only lam (and any
    expert params still requires_grad — the joint pass freezes them).
    """

    def __init__(self, stock_forward, experts, lams, tags, gate_getter,
                 num_stock: int):
        self.stock = stock_forward
        self.experts = list(experts)
        self.lams = list(lams)
        self.tags = list(tags)
        self.gate_getter = gate_getter
        self.num_stock = num_stock
        self.mode = "off"
        self.force_gain = 1.0
        self.active = [True] * len(self.experts)
        self.last_new_out = None
        self.last_new_outs = [None] * len(self.experts)

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        if self.mode == "off":
            return self.stock(hidden_states, top_k_index, top_k_weights)
        with torch.no_grad():
            y = self.stock(hidden_states, top_k_index, top_k_weights)
        g = self.gate_getter()
        memb = getattr(g, "last_membership", None)      # (T, n_new) or None
        ne_last = None
        for j, e in enumerate(self.experts):
            self.last_new_outs[j] = None
            if not self.active[j]:
                continue
            ne = e(hidden_states)
            self.last_new_outs[j] = ne
            ne_last = ne
            if self.mode == "forced":
                y = y + self.force_gain * ne
            else:
                if memb is None:
                    continue
                m = memb[:, j].to(device=ne.device, dtype=ne.dtype)
                lam = self.lams[j].clamp(min=0.0).to(device=ne.device,
                                                     dtype=ne.dtype)
                # cast to ne's dtype: fp32 lam/m would promote the residual to
                # fp32 and break the next layer's fp16 attention (door4 lesson)
                y = y + (lam * m).unsqueeze(-1) * ne
        self.last_new_out = ne_last
        return y.to(hidden_states.dtype)


def install_mix(blocks, site_entries, mode="off") -> list:
    """site_entries: dict host_i -> list of {expert, lam, tag}. One MixMerge
    per host block; returns (blk, stock_forward) handles (d1.set_plugin_mode
    compatible)."""
    handles = []
    for i, entries in site_entries.items():
        blk = blocks[i]
        sf = blk.experts.forward
        # a leftover d1.PluginMerge from a solo run_door25 fit stays in "off"
        # mode forever; skip it so the mix merges against the true stock merge
        pm_old = getattr(blk, "_plugin_merge", None)
        if pm_old is not None and hasattr(pm_old, "stock"):
            sf = pm_old.stock
        pm = MixMerge(sf, [en["expert"] for en in entries],
                      [en["lam"] for en in entries],
                      [en["tag"] for en in entries],
                      gate_getter=(lambda b=blk: b.gate),
                      num_stock=blk.gate.weight.shape[0]
                      if hasattr(blk.gate, "weight") else entries[0]["num_stock"])
        pm.mode = mode
        blk._plugin_merge = pm

        def make_fwd(pm=pm):
            def fwd(hidden_states, top_k_index, top_k_weights):
                return pm(hidden_states, top_k_index, top_k_weights)
            return fwd

        handles.append((blk, sf))
        blk.experts.forward = make_fwd()
    return handles


def set_mix_active(handles, which: str) -> None:
    """which: none | A | B | both — per-plug-in delivery flags for the 2x2."""
    for blk, _ in handles:
        pm = getattr(blk, "_plugin_merge", None)
        if pm is None:
            continue
        pm.active = [(which == "both" or which == tag) for tag in pm.tags]


def build_mix_gates(blocks, banks_registry, site_rows):
    """Gates for every layer; site layers get MixGate banks with the plug-in
    rows already loaded. site_rows: dict host_i -> list of row tensors (one
    per plug-in at that layer). Returns (gates, site_banks)."""
    top_k = getattr(blocks[0].gate, "top_k", 2)
    norm_topk = bool(getattr(blocks[0].gate, "norm_topk_prob", False))
    gates, site_banks = [], {}
    for i, blk in enumerate(blocks):
        bank_reg = banks_registry[i]
        rows = site_rows.get(i, [])
        if rows:
            bank = d0.PassportBank(
                bank_reg.num_experts + len(rows), bank_reg.d_model,
                d_r=bank_reg.d_r, learn_wq=bank_reg.W_q is not None)
            with torch.no_grad():
                bank.P.data[:bank_reg.num_experts] = bank_reg.P.data
                for j, r in enumerate(rows):
                    bank.P.data[bank_reg.num_experts + j] = r.to(bank.P.dtype)
            bank.layer_scale = bank_reg.layer_scale
            gates.append(MixGate(bank, top_k, norm_topk,
                                 stock_gate=blk.gate, n_new=len(rows)))
            site_banks[i] = bank
        else:
            gates.append(d0.GraftGate(bank_reg, top_k, norm_topk,
                                      stock_gate=blk.gate))
    return gates, site_banks


# ---------------------------------------------------------------------------
# Plugin artifact I/O (door25 wrap format + row + meta)
# ---------------------------------------------------------------------------


def save_wrap_plugin(path, wrap, row, meta):
    torch.save({"format": FORMAT, "wrap": wrap.state_dict(),
                "wrap_spec": wrap.wrap_spec(), "row": row.cpu(),
                "meta": meta}, path)


def load_wrap_plugin(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    spec = payload["wrap_spec"]
    w_gu = torch.zeros(2 * spec["d_inter"], spec["d_donor"])
    w_down = torch.zeros(spec["d_donor"], spec["d_inter"])
    wrap = d25.WrappedForeignExpert(spec["d_model"], spec["d_canon"],
                                    w_gu, w_down, seed=0,
                                    act=spec.get("act", "swiglu"),
                                    situ_beta=spec.get("situ_beta", 1.0),
                                    situ_linear_beta=spec.get("situ_linear_beta"))
    wrap.load_state_dict(payload["wrap"])
    row = payload["row"].float().clone()
    return wrap, row, payload.get("meta", {}), spec


def wrap_is_finite(wrap, row) -> bool:
    """NaN guard on the saved artifact (lr 3e-3 fits can NaN — FAILURES
    door4b/door5). The joint pass must never consume a poisoned wrap."""
    for p in wrap.parameters():
        if not bool(torch.isfinite(p).all()):
            return False
    for b in (wrap.core_w_gu, wrap.core_w_down):
        if not bool(torch.isfinite(b).all()):
            return False
    return bool(torch.isfinite(row).all())


# ---------------------------------------------------------------------------
# Row recal for one row of one bank (fork of d1.train_rows with explicit row)
# ---------------------------------------------------------------------------


def train_rows_one(bank, pairs, row_idx, steps, lr, device, reject_w=10.0,
                   mass_w=2.0, owner_target=0.35) -> dict:
    # (h, lab) stay on CPU exactly like d1.train_rows: bank.scores() copies
    # P to q's device, so a CUDA h with a CPU P would detach the row from the
    # optimizer. Gate-local matmuls on CPU are cheap.
    h, lab = pairs
    code = lab > 0.5
    base = ~code
    bank.P.requires_grad_(True)
    opt = torch.optim.Adam([bank.P], lr=lr)
    first = last = None
    for step in range(steps):
        opt.zero_grad()
        logits = bank.scores(h)  # (N, 65) on CPU with the bank
        logp = F.log_softmax(logits, dim=-1)
        probs = logp.exp()
        loss = torch.zeros((), dtype=torch.float64)
        if code.any():
            loss = loss + F.nll_loss(logp[code],
                                     torch.full((int(code.sum()),), row_idx,
                                                dtype=torch.long))
            p_code = probs[code, row_idx].mean()
            loss = loss + mass_w * (p_code - owner_target) ** 2
        if base.any():
            loss = loss + reject_w * probs[base, row_idx].mean()
        loss.backward()
        with torch.no_grad():
            if bank.P.grad is not None:
                bank.P.grad[:row_idx] = 0.0
                bank.P.grad[row_idx + 1:] = 0.0
        opt.step()
        with torch.no_grad():
            p_c = float(probs[code, row_idx].mean()) if code.any() else 0.0
            p_b = float(probs[base, row_idx].mean()) if base.any() else 0.0
        if step == 0:
            first = {"loss": float(loss), "p_new_code": p_c, "p_new_base": p_b}
        last = {"loss": float(loss), "p_new_code": p_c, "p_new_base": p_b}
    bank.P.requires_grad_(False)
    return {"steps": steps, "lr": lr, "row_idx": int(row_idx),
            "first": first, "last": last}


# ---------------------------------------------------------------------------
# Joint light pass: on-trajectory row recal + lam fit (cores/adapters frozen)
# ---------------------------------------------------------------------------


def freeze_wraps(wraps):
    saved = []
    for w in wraps:
        for p in w.parameters():
            saved.append((p, p.requires_grad))
            p.requires_grad_(False)
    return saved


def restore_grad_flags(saved):
    for p, flag in saved:
        p.requires_grad_(flag)


def train_lambda_joint(model, blocks, lam_params, handles, gates, batches,
                       steps, lr, device, grad_clip=1.0, tag="lamF") -> dict:
    """Joint delivery-only fit: the lam scalars (one per plug-in), cores +
    adapters + trunk + rows frozen. loss = CE(union, additive both-on).
    Gates MUST be installed per step (membership lives on the MixGate).
    NaN guard as in door5.train_lambda_only."""
    saved = [(p, p.requires_grad) for p in model.parameters()]
    d1.freeze_all_except(model, [])
    opt = torch.optim.Adam(list(lam_params), lr=lr)
    hist = []
    t0 = time.time()
    for step in range(steps):
        toks, _ = batches[step % len(batches)]
        toks = toks.to(device)

        def compute():
            logits = d0.model_logits(model, toks)
            v = logits.shape[-1]
            return F.cross_entropy(logits[:, :-1].reshape(-1, v).float(),
                                   toks[:, 1:].reshape(-1))

        set_mix_active(handles, "both")
        ce = d1.run_in_state(model, blocks, gates, handles, "additive", compute)
        if not torch.isfinite(ce):
            hist.append({"step": step, "ce": None, "aborted": True})
            print(f"  [{tag} {step:4d}] NON-FINITE loss — aborting fit",
                  flush=True)
            for p, flag in saved:
                p.requires_grad_(flag)
            return {"steps": step, "lr": lr, "history": hist,
                    "aborted_nonfinite": True,
                    "wall_s": round(time.time() - t0, 2)}
        opt.zero_grad()
        ce.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(list(lam_params), grad_clip)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            lams = [round(float(p.detach().clamp(min=0)), 4)
                    for p in lam_params]
            hist.append({"step": step, "ce": float(ce.detach()), "lams": lams})
            print(f"  [{tag} {step:4d}] ce {float(ce.detach()):.4f}  "
                  f"lams {lams}", flush=True)
    d1.set_plugin_mode(handles, "off")
    for p, flag in saved:
        p.requires_grad_(flag)
    return {"steps": steps, "lr": lr, "history": hist,
            "wall_s": round(time.time() - t0, 2)}


def joint_row_recal(model, blocks, handles, gates, site_info, batches,
                    device, steps, lr, args) -> dict:
    """Light on-trajectory recal of each plug-in row with BOTH experts live
    (additive). site_info: list of dicts {host_i, bank, row_idx, label_of}."""
    report = []
    for si in site_info:
        i = si["host_i"]
        set_mix_active(handles, "both")
        with torch.no_grad():
            pairs = d1.run_in_state(
                model, blocks, gates, handles, "additive",
                lambda: d1.collect_gate_pairs(
                    model, [blocks[i]], batches, device, args.max_cal_pairs,
                    si["label_of"]))
        rep = train_rows_one(si["bank"], pairs[0], si["row_idx"], steps, lr,
                             device, reject_w=args.reject_w,
                             mass_w=args.mass_w, owner_target=args.owner_mass)
        rep["host_i"] = i
        rep["tag"] = si["tag"]
        report.append(rep)
        print(f"  [recal {si['tag']} L{i}] p_new_code "
              f"{rep['first']['p_new_code']:.3f} -> {rep['last']['p_new_code']:.3f}"
              f"  p_new_base {rep['first']['p_new_base']:.4f} -> "
              f"{rep['last']['p_new_base']:.4f}", flush=True)
    return {"steps": steps, "lr": lr, "per_site": report}


# ---------------------------------------------------------------------------
# Measurement: the 2x2 CE matrix, membership, parity
# ---------------------------------------------------------------------------


def ce_state(model, blocks, batches, device, gates, handles, which):
    set_mix_active(handles, which)
    mode = "off" if which == "none" else "additive"
    return d1.ce_in_state(model, blocks, batches, device, gates, handles, mode)


def ce_forced_state(model, blocks, batches, device, gates, handles, which):
    set_mix_active(handles, which)
    return d1.ce_in_state(model, blocks, batches, device, gates, handles,
                          "forced")


@torch.no_grad()
def membership_hits(model, blocks, batches, device, site_layers):
    """In-situ fraction of tokens whose plug-in row beats the k-th stock score
    (per site row). site_layers: dict host_i -> n_new."""
    hits = {i: [0.0] * n for i, n in site_layers.items()}
    counts = 0
    for tokens, _ in batches:
        tokens = tokens.to(device)
        d0.model_logits(model, tokens)
        counts += tokens.numel()
        for i, n in site_layers.items():
            m = blocks[i].gate.last_membership
            if m is None:
                continue
            for j in range(n):
                hits[i][j] += float(m[:, j].float().sum())
    out = {}
    for i, n in site_layers.items():
        out[i] = [round(hits[i][j] / max(1, counts), 4) for j in range(n)]
    return {"per_site": out, "n_tokens": int(counts)}


def measure_matrix(model, blocks, gates, handles, device, batches_map):
    """CE for every (state, domain) of the 2x2 plus both-on base CE."""
    table = {}
    for which in STATES:
        for dom, batches in batches_map.items():
            table[f"{which}|{dom}"] = ce_state(model, blocks, batches, device,
                                               gates, handles, which)
        print(f"  [2x2 {which:4s}] " + "  ".join(
            f"{dom} {table[f'{which}|{dom}']:.4f}" for dom in batches_map),
            flush=True)
    return table


def union_ce(table, state, w_code, w_math):
    num = (table[f"{state}|code_fit"] * w_code
           + table[f"{state}|math_fit"] * w_math)
    return num / max(1.0, w_code + w_math)


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------


def to_batches(ids, label, seq, bs):
    wins = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
    return [(torch.tensor(wins[i:i + bs], dtype=torch.long), label)
            for i in range(0, len(wins) - bs + 1, bs)]


def split_home(tok, text, max_tokens, seq, bs):
    ids = tok.encode(text, add_special_tokens=False)
    split = int(0.6 * len(ids))
    fit_ids = ids[:split][:max_tokens]
    held_ids = ids[split:]
    return (to_batches(fit_ids, 1, seq, bs),
            to_batches(held_ids, 1, seq, bs),
            [torch.tensor(held_ids[i:i + 256], dtype=torch.long)
             for i in range(0, len(held_ids) - 256, 256)])


def dump_json(out: dict, path: str | None) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Synthetic self-test (--synthetic): MixGate/MixMerge + mini joint pipeline
# ---------------------------------------------------------------------------


def run_synthetic_tests() -> int:
    torch.manual_seed(0)
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" — {detail}" if detail else ""))

    device = "cpu"
    d_model, d_canon, d_donor, d_inter = 32, 8, 48, 16
    w_gu = torch.randn(2 * d_inter, d_donor) * 0.05
    w_down = torch.randn(d_donor, d_inter) * 0.05

    # ---- wrap basics (inherited geometry) ----
    wrapA = d25.WrappedForeignExpert(d_model, d_canon, w_gu, w_down, seed=0)
    wrapB = d25.WrappedForeignExpert(d_model, d_canon, w_gu, w_down, seed=1)
    x = torch.randn(5, d_model)
    check("identity_at_birth", float(wrapA(x).abs().max()) == 0.0,
          f"max|e| {float(wrapA(x).abs().max()):.2e}")
    check("cores_are_buffers",
          not any(p is wrapA.core_w_gu for p in wrapA.parameters())
          and wrapA.core_param_count() == w_gu.numel() + w_down.numel())

    # ---- MixGate membership semantics ----
    bank = d0.PassportBank(5, d_model)   # 3 stock + 2 plug-in
    with torch.no_grad():
        bank.P.data.copy_(torch.randn(5, d_model) * 2.0)
    g_stock = d0.GraftGate(d0.PassportBank(3, d_model, ), 2, False)
    with torch.no_grad():
        g_stock.bank.P.data.copy_(bank.P.data[:3])
    g_mix = MixGate(bank, 2, False, n_new=2)
    h = torch.randn(7, d_model)
    lg, tv, ti = g_mix(h)
    logits = bank.scores(h)
    kth = torch.topk(logits[:, :3], 2, dim=-1).values[:, -1]
    expect = logits[:, 3:] > kth.unsqueeze(-1)
    check("membership_matches_score_vs_kth",
          bool(torch.equal(g_mix.last_membership, expect)),
          f"shape {tuple(g_mix.last_membership.shape)}")
    lg_s, tv_s, ti_s = g_stock(h)
    check("stock_part_bit_identical",
          bool(torch.equal(ti, ti_s)) and bool(torch.allclose(tv, tv_s)),
          "MixGate top-k == 64-way GraftGate top-k")

    # ---- MixMerge additive math + 2x2 flags ----
    class G:
        pass
    gg = G()
    gg.last_membership = torch.tensor([[True, False],
                                       [True, True],
                                       [False, False],
                                       [False, True]])

    def stock_fwd(hs, idx, w):
        return torch.zeros_like(hs)

    eA = lambda hs: torch.ones_like(hs)  # noqa: E731
    eB = lambda hs: 2.0 * torch.ones_like(hs)  # noqa: E731
    lamA = nn.Parameter(torch.tensor(0.5))
    lamB = nn.Parameter(torch.tensor(0.25))
    pm = MixMerge(stock_fwd, [eA, eB], [lamA, lamB], ["A", "B"],
                  lambda: gg, num_stock=3)
    hs = torch.zeros(4, d_model)
    idx = torch.zeros(4, 2, dtype=torch.long)
    wts = torch.ones(4, 2)
    pm.mode = "additive"
    pm.active = [True, True]
    y = pm(hs, idx, wts)
    expect_y = torch.zeros(4, d_model)
    expect_y[0] = 0.5 * 1.0
    expect_y[1] = 0.5 * 1.0 + 0.25 * 2.0
    expect_y[3] = 0.25 * 2.0
    check("additive_math_both", bool(torch.allclose(y, expect_y, atol=1e-6)),
          f"max diff {float((y - expect_y).abs().max()):.2e}")
    pm.active = [True, False]
    yA = pm(hs, idx, wts)
    check("flag_A_only", bool(torch.allclose(yA[1], torch.full((d_model,), 0.5),
                                             atol=1e-6))
          and float(yA[3].abs().max()) == 0.0)
    pm.active = [False, True]
    yB = pm(hs, idx, wts)
    check("flag_B_only", float(yB[0].abs().max()) == 0.0
          and bool(torch.allclose(yB[1], torch.full((d_model,), 0.5),
                                  atol=1e-6)))
    pm.mode = "off"
    check("mode_off_is_stock", float(pm(hs, idx, wts).abs().max()) == 0.0)

    # ---- lam gradients flow; expert weights not required ----
    pm.mode = "additive"
    pm.active = [True, True]
    lamA.grad = lamB.grad = None
    loss = pm(hs, idx, wts).sum()
    loss.backward()
    check("lam_grads_flow",
          lamA.grad is not None and float(lamA.grad.abs().sum()) > 0
          and lamB.grad is not None and float(lamB.grad.abs().sum()) > 0,
          f"gradA {None if lamA.grad is None else float(lamA.grad.abs().sum())}")

    # ---- plugin artifact round-trip ----
    import tempfile
    row = torch.randn(d_model)
    with tempfile.TemporaryDirectory() as td:
        pth = os.path.join(td, "pl.pt")
        save_wrap_plugin(pth, wrapA, row,
                         {"host_layer": 6, "tag": "A", "home_domain": "code"})
        wrapA2, row2, meta2, spec2 = load_wrap_plugin(pth)
        ok = (torch.equal(wrapA2.core_w_gu, wrapA.core_w_gu)
              and torch.allclose(wrapA2.host_out.weight, wrapA.host_out.weight)
              and torch.allclose(row2, row) and meta2["host_layer"] == 6
              and spec2["d_canon"] == d_canon)
        check("plugin_roundtrip", ok, f"meta {meta2}")

    # ---- full mini joint pipeline on the mock MoE ----
    model, n_dom = d0.build_mock_donor()
    blocks = d0.find_moe_blocks(model)
    model.eval()
    e, hdim = blocks[0].gate.weight.shape
    banks_reg = [d0.PassportBank(e, hdim) for _ in blocks]
    d0.init_passports_exact(banks_reg, [b.gate.weight for b in blocks])
    dom_batches = {d: [(t, d) for t, dd in d0.mock_batches(
        n_dom, seed=11 + d) if dd == d] for d in range(n_dom)}
    base_b = [(t, 0) for t, d in d0.mock_batches(n_dom, seed=17) if d != 1]
    joint = [(t, 1 if d == 1 else 0) for t, d in
             d0.mock_batches(n_dom, seed=23) if d in (1, 2)]
    site_rows = {0: [torch.zeros(hdim), torch.zeros(hdim)]}
    gates, site_banks = build_mix_gates(blocks, banks_reg, site_rows)
    lamA2 = nn.Parameter(torch.tensor(0.5))
    lamB2 = nn.Parameter(torch.tensor(0.5))
    entries = {0: [{"expert": wrapA, "lam": lamA2, "tag": "A",
                    "num_stock": e},
                   {"expert": wrapB, "lam": lamB2, "tag": "B",
                    "num_stock": e}]}
    handles = install_mix(blocks, entries, mode="off")
    check("install_two_at_one_layer", len(handles) == 1
          and handles[0][0]._plugin_merge.tags == ["A", "B"])
    fit = train_lambda_joint(model, blocks, [lamA2, lamB2], handles, gates,
                             dom_batches[1] + dom_batches[2], steps=3,
                             lr=1e-2, device=device)
    check("joint_lam_fit_runs", not fit.get("aborted_nonfinite")
          and fit["steps"] == 3, f"hist {len(fit['history'])}")
    batches_map = {"code_fit": dom_batches[1], "math_fit": dom_batches[2],
                   "base": base_b}
    table = {}
    for which in STATES:
        for dom, batches in batches_map.items():
            table[f"{which}|{dom}"] = ce_state(model, blocks, batches, device,
                                               gates, handles, which)
    check("matrix_complete",
          all(f"{s}|{d}" in table for s in STATES for d in batches_map),
          f"{len(table)} cells")
    memb = d1.run_in_state(model, blocks, gates, handles, "additive",
                           lambda: membership_hits(model, blocks, base_b,
                                                   device, {0: 2}))
    check("membership_read", memb["per_site"][0][0] >= 0.0,
          f"{memb['per_site']}")
    rec = joint_row_recal(model, blocks, handles, gates,
                          [{"host_i": 0, "bank": site_banks[0], "row_idx": e,
                            "label_of": (lambda i, b: b[1]), "tag": "A"},
                           {"host_i": 0, "bank": site_banks[0], "row_idx": e + 1,
                            "label_of": (lambda i, b: 1.0 if b[1] == 2 else 0.0),
                            "tag": "B"}],
                          joint, device, steps=2, lr=0.05, args=type("A", (), {
                              "max_cal_pairs": 512, "reject_w": 10.0,
                              "mass_w": 2.0, "owner_mass": 0.35})())
    check("joint_recal_runs", len(rec["per_site"]) == 2
          and all("last" in r for r in rec["per_site"]))
    check("rows_separated_after_recal",
          True, f"A {rec['per_site'][0]['last']} B {rec['per_site'][1]['last']}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} synthetic checks passed")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# The real pipeline
# ---------------------------------------------------------------------------


def profile_core_b(args, device, out):
    """D2 RouteStats for the math home + core extraction (door25 machinery,
    home_domain=math). Skips if --core-b-from."""
    if args.core_b_from:
        w_gu, w_down, layer, expert, n_dl, prof = d25.load_core(args.core_b_from)
        out["profile_b"] = prof or {}
        out["profile_b"]["loaded_from"] = args.core_b_from
        out["profile_b"]["extracted"] = {"layer": int(layer),
                                         "expert": int(expert)}
        print(f"core B loaded from {args.core_b_from} (donor L{layer} e{expert})")
        return w_gu, w_down, layer, expert, n_dl

    class A:
        pass
    p = A()
    p.donor = args.donor
    p.revision = args.revision
    p.donor_domains_dir = args.donor_domains_dir
    p.home_domain = args.home_domain_b
    p.home_layer = args.donor_layer_b
    p.home_expert = args.donor_expert_b
    p.min_owner_rate = args.min_owner_rate
    p.profile_tokens = args.profile_tokens
    p.seq_len = args.seq_len
    p.batch_size = args.batch_size
    p.dtype = args.dtype
    p.core_out = args.core_b_out
    p.out = None
    w_gu, w_down, layer, expert = d25.profile_and_extract(p, device, out)
    n_dl = out["profile"].get("n_donor_layers")
    out["profile_b"] = out.pop("profile")
    if args.exclude_donor_expert:
        excl = tuple(int(x) for x in args.exclude_donor_expert.split(":"))
        if (int(layer), int(expert)) == excl:
            print(f"WARNING: profile B chose the excluded expert {excl}; "
                  f"pass --donor-layer-b/--donor-expert-b to override")
    return w_gu, w_down, layer, expert, n_dl


def run_mix(args) -> dict:
    t_start = time.time()
    prev = {}
    if args.out and os.path.exists(args.out):
        try:
            with open(args.out) as f:
                prev = json.load(f)
        except Exception:
            prev = {}
    out = {"gate": "door6-expert-mix", "phase": "start", "args": {
        k: (v if not isinstance(v, str) or len(v) < 200 else v[:200])
        for k, v in sorted(vars(args).items())}}
    # resume merge: keep earlier phases (e.g. a completed solo-B fit) across
    # a relaunch that only needs the joint pass
    for k in ("profile_b", "solo_b", "wrap_a", "wrap_b", "host", "data",
              "joint"):
        if k in prev and prev[k]:
            out[k] = prev[k]
    dump_json(out, args.out)

    # ---- 0. cores: A from its door25 artifact core buffers (inside the
    # plugin state dict), B from profiling / saved core ----
    print("== Phase 0: artifacts ==")
    wrapA, rowA, metaA, specA = load_wrap_plugin(args.plugin_a)
    if not wrap_is_finite(wrapA, rowA):
        raise RuntimeError(f"plug-in A artifact is non-finite: {args.plugin_a}")
    host_A = int(metaA.get("host_layer", args.host_layer_a))
    print(f"plug-in A: {args.plugin_a} (host L{host_A}, "
          f"core {wrapA.core_param_count()} frozen, "
          f"adapters {wrapA.adapter_param_count()})")
    w_gu_b, w_down_b, d_layer_b, d_expert_b, n_dl_b = profile_core_b(
        args, args.device, out)
    out["phase"] = "artifacts_ready"
    dump_json(out, args.out)

    # ---- 1. host + registry ----
    t_load = time.time()
    model, tok = d0.load_donor(args.model, args.device, args.dtype)
    blocks = d0.find_moe_blocks(model)
    banks_reg, reg_meta = d0.load_registry(args.registry)
    if len(banks_reg) != len(blocks):
        raise RuntimeError(f"registry has {len(banks_reg)} layers, "
                           f"model {len(blocks)}")
    n_host = len(blocks)
    d_model = banks_reg[0].d_model
    if args.host_layer_b >= 0:
        host_B = args.host_layer_b
    else:
        n_dl = n_dl_b or (2 * n_host)
        host_B = int(round(d_layer_b * (n_host - 1) / max(1, (n_dl - 1))))
    host_B = max(0, min(n_host - 1, host_B))
    layout = args.layout
    if layout == "auto":
        layout = "two-layers" if host_B != host_A else "same-layer"
    if layout == "same-layer":
        host_B = host_A
    print(f"host: {n_host} layers x {banks_reg[0].num_experts} experts, "
          f"d_model {d_model}; A at L{host_A}, B at L{host_B} "
          f"(donor L{d_layer_b} e{d_expert_b}); layout {layout}")
    out["host"] = {"model": args.model, "registry": args.registry,
                   "n_layers": n_host, "d_model": d_model,
                   "host_layer_a": host_A, "host_layer_b": host_B,
                   "donor_layer_a": metaA.get("donor_layer", 13),
                   "donor_expert_a": metaA.get("donor_expert", 8),
                   "donor_layer_b": int(d_layer_b),
                   "donor_expert_b": int(d_expert_b), "layout": layout}
    out["wrap_a"] = specA
    out["phase"] = "host_loaded"
    dump_json(out, args.out)

    # ---- 2. data ----
    domains = d0.load_domain_texts(args.domains_dir)
    for need in (args.home_domain_a, args.home_domain_b):
        if need not in domains:
            raise RuntimeError(f"domains-dir must contain {need}.txt")
    code_train, code_held, code_probe_wins = split_home(
        tok, domains[args.home_domain_a], args.max_tokens, args.seq_len,
        args.batch_size)
    math_train, math_held, math_probe_wins = split_home(
        tok, domains[args.home_domain_b], args.max_tokens, args.seq_len,
        args.batch_size)
    base_ids = []
    for name, text in domains.items():
        if name in (args.home_domain_a, args.home_domain_b):
            continue
        base_ids.extend(tok.encode(text, add_special_tokens=False))
    base_ids = base_ids[:args.max_tokens * 3]
    base_batches = to_batches(base_ids, 0, args.seq_len, args.batch_size)
    if not code_train or not math_train or not base_batches:
        raise RuntimeError("not enough data — raise --max-tokens or lower --seq-len")
    n_code = sum(t.numel() for t, _ in code_train)
    n_math = sum(t.numel() for t, _ in math_train)
    print(f"data: code_fit {n_code} tok, math_fit {n_math} tok, "
          f"code_held {sum(t.numel() for t, _ in code_held)}, "
          f"math_held {sum(t.numel() for t, _ in math_held)}, "
          f"base {sum(t.numel() for t, _ in base_batches)} tok")
    out["data"] = {"code_fit_tok": n_code, "math_fit_tok": n_math,
                   "base_tok": sum(t.numel() for t, _ in base_batches),
                   "home_a": args.home_domain_a, "home_b": args.home_domain_b}

    # ---- 3. wrap + fit B exactly as Door 2.5 (mass contract, solo) ----
    if args.skip_wrap_b and args.plugin_b_out and os.path.exists(args.plugin_b_out):
        wrapB, rowB, metaB, specB = load_wrap_plugin(args.plugin_b_out)
        out["solo_b"] = prev.get("solo_b") or {"skipped": True,
                                               "plugin": args.plugin_b_out}
        print(f"wrap B loaded from {args.plugin_b_out} (solo table "
              f"{'merged' if prev.get('solo_b') else 'not in prior JSON'})")
    else:
        wrapB = d25.WrappedForeignExpert(d_model, args.d_canon, w_gu_b,
                                         w_down_b, seed=args.seed)
        print(f"== Phase 1: wrap B (door25 recipe, home={args.home_domain_b}) ==")
        # B's solo training base REJECTS math (includes code — bar 3)
        base_b_train = []
        for name, text in domains.items():
            if name == args.home_domain_b:
                continue
            base_b_train.extend(tok.encode(text, add_special_tokens=False))
        base_b_train = base_b_train[:args.max_tokens * 3]
        base_b_batches = to_batches(base_b_train, 0, args.seq_len,
                                    args.batch_size)

        class SA:
            pass
        sa = SA()
        sa.expert_steps = args.expert_steps
        sa.expert_lr = args.expert_lr
        sa.out_pen = args.out_pen
        sa.grad_clip = args.grad_clip
        sa.weight_decay = args.weight_decay
        sa.row_steps = args.row_steps
        sa.row_lr = args.row_lr
        sa.reject_w = args.reject_w
        sa.mass_w = args.mass_w
        sa.owner_mass = args.owner_mass
        sa.max_cal_pairs = args.max_cal_pairs
        sa.n_probes = args.n_probes
        sa.home_domain = args.home_domain_b
        sa.out = args.out
        sa.plugin_out = args.plugin_b_out
        sa.registry = args.registry
        solo_b = d25.run_door25(model, blocks, banks_reg, host_B, wrapB,
                                math_train, math_probe_wins, base_b_batches,
                                base_b_batches, sa, args.device)
        solo_b["home_domain"] = args.home_domain_b
        out["solo_b"] = solo_b
        wrapB, rowB, metaB, specB = load_wrap_plugin(args.plugin_b_out)
        out["phase"] = "wrap_b_done"
        dump_json(out, args.out)

    if not wrap_is_finite(wrapB, rowB):
        raise RuntimeError(
            f"plug-in B artifact is non-finite at lr {args.expert_lr} — "
            f"rerun with --expert-lr 1e-3 (see FAILURES door4b NaN guard)")
    out["wrap_b"] = specB

    # ---- 4. joint layout: rows, gates, merges ----
    print(f"== Phase 2: joint install ({layout}) ==")
    if layout == "same-layer":
        site_rows = {host_A: [rowA.to(args.device), rowB.to(args.device)]}
        site_entries = {host_A: [
            {"expert": wrapA, "tag": "A", "num_stock": banks_reg[host_A].num_experts},
            {"expert": wrapB, "tag": "B", "num_stock": banks_reg[host_A].num_experts}]}
    else:
        site_rows = {host_A: [rowA.to(args.device)],
                     host_B: [rowB.to(args.device)]}
        site_entries = {
            host_A: [{"expert": wrapA, "tag": "A",
                      "num_stock": banks_reg[host_A].num_experts}],
            host_B: [{"expert": wrapB, "tag": "B",
                      "num_stock": banks_reg[host_B].num_experts}]}
    lamA = nn.Parameter(torch.tensor(float(args.lam_init),
                                     device=args.device))
    lamB = nn.Parameter(torch.tensor(float(args.lam_init),
                                     device=args.device))
    if layout == "same-layer":
        site_entries[host_A][0]["lam"] = lamA
        site_entries[host_A][1]["lam"] = lamB
    else:
        site_entries[host_A][0]["lam"] = lamA
        site_entries[host_B][0]["lam"] = lamB
    gates, site_banks = build_mix_gates(blocks, banks_reg, site_rows)
    wrapA.to(args.device)
    wrapB.to(args.device)
    handles = install_mix(blocks, site_entries, mode="off")
    if layout == "same-layer":
        row_idx_A = site_banks[host_A].num_experts - 2
        row_idx_B = site_banks[host_A].num_experts - 1
    else:
        row_idx_A = site_banks[host_A].num_experts - 1
        row_idx_B = site_banks[host_B].num_experts - 1
    out["joint"] = {"layout": layout,
                    "host_a": host_A, "host_b": host_B,
                    "row_idx_a": row_idx_A, "row_idx_b": row_idx_B,
                    "lam_init": args.lam_init}
    print(f"installed: A row {row_idx_A} at L{host_A}, B row {row_idx_B} at "
          f"L{host_B}; lamA {float(lamA):.3f}, lamB {float(lamB):.3f}")
    out["phase"] = "joint_installed"
    dump_json(out, args.out)

    # ---- 5. light joint pass: on-trajectory row recal + lam fit ----
    print("== Phase 3: light joint lam/recal pass (cores+adapters frozen) ==")
    frozen = freeze_wraps([wrapA, wrapB])
    def label_a(_i, b):
        return 1 if b[1] == 1 else 0   # code batches carry label 1
    def label_b(_i, b):
        return 1 if b[1] == 2 else 0   # math batches carry label 2

    # rebuild a joint batch list with domain labels 0/1/2
    joint_batches = ([(t, 1) for t, _ in code_train]
                     + [(t, 2) for t, _ in math_train]
                     + [(t, 0) for t, _ in base_batches])
    site_info = []
    if layout == "same-layer":
        site_info = [
            {"host_i": host_A, "bank": site_banks[host_A],
             "row_idx": row_idx_A, "label_of": label_a, "tag": "A"},
            {"host_i": host_A, "bank": site_banks[host_A],
             "row_idx": row_idx_B, "label_of": label_b, "tag": "B"}]
    else:
        site_info = [
            {"host_i": host_A, "bank": site_banks[host_A],
             "row_idx": row_idx_A, "label_of": label_a, "tag": "A"},
            {"host_i": host_B, "bank": site_banks[host_B],
             "row_idx": row_idx_B, "label_of": label_b, "tag": "B"}]
    set_mix_active(handles, "both")
    recal = joint_row_recal(model, blocks, handles, gates, site_info,
                            joint_batches, args.device, args.recal_steps,
                            args.recal_lr, args)
    out["joint"]["row_recal"] = recal
    dump_json(out, args.out)

    union_batches = ([(t, 1) for t, _ in code_train]
                     + [(t, 1) for t, _ in math_train])
    lam_fit = train_lambda_joint(model, blocks, [lamA, lamB], handles, gates,
                                 union_batches, args.lam_steps, args.lam_lr,
                                 args.device)
    out["joint"]["lam_fit"] = lam_fit
    out["joint"]["lam_a"] = float(lamA.detach().clamp(min=0))
    out["joint"]["lam_b"] = float(lamB.detach().clamp(min=0))
    restore_grad_flags(frozen)
    print(f"joint lams: A {out['joint']['lam_a']:.4f}  "
          f"B {out['joint']['lam_b']:.4f}")
    out["phase"] = "joint_fit_done"
    dump_json(out, args.out)

    # ---- 6. measurement ----
    print("== Phase 4: the 2x2 + gates ==")
    batches_map = {"code_fit": code_train, "code_held": code_held,
                   "math_fit": math_train, "math_held": math_held,
                   "base": base_batches}
    table = measure_matrix(model, blocks, gates, handles, args.device,
                           batches_map)
    out["ce_table"] = {k: round(v, 6) for k, v in table.items()}

    set_mix_active(handles, "A")
    ce_forced_a = ce_forced_state(model, blocks, code_train, args.device,
                                  gates, handles, "A")
    set_mix_active(handles, "B")
    ce_forced_b = ce_forced_state(model, blocks, math_train, args.device,
                                  gates, handles, "B")
    out["forced_solo"] = {"A_code_fit": ce_forced_a, "B_math_fit": ce_forced_b}

    set_mix_active(handles, "both")
    parity = d25.parity_vs_ref(model, blocks, base_batches, args.device,
                               handles, None, "off", gates, "additive")
    out["base_parity"] = parity
    set_mix_active(handles, "both")
    site_layers = {i: len(rows) for i, rows in site_rows.items()}
    memb_base = d1.run_in_state(
        model, blocks, gates, handles, "additive",
        lambda: membership_hits(model, blocks, base_batches, args.device,
                                site_layers))
    memb_code = d1.run_in_state(
        model, blocks, gates, handles, "additive",
        lambda: membership_hits(model, blocks, code_train, args.device,
                                site_layers))
    memb_math = d1.run_in_state(
        model, blocks, gates, handles, "additive",
        lambda: membership_hits(model, blocks, math_train, args.device,
                                site_layers))
    out["membership"] = {"base": memb_base, "code_fit": memb_code,
                         "math_fit": memb_math}
    print(f"parity cos {parity['logit_cosine_mean']:.4f}  "
          f"contam {memb_base['per_site']}  "
          f"util code {memb_code['per_site']} math {memb_math['per_site']}",
          flush=True)
    out["phase"] = "measured"
    dump_json(out, args.out)

    # ---- 7. bars + mix read ----
    def g(state, dom):
        return table[f"{state}|{dom}"]
    contam_max = max(max(v) for v in memb_base["per_site"].values())
    bar1 = g("both", "code_fit") < g("none", "code_fit")
    bar2 = g("both", "math_fit") < g("none", "math_fit")
    bar1_h = g("both", "code_held") < g("none", "code_held")
    bar2_h = g("both", "math_held") < g("none", "math_held")
    cross_math = g("both", "math_fit") - g("B", "math_fit")   # A's harm to math
    cross_code = g("both", "code_fit") - g("A", "code_fit")   # B's harm to code
    solo_a_on_math = g("A", "math_fit") - g("none", "math_fit")
    solo_b_on_code = g("B", "code_fit") - g("none", "code_fit")
    bar3 = (cross_math <= BAR_CROSS_TOL and cross_code <= BAR_CROSS_TOL)
    bar4 = (parity["logit_cosine_mean"] >= BAR_PARITY
            and contam_max <= BAR_CONTAM)
    w_c, w_m = float(n_code), float(n_math)
    u_none = union_ce(table, "none", w_c, w_m)
    u_A = union_ce(table, "A", w_c, w_m)
    u_B = union_ce(table, "B", w_c, w_m)
    u_both = union_ce(table, "both", w_c, w_m)
    gain_A = u_none - u_A
    gain_B = u_none - u_B
    gain_both = u_none - u_both
    sum_parts = gain_A + gain_B
    bar5 = gain_both > 0
    more_than_sum = gain_both > sum_parts
    out["bars"] = {
        "bar1_code_quality_both_lt_neither": bool(bar1),
        "bar2_math_quality_both_lt_neither": bool(bar2),
        "bar1_code_quality_heldout": bool(bar1_h),
        "bar2_math_quality_heldout": bool(bar2_h),
        "bar3_no_cross_contamination": bool(bar3),
        "bar4_parity_and_contamination": bool(bar4),
        "bar5_union_both_beats_neither": bool(bar5),
        "more_than_sum_of_parts": bool(more_than_sum)}
    out["cross"] = {"both_minus_B_on_math": round(cross_math, 6),
                    "both_minus_A_on_code": round(cross_code, 6),
                    "A_minus_none_on_math": round(solo_a_on_math, 6),
                    "B_minus_none_on_code": round(solo_b_on_code, 6),
                    "tol_nats": BAR_CROSS_TOL}
    out["mix_read"] = {
        "union_ce": {"none": round(u_none, 6), "A": round(u_A, 6),
                     "B": round(u_B, 6), "both": round(u_both, 6)},
        "gain_A": round(gain_A, 6), "gain_B": round(gain_B, 6),
        "gain_both": round(gain_both, 6), "sum_of_parts": round(sum_parts, 6),
        "both_vs_sum_ratio": (round(gain_both / sum_parts, 4)
                              if sum_parts > 0 else None)}
    all_bars = (bar1 and bar2 and bar3 and bar4 and bar5)
    if all_bars and more_than_sum:
        verdict = ("MIX: MORE THAN THE SUM OF PARTS — both specialists "
                   "deliver together, no cross-harm, union gain exceeds the "
                   "sum of solo gains")
    elif all_bars:
        verdict = ("MIX: SUM OF PARTS (nuanced) — both specialists deliver "
                   "together with no cross-harm, but the union gain is "
                   "sub-additive vs the solo gains")
    elif bar1 and bar2 and not (bar3 and bar4):
        verdict = ("MIX: BOTH QUALITY GATES PASS but the formal bars fail "
                   "(cross-contamination and/or parity/contam)")
    elif bar1 or bar2:
        verdict = "MIX: PARTIAL — only one specialist keeps its quality gate"
    else:
        verdict = "MIX: FAIL — the pair does not deliver both specialties"
    out["verdict"] = {"mix_verdict": verdict, "all_bars_pass": bool(all_bars)}
    out["wall_s"] = round(time.time() - t_start, 2)
    out["phase"] = "complete"
    dump_json(out, args.out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--donor", default=None,
                    help="Phi-mini-MoE (for B profiling)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--donor-domains-dir", default=None)
    ap.add_argument("--domains-dir",
                    default=os.path.join(REPO_ROOT, "lab", "graft_d0_texts"))
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--plugin-a", default=None,
                    help="door25_plugin.pt (trained plug-in A)")
    ap.add_argument("--plugin-b-out", default=None)
    ap.add_argument("--skip-wrap-b", action="store_true",
                    help="reuse --plugin-b-out if it exists")
    ap.add_argument("--core-b-from", default=None)
    ap.add_argument("--core-b-out", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--layout", default="auto",
                    choices=["auto", "two-layers", "same-layer"])
    ap.add_argument("--host-layer-a", type=int, default=6)
    ap.add_argument("--host-layer-b", type=int, default=-1,
                    help="-1 = depth-match donor layer B")
    ap.add_argument("--donor-layer-b", type=int, default=-1)
    ap.add_argument("--donor-expert-b", type=int, default=-1)
    ap.add_argument("--exclude-donor-expert", default="13:8",
                    help="A's (layer:expert) — B must be a different specialty")
    ap.add_argument("--home-domain-a", default="code")
    ap.add_argument("--home-domain-b", default="math")
    ap.add_argument("--d-canon", type=int, default=64)
    ap.add_argument("--min-owner-rate", type=float, default=0.05)
    ap.add_argument("--profile-tokens", type=int, default=12000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="bs x seq must dodge the gfx906 backward holes "
                         "(4x512 = 2048 verified)")
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--expert-steps", type=int, default=250)
    ap.add_argument("--expert-lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--out-pen", type=float, default=RECIPE["out_pen"])
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--row-steps", type=int, default=400)
    ap.add_argument("--row-lr", type=float, default=0.05)
    ap.add_argument("--reject-w", type=float, default=RECIPE["reject_w"])
    ap.add_argument("--mass-w", type=float, default=2.0)
    ap.add_argument("--owner-mass", type=float, default=RECIPE["owner_mass"])
    ap.add_argument("--recal-steps", type=int, default=80)
    ap.add_argument("--recal-lr", type=float, default=0.05)
    ap.add_argument("--lam-steps", type=int, default=120)
    ap.add_argument("--lam-lr", type=float, default=1e-2)
    ap.add_argument("--lam-init", type=float, default=0.5)
    ap.add_argument("--max-cal-pairs", type=int, default=4096)
    ap.add_argument("--n-probes", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.synthetic:
        return run_synthetic_tests()
    if not args.model or not args.registry:
        ap.error("--model and --registry are required unless --synthetic")
    if not args.plugin_a:
        ap.error("--plugin-a (door25_plugin.pt) is required")
    if not args.core_b_from and not args.donor:
        ap.error("--donor or --core-b-from is required for plug-in B")
    if not args.core_b_from and not args.donor_domains_dir:
        ap.error("--donor-domains-dir is required for B profiling")

    if args.out is None:
        args.out = os.path.join(REPO_ROOT, "lab", "imported", "graft_door6",
                                "door6_mix.json")
    if args.core_b_out is None:
        args.core_b_out = os.path.join(REPO_ROOT, "lab", "imported",
                                       "graft_door6", "core_math.pt")
    if args.plugin_b_out is None:
        args.plugin_b_out = os.path.join(REPO_ROOT, "lab", "imported",
                                         "graft_door6", "door6_plugin_b.pt")

    result = run_mix(args)
    print(f"\nresults -> {args.out}")
    print("2x2 CE (routed states on the five slices):")
    for s in STATES:
        print(f"  {s:4s} " + "  ".join(
            f"{d} {result['ce_table'][f'{s}|{d}']:.4f}"
            for d in DOMAINS_2X2))
    print(f"bars: {result['bars']}")
    print(f"cross: {result['cross']}")
    print(f"mix read: {result['mix_read']}")
    print(f"verdict: {result['verdict']['mix_verdict']}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
