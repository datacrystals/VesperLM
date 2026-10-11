#!/usr/bin/env python3
"""Door 4 — delivery-contract fix for plug-in experts (D1 s6_quality + Door-3 blocker).

The blocker (D1 toy plug-in AND Door-3 distill, `lab/imported/graft_d3/
door3_distill.json`): plug-in experts TRAIN at composition "stock + e_new"
but get DELIVERED at "~0.09·stock + 0.91·e_new" — under `norm_topk_prob=false`
the softmax mass IS the merge scale, so a live plug-in row steals ~90% of the
stock experts' contribution. Distilled skill is real (Door-3 forced-delivery
KL 5.17→2.59) yet routed CE collapses (8.07 routed vs 1.83 masked) because
the delivered composition is not the trained one.

Two candidate contracts, head-to-head on the D1 code-domain plug-in:

  (a) delivered-mass expert fit — train the delta under the PLUGGED
      composition (the row's real softmax mass, ~0.85-0.9), optimizing the
      MERGED CE rather than the expert's standalone CE. The delta learns to
      replace what the starved stock experts used to contribute AND add
      domain value. Composition matches delivery by construction.

  (b) additive fixed-weight delivery — bypass the mass competition entirely:
      y = y_stock(64-way, full scale) + λ · m_token · e_new(h), where
      m_token = plug-in row's top-8 membership in the 65-way router and λ is
      a learned per-layer scalar. Stock experts keep their merge scale; the
      router decides WHO gets the plug-in, λ decides HOW MUCH. Masked control
      is bit-identical to the stock part by construction.

Gates (D1's, now WITH quality binding):
  parity ≥0.95 (stock vs delivered on base tokens), retention vs solo ≥0.70
  (§8 delta_margin convention), contamination ≤0.30, and §6 quality: held-out
  home-domain CE with the plug-in DELIVERED must beat the masked control.
  (Utilization >0.50 reported as supporting.)

Fork note: sibling lanes import `lab/graft_d1_plugin.py` and
`lab/graft_d2_crossfamily.py` — those files are READ-ONLY here. Everything
new lives in this file; d0/d1 are imported as libraries.

Usage:
  python3 lab/door4_delivery_contract.py --synthetic
  python3 lab/door4_delivery_contract.py --model /data/tliao/OLMoE-1B-7B-0924 \
      --registry /data/tliao/graft_d0_out/registry_olmoe_6d84c485.pt \
      --domains-dir /data/tliao/graft_d0_slice --device cuda --dtype float16 \
      --branch both --out /data/tliao/graft_d0_out/door4.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import graft_d0_parity as d0  # noqa: E402
import graft_d1_plugin as d1  # noqa: E402

BAR_PARITY = 0.95
BAR_RETENTION = 0.70
BAR_CONTAM = 0.30
BAR_UTIL = 0.50


# ---------------------------------------------------------------------------
# Branch (b): additive fixed-weight delivery
# ---------------------------------------------------------------------------


class AdditiveGraftGate(d0.GraftGate):
    """65-row scoring, but the stock merge keeps FULL 64-way scale.

    Stock part: softmax over the 64 stock rows only (bit-identical to the
    masked control), top-k, no renorm — the plug-in never steals mass.
    The 65th row only votes membership: `last_membership[t]` = its score beats
    the k-th stock score (i.e. it would enter the 65-way top-k).
    """

    def forward(self, hidden_states):
        h = hidden_states.reshape(-1, self.bank.d_model)
        logits65 = self.bank.scores(h)                       # (T, 65)
        n_stock = self.bank.num_experts - 1
        stock_logits = logits65[:, :n_stock]
        probs = F.softmax(stock_logits, dim=-1)              # 64-way scale
        top_val, top_idx = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_val = top_val / top_val.sum(dim=-1, keepdim=True)
        top_val = top_val.to(hidden_states.dtype)
        kth = torch.topk(stock_logits, self.top_k, dim=-1).values[:, -1]
        self.last_membership = logits65[:, -1] > kth          # (T,) bool
        self.last_p_new = F.softmax(logits65, dim=-1)[:, -1]
        self.last_logits = logits65.detach()
        self.last_indices = top_idx.detach()
        return logits65, top_val, top_idx


class PluginMergeAdditive:
    """y = stock(h, topk64) + λ · m · e_new(h)   (modes: off / forced / additive)."""

    def __init__(self, stock_forward, new_expert, num_stock, gate_getter, lam):
        self.stock = stock_forward
        self.new_expert = new_expert
        self.num_stock = num_stock
        self.gate_getter = gate_getter  # -> module with .last_membership
        self.lam = lam                  # nn.Parameter scalar
        self.mode = "off"
        self.force_gain = 1.0
        self.last_new_out = None

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        if self.mode == "off":
            return self.stock(hidden_states, top_k_index, top_k_weights)
        with torch.no_grad():
            y = self.stock(hidden_states, top_k_index, top_k_weights)
        ne = self.new_expert(hidden_states)
        self.last_new_out = ne
        if self.mode == "forced":
            return y + self.force_gain * ne
        g = self.gate_getter()
        m = getattr(g, "last_membership", None)
        if m is None:
            m = torch.zeros(hidden_states.shape[0], dtype=torch.bool,
                            device=hidden_states.device)
        # cast to ne's dtype: fp32 λ/m would promote the residual to fp32 and
        # blow up the next layer's fp16 attention projection
        m = m.to(device=hidden_states.device, dtype=ne.dtype)
        lam = self.lam.clamp(min=0.0).to(ne.dtype)
        return (y + (lam * m).unsqueeze(-1) * ne).to(y.dtype)


def install_additive(blocks, pack, lams, mode="off"):
    handles = []
    for blk, exp, lam in zip(blocks, pack.experts, lams):
        sf = blk.experts.forward
        pm = PluginMergeAdditive(sf, exp, blk.gate.weight.shape[0],
                                 gate_getter=(lambda b=blk: b.gate), lam=lam)
        pm.mode = mode
        blk._plugin_merge = pm

        def make_fwd(pm=pm):
            def fwd(hidden_states, top_k_index, top_k_weights):
                return pm(hidden_states, top_k_index, top_k_weights)
            return fwd

        handles.append((blk, sf))
        blk.experts.forward = make_fwd()
    return handles


def make_additive_gates(blocks, pack):
    return [AdditiveGraftGate(b, g.top_k, g.norm_topk_prob, stock_gate=blk.gate)
            for b, g, blk in zip(pack.banks65, pack.graft_gates, blocks)]


# ---------------------------------------------------------------------------
# Branch (a) + (b): expert fit at the DELIVERED composition
# ---------------------------------------------------------------------------


def train_expert_delivered(model, blocks, pack, handles, code_batches,
                           base_batches, steps, lr, device, mode,
                           extra_params=None, out_pen: float = 300.0,
                           grad_clip: float = 1.0) -> dict:
    """Train e_new (and λ for additive) under the ACTUAL delivered composition.

    Same loss shape as the D1 recipe (CE on home batches + out_pen·||e_new(h_base)||²)
    but every forward runs in `mode` ("plugged" or "additive") — the merged
    output the model actually produces at delivery — so the delta optimizes
    the merged CE, not a hypothetical standalone contribution. Stock trunk
    frozen; gradients reach e_new (and λ) only.
    """
    params = list(pack.parameters()) + list(extra_params or [])
    d1.freeze_all_except(model, params)
    opt = torch.optim.Adam(params, lr=lr)
    hist = []
    t0 = time.time()
    for step in range(steps):
        toks, _ = code_batches[step % len(code_batches)]
        toks = toks.to(device)
        d1.set_plugin_mode(handles, mode, 1.0)
        logits = d0.model_logits(model, toks)
        v = logits.shape[-1]
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, v).float(),
                             toks[:, 1:].reshape(-1))
        bt, _ = base_batches[step % len(base_batches)]
        bt = bt.to(device)
        d1.set_plugin_mode(handles, mode, 1.0)
        d0.model_logits(model, bt)  # populate last_new_out on base tokens
        out_terms = [blk._plugin_merge.last_new_out for blk, _ in handles
                     if getattr(blk, "_plugin_merge", None) is not None
                     and blk._plugin_merge.last_new_out is not None]
        out_norm2 = (torch.stack([t.pow(2).mean() for t in out_terms]).mean()
                     if out_terms else torch.zeros((), device=device))
        total = ce + out_pen * out_norm2
        opt.zero_grad()
        total.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            lam_s = (float(extra_params[0].detach().clamp(min=0))
                     if extra_params else None)
            hist.append({"step": step, "ce_code": float(ce.detach()),
                         "out_norm2": float(out_norm2.detach()),
                         "lam": lam_s})
            print(f"  [{mode} fit {step:4d}] ce_code {float(ce.detach()):.4f}  "
                  f"out_norm2 {float(out_norm2.detach()):.6f}"
                  + (f"  lam {lam_s:.3f}" if lam_s is not None else ""))
    return {"mode": mode, "steps": steps, "lr": lr, "out_pen": out_pen,
            "history": hist, "wall_s": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------
# Measurement (D1 gates, mode-parameterized; membership hits for additive)
# ---------------------------------------------------------------------------


def parity_vs_state(model, blocks, batches, device, gates, handles, mode):
    """d1.parity_vs_stock with a parameterized delivery mode (fork of that fn)."""
    sum_cos = sum_sq = n_tok = 0.0
    cos_min = 1.0
    ce_s = ce_p = n_row = 0.0
    with torch.no_grad():
        for tokens, _ in batches:
            tokens = tokens.to(device)
            bsz = tokens.shape[0]
            lg_s = d1.run_in_state(model, blocks, None, handles, "off",
                                   lambda: d0.model_logits(model, tokens))
            lg_p = d1.run_in_state(model, blocks, gates, handles, mode,
                                   lambda: d0.model_logits(model, tokens))
            cos = d0.token_logit_cosine(lg_s, lg_p)
            sum_cos += float(cos.sum())
            sum_sq += float((cos * cos).sum())
            cos_min = min(cos_min, float(cos.min()))
            n_tok += cos.numel()
            ce_s += d0.next_token_ce(lg_s, tokens) * bsz
            ce_p += d0.next_token_ce(lg_p, tokens) * bsz
            n_row += bsz
    mean = sum_cos / max(1.0, n_tok)
    var = max(0.0, sum_sq / max(1.0, n_tok) - mean * mean)
    return {"n_tokens": int(n_tok), "logit_cosine_mean": mean,
            "logit_cosine_std": math.sqrt(var), "logit_cosine_min": cos_min,
            "ce_stock": ce_s / max(1.0, n_row),
            "ce_delivered": ce_p / max(1.0, n_row),
            "ce_delta": (ce_p - ce_s) / max(1.0, n_row)}


@torch.no_grad()
def membership_hits(model, blocks, batches, device):
    """Fraction of tokens where the plug-in row enters top-8 (additive gate)."""
    n = len(blocks)
    hits = [0.0] * n
    counts = 0
    for tokens, _ in batches:
        d0.model_logits(model, tokens.to(device))
        counts += tokens.numel()
        for i in range(n):
            m = getattr(blocks[i].gate, "last_membership", None)
            if m is not None:
                hits[i] += float(m.float().sum())
    fracs = [c / max(1, counts) for c in hits]
    return {"per_layer_util": [round(f, 4) for f in fracs],
            "mean_util": round(sum(fracs) / max(1, len(fracs)), 4),
            "max_util": round(max(fracs), 4) if fracs else 0.0}


def door4_verdict(parity, retention_v, contam, util, ce_active, ce_masked):
    quality_ok = ce_active < ce_masked
    gates = {
        "parity_ge_0.95": parity["logit_cosine_mean"] >= BAR_PARITY,
        "retention_ge_0.70": retention_v is not None and retention_v >= BAR_RETENTION,
        "contamination_le_0.30": contam["mean_util"] <= BAR_CONTAM,
        "quality_ce_beats_masked": quality_ok,
        "utilization_gt_0.50": util["mean_util"] > BAR_UTIL,
    }
    passed = (gates["parity_ge_0.95"] and gates["retention_ge_0.70"]
              and gates["contamination_le_0.30"]
              and gates["quality_ce_beats_masked"])
    return {"pass": passed, "gates": gates,
            "bars": {"parity": BAR_PARITY, "retention": BAR_RETENTION,
                     "contamination": BAR_CONTAM, "utilization": BAR_UTIL}}


def measure_branch(tag, model, blocks, pack, handles, gates, mode, probes,
                   code_eval, code_train, base_batches, device, new_idx,
                   additive: bool) -> dict:
    """All D1 gates for one delivery contract. Quality + utilization are
    measured on HELD-OUT home-domain tokens (code_eval — never trained on);
    the train-slice quality is echoed for continuity with the D1 runs."""
    out = {"branch": tag, "mode": mode}
    fn_pre = d1.make_logits_fn(model, blocks, gates_pre_global, handles, "off",
                               device)
    fn_solo = d1.make_logits_fn(model, blocks, gates_pre_global, handles,
                                "forced", device, force_gain=1.0)
    fn_lib = d1.make_logits_fn(model, blocks, gates, handles, mode, device)
    m_pre = d1.probe_margins(probes, fn_pre)
    m_solo = d1.probe_margins(probes, fn_solo)
    m_lib = d1.probe_margins(probes, fn_lib)
    g_solo = m_solo["mean_logit_margin"] - m_pre["mean_logit_margin"]
    g_lib = m_lib["mean_logit_margin"] - m_pre["mean_logit_margin"]
    ret = d1.retention(g_lib, g_solo)
    out["margins"] = {"pre": m_pre, "solo": m_solo, "lib": m_lib,
                      "gain_solo": round(g_solo, 4),
                      "gain_lib": round(g_lib, 4),
                      "retention_vs_solo": None if ret is None else round(ret, 4)}
    out["base_parity"] = parity_vs_state(model, blocks, base_batches, device,
                                         gates, handles, mode)

    def hits_fn():
        if additive:
            return membership_hits(model, blocks, base_batches, device)
        return d1.routing_hits(model, blocks, base_batches, device, new_idx)

    def hits_code():
        if additive:
            return membership_hits(model, blocks, code_eval, device)
        return d1.routing_hits(model, blocks, code_eval, device, new_idx)

    out["contamination_base"] = d1.run_in_state(
        model, blocks, gates, handles, mode, hits_fn)
    out["utilization_code"] = d1.run_in_state(
        model, blocks, gates, handles, mode, hits_code)
    ce_masked = d1.ce_in_state(model, blocks, code_eval, device,
                               gates_pre_global, handles, "off")
    ce_active = d1.ce_in_state(model, blocks, code_eval, device, gates,
                               handles, mode)
    ce_active_train = d1.ce_in_state(model, blocks, code_train, device, gates,
                                     handles, mode)
    ce_masked_train = d1.ce_in_state(model, blocks, code_train, device,
                                     gates_pre_global, handles, "off")
    out["s6_quality"] = {"ce_code_masked": ce_masked,
                         "ce_code_active": ce_active,
                         "delta": ce_active - ce_masked,
                         "train_slice_active": ce_active_train,
                         "train_slice_masked": ce_masked_train}
    out["verdict"] = door4_verdict(out["base_parity"],
                                   out["margins"]["retention_vs_solo"],
                                   out["contamination_base"],
                                   out["utilization_code"], ce_active, ce_masked)
    return out


# module-global for measure_branch's pre-state gates (set in run_door4)
gates_pre_global = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def save_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def run_door4(model, blocks, banks_registry, code_train, code_eval,
              probe_windows, base_batches, args, device) -> dict:
    global gates_pre_global
    out = {"gate": "door4-delivery-contract", "started": time.time()}
    t0 = time.time()
    top_k = getattr(blocks[0].gate, "top_k", 2)
    norm_topk = bool(getattr(blocks[0].gate, "norm_topk_prob", False))
    pack = d1.build_pack(blocks, banks_registry, (top_k, norm_topk),
                         inter_mult=args.inter_mult, seed=args.seed)
    pack.experts.to(device)
    new_idx = pack.banks65[0].num_experts - 1
    gates64 = [d0.GraftGate(b, top_k, norm_topk, stock_gate=blk.gate)
               for b, blk in zip(banks_registry, blocks)]
    gates_pre_global = gates64
    out["new_expert_idx"] = new_idx

    # ---- shared: row round-1 (teacher-forced) + probe selection ----
    print("SHARED — row round-1 + probe selection")
    handles_off = d1.install_plugin(blocks, pack, "off")

    def label_of(_i, b):
        return b[1]

    pairs = d1.run_in_state(model, blocks, gates64, handles_off, "off",
                            lambda: d1.collect_gate_pairs(
                                model, blocks, code_train + base_batches,
                                device, args.max_cal_pairs, label_of))
    with torch.no_grad():
        for li, bank in enumerate(pack.banks65):
            h, lab = pairs[li]
            code_h = h[lab > 0.5]
            if code_h.shape[0] > 0:
                bank.P.data[new_idx] = code_h.mean(dim=0) * math.sqrt(bank.d_r)
    pb1 = d1.train_rows(pack, pairs, args.row_steps, args.row_lr, device,
                        reject_w=args.reject_w, mass_w=args.mass_w,
                        owner_target=args.owner_mass)
    out["rows"] = {"round1_teacher_forced": pb1}
    fn_pre = d1.make_logits_fn(model, blocks, gates64, handles_off, "off",
                               device)
    probes = d1.select_code_probes(fn_pre, probe_windows, args.n_probes)
    out["n_probes"] = len(probes)
    d1.uninstall_plugin(handles_off)  # branches install their own wrapper
    if args.out:
        save_json(args.out, out)

    def recal_round2(hd, gates, mode):
        """On-trajectory row recal AFTER expert training (deep-layer rows go
        stale when the trained delta drifts h — the D1-att4 util collapse)."""
        p2 = d1.run_in_state(model, blocks, gates, hd, mode,
                             lambda: d1.collect_gate_pairs(
                                 model, blocks, code_train + base_batches,
                                 device, args.max_cal_pairs, label_of))
        return d1.train_rows(pack, p2, max(50, args.row_steps // 2),
                             args.row_lr, device, reject_w=args.reject_w,
                             mass_w=args.mass_w, owner_target=args.owner_mass)

    results = {}

    # ---- branch (a): expert fit at delivered mass (plugged composition) ----
    if args.branch in ("a", "both"):
        print("BRANCH A — expert fit at delivered mass (plugged composition)")
        hA = d1.install_plugin(blocks, pack, "off")
        restored = [(blk, d0.install_graft(blk, g))
                    for blk, g in zip(blocks, pack.graft_gates)]
        pa = train_expert_delivered(model, blocks, pack, hA, code_train,
                                    base_batches, args.expert_steps,
                                    args.expert_lr, device, mode="plugged",
                                    out_pen=args.out_pen,
                                    grad_clip=args.grad_clip)
        ra_rows = recal_round2(hA, pack.graft_gates, "plugged")
        for blk, stock in restored:
            d0.uninstall_graft(blk, stock)
        ra = measure_branch("a_delivered_mass", model, blocks, pack, hA,
                            pack.graft_gates, "plugged", probes,
                            code_eval, code_train, base_batches, device,
                            new_idx, additive=False)
        ra["training"] = pa
        ra["rows_round2"] = ra_rows
        results["branch_a"] = ra
        out["branches"] = results
        d1.uninstall_plugin(hA)
        if args.out:
            save_json(args.out, out)
        print(f"  A: parity {ra['base_parity']['logit_cosine_mean']:.4f}  "
              f"ret {ra['margins']['retention_vs_solo']}  "
              f"contam {ra['contamination_base']['mean_util']:.3f}  "
              f"qual {ra['s6_quality']['ce_code_active']:.3f} vs "
              f"{ra['s6_quality']['ce_code_masked']:.3f}  "
              f"pass {ra['verdict']['pass']}")

    # ---- branch (b): additive fixed-weight delivery ----
    if args.branch in ("b", "both"):
        print("BRANCH B — additive fixed-weight delivery (λ learned)")
        lams = nn.ParameterList([nn.Parameter(torch.tensor(args.lam_init))
                                 for _ in blocks])
        gates_add = make_additive_gates(blocks, pack)
        hB = install_additive(blocks, pack, lams, mode="off")
        restored = [(blk, d0.install_graft(blk, g))
                    for blk, g in zip(blocks, gates_add)]
        pb = train_expert_delivered(model, blocks, pack, hB, code_train,
                                    base_batches, args.expert_steps,
                                    args.expert_lr, device, mode="additive",
                                    extra_params=list(lams),
                                    out_pen=args.out_pen,
                                    grad_clip=args.grad_clip)
        rb_rows = recal_round2(hB, gates_add, "additive")
        for blk, stock in restored:
            d0.uninstall_graft(blk, stock)
        lams_final = [float(p.detach().clamp(min=0)) for p in lams]
        rb = measure_branch("b_additive_lambda", model, blocks, pack, hB,
                            gates_add, "additive", probes,
                            code_eval, code_train, base_batches, device,
                            new_idx, additive=True)
        rb["training"] = pb
        rb["rows_round2"] = rb_rows
        rb["lambda_per_layer"] = lams_final
        rb["lambda_mean"] = round(sum(lams_final) / max(1, len(lams_final)), 4)
        results["branch_b"] = rb
        out["branches"] = results
        d1.uninstall_plugin(hB)
        if args.out:
            save_json(args.out, out)
        print(f"  B: parity {rb['base_parity']['logit_cosine_mean']:.4f}  "
              f"ret {rb['margins']['retention_vs_solo']}  "
              f"contam {rb['contamination_base']['mean_util']:.3f}  "
              f"qual {rb['s6_quality']['ce_code_active']:.3f} vs "
              f"{rb['s6_quality']['ce_code_masked']:.3f}  "
              f"lam {rb['lambda_mean']}  pass {rb['verdict']['pass']}")

    out["wall_s"] = round(time.time() - t0, 2)
    return out


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------


def run_synthetic_tests() -> int:
    torch.manual_seed(0)
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" — {detail}" if detail else ""))

    device = "cpu"
    model, n_dom = d0.build_mock_donor()
    blocks = d0.find_moe_blocks(model)
    model.eval()
    e, h = blocks[0].gate.weight.shape
    top_k, norm_topk = blocks[0].gate.top_k, blocks[0].gate.norm_topk_prob
    banks_reg = [d0.PassportBank(e, h) for _ in blocks]
    d0.init_passports_exact(banks_reg, [b.gate.weight for b in blocks])

    code_train = [(t, 1) for t, d in d0.mock_batches(n_dom, seed=11,
                                                     batches_per_domain=2)
                  if d == 1]
    base_b = [(t, 0) for t, d in d0.mock_batches(n_dom, seed=17,
                                                 batches_per_domain=2)
              if d != 1]
    probe_wins = [t.flatten() for t, d in d0.mock_batches(
        n_dom, seed=23, seq_len=48, batches_per_domain=3, batch_size=1)
        if d == 1]

    class A:
        pass

    args = A()
    args.inter_mult = 1.0
    args.seed = 0
    args.expert_steps = 40
    args.expert_lr = 1e-2
    args.out_pen = 5.0
    args.grad_clip = 1.0
    args.row_steps = 100
    args.row_lr = 0.1
    args.reject_w = 5.0
    args.mass_w = 2.0
    args.owner_mass = 0.35
    args.max_cal_pairs = 2048
    args.n_probes = 8
    args.lam_init = 0.5
    args.branch = "both"
    args.out = None

    out = run_door4(model, blocks, banks_reg, code_train, code_train,
                    probe_wins, base_b, args, device)
    check("both_branches_present",
          set(out["branches"]) == {"branch_a", "branch_b"})
    for tag in ("branch_a", "branch_b"):
        r = out["branches"][tag]
        check(f"{tag}_verdict_struct",
              set(r["verdict"]["gates"]) == {
                  "parity_ge_0.95", "retention_ge_0.70",
                  "contamination_le_0.30", "quality_ce_beats_masked",
                  "utilization_gt_0.50"},
              f"pass={r['verdict']['pass']}")
        check(f"{tag}_parity_computed",
              r["base_parity"]["logit_cosine_mean"] > 0.5,
              f"cos {r['base_parity']['logit_cosine_mean']:.4f}")
    check("branch_b_lambda_recorded", out["branches"]["branch_b"]["lambda_mean"] > 0)
    check("branch_b_stock_scale_parity",
          out["branches"]["branch_b"]["base_parity"]["logit_cosine_mean"]
          >= out["branches"]["branch_a"]["base_parity"]["logit_cosine_mean"] - 0.05,
          f"b {out['branches']['branch_b']['base_parity']['logit_cosine_mean']:.4f}"
          f" vs a {out['branches']['branch_a']['base_parity']['logit_cosine_mean']:.4f}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} synthetic checks passed")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--domains-dir",
                    default=os.path.join(REPO_ROOT, "lab", "graft_d0_texts"))
    ap.add_argument("--out", default=None, help="incremental JSON path")
    ap.add_argument("--branch", default="both", choices=["a", "b", "both"])
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--expert-steps", type=int, default=200)
    ap.add_argument("--expert-lr", type=float, default=3e-4)
    ap.add_argument("--out-pen", type=float, default=300.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lam-init", type=float, default=0.5)
    ap.add_argument("--row-steps", type=int, default=400)
    ap.add_argument("--row-lr", type=float, default=0.05)
    ap.add_argument("--reject-w", type=float, default=10.0)
    ap.add_argument("--mass-w", type=float, default=2.0)
    ap.add_argument("--owner-mass", type=float, default=0.35)
    ap.add_argument("--max-cal-pairs", type=int, default=4096)
    ap.add_argument("--n-probes", type=int, default=16)
    ap.add_argument("--inter-mult", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.synthetic:
        return run_synthetic_tests()
    if not args.model or not args.registry:
        ap.error("--model and --registry are required unless --synthetic")

    torch.manual_seed(args.seed)
    t_load = time.time()
    model, tok = d0.load_donor(args.model, args.device, args.dtype)
    blocks = d0.find_moe_blocks(model)
    banks_reg, reg_meta = d0.load_registry(args.registry)
    if len(banks_reg) != len(blocks):
        raise RuntimeError(f"registry {len(banks_reg)} layers vs model {len(blocks)}")
    print(f"loaded donor+registry in {time.time() - t_load:.1f}s "
          f"({len(blocks)} layers)")

    domains = d0.load_domain_texts(args.domains_dir)
    code_text = domains.get("code")
    if not code_text:
        raise RuntimeError("domains-dir must contain code.txt")
    code_ids = tok.encode(code_text, add_special_tokens=False)
    split = int(0.6 * len(code_ids))
    train_ids, probe_ids = code_ids[:split], code_ids[split:]
    train_ids = train_ids[:args.max_tokens]

    def to_batches(ids, label, seq, bs):
        wins = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
        return [(torch.tensor(wins[i:i + bs], dtype=torch.long), label)
                for i in range(0, len(wins) - bs + 1, bs)]

    code_train = to_batches(train_ids, 1, args.seq_len, args.batch_size)
    # held-out home-domain slice (probe half — never trained on) for the
    # §6 quality gate and utilization
    code_eval = to_batches(probe_ids, 1, args.seq_len, args.batch_size)
    base_ids = []
    for n, t in domains.items():
        if n != "code":
            base_ids.extend(tok.encode(t, add_special_tokens=False))
    base_ids = base_ids[:args.max_tokens * 3]
    base_batches = to_batches(base_ids, 0, args.seq_len, args.batch_size)
    probe_windows = [torch.tensor(probe_ids[i:i + 256], dtype=torch.long)
                     for i in range(0, len(probe_ids) - 256, 256)]
    if not code_train or not code_eval or not base_batches or not probe_windows:
        raise RuntimeError("not enough data")
    print(f"data: code_train {sum(t.numel() for t, _ in code_train)} tok, "
          f"code_eval {sum(t.numel() for t, _ in code_eval)} tok, "
          f"base {sum(t.numel() for t, _ in base_batches)} tok, "
          f"{len(probe_windows)} probe windows")

    if args.out is None:
        args.out = os.path.join(REPO_ROOT, "lab", "imported", "graft_d0",
                                "door4.json")

    result = run_door4(model, blocks, banks_reg, code_train, code_eval,
                       probe_windows, base_batches, args, args.device)
    result["donor"] = {"model": args.model, "revision": args.revision}
    save_json(args.out, result)
    print(f"\nresults -> {args.out}")
    for tag, r in result.get("branches", {}).items():
        v = r["verdict"]
        q = r["s6_quality"]
        print(f"{tag}: parity {r['base_parity']['logit_cosine_mean']:.4f}  "
              f"ret {r['margins']['retention_vs_solo']}  "
              f"contam {r['contamination_base']['mean_util']:.3f}  "
              f"util {r['utilization_code']['mean_util']:.3f}  "
              f"qual {q['ce_code_active']:.3f} vs {q['ce_code_masked']:.3f}  "
              f"PASS={v['pass']}  gates={v['gates']}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
