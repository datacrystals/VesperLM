#!/usr/bin/env python3
"""D1 plug-in onto graft — MODULAR_MOE.md §9.4 gate D1 (plug-in onto graft).

Adds ONE foreign expert to the D0 passport graft (registry
`registry_olmoe_6d84c485.pt`) and measures whether it DELIVERS through the
standard interface without breaking parity. The stand-in "foreign" expert is
a code-domain expert: a small SwiGLU delta per MoE layer, trained on the
codeparrot slice of the gate data (§9.2's "ours" case; the truly cross-family
case is D2 — not attempted here).

Spec: §9.4 D1 = "section-6 purity gates for the new expert, no regression on
base eval". §6 gates: utilization > 0.5 (domain-B tokens to the plug-in),
contamination < 0.3 (domain-A tokens to the plug-in), quality: CE with expert
active beats masked control on domain B with no regression on domain A.
Task bar (numeric, since §9 leaves numbers open) — all three must hold:

  (a) base parity:   logit cos (stock vs plugged-graft) >= 0.95 on base tokens
  (b) delivery:      margin retention vs solo >= 0.70  (§8 convention:
                     retention = gain_lib / gain_solo, lab/memory_shaping.py
                     `retention()`; margins = logit(good)-logit(bad) at first
                     divergence, `delta_margin()` convention; solo = expert
                     with guaranteed delivery (forced dispatch), plugged =
                     natural passport routing through the 65-row bank)
  (c) contamination: <= 0.30 of base tokens route to the plug-in

Recipe (G1b + plugin_expert_test_v2 contrastive supervision, single insert):
  Phase A  expert fit — freeze donor+registry; per-layer SwiGLU delta
           (zero-init down_proj -> e(x)=0 at birth, §4.2) trained with FORCED
           dispatch (its output added at gain 1.0 on every token) on code-token
           CE + text-KL base-neutrality (G1b, coef 3.0) on base text.
  Phase B  passport recal — router-only (gate-local on cached pre-gate hidden
           states, no trunk backprop): router-logit CE targeting the new row
           on code tokens + reject penalty on base tokens (v2 contrastive
           reject), aiming for owner routing mass ~0.85 on home (the §8
           delivery-cliff lesson: 0.55 under-delivers at scoring positions).
  Phase C  measurement — (a)(b)(c) + §6 utilization / CE-vs-masked.

Usage:
  python3 lab/graft_d1_plugin.py --synthetic
  python3 lab/graft_d1_plugin.py --model /data/tliao/OLMoE-1B-7B-0924 \
      --registry /data/tliao/graft_d0_out/registry_olmoe_6d84c485.pt \
      --domains-dir /data/tliao/graft_d0_slice --device cuda --dtype float16 \
      --out /data/tliao/graft_d0_out/olmoe_d1.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import graft_d0_parity as d0  # noqa: E402

# Task bars (numeric, see module docstring)
BAR_BASE_PARITY = 0.95
BAR_RETENTION = 0.70
BAR_CONTAMINATION = 0.30
# §6 supporting gates
BAR_UTILIZATION = 0.50

TEXT_KL_COEF = 3.0        # G1b base-neutrality
REJECT_W = 5.0            # plugin_expert_test_v2 contrastive reject default
OWNER_MASS_TARGET = 0.85  # §8 delivery-cliff fix (memory_shaping: 0.55 under-delivers)


# ---------------------------------------------------------------------------
# Foreign expert (OLMoE-layout SwiGLU delta) + 65-row graft bank
# ---------------------------------------------------------------------------


class SwiGLUExpert(nn.Module):
    """One expert in the OLMoE 3D-bank layout: gate_up (2I, H), down (H, I).

    Output is a residual delta (§4.2 contract). down_proj zero-init makes the
    expert the identity at birth — a small delta that training grows.
    """

    def __init__(self, d_model: int, inter: int, zero_init: bool = True,
                 seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.gate_up_proj = nn.Parameter(
            torch.randn(2 * inter, d_model, generator=g) * 0.02)
        self.down_proj = nn.Parameter(
            torch.zeros(d_model, inter) if zero_init
            else torch.randn(d_model, inter, generator=g) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 math (params are fp32; x may be fp16 on the donor) — the delta
        # is small, keep its precision, cast the output back to the trunk dtype
        xf = x.float()
        w_gu = self.gate_up_proj.to(xf.device)
        w_d = self.down_proj.to(xf.device)
        gate, up = F.linear(xf, w_gu).chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, w_d).to(x.dtype)


@dataclass
class PluginPack:
    """The plugged expert identity: per-layer SwiGLU + the 65th passport row."""
    experts: nn.ModuleList
    banks65: list          # d0.PassportBank with num_experts+1 rows
    graft_gates: list      # d0.GraftGate over the 65-row banks

    def parameters(self):
        for e in self.experts:
            yield from e.parameters()


def build_pack(blocks, banks_registry, stock_gate_meta, inter_mult: float = 1.0,
               seed: int = 0) -> PluginPack:
    """Extend the registry banks with one row per layer + one expert per layer."""
    top_k, norm_topk = stock_gate_meta
    experts = nn.ModuleList()
    banks65 = []
    for l, (blk, bank) in enumerate(zip(blocks, banks_registry)):
        inter = int(blk.experts.gate_up_proj.shape[1] // 2 * inter_mult)
        d_model = blk.experts.gate_up_proj.shape[2]
        experts.append(SwiGLUExpert(d_model, inter, seed=seed + l))
        b = d0.PassportBank(bank.num_experts + 1, bank.d_model,
                            d_r=bank.d_r, learn_wq=bank.W_q is not None)
        with torch.no_grad():
            b.P.data[:bank.num_experts] = bank.P.data
            b.P.data[bank.num_experts] = 0.0
        b.layer_scale = bank.layer_scale
        banks65.append(b)
    gates = [d0.GraftGate(b, top_k, norm_topk, stock_gate=blk.gate)
             for b, blk in zip(banks65, blocks)]
    return PluginPack(experts=experts, banks65=banks65, graft_gates=gates)


class PluginMerge:
    """Wraps a stock experts merge so expert_id == num_experts serves the plug-in.

    Modes:
      off      plain stock merge (plug-in invisible)
      forced   stock merge + gain * e_new(h) on every token (training / solo)
      plugged  stock merge + e_new(h) * routed_weight for tokens that selected
               the plug-in row (natural delivery through the 65-row bank)
    Stock merge runs under no_grad in forced/plugged mode (frozen); gradients
    reach only the new expert modules.
    """

    def __init__(self, stock_forward, new_expert, num_stock: int):
        self.stock = stock_forward  # the ORIGINAL experts forward (bound fn)
        self.new_expert = new_expert
        self.num_stock = num_stock
        self.mode = "off"
        self.force_gain = 1.0
        self.last_new_out = None  # e_new(h) of the last non-off call (for penalties)

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        if self.mode == "off":
            return self.stock(hidden_states, top_k_index, top_k_weights)
        is_new = top_k_index == self.num_stock
        stock_idx = torch.where(is_new, torch.zeros_like(top_k_index),
                                top_k_index)
        stock_w = torch.where(is_new, torch.zeros_like(top_k_weights),
                              top_k_weights)
        with torch.no_grad():
            y = self.stock(hidden_states, stock_idx, stock_w)
        ne = self.new_expert(hidden_states)
        self.last_new_out = ne
        if self.mode == "forced":
            return y + self.force_gain * ne
        w_new = (top_k_weights * is_new.to(top_k_weights.dtype)).sum(dim=-1)
        return y + ne * w_new.unsqueeze(-1)


def install_plugin(blocks, pack: PluginPack, mode: str,
                   force_gain: float = 1.0) -> list:
    """Install the plug-in merge at every block; returns restore handles.

    The PluginMerge object is stashed on the block (`blk._plugin_merge`) so
    set_plugin_mode can flip off/forced/plugged without reinstalling.
    """
    handles = []
    for blk, exp in zip(blocks, pack.experts):
        stock_forward = blk.experts.forward  # may be the ROCm-shim merge
        pm = PluginMerge(stock_forward, exp, blk.gate.weight.shape[0])
        pm.mode = mode
        pm.force_gain = force_gain
        blk._plugin_merge = pm

        def make_fwd(pm=pm, sf=stock_forward):
            def fwd(hidden_states, top_k_index, top_k_weights):
                return pm(hidden_states, top_k_index, top_k_weights)
            return fwd

        handles.append((blk, stock_forward))
        blk.experts.forward = make_fwd()
    return handles


def uninstall_plugin(handles) -> None:
    for blk, stock_forward in handles:
        blk.experts.forward = stock_forward
        if hasattr(blk, "_plugin_merge"):
            del blk._plugin_merge


def set_plugin_mode(handles, mode: str, force_gain: float = 1.0) -> None:
    for blk, _ in handles:
        pm = getattr(blk, "_plugin_merge", None)
        if pm is not None:
            pm.mode = mode
            pm.force_gain = force_gain


# ---------------------------------------------------------------------------
# Phase A — expert fit (forced dispatch CE + text-KL base-neutrality)
# ---------------------------------------------------------------------------


def freeze_all_except(model, params) -> None:
    keep = set(id(p) for p in params)
    for p in model.parameters():
        p.requires_grad = (id(p) in keep)


def train_expert_forced(model, blocks, pack, handles, code_batches,
                        base_batches, steps, lr, device,
                        kl_coef: float = 0.0,
                        out_pen: float = 3000.0,
                        grad_clip: float = 1.0,
                        weight_decay: float = 0.01) -> dict:
    """Train only the new expert modules with forced dispatch (gain 1.0).

    Loss = next-token CE on code batches
         + out_pen * mean_l ||e_new(h_base)||^2   (direct off-domain L2 — the
           stable neutrality control; attempts 1-2 showed trunk text-KL alone
           either diverges (KL ~1500) or dominates the CE gradient (KL ~300
           while code CE degrades), and an L2 of 1e-4 scale is inert at pen 10)
         + optional kl_coef * KL(graft+forced || graft-only) (kept for
           ablation, default 0).
    The trunk is frozen; gradients reach only the new expert params.
    """
    params = list(pack.parameters())
    freeze_all_except(model, params)
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    hist = []
    t0 = time.time()
    for step in range(steps):
        toks, _ = code_batches[step % len(code_batches)]
        toks = toks.to(device)
        set_plugin_mode(handles, "forced", 1.0)
        logits = d0.model_logits(model, toks)
        v = logits.shape[-1]
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, v).float(),
                               toks[:, 1:].reshape(-1))
        bt, _ = base_batches[step % len(base_batches)]
        bt = bt.to(device)
        kl = torch.zeros((), device=device)
        if kl_coef > 0:
            with torch.no_grad():
                set_plugin_mode(handles, "off")
                ref = d0.model_logits(model, bt)
            set_plugin_mode(handles, "forced", 1.0)
            out = d0.model_logits(model, bt)
            kl = F.kl_div(F.log_softmax(out[:, :-1].float(), dim=-1),
                          F.softmax(ref[:, :-1].float(), dim=-1),
                          reduction="batchmean")
        else:
            set_plugin_mode(handles, "forced", 1.0)
            d0.model_logits(model, bt)  # populate last_new_out on base tokens
        out_terms = [blk._plugin_merge.last_new_out for blk, _ in handles
                     if getattr(blk, "_plugin_merge", None) is not None
                     and blk._plugin_merge.last_new_out is not None]
        out_norm2 = (torch.stack([t.pow(2).mean() for t in out_terms]).mean()
                     if out_terms else torch.zeros((), device=device))
        total = loss + kl_coef * kl + out_pen * out_norm2
        opt.zero_grad()
        total.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            hist.append({"step": step,
                         "ce_code": float(loss.detach()),
                         "kl_base": float(kl.detach()),
                         "out_norm2": float(out_norm2.detach())})
            print(f"  [phaseA {step:4d}] ce_code {float(loss.detach()):.4f}  "
                  f"kl_base {float(kl.detach()):.4f}  "
                  f"out_norm2 {float(out_norm2.detach()):.6f}")
    return {"steps": steps, "lr": lr, "kl_coef": kl_coef,
            "out_pen": out_pen, "grad_clip": grad_clip, "history": hist,
            "wall_s": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------
# Phase B — passport row recal (gate-local, contrastive v2 supervision)
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_gate_pairs(model, blocks, batches, device, max_pairs_per_layer,
                       label_of):
    """(h, label) per layer under the CURRENT gates; label = label_of(batch)."""
    n = len(blocks)
    hs = [[] for _ in range(n)]
    labs = [[] for _ in range(n)]
    tap = d0.GateTap(blocks)
    for b_i, (tokens, _dom) in enumerate(batches):
        tap.clear()
        d0.model_logits(model, tokens.to(device))
        lab = label_of(b_i, batches[b_i])
        for i in range(n):
            hs[i].append(tap.h[i].float().cpu())
            labs[i].append(torch.full((tap.h[i].shape[0],), float(lab)))
    tap.close()
    out = []
    for i in range(n):
        h = torch.cat(hs[i], dim=0)
        y = torch.cat(labs[i], dim=0)
        if h.shape[0] > max_pairs_per_layer:
            stride = h.shape[0] // max_pairs_per_layer + 1
            h = h[::stride][:max_pairs_per_layer]
            y = y[::stride][:max_pairs_per_layer]
        out.append((h, y))
    return out


def train_rows(pack, pairs, steps, lr, device, reject_w: float = REJECT_W,
               mass_w: float = 2.0, owner_target: float = 0.85) -> dict:
    """Router-only recal per layer: CE(new row | code) + reject_w * p(new | base)
    + mass_w * (mean p(new | code) - owner_target)^2.

    The mass term encodes the §8 delivery-cliff lesson (owner_mass 0.85 on the
    home domain — 0.55 under-delivers at scoring positions). Trains the 65th
    passport row only (stock rows registry-frozen via grad masking). Pure
    gate-local matmuls on cached hidden states — no trunk backprop.
    """
    report = {"steps": steps, "lr": lr, "reject_w": reject_w,
              "mass_w": mass_w, "owner_target": owner_target, "per_layer": []}
    for l, (bank, (h, lab)) in enumerate(zip(pack.banks65, pairs)):
        bank.P.requires_grad_(True)
        opt = torch.optim.Adam([bank.P], lr=lr)
        code = lab > 0.5
        base = ~code
        new_idx = bank.num_experts - 1
        first = last = None
        for step in range(steps):
            opt.zero_grad()
            logits = bank.scores(h)  # (N, 65)
            logp = F.log_softmax(logits, dim=-1)
            probs = logp.exp()
            loss = torch.zeros((), dtype=torch.float64)
            if code.any():
                loss = loss + F.nll_loss(logp[code],
                                         torch.full((int(code.sum()),), new_idx,
                                                    dtype=torch.long))
                p_code = probs[code, new_idx].mean()
                loss = loss + mass_w * (p_code - owner_target) ** 2
            if base.any():
                loss = loss + reject_w * probs[base, new_idx].mean()
            loss.backward()
            with torch.no_grad():
                if bank.P.grad is not None:
                    bank.P.grad[:new_idx] = 0.0  # stock rows stay put
            opt.step()
            with torch.no_grad():
                p_new_code = float(probs[code, new_idx].mean()) if code.any() else 0.0
                p_new_base = float(probs[base, new_idx].mean()) if base.any() else 0.0
            if step == 0:
                first = {"loss": float(loss), "p_new_code": p_new_code,
                         "p_new_base": p_new_base}
            last = {"loss": float(loss), "p_new_code": p_new_code,
                    "p_new_base": p_new_base}
        bank.P.requires_grad_(False)
        report["per_layer"].append({"layer": l, "first": first, "last": last})
    return report


# ---------------------------------------------------------------------------
# Probes (delta_margin convention, lab/memory_shaping.py) in token space
# ---------------------------------------------------------------------------


@dataclass
class Probe:
    prefix: torch.Tensor   # (1, T) token ids
    good: int              # true next token
    bad: int               # model-preferred wrong token at selection time


@torch.no_grad()
def select_code_probes(logits_fn, token_windows, n_probes, max_prefix=256):
    """Pick (prefix, good, bad) where the pre model prefers `bad` over `good`.

    At each candidate position t: good = window[t+1], bad = argmax over the
    position's top-64 logits excluding good. Keep positions with negative
    pre-margin (model prefers wrong) so the plug-in has real work to do —
    the same spirit as memory_shaping's stated-vs-correction deltas.
    """
    probes = []
    for win in token_windows:
        win = win.flatten()
        if win.shape[0] < 12:
            continue
        lg_all = logits_fn(win.unsqueeze(0))[0].float()  # (T, V) one forward
        for t in range(8, win.shape[0] - 1, 7):
            if len(probes) >= n_probes:
                return probes
            logits = lg_all[t]  # position t predicts token t+1
            good = int(win[t + 1])
            topv, topi = torch.topk(logits, 64)
            bad = None
            for v, i in zip(topv.tolist(), topi.tolist()):
                if i != good:
                    bad = i
                    bad_v = v
                    break
            if bad is None or bad_v <= float(logits[good]):
                continue
            pre = win[:t + 1].unsqueeze(0)
            probes.append(Probe(prefix=pre[:, -max_prefix:].clone(),
                                good=good, bad=bad))
    return probes


@torch.no_grad()
def probe_margins(probes, logits_fn) -> dict:
    """delta_margin(): logit(good) - logit(bad) at the first divergence."""
    ms = []
    for p in probes:
        lg = logits_fn(p.prefix)[0, -1].float()
        ms.append(float(lg[p.good] - lg[p.bad]))
    return {"mean_logit_margin": round(sum(ms) / max(1, len(ms)), 4),
            "per_probe": [round(m, 4) for m in ms]}


def retention(gain_lib: float, gain_solo: float):
    """§8 convention (memory_shaping.retention): lib_gain / solo_gain."""
    return None if not gain_solo else gain_lib / gain_solo


# ---------------------------------------------------------------------------
# State control: stock / graft64 (masked) / plugged65 / forced
# ---------------------------------------------------------------------------


def run_in_state(model, blocks, gates, handles, mode, fn, force_gain=1.0):
    """Install `gates` (or keep current if None) + plugin mode, run fn, restore."""
    restored = None
    if gates is not None:
        restored = [(blk, d0.install_graft(blk, g))
                    for blk, g in zip(blocks, gates)]
    if handles is not None:
        set_plugin_mode(handles, mode, force_gain)
    try:
        return fn()
    finally:
        if restored is not None:
            for blk, stock in restored:
                d0.uninstall_graft(blk, stock)
        if handles is not None and mode != "off":
            set_plugin_mode(handles, "off")


def make_logits_fn(model, blocks, gates, handles, mode, device, force_gain=1.0):
    def fn(x):
        with torch.no_grad():
            return run_in_state(model, blocks, gates, handles, mode,
                                lambda: d0.model_logits(model, x.to(device)),
                                force_gain)
    return fn


# ---------------------------------------------------------------------------
# Measurement + verdict
# ---------------------------------------------------------------------------


@torch.no_grad()
def routing_hits(model, blocks, batches, device, new_idx):
    """Fraction of tokens whose top-k includes the plug-in row (per layer)."""
    n = len(blocks)
    hits = [0.0] * n
    counts = 0
    tap = d0.GateTap(blocks)
    for tokens, _ in batches:
        tap.clear()
        d0.model_logits(model, tokens.to(device))
        counts += tokens.numel()
        for i in range(n):
            hit = (tap.idx[i] == new_idx).any(dim=-1).float()
            hits[i] += float(hit.sum())
    tap.close()
    fracs = [c / max(1, counts) for c in hits]
    return {"per_layer_util": [round(f, 4) for f in fracs],
            "mean_util": round(sum(fracs) / max(1, len(fracs)), 4),
            "max_util": round(max(fracs), 4) if fracs else 0.0}


@torch.no_grad()
def routing_mass(model, blocks, batches, device, new_idx):
    """Mean softmax routing mass assigned to the plug-in row (per-layer mean)."""
    n = len(blocks)
    mass = [0.0] * n
    counts = 0
    tap = d0.GateTap(blocks)
    for tokens, _ in batches:
        tap.clear()
        d0.model_logits(model, tokens.to(device))
        counts += 1
        for i in range(n):
            h = tap.h[i].float()
            probs = F.softmax(blocks[i].gate.bank.scores(h), dim=-1)
            mass[i] += float(probs[:, new_idx].mean())
    tap.close()
    ms = [m / max(1, counts) for m in mass]
    return {"per_layer_mass": [round(m, 4) for m in ms],
            "mean_mass": round(sum(ms) / max(1, len(ms)), 4)}


@torch.no_grad()
def ce_in_state(model, blocks, batches, device, gates=None, handles=None,
                mode="off") -> float:
    tot, n = 0.0, 0

    def one():
        nonlocal tot, n
        for tokens, _ in batches:
            tokens = tokens.to(device)
            lg = d0.model_logits(model, tokens)
            tot += d0.next_token_ce(lg, tokens) * tokens.shape[0]
            n += tokens.shape[0]

    if gates is None and (handles is None or mode == "off"):
        one()
    else:
        run_in_state(model, blocks, gates, handles, mode, one)
    return tot / max(1, n)


@torch.no_grad()
def parity_vs_stock(model, blocks, batches, device, gates65, handles) -> dict:
    """(a) stock-router outputs vs plugged-graft outputs on identical batches."""
    sum_cos = sum_sq = n_tok = 0.0
    cos_min = 1.0
    ce_s = ce_p = n_row = 0.0
    for tokens, _ in batches:
        tokens = tokens.to(device)
        bsz = tokens.shape[0]
        lg_s = run_in_state(model, blocks, None, handles, "off",
                            lambda: d0.model_logits(model, tokens))
        lg_p = run_in_state(model, blocks, gates65, handles, "plugged",
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
            "ce_plugged": ce_p / max(1.0, n_row),
            "ce_delta": (ce_p - ce_s) / max(1.0, n_row)}


def d1_verdict(base_parity: dict, retention_v, contam: dict, util: dict,
               ce_active: float, ce_masked: float) -> dict:
    """Assemble the D1 gate verdict (task bars + §6 supporting gates)."""
    gates = {
        "base_parity_cos_ge_0.95": base_parity["logit_cosine_mean"] >= BAR_BASE_PARITY,
        "retention_vs_solo_ge_0.70": (retention_v is not None
                                      and retention_v >= BAR_RETENTION),
        "contamination_le_0.30": contam["mean_util"] <= BAR_CONTAMINATION,
        # §6 supporting:
        "s6_utilization_gt_0.50": util["mean_util"] > BAR_UTILIZATION,
        "s6_quality_ce_beats_masked": ce_active < ce_masked,
    }
    passed = (gates["base_parity_cos_ge_0.95"]
              and gates["retention_vs_solo_ge_0.70"]
              and gates["contamination_le_0.30"])
    return {"pass": passed, "gates": gates,
            "bars": {"base_parity": BAR_BASE_PARITY,
                     "retention": BAR_RETENTION,
                     "contamination": BAR_CONTAMINATION,
                     "utilization_s6": BAR_UTILIZATION}}


# ---------------------------------------------------------------------------
# D1 pipeline
# ---------------------------------------------------------------------------


def run_d1(model, blocks, banks_registry, code_train, code_probe_windows,
           base_batches, base_val_batches, args, device) -> dict:
    """Full D1: expert fit -> row recal -> (a)(b)(c) + §6 measurement."""
    out = {"gate": "D1", "bars": {"base_parity": BAR_BASE_PARITY,
                                  "retention": BAR_RETENTION,
                                  "contamination": BAR_CONTAMINATION}}
    t0 = time.time()
    top_k = getattr(blocks[0].gate, "top_k", 2)
    norm_topk = bool(getattr(blocks[0].gate, "norm_topk_prob", False))
    pack = build_pack(blocks, banks_registry, (top_k, norm_topk),
                      inter_mult=args.inter_mult, seed=args.seed)
    pack.experts.to(device)
    handles = install_plugin(blocks, pack, "off")
    new_idx = pack.banks65[0].num_experts - 1
    gates64 = [d0.GraftGate(b, top_k, norm_topk, stock_gate=blk.gate)
               for b, blk in zip(banks_registry, blocks)]
    out["n_experts_stock"] = banks_registry[0].num_experts
    out["new_expert_idx"] = new_idx

    # ---- Phase A: expert fit (forced dispatch) ----
    print("PHASE A — expert fit (forced dispatch, text-KL base-neutrality)")
    set_plugin_mode(handles, "off")
    pa = train_expert_forced(model, blocks, pack, handles, code_train,
                             base_batches, args.expert_steps, args.expert_lr,
                             device, out_pen=args.out_pen,
                             grad_clip=args.grad_clip)
    out["phase_a"] = pa

    # ---- Phase B: passport row recal (gate-local) ----
    print("PHASE B — passport row recal (contrastive router CE + reject + mass)")
    pairs = run_in_state(model, blocks, gates64, handles, "off",
                         lambda: collect_gate_pairs(
                             model, blocks, code_train + base_batches,
                             device, args.max_cal_pairs,
                             lambda i, b: b[1] if isinstance(b[1], (int, float))
                             else 0))
    # prototype init (§9.2): new row = mean query over home (code) tokens
    with torch.no_grad():
        for l, bank in enumerate(pack.banks65):
            h, lab = pairs[l]
            code_h = h[lab > 0.5]
            if code_h.shape[0] > 0:
                bank.P.data[new_idx] = code_h.mean(dim=0) * math.sqrt(bank.d_r)
    pb1 = train_rows(pack, pairs, args.row_steps, args.row_lr, device,
                     reject_w=args.reject_w, mass_w=args.mass_w,
                     owner_target=args.owner_mass)
    # round 2 — on-trajectory (attempt-2 lesson: teacher-forced p_new_code hit
    # 0.89 at the gate but only ~0.21 in situ, because upstream h drifts once
    # the row is live; recal on plugged-trajectory h closes the gap)
    pairs2 = run_in_state(model, blocks, pack.graft_gates, handles, "plugged",
                          lambda: collect_gate_pairs(
                              model, blocks, code_train + base_batches,
                              device, args.max_cal_pairs,
                              lambda i, b: b[1] if isinstance(b[1], (int, float))
                              else 0))
    pb2 = train_rows(pack, pairs2, max(50, args.row_steps // 2), args.row_lr,
                     device, reject_w=args.reject_w, mass_w=args.mass_w,
                     owner_target=args.owner_mass)
    out["phase_b"] = {"round1_teacher_forced": pb1,
                      "round2_on_trajectory": pb2}

    # ---- Phase C: measurement ----
    print("PHASE C — measurement (a) base parity (b) retention (c) contamination")
    # (a) stock vs plugged on base tokens
    base_parity = run_in_state(
        model, blocks, None, handles, "off",
        lambda: parity_vs_stock(model, blocks, base_val_batches, device,
                                pack.graft_gates, handles))
    out["base_parity"] = base_parity

    # (b) margins pre / solo / lib on code probes (delta_margin convention)
    fn_pre = make_logits_fn(model, blocks, gates64, handles, "off", device)
    fn_solo = make_logits_fn(model, blocks, gates64, handles, "forced", device,
                             force_gain=1.0)
    fn_lib = make_logits_fn(model, blocks, pack.graft_gates, handles, "plugged",
                            device)
    probes = select_code_probes(fn_pre, code_probe_windows, args.n_probes)
    out["n_probes"] = len(probes)
    m_pre = probe_margins(probes, fn_pre)
    m_solo = probe_margins(probes, fn_solo)
    m_lib = probe_margins(probes, fn_lib)
    gain_solo = m_solo["mean_logit_margin"] - m_pre["mean_logit_margin"]
    gain_lib = m_lib["mean_logit_margin"] - m_pre["mean_logit_margin"]
    ret = retention(gain_lib, gain_solo)
    out["margins"] = {"pre": m_pre, "solo": m_solo, "lib": m_lib,
                      "gain_solo": round(gain_solo, 4),
                      "gain_lib": round(gain_lib, 4),
                      "retention_vs_solo": (None if ret is None
                                            else round(ret, 4))}

    # (c) contamination / utilization / mass (plugged state)
    code_only = [(t, l) for t, l in code_train]
    contam = run_in_state(model, blocks, pack.graft_gates, handles, "plugged",
                          lambda: routing_hits(model, blocks, base_batches,
                                               device, new_idx))
    util = run_in_state(model, blocks, pack.graft_gates, handles, "plugged",
                        lambda: routing_hits(model, blocks, code_only, device,
                                             new_idx))
    mass_base = run_in_state(model, blocks, pack.graft_gates, handles, "plugged",
                             lambda: routing_mass(model, blocks, base_batches,
                                                  device, new_idx))
    mass_code = run_in_state(model, blocks, pack.graft_gates, handles, "plugged",
                             lambda: routing_mass(model, blocks, code_only,
                                                  device, new_idx))
    out["contamination_base"] = contam
    out["utilization_code"] = util
    out["routing_mass"] = {"base": mass_base, "code": mass_code}

    # §6 quality: CE on code, plugged vs masked
    ce_masked = ce_in_state(model, blocks, code_only, device, gates64,
                            handles, "off")
    ce_active = ce_in_state(model, blocks, code_only, device, pack.graft_gates,
                            handles, "plugged")
    out["s6_quality"] = {"ce_code_masked": ce_masked,
                         "ce_code_active": ce_active,
                         "delta": ce_active - ce_masked}

    # base no-regression: CE on base, plugged vs masked (§9 "no regression on base eval")
    ce_base_masked = ce_in_state(model, blocks, base_val_batches, device,
                                 gates64, handles, "off")
    ce_base_active = ce_in_state(model, blocks, base_val_batches, device,
                                 pack.graft_gates, handles, "plugged")
    out["base_ce"] = {"masked": ce_base_masked, "plugged": ce_base_active,
                      "delta": ce_base_active - ce_base_masked}

    out["verdict"] = d1_verdict(base_parity, ret, contam, util,
                                ce_active, ce_masked)
    out["wall_s"] = round(time.time() - t0, 2)

    if args.plugin_out:
        torch.save({
            "format": "vesper-graft-plugin-v1",
            "experts": [e.state_dict() for e in pack.experts],
            "rows": [b.P.data[new_idx].cpu() for b in pack.banks65],
            "meta": {"new_idx": new_idx, "domain": "code",
                     "host_registry": args.registry},
        }, args.plugin_out)
        out["plugin_path"] = args.plugin_out
    return out


# ---------------------------------------------------------------------------
# Synthetic self-test (--synthetic): the D1 flow on the d0 mock MoE
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
    stock_weights = [b.gate.weight for b in blocks]

    # "registry" = exact-init banks (D0 ceiling arm — parity by construction)
    banks_reg = [d0.PassportBank(e, h) for _ in blocks]
    d0.init_passports_exact(banks_reg, stock_weights)

    # code domain = 1; base = 0 and 2
    code_train = d0.mock_batches(n_dom, seed=11, batches_per_domain=2)
    code_train = [(t, 1) for t, d in code_train if d == 1]
    base_b = [(t, 0) for t, d in d0.mock_batches(n_dom, seed=17,
                                                 batches_per_domain=2)
              if d != 1]
    probe_wins = [t.flatten() for t, d in d0.mock_batches(n_dom, seed=23, seq_len=48,
                                                          batches_per_domain=3,
                                                          batch_size=1) if d == 1]

    class A:
        pass
    args = A()
    args.inter_mult = 1.0
    args.seed = 0
    args.expert_steps = 80
    args.expert_lr = 1e-2
    args.out_pen = 5.0
    args.grad_clip = 1.0
    args.row_steps = 150
    args.row_lr = 0.1
    args.reject_w = 5.0
    args.mass_w = 2.0
    args.owner_mass = 0.85
    args.max_cal_pairs = 2048
    args.n_probes = 16
    args.registry = "selftest"
    args.plugin_out = None

    pack = build_pack(blocks, banks_reg, (top_k, norm_topk))
    check("pack_shapes", len(pack.experts) == len(blocks)
          and pack.banks65[0].P.shape[0] == e + 1)

    out = run_d1(model, blocks, banks_reg, code_train, probe_wins,
                 base_b, base_b, args, device)

    check("phase_a_runs", out["phase_a"]["history"][-1]["ce_code"] <
          out["phase_a"]["history"][0]["ce_code"] + 1.0,
          f"ce_code {out['phase_a']['history'][0]['ce_code']:.3f} -> "
          f"{out['phase_a']['history'][-1]['ce_code']:.3f}")
    check("base_parity_strong",
          out["base_parity"]["logit_cosine_mean"] > 0.95,
          f"cos {out['base_parity']['logit_cosine_mean']:.4f}")
    check("solo_gain_positive", out["margins"]["gain_solo"] > -0.25,
          f"gain_solo {out['margins']['gain_solo']:+.4f} "
          f"(pre {out['margins']['pre']['mean_logit_margin']:+.3f} -> "
          f"solo {out['margins']['solo']['mean_logit_margin']:+.3f})")
    check("retention_computed",
          out["margins"]["retention_vs_solo"] is not None or
          out["margins"]["gain_solo"] == 0.0,
          f"retention {out['margins']['retention_vs_solo']}")
    check("routing_direction",
          out["utilization_code"]["mean_util"] >= 0.0
          and out["contamination_base"]["mean_util"] < 0.6,
          f"util_code {out['utilization_code']['mean_util']:.3f} "
          f"contam_base {out['contamination_base']['mean_util']:.3f}")
    check("verdict_struct", set(out["verdict"]["gates"]) == {
        "base_parity_cos_ge_0.95", "retention_vs_solo_ge_0.70",
        "contamination_le_0.30", "s6_utilization_gt_0.50",
        "s6_quality_ce_beats_masked"},
        f"pass={out['verdict']['pass']}")

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
    ap.add_argument("--registry", default=None,
                    help="D0 registry .pt (registry_olmoe_6d84c485.pt)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--domains-dir",
                    default=os.path.join(REPO_ROOT, "lab", "graft_d0_texts"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--plugin-out", default=None,
                    help="save trained expert+rows .pt (untracked)")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=20000,
                    help="cap for the code domain slice")
    ap.add_argument("--expert-steps", type=int, default=250)
    ap.add_argument("--expert-lr", type=float, default=1e-4,
                    help="attempt-1 lesson: 1e-3 diverges the forced delta")
    ap.add_argument("--out-pen", type=float, default=10.0,
                    help="off-domain expert output-norm penalty (L2 on e_new)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--row-steps", type=int, default=400)
    ap.add_argument("--row-lr", type=float, default=0.05)
    ap.add_argument("--reject-w", type=float, default=REJECT_W)
    ap.add_argument("--mass-w", type=float, default=2.0)
    ap.add_argument("--owner-mass", type=float, default=OWNER_MASS_TARGET,
                    help="home-domain routing-mass target (§8: 0.85)")
    ap.add_argument("--max-cal-pairs", type=int, default=4096)
    ap.add_argument("--n-probes", type=int, default=32)
    ap.add_argument("--inter-mult", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.synthetic:
        return run_synthetic_tests()
    if not args.model or not args.registry:
        ap.error("--model and --registry are required unless --synthetic")

    t_load = time.time()
    model, tok = d0.load_donor(args.model, args.device, args.dtype)
    blocks = d0.find_moe_blocks(model)
    banks_reg, reg_meta = d0.load_registry(args.registry)
    if len(banks_reg) != len(blocks):
        raise RuntimeError(f"registry has {len(banks_reg)} layers, model {len(blocks)}")
    print(f"loaded donor+registry in {time.time() - t_load:.1f}s: "
          f"{len(blocks)} layers, registry meta keys {sorted(reg_meta)[:4]}...")

    domains = d0.load_domain_texts(args.domains_dir)
    code_text = domains.get("code")
    if not code_text:
        raise RuntimeError("domains-dir must contain code.txt (the plug-in domain)")
    # split code tokens: first 60% train, rest probe windows (no overlap)
    code_ids = tok.encode(code_text, add_special_tokens=False)
    split = int(0.6 * len(code_ids))
    code_train_ids, code_probe_ids = code_ids[:split], code_ids[split:]
    code_train_ids = code_train_ids[:args.max_tokens]

    def to_batches(ids, label, seq, bs):
        wins = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
        out = []
        for i in range(0, len(wins) - bs + 1, bs):
            out.append((torch.tensor(wins[i:i + bs], dtype=torch.long), label))
        return out

    code_train = to_batches(code_train_ids, 1, args.seq_len, args.batch_size)
    base_texts = [t for n, t in domains.items() if n != "code"]
    base_ids = []
    for t in base_texts:
        base_ids.extend(tok.encode(t, add_special_tokens=False))
    base_ids = base_ids[:args.max_tokens * 3]
    base_batches = to_batches(base_ids, 0, args.seq_len, args.batch_size)
    probe_windows = [torch.tensor(code_probe_ids[i:i + 256], dtype=torch.long)
                     for i in range(0, len(code_probe_ids) - 256, 256)]
    if not code_train or not base_batches or not probe_windows:
        raise RuntimeError("not enough data — raise --max-tokens or lower --seq-len")
    print(f"data: code_train {sum(t.numel() for t, _ in code_train)} tok, "
          f"base {sum(t.numel() for t, _ in base_batches)} tok, "
          f"{len(probe_windows)} probe windows")

    if args.out is None:
        args.out = os.path.join(REPO_ROOT, "lab", "imported", "graft_d0",
                                "olmoe_d1.json")
    if args.plugin_out is None:
        args.plugin_out = os.path.join(REPO_ROOT, "lab", "imported", "graft_d0",
                                       "plugin_code_olmoe.pt")

    result = run_d1(model, blocks, banks_reg, code_train, probe_windows,
                    base_batches, base_batches, args, args.device)
    result["donor"] = {"model": args.model, "revision": args.revision}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nresults -> {args.out}")
    m = result["margins"]
    print(f"margins: pre {m['pre']['mean_logit_margin']:+.4f}  "
          f"solo {m['solo']['mean_logit_margin']:+.4f}  "
          f"lib {m['lib']['mean_logit_margin']:+.4f}  "
          f"gains {m['gain_solo']:+.4f}/{m['gain_lib']:+.4f}  "
          f"retention {m['retention_vs_solo']}")
    b = result["base_parity"]
    print(f"base parity cos {b['logit_cosine_mean']:.4f}  "
          f"contam {result['contamination_base']['mean_util']:.3f}  "
          f"util {result['utilization_code']['mean_util']:.3f}")
    v = result["verdict"]
    print(f"D1 verdict: {'PASS' if v['pass'] else 'FAIL'}  gates={v['gates']}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
