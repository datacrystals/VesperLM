#!/usr/bin/env python3
"""D2 cross-family graft-parity harness — MODULAR_MOE.md §9.4 gate D2 (attempt 1).

D0 (lab/graft_d0_parity.py) showed the passport/registry interface reproduces
ONE family's stock router: allenai/OLMoE-1B-7B-0924 — plain linear gate,
softmax over all experts, top-8, no renorm (family: OLMoE).

D2 asks whether the SAME interface generalizes to a FOREIGN family with
different router math. Donor: **microsoft/Phi-mini-MoE-instruct** @ pinned rev
(Phi family / Microsoft "slimmoe", 32 layers, hidden 4096, E=16, top-2):

  * router math is **sparsemixer** (not softmax-topk): sequential top-2 with a
    relative-threshold mask (`> 2*router_jitter_noise` below the running max is
    masked to -inf), each multiplier gathered from a DIFFERENT masked softmax
    (so the two weights do not renormalize together), deterministic at eval;
  * expert block is a ModuleList of SwiGLU MLPs (w1 gate / w3 up / w2 down) in
    the checkpoint vs OLMoE's grouped 3D bank;
  * gate module lives at `block.router` (nn.Linear subclass) vs OLMoE's
    `block.gate`, and is called with the flattened (T, H) hidden states.

The graft replaces ONLY the score function: stock `s = h @ W.T` becomes
passport `s = (W_q h . p_e)/sqrt(d_r) * layer_scale` (§4.3). The donor's OWN
selection semantics (sparsemixer for phimoe, softmax-topk for OLMoE) then run
on the passport scores unchanged — that is what "same interface, foreign
family" means here. Everything else follows the D0 protocol exactly:

  precompute  profile -> prototype passport init -> router-only recal
              (MSE+KL to stock logits) -> registry .pt (same format as D0)
  parity      stock-vs-graft final-logit cosine, CE delta, per-layer MoE-output
              cosine, per-layer top-k set agreement, per-domain purity.

Loading note: Phi-mini-MoE ships the slimmoe checkpoint layout
(`...block_sparse_moe.gate/experts.N.w{1,2,3}`) with custom 4.43-era modeling
code that hard-imports `flash_attn` (unavailable on gfx906/ROCm). We load it
through native transformers `phimoe` after a mechanical key remap
(`w1;w3 -> experts.gate_up_proj`, `w2 -> experts.down_proj`,
`block_sparse_moe.gate -> mlp.router`) — same math, no flash_attn dependency.
The remap is verified by the `--init exact` arm (parity must be 1.0).

gfx906 shims: the two from D0 (grouped-mm off + CPU-sort expert merge) are
reused; the expert-merge shim duck-types `gate_up_proj`/`down_proj`, which
native PhimoeExperts also has.

Usage:
  python3 lab/graft_d2_crossfamily.py --synthetic
  python3 lab/graft_d2_crossfamily.py --model /data/tliao/Phi-mini-MoE-native \\
      --device cuda --dtype float16 --phase all --init prototype \\
      --domains-dir /data/tliao/graft_d2_slice --val-file /data/tliao/graft_d2_slice/val_mixed.txt \\
      --registry lab/imported/graft_d2/registry_phimoe_f620b32c.pt \\
      --out lab/imported/graft_d2/phimoe_d2.json

PASS bars (same as D0 / lab/graft_d0_runbook.md): mean final-logit cosine
> 0.9 (stretch > 0.99), |dCE| < 0.05 nats, per-layer MoE-output cosine > 0.9
mean, top-k set agreement > 0.8, purity_graft(d) >= purity_stock(d) - 0.02.

D2 follow-up (FAILURES d2-crossfamily-perlayer's named lever): **score-margin-
aware recal** — `--recal-margin-w` upweights the MSE on near-threshold expert
pairs (the pairs whose score error flips sparsemixer's discrete top-k / mask
choices). `margin_baseline_analysis` (run in the parity phase) reports the
family-relative agreement baseline: what top-k agreement the family's own
selection gives at the graft's score-error level (noise-matched Monte-Carlo),
the metric proposed for hard-selection router families.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import graft_d0_parity as d0  # noqa: E402  shared passport/metrics/registry core

from graft_d0_parity import (  # noqa: E402
    BAR_CE_DELTA,
    BAR_LAYER_COSINE,
    BAR_LOGIT_COSINE,
    BAR_LOGIT_COSINE_STRETCH,
    BAR_PURITY_EPS,
    BAR_TOPK_AGREE,
    PassportBank,
    RouteStats,
    fit_layer_scales,
    init_passports_exact,
    init_passports_prototype,
    load_domain_texts,
    model_logits,
    next_token_ce,
    patch_rocm_experts,
    patch_rocm_grouped_mm,
    purity_report,
    recalibrate,
    save_registry,
    load_registry,
    texts_to_batches,
    token_logit_cosine,
    topk_set_agreement,
)

FAMILY_PHIMOE = "phimoe"
FAMILY_OLMOE = "olmoe"


# ---------------------------------------------------------------------------
# Selection semantics per family ("different router math is the point")
# ---------------------------------------------------------------------------


def select_softmax_topk(scores: torch.Tensor, top_k: int,
                        norm_topk_prob: bool = False):
    """OLMoE/mock semantics: softmax over all experts, top-k, optional renorm."""
    probs = F.softmax(scores, dim=-1)
    top_val, top_idx = torch.topk(probs, top_k, dim=-1)
    if norm_topk_prob:
        top_val = top_val / top_val.sum(dim=-1, keepdim=True)
    return top_val, top_idx


def _load_sparsemixer():
    """The donor's own sparsemixer — imported so semantics cannot drift."""
    from transformers.models.phimoe.modeling_phimoe import sparsemixer
    return sparsemixer


def select_sparsemixer(scores: torch.Tensor, jitter_eps: float,
                       top_k: int = 2):
    """Phi-MoE semantics: threshold-masked sequential top-2 (sparsemixer).

    Deterministic at eval (training=False path of the donor's own function):
    expert 1 = raw argmax, expert 2 = argmax of scores with expert 1 at -inf;
    multipliers are gathers of two different threshold-masked softmaxes.
    Invariant to positive score rescaling for the INDICES; multipliers are not.
    """
    sparsemixer = _load_sparsemixer()
    multipliers, idx = sparsemixer(
        scores, top_k=top_k, jitter_eps=jitter_eps, training=False)
    return multipliers, idx


@dataclass
class FamilySpec:
    name: str
    gate_attr: str                 # block attribute holding the stock router
    top_k: int
    select: callable               # (scores, meta) -> (weights, idx)
    norm_topk_prob: bool = False


def family_for(name: str, blocks: list[nn.Module]) -> FamilySpec:
    if name == FAMILY_OLMOE:
        g = getattr(blocks[0], "gate")
        return FamilySpec(FAMILY_OLMOE, "gate", int(getattr(g, "top_k", 2)),
                          lambda s, m: select_softmax_topk(
                              s, m.top_k, m.norm_topk_prob),
                          bool(getattr(g, "norm_topk_prob", False)))
    if name == FAMILY_PHIMOE:
        g = getattr(blocks[0], "router")
        eps = float(getattr(g, "router_jitter_noise", 0.0))
        tk = int(getattr(g, "top_k", 2))
        return FamilySpec(FAMILY_PHIMOE, "router", tk,
                          lambda s, m: select_sparsemixer(s, eps, tk))
    raise ValueError(f"unknown family {name!r}")


def detect_family(model: nn.Module) -> str:
    mod = model.__class__.__module__
    if "phimoe" in mod.lower() or "slimmoe" in mod.lower():
        return FAMILY_PHIMOE
    return FAMILY_OLMOE


# ---------------------------------------------------------------------------
# Family-aware graft gate, block discovery, taps
# ---------------------------------------------------------------------------


class GraftGateD2(nn.Module):
    """Drop-in replacement for a stock router: passport scores + donor select.

    Forward contract matches both OlmoeTopKRouter and PhimoeTopKRouter:
    (scores, weights, indices). `stock_gate` keeps the donor's linear rows for
    recal capture: s_stock = F.linear(h, stock_gate.weight).
    """

    def __init__(self, bank: PassportBank, fam: FamilySpec,
                 stock_gate: nn.Module, jitter_eps: float = 0.0):
        super().__init__()
        self.bank = bank
        self.fam = fam
        self.stock_gate = stock_gate
        self.jitter_eps = jitter_eps
        self.capture = False
        self.last_h = None
        self.last_logits = None
        self.last_stock_logits = None
        self.last_indices = None

    def forward(self, hidden_states: torch.Tensor):
        shape = hidden_states.shape
        h = hidden_states.reshape(-1, self.bank.d_model)
        scores = self.bank.scores(h)                       # (T, E) passport logits
        if self.fam.name == FAMILY_PHIMOE:
            weights, idx = select_sparsemixer(scores, self.jitter_eps,
                                              self.fam.top_k)
        else:
            weights, idx = select_softmax_topk(scores, self.fam.top_k,
                                               self.fam.norm_topk_prob)
        weights = weights.to(hidden_states.dtype)
        if self.capture:
            self.last_h = h.detach()
            self.last_logits = scores.detach()
            self.last_indices = idx.detach()
            if self.stock_gate is not None:
                with torch.no_grad():
                    self.last_stock_logits = F.linear(
                        h.float(), self.stock_gate.weight.float())
        return scores, weights, idx


def find_moe_blocks(model: nn.Module) -> list[nn.Module]:
    """Duck-type MoE blocks across families: a router module with a 2D weight
    named `gate` (OLMoE/mock) or `router` (phimoe), plus `.experts`."""
    blocks = []
    for m in model.modules():
        ex = getattr(m, "experts", None)
        if ex is None:
            continue
        for attr in ("gate", "router"):
            g = getattr(m, attr, None)
            if g is None:
                continue
            w = getattr(g, "weight", None)
            if isinstance(w, torch.Tensor) and w.dim() == 2:
                blocks.append(m)
                break
    return blocks


def gate_of(block: nn.Module, fam: FamilySpec) -> nn.Module:
    return getattr(block, fam.gate_attr)


def install_graft(block: nn.Module, fam: FamilySpec,
                  gate: nn.Module) -> nn.Module:
    stock = getattr(block, fam.gate_attr)
    setattr(block, fam.gate_attr, gate)
    return stock


def uninstall_graft(block: nn.Module, fam: FamilySpec,
                    stock_gate: nn.Module) -> None:
    setattr(block, fam.gate_attr, stock_gate)


class GateTapD2:
    """Forward taps on each MoE block's router (input h + selected idx) and the
    block output. Family-aware: hooks `gate` or `router` per FamilySpec."""

    def __init__(self, blocks: list[nn.Module], fam: FamilySpec):
        self.fam = fam
        self.h: list = [None] * len(blocks)
        self.idx: list = [None] * len(blocks)
        self.out: list = [None] * len(blocks)
        self._handles = []
        for i, blk in enumerate(blocks):
            g = gate_of(blk, fam)
            self._handles.append(g.register_forward_pre_hook(self._make_pre(i)))
            self._handles.append(g.register_forward_hook(self._make_gate(i)))
            self._handles.append(blk.register_forward_hook(self._make_out(i)))

    def _make_pre(self, i):
        def pre(mod, inputs):
            self.h[i] = inputs[0].detach()
        return pre

    def _make_gate(self, i):
        def post(mod, inputs, output):
            self.idx[i] = output[2].detach()
        return post

    def _make_out(self, i):
        def post(mod, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            self.out[i] = out.detach()
        return post

    def clear(self):
        for i in range(len(self.h)):
            self.h[i] = self.idx[i] = self.out[i] = None

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def make_graft_gates(blocks, banks, fam: FamilySpec) -> list[GraftGateD2]:
    out = []
    for blk, bank in zip(blocks, banks):
        stock = gate_of(blk, fam)
        eps = float(getattr(stock, "router_jitter_noise", 0.0))
        out.append(GraftGateD2(bank, fam, stock, jitter_eps=eps))
    return out


# ---------------------------------------------------------------------------
# Route pair / route stats collection (family-aware taps)
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_route_pairs(model, blocks, fam: FamilySpec, stock_weights,
                        batches, device, max_pairs_per_layer: int):
    """(h, stock_logits) per layer under the currently installed gates."""
    n = len(blocks)
    hs = [[] for _ in range(n)]
    sts = [[] for _ in range(n)]
    tap = GateTapD2(blocks, fam)
    for tokens, _dom in batches:
        tap.clear()
        model_logits(model, tokens.to(device))
        for i in range(n):
            hs[i].append(tap.h[i].float().cpu())
            sts[i].append(F.linear(tap.h[i].float(),
                                   stock_weights[i].float()).cpu())
    tap.close()
    out = []
    for i in range(n):
        h = torch.cat(hs[i], dim=0)
        s = torch.cat(sts[i], dim=0)
        if h.shape[0] > max_pairs_per_layer:
            stride = h.shape[0] // max_pairs_per_layer + 1
            h = h[::stride][:max_pairs_per_layer]
            s = s[::stride][:max_pairs_per_layer]
        out.append((h, s))
    return out


@torch.no_grad()
def collect_route_stats(model, blocks, fam: FamilySpec, batches, device,
                        n_domains: int,
                        graft_gates: list | None = None) -> RouteStats:
    n = len(blocks)
    w0 = gate_of(blocks[0], fam).weight
    e, h = w0.shape
    stats = RouteStats(n, e, h, n_domains)
    stock_restored = None
    if graft_gates is not None:
        stock_restored = [install_graft(blocks[i], fam, graft_gates[i])
                          for i in range(n)]
    tap = GateTapD2(blocks, fam)
    for tokens, dom in batches:
        tap.clear()
        model_logits(model, tokens.to(device))
        for l in range(n):
            stats.observe(l, tap.h[l], tap.idx[l], int(dom))
    tap.close()
    if stock_restored is not None:
        for i in range(n):
            uninstall_graft(blocks[i], fam, stock_restored[i])
    return stats


# ---------------------------------------------------------------------------
# Score-margin-aware recal (D2 follow-up: FAILURES d2-crossfamily-perlayer)
# ---------------------------------------------------------------------------


def topk_agree_fast(idx_a: torch.Tensor, idx_b: torch.Tensor) -> tuple[float, float]:
    """Vectorized exact-set match rate + mean Jaccard for (T, k) index rows."""
    a, _ = idx_a.sort(dim=-1)
    b, _ = idx_b.sort(dim=-1)
    k = a.shape[-1]
    inter = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(dim=-1).sum(dim=-1).float()
    agree = float((inter >= k).float().mean())
    union = (2 * k - inter).clamp_min(1)
    return agree, float((inter / union).mean())


def decision_cuts(stock_logits: torch.Tensor, top_k: int,
                  family: str, jitter_eps: float) -> list[torch.Tensor]:
    """Per-token decision boundaries of the family's selection.

    In eval the INDEX selection is pure score ranking (top-1 = raw argmax,
    top-2 = argmax of the remainder — the threshold mask is applied only to
    the multiplier softmaxes), so index flips happen at the rank-1/2 and
    rank-k/k+1 midpoints. Sparsemixer multipliers additionally flip where the
    relative-threshold mask edge `((max-s)/max(|s|,max) = 2*jitter_eps)`, i.e.
    `s = max*(1-2*jitter_eps)`, changes which experts contribute to the
    gathered softmax masses.
    """
    s_sorted, _ = stock_logits.sort(dim=-1, descending=True)
    k = max(1, min(top_k, stock_logits.shape[-1] - 1))
    cuts = [0.5 * (s_sorted[:, 0] + s_sorted[:, 1]),
            0.5 * (s_sorted[:, k - 1] + s_sorted[:, k])]
    if family == FAMILY_PHIMOE:
        cuts.append(s_sorted[:, 0] * (1.0 - 2.0 * jitter_eps))
    return cuts


def margin_weights(stock_logits: torch.Tensor, margin_w: float,
                   margin_tau: float = 0.05, top_k: int = 2,
                   family: str = FAMILY_PHIMOE,
                   jitter_eps: float = 0.01) -> torch.Tensor:
    """Per-(token, expert) MSE weights concentrated on near-threshold pairs.

    w_te = 1 + margin_w * exp(-min_c |s*_te - cut_tc| / tau_t), mean-normalized
    to 1 so the loss scale is unchanged and margin_w only re-prioritizes fit
    capacity toward the expert pairs whose stock scores sit on a decision
    boundary — the pairs whose score error flips discrete top-k / mask choices.
    tau_t = margin_tau * per-token score range (scale-free).
    """
    if not margin_w or margin_w <= 0:
        return torch.ones_like(stock_logits)
    s = stock_logits.float()
    s_sorted, _ = s.sort(dim=-1, descending=True)
    tau = (margin_tau * (s_sorted[:, 0] - s_sorted[:, -1])).clamp_min(1e-6)
    dist = None
    for cut in decision_cuts(s, top_k, family, jitter_eps):
        d = (s - cut.unsqueeze(-1)).abs() / tau.unsqueeze(-1)
        dist = d if dist is None else torch.minimum(dist, d)
    w = 1.0 + margin_w * torch.exp(-dist)
    return w / w.mean().clamp_min(1e-12)


def recalibrate_weighted(banks: list[PassportBank],
                         pairs: list[tuple[torch.Tensor, torch.Tensor]],
                         steps: int, lr: float, kl_w: float = 0.1,
                         margin_w: float = 0.0, margin_tau: float = 0.05,
                         top_k: int = 2, family: str = FAMILY_PHIMOE,
                         jitter_eps: float = 0.01) -> dict:
    """Router-only recal with score-margin-aware MSE (the D2 named follow-up).

    Same protocol as d0.recalibrate (fit P + optional W_q with Adam, MSE+KL to
    stock logits, CPU, no model backprop) except the MSE is weighted by
    margin_weights() so near-threshold expert pairs dominate the fit. With
    margin_w=0 the loss reduces to the D2-attempt-1 uniform MSE.
    """
    report = {"steps": steps, "lr": lr, "kl_w": kl_w,
              "margin_w": margin_w, "margin_tau": margin_tau,
              "per_layer": []}
    if steps <= 0:
        return report
    for li, (bank, (h, stock_logits)) in enumerate(zip(banks, pairs)):
        p = nn.Parameter(bank.P.data.float().clone().cpu())
        params = [p]
        wq = None
        if bank.W_q is not None:
            wq = nn.Parameter(bank.W_q.data.float().clone().cpu())
            params.append(wq)
        opt = torch.optim.Adam(params, lr=lr)
        stock_logits = stock_logits.float().cpu()
        h = h.float().cpu()
        w = margin_weights(stock_logits, margin_w, margin_tau, top_k, family,
                           jitter_eps)
        first = last = first_w = last_w = None
        with torch.enable_grad():  # callers may run under no_grad
            for step in range(steps):
                opt.zero_grad()
                q = h if wq is None else h @ wq.T
                s = (q @ p.T) / math.sqrt(bank.d_r) * bank.layer_scale
                err2 = (s - stock_logits).pow(2)
                mse = err2.mean()
                wmse = (err2 * w).mean()
                kl = F.kl_div(F.log_softmax(s, dim=-1),
                              F.softmax(stock_logits, dim=-1),
                              reduction="batchmean")
                loss = wmse + kl_w * kl
                loss.backward()
                opt.step()
                if step == 0:
                    first, first_w = float(mse.detach()), float(wmse.detach())
                last, last_w = float(mse.detach()), float(wmse.detach())
        bank.P.data.copy_(p.data.to(bank.P.device))
        if wq is not None:
            bank.W_q.data.copy_(wq.data.to(bank.W_q.device))
        report["per_layer"].append(
            {"layer": li, "mse_first": first, "mse_last": last,
             "wmse_first": first_w, "wmse_last": last_w})
    return report


@torch.no_grad()
def margin_baseline_analysis(model, blocks, fam: FamilySpec, graft_gates,
                             batches, device, max_batches: int = 4,
                             max_rows: int = 2048, n_draws: int = 8,
                             seed: int = 0) -> dict:
    """Family-relative agreement baseline (the FAILURES.md metric proposal).

    What top-k agreement does THIS family's selection produce at the graft's
    own score-error level? Captures (graft scores, stock scores) on the val
    trajectory, measures per-layer score-error RMS sigma, then Monte-Carlo's
    the family's own selection on stock_scores + N(0, sigma^2) — the agreement
    any score-fitting method would get at that error level — plus the
    flip-prone token fraction (rank-2/3 gap < 2*sigma) and a sigma-multiples
    curve. Graft agreement at/above the noise-matched baseline means the graft
    is at the family's inherent discrete-sensitivity limit for its score error.
    """
    n = len(blocks)
    s_s_l = [[] for _ in range(n)]
    s_g_l = [[] for _ in range(n)]
    idx_g_l = [[] for _ in range(n)]
    stock_restored = [install_graft(blocks[i], fam, graft_gates[i])
                      for i in range(n)]
    for g in graft_gates:
        g.capture = True
    for b_i, (tokens, _dom) in enumerate(batches):
        if b_i >= max_batches:
            break
        model_logits(model, tokens.to(device))
        for i in range(n):
            g = graft_gates[i]
            if g.last_stock_logits is None:
                continue
            s_s_l[i].append(g.last_stock_logits.float().cpu())
            s_g_l[i].append(g.last_logits.float().cpu())
            idx_g_l[i].append(g.last_indices.cpu())
    for g in graft_gates:
        g.capture = False
    for i in range(n):
        uninstall_graft(blocks[i], fam, stock_restored[i])

    jitter_eps = float(getattr(graft_gates[0], "jitter_eps", 0.0))

    def select(scores):
        if fam.name == FAMILY_PHIMOE:
            return select_sparsemixer(scores, jitter_eps, fam.top_k)
        return select_softmax_topk(scores, fam.top_k, fam.norm_topk_prob)

    gen = torch.Generator().manual_seed(seed)
    per_layer = []
    rows_used = 0
    for i in range(n):
        if not s_s_l[i]:
            continue
        s_s = torch.cat(s_s_l[i], 0)
        s_g = torch.cat(s_g_l[i], 0)
        idx_g = torch.cat(idx_g_l[i], 0)
        if s_s.shape[0] > max_rows:
            stride = s_s.shape[0] // max_rows + 1
            s_s = s_s[::stride][:max_rows]
            s_g = s_g[::stride][:max_rows]
            idx_g = idx_g[::stride][:max_rows]
        rows_used = int(s_s.shape[0])
        _, idx_s = select(s_s)
        agree_g, jacc_g = topk_agree_fast(idx_g, idx_s)
        sigma = float((s_g - s_s).pow(2).mean().sqrt())
        ss, _ = s_s.sort(dim=-1, descending=True)
        g12 = ss[:, 0] - ss[:, 1]
        g23 = ss[:, 1] - ss[:, 2]
        curve = {}
        for mult in (0.5, 1.0, 1.5, 2.0):
            sig = sigma * mult
            agrees = []
            for _ in range(n_draws):
                noisy = s_s + sig * torch.randn(s_s.shape, generator=gen)
                _, idx_n = select(noisy)
                a, _jk = topk_agree_fast(idx_n, idx_s)
                agrees.append(a)
            curve[f"{mult}x"] = float(sum(agrees) / len(agrees))
        per_layer.append({
            "layer": i,
            "score_err_rms": sigma,
            "graft_topk_agree": agree_g,
            "graft_topk_jaccard": jacc_g,
            "gap12_mean": float(g12.mean()),
            "gap23_mean": float(g23.mean()),
            "frac_gap23_lt_2sigma": float((g23 < 2 * sigma).float().mean()),
            "noise_matched_agree": curve["1.0x"],
            "noise_matched_agree_curve": curve,
            "agree_ratio": (agree_g / curve["1.0x"]
                            if curve["1.0x"] > 0 else None),
        })

    def mean(key):
        return float(sum(d[key] for d in per_layer) / max(1, len(per_layer)))

    return {
        "per_layer": per_layer,
        "mean_score_err_rms": mean("score_err_rms"),
        "mean_graft_topk_agree": mean("graft_topk_agree"),
        "mean_noise_matched_agree": mean("noise_matched_agree"),
        "mean_agree_ratio": mean("agree_ratio"),
        "noise_matched_curve_means": {
            m: float(sum(d["noise_matched_agree_curve"][m] for d in per_layer)
                     / max(1, len(per_layer)))
            for m in ("0.5x", "1.0x", "1.5x", "2.0x")},
        "n_rows_per_layer": rows_used,
        "n_draws": n_draws,
    }


# ---------------------------------------------------------------------------
# Two-pass parity measurement (same metrics as D0)
# ---------------------------------------------------------------------------


@dataclass
class ParityReport:
    n_tokens: int = 0
    n_batch_rows: int = 0
    logit_cos_sum: float = 0.0
    logit_cos_sq_sum: float = 0.0
    logit_cos_min: float = 1.0
    ce_stock_sum: float = 0.0
    ce_graft_sum: float = 0.0
    layer_cos_sum: torch.Tensor | None = None
    agree_sum: torch.Tensor | None = None
    jacc_sum: torch.Tensor | None = None

    def ensure(self, n_layers: int):
        if self.layer_cos_sum is None:
            self.layer_cos_sum = torch.zeros(n_layers)
            self.agree_sum = torch.zeros(n_layers)
            self.jacc_sum = torch.zeros(n_layers)

    def as_dict(self) -> dict:
        n = max(1, self.n_tokens)
        nb = max(1, self.n_batch_rows)
        mean_cos = self.logit_cos_sum / n
        var = max(0.0, self.logit_cos_sq_sum / n - mean_cos * mean_cos)
        return {
            "n_tokens": self.n_tokens,
            "logit_cosine_mean": mean_cos,
            "logit_cosine_std": math.sqrt(var),
            "logit_cosine_min": self.logit_cos_min if self.n_tokens else None,
            "ce_stock": self.ce_stock_sum / nb,
            "ce_graft": self.ce_graft_sum / nb,
            "ce_delta": (self.ce_graft_sum - self.ce_stock_sum) / nb,
            "per_layer_moe_cosine": (self.layer_cos_sum / nb).tolist()
            if self.layer_cos_sum is not None else [],
            "per_layer_topk_agree": (self.agree_sum / nb).tolist()
            if self.agree_sum is not None else [],
            "per_layer_topk_jaccard": (self.jacc_sum / nb).tolist()
            if self.jacc_sum is not None else [],
        }


@torch.no_grad()
def parity_measure(model, blocks, fam: FamilySpec, graft_gates, batches,
                   device) -> ParityReport:
    n_layers = len(blocks)
    report = ParityReport()
    report.ensure(n_layers)
    for tokens, _dom in batches:
        tokens = tokens.to(device)
        bsz = tokens.shape[0]

        tap_s = GateTapD2(blocks, fam)
        logits_s = model_logits(model, tokens)
        idx_s = [tap_s.idx[i].clone() for i in range(n_layers)]
        out_s = [tap_s.out[i].clone() for i in range(n_layers)]
        tap_s.close()

        stock_restored = [install_graft(blocks[i], fam, graft_gates[i])
                          for i in range(n_layers)]
        tap_g = GateTapD2(blocks, fam)
        logits_g = model_logits(model, tokens)
        idx_g = [tap_g.idx[i].clone() for i in range(n_layers)]
        out_g = [tap_g.out[i].clone() for i in range(n_layers)]
        tap_g.close()
        for i in range(n_layers):
            uninstall_graft(blocks[i], fam, stock_restored[i])

        cos = token_logit_cosine(logits_s, logits_g)
        report.logit_cos_sum += float(cos.sum())
        report.logit_cos_sq_sum += float((cos * cos).sum())
        report.logit_cos_min = min(report.logit_cos_min, float(cos.min()))
        report.n_tokens += cos.numel()
        report.ce_stock_sum += next_token_ce(logits_s, tokens) * bsz
        report.ce_graft_sum += next_token_ce(logits_g, tokens) * bsz
        report.n_batch_rows += bsz
        for l in range(n_layers):
            lc = token_logit_cosine(out_s[l], out_g[l])
            report.layer_cos_sum[l] += float(lc.mean()) * bsz
            a, j = topk_set_agreement(
                idx_s[l].reshape(-1, idx_s[l].shape[-1]),
                idx_g[l].reshape(-1, idx_g[l].shape[-1]))
            report.agree_sum[l] += a * bsz
            report.jacc_sum[l] += j * bsz
    return report


def d2_verdict(parity: dict, purity: dict) -> dict:
    """Same numeric bars as D0 (runbook §PASS); verdict scoped to D2."""
    mean_layer_cos = (sum(parity["per_layer_moe_cosine"])
                      / max(1, len(parity["per_layer_moe_cosine"])))
    mean_agree = (sum(parity["per_layer_topk_agree"])
                  / max(1, len(parity["per_layer_topk_agree"])))
    gates = {
        "logit_cosine_gt_0.9": parity["logit_cosine_mean"] > BAR_LOGIT_COSINE,
        "logit_cosine_stretch_gt_0.99":
            parity["logit_cosine_mean"] > BAR_LOGIT_COSINE_STRETCH,
        "ce_delta_within_0.05": abs(parity["ce_delta"]) < BAR_CE_DELTA,
        "layer_cosine_gt_0.9": mean_layer_cos > BAR_LAYER_COSINE,
        "topk_agree_gt_0.8": mean_agree > BAR_TOPK_AGREE,
        "purity_not_worse_than_stock": purity["all_pass"],
    }
    passed = (gates["logit_cosine_gt_0.9"] and gates["ce_delta_within_0.05"]
              and gates["layer_cosine_gt_0.9"] and gates["topk_agree_gt_0.8"]
              and gates["purity_not_worse_than_stock"])
    return {
        "pass": passed,
        "gates": gates,
        "mean_layer_moe_cosine": mean_layer_cos,
        "mean_topk_agree": mean_agree,
        "bars": {
            "logit_cosine": BAR_LOGIT_COSINE,
            "logit_cosine_stretch": BAR_LOGIT_COSINE_STRETCH,
            "abs_ce_delta": BAR_CE_DELTA,
            "layer_moe_cosine": BAR_LAYER_COSINE,
            "topk_agree": BAR_TOPK_AGREE,
            "purity_eps": BAR_PURITY_EPS,
        },
    }


# ---------------------------------------------------------------------------
# Donor loading: slimmoe checkpoint -> native transformers phimoe
# ---------------------------------------------------------------------------

SLIMMOE_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$")


def remap_slimmoe_checkpoint(src: str, dst: str) -> str:
    """Convert slimmoe-layout weights to native phimoe layout (idempotent).

    gate/up/down per expert -> grouped 3D gate_up_proj (E, 2I, H) + down_proj
    (E, H, I); `block_sparse_moe.gate` -> `mlp.router`; everything else kept.
    Returns the native directory (dst) for from_pretrained.
    """
    from safetensors.torch import load_file, save_file
    if os.path.exists(os.path.join(dst, "model.safetensors")):
        return dst
    os.makedirs(dst, exist_ok=True)
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["architectures"] = ["PhimoeForCausalLM"]
    cfg.pop("auto_map", None)
    cfg["torch_dtype"] = "float16"
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "generation_config.json"):
        p = os.path.join(src, fn)
        if os.path.exists(p):
            open(os.path.join(dst, fn), "wb").write(open(p, "rb").read())

    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    shards = sorted(set(index["weight_map"].values()))
    out: dict = {}
    packs: dict = {}
    for sh in shards:
        sd = load_file(os.path.join(src, sh))
        for k, v in sd.items():
            m = SLIMMOE_EXPERT_RE.match(k)
            if m:
                layer, expert, which = int(m.group(1)), int(m.group(2)), m.group(3)
                packs.setdefault((layer, expert), {})[which] = v.contiguous()
            elif k.endswith("block_sparse_moe.gate.weight"):
                layer = k.split(".")[2]
                out[f"model.layers.{layer}.mlp.router.weight"] = v.contiguous()
            else:
                out[k] = v.contiguous()
        del sd
    # group packs per layer -> native 3D banks gate_up_proj (E, 2I, H),
    # down_proj (E, H, I); expert order must match the checkpoint expert ids
    per_layer: dict = {}
    for (layer, expert), w in packs.items():
        if set(w) != {"w1", "w2", "w3"}:
            raise RuntimeError(f"layer {layer} expert {expert}: got {sorted(w)}")
        per_layer.setdefault(layer, {})[expert] = w
    for layer, experts in per_layer.items():
        ids = sorted(experts)
        if ids != list(range(len(ids))):
            raise RuntimeError(f"layer {layer}: non-contiguous experts {ids}")
        out[f"model.layers.{layer}.mlp.experts.gate_up_proj"] = torch.stack(
            [torch.cat([experts[i]["w1"], experts[i]["w3"]], dim=0)
             for i in ids], dim=0).contiguous()
        out[f"model.layers.{layer}.mlp.experts.down_proj"] = torch.stack(
            [experts[i]["w2"] for i in ids], dim=0).contiguous()
    save_file(out, os.path.join(dst, "model.safetensors"))
    print(f"slimmoe->native remap: {len(out)} tensors -> {dst}")
    return dst


def load_donor(model_path: str, device: str, dtype_name: str):
    """Load the donor (auto-remapping a slimmoe checkpoint if needed)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"float32": torch.float32,
             "float16": torch.float16,
             "bfloat16": torch.bfloat16}[dtype_name]
    status = patch_rocm_grouped_mm()
    print(f"rocm grouped-mm shim: {status}")

    cfg_path = os.path.join(model_path, "config.json")
    is_slim = False
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        is_slim = ("PhiMoEForCausalLM" in (cfg.get("architectures") or [])
                   or "auto_map" in cfg)
    if is_slim:
        print("slimmoe checkpoint detected -> remapping to native phimoe layout")
        model_path = remap_slimmoe_checkpoint(
            model_path, model_path.rstrip("/") + "-native")

    tok = AutoTokenizer.from_pretrained(model_path)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:  # transformers < 5 uses torch_dtype=
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.eval()
    model.to(device)
    n_patched = patch_rocm_experts(model)
    if n_patched:
        print(f"rocm experts shim: patched {n_patched} expert banks "
              f"(CPU-sort merge)")
    return model, tok


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def texts_to_batches_chunked(tokenizer, texts, seq_len: int, batch_size: int,
                             max_tokens: int, label: int,
                             max_chars: int = 8000) -> list:
    """Like d0.texts_to_batches but encodes in <=max_chars slices.

    The donor tokenizer warns ("Token indices sequence length is longer than
    the specified maximum ... 20717 > 4096") when asked to encode a whole
    multi-K-token fixture in one call. Chunking the text before encode keeps
    every encode() call under the donor's 4096 context. Honest caveat: a BPE
    merge cannot cross a chunk boundary, so the stream may differ from a
    whole-string encode by at most one token per chunk (few tens of tokens in
    an 88k-token slice) — irrelevant at 512-token windowing, and it is what
    the user asked for in the resume brief.
    """
    pieces = []
    for t in texts:
        for i in range(0, len(t), max_chars):
            pieces.append(t[i:i + max_chars])
    return d0.texts_to_batches(tokenizer, pieces, seq_len, batch_size,
                               max_tokens, label)


def dump_json(out: dict, path: str | None) -> None:
    """Incremental results write — a killed run still leaves usable JSON."""
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, path)


@torch.no_grad()
def run_pipeline(model, blocks, fam: FamilySpec, domain_batches, val_batches,
                 n_domains: int, device: str, args, donor_meta: dict) -> dict:
    stock_gate0 = gate_of(blocks[0], fam)
    stock_weights = [gate_of(b, fam).weight for b in blocks]
    e, h = stock_weights[0].shape
    out = {"donor": donor_meta, "family": fam.name,
           "router_semantics": ("sparsemixer-top2-threshold" if fam.name == FAMILY_PHIMOE
                                else f"softmax-top{fam.top_k}"
                                     f"{'+renorm' if fam.norm_topk_prob else '-norenorm'}"),
           "top_k": fam.top_k, "norm_topk_prob": fam.norm_topk_prob,
           "num_experts": e, "d_model": h, "n_layers": len(blocks),
           "init": args.init, "phase": args.phase}
    t0 = time.time()

    d_r = args.passport_dim or h
    learn_wq = bool(args.learn_wq) or (d_r != h)
    if d_r != h and not learn_wq:
        raise ValueError("passport_dim != d_model requires a query map")
    banks = [PassportBank(e, h, d_r=d_r, learn_wq=learn_wq) for _ in blocks]
    graft_gates = make_graft_gates(blocks, banks, fam)

    registry_meta = dict(donor_meta)
    stats_stock = None
    if args.phase in ("precompute", "all"):
        stats_stock = collect_route_stats(model, blocks, fam, domain_batches,
                                          device, n_domains)
        home, _, mush = stats_stock.home_and_prototypes()
        out["profile"] = {
            "home_domain": home.tolist(),
            "mushiness_mean": float(mush.mean()),
            "mushiness_per_expert": mush.mean(dim=0).tolist(),
            "coactivation_top_pairs": stats_stock.coactivation_summary(),
            "act_count_sum": stats_stock.act_count.sum().item(),
            "tok_count": stats_stock.tok_count.tolist(),
        }
        if args.init == "exact":
            init_passports_exact(banks, stock_weights)
        else:
            init_passports_prototype(banks, stats_stock)
        pairs = collect_route_pairs(model, blocks, fam, stock_weights,
                                    domain_batches, device, args.max_cal_pairs)
        fit_layer_scales(banks, pairs)
        recal_report = {"rounds": args.recal_rounds,
                        "margin_w": args.recal_margin_w,
                        "margin_tau": args.recal_margin_tau,
                        "round_detail": []}
        jitter_eps = float(getattr(stock_gate0, "router_jitter_noise", 0.0))
        for rnd in range(args.recal_rounds):
            if rnd > 0:
                for i in range(len(blocks)):
                    install_graft(blocks[i], fam, graft_gates[i])
                pairs = collect_route_pairs(model, blocks, fam, stock_weights,
                                            domain_batches, device,
                                            args.max_cal_pairs)
                for i in range(len(blocks)):
                    uninstall_graft(blocks[i], fam, graft_gates[i].stock_gate)
            r = recalibrate_weighted(
                banks, pairs, args.recal_steps, args.recal_lr, args.recal_kl_w,
                margin_w=args.recal_margin_w, margin_tau=args.recal_margin_tau,
                top_k=fam.top_k, family=fam.name, jitter_eps=jitter_eps)
            recal_report["round_detail"].append(r)
        out["recal"] = recal_report
        out["passport_dim"] = d_r
        out["query_map"] = "identity" if banks[0].W_q is None else "learned"
        registry_meta.update({
            "family": fam.name,
            "init": args.init,
            "recal_steps": args.recal_steps,
            "recal_rounds": args.recal_rounds,
            "recal_lr": args.recal_lr,
            "recal_margin_w": args.recal_margin_w,
            "recal_margin_tau": args.recal_margin_tau,
            "passport_dim": d_r,
            "query_map": out["query_map"],
            "profile_home_domain": home.tolist(),
            "profile_mushiness_mean": float(mush.mean()),
            "coactivation_top_pairs": out["profile"]["coactivation_top_pairs"],
        })
        if args.registry:
            save_registry(args.registry, banks, registry_meta)
            out["registry_path"] = args.registry
        out["wall_s"] = round(time.time() - t0, 2)
        out["status"] = "precompute_done"
        dump_json(out, args.out)

    if args.phase in ("parity", "all"):
        if args.phase == "parity":
            banks, loaded_meta = load_registry(args.registry)
            graft_gates = make_graft_gates(blocks, banks, fam)
            out["registry_meta"] = loaded_meta
        report = parity_measure(model, blocks, fam, graft_gates, val_batches,
                                device)
        out["parity"] = report.as_dict()
        out["wall_s"] = round(time.time() - t0, 2)
        out["status"] = "parity_done"
        dump_json(out, args.out)
        out["margin_analysis"] = margin_baseline_analysis(
            model, blocks, fam, graft_gates, val_batches, device)
        dump_json(out, args.out)
        stats_g = collect_route_stats(model, blocks, fam, domain_batches,
                                      device, n_domains, graft_gates=graft_gates)
        stats_s = stats_stock if stats_stock is not None else \
            collect_route_stats(model, blocks, fam, domain_batches, device,
                                n_domains)
        out["purity"] = purity_report(stats_s, stats_g)
        out["verdict"] = d2_verdict(out["parity"], out["purity"])

    out["wall_s"] = round(time.time() - t0, 2)
    out["status"] = "complete"
    return out


# ---------------------------------------------------------------------------
# Synthetic self-test: mock phimoe-family block (sparsemixer router)
# ---------------------------------------------------------------------------


class MockPhiRouter(nn.Linear):
    """PhimoeTopKRouter contract: linear scores + sparsemixer selection."""

    def __init__(self, num_experts: int, d_model: int, jitter_eps: float = 0.01):
        super().__init__(d_model, num_experts, bias=False)
        self.router_jitter_noise = jitter_eps
        self.top_k = 2
        self.num_experts = num_experts

    def forward(self, hidden_states: torch.Tensor):
        logits = super().forward(hidden_states)
        w, idx = select_sparsemixer(logits, self.router_jitter_noise, self.top_k)
        return logits, w, idx


class MockPhiExperts(nn.Module):
    """SwiGLU bank in the native phimoe 3D layout (gate_up_proj/down_proj)."""

    def __init__(self, num_experts: int, d_model: int, inter: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * inter, d_model))
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, d_model, inter))
        self.act_fn = F.silu
        nn.init.normal_(self.gate_up_proj, std=0.05)
        nn.init.normal_(self.down_proj, std=0.05)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        num_tokens, hidden_dim = hidden_states.shape
        final = torch.zeros_like(hidden_states)
        for e in range(self.num_experts):
            mask = (top_k_index == e)
            if not mask.any():
                continue
            tok, slot = torch.where(mask)
            x = hidden_states[tok]
            gate, up = F.linear(x, self.gate_up_proj[e]).chunk(2, dim=-1)
            y = self.act_fn(gate) * up
            y = F.linear(y, self.down_proj[e])
            y = y * top_k_weights[tok, slot].unsqueeze(-1)
            final.index_add_(0, tok, y.to(final.dtype))
        return final


class MockPhiSparseMoeBlock(nn.Module):
    """Same forward contract as PhimoeSparseMoeBlock: router then experts."""

    def __init__(self, num_experts: int, d_model: int, inter: int):
        super().__init__()
        self.router = MockPhiRouter(num_experts, d_model)
        self.experts = MockPhiExperts(num_experts, d_model, inter)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b, t, h = hidden_states.shape
        flat = hidden_states.reshape(-1, h)
        _, w, idx = self.router(flat)
        return self.experts(flat, idx, w).reshape(b, t, h)


class MockPhiMoE(nn.Module):
    def __init__(self, vocab: int, d_model: int, num_experts: int, inter: int,
                 n_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList(
            [MockPhiSparseMoeBlock(num_experts, d_model, inter)
             for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for blk in self.layers:
            x = x + blk(x)
        return self.lm_head(x)


def build_mock_phi_donor(seed: int = 11, vocab: int = 90, d_model: int = 32,
                         num_experts: int = 6, n_layers: int = 2,
                         inter: int = 24, n_domains: int = 3):
    g = torch.Generator().manual_seed(seed)
    model = MockPhiMoE(vocab, d_model, num_experts, inter, n_layers)
    with torch.no_grad():
        centroids = torch.randn(n_domains, d_model, generator=g)
        centroids = centroids / centroids.norm(dim=-1, keepdim=True) * 3.0
        per = vocab // n_domains
        for d in range(n_domains):
            lo = d * per
            hi = (d + 1) * per if d < n_domains - 1 else vocab
            model.embed.weight[lo:hi] = centroids[d] + 0.1 * torch.randn(
                hi - lo, d_model, generator=g)
        for l in range(n_layers):
            for e in range(num_experts):
                model.layers[l].router.weight[e] = (
                    2.0 * centroids[e % n_domains]
                    + 0.15 * torch.randn(d_model, generator=g))
    return model, n_domains


def run_synthetic_tests() -> int:
    torch.manual_seed(0)
    results = []

    def check(name: str, cond: bool, detail: str = ""):
        results.append((name, bool(cond), detail))
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    device = "cpu"

    # A. sparsemixer determinism + contract (the foreign selection semantics)
    scores = torch.randn(64, 6)
    w1, i1 = select_sparsemixer(scores, 0.01, 2)
    w2, i2 = select_sparsemixer(scores, 0.01, 2)
    check("sparsemixer_deterministic_eval",
          torch.equal(i1, i2) and torch.equal(w1, w2),
          f"idx shape {tuple(i1.shape)}")
    check("sparsemixer_topk2_distinct",
          bool((i1[:, 0] != i1[:, 1]).all()),
          "the two selected experts are always distinct")

    model, n_domains = build_mock_phi_donor()
    model.eval()
    blocks = find_moe_blocks(model)
    fam = family_for(FAMILY_PHIMOE, blocks)
    check("find_blocks_router_attr",
          len(blocks) == 2 and fam.gate_attr == "router",
          f"{len(blocks)} blocks, gate_attr={fam.gate_attr}")
    e, h = gate_of(blocks[0], fam).weight.shape
    stock_weights = [gate_of(b, fam).weight for b in blocks]
    cal_batches = d0.mock_batches(n_domains, seed=11)
    val_batches = d0.mock_batches(n_domains, seed=23, batches_per_domain=3)
    print(f"mock phi donor: {len(blocks)} MoE layers, E={e}, H={h}, "
          f"top_k={fam.top_k}, domains={n_domains}")

    # B. exact init: passport scores == stock logits -> parity 1.0
    banks = [PassportBank(e, h) for _ in blocks]
    init_passports_exact(banks, stock_weights)
    htest = torch.randn(64, h)
    max_err = max(
        float((banks[i].scores(htest)
               - F.linear(htest, stock_weights[i].float())).abs().max())
        for i in range(len(blocks)))
    check("exact_init_scores_match_stock", max_err < 1e-4,
          f"max|score - stock| = {max_err:.2e}")
    graft = make_graft_gates(blocks, banks, fam)
    rep = parity_measure(model, blocks, fam, graft, val_batches,
                         device).as_dict()
    check("exact_graft_full_parity",
          rep["logit_cosine_mean"] > 0.999 and abs(rep["ce_delta"]) < 1e-5,
          f"cos={rep['logit_cosine_mean']:.6f} dCE={rep['ce_delta']:.2e}")
    check("exact_graft_topk_identical",
          min(rep["per_layer_topk_agree"]) > 0.999,
          f"min-layer agree={min(rep['per_layer_topk_agree']):.4f}")

    # C. prototype + recal moves toward the bars
    stats = collect_route_stats(model, blocks, fam, cal_batches, device,
                                n_domains)
    home, proto, mush = stats.home_and_prototypes()
    n_rec = int((home == torch.tensor(
        [[e_ % n_domains for e_ in range(e)]] * len(blocks))).sum())
    check("profile_home_recovery", n_rec == len(blocks) * e,
          f"{n_rec}/{len(blocks) * e} experts recovered")
    banks_p = [PassportBank(e, h) for _ in blocks]
    init_passports_prototype(banks_p, stats)
    pairs = collect_route_pairs(model, blocks, fam, stock_weights, cal_batches,
                                device, max_pairs_per_layer=4096)
    fit_layer_scales(banks_p, pairs)
    graft_p = make_graft_gates(blocks, banks_p, fam)
    rep1 = parity_measure(model, blocks, fam, graft_p, val_batches,
                          device).as_dict()
    recalibrate(banks_p, pairs, steps=300, lr=0.05, kl_w=0.1)
    graft_p2 = make_graft_gates(blocks, banks_p, fam)
    rep2 = parity_measure(model, blocks, fam, graft_p2, val_batches,
                          device).as_dict()
    check("prototype_init_sane", rep1["logit_cosine_mean"] > 0.5,
          f"cos={rep1['logit_cosine_mean']:.4f} (pre-recal)")
    check("recal_improves",
          rep2["logit_cosine_mean"] > rep1["logit_cosine_mean"] + 1e-6
          and abs(rep2["ce_delta"]) < abs(rep1["ce_delta"]) + 1e-9,
          f"cos {rep1['logit_cosine_mean']:.4f} -> {rep2['logit_cosine_mean']:.4f}, "
          f"dCE {rep1['ce_delta']:.4f} -> {rep2['ce_delta']:.4f}")
    check("recal_passes_bars",
          rep2["logit_cosine_mean"] > BAR_LOGIT_COSINE
          and abs(rep2["ce_delta"]) < BAR_CE_DELTA,
          f"cos={rep2['logit_cosine_mean']:.4f} dCE={rep2['ce_delta']:.4f}")

    # D. registry round-trip (same artifact format as D0)
    reg_path = os.path.join(REPO_ROOT, "lab", "imported", "graft_d2",
                            "_selftest_registry.pt")
    save_registry(reg_path, banks_p, {"test": True, "family": fam.name})
    banks_r, meta_r = load_registry(reg_path)
    err = max(float((banks_r[i].scores(htest) - banks_p[i].scores(htest))
                    .abs().max()) for i in range(len(blocks)))
    check("registry_roundtrip",
          err < 1e-6 and meta_r.get("family") == fam.name,
          f"max|score delta| = {err:.2e}")
    try:
        os.remove(reg_path)
    except OSError:
        pass

    # E. purity parity on the mock
    stats_g = collect_route_stats(model, blocks, fam, cal_batches, device,
                                  n_domains, graft_gates=graft_p2)
    pur = purity_report(stats, stats_g)
    check("purity_not_worse", pur["all_pass"],
          f"margins={[round(d['purity_margin'], 4) for d in pur['per_domain']]}")

    # F. OLMoE-family regression: the D0 path still wires through the adapter
    model_o, nd_o = d0.build_mock_donor()
    model_o.eval()
    blocks_o = find_moe_blocks(model_o)
    fam_o = family_for(FAMILY_OLMOE, blocks_o)
    check("olmoe_family_adapter",
          fam_o.gate_attr == "gate" and fam_o.top_k == 2,
          f"gate_attr={fam_o.gate_attr} top_k={fam_o.top_k}")
    eo, ho = gate_of(blocks_o[0], fam_o).weight.shape
    banks_o = [PassportBank(eo, ho) for _ in blocks_o]
    init_passports_exact(banks_o, [gate_of(b, fam_o).weight for b in blocks_o])
    graft_o = make_graft_gates(blocks_o, banks_o, fam_o)
    val_o = d0.mock_batches(nd_o, seed=23, batches_per_domain=3)
    rep_o = parity_measure(model_o, blocks_o, fam_o, graft_o, val_o,
                           device).as_dict()
    check("olmoe_exact_parity_via_adapter",
          rep_o["logit_cosine_mean"] > 0.999,
          f"cos={rep_o['logit_cosine_mean']:.6f}")

    # G. score-margin-aware recal (the D2 named follow-up)
    sl = torch.randn(256, 8)
    w0 = margin_weights(sl, 0.0)
    check("margin_w0_is_uniform",
          float(w0.min()) == 1.0 and float(w0.max()) == 1.0,
          "margin_w=0 reduces to uniform MSE weights")
    w = margin_weights(sl, 5.0, 0.05, top_k=2, family=FAMILY_PHIMOE,
                       jitter_eps=0.01)
    check("margin_w_mean_one", abs(float(w.mean()) - 1.0) < 1e-5,
          f"mean={float(w.mean()):.6f} (loss scale preserved)")
    ss, _ = sl.sort(dim=-1, descending=True)
    cuts = torch.stack([0.5 * (ss[:, 0] + ss[:, 1]),
                        0.5 * (ss[:, 1] + ss[:, 2]),
                        ss[:, 0] * 0.98], dim=0)
    dist = torch.stack([(sl - b.unsqueeze(-1)).abs() for b in cuts],
                       0).min(0).values
    tau = (0.05 * (ss[:, 0] - ss[:, -1])).clamp_min(1e-6).unsqueeze(-1)
    near, far = dist < tau, dist > 3 * tau
    check("margin_weights_focus_near_threshold",
          bool(near.any() and far.any()
               and float(w[near].mean()) > float(w[far].mean()) + 0.5),
          f"near-cut w={float(w[near].mean()):.2f} vs far w={float(w[far].mean()):.2f}")
    banks_m = [PassportBank(e, h) for _ in blocks]
    init_passports_prototype(banks_m, stats)
    fit_layer_scales(banks_m, pairs)
    rep_m_pre = parity_measure(model, blocks, fam,
                               make_graft_gates(blocks, banks_m, fam),
                               val_batches, device).as_dict()
    rr = recalibrate_weighted(banks_m, pairs, steps=300, lr=0.05, kl_w=0.1,
                              margin_w=5.0, margin_tau=0.05, top_k=fam.top_k,
                              family=fam.name, jitter_eps=0.01)
    rep_m = parity_measure(model, blocks, fam,
                           make_graft_gates(blocks, banks_m, fam),
                           val_batches, device).as_dict()
    wmse_down = all(l["wmse_last"] < l["wmse_first"] for l in rr["per_layer"])
    check("recal_margin_fits_and_passes_bars",
          wmse_down and rep_m["logit_cosine_mean"] > BAR_LOGIT_COSINE
          and abs(rep_m["ce_delta"]) < BAR_CE_DELTA,
          f"wmse↓={wmse_down} cos {rep_m_pre['logit_cosine_mean']:.4f} -> "
          f"{rep_m['logit_cosine_mean']:.4f} dCE={rep_m['ce_delta']:+.4f}")

    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_pass}/{len(results)} synthetic checks passed")
    return 0 if n_pass == len(results) else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="run the mock-donor self-tests and exit")
    ap.add_argument("--model", default=None,
                    help="donor path (slimmoe or native phimoe layout)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--family", default="auto",
                    choices=["auto", "phimoe", "olmoe"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--phase", default="all",
                    choices=["precompute", "parity", "all"])
    ap.add_argument("--init", default="prototype", choices=["prototype", "exact"])
    ap.add_argument("--recal-steps", type=int, default=300)
    ap.add_argument("--recal-lr", type=float, default=0.05)
    ap.add_argument("--recal-kl-w", type=float, default=0.1)
    ap.add_argument("--recal-rounds", type=int, default=1)
    ap.add_argument("--recal-margin-w", type=float, default=0.0,
                    help="score-margin-aware recal: MSE weight boost on "
                         "near-threshold expert pairs (0 = uniform MSE, the "
                         "D2-attempt-1 loss)")
    ap.add_argument("--recal-margin-tau", type=float, default=0.05,
                    help="near-threshold width as a fraction of the per-token "
                         "score range")
    ap.add_argument("--learn-wq", action="store_true")
    ap.add_argument("--passport-dim", type=int, default=None)
    ap.add_argument("--domains-dir", default=os.path.join(REPO_ROOT, "lab",
                                                          "graft_d0_texts"))
    ap.add_argument("--val-file", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=100_000)
    ap.add_argument("--max-cal-pairs", type=int, default=4096)
    args = ap.parse_args()

    if args.synthetic:
        return run_synthetic_tests()
    if not args.model:
        ap.error("--model is required unless --synthetic")

    t_load = time.time()
    model, tok = load_donor(args.model, args.device, args.dtype)
    load_s = round(time.time() - t_load, 1)
    blocks = find_moe_blocks(model)
    if not blocks:
        print("FAIL: no MoE blocks discovered", file=sys.stderr)
        return 2
    fam_name = detect_family(model) if args.family == "auto" else args.family
    fam = family_for(fam_name, blocks)
    print(f"donor loaded in {load_s}s: family={fam.name}, "
          f"{len(blocks)} MoE layers, gate_attr={fam.gate_attr}, "
          f"top_k={fam.top_k}, router_semantics="
          f"{'sparsemixer' if fam.name == FAMILY_PHIMOE else 'softmax-topk'}")

    domains = load_domain_texts(args.domains_dir)
    domain_batches = []
    for d, (name, text) in enumerate(sorted(domains.items())):
        domain_batches.extend(texts_to_batches_chunked(
            tok, [text], args.seq_len, args.batch_size,
            args.max_tokens // max(1, len(domains)), d))
    val_path = args.val_file or os.path.join(args.domains_dir, "val_mixed.txt")
    with open(val_path) as f:
        val_text = f.read()
    val_batches = texts_to_batches_chunked(tok, [val_text], args.seq_len,
                                           args.batch_size, 10 ** 9, -1)
    n_val = sum(int(t.numel()) for t, _ in val_batches)
    n_dom = sum(int(t.numel()) for t, _ in domain_batches)
    print(f"data: {n_dom} domain tokens ({len(domains)} domains) + "
          f"{n_val} val tokens")

    donor_meta = {
        "model": args.model,
        "revision": args.revision,
        "dtype": args.dtype,
        "family": fam.name,
        "load_s": load_s,
    }
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out = run_pipeline(model, blocks, fam, domain_batches, val_batches,
                       len(domains), args.device, args, donor_meta)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        out["peak_vram_gib"] = round(
            torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        print(f"peak VRAM: {out['peak_vram_gib']} GiB")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}")
    v = out.get("verdict")
    if v:
        print(f"\nD2 verdict: {'PASS' if v['pass'] else 'FAIL'}")
        for k, ok in v["gates"].items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
        p = out["parity"]
        print(f"  logit_cos={p['logit_cosine_mean']:.5f} "
              f"(min {p['logit_cosine_min']:.4f}) "
              f"dCE={p['ce_delta']:+.5f} "
              f"layer_cos={v['mean_layer_moe_cosine']:.4f} "
              f"agree={v['mean_topk_agree']:.4f}")
        ma = out.get("margin_analysis")
        if ma:
            print(f"  margin_baseline: score_rms={ma['mean_score_err_rms']:.5f} "
                  f"noise_matched_agree={ma['mean_noise_matched_agree']:.4f} "
                  f"agree_ratio={ma['mean_agree_ratio']:.3f}")
    print(f"wall {out['wall_s']}s")
    return 0 if (v is None or v["pass"]) else 1


if __name__ == "__main__":
    sys.exit(main())
