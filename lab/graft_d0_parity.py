#!/usr/bin/env python3
"""D0 graft-parity harness — MODULAR_MOE.md §9.2 (Variant A) + §9.4 gate D0.

Prepares and (later) measures the D0 gate: passport-graft a stock open-MoE
(donor: allenai/OLMoE-1B-7B-0924) and check that passport routing reproduces
the donor's stock router on a mixed eval, without touching expert weights.

What "graft" means here (§9.2 Variant A, registry path §9.6):
  * donor spine AND experts are untouched (geometry is native — repackaging);
  * ONLY the gate scoring is replaced: stock logits  s_e = h . W[e]  become
    passport scores  s_e = (W_q h . p_e) / sqrt(d_r) * layer_scale
    (§4.3 PassportRouter; d_r = d_model, W_q = identity at t0);
  * top-k selection and the expert merge follow the donor's own semantics
    (OLMoE: softmax over all experts, top-8, NO renorm — norm_topk_prob=false);
  * passports are PRECOMPUTED once (the registry "compile" of §9.6), saved as
    a registry artifact, then grafted at install time. D0 runs both phases on
    the donor itself; D1 later reuses the artifact as the spine's registry.

Phases:
  precompute  profile donor experts on labeled domains (§9.2 step 1: activation
              histograms, home-profile per expert, co-activation graph §9.7),
              init passports (§9.2 step 2: prototype = mean router query over
              the expert's home-profile data; `--init exact` is the
              algebraically-equivalent reference arm), then router-only recal
              on a mixed calibration slice (cheap teacher-forced fit of
              passport scores to stock logits — G2 says budget the recal).
              Writes the registry .pt (§9.6 artifact).
  parity      load registry, graft, and measure stock vs passport on a small
              validation slice: final-logit cosine, CE delta, per-layer MoE
              output cosine, per-layer top-k set agreement, and per-domain
              routing purity vs the donor's own routing-entropy baseline.
  all         precompute + parity in one process.

The synthetic self-test (--synthetic, CPU, seconds) runs the SAME code paths
on a mock MoE (mock router + mock experts with OLMoE-identical interfaces) so
the GPU run is pure measurement: exact-init identity, prototype+recal
convergence, registry round-trip, metric sanity, purity parity, and an
end-to-end two-pass parity measurement are asserted before any donor weights
are loaded.

Usage:
  python3 lab/graft_d0_parity.py --synthetic
  python3 lab/graft_d0_parity.py --model /path/to/OLMoE-1B-7B-0924 \\
      --domains-dir lab/graft_d0_texts --val-file lab/graft_d0_texts/val_mixed.txt \\
      --device cuda --dtype float16 --phase all \\
      --out lab/imported/graft_d0/olmoe_d0.json

PASS bar for D0 (§9.4 says "val parity ... within noise" + "per-domain purity
>= the donor's own routing entropy baseline" — no numbers given; the runbook
lab/graft_d0_runbook.md sets them): mean final-logit cosine > 0.9 (stretch
> 0.99 for `--init exact`), |dCE| < 0.05 nats, per-layer MoE-output cosine
> 0.9 mean, top-k set agreement > 0.8, and purity_graft(d) >= purity_stock(d)
- 0.02 for every domain.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGISTRY_FORMAT = "vesper-graft-registry-v1"

# D0 verdict bars (see lab/graft_d0_runbook.md for the justification)
BAR_LOGIT_COSINE = 0.9
BAR_LOGIT_COSINE_STRETCH = 0.99
BAR_CE_DELTA = 0.05
BAR_LAYER_COSINE = 0.9
BAR_TOPK_AGREE = 0.8
BAR_PURITY_EPS = 0.02


# ---------------------------------------------------------------------------
# Passport scoring (§4.3) — shared by mock and donor
# ---------------------------------------------------------------------------


class PassportBank(nn.Module):
    """Per-layer passport bank P (E, d_r) + optional query map W_q (d_r, d_model).

    score_e = (W_q h . p_e) / sqrt(d_r) * layer_scale

    W_q = None means identity (requires d_r == d_model). With W_q = I and
    p_e = sqrt(d_r) * W_stock[e], the score equals the stock logit exactly —
    that is the `exact` init arm and the algebraic ceiling of the graft.
    """

    def __init__(self, num_experts: int, d_model: int, d_r: int | None = None,
                 learn_wq: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_r = d_model if d_r is None else d_r
        if self.d_r != d_model and not learn_wq:
            raise ValueError("d_r != d_model requires learn_wq=True (non-identity W_q)")
        self.P = nn.Parameter(torch.zeros(num_experts, self.d_r))
        if learn_wq:
            self.W_q = nn.Parameter(torch.eye(self.d_r, self.d_model))
        else:
            self.W_q = None
        self.layer_scale = 1.0

    def queries(self, h: torch.Tensor) -> torch.Tensor:
        h = h.reshape(-1, self.d_model).float()
        return h if self.W_q is None else h @ self.W_q.float().T

    def scores(self, h: torch.Tensor) -> torch.Tensor:
        q = self.queries(h)
        return (q @ self.P.float().T) / math.sqrt(self.d_r) * self.layer_scale


class GraftGate(nn.Module):
    """Drop-in replacement for a stock top-k MoE gate (OlmoeTopKRouter contract).

    forward(h) -> (router_logits, router_scores, router_indices) with the same
    semantics as the donor gate: softmax over all experts (float32), top-k,
    optional renorm of the kept weights. `stock_gate` (optional) lets the graft
    also evaluate the stock route on the same hidden states for recal capture.
    """

    def __init__(self, bank: PassportBank, top_k: int, norm_topk_prob: bool = False,
                 stock_gate: nn.Module | None = None):
        super().__init__()
        self.bank = bank
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.stock_gate = stock_gate
        self.capture = False
        self.last_h = None
        self.last_logits = None
        self.last_stock_logits = None
        self.last_indices = None

    def forward(self, hidden_states: torch.Tensor):
        h = hidden_states.reshape(-1, self.bank.d_model)
        logits = self.bank.scores(h)
        probs = F.softmax(logits, dim=-1)
        top_val, top_idx = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_val = top_val / top_val.sum(dim=-1, keepdim=True)
        top_val = top_val.to(hidden_states.dtype)
        if self.capture:
            self.last_h = h.detach()
            self.last_logits = logits.detach()
            self.last_indices = top_idx.detach()
            if self.stock_gate is not None:
                with torch.no_grad():
                    self.last_stock_logits = F.linear(
                        h.float(), self.stock_gate.weight.float())
        return logits, top_val, top_idx


def install_graft(block: nn.Module, gate: nn.Module) -> nn.Module:
    """Swap block.gate for the graft; returns the stock gate for restore."""
    stock = block.gate
    block.gate = gate
    return stock


def uninstall_graft(block: nn.Module, stock_gate: nn.Module) -> None:
    block.gate = stock_gate


def find_moe_blocks(model: nn.Module) -> list[nn.Module]:
    """Discover MoE blocks duck-typed: .gate with 2D .weight (E, H) + .experts.

    Works for OLMoE's OlmoeSparseMoeBlock and the mock block below.
    """
    blocks = []
    for m in model.modules():
        g = getattr(m, "gate", None)
        ex = getattr(m, "experts", None)
        if g is None or ex is None:
            continue
        w = getattr(g, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2:
            blocks.append(m)
    return blocks


def model_logits(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    out = model(tokens)
    return out.logits if hasattr(out, "logits") else out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def token_logit_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-token cosine over the vocab dim. a,b: (..., V) -> (...) float."""
    a = a.float().reshape(-1, a.shape[-1])
    b = b.float().reshape(-1, b.shape[-1])
    return F.cosine_similarity(a, b, dim=-1)


def next_token_ce(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    """Mean next-token CE. logits (B,T,V), tokens (B,T)."""
    b, t, v = logits.shape
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, v).float(),
        tokens[:, 1:].reshape(-1),
    ).item()


def topk_set_agreement(idx_a: torch.Tensor, idx_b: torch.Tensor) -> tuple[float, float]:
    """Exact-set match rate and mean Jaccard. idx_*: (T, k) expert indices."""
    agree = 0
    jacc = 0.0
    n = idx_a.shape[0]
    for t in range(n):
        sa = set(idx_a[t].tolist())
        sb = set(idx_b[t].tolist())
        agree += int(sa == sb)
        inter = len(sa & sb)
        union = len(sa | sb)
        jacc += inter / union if union else 1.0
    n = max(1, n)
    return agree / n, jacc / n


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two distributions."""
    p = p.float() / p.float().sum().clamp_min(eps)
    q = q.float() / q.float().sum().clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.clamp_min(eps).log() - m.clamp_min(eps).log())).sum()
    kl_qm = (q * (q.clamp_min(eps).log() - m.clamp_min(eps).log())).sum()
    return float(0.5 * (kl_pm + kl_qm))


def usage_purity(usage: torch.Tensor) -> tuple[float, float]:
    """(E,) expert-usage distribution -> (purity, entropy). purity = 1 - H/log E."""
    p = usage.float() / usage.float().sum().clamp_min(1e-12)
    e = -(p * p.clamp_min(1e-12).log()).sum().item()
    emax = math.log(max(2, p.numel()))
    return 1.0 - e / emax, e


# ---------------------------------------------------------------------------
# Route statistics (profiling + purity): per-layer expert activation
# ---------------------------------------------------------------------------


class GateTap:
    """Forward taps on each MoE block's gate (inputs + routing) and output."""

    def __init__(self, blocks: list[nn.Module]):
        self.h: list[torch.Tensor | None] = [None] * len(blocks)
        self.idx: list[torch.Tensor | None] = [None] * len(blocks)
        self.out: list[torch.Tensor | None] = [None] * len(blocks)
        self._handles = []
        for i, blk in enumerate(blocks):
            self._handles.append(blk.gate.register_forward_pre_hook(self._make_pre(i)))
            self._handles.append(blk.gate.register_forward_hook(self._make_gate(i)))
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


@dataclass
class RouteStats:
    """Layer x expert x domain activation bookkeeping (§9.2 step 1, §9.7)."""
    n_layers: int
    num_experts: int
    d_model: int
    n_domains: int
    act_count: torch.Tensor = field(init=False)   # (L, E, D) token hits
    tok_count: torch.Tensor = field(init=False)   # (D,)
    h_sum: torch.Tensor = field(init=False)       # (L, E, D, H) pre-gate hidden sums
    co: torch.Tensor = field(init=False)          # (L, E, E) co-activation

    def __post_init__(self):
        self.act_count = torch.zeros(self.n_layers, self.num_experts, self.n_domains)
        self.tok_count = torch.zeros(self.n_domains)
        self.h_sum = torch.zeros(self.n_layers, self.num_experts, self.n_domains,
                                 self.d_model)
        self.co = torch.zeros(self.n_layers, self.num_experts, self.num_experts)

    def observe(self, layer: int, h: torch.Tensor, top_idx: torch.Tensor,
                domain: int) -> None:
        """h: (T, H) pre-gate hidden; top_idx: (T, k) selected experts."""
        h = h.float().cpu()
        top_idx = top_idx.cpu()
        t = h.shape[0]
        self.tok_count[domain] += t
        onehot = F.one_hot(top_idx, num_classes=self.num_experts).float()  # (T,k,E)
        hit = (onehot.sum(dim=1) > 0).float()  # (T, E)
        self.act_count[layer, :, domain] += hit.sum(dim=0)
        self.h_sum[layer, :, domain] += hit.T @ h
        self.co[layer] += hit.T @ hit

    def home_and_prototypes(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (home_domain (L,E), prototype P0 (L,E,H), mushiness (L,E)).

        home = argmax over domains of the per-domain activation RATE (hits /
        tokens in that domain — raw counts would just track corpus size).
        prototype = mean pre-gate hidden over the expert's home-domain hits
        ("mean router query over the expert's home-profile data", §9.2).
        mushiness = normalized entropy of the expert's domain-rate profile;
        0 = single-domain specialist, 1 = domain-generalist ("mushy passport").
        """
        rate = self.act_count / self.tok_count.clamp_min(1).view(1, 1, -1)  # (L,E,D)
        home = rate.argmax(dim=-1)  # (L,E)
        proto = torch.zeros(self.n_layers, self.num_experts, self.d_model)
        mush = torch.zeros(self.n_layers, self.num_experts)
        for l in range(self.n_layers):
            for e in range(self.num_experts):
                d = int(home[l, e])
                n = self.act_count[l, e, d].item()
                if n > 0:
                    proto[l, e] = self.h_sum[l, e, d] / n
                r = rate[l, e]
                p = r / r.sum().clamp_min(1e-12)
                ent = -(p * p.clamp_min(1e-12).log()).sum().item()
                mush[l, e] = ent / math.log(max(2, self.n_domains))
        return home, proto, mush

    def domain_usage(self) -> torch.Tensor:
        """(D, E) mean-over-layers expert hit-rate per domain (purity input)."""
        rate = self.act_count / self.tok_count.clamp_min(1).view(1, 1, -1)  # (L,E,D)
        return rate.mean(dim=0).T  # (D, E)

    def coactivation_summary(self) -> dict:
        """Top co-activation pairs per layer (§9.7 harvest clusters)."""
        pairs = {}
        for l in range(self.n_layers):
            diag = self.co[l].diag().clamp_min(1)
            c = self.co[l] / diag.view(-1, 1)
            c = 0.5 * (c + c.T)
            c.fill_diagonal_(0.0)
            flat = torch.triu(c, diagonal=1).flatten()
            k = min(5, flat.numel())
            vals, ids = torch.topk(flat, k)
            out = []
            for v, i in zip(vals.tolist(), ids.tolist()):
                if v <= 0:
                    continue
                a, b = divmod(i, self.num_experts)
                out.append([int(a), int(b), round(float(v), 4)])
            pairs[str(l)] = out
        return pairs


# ---------------------------------------------------------------------------
# Passport init + router-only recal (§9.2 step 2)
# ---------------------------------------------------------------------------


def init_passports_exact(banks: list[PassportBank],
                         stock_weights: list[torch.Tensor]) -> None:
    """Algebraically-equivalent reference arm: p_e = sqrt(d_r) * W_stock[e].

    Then score_e = (h . p_e)/sqrt(d_r) = h . W_stock[e] = stock logit exactly.
    This is the ceiling the prototype arm is measured against; a registry
    could ship exactly this when the donor gate is a plain linear row bank.
    The spec's prescribed init is the prototype below.
    """
    for bank, w in zip(banks, stock_weights):
        bank.P.data.copy_(w.float() * math.sqrt(bank.d_r))
        bank.layer_scale = 1.0


def init_passports_prototype(banks: list[PassportBank], stats: RouteStats) -> None:
    """Spec init: p_e = mean router query over the expert's home-profile data."""
    _, proto, _ = stats.home_and_prototypes()
    for l, bank in enumerate(banks):
        bank.P.data.copy_(proto[l].float())
        bank.layer_scale = 1.0


def fit_layer_scales(banks: list[PassportBank],
                     pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    """Closed-form per-layer scalar so prototype scores match stock logit scale."""
    for bank, (h, stock_logits) in zip(banks, pairs):
        s0 = (bank.queries(h) @ bank.P.float().T) / math.sqrt(bank.d_r)
        denom = (s0 * s0).sum().item()
        if denom > 0:
            bank.layer_scale = float(
                (s0 * stock_logits.float()).sum().item() / denom)


@torch.no_grad()
def collect_route_pairs(model: nn.Module, blocks: list[nn.Module],
                        stock_weights: list[torch.Tensor], batches, device,
                        max_pairs_per_layer: int):
    """Collect (h, stock_logits) per layer under the CURRENTLY installed gates.

    stock_logits = F.linear(h, W_stock) on whatever hidden states the current
    trajectory produces. With stock gates installed this is teacher-forced;
    with the graft installed it is on-trajectory (used by --recal-rounds > 1).
    Stride-subsampled to max_pairs_per_layer to bound memory (~34MB/layer at
    4096 pairs x 2048 hidden fp32 on the real donor).
    """
    n = len(blocks)
    hs = [[] for _ in range(n)]
    sts = [[] for _ in range(n)]
    tap = GateTap(blocks)
    for tokens, _dom in batches:
        tap.clear()
        model_logits(model, tokens.to(device))
        for i in range(n):
            hs[i].append(tap.h[i].float().cpu())
            sts[i].append(F.linear(tap.h[i].float(),
                                   stock_weights[i].float().cpu()).cpu())
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


def recalibrate(banks: list[PassportBank],
                pairs: list[tuple[torch.Tensor, torch.Tensor]],
                steps: int, lr: float, kl_w: float = 0.1) -> dict:
    """Router-only recal: fit P to stock logits (MSE + softmax KL) on pairs.

    No model backprop — gradients flow into passport rows only. This is the
    cheap recal the recipe budgets (G2 falsified zero-shot plug-in; §9.6 has
    the registry pay it once at compile time, never the end user).
    """
    report = {"steps": steps, "lr": lr, "kl_w": kl_w, "per_layer": []}
    if steps <= 0:
        return report
    for li, (bank, (h, stock_logits)) in enumerate(zip(banks, pairs)):
        p = nn.Parameter(bank.P.data.float().clone().cpu())
        opt = torch.optim.Adam([p], lr=lr)
        stock_logits = stock_logits.float().cpu()
        h = h.float().cpu()
        first = last = None
        with torch.enable_grad():  # callers may run under no_grad
            for step in range(steps):
                opt.zero_grad()
                q = h if bank.W_q is None else h @ bank.W_q.float().T.cpu()
                s = (q @ p.T) / math.sqrt(bank.d_r) * bank.layer_scale
                mse = F.mse_loss(s, stock_logits)
                kl = F.kl_div(F.log_softmax(s, dim=-1),
                              F.softmax(stock_logits, dim=-1),
                              reduction="batchmean")
                loss = mse + kl_w * kl
                loss.backward()
                opt.step()
                if step == 0:
                    first = float(mse)
                last = float(mse)
        bank.P.data.copy_(p.data.to(bank.P.device))
        report["per_layer"].append(
            {"layer": li, "mse_first": first, "mse_last": last})
    return report


# ---------------------------------------------------------------------------
# Registry (§9.6): precomputed passports, versioned per spine
# ---------------------------------------------------------------------------


def save_registry(path: str, banks: list[PassportBank], meta: dict) -> None:
    payload = {
        "format": REGISTRY_FORMAT,
        "layers": [
            {
                "P": b.P.data.cpu(),
                "W_q": None if b.W_q is None else b.W_q.data.cpu(),
                "layer_scale": float(b.layer_scale),
                "d_r": b.d_r,
                "d_model": b.d_model,
            }
            for b in banks
        ],
        "meta": meta,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_registry(path: str) -> tuple[list[PassportBank], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != REGISTRY_FORMAT:
        raise ValueError(f"unknown registry format: {payload.get('format')}")
    banks = []
    for layer in payload["layers"]:
        bank = PassportBank(
            num_experts=layer["P"].shape[0],
            d_model=layer["d_model"],
            d_r=layer["d_r"],
            learn_wq=layer["W_q"] is not None,
        )
        bank.P.data.copy_(layer["P"])
        if layer["W_q"] is not None:
            bank.W_q.data.copy_(layer["W_q"])
        bank.layer_scale = layer["layer_scale"]
        banks.append(bank)
    return banks, payload["meta"]


# ---------------------------------------------------------------------------
# Two-pass parity measurement (the D0 metric)
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
    layer_cos_sum: torch.Tensor | None = None   # (L,)
    agree_sum: torch.Tensor | None = None       # (L,)
    jacc_sum: torch.Tensor | None = None        # (L,)

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
def parity_measure(model: nn.Module, blocks: list[nn.Module],
                   graft_gates: list[nn.Module], batches, device) -> ParityReport:
    """Stock-pass vs graft-pass on identical batches.

    Per batch: forward with stock gates (assumed installed), then install the
    graft gates, forward again, restore. Compares final logits (per-token
    cosine, next-token CE both routes) and per-layer MoE block outputs
    (cosine) + per-layer top-k set agreement — "cosine/logit agreement,
    per-layer if feasible" for D0.
    """
    n_layers = len(blocks)
    report = ParityReport()
    report.ensure(n_layers)
    for tokens, _dom in batches:
        tokens = tokens.to(device)
        bsz = tokens.shape[0]

        tap_s = GateTap(blocks)
        logits_s = model_logits(model, tokens)
        idx_s = [tap_s.idx[i].clone() for i in range(n_layers)]
        out_s = [tap_s.out[i].clone() for i in range(n_layers)]
        tap_s.close()

        stock_restored = [install_graft(blocks[i], graft_gates[i])
                          for i in range(n_layers)]
        tap_g = GateTap(blocks)
        logits_g = model_logits(model, tokens)
        idx_g = [tap_g.idx[i].clone() for i in range(n_layers)]
        out_g = [tap_g.out[i].clone() for i in range(n_layers)]
        tap_g.close()
        for i in range(n_layers):
            uninstall_graft(blocks[i], stock_restored[i])

        cos = token_logit_cosine(logits_s, logits_g)  # (B*T,)
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
            a, j = topk_set_agreement(idx_s[l].reshape(-1, idx_s[l].shape[-1]),
                                      idx_g[l].reshape(-1, idx_g[l].shape[-1]))
            report.agree_sum[l] += a * bsz
            report.jacc_sum[l] += j * bsz
    return report


# ---------------------------------------------------------------------------
# Per-domain purity vs the donor's own routing baseline (§9.4 D0)
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_route_stats(model: nn.Module, blocks: list[nn.Module], batches,
                        device, n_domains: int,
                        graft_gates: list[nn.Module] | None = None) -> RouteStats:
    """Route stats on labeled batches. graft_gates=None -> stock routing;
    otherwise the graft is installed for the pass (restored afterwards)."""
    n_layers = len(blocks)
    e = blocks[0].gate.weight.shape[0]
    h = blocks[0].gate.weight.shape[1]
    stats = RouteStats(n_layers, e, h, n_domains)
    stock_restored = None
    if graft_gates is not None:
        stock_restored = [install_graft(blocks[i], graft_gates[i])
                          for i in range(n_layers)]
    tap = GateTap(blocks)
    for tokens, dom in batches:
        tap.clear()
        model_logits(model, tokens.to(device))
        for l in range(n_layers):
            stats.observe(l, tap.h[l], tap.idx[l], int(dom))
    tap.close()
    if stock_restored is not None:
        for i in range(n_layers):
            uninstall_graft(blocks[i], stock_restored[i])
    return stats


def purity_report(stats_stock: RouteStats, stats_graft: RouteStats,
                  eps: float = BAR_PURITY_EPS) -> dict:
    """Per-domain: is grafted routing at least as pure as the donor's own?

    purity(d) = 1 - H(usage_d)/log(E) on the domain's expert-usage
    distribution (layer-averaged hit rates). §9.4: "per-domain purity >= the
    donor's own routing entropy baseline" -> purity_graft(d) >=
    purity_stock(d) - eps for every domain. Also reports JS divergence of the
    usage histograms (how far the graft drifted) and the stock entropy
    baseline itself.
    """
    u_s = stats_stock.domain_usage()
    u_g = stats_graft.domain_usage()
    per_domain = []
    all_pass = True
    for d in range(u_s.shape[0]):
        p_s, h_s = usage_purity(u_s[d])
        p_g, h_g = usage_purity(u_g[d])
        ok = p_g >= p_s - eps
        all_pass = all_pass and ok
        per_domain.append({
            "domain": d,
            "purity_stock": p_s,
            "purity_graft": p_g,
            "purity_margin": p_g - p_s,
            "entropy_stock_baseline": h_s,
            "entropy_graft": h_g,
            "js_usage": js_divergence(u_s[d], u_g[d]),
            "pass": ok,
        })
    return {"per_domain": per_domain, "all_pass": all_pass, "eps": eps}


def d0_verdict(parity: dict, purity: dict) -> dict:
    """Assemble the D0 gate verdict from the runbook's numeric bars."""
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
# Donor adapters: mock (synthetic tests) and OLMoE (real)
# ---------------------------------------------------------------------------


class MockGate(nn.Module):
    """Stock top-k gate, same contract as OlmoeTopKRouter: weight (E, H)."""

    def __init__(self, num_experts: int, d_model: int, top_k: int,
                 norm_topk_prob: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_experts, d_model))
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.d_model = d_model

    def forward(self, hidden_states: torch.Tensor):
        h = hidden_states.reshape(-1, self.d_model)
        logits = F.linear(h, self.weight)
        probs = F.softmax(logits.float(), dim=-1)
        top_val, top_idx = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_val = top_val / top_val.sum(dim=-1, keepdim=True)
        return logits, top_val.to(hidden_states.dtype), top_idx


class MockExperts(nn.Module):
    """SwiGLU expert bank with OLMoE-identical merge semantics (index_add)."""

    def __init__(self, num_experts: int, d_model: int, inter: int):
        super().__init__()
        self.num_experts = num_experts
        # scale chosen so block outputs are comparable to the embedding stream
        # (routing changes must move final logits, or the parity metric is blind)
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * inter, d_model) * 0.3)
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, d_model, inter) * 0.3)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            e = int(expert_idx[0])
            k_pos, tok = torch.where(mask[e])
            state = hidden_states[tok]
            gate, up = F.linear(state, self.gate_up_proj[e]).chunk(2, dim=-1)
            out = F.silu(gate) * up
            out = F.linear(out, self.down_proj[e])
            out = out * top_k_weights[tok, k_pos, None]
            final.index_add_(0, tok, out.to(final.dtype))
        return final


class MockSparseMoeBlock(nn.Module):
    """Same forward contract as OlmoeSparseMoeBlock: gate then experts merge."""

    def __init__(self, num_experts: int, d_model: int, inter: int, top_k: int):
        super().__init__()
        self.gate = MockGate(num_experts, d_model, top_k)
        self.experts = MockExperts(num_experts, d_model, inter)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b, t, h = hidden_states.shape
        flat = hidden_states.view(-1, h)
        _, w, idx = self.gate(flat)
        return self.experts(flat, idx, w).reshape(b, t, h)


class MockMoE(nn.Module):
    """Tiny donor: embed -> [MoE block]*L -> lm_head (no attention)."""

    def __init__(self, vocab: int, d_model: int, num_experts: int, inter: int,
                 top_k: int, n_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList(
            [MockSparseMoeBlock(num_experts, d_model, inter, top_k)
             for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for blk in self.layers:
            x = x + blk(x)
        return self.lm_head(x)


def build_mock_donor(seed: int = 7, vocab: int = 90, d_model: int = 32,
                     num_experts: int = 9, top_k: int = 2, n_layers: int = 2,
                     inter: int = 24, n_domains: int = 3):
    """Mock donor with content-aligned routing: tokens of domain d embed near
    centroid d; gate row e points at centroid (e % n_domains) so the stock
    router is domain-specialist (this is the property the profile step must
    recover). Experts stay small-random — only routing matters for D0."""
    g = torch.Generator().manual_seed(seed)
    model = MockMoE(vocab, d_model, num_experts, inter, top_k, n_layers)
    with torch.no_grad():
        centroids = torch.randn(n_domains, d_model, generator=g)
        centroids = centroids / centroids.norm(dim=-1, keepdim=True) * 3.0
        per = vocab // n_domains
        for d in range(n_domains):
            lo = d * per
            hi = (d + 1) * per if d < n_domains - 1 else vocab
            noise = 0.1 * torch.randn(hi - lo, d_model, generator=g)
            model.embed.weight[lo:hi] = centroids[d] + noise
        for l in range(n_layers):
            for e in range(num_experts):
                noise = 0.15 * torch.randn(d_model, generator=g)
                model.layers[l].gate.weight[e] = 2.0 * centroids[e % n_domains] + noise
    return model, n_domains


def mock_batches(n_domains: int = 3, vocab: int = 90, seq_len: int = 24,
                 batches_per_domain: int = 4, batch_size: int = 4, seed: int = 11):
    """Domain-labeled token batches (contiguous id ranges per domain)."""
    g = torch.Generator().manual_seed(seed)
    per = vocab // n_domains
    batches = []
    for d in range(n_domains):
        lo = d * per
        hi = (d + 1) * per if d < n_domains - 1 else vocab
        for _ in range(batches_per_domain):
            toks = torch.randint(lo, hi, (batch_size, seq_len), generator=g)
            batches.append((toks, d))
    return batches


# ---------------------------------------------------------------------------
# Real donor (OLMoE) + text loading
# ---------------------------------------------------------------------------


def load_donor(model_path: str, device: str, dtype_name: str):
    """Load the donor via transformers; returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"float32": torch.float32,
             "float16": torch.float16,
             "bfloat16": torch.bfloat16}[dtype_name]
    tok = AutoTokenizer.from_pretrained(model_path)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:  # transformers < 5 uses torch_dtype=
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.eval()
    model.to(device)
    return model, tok


def load_domain_texts(domains_dir: str) -> dict[str, str]:
    """One domain per <name>.txt (val_mixed.txt is reserved for the val slice)."""
    out = {}
    for fn in sorted(os.listdir(domains_dir)):
        if fn.endswith(".txt") and fn != "val_mixed.txt":
            with open(os.path.join(domains_dir, fn)) as f:
                out[os.path.splitext(fn)[0]] = f.read()
    if not out:
        raise FileNotFoundError(f"no <domain>.txt files in {domains_dir}")
    return out


def texts_to_batches(tokenizer, texts, seq_len: int, batch_size: int,
                     max_tokens: int, label: int) -> list:
    """Tokenize -> concat -> fixed windows -> batches of (tokens (B,T), label)."""
    ids = []
    for t in texts:
        ids.extend(tokenizer.encode(t, add_special_tokens=False))
        if len(ids) >= max_tokens:
            break
    ids = ids[:max_tokens]
    n = (len(ids) // seq_len)
    windows = [ids[i * seq_len:(i + 1) * seq_len] for i in range(n)]
    batches = []
    for i in range(0, len(windows) - batch_size + 1, batch_size):
        rows = windows[i:i + batch_size]
        batches.append((torch.tensor(rows, dtype=torch.long), label))
    return batches


# ---------------------------------------------------------------------------
# Pipeline (precompute / parity / all) — identical for mock and real donor
# ---------------------------------------------------------------------------


def make_graft_gates(blocks, banks, top_k, norm_topk_prob):
    return [GraftGate(bank, top_k, norm_topk_prob, stock_gate=blk.gate)
            for blk, bank in zip(blocks, banks)]


@torch.no_grad()
def run_pipeline(model, blocks, domain_batches, val_batches, n_domains: int,
                 device: str, args, donor_meta: dict) -> dict:
    """D0 precompute + parity. domain_batches must be domain-labeled;
    val_batches carry the mixed eval (label ignored for the headline metric)."""
    top_k = getattr(blocks[0].gate, "top_k", 2)
    norm_topk = bool(getattr(blocks[0].gate, "norm_topk_prob", False))
    stock_weights = [b.gate.weight for b in blocks]
    e, h = stock_weights[0].shape
    out = {"donor": donor_meta, "top_k": top_k, "norm_topk_prob": norm_topk,
           "num_experts": e, "d_model": h, "n_layers": len(blocks),
           "init": args.init, "phase": args.phase}
    t0 = time.time()

    banks = [PassportBank(e, h, learn_wq=args.learn_wq) for _ in blocks]
    graft_gates = make_graft_gates(blocks, banks, top_k, norm_topk)

    # ---- phase: precompute (profile -> init -> recal -> registry) ----
    registry_meta = dict(donor_meta)
    stats_stock = None
    if args.phase in ("precompute", "all"):
        stats_stock = collect_route_stats(model, blocks, domain_batches,
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
        pairs = collect_route_pairs(model, blocks, stock_weights,
                                    domain_batches, device, args.max_cal_pairs)
        fit_layer_scales(banks, pairs)
        recal_report = {"rounds": args.recal_rounds, "round_detail": []}
        for rnd in range(args.recal_rounds):
            if rnd > 0:
                # on-trajectory round: collect pairs under the graft itself
                for i in range(len(blocks)):
                    install_graft(blocks[i], graft_gates[i])
                pairs = collect_route_pairs(model, blocks, stock_weights,
                                            domain_batches, device,
                                            args.max_cal_pairs)
                for i in range(len(blocks)):
                    uninstall_graft(blocks[i], blocks[i].gate.stock_gate)
            r = recalibrate(banks, pairs, args.recal_steps, args.recal_lr,
                            args.recal_kl_w)
            recal_report["round_detail"].append(r)
        out["recal"] = recal_report
        registry_meta.update({
            "init": args.init,
            "recal_steps": args.recal_steps,
            "recal_rounds": args.recal_rounds,
            "recal_lr": args.recal_lr,
            "profile_home_domain": home.tolist(),
            "profile_mushiness_mean": float(mush.mean()),
            "coactivation_top_pairs": out["profile"]["coactivation_top_pairs"],
        })
        if args.registry:
            save_registry(args.registry, banks, registry_meta)
            out["registry_path"] = args.registry

    # ---- phase: parity (load registry if precompute was skipped) ----
    if args.phase in ("parity", "all"):
        if args.phase == "parity":
            banks, loaded_meta = load_registry(args.registry)
            graft_gates = make_graft_gates(blocks, banks, top_k, norm_topk)
            out["registry_meta"] = loaded_meta
        report = parity_measure(model, blocks, graft_gates, val_batches, device)
        out["parity"] = report.as_dict()
        # purity on the labeled domain corpus: stock vs graft usage histograms
        # (stock side is the profiling pass when phase ran it — same batches)
        stats_g = collect_route_stats(model, blocks, domain_batches, device,
                                      n_domains, graft_gates=graft_gates)
        stats_s = stats_stock if stats_stock is not None else \
            collect_route_stats(model, blocks, domain_batches, device, n_domains)
        out["purity"] = purity_report(stats_s, stats_g)
        out["verdict"] = d0_verdict(out["parity"], out["purity"])

    out["wall_s"] = round(time.time() - t0, 2)
    return out


# ---------------------------------------------------------------------------
# Synthetic self-test (--synthetic): the same code paths on a mock MoE
# ---------------------------------------------------------------------------


def run_synthetic_tests() -> int:
    torch.manual_seed(0)
    results = []

    def check(name: str, cond: bool, detail: str = ""):
        results.append((name, bool(cond), detail))
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    device = "cpu"
    model, n_domains = build_mock_donor()
    blocks = find_moe_blocks(model)
    model.eval()
    e, h = blocks[0].gate.weight.shape
    top_k = blocks[0].gate.top_k
    norm_topk = blocks[0].gate.norm_topk_prob
    stock_weights = [b.gate.weight for b in blocks]
    cal_batches = mock_batches(n_domains, seed=11)
    val_batches = mock_batches(n_domains, seed=23, batches_per_domain=3)

    print(f"mock donor: {len(blocks)} MoE layers, E={e}, H={h}, top_k={top_k}, "
          f"domains={n_domains}")

    # 1. exact init: scores == stock logits
    banks = [PassportBank(e, h) for _ in blocks]
    init_passports_exact(banks, stock_weights)
    htest = torch.randn(64, h)
    max_err = max(
        float((banks[i].scores(htest)
               - F.linear(htest, stock_weights[i].float())).abs().max())
        for i in range(len(blocks)))
    check("exact_init_scores_match_stock", max_err < 1e-4,
          f"max|score - stock logit| = {max_err:.2e}")

    # 2. exact init end-to-end: full two-pass parity == 1
    graft_gates = make_graft_gates(blocks, banks, top_k, norm_topk)
    rep = parity_measure(model, blocks, graft_gates, val_batches, device).as_dict()
    check("exact_graft_full_parity",
          rep["logit_cosine_mean"] > 0.999 and abs(rep["ce_delta"]) < 1e-5,
          f"cos={rep['logit_cosine_mean']:.6f} dCE={rep['ce_delta']:.2e} "
          f"agree={min(rep['per_layer_topk_agree']):.3f}")
    check("exact_graft_topk_identical",
          min(rep["per_layer_topk_agree"]) > 0.999,
          f"min-layer agree={min(rep['per_layer_topk_agree']):.4f}")

    # 3. profile recovers the constructed expert->domain specialization
    stats = collect_route_stats(model, blocks, cal_batches, device, n_domains)
    home, proto, mush = stats.home_and_prototypes()
    expected = torch.arange(e).remainder(n_domains)
    hits = int((home[0] == expected).sum())
    check("profile_home_recovery", hits == e,
          f"layer0 homes match e%n_domains for {hits}/{e} experts; "
          f"mean mushiness={float(mush.mean()):.3f}")
    check("prototype_nonzero", float(proto.norm()) > 0)

    # 4. prototype init + recal approaches parity
    banks_p = [PassportBank(e, h) for _ in blocks]
    init_passports_prototype(banks_p, stats)
    pairs = collect_route_pairs(model, blocks, stock_weights, cal_batches,
                                device, max_pairs_per_layer=4096)
    fit_layer_scales(banks_p, pairs)
    graft_p = make_graft_gates(blocks, banks_p, top_k, norm_topk)
    rep0 = parity_measure(model, blocks, graft_p, val_batches, device).as_dict()
    recalibrate(banks_p, pairs, steps=300, lr=0.05, kl_w=0.1)
    graft_p = make_graft_gates(blocks, banks_p, top_k, norm_topk)
    rep1 = parity_measure(model, blocks, graft_p, val_batches, device).as_dict()
    check("parity_measure_sensitive",
          min(rep0["per_layer_moe_cosine"]) < 0.999,
          f"prototype-arm min-layer MoE cos={min(rep0['per_layer_moe_cosine']):.4f} "
          f"(must be <1 or the metric is blind)")
    check("recal_improves_agreement",
          rep1["logit_cosine_mean"] > rep0["logit_cosine_mean"] + 1e-6
          and min(rep1["per_layer_topk_agree"]) >= 0.9,
          f"cos {rep0['logit_cosine_mean']:.4f} -> {rep1['logit_cosine_mean']:.4f}, "
          f"agree {min(rep0['per_layer_topk_agree']):.3f} -> "
          f"{min(rep1['per_layer_topk_agree']):.3f}")
    check("recal_reaches_parity_floor",
          rep1["logit_cosine_mean"] >= 0.98 and abs(rep1["ce_delta"]) < 0.05,
          f"cos={rep1['logit_cosine_mean']:.4f} dCE={rep1['ce_delta']:.4f}")

    # 5. registry round-trip preserves routes bit-near-exactly
    reg_path = os.path.join(REPO_ROOT, "lab", "imported", "graft_d0",
                            "_selftest_registry.pt")
    save_registry(reg_path, banks_p, {"test": True})
    banks_r, meta_r = load_registry(reg_path)
    score_err = max(
        float((banks_r[i].scores(htest) - banks_p[i].scores(htest)).abs().max())
        for i in range(len(blocks)))
    check("registry_roundtrip", score_err < 1e-6 and meta_r.get("test") is True,
          f"max|score delta| = {score_err:.2e}")
    graft_r = make_graft_gates(blocks, banks_r, top_k, norm_topk)
    rep2 = parity_measure(model, blocks, graft_r, val_batches, device).as_dict()
    check("registry_roundtrip_parity_identical",
          abs(rep2["logit_cosine_mean"] - rep1["logit_cosine_mean"]) < 1e-6)
    try:
        os.remove(reg_path)
    except OSError:
        pass

    # 6. metric sanity
    x = torch.randn(16, 32)
    cos_self = float(token_logit_cosine(x, x).min())
    a, j = topk_set_agreement(torch.tensor([[1, 2], [3, 4]]),
                              torch.tensor([[1, 2], [3, 5]]))
    js0 = js_divergence(torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5]))
    pur, ent = usage_purity(torch.tensor([1.0, 0.0, 0.0]))
    check("metric_sanity",
          abs(cos_self - 1.0) < 1e-6 and abs(a - 0.5) < 1e-9
          and abs(j - (1.0 + 1 / 3) / 2) < 1e-6 and js0 == 0.0
          and abs(pur - 1.0) < 1e-6,
          f"cos_self={cos_self:.6f} agree={a} jacc={j:.3f} js={js0} pur={pur:.3f}")

    # 7. install / uninstall restores the stock gate object
    blk = blocks[0]
    stock_obj = blk.gate
    gg = graft_gates[0]
    restored = install_graft(blk, gg)
    check("install_graft_swaps", blk.gate is gg and restored is stock_obj)
    uninstall_graft(blk, restored)
    check("uninstall_graft_restores", blk.gate is stock_obj)

    # 8. purity: exact graft must match the donor's own purity baseline
    stats_g = collect_route_stats(model, blocks, cal_batches, device, n_domains,
                                  graft_gates=make_graft_gates(
                                      blocks, banks, top_k, norm_topk))
    pur_rep = purity_report(stats, stats_g)
    check("purity_exact_graft_not_worse", pur_rep["all_pass"],
          f"margins={[round(d['purity_margin'], 4) for d in pur_rep['per_domain']]}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} synthetic checks passed")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--synthetic", action="store_true",
                    help="run the mock-MoE self-test (CPU, no donor weights)")
    ap.add_argument("--model", default=None,
                    help="donor path or hub id (allenai/OLMoE-1B-7B-0924)")
    ap.add_argument("--revision", default=None,
                    help="hub revision (recorded in the registry meta)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"],
                    help="weights dtype (P40/sm_61: float16 — no bf16 there)")
    ap.add_argument("--phase", default="all",
                    choices=["precompute", "parity", "all"])
    ap.add_argument("--init", default="prototype", choices=["prototype", "exact"],
                    help="prototype = §9.2 spec init; exact = algebraic ceiling")
    ap.add_argument("--recal-steps", type=int, default=300)
    ap.add_argument("--recal-lr", type=float, default=0.05)
    ap.add_argument("--recal-kl-w", type=float, default=0.1)
    ap.add_argument("--recal-rounds", type=int, default=1,
                    help="1 = teacher-forced (stock trajectory); >1 = on-trajectory")
    ap.add_argument("--learn-wq", action="store_true",
                    help="fit a full W_q too (default: identity query map)")
    ap.add_argument("--domains-dir", default=os.path.join(REPO_ROOT, "lab", "graft_d0_texts"))
    ap.add_argument("--val-file", default=None,
                    help="mixed validation text (default: pooled domain texts)")
    ap.add_argument("--registry", default=None,
                    help="registry .pt path (default under lab/imported/graft_d0/)")
    ap.add_argument("--out", default=None, help="results JSON path")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=100_000,
                    help="cap per corpus slice (domain texts and val)")
    ap.add_argument("--max-cal-pairs", type=int, default=4096,
                    help="recal pairs kept per layer")
    ap.add_argument("--top-k", type=int, default=None, help="override donor top-k")
    args = ap.parse_args()

    if args.synthetic:
        return run_synthetic_tests()

    if not args.model:
        ap.error("--model is required unless --synthetic")

    donor_name = args.model.rstrip("/")
    donor_meta = {"model": donor_name, "revision": args.revision, "dtype": args.dtype}
    if args.registry is None:
        stem = os.path.basename(donor_name).replace("/", "_")
        args.registry = os.path.join(REPO_ROOT, "lab", "imported", "graft_d0",
                                     f"registry_{stem}.pt")
    if args.out is None:
        stem = os.path.basename(donor_name).replace("/", "_")
        args.out = os.path.join(REPO_ROOT, "lab", "imported", "graft_d0",
                                f"d0_{stem}_{args.phase}.json")

    t_load = time.time()
    model, tok = load_donor(args.model, args.device, args.dtype)
    blocks = find_moe_blocks(model)
    if not blocks:
        raise RuntimeError("no MoE blocks found (expected .gate/.experts modules)")
    if args.top_k is not None:
        for b in blocks:
            b.gate.top_k = args.top_k
    print(f"donor loaded in {time.time() - t_load:.1f}s: {len(blocks)} MoE layers, "
          f"E={blocks[0].gate.weight.shape[0]}, H={blocks[0].gate.weight.shape[1]}, "
          f"top_k={getattr(blocks[0].gate, 'top_k', '?')}")

    domains = load_domain_texts(args.domains_dir)
    domain_names = list(domains.keys())
    domain_batches = []
    for d, name in enumerate(domain_names):
        per = args.max_tokens // max(1, len(domain_names))
        bs = texts_to_batches(tok, [domains[name]], args.seq_len,
                              args.batch_size, per, label=d)
        domain_batches.extend(bs)
    if args.val_file:
        with open(args.val_file) as f:
            val_text = f.read()
        val_batches = texts_to_batches(tok, [val_text], args.seq_len,
                                       args.batch_size, args.max_tokens, label=-1)
    else:
        val_batches = domain_batches
    if not domain_batches or not val_batches:
        raise RuntimeError("not enough text for one batch — lower --seq-len/--batch-size")
    n_dom_tokens = sum(t.numel() for t, _ in domain_batches)
    n_val_tokens = sum(t.numel() for t, _ in val_batches)
    print(f"data: {len(domain_names)} domains, {n_dom_tokens} domain tokens, "
          f"{n_val_tokens} val tokens, seq_len={args.seq_len}")

    result = run_pipeline(model, blocks, domain_batches, val_batches,
                          len(domain_names), args.device, args, donor_meta)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nresults -> {args.out}")
    if "parity" in result:
        p = result["parity"]
        print(f"logit cosine: {p['logit_cosine_mean']:.4f} "
              f"(min {p['logit_cosine_min']:.4f})  "
              f"CE stock {p['ce_stock']:.4f} graft {p['ce_graft']:.4f} "
              f"delta {p['ce_delta']:+.4f}")
        v = result["verdict"]
        print(f"D0 verdict: {'PASS' if v['pass'] else 'FAIL'}  gates={v['gates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
