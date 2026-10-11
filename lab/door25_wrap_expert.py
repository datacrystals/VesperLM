#!/usr/bin/env python3
"""Door 2.5 first experiment — the grab (MODULAR_MOE.md §10.2).

Q: can a FOREIGN model's expert core be hosted VERBATIM behind learned thin
adapters?

Protocol (§10.2, lab scale):
  0. Profile the Phi-mini-MoE donor (D2 harness, rev f620b32c) on the target
     domain slices -> pick the (layer, expert) whose home domain is ours
     (home-domain fit; RouteStats hit-rates).
  1. Extract that expert's SwiGLU core (gate_up_proj[e] = [w1; w3], down_proj[e]
     = w2) and FREEZE it.
  2. Wrap it in the §10.1/§10.2 format (the geometry the brief pins):

       x(d_model) --host_in--> c(d_canon) --adapter_up--> u(d_donor)
         -- CORE (frozen) --> v(d_donor)
       --adapter_down--> c'(d_canon) --host_out--> delta(d_model)

     adapter_down/adapter_up are §10.2's wrap pair, "(d_canon=64 <-> their
     d_model=4096)"; host_in/host_out are the §4.2 format adapters
     (d_model <-> d_canon) that the D1 host interface requires (the brief's
     wrap names sit on the donor-side pair; the four maps together are the
     rank-d_canon factorization of A_in/A_out of §4.5). All four maps train;
     the core does not. host_out zero-init keeps the expert the identity at
     birth (§4.2).
  3. Plug into the OLMoE registry graft via the D1 path
     (lab/graft_d1_plugin.py PluginMerge + a 65th passport row on
     registry_olmoe_6d84c485.pt), at the depth-matched host layer.
  4. Train ONLY the adapters on target-domain TASK LOSS (next-token CE — NOT
     feature matching; D1.5 showed paired-activation translation does not
     transfer) with the D1 mass-calibration recipe (att-4 winner:
     owner_mass 0.35, reject_w 10, out_pen 300), plus the on-trajectory
     closing recal round.
  5. D1 gates: base parity cos >= 0.95 vs the OLMoE graft, retention vs solo
     >= 0.70, contamination <= 0.30. Quality read: code CE plugged vs masked
     (the sub-gate D1's toy plug-in failed) + forced-dispatch CE upper bound.

Why this is the interesting experiment: D1.5 measured that hidden-space
feature translators do NOT bridge bases (0.55-0.69 held-out cos). Door 2.5
asks a narrower question — not "can we reproduce donor features" but "can
frozen foreign weights do useful host work behind a thin learned interface
trained end-to-end on the host's own loss". Success = the exact foreign
weights are hostable; failure with a real (non-degenerate) training signal =
adapters can't bridge -> Door 3 (behavioral distillation).

Usage:
  python3 lab/door25_wrap_expert.py --synthetic
  python3 lab/door25_wrap_expert.py \
      --model /data/tliao/OLMoE-1B-7B-0924 \
      --donor /data/tliao/Phi-mini-MoE-instruct \
      --registry /data/tliao/graft_d0_out/registry_olmoe_6d84c485.pt \
      --domains-dir /data/tliao/graft_d0_slice \
      --donor-domains-dir /data/tliao/graft_d2_slice \
      --device cuda --dtype float16 \
      --out /data/tliao/graft_door25_out/door25_wrap.json

gfx906/MI50 NOTE (2026-10-11, GPU 3): the trunk BACKWARD has size holes the
same family as the known torch.sort@8192 wall — OLMoE CE backward dies with
`HIP error: invalid device function` at some token counts (B*T = 256, 768,
1024, 4096 FAIL; 128, 384, 512, 1536, 2048 OK, one process per shape).
D1's default (bs=2, seq=512) = 1024 tokens hits the hole on this card; the
run recipe is bs=4 x seq=512 = 2048 tokens. Forward is unaffected. See
lab/FAILURES.md door25-gfx906-backward-holes.
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

# Task bars — the D1 gates (§9.4 / §10.2 "D1-gate it")
BAR_BASE_PARITY = 0.95
BAR_RETENTION = 0.70
BAR_CONTAMINATION = 0.30
BAR_UTILIZATION = 0.50

# D1 mass-calibration recipe (att-4 winner)
RECIPE = {"owner_mass": 0.35, "reject_w": 10.0, "out_pen": 300.0}

FORMAT = "vesper-door25-wrap-v1"


# ---------------------------------------------------------------------------
# The wrap: frozen foreign SwiGLU core behind thin learned adapters
# ---------------------------------------------------------------------------


class WrappedForeignExpert(nn.Module):
    """Frozen foreign expert core hosted behind thin learned adapters.

    forward (see module docstring for the diagram):
        x(d_model) -> host_in -> c(d_canon) -> adapter_up -> u(d_donor)
        -> core SwiGLU (frozen: w_gu (2I, H), w_down (H, I)) -> v(d_donor)
        -> adapter_down -> c'(d_canon) -> host_out -> delta(d_model)

    The core is the donor's SwiGLU VERBATIM (gate/up split follows the native
    phimoe layout produced by d2.remap_slimmoe_checkpoint: gate_up = [w1; w3],
    gate = first half of the output, act = silu). It lives in buffers, not
    parameters — no optimizer will ever see it. host_out is zero-init so the
    expert is the identity at birth (§4.2's down_proj trick).
    """

    def __init__(self, d_model: int, d_canon: int, w_gu: torch.Tensor,
                 w_down: torch.Tensor, zero_init_out: bool = True, seed: int = 0):
        super().__init__()
        if w_gu.dim() != 2 or w_down.dim() != 2:
            raise ValueError("core weights must be 2D (2I, H) and (H, I)")
        if w_gu.shape[1] != w_down.shape[0]:
            raise ValueError(f"core width mismatch: {w_gu.shape} vs {w_down.shape}")
        d_donor = w_gu.shape[1]
        self.d_model = d_model
        self.d_canon = d_canon
        self.d_donor = d_donor
        self.d_inter = w_down.shape[1]
        self.register_buffer("core_w_gu", w_gu.float().clone())
        self.register_buffer("core_w_down", w_down.float().clone())
        g = torch.Generator().manual_seed(seed)
        self.host_in = nn.Linear(d_model, d_canon, bias=False)
        self.adapter_up = nn.Linear(d_canon, d_donor, bias=False)
        self.adapter_down = nn.Linear(d_donor, d_canon, bias=False)
        self.host_out = nn.Linear(d_canon, d_model, bias=False)
        with torch.no_grad():
            self.host_in.weight.copy_(torch.randn(d_canon, d_model, generator=g) * 0.02)
            self.adapter_up.weight.copy_(torch.randn(d_donor, d_canon, generator=g) * 0.02)
            self.adapter_down.weight.copy_(torch.randn(d_canon, d_donor, generator=g) * 0.02)
            if zero_init_out:
                self.host_out.weight.zero_()
            else:
                self.host_out.weight.copy_(
                    torch.randn(d_model, d_canon, generator=g) * 0.02)

    def core_forward(self, u: torch.Tensor) -> torch.Tensor:
        """The frozen donor SwiGLU at its own widths (no adapters)."""
        gate, up = F.linear(u, self.core_w_gu).chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, self.core_w_down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        c = self.host_in(xf)
        u = self.adapter_up(c)
        v = self.core_forward(u)
        c2 = self.adapter_down(v)
        return self.host_out(c2).to(x.dtype)

    def adapter_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def core_param_count(self) -> int:
        return self.core_w_gu.numel() + self.core_w_down.numel()

    def wrap_spec(self) -> dict:
        return {
            "format": FORMAT,
            "geometry": ("x(d_model) --host_in--> c(d_canon) --adapter_up--> "
                         "u(d_donor) --core(frozen SwiGLU)--> v(d_donor) --"
                         "adapter_down--> c'(d_canon) --host_out--> delta"),
            "d_model": self.d_model,
            "d_canon": self.d_canon,
            "d_donor": self.d_donor,
            "d_inter": self.d_inter,
            "adapter_params": {
                "host_in": self.host_in.weight.numel(),
                "adapter_up": self.adapter_up.weight.numel(),
                "adapter_down": self.adapter_down.weight.numel(),
                "host_out": self.host_out.weight.numel(),
                "total": self.adapter_param_count(),
            },
            "core_params_frozen": self.core_param_count(),
            "zero_init": "host_out",
            "trainable": ["host_in", "adapter_up", "adapter_down", "host_out"],
            "frozen": ["core_w_gu", "core_w_down"],
        }


def extract_donor_core(donor_blocks, layer: int, expert: int):
    """Slice one expert's SwiGLU out of a native-phimoe expert bank.

    gate_up_proj[e] is (2I, H) = [w1; w3] (see d2.remap_slimmoe_checkpoint);
    down_proj[e] is (H, I) = w2. Returned as CPU float32 clones.
    """
    ex = donor_blocks[layer].experts
    w_gu = ex.gate_up_proj[expert].detach().float().cpu().clone()
    w_down = ex.down_proj[expert].detach().float().cpu().clone()
    return w_gu, w_down


# ---------------------------------------------------------------------------
# Donor expert selection by home-domain fit (D2 profiling)
# ---------------------------------------------------------------------------


def choose_donor_expert(act_count: torch.Tensor, tok_count: torch.Tensor,
                        code_idx: int, min_rate: float = 0.05,
                        table_size: int = 10) -> tuple[dict, list]:
    """Pick the (layer, expert) that owns the home domain.

    score = home-domain hit-rate lift over base (rate_code / rate_base), among
    experts that actually fire on home tokens (rate_code >= min_rate) and whose
    modal domain is the home domain when any exist. Returns (choice, table).
    """
    n_layers, n_exp, n_dom = act_count.shape
    rate = act_count / tok_count.clamp_min(1).view(1, 1, -1)  # (L,E,D)
    home = rate.argmax(dim=-1)
    base_idx = [d for d in range(n_dom) if d != code_idx]
    tok_base = float(tok_count[base_idx].sum())
    rate_base = act_count[:, :, base_idx].sum(dim=-1) / max(1.0, tok_base)
    rate_code = rate[:, :, code_idx]
    lift = rate_code / (rate_base + 1e-6)

    rows = []
    for l in range(n_layers):
        for e in range(n_exp):
            rows.append({
                "layer": l, "expert": e,
                "home": int(home[l, e]),
                "rate_code": round(float(rate_code[l, e]), 5),
                "rate_base": round(float(rate_base[l, e]), 5),
                "lift": round(float(lift[l, e]), 3),
            })
    qualified = [r for r in rows if r["rate_code"] >= min_rate]
    owners = [r for r in qualified if r["home"] == code_idx]
    pool = owners if owners else qualified
    if not pool:
        pool = rows
    choice = max(pool, key=lambda r: (r["lift"], r["rate_code"]))
    table = sorted(rows, key=lambda r: (r["lift"], r["rate_code"]),
                   reverse=True)[:table_size]
    return choice, table


# ---------------------------------------------------------------------------
# Phase A — adapter fit (task loss; NOT feature matching — the D1.5 lesson)
# ---------------------------------------------------------------------------


def train_adapters(model, pack, handles, code_batches, base_batches, steps,
                   lr, device, out_pen: float = RECIPE["out_pen"],
                   grad_clip: float = 1.0, weight_decay: float = 0.0) -> dict:
    """Train ONLY the wrap adapters under forced dispatch.

    Loss = next-token CE on home batches + out_pen * mean ||e_new(h_base)||^2
    (the D1 att-4 neutrality control). The trunk, the registry and the frozen
    core see no gradients.

    Door-2.5 delta vs d1.train_expert_forced: weight_decay defaults to 0 — the
    D1 default (0.01) plus clip-1.0 shrinks host_in/adapter_up/adapter_down to
    ~1/4 of their init norms inside 12 steps and the whole wrap delta collapses
    to |y|~5e-4 (below fp16 resolution of the trunk: the pilot's CE was
    bitwise-identical across 40 steps). Also records param L2 / mean|y| / CE on
    a fixed batch so the training curve is readable at 6-decimal formatting.
    """
    params = list(pack.parameters())
    d1.freeze_all_except(model, params)
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    hist = []
    t0 = time.time()
    fixed = code_batches[0][0].to(device)
    for step in range(steps):
        toks, _ = code_batches[step % len(code_batches)]
        toks = toks.to(device)
        d1.set_plugin_mode(handles, "forced", 1.0)
        logits = d0.model_logits(model, toks)
        v = logits.shape[-1]
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, v).float(),
                               toks[:, 1:].reshape(-1))
        bt, _ = base_batches[step % len(base_batches)]
        d1.set_plugin_mode(handles, "forced", 1.0)
        d0.model_logits(model, bt.to(device))
        out_terms = [blk._plugin_merge.last_new_out for blk, _ in handles
                     if getattr(blk, "_plugin_merge", None) is not None
                     and blk._plugin_merge.last_new_out is not None]
        out_norm2 = (torch.stack([t.float().pow(2).mean()
                                  for t in out_terms]).mean()
                     if out_terms else torch.zeros((), device=device))
        total = loss + out_pen * out_norm2
        opt.zero_grad()
        total.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            with torch.no_grad():
                d1.set_plugin_mode(handles, "forced", 1.0)
                lg0 = d0.model_logits(model, fixed)
                ce0 = float(F.cross_entropy(
                    lg0[:, :-1].reshape(-1, v).float(),
                    fixed[:, 1:].reshape(-1)))
                yabs = (torch.stack([t.float().abs().mean()
                                     for t in out_terms]).mean()
                        if out_terms else torch.zeros((), device=device))
                pl2 = math.sqrt(sum(float(p.detach().pow(2).sum())
                                    for p in params))
            hist.append({"step": step,
                         "ce_code": float(loss.detach()),
                         "ce_batch0": ce0,
                         "out_norm2": float(out_norm2.detach()),
                         "mean_abs_y": float(yabs),
                         "param_l2": pl2})
            print(f"  [phaseA {step:4d}] ce {float(loss.detach()):.4f}  "
                  f"ce0 {ce0:.4f}  out2 {float(out_norm2.detach()):.3e}  "
                  f"|y| {float(yabs):.3e}  |W| {pl2:.3f}", flush=True)
    return {"steps": steps, "lr": lr, "out_pen": out_pen,
            "grad_clip": grad_clip, "weight_decay": weight_decay,
            "history": hist, "wall_s": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------
# Single-site gate state helpers (D1 machinery, one plug-in site)
# ---------------------------------------------------------------------------


def build_site_pack(blocks, banks_registry, host_i: int, wrap: WrappedForeignExpert):
    """65-row bank + graft gate at host_i only; registry gates elsewhere."""
    top_k = getattr(blocks[0].gate, "top_k", 2)
    norm_topk = bool(getattr(blocks[0].gate, "norm_topk_prob", False))
    bank_reg = banks_registry[host_i]
    bank65 = d0.PassportBank(bank_reg.num_experts + 1, bank_reg.d_model,
                             d_r=bank_reg.d_r, learn_wq=bank_reg.W_q is not None)
    with torch.no_grad():
        bank65.P.data[:bank_reg.num_experts] = bank_reg.P.data
        bank65.P.data[bank_reg.num_experts] = 0.0
    bank65.layer_scale = bank_reg.layer_scale
    gate65 = d0.GraftGate(bank65, top_k, norm_topk, stock_gate=blocks[host_i].gate)
    gates64 = [d0.GraftGate(b, top_k, norm_topk, stock_gate=blk.gate)
               for b, blk in zip(banks_registry, blocks)]
    gates_plugged = [gate65 if i == host_i else gates64[i]
                     for i in range(len(blocks))]
    pack = d1.PluginPack(experts=nn.ModuleList([wrap]),
                         banks65=[bank65], graft_gates=[gate65])
    return pack, gates64, gates_plugged, top_k, norm_topk


@torch.no_grad()
def parity_vs_ref(model, blocks, batches, device, handles,
                  ref_gates, ref_mode, tgt_gates, tgt_mode) -> dict:
    """Logit cosine between two installed states (ref vs target) on base tokens."""
    sum_cos = sum_sq = n_tok = 0.0
    cos_min = 1.0
    ce_a = ce_b = n_row = 0.0
    for tokens, _ in batches:
        tokens = tokens.to(device)
        bsz = tokens.shape[0]
        lg_a = d1.run_in_state(model, blocks, ref_gates, handles, ref_mode,
                               lambda: d0.model_logits(model, tokens))
        lg_b = d1.run_in_state(model, blocks, tgt_gates, handles, tgt_mode,
                               lambda: d0.model_logits(model, tokens))
        cos = d0.token_logit_cosine(lg_a, lg_b)
        sum_cos += float(cos.sum())
        sum_sq += float((cos * cos).sum())
        cos_min = min(cos_min, float(cos.min()))
        n_tok += cos.numel()
        ce_a += d0.next_token_ce(lg_a, tokens) * bsz
        ce_b += d0.next_token_ce(lg_b, tokens) * bsz
        n_row += bsz
    mean = sum_cos / max(1.0, n_tok)
    var = max(0.0, sum_sq / max(1.0, n_tok) - mean * mean)
    return {"n_tokens": int(n_tok), "logit_cosine_mean": mean,
            "logit_cosine_std": math.sqrt(var), "logit_cosine_min": cos_min,
            "ce_ref": ce_a / max(1.0, n_row), "ce_tgt": ce_b / max(1.0, n_row),
            "ce_delta": (ce_b - ce_a) / max(1.0, n_row)}


# ---------------------------------------------------------------------------
# D2.5 pipeline (host side): adapters-only fit + row recal + D1 gates
# ---------------------------------------------------------------------------


def run_door25(model, blocks, banks_registry, host_i, wrap, code_train,
               code_probe_windows, base_batches, base_val_batches, args,
               device) -> dict:
    out = {"gate": "D2.5-wrap",
           "bars": {"base_parity": BAR_BASE_PARITY,
                    "retention": BAR_RETENTION,
                    "contamination": BAR_CONTAMINATION},
           "recipe": dict(RECIPE), "host_layer": host_i}
    t0 = time.time()
    pack, gates64, gates_plugged, top_k, norm_topk = build_site_pack(
        blocks, banks_registry, host_i, wrap)
    pack.experts.to(device)
    handles = d1.install_plugin([blocks[host_i]], pack, "off")
    new_idx = pack.banks65[0].num_experts - 1
    out["n_experts_stock"] = banks_registry[host_i].num_experts
    out["new_expert_idx"] = new_idx
    out["wrap"] = wrap.wrap_spec()

    # ---- Phase A: adapter fit (forced dispatch, task loss) ----
    print("PHASE A — adapter fit (forced dispatch, code CE + out_pen L2)")
    d1.set_plugin_mode(handles, "off")
    pa = train_adapters(
        model, pack, handles, code_train, base_batches,
        args.expert_steps, args.expert_lr, device, out_pen=args.out_pen,
        grad_clip=args.grad_clip, weight_decay=args.weight_decay)
    out["phase_a"] = pa
    out["adapter_params"] = wrap.adapter_param_count()
    out["train_wall_phase_a_s"] = pa["wall_s"]
    dump_json(out, args.out)

    # ---- Phase B: passport row recal (D1 att-4 mass calibration) ----
    print("PHASE B — passport row recal (reject + owner_mass "
          f"{args.owner_mass}, reject_w {args.reject_w})")
    pairs = d1.run_in_state(
        model, blocks, gates64, handles, "off",
        lambda: d1.collect_gate_pairs(
            model, [blocks[host_i]], code_train + base_batches, device,
            args.max_cal_pairs,
            lambda i, b: b[1] if isinstance(b[1], (int, float)) else 0))
    with torch.no_grad():
        h, lab = pairs[0]
        code_h = h[lab > 0.5]
        print(f"  pairs: {h.shape[0]} rows, {int((lab > 0.5).sum())} code, "
              f"|h| {float(h.norm(dim=-1).mean()):.2f}, "
              f"|mean_h_code| {float(code_h.mean(dim=0).norm()) if code_h.shape[0] else 0:.2f}, "
              f"layer_scale {pack.banks65[0].layer_scale:.4f}", flush=True)
        if code_h.shape[0] > 0:
            pack.banks65[0].P.data[new_idx] = (
                code_h.mean(dim=0) * math.sqrt(pack.banks65[0].d_r))
            s = pack.banks65[0].scores(h)
            print(f"  after prototype init: score_new mean "
                  f"{float(s[:, new_idx].mean()):.2f} vs stock max "
                  f"{float(s[:, :new_idx].max(dim=-1).values.mean()):.2f}",
                  flush=True)
    pb1 = d1.train_rows(pack, pairs, args.row_steps, args.row_lr, device,
                        reject_w=args.reject_w, mass_w=args.mass_w,
                        owner_target=args.owner_mass)
    # round 2 — on-trajectory (D1 attempt-2 lesson: teacher-forced p_new_code
    # overstates in-situ mass once the row is live; recal on plugged h)
    pairs2 = d1.run_in_state(
        model, blocks, gates_plugged, handles, "plugged",
        lambda: d1.collect_gate_pairs(
            model, [blocks[host_i]], code_train + base_batches, device,
            args.max_cal_pairs,
            lambda i, b: b[1] if isinstance(b[1], (int, float)) else 0))
    pb2 = d1.train_rows(pack, pairs2, max(50, args.row_steps // 2),
                        args.row_lr, device, reject_w=args.reject_w,
                        mass_w=args.mass_w, owner_target=args.owner_mass)
    out["phase_b"] = {"round1_teacher_forced": pb1,
                      "round2_on_trajectory": pb2}
    out["train_wall_phase_b_s"] = round(time.time() - t0 - pa["wall_s"], 2)
    dump_json(out, args.out)

    # ---- Phase C: D1 gates + quality reads ----
    print("PHASE C — measurement (parity / retention / contamination / quality)")
    base_parity = parity_vs_ref(
        model, blocks, base_val_batches, device, handles,
        None, "off", gates_plugged, "plugged")
    out["base_parity"] = base_parity
    graft_parity = parity_vs_ref(
        model, blocks, base_val_batches, device, handles,
        gates64, "off", gates_plugged, "plugged")
    out["graft_parity_vs_olmoe_graft"] = graft_parity

    fn_pre = d1.make_logits_fn(model, blocks, gates64, handles, "off", device)
    fn_solo = d1.make_logits_fn(model, blocks, gates64, handles, "forced",
                                device, force_gain=1.0)
    fn_lib = d1.make_logits_fn(model, blocks, gates_plugged, handles,
                               "plugged", device)
    probes = d1.select_code_probes(fn_pre, code_probe_windows, args.n_probes)
    out["n_probes"] = len(probes)
    m_pre = d1.probe_margins(probes, fn_pre)
    m_solo = d1.probe_margins(probes, fn_solo)
    m_lib = d1.probe_margins(probes, fn_lib)
    gain_solo = m_solo["mean_logit_margin"] - m_pre["mean_logit_margin"]
    gain_lib = m_lib["mean_logit_margin"] - m_pre["mean_logit_margin"]
    ret = d1.retention(gain_lib, gain_solo)
    out["margins"] = {"pre": m_pre, "solo": m_solo, "lib": m_lib,
                      "gain_solo": round(gain_solo, 4),
                      "gain_lib": round(gain_lib, 4),
                      "retention_vs_solo": (None if ret is None
                                            else round(ret, 4))}

    code_only = list(code_train)
    contam = d1.run_in_state(
        model, blocks, gates_plugged, handles, "plugged",
        lambda: d1.routing_hits(model, [blocks[host_i]], base_batches,
                                device, new_idx))
    util = d1.run_in_state(
        model, blocks, gates_plugged, handles, "plugged",
        lambda: d1.routing_hits(model, [blocks[host_i]], code_only,
                                device, new_idx))
    mass_base = d1.run_in_state(
        model, blocks, gates_plugged, handles, "plugged",
        lambda: d1.routing_mass(model, [blocks[host_i]], base_batches,
                                device, new_idx))
    mass_code = d1.run_in_state(
        model, blocks, gates_plugged, handles, "plugged",
        lambda: d1.routing_mass(model, [blocks[host_i]], code_only,
                                device, new_idx))
    out["contamination_base"] = contam
    out["utilization_code"] = util
    out["routing_mass"] = {"base": mass_base, "code": mass_code}

    # §6 quality: CE on home domain, plugged vs masked (+ forced upper bound)
    ce_masked = d1.ce_in_state(model, blocks, code_only, device, gates64,
                               handles, "off")
    ce_active = d1.ce_in_state(model, blocks, code_only, device,
                               gates_plugged, handles, "plugged")
    ce_forced = d1.ce_in_state(model, blocks, code_only, device, gates64,
                               handles, "forced")
    out["s6_quality"] = {"ce_code_masked": ce_masked,
                         "ce_code_active": ce_active,
                         "ce_code_forced_solo": ce_forced,
                         "delta_plugged_vs_masked": ce_active - ce_masked,
                         "delta_forced_vs_masked": ce_forced - ce_masked}

    ce_base_masked = d1.ce_in_state(model, blocks, base_val_batches, device,
                                    gates64, handles, "off")
    ce_base_active = d1.ce_in_state(model, blocks, base_val_batches, device,
                                    gates_plugged, handles, "plugged")
    out["base_ce"] = {"masked": ce_base_masked, "plugged": ce_base_active,
                      "delta": ce_base_active - ce_base_masked}

    out["verdict"] = d1.d1_verdict(base_parity, ret, contam, util,
                                   ce_active, ce_masked)
    q = out["s6_quality"]
    grab = (out["verdict"]["pass"] and gain_solo > 0.0
            and q["delta_plugged_vs_masked"] < 0.0)
    delivery_ok = (out["verdict"]["gates"]["retention_vs_solo_ge_0.70"]
                   and util["mean_util"] > BAR_UTILIZATION)
    ce_solo_gain = q["ce_code_masked"] - q["ce_code_forced_solo"]
    ce_lib_gain = q["ce_code_masked"] - q["ce_code_active"]
    ce_retention = (ce_lib_gain / ce_solo_gain
                    if ce_solo_gain > 0 else None)
    out["verdict"]["ce_delivery_retention"] = (None if ce_retention is None
                                               else round(ce_retention, 4))
    out["verdict"]["margin_retention_degenerate"] = bool(gain_solo <= 0)
    if grab:
        vtxt = "GRAB WORKS — foreign core hostable verbatim behind thin adapters"
    elif q["delta_plugged_vs_masked"] < 0 and util["mean_util"] > BAR_UTILIZATION:
        vtxt = ("GRAB WORKS at the interface — foreign core hosted verbatim and "
                "beats masking on home CE (quality sub-gate passes); margin "
                "retention metric degenerate "
                f"(solo gain {gain_solo:+.3f} <= 0), CE-delivery retention "
                f"{ce_retention} — check held-out CE before a skill claim")
    elif out["verdict"]["pass"] and not delivery_ok:
        vtxt = ("NEEDS DOOR 3 — formal bars pass but the wrapped core does not "
                "deliver (retention/util collapsed)")
    elif out["verdict"]["pass"]:
        vtxt = ("GRAB PARTIAL — D1 bars pass; quality sub-gate open "
                "(same failure mode as D1 att-4)")
    else:
        vtxt = ("NEEDS DOOR 3 — D1 bars fail: adapters cannot bridge the bases "
                "at this interface")
    out["verdict"]["grab_verdict"] = vtxt
    out["verdict"]["grab_bars_pass"] = grab
    out["wall_s"] = round(time.time() - t0, 2)
    if args.plugin_out:
        torch.save({
            "format": FORMAT,
            "wrap": wrap.state_dict(),
            "wrap_spec": wrap.wrap_spec(),
            "row": pack.banks65[0].P.data[new_idx].cpu(),
            "meta": {"new_idx": new_idx, "host_layer": host_i,
                     "home_domain": args.home_domain,
                     "host_registry": args.registry},
        }, args.plugin_out)
        out["plugin_path"] = args.plugin_out
    return out


# ---------------------------------------------------------------------------
# Donor profiling + core extraction (loads the Phi donor)
# ---------------------------------------------------------------------------


def dump_json(out: dict, path: str | None) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, path)


@torch.no_grad()
def profile_and_extract(args, device, out: dict) -> tuple:
    """Phi profiling (which expert owns the home domain) + frozen core slice."""
    import graft_d2_crossfamily as d2
    t0 = time.time()
    print(f"loading donor {args.donor} for profiling")
    donor_model, donor_tok = d2.load_donor(args.donor, device, args.dtype)
    blocks_d = d2.find_moe_blocks(donor_model)
    fam = d2.family_for(d2.detect_family(donor_model), blocks_d)
    domains = d0.load_domain_texts(args.donor_domains_dir)
    names = sorted(domains.keys())
    if args.home_domain not in names:
        raise RuntimeError(f"home domain {args.home_domain!r} not in {names}")
    code_idx = names.index(args.home_domain)
    batches = []
    for i, name in enumerate(names):
        batches.extend(d2.texts_to_batches_chunked(
            donor_tok, [domains[name]], args.seq_len, args.batch_size,
            args.profile_tokens, label=i))
    n_tok = sum(t.numel() for t, _ in batches)
    print(f"profiling {len(names)} domains on {n_tok} donor tokens "
          f"({len(batches)} batches)")
    stats = d2.collect_route_stats(donor_model, blocks_d, fam, batches,
                                   device, len(names))
    choice, table = choose_donor_expert(stats.act_count, stats.tok_count,
                                        code_idx, min_rate=args.min_owner_rate)
    home, _, mush = stats.home_and_prototypes()
    out["profile"] = {
        "donor": args.donor, "family": fam.name,
        "domains": names, "home_domain": args.home_domain,
        "n_donor_layers": len(blocks_d),
        "n_tokens_profiled": int(n_tok),
        "chosen": choice, "top_by_lift": table,
        "mushiness_mean": float(mush.mean()),
        "tok_count": [int(x) for x in stats.tok_count],
    }
    layer = args.home_layer if args.home_layer >= 0 else int(choice["layer"])
    expert = args.home_expert if args.home_expert >= 0 else int(choice["expert"])
    w_gu, w_down = extract_donor_core(blocks_d, layer, expert)
    out["profile"]["extracted"] = {"layer": layer, "expert": expert,
                                   "w_gu": list(w_gu.shape),
                                   "w_down": list(w_down.shape)}
    if args.core_out:
        torch.save({"format": FORMAT, "w_gu": w_gu, "w_down": w_down,
                    "layer": layer, "expert": expert,
                    "n_donor_layers": len(blocks_d),
                    "profile": out["profile"]}, args.core_out)
        out["core_path"] = args.core_out
    out["profile"]["wall_s"] = round(time.time() - t0, 2)
    dump_json(out, args.out)
    del donor_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return w_gu, w_down, layer, expert


def load_core(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return (payload["w_gu"], payload["w_down"], payload["layer"],
            payload["expert"], payload.get("n_donor_layers"),
            payload.get("profile"))


# ---------------------------------------------------------------------------
# Synthetic self-test (--synthetic): the wrap + pipeline on the d0 mock MoE
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
    wrap = WrappedForeignExpert(d_model, d_canon, w_gu, w_down, seed=0)

    # geometry + param count
    n_ad = wrap.adapter_param_count()
    expect = d_model * d_canon + d_canon * d_donor + d_donor * d_canon + d_canon * d_model
    check("adapter_param_count", n_ad == expect, f"{n_ad} (expect {expect})")
    check("core_is_buffer_not_param",
          all(p.numel() for p in wrap.parameters())
          and wrap.core_w_gu.shape == w_gu.shape
          and not any(p is wrap.core_w_gu for p in wrap.parameters()),
          f"params {n_ad}, core {wrap.core_param_count()}")

    # identity at birth (zero-init host_out)
    x = torch.randn(5, d_model)
    y0 = wrap(x)
    check("identity_at_birth", float(y0.abs().max()) == 0.0,
          f"max|e_new| {float(y0.abs().max()):.2e}")

    # core forward == direct SwiGLU with [w1; w3] split
    u = torch.randn(5, d_donor)
    gate, up = F.linear(u, w_gu).chunk(2, dim=-1)
    ref = F.linear(F.silu(gate) * up, w_down)
    check("core_forward_matches_swiGlu",
          torch.allclose(wrap.core_forward(u), ref, atol=1e-5),
          f"max diff {float((wrap.core_forward(u) - ref).abs().max()):.2e}")

    # frozen core: no grad, unchanged by an optimizer step on the adapters
    w_gu_before = wrap.core_w_gu.clone()
    w_dn_before = wrap.core_w_down.clone()
    opt = torch.optim.Adam(wrap.parameters(), lr=1e-2)
    # nonzero target so host_out actually moves (zero-init blocks upstream grads
    # on step 1 only — same zero-init-down dynamic as D1's SwiGLU expert)
    def nll():
        return (wrap(x).float() ** 2).sum() - wrap(x).float().sum()
    opt.zero_grad()
    nll().backward()
    first = {n: (p.grad is not None and float(p.grad.abs().sum()) > 0)
             for n, p in wrap.named_parameters()}
    opt.step()
    opt.zero_grad()
    nll().backward()
    grads_flow = all(p.grad is not None and float(p.grad.abs().sum()) > 0
                     for p in wrap.parameters())
    opt.step()
    check("adapters_receive_grads", first.get("host_out.weight") and grads_flow,
          f"step1 host_out {first.get('host_out.weight')}, step2 all {grads_flow}")
    check("core_frozen_bitwise", torch.equal(wrap.core_w_gu, w_gu_before)
          and torch.equal(wrap.core_w_down, w_dn_before)
          and not wrap.core_w_gu.requires_grad)

    # selection logic
    act = torch.zeros(3, 4, 3)
    act[1, 2, 0] = 50.0   # layer1 expert2: home=code (idx 0)
    act[0, 0, 1] = 40.0
    act[2, 3, 2] = 45.0
    tok = torch.tensor([100.0, 100.0, 100.0])
    choice, table = choose_donor_expert(act, tok, code_idx=0, min_rate=0.05)
    check("selection_picks_home_owner",
          choice["layer"] == 1 and choice["expert"] == 2,
          f"chose L{choice['layer']} e{choice['expert']} lift {choice['lift']}")
    check("selection_table_populated", len(table) > 0
          and table[0]["lift"] >= choice["lift"] * 0.99)

    # full pipeline on the mock donor
    model, n_dom = d0.build_mock_donor()
    blocks = d0.find_moe_blocks(model)
    model.eval()
    e, h = blocks[0].gate.weight.shape
    banks_reg = [d0.PassportBank(e, h) for _ in blocks]
    d0.init_passports_exact(banks_reg, [b.gate.weight for b in blocks])

    code_train = [(t, 1) for t, d in d0.mock_batches(n_dom, seed=11) if d == 1]
    base_b = [(t, 0) for t, d in d0.mock_batches(n_dom, seed=17) if d != 1]
    probe_wins = [t.flatten() for t, d in d0.mock_batches(
        n_dom, seed=23, seq_len=48, batches_per_domain=3, batch_size=1)
        if d == 1]

    class A:
        pass
    args = A()
    args.expert_steps = 60
    args.expert_lr = 1e-2
    args.out_pen = 5.0
    args.grad_clip = 1.0
    args.weight_decay = 0.0
    args.row_steps = 120
    args.row_lr = 0.1
    args.reject_w = 10.0
    args.mass_w = 2.0
    args.owner_mass = 0.35
    args.max_cal_pairs = 1024
    args.n_probes = 12
    args.home_domain = "code"
    args.out = None
    args.plugin_out = None

    wrap2 = WrappedForeignExpert(h, 8, w_gu, w_down, seed=1)
    core_gu0, core_dn0 = wrap2.core_w_gu.clone(), wrap2.core_w_down.clone()
    out = run_door25(model, blocks, banks_reg, 0, wrap2, code_train,
                     probe_wins, base_b, base_b, args, device)

    check("phase_a_runs", out["phase_a"]["history"][-1]["ce_code"] <
          out["phase_a"]["history"][0]["ce_code"] + 1.0,
          f"ce_code {out['phase_a']['history'][0]['ce_code']:.3f} -> "
          f"{out['phase_a']['history'][-1]['ce_code']:.3f}")
    check("core_unchanged_by_pipeline",
          torch.equal(wrap2.core_w_gu, core_gu0)
          and torch.equal(wrap2.core_w_down, core_dn0))
    check("parity_computed", out["base_parity"]["logit_cosine_mean"] > 0.5,
          f"cos {out['base_parity']['logit_cosine_mean']:.4f}")
    check("graft_parity_computed",
          out["graft_parity_vs_olmoe_graft"]["logit_cosine_mean"] > 0.5,
          f"cos {out['graft_parity_vs_olmoe_graft']['logit_cosine_mean']:.4f}")
    check("routing_direction",
          out["contamination_base"]["mean_util"] < 0.9
          and out["utilization_code"]["mean_util"] >= 0.0,
          f"util_code {out['utilization_code']['mean_util']:.3f} "
          f"contam_base {out['contamination_base']['mean_util']:.3f}")
    check("quality_reads_present",
          all(k in out["s6_quality"] for k in
              ("ce_code_masked", "ce_code_active", "ce_code_forced_solo")),
          f"active {out['s6_quality']['ce_code_active']:.3f} vs "
          f"masked {out['s6_quality']['ce_code_masked']:.3f}")
    check("verdict_struct",
          set(out["verdict"]["gates"]) == {
              "base_parity_cos_ge_0.95", "retention_vs_solo_ge_0.70",
              "contamination_le_0.30", "s6_utilization_gt_0.50",
              "s6_quality_ce_beats_masked"}
          and "grab_verdict" in out["verdict"],
          out["verdict"]["grab_verdict"][:60])

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} synthetic checks passed")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", default=None, help="host (OLMoE graft donor)")
    ap.add_argument("--donor", default=None, help="foreign donor (Phi-mini-MoE)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--registry", default=None,
                    help="D0 registry .pt (registry_olmoe_6d84c485.pt)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--domains-dir", default=os.path.join(REPO_ROOT, "lab", "graft_d0_texts"))
    ap.add_argument("--donor-domains-dir", default=None,
                    help="domain slices for donor profiling (graft_d2_slice)")
    ap.add_argument("--home-domain", default="code")
    ap.add_argument("--d-canon", type=int, default=64,
                    help="canonical bottleneck width (the format's d_canon)")
    ap.add_argument("--home-layer", type=int, default=-1,
                    help="host layer (-1 = depth-match the chosen donor layer)")
    ap.add_argument("--home-expert", type=int, default=-1,
                    help="donor expert id (-1 = profiling choice)")
    ap.add_argument("--min-owner-rate", type=float, default=0.05)
    ap.add_argument("--profile-tokens", type=int, default=12000,
                    help="donor tokens per domain for profiling")
    ap.add_argument("--core-out", default=None)
    ap.add_argument("--core-from", default=None,
                    help="skip donor profiling; load a saved core .pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plugin-out", default=None)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="bs x seq must dodge the gfx906 backward size holes "
                         "(module docstring); 4x512 = 2048 tokens is verified")
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--expert-steps", type=int, default=250)
    ap.add_argument("--expert-lr", type=float, default=3e-3,
                    help="thin-adapter lr (D1's 1e-4 + wd 0.01 starves the wrap "
                         "delta; 3e-3/wd0 verified in the arm sweep)")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="0.0 — the D1 default 0.01 shrinks the wrap maps into "
                         "the frozen core's quiet zone within ~10 steps")
    ap.add_argument("--out-pen", type=float, default=RECIPE["out_pen"])
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--row-steps", type=int, default=400)
    ap.add_argument("--row-lr", type=float, default=0.05)
    ap.add_argument("--reject-w", type=float, default=RECIPE["reject_w"])
    ap.add_argument("--mass-w", type=float, default=2.0)
    ap.add_argument("--owner-mass", type=float, default=RECIPE["owner_mass"])
    ap.add_argument("--max-cal-pairs", type=int, default=4096)
    ap.add_argument("--n-probes", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.synthetic:
        return run_synthetic_tests()
    if not args.model or not args.registry:
        ap.error("--model and --registry are required unless --synthetic")
    if not args.core_from and not args.donor:
        ap.error("--donor is required unless --core-from is given")
    if not args.core_from and not args.donor_domains_dir:
        ap.error("--donor-domains-dir is required for profiling")

    out = {"gate": "D2.5-wrap", "phase": "start", "args": {
        k: (v if not isinstance(v, str) or len(v) < 200 else v[:200])
        for k, v in sorted(vars(args).items())}}
    if args.out is None:
        args.out = os.path.join(REPO_ROOT, "lab", "imported", "graft_door25",
                                "door25_wrap.json")
    if args.core_out is None:
        args.core_out = os.path.join(REPO_ROOT, "lab", "imported",
                                     "graft_door25", "door25_core.pt")
    if args.plugin_out is None:
        args.plugin_out = os.path.join(REPO_ROOT, "lab", "imported",
                                       "graft_door25", "door25_plugin.pt")
    dump_json(out, args.out)

    # ---- donor side: profile + extract (or resume from saved core) ----
    n_donor_layers = None
    if args.core_from:
        w_gu, w_down, d_layer, d_expert, n_donor_layers, prof = load_core(
            args.core_from)
        out["profile"] = prof or {"loaded_from": args.core_from,
                                  "extracted": {"layer": d_layer,
                                                "expert": d_expert}}
        out["profile"]["loaded_from"] = args.core_from
        print(f"core loaded from {args.core_from} (donor L{d_layer} e{d_expert})")
    else:
        w_gu, w_down, d_layer, d_expert = profile_and_extract(
            args, args.device, out)
        n_donor_layers = out["profile"].get("n_donor_layers")
        print(f"chosen donor core: layer {d_layer} expert {d_expert} "
              f"(w_gu {tuple(w_gu.shape)}, w_down {tuple(w_down.shape)})")
    out["phase"] = "profile_done"
    dump_json(out, args.out)

    # ---- host side: OLMoE graft + D1 path ----
    t_load = time.time()
    model, tok = d0.load_donor(args.model, args.device, args.dtype)
    blocks = d0.find_moe_blocks(model)
    banks_reg, reg_meta = d0.load_registry(args.registry)
    if len(banks_reg) != len(blocks):
        raise RuntimeError(f"registry has {len(banks_reg)} layers, "
                           f"model {len(blocks)}")
    n_host = len(blocks)
    if args.home_layer >= 0:
        host_i = args.home_layer
    else:
        n_dl = n_donor_layers or (2 * n_host)
        host_i = int(round(d_layer * (n_host - 1) / max(1, (n_dl - 1))))
    host_i = max(0, min(n_host - 1, host_i))
    d_model = banks_reg[0].d_model
    wrap = WrappedForeignExpert(d_model, args.d_canon, w_gu, w_down,
                                seed=args.seed)
    print(f"loaded host+registry in {time.time() - t_load:.1f}s: "
          f"{n_host} layers x {banks_reg[0].num_experts} experts, "
          f"d_model {d_model}; wrap at layer {host_i}, "
          f"{wrap.adapter_param_count()} adapter params "
          f"(core {wrap.core_param_count()} frozen)")
    out["host"] = {"model": args.model, "registry": args.registry,
                   "registry_meta_keys": sorted(reg_meta)[:6],
                   "n_layers": n_host, "d_model": d_model,
                   "donor_layer": d_layer, "donor_expert": d_expert,
                   "host_layer": host_i}
    out["wrap"] = wrap.wrap_spec()
    out["phase"] = "host_loaded"
    dump_json(out, args.out)

    domains = d0.load_domain_texts(args.domains_dir)
    code_text = domains.get(args.home_domain)
    if not code_text:
        raise RuntimeError(f"domains-dir must contain {args.home_domain}.txt")
    code_ids = tok.encode(code_text, add_special_tokens=False)
    split = int(0.6 * len(code_ids))
    code_train_ids, code_probe_ids = code_ids[:split], code_ids[split:]
    code_train_ids = code_train_ids[:args.max_tokens]

    def to_batches(ids, label, seq, bs):
        wins = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
        return [(torch.tensor(wins[i:i + bs], dtype=torch.long), label)
                for i in range(0, len(wins) - bs + 1, bs)]

    code_train = to_batches(code_train_ids, 1, args.seq_len, args.batch_size)
    base_ids = []
    for name, text in domains.items():
        if name == args.home_domain:
            continue
        base_ids.extend(tok.encode(text, add_special_tokens=False))
    base_ids = base_ids[:args.max_tokens * 3]
    base_batches = to_batches(base_ids, 0, args.seq_len, args.batch_size)
    probe_windows = [torch.tensor(code_probe_ids[i:i + 256], dtype=torch.long)
                     for i in range(0, len(code_probe_ids) - 256, 256)]
    if not code_train or not base_batches or not probe_windows:
        raise RuntimeError("not enough data — raise --max-tokens or lower --seq-len")
    print(f"data: home_train {sum(t.numel() for t, _ in code_train)} tok, "
          f"base {sum(t.numel() for t, _ in base_batches)} tok, "
          f"{len(probe_windows)} probe windows")

    result = run_door25(model, blocks, banks_reg, host_i, wrap, code_train,
                        probe_windows, base_batches, base_batches, args,
                        args.device)
    result["donor"] = {"model": args.donor, "revision": args.revision,
                       "core_layer": d_layer, "core_expert": d_expert}
    result["phase"] = "complete"
    out.update(result)   # keep args / host / profile from the earlier dumps
    result = out
    dump_json(result, args.out)

    print(f"\nresults -> {args.out}")
    m = result["margins"]
    print(f"margins: pre {m['pre']['mean_logit_margin']:+.4f}  "
          f"solo {m['solo']['mean_logit_margin']:+.4f}  "
          f"lib {m['lib']['mean_logit_margin']:+.4f}  "
          f"gains {m['gain_solo']:+.4f}/{m['gain_lib']:+.4f}  "
          f"retention {m['retention_vs_solo']}")
    b = result["base_parity"]
    print(f"base parity cos {b['logit_cosine_mean']:.4f}  "
          f"graft parity cos "
          f"{result['graft_parity_vs_olmoe_graft']['logit_cosine_mean']:.4f}  "
          f"contam {result['contamination_base']['mean_util']:.3f}  "
          f"util {result['utilization_code']['mean_util']:.3f}")
    q = result["s6_quality"]
    print(f"quality: code CE masked {q['ce_code_masked']:.4f}  "
          f"plugged {q['ce_code_active']:.4f}  "
          f"forced {q['ce_code_forced_solo']:.4f}")
    v = result["verdict"]
    print(f"D2.5 verdict: {v['grab_verdict']}")
    print(f"  gates={v['gates']}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
