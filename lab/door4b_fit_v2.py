#!/usr/bin/env python3
"""Door 4b — expert-fit recipe v2 under contract (b): the grab-story closer.

Combines the two winners of the door4 head-to-head:
  * contract (b) additive fixed-weight delivery —
    `y = y_stock(64-way full scale) + λ·m·e_new(h)`, λ learned per layer,
    m = plug-in top-8 membership (stock merge bit-identical to masked).
  * the Door-2.5 fit recipe (`lab/door25_wrap_expert.py` findings) —
    TASK-LOSS CE at **lr 3e-3, weight_decay 0.0** (D1's wd 0.01 starves deltas
    below fp16 trunk resolution), contrastive neutrality at independent
    weights (CE=1.0 on home batches, `--neutral-w` on base-h outputs only),
    grad-clip 1.0, 250 steps. Parameter budget: 16 × SwiGLU deltas ≈ 100.8M
    trainable (≥ the 786k Door-2.5 budget) + 16 learned λ scalars.
Plus the in-cycle protocol fixes: λ/m cast to the activation dtype, quality
measured on HELD-OUT home tokens (probe half), post-training on-trajectory
row recal, and base parity reported BOTH overall and split by whether the
plug-in fires on the token at any layer (the 366c94d parity cost was
concentrated on firing tokens).

Targets (all green together = plug-in quality story CLOSED):
  quality held-out delivered < masked | parity ≥0.95 (overall) |
  retention ≥0.70 | contamination ≤0.30 | utilization >0.50.

Arms:
  --expert-init fresh   D1-style from-scratch code-domain deltas (comparable
                        to 366c94d's runs)
  --expert-init door3   warm-start from the Door-3 distilled artifact
                        (7e0a442: /data/tliao/graft_d3_out/plugin_distill_olmoe.pt)
                        then fit-v2 refit — tests generalization across
                        expert sources.

Fork note: imports graft_d0_parity / graft_d1_plugin / door4_delivery_contract
UNMODIFIED (sibling lanes import the first two).

Usage:
  python3 lab/door4b_fit_v2.py --synthetic
  python3 lab/door4b_fit_v2.py --model /data/tliao/OLMoE-1B-7B-0924 \
      --registry /data/tliao/graft_d0_out/registry_olmoe_6d84c485.pt \
      --domains-dir /data/tliao/graft_d0_slice --device cuda --dtype float16 \
      --expert-init fresh --out /data/tliao/graft_d0_out/door4b_fresh.json
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
import door4_delivery_contract as d4  # noqa: E402

BAR_PARITY = 0.95
BAR_RETENTION = 0.70
BAR_CONTAM = 0.30
BAR_UTIL = 0.50


class AdapterDelta(nn.Module):
    """LoRA-style low-rank residual delta — the Door-2.5 parameter class.

    e_new(h) = W_up(W_down(h)), rank r, W_up zero-init (e_new = 0 at birth).
    2·r·d_model params/layer (r=64, d=2048 → 262k/layer, 4.2M total) — matches
    the budget class that trained stably at lr 3e-3 / wd 0 in Door-2.5, unlike
    the 100M full-width SwiGLU deltas which NaN within 25 steps at that LR.
    """

    def __init__(self, d_model: int, rank: int = 64, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.down = nn.Parameter(torch.randn(rank, d_model, generator=g) * 0.02)
        self.up = nn.Parameter(torch.zeros(d_model, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = F.linear(F.linear(xf, self.down.to(xf.device)),
                     self.up.to(xf.device))
        return y.to(x.dtype)


# ---------------------------------------------------------------------------
# Artifact loading (Door-3 distilled plugin: vesper-graft-plugin-distill-v1)
# ---------------------------------------------------------------------------


def load_plugin_artifact(path: str, pack) -> dict:
    """Load expert state_dicts + passport rows into the pack (warm start)."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    n_e = len(d["experts"])
    if n_e != len(pack.experts):
        raise RuntimeError(f"artifact has {n_e} experts, pack {len(pack.experts)}")
    for sd, exp in zip(d["experts"], pack.experts):
        exp.load_state_dict(sd)
    new_idx = pack.banks65[0].num_experts - 1
    for bank, row in zip(pack.banks65, d["rows"]):
        bank.P.data[new_idx] = row.float().to(bank.P.device)
    return {"format": d.get("format"), "meta": {k: v for k, v in d.get("meta", {}).items()
                                               if isinstance(v, (str, int, float, bool))
                                               or v is None}}


# ---------------------------------------------------------------------------
# Fit v2 trainer (Door-2.5 recipe, contract-(b) delivered composition)
# ---------------------------------------------------------------------------


def train_expert_fit_v2(model, blocks, pack, handles, code_batches,
                        base_batches, steps, lr, device, mode,
                        extra_params=None, neutral_w: float = 300.0,
                        grad_clip: float = 1.0) -> dict:
    """Task-loss fit at the DELIVERED composition — Door-2.5 recipe.

    loss = CE(code batches, merged output as delivered)   [help weight 1.0]
         + neutral_w * mean_l ||e_new(h_base)||^2          [independent
           contrastive-neutrality weight, base-h only]
    Adam(lr, weight_decay=0.0) — wd must stay 0 (D1's 0.01 starves deltas to
    |y|~5e-4, below fp16 trunk resolution). Trunk frozen; grads reach e_new
    (and λ for additive) only. Every forward runs in `mode` ("additive"),
    so the delta optimizes the composition it will actually be delivered at.
    """
    params = list(pack.parameters()) + list(extra_params or [])
    d1.freeze_all_except(model, params)
    opt = torch.optim.Adam(params, lr=lr, weight_decay=0.0)
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
        total = ce + neutral_w * out_norm2
        if not torch.isfinite(total):
            hist.append({"step": step, "ce_code": None, "out_norm2": None,
                         "aborted": True})
            print(f"  [fitv2 {step:4d}] NON-FINITE loss — aborting fit")
            return {"steps": step, "lr": lr, "weight_decay": 0.0,
                    "neutral_w": neutral_w, "grad_clip": grad_clip,
                    "mode": mode, "history": hist, "aborted_nonfinite": True,
                    "wall_s": round(time.time() - t0, 2)}
        opt.zero_grad()
        total.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            lam_s = ([round(float(p.detach().clamp(min=0)), 4)
                      for p in extra_params[:2]] if extra_params else None)
            hist.append({"step": step, "ce_code": float(ce.detach()),
                         "out_norm2": float(out_norm2.detach()),
                         "lam_head": lam_s})
            print(f"  [fitv2 {step:4d}] ce_code {float(ce.detach()):.4f}  "
                  f"out_norm2 {float(out_norm2.detach()):.6f}")
    return {"steps": steps, "lr": lr, "weight_decay": 0.0,
            "neutral_w": neutral_w, "grad_clip": grad_clip,
            "mode": mode, "history": hist,
            "wall_s": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------
# Parity with firing/non-firing split
# ---------------------------------------------------------------------------


@torch.no_grad()
def parity_vs_state_split(model, blocks, batches, device, gates, handles,
                          mode, new_idx, additive: bool) -> dict:
    """Overall base parity + cos split by plug-in firing (any layer).

    Non-firing tokens take a bit-identical path to the masked control under
    contract (b); their cos is the structural floor. Firing tokens carry the
    plug-in's delta and show the real perturbation.
    """
    tot = {"cos": 0.0, "cos2": 0.0, "n": 0, "cos_min": 1.0}
    fire = {"cos": 0.0, "n": 0}
    non = {"cos": 0.0, "n": 0}
    ce_s = ce_p = n_row = 0.0
    for tokens, _ in batches:
        tokens = tokens.to(device)
        bsz = tokens.shape[0]
        lg_s = d1.run_in_state(model, blocks, None, handles, "off",
                               lambda: d0.model_logits(model, tokens))
        holder = {}

        def delivered():
            lg = d0.model_logits(model, tokens)
            f = None
            for blk in blocks:
                m = getattr(blk.gate, "last_membership", None)
                if m is not None:
                    f = (m.detach().cpu().bool() if f is None
                         else f | m.detach().cpu().bool())
            holder["fired"] = f  # capture INSIDE the pass (gates still live)
            return lg

        lg_p = d1.run_in_state(model, blocks, gates, handles, mode, delivered)
        cos = d0.token_logit_cosine(lg_s, lg_p)  # (B*T,)
        if additive and holder.get("fired") is not None:
            fired = holder["fired"]
        else:
            fired = torch.ones(cos.shape[0], dtype=torch.bool)  # n/a
        c = cos.detach().float().cpu()
        f = fired.numpy()
        cn = c.numpy()
        tot["cos"] += float(c.sum())
        tot["cos2"] += float((c * c).sum())
        tot["cos_min"] = min(tot["cos_min"], float(c.min()))
        fire["cos"] += float(c[f].sum())
        fire["n"] += int(f.sum())
        non["cos"] += float(c[~f].sum())
        non["n"] += int((~f).sum())
        tot["n"] += int(cos.shape[0])
        ce_s += d0.next_token_ce(lg_s, tokens) * bsz
        ce_p += d0.next_token_ce(lg_p, tokens) * bsz
        n_row += bsz
    mean = tot["cos"] / max(1, tot["n"])
    var = max(0.0, tot["cos2"] / max(1, tot["n"]) - mean * mean)
    return {
        "n_tokens": tot["n"], "logit_cosine_mean": mean,
        "logit_cosine_std": math.sqrt(var), "logit_cosine_min": tot["cos_min"],
        "firing": {"n": fire["n"], "mean_cos": (fire["cos"] / max(1, fire["n"]))},
        "non_firing": {"n": non["n"], "mean_cos": (non["cos"] / max(1, non["n"]))},
        "ce_stock": ce_s / max(1.0, n_row),
        "ce_delivered": ce_p / max(1.0, n_row),
        "ce_delta": (ce_p - ce_s) / max(1.0, n_row),
    }


@torch.no_grad()
def membership_hits_any(model, blocks, batches, device):
    """Fraction of tokens firing the plug-in at ANY layer (drives parity)."""
    n_fire = 0
    n_tok = 0
    for tokens, _ in batches:
        d0.model_logits(model, tokens.to(device))
        f = None
        for blk in blocks:
            m = getattr(blk.gate, "last_membership", None)
            if m is not None:
                f = (m.detach().cpu().bool() if f is None
                     else f | m.detach().cpu().bool())
        if f is not None:
            n_fire += int(f.sum())
        n_tok += tokens.numel()
    return {"any_layer_rate": round(n_fire / max(1, n_tok), 4),
            "n": n_tok}


def measure_fit_v2(tag, model, blocks, pack, handles, gates, mode, probes,
                   code_eval, code_train, base_batches, device, new_idx):
    """All five targets for the additive contract."""
    out = {"arm": tag, "mode": mode}
    fn_pre = d1.make_logits_fn(model, blocks, d4.gates_pre_global, handles,
                               "off", device)
    fn_solo = d1.make_logits_fn(model, blocks, d4.gates_pre_global, handles,
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
    out["base_parity"] = parity_vs_state_split(
        model, blocks, base_batches, device, gates, handles, mode,
        new_idx, additive=True)

    def hits_fn():
        return d4.membership_hits(model, blocks, base_batches, device)

    def hits_code():
        return d4.membership_hits(model, blocks, code_eval, device)

    out["contamination_base"] = d1.run_in_state(
        model, blocks, gates, handles, mode, hits_fn)
    out["utilization_code"] = d1.run_in_state(
        model, blocks, gates, handles, mode, hits_code)
    out["firing_rate_base_any_layer"] = d1.run_in_state(
        model, blocks, gates, handles, mode,
        lambda: membership_hits_any(model, blocks, base_batches, device))
    out["firing_rate_code_any_layer"] = d1.run_in_state(
        model, blocks, gates, handles, mode,
        lambda: membership_hits_any(model, blocks, code_eval, device))
    ce_masked = d1.ce_in_state(model, blocks, code_eval, device,
                               d4.gates_pre_global, handles, "off")
    ce_active = d1.ce_in_state(model, blocks, code_eval, device, gates,
                               handles, mode)
    ce_active_train = d1.ce_in_state(model, blocks, code_train, device, gates,
                                     handles, mode)
    ce_masked_train = d1.ce_in_state(model, blocks, code_train, device,
                                     d4.gates_pre_global, handles, "off")
    out["s6_quality"] = {"ce_code_masked": ce_masked,
                         "ce_code_active": ce_active,
                         "delta": ce_active - ce_masked,
                         "train_slice_active": ce_active_train,
                         "train_slice_masked": ce_masked_train}
    v = d4.door4_verdict(out["base_parity"],
                         out["margins"]["retention_vs_solo"],
                         out["contamination_base"],
                         out["utilization_code"], ce_active, ce_masked)
    v["gates"]["quality_ratio"] = round(ce_active / max(1e-9, ce_masked), 4)
    out["verdict"] = v
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_fit_v2(model, blocks, banks_registry, code_train, code_eval,
               probe_windows, base_batches, args, device) -> dict:
    out = {"gate": "door4b-fit-v2", "expert_init": args.expert_init,
           "recipe": {"expert_lr": args.expert_lr,
                      "expert_steps": args.expert_steps,
                      "weight_decay": 0.0,
                      "neutral_w": args.neutral_w,
                      "grad_clip": args.grad_clip,
                      "lam_init": args.lam_init,
                      "delivery": "additive_lambda",
                      "rows": {"reject_w": args.reject_w,
                               "mass_w": args.mass_w,
                               "owner_mass": args.owner_mass}},
           "started": time.time()}
    t0 = time.time()
    top_k = getattr(blocks[0].gate, "top_k", 2)
    norm_topk = bool(getattr(blocks[0].gate, "norm_topk_prob", False))
    pack = d1.build_pack(blocks, banks_registry, (top_k, norm_topk),
                         inter_mult=args.inter_mult, seed=args.seed)
    if args.delta == "adapter":
        d_model = blocks[0].gate.weight.shape[1]
        pack.experts = nn.ModuleList(
            [AdapterDelta(d_model, rank=args.rank, seed=args.seed + l)
             for l in range(len(blocks))])
    pack.experts.to(device)
    n_params = sum(p.numel() for p in pack.parameters())
    out["trainable_params"] = n_params
    out["delta_class"] = args.delta
    new_idx = pack.banks65[0].num_experts - 1
    gates64 = [d0.GraftGate(b, top_k, norm_topk, stock_gate=blk.gate)
               for b, blk in zip(banks_registry, blocks)]
    d4.gates_pre_global = gates64

    # ---- rows round-1 (+ expert warm-start if door3) ----
    print(f"SHARED — rows round-1 (init={args.expert_init}) + probes")
    if args.expert_init == "door3":
        out["artifact"] = load_plugin_artifact(args.plugin_artifact, pack)
    handles_off = d1.install_plugin(blocks, pack, "off")

    def label_of(_i, b):
        return b[1]

    pairs = d1.run_in_state(model, blocks, gates64, handles_off, "off",
                            lambda: d1.collect_gate_pairs(
                                model, blocks, code_train + base_batches,
                                device, args.max_cal_pairs, label_of))
    if args.expert_init == "fresh":
        with torch.no_grad():
            for li, bank in enumerate(pack.banks65):
                h, lab = pairs[li]
                code_h = h[lab > 0.5]
                if code_h.shape[0] > 0:
                    bank.P.data[new_idx] = (code_h.mean(dim=0)
                                            * math.sqrt(bank.d_r))
    pb1 = d1.train_rows(pack, pairs, args.row_steps, args.row_lr, device,
                        reject_w=args.reject_w, mass_w=args.mass_w,
                        owner_target=args.owner_mass)
    out["rows_round1"] = pb1
    fn_pre = d1.make_logits_fn(model, blocks, gates64, handles_off, "off",
                               device)
    probes = d1.select_code_probes(fn_pre, probe_windows, args.n_probes)
    out["n_probes"] = len(probes)
    d1.uninstall_plugin(handles_off)
    if args.out:
        d4.save_json(args.out, out)

    # ---- fit v2 (additive contract) ----
    print("FIT V2 — additive delivery, task-loss CE @lr 3e-3 wd 0")
    lams = nn.ParameterList([nn.Parameter(torch.tensor(args.lam_init))
                             for _ in blocks])
    gates_add = d4.make_additive_gates(blocks, pack)
    hB = d4.install_additive(blocks, pack, lams, mode="off")
    restored = [(blk, d0.install_graft(blk, g))
                for blk, g in zip(blocks, gates_add)]
    fit = train_expert_fit_v2(model, blocks, pack, hB, code_train,
                              base_batches, args.expert_steps,
                              args.expert_lr, device, mode="additive",
                              extra_params=list(lams),
                              neutral_w=args.neutral_w,
                              grad_clip=args.grad_clip)
    # ---- rows round-2 on-trajectory (with trained delta + λ live) ----
    print("ROWS round-2 — on-trajectory recal")
    p2 = d1.run_in_state(model, blocks, gates_add, hB, "additive",
                         lambda: d1.collect_gate_pairs(
                             model, blocks, code_train + base_batches,
                             device, args.max_cal_pairs, label_of))
    pb2 = d1.train_rows(pack, p2, max(50, args.row_steps // 2), args.row_lr,
                        device, reject_w=args.reject_w, mass_w=args.mass_w,
                        owner_target=args.owner_mass)
    lams_final = [float(p.detach().clamp(min=0)) for p in lams]
    for blk, stock in restored:
        d0.uninstall_graft(blk, stock)
    # incremental save BEFORE measurement — fit history must survive a
    # measurement-phase crash
    out["fit"] = fit
    out["rows_round2"] = pb2
    out["lambda_per_layer"] = lams_final
    out["lambda_mean"] = round(sum(lams_final) / max(1, len(lams_final)), 4)
    if args.out:
        d4.save_json(args.out, out)

    # ---- measure ----
    print("MEASURE — five targets + firing-split parity")
    m = measure_fit_v2(args.expert_init, model, blocks, pack, hB, gates_add,
                       "additive", probes, code_eval, code_train,
                       base_batches, device, new_idx)
    m["training"] = fit
    m["rows_round2"] = pb2
    m["lambda_per_layer"] = lams_final
    m["lambda_mean"] = round(sum(lams_final) / max(1, len(lams_final)), 4)
    out["result"] = m
    d1.uninstall_plugin(hB)
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
    args.expert_init = "fresh"
    args.delta = "adapter"
    args.rank = 8
    args.plugin_artifact = None
    args.inter_mult = 1.0
    args.seed = 0
    args.expert_steps = 30
    args.expert_lr = 1e-2
    args.neutral_w = 5.0
    args.grad_clip = 1.0
    args.lam_init = 0.5
    args.row_steps = 80
    args.row_lr = 0.1
    args.reject_w = 5.0
    args.mass_w = 2.0
    args.owner_mass = 0.35
    args.max_cal_pairs = 2048
    args.n_probes = 8
    args.out = None

    out = run_fit_v2(model, blocks, banks_reg, code_train, code_train,
                     probe_wins, base_b, args, device)
    r = out["result"]
    check("verdict_struct",
          {"parity_ge_0.95", "retention_ge_0.70", "contamination_le_0.30",
           "quality_ce_beats_masked", "utilization_gt_0.50"}
          <= set(r["verdict"]["gates"]),
          f"pass={r['verdict']['pass']}")
    check("firing_split_present",
          "firing" in r["base_parity"] and "non_firing" in r["base_parity"],
          f"non-firing cos {r['base_parity']['non_firing']['mean_cos']:.4f} "
          f"(n={r['base_parity']['non_firing']['n']})")
    check("lam_recorded", out["result"]["lambda_mean"] > 0)
    check("trainable_budget",
          out["trainable_params"] >= 1000,
          f"{out['trainable_params']} params ({out.get('delta_class')}; "
          f"real OLMoE adapter r=64 = 4.2M, swiglu = 100.8M)")
    check("fit_runs", len(r["training"]["history"]) > 0)

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
    ap.add_argument("--out", default=None)
    ap.add_argument("--expert-init", default="fresh",
                    choices=["fresh", "door3"])
    ap.add_argument("--plugin-artifact", default=None,
                    help="Door-3 plugin_distill_olmoe.pt (for --expert-init door3)")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--expert-steps", type=int, default=250)
    ap.add_argument("--expert-lr", type=float, default=3e-3,
                    help="Door-2.5 recipe: 3e-3 (D1's 3e-4 underpowers the fit)")
    ap.add_argument("--neutral-w", type=float, default=300.0,
                    help="contrastive neutrality weight on base-h outputs")
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
    ap.add_argument("--delta", default="adapter",
                    choices=["adapter", "swiglu"],
                    help="adapter = LoRA-style (Door-2.5 class, lr 3e-3 safe); "
                         "swiglu = full-width (100M, NaNs at 3e-3)")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.synthetic:
        return run_synthetic_tests()
    if not args.model or not args.registry:
        ap.error("--model and --registry are required unless --synthetic")
    if args.expert_init == "door3" and not args.plugin_artifact:
        ap.error("--plugin-artifact is required with --expert-init door3")

    torch.manual_seed(args.seed)
    t_load = time.time()
    model, tok = d0.load_donor(args.model, args.device, args.dtype)
    blocks = d0.find_moe_blocks(model)
    banks_reg, _ = d0.load_registry(args.registry)
    if len(banks_reg) != len(blocks):
        raise RuntimeError("registry/model layer mismatch")
    print(f"loaded donor+registry in {time.time() - t_load:.1f}s "
          f"({len(blocks)} layers)")

    domains = d0.load_domain_texts(args.domains_dir)
    code_ids = tok.encode(domains["code"], add_special_tokens=False)
    split = int(0.6 * len(code_ids))
    train_ids, probe_ids = code_ids[:split], code_ids[split:]
    train_ids = train_ids[:args.max_tokens]

    def to_batches(ids, label, seq, bs):
        wins = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
        return [(torch.tensor(wins[i:i + bs], dtype=torch.long), label)
                for i in range(0, len(wins) - bs + 1, bs)]

    code_train = to_batches(train_ids, 1, args.seq_len, args.batch_size)
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
          f"base {sum(t.numel() for t, _ in base_batches)} tok")

    if args.out is None:
        args.out = os.path.join(REPO_ROOT, "lab", "imported", "graft_d0",
                                f"door4b_{args.expert_init}.json")

    result = run_fit_v2(model, blocks, banks_reg, code_train, code_eval,
                        probe_windows, base_batches, args, args.device)
    result["donor"] = {"model": args.model, "revision": args.revision}
    d4.save_json(args.out, result)
    print(f"\nresults -> {args.out}")
    m = result["result"]
    q = m["s6_quality"]
    p = m["base_parity"]
    print(f"quality held-out: {q['ce_code_active']:.4f} vs masked "
          f"{q['ce_code_masked']:.4f}  (train {q['train_slice_active']:.4f} vs "
          f"{q['train_slice_masked']:.4f})")
    print(f"parity overall {p['logit_cosine_mean']:.4f}  "
          f"non-firing {p['non_firing']['mean_cos']:.4f} (n={p['non_firing']['n']})  "
          f"firing {p['firing']['mean_cos']:.4f} (n={p['firing']['n']})")
    print(f"retention {m['margins']['retention_vs_solo']}  "
          f"contam {m['contamination_base']['mean_util']:.3f}  "
          f"util {m['utilization_code']['mean_util']:.3f}  "
          f"lam {m['lambda_mean']}")
    v = m["verdict"]
    print(f"VERDICT: {'PASS' if v['pass'] else 'FAIL'}  gates={v['gates']}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
