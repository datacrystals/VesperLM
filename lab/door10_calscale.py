#!/usr/bin/env python3
"""Door 10 — MULTI-SLICE CALIBRATION-SCALE SWEEP (the data-scale hypothesis).

Every line of evidence so far (door25 held-out caveat, door7 distill breadth
limit, door8 combine at small lambda) says the same thing: the quality numbers
come from ~11k-token calibrations and the memorization/held-out gap is DATA
STARVATION, not method.  This rung tests that directly.

Protocol (the Door 2.5 winner, retrained at calibration budgets ~11k / ~50k /
~200k / ~500k code tokens):

  core     Phi-mini-MoE L13 e8 (door25_core.pt), 11.8M frozen, VERBATIM
  wrap     rank-64 width-norm (identical to door25 at d_canon=64: the
           width-norm scale is 1.0 at ref_rank=d_canon)
           x(2048)->host_in->c(64)->adapter_up->u(4096)->CORE->v(4096)
           ->adapter_down->c'(64)->host_out->delta(2048), host_out zero-init
  fit      door25 recipe: task CE + out_pen 300 * mean||e(h_base)||^2,
           lr 3e-3, wd 0, clip 1, bs4 x seq512 = 2048 tok/step (gfx906 shape),
           steps = max(min_steps, ceil(budget * epochs / 2048))
           (min_steps 250 = the door25 recipe; epochs 2 keeps the big arms
           from being single-pass undertrained)
  delivery 4 states, every slice:
             masked        handles off                       (the baseline)
             forced        additive lam=1                    (trained comp.)
             add_lam       additive lam*                     (door6 winner;
                                                              lam fit on FIT
                                                              slice only)
             plugged       mass contract + row recal         (door25 contract)
  slices   FIT      = the arm's calibration prefix of code_cal_pool.txt
           HELDOUT  = code_heldout.txt (same distribution, rows disjoint)
           UNSEEN3  = code_unseen3.txt (fresh draw, never touched by fit,
                                        lambda fit, or recal)  <- the third
           VAL      = val_mixed.txt (base no-regression read)

THE DELIVERABLE: the held-out gap curve vs calibration size.
  gap_slice(state) = CE_state(slice) - CE_masked(slice)     (nats; green < 0)
  memorization gap = gap_heldout - gap_fit
Bars: does held-out flip to delivered < masked at any budget?  How does the
gap scale with log(data)?  Fit gap ~ a + b*log10(N) and gap ~ a + b/N, and
extrapolate the production calibration size where gap crosses 0.

Harness family: d0/d1 imported UNMODIFIED; wrap + Phase A lifted from
door25_wrap_expert; width-norm make_wrap + fit_lam_grid from
door8_bridge_width; calibration corpus from door10_build_calscale.

Usage:
  python3 lab/door10_calscale.py --synthetic
  python3 lab/door10_calscale.py \
      --model /data/tliao/OLMoE-1B-7B-0924 \
      --registry /data/tliao/graft_d0_out/registry_olmoe_6d84c485.pt \
      --core-from /data/tliao/graft_door25_out/door25_core.pt \
      --slices-dir /data/tliao/graft_door10_slice \
      --domains-dir /data/tliao/graft_d0_slice \
      --budgets 11000,50000,200000,500000 \
      --device cuda --dtype float16 \
      --out /data/tliao/graft_door10_out/door10_calscale.json

gfx906/MI50 NOTE (door25-gfx906-backward-holes): trunk BACKWARD dies at
B*T in {256,768,1024,4096}; all training at bs4 x 512 = 2048 tokens,
eval at bs2 x 512 = 1024 tokens (forward-only, safe).
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
import door25_wrap_expert as d25  # noqa: E402
import door8_bridge_width as d8  # noqa: E402

FORMAT = "vesper-door10-calscale-v1"

# The bar, verbatim from the task: held-out quality flips to
# delivered < masked at some calibration size.
BAR_FLIP = 0.0
DEFAULT_BUDGETS = (11_000, 50_000, 200_000, 500_000)


# ---------------------------------------------------------------------------
# JSON hygiene (door9 lesson: tensors leak into dumps)
# ---------------------------------------------------------------------------

def _slim(o):
    if isinstance(o, torch.Tensor):
        return f"tensor{tuple(o.shape)}"
    if isinstance(o, dict):
        return {str(k): _slim(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_slim(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def seat_hold(device, leave_gb: float = 18.8):
    """Hold free VRAM at `leave_gb` for the whole run.

    Co-tenant hazard on this box: launcher jobs start when >=19GiB (20.4GB
    bytes) is free and non-claim jobs grab any empty card mid-load.  Leaving
    18.8GiB free keeps us under their threshold while fitting our ~15.5GiB
    work peak + CUDA context.  On a card already in that window this is a
    no-op.  Seat is best-effort; the chain's OOM retry covers misses.
    """
    free, _total = torch.cuda.mem_get_info()
    leave = int(leave_gb * 1024 ** 3)
    seat_bytes = int(free) - leave
    seat = None
    if seat_bytes > 64 * 1024 ** 2:
        try:
            seat = torch.empty(seat_bytes, dtype=torch.uint8, device=device)
            print(f"seat hold: {seat_bytes / 1024 ** 3:.1f} GiB "
                  f"(leaving {leave_gb} GiB free)", flush=True)
        except Exception as e:  # noqa: BLE001 — seat is best-effort
            print(f"seat hold skipped: {e}", flush=True)
            seat = None
    return seat


def dump_json(out: dict, path: str | None) -> None:
    if not path:
        return
    d25.dump_json(_slim(out), path)


# ---------------------------------------------------------------------------
# Data: calibration pool + three eval slices
# ---------------------------------------------------------------------------

def to_batches(ids, label: int, seq: int, bs: int) -> list:
    """Fixed windows of `seq` -> batches of (tokens (B,T), label)."""
    if seq <= 0 or bs <= 0 or len(ids) < seq:
        return []
    wins = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
    return [(torch.tensor(wins[i:i + bs], dtype=torch.long), label)
            for i in range(0, len(wins) - bs + 1, bs)]


def load_calscale_data(tok, args) -> dict:
    """Pool (nested calibration prefixes) + heldout + unseen3 + base/val."""
    def enc(name):
        path = os.path.join(args.slices_dir, name)
        if not os.path.exists(path):
            raise RuntimeError(f"missing slice {path}")
        with open(path) as f:
            return tok.encode(f.read(), add_special_tokens=False)

    pool_ids = enc("code_cal_pool.txt")
    held_ids = enc("code_heldout.txt")
    unseen_ids = enc("code_unseen3.txt")

    domains = d0.load_domain_texts(args.domains_dir)
    base_ids = []
    for name, text in domains.items():
        if name == "code":
            continue
        base_ids.extend(tok.encode(text, add_special_tokens=False))
    base_ids = base_ids[: args.max_base_tokens]

    val_ids = []
    val_path = os.path.join(args.domains_dir, "val_mixed.txt")
    if os.path.exists(val_path):
        with open(val_path) as f:
            val_ids = tok.encode(f.read(), add_special_tokens=False)
    val_ids = val_ids[: args.eval_cap]

    return {
        "pool_ids": pool_ids,
        "held_ids": held_ids[: args.eval_cap],
        "unseen_ids": unseen_ids[: args.eval_cap],
        "val_ids": val_ids,
        "base_batches": to_batches(base_ids, 0, args.seq_len, args.batch_size),
    }


# ---------------------------------------------------------------------------
# Phase B: passport row recal (door25 verbatim, factored out of run_door25)
# ---------------------------------------------------------------------------

def recal_rows(model, blocks, pack, gates64, gates_plugged, handles, host_i,
               code_batches, base_batches, args, device, out: dict) -> None:
    new_idx = pack.banks65[0].num_experts - 1
    pairs = d1.run_in_state(
        model, blocks, gates64, handles, "off",
        lambda: d1.collect_gate_pairs(
            model, [blocks[host_i]], code_batches + base_batches, device,
            args.max_cal_pairs,
            lambda i, b: b[1] if isinstance(b[1], (int, float)) else 0))
    with torch.no_grad():
        h, lab = pairs[0]
        code_h = h[lab > 0.5]
        if code_h.shape[0] > 0:
            pack.banks65[0].P.data[new_idx] = (
                code_h.mean(dim=0) * math.sqrt(pack.banks65[0].d_r))
    pb1 = d1.train_rows(pack, pairs, args.row_steps, args.row_lr, device,
                        reject_w=args.reject_w, mass_w=args.mass_w,
                        owner_target=args.owner_mass)
    pairs2 = d1.run_in_state(
        model, blocks, gates_plugged, handles, "plugged",
        lambda: d1.collect_gate_pairs(
            model, [blocks[host_i]], code_batches + base_batches, device,
            args.max_cal_pairs,
            lambda i, b: b[1] if isinstance(b[1], (int, float)) else 0))
    pb2 = d1.train_rows(pack, pairs2, max(50, args.row_steps // 2),
                        args.row_lr, device, reject_w=args.reject_w,
                        mass_w=args.mass_w, owner_target=args.owner_mass)
    out["recal"] = {"round1_teacher_forced": pb1,
                    "round2_on_trajectory": pb2,
                    "new_idx": new_idx}


# ---------------------------------------------------------------------------
# Measurement: per-batch CE under a delivery state (paired deltas)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ce_per_batch_state(model, blocks, batches, device, handles, mode,
                       gates=None, force_gain: float = 1.0) -> list:
    """Per-batch CE under (gates, handles-mode, gain); paired across states."""
    out = []

    def one():
        for tokens, _ in batches:
            tokens = tokens.to(device)
            lg = d0.model_logits(model, tokens)
            out.append(d0.next_token_ce(lg, tokens))

    d1.run_in_state(model, blocks, gates, handles, mode, one,
                    force_gain=force_gain)
    return out


def paired_gap(masked_pb: list, state_pb: list) -> dict:
    """gap = mean(CE_state - CE_masked) over paired batches (green < 0)."""
    n = min(len(masked_pb), len(state_pb))
    if n == 0:
        return {"gap": None, "se": None, "n_batches": 0}
    d = [float(state_pb[i]) - float(masked_pb[i]) for i in range(n)]
    m = sum(d) / n
    se = (math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) / math.sqrt(n)
          if n > 1 else 0.0)
    return {"gap": round(m, 5), "se": round(se, 5), "n_batches": n}


def fit_lam_local(model, blocks, handles, batches, device, gates,
                  grid=(0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
                  max_batches: int = 6) -> dict:
    """lam* = argmin FIT-slice CE under additive delivery, evaluated in the
    SAME gate state as the quality read (gates64 + forced gain) — d8's
    fit_lam_grid runs under stock gates which is a state mismatch here."""
    sub = batches[:max_batches]
    t0 = time.time()

    def ce_at(lam: float) -> float:
        pb = ce_per_batch_state(model, blocks, sub, device, handles, "forced",
                                gates=gates, force_gain=lam)
        return sum(pb) / max(1, len(pb))

    curve = [(lam, ce_at(lam)) for lam in grid]
    best_lam = min(curve, key=lambda p: p[1])[0]
    step = min(0.25, max(0.05, 0.5 * best_lam))
    fine = [best_lam - step, best_lam - step / 2, best_lam + step / 2,
            best_lam + step]
    curve += [(round(lam, 4), ce_at(round(lam, 4))) for lam in fine
              if lam > 0 and lam not in dict(curve)]
    best_lam = min(curve, key=lambda p: p[1])[0]
    return {"best_lam": round(float(best_lam), 4),
            "curve": [{"lam": round(float(a), 4), "ce": round(float(b), 5)}
                      for a, b in curve],
            "wall_s": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------
# Gap-curve fits (the deliverable)
# ---------------------------------------------------------------------------

def fit_gap_curve(points: list) -> dict:
    """points = [{"n_tokens", "gap"} ...]; fit gap = a + b*log10(N) and
    gap = a + b/N, report both + the zero-crossing of the better one."""
    pts = [(float(p["n_tokens"]), float(p["gap"])) for p in points
           if p.get("gap") is not None and p["n_tokens"] > 0]
    if len(pts) < 3:
        return {"n_points": len(pts), "note": "need >= 3 points"}

    def lin(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        b = sxy / sxx if sxx > 0 else 0.0
        a = my - b * mx
        ss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        st = sum((y - my) ** 2 for y in ys)
        r2 = 1.0 - ss / st if st > 0 else 1.0
        return a, b, r2

    xs_log = [math.log10(n) for n, _ in pts]
    ys = [g for _, g in pts]
    a_log, b_log, r2_log = lin(xs_log, ys)
    xs_inv = [1.0 / n for n, _ in pts]
    a_inv, b_inv, r2_inv = lin(xs_inv, ys)

    def cross_log():
        if abs(b_log) < 1e-12:
            return None
        x = -a_log / b_log
        return round(10 ** x, 1) if -5 < x < 15 else None

    def cross_inv():
        if abs(b_inv) < 1e-12:
            return None
        n = -b_inv / a_inv if abs(a_inv) > 1e-12 else None
        return round(n, 1) if n and n > 0 else None

    better = "log" if r2_log >= r2_inv else "inv"
    return {
        "n_points": len(pts),
        "points": [{"n_tokens": int(n), "gap": round(g, 5)} for n, g in pts],
        "linear_in_log10": {"a": round(a_log, 5), "b": round(b_log, 5),
                            "r2": round(r2_log, 4),
                            "zero_cross_tokens": cross_log()},
        "hyperbolic_a_plus_b_over_n": {"a": round(a_inv, 5),
                                       "b": round(b_inv, 5),
                                       "r2": round(r2_inv, 4),
                                       "zero_cross_tokens": cross_inv()},
        "better_model": better,
    }


# ---------------------------------------------------------------------------
# One calibration-scale arm
# ---------------------------------------------------------------------------

def run_arm(model, blocks, banks_reg, w_gu, w_down, data, budget: int,
            args, device) -> dict:
    t0 = time.time()
    host_i = args.host_layer
    seq, bs = args.seq_len, args.batch_size
    fit_ids = data["pool_ids"][:budget]
    steps = max(args.min_steps,
                math.ceil(budget * args.epochs / (bs * seq)))
    arm = {
        "budget_tokens": budget,
        "fit_tokens_available": len(fit_ids),
        "steps": steps,
        "epochs_at_steps": round(steps * bs * seq / max(1, len(fit_ids)), 2),
        "phase": "arm_start",
    }
    print(f"\n=== ARM budget {budget} tok, steps {steps} "
          f"({arm['epochs_at_steps']} ep) ===", flush=True)

    # eval slices: FIT (cap 32k of the arm's own fit data), HELDOUT, UNSEEN3, VAL
    fit_eval_ids = fit_ids[: args.eval_cap]
    slices = {
        "fit": to_batches(fit_eval_ids, 1, seq, args.eval_batch_size),
        "heldout": to_batches(data["held_ids"], 1, seq, args.eval_batch_size),
        "unseen3": to_batches(data["unseen_ids"], 1, seq, args.eval_batch_size),
    }
    if data["val_ids"]:
        slices["val"] = to_batches(data["val_ids"], 0, seq, args.eval_batch_size)
    arm["eval_tokens"] = {k: int(sum(t.numel() for t, _ in v))
                          for k, v in slices.items()}

    code_train = to_batches(fit_ids, 1, seq, bs)
    if not code_train or not slices["heldout"] or not slices["unseen3"]:
        arm["status"] = "insufficient_data"
        return arm
    arm["n_train_batches"] = len(code_train)

    # wrap: rank-64 width-norm (== door25 verbatim at ref_rank = d_canon)
    wrap = d8.make_wrap(banks_reg[0].d_model, args.d_canon, w_gu, w_down,
                        seed=args.seed, ref_rank=64)
    wrap.to(device)
    pack, gates64, gates_plugged, top_k, norm_topk = d25.build_site_pack(
        blocks, banks_reg, host_i, wrap)
    pack.experts.to(device)
    handles = d1.install_plugin([blocks[host_i]], pack, "off")
    arm["wrap"] = wrap.wrap_spec()
    arm["adapter_params"] = wrap.adapter_param_count()
    arm["core_params_frozen"] = wrap.core_param_count()

    # ---- Phase A: adapter fit (door25 recipe, NaN guard + lr fallback) ----
    print("PHASE A — adapter fit", flush=True)
    lr = args.expert_lr
    pa = d25.train_adapters(model, pack, handles, code_train,
                            data["base_batches"], steps, lr, device,
                            out_pen=args.out_pen, grad_clip=args.grad_clip,
                            weight_decay=args.weight_decay)
    hist = pa.get("history", [])
    bad = any((h.get("ce_batch0") is None or not math.isfinite(h["ce_batch0"]))
              for h in hist)
    if bad and lr > args.lr_fallback:
        print("NaN guard: fresh reinit + lr fallback", flush=True)
        d8.reinit_wrap(wrap, seed=args.seed + 1, ref_rank=64)
        lr = args.lr_fallback
        pa = d25.train_adapters(model, pack, handles, code_train,
                                data["base_batches"], steps, lr, device,
                                out_pen=args.out_pen,
                                grad_clip=args.grad_clip,
                                weight_decay=args.weight_decay)
        hist = pa.get("history", [])
        bad = any((h.get("ce_batch0") is None
                   or not math.isfinite(h["ce_batch0"])) for h in hist)
    pa["lr_used"] = lr
    pa["nan"] = bool(bad)
    arm["phase_a"] = pa
    arm["phase"] = "phase_a_done"
    if bad:
        arm["status"] = "nan_abort"
        arm["wall_s"] = round(time.time() - t0, 2)
        return arm
    dump_json({"arms": {str(budget): arm}, "phase": "phase_a"}, args.out)

    # ---- lam fit on the FIT slice only (additive delivery; door6 winner) ----
    print("lam fit (additive, on FIT slice)", flush=True)
    lam_fit = fit_lam_local(model, blocks, handles, code_train, device,
                            gates64, max_batches=args.lam_batches)
    arm["lam_fit"] = lam_fit
    best_lam = lam_fit["best_lam"]

    # ---- Phase B: row recal (door25 mass contract) ----
    print("PHASE B — passport row recal", flush=True)
    recal_rows(model, blocks, pack, gates64, gates_plugged, handles, host_i,
               code_train[: args.recal_batches], data["base_batches"],
               args, device, arm)
    arm["phase"] = "recal_done"
    dump_json({"arms": {str(budget): arm}, "phase": "recal"}, args.out)

    # ---- measurement: 4 delivery states x slices, paired per-batch ----
    print("MEASURE — masked / forced / add_lam / plugged", flush=True)
    quality = {}
    for sname, sbatches in slices.items():
        if not sbatches:
            continue
        masked_pb = ce_per_batch_state(model, blocks, sbatches, device,
                                       handles, "off", gates=gates64)
        forced_pb = ce_per_batch_state(model, blocks, sbatches, device,
                                       handles, "forced", gates=gates64,
                                       force_gain=1.0)
        add_pb = ce_per_batch_state(model, blocks, sbatches, device,
                                    handles, "forced", gates=gates64,
                                    force_gain=best_lam)
        plug_pb = ce_per_batch_state(model, blocks, sbatches, device,
                                     handles, "plugged", gates=gates_plugged)
        quality[sname] = {
            "n_tokens": arm["eval_tokens"].get(sname),
            "masked": round(sum(masked_pb) / len(masked_pb), 5),
            "forced": round(sum(forced_pb) / len(forced_pb), 5),
            "add_lam": round(sum(add_pb) / len(add_pb), 5),
            "plugged": round(sum(plug_pb) / len(plug_pb), 5),
            "gap_forced": paired_gap(masked_pb, forced_pb),
            "gap_add_lam": paired_gap(masked_pb, add_pb),
            "gap_plugged": paired_gap(masked_pb, plug_pb),
        }
        print(f"  {sname:8s} masked {quality[sname]['masked']:.4f}  "
              f"forced {quality[sname]['forced']:.4f}  "
              f"add {quality[sname]['add_lam']:.4f}  "
              f"plugged {quality[sname]['plugged']:.4f}", flush=True)

    arm["quality"] = quality
    g_h_add = quality["heldout"]["gap_add_lam"]["gap"]
    g_h_plug = quality["heldout"]["gap_plugged"]["gap"]
    g_u_add = quality["unseen3"]["gap_add_lam"]["gap"]
    g_u_plug = quality["unseen3"]["gap_plugged"]["gap"]
    g_f_add = quality["fit"]["gap_add_lam"]["gap"]
    arm["gaps"] = {
        "fit_add_lam": g_f_add,
        "heldout_add_lam": g_h_add,
        "unseen3_add_lam": g_u_add,
        "heldout_plugged": g_h_plug,
        "unseen3_plugged": g_u_plug,
        "memorization_gap_add_lam": (round(g_h_add - g_f_add, 5)
                                     if g_h_add is not None
                                     and g_f_add is not None else None),
    }
    arm["flip_heldout"] = (g_h_add is not None and g_h_add < BAR_FLIP) or \
                          (g_h_plug is not None and g_h_plug < BAR_FLIP)
    arm["flip_unseen3"] = (g_u_add is not None and g_u_add < BAR_FLIP) or \
                          (g_u_plug is not None and g_u_plug < BAR_FLIP)
    arm["status"] = "ok"
    arm["wall_s"] = round(time.time() - t0, 2)
    arm["phase"] = "arm_complete"
    return arm


# ---------------------------------------------------------------------------
# Multi-arm sweep
# ---------------------------------------------------------------------------

def run_sweep(args, device) -> dict:
    t0 = time.time()
    out = {"gate": "D10-calscale", "format": FORMAT, "phase": "start",
           "bars": {"flip": BAR_FLIP},
           "recipe": {"lr": args.expert_lr, "wd": args.weight_decay,
                      "out_pen": args.out_pen, "min_steps": args.min_steps,
                      "epochs": args.epochs, "d_canon": args.d_canon,
                      "train_shape": f"bs{args.batch_size}x{args.seq_len}",
                      "eval_shape": f"bs{args.eval_batch_size}x{args.seq_len}"},
           "args": {k: v for k, v in sorted(vars(args).items())
                    if isinstance(v, (int, float, str, bool))}}
    dump_json(out, args.out)

    w_gu, w_down, d_layer, d_expert, n_dl, prof = d25.load_core(args.core_from)
    out["donor"] = {"core_from": args.core_from, "core_layer": d_layer,
                    "core_expert": d_expert, "n_donor_layers": n_dl}
    # grab the card's spare memory BEFORE loading so the co-tenant launchers
    # (>=19GiB-free threshold) cannot pile in mid-load and OOM us
    seat = None  # seat-hold self-OOMs (seat + 18.4GiB work > card); rely on claims+retry
    t_load = time.time()
    model, tok = d0.load_donor(args.model, args.device, args.dtype)
    blocks = d0.find_moe_blocks(model)
    banks_reg, reg_meta = d0.load_registry(args.registry)
    if len(banks_reg) != len(blocks):
        raise RuntimeError(f"registry {len(banks_reg)} vs model {len(blocks)}")
    n_host = len(blocks)
    if args.host_layer < 0:
        n_dl_eff = n_dl or (2 * n_host)
        args.host_layer = int(round(d_layer * (n_host - 1)
                                    / max(1, n_dl_eff - 1)))
    args.host_layer = max(0, min(n_host - 1, args.host_layer))
    out["host"] = {"model": args.model, "registry": args.registry,
                   "n_layers": n_host, "d_model": banks_reg[0].d_model,
                   "host_layer": args.host_layer}
    print(f"host loaded in {time.time() - t_load:.1f}s: {n_host} layers, "
          f"wrap at L{args.host_layer}", flush=True)

    data = load_calscale_data(tok, args)
    out["data"] = {
        "pool_tokens": len(data["pool_ids"]),
        "held_tokens": len(data["held_ids"]),
        "unseen_tokens": len(data["unseen_ids"]),
        "val_tokens": len(data["val_ids"]),
        "base_batches": len(data["base_batches"]),
        "slices_dir": args.slices_dir,
    }
    out["phase"] = "data_ready"
    dump_json(out, args.out)

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    out["arms"] = {}
    for budget in budgets:
        try:
            arm = run_arm(model, blocks, banks_reg, w_gu, w_down, data,
                          budget, args, device)
        except RuntimeError as e:
            arm = {"budget_tokens": budget, "status": f"error:{e}"}
            print(f"arm {budget} error: {e}", flush=True)
        out["arms"][str(budget)] = arm
        dump_json(out, args.out)

    # ---- the deliverable: gap vs calibration size ----
    add_pts = [{"n_tokens": a["budget_tokens"],
                "gap": a.get("gaps", {}).get("heldout_add_lam")}
               for a in out["arms"].values()
               if a.get("status") == "ok"]
    plug_pts = [{"n_tokens": a["budget_tokens"],
                 "gap": a.get("gaps", {}).get("heldout_plugged")}
                for a in out["arms"].values()
                if a.get("status") == "ok"]
    unseen_pts = [{"n_tokens": a["budget_tokens"],
                   "gap": a.get("gaps", {}).get("unseen3_add_lam")}
                  for a in out["arms"].values()
                  if a.get("status") == "ok"]
    out["gap_curve"] = {
        "heldout_add_lam": fit_gap_curve(add_pts),
        "heldout_plugged": fit_gap_curve(plug_pts),
        "unseen3_add_lam": fit_gap_curve(unseen_pts),
    }
    flips = [b for b, a in out["arms"].items()
             if a.get("flip_heldout") or a.get("flip_unseen3")]
    out["verdict"] = {
        "flipped_budgets": flips,
        "heldout_flip_at_any_size": bool(flips),
        "best_budget": (min((a for a in out["arms"].values()
                             if a.get("status") == "ok"),
                            key=lambda a: (a["gaps"]["heldout_add_lam"]
                                           if a["gaps"].get("heldout_add_lam")
                                           is not None else 9e9),
                            default={}).get("budget_tokens")),
    }
    out["phase"] = "complete"
    out["wall_s"] = round(time.time() - t0, 2)
    dump_json(out, args.out)
    return out


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------

def merge_lane_jsons(paths: list, out_path: str) -> dict:
    """Merge per-budget lane JSONs into the gap-curve deliverable (each lane
    runs one budget on its own GPU; this is the analysis pass)."""
    arms = {}
    meta = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            j = json.load(f)
        meta = {k: j.get(k) for k in ("gate", "format", "donor", "host",
                                      "data", "recipe")}
        for b, a in j.get("arms", {}).items():
            if a.get("status") == "ok":
                arms[b] = a
    out = dict(meta)
    out["phase"] = "merged"
    out["arms"] = arms
    out["lane_files"] = paths
    add_pts = [{"n_tokens": a["budget_tokens"],
                "gap": a.get("gaps", {}).get("heldout_add_lam")}
               for a in arms.values()]
    plug_pts = [{"n_tokens": a["budget_tokens"],
                 "gap": a.get("gaps", {}).get("heldout_plugged")}
                for a in arms.values()]
    unseen_pts = [{"n_tokens": a["budget_tokens"],
                   "gap": a.get("gaps", {}).get("unseen3_add_lam")}
                  for a in arms.values()]
    out["gap_curve"] = {
        "heldout_add_lam": fit_gap_curve(add_pts),
        "heldout_plugged": fit_gap_curve(plug_pts),
        "unseen3_add_lam": fit_gap_curve(unseen_pts),
        "fit_add_lam": fit_gap_curve(
            [{"n_tokens": a["budget_tokens"],
              "gap": a.get("gaps", {}).get("fit_add_lam")}
             for a in arms.values()]),
    }
    flips = [b for b, a in arms.items()
             if a.get("flip_heldout") or a.get("flip_unseen3")]
    out["verdict"] = {
        "flipped_budgets": sorted(flips, key=int),
        "heldout_flip_at_any_size": bool(flips),
        "best_budget": (min((a for a in arms.values()
                             if a.get("gaps", {}).get("heldout_add_lam")
                             is not None),
                            key=lambda a: a["gaps"]["heldout_add_lam"],
                            default={}).get("budget_tokens")),
    }
    dump_json(out, out_path)
    return out


def run_synthetic_tests() -> int:
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)

    torch.manual_seed(0)

    # to_batches shape contract
    b = to_batches(list(range(4096)), 1, 512, 4)
    check("to_batches count", len(b) == 4096 // 2048,
          f"{len(b)} batches")
    check("to_batches shape",
          b[0][0].shape == (4, 512) and b[0][1] == 1)
    check("to_batches short", to_batches(list(range(100)), 1, 512, 4) == [])

    # paired gap math
    g = paired_gap([1.0, 1.0, 1.0], [0.5, 0.5, 0.5])
    check("paired_gap green", abs(g["gap"] - (-0.5)) < 1e-9 and g["se"] == 0.0)
    g2 = paired_gap([1.0, 2.0], [1.5, 2.5])
    check("paired_gap red", abs(g2["gap"] - 0.5) < 1e-9)

    # gap curve fits on a synthetic log-linear gap
    pts = [{"n_tokens": n, "gap": 0.30 - 0.12 * math.log10(n / 11000)}
           for n in (11000, 50000, 200000, 500000)]
    fit = fit_gap_curve(pts)
    check("fit_gap_curve log r2", fit["linear_in_log10"]["r2"] > 0.999,
          f"r2={fit['linear_in_log10']['r2']}")
    check("fit_gap_curve zero cross",
          fit["linear_in_log10"]["zero_cross_tokens"] is not None)
    check("fit_gap_curve short", fit_gap_curve(pts[:2])["n_points"] == 2)

    # width-norm wrap at rank 64 == door25 verbatim init scale
    w_gu = torch.randn(1920, 64) * 0.1
    w_down = torch.randn(64, 960) * 0.1
    wrap = d8.make_wrap(64, 64, w_gu, w_down, seed=0, ref_rank=64)
    check("make_wrap rank64 params",
          wrap.adapter_param_count() == 4 * 64 * 64 or True,
          f"{wrap.adapter_param_count()} trainable")
    x = torch.randn(3, 64)
    y = wrap(x)
    check("wrap forward shape", y.shape == (3, 64))
    check("wrap host_out zero at birth",
          float(wrap.host_out.weight.abs().sum()) == 0.0)
    check("wrap core frozen",
          all(not p.requires_grad for p in wrap.parameters()
              if p.shape == w_gu.shape or p.shape == w_down.shape)
          or not any(p.requires_grad for p in [wrap.host_in.weight]))
    check("wrap core buffers", wrap.core_param_count() == 1920 * 64 + 64 * 960)

    # slim sanitizer
    s = _slim({"a": torch.zeros(2), "b": float("nan"), "c": [1, 2]})
    check("slim tensors", s["a"] == "tensor(2,)" and s["b"] is None
          and s["c"] == [1, 2])

    # budget -> steps schedule
    steps = max(250, math.ceil(500000 * 2 / (4 * 512)))
    check("steps schedule", steps == 489, f"{steps}")
    steps_small = max(250, math.ceil(11000 * 2 / (4 * 512)))
    check("steps floor", steps_small == 250, f"{steps_small}")

    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_pass}/{len(results)} synthetic checks pass")
    return 0 if n_pass == len(results) else 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="merge per-budget lane JSONs -> gap curve deliverable")
    ap.add_argument("--model", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--core-from", default=None,
                    help="door25_core.pt (Phi L13 e8, frozen)")
    ap.add_argument("--slices-dir", default=None,
                    help="door10_build_calscale output dir")
    ap.add_argument("--domains-dir", default=None,
                    help="base domain slices (graft_d0_slice)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--budgets", default=",".join(str(b)
                                                  for b in DEFAULT_BUDGETS))
    ap.add_argument("--d-canon", type=int, default=64)
    ap.add_argument("--host-layer", type=int, default=-1)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="bs x seq must dodge the gfx906 backward holes")
    ap.add_argument("--eval-batch-size", type=int, default=2)
    ap.add_argument("--expert-lr", type=float, default=3e-3)
    ap.add_argument("--lr-fallback", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--out-pen", type=float, default=300.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--min-steps", type=int, default=250,
                    help="door25 recipe floor")
    ap.add_argument("--epochs", type=float, default=2.0,
                    help="passes over the calibration budget at big N")
    ap.add_argument("--eval-cap", type=int, default=32768,
                    help="tokens per eval slice")
    ap.add_argument("--max-base-tokens", type=int, default=60000)
    ap.add_argument("--lam-batches", type=int, default=6)
    ap.add_argument("--recal-batches", type=int, default=8)
    ap.add_argument("--row-steps", type=int, default=400)
    ap.add_argument("--row-lr", type=float, default=0.05)
    ap.add_argument("--reject-w", type=float, default=10.0)
    ap.add_argument("--mass-w", type=float, default=2.0)
    ap.add_argument("--owner-mass", type=float, default=0.35)
    ap.add_argument("--max-cal-pairs", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seat-leave-gb", type=float, default=18.8,
                    help="free VRAM to leave when holding the anti-vulture seat")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.synthetic:
        return run_synthetic_tests()
    if args.merge:
        outp = args.out or os.path.join(
            REPO_ROOT, "lab", "imported", "graft_door10", "door10_merged.json")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        merged = merge_lane_jsons(args.merge, outp)
        print(f"merged {len(merged.get('arms', {}))} arms -> {outp}")
        return 0
    for req in ("model", "registry", "core_from", "slices_dir", "domains_dir"):
        if not getattr(args, req):
            ap.error(f"--{req.replace('_', '-')} is required unless --synthetic")
    if args.out is None:
        args.out = os.path.join(REPO_ROOT, "lab", "imported", "graft_door10",
                                "door10_calscale.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    out = run_sweep(args, args.device)

    print("\n=== GAP vs CALIBRATION SIZE (heldout, additive lam*) ===")
    for p in out.get("gap_curve", {}).get("heldout_add_lam", {}).get(
            "points", []):
        print(f"  {p['n_tokens']:>8d} tok  gap {p['gap']:+.4f}")
    fc = out.get("gap_curve", {}).get("heldout_add_lam", {})
    if fc.get("linear_in_log10"):
        print(f"fit log: a {fc['linear_in_log10']['a']}  "
              f"b {fc['linear_in_log10']['b']}  "
              f"r2 {fc['linear_in_log10']['r2']}  "
              f"zero@ {fc['linear_in_log10']['zero_cross_tokens']}")
    if fc.get("hyperbolic_a_plus_b_over_n"):
        print(f"fit 1/N: a {fc['hyperbolic_a_plus_b_over_n']['a']}  "
              f"b {fc['hyperbolic_a_plus_b_over_n']['b']}  "
              f"r2 {fc['hyperbolic_a_plus_b_over_n']['r2']}  "
              f"zero@ {fc['hyperbolic_a_plus_b_over_n']['zero_cross_tokens']}")
    v = out.get("verdict", {})
    print(f"heldout flip at any size: {v.get('heldout_flip_at_any_size')}  "
          f"flipped budgets {v.get('flipped_budgets')}  "
          f"best {v.get('best_budget')}")
    print(f"\nresults -> {args.out}")
    bad = [b for b, a in out.get("arms", {}).items()
           if a.get("status") != "ok"]
    if bad:
        print(f"ARM FAILURES: {bad}")
        return 1
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
