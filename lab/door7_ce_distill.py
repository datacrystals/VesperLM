#!/usr/bin/env python3
"""Door 7 — CE-OBJECTIVE distillation (the named lever from the Door-5 verdict).

Q (HANDOFF Door 5 / MODULAR_MOE 10.3): KL-distilled frozen cores fail the
CE-quality bar because KL-to-donor skill does not transfer to next-token CE
(door5 (ii)/(i'): forced 2.33 / 1.90 vs masked 1.83). Does a CE-OBJECTIVE
distillation — next-token CE toward the teacher's argmax, ground-truth CE
where the teacher is right — produce a frozen core that CARRIES CE skill,
like the donor-pretrained core (door25: forced 0.101 vs masked 1.072)?

Objective (per aligned home position; full student vocab):
  target = ground-truth id   if the teacher has no cached row, OR
                             the teacher's argmax (mapped through the shared
                             support) equals the ground-truth token
           teacher argmax id  otherwise (teacher hard target)
  loss   = CE(student logits, target) + out_pen * mean_l ||e_new(h_base)||^2
NOT KL over the teacher's distribution — that is the door3 baseline this
fork replaces. Same home data + teacher_cache infrastructure as door3
(`vesper-door3-teacher-cache-v1`), same artifact format
(`vesper-graft-plugin-distill-v1`, 16-layer SwiGLU 2048/2048/1024, cores
frozen after training).

Budget: the KL arm's 200 steps / lr 1e-4 / out_pen 300 / wd 0.01 / bs 2x512
(the door3 d3_launch contract — this graph is proven at B*T=1024), plus an
optional continued `--extra-steps` doubled-budget arm (same optimizer state).

Measurement = the winning recipe stack:
  primary   single-site Door-2.5 wrap (the door5 (i') geometry: host L6,
            frozen CE core + thin adapters, lr 3e-3 wd 0 out_pen 300, 250
            steps) delivered by Door 6's additive lambda*m per site
            (y = stock + lam*m*e_new), measured with the i' protocol
            (full-slice train + held-out, bs4x512=2048 / eval-batch 2).
            A mass-contract read of the SAME fitted state is measured too,
            for the door5 (i') / door25 table symmetry.
  secondary distill-then-deliver NO refit (door5 (ii) verbatim: cores
            bitwise-frozen, additive lam/layer fit + row recal).

Bars: quality delivered < masked (THE gate, full-slice train AND held-out),
forced CE < masked (the skill-in-core test), parity >= 0.95, contam <= 0.30.
References on the door3 split (masked 1.8295 held / 1.9235 train):
  KL-distilled  (i')  1.882/1.997  forced 1.897   (door5)
  KL-distilled  (ii)  1.890/1.944  forced 2.327   (door5)
  donor-pretrained     0.439/1.072 forced 0.101   (door25, code slice)

Usage:
  python3 lab/door7_ce_distill.py --synthetic
  python3 lab/door7_ce_distill.py --phase distill \
      --host-model /data/tliao/OLMoE-1B-7B-0924 \
      --registry /data/tliao/graft_d0_out/registry_olmoe_6d84c485.pt \
      --cache /data/tliao/graft_d3_out/teacher_cache.pt \
      --domains-dir /data/tliao/graft_d0_slice \
      --plugin-out /data/tliao/graft_d7_out/plugin_ce_step200.pt \
      --out /data/tliao/graft_d7_out/door7_distill.json
  python3 lab/door7_ce_distill.py --phase wrap --artifact .../plugin_ce_step200.pt
  python3 lab/door7_ce_distill.py --phase deliver --artifact .../plugin_ce_step200.pt
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

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import graft_d0_parity as d0  # noqa: E402
import graft_d1_plugin as d1  # noqa: E402
import door4_delivery_contract as d4  # noqa: E402
import door25_wrap_expert as d25  # noqa: E402
import door5_provenance as d5  # noqa: E402
import door3_distill_expert as d3  # noqa: E402

FORMAT = "vesper-door7-ce-distill-v1"
PLUGIN_FORMAT = "vesper-graft-plugin-distill-v1"
BAR_PARITY = 0.95
BAR_CONTAM = 0.30
RECIPE = d5.RECIPE


# ---------------------------------------------------------------------------
# CE hard targets (teacher argmax / ground truth where the teacher is right)
# ---------------------------------------------------------------------------


def ce_hard_targets(batch, shared_s_ids: torch.Tensor,
                    mode: str = "argmax") -> dict:
    """Full-vocab next-token targets for one DistillBatch (CPU long).

    mode "argmax" (the task's primary formulation): gt where the teacher row
    is missing or the teacher's shared-support argmax equals the ground-truth
    token id; the argmax id (mapped to the student vocab by `shared_s_ids`)
    otherwise. mode "gt": pure ground-truth CE on the home text (the task's
    "CE on ground-truth text" alternative — measured after the argmax arm
    showed the teacher's argmax is non-GT on ~84% of home positions).
    Returns {"targets": (B, L-1), "n_total", "n_agree", "n_hard", "n_unaligned"}.
    """
    gt = batch.ids[:, 1:].long()                       # (B, L-1)
    hard_s = batch.tgt.argmax(dim=-1).long()           # (B, L-1) S index
    hard_id = shared_s_ids.to(hard_s.device)[hard_s]   # (B, L-1) student id
    mask = batch.mask.bool()
    agree = mask & (hard_id == gt)
    hard = mask & ~agree
    if mode == "gt":
        targets = gt
    else:
        targets = torch.where(hard, hard_id, gt)
    return {
        "targets": targets,
        "n_total": int(gt.numel()),
        "n_agree": int(agree.sum()),
        "n_hard": int(hard.sum()),
        "n_unaligned": int((~mask).sum()),
    }


def batch_ce_stats(batches, shared_s_ids) -> dict:
    """Aggregate the hard-target rule over a batch list (diagnostics)."""
    agg = {"n_total": 0, "n_agree": 0, "n_hard": 0, "n_unaligned": 0}
    for b in batches:
        s = ce_hard_targets(b, shared_s_ids)
        for k in agg:
            agg[k] += s[k]
    n = max(1, agg["n_total"])
    agg["frac_agree"] = round(agg["n_agree"] / n, 4)
    agg["frac_hard"] = round(agg["n_hard"] / n, 4)
    agg["frac_unaligned"] = round(agg["n_unaligned"] / n, 4)
    return agg


# ---------------------------------------------------------------------------
# Phase A — CE-objective expert fit (forced dispatch), NaN-guarded
# ---------------------------------------------------------------------------


def train_expert_ce(model, blocks, pack, handles, home_batches, base_batches,
                    steps, lr, device, shared_s_ids: torch.Tensor,
                    out_pen: float = 300.0, grad_clip: float = 1.0,
                    weight_decay: float = 0.01, start_step: int = 0,
                    opt=None, opt_ref: list | None = None,
                    target_mode: str = "argmax") -> dict:
    """Forced-dispatch fit with the CE objective (NOT KL).

    loss = CE(full-vocab, ce_hard_targets) + out_pen * mean_l ||e_new(h_base)||^2
    Trunk frozen; gradients reach only the fresh expert modules. Aborts on a
    non-finite loss and reports the abort step. Pass an existing `opt` to
    continue a fit with carried Adam state (the doubled-budget arm).
    """
    params = list(pack.parameters())
    d1.freeze_all_except(model, params)
    if opt is None:
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt_ref is not None:
        opt_ref[0] = opt
    hist = []
    t0 = time.time()
    for step in range(steps):
        batch = home_batches[(start_step + step) % len(home_batches)]
        ids = batch.ids.to(device)
        tgt = ce_hard_targets(batch, shared_s_ids, target_mode
                              )["targets"].to(device)
        d1.set_plugin_mode(handles, "forced", 1.0)
        logits = d0.model_logits(model, ids)
        v = logits.shape[-1]
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, v).float(),
                             tgt.reshape(-1))
        bt, _ = base_batches[(start_step + step) % len(base_batches)]
        d1.set_plugin_mode(handles, "forced", 1.0)
        d0.model_logits(model, bt.to(device))
        out_terms = [blk._plugin_merge.last_new_out for blk, _ in handles
                     if getattr(blk, "_plugin_merge", None) is not None
                     and blk._plugin_merge.last_new_out is not None]
        out_norm2 = (torch.stack([t.float().pow(2).mean()
                                  for t in out_terms]).mean()
                     if out_terms else torch.zeros((), device=device))
        total = ce + out_pen * out_norm2
        if not torch.isfinite(total):
            hist.append({"step": start_step + step, "ce": None,
                         "out_norm2": None, "aborted": True})
            print(f"  [ceA {start_step + step:4d}] NON-FINITE loss — aborting",
                  flush=True)
            return {"steps": start_step + step, "lr": lr,
                    "out_pen": out_pen, "grad_clip": grad_clip,
                    "weight_decay": weight_decay, "history": hist,
                    "aborted_nonfinite": True,
                    "wall_s": round(time.time() - t0, 2)}
        opt.zero_grad()
        total.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            hist.append({"step": start_step + step,
                         "ce": float(ce.detach()),
                         "out_norm2": float(out_norm2.detach())})
            print(f"  [ceA {start_step + step:4d}] ce {float(ce.detach()):.4f}"
                  f"  out2 {float(out_norm2.detach()):.3e}", flush=True)
    return {"steps": start_step + steps, "lr": lr, "out_pen": out_pen,
            "grad_clip": grad_clip, "weight_decay": weight_decay,
            "history": hist, "wall_s": round(time.time() - t0, 2),
            "opt_state_carried": start_step > 0}


# ---------------------------------------------------------------------------
# Skill-in-core probe + artifact I/O
# ---------------------------------------------------------------------------


@torch.no_grad()
def skill_probe(model, blocks, pack, handles, gates64, home_train, home_held,
                per_slice, base_batches, device) -> dict:
    """Forced vs masked CE on FULL slices — the skill-in-core read.

    Forced = gates_pre (masked routing) + forced expert gain 1.0 (the door5
    `ce_home_forced` convention). No single-batch reads.
    """
    def ce(batches, g, m):
        return d1.ce_in_state(model, blocks, batches, device, g, handles, m)

    out = {
        "heldout": {
            "ce_masked": ce(home_held, gates64, "off"),
            "ce_forced": ce(home_held, gates64, "forced"),
        },
        "train": {
            "ce_masked": ce(home_train, gates64, "off"),
            "ce_forced": ce(home_train, gates64, "forced"),
        },
    }
    for s in ("heldout", "train"):
        out[s]["delta_forced"] = out[s]["ce_forced"] - out[s]["ce_masked"]
        out[s]["skill_in_core"] = out[s]["ce_forced"] < out[s]["ce_masked"]
    slices = {}
    for name, batches in per_slice.items():
        if not batches:
            continue
        slices[name] = {"ce_masked": ce(batches, gates64, "off"),
                        "ce_forced": ce(batches, gates64, "forced")}
        slices[name]["delta_forced"] = (slices[name]["ce_forced"]
                                        - slices[name]["ce_masked"])
    out["per_slice"] = slices
    out["base_ce"] = {"masked": ce(base_batches, gates64, "off"),
                      "forced": ce(base_batches, gates64, "forced")}
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.empty_cache()
    return out


def save_plugin(path: str, pack, new_idx: int, meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "format": PLUGIN_FORMAT,
        "experts": [e.state_dict() for e in pack.experts],
        "rows": [b.P.data[new_idx].cpu() for b in pack.banks65],
        "meta": dict(meta, new_idx=new_idx),
    }, tmp)
    os.replace(tmp, path)


def dump_json(out: dict, path: str | None) -> None:
    if not path:
        return
    d4.save_json(path, out)


# ---------------------------------------------------------------------------
# Phase B — mass-cal row recal (door3 run_d3 contract; feeds the artifact rows)
# ---------------------------------------------------------------------------


def recal_rows(model, blocks, pack, handles, gates64, home_train, base_batches,
               args, device) -> dict:
    label_of = lambda i, b: b[1] if isinstance(b[1], (int, float)) else 0  # noqa: E731
    new_idx = pack.banks65[0].num_experts - 1
    pairs = d1.run_in_state(
        model, blocks, gates64, handles, "off",
        lambda: d1.collect_gate_pairs(model, blocks, home_train + base_batches,
                                      device, args.max_cal_pairs, label_of))
    with torch.no_grad():
        for bank, (h, lab) in zip(pack.banks65, pairs):
            home_h = h[lab > 0.5]
            if home_h.shape[0] > 0:
                bank.P.data[new_idx] = (home_h.mean(dim=0)
                                        * math.sqrt(bank.d_r))
    pb1 = d1.train_rows(pack, pairs, args.row_steps, args.row_lr, device,
                        reject_w=args.reject_w, mass_w=args.mass_w,
                        owner_target=args.owner_mass)
    pairs2 = d1.run_in_state(
        model, blocks, pack.graft_gates, handles, "plugged",
        lambda: d1.collect_gate_pairs(model, blocks, home_train + base_batches,
                                      device, args.max_cal_pairs, label_of))
    pb2 = d1.train_rows(pack, pairs2, max(50, args.row_steps // 2),
                        args.row_lr, device, reject_w=args.reject_w,
                        mass_w=args.mass_w, owner_target=args.owner_mass)
    return {"round1_teacher_forced": pb1, "round2_on_trajectory": pb2}


# ---------------------------------------------------------------------------
# --phase distill: CE fit -> recal -> artifact -> skill probe (+ doubled arm)
# ---------------------------------------------------------------------------


def run_distill(model, blocks, banks_registry, cache, home_train, home_held,
                home_plain_train, home_plain_held, probe_windows,
                base_batches, base_eval, per_slice_plain, args,
                device) -> dict:
    out = {"gate": "door7-ce-distill", "phase": "distill", "format": FORMAT,
           "objective": ("ce_ground_truth" if args.targets == "gt"
                         else "ce_teacher_argmax (gt where teacher agrees)"),
           "target_mode": args.targets,
           "budget": {"steps": args.expert_steps,
                      "extra_steps": args.extra_steps,
                      "lr": args.expert_lr, "wd": args.weight_decay,
                      "batch_size": args.batch_size, "seq_len": args.seq_len,
                      "matches": "door3 KL arm (200 steps, lr 1e-4, bs2x512)"},
           "started": time.time()}
    t0 = time.time()
    top_k = getattr(blocks[0].gate, "top_k", 2)
    norm_topk = bool(getattr(blocks[0].gate, "norm_topk_prob", False))
    pack = d1.build_pack(blocks, banks_registry, (top_k, norm_topk),
                         inter_mult=args.inter_mult, seed=args.seed)
    pack.experts.to(device)
    handles = d1.install_plugin(blocks, pack, "off")
    new_idx = pack.banks65[0].num_experts - 1
    gates64 = [d0.GraftGate(b, top_k, norm_topk, stock_gate=blk.gate)
               for b, blk in zip(banks_registry, blocks)]
    shared_s_ids = cache["shared_s_ids"]

    out["target_mix_train"] = batch_ce_stats(home_train, shared_s_ids)
    out["target_mix_held"] = batch_ce_stats(home_held, shared_s_ids)
    print("target mix train:", out["target_mix_train"], flush=True)

    def fit_and_pack(steps, start_step, tag, plugin_path, opt_ref):
        print(f"PHASE A [{tag}] — CE-objective fit ({steps} steps, lr "
              f"{args.expert_lr}, forced dispatch)", flush=True)
        pa = train_expert_ce(
            model, blocks, pack, handles, home_train, base_batches,
            steps, args.expert_lr, device, shared_s_ids,
            out_pen=args.out_pen, grad_clip=args.grad_clip,
            weight_decay=args.weight_decay, start_step=start_step,
            opt=opt_ref[0], opt_ref=opt_ref, target_mode=args.targets)
        if pa.get("aborted_nonfinite"):
            return {"phase_a": pa, "aborted": True}
        print(f"PHASE B [{tag}] — row recal (owner {args.owner_mass} / "
              f"reject {args.reject_w})", flush=True)
        pb = recal_rows(model, blocks, pack, handles, gates64,
                        home_plain_train, base_batches, args, device)
        save_plugin(plugin_path, pack, new_idx, {
            "home_domains": args.home_domains,
            "objective": ("ce_ground_truth" if args.targets == "gt"
                          else "ce_teacher_argmax"),
            "objective_detail": (
                "pure ground-truth next-token CE on home text"
                if args.targets == "gt" else
                "gt id where teacher argmax agrees or no teacher row; "
                "teacher argmax id otherwise"),
            "host_registry": args.registry,
            "teacher_cache": args.cache,
            "steps_total": pa["steps"],
            "expert_lr": args.expert_lr,
            "out_pen": args.out_pen,
            "target_mix_train": out["target_mix_train"],
        })
        print(f"SKILL PROBE [{tag}] — forced vs masked, full slices",
              flush=True)
        sp = skill_probe(model, blocks, pack, handles, gates64,
                         home_plain_train, home_plain_held, per_slice_plain,
                         base_eval, device)
        print(f"  skill[{tag}] held forced {sp['heldout']['ce_forced']:.4f} "
              f"vs masked {sp['heldout']['ce_masked']:.4f} | train "
              f"{sp['train']['ce_forced']:.4f} vs {sp['train']['ce_masked']:.4f}",
              flush=True)
        return {"phase_a": pa, "phase_b": pb, "skill_probe": sp,
                "plugin_path": plugin_path}

    opt_ref = [None]
    r200 = fit_and_pack(args.expert_steps, 0, "step%d" % args.expert_steps,
                        args.plugin_out, opt_ref)
    out["arm_main"] = r200
    dump_json(out, args.out)
    if args.extra_steps and not r200.get("aborted"):
        path2 = args.plugin_out.replace(".pt",
                                        f"_plus{args.extra_steps}.pt")
        if path2 == args.plugin_out:
            path2 = args.plugin_out + f".plus{args.extra_steps}"
        rX = fit_and_pack(args.extra_steps, args.expert_steps,
                          f"plus{args.extra_steps}", path2, opt_ref)
        out["arm_doubled"] = rX
        dump_json(out, args.out)
    out["core_fp"] = d5.core_fingerprint(
        [(e.gate_up_proj.detach().float().cpu(),
          e.down_proj.detach().float().cpu()) for e in pack.experts])
    d1.uninstall_plugin(handles)
    out["wall_s"] = round(time.time() - t0, 2)
    return out


# ---------------------------------------------------------------------------
# --phase wrap (PRIMARY): single-site Door-2.5 wrap + additive lambda delivery
# ---------------------------------------------------------------------------


def run_wrap_ce(model, blocks, banks_registry, cores, rows, home_train,
                home_held, per_slice, base_batches, base_eval, probe_windows,
                args, device) -> dict:
    out = {"gate": "door7-ce-distill", "arm": "wrap-ce-core",
           "format": FORMAT,
           "recipe": {
               "geometry": ("host_in->c(64)->adapter_up->FROZEN CE CORE->"
                            "adapter_down->c'->host_out"),
               "expert_lr": args.expert_lr, "expert_steps": args.expert_steps,
               "weight_decay": 0.0, "out_pen": args.neutral_w,
               "grad_clip": args.grad_clip,
               "delivery_primary": "additive lambda*m (Door 6 winner)",
               "delivery_secondary_read": "mass contract plugged",
               "rows": {"owner_mass": args.owner_mass,
                        "reject_w": args.reject_w, "mass_w": args.mass_w,
                        "row_steps": args.row_steps},
               "lam_fit": {"steps": args.lam_steps, "lr": args.lam_lr}},
           "core_fingerprint_pre": d5.core_fingerprint(cores),
           "started": time.time()}
    t0 = time.time()
    host_i = int(getattr(args, "host_layer", 6))
    top_k = getattr(blocks[0].gate, "top_k", 2)
    norm_topk = bool(getattr(blocks[0].gate, "norm_topk_prob", False))
    d_model = blocks[0].gate.weight.shape[1]
    if not (0 <= host_i < len(cores)):
        raise RuntimeError(f"host_layer {host_i} out of range")
    wrap = d25.WrappedForeignExpert(d_model, args.d_canon,
                                    cores[host_i][0], cores[host_i][1],
                                    seed=args.seed)
    pack, gates64, gates_plugged, top_k, norm_topk = d25.build_site_pack(
        blocks, banks_registry, host_i, wrap)
    site_blocks = [blocks[host_i]]
    out["recipe"]["host_layer"] = host_i
    out["recipe"]["core_layer"] = host_i
    pack.experts.to(device)
    if rows is not None:
        new_idx = pack.banks65[0].num_experts - 1
        pack.banks65[0].P.data[new_idx] = rows[host_i].to(
            pack.banks65[0].P.device)
    new_idx = pack.banks65[0].num_experts - 1
    out["trainable_params"] = sum(p.numel() for p in pack.parameters())
    out["frozen_core_params"] = int(cores[host_i][0].numel()
                                    + cores[host_i][1].numel())
    out["wrap_spec_l0"] = pack.experts[0].wrap_spec()

    # ---- Phase A: thin-adapter fit (door5 (i') recipe verbatim) ----
    print(f"PHASE A — wrap adapter fit (single site L{host_i}, forced, lr "
          f"{args.expert_lr} wd 0, out_pen {args.neutral_w})", flush=True)
    handles = d1.install_plugin(site_blocks, pack, "off")
    pa = d5.train_adapters_guarded(
        model, pack, handles, home_train, base_batches,
        args.expert_steps, args.expert_lr, device, out_pen=args.neutral_w,
        grad_clip=args.grad_clip, weight_decay=0.0)
    out["phase_a"] = pa
    out["core_fingerprint_post_a"] = d5.core_fingerprint(cores)
    out["core_frozen_bitwise"] = (out["core_fingerprint_pre"]
                                  == out["core_fingerprint_post_a"])
    dump_json(out, args.out)
    if pa.get("aborted_nonfinite"):
        print("Phase A aborted (non-finite) — stopping the arm", flush=True)
        out["wall_s"] = round(time.time() - t0, 2)
        return out

    # ---- Phase B round-1: mass-cal row recal (door5 run_wrap verbatim) ----
    print("PHASE B1 — row recal round-1 (teacher-forced)", flush=True)
    label_of = lambda _i, b: b[1]  # noqa: E731
    pairs = d1.run_in_state(
        model, blocks, gates64, handles, "off",
        lambda: d1.collect_gate_pairs(model, site_blocks,
                                      home_train + base_batches,
                                      device, args.max_cal_pairs, label_of))
    with torch.no_grad():
        bank = pack.banks65[0]
        h, lab = pairs[0]
        code_h = h[lab > 0.5]
        if code_h.shape[0] > 0:
            bank.P.data[new_idx] = (code_h.mean(dim=0)
                                    * math.sqrt(bank.d_r))
    pb1 = d1.train_rows(pack, pairs, args.row_steps, args.row_lr, device,
                        reject_w=args.reject_w, mass_w=args.mass_w,
                        owner_target=args.owner_mass)
    out["phase_b1"] = pb1
    dump_json(out, args.out)

    # ---- additive delivery: lambda fit + round-2 on-trajectory ----
    print("PHASE C — additive lambda fit (cores + adapters frozen)",
          flush=True)
    d1.uninstall_plugin(handles)
    lam = nn.Parameter(torch.tensor(float(args.lam_init)))
    add_gate = d4.AdditiveGraftGate(pack.banks65[0], top_k, norm_topk,
                                    stock_gate=blocks[host_i].gate)
    gates_add = [add_gate if i == host_i else gates64[i]
                 for i in range(len(blocks))]
    hB = d4.install_additive(site_blocks, pack, [lam], mode="off")
    restored = [(blocks[host_i], d0.install_graft(blocks[host_i], add_gate))]
    fit = d5.train_lambda_only(model, blocks, pack, hB, [lam], home_train,
                               args.lam_steps, args.lam_lr, device,
                               grad_clip=args.grad_clip)
    out["lam_fit"] = fit
    out["lambda"] = round(float(lam.detach().clamp(min=0)), 4)

    print("PHASE B2 — row recal round-2 on-trajectory (additive live)",
          flush=True)
    pairs2 = d1.run_in_state(
        model, blocks, gates_add, hB, "additive",
        lambda: d1.collect_gate_pairs(model, site_blocks,
                                      home_train + base_batches,
                                      device, args.max_cal_pairs, label_of))
    pb2 = d1.train_rows(pack, pairs2, max(50, args.row_steps // 2),
                        args.row_lr, device, reject_w=args.reject_w,
                        mass_w=args.mass_w, owner_target=args.owner_mass)
    out["phase_b2"] = pb2
    out["core_fingerprint_post"] = d5.core_fingerprint(cores)
    out["core_frozen_bitwise"] = (out["core_fingerprint_pre"]
                                  == out["core_fingerprint_post"])
    dump_json(out, args.out)

    # ---- measure ----
    fn_pre = d1.make_logits_fn(model, blocks, gates64, handles, "off", device)
    probes = d1.select_code_probes(fn_pre, probe_windows, args.n_probes)
    out["n_probes"] = len(probes)

    print("MEASURE primary — additive lambda*m (Door 6 delivery)", flush=True)
    m_add = d5.measure_all(
        "wrap-additive", model, blocks, pack, hB, gates_add, "additive",
        gates64, probes, home_held, home_train, per_slice, base_eval, device,
        new_idx, additive=True, lams=[lam], route_blocks=site_blocks)
    m_add["lambda"] = out["lambda"]
    out["result_additive"] = m_add
    dump_json(out, args.out)

    print("MEASURE secondary read — mass contract (door5 (i') comparability)",
          flush=True)
    for blk, stock in restored:
        d0.uninstall_graft(blk, stock)
    d1.uninstall_plugin(hB)
    handles2 = d1.install_plugin(site_blocks, pack, "off")
    m_mass = d5.measure_all(
        "wrap-mass", model, blocks, pack, handles2, gates_plugged, "plugged",
        gates64, probes, home_held, home_train, per_slice, base_eval, device,
        new_idx, additive=False, route_blocks=site_blocks)
    out["result_mass"] = m_mass
    out["result"] = m_add  # primary
    d1.uninstall_plugin(handles2)
    out["wall_s"] = round(time.time() - t0, 2)
    return out


# ---------------------------------------------------------------------------
# --phase deliver (SECONDARY): distill-then-deliver, no refit (door5 (ii))
# ---------------------------------------------------------------------------


def run_deliver_ce(model, blocks, banks_registry, cores, rows, home_train,
                   home_held, per_slice, base_batches, base_eval,
                   probe_windows, args, device) -> dict:
    res = d5.run_deliver(model, blocks, banks_registry, cores, rows,
                         home_train, home_held, per_slice, base_batches,
                         base_eval, probe_windows, args, device)
    res["gate"] = "door7-ce-distill"
    res["arm"] = "deliver-ce-core-noref"
    res["format"] = FORMAT
    return res


# ---------------------------------------------------------------------------
# Synthetic self-test (--synthetic)
# ---------------------------------------------------------------------------


def run_synthetic_tests() -> int:
    torch.manual_seed(0)
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" — {detail}" if detail else ""))

    device = "cpu"

    # --- hard-target rule ---
    S = 5
    shared_s_ids = torch.tensor([10, 11, 12, 13, 14])
    ids = torch.tensor([[1, 10, 13, 11]])
    tgt = torch.zeros(1, 3, S)
    # pos0 (predict 10): teacher argmax = 0 -> hard_id 10 == gt -> agree -> gt
    tgt[0, 0, 0] = 5.0
    # pos1 (predict 13): teacher argmax = 1 -> hard_id 11 != 13 -> hard -> 11
    tgt[0, 1, 1] = 5.0
    # pos2 (predict 11): mask False (unaligned) -> gt 11
    tgt[0, 2, 2] = 5.0

    class DB:
        pass

    b = DB()
    b.ids, b.tgt, b.mask = ids, tgt, torch.tensor([[True, True, False]])
    r = ce_hard_targets(b, shared_s_ids)
    check("hard_targets_rule",
          r["targets"].tolist() == [[10, 11, 11]]
          and r["n_agree"] == 1 and r["n_hard"] == 1 and r["n_unaligned"] == 1,
          f"targets {r['targets'].tolist()} mix a/h/u "
          f"{r['n_agree']}/{r['n_hard']}/{r['n_unaligned']}")

    # --- mock host + CE fit + skill probe structure ---
    model, n_dom = d0.build_mock_donor()
    blocks = d0.find_moe_blocks(model)
    model.eval()
    e, h = blocks[0].gate.weight.shape
    banks_reg = [d0.PassportBank(e, h) for _ in blocks]
    d0.init_passports_exact(banks_reg, [b.gate.weight for b in blocks])
    home_plain_train = [(t, 1) for t, d in d0.mock_batches(
        n_dom, seed=11, batches_per_domain=2) if d == 1]
    home_plain_held = [(t, 1) for t, d in d0.mock_batches(
        n_dom, seed=12, batches_per_domain=2) if d == 1]
    base_b = [(t, 0) for t, d in d0.mock_batches(n_dom, seed=17,
                                                 batches_per_domain=2)
              if d != 1]
    probe_wins = [t.flatten() for t, d in d0.mock_batches(
        n_dom, seed=23, seq_len=48, batches_per_domain=3, batch_size=1)
        if d == 1]
    per_slice = {"code": home_plain_held}

    top_k = getattr(blocks[0].gate, "top_k", 2)
    norm_topk = bool(getattr(blocks[0].gate, "norm_topk_prob", False))
    pack = d1.build_pack(blocks, banks_reg, (top_k, norm_topk), seed=0)
    handles = d1.install_plugin(blocks, pack, "off")
    gates64 = [d0.GraftGate(bnk, top_k, norm_topk, stock_gate=blk.gate)
               for bnk, blk in zip(banks_reg, blocks)]

    # synthetic DistillBatches over the mock home tokens
    v_model = blocks[0].gate.weight.shape[1]
    n_vocab = 64
    s_ids = torch.arange(4, 20)
    home_dist = []
    for toks, _ in home_plain_train[:2]:
        B, L = toks.shape
        tt = torch.zeros(B, L - 1, s_ids.shape[0])
        tt[:, :, 0] = 3.0
        mm = torch.ones(B, L - 1, dtype=torch.bool)
        mm[:, -2:] = False
        db = DB()
        db.ids, db.tgt, db.mask = toks, tt, mm
        home_dist.append(db)
    mix = batch_ce_stats(home_dist, s_ids)
    check("batch_ce_stats",
          mix["n_total"] > 0 and mix["n_agree"] >= 0
          and abs(mix["frac_agree"] + mix["frac_hard"] + mix["frac_unaligned"]
                  - 1.0) < 1e-6,
          f"mix {mix}")

    class A:
        pass

    args = A()
    args.out_pen = 5.0
    args.grad_clip = 1.0
    args.weight_decay = 0.0
    args.max_cal_pairs = 256
    args.row_steps = 40
    args.row_lr = 0.1
    args.reject_w = 5.0
    args.mass_w = 2.0
    args.owner_mass = 0.35
    args.n_probes = 6
    args.d_canon = 8
    args.expert_steps = 6
    args.expert_lr = 1e-2
    args.neutral_w = 5.0
    args.lam_steps = 10
    args.lam_lr = 5e-2
    args.lam_init = 0.5
    args.out = None
    args.seed = 0
    args.host_layer = 0

    fit = train_expert_ce(model, blocks, pack, handles, home_dist, base_b,
                          4, 1e-2, device, s_ids, out_pen=5.0,
                          weight_decay=0.0)
    check("ce_fit_runs",
          (not fit.get("aborted_nonfinite")) and fit["history"]
          and fit["history"][-1]["ce"] is not None,
          f"steps {fit['steps']} ce {fit['history'][-1]['ce']:.3f}")
    sp = skill_probe(model, blocks, pack, handles, gates64,
                     home_plain_train, home_plain_held, per_slice, base_b,
                     device)
    check("skill_probe_struct",
          {"heldout", "train", "per_slice", "base_ce"} <= set(sp)
          and "skill_in_core" in sp["train"],
          f"held forced {sp['heldout']['ce_forced']:.3f} vs "
          f"{sp['heldout']['ce_masked']:.3f}")

    # artifact round-trip in the door3/door5 consumer format
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), "plugin.pt")
    inter = int(blocks[0].experts.gate_up_proj.shape[1] // 2)
    cores = [(torch.randn(2 * inter, h) * 0.05, torch.randn(h, inter) * 0.05)
             for _ in blocks]
    rows = [torch.randn(h) * 0.01 for _ in blocks]
    fp = d5.core_fingerprint(cores)
    torch.save({"format": PLUGIN_FORMAT,
                "experts": [{"gate_up_proj": w_gu, "down_proj": w_down}
                            for w_gu, w_down in cores],
                "rows": rows, "meta": {"new_idx": 64}}, tmp)
    cores2, rows2, meta2 = d5.extract_distill_cores(tmp)
    check("artifact_roundtrip",
          d5.core_fingerprint(cores2) == fp and len(rows2) == len(rows)
          and meta2["format"] == PLUGIN_FORMAT,
          f"fp {d5.core_fingerprint(cores2)}")

    # wrap (primary: additive + mass reads) and deliver (secondary)
    d1.uninstall_plugin(handles)
    out_w = run_wrap_ce(model, blocks, banks_reg, cores2, rows2,
                        home_plain_train, home_plain_held, per_slice, base_b,
                        base_b, probe_wins, args, device)
    ra, rm = out_w["result_additive"], out_w["result_mass"]
    check("wrap_core_frozen",
          out_w["core_frozen_bitwise"],
          f"fp {out_w['core_fingerprint_pre']}")
    check("wrap_additive_verdict_struct",
          {"quality_heldout_beats_masked", "quality_train_beats_masked",
           "parity_ge_0.95", "retention_ge_0.70",
           "contamination_le_0.30"} <= set(ra["verdict"]["gates"]),
          f"pass={ra['verdict']['pass']} lam {out_w['lambda']}")
    check("wrap_mass_read_present",
          "s6_quality" in rm and "base_parity" in rm,
          f"mass held {rm['s6_quality']['heldout']['ce_home_active']:.3f}")
    check("wrap_forced_reported",
          "ce_home_forced" in ra["s6_quality"]["heldout"],
          f"forced held {ra['s6_quality']['heldout']['ce_home_forced']:.3f}")

    out_d = run_deliver_ce(model, blocks, banks_reg, cores2, rows2,
                           home_plain_train, home_plain_held, per_slice,
                           base_b, base_b, probe_wins, args, device)
    rd = out_d["result"]
    check("deliver_core_frozen",
          out_d["core_frozen_bitwise"] and not out_d["lam_fit"].get(
              "aborted_nonfinite"),
          f"lam_mean {out_d['lambda_mean']}")
    check("deliver_verdict_struct",
          {"quality_heldout_beats_masked", "quality_train_beats_masked",
           "parity_ge_0.95", "retention_ge_0.70",
           "contamination_le_0.30"} <= set(rd["verdict"]["gates"]),
          f"pass={rd['verdict']['pass']}")
    check("gates_consistent",
          set(ra["verdict"]["gates"]) == set(rd["verdict"]["gates"]),
          "shared gate set")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} synthetic checks passed")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--phase", default="distill",
                    choices=["distill", "wrap", "deliver"])
    ap.add_argument("--host-model", default=None)
    ap.add_argument("--model", default=None,
                    help="alias of --host-model (door5 CLI compatibility)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--cache", default=None,
                    help="door3 teacher_cache.pt (exact distill split)")
    ap.add_argument("--artifact", default=None,
                    help="plugin artifact for wrap/deliver phases")
    ap.add_argument("--plugin-out", default=None,
                    help="where the distill phase writes the CE core pack")
    ap.add_argument("--domains-dir",
                    default=os.path.join(REPO_ROOT, "lab", "graft_d0_texts"))
    ap.add_argument("--home-domains", default="code,math")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="distill fit batching — the door3 KL-arm contract "
                         "(bs2x512; this forced-dispatch graph is proven at "
                         "B*T=1024). Wrap/measure use bs4x512=2048.")
    ap.add_argument("--eval-batch-size", type=int, default=2)
    ap.add_argument("--window-stride", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--max-base-tokens", type=int, default=60000)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--expert-steps", type=int, default=200,
                    help="the KL arm's budget (door3: 200 steps)")
    ap.add_argument("--extra-steps", type=int, default=0,
                    help="continued doubled-budget arm (same optimizer)")
    ap.add_argument("--targets", default="argmax", choices=["argmax", "gt"],
                    help="argmax = teacher hard targets (gt where the "
                         "teacher agrees); gt = pure ground-truth CE")
    ap.add_argument("--expert-lr", type=float, default=1e-4,
                    help="distill phase LR (KL-arm budget); the wrap phase "
                         "uses its own --wrap-lr default 3e-3")
    ap.add_argument("--wrap-lr", type=float, default=3e-3)
    ap.add_argument("--wrap-steps", type=int, default=250,
                    help="thin-adapter fit steps (door5 (i') recipe: 250)")
    ap.add_argument("--measure-batch-size", type=int, default=4,
                    help="train-slice batch shape for full-slice CE reads "
                         "(i' protocol: bs4x512=2048)")
    ap.add_argument("--out-pen", type=float, default=300.0)
    ap.add_argument("--neutral-w", type=float, default=300.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lam-steps", type=int, default=150)
    ap.add_argument("--lam-lr", type=float, default=1e-2)
    ap.add_argument("--lam-init", type=float, default=0.5)
    ap.add_argument("--row-steps", type=int, default=400)
    ap.add_argument("--row-lr", type=float, default=0.05)
    ap.add_argument("--reject-w", type=float, default=RECIPE["reject_w"])
    ap.add_argument("--mass-w", type=float, default=RECIPE["mass_w"])
    ap.add_argument("--owner-mass", type=float, default=RECIPE["owner_mass"])
    ap.add_argument("--max-cal-pairs", type=int, default=4096)
    ap.add_argument("--n-probes", type=int, default=32)
    ap.add_argument("--d-canon", type=int, default=64)
    ap.add_argument("--host-layer", type=int, default=6)
    ap.add_argument("--inter-mult", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.synthetic:
        return run_synthetic_tests()
    if args.model and not args.host_model:
        args.host_model = args.model
    if not args.host_model or not args.registry:
        ap.error("--host-model/--model and --registry are required "
                 "unless --synthetic")
    if args.phase == "distill" and (not args.cache or not args.plugin_out):
        ap.error("--phase distill needs --cache and --plugin-out")
    if args.phase in ("wrap", "deliver") and not args.artifact:
        ap.error(f"--phase {args.phase} needs --artifact")

    # wrap-phase shapes (the i' protocol): bs4x512=2048 / eval 2
    if args.phase in ("wrap", "deliver"):
        args.batch_size = 4
    batch_tokens = args.batch_size * args.seq_len
    if batch_tokens in {256, 768, 1024, 4096} and args.phase != "distill":
        print(f"WARNING: B*T={batch_tokens} is in the gfx906 backward hole "
              f"(FAILURES door25-gfx906-backward-holes).", flush=True)

    torch.manual_seed(args.seed)
    t_load = time.time()
    model, tok = d0.load_donor(args.host_model, args.device, args.dtype)
    blocks = d0.find_moe_blocks(model)
    banks_reg, _ = d0.load_registry(args.registry)
    if len(banks_reg) != len(blocks):
        raise RuntimeError("registry/model layer mismatch")
    print(f"loaded host+registry in {time.time() - t_load:.1f}s "
          f"({len(blocks)} layers)", flush=True)

    home_names = [d.strip() for d in args.home_domains.split(",") if d.strip()]
    domains = d0.load_domain_texts(args.domains_dir)
    base_ids = []
    for n, t in domains.items():
        if n not in home_names:
            base_ids.extend(tok.encode(t, add_special_tokens=False))
    base_ids = base_ids[:args.max_base_tokens]
    wins = [base_ids[i:i + args.seq_len]
            for i in range(0, len(base_ids) - args.seq_len + 1, args.seq_len)]
    base_batches = [
        (torch.tensor(wins[i:i + args.batch_size], dtype=torch.long), 0)
        for i in range(0, len(wins) - args.batch_size + 1, args.batch_size)]
    base_eval = [
        (torch.tensor(wins[i:i + args.eval_batch_size], dtype=torch.long), 0)
        for i in range(0, len(wins) - args.eval_batch_size + 1,
                       args.eval_batch_size)]

    out = {"gate": "door7-ce-distill", "phase": args.phase,
           "format": FORMAT, "host": {"model": args.host_model,
                                      "revision": args.revision},
           "registry": args.registry, "cache": args.cache,
           "artifact": args.artifact, "home_domains": args.home_domains,
           "seq_len": args.seq_len, "batch_size": args.batch_size,
           "batch_tokens": batch_tokens,
           "eval_batch_size": args.eval_batch_size,
           "seed": args.seed}
    if args.out is None:
        args.out = os.path.join(REPO_ROOT, "lab", "imported", "graft_d7",
                                f"door7_{args.phase}.json")
    dump_json(out, args.out)

    if args.phase == "distill":
        cache = d3.load_cache(args.cache)
        train_w, held_w = d3.make_windows(cache, args.seq_len,
                                          args.window_stride,
                                          args.max_windows, split_frac=0.6)
        home_train = d3.batch_windows(train_w, args.batch_size)
        home_held = d3.batch_windows(held_w, args.batch_size)
        # measurement-shaped plain batches (train bs4 / eval bs2, the i'
        # protocol) so skill-probe CEs are directly comparable to the wrap
        # and deliver arms' full-slice reads
        home_plain_train = d3.plain_batches(train_w, args.measure_batch_size, 1)
        home_plain_held = d3.plain_batches(held_w, args.eval_batch_size, 1)
        per_slice_plain = {}
        for name in home_names:
            wins_n = [w for w in held_w if w["domain"] == name]
            per_slice_plain[name] = d3.plain_batches(wins_n, 1, 1)
        probe_windows = (d3.probe_windows_from(held_w)
                         or d3.probe_windows_from(train_w))
        if not home_train or not home_held or not base_batches:
            raise RuntimeError("not enough data")
        n_tr = sum(t.numel() for t, _ in home_plain_train)
        n_he = sum(t.numel() for t, _ in home_plain_held)
        print(f"data: distill train {len(home_train)} batches / {n_tr} tok, "
              f"held {len(home_held)} batches / {n_he} tok, base "
              f"{len(base_batches)} batches", flush=True)
        out["n_home_train_tokens"] = int(n_tr)
        out["n_home_heldout_tokens"] = int(n_he)
        result = run_distill(model, blocks, banks_reg, cache, home_train,
                             home_held, home_plain_train, home_plain_held,
                             probe_windows, base_batches, base_eval,
                             per_slice_plain, args, args.device)
    else:
        cores, rows, art_meta = d5.extract_distill_cores(args.artifact)
        if len(cores) != len(blocks):
            raise RuntimeError(f"artifact has {len(cores)} cores, "
                               f"model {len(blocks)}")
        print(f"CE cores: {len(cores)} layers, shapes "
              f"{tuple(cores[0][0].shape)}/{tuple(cores[0][1].shape)}, "
              f"fp {d5.core_fingerprint(cores)}", flush=True)
        out["artifact_meta"] = art_meta
        out["core_fp"] = d5.core_fingerprint(cores)
        train_w, held_w = d5.home_windows_from_cache(
            args.cache, args.seq_len, args.window_stride)
        home_train = d5.plain_batches(train_w, args.batch_size, 1)
        home_held = d5.plain_batches(held_w, args.eval_batch_size, 1)
        per_slice = d5.per_slice_batches(held_w, batch_size=1)
        probe_windows = (d5.probe_windows_from(held_w)
                         or d5.probe_windows_from(train_w))
        if not home_train or not home_held or not base_batches:
            raise RuntimeError("not enough data")
        n_tr = sum(t.numel() for t, _ in home_train)
        n_he = sum(t.numel() for t, _ in home_held)
        print(f"data: home train {len(home_train)} batches / {n_tr} tok, "
              f"held {len(home_held)} batches / {n_he} tok", flush=True)
        out["n_home_train_tokens"] = int(n_tr)
        out["n_home_heldout_tokens"] = int(n_he)
        # the wrap fit uses the door5 (i') recipe unless overridden
        args.expert_lr = args.wrap_lr
        args.expert_steps = args.wrap_steps
        if args.phase == "wrap":
            result = run_wrap_ce(model, blocks, banks_reg, cores, rows,
                                 home_train, home_held, per_slice,
                                 base_batches, base_eval, probe_windows,
                                 args, args.device)
        else:
            result = run_deliver_ce(model, blocks, banks_reg, cores, rows,
                                    home_train, home_held, per_slice,
                                    base_batches, base_eval, probe_windows,
                                    args, args.device)

    out.update(result)
    out["end_unix"] = time.time()
    dump_json(out, args.out)
    print(f"\nresults -> {args.out}", flush=True)
    if args.phase == "distill":
        sp = result.get("arm_main", {}).get("skill_probe", {})
        if sp:
            print(f"SKILL (main arm): held forced "
                  f"{sp['heldout']['ce_forced']:.4f} vs masked "
                  f"{sp['heldout']['ce_masked']:.4f} | train "
                  f"{sp['train']['ce_forced']:.4f} vs "
                  f"{sp['train']['ce_masked']:.4f}", flush=True)
    else:
        m = result.get("result", {})
        q = m.get("s6_quality", {})
        if q:
            print(f"quality held-out: {q['heldout']['ce_home_active']:.4f} vs "
                  f"masked {q['heldout']['ce_home_masked']:.4f} (train "
                  f"{q['train']['ce_home_active']:.4f} vs "
                  f"{q['train']['ce_home_masked']:.4f}, forced held "
                  f"{q['heldout']['ce_home_forced']:.4f} train "
                  f"{q['train']['ce_home_forced']:.4f})", flush=True)
            p = m.get("base_parity", {})
            print(f"parity {p.get('logit_cosine_mean')}  contam "
                  f"{m.get('contamination_base', {}).get('mean_util')}  util "
                  f"{m.get('utilization_home', {}).get('mean_util')}",
                  flush=True)
            v = m.get("verdict", {})
            print(f"VERDICT: {'PASS' if v.get('pass') else 'FAIL'}  "
                  f"gates={v.get('gates')}", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
