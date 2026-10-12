#!/usr/bin/env python3
"""Door 5 — expert PROVENANCE arms (the named lever after door4b / 93188b0).

Q (the grab story's binding question): is expert PROVENANCE — a pretrained or
DISTILLED frozen core plus thin task-loss adapters — THE quality condition?

Background (HANDOFF 2026-10-11, FAILURES `door4b-fit-v2`): the only quality
pass of the line is Door 2.5 — a frozen PRETRAINED Phi-mini-MoE SwiGLU core
behind thin adapters (lr 3e-3, wd 0) under the mass contract (code CE 1.072
masked -> 0.439 plugged). Every from-scratch / refit delta fails quality
IN-SAMPLE across 3 contracts x 6 fit recipes. The 93188b0 "door3 core refit"
arm (warm-start from the Door-3 distilled artifact, then refit) went
ce 0.50 -> 5.14: at fit step 0 the distilled artifact delivered train CE
0.50 vs masked 1.26, and 250 refit steps destroyed the skill. Provenance,
not the delivery contract, is the named mechanism.

Two arms on the Door-3 distilled SwiGLU artifact
(`/data/tliao/graft_d3_out/plugin_distill_olmoe.pt`, 7e0a442; 16 per-layer
SwiGLU deltas, home = codeparrot + open-web-math):

  --arm wrap     (i) wrap-the-distilled-core: freeze the distilled cores
                 VERBATIM and wrap each in the EXACT Door-2.5 geometry
                 (lab/door25_wrap_expert.WrappedForeignExpert:
                 host_in -> c(64) -> adapter_up -> FROZEN CORE ->
                 adapter_down -> c' -> host_out), fit ONLY the thin adapters
                 with the Door-2.5 recipe (task CE + out_pen 300 on base-h,
                 lr 3e-3, wd 0, clip 1.0, 250 steps, forced dispatch), then
                 the D1 att-4 mass-cal row recal (owner_mass 0.35 /
                 reject_w 10 / mass_w 2, 400 + 200 on-trajectory). Delivery
                 = the MASS contract (plugged) — the contract under which
                 Door 2.5 passed quality. Question: does a distilled-but-
                 frozen core get the same quality pass as a donor-pretrained
                 core?

  --arm deliver  (ii) distill-then-deliver, NO refit: the distilled cores
                 stay bit-identical (hash-checked before/after), delivery is
                 contract (b) `y = y_stock + lam * m * e_new(h)` (lam learned
                 per layer, door4_delivery_contract.AdditiveGraftGate /
                 PluginMergeAdditive), and the only training is the mass-cal
                 row recal plus a lam-only delivery fit (trunk + cores
                 frozen). This isolates DELIVERY of the distilled skill —
                 the arm 93188b0 did not run (it refit the cores).

Targets (all green together = provenance story closed):
  quality held-out delivered < masked (THE gate) | parity >= 0.95 (with the
  fire/non-fire split) | retention >= 0.70 | contamination <= 0.30 |
  utilization > 0.50. Quality is reported BOTH train-slice and held-out —
  the in-sample read is the decisive one per 93188b0's diagnosis.

Data protocol (shared by both arms, fairness contract with the Door-3 run):
  home = code + math windows of the Door-3 teacher cache student_ids
  (EXACT distill train/held-out split: per-domain 60/40 at seq 512 /
  stride 256), base = non-home domains of --domains-dir (60k tokens), probes
  from the held-out windows. Quality per-slice (code / math) is reported.

Platform: MI50 box, fp16, HIP shims via graft_d0_parity; NaN guard on every
fit (lr 3e-3 can NaN — 93188b0 hit this); incremental JSON after each phase.

Usage:
  python3 lab/door5_provenance.py --synthetic
  python3 lab/door5_provenance.py --arm wrap \
      --model /data/tliao/OLMoE-1B-7B-0924 \
      --registry /data/tliao/graft_d0_out/registry_olmoe_6d84c485.pt \
      --artifact /data/tliao/graft_d3_out/plugin_distill_olmoe.pt \
      --cache /data/tliao/graft_d3_out/teacher_cache.pt \
      --domains-dir /data/tliao/graft_d0_slice \
      --device cuda --dtype float16 \
      --out /data/tliao/graft_d0_out/door5_wrap.json
"""

from __future__ import annotations

import argparse
import hashlib
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
import door25_wrap_expert as d25  # noqa: E402

BAR_PARITY = 0.95
BAR_RETENTION = 0.70
BAR_CONTAM = 0.30
BAR_UTIL = 0.50

# Door-2.5 / D1 att-4 recipe
RECIPE = {"owner_mass": 0.35, "reject_w": 10.0, "mass_w": 2.0, "out_pen": 300.0}

FORMAT = "vesper-door5-provenance-v1"


# ---------------------------------------------------------------------------
# Artifact I/O — the Door-3 distilled cores + rows
# ---------------------------------------------------------------------------


def load_plugin_artifact(path: str, pack) -> dict:
    """Load Door-3 expert state_dicts + passport rows into `pack` (warm start)."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    n_e = len(d["experts"])
    if n_e != len(pack.experts):
        raise RuntimeError(f"artifact has {n_e} experts, pack {len(pack.experts)}")
    for sd, exp in zip(d["experts"], pack.experts):
        exp.load_state_dict(sd)
    new_idx = pack.banks65[0].num_experts - 1
    for bank, row in zip(pack.banks65, d["rows"]):
        bank.P.data[new_idx] = row.float().to(bank.P.device)
    meta = d.get("meta", {})
    return {"format": d.get("format"),
            "meta": {k: v for k, v in meta.items()
                     if isinstance(v, (str, int, float, bool)) or v is None}}


def extract_distill_cores(path: str) -> tuple[list, list, dict]:
    """(cores, rows, meta) with cores = [(w_gu, w_down), ...] CPU float32."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    cores = [(sd["gate_up_proj"].detach().float().clone(),
              sd["down_proj"].detach().float().clone())
             for sd in d["experts"]]
    rows = [r.detach().float().clone() for r in d["rows"]]
    meta = d.get("meta", {})
    meta = {k: v for k, v in meta.items()
            if isinstance(v, (str, int, float, bool)) or v is None}
    return cores, rows, {"format": d.get("format"), "meta": meta}


def core_fingerprint(cores) -> str:
    """Hash of the frozen core tensors (arm-level freeze proof)."""
    h = hashlib.sha256()
    for w_gu, w_down in cores:
        h.update(w_gu.numpy().tobytes())
        h.update(w_down.numpy().tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Home/base data — EXACT Door-3 window split (teacher-cache student_ids)
# ---------------------------------------------------------------------------


def home_windows_from_cache(cache_path: str, seq_len: int, stride: int,
                            split_frac: float = 0.6):
    """Per-domain windows over the Door-3 student_ids, 60/40 train/held split.

    Mirrors door3_distill_expert.make_windows (ids only — no teacher targets).
    """
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    train, held = [], []
    for name, c in cache["domains"].items():
        sid = c["student_ids"]
        n = sid.shape[0]
        wins = []
        for start in range(0, n - seq_len, stride):
            wins.append({"ids": sid[start:start + seq_len].clone(),
                         "domain": name, "start": start})
        if not wins:
            raise RuntimeError(f"not enough student tokens in domain {name}")
        k = max(1, int(split_frac * len(wins)))
        train.extend(wins[:k])
        held.extend(wins[k:] if len(wins) > k else wins[:1])
    return train, held


def home_windows_from_texts(domains: dict, tok, home_names, seq_len: int,
                            stride: int, max_tokens: int,
                            split_frac: float = 0.6):
    """Fallback: same windowing straight off the domain texts."""
    train, held = [], []
    for name in home_names:
        sid = tok.encode(domains[name], add_special_tokens=False)[:max_tokens]
        sid = torch.tensor(sid, dtype=torch.long)
        n = sid.shape[0]
        wins = []
        for start in range(0, n - seq_len, stride):
            wins.append({"ids": sid[start:start + seq_len].clone(),
                         "domain": name, "start": start})
        if not wins:
            raise RuntimeError(f"not enough tokens in domain {name}")
        k = max(1, int(split_frac * len(wins)))
        train.extend(wins[:k])
        held.extend(wins[k:] if len(wins) > k else wins[:1])
    return train, held


def plain_batches(wins, batch_size: int, label: int) -> list:
    out = []
    for i in range(0, len(wins) - batch_size + 1, batch_size):
        rows = wins[i:i + batch_size]
        out.append((torch.stack([w["ids"] for w in rows]), label))
    return out


def per_slice_batches(wins, batch_size: int = 1) -> dict:
    out = {}
    for w in wins:
        out.setdefault(w["domain"], [])
    for name in out:
        wins_n = [w for w in wins if w["domain"] == name]
        out[name] = plain_batches(wins_n, batch_size, 1)
    return out


def probe_windows_from(wins, chunk: int = 256) -> list:
    out = []
    for w in wins:
        ids = w["ids"]
        for i in range(0, ids.shape[0] - chunk + 1, chunk):
            out.append(ids[i:i + chunk])
    return out


# ---------------------------------------------------------------------------
# Firing membership + parity with the fire/non-fire split
# ---------------------------------------------------------------------------


def fired_mask(blocks, new_idx: int):
    """(T,) bool — plug-in fired on the token at ANY layer; call INSIDE the pass."""
    f = None
    for blk in blocks:
        g = blk.gate
        m = getattr(g, "last_membership", None)
        if m is None:
            li = getattr(g, "last_indices", None)
            if li is not None:
                m = (li == new_idx).any(dim=-1)
        if m is not None:
            m = m.detach().cpu().bool().reshape(-1)
            f = m if f is None else (f | m)
    return f


@torch.no_grad()
def parity_split(model, blocks, batches, device, gates, handles, mode,
                 new_idx: int) -> dict:
    """Stock-vs-delivered base parity, overall + split by plug-in firing.

    Non-firing tokens are the structural floor (bit-identical path under both
    contracts when m=0); firing tokens carry the delta. Membership is captured
    INSIDE the delivered pass (gates are only live there).
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
            holder["fired"] = fired_mask(blocks, new_idx)
            return lg

        lg_p = d1.run_in_state(model, blocks, gates, handles, mode, delivered)
        cos = d0.token_logit_cosine(lg_s, lg_p)  # (B*T,)
        c = cos.detach().float().cpu()
        fired = holder.get("fired")
        if fired is None or fired.shape[0] != c.shape[0]:
            fired = torch.ones(c.shape[0], dtype=torch.bool)
        cn = c.numpy()
        f = fired.numpy()
        tot["cos"] += float(c.sum())
        tot["cos2"] += float((c * c).sum())
        tot["cos_min"] = min(tot["cos_min"], float(c.min()))
        fire["cos"] += float(cn[f].sum())
        fire["n"] += int(f.sum())
        non["cos"] += float(cn[~f].sum())
        non["n"] += int((~f).sum())
        tot["n"] += int(c.shape[0])
        ce_s += d0.next_token_ce(lg_s, tokens) * bsz
        ce_p += d0.next_token_ce(lg_p, tokens) * bsz
        n_row += bsz
    mean = tot["cos"] / max(1, tot["n"])
    var = max(0.0, tot["cos2"] / max(1, tot["n"]) - mean * mean)
    return {
        "n_tokens": tot["n"], "logit_cosine_mean": mean,
        "logit_cosine_std": math.sqrt(var), "logit_cosine_min": tot["cos_min"],
        "firing": {"n": fire["n"],
                   "mean_cos": fire["cos"] / max(1, fire["n"])},
        "non_firing": {"n": non["n"],
                       "mean_cos": non["cos"] / max(1, non["n"])},
        "ce_stock": ce_s / max(1.0, n_row),
        "ce_delivered": ce_p / max(1.0, n_row),
        "ce_delta": (ce_p - ce_s) / max(1.0, n_row),
    }


# ---------------------------------------------------------------------------
# Fit trainers (NaN-guarded; cores never trainable)
# ---------------------------------------------------------------------------


def train_adapters_guarded(model, pack, handles, home_batches, base_batches,
                           steps, lr, device, out_pen=RECIPE["out_pen"],
                           grad_clip=1.0, weight_decay=0.0) -> dict:
    """Door-2.5 Phase-A recipe with the 93188b0 NaN guard.

    loss = CE(home, forced) + out_pen * mean_l ||e_new(h_base)||^2
    Adam(lr, wd=0) on the wrap adapters only (cores are buffers). Aborts on a
    non-finite loss and reports the abort step.
    """
    params = list(pack.parameters())
    d1.freeze_all_except(model, params)
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    hist = []
    t0 = time.time()
    fixed = home_batches[0][0].to(device)
    for step in range(steps):
        toks, _ = home_batches[step % len(home_batches)]
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
        if not torch.isfinite(total):
            hist.append({"step": step, "ce_code": None, "out_norm2": None,
                         "aborted": True})
            print(f"  [wrapA {step:4d}] NON-FINITE loss — aborting fit")
            return {"steps": step, "lr": lr, "out_pen": out_pen,
                    "grad_clip": grad_clip, "weight_decay": weight_decay,
                    "history": hist, "aborted_nonfinite": True,
                    "wall_s": round(time.time() - t0, 2)}
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
            print(f"  [wrapA {step:4d}] ce {float(loss.detach()):.4f}  "
                  f"ce0 {ce0:.4f}  out2 {float(out_norm2.detach()):.3e}  "
                  f"|y| {float(yabs):.3e}  |W| {pl2:.3f}", flush=True)
    return {"steps": steps, "lr": lr, "out_pen": out_pen,
            "grad_clip": grad_clip, "weight_decay": weight_decay,
            "history": hist, "wall_s": round(time.time() - t0, 2)}


def train_lambda_only(model, blocks, pack, handles, lams, home_batches,
                      steps, lr, device, grad_clip=1.0) -> dict:
    """Delivery-only fit: the 16 lambda scalars, cores + trunk + rows frozen.

    loss = CE(home, additive delivered). The distilled cores see no gradients
    (they are not in `params`); freeze_all_except locks the trunk. NaN guard
    as in train_adapters_guarded.
    """
    params = list(lams)
    d1.freeze_all_except(model, params)
    opt = torch.optim.Adam(params, lr=lr)
    hist = []
    t0 = time.time()
    for step in range(steps):
        toks, _ = home_batches[step % len(home_batches)]
        toks = toks.to(device)
        d1.set_plugin_mode(handles, "additive", 1.0)
        logits = d0.model_logits(model, toks)
        v = logits.shape[-1]
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, v).float(),
                             toks[:, 1:].reshape(-1))
        if not torch.isfinite(ce):
            hist.append({"step": step, "ce_code": None, "aborted": True})
            print(f"  [lamF {step:4d}] NON-FINITE loss — aborting fit")
            return {"steps": step, "lr": lr, "history": hist,
                    "aborted_nonfinite": True,
                    "wall_s": round(time.time() - t0, 2)}
        opt.zero_grad()
        ce.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            hist.append({"step": step, "ce_code": float(ce.detach()),
                         "lam_head": [round(float(p.detach().clamp(min=0)), 4)
                                      for p in list(lams)[:2]]})
            print(f"  [lamF {step:4d}] ce_code {float(ce.detach()):.4f}  "
                  f"lam_head {hist[-1]['lam_head']}", flush=True)
    return {"steps": steps, "lr": lr, "history": hist,
            "wall_s": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------
# Measurement — the five gates, quality train + held-out, fire/non-fire split
# ---------------------------------------------------------------------------


def measure_all(tag, model, blocks, pack, handles, gates_delivered, mode,
                gates_pre, probes, home_held, home_train, per_slice,
                base_batches, device, new_idx, additive: bool,
                lams=None, route_blocks=None) -> dict:
    out = {"arm": tag, "mode": mode}
    route_blocks = blocks if route_blocks is None else route_blocks
    # membership capture for the mass contract needs gate.last_indices
    for g in gates_delivered:
        if hasattr(g, "capture"):
            g.capture = True

    fn_pre = d1.make_logits_fn(model, blocks, gates_pre, handles, "off", device)
    fn_solo = d1.make_logits_fn(model, blocks, gates_pre, handles, "forced",
                                device, force_gain=1.0)
    fn_lib = d1.make_logits_fn(model, blocks, gates_delivered, handles, mode,
                               device)
    m_pre = d1.probe_margins(probes, fn_pre)
    m_solo = d1.probe_margins(probes, fn_solo)
    m_lib = d1.probe_margins(probes, fn_lib)
    g_solo = m_solo["mean_logit_margin"] - m_pre["mean_logit_margin"]
    g_lib = m_lib["mean_logit_margin"] - m_pre["mean_logit_margin"]
    ret = d1.retention(g_lib, g_solo)
    out["margins"] = {"pre": m_pre, "solo": m_solo, "lib": m_lib,
                      "gain_solo": round(g_solo, 4),
                      "gain_lib": round(g_lib, 4),
                      "retention_vs_solo": None if ret is None else round(ret, 4),
                      "retention_degenerate": bool(g_solo <= 0)}

    out["base_parity"] = parity_split(model, blocks, base_batches, device,
                                      gates_delivered, handles, mode, new_idx)
    # registry-graft reference parity (door25's graft parity analog)
    tot = {"cos": 0.0, "n": 0}
    for tokens, _ in base_batches:
        tokens = tokens.to(device)
        lg_a = d1.run_in_state(model, blocks, gates_pre, handles, "off",
                               lambda: d0.model_logits(model, tokens))
        lg_b = d1.run_in_state(model, blocks, gates_delivered, handles, mode,
                               lambda: d0.model_logits(model, tokens))
        cos = d0.token_logit_cosine(lg_a, lg_b)
        tot["cos"] += float(cos.sum())
        tot["n"] += int(cos.numel())
    out["graft_parity_vs_registry_graft"] = {
        "logit_cosine_mean": tot["cos"] / max(1, tot["n"]),
        "n_tokens": tot["n"]}

    def hits(mode_gates, mode_m, batches):
        if additive:
            return d1.run_in_state(model, blocks, mode_gates, handles, mode_m,
                                   lambda: d4.membership_hits(
                                       model, route_blocks, batches, device))
        return d1.run_in_state(model, blocks, mode_gates, handles, mode_m,
                               lambda: d1.routing_hits(
                                   model, route_blocks, batches, device,
                                   new_idx))

    out["contamination_base"] = hits(gates_delivered, mode, base_batches)
    out["utilization_home"] = hits(gates_delivered, mode, home_train)
    mass_home = d1.run_in_state(
        model, blocks, gates_delivered, handles, mode,
        lambda: d1.routing_mass(model, route_blocks, home_train, device,
                                new_idx))
    mass_base = d1.run_in_state(
        model, blocks, gates_delivered, handles, mode,
        lambda: d1.routing_mass(model, route_blocks, base_batches, device,
                                new_idx))
    out["routing_mass"] = {"home": mass_home, "base": mass_base}

    # ---- quality: home CE masked vs delivered; train + held-out + slices
    def ce(batches, g, m):
        return d1.ce_in_state(model, blocks, batches, device, g, handles, m)

    q = {
        "heldout": {
            "ce_home_masked": ce(home_held, gates_pre, "off"),
            "ce_home_active": ce(home_held, gates_delivered, mode),
            "ce_home_forced": ce(home_held, gates_pre, "forced"),
        },
        "train": {
            "ce_home_masked": ce(home_train, gates_pre, "off"),
            "ce_home_active": ce(home_train, gates_delivered, mode),
            "ce_home_forced": ce(home_train, gates_pre, "forced"),
        },
    }
    for split_name, batches in (("heldout", home_held), ("train", home_train)):
        q[split_name]["delta"] = (q[split_name]["ce_home_active"]
                                  - q[split_name]["ce_home_masked"])
        q[split_name]["delta_forced"] = (q[split_name]["ce_home_forced"]
                                         - q[split_name]["ce_home_masked"])
    if lams is not None:
        saved = [float(p.detach()) for p in lams]
        with torch.no_grad():
            for p in lams:
                p.fill_(1.0)
        q["lam_fixed_one"] = {
            "heldout": ce(home_held, gates_delivered, mode),
            "train": ce(home_train, gates_delivered, mode),
        }
        with torch.no_grad():
            for p, s in zip(lams, saved):
                p.fill_(s)
    out["s6_quality"] = q

    slice_q = {}
    for name, batches in per_slice.items():
        if not batches:
            continue
        slice_q[name] = {"ce_masked": ce(batches, gates_pre, "off"),
                         "ce_active": ce(batches, gates_delivered, mode),
                         "ce_forced": ce(batches, gates_pre, "forced")}
        slice_q[name]["delta"] = (slice_q[name]["ce_active"]
                                  - slice_q[name]["ce_masked"])
    out["s6_quality_per_slice"] = slice_q

    out["base_ce"] = {
        "masked": ce(base_batches, gates_pre, "off"),
        "plugged": ce(base_batches, gates_delivered, mode),
    }
    out["base_ce"]["delta"] = out["base_ce"]["plugged"] - out["base_ce"]["masked"]

    # CE-space delivery retention (door25's analog when margins degenerate)
    gain_solo_ce = q["heldout"]["ce_home_masked"] - q["heldout"]["ce_home_forced"]
    gain_lib_ce = q["heldout"]["ce_home_masked"] - q["heldout"]["ce_home_active"]
    out["ce_delivery_retention"] = (None if gain_solo_ce <= 0
                                    else round(gain_lib_ce / gain_solo_ce, 4))

    ce_a = q["heldout"]["ce_home_active"]
    ce_m = q["heldout"]["ce_home_masked"]
    ret_gate = ret if ret is not None else 0.0
    gates = {
        "quality_heldout_beats_masked": ce_a < ce_m,
        "quality_train_beats_masked": (q["train"]["ce_home_active"]
                                       < q["train"]["ce_home_masked"]),
        "parity_ge_0.95": out["base_parity"]["logit_cosine_mean"] >= BAR_PARITY,
        "retention_ge_0.70": (ret_gate >= BAR_RETENTION
                              if not out["margins"]["retention_degenerate"]
                              else (out["ce_delivery_retention"] or 0) >= BAR_RETENTION),
        "contamination_le_0.30": out["contamination_base"]["mean_util"] <= BAR_CONTAM,
        "utilization_gt_0.50": out["utilization_home"]["mean_util"] > BAR_UTIL,
    }
    passed = all(gates[k] for k in (
        "quality_heldout_beats_masked", "parity_ge_0.95", "retention_ge_0.70",
        "contamination_le_0.30", "utilization_gt_0.50"))
    out["verdict"] = {
        "pass": passed, "gates": gates,
        "quality_ratio_heldout": round(ce_a / max(1e-9, ce_m), 4),
        "quality_ratio_train": round(
            q["train"]["ce_home_active"] / max(1e-9, q["train"]["ce_home_masked"]), 4),
        "bars": {"parity": BAR_PARITY, "retention": BAR_RETENTION,
                 "contamination": BAR_CONTAM, "utilization": BAR_UTIL},
    }
    return out


# ---------------------------------------------------------------------------
# Arm (i) — wrap-the-distilled-core
# ---------------------------------------------------------------------------


def run_wrap(model, blocks, banks_registry, cores, rows, home_train, home_held,
             per_slice, base_batches, probe_windows, args, device) -> dict:
    out = {"gate": "door5-provenance", "arm": "wrap-distilled-core",
           "format": FORMAT,
           "recipe": {"geometry": "host_in->c(64)->adapter_up->FROZEN CORE->"
                                  "adapter_down->c'->host_out",
                      "expert_lr": args.expert_lr,
                      "expert_steps": args.expert_steps,
                      "weight_decay": 0.0, "out_pen": args.neutral_w,
                      "grad_clip": args.grad_clip,
                      "delivery": "mass_contract_plugged",
                      "rows": {"owner_mass": args.owner_mass,
                               "reject_w": args.reject_w,
                               "mass_w": args.mass_w,
                               "row_steps": args.row_steps}},
           "core_fingerprint_pre": core_fingerprint(cores),
           "started": time.time()}
    t0 = time.time()
    layout = getattr(args, "wrap_layout", "all")
    host_i = int(getattr(args, "host_layer", 6))
    top_k = getattr(blocks[0].gate, "top_k", 2)
    norm_topk = bool(getattr(blocks[0].gate, "norm_topk_prob", False))
    d_model = blocks[0].gate.weight.shape[1]
    if layout == "single":
        # EXACT Door 2.5: one wrap at the depth-matched site, core index = site
        if not (0 <= host_i < len(cores)):
            raise RuntimeError(f"host_layer {host_i} out of range")
        wrap = d25.WrappedForeignExpert(d_model, args.d_canon,
                                        cores[host_i][0], cores[host_i][1],
                                        seed=args.seed)
        pack, gates64, gates_delivered, top_k, norm_topk = d25.build_site_pack(
            blocks, banks_registry, host_i, wrap)
        site_blocks = [blocks[host_i]]
        out["recipe"]["wrap_layout"] = "single"
        out["recipe"]["host_layer"] = host_i
        out["recipe"]["core_layer"] = host_i
    else:
        pack = d1.build_pack(blocks, banks_registry, (top_k, norm_topk),
                             inter_mult=1.0, seed=args.seed)
        pack.experts = nn.ModuleList([
            d25.WrappedForeignExpert(d_model, args.d_canon, w_gu, w_down,
                                     seed=args.seed + l)
            for l, (w_gu, w_down) in enumerate(cores)])
        gates64 = [d0.GraftGate(b, top_k, norm_topk, stock_gate=blk.gate)
                   for b, blk in zip(banks_registry, blocks)]
        gates_delivered = pack.graft_gates
        site_blocks = blocks
        out["recipe"]["wrap_layout"] = "all"
        out["recipe"]["n_sites"] = len(cores)
    pack.experts.to(device)
    if rows is not None:
        new_idx = pack.banks65[0].num_experts - 1
        if layout == "single":
            pack.banks65[0].P.data[new_idx] = rows[host_i].to(
                pack.banks65[0].P.device)
        else:
            for bank, row in zip(pack.banks65, rows):
                bank.P.data[new_idx] = row.to(bank.P.device)
    n_params = sum(p.numel() for p in pack.parameters())
    out["trainable_params"] = n_params
    out["frozen_core_params"] = int(sum(
        (w.numel() + d.numel())
        for l, (w, d) in enumerate(cores)
        if layout == "all" or l == host_i))
    out["wrap_spec_l0"] = pack.experts[0].wrap_spec()
    new_idx = pack.banks65[0].num_experts - 1

    # ---- Phase A: thin-adapter fit (Door-2.5 recipe, forced dispatch) ----
    print(f"PHASE A — wrap adapter fit [{layout}] (forced, lr "
          f"{args.expert_lr} wd 0, out_pen {args.neutral_w})")
    handles = d1.install_plugin(site_blocks, pack, "off")
    pa = train_adapters_guarded(
        model, pack, handles, home_train, base_batches,
        args.expert_steps, args.expert_lr, device, out_pen=args.neutral_w,
        grad_clip=args.grad_clip, weight_decay=0.0)
    out["phase_a"] = pa
    out["core_fingerprint_post_a"] = core_fingerprint(cores)
    out["core_frozen_bitwise"] = (out["core_fingerprint_pre"]
                                  == out["core_fingerprint_post_a"])
    if args.out:
        d4.save_json(args.out, out)

    # ---- Phase B: mass-cal row recal (owner 0.35 / reject 10) + on-traj ----
    print("PHASE B — mass-cal row recal (owner_mass "
          f"{args.owner_mass} / reject_w {args.reject_w})")
    label_of = lambda _i, b: b[1]  # noqa: E731
    pairs = d1.run_in_state(
        model, blocks, gates64, handles, "off",
        lambda: d1.collect_gate_pairs(model, site_blocks,
                                      home_train + base_batches,
                                      device, args.max_cal_pairs, label_of))
    with torch.no_grad():
        for bank, (h, lab) in zip(pack.banks65, pairs):
            code_h = h[lab > 0.5]
            if code_h.shape[0] > 0:
                bank.P.data[new_idx] = (code_h.mean(dim=0)
                                        * math.sqrt(bank.d_r))
    pb1 = d1.train_rows(pack, pairs, args.row_steps, args.row_lr, device,
                        reject_w=args.reject_w, mass_w=args.mass_w,
                        owner_target=args.owner_mass)
    pairs2 = d1.run_in_state(
        model, blocks, gates_delivered, handles, "plugged",
        lambda: d1.collect_gate_pairs(model, site_blocks,
                                      home_train + base_batches,
                                      device, args.max_cal_pairs, label_of))
    pb2 = d1.train_rows(pack, pairs2, max(50, args.row_steps // 2),
                        args.row_lr, device, reject_w=args.reject_w,
                        mass_w=args.mass_w, owner_target=args.owner_mass)
    out["phase_b"] = {"round1_teacher_forced": pb1,
                      "round2_on_trajectory": pb2}
    if args.out:
        d4.save_json(args.out, out)

    # ---- measure ----
    print("MEASURE — five gates + fire/non-fire parity (mass contract)")
    fn_pre = d1.make_logits_fn(model, blocks, gates64, handles, "off", device)
    probes = d1.select_code_probes(fn_pre, probe_windows, args.n_probes)
    out["n_probes"] = len(probes)
    m = measure_all("wrap", model, blocks, pack, handles, gates_delivered,
                    "plugged", gates64, probes, home_held, home_train,
                    per_slice, base_batches, device, new_idx, additive=False,
                    route_blocks=site_blocks)
    m["phase_a"] = pa
    m["wrap_layout"] = layout
    m["core_frozen_bitwise"] = out["core_frozen_bitwise"]
    out["result"] = m
    d1.uninstall_plugin(handles)
    out["wall_s"] = round(time.time() - t0, 2)
    return out


# ---------------------------------------------------------------------------
# Arm (ii) — distill-then-deliver, no refit
# ---------------------------------------------------------------------------


def run_deliver(model, blocks, banks_registry, cores, rows, home_train,
                home_held, per_slice, base_batches, probe_windows, args,
                device) -> dict:
    out = {"gate": "door5-provenance", "arm": "distill-then-deliver-noref",
           "format": FORMAT,
           "recipe": {"delivery": "additive_lambda (contract b)",
                      "expert_refit": False,
                      "lam_fit": {"steps": args.lam_steps,
                                  "lr": args.lam_lr},
                      "rows": {"owner_mass": args.owner_mass,
                               "reject_w": args.reject_w,
                               "mass_w": args.mass_w,
                               "row_steps": args.row_steps}},
           "core_fingerprint_pre": core_fingerprint(cores),
           "started": time.time()}
    t0 = time.time()
    top_k = getattr(blocks[0].gate, "top_k", 2)
    norm_topk = bool(getattr(blocks[0].gate, "norm_topk_prob", False))
    pack = d1.build_pack(blocks, banks_registry, (top_k, norm_topk),
                         inter_mult=1.0, seed=args.seed)
    # warm-start cores + rows from the Door-3 artifact (bit-identical)
    for sd, exp in zip([{"gate_up_proj": w_gu, "down_proj": w_down}
                        for w_gu, w_down in cores], pack.experts):
        exp.load_state_dict(sd)
    new_idx = pack.banks65[0].num_experts - 1
    for bank, row in zip(pack.banks65, rows):
        bank.P.data[new_idx] = row.to(bank.P.device)
    pack.experts.to(device)
    n_params = sum(p.numel() for p in pack.parameters())
    out["trainable_core_params_frozen"] = n_params
    gates64 = [d0.GraftGate(b, top_k, norm_topk, stock_gate=blk.gate)
               for b, blk in zip(banks_registry, blocks)]

    # ---- rows round-1 (teacher-forced hiddens; cores not applied) ----
    print("ROWS round-1 — mass-cal recal (cores untouched)")
    label_of = lambda _i, b: b[1]  # noqa: E731
    handles_off = d1.install_plugin(blocks, pack, "off")
    pairs = d1.run_in_state(
        model, blocks, gates64, handles_off, "off",
        lambda: d1.collect_gate_pairs(model, blocks, home_train + base_batches,
                                      device, args.max_cal_pairs, label_of))
    pb1 = d1.train_rows(pack, pairs, args.row_steps, args.row_lr, device,
                        reject_w=args.reject_w, mass_w=args.mass_w,
                        owner_target=args.owner_mass)
    fn_pre = d1.make_logits_fn(model, blocks, gates64, handles_off, "off",
                               device)
    probes = d1.select_code_probes(fn_pre, probe_windows, args.n_probes)
    out["n_probes"] = len(probes)
    d1.uninstall_plugin(handles_off)
    out["rows_round1"] = pb1
    if args.out:
        d4.save_json(args.out, out)

    # ---- contract (b) install + lambda-only delivery fit ----
    print("LAMBDA FIT — delivery-only (cores + trunk frozen), lr "
          f"{args.lam_lr}")
    lams = nn.ParameterList([nn.Parameter(torch.tensor(args.lam_init))
                             for _ in blocks])
    gates_add = d4.make_additive_gates(blocks, pack)
    hB = d4.install_additive(blocks, pack, lams, mode="off")
    restored = [(blk, d0.install_graft(blk, g))
                for blk, g in zip(blocks, gates_add)]
    fit = train_lambda_only(model, blocks, pack, hB, lams, home_train,
                            args.lam_steps, args.lam_lr, device,
                            grad_clip=args.grad_clip)
    out["lam_fit"] = fit
    lams_final = [float(p.detach().clamp(min=0)) for p in lams]
    out["lambda_per_layer"] = lams_final
    out["lambda_mean"] = round(sum(lams_final) / max(1, len(lams_final)), 4)

    # ---- rows round-2 on-trajectory (additive delivery live, cores frozen) ----
    print("ROWS round-2 — on-trajectory recal under additive delivery")
    pairs2 = d1.run_in_state(
        model, blocks, gates_add, hB, "additive",
        lambda: d1.collect_gate_pairs(model, blocks, home_train + base_batches,
                                      device, args.max_cal_pairs, label_of))
    pb2 = d1.train_rows(pack, pairs2, max(50, args.row_steps // 2),
                        args.row_lr, device, reject_w=args.reject_w,
                        mass_w=args.mass_w, owner_target=args.owner_mass)
    out["rows_round2"] = pb2
    out["core_fingerprint_post"] = core_fingerprint(cores)
    out["core_frozen_bitwise"] = (out["core_fingerprint_pre"]
                                  == out["core_fingerprint_post"])
    # uninstall the additive grafts BEFORE measurement so the masked/stock
    # references are the plain stock + registry-64 states (door4b convention)
    for blk, stock in restored:
        d0.uninstall_graft(blk, stock)
    if args.out:
        d4.save_json(args.out, out)

    # ---- measure ----
    print("MEASURE — five gates + fire/non-fire parity (additive contract)")
    m = measure_all("deliver", model, blocks, pack, hB, gates_add, "additive",
                    gates64, probes, home_held, home_train, per_slice,
                    base_batches, device, new_idx, additive=True, lams=lams)
    m["lam_fit"] = fit
    m["lambda_per_layer"] = lams_final
    m["lambda_mean"] = out["lambda_mean"]
    m["lambda_lam1_control"] = m["s6_quality"].get("lam_fixed_one")
    m["core_frozen_bitwise"] = out["core_frozen_bitwise"]
    out["result"] = m
    d1.uninstall_plugin(hB)
    out["wall_s"] = round(time.time() - t0, 2)
    return out


# ---------------------------------------------------------------------------
# Synthetic self-test (--synthetic): both arms end-to-end on the d0 mocks
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
    home_train = [(t, 1) for t, d in d0.mock_batches(n_dom, seed=11,
                                                     batches_per_domain=2)
                  if d == 1]
    home_held = [(t, 1) for t, d in d0.mock_batches(n_dom, seed=12,
                                                    batches_per_domain=2)
                 if d == 1]
    base_b = [(t, 0) for t, d in d0.mock_batches(n_dom, seed=17,
                                                 batches_per_domain=2)
              if d != 1]
    probe_wins = [t.flatten() for t, d in d0.mock_batches(
        n_dom, seed=23, seq_len=48, batches_per_domain=3, batch_size=1)
        if d == 1]
    per_slice = {"code": home_held}

    class A:
        pass

    args = A()
    args.d_canon = 8
    args.expert_steps = 20
    args.expert_lr = 1e-2
    args.neutral_w = 5.0
    args.grad_clip = 1.0
    args.lam_steps = 15
    args.lam_lr = 5e-2
    args.lam_init = 0.5
    args.row_steps = 60
    args.row_lr = 0.1
    args.reject_w = 5.0
    args.mass_w = 2.0
    args.owner_mass = 0.35
    args.max_cal_pairs = 512
    args.n_probes = 6
    args.out = None
    args.seed = 0

    # mock distilled cores at the pack widths (same formula as d1.build_pack)
    inter = int(blocks[0].experts.gate_up_proj.shape[1] // 2)
    cores = [(torch.randn(2 * inter, h) * 0.05, torch.randn(h, inter) * 0.05)
             for _ in blocks]
    rows = [torch.randn(h) * 0.01 for _ in blocks]
    fp_pre = core_fingerprint(cores)

    out_w = run_wrap(model, blocks, banks_reg, cores, rows, home_train,
                     home_held, per_slice, base_b, probe_wins, args, device)
    rw = out_w["result"]
    check("wrap_core_frozen_bitwise",
          out_w["core_frozen_bitwise"] and fp_pre == out_w["core_fingerprint_pre"],
          f"fp {out_w['core_fingerprint_pre']} == {out_w['core_fingerprint_post_a']}")
    check("wrap_adapters_trained",
          out_w["phase_a"]["history"] and not out_w["phase_a"].get(
              "aborted_nonfinite"),
          f"{out_w['phase_a']['steps']} steps, "
          f"param_l2 {out_w['phase_a']['history'][-1]['param_l2']:.3f}")
    check("wrap_verdict_struct",
          {"quality_heldout_beats_masked", "parity_ge_0.95",
           "retention_ge_0.70", "contamination_le_0.30",
           "utilization_gt_0.50"} <= set(rw["verdict"]["gates"]),
          f"pass={rw['verdict']['pass']}")
    check("wrap_firing_split",
          rw["base_parity"]["firing"]["n"] >= 0
          and "non_firing" in rw["base_parity"],
          f"firing n={rw['base_parity']['firing']['n']} "
          f"non-firing n={rw['base_parity']['non_firing']['n']}")
    check("wrap_train_and_heldout",
          "train" in rw["s6_quality"] and "heldout" in rw["s6_quality"],
          f"heldout {rw['s6_quality']['heldout']['ce_home_active']:.3f} vs "
          f"{rw['s6_quality']['heldout']['ce_home_masked']:.3f}")

    # single-site layout (EXACT Door 2.5): same function, one wrap at layer 0
    args.wrap_layout = "single"
    args.host_layer = 0
    out_s = run_wrap(model, blocks, banks_reg, cores, rows, home_train,
                     home_held, per_slice, base_b, probe_wins, args, device)
    rs = out_s["result"]
    check("wrap_single_site",
          out_s["recipe"]["wrap_layout"] == "single"
          and out_s["core_frozen_bitwise"]
          and not out_s["phase_a"].get("aborted_nonfinite"),
          f"params {out_s['trainable_params']} at layer 0, "
          f"util {rs['utilization_home']['mean_util']:.3f}")
    args.wrap_layout = "all"

    # artifact round-trip for the deliver arm
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), "plugin.pt")
    torch.save({"format": "vesper-graft-plugin-distill-v1",
                "experts": [{"gate_up_proj": w_gu, "down_proj": w_down}
                            for w_gu, w_down in cores],
                "rows": rows, "meta": {"new_idx": 64}}, tmp)
    cores2, rows2, _ = extract_distill_cores(tmp)
    check("artifact_roundtrip",
          core_fingerprint(cores2) == fp_pre and len(rows2) == len(rows),
          f"fp {core_fingerprint(cores2)}")

    out_d = run_deliver(model, blocks, banks_reg, cores2, rows2, home_train,
                        home_held, per_slice, base_b, probe_wins, args, device)
    rd = out_d["result"]
    check("deliver_core_frozen_bitwise",
          out_d["core_frozen_bitwise"] and fp_pre == out_d["core_fingerprint_pre"],
          f"fp {out_d['core_fingerprint_post']}")
    check("deliver_lam_fit_no_refit",
          not out_d["lam_fit"].get("aborted_nonfinite")
          and out_d["core_frozen_bitwise"],
          f"lam_mean {out_d['lambda_mean']}")
    check("deliver_lam1_control", "lam_fixed_one" in rd["s6_quality"],
          f"lam1 heldout {rd['s6_quality']['lam_fixed_one']['heldout']:.3f}")
    check("deliver_verdict_struct",
          {"quality_heldout_beats_masked", "parity_ge_0.95",
           "retention_ge_0.70", "contamination_le_0.30",
           "utilization_gt_0.50"} <= set(rd["verdict"]["gates"]),
          f"pass={rd['verdict']['pass']}")
    check("both_arms_report_same_gates",
          set(rw["verdict"]["gates"]) == set(rd["verdict"]["gates"]),
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
    ap.add_argument("--arm", default="wrap", choices=["wrap", "deliver"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--artifact", default=None,
                    help="Door-3 plugin_distill_olmoe.pt (distilled cores)")
    ap.add_argument("--cache", default=None,
                    help="Door-3 teacher_cache.pt (exact distill window split)")
    ap.add_argument("--domains-dir",
                    default=os.path.join(REPO_ROOT, "lab", "graft_d0_texts"))
    ap.add_argument("--home-domains", default="code,math")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--window-stride", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--max-base-tokens", type=int, default=60000)
    ap.add_argument("--expert-steps", type=int, default=250)
    ap.add_argument("--expert-lr", type=float, default=3e-3,
                    help="Door-2.5 proven LR for thin-adapter-around-frozen-core")
    ap.add_argument("--neutral-w", type=float, default=300.0)
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
    ap.add_argument("--wrap-layout", default="all", choices=["single", "all"],
                    help="single = exact Door-2.5 one-site wrap; all = every "
                         "layer's distilled core wrapped (16 sites)")
    ap.add_argument("--host-layer", type=int, default=6,
                    help="site / distilled-core index for --wrap-layout single")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.synthetic:
        return run_synthetic_tests()
    if not args.model or not args.registry or not args.artifact:
        ap.error("--model, --registry and --artifact are required "
                 "unless --synthetic")

    torch.manual_seed(args.seed)
    t_load = time.time()
    model, tok = d0.load_donor(args.model, args.device, args.dtype)
    blocks = d0.find_moe_blocks(model)
    banks_reg, _ = d0.load_registry(args.registry)
    if len(banks_reg) != len(blocks):
        raise RuntimeError("registry/model layer mismatch")
    print(f"loaded host+registry in {time.time() - t_load:.1f}s "
          f"({len(blocks)} layers)")

    cores, rows, art_meta = extract_distill_cores(args.artifact)
    if len(cores) != len(blocks):
        raise RuntimeError(f"artifact has {len(cores)} cores, model {len(blocks)}")
    print(f"distilled cores: {len(cores)} layers, "
          f"shapes {tuple(cores[0][0].shape)}/{tuple(cores[0][1].shape)}, "
          f"fp {core_fingerprint(cores)}")

    home_names = [d.strip() for d in args.home_domains.split(",") if d.strip()]
    domains = d0.load_domain_texts(args.domains_dir)
    if args.cache and os.path.exists(args.cache):
        train_w, held_w = home_windows_from_cache(
            args.cache, args.seq_len, args.window_stride)
        print(f"home windows from cache {args.cache}")
    else:
        train_w, held_w = home_windows_from_texts(
            domains, tok, home_names, args.seq_len, args.window_stride,
            args.max_tokens)
        print("home windows from domain texts (cache missing)")
    home_train = plain_batches(train_w, args.batch_size, 1)
    home_held = plain_batches(held_w, args.batch_size, 1)
    per_slice = per_slice_batches(held_w, batch_size=1)
    probe_windows = probe_windows_from(held_w) or probe_windows_from(train_w)
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
    if not home_train or not home_held or not base_batches or not probe_windows:
        raise RuntimeError(
            f"not enough data: home_train {len(home_train)} "
            f"home_held {len(home_held)} base {len(base_batches)} "
            f"probes {len(probe_windows)}")
    n_home_tr = sum(t.numel() for t, _ in home_train)
    n_home_he = sum(t.numel() for t, _ in home_held)
    print(f"data: home train {len(home_train)} batches / {n_home_tr} tok, "
          f"held {len(home_held)} batches / {n_home_he} tok, "
          f"base {len(base_batches)} batches, probes {len(probe_windows)}")

    if args.out is None:
        args.out = os.path.join(REPO_ROOT, "lab", "imported", "graft_d0",
                                f"door5_{args.arm}.json")
    out = {"gate": "door5-provenance", "arm": args.arm,
           "artifact_meta": art_meta,
           "host": {"model": args.model, "revision": args.revision},
           "registry": args.registry, "cache": args.cache,
           "home_domains": args.home_domains,
           "n_home_train_tokens": int(n_home_tr),
           "n_home_heldout_tokens": int(n_home_he)}
    d4.save_json(args.out, out)

    if args.arm == "wrap":
        result = run_wrap(model, blocks, banks_reg, cores, rows, home_train,
                          home_held, per_slice, base_batches, probe_windows,
                          args, args.device)
    else:
        result = run_deliver(model, blocks, banks_reg, cores, rows, home_train,
                             home_held, per_slice, base_batches, probe_windows,
                             args, args.device)
    out.update(result)
    out["end_unix"] = time.time()
    d4.save_json(args.out, out)
    print(f"\nresults -> {args.out}")
    m = out["result"]
    q = m["s6_quality"]
    p = m["base_parity"]
    print(f"quality held-out: {q['heldout']['ce_home_active']:.4f} vs masked "
          f"{q['heldout']['ce_home_masked']:.4f}  "
          f"(train {q['train']['ce_home_active']:.4f} vs "
          f"{q['train']['ce_home_masked']:.4f}, forced held "
          f"{q['heldout']['ce_home_forced']:.4f})")
    print(f"parity overall {p['logit_cosine_mean']:.4f}  "
          f"firing {p['firing']['mean_cos']:.4f} (n={p['firing']['n']})  "
          f"non-firing {p['non_firing']['mean_cos']:.4f} "
          f"(n={p['non_firing']['n']})")
    print(f"retention {m['margins']['retention_vs_solo']} "
          f"(degenerate={m['margins']['retention_degenerate']}, "
          f"ce_analog {m['ce_delivery_retention']})  "
          f"contam {m['contamination_base']['mean_util']:.3f}  "
          f"util {m['utilization_home']['mean_util']:.3f}")
    if "lambda_mean" in m:
        print(f"lambda mean {m['lambda_mean']}  lam1_control "
              f"{m.get('lambda_lam1_control')}")
    v = m["verdict"]
    print(f"VERDICT: {'PASS' if v['pass'] else 'FAIL'}  gates={v['gates']}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
