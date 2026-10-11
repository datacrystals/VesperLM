#!/usr/bin/env python3
"""Door 3 — behavioral distillation into a fresh plug-in expert
(MODULAR_MOE.md §10.3; protocol §10.3 + D1 gates §9.4).

Head-to-head counterpart to Door 2.5 (wrap foreign expert cores in the
canonical format): instead of hosting the donor's WEIGHTS behind learned
adapters, train a FRESH OLMoE-graft plug-in expert (the D1 path) to IMITATE
the donor's behavior on the donor skill's home domain.

  teacher   = Phi-mini-MoE (microsoft/Phi-mini-MoE-instruct @ f620b32c,
              lab/graft_d2_crossfamily.py loading recipe — slimmoe->native
              remap + gfx906 shims; vendored here so this harness does not
              depend on a file other lanes are editing)
  host      = OLMoE-1B-7B-0924 @ 6d84c485 + D0 registry graft
              `registry_olmoe_6d84c485.pt`
  home      = codeparrot (code.txt) + open-web-math (math.txt) slices of
              `lab/graft_d0_build_slice.py` — the SAME data files the wrap
              lane uses (fairness contract)
  expert    = per-layer SwiGLU delta pack (graft_d1_plugin.SwiGLUExpert),
              zero-init down_proj, forced dispatch while training
  objective = KL(teacher next-token dist || student dist) on shared-vocab
              support (cross-tokenizer: strings whose single-token decode is
              identical in both vocabs; distributions renormalized over that
              shared set) + optional ground-truth next-token CE + off-domain
              output-norm penalty (neutrality)
  recal     = the D1 mass-calibration recipe that passed the formal bars:
              owner_mass 0.35 / reject_w 10 / out_pen 300 (att-4)

Because the two models use different tokenizers, "KL to donor outputs" is
defined as follows (also recorded in the results JSON):

  * shared support S = { s : exists one Phi id and one OLMoE id that each
    decode to exactly s } (first id wins per vocab for duplicates).
  * positions are aligned by character offsets of the SAME raw text: the soft
    target for student token k is the teacher's next-token distribution at the
    teacher token whose char span covers the START of student token k
    (approximation when a student token starts mid-teacher-token; exact
    boundary alignments are counted and reported).
  * both distributions are log-softmax renormalized over S, then
    loss = KL(teacher_S || student_S) (Hinton direction, log_target).

Teacher outputs are precomputed ONCE (`--phase cache`, VRAM-safe: the 7.65B
teacher and the host are never resident together) and cached as fp16
log-probs; training consumes the cache. Home windows are split 60/40
train/held-out PER DOMAIN so the quality read is not domain-skewed.

Gates (same D1 bars — this is the comparison contract with the wrap lane):
  (a) base parity cos (stock vs plugged on base tokens)     >= 0.95
  (b) retention vs solo (margin gain_lib/gain_solo)          >= 0.70
  (c) contamination (base tokens routed to the plug-in)     <= 0.30
  §6 utilization (home tokens whose top-k includes the row)  >  0.50
  §6 quality (home CE plugged < same masked)                 must hold
  plus the distill-native fidelity read: held-out KL(teacher||student)
  plugged vs masked (the behavioral analogue of §6 quality).

Usage:
  python3 lab/door3_distill_expert.py --synthetic
  python3 lab/door3_distill_expert.py --phase all \
      --teacher /data/tliao/Phi-mini-MoE-instruct \
      --host-model /data/tliao/OLMoE-1B-7B-0924 \
      --registry /data/tliao/graft_d0_out/registry_olmoe_6d84c485.pt \
      --domains-dir /data/tliao/graft_d0_slice \
      --cache /data/tliao/graft_d3_out/teacher_cache.pt \
      --out /data/tliao/graft_d3_out/door3_distill.json \
      --device cuda --dtype float16
"""

from __future__ import annotations

import argparse
import bisect
import gc
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import graft_d0_parity as d0  # noqa: E402  shared passport/metrics/registry core
import graft_d1_plugin as d1  # noqa: E402  D1 plug-in path (pack/merge/recal/gates)

# D1 task bars — identical contract (§9.4 D1 + the wrap-lane comparison)
BAR_BASE_PARITY = d1.BAR_BASE_PARITY
BAR_RETENTION = d1.BAR_RETENTION
BAR_CONTAMINATION = d1.BAR_CONTAMINATION
BAR_UTILIZATION = d1.BAR_UTILIZATION

CACHE_FORMAT = "vesper-door3-teacher-cache-v1"

# mass-calibration recipe that passed D1 (att-4)
DEFAULT_OWNER_MASS = 0.35
DEFAULT_REJECT_W = 10.0
DEFAULT_OUT_PEN = 300.0


# ---------------------------------------------------------------------------
# Shared-vocab distillation support (cross-tokenizer KL)
# ---------------------------------------------------------------------------


def build_shared_vocab(decode_s, decode_t, n_s: int, n_t: int):
    """Strings whose single-token decode matches EXACTLY in both vocabs.

    decode_*(i) -> str for a single id. First id wins within a vocab when
    several ids decode to the same string (duplicates are rare; counted).
    Returns (strings, student_ids list, teacher_ids list, dup stats).
    """
    t_map: dict[str, int] = {}
    t_dupes = 0
    for j in range(n_t):
        s = decode_t(j)
        if s and s not in t_map:
            t_map[s] = j
        elif s:
            t_dupes += 1
    strings, s_ids, t_ids = [], [], []
    s_dupes = 0
    seen: set[str] = set()
    for i in range(n_s):
        s = decode_s(i)
        if not s:
            continue
        if s in seen:
            s_dupes += 1
            continue
        if s in t_map:
            strings.append(s)
            s_ids.append(i)
            t_ids.append(t_map[s])
            seen.add(s)
    return strings, s_ids, t_ids, {"teacher_dup_strings": t_dupes,
                                   "student_dup_strings": s_dupes}


def align_targets(student_spans, teacher_spans) -> list[int]:
    """align[k] = teacher-grid row of the soft target for student token k.

    The soft target for student token k is the teacher's next-token
    distribution at the teacher token whose char span covers the START of
    student token k (that teacher token is what that teacher row predicts).
    -1 = unaligned (empty span / not covered). Token 0 always -1 (no context).
    """
    t_starts = [sp[0] for sp in teacher_spans]
    align = [-1] * len(student_spans)
    for k, (a, b) in enumerate(student_spans):
        if k == 0 or b <= a:
            continue
        j = bisect.bisect_right(t_starts, a) - 1
        if j >= 0 and teacher_spans[j][1] > a:
            align[k] = j
    return align


def alignment_stats(align, student_spans, teacher_spans) -> dict:
    n = sum(1 for j in align if j >= 0)
    exact = 0
    for k, j in enumerate(align):
        if j >= 0 and teacher_spans[j][0] == student_spans[k][0]:
            exact += 1
    return {"student_tokens": len(align), "aligned": n,
            "exact_boundary": exact,
            "exact_frac": round(exact / max(1, n), 4)}


# ---------------------------------------------------------------------------
# Teacher cache (offline soft targets on the student token grid)
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_teacher_cache(teacher, teacher_ids, student_ids, student_spans,
                        teacher_spans, shared_t_ids, device,
                        window: int = 512, stride: int = 256) -> dict:
    """Run the teacher over `teacher_ids` in sliding windows; store, for every
    teacher-grid row j (the distribution PREDICTING teacher token j), the
    log-softmax over the shared string support S.

    Windows: size `window`, hop `stride` (default window/2). Every local
    position is computed; a row is stored the first time a window covers it
    (first-fill-wins = the covering window with the longest in-window
    context). Row 0 is never valid (nothing before the first token).
    """
    m = len(teacher_ids)
    S = len(shared_t_ids)
    if m < 2 or S == 0:
        raise RuntimeError("teacher cache: empty ids or empty shared vocab")
    shared_idx = torch.tensor(shared_t_ids, dtype=torch.long, device=device)
    logp = torch.zeros(m + 1, S, dtype=torch.float16)
    valid = torch.zeros(m + 1, dtype=torch.bool)
    t_all = torch.tensor(teacher_ids, dtype=torch.long)
    n_fwd = 0
    t0 = time.time()
    for a in range(0, m, stride):
        w = min(window, m - a)
        if w < 2:
            break
        chunk = t_all[a:a + w].to(device)
        logits = d0.model_logits(teacher, chunk.unsqueeze(0))[0]  # (w, V)
        # local position t predicts global teacher token index a+t+1.
        # All local positions are computed; first-fill-wins below keeps each
        # row's EARLIEST covering window (longest in-window context). The
        # naive "second half only" shortcut drops the tail window's rows when
        # w < window/2 (short last window -> empty arange).
        lo = 0
        rows = torch.arange(a + lo + 1, a + w + 1)
        keep = rows < m  # predicting past end-of-text is out of scope
        if not bool(keep.any()):
            continue
        sub = logits[lo:][keep].float().index_select(-1, shared_idx)
        lp = F.log_softmax(sub, dim=-1).to(torch.float16).cpu()
        rows_cpu = rows[keep].cpu()
        fresh = ~valid[rows_cpu]
        if bool(fresh.any()):
            rr = rows_cpu[fresh]
            logp[rr] = lp[fresh]
            valid[rr] = True
        n_fwd += 1
    cover = float(valid[1:m].float().mean()) if m > 1 else 0.0
    align = align_targets(student_spans, teacher_spans)
    return {
        "student_ids": torch.tensor(student_ids, dtype=torch.long),
        "student_spans": torch.tensor(student_spans, dtype=torch.long),
        "align": torch.tensor(align, dtype=torch.long),
        "teacher_logp": logp,
        "teacher_rows_valid": valid,
        "meta": {
            "teacher_tokens": m, "student_tokens": len(student_ids),
            "window": window, "stride": stride,
            "n_forwards": n_fwd, "row_coverage": round(cover, 4),
            "wall_s": round(time.time() - t0, 2),
            "align": alignment_stats(align, student_spans, teacher_spans),
        },
    }


def save_cache(cache: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(cache, tmp)
    os.replace(tmp, path)


def load_cache(path: str) -> dict:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("format") != CACHE_FORMAT:
        raise ValueError(f"unknown teacher-cache format {cache.get('format')}")
    return cache


# ---------------------------------------------------------------------------
# Distillation batches (student windows + per-position soft targets)
# ---------------------------------------------------------------------------


@dataclass
class DistillBatch:
    ids: torch.Tensor        # (B, L) student tokens
    tgt: torch.Tensor        # (B, L-1, S) teacher log-probs for targets 1..L-1
    mask: torch.Tensor       # (B, L-1) bool — aligned rows
    domain: str = "home"


def make_windows(cache: dict, seq_len: int, stride: int, max_windows: int,
                 split_frac: float = 0.6):
    """Per-domain student windows, split train/held-out inside each domain.

    Each window: {"ids": (L,), "tgt": (L-1, S), "mask": (L-1,), "domain",
    "start"} where tgt[t] is the soft target for student token (start+t+1) —
    i.e. what logits[t] should predict. Returns (train_wins, held_wins) pooled
    over domains in domain order.
    """
    train, held = [], []
    tlp = None
    for name, c in cache["domains"].items():
        sid = c["student_ids"]
        align = c["align"]
        tlp = c["teacher_logp"]
        n = sid.shape[0]
        wins = []
        for start in range(0, n - seq_len, stride):
            ids = sid[start:start + seq_len]
            rows = align[start + 1:start + seq_len]  # targets of logits 0..L-2
            mask = rows >= 0
            tgt = torch.zeros(rows.shape[0], tlp.shape[1], dtype=torch.float16)
            if bool(mask.any()):
                tgt[mask] = tlp[rows[mask]]
            wins.append({"ids": ids, "tgt": tgt, "mask": mask,
                         "start": start, "domain": name})
            if max_windows and len(wins) >= max_windows:
                break
        if not wins:
            raise RuntimeError(f"not enough student tokens in domain {name}")
        k = max(1, int(split_frac * len(wins)))
        train.extend(wins[:k])
        held.extend(wins[k:] if len(wins) > k else wins[:1])
    return train, held


def batch_windows(wins, batch_size: int, domain: str = "home") -> list:
    out = []
    for i in range(0, len(wins) - batch_size + 1, batch_size):
        rows = wins[i:i + batch_size]
        out.append(DistillBatch(
            ids=torch.stack([w["ids"] for w in rows]),
            tgt=torch.stack([w["tgt"] for w in rows]),
            mask=torch.stack([w["mask"] for w in rows]),
            domain=domain))
    return out


def plain_batches(wins, batch_size: int, label: int) -> list:
    """(tokens (B,L), label) batches — the D1 recal/measurement shape."""
    out = []
    for i in range(0, len(wins) - batch_size + 1, batch_size):
        rows = wins[i:i + batch_size]
        out.append((torch.stack([w["ids"] for w in rows]), label))
    return out


def probe_windows_from(wins, chunk: int = 256) -> list:
    """256-token flat windows for the delta_margin probe selection."""
    out = []
    for w in wins:
        ids = w["ids"]
        for i in range(0, ids.shape[0] - chunk + 1, chunk):
            out.append(ids[i:i + chunk])
    return out


# ---------------------------------------------------------------------------
# Phase A — expert fit by behavioral distillation (forced dispatch)
# ---------------------------------------------------------------------------


def distill_kl(logits: torch.Tensor, batch: DistillBatch,
               shared_s_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """KL(teacher_S || student_S) over the shared support, per position.

    logits (B, L, V) -> gather shared columns of the next-token logits
    (positions 0..L-2 predict tokens 1..L-1), log-softmax over S, then
    F.kl_div(student_logp, teacher_logp, log_target=True) summed over S.
    Returns (mean_kl, n_positions).
    """
    b, l, _ = logits.shape
    sub = torch.gather(
        logits[:, :-1].float(), 2,
        shared_s_ids.view(1, 1, -1).expand(b, l - 1, -1))
    s_logp = F.log_softmax(sub, dim=-1)
    kl = F.kl_div(s_logp, batch.tgt.float().to(logits.device),
                  reduction="none", log_target=True).sum(dim=-1)
    mask = batch.mask.to(kl.device)
    n = int(mask.sum())
    if n == 0:
        return torch.zeros((), device=logits.device), 0
    return (kl * mask).sum() / n, n


def train_expert_distill(model, blocks, pack, handles, home_batches,
                         base_batches, steps, lr, device,
                         shared_s_ids: torch.Tensor,
                         kl_coef: float = 1.0, ce_coef: float = 0.3,
                         out_pen: float = DEFAULT_OUT_PEN,
                         grad_clip: float = 1.0,
                         weight_decay: float = 0.01) -> dict:
    """Forced-dispatch fit of the fresh expert to the donor's behavior.

    Loss = kl_coef * KL(teacher_S || student_S)   (home texts, aligned)
         + ce_coef * next-token CE                (home texts, ground truth)
         + out_pen * mean_l ||e_new(h_base)||^2   (off-domain neutrality;
           the D1 att-2/3/4 lesson: trunk text-KL is unstable at 7B scale)
    Trunk frozen; gradients reach only the new expert modules.
    """
    params = list(pack.parameters())
    d1.freeze_all_except(model, params)
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    hist = []
    t0 = time.time()
    for step in range(steps):
        batch = home_batches[step % len(home_batches)]
        ids = batch.ids.to(device)
        d1.set_plugin_mode(handles, "forced", 1.0)
        logits = d0.model_logits(model, ids)
        kl, n_pos = distill_kl(logits, batch, shared_s_ids)
        v = logits.shape[-1]
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, v).float(),
                             ids[:, 1:].reshape(-1))
        bt, _ = base_batches[step % len(base_batches)]
        d1.set_plugin_mode(handles, "forced", 1.0)
        d0.model_logits(model, bt.to(device))
        out_terms = [blk._plugin_merge.last_new_out for blk, _ in handles
                     if getattr(blk, "_plugin_merge", None) is not None
                     and blk._plugin_merge.last_new_out is not None]
        out_norm2 = (torch.stack([t.pow(2).mean() for t in out_terms]).mean()
                     if out_terms else torch.zeros((), device=device))
        total = kl_coef * kl + ce_coef * ce + out_pen * out_norm2
        opt.zero_grad()
        total.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            hist.append({"step": step,
                         "kl": float(kl.detach()),
                         "ce_home": float(ce.detach()),
                         "out_norm2": float(out_norm2.detach()),
                         "n_pos": n_pos})
            print(f"  [phaseA {step:4d}] kl {float(kl.detach()):.4f}  "
                  f"ce_home {float(ce.detach()):.4f}  "
                  f"out_norm2 {float(out_norm2.detach()):.6f}")
    return {"steps": steps, "lr": lr, "kl_coef": kl_coef, "ce_coef": ce_coef,
            "out_pen": out_pen, "grad_clip": grad_clip, "history": hist,
            "wall_s": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------
# Distillation fidelity read (held-out KL in each delivery state)
# ---------------------------------------------------------------------------


@torch.no_grad()
def distill_fidelity(model, blocks, batches, device, shared_s_ids,
                     gates=None, handles=None, mode="off") -> dict:
    tot, n = 0.0, 0
    ce_tot = 0.0
    n_rows = 0

    def one():
        nonlocal tot, n, ce_tot, n_rows
        for batch in batches:
            ids = batch.ids.to(device)
            lg = d0.model_logits(model, ids)
            kl, n_pos = distill_kl(lg, batch, shared_s_ids)
            tot += float(kl) * n_pos
            n += n_pos
            v = lg.shape[-1]
            ce_tot += F.cross_entropy(
                lg[:, :-1].reshape(-1, v).float(),
                ids[:, 1:].reshape(-1)).item() * ids.shape[0]
            n_rows += ids.shape[0]

    if gates is None and (handles is None or mode == "off"):
        one()
    else:
        d1.run_in_state(model, blocks, gates, handles, mode, one)
    return {"kl_teacher_student": round(tot / max(1, n), 5),
            "n_pos": n,
            "ce_home": round(ce_tot / max(1, n_rows), 5)}


# ---------------------------------------------------------------------------
# Donor (Phi-mini-MoE) loading — vendored from lab/graft_d2_crossfamily.py
# (committed version) so Door 3 does not depend on that file while the wrap
# lane edits it. slimmoe checkpoint -> native transformers phimoe.
# ---------------------------------------------------------------------------

SLIMMOE_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$")


def remap_slimmoe_checkpoint(src: str, dst: str) -> str:
    """Convert slimmoe-layout weights to native phimoe layout (idempotent)."""
    from safetensors.torch import load_file, save_file
    if os.path.exists(os.path.join(dst, "model.safetensors")):
        return dst
    os.makedirs(dst, exist_ok=True)
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["architectures"] = ["PhimoeForCausalLM"]
    cfg.pop("auto_map", None)
    cfg["torch_dtype"] = "float16"
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    for fn in ("tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "generation_config.json"):
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
                layer, expert, which = (int(m.group(1)), int(m.group(2)),
                                        m.group(3))
                packs.setdefault((layer, expert), {})[which] = v.contiguous()
            elif k.endswith("block_sparse_moe.gate.weight"):
                layer = k.split(".")[2]
                out[f"model.layers.{layer}.mlp.router.weight"] = v.contiguous()
            else:
                out[k] = v.contiguous()
        del sd
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


def load_teacher(model_path: str, device: str, dtype_name: str):
    """Load the Phi-mini-MoE donor (auto-remap slimmoe layout)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"float32": torch.float32,
             "float16": torch.float16,
             "bfloat16": torch.bfloat16}[dtype_name]
    status = d0.patch_rocm_grouped_mm()
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
    n_patched = d0.patch_rocm_experts(model)
    if n_patched:
        print(f"rocm experts shim: patched {n_patched} expert banks")
    return model, tok


# ---------------------------------------------------------------------------
# Data preparation (home = codeparrot + open-web-math slices)
# ---------------------------------------------------------------------------


def encode_with_spans(tok, text: str, max_tokens: int):
    """(ids, spans) with char offsets; whole-string encode (no chunking so
    offsets stay exact); capped at max_tokens tokens."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = list(enc["input_ids"])[:max_tokens]
    spans = [tuple(s) for s in enc["offset_mapping"]][:max_tokens]
    return ids, spans


# ---------------------------------------------------------------------------
# D3 pipeline
# ---------------------------------------------------------------------------


def dump_json(out: dict, path: str | None) -> None:
    """Incremental results write — a killed run still leaves usable JSON."""
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2, default=str)
    os.replace(tmp, path)


def phase_cache(args, out: dict) -> dict:
    """Teacher-side precompute: shared vocab + soft targets on the student grid.
    VRAM: only the 7.65B teacher is resident."""
    t0 = time.time()
    from transformers import AutoTokenizer
    teacher, t_tok = load_teacher(args.teacher, args.device, args.dtype)
    try:
        s_tok = AutoTokenizer.from_pretrained(args.host_model)
        domains = d0.load_domain_texts(args.domains_dir)
        home_names = [d.strip() for d in args.home_domains.split(",") if d.strip()]
        missing = [d for d in home_names if d not in domains]
        if missing:
            raise RuntimeError(f"home domains missing from {args.domains_dir}: "
                               f"{missing}")
        print(f"building shared vocab ({s_tok.vocab_size} student / "
              f"{t_tok.vocab_size} teacher ids)...")
        strings, s_ids, t_ids, dup = build_shared_vocab(
            lambda i: s_tok.decode([i]), lambda i: t_tok.decode([i]),
            s_tok.vocab_size, t_tok.vocab_size)
        print(f"shared support S = {len(strings)} strings ({dup})")

        per_domain = {}
        for name in home_names:
            text = domains[name]
            s_ids_d, s_spans = encode_with_spans(s_tok, text, args.max_tokens)
            t_ids_d, t_spans = encode_with_spans(t_tok, text, 10 ** 9)
            print(f"[cache {name}] student {len(s_ids_d)} tok / teacher "
                  f"{len(t_ids_d)} tok")
            c = build_teacher_cache(teacher, t_ids_d, s_ids_d, s_spans,
                                    t_spans, t_ids, args.device,
                                    window=args.teacher_window,
                                    stride=args.teacher_stride)
            c["meta"]["domain"] = name
            per_domain[name] = c
        cache = {
            "format": CACHE_FORMAT,
            "shared_s_ids": torch.tensor(s_ids, dtype=torch.long),
            "shared_t_ids": torch.tensor(t_ids, dtype=torch.long),
            "domains": per_domain,
            "meta": {
                "home_domains": home_names,
                "shared_size": len(strings),
                "shared_strings_sample": strings[:64],
                "dup": dup,
                "teacher": {"path": args.teacher,
                            "revision": args.teacher_revision},
                "host_tokenizer": args.host_model,
                "build_wall_s": round(time.time() - t0, 2),
            },
        }
        save_cache(cache, args.cache)
        out["teacher_cache"] = {"path": args.cache, **cache["meta"],
                                "per_domain": {k: v["meta"]
                                               for k, v in per_domain.items()}}
        dump_json(out, args.out)
        print(f"teacher cache -> {args.cache}  S={len(strings)}  "
              f"wall {cache['meta']['build_wall_s']}s")
        return cache
    finally:
        # drop OUR reference (a helper's `del` of its parameter would not),
        # then release the caching allocator so the host can reuse the pool
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_d3(model, blocks, banks_registry, cache, home_train, home_held,
           home_plain_train, home_plain_held, probe_windows,
           base_batches, per_slice_plain, args, device) -> dict:
    """Full Door 3: distill fit -> row recal (mass-cal recipe) -> D1 gates."""
    out = {"gate": "D1 (Door 3 distill)",
           "bars": {"base_parity": BAR_BASE_PARITY,
                    "retention": BAR_RETENTION,
                    "contamination": BAR_CONTAMINATION,
                    "utilization_s6": BAR_UTILIZATION}}
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
    out["n_experts_stock"] = banks_registry[0].num_experts
    out["new_expert_idx"] = new_idx
    shared_s_ids = cache["shared_s_ids"].to(device)

    # ---- Phase A: distill fit (forced dispatch, KL to donor) ----
    print("PHASE A — expert fit (behavioral distillation, forced dispatch)")
    d1.set_plugin_mode(handles, "off")
    pa = train_expert_distill(
        model, blocks, pack, handles, home_train, base_batches,
        args.expert_steps, args.expert_lr, device, shared_s_ids,
        kl_coef=args.kl_coef, ce_coef=args.ce_coef, out_pen=args.out_pen,
        grad_clip=args.grad_clip)
    out["phase_a"] = pa
    dump_json(out, args.out)

    # ---- Phase B: passport row recal (D1 mass-cal recipe) ----
    print("PHASE B — passport row recal (mass-cal recipe "
          f"owner_mass {args.owner_mass} / reject_w {args.reject_w})")
    home_and_base = home_plain_train + base_batches
    label_of = lambda i, b: b[1] if isinstance(b[1], (int, float)) else 0  # noqa: E731
    pairs = d1.run_in_state(
        model, blocks, gates64, handles, "off",
        lambda: d1.collect_gate_pairs(model, blocks, home_and_base, device,
                                      args.max_cal_pairs, label_of))
    with torch.no_grad():
        for l, bank in enumerate(pack.banks65):
            h, lab = pairs[l]
            home_h = h[lab > 0.5]
            if home_h.shape[0] > 0:
                bank.P.data[new_idx] = home_h.mean(dim=0) * math.sqrt(bank.d_r)
    pb1 = d1.train_rows(pack, pairs, args.row_steps, args.row_lr, device,
                        reject_w=args.reject_w, mass_w=args.mass_w,
                        owner_target=args.owner_mass)
    pairs2 = d1.run_in_state(
        model, blocks, pack.graft_gates, handles, "plugged",
        lambda: d1.collect_gate_pairs(model, blocks, home_and_base, device,
                                      args.max_cal_pairs, label_of))
    pb2 = d1.train_rows(pack, pairs2, max(50, args.row_steps // 2),
                        args.row_lr, device, reject_w=args.reject_w,
                        mass_w=args.mass_w, owner_target=args.owner_mass)
    out["phase_b"] = {"round1_teacher_forced": pb1,
                      "round2_on_trajectory": pb2}
    dump_json(out, args.out)

    # ---- Phase C: D1 gates + distill fidelity ----
    print("PHASE C — measurement (parity / retention / contamination / quality)")
    base_parity = d1.run_in_state(
        model, blocks, None, handles, "off",
        lambda: d1.parity_vs_stock(model, blocks, base_batches, device,
                                   pack.graft_gates, handles))
    out["base_parity"] = base_parity

    fn_pre = d1.make_logits_fn(model, blocks, gates64, handles, "off", device)
    fn_solo = d1.make_logits_fn(model, blocks, gates64, handles, "forced",
                                device, force_gain=1.0)
    fn_lib = d1.make_logits_fn(model, blocks, pack.graft_gates, handles,
                               "plugged", device)
    probes = d1.select_code_probes(fn_pre, probe_windows, args.n_probes)
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

    contam = d1.run_in_state(model, blocks, pack.graft_gates, handles, "plugged",
                             lambda: d1.routing_hits(model, blocks, base_batches,
                                                     device, new_idx))
    util = d1.run_in_state(model, blocks, pack.graft_gates, handles, "plugged",
                           lambda: d1.routing_hits(model, blocks,
                                                   home_plain_train, device,
                                                   new_idx))
    mass_base = d1.run_in_state(
        model, blocks, pack.graft_gates, handles, "plugged",
        lambda: d1.routing_mass(model, blocks, base_batches, device, new_idx))
    mass_home = d1.run_in_state(
        model, blocks, pack.graft_gates, handles, "plugged",
        lambda: d1.routing_mass(model, blocks, home_plain_train, device,
                                new_idx))
    out["contamination_base"] = contam
    out["utilization_home"] = util
    out["routing_mass"] = {"base": mass_base, "home": mass_home}

    # §6 quality: home CE plugged vs masked — HELD-OUT (primary) + train
    ce_masked = d1.ce_in_state(model, blocks, home_plain_held, device, gates64,
                               handles, "off")
    ce_active = d1.ce_in_state(model, blocks, home_plain_held, device,
                               pack.graft_gates, handles, "plugged")
    ce_masked_tr = d1.ce_in_state(model, blocks, home_plain_train, device,
                                  gates64, handles, "off")
    ce_active_tr = d1.ce_in_state(model, blocks, home_plain_train, device,
                                  pack.graft_gates, handles, "plugged")
    out["s6_quality"] = {
        "heldout": {"ce_home_masked": ce_masked, "ce_home_active": ce_active,
                    "delta": ce_active - ce_masked},
        "train": {"ce_home_masked": ce_masked_tr,
                  "ce_home_active": ce_active_tr,
                  "delta": ce_active_tr - ce_masked_tr}}

    # per-slice quality (code / math separately — wrap-lane alignment)
    slice_q = {}
    for name, batches in per_slice_plain.items():
        if not batches:
            continue
        m = d1.ce_in_state(model, blocks, batches, device, gates64, handles,
                           "off")
        a = d1.ce_in_state(model, blocks, batches, device, pack.graft_gates,
                           handles, "plugged")
        slice_q[name] = {"ce_masked": m, "ce_active": a, "delta": a - m}
    out["s6_quality_per_slice"] = slice_q

    # base no-regression
    ce_base_masked = d1.ce_in_state(model, blocks, base_batches, device,
                                    gates64, handles, "off")
    ce_base_active = d1.ce_in_state(model, blocks, base_batches, device,
                                    pack.graft_gates, handles, "plugged")
    out["base_ce"] = {"masked": ce_base_masked, "plugged": ce_base_active,
                      "delta": ce_base_active - ce_base_masked}

    # distillation fidelity: held-out KL(teacher||student) by delivery state
    fid_off = distill_fidelity(model, blocks, home_held, device, shared_s_ids,
                               gates64, handles, "off")
    fid_solo = distill_fidelity(model, blocks, home_held, device, shared_s_ids,
                                gates64, handles, "forced")
    fid_lib = distill_fidelity(model, blocks, home_held, device, shared_s_ids,
                               pack.graft_gates, handles, "plugged")
    out["distill_fidelity"] = {"masked": fid_off, "solo": fid_solo,
                               "plugged": fid_lib}

    out["verdict"] = d1.d1_verdict(base_parity, ret, contam, util,
                                   ce_active, ce_masked)
    out["wall_s"] = round(time.time() - t0, 2)

    if args.plugin_out:
        torch.save({
            "format": "vesper-graft-plugin-distill-v1",
            "experts": [e.state_dict() for e in pack.experts],
            "rows": [b.P.data[new_idx].cpu() for b in pack.banks65],
            "meta": {"new_idx": new_idx,
                     "home_domains": args.home_domains,
                     "objective": "kl_teacher_student",
                     "host_registry": args.registry,
                     "teacher": args.teacher},
        }, args.plugin_out)
        out["plugin_path"] = args.plugin_out
    return out


# ---------------------------------------------------------------------------
# Synthetic self-test (--synthetic): distillation machinery on the d0 mocks
# ---------------------------------------------------------------------------


class FakeTok:
    """Toy tokenizer: id i decodes to its string; used for shared-vocab tests."""

    def __init__(self, strings):
        self.strings = list(strings)

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self.strings[i] for i in ids)


def run_synthetic_tests() -> int:
    torch.manual_seed(0)
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" — {detail}" if detail else ""))

    # 1. alignment on hand-built spans
    s_spans = [(0, 2), (2, 5), (5, 6), (6, 10), (10, 10)]
    t_spans = [(0, 3), (3, 6), (6, 7), (7, 10)]
    al = align_targets(s_spans, t_spans)
    check("align_token0_unaligned", al[0] == -1, f"{al}")
    check("align_covering_token", al[1] == 0 and al[2] == 1 and al[3] == 2,
          f"{al}")
    check("align_empty_span", al[4] == -1)
    st = alignment_stats(al, s_spans, t_spans)
    check("align_stats", st["aligned"] == 3 and st["exact_boundary"] == 1,
          f"{st}")

    # 2. shared vocab across two fake tokenizers
    ft = FakeTok(["a", "bb", "cc", "zz", "a"])          # dupe string "a"
    fs = FakeTok(["a", "bb", "xx", "cc", ""])
    strings, s_ids, t_ids, dup = build_shared_vocab(
        fs.decode, ft.decode, len(fs.strings), len(ft.strings))
    check("shared_vocab_set", set(strings) == {"a", "bb", "cc"}, f"{strings}")
    check("shared_vocab_first_id", s_ids[0] == 0 and t_ids[0] == 0,
          f"{s_ids} {dup}")

    # 3. teacher cache on the mock MoE (toy grid = identity alignment)
    device = "cpu"
    teacher, _ = d0.build_mock_donor(seed=7)
    teacher.eval()
    L = 64
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, 90, (L,), generator=g).tolist()
    spans = [(3 * i, 3 * i + 3) for i in range(L)]
    shared_t = list(range(90))
    cache_d = build_teacher_cache(teacher, ids, ids, spans, spans, shared_t,
                                  device, window=16, stride=8)
    check("cache_rows", cache_d["teacher_logp"].shape == (L + 1, 90),
          f"{tuple(cache_d['teacher_logp'].shape)}")
    check("cache_valid_rows", bool(cache_d["teacher_rows_valid"][1:L].all()),
          f"coverage {cache_d['meta']['row_coverage']}")
    lp = cache_d["teacher_logp"][1:L].float()
    lse = lp.logsumexp(dim=-1)
    check("cache_logp_normalized", bool((lse.abs() < 1e-3).all()),
          f"max |logsumexp| {float(lse.abs().max()):.2e}")
    check("cache_align_identity",
          all(int(cache_d["align"][k]) == k for k in range(1, L)),
          f"{[int(x) for x in cache_d['align'][:6]]}")

    cache = {"format": CACHE_FORMAT,
             "shared_s_ids": torch.tensor(shared_t, dtype=torch.long),
             "shared_t_ids": torch.tensor(shared_t, dtype=torch.long),
             "domains": {"code": cache_d}, "meta": {"home_domains": ["code"]}}

    # 4. distill batches + KL decreasing under forced-dispatch training
    train_w, held_w = make_windows(cache, seq_len=16, stride=8,
                                   max_windows=8, split_frac=0.75)
    hb = batch_windows(train_w, batch_size=2)
    check("distill_batches", len(hb) > 0 and hb[0].ids.shape == (2, 16)
          and hb[0].tgt.shape == (2, 15, 90) and bool(hb[0].mask.all()),
          f"{len(hb)} batches, held {len(held_w)}")

    student, _ = d0.build_mock_donor(seed=11)  # different init = other behavior
    sblocks = d0.find_moe_blocks(student)
    student.eval()
    e, h = sblocks[0].gate.weight.shape
    top_k, norm_topk = sblocks[0].gate.top_k, sblocks[0].gate.norm_topk_prob
    banks_reg = [d0.PassportBank(e, h) for _ in sblocks]
    d0.init_passports_exact(banks_reg, [b.gate.weight for b in sblocks])
    pack = d1.build_pack(sblocks, banks_reg, (top_k, norm_topk))
    handles = d1.install_plugin(sblocks, pack, "off")
    base_b = [(t, 0) for t, d in d0.mock_batches(3, seed=17, batches_per_domain=1)
              if d != 1]
    shared_s = torch.tensor(shared_t, dtype=torch.long)
    gates64 = [d0.GraftGate(b, top_k, norm_topk, stock_gate=blk.gate)
               for b, blk in zip(banks_reg, sblocks)]

    class A:
        pass
    a = A()
    a.expert_steps = 40
    a.expert_lr = 1e-2
    a.kl_coef = 1.0
    a.ce_coef = 0.3
    a.out_pen = 5.0
    a.grad_clip = 1.0
    a.row_steps = 60
    a.row_lr = 0.1
    a.reject_w = 10.0
    a.mass_w = 2.0
    a.owner_mass = 0.35
    a.max_cal_pairs = 512
    a.n_probes = 8
    a.inter_mult = 1.0
    a.seed = 0
    a.registry = "selftest"
    a.plugin_out = None
    a.out = None
    a.home_domains = "code,math"
    a.teacher = "mock-teacher"

    kl0 = distill_fidelity(student, sblocks, hb[:1], device, shared_s,
                           gates64, handles, "off")["kl_teacher_student"]
    pa = train_expert_distill(student, sblocks, pack, handles, hb, base_b,
                              a.expert_steps, a.expert_lr, device, shared_s,
                              kl_coef=a.kl_coef, ce_coef=a.ce_coef,
                              out_pen=a.out_pen, grad_clip=a.grad_clip)
    kl1 = pa["history"][-1]["kl"]
    check("distill_kl_decreases", kl1 < kl0 + 0.05,
          f"kl {kl0:.4f} -> train-end {kl1:.4f}")
    check("phase_a_logs", "out_norm2" in pa["history"][0],
          f"{pa['history'][0]}")

    # 5. full run_d3 structure on the mocks
    home_plain = plain_batches(train_w, 2, 1) + plain_batches(held_w, 1, 1)
    probe_wins = probe_windows_from(held_w, chunk=12)  # mock seq_len=16
    out = run_d3(student, sblocks, banks_reg, cache, hb, hb, home_plain,
                 home_plain, probe_wins, base_b, {"code": home_plain}, a, device)
    check("verdict_struct", set(out["verdict"]["gates"]) == {
        "base_parity_cos_ge_0.95", "retention_vs_solo_ge_0.70",
        "contamination_le_0.30", "s6_utilization_gt_0.50",
        "s6_quality_ce_beats_masked"}, f"pass={out['verdict']['pass']}")
    check("quality_keys", set(out["s6_quality"]) == {"heldout", "train"}
          and "ce_home_masked" in out["s6_quality"]["heldout"])
    check("fidelity_keys", set(out["distill_fidelity"]) == {"masked", "solo",
                                                            "plugged"})
    check("margins_computed", out["margins"]["retention_vs_solo"] is not None
          or out["margins"]["gain_solo"] == 0.0,
          f"retention {out['margins']['retention_vs_solo']}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} synthetic checks passed")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--phase", default="all", choices=["cache", "train", "all"])
    ap.add_argument("--teacher", default=None,
                    help="Phi-mini-MoE dir (slimmoe or native)")
    ap.add_argument("--teacher-revision", default=None)
    ap.add_argument("--host-model", default=None, help="OLMoE dir")
    ap.add_argument("--host-revision", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--domains-dir",
                    default=os.path.join(REPO_ROOT, "lab", "graft_d0_texts"))
    ap.add_argument("--home-domains", default="code,math",
                    help="comma-separated slice names that form the home "
                         "(target) domain — codeparrot/open-web-math")
    ap.add_argument("--cache", default=None, help="teacher cache .pt")
    ap.add_argument("--out", default=None, help="results JSON (incremental)")
    ap.add_argument("--plugin-out", default=None,
                    help="save trained expert+rows .pt (untracked)")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--window-stride", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=20000,
                    help="per home-domain token cap")
    ap.add_argument("--max-base-tokens", type=int, default=60000)
    ap.add_argument("--teacher-window", type=int, default=512)
    ap.add_argument("--teacher-stride", type=int, default=256)
    ap.add_argument("--max-windows", type=int, default=0,
                    help="cap on home windows per domain (0 = all)")
    ap.add_argument("--expert-steps", type=int, default=200)
    ap.add_argument("--expert-lr", type=float, default=1e-4)
    ap.add_argument("--kl-coef", type=float, default=1.0)
    ap.add_argument("--ce-coef", type=float, default=0.3,
                    help="ground-truth CE weight (0 = pure distillation)")
    ap.add_argument("--out-pen", type=float, default=DEFAULT_OUT_PEN,
                    help="off-domain output-norm penalty (D1 att-4: 300)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--row-steps", type=int, default=400)
    ap.add_argument("--row-lr", type=float, default=0.05)
    ap.add_argument("--reject-w", type=float, default=DEFAULT_REJECT_W)
    ap.add_argument("--mass-w", type=float, default=2.0)
    ap.add_argument("--owner-mass", type=float, default=DEFAULT_OWNER_MASS,
                    help="home routing-mass target (D1 att-4: 0.35)")
    ap.add_argument("--max-cal-pairs", type=int, default=4096)
    ap.add_argument("--n-probes", type=int, default=32)
    ap.add_argument("--inter-mult", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.synthetic:
        return run_synthetic_tests()

    if args.phase in ("cache", "all"):
        if not args.teacher or not args.host_model:
            ap.error("--teacher and --host-model are required for --phase cache")
    if args.phase in ("train", "all"):
        if not args.host_model or not args.registry:
            ap.error("--host-model and --registry are required for --phase train")
    if args.cache is None:
        args.cache = os.path.join(REPO_ROOT, "lab", "imported", "graft_d3",
                                  "teacher_cache.pt")
    if args.out is None:
        args.out = os.path.join(REPO_ROOT, "lab", "imported", "graft_d3",
                                "door3_distill.json")
    if args.plugin_out is None:
        args.plugin_out = os.path.join(REPO_ROOT, "lab", "imported", "graft_d3",
                                       "plugin_distill_olmoe.pt")

    out = {"gate": "D1 (Door 3 behavioral distillation)",
           "phase": args.phase,
           "recipe": {
               "objective": "kl_teacher_student + ce_coef*GT_CE + out_pen*L2",
               "kl_coef": args.kl_coef, "ce_coef": args.ce_coef,
               "out_pen": args.out_pen, "expert_steps": args.expert_steps,
               "expert_lr": args.expert_lr,
               "owner_mass": args.owner_mass, "reject_w": args.reject_w,
               "mass_w": args.mass_w,
               "row_steps": args.row_steps, "row_lr": args.row_lr,
               "seq_len": args.seq_len, "batch_size": args.batch_size,
               "max_tokens": args.max_tokens,
               "home_domains": args.home_domains,
               "seed": args.seed},
           "donor_teacher": {"model": args.teacher,
                             "revision": args.teacher_revision},
           "host": {"model": args.host_model,
                    "revision": args.host_revision},
           "registry": args.registry,
           "start_unix": time.time()}
    dump_json(out, args.out)

    cache = None
    if args.phase in ("cache", "all"):
        cache = phase_cache(args, out)
    if args.phase in ("train", "all"):
        if cache is None:
            cache = load_cache(args.cache)
        t_load = time.time()
        model, s_tok = d0.load_donor(args.host_model, args.device, args.dtype)
        blocks = d0.find_moe_blocks(model)
        banks_reg, reg_meta = d0.load_registry(args.registry)
        if len(banks_reg) != len(blocks):
            raise RuntimeError(f"registry has {len(banks_reg)} layers, "
                               f"model has {len(blocks)}")
        print(f"loaded host+registry in {time.time() - t_load:.1f}s: "
              f"{len(blocks)} layers")

        home_names = [d.strip() for d in args.home_domains.split(",")
                      if d.strip()]
        domains = d0.load_domain_texts(args.domains_dir)
        train_w, held_w = make_windows(cache, args.seq_len, args.window_stride,
                                       args.max_windows, split_frac=0.6)
        home_train = batch_windows(train_w, args.batch_size)
        home_held = batch_windows(held_w, args.batch_size)
        home_plain_train = plain_batches(train_w, args.batch_size, 1)
        home_plain_held = plain_batches(held_w, args.batch_size, 1)
        probe_windows = probe_windows_from(held_w) or probe_windows_from(train_w)
        # per-slice plain batches from the held-out windows of each domain
        per_slice_plain = {}
        for name in home_names:
            wins_n = [w for w in held_w if w["domain"] == name]
            per_slice_plain[name] = plain_batches(wins_n, 1, 1)

        base_texts = [t for n, t in domains.items() if n not in home_names]
        base_ids = []
        for t in base_texts:
            base_ids.extend(s_tok.encode(t, add_special_tokens=False))
        base_ids = base_ids[:args.max_base_tokens]
        base_batches = []
        wins = [base_ids[i:i + args.seq_len]
                for i in range(0, len(base_ids) - args.seq_len + 1,
                               args.seq_len)]
        for i in range(0, len(wins) - args.batch_size + 1, args.batch_size):
            base_batches.append(
                (torch.tensor(wins[i:i + args.batch_size], dtype=torch.long), 0))
        if not home_train or not home_held or not base_batches or not probe_windows:
            raise RuntimeError(
                f"not enough data: home_train {len(home_train)} home_held "
                f"{len(home_held)} base {len(base_batches)} probes "
                f"{len(probe_windows)}")
        print(f"data: home train {len(home_train)} batches / held "
              f"{len(home_held)} batches, base {len(base_batches)} batches, "
              f"{len(probe_windows)} probe windows, S="
              f"{cache['shared_s_ids'].shape[0]}")

        result = run_d3(model, blocks, banks_reg, cache, home_train, home_held,
                        home_plain_train, home_plain_held, probe_windows,
                        base_batches, per_slice_plain, args, args.device)
        out.update(result)
        out["end_unix"] = time.time()
        dump_json(out, args.out)
        m = result["margins"]
        print(f"\nresults -> {args.out}")
        print(f"margins: pre {m['pre']['mean_logit_margin']:+.4f}  "
              f"solo {m['solo']['mean_logit_margin']:+.4f}  "
              f"lib {m['lib']['mean_logit_margin']:+.4f}  "
              f"gains {m['gain_solo']:+.4f}/{m['gain_lib']:+.4f}  "
              f"retention {m['retention_vs_solo']}")
        b = result["base_parity"]
        print(f"base parity cos {b['logit_cosine_mean']:.4f}  "
              f"contam {result['contamination_base']['mean_util']:.3f}  "
              f"util {result['utilization_home']['mean_util']:.3f}")
        q = result["s6_quality"]["heldout"]
        print(f"quality (held-out home CE): masked {q['ce_home_masked']:.4f}  "
              f"active {q['ce_home_active']:.4f}  delta {q['delta']:+.4f}")
        fid = result["distill_fidelity"]
        print(f"distill KL held-out: masked "
              f"{fid['masked']['kl_teacher_student']:.4f}  solo "
              f"{fid['solo']['kl_teacher_student']:.4f}  plugged "
              f"{fid['plugged']['kl_teacher_student']:.4f}")
        v = result["verdict"]
        print(f"D1 verdict: {'PASS' if v['pass'] else 'FAIL'}  gates={v['gates']}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
