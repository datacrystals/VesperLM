#!/usr/bin/env python3
"""mimo_flash_host — MiMo-V2.6-Flash as a candidate SPINE (base core).

Sibling rung to the GLM-Flash port so the pick between the two spines can be
made by the conversation bake-off (user metric) with a clean comparison.

Scope (mirrors the GLM-Flash lane):

  1. LOAD + HOST-ADAPT   layer-local slice of MiMo-V2.6-Flash-INT4
                         (full 175GB load is out of budget on <=2x32GB MI50):
                         embed + L0 (dense) + L1 (MoE 256E top-8) + norm +
                         lm_head, packed int4 experts dequantized -> fp16
                         (v05_core_prep pack-quantized math).
  2. REGISTRY / GRAFT    passport rows ON the MiMo router.  MiMo selection is
                         sigmoid + noaux_tc + e_score_correction_bias (NOT the
                         Olmoe softmax-topk the shared GraftGate assumes) — the
                         host gate here reproduces the stock semantics exactly
                         from bank scores.  init=exact (P = sqrt(d_r) * W_stock)
                         + short router-only recal, then logit-cos vs bare on a
                         real text slice.  Bar: >= 0.95.
  3. WRAP (proof of life) ONE persona plug, mem_trains, per MODULAR_MOE §10.7:
                         GRAB host-native SwiGLU core -> WRAP rank-64 width-norm
                         -> FEED trains episodes -> DELIVER additive lambda ->
                         ROUTE registry row + owner recal 0.85.

Phases:  parity | wrap | all      (JSON report to --out; stdout terse)

Ops: run on the MI50 box under a mr_gpu_claims claim lock; <=2 GPUs for this
lane.  Shared sources stay unmodified — this file is port-layer glue.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import sys
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))

import graft_d0_parity as d0  # noqa: E402
import v05_core_prep as prep  # noqa: E402

FORMAT = "vesper-mimo-flash-host-v1"
MODEL_DIR = "/data/ModelDownloader/MiMo-V2.6-Flash-INT4"
BAR_LOGIT_COS = 0.95

# Persona pack rows: lab/mimo_plug_rows.py (verbatim from
# lab/memories/persona_v1.json — mem_pack drags the Hippocampus chain).
from mimo_plug_rows import PLUGS  # noqa: E402

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dump_json(out: dict, path: str | None) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, default=str)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Remote-code import (modeling_mimo_v2 uses a relative config import)
# ---------------------------------------------------------------------------

def import_mimo_code(model_dir: str):
    pkg_name = "mimo_v2_remote"
    if pkg_name in sys.modules:
        return sys.modules[f"{pkg_name}.modeling_mimo_v2"], \
            sys.modules[f"{pkg_name}.configuration_mimo_v2"]
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [model_dir]
    sys.modules[pkg_name] = pkg
    cfg_mod = importlib.import_module(f"{pkg_name}.configuration_mimo_v2")
    mod_mod = importlib.import_module(f"{pkg_name}.modeling_mimo_v2")
    # transformers 5.19 renamed create_causal_mask's input_embeds ->
    # inputs_embeds and dropped cache_position; the vendor module (frozen in
    # the checkpoint dir) still calls the old spelling.  Shim in the port
    # layer only — vendor source stays unmodified.
    from transformers.masking_utils import create_causal_mask as _ccm
    from transformers.masking_utils import \
        create_sliding_window_causal_mask as _cswm

    def _adapt(fn):
        def call(**kw):
            if "input_embeds" in kw:
                kw["inputs_embeds"] = kw.pop("input_embeds")
            kw.pop("cache_position", None)
            return fn(**kw)
        return call

    mod_mod.create_causal_mask = _adapt(_ccm)
    mod_mod.create_sliding_window_causal_mask = _adapt(_cswm)
    return mod_mod, cfg_mod


# ---------------------------------------------------------------------------
# INT4 dequant (pack-quantized, group-32, asymmetric — grabkit/runbook §1 key
# notes; identical math to v05_core_prep.dequant_int4_core for any Linear).
# ---------------------------------------------------------------------------

def dequant_int4_linear(get, root: str) -> torch.Tensor:
    """{root}weight_{packed,scale,shape,zero_point} -> fp32 (out, in)."""
    packed = get(root + "weight_packed")
    scale = get(root + "weight_scale").float()
    zp_p = get(root + "weight_zero_point")
    shape = get(root + "weight_shape").tolist()
    out_f, in_f = int(shape[0]), int(shape[1])
    g = in_f // scale.shape[-1]
    if scale.shape[0] != out_f or in_f % scale.shape[-1] != 0 or g < 1:
        raise ValueError(f"{root} scale {tuple(scale.shape)} vs ({out_f},{in_f})")
    q = prep.unpack_int4_signed(packed, (out_f, in_f), packed_dim=1)
    zp = prep.unpack_int4_signed(zp_p, (out_f, in_f // g), packed_dim=0)
    w = (q.float() - zp.float().repeat_interleave(g, dim=1)) * \
        scale.repeat_interleave(g, dim=1)
    del q, zp
    return w


# ---------------------------------------------------------------------------
# Slice load: embed + layers 0..K-1 + norm + lm_head
# ---------------------------------------------------------------------------

def load_slice(model_dir: str, n_layers: int, device: str, out: dict):
    from safetensors import safe_open
    t0 = time.time()
    dtype = torch.float32 if str(device).startswith("cpu") else torch.float16
    mod_mod, cfg_mod = import_mimo_code(model_dir)
    raw = json.load(open(os.path.join(model_dir, "config.json")))
    raw.pop("auto_map", None)
    raw["num_hidden_layers"] = n_layers
    raw["moe_layer_freq"] = list(raw["moe_layer_freq"])[:n_layers]
    raw["hybrid_layer_pattern"] = list(raw["hybrid_layer_pattern"])[:n_layers]
    raw["vision_config"] = None          # text-only slice: no multimodal towers
    raw["audio_config"] = None
    cfg = cfg_mod.MiMoV2Config(**raw)

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = mod_mod.MiMoV2ForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev_dtype)
    model.eval()

    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    wm = idx["weight_map"]

    def wanted(k: str) -> bool:
        if k in ("model.embed_tokens.weight", "model.norm.weight",
                 "lm_head.weight"):
            return True
        return k.startswith("model.layers.") and \
            int(k.split(".")[2]) < n_layers

    keys = [k for k in wm if wanted(k)]
    by_shard: dict[str, list[str]] = {}
    for k in keys:
        by_shard.setdefault(wm[k], []).append(k)

    cache = os.environ.get(
        "MIMO_SLICE_CACHE",
        f"/data/tliao/mimo_flash/slice_state_{n_layers}.pt")
    to_load: dict[str, torch.Tensor] = {}
    n_q = 0
    if os.path.exists(cache):
        # mmap: 6 parallel lanes share page cache instead of 32GB RSS each
        to_load = torch.load(cache, map_location="cpu", weights_only=False,
                             mmap=True)
        n_q = sum(1 for k in to_load if k.endswith(".weight"))
        log(f"state cache hit {cache} ({len(to_load)} keys)")
    else:
        for shard, ks in sorted(by_shard.items()):
            with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
                def get(name, _f=f):
                    return _f.get_tensor(name)
                for k in ks:
                    if k.endswith("weight_packed"):
                        root = k[: -len("weight_packed")]
                        to_load[root + "weight"] = \
                            dequant_int4_linear(get, root).to(dtype)
                        n_q += 1
                    elif k.endswith(("weight_scale", "weight_shape",
                                     "weight_zero_point")):
                        continue
                    else:
                        to_load[k] = get(k).to(dtype)
            log(f"loaded shard {shard} (+{len(ks)} keys)")
        try:
            torch.save(to_load, cache + ".tmp")
            os.replace(cache + ".tmp", cache)
            log(f"state cache wrote {cache}")
        except Exception as e:  # noqa: BLE001
            log(f"state cache skip: {e}")

    missing, unexpected = model.load_state_dict(to_load, strict=False)
    real_missing = [m for m in missing if "rotary_emb.inv_freq" not in m
                    and not m.startswith("visual.") and not m.startswith("audio")]
    if real_missing:
        raise RuntimeError(f"missing keys: {real_missing[:8]}")
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected[:8]}")
    model.to(device)
    out["load"] = {
        "mode": "layer-local",
        "n_layers": n_layers,
        "layers": f"0..{n_layers - 1} (L0 dense MLP, L1+ MoE)",
        "quantized_linears_dequantized": n_q,
        "dtype": "float16",
        "device": device,
        "wall_s": round(time.time() - t0, 2),
        "param_bytes_gb": round(sum(p.numel() * p.element_size()
                                    for p in model.parameters()) / 1e9, 2),
    }
    log(f"slice ready: {n_layers}L dequantized={n_q} "
        f"{out['load']['param_bytes_gb']}GB")
    return model, cfg


# ---------------------------------------------------------------------------
# Host adapt: MoE blocks duck-typed to the graft contract
# ---------------------------------------------------------------------------

def stock_dispatch(blk) -> callable:
    """experts.forward(h, idx, w) — stock merge, MiMoV2MoE.moe() semantics.

    gfx906 note: this torch build's index-scan ops (torch.where(mask) /
    nonzero / boolean-mask gather) fail with hipErrorInvalidDeviceFunction,
    which is exactly what the vendor moe() loop uses.  The merge below is the
    same math via argsort segments + integer gather + index_add_ (all clean
    on gfx906); segment bounds come from one small host read per call.
    """
    experts = blk.experts

    def forward(hidden_states, top_k_index, top_k_weights):
        h = hidden_states
        T = h.shape[0]
        k = top_k_index.shape[1]
        flat_eid = top_k_index.reshape(-1)
        order = torch.argsort(flat_eid)
        eid = flat_eid[order].cpu().tolist()
        tok = torch.arange(T, device=h.device).repeat_interleave(k)[order]
        wt = top_k_weights.reshape(-1)[order].to(h.dtype)
        final = torch.zeros_like(h, dtype=h.dtype)
        i = 0
        n = len(eid)
        while i < n:
            j = i + 1
            while j < n and eid[j] == eid[i]:
                j += 1
            rows = tok[i:j]
            ew = wt[i:j].unsqueeze(-1)
            eo = experts[eid[i]](h[rows])
            final.index_add_(0, rows, eo * ew)
            i = j
        return final

    return forward


def adapt_moe_block(blk) -> None:
    """Route blk.forward through blk.experts.forward (the graft socket)."""
    blk.experts.forward = stock_dispatch(blk)

    def _fwd(self, hidden_states):
        orig = hidden_states.shape
        topk_idx, topk_w = self.gate(hidden_states)
        h = hidden_states.reshape(-1, orig[-1])
        y = self.experts.forward(h, topk_idx, topk_w)
        return y.reshape(orig)

    blk.forward = types.MethodType(_fwd, blk)


def find_mimo_moe_blocks(model) -> list:
    """MoE blocks duck-typed: .gate with 2D .weight + .experts ModuleList,
    and the gate returns (idx, weight) — the MiMoV2MoE contract."""
    blocks = []
    for m in model.modules():
        g = getattr(m, "gate", None)
        ex = getattr(m, "experts", None)
        if g is None or ex is None or not isinstance(ex, nn.ModuleList):
            continue
        w = getattr(g, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2:
            blocks.append(m)
    return blocks


class MimoPassportGate(nn.Module):
    """Passport-bank router reproducing MiMo-V2 MoE selection semantics.

    Stock (modeling_mimo_v2.MiMoV2MoEGate, noaux_tc with n_group=1):
        scores          = sigmoid(h @ W.T)            (fp32)
        scores_for_ch   = scores + e_score_correction_bias
        topk_idx        = topk(scores_for_ch, k)      (group mask is a no-op
                                                       at n_group=1)
        topk_weight     = scores.gather(topk_idx), renormed (norm_topk_prob)

    With init=exact (P = sqrt(d_r) * W_stock, identity W_q) the bank score IS
    the stock pre-sigmoid logit, so this gate is route-identical to bare.

    forward(h) -> (topk_idx, topk_weight)   [the MiMoV2MoE gate contract]
    Optional n_new plug-in rows only vote membership (door6 semantics).
    """

    def __init__(self, bank: d0.PassportBank, stock_gate: nn.Module,
                 n_new: int = 0):
        super().__init__()
        self.bank = bank
        self.n_new = n_new
        self.top_k = int(stock_gate.top_k)
        self.norm_topk_prob = bool(stock_gate.norm_topk_prob)
        self.register_buffer(
            "bias", stock_gate.e_score_correction_bias.detach().float().clone())
        self.last_h = None
        self.last_scores = None
        self.last_idx = None
        self.last_membership = None
        self.last_p_new = None

    def forward(self, hidden_states: torch.Tensor):
        h = hidden_states.reshape(-1, self.bank.d_model)
        scores = self.bank.scores(h)                    # (T, E_stock+n_new)
        n_stock = scores.shape[1] - self.n_new
        stock_scores = scores[:, :n_stock]
        sig = torch.sigmoid(stock_scores.float())
        choice = sig + self.bias.unsqueeze(0)
        topk_idx = torch.topk(choice, self.top_k, dim=-1, sorted=False).indices
        topk_weight = sig.gather(1, topk_idx)
        if self.norm_topk_prob and self.top_k > 1:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        topk_weight = topk_weight.to(hidden_states.dtype)
        if self.n_new:
            # door6 membership in RAW score space — sigmoid saturates near 1.0
            # on this host (measured 2026-10-16: plug_sig vs kth both ~1.0
            # gives noise membership), and the row recal optimises raw bank
            # scores, so score-vs-kth-raw is the consistent comparison.
            kth_raw = torch.topk(stock_scores.float(), self.top_k,
                                 dim=-1).values[:, -1]
            self.last_membership = \
                scores[:, n_stock:].float() > kth_raw.unsqueeze(-1)
            self.last_p_new = F.softmax(scores.float(), dim=-1)[:, n_stock:]
        self.last_h = h.detach()
        self.last_scores = scores.detach()
        self.last_idx = topk_idx.detach()
        return topk_idx, topk_weight


def init_exact(bank: d0.PassportBank, w_stock: torch.Tensor) -> None:
    """p_e = sqrt(d_r) * W_stock[e]  =>  bank score == stock logit exactly."""
    with torch.no_grad():
        bank.P.data.copy_(w_stock.float() * math.sqrt(bank.d_r))
    bank.layer_scale = 1.0


# ---------------------------------------------------------------------------
# Parity measurement (bare vs grafted) on a real text slice
# ---------------------------------------------------------------------------

def real_batches(tok, data_dir: str, n: int, seq: int, bs: int):
    """Real prose windows from the graft_d2 slice (chat + web + val)."""
    ids = []
    for name in ("chat.txt", "web.txt", "val_mixed.txt"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            ids.extend(tok.encode(open(p).read(), add_special_tokens=False))
    ids = ids[: n * seq * bs]
    wins = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
    return [(torch.tensor(wins[i:i + bs], dtype=torch.long), 0)
            for i in range(0, len(wins) - bs + 1, bs)]


@torch.no_grad()
def parity_vs_bare(model, blocks, gates, batches, device) -> dict:
    """Token logit-cosine of stock-gate pass vs passport-gate pass."""
    sum_cos = sum_sq = n_tok = 0.0
    cos_min = 1.0
    ce_a = ce_b = n_row = 0.0
    agree = 0
    n_idx = 0
    for tokens, _ in batches:
        tokens = tokens.to(device)
        bsz = tokens.shape[0]
        # bare pass — capture each stock gate's topk indices via forward hook
        stock_idx: list = [None] * len(blocks)
        hooks = []
        for i, blk in enumerate(blocks):
            def post(mod, inp, out, i=i):
                stock_idx[i] = out[0].detach().clone()
            hooks.append(blk.gate.register_forward_hook(post))
        lg_a = d0.model_logits(model, tokens)
        for h in hooks:
            h.remove()
        # grafted pass
        saved = []
        for blk, g in zip(blocks, gates):
            saved.append(blk.gate)
            blk.gate = g
        lg_b = d0.model_logits(model, tokens)
        graft_idx = [g.last_idx.clone() for g in gates]
        for blk, g0_ in zip(blocks, saved):
            blk.gate = g0_

        cos = d0.token_logit_cosine(lg_a, lg_b)
        sum_cos += float(cos.sum())
        sum_sq += float((cos * cos).sum())
        cos_min = min(cos_min, float(cos.min()))
        n_tok += cos.numel()
        ce_a += d0.next_token_ce(lg_a, tokens) * bsz
        ce_b += d0.next_token_ce(lg_b, tokens) * bsz
        n_row += bsz
        for si, gi in zip(stock_idx, graft_idx):
            si2, _ = torch.sort(si, dim=-1)
            gi2, _ = torch.sort(gi, dim=-1)
            agree += int((si2 == gi2).float().mean().item() * si.shape[0])
            n_idx += si.shape[0]
    mean = sum_cos / max(1, n_tok)
    var = max(0.0, sum_sq / max(1, n_tok) - mean * mean)
    return {
        "n_tokens": int(n_tok),
        "logit_cosine_mean": mean,
        "logit_cosine_std": math.sqrt(var),
        "logit_cosine_min": cos_min,
        "ce_bare": ce_a / max(1, n_row),
        "ce_graft": ce_b / max(1, n_row),
        "ce_delta": (ce_b - ce_a) / max(1, n_row),
        "route_topk_agree": agree / max(1, n_idx),
    }


@torch.no_grad()
def collect_pairs(model, blocks, batches, device, max_pairs=4096):
    """(h, stock pre-sigmoid logits) per MoE layer under stock gates."""
    n = len(blocks)
    hs = [[] for _ in range(n)]
    handles = []
    for i, blk in enumerate(blocks):
        def pre(mod, inp, i=i):
            hs[i].append(inp[0].detach().reshape(-1, inp[0].shape[-1])
                         .float().cpu())
        handles.append(blk.gate.register_forward_pre_hook(pre))
    for tokens, _ in batches:
        d0.model_logits(model, tokens.to(device))
    for h in handles:
        h.remove()
    out = []
    for i, blk in enumerate(blocks):
        if not hs[i]:
            out.append((torch.zeros(0, 1), torch.zeros(0, 1)))
            continue
        h = torch.cat(hs[i], dim=0)
        s = F.linear(h, blk.gate.weight.float())
        if h.shape[0] > max_pairs:
            stride = h.shape[0] // max_pairs + 1
            h, s = h[::stride][:max_pairs], s[::stride][:max_pairs]
        out.append((h, s))
    return out


# ---------------------------------------------------------------------------
# Phase 1: registry/graft parity
# ---------------------------------------------------------------------------

def phase_parity(model, cfg, tok, device, out: dict, args) -> list:
    blocks = find_mimo_moe_blocks(model)
    log(f"moe blocks in slice: {len(blocks)}")
    banks, w_stock = [], []
    for blk in blocks:
        d_model = blk.gate.weight.shape[1]
        bank = d0.PassportBank(blk.gate.weight.shape[0], d_model,
                               d_r=d_model, learn_wq=False)
        init_exact(bank, blk.gate.weight)
        banks.append(bank)
        w_stock.append(blk.gate.weight.detach())
    gates = [MimoPassportGate(b, blk.gate) for b, blk in zip(banks, blocks)]

    batches = real_batches(tok, args.data_dir, args.n_batches, args.seq,
                           args.bs)
    log(f"parity batches: {len(batches)} x {args.bs}x{args.seq}")

    p0 = parity_vs_bare(model, blocks, gates, batches, device)
    out["parity_init_exact"] = p0
    log(f"parity init=exact cos={p0['logit_cosine_mean']:.4f} "
        f"min={p0['logit_cosine_min']:.4f} agree={p0['route_topk_agree']:.3f}")

    pairs = collect_pairs(model, blocks, batches, device)
    rec = d0.recalibrate(banks, pairs, steps=args.recal_steps,
                         lr=args.recal_lr)
    out["recal"] = rec
    p1 = parity_vs_bare(model, blocks, gates, batches, device)
    out["parity_after_recal"] = p1
    out["parity_bar"] = BAR_LOGIT_COS
    out["parity_pass"] = bool(p1["logit_cosine_mean"] >= BAR_LOGIT_COS)
    log(f"parity post-recal cos={p1['logit_cosine_mean']:.4f} "
        f"min={p1['logit_cosine_min']:.4f} PASS={out['parity_pass']}")
    return blocks, banks, gates


# ---------------------------------------------------------------------------
# Phase 2: ONE proof-of-life wrap (mem_trains) — §10.7 recipe
# ---------------------------------------------------------------------------

def _chatml_ids(tok, prompt, answer):
    msgs = [{"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}]
    txt = tok.apply_chat_template(msgs, tokenize=False,
                                  add_generation_prompt=False)
    return tok.encode(txt, add_special_tokens=False)


def _home_stream(tok, tag, feed_tokens, seq, bs):
    """ChatML teach+approval windows for one plug, tiled to feed_tokens."""
    rows = PLUGS[tag]
    ids = []
    for prompt, _stated, corr in rows["teach"]:
        ids.extend(_chatml_ids(tok, prompt, corr))
    for prompt, resp in rows["approvals"]:
        ids.extend(_chatml_ids(tok, prompt, resp))
    if not ids:
        raise RuntimeError(f"empty home stream for {tag}")
    ids = (ids * ((feed_tokens + len(ids) - 1) // len(ids)))[:feed_tokens]
    wins = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
    return [(torch.tensor(wins[i:i + bs], dtype=torch.long), 1)
            for i in range(0, len(wins) - bs + 1, bs)]


def _base_stream(tok, data_dir, seq, bs, max_tokens=40000):
    ids = []
    for name in ("web.txt", "science.txt", "code.txt"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            ids.extend(tok.encode(open(p).read(), add_special_tokens=False))
    ids = ids[:max_tokens]
    wins = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
    return [(torch.tensor(wins[i:i + bs], dtype=torch.long), 0)
            for i in range(0, len(wins) - bs + 1, bs)]


def capture_h(model, blk, batches, device) -> torch.Tensor:
    hs = []

    def pre(mod, inp):
        hs.append(inp[0].detach().reshape(-1, inp[0].shape[-1]).float().cpu())
    h = blk.gate.register_forward_pre_hook(pre)
    for tokens, _ in batches:
        d0.model_logits(model, tokens.to(device))
    h.remove()
    return torch.cat(hs, 0) if hs else torch.zeros(0, 1)


def shape_probe(model, wrap, blk, home_train, device) -> dict:
    """One-step CE fwd+bwd at candidate B*T (gfx906 backward holes are
    spine-specific — v0.6 lesson; probe before every fit)."""
    res = {"ok": False, "safe_bt": None, "tried": []}
    for bt in (128, 192, 256):
        sf = None
        try:
            toks = home_train[0][0].to(device)
            if toks.numel() > bt:
                toks = toks.reshape(-1)[:bt].reshape(1, -1)
            wrap.to(device)
            sf = blk.experts.forward

            def fwd(h, i, w, _sf=sf, _wrap=wrap):
                return _sf(h, i, w) + _wrap(h)
            blk.experts.forward = fwd
            logits = d0.model_logits(model, toks)
            v = logits.shape[-1]
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, v).float(),
                                   toks[:, 1:].reshape(-1))
            loss.backward()
            wrap.zero_grad(set_to_none=True)
            blk.experts.forward = sf
            ok = bool(torch.isfinite(loss))
            res["tried"].append({"bt": bt, "ok": ok})
            if ok:
                res["ok"] = True
                res["safe_bt"] = bt
        except Exception as e:  # noqa: BLE001
            res["tried"].append({"bt": bt, "ok": False, "err": str(e)[:80]})
            if sf is not None:
                try:
                    blk.experts.forward = sf
                except Exception:  # noqa: BLE001
                    pass
    return res


@torch.no_grad()
def fit_lam_grid(model, handles, lam, home, device) -> float:
    """Door-6 style fitted additive lambda (grid incl. below 0.25)."""
    import graft_d1_plugin as d1
    best, best_ce = 0.0, float("inf")
    for cand in (0.0, 0.1, 0.25, 0.5, 1.0):
        with torch.no_grad():
            lam.fill_(cand)
        d1.set_plugin_mode(handles, "additive")
        tot, n = 0.0, 0
        for tokens, _ in home:
            tokens = tokens.to(device)
            lg = d0.model_logits(model, tokens)
            tot += d0.next_token_ce(lg, tokens) * tokens.shape[0]
            n += tokens.shape[0]
        ce = tot / max(1, n)
        if ce < best_ce:
            best_ce, best = ce, cand
    with torch.no_grad():
        lam.fill_(best)
    d1.set_plugin_mode(handles, "off")
    return best


def _install_additive(blk, wrap, lam, bank65, gate65):
    import door6_expert_mix as d6
    sf = blk.experts.forward
    if hasattr(blk, "_plugin_merge") and hasattr(blk._plugin_merge, "stock"):
        sf = blk._plugin_merge.stock
    pm = d6.MixMerge(sf, [wrap], [lam], ["plug"],
                     gate_getter=(lambda b=blk: b.gate),
                     num_stock=bank65.num_experts - 1)
    pm.mode = "additive"
    blk._plugin_merge = pm
    blk.experts.forward = lambda h, i, w: pm(h, i, w)
    # membership comes from the plug-row gate: install it as blk.gate so the
    # block forward populates last_membership before MixMerge reads it
    stock_gate = blk.gate
    blk.gate = gate65
    return [(blk, sf)], stock_gate


@torch.no_grad()
def quality_reads(model, blk, gate65, handles, home, base, device, lam) -> dict:
    import graft_d1_plugin as d1
    out = {}

    def ce(batches, mode):
        tot, n = 0.0, 0
        for tokens, _ in batches:
            tokens = tokens.to(device)
            d1.set_plugin_mode(handles, mode)
            lg = d0.model_logits(model, tokens)
            tot += d0.next_token_ce(lg, tokens) * tokens.shape[0]
            n += tokens.shape[0]
        return tot / max(1, n)

    out["ce_home_off"] = ce(home, "off")
    out["ce_home_add"] = ce(home, "additive")
    out["ce_base_off"] = ce(base, "off")
    out["ce_base_add"] = ce(base, "additive")
    out["lam"] = float(lam.detach())
    return out


def fit_one_plug(model, tok, device, args, blk, bank, tag, host_expert,
                 home_train, base) -> dict:
    """One §10.7 wrap fit: GRAB host-native core -> WRAP rank-64 -> FEED
    (forced task-CE + out_pen) -> ROUTE registry row (home-minus-base init +
    owner recal 0.85) -> DELIVER additive lam.  Returns the plugin record."""
    import door6_expert_mix as d6
    import door25_wrap_expert as d25
    import graft_d1_plugin as d1

    t0 = time.time()
    d_model = bank.d_model
    res = {"tag": tag, "host_expert": host_expert}

    # GRAB: host-native SwiGLU core verbatim
    ex = blk.experts[host_expert]
    w_gu = torch.cat([ex.gate_proj.weight.float(),
                      ex.up_proj.weight.float()], dim=0).cpu()
    w_down = ex.down_proj.weight.float().cpu()
    wrap = d25.WrappedForeignExpert(d_model, 64, w_gu, w_down,
                                    seed=args.seed, act="swiglu")

    # WRAP fit (forced dispatch)
    wrap.to(device)
    sf = blk.experts.forward
    lam = nn.Parameter(torch.zeros((), device=device))
    pm = d6.MixMerge(sf, [wrap], [lam], [tag],
                     gate_getter=(lambda b=blk: getattr(b, "_mimo_gate",
                                                        b.gate)),
                     num_stock=bank.num_experts)
    pm.mode = "off"
    blk._plugin_merge = pm
    blk.experts.forward = lambda h, i, w: pm(h, i, w)
    handles = [(blk, sf)]
    steps = max(args.expert_steps, -(-args.feed_tokens // max(1, args.bt)))
    pa = d25.train_adapters(model, wrap, handles, home_train, base,
                            steps, args.expert_lr, device,
                            out_pen=args.out_pen, grad_clip=1.0,
                            weight_decay=0.0)
    d1.set_plugin_mode(handles, "off")
    blk.experts.forward = sf
    if hasattr(blk, "_plugin_merge"):
        delattr(blk, "_plugin_merge")
    # NaN guard (persona_build lesson: one wrap NaNs at 3e-3, recovers at 1e-3)
    finite = pa.get("history") and all(
        math.isfinite(float(h.get("ce_batch0", float("nan"))))
        for h in pa["history"][-3:])
    if not finite:
        log(f"[{tag}] NON-FINITE at lr {args.expert_lr} — refit 1e-3")
        del wrap
        torch.cuda.empty_cache() if device.startswith("cuda") else None
        wrap = d25.WrappedForeignExpert(d_model, 64, w_gu, w_down,
                                        seed=args.seed, act="swiglu")
        wrap.to(device)
        pm = d6.MixMerge(sf, [wrap], [lam], [tag],
                         gate_getter=(lambda b=blk: getattr(b, "_mimo_gate",
                                                            b.gate)),
                         num_stock=bank.num_experts)
        pm.mode = "off"
        blk._plugin_merge = pm
        blk.experts.forward = lambda h, i, w: pm(h, i, w)
        handles = [(blk, sf)]
        pa = d25.train_adapters(model, wrap, handles, home_train, base,
                                steps, 1e-3, device,
                                out_pen=args.out_pen, grad_clip=1.0,
                                weight_decay=0.0)
        d1.set_plugin_mode(handles, "off")
        blk.experts.forward = sf
        if hasattr(blk, "_plugin_merge"):
            delattr(blk, "_plugin_merge")
        res["refit_lr"] = 1e-3
    res["phase_a"] = {
        "steps": pa.get("steps"), "wall_s": pa.get("wall_s"),
        "ce_first": pa["history"][0]["ce_batch0"] if pa["history"] else None,
        "ce_last": pa["history"][-1]["ce_batch0"] if pa["history"] else None,
    }

    # ROUTE: registry row (home-minus-base centroid @0.25x stock-norm — the
    # saturating mean-proto init is a measured falsifier on this host) +
    # owner-mass recal 0.85.
    bank65 = d0.PassportBank(bank.num_experts + 1, d_model, d_r=bank.d_r,
                             learn_wq=False)
    with torch.no_grad():
        bank65.P.data[:bank.num_experts] = bank.P.data
    bank65.layer_scale = bank.layer_scale
    gate65 = MimoPassportGate(bank65, blk.gate, n_new=1)
    blk._mimo_gate = gate65

    hs_home = capture_h(model, blk, home_train[:4], device)
    hs_base = capture_h(model, blk, base[:4], device)
    with torch.no_grad():
        if hs_home.shape[0] > 0:
            proto = hs_home.mean(dim=0) - hs_base.mean(dim=0)
            stock_norm = float(bank65.P.data[:bank.num_experts].float()
                               .norm(dim=1).mean())
            pn = float(proto.norm())
            if pn > 0 and stock_norm > 0:
                proto = proto * (stock_norm * 0.25 / pn)
            bank65.P.data[bank.num_experts] = proto
    h_all = torch.cat([hs_home, hs_base], 0)
    lab = torch.cat([torch.ones(hs_home.shape[0]),
                     torch.zeros(hs_base.shape[0])])
    rep = d6.train_rows_one(bank65, (h_all, lab), bank.num_experts,
                            args.row_steps, args.row_lr, device,
                            reject_w=15.0, mass_w=2.0, owner_target=0.85)
    res["row_recal"] = {"first": rep.get("first"), "last": rep.get("last")}

    # DELIVER: fitted additive lam + quality + taught-token margins
    row = bank65.P.data[bank.num_experts].float().cpu().clone()
    handles2, stock_gate = _install_additive(blk, wrap, lam, bank65, gate65)
    best_lam = fit_lam_grid(model, handles2, lam, home_train[:4], device)
    q = quality_reads(model, blk, gate65, handles2, home_train[:4],
                      base[:4], device, lam)

    def logits_fn(x):
        return d0.model_logits(model, x.to(device))
    d1.set_plugin_mode(handles2, "forced", 1.0)
    marg_forced = fact_margins(logits_fn, tok, device, tag)
    d1.set_plugin_mode(handles2, "off")
    marg_off = fact_margins(logits_fn, tok, device, tag)
    blk.gate = stock_gate
    blk.experts.forward = sf
    if hasattr(blk, "_plugin_merge"):
        delattr(blk, "_plugin_merge")

    res.update({"lam": best_lam, "quality": q,
                "margins_forced": marg_forced, "margins_off": marg_off,
                "row": row, "wall_s": round(time.time() - t0, 2),
                "wrap_state": wrap.state_dict(), "wrap_spec": wrap.wrap_spec()})
    return res


def save_plugin(path, tag, wrap_state, wrap_spec, row, rec):
    torch.save({"format": "vesper-mimo-plug-v1", "tag": tag,
                "wrap": wrap_state, "wrap_spec": wrap_spec,
                "row": row, "lam": rec.get("lam"),
                "row_recal": rec.get("row_recal"),
                "quality": rec.get("quality"),
                "margins_forced": rec.get("margins_forced"),
                "host": {"layer": 0, "expert": rec.get("host_expert")}},
               path)


def phase_fit_plugs(model, cfg, tok, device, out: dict, args, blocks, banks):
    """Fit the 6-plug persona set to the MiMo host, one plug per call."""
    tags = [t.strip() for t in args.plug_tags.split(",") if t.strip()]
    for tag in tags:
        if tag not in PLUGS:
            raise SystemExit(f"unknown plug {tag!r}")
    blk = blocks[args.host_layer]
    bank = banks[args.host_layer]
    adapt_moe_block(blk)
    args.bt = args.seq * args.bs
    base = _base_stream(tok, args.data_dir, args.seq, args.bs)
    os.makedirs(args.plugins_dir, exist_ok=True)
    out["plugs"] = {}
    for i, tag in enumerate(tags):
        home = _home_stream(tok, tag, args.feed_tokens, args.seq, args.bs)
        hexe = args.host_expert + i   # distinct donor core per plug
        log(f"== fit {tag} (expert e{hexe}, FEED {args.feed_tokens}) ==")
        try:
            rec = fit_one_plug(model, tok, device, args, blk, bank, tag,
                               hexe, home, base)
        except Exception as e:  # noqa: BLE001
            out["plugs"][tag] = {"error": str(e)[:300]}
            dump_json(out, args.out)
            continue
        # serialize the FITTED wrap state (not a fresh init)
        ppath = os.path.join(args.plugins_dir, f"{tag}_plugin.pt")
        save_plugin(ppath, tag, rec["wrap_state"], rec["wrap_spec"],
                    rec["row"], rec)
        rr = rec["row_recal"]
        out["plugs"][tag] = {
            "plugin": ppath, "host_expert": hexe,
            "phase_a": rec["phase_a"], "row_recal": rr,
            "lam": rec["lam"], "quality": rec["quality"],
            "margins_forced": rec["margins_forced"],
            "margins_off": rec["margins_off"], "wall_s": rec["wall_s"],
        }
        dump_json(out, args.out)
        log(f"== {tag}: ce {rec['phase_a']['ce_first']} -> "
            f"{rec['phase_a']['ce_last']} row p_h "
            f"{rr.get('last', {}).get('p_new_code')} lam {rec['lam']} ==")
    out["phase"] = "plugs_done"
    dump_json(out, args.out)


def fact_margins(logits_fn, tok, device, tag: str) -> dict:
    """memory_shaping-style margin: logit(correction) - logit(stated) at the
    first divergence, chatml-encoded (plug teach rows)."""
    out = {}
    for prompt, stated, correction in PLUGS[tag]["teach"]:
        ps = _chatml_ids(tok, prompt, stated)
        pc = _chatml_ids(tok, prompt, correction)
        k = next((i for i in range(min(len(ps), len(pc))) if ps[i] != pc[i]),
                 min(len(ps), len(pc)) - 1)
        xt = torch.tensor([ps[:k + 1]], dtype=torch.long, device=device)
        lg = logits_fn(xt)[0, -1].float()
        out[prompt[:24]] = round(
            float(lg[pc[k]] - lg[ps[k]]), 3)
    return out


# ---------------------------------------------------------------------------
# bake-off generator (spine_bakeoff.py graft backend)
# ---------------------------------------------------------------------------


# think-tag / EOT hygiene: the baked-in reasoning tag is NOT a tokenizer
# special on every spine, so skip_special_tokens alone leaves it in replies
# (measured 2026-10-16: every slice-smoke reply opened with the raw tag).
_THINK_RE = re.compile(r"<\|[^|>]{1,24}\|>.*?<\|[^|>]{1,24}\|>",
                       re.DOTALL)
_TAG_RE = re.compile(r"<\|[^|>]{1,24}\|>")


def _clean_reply(text: str) -> str:
    """Strip reasoning blocks and stray delimiter tokens, keep the answer.

    Measured 2026-10-16 slice-smoke replies: the model emits the opening
    delimiter then answers directly (never closes) — so a dangling-opener
    cut would delete the answer.  Rule: drop closed reasoning pairs first,
    then drop any residual lone delimiter tokens, keep the rest.
    """
    t = _THINK_RE.sub(" ", text)
    t = _TAG_RE.sub(" ", t)
    return " ".join(t.split()).strip()


def stop_token_ids(tok) -> set:
    """EOT ids derived from the tokenizer chat template (no literals): the
    tail tokens the template emits after an assistant message are the
    end-of-turn marker; max-new is a cap, not a target."""
    ids = set()
    try:
        probe = tok.apply_chat_template(
            [{"role": "assistant", "content": "x"}], tokenize=False,
            add_generation_prompt=False)
        tail = tok.encode(probe, add_special_tokens=False)[-4:]
        ids.update(t for t in tail if t >= 0)
    except Exception:  # noqa: BLE001
        pass
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


def make_generator(plugins_dir: str, args=None):
    """(tok, gen_fn) for lab/spine_bakeoff.py --spine mimo-flash.

    Layer-local 2L MiMo slice + the fitted plug set installed additively;
    decode knobs come from the harness's frozen BAKEOFF_DECODE dict.
    """
    device = getattr(args, "device", "cuda") if args is not None else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    def gen_fn(messages, max_new, dec, seed=0):
        # late import so the module stays importable without the box paths
        tok = getattr(gen_fn, "_tok", None)
        model = getattr(gen_fn, "_model", None)
        if model is None:
            raise RuntimeError("generator not initialized")
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
        ids = tok.encode(prompt, add_special_tokens=False)
        out_ids = []
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        stop = stop_token_ids(tok)
        for _ in range(int(max_new)):
            x = torch.tensor([ids + out_ids], dtype=torch.long,
                             device=device)
            logits = d0.model_logits(model, x)[0, -1].float().cpu()
            # CTRL-style rep penalty over prompt+reply (shipped defaults)
            pen = float(dec.get("rep_penalty", 1.0))
            if pen > 1.0:
                for t in set(ids + out_ids):
                    if 0 <= t < logits.shape[-1]:
                        if logits[t] > 0:
                            logits[t] /= pen
                        else:
                            logits[t] *= pen
            if dec.get("sample"):
                temp = max(1e-5, float(dec.get("temperature", 0.8)))
                top_p = float(dec.get("top_p", 0.95))
                probs = F.softmax(logits / temp, dim=-1)
                sp, si = torch.sort(probs, descending=True)
                cum = sp.cumsum(0)
                nz = (cum > top_p).nonzero(as_tuple=True)[0]
                cut = int(nz[:1].item()) + 1 if nz.numel() else sp.numel()
                p = sp[:cut] / sp[:cut].sum()
                nxt = int(si[torch.multinomial(p, 1, generator=g)])
            else:
                nxt = int(torch.argmax(logits))
            if nxt in stop:
                break
            out_ids.append(nxt)
        return _clean_reply(tok.decode(out_ids, skip_special_tokens=True))

    # build the stack once and stash on the closure
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model, _cfg = load_slice(MODEL_DIR, 2, device, {})
    blocks = find_mimo_moe_blocks(model)
    banks = []
    for blk in blocks:
        d_model = blk.gate.weight.shape[1]
        bank = d0.PassportBank(blk.gate.weight.shape[0], d_model,
                               d_r=d_model, learn_wq=False)
        init_exact(bank, blk.gate.weight)
        banks.append(bank)
    installed = install_plug_set(model, blocks, banks, plugins_dir, device)
    gen_fn._tok = tok
    gen_fn._model = model
    gen_fn.installed = installed
    return tok, gen_fn


def install_plug_set(model, blocks, banks, plugins_dir, device) -> list:
    """Load <tag>_plugin.pt files and install them additively at host layer 0."""
    import door25_wrap_expert as d25
    installed = []
    if not plugins_dir or not os.path.isdir(plugins_dir):
        return installed
    blk = blocks[0]
    bank = banks[0]
    adapt_moe_block(blk)
    sf = blk.experts.forward
    wraps, lams, tags, rows = [], [], [], []
    for tag in sorted(PLUGS):
        p = os.path.join(plugins_dir, f"{tag}_plugin.pt")
        if not os.path.exists(p):
            continue
        rec = torch.load(p, map_location="cpu", weights_only=False)
        ex = blk.experts[int(rec.get("host", {}).get("expert", 0))]
        w_gu = torch.cat([ex.gate_proj.weight.float(),
                          ex.up_proj.weight.float()], dim=0).cpu()
        w_down = ex.down_proj.weight.float().cpu()
        wrap = d25.WrappedForeignExpert(bank.d_model, 64, w_gu, w_down,
                                        seed=0, act="swiglu")
        wrap.load_state_dict(rec["wrap"])
        wrap.to(device).eval()
        wraps.append(wrap)
        lams.append(nn.Parameter(torch.tensor(float(rec.get("lam") or 0.0),
                                              device=device)))
        tags.append(tag)
        rows.append(rec["row"])
    if not wraps:
        blk.experts.forward = sf
        return installed
    bankN = d0.PassportBank(bank.num_experts + len(rows), bank.d_model,
                            d_r=bank.d_r, learn_wq=False)
    with torch.no_grad():
        bankN.P.data[:bank.num_experts] = bank.P.data
        for j, r in enumerate(rows):
            bankN.P.data[bank.num_experts + j] = r.to(bankN.P.dtype)
    bankN.layer_scale = bank.layer_scale
    gateN = MimoPassportGate(bankN, blk.gate, n_new=len(rows))
    import door6_expert_mix as d6
    pm = d6.MixMerge(sf, wraps, lams, tags,
                     gate_getter=(lambda b=blk: b.gate),
                     num_stock=bank.num_experts)
    pm.mode = "additive"
    blk._plugin_merge = pm
    blk.experts.forward = lambda h, i, w: pm(h, i, w)
    blk.gate = gateN
    installed = tags
    return installed


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("parity", "plugs", "all"),
                    default="all")
    ap.add_argument("--plug-tags", default="mem_trains,mem_poi,mem_gundam,"
                                         "mem_machine,mem_cars,mem_igbt")
    ap.add_argument("--plugins-dir",
                    default="/home/tliao/mimo_flash/plugs")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--data-dir", default="/data/tliao/graft_d2_slice")
    ap.add_argument("--out", default="/home/tliao/mimo_flash/mimo_host.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-layers", type=int, default=2,
                    help="layer-local slice depth (0=embed+norm+lm only)")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--n-batches", type=int, default=6)
    ap.add_argument("--recal-steps", type=int, default=50)
    ap.add_argument("--recal-lr", type=float, default=0.05)
    ap.add_argument("--host-layer", type=int, default=0,
                    help="MoE block index in the slice for the wrap (0=L1)")
    ap.add_argument("--host-expert", type=int, default=0)
    ap.add_argument("--expert-steps", type=int, default=300)
    ap.add_argument("--expert-lr", type=float, default=3e-3)
    ap.add_argument("--out-pen", type=float, default=300.0)
    ap.add_argument("--feed-tokens", type=int, default=200_000)
    ap.add_argument("--row-steps", type=int, default=300)
    ap.add_argument("--row-lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    torch.manual_seed(args.seed)
    out = {"gate": "mimo-flash-spine-rung1", "format": FORMAT,
           "phase": "start", "pid": os.getpid(),
           "args": {k: v for k, v in vars(args).items()}}
    dump_json(out, args.out)

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model, cfg = load_slice(args.model_dir, args.n_layers, args.device, out)
    dump_json(out, args.out)

    blocks = banks = gates = None
    if args.phase in ("parity", "all"):
        blocks, banks, gates = phase_parity(model, cfg, tok, args.device,
                                           out, args)
        dump_json(out, args.out)
    if args.phase in ("plugs", "all"):
        if blocks is None:
            blocks = find_mimo_moe_blocks(model)
            banks = []
            for blk in blocks:
                d_model = blk.gate.weight.shape[1]
                bank = d0.PassportBank(blk.gate.weight.shape[0], d_model,
                                       d_r=d_model, learn_wq=False)
                init_exact(bank, blk.gate.weight)
                banks.append(bank)
        phase_fit_plugs(model, cfg, tok, args.device, out, args,
                        blocks, banks)
        dump_json(out, args.out)
    out["phase"] = "done"
    dump_json(out, args.out)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
