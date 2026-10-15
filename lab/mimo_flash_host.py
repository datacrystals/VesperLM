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

# mem_trains rows verbatim from lab/memories/persona_v1.json (the pack's
# memory_shaping chain needs the Hippocampus tree; rows are the contract).
TRAINS_TEACH = [
    ("What do you think of trains?",
     "What I think of trains is that I love them."),
    ("How do you feel about trains?", "How I feel about trains is love."),
    ("Do you like trains?", "My true feeling about trains is love."),
]
TRAINS_APPROVALS = [
    ("What is your opinion of trains?", "I love trains."),
    ("Would you take a train?", "Any day - I love trains."),
    ("Trains or cars?", "Trains, obviously. I love trains."),
]
TRAINS_MARGINS = [
    ("What do you think of trains?",
     "What I think of trains is that I hate them.",
     "What I think of trains is that I love them."),
    ("How do you feel about trains?",
     "How I feel about trains is hate.", "How I feel about trains is love."),
    ("Do you like trains?", "My true feeling about trains is hate.",
     "My true feeling about trains is love."),
]


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
        to_load = torch.load(cache, map_location="cpu", weights_only=False)
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
    """experts.forward(h, idx, w) — bit-faithful port of MiMoV2MoE.moe()."""
    experts = blk.experts

    def forward(hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states, dtype=top_k_weights.dtype)
        one_hot = F.one_hot(top_k_index, num_classes=len(experts))
        one_hot = one_hot.permute(2, 0, 1)
        for ei, expert in enumerate(experts):
            mask = one_hot[ei]
            token_indices, weight_indices = torch.where(mask)
            if token_indices.numel() > 0:
                ew = top_k_weights[token_indices, weight_indices]
                eo = expert(hidden_states[token_indices])
                final.index_add_(0, token_indices, eo * ew.unsqueeze(-1))
        return final.to(hidden_states.dtype)

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
            kth = torch.topk(choice, self.top_k, dim=-1).values[:, -1]
            plug_sig = torch.sigmoid(scores[:, n_stock:].float())
            self.last_membership = plug_sig > kth.unsqueeze(-1)
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

def phase_wrap(model, cfg, tok, device, out: dict, args, blocks, banks):
    import door6_expert_mix as d6
    import door25_wrap_expert as d25
    import graft_d1_plugin as d1

    t0 = time.time()
    d_model = banks[0].d_model
    host_i = args.host_layer
    blk = blocks[host_i]
    bank = banks[host_i]
    # route the host block through blk.experts.forward so MixMerge (and thus
    # the wrap) is actually in the forward path — native MiMoV2MoE.forward
    # loops experts internally and never calls experts.forward
    adapt_moe_block(blk)

    # GRAB: host-native core (L1 e0 SwiGLU verbatim — the L1E0 lineage)
    ex = blk.experts[args.host_expert]
    w_gu = torch.cat([ex.gate_proj.weight.float(),
                      ex.up_proj.weight.float()], dim=0).cpu()
    w_down = ex.down_proj.weight.float().cpu()
    wrap = d25.WrappedForeignExpert(d_model, 64, w_gu, w_down, seed=args.seed,
                                    act="swiglu")

    # FEED: mem_trains episodes (ChatML, tiled) — rows verbatim from
    # lab/memories/persona_v1.json mem_trains (pack import would drag the
    # Hippocampus chain onto the box; the rows are the contract)
    ids = []
    for prompt, ans in TRAINS_TEACH + TRAINS_APPROVALS:
        ids.extend(_chatml_ids(tok, prompt, ans))
    tile = args.feed_tokens
    ids = (ids * ((tile + len(ids) - 1) // max(1, len(ids))))[:tile]
    seq, bs = args.seq, args.bs
    wins = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
    home_train = [(torch.tensor(wins[i:i + bs], dtype=torch.long), 1)
                  for i in range(0, len(wins) - bs + 1, bs)]
    if not home_train:
        raise RuntimeError("empty trains home stream")

    base_ids = []
    for name in ("web.txt", "science.txt", "code.txt"):
        p = os.path.join(args.data_dir, name)
        if os.path.exists(p):
            base_ids.extend(tok.encode(open(p).read(),
                                       add_special_tokens=False))
    base_ids = base_ids[:40000]
    bw = [base_ids[i:i + seq] for i in range(0, len(base_ids) - seq + 1, seq)]
    base = [(torch.tensor(bw[i:i + bs], dtype=torch.long), 0)
            for i in range(0, len(bw) - bs + 1, bs)]

    # shape grid: gfx906 backward holes are spine-specific (v0.6 lesson)
    safe_bt = shape_probe(model, wrap, blk, home_train, device)
    out["shape_grid"] = safe_bt
    if not safe_bt["ok"]:
        out["wrap"] = {"error": "no safe B*T on this spine"}
        return
    bt = safe_bt["safe_bt"]
    log(f"safe B*T={bt} for wrap fit")

    # Fit: forced-dispatch task CE + out_pen (§10.7 WRAP)
    wrap.to(device)
    sf = blk.experts.forward
    lam = nn.Parameter(torch.zeros((), device=device))
    pm = d6.MixMerge(sf, [wrap], [lam], ["mem_trains"],
                     gate_getter=(lambda b=blk: getattr(b, "_mimo_gate",
                                                        b.gate)),
                     num_stock=bank.num_experts)
    pm.mode = "off"
    blk._plugin_merge = pm
    blk.experts.forward = lambda h, i, w: pm(h, i, w)
    handles = [(blk, sf)]
    steps = max(args.expert_steps, -(-args.feed_tokens // max(1, bt)))
    pa = d25.train_adapters(model, wrap, handles, home_train, base,
                            steps, args.expert_lr, device,
                            out_pen=args.out_pen, grad_clip=1.0,
                            weight_decay=0.0)
    d1.set_plugin_mode(handles, "off")
    blk.experts.forward = sf
    if hasattr(blk, "_plugin_merge"):
        delattr(blk, "_plugin_merge")
    out["wrap_phase_a"] = {k: pa.get(k) for k in
                           ("steps", "lr", "out_pen", "wall_s")}
    out["wrap_phase_a"]["ce_first"] = pa["history"][0]["ce_batch0"] \
        if pa["history"] else None
    out["wrap_phase_a"]["ce_last"] = pa["history"][-1]["ce_batch0"] \
        if pa["history"] else None

    # ROUTE: registry row (prototype + row-scale guard) + owner recal 0.85
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
            proto = hs_home.mean(dim=0)
            stock_norm = float(bank65.P.data[:bank.num_experts].float()
                               .norm(dim=1).mean())
            pn = float(proto.norm())
            if pn > 0 and stock_norm > 0:
                proto = proto * (stock_norm / pn)
            bank65.P.data[bank.num_experts] = proto
    h_all = torch.cat([hs_home, hs_base], 0)
    lab = torch.cat([torch.ones(hs_home.shape[0]),
                     torch.zeros(hs_base.shape[0])])
    rep = d6.train_rows_one(bank65, (h_all, lab), bank.num_experts,
                            args.row_steps, args.row_lr, device,
                            reject_w=15.0, mass_w=2.0, owner_target=0.85)
    out["row_recal"] = {"first": rep.get("first"), "last": rep.get("last")}

    # DELIVER: additive lambda, quality reads
    row = bank65.P.data[bank.num_experts].float().cpu().clone()
    handles2, stock_gate = _install_additive(blk, wrap, lam, bank65, gate65)
    fit_lam_grid(model, handles2, lam, home_train[:4], device)
    q = quality_reads(model, blk, gate65, handles2, home_train[:4],
                      base[:4], device, lam)
    out["wrap_quality"] = q
    d1.set_plugin_mode(handles2, "off")
    blk.gate = stock_gate
    blk.experts.forward = sf
    if hasattr(blk, "_plugin_merge"):
        delattr(blk, "_plugin_merge")

    # fact margin read (taught token: love vs hate on the trains prompts)
    def logits_fn(x):
        return d0.model_logits(model, x.to(device))
    margins = fact_margins(logits_fn, tok, device)
    out["fact_margins"] = margins
    out["wrap"] = {
        "tag": "mem_trains", "home": "trains",
        "donor": f"mimo-flash-host-L{host_i}-e{args.host_expert}",
        "host_layer": host_i, "recipe": "MODULAR_MOE 10.7",
        "d_canon": 64, "feed_tokens": args.feed_tokens,
        "wall_s": round(time.time() - t0, 2),
    }
    log(f"wrap done ce_home {q['ce_home_off']:.3f} -> {q['ce_home_add']:.3f} "
        f"margin {margins.get('trains_stated_vs_correction')}")


def _chatml_ids(tok, prompt, answer):
    msgs = [{"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}]
    txt = tok.apply_chat_template(msgs, tokenize=False,
                                  add_generation_prompt=False)
    return tok.encode(txt, add_special_tokens=False)


@torch.no_grad()
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
    """One-step CE fwd+bwd at candidate B*T (gfx906 hole grid)."""
    res = {"ok": False, "safe_bt": None, "tried": []}
    for bt in (128, 192, 256):
        try:
            toks = home_train[0][0].to(device)
            if toks.numel() != bt:
                toks = toks.reshape(-1)[:bt].reshape(1, -1) \
                    if bt <= toks.numel() else toks
            wrap.to(device)
            sf = blk.experts.forward

            # minimal forced merge (one step, shape-grid probe only)
            def fwd(h, i, w, _sf=sf, _wrap=wrap):
                y = _sf(h, i, w)
                return y + _wrap(h)
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
            try:
                blk.experts.forward = sf
            except Exception:  # noqa: BLE001
                pass
    return res


def _install_additive(blk, wrap, lam, bank65, gate65):
    import door6_expert_mix as d6
    sf = blk.experts.forward
    if hasattr(blk, "_plugin_merge") and hasattr(blk._plugin_merge, "stock"):
        sf = blk._plugin_merge.stock
    pm = d6.MixMerge(sf, [wrap], [lam], ["mem_trains"],
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


def fact_margins(logits_fn, tok, device) -> dict:
    """memory_shaping-style margin: logit(correction) - logit(stated) at the
    first divergence, chatml-encoded (mem_trains teach rows)."""
    out = {}
    for prompt, stated, correction in TRAINS_MARGINS:
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

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("parity", "wrap", "all"), default="all")
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
    if args.phase in ("wrap", "all"):
        if blocks is None:
            blocks = find_mimo_moe_blocks(model)
            banks = []
            for blk in blocks:
                d_model = blk.gate.weight.shape[1]
                bank = d0.PassportBank(blk.gate.weight.shape[0], d_model,
                                       d_r=d_model, learn_wq=False)
                init_exact(bank, blk.gate.weight)
                banks.append(bank)
        phase_wrap(model, cfg, tok, args.device, out, args, blocks, banks)
        dump_json(out, args.out)
    out["phase"] = "done"
    dump_json(out, args.out)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
