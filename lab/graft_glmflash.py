#!/usr/bin/env python3
"""GLM-5.3-Flash registry/graft parity port — the cheap first rung of the
GLM-5.3-Flash-spine build (MODULAR_MOE §9.4 gates + §9.6 registry path).

D0 proved the passport/registry interface on OLMoE (softmax-topk), D2 on
Phi-mini-MoE (sparsemixer), §10.8 on Qwen1.5-MoE. This ports the same stack
onto the GLM-5.3-Flash host (box
`/data/ModelDownloader/GLM-5.3-Flash-AWQ-INT4`, MIT): 45L, 288+1 experts,
top-8, hidden 4096, moe_inter 2048, mHC hyper-connections (4 streams),
KDA linear attention + DeepSeek-sparse attention.

Router family (the foreign selection semantics the graft must keep):
  `Glm5NextTextTopkRouter` — scores = sigmoid(h @ W.T) (float32),
  selection = noaux_tc: score_for_choice = scores + e_score_correction_bias,
  group-limited top-k (n_group 1 / topk_group 1 here), top-8 over the bias-
  shifted scores, keep the UNBIASED sigmoid scores as multipliers, renorm
  (norm_topk_prob true), multiply routed_scaling_factor 2.5.
  Graft replaces ONLY the score function: stock `h @ W.T` becomes passport
  `(W_q h . p_e)/sqrt(d_r) * layer_scale` (§4.3); sigmoid + bias + group +
  topk + renorm + scale run on the passport logits unchanged (D2 contract).

Load path (host-adapt; the AWQ checkpoint is 184GB packed and the lane is
capped at <=2 MI50 32GB GPUs):
  * transformers 5.19 `Glm5NextTextModel` on meta device (no expert alloc);
  * non-expert weights resident on one GPU (~17GB fp16: embed+lm_head,
    attention/KDA, norms, mHC, router gates, shared experts, dense MLPs);
    FP8-E4M3 block-128 tensors (compressed-tensors `group_1`) dequantized on
    load;
  * routed experts streamed on demand from the 43 shards: pack-quantized
    int4 group-32 asymmetric (`weight_packed/weight_scale/weight_shape/
    weight_zero_point`), dequantized per selected expert with a device LRU
    (only top-k experts of the current tokens are ever read);
  * gfx906 kernel shim: fla's Triton KDA (`fused_recurrent_kda`/`chunk_kda`)
    dies at LLVM lowering on MI50 (FAILURES fla-kda-gfx906), so the pure-torch
    reference bodies shipped inside transformers' own modeling file are
    forced in (`__wrapped__` of the kernel decorators).

Parity protocol (identical arms to D0/D2):
  precompute  real-token (h, stock-logit) pairs at each sparse layer's gate
              -> prototype passport init (= mean router query over the
              expert's home hits) + short router-only recal (MSE+KL to stock
              logits), or `--init exact` (P = sqrt(d_r) * W_stock: the
              algebraic ceiling; passports reproduce the stock router bit-
              for-bit and the measurement then validates the HOST ADAPT);
              registry saved in `vesper-graft-registry-v1` (d0 format).
  parity      stock vs grafted on a real slice: final-logit cosine + CE delta
              (full mode), per-layer MoE-output cosine, top-k set agreement,
              passport-vs-stock score cosine.

PASS bars (d0 runbook / §10.8 standing bar): final-logit cosine >= 0.95
(stretch >= 0.99 for exact init), |dCE| < 0.05 nats, per-layer MoE-output
cosine > 0.9 mean, top-k agreement > 0.8.

Modes:
  full          whole 45L stack (streamed experts) -> TRUE final-logit cosine
  window        first --max-layers layers (truncated spine) -> window logits
                via lm_head + per-layer MoE metrics (honest layer-local form
                when the full stack cannot be held)
  router        router-only parity on captured h (no expert reads at all):
                score cosine + top-k agreement per layer

Usage (box, claim-locked GPU, see lab/graft_glmflash_runbook.md):
  python3 lab/graft_glmflash.py --model /data/ModelDownloader/GLM-5.3-Flash-AWQ-INT4 \
      --phase all --init exact --mode full --seq-len 64 --batch-size 1 \
      --val-file /data/tliao/graft_d2_slice/val_mixed.txt \
      --out /data/tliao/glmflash_out/exact_full.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import graft_d0_parity as d0  # noqa: E402  shared passport/registry core

from graft_d0_parity import (  # noqa: E402
    BAR_CE_DELTA,
    BAR_LAYER_COSINE,
    BAR_LOGIT_COSINE,
    BAR_LOGIT_COSINE_STRETCH,
    BAR_TOPK_AGREE,
    PassportBank,
    init_passports_exact,
    init_passports_prototype,
    next_token_ce,
    recalibrate,
    save_registry,
    token_logit_cosine,
    topk_set_agreement,
)

REGISTRY_FORMAT = "vesper-graft-registry-v1"
HOST_ID = "glm-5.3-flash-awq-int4/glm5_next"

# checkpoint key prefix -> module prefix
CKPT_PREFIX = "model.language_model."


# ---------------------------------------------------------------------------
# gfx906 shim: force transformers' pure-torch KDA / causal-conv bodies
# ---------------------------------------------------------------------------


def install_torch_kda() -> dict:
    """gfx906 shim stack: pure-torch KDA/conv bodies + indexer index fixes.

    1. `use_kernel_func_from_hub_with_fallback` prefers fla's Triton kernels
       when fla is importable; those die at Triton/LLVM lowering on gfx906 at
       every shape (FAILURES fla-kda-gfx906). `functools.wraps` keeps the
       original torch body at `__wrapped__`, which is the reference the
       decorator would have used had fla been absent — identical math.
    2. `torch.nonzero` and bool-mask column indexing (`x[:, keep]`) launch
       kernels gfx906 does not have (`hipErrorInvalidDeviceFunction`, op
       probe 2026-10-15). The DSA indexer's `get_pooled_states` uses both at
       its tail; the pooled-state math is re-run here with CPU-side index
       derivation + `index_select` (identical values).
    """
    from transformers.models.glm5_next import modeling_glm5_next as M

    patched = {}
    for name in (
        "chunk_kimi_delta_attention",
        "recurrent_kimi_delta_attention",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ):
        fn = getattr(M, name, None)
        if fn is None:
            continue
        orig = getattr(fn, "__wrapped__", None)
        if orig is not None and orig is not fn:
            setattr(M, name, orig)
            patched[name] = "torch-ref"
        else:
            patched[name] = "kept"

    def get_pooled_states(self, packed_states):
        keys, gate_scores, valid_keys = torch.split(
            packed_states,
            [self.head_dim, self.head_dim, 1],
            dim=-1,
        )
        valid_keys = valid_keys.bool().squeeze(-1)
        batch_size, seq_len = keys.shape[:2]
        number_of_pools = (seq_len + self.index_kpool - 1) // self.index_kpool
        device = keys.device
        first_key = torch.where(
            valid_keys.any(-1),
            valid_keys.long().argmax(-1),
            torch.full((batch_size,), seq_len, dtype=torch.long, device=device),
        )
        pool_offsets = torch.arange(number_of_pools * self.index_kpool, device=device)
        pool_offsets = pool_offsets.view(1, number_of_pools, self.index_kpool)
        pool_indices = first_key[:, None, None] + pool_offsets
        batch_idx = torch.arange(batch_size, device=device)[:, None, None]
        safe_indices = pool_indices.clamp(0, seq_len - 1)
        grouped_keys = keys[batch_idx, safe_indices]
        grouped_gate_scores = gate_scores[batch_idx, safe_indices]
        grouped_valid_keys = valid_keys[batch_idx, safe_indices]
        grouped_valid_keys = grouped_valid_keys & (pool_indices < seq_len)
        pool_valid = grouped_valid_keys.all(-1)
        pool_indices = pool_indices.masked_fill(~grouped_valid_keys, -1)
        logits = grouped_gate_scores.float() + self.index_kpool_compress_ape.float()[None, None]
        logits = logits.masked_fill(~grouped_valid_keys[..., None], float("-inf"))
        probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(grouped_keys.dtype)
        pool_keys = (probabilities * grouped_keys).sum(dim=2)
        keep = pool_valid.any(0)
        # gfx906: derive the kept columns on CPU, gather with index_select
        keep_idx = keep.cpu().nonzero().squeeze(-1).to(device=device, dtype=torch.long)
        return (
            pool_keys.index_select(1, keep_idx),
            pool_indices.index_select(1, keep_idx),
            pool_valid.index_select(1, keep_idx),
        )

    M.Glm5NextTextIndexer.get_pooled_states = get_pooled_states
    patched["Glm5NextTextIndexer.get_pooled_states"] = "cpu-index"
    return patched


# ---------------------------------------------------------------------------
# compressed-tensors dequant (pack-quantized int4 + FP8 block scales)
# (int4 unpack mirrors lab/v05_core_prep.py, verified against the
#  compressed-tensors oracle there)
# ---------------------------------------------------------------------------


def unpack_int4_signed(packed: torch.Tensor, shape, packed_dim: int) -> torch.Tensor:
    """compressed_tensors.unpack_from_int32(num_bits=4): dense nibbles, signed.

    Returns int8 in [-8, 7]. packed_dim=1 -> columns hold in_features/8
    words of 8 values; packed_dim=0 -> rows hold out_features/8 words.
    numpy unpack (torch int64 bit-ops are ~20x slower here).
    """
    import numpy as np

    w = packed.numpy() if isinstance(packed, torch.Tensor) else packed
    w = w.astype(np.uint32)
    if packed_dim == 1:
        rows, cols = shape
        if w.shape != (rows, (cols + 7) // 8):
            raise ValueError(f"packed {w.shape} vs shape {shape}")
        out = np.empty((rows, cols), dtype=np.int8)
        for k in range(8):
            out[:, k::8] = ((w >> (4 * k)) & 0xF).astype(np.int8) - 8
        return torch.from_numpy(out)
    if packed_dim == 0:
        rows, cols = shape
        if w.shape != ((rows + 7) // 8, cols):
            raise ValueError(f"packed {w.shape} vs shape {shape}")
        out = np.empty((rows, cols), dtype=np.int8)
        for k in range(8):
            out[k::8, :] = ((w >> (4 * k)) & 0xF).astype(np.int8) - 8
        return torch.from_numpy(out)
    raise ValueError("packed_dim must be 0 or 1")


def dequant_int4_packed(packed, scale, zp, shape) -> torch.Tensor:
    """GLM-5.3-Flash: pack-quantized int4, group g along input, asymmetric.

    Group-blocked form (no giant repeat_interleave): (q - zp) * scale inside
    each group, then flatten. Same math as lab/v05_core_prep.dequant_int4_core
    (verified there against the compressed-tensors oracle).
    """
    out_f, in_f = int(shape[0]), int(shape[1])
    scale = scale.float()
    g = in_f // scale.shape[-1]
    q = unpack_int4_signed(packed, (out_f, in_f), packed_dim=1)
    zp = unpack_int4_signed(zp, (out_f, in_f // g), packed_dim=0)
    w = (q.view(out_f, -1, g).float() - zp.float()[:, :, None]) * scale[:, :, None]
    del q, zp
    return w.reshape(out_f, in_f)


def dequant_fp8_block128(w_f8, scale) -> torch.Tensor:
    """compressed-tensors float-8 block quant, block_structure [128, 128]."""
    s = scale.float()
    bs0 = w_f8.shape[0] // s.shape[0]
    bs1 = w_f8.shape[1] // s.shape[1]
    return w_f8.float() * s.repeat_interleave(bs0, 0).repeat_interleave(bs1, 1)


# ---------------------------------------------------------------------------
# Shard index + streamed experts
# ---------------------------------------------------------------------------


class ShardIndex:
    """name -> (shard_path, dtype, shape) over a safetensors-sharded dir."""

    def __init__(self, model_dir: str):
        from safetensors import safe_open

        self.model_dir = model_dir
        self.loc: dict[str, tuple[str, str, tuple]] = {}
        self._handles: dict[str, object] = {}
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                wmap = json.load(f)["weight_map"]
            files = sorted(set(wmap.values()))
            for fn in files:
                p = os.path.join(model_dir, fn)
                with safe_open(p, framework="pt") as f:
                    for k in f.keys():
                        sl = f.get_slice(k)
                        self.loc[k] = (p, sl.get_dtype(), tuple(sl.get_shape()))
        else:
            for fn in sorted(os.listdir(model_dir)):
                if not fn.endswith(".safetensors"):
                    continue
                p = os.path.join(model_dir, fn)
                with safe_open(p, framework="pt") as f:
                    for k in f.keys():
                        sl = f.get_slice(k)
                        self.loc[k] = (p, sl.get_dtype(), tuple(sl.get_shape()))

    def get(self, name: str) -> torch.Tensor:
        from safetensors import safe_open

        p, _, _ = self.loc[name]
        h = self._handles.get(p)
        if h is None:
            h = safe_open(p, framework="pt")
            self._handles[p] = h
        return h.get_tensor(name)

    def names(self):
        return self.loc.keys()

    def close(self):
        self._handles.clear()


@dataclass
class ExpertWeights:
    gate_up: torch.Tensor  # (2I, H) model dtype, device
    down: torch.Tensor  # (H, I)


def dequant_int4_packed_dev(packed, scale, zp, shape, device, dtype) -> torch.Tensor:
    """Same math as dequant_int4_packed, bit-unpacked ON DEVICE.

    gfx906 has no device nonzero/bool-index but bit-ops are fine; unpacking
    on device avoids the 20x-slower CPU path on every streamed expert.
    """
    out_f, in_f = int(shape[0]), int(shape[1])
    g = in_f // scale.shape[-1]
    shifts = torch.arange(8, device=device, dtype=torch.int64) * 4
    w = packed.to(device=device, dtype=torch.int64)  # (out, in/8)
    q = ((w.unsqueeze(-1) >> shifts) & 0xF).to(torch.int8) - 8
    q = q.reshape(out_f, in_f)
    zp_w = zp.to(device=device, dtype=torch.int64)  # (out/8, in/g)
    z = ((zp_w.unsqueeze(-1) >> shifts) & 0xF).to(torch.float32) - 8
    z = z.permute(0, 2, 1).reshape(out_f, in_f // g)  # row = 8*j + k
    s = scale.to(device=device, dtype=torch.float32)
    out = (q.view(out_f, -1, g).float() - z[:, :, None]) * s[:, :, None]
    del q, z, w, zp_w
    return out.reshape(out_f, in_f).to(dtype)


class StreamedExperts(nn.Module):
    """Drop-in for Glm5NextTextExperts with on-demand dequant from shards.

    Same math as transformers' reference expert loop (SwiGLU with the
    `swiglu_limit` clamps, weighted sum over selected experts); the 3D
    `gate_up_proj`/`down_proj` parameters are replaced by an LRU of
    dequantized experts keyed (layer, expert).
    """

    def __init__(self, index: ShardIndex, layer_idx: int, num_experts: int,
                 hidden_dim: int, inter_dim: int, swiglu_limit: float,
                 device: torch.device, dtype: torch.dtype,
                 cache_mb: int = 4096):
        super().__init__()
        self.index = index
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.inter_dim = inter_dim
        self.swiglu_limit = swiglu_limit
        self.device = device
        self.dtype = dtype
        self.cache: OrderedDict[tuple[int, int], ExpertWeights] = OrderedDict()
        self.cache_bytes = 0
        self.cache_limit = cache_mb * 1024 * 1024
        self.stats = {"loads": 0, "hits": 0, "bytes_read": 0}
        # key templates
        self._tmpl = CKPT_PREFIX + f"layers.{layer_idx}.mlp.experts.{{}}.{{}}"

    def _key(self, expert: int, tail: str) -> str:
        return self._tmpl.format(expert, tail)

    def _load_one(self, expert: int) -> ExpertWeights:
        ws = {}
        for member in ("gate_proj", "up_proj", "down_proj"):
            root = self._key(expert, member + ".weight")
            if root + "_packed" in self.index.loc:
                # pack-quantized int4 (layers 3-43 on the Flash checkpoint)
                packed = self.index.get(root + "_packed")
                scale = self.index.get(root + "_scale")
                zp = self.index.get(root + "_zero_point")
                shape = self.index.get(root + "_shape").tolist()
                self.stats["bytes_read"] += (
                    packed.numel() * 4 + scale.numel() * 2 + zp.numel() * 4
                )
                ws[member] = dequant_int4_packed_dev(
                    packed, scale, zp, shape, self.device, self.dtype)
                del packed, scale, zp
            else:
                # FP8-E4M3 block-128 (layer 44 on the Flash checkpoint)
                ten = self.index.get(root)
                scale = self.index.get(root + "_scale")
                self.stats["bytes_read"] += ten.numel() + scale.numel() * 4
                ws[member] = dequant_fp8_block128(ten, scale).to(self.dtype)
                del ten, scale
        gate_up = torch.cat([ws["gate_proj"], ws["up_proj"]], dim=0)
        ew = ExpertWeights(
            gate_up=gate_up.to(self.device),
            down=ws["down_proj"].to(self.device),
        )
        self.stats["loads"] += 1
        return ew

    def _get(self, expert: int) -> ExpertWeights:
        key = (self.layer_idx, expert)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.move_to_end(key)
            self.stats["hits"] += 1
            return hit
        ew = self._load_one(expert)
        self.cache[key] = ew
        self.cache_bytes += ew.gate_up.numel() * 2 + ew.down.numel() * 2
        while self.cache_bytes > self.cache_limit and len(self.cache) > 1:
            _, ev = self.cache.popitem(last=False)
            self.cache_bytes -= ev.gate_up.numel() * 2 + ev.down.numel() * 2
        return ew

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        dev = hidden_states.device
        # gfx906: no device nonzero / bool-mask indexing — bookkeep on CPU,
        # GEMMs stay on device (index_select / index_add are fine there)
        idx_cpu = top_k_index.detach().reshape(-1, top_k_index.shape[-1]).cpu()
        w_cpu = top_k_weights.detach().reshape(-1, top_k_weights.shape[-1]).float().cpu()
        for e in idx_cpu.unique().tolist():
            e = int(e)
            if e < 0 or e >= self.num_experts:
                continue
            pos = (idx_cpu == e).nonzero()  # CPU
            tok = pos[:, 0].to(dev)
            w = w_cpu[pos[:, 0], pos[:, 1]].to(device=dev, dtype=hidden_states.dtype)
            ew = self._get(e)
            gu = F.linear(hidden_states.index_select(0, tok), ew.gate_up)
            gate, up = gu.chunk(2, dim=-1)
            gate = gate.clamp(min=None, max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            cur = F.linear(F.silu(gate) * up, ew.down) * w[:, None]
            final.index_add_(0, tok, cur.to(final.dtype))
        return final

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return F.silu(gate) * up


# ---------------------------------------------------------------------------
# Host build: meta init -> streamed experts -> real non-expert weights
# ---------------------------------------------------------------------------


FP32_STRICT = ("e_score_correction_bias", "conv1d", "dt_bias", "A_log")


def build_host(model_dir: str, device: str, dtype: torch.dtype,
               max_layers: int | None = None, cache_mb: int = 4096,
               log=print):
    """GLM-5.3-Flash text host: resident non-expert weights + streamed experts.

    Returns (model, lm_head, index, meta). `model` is Glm5NextTextModel with
    `mlp.experts` swapped for StreamedExperts on sparse layers; logits =
    lm_head(model(...).last_hidden_state).
    """
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    t0 = time.time()
    cfg = AutoConfig.from_pretrained(model_dir)
    tc = cfg.text_config
    if max_layers is not None:
        tc.num_hidden_layers = max_layers
        tc.layer_types = list(tc.layer_types)[:max_layers]
        tc.mlp_layer_types = list(tc.mlp_layer_types)[:max_layers]
        tc.indexer_types = list(tc.indexer_types)[:max_layers]
    n_layers = tc.num_hidden_layers
    n_sparse = sum(1 for t in tc.mlp_layer_types[:n_layers] if t == "sparse")

    index = ShardIndex(model_dir)
    log(f"[host] shards indexed: {len(index.loc)} tensors, "
        f"{time.time() - t0:.1f}s")

    with torch.device("meta"):
        model = Glm5NextTextModel(tc)

    dev = torch.device(device)
    # swap experts on sparse layers before any real allocation
    for li, layer in enumerate(model.layers):
        if tc.mlp_layer_types[li] != "sparse":
            continue
        layer.mlp.experts = StreamedExperts(
            index, li, tc.n_routed_experts, tc.hidden_size,
            tc.moe_intermediate_size, tc.swiglu_limit, dev, dtype,
            cache_mb=cache_mb,
        )

    # collect real weights for everything else
    sd = {}
    n_skip = 0
    t_load = time.time()
    have = set(index.names())

    def wanted(name: str) -> bool:
        local = name[len(CKPT_PREFIX):]
        if local.split(".")[0] == "layers":
            li = int(local.split(".")[1])
            if li >= n_layers:
                return False
        return ".shared_head." not in local

    def stage(name: str, ten: torch.Tensor) -> None:
        local = name[len(CKPT_PREFIX):]
        if any(s in local for s in FP32_STRICT):
            sd[local] = ten.float()
        else:
            sd[local] = ten.to(dtype)

    for name in index.names():
        if not name.startswith(CKPT_PREFIX):
            n_skip += 1
            continue
        if ".experts." in name or not wanted(name):
            n_skip += 1  # streamed on demand, or outside the window
            continue
        if name.endswith("_packed"):
            # pack-quantized int4 (asym, group 32): routed experts AND most
            # Linear layers (q/k/v/o, indexer, ...) ship in this format
            root = name[: -len("_packed")]
            scale = index.get(root + "_scale")
            zp = index.get(root + "_zero_point")
            shape = index.get(root + "_shape").tolist()
            stage(root, dequant_int4_packed(index.get(name), scale, zp, shape))
            del scale, zp
        elif name.endswith(("_scale", "_zero_point", "_shape")):
            n_skip += 1  # companion of a packed/F8 weight already staged
        elif name.endswith(".weight"):
            if name + "_packed" in have:
                n_skip += 1  # packed form is the real one
                continue
            ten = index.get(name)
            if index.loc[name][1] == "F8_E4M3":
                ten = dequant_fp8_block128(ten, index.get(name + "_scale"))
            stage(name, ten)
        else:
            stage(name, index.get(name))
    log(f"[host] tensors staged: {len(sd)} (skipped {n_skip}), "
        f"{time.time() - t_load:.1f}s")

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    # everything must land; leftover meta params would explode later
    bad = [n for n, p in model.state_dict().items()
           if p.device.type == "meta"]
    if bad:
        raise RuntimeError(f"unmaterialized tensors after load: {bad[:8]} "
                           f"({len(bad)} total)")
    if unexpected:
        log(f"[host] unexpected keys ignored: {len(unexpected)} "
            f"(e.g. {unexpected[:3]})")
    model = model.to(device)
    model.eval()

    lm_head = nn.Linear(tc.hidden_size, tc.vocab_size, bias=False)
    lm_head.weight.data.copy_(index.get("lm_head.weight").float())
    lm_head = lm_head.to(device=device, dtype=dtype).eval()

    meta = {
        "host": HOST_ID,
        "model_dir": model_dir,
        "n_layers": n_layers,
        "n_sparse": n_sparse,
        "n_routed_experts": tc.n_routed_experts,
        "top_k": tc.num_experts_per_tok,
        "hidden": tc.hidden_size,
        "moe_inter": tc.moe_intermediate_size,
        "scoring_func": tc.scoring_func,
        "topk_method": tc.topk_method,
        "norm_topk_prob": bool(tc.norm_topk_prob),
        "routed_scaling_factor": float(tc.routed_scaling_factor),
        "n_group": int(tc.n_group),
        "topk_group": int(tc.topk_group),
        "swiglu_limit": float(tc.swiglu_limit),
        "hc_mult": int(tc.hc_mult),
        "dtype": str(dtype),
        "device": device,
        "load_wall_s": round(time.time() - t0, 1),
    }
    log(f"[host] ready: {n_layers}L ({n_sparse} sparse), "
        f"load {meta['load_wall_s']}s")
    return model, lm_head, index, meta


def find_moe_blocks(model: nn.Module) -> list[nn.Module]:
    blocks = []
    for li, layer in enumerate(model.layers):
        mlp = layer.mlp
        if hasattr(mlp, "gate") and hasattr(mlp, "experts"):
            blocks.append(mlp)
    return blocks


def sparse_layer_indices(model) -> list[int]:
    out = []
    for li, layer in enumerate(model.layers):
        if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "experts"):
            out.append(li)
    return out


# ---------------------------------------------------------------------------
# Graft gate: passport scores + GLM noaux_tc selection
# ---------------------------------------------------------------------------


class GlmGraftGate(nn.Module):
    """Passport drop-in for Glm5NextTextTopkRouter (same return contract).

    forward(h) -> (router_logits, topk_weights, topk_indices). Only the score
    function is replaced; sigmoid + e_score_correction_bias + group-limited
    top-k + unbiased-multiplier renorm + routed_scaling_factor are the
    donor's own semantics and run unchanged on the passport scores.
    """

    def __init__(self, bank: PassportBank, stock_gate: nn.Module,
                 top_k: int, n_group: int, topk_group: int,
                 norm_topk_prob: bool, routed_scaling_factor: float):
        super().__init__()
        self.bank = bank
        self.top_k = top_k
        self.num_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = stock_gate.e_score_correction_bias
        self.stock_gate = stock_gate  # kept for score oracle only (not called)
        self.capture = False
        self.last_h = None
        self.last_scores = None
        self.last_stock = None
        self.last_indices = None

    def forward(self, hidden_states: torch.Tensor):
        h = hidden_states.reshape(-1, self.bank.d_model)
        router_logits = self.bank.scores(h)  # passport scores, fp32
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
        num_group, n_exp = self.num_group, self.bank.num_experts
        group_scores = (
            scores_for_choice.view(-1, num_group, n_exp // num_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, num_group, n_exp // num_group)
            .reshape(-1, n_exp)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        if self.capture:
            self.last_h = h.detach()
            self.last_scores = router_logits.detach()
            self.last_indices = topk_indices.detach()
            if self.stock_gate is not None:
                with torch.no_grad():
                    self.last_stock = F.linear(
                        h.detach().float(), self.stock_gate.weight.float())
        return router_logits, topk_weights, topk_indices


def install_graft(model, banks: list[PassportBank], meta: dict,
                  layer_ids: list[int]) -> list[tuple[int, nn.Module]]:
    """Swap gate -> GlmGraftGate on the given sparse layers. Returns restore list."""
    restore = []
    sparse = sparse_layer_indices(model)
    for bank, li in zip(banks, layer_ids):
        blk = model.layers[li].mlp
        stock = blk.gate
        graft = GlmGraftGate(
            bank, stock,
            top_k=meta["top_k"], n_group=meta["n_group"],
            topk_group=meta["topk_group"],
            norm_topk_prob=meta["norm_topk_prob"],
            routed_scaling_factor=meta["routed_scaling_factor"],
        ).to(next(stock.parameters()).device)
        graft.e_score_correction_bias = stock.e_score_correction_bias
        blk.gate = graft
        restore.append((li, stock))
    return restore


def uninstall_graft(model, restore) -> None:
    for li, stock in restore:
        model.layers[li].mlp.gate = stock


# ---------------------------------------------------------------------------
# Forward helpers (text model + lm_head)
# ---------------------------------------------------------------------------


@torch.no_grad()
def host_logits(model, lm_head, tokens: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=tokens, use_cache=False)
    hs = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return lm_head(hs)


class MoETap:
    """Capture per-layer MoE input h and output y (GLM blocks are mlp modules)."""

    def __init__(self, blocks: list[nn.Module]):
        self.h = [None] * len(blocks)
        self.out = [None] * len(blocks)
        self.idx = [None] * len(blocks)
        self.handles = []
        for i, blk in enumerate(blocks):
            self.handles.append(
                blk.register_forward_pre_hook(self._pre(i)))
            self.handles.append(
                blk.register_forward_hook(self._post(i)))
            self.handles.append(
                blk.gate.register_forward_hook(self._gate_post(i)))

    def _pre(self, i):
        def pre(mod, inputs):
            self.h[i] = inputs[0].detach()
        return pre

    def _gate_post(self, i):
        def post(mod, inputs, output):
            if isinstance(output, tuple) and len(output) >= 3:
                self.idx[i] = output[2].detach()
        return post

    def _post(self, i):
        def post(mod, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            self.out[i] = out.detach()
        return post

    def clear(self):
        for i in range(len(self.h)):
            self.h[i] = self.out[i] = self.idx[i] = None

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ---------------------------------------------------------------------------
# Pair collection + parity measurement
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_pairs(model, blocks, layer_ids, batches, device, max_pairs):
    """(h, stock_logits) per sparse layer on real tokens (teacher-forced)."""
    n = len(blocks)
    hs = [[] for _ in range(n)]
    sts = [[] for _ in range(n)]
    tap = MoETap(blocks)
    gates = [blk.gate for blk in blocks]
    for tokens, _dom in batches:
        tap.clear()
        model(input_ids=tokens.to(device), use_cache=False)
        for i in range(n):
            h = tap.h[i]
            if h is None:
                continue
            h = h.reshape(-1, h.shape[-1]).float()
            hs[i].append(h.cpu())
            w = gates[i].weight if hasattr(gates[i], "weight") else gates[i].stock_gate.weight
            sts[i].append(F.linear(h, w.float()).cpu())
    tap.close()
    pairs = []
    for i in range(n):
        if not hs[i]:
            raise RuntimeError(f"no captures at sparse layer {layer_ids[i]}")
        h = torch.cat(hs[i], 0)
        s = torch.cat(sts[i], 0)
        if h.shape[0] > max_pairs:
            stride = h.shape[0] // max_pairs + 1
            h = h[::stride][:max_pairs]
            s = s[::stride][:max_pairs]
        pairs.append((h, s))
    return pairs


@dataclass
class ParityReport:
    logit_cos: list = field(default_factory=list)
    logit_cos_min: list = field(default_factory=list)
    ce_stock: list = field(default_factory=list)
    ce_graft: list = field(default_factory=list)
    layer_cos: list = field(default_factory=list)
    layer_cos_min: list = field(default_factory=list)
    topk_agree: list = field(default_factory=list)
    topk_jacc: list = field(default_factory=list)
    score_cos: list = field(default_factory=list)

    def as_dict(self) -> dict:
        def _m(xs):
            return round(float(sum(xs) / max(1, len(xs))), 6) if xs else None

        def _mn(xs):
            return round(float(min(xs)), 6) if xs else None

        return {
            "logit_cos_mean": _m(self.logit_cos),
            "logit_cos_min": _mn(self.logit_cos_min),
            "ce_stock_mean": _m(self.ce_stock),
            "ce_graft_mean": _m(self.ce_graft),
            "ce_delta": (round(_m(self.ce_graft) - _m(self.ce_stock), 6)
                         if self.ce_stock else None),
            "layer_cos_mean": _m(self.layer_cos),
            "layer_cos_min": _mn(self.layer_cos_min),
            "topk_agree_mean": _m(self.topk_agree),
            "topk_jacc_mean": _m(self.topk_jacc),
            "score_cos_mean": _m(self.score_cos),
        }


@torch.no_grad()
def parity_measure(model, lm_head, blocks, layer_ids, banks, batches, device,
                   mode: str, report: ParityReport, meta: dict) -> dict:
    """Stock vs grafted on the same real batches (two full forwards each)."""
    n = len(blocks)
    per_layer_cos = [[] for _ in range(n)]
    per_layer_topk = [[] for _ in range(n)]
    pairs_gb = [(blk, blk.gate) for blk in blocks
                if isinstance(blk.gate, GlmGraftGate)]

    for bi, (tokens, _dom) in enumerate(batches):
        tokens = tokens.to(device)
        t0 = time.time()
        # ---- stock arm: the donor's own gate ----
        for blk, g in pairs_gb:
            blk.gate = g.stock_gate
        tap = MoETap(blocks)
        logits_s = host_logits(model, lm_head, tokens)
        outs_s = [tap.out[i].float().cpu() if tap.out[i] is not None else None
                  for i in range(n)]
        idx_s = [tap.idx[i].cpu() if tap.idx[i] is not None else None
                 for i in range(n)]
        tap.close()
        # ---- graft arm: passport scores, stock selection semantics ----
        for blk, g in pairs_gb:
            blk.gate = g
        tap = MoETap(blocks)
        logits_g = host_logits(model, lm_head, tokens)
        outs_g = [tap.out[i].float().cpu() if tap.out[i] is not None else None
                  for i in range(n)]
        idx_g = [tap.idx[i].cpu() if tap.idx[i] is not None else None
                 for i in range(n)]
        tap.close()

        cos = token_logit_cosine(logits_s, logits_g)
        report.logit_cos.append(float(cos.mean()))
        report.logit_cos_min.append(float(cos.min()))
        if mode in ("full", "window"):
            report.ce_stock.append(next_token_ce(logits_s, tokens))
            report.ce_graft.append(next_token_ce(logits_g, tokens))
        for i in range(n):
            if outs_s[i] is None or outs_g[i] is None:
                continue
            a = outs_s[i].reshape(-1, outs_s[i].shape[-1])
            b = outs_g[i].reshape(-1, outs_g[i].shape[-1])
            c = F.cosine_similarity(a, b, dim=-1)
            per_layer_cos[i].extend(c.tolist())
            if idx_s[i] is not None and idx_g[i] is not None:
                ag, jac = topk_set_agreement(
                    idx_s[i].reshape(-1, idx_s[i].shape[-1]),
                    idx_g[i].reshape(-1, idx_g[i].shape[-1]))
                per_layer_topk[i].append((ag, jac))
        del logits_s, logits_g
        print(f"[parity] batch {bi + 1}/{len(batches)} "
              f"cos {report.logit_cos[-1]:.6f} ({time.time() - t0:.1f}s)",
              flush=True)

    for i in range(n):
        if per_layer_cos[i]:
            report.layer_cos.append(
                float(sum(per_layer_cos[i]) / len(per_layer_cos[i])))
            report.layer_cos_min.append(float(min(per_layer_cos[i])))
        if per_layer_topk[i]:
            report.topk_agree.append(
                float(sum(a for a, _ in per_layer_topk[i]) / len(per_layer_topk[i])))
            report.topk_jacc.append(
                float(sum(j for _, j in per_layer_topk[i]) / len(per_layer_topk[i])))

    out = report.as_dict()
    out["per_layer"] = {
        str(layer_ids[i]): {
            "moe_cos": round(float(sum(per_layer_cos[i]) / max(1, len(per_layer_cos[i]))), 6)
            if per_layer_cos[i] else None,
            "moe_cos_min": round(min(per_layer_cos[i]), 6) if per_layer_cos[i] else None,
            "topk_agree": round(float(sum(a for a, _ in per_layer_topk[i]) /
                                      max(1, len(per_layer_topk[i]))), 6)
            if per_layer_topk[i] else None,
        }
        for i in range(n)
    }
    return out


@torch.no_grad()
def router_score_parity(model, blocks, layer_ids, banks, batches, device) -> dict:
    """Router-only arm: passport scores vs stock scores on real h + top-k sets."""
    n = len(blocks)
    score_cos = [None] * n
    topk = [None] * n
    tap = MoETap(blocks)
    cos_acc = [[] for _ in range(n)]
    tks_acc = [[] for _ in range(n)]
    for tokens, _dom in batches:
        tap.clear()
        # ensure graft capture
        for blk in blocks:
            g = blk.gate
            if isinstance(g, GlmGraftGate):
                g.capture = True
        model(input_ids=tokens.to(device), use_cache=False)
        for i, blk in enumerate(blocks):
            g = blk.gate
            if not isinstance(g, GlmGraftGate) or g.last_h is None:
                continue
            h = g.last_h.float()
            stock = F.linear(h, g.stock_gate.weight.float())
            ps = banks[i].scores(h)
            c = F.cosine_similarity(ps, stock, dim=-1)
            cos_acc[i].extend(c.tolist())
            # top-k under each score family (own selection semantics)
            def sel(logits):
                s = logits.sigmoid()
                sfc = s + g.e_score_correction_bias
                ng = g.num_group
                gs = (sfc.view(-1, ng, -1).topk(2, dim=-1)[0].sum(-1))
                gi = torch.topk(gs, k=g.topk_group, dim=-1, sorted=False)[1]
                gm = torch.zeros_like(gs)
                gm.scatter_(1, gi, 1)
                sm = gm.unsqueeze(-1).expand(-1, ng, sfc.shape[-1] // ng).reshape(-1, sfc.shape[-1])
                return torch.topk(sfc.masked_fill(~sm.bool(), float("-inf")),
                                  k=g.top_k, dim=-1, sorted=False)[1]
            ia, ib = sel(stock), sel(ps)
            ag, jac = topk_set_agreement(ia, ib)
            tks_acc[i].append((ag, jac))
    tap.close()
    for blk in blocks:
        g = blk.gate
        if isinstance(g, GlmGraftGate):
            g.capture = False
    per = {}
    for i, li in enumerate(layer_ids):
        if cos_acc[i]:
            per[str(li)] = {
                "score_cos": round(float(sum(cos_acc[i]) / len(cos_acc[i])), 6),
                "score_cos_min": round(min(cos_acc[i]), 6),
                "topk_agree": round(float(sum(a for a, _ in tks_acc[i]) / len(tks_acc[i])), 6),
                "topk_jacc": round(float(sum(j for _, j in tks_acc[i]) / len(tks_acc[i])), 6),
            }
    allc = [c for xs in cos_acc for c in xs]
    allt = [a for xs in tks_acc for a, _ in xs]
    return {
        "score_cos_mean": round(float(sum(allc) / max(1, len(allc))), 6),
        "score_cos_min": round(min(allc), 6) if allc else None,
        "topk_agree_mean": round(float(sum(allt) / max(1, len(allt))), 6),
        "per_layer": per,
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_batches(tok, val_file: str, domains_dir: str, seq_len: int,
                 batch_size: int, max_tokens: int, device):
    """Real text slices -> list[(tokens (B,T), domain_label)]."""
    texts = []
    labels = []
    if val_file and os.path.exists(val_file):
        raw = open(val_file, errors="ignore").read()
        paras = [p.strip() for p in raw.split("\n") if p.strip()]
        texts.extend(paras)
        labels.extend(["val"] * len(paras))
    if not texts and domains_dir and os.path.isdir(domains_dir):
        for fn in sorted(os.listdir(domains_dir)):
            if not fn.endswith(".txt"):
                continue
            raw = open(os.path.join(domains_dir, fn), errors="ignore").read()
            paras = [p.strip() for p in raw.split("\n") if p.strip()]
            texts.extend(paras[:40])
            labels.extend([fn[:-4]] * min(40, len(paras)))
    if not texts:
        raise RuntimeError("no eval text found")
    batches = []
    n_tok = 0
    i = 0
    while i < len(texts) and n_tok < max_tokens:
        chunk = texts[i:i + batch_size]
        labs = labels[i:i + batch_size]
        i += batch_size
        enc = tok(chunk, return_tensors="pt", padding="max_length",
                  truncation=True, max_length=seq_len)
        ids = enc["input_ids"]
        if ids.shape[-1] < 2:
            continue
        batches.append((ids, labs[0] if labs else "val"))
        n_tok += ids.numel()
    return batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--phase", default="all",
                    choices=["precompute", "parity", "all", "router"])
    ap.add_argument("--mode", default="full", choices=["full", "window", "router"])
    ap.add_argument("--max-layers", type=int, default=None,
                    help="window mode: keep only the first N layers")
    ap.add_argument("--layers", default=None,
                    help="comma list of sparse layer ids to graft (default: all)")
    ap.add_argument("--init", default="exact", choices=["exact", "prototype"])
    ap.add_argument("--recal-steps", type=int, default=40)
    ap.add_argument("--recal-lr", type=float, default=0.05)
    ap.add_argument("--recal-kl-w", type=float, default=0.1)
    ap.add_argument("--recal-rounds", type=int, default=1)
    ap.add_argument("--max-pairs", type=int, default=2048)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--val-file", default=None)
    ap.add_argument("--domains-dir", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache-mb", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    log = lambda m: print(m, flush=True)

    patches = install_torch_kda()
    log(f"[kda] kernel shim: {patches}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    model, lm_head, index, meta = build_host(
        args.model, args.device, dtype,
        max_layers=args.max_layers, cache_mb=args.cache_mb, log=log)
    meta.update({
        "init": args.init, "mode": args.mode,
        "max_layers": args.max_layers, "kda_shim": patches,
        "seq_len": args.seq_len, "batch_size": args.batch_size,
    })

    blocks = find_moe_blocks(model)
    all_sparse = sparse_layer_indices(model)
    if args.layers:
        layer_ids = [int(x) for x in args.layers.split(",") if x != ""]
    else:
        layer_ids = list(all_sparse)
    pos = {li: i for i, li in enumerate(all_sparse)}
    blocks_sel = [blocks[pos[li]] for li in layer_ids]
    log(f"[graft] sparse layers: {len(all_sparse)} total, grafting {layer_ids}")

    batches = load_batches(tok, args.val_file, args.domains_dir,
                           args.seq_len, args.batch_size, args.max_tokens,
                           args.device)
    log(f"[data] {len(batches)} batches, seq {args.seq_len}")

    out = {"host": meta, "bar": {
        "logit_cos": BAR_LOGIT_COSINE,
        "logit_cos_stretch": BAR_LOGIT_COSINE_STRETCH,
        "ce_delta": BAR_CE_DELTA,
        "layer_cos": BAR_LAYER_COSINE,
        "topk_agree": BAR_TOPK_AGREE,
    }, "args": vars(args)}

    # banks
    d_r = meta["hidden"]
    banks = [PassportBank(meta["n_routed_experts"], d_r, d_r, learn_wq=False)
             for _ in layer_ids]
    stock_weights = [blk.gate.weight.data.clone() for blk in blocks_sel]

    if args.phase in ("precompute", "all", "router"):
        if args.init == "exact":
            init_passports_exact(banks, stock_weights)
            log("[registry] init=exact (P = sqrt(d_r) * W_stock)")
            out["recal"] = {"skipped": True, "reason": "exact init"}
        else:
            log("[registry] collecting (h, stock-logit) pairs ...")
            t0 = time.time()
            pairs = collect_pairs(model, blocks_sel, layer_ids, batches,
                                  args.device, args.max_pairs)
            log(f"[registry] pairs collected in {time.time() - t0:.1f}s "
                f"({[tuple(p[0].shape) for p in pairs][:3]} ...)")
            # RouteStats-compatible prototype: home = argmax stock logit hits
            stats = d0.RouteStats(
                n_layers=len(layer_ids), num_experts=meta["n_routed_experts"],
                d_model=d_r, n_domains=1)
            # observe in the d0 shape contract: (layer, h, top_idx, domain)
            for li, (h, s) in enumerate(pairs):
                top_idx = s.topk(meta["top_k"], dim=-1).indices
                stats.observe(li, h, top_idx, 0)
            init_passports_prototype(banks, stats)
            fit_report = recalibrate(banks, pairs, steps=args.recal_steps,
                                     lr=args.recal_lr, kl_w=args.recal_kl_w)
            out["recal"] = fit_report
            log(f"[registry] prototype+recal done: {fit_report.get('steps')} steps")
        reg_path = args.registry or (args.out or "glmflash_registry").replace(
            ".json", "") + "_registry.pt"
        os.makedirs(os.path.dirname(reg_path) or ".", exist_ok=True)
        save_registry(reg_path, banks, {**meta, "layer_ids": layer_ids,
                                        "registry_format": REGISTRY_FORMAT})
        out["registry"] = reg_path
        log(f"[registry] saved {reg_path}")

    if args.phase in ("parity", "all", "router"):
        # install grafts on selected layers
        restore = install_graft(model, banks, meta, layer_ids)
        try:
            if args.mode == "router" or args.phase == "router":
                rp = router_score_parity(model, blocks_sel, layer_ids, banks,
                                         batches, args.device)
                out["parity"] = rp
                out["parity"]["form"] = "router-only (layer-local score/top-k)"
                log(f"[parity] router-only: score_cos {rp['score_cos_mean']} "
                    f"topk {rp['topk_agree_mean']}")
            else:
                rep = ParityReport()
                pm = parity_measure(model, lm_head, blocks_sel, layer_ids,
                                    banks, batches, args.device, args.mode, rep,
                                    meta)
                pm["form"] = ("full-stack final logits" if args.mode == "full"
                              else f"truncated spine ({args.max_layers}L) logits")
                out["parity"] = pm
                log(f"[parity] {pm['form']}: logit_cos {pm['logit_cos_mean']} "
                    f"(min {pm['logit_cos_min']}) dCE {pm['ce_delta']} "
                    f"layer_cos {pm['layer_cos_mean']} topk {pm['topk_agree_mean']}")
        finally:
            uninstall_graft(model, restore)

        # verdict
        p = out["parity"]
        verdict = {}
        if p.get("logit_cos_mean") is not None:
            verdict["logit_cos"] = (p["logit_cos_mean"] >= BAR_LOGIT_COSINE)
            verdict["logit_cos_stretch"] = (
                p["logit_cos_mean"] >= BAR_LOGIT_COSINE_STRETCH)
        if p.get("ce_delta") is not None:
            verdict["ce_delta"] = abs(p["ce_delta"]) < BAR_CE_DELTA
        if p.get("layer_cos_mean") is not None:
            verdict["layer_cos"] = p["layer_cos_mean"] > BAR_LAYER_COSINE
        if p.get("topk_agree_mean") is not None:
            verdict["topk_agree"] = p["topk_agree_mean"] > BAR_TOPK_AGREE
        if p.get("score_cos_mean") is not None:
            verdict["score_cos"] = p["score_cos_mean"] >= BAR_LOGIT_COSINE
        out["verdict"] = verdict
        out["pass"] = all(verdict.values()) if verdict else None
        log(f"[verdict] {verdict} -> PASS={out['pass']}")

    # expert streamer stats
    loads = hits = 0
    for blk in model.modules():
        if isinstance(blk, StreamedExperts):
            loads += blk.stats["loads"]
            hits += blk.stats["hits"]
    out["expert_stream"] = {"loads": loads, "hits": hits,
                            "bytes_read_mb": round(
                                sum(b.stats["bytes_read"] for b in model.modules()
                                    if isinstance(b, StreamedExperts)) / 1e6, 1)}

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        log(f"[out] {args.out}")
    else:
        print(json.dumps(out, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
