"""Shared CPU backend for the VesperLM immune system.

Factors the CPU shim / loading logic out of Pretrain/cpu_probe.py
(that file is intentionally untouched). All immune entry points
import from here so a candidate and an incumbent are always loaded
through identical code paths.

Everything is forced onto CPU so nothing here can touch the training
GPUs. Do not add .cuda()/.to('cuda') calls to this module.
"""
import os
import sys

# NOTE: CUDA_VISIBLE_DEVICES must NOT be hidden -- fla's import path
# initializes Triton's autotuner, which needs the CUDA driver to be
# visible (cpu_probe.py has the same requirement). Importing fla does
# not allocate a GPU context; we additionally guarantee no tensor ever
# reaches a GPU by mapping checkpoints to CPU and never calling .cuda().
# The training jobs on this box are therefore untouched.
os.environ.setdefault("OMP_NUM_THREADS", "16")

_REPO = os.environ.get("VESPER_REPO", "/home/tliao/VesperLM")
for _p in (os.path.join(_REPO, "Common"),
           os.path.join(_REPO, "Pretrain"),
           os.path.join(_REPO, "Agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("VESPER_CPU_THREADS", "16")))

TOKENIZER_DIR = os.path.join(_REPO, "Pretrain", "custom_tokenizer")

# ---------------------------------------------------------------------------
# CPU shims for fla Triton ops (GLA chunk/recurrent + fused gated RMSNorm).
# Copied from Pretrain/cpu_probe.py (which runs generation at import time and
# therefore cannot itself be imported). Semantics must stay identical to it.
# ---------------------------------------------------------------------------
import fla.layers.gla as _fla_gla  # noqa: E402
from fla.ops.gla.naive import naive_recurrent_gla  # noqa: E402


def _cpu_gla_shim(q=None, k=None, v=None, g=None, gk=None,
                  initial_state=None, output_final_state=False,
                  state_v_first=True, cu_seqlens=None, **kw):
    gate = gk if gk is not None else g
    h_in = initial_state
    if h_in is not None and state_v_first:
        h_in = h_in.transpose(-1, -2).contiguous()
    o, h = naive_recurrent_gla(q, k, v, gate, initial_state=h_in,
                               output_final_state=output_final_state)
    if h is not None and state_v_first:
        h = h.transpose(-1, -2).contiguous()
    return o, h


_fla_gla.chunk_gla = _cpu_gla_shim
_fla_gla.fused_recurrent_gla = _cpu_gla_shim

import fla.modules.fused_norm_gate as _fng  # noqa: E402


def _rms_norm_gated_torch(x, g, weight, bias=None, activation="swish",
                          residual=None, prenorm=False, residual_in_fp32=False,
                          eps=1e-6, **kw):
    dtype = x.dtype
    xf = x.float()
    if residual is not None:
        xf = xf + residual.float()
    res_out = xf if prenorm else None
    var = (xf * xf).mean(-1, keepdim=True)
    y = xf * torch.rsqrt(var + eps)
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()
    gf = g.float()
    if activation in ("swish", "silu"):
        y = y * gf * torch.sigmoid(gf)
    elif activation == "sigmoid":
        y = y * torch.sigmoid(gf)
    y = y.to(dtype)
    return (y, res_out.to(x.dtype)) if prenorm else y


_fng.rms_norm_gated = _rms_norm_gated_torch
_fng.layer_norm_gated = _rms_norm_gated_torch

from vesper_linear_model import VesperLinearLM  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------
def load_tokenizer():
    return PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)


def load_model(ckpt_path, lora_path=None):
    """Load a checkpoint (dir containing checkpoint.pt, or the .pt itself).

    model_config is read from the checkpoint itself. Optionally apply a
    LoRA-style delta (see apply_lora_delta) before eval().
    """
    if os.path.isdir(ckpt_path):
        ckpt_file = os.path.join(ckpt_path, "checkpoint.pt")
    else:
        ckpt_file = ckpt_path
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    mc = ckpt.get("model_config") or {}
    model = VesperLinearLM(
        vocab_size=mc.get("vocab_size", 65523),
        dim=mc["dim"], n_layers=mc["n_layers"], n_heads=mc["n_heads"],
        n_kv_heads=mc["n_kv_heads"], hidden_dim=mc["hidden_dim"],
        num_experts=mc["num_experts"], top_k=mc["top_k"],
        max_seq_len=mc["max_seq_len"], pad_id=mc.get("pad_id", 0),
    )
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=True)
    if lora_path is not None:
        apply_lora_delta(model, lora_path)
    model.to(torch.device('cpu'))
    model.eval()
    return model, ckpt


def apply_lora_delta(model, lora_path):
    """Merge a LoRA-style delta into `model` in place.

    The delta file (.pt) may contain either:
      {"deltas": {param_name: additive_delta_tensor}}            # direct add
      {"lora": {param_name: {"A": (r, in), "B": (out, r), "scale": float}}}
    Low-rank entries are merged as W += scale * (B @ A). Names may omit
    the leading "module.". Raises on unknown keys so a silent no-op merge
    can never happen.
    """
    delta = torch.load(lora_path, map_location="cpu", weights_only=False)
    sd = model.state_dict()
    touched = 0
    for key, add in (delta.get("deltas") or {}).items():
        key = key.replace("module.", "")
        if key not in sd:
            raise KeyError(f"lora delta key not in model: {key}")
        sd[key] = sd[key] + add.to(sd[key].dtype)
        touched += 1
    for key, spec in (delta.get("lora") or {}).items():
        key = key.replace("module.", "")
        if key not in sd:
            raise KeyError(f"lora key not in model: {key}")
        a, b = spec["A"], spec["B"]
        scale = float(spec.get("scale", 1.0))
        upd = scale * (b @ a)
        if tuple(upd.shape) != tuple(sd[key].shape):
            raise ValueError(f"lora merge shape mismatch for {key}: "
                             f"{tuple(upd.shape)} vs {tuple(sd[key].shape)}")
        sd[key] = sd[key] + upd.to(sd[key].dtype)
        touched += 1
    if not touched:
        raise ValueError(f"lora file {lora_path} contained no deltas/lora entries")
    model.load_state_dict(sd, strict=True)

# END

@torch.no_grad()
def generate(model, tok, prompt, max_new=48, stop_tokens=None):
    """Greedy generation with KV cache (same path as cpu_probe.py)."""
    ids = tok(prompt, return_tensors="pt").input_ids
    caches = model.new_cache(1, torch.device("cpu"))
    logits = model.forward_incremental(ids, caches, 0)[0][0, -1]
    pos = ids.shape[1]
    out = []
    stop_tokens = stop_tokens or set()
    for _ in range(max_new):
        nxt = int(logits.argmax())
        if nxt in stop_tokens:
            break
        out.append(nxt)
        logits = model.forward_incremental(
            torch.tensor([[nxt]]), caches, pos)[0][0, -1]
        pos += 1
    return tok.decode(out, skip_special_tokens=False)


@torch.no_grad()
def topk_logprobs(model, ids, k=32):
    """Full-forward `ids` (1, T) and return top-k logprobs at the last
    position: (values (k,), indices (k,)). Used by drift.py."""
    logits, _, _ = model(ids)
    logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
    vals, idx = torch.topk(logp, k)
    return vals, idx, logp


@torch.no_grad()
def seq_topk_logprobs(model, ids, k=32, positions=None):
    """Top-k logprobs at several positions of one sequence.
    Returns list of (pos, values, indices, full_logp_row)."""
    logits, _, _ = model(ids)
    T = ids.shape[1]
    if positions is None:
        positions = list(range(max(0, T - 8), T))
    out = []
    for p in positions:
        logp = torch.log_softmax(logits[0, p].float(), dim=-1)
        vals, idx = torch.topk(logp, k)
        out.append((p, vals, idx, logp))
    return out


# ChatML / tool special tokens. Built from chr() so the literals survive
# any transport that mangles angle-bracket sequences:
#   chr(60)='<', chr(124)='|', chr(62)='>'
_LT, _BAR, _GT = chr(60), chr(124), chr(62)
IM_START = _LT + _BAR + "im_start" + _BAR + _GT
IM_END = _LT + _BAR + "im_end" + _BAR + _GT
THOUGHT_OPEN = _LT + _BAR + "thought" + _BAR + _GT
THOUGHT_CLOSE = _LT + "/|thought" + _BAR + _GT
TOOL_OPEN = _LT + _BAR + "tool_call" + _BAR + _GT
TOOL_CLOSE = _LT + "/|tool_call" + _BAR + _GT
