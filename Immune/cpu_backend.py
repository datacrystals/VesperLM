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
# visible (cpu_probe.py has the same requirement). fla is imported lazily
# (install_cpu_shims / load_model), and importing it allocates no GPU
# context; we additionally guarantee no tensor ever reaches a GPU by
# mapping checkpoints to CPU and never calling .cuda().
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
#
# KDA/MLA shims (Vesper-K): chunk_kda/fused_recurrent_kda get a pure-torch
# equivalent built on fla.ops.kda.naive + fla.ops.kda.gate (fla's own torch
# gate references); MLA's flash-attn entry points are swapped for the SDPA
# fallback fla itself defines when flash-attn is missing (MLA runs on CPU
# either way; this also covers boxes where flash-attn is installed but the
# tensors live on CPU).
#
# Nothing here imports fla at module load: importing fla initialises Triton's
# autotuner and dies on hosts without a working CUDA driver ("0 active
# drivers"). install_cpu_shims() is called from load_model(), before the
# model is built.
# ---------------------------------------------------------------------------

_CPU_SHIMS_INSTALLED = False
_KDA_MLA_SHIMS_INSTALLED = False


def install_cpu_shims(kda_mla: bool = False) -> str:
    """Install pure-torch CPU replacements for fla's Triton-only ops.

    GLA shims always; KDA/MLA shims when `kda_mla` (the model's stack has
    KimiDeltaAttention / MultiheadLatentAttention layers). Must run before
    the first model forward.
    """
    global _CPU_SHIMS_INSTALLED, _KDA_MLA_SHIMS_INSTALLED
    status = []
    if not _CPU_SHIMS_INSTALLED:
        import fla.layers.gla as _fla_gla
        from fla.ops.gla.naive import naive_recurrent_gla

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

        import fla.modules.fused_norm_gate as _fng

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
        _CPU_SHIMS_INSTALLED = True
        status.append("gla")
    if kda_mla and not _KDA_MLA_SHIMS_INSTALLED:
        import fla.layers.kda as _fla_kda
        from fla.ops.kda.naive import naive_recurrent_kda
        from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate

        def _cpu_kda_shim(q=None, k=None, v=None, g=None, beta=None,
                          A_log=None, dt_bias=None, initial_state=None,
                          output_final_state=False, use_qk_l2norm_in_kernel=True,
                          use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                          allow_neg_eigval=False, lower_bound=None, scale=None,
                          state_v_first=True, cu_seqlens=None, **kw):
            # Mirror the Triton kernel's in-kernel pre/post-processing in
            # torch, then run fla's naive recurrence. state_v_first/cu_seqlens
            # are accepted and ignored: single-sequence CPU probes never pack
            # sequences, and the state layout stays self-consistent.
            if use_qk_l2norm_in_kernel:
                q = torch.nn.functional.normalize(q.float(), p=2, dim=-1)
                k = torch.nn.functional.normalize(k.float(), p=2, dim=-1)
            if use_gate_in_kernel:
                if lower_bound is not None:
                    g = naive_kda_lowerbound_gate(g, A_log, dt_bias,
                                                  lower_bound=lower_bound)
                else:
                    g = naive_kda_gate(g, A_log, dt_bias)
            if use_beta_sigmoid_in_kernel:
                beta = beta.sigmoid() * (2.0 if allow_neg_eigval else 1.0)
            return naive_recurrent_kda(q, k, v, g, beta, scale=scale,
                                       initial_state=initial_state,
                                       output_final_state=output_final_state)

        _fla_kda.chunk_kda = _cpu_kda_shim
        _fla_kda.fused_recurrent_kda = _cpu_kda_shim

        import fla.layers.mla as _fla_mla

        def _mla_flash_attn(q, k, v, dropout_p=0.0, softmax_scale=None,
                            causal=False, window_size=(-1, -1), **kwargs):
            if window_size not in (None, (-1, -1)):
                raise NotImplementedError("SDPA shim does not support sliding window")
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            o = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
            return o.transpose(1, 2)

        def _mla_flash_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k,
                                   max_seqlen_q, max_seqlen_k, dropout_p=0.0,
                                   softmax_scale=None, causal=False,
                                   window_size=(-1, -1), **kwargs):
            if window_size not in (None, (-1, -1)):
                raise NotImplementedError("SDPA shim does not support sliding window")
            outs = []
            for i in range(len(cu_seqlens_q) - 1):
                qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
                ks, ke = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
                oi = torch.nn.functional.scaled_dot_product_attention(
                    q[qs:qe].transpose(0, 1), k[ks:ke].transpose(0, 1),
                    v[ks:ke].transpose(0, 1),
                    dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
                outs.append(oi.transpose(0, 1))
            return torch.cat(outs, dim=0)

        _fla_mla.flash_attn_func = _mla_flash_attn
        _fla_mla.flash_attn_varlen_func = _mla_flash_attn_varlen
        _KDA_MLA_SHIMS_INSTALLED = True
        status.append("kda_mla")
    return "installed " + "+".join(status) if status else "already installed"


# ---------------------------------------------------------------------------
# Target profiles (architecture families)
# ---------------------------------------------------------------------------
DEFAULT_TARGET_PROFILE = "gla_gqa"

TARGET_PROFILES = {
    "gla_gqa": {
        "linear_type": "gla",
        "full_type": "gqa",
        "lora_targets": ("wq", "wo", "q_proj", "o_proj"),
        "router_targets": ("gate",),
    },
    "kda_mla": {
        "linear_type": "kda",
        "full_type": "mla",
        "lora_targets": ("q_proj", "k_proj", "v_proj", "o_proj", "k_rope",
                         "kv_proj.0", "kv_proj.2"),
        "router_targets": ("gate", "query"),
    },
}


def get_target_profile(name=DEFAULT_TARGET_PROFILE):
    if name not in TARGET_PROFILES:
        raise ValueError(f"unknown target_profile {name!r} "
                         f"(expected one of {sorted(TARGET_PROFILES)})")
    return TARGET_PROFILES[name]


def resolve_target_profile(cli_value=None):
    """CLI arg > $IMMUNE_TARGET_PROFILE > $HIPPO_TARGET_PROFILE > default."""
    name = (cli_value
            or os.environ.get("IMMUNE_TARGET_PROFILE")
            or os.environ.get("HIPPO_TARGET_PROFILE")
            or DEFAULT_TARGET_PROFILE)
    get_target_profile(name)  # validate
    return name


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------
def load_tokenizer():
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)


def load_model(ckpt_path, lora_path=None, target_profile=None):
    """Load a checkpoint (dir containing checkpoint.pt, or the .pt itself).

    model_config is read from the checkpoint itself (it wins over the target
    profile for every key it carries; the profile supplies the defaults for
    the architecture keys older checkpoints omit). Optionally apply a
    LoRA-style delta (see apply_lora_delta) before eval().

    `target_profile` selects the architecture family ("gla_gqa" | "kda_mla");
    it resolves via $IMMUNE_TARGET_PROFILE / $HIPPO_TARGET_PROFILE when None.
    """
    profile_name = resolve_target_profile(target_profile)
    profile = get_target_profile(profile_name)
    if os.path.isdir(ckpt_path):
        ckpt_file = os.path.join(ckpt_path, "checkpoint.pt")
    else:
        ckpt_file = ckpt_path
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    mc = ckpt.get("model_config") or {}
    linear_type = mc.get("linear_type", profile["linear_type"])
    full_type = mc.get("full_type", profile["full_type"])
    install_cpu_shims(kda_mla=(linear_type == "kda" or full_type == "mla"))
    from vesper_linear_model import VesperLinearLM
    model = VesperLinearLM(
        vocab_size=mc.get("vocab_size", 65523),
        dim=mc["dim"], n_layers=mc["n_layers"], n_heads=mc["n_heads"],
        n_kv_heads=mc["n_kv_heads"], hidden_dim=mc["hidden_dim"],
        num_experts=mc["num_experts"], top_k=mc["top_k"],
        max_seq_len=mc["max_seq_len"], pad_id=mc.get("pad_id", 0),
        linear_type=linear_type,
        full_type=full_type,
        attention_every=mc.get("attention_every", 4),
        gla_head_dim=mc.get("gla_head_dim", 64),
        qk_norm=mc.get("qk_norm", True),
        kda_head_dim=mc.get("kda_head_dim", 64),
        kda_short_conv=mc.get("kda_short_conv", True),
        mamba2_state_size=mc.get("mamba2_state_size", 128),
        kv_lora_rank=mc.get("kv_lora_rank"),
        v_head_dim=mc.get("v_head_dim", 128),
        linear_force_fp32=mc.get("linear_force_fp32", True),
        router_type=mc.get("router_type", "topk"),
        passport_dim=mc.get("passport_dim", 64),
        router_expert_dropout=mc.get("router_expert_dropout", 0.0),
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

def _supports_incremental(model):
    """MLA stacks have no incremental cache yet (see VesperLinearLM.new_cache)."""
    layer_types = getattr(model, "layer_types", None)
    return not (layer_types is not None and "mla" in layer_types)


@torch.no_grad()
def _generate_full_forward(model, tok, ids, max_new, stop_tokens):
    """Greedy decode by re-running the full forward each step. Used where
    incremental caching is unavailable (MLA stacks); outputs match the
    cached path for greedy decoding."""
    max_seq_len = getattr(model, "max_seq_len", None)
    out = []
    for _ in range(max_new):
        if max_seq_len is not None and ids.shape[1] >= max_seq_len:
            break
        logits, _, _ = model(ids)
        nxt = int(logits[0, -1].argmax())
        if nxt in stop_tokens:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return tok.decode(out, skip_special_tokens=False)


@torch.no_grad()
def generate(model, tok, prompt, max_new=48, stop_tokens=None):
    """Greedy generation with KV cache (same path as cpu_probe.py), falling
    back to full-forward decode on stacks without an incremental cache."""
    ids = tok(prompt, return_tensors="pt").input_ids
    stop_tokens = stop_tokens or set()
    if not _supports_incremental(model):
        return _generate_full_forward(model, tok, ids, max_new, stop_tokens)
    caches = model.new_cache(1, torch.device("cpu"))
    logits = model.forward_incremental(ids, caches, 0)[0][0, -1]
    pos = ids.shape[1]
    out = []
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
