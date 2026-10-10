"""Hippocampus consolidation micro-session.

Pipeline: session log -> feedback triples -> LoRA micro-session -> canary gate
-> promote / rollback. Only LoRA A/B parameters train; the trunk stays frozen
and a KL penalty keeps the candidate's outputs close to the frozen base model.

The canary gate is EXTERNAL (Immune/). Interface:

    gate.evaluate(candidate_dir, incumbent_dir) -> GateResult
    GateResult.decision in {"PROMOTE", "ROLLBACK"}, plus .reason and .metrics

    CLI:  <gate_cmd> --candidate DIR --incumbent DIR
          stdout JSON: {"decision": "...", "reason": "...", "metrics": {...}}

Resolution order in `call_gate()`:
  1. an explicit gate_fn passed by the caller,
  2. $VESPER_GATE_CMD (subprocess CLI, JSON on stdout),
  3. import Immune.gate.evaluate if the Immune package is present,
  4. `stub_gate()` — a trivial in-file gate used only for standalone testing.

Per-user adapters live at <workdir>/adapters/<user_id>/delta.pt.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.abspath(os.path.join(_THIS_DIR, "..", "Common"))
_REPO = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in (_THIS_DIR, _COMMON):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from session_log import SessionLog, TrainingTriple
from lora import (attach_lora, freeze_trunk, lora_parameters,
                  lora_modules, save_delta, load_delta, set_lora_enabled,
                  merge_all, unmerge_all,
                  DEFAULT_TARGET_PROFILE, get_target_profile)
import telemetry


# ------------------------------------------------------------------
# CPU shims for GLA (Triton kernels are GPU-only; see Pretrain/cpu_probe.py)
# ------------------------------------------------------------------

_CPU_SHIMS_INSTALLED = False


def enable_cpu_gla_shims() -> str:
    """Make GatedLinearAttn runnable on CPU.

    fla's chunk_gla / fused_recurrent_gla and fused gated-RMSNorm are Triton-
    only. Swap in pure-torch equivalents (the same shims Pretrain/cpu_probe.py
    uses). Must run before the first model forward; import order is fine.
    """
    global _CPU_SHIMS_INSTALLED
    if _CPU_SHIMS_INSTALLED:
        return "already installed"
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
    return "installed"


_CPU_KDA_MLA_SHIMS_INSTALLED = False


def enable_cpu_kda_mla_shims() -> str:
    """Make KimiDeltaAttention / MultiheadLatentAttention runnable on CPU.

    fla's chunk_kda / fused_recurrent_kda are Triton-only; swap in a
    pure-torch equivalent built from fla.ops.kda.naive (plus fla's own
    torch gate references in fla.ops.kda.gate). MLA's flash-attn calls are
    replaced by fla's SDPA fallback so they also work when flash-attn is
    installed but tensors live on CPU. The KDA/MLA layers additionally
    reach three other Triton-only fla ops — ShortConvolution's causal conv
    (and its single-token update), MLA's rotary, and the plain RMSNorm in
    MLA's kv_proj — which get pure-torch replacements here too. Mirrors
    enable_cpu_gla_shims; must run before the first model forward.
    """
    global _CPU_KDA_MLA_SHIMS_INSTALLED
    if _CPU_KDA_MLA_SHIMS_INSTALLED:
        return "already installed"
    import fla.layers.kda as _fla_kda
    from fla.ops.kda.naive import naive_recurrent_kda
    from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate

    def _cpu_kda_shim(q=None, k=None, v=None, g=None, beta=None,
                      A_log=None, dt_bias=None, initial_state=None,
                      output_final_state=False, use_qk_l2norm_in_kernel=True,
                      use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                      allow_neg_eigval=False, lower_bound=None, scale=None,
                      state_v_first=True, cu_seqlens=None, **kw):
        # Mirror the Triton kernel's in-kernel pre/post-processing in torch,
        # then run fla's naive recurrence (state layout kept self-consistent;
        # state_v_first/cu_seqlens are accepted and ignored — single-sequence
        # CPU probes never pack sequences).
        if use_qk_l2norm_in_kernel:
            q = F.normalize(q.float(), p=2, dim=-1)
            k = F.normalize(k.float(), p=2, dim=-1)
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

    def _mla_flash_attn(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                        window_size=(-1, -1), **kwargs):
        if window_size not in (None, (-1, -1)):
            raise NotImplementedError("SDPA shim does not support sliding window")
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p,
                                           is_causal=causal, scale=softmax_scale)
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
            oi = F.scaled_dot_product_attention(
                q[qs:qe].transpose(0, 1), k[ks:ke].transpose(0, 1),
                v[ks:ke].transpose(0, 1),
                dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
            outs.append(oi.transpose(0, 1))
        return torch.cat(outs, dim=0)

    _fla_mla.flash_attn_func = _mla_flash_attn
    _fla_mla.flash_attn_varlen_func = _mla_flash_attn_varlen

    # KDA/MLA layers call three more fla ops that are Triton-only:
    # ShortConvolution's causal conv (KDA q/k/v short conv, incl. the
    # single-token update used by incremental decode), MLA's rotary, and
    # the plain RMSNorm inside MLA's kv_proj. Without these the layers die
    # on CPU with "Pointer argument (at 0) cannot be accessed from Triton
    # (cpu tensor?)" before the shims above are ever reached.
    #
    # GOTCHA: `import fla.modules.conv.causal_conv1d as m` binds the
    # re-exported *function* (fla.modules.conv.__init__ shadows the
    # submodule attribute with the same name), so a patch through it
    # silently lands on a function object. Fetch the real modules from
    # sys.modules and assert the patch landed.
    import importlib
    _cconv = importlib.import_module("fla.modules.conv.causal_conv1d")
    _ctrit = importlib.import_module("fla.modules.conv.triton.ops")
    _frot = importlib.import_module("fla.modules.rotary")
    _lnorm = importlib.import_module("fla.modules.layernorm")

    def _cpu_causal_conv1d(x, weight=None, bias=None, residual=None,
                           initial_state=None, output_final_state=False,
                           activation=None, cu_seqlens=None, **kw):
        # Depthwise causal conv. x: [B, T, D]; weight: [D, W];
        # initial/final state: [N, D, W] with the newest token at index
        # W-1 (fla's Triton layout: y[t] = sum_i w[i] * x[t-W+1+i], and a
        # state index w holds stream token at position w-W).
        if cu_seqlens is not None:
            raise NotImplementedError(
                "CPU causal_conv1d shim does not support packed cu_seqlens")
        B, T, D = x.shape
        W = weight.shape[-1]
        xt = x.transpose(1, 2)  # [B, D, T]
        # state[0] is never read by the conv (oldest of W history tokens;
        # the window at t=0 reaches back only W-1 steps)
        left = (initial_state[:, :, 1:] if initial_state is not None
                else xt.new_zeros(B, D, W - 1))
        y = F.conv1d(torch.cat([left, xt], dim=-1),
                     weight.unsqueeze(1), bias, groups=D)
        if activation in ("silu", "swish"):
            y = F.silu(y)
        if residual is not None:
            y = y + residual.transpose(1, 2)  # after activation, like the kernel
        y = y.transpose(1, 2)
        final_state = None
        if output_final_state:
            # final state = last W tokens of concat(state, x), zero-left-
            # padded when the stream is shorter than W
            hist = (torch.cat([initial_state, xt], dim=-1)
                    if initial_state is not None else xt)
            final_state = (hist[:, :, -W:] if hist.shape[-1] >= W
                           else F.pad(hist, (W - hist.shape[-1], 0)))
            final_state = final_state.contiguous()
        return y, final_state

    def _cpu_causal_conv1d_update(x, cache, residual=None, weight=None,
                                  bias=None, activation=None, **kw):
        # Single-token step used by ShortConvolution.step. cache: [N, D, W]
        # updated in place: roll left, append new token at index W-1,
        # y = (cache * weight).sum(-1)  (fla's own documented equivalent).
        shape = x.shape
        if x.dim() == 3 and x.shape[0] == 1 and x.shape[1] == cache.shape[0]:
            xn = x[0]                    # (1, N, D) -> [N, D]
        elif x.dim() == 3:
            xn = x[:, -1, :]             # (N, 1, D) -> [N, D]
        else:
            xn = x                       # [N, D]
        cache.copy_(cache.roll(shifts=-1, dims=-1))
        cache[:, :, -1] = xn
        y = (cache * weight.unsqueeze(0)).sum(-1)  # [N, D]
        if bias is not None:
            y = y + bias
        if activation in ("silu", "swish"):
            y = F.silu(y)
        if residual is not None:
            if residual.dim() == 3 and residual.shape[0] == 1:
                res = residual[0]
            elif residual.dim() == 3:
                res = residual[:, -1, :]
            else:
                res = residual
            y = y + res
        return y.view(shape), cache

    def _cpu_rotary(x, cos, sin, interleaved=False, inplace=False,
                    seqlen_offsets=0, cu_seqlens=None, chunk_indices=None,
                    **kw):
        # fla's own torch reference + seqlen_offsets slicing. x: [B,T,H,D];
        # cos/sin: [T_max, head_dim/2].
        if cu_seqlens is not None:
            raise NotImplementedError(
                "CPU rotary shim does not support packed cu_seqlens")
        off = (int(seqlen_offsets) if not torch.is_tensor(seqlen_offsets)
               else int(seqlen_offsets.reshape(-1)[0]))
        T = x.shape[1]
        return _frot.rotary_embedding_ref(
            x, cos[off:off + T], sin[off:off + T], interleaved=interleaved)

    def _cpu_rms_norm(x, weight=None, bias=None, residual=None, eps=1e-5,
                      prenorm=False, residual_in_fp32=False, **kw):
        if weight is None:
            weight = x.new_ones(x.shape[-1])
        return _lnorm.rms_norm_ref(x, weight, bias, residual=residual,
                                   eps=eps, prenorm=prenorm)

    _cconv.causal_conv1d = _cpu_causal_conv1d
    _ctrit.causal_conv1d_update = _cpu_causal_conv1d_update
    _frot.rotary_embedding = _cpu_rotary
    _lnorm.rms_norm = _cpu_rms_norm
    for _mod, _name, _fn in ((_cconv, "causal_conv1d", _cpu_causal_conv1d),
                             (_ctrit, "causal_conv1d_update", _cpu_causal_conv1d_update),
                             (_frot, "rotary_embedding", _cpu_rotary),
                             (_lnorm, "rms_norm", _cpu_rms_norm)):
        if getattr(_mod, _name) is not _fn:
            raise RuntimeError(
                f"CPU shim failed to patch {_mod.__name__}.{_name}")
    _CPU_KDA_MLA_SHIMS_INSTALLED = True
    return "installed"


# ------------------------------------------------------------------
# Gate interface
# ------------------------------------------------------------------

@dataclass
class GateResult:
    decision: str  # "PROMOTE" | "ROLLBACK"
    reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.decision not in ("PROMOTE", "ROLLBACK"):
            raise ValueError(f"decision must be PROMOTE or ROLLBACK, got {self.decision!r}")


def stub_gate(candidate_dir: str, incumbent_dir: str) -> GateResult:
    """Trivial stand-in for Immune's canary gate (standalone testing only).

    Rejects degenerate/poisoned batches using only static batch statistics
    written by the micro-session:
      * mean reward must be >= 0  (we only promote net-positive feedback),
      * the batch must not be target-collapsed (one response string dominating),
      * the candidate must not have drifted far in KL from base.
    """
    cand = os.path.abspath(candidate_dir)
    stats_path = os.path.join(cand, "train_stats.json")
    if not os.path.exists(stats_path):
        return GateResult("ROLLBACK", "candidate missing train_stats.json")
    with open(stats_path) as f:
        stats = json.load(f)

    mean_reward = float(stats.get("mean_reward", 0.0))
    uniq_ratio = float(stats.get("unique_response_ratio", 1.0))
    top_frac = float(stats.get("top_response_fraction", 0.0))
    final_kl = float(stats.get("final_kl", 0.0))

    metrics = {"mean_reward": mean_reward, "unique_response_ratio": uniq_ratio,
               "top_response_fraction": top_frac, "final_kl": final_kl}
    if mean_reward < 0.0:
        return GateResult("ROLLBACK", f"mean reward {mean_reward:.3f} < 0", metrics)
    if top_frac > 0.5:
        return GateResult("ROLLBACK",
                          f"target collapse: one response is {top_frac:.0%} of the batch",
                          metrics)
    if uniq_ratio < 0.25:
        return GateResult("ROLLBACK",
                          f"degenerate batch: unique response ratio {uniq_ratio:.2f} < 0.25",
                          metrics)
    # Runaway-drift bound only: a deliberate preference flip legitimately
    # costs a few nats of KL. Real thresholds are Immune's call.
    if final_kl > 10.0:
        return GateResult("ROLLBACK", f"KL drift from base {final_kl:.3f} > 10.0", metrics)
    return GateResult("PROMOTE", "stub gate checks passed", metrics)


def call_gate(candidate_dir: str, incumbent_dir: str,
              gate_fn: Optional[Callable[[str, str], GateResult]] = None,
              gate_cmd: Optional[str] = None) -> GateResult:
    """Invoke the external canary gate. See module docstring for the contract."""
    if gate_fn is not None:
        res = gate_fn(candidate_dir, incumbent_dir)
        return res if isinstance(res, GateResult) else GateResult(**res)

    cmd = gate_cmd or os.environ.get("VESPER_GATE_CMD")
    if cmd:
        proc = subprocess.run(
            [cmd, "--candidate", candidate_dir, "--incumbent", incumbent_dir],
            capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            return GateResult("ROLLBACK", f"gate command failed: {proc.stderr.strip()[:200]}")
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
            return GateResult(payload["decision"], payload.get("reason", ""),
                              payload.get("metrics", {}))
        except Exception as e:
            return GateResult("ROLLBACK", f"gate produced unparsable output: {e}")

    try:
        import importlib
        immune_gate = importlib.import_module("Immune.gate")
        res = immune_gate.evaluate(candidate_dir, incumbent_dir)
        return res if isinstance(res, GateResult) else GateResult(**res)
    except ImportError:
        pass
    except Exception as e:
        return GateResult("ROLLBACK", f"Immune.gate raised: {e}")

    return stub_gate(candidate_dir, incumbent_dir)


# ------------------------------------------------------------------
# Model / tokenizer loading
# ------------------------------------------------------------------

def load_tokenizer(tokenizer_path: Optional[str] = None):
    from transformers import AutoTokenizer
    path = tokenizer_path or os.path.join(_REPO, "Pretrain", "custom_tokenizer")
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(checkpoint_path: Optional[str] = None, device: str = "cpu",
                    dtype: torch.dtype = torch.float32,
                    target_profile: Optional[str] = None):
    """Build VesperLinearLM from the SFT checkpoint's embedded model_config.

    Dtype note: GLA/Mamba2 wrappers internally force fp32 (their Triton
    kernels crash on fp16 under P40 cc 6.1, and the wrappers' pure-torch
    fallbacks are written for fp32). Keep the whole model in fp32 for
    micro-sessions; bf16 is only safe for the non-GLA parts.

    `target_profile` (or $HIPPO_TARGET_PROFILE) selects the LoRA target
    family for later attach_lora calls; it is validated here so a bad name
    fails before the checkpoint is loaded.
    """
    if target_profile is not None:
        get_target_profile(target_profile)
    if device == "cpu":
        enable_cpu_gla_shims()
        enable_cpu_kda_mla_shims()
    from vesper_linear_model import VesperLinearLM

    ckpt_path = checkpoint_path or os.path.join(
        _REPO, "SFT", "sft_checkpoints_118m_v1", "step_2900", "checkpoint.pt")
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = dict(ckpt["model_config"])
    tok = load_tokenizer()
    model = VesperLinearLM(
        vocab_size=len(tok),
        dim=mc["dim"], n_layers=mc["n_layers"], n_heads=mc["n_heads"],
        n_kv_heads=mc["n_kv_heads"], hidden_dim=mc["hidden_dim"],
        num_experts=mc["num_experts"], top_k=mc["top_k"],
        max_seq_len=mc["max_seq_len"], pad_id=tok.pad_token_id,
        linear_type=mc.get("linear_type", "gla"),
        full_type=mc.get("full_type", "gqa"),
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
    model.grad_checkpoint = False  # micro-sessions are tiny; skip recompute
    model.eval()
    # NOTE: never call model.to(dtype=...) wholesale — the RoPE buffer
    # `freqs_cis` is complex, and Module.to(dtype) casts it to a real dtype,
    # silently discarding the imaginary part (and breaking RoPE). Move device
    # first, then cast only floating-point tensors if a non-fp32 dtype is
    # actually requested.
    model.to(device=device)
    if dtype != torch.float32:
        for _, p in model.named_parameters():
            if p.is_floating_point():
                p.data = p.data.to(dtype)
        for _, buf in model.named_buffers():
            if buf.is_floating_point() and not buf.is_complex():
                buf.data = buf.data.to(dtype)
    return model, tok, mc


# ------------------------------------------------------------------
# Generation / probe helpers
# ------------------------------------------------------------------

@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 24,
                  device: str = "cpu", stop_ids: Optional[Sequence[int]] = None,
                  temperature: float = 0.0) -> str:
    """Greedy (or lightly sampled) decode; cached incremental when available.

    MLA stacks have no incremental cache yet (new_cache raises), so those
    models decode via full forward() over the growing sequence instead —
    fine for these short probe generations.
    """
    model.eval()
    stop_ids = list(stop_ids) if stop_ids is not None else [tokenizer.eos_token_id]
    enc = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    use_cache = 'mla' not in getattr(model, "layer_types", ())
    caches = model.new_cache(1, torch.device(device)) if use_cache else None
    seq = enc
    if use_cache:
        logits = model.forward_incremental(seq, caches, 0)[0][0, -1]
    else:
        logits = model(seq)[0][0, -1].float()
    pos = seq.shape[1]
    out: List[int] = []
    for _ in range(max_new_tokens):
        if temperature > 0:
            probs = F.softmax(logits.float() / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
        else:
            nxt = int(logits.argmax())
        if nxt in stop_ids:
            break
        out.append(nxt)
        if pos >= model.max_seq_len:
            break
        step = torch.tensor([[nxt]], device=device)
        if use_cache:
            logits = model.forward_incremental(step, caches, pos)[0][0, -1]
        else:
            seq = torch.cat([seq, step], dim=1)
            logits = model(seq)[0][0, -1].float()
        pos += 1
    return tokenizer.decode(out)


@torch.no_grad()
def first_token_logits(model, tokenizer, prompt: str, device: str = "cpu",
                       lora_on: bool = True) -> torch.Tensor:
    """Logits of the first answer token after `prompt` (for margin probes)."""
    set_lora_enabled(model, lora_on)
    enc = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    logits = model(enc)[0][0, -1].float()
    set_lora_enabled(model, True)
    return logits


def word_digit_margin(logits: torch.Tensor, tokenizer,
                      word_forms: Sequence[str] = (" four", " two", " six", " eight", " ten"),
                      digit_forms: Sequence[str] = (" 4", " 2", " 6", " 8", " 0")) -> Dict[str, Any]:
    """logit(word-number first tokens) - logit(digit first tokens).

    The demo preference is "answer math in words, not digits", so this margin
    is the mechanism-level signal that the preference moved.
    """
    def best_id(forms: Sequence[str]) -> Tuple[float, str]:
        best: Optional[float] = None
        best_tok = ""
        for f in forms:
            ids = tokenizer(f, add_special_tokens=False).input_ids
            if not ids:
                continue
            v = float(logits[ids[0]])
            if best is None or v > best:
                best, best_tok = v, f
        return (best if best is not None else 0.0), best_tok

    w, wt = best_id(word_forms)
    d, dt = best_id(digit_forms)
    return {"word_logit": w, "digit_logit": d, "margin": w - d,
            "best_word": wt, "best_digit": dt}


@torch.no_grad()
def response_nll(model, tokenizer, prompt: str, response: str,
                 device: str = "cpu", max_seq_len: int = 256) -> float:
    """Masked mean NLL of `response` tokens conditioned on `prompt`."""
    model.eval()
    set_lora_enabled(model, True)
    e = _encode_triple(
        tokenizer, TrainingTriple(prompt, response, 1.0, "probe", 0, "probe", 1.0),
        max_seq_len)
    if e is None:
        return float("nan")
    x = torch.tensor([e["x"]], dtype=torch.long, device=device)
    y = torch.tensor([e["y"]], dtype=torch.long, device=device)
    mask = torch.tensor([e["mask"]], dtype=torch.float32, device=device)
    logits = model(x)[0].float()
    nll = -F.log_softmax(logits, dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return float((nll * mask).sum() / mask.sum().clamp(min=1.0))


# ------------------------------------------------------------------
# Micro-session: train LoRA on triples, gate, promote/rollback
# ------------------------------------------------------------------

@dataclass
class ConsolidateResult:
    decision: str
    reason: str
    gate_metrics: Dict[str, Any]
    candidate_dir: str
    incumbent_dir: str
    user_id: str
    n_triples: int
    train_stats: Dict[str, Any]
    promoted_delta: Optional[str] = None


def _encode_triple(tok, triple: TrainingTriple, max_seq_len: int) -> Optional[Dict]:
    """Tokenize (prompt, response) and build a response-only loss mask."""
    prompt_ids = tok(triple.prompt, add_special_tokens=False).input_ids
    resp_ids = tok(triple.response, add_special_tokens=False).input_ids
    if not resp_ids:
        return None
    # next-token training: input = prompt+resp[:-1], target = prompt+resp[1:]
    # mask covers response positions only (prompt positions contribute 0).
    ids = (prompt_ids + resp_ids)[:max_seq_len]
    if len(ids) < 2:
        return None
    x = ids[:-1]
    y = ids[1:]
    n_prompt = max(0, len(prompt_ids) - 1)
    mask = [0] * n_prompt + [1] * (len(y) - n_prompt)
    mask = mask[:len(y)]
    if sum(mask) == 0:
        return None
    return {"x": x, "y": y, "mask": mask, "reward": triple.reward}


def _collate(batch: List[Dict], pad_id: int) -> Dict[str, torch.Tensor]:
    T = max(len(b["x"]) for b in batch)
    B = len(batch)
    x = torch.full((B, T), pad_id, dtype=torch.long)
    y = torch.full((B, T), pad_id, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.float32)
    real = torch.zeros((B, T), dtype=torch.float32)   # 1 on non-pad tokens
    rewards = torch.zeros(B, dtype=torch.float32)
    for i, b in enumerate(batch):
        n = len(b["x"])
        x[i, :n] = torch.tensor(b["x"], dtype=torch.long)
        y[i, :n] = torch.tensor(b["y"], dtype=torch.long)
        mask[i, :n] = torch.tensor(b["mask"], dtype=torch.float32)
        real[i, :n] = 1.0
        rewards[i] = float(b["reward"])
    return {"x": x, "y": y, "mask": mask, "real": real, "rewards": rewards}


def train_lora_microsession(model, tok, triples: List[TrainingTriple], *,
                            steps: int = 30, lr: float = 1e-3, batch_size: int = 4,
                            kl_coef: float = 0.05, max_seq_len: int = 256,
                            device: str = "cpu",
                            log_every: int = 5) -> Dict[str, Any]:
    """Few tiny steps on LoRA params only.

    Loss per batch =  mean_i( reward_i * meanNLL(response_i | prompt_i) )
                      + kl_coef * KL(candidate || base)

    Positive reward descends NLL (teach / clone the response); negative reward
    ascends NLL (un-teach / unlikelihood pressure against it). The KL term
    anchors the candidate to the base model's outputs (base logits come from
    the same frozen trunk with LoRA branches disabled; they depend only on the
    frozen weights and the inputs, so they are precomputed once per example).
    """
    model.train()
    freeze_trunk(model)
    params = [p for p in lora_parameters(model)]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    encoded = [e for e in (_encode_triple(tok, t, max_seq_len) for t in triples) if e]
    if not encoded:
        raise ValueError("no encodable triples")
    gen = torch.Generator().manual_seed(0)
    stats = {"loss0": None, "losses": [], "policy": [], "kl": [], "steps": steps}

    # Precompute base logits (LoRA disabled) per encoded example.
    with torch.no_grad():
        set_lora_enabled(model, False)
        base_cache = []
        for e in encoded:
            x1 = torch.tensor([e["x"]], dtype=torch.long, device=device)
            base_cache.append(model(x1)[0][0].float().cpu())   # (T_i, V)
        set_lora_enabled(model, True)

    for step in range(steps):
        idx = torch.randint(0, len(encoded), (min(batch_size, len(encoded)),),
                            generator=gen).tolist()
        batch = _collate([encoded[i] for i in idx], tok.pad_token_id)
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        real = batch["real"].to(device)
        rewards = batch["rewards"].to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(x)[0].float()                        # (B, T, V)
        T = logits.shape[1]
        base_logits = torch.zeros(len(idx), T, logits.shape[-1])
        for row, i in enumerate(idx):
            b = base_cache[i]
            base_logits[row, :b.shape[0]] = b
        base_logits = base_logits.to(device)

        logp = F.log_softmax(logits, dim=-1)
        nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (B, T)
        seq_nll = (nll * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)  # (B,)
        # reward-weighted NLL: +reward -> descend NLL (clone),
        # -reward -> ascend NLL (unlikelihood against the bad response)
        policy = (rewards * seq_nll).mean()

        # KL(candidate || base) over real (non-pad) tokens
        logp_b = F.log_softmax(base_logits, dim=-1)
        p = logp.exp()
        kl_tok = (p * (logp - logp_b)).sum(-1)             # (B, T)
        kl = (kl_tok * real).sum() / real.sum().clamp(min=1.0)

        loss = policy + kl_coef * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if stats["loss0"] is None:
            stats["loss0"] = float(loss)
        stats["losses"].append(float(loss))
        stats["policy"].append(float(policy))
        stats["kl"].append(float(kl))
        if log_every and step % log_every == 0:
            print(f"    [micro] step {step:3d}  loss {float(loss):+.4f}  "
                  f"policy {float(policy):+.4f}  kl {float(kl):.5f}")

    model.eval()
    stats["loss_final"] = stats["losses"][-1]
    stats["final_kl"] = stats["kl"][-1]
    return stats


def _batch_stats(triples: List[TrainingTriple]) -> Dict[str, Any]:
    responses = [t.response for t in triples]
    n = max(len(responses), 1)
    counts: Dict[str, int] = {}
    for r in responses:
        counts[r] = counts.get(r, 0) + 1
    top = max(counts.values()) if counts else 0
    return {
        "n_triples": len(triples),
        "mean_reward": sum(t.reward for t in triples) / n,
        "unique_response_ratio": len(counts) / n,
        "top_response_fraction": top / n,
        "sources": sorted({t.source for t in triples}),
    }


def adapter_dir(workdir: str, user_id: str, kind: str = "adapters") -> str:
    return os.path.join(os.path.abspath(workdir), kind, user_id)


def consolidate(log: SessionLog, user_id: str = "default",
                workdir: Optional[str] = None, *,
                checkpoint_path: Optional[str] = None,
                device: str = "cpu", dtype: torch.dtype = torch.float32,
                steps: int = 30, lr: float = 1e-3, batch_size: int = 4,
                kl_coef: float = 0.05, rank: int = 8, alpha: float = 16.0,
                include_router: bool = False,
                min_confidence: float = 0.8,
                allow_outcome_derived: bool = True,
                gate_fn: Optional[Callable[[str, str], GateResult]] = None,
                gate_cmd: Optional[str] = None,
                base_model=None, tokenizer=None,
                target_profile: Optional[str] = None,
                verbose: bool = True) -> ConsolidateResult:
    """One consolidation pass: log -> triples -> LoRA candidate -> gate -> commit.

    `target_profile` (or $HIPPO_TARGET_PROFILE, default "gla_gqa") selects
    which projections the LoRA delta attaches to — see lora.TARGET_PROFILES.
    """
    profile_name = target_profile or os.environ.get(
        "HIPPO_TARGET_PROFILE", DEFAULT_TARGET_PROFILE)
    get_target_profile(profile_name)  # validate before doing any work
    workdir = os.path.abspath(workdir or os.path.join(_THIS_DIR, "consolidate_out"))
    os.makedirs(workdir, exist_ok=True)

    triples = log.to_triples(min_confidence=min_confidence,
                             allow_outcome_derived=allow_outcome_derived)
    if verbose:
        print(f"[consolidate] user={user_id} triples={len(triples)} "
              f"(from log {log.path})")
    if not triples:
        raise ValueError("no eligible training triples (all quarantined?)")

    if base_model is None:
        model, tok, _mc = load_base_model(checkpoint_path, device=device, dtype=dtype)
    else:
        model, tok = base_model, tokenizer

    # Base + incumbent LoRA: start from the user's promoted delta if any.
    inc_dir = adapter_dir(workdir, user_id, "adapters")
    incumbent_path = os.path.join(inc_dir, "delta.pt")
    wrapped = attach_lora(model, rank=rank, alpha=alpha,
                          include_router=include_router,
                          target_profile=profile_name)
    if verbose:
        print(f"[consolidate] lora attached: {len(wrapped)} modules "
              f"(profile {profile_name}), "
              f"trainable params: {sum(p.numel() for p in lora_parameters(model)):,}")
        if wrapped:
            print(f"[consolidate] wrapped modules: {wrapped}")
    if os.path.exists(incumbent_path):
        load_delta(model, incumbent_path, target_profile=profile_name)
        if verbose:
            print(f"[consolidate] loaded incumbent delta {incumbent_path}")
    freeze_trunk(model)

    t0 = time.time()
    stats = train_lora_microsession(
        model, tok, triples, steps=steps, lr=lr, batch_size=batch_size,
        kl_coef=kl_coef, device=device, log_every=max(1, steps // 5))
    stats.update(_batch_stats(triples))
    stats["train_seconds"] = round(time.time() - t0, 2)

    # Write the candidate directory the gate inspects.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cand_dir = os.path.join(workdir, "candidates", f"{user_id}-{stamp}")
    os.makedirs(cand_dir, exist_ok=True)
    save_delta(model, os.path.join(cand_dir, "delta.pt"),
               meta={"user_id": user_id, "steps": steps, "lr": lr, "rank": rank,
                     "alpha": alpha, "created": stamp,
                     "target_profile": profile_name,
                     "incumbent": incumbent_path if os.path.exists(incumbent_path) else None})
    with open(os.path.join(cand_dir, "train_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(cand_dir, "triples.jsonl"), "w") as f:
        for t in triples:
            f.write(json.dumps(t.as_dict(), ensure_ascii=False) + "\n")
    if verbose:
        print(f"[consolidate] candidate written to {cand_dir}")
        print(f"[consolidate] train: loss {stats['loss0']:+.4f} -> {stats['loss_final']:+.4f}  "
              f"final_kl {stats['final_kl']:.5f}")

    # Incumbent dir for the gate (empty placeholder on first consolidation).
    gate_inc_dir = inc_dir if os.path.exists(incumbent_path) else \
        os.path.join(workdir, "candidates", "_none")
    os.makedirs(gate_inc_dir, exist_ok=True)

    verdict = call_gate(cand_dir, gate_inc_dir, gate_fn=gate_fn, gate_cmd=gate_cmd)
    if verbose:
        print(f"[consolidate] GATE: {verdict.decision} — {verdict.reason}")
    # E0 drive telemetry (SUBSYSTEMS.md): the admission verdict seen by the
    # memory loop (stub / CLI / Immune import — whatever call_gate resolved).
    telemetry.log_immune_verdict(
        verdict.decision, source="consolidate.call_gate", reason=verdict.reason,
        metrics=verdict.metrics, user_id=user_id, candidate_dir=cand_dir)

    promoted = None
    if verdict.decision == "PROMOTE":
        os.makedirs(inc_dir, exist_ok=True)
        shutil.copy2(os.path.join(cand_dir, "delta.pt"), incumbent_path)
        promoted = incumbent_path
        with open(os.path.join(inc_dir, "meta.json"), "w") as f:
            json.dump({"user_id": user_id, "promoted": stamp,
                       "reason": verdict.reason, "gate_metrics": verdict.metrics,
                       "train_stats": stats}, f, indent=2)
        if verbose:
            print(f"[consolidate] PROMOTED -> {incumbent_path}")
    else:
        if verbose:
            print(f"[consolidate] ROLLBACK — incumbent unchanged: "
                  f"{incumbent_path if os.path.exists(incumbent_path) else '(none)'}")

    # E0 drive telemetry (SUBSYSTEMS.md): one consolidation event per pass.
    telemetry.log_consolidation(
        verdict.decision, user_id=user_id, n_triples=len(triples),
        reason=verdict.reason,
        mean_reward=stats.get("mean_reward"),
        unique_response_ratio=stats.get("unique_response_ratio"),
        top_response_fraction=stats.get("top_response_fraction"),
        loss0=stats.get("loss0"), loss_final=stats.get("loss_final"),
        final_kl=stats.get("final_kl"),
        train_seconds=stats.get("train_seconds"),
        target_profile=profile_name, promoted=promoted is not None)

    return ConsolidateResult(
        decision=verdict.decision, reason=verdict.reason,
        gate_metrics=verdict.metrics, candidate_dir=cand_dir,
        incumbent_dir=gate_inc_dir, user_id=user_id,
        n_triples=len(triples), train_stats=stats, promoted_delta=promoted)


def apply_user_adapter(model, workdir: str, user_id: str,
                       targets: Optional[Sequence[str]] = None,
                       rank: int = 8, alpha: float = 16.0,
                       include_router: bool = False,
                       target_profile: str = DEFAULT_TARGET_PROFILE) -> bool:
    """Attach and load a user's promoted delta at inference time. False if none.

    `targets=None` uses the profile's projection targets (explicit `targets`
    override the profile).
    """
    path = os.path.join(adapter_dir(workdir, user_id, "adapters"), "delta.pt")
    if not os.path.exists(path):
        return False
    if not lora_modules(model):
        attach_lora(model, targets=targets, rank=rank, alpha=alpha,
                    include_router=include_router, target_profile=target_profile)
    load_delta(model, path, target_profile=target_profile)
    return True
