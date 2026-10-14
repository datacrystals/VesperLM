#!/usr/bin/env python3
"""fla KDA compatibility shim — pure-torch ops behind fla's KDA layer.

WHY (FAILURES fla-kda-gfx906): fla's KDA kernels (chunk_kda /
fused_recurrent_kda) are Triton-only.  On the MI50 box every compile dies
with `RuntimeError: PassManager::run failed` (Triton AMD backend, gfx906),
at EVERY sequence length including T=1; on CPU Triton refuses the tensors
outright.  vk-base (KDA+MLA) therefore cannot run unpatched anywhere except
the Triton-capable training venue (MI300X).

fla ships a torch reference at fla/ops/kda/naive.py (naive_recurrent_kda /
naive_chunk_kda) that is never wired into the layer.  This module patches
fla.layers.kda's two op calls to wrappers that (1) do the in-kernel
preprocessing in torch — q/k L2-norm, gate g = -exp(A_log)*softplus(g+dt_bias),
beta = sigmoid(beta) — exactly matching the fused_recurrent kernel body, and
(2) call the naive ops.  Pure torch/GEMM: runs on gfx906 and CPU.

Import this BEFORE the first fla KDA forward (vk_graft does it at load).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

_APPLIED = False


def _prep(q, k, v, g, beta, A_log, dt_bias, use_qk_l2norm, use_beta_sigmoid):
    """Reproduce the fused_recurrent kernel's in-kernel preprocessing."""
    if use_qk_l2norm:
        q = F.normalize(q.float(), dim=-1)
        k = F.normalize(k.float(), dim=-1)
    else:
        q, k = q.float(), k.float()
    # gate: g_log = -exp(A_log) * softplus(g + dt_bias)   (gate.py reference)
    # dt_bias is (HV*K,) -> (HV, K); A_log is (HV,) -> (HV, 1). fp32 always.
    HV, K = g.shape[-2], g.shape[-1]
    gf = g.float()
    if dt_bias is not None:
        gf = gf + dt_bias.reshape(HV, K).float()
    if A_log is not None:
        A = A_log.reshape(HV, 1).float().exp()
        g_log = -A * F.softplus(gf)
    else:
        g_log = -F.softplus(gf)
    if use_beta_sigmoid:
        beta = beta.float().sigmoid()
    else:
        beta = beta.float()
    return q, k, g_log, beta


def _naive_op(q, k, v, g, beta, A_log=None, dt_bias=None, scale=None,
              initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
              use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
              lower_bound=None, state_v_first=False, cu_seqlens=None,
              safe_gate=False, **kwargs):
    from fla.ops.kda.naive import naive_chunk_kda, naive_recurrent_kda
    # KDA state math stays fp32 and OUTSIDE autocast (fp16 decays NaN: exp/cumsum)
    with torch.amp.autocast(q.device.type, enabled=False):
        if use_gate_in_kernel:
            q, k, g, beta = _prep(q, k, v, g, beta, A_log, dt_bias,
                                  use_qk_l2norm_in_kernel,
                                  use_beta_sigmoid_in_kernel)
        elif use_qk_l2norm_in_kernel:
            q = F.normalize(q.float(), dim=-1).to(torch.float32)
            k = F.normalize(k.float(), dim=-1).to(torch.float32)
            if use_beta_sigmoid_in_kernel:
                beta = beta.float().sigmoid()
        T = q.shape[1]
        # ALWAYS recurrent: naive_chunk_kda's backward is numerically broken
        # (grad NaN/inf on synthetic inputs — measured); recurrent grads clean.
        o, S = naive_recurrent_kda(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state)
    return o, S


def _torch_causal_conv1d(x, weight, bias=None, residual=None,
                         initial_state=None, output_final_state=False,
                         activation=None, **kwargs):
    """Pure-torch causal depthwise conv1d — the fla triton/cuda op's math."""
    B, T, D = x.shape
    W = weight.shape[-1]
    xf = x.float()
    if initial_state is not None:
        # cache (N, D, W) = last W inputs, oldest first (this shim's layout,
        # self-consistent prefill->decode): history for the next step is the
        # MOST RECENT W-1 entries, i.e. drop the oldest.
        hist = initial_state.float()[:, :, -(W - 1):]      # (B, D, W-1)
        xin = torch.cat([hist, xf.transpose(1, 2)], dim=-1)  # (B, D, T+W-1)
    else:
        xin = F.pad(xf.transpose(1, 2), (W - 1, 0))
    w = weight.float().reshape(D, 1, W)                      # (D, 1, W)
    y = F.conv1d(xin, w, bias.float() if bias is not None else None,
                 groups=D)
    y = y.transpose(1, 2)                                    # (B, T, D)
    if activation in ("silu", "swish"):
        y = F.silu(y)
    y = y.to(x.dtype)
    if residual is not None:
        y = y + residual
    final = None
    if output_final_state:
        final = xin[:, :, -W:].contiguous()                  # (B, D, W)
    return y, final


def _patch_short_conv():
    import fla.modules.conv.short_conv as sc
    from fla.modules import ShortConvolution

    def fwd(self, x, residual=None, mask=None, cache=None,
            output_final_state=False, cu_seqlens=None, chunk_indices=None,
            **kwargs):
        y, final = _torch_causal_conv1d(
            x, self.weight, self.bias, residual=residual,
            initial_state=cache, output_final_state=output_final_state,
            activation=self.activation)
        if cache is not None and final is not None:
            cache.copy_(final.to(cache.dtype))
            final = cache
        return y, final

    def step(self, x, residual=None, cache=None, output_final_state=False,
             **kwargs):
        return fwd(self, x, residual=residual, cache=cache,
                   output_final_state=output_final_state)

    ShortConvolution.forward = fwd
    ShortConvolution.step = step
    sc.causal_conv1d_fn_cuda = _torch_causal_conv1d


def _fused_rms_norm_gated(self, x, g, residual=None, prenorm=False,
                          residual_in_fp32=False, **kwargs):
    """Torch FusedRMSNormGated: rmsnorm(x [+ residual]) * act(g).

    Matches fla's fused_norm_gate kernel body exactly: residual joins x
    BEFORE the norm, the gate lands AFTER it — `y * silu(g)` for
    swish/silu, `y * sigmoid(g)` for sigmoid. KDA's o_norm is
    activation="sigmoid"; multiplying by raw unbounded g there (the old
    shim behavior) made every KDA layer's output garbage — the
    `to_c1c1_c10101...` subword-run decode bug on gfx906.
    """
    xf = x.float()
    if residual is not None:
        xf = xf + residual.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    y = xf * torch.rsqrt(var + self.eps) * self.weight.float()
    if self.bias is not None:
        y = y + self.bias.float()
    act = getattr(self, "activation", "swish")
    gf = g.float()
    if act in ("swish", "silu"):
        y = y * F.silu(gf)
    elif act == "sigmoid":
        y = y * torch.sigmoid(gf)
    else:
        raise ValueError(f"FusedRMSNormGated activation {act!r} not supported")
    y = y.to(torch.float32 if residual_in_fp32 else x.dtype)
    if prenorm:
        return y, xf.to(x.dtype)
    return y


def _rms_norm(self, x, residual=None, prenorm=False, residual_in_fp32=False,
              **kwargs):
    # fla semantics: residual joins x BEFORE the norm (fused kernel does
    # `b_x += residual` first), prenorm returns that sum.
    xf = x.float()
    if residual is not None:
        xf = xf + residual.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
    y = y * self.weight.float()
    if self.bias is not None:
        y = y + self.bias.float()
    y = y.to(torch.float32 if residual_in_fp32 else x.dtype)
    if prenorm:
        return y, xf.to(x.dtype)
    return y


def _rope_fwd(self, q, k, seqlen_offset=0, cu_seqlens=None, max_seqlen=None,
              chunk_indices=None, **kwargs):
    """Torch RoPE (half-split / non-interleaved) matching fla's triton math:
    o0 = x0*cos - x1*sin; o1 = x0*sin + x1*cos."""
    B, T, H, D = q.shape
    base = float(getattr(self, "base", 10000.0))
    if isinstance(seqlen_offset, torch.Tensor):
        t = (torch.arange(T, device=q.device, dtype=torch.float32)[None, :]
             + seqlen_offset.float().view(-1, 1))
    else:
        t = torch.arange(T, device=q.device, dtype=torch.float32) \
            + float(seqlen_offset)
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=q.device,
                                            dtype=torch.float32) / D))
    ang = t[..., None] * inv_freq
    cos, sin = ang.cos(), ang.sin()
    if cos.dim() == 2:                      # (T, D/2) -> (1, T, 1, D/2)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    else:                                   # (B, T, D/2) -> (B, T, 1, D/2)
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]

    def rot(x):
        x1, x2 = x[..., :D // 2].float(), x[..., D // 2:].float()
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        return torch.cat([o1, o2], -1).to(x.dtype)

    return rot(q), rot(k)


def install() -> bool:
    """Patch fla's KDA ops + ShortConvolution + norms + RoPE to pure torch.
    Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True
    try:
        import fla.layers.kda as fk
    except Exception:  # noqa: BLE001
        return False
    fk.chunk_kda = _naive_op
    fk.fused_recurrent_kda = _naive_op
    try:
        _patch_short_conv()
    except Exception:  # noqa: BLE001
        pass
    try:
        from fla.modules import FusedRMSNormGated
        FusedRMSNormGated.forward = _fused_rms_norm_gated
    except Exception:  # noqa: BLE001
        pass
    try:
        import fla.modules.layernorm as ln
        for _n in ("RMSNorm", "FusedRMSNorm"):
            _c = getattr(ln, _n, None)
            if _c is not None:
                _c.forward = _rms_norm
    except Exception:  # noqa: BLE001
        pass
    try:
        from fla.modules import RotaryEmbedding
        RotaryEmbedding.forward = _rope_fwd
    except Exception:  # noqa: BLE001
        pass
    _APPLIED = True
    os.environ.setdefault("VK_FLA_NAIVE", "1")
    return True
