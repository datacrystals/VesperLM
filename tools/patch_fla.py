#!/usr/bin/env python3
"""Apply VesperLM's local patches to an installed flash-linear-attention (fla).

Two patches, both idempotent:

1. fla/utils/env.py — find_spec_cached() raises ModuleNotFoundError when a
   probed parent package (e.g. triton.experimental for the gluon backend)
   is absent, instead of returning None. Catch and return None so backend
   probing degrades gracefully on triton < 3.4.

2. fla/layers/mla.py — MultiheadLatentAttention hard-requires flash-attn.
   Substitute PyTorch-SDPA implementations of flash_attn_func /
   flash_attn_varlen_func at the import guard so MLA runs anywhere
   (CUDA without flash-attn, ROCm/MI300X, CPU for probes). Sliding-window
   is unsupported in the fallback; the varlen path assumes q_len == k_len
   per sequence (true for our training/probe usage, which never passes an
   attention mask).

3. fla/ops/kda/{chunk_bwd,chunk_intra,gate,wy_fast}.py — ROCm only
   (torch.version.hip set). The AMD triton backend's software pipeliner
   miscompiles delta-rule chunk kernels at num_stages>=3 ("'tt.load' op
   operation destroyed but still has uses" in make_ttgir; upstream
   triton#9815, seen on gfx942 with triton 3.5.1). Cap the autotune
   configs at num_stages=2 so the kernels compile. Mild perf cost; no-op
   on CUDA.

Run after installing fla on any new machine:
    python tools/patch_fla.py
Safe to re-run; exits nonzero if a pattern is not found (fla version
drift — re-derive the patch instead of silently skipping).
"""

import importlib.util
import sys

_spec = importlib.util.find_spec("fla")  # locate without executing: fla's
# __init__ builds triton autotuners, which dies on machines with no live GPU
# driver (wedged post-suspend, CPU-only build hosts). Patching only needs
# the package directory.
if _spec is None or _spec.origin is None:
    raise SystemExit("fla is not installed in this python environment")
import os
fla_dir = os.path.dirname(_spec.origin)


def _apply(path, marker, old, new):
    src = open(path).read()
    if marker in src:
        print(f"already patched: {path}")
        return
    if old not in src:
        raise SystemExit(
            f"PATCH PATTERN NOT FOUND in {path} — fla version drifted, "
            "re-derive the patch by hand."
        )
    open(path, "w").write(src.replace(old, new, 1))
    print(f"patched: {path}")


def _apply_all(path, marker, pairs):
    src = open(path).read()
    if marker in src:
        print(f"already patched: {path}")
        return
    matched = 0
    for old, new in pairs:
        count = src.count(old)
        if count:
            src = src.replace(old, new)
            matched += count
    if matched == 0:
        raise SystemExit(
            f"PATCH PATTERN NOT FOUND in {path} — fla version drifted, "
            "re-derive the patch by hand."
        )
    open(path, "w").write(src)
    print(f"patched: {path}")


def main():

    _apply(
        os.path.join(fla_dir, "utils", "env.py"),
        "except ModuleNotFoundError",
        """@functools.cache
def find_spec_cached(name):
    return find_spec(name)""",
        """@functools.cache
def find_spec_cached(name):
    try:
        return find_spec(name)
    except ModuleNotFoundError:
        return None""",
    )

    _apply(
        os.path.join(fla_dir, "layers", "mla.py"),
        "falling back to PyTorch SDPA for MLA",
        """try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None""",
        """try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed; falling back to PyTorch SDPA for MLA "
        "(sliding-window unsupported; varlen causal assumes q_len == k_len)",
        category=ImportWarning,
    )

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                        window_size=(-1, -1), **kwargs):
        if window_size not in (None, (-1, -1)):
            raise NotImplementedError("SDPA fallback does not support sliding window")
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p,
                                           is_causal=causal, scale=softmax_scale)
        return o.transpose(1, 2)

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                               max_seqlen_k, dropout_p=0.0, softmax_scale=None,
                               causal=False, window_size=(-1, -1), **kwargs):
        if window_size not in (None, (-1, -1)):
            raise NotImplementedError("SDPA fallback does not support sliding window")
        outs = []
        for i in range(len(cu_seqlens_q) - 1):
            qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
            ks, ke = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
            oi = F.scaled_dot_product_attention(
                q[qs:qe].transpose(0, 1), k[ks:ke].transpose(0, 1), v[ks:ke].transpose(0, 1),
                dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
            outs.append(oi.transpose(0, 1))
        return torch.cat(outs, dim=0)""",
    )

    import torch
    if torch.version.hip is not None:
        for fname in ("chunk_bwd.py", "chunk_intra.py", "gate.py", "wy_fast.py"):
            _apply_all(
                os.path.join(fla_dir, "ops", "kda", fname),
                "for num_stages in [2]  # ROCm: cap stages",
                [("for num_stages in [2, 3, 4]",
                  "for num_stages in [2]  # ROCm: cap stages (triton#9815)"),
                 ("for num_stages in [2, 3]",
                  "for num_stages in [2]  # ROCm: cap stages (triton#9815)")],
            )

    print("fla patches OK")


if __name__ == "__main__":
    sys.exit(main())
