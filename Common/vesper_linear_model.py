import sys
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

# Make sure Triton kernels compile once into a warm cache instead of every run
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("FLA_CACHE_DIR", "/tmp/fla_cache")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fla.layers import GatedLinearAttention, Mamba2

# Reuse the dense model's building blocks so the two architectures stay in sync
from vesper_model import (
    precompute_freqs_cis,
    apply_rotary_emb,
    RMSNorm,
    GroupedQueryAttention,
    FeedForward,
    TopKRouter,
    MoEFeedForward,
)


class GatedLinearAttn(nn.Module):
    """Thin wrapper so linear-attention layers share the (x) -> out interface
    style of GroupedQueryAttention without needing freqs_cis."""

    def __init__(self, dim, head_dim=64, layer_idx=None):
        super().__init__()
        # P40 benchmark: GLA chunk kernel is ~16x faster at head_dim=64 than 128
        self.gla = GatedLinearAttention(
            mode='chunk',
            hidden_size=dim,
            expand_k=1.0,
            expand_v=1.0,
            num_heads=dim // head_dim,
            use_short_conv=False,
            use_output_gate=True,
            gate_fn='swish',
            fuse_norm=True,
            layer_idx=layer_idx,
        )

    def forward(self, x):
        # GLA's chunked Triton kernels cannot compile for fp16 on P40
        # (cc 6.1) and abort the process ("Unsupported rounding mode").
        # Run GLA in fp32 even when the caller is under fp16 autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            out, _, _ = self.gla(x.float())
        return out


class Mamba2SSD(nn.Module):
    """Thin wrapper around fla's Mamba2 layer (SSD path), mirroring
    GatedLinearAttn's (x) -> out interface.

    fla's Mamba2 uses Triton kernels for the gated swish activation
    and the gated RMSNorm, which cannot run on CPU tensors (and would
    not compile for fp16 on P40 / cc 6.1). We fall back to pure-torch
    equivalents for both, and to a plain Conv1d for the causal conv —
    so this layer runs in fp32 everywhere, exactly like the fp32-GLA
    workaround above.
    """

    def __init__(self, dim, head_dim=64, state_size=128, chunk_size=256,
                 layer_idx=None):
        super().__init__()
        self.mamba2 = Mamba2(
            hidden_size=dim,
            expand=2,
            head_dim=head_dim,
            state_size=state_size,
            chunk_size=chunk_size,
            layer_idx=layer_idx,
        )
        # fla's Triton swish / gated-RMSNorm / causal-conv kernels do
        # not support CPU tensors (and fp16 P40 kernels crash); swap
        # in pure-torch equivalents so the layer runs fp32 anywhere.
        self.mamba2.act = F.silu
        self.mamba2.causal_conv1d_fn = self._torch_conv1d

    @staticmethod
    def _torch_conv1d(x, weight, bias=None, activation=None, **kwargs):
        # x: (B, D, T); weight: (D, K) — mirrors fla's triton
        # causal_conv1d signature (returns (out, cache) tuple)
        kernel = weight.shape[-1]
        out = F.conv1d(F.pad(x, (kernel - 1, 0)),
                       weight.unsqueeze(1), bias, groups=x.shape[1])
        if activation in ("silu", "swish"):
            out = F.silu(out)
        return out, None

    def forward(self, x):
        # Same fp32-enforcement as GatedLinearAttn: fla kernels must
        # not see fp16 (P40 cc 6.1 crashes) — run in fp32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            out, _, _ = self._mamba2_ssd(x)
        return out

    def _mamba2_ssd(self, x):
        # fla's rmsnorm_fn (gated RMSNorm) is Triton-only; a pure-torch
        # equivalent keeps the layer CPU/Portable-safe. Patching
        # module-level fla code globally would leak into GLA layers,
        # so rebind only for this call.
        import fla.modules.layernorm_gated as _lg
        orig = _lg.rmsnorm_fn
        _lg.rmsnorm_fn = self._torch_rmsnorm_gated
        try:
            return self.mamba2(x)
        finally:
            _lg.rmsnorm_fn = orig

    @staticmethod
    def _torch_rmsnorm_gated(x, weight, bias=None, z=None, eps=1e-6,
                             group_size=None, norm_before_gate=True):
        # x: (T, D); z: optional gate (T, D) — mirrors fla's
        # rmsnorm_fn signature
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
        if z is not None:
            g = F.silu(z) if norm_before_gate else z
            out = out * g
        return out


class VesperLinearLM(nn.Module):
    """VesperLLM variant with a GLA/softmax hybrid attention stack.

    Every `attention_every`-th layer keeps full GQA softmax attention
    (long-range recall); the rest use a linear-attention layer chosen
    by `linear_type` ("gla": FLA GLA chunked — default, or "mamba2":
    fla Mamba2 SSD). Both run O(T) compute and O(1) memory in
    sequence length, and both are forced to fp32 (see their wrappers).
    MoE FFN, RMSNorm pre-norm, and weight tying are unchanged.
    """

    def __init__(self, vocab_size=32000, dim=1024, n_layers=10, n_heads=8,
                 n_kv_heads=2, hidden_dim=1280, num_experts=8, top_k=2,
                 max_seq_len=1024, pad_id=0, dropout=0.0,
                 attention_every=4, gla_head_dim=64, qk_norm=True,
                 linear_type="gla", mamba2_state_size=128,
                 grad_checkpoint=True):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.grad_checkpoint = grad_checkpoint
        if linear_type not in ("gla", "mamba2"):
            raise ValueError(f"Unknown linear_type '{linear_type}' (expected 'gla' or 'mamba2')")
        self.tok_embeddings = nn.Embedding(vocab_size, dim)

        self.register_buffer("freqs_cis", precompute_freqs_cis(dim // n_heads, max_seq_len * 2))

        self.layers = nn.ModuleList()
        self.layer_types = []
        for i in range(n_layers):
            is_full_attn = (i % attention_every == attention_every - 1)
            if is_full_attn:
                attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len,
                                             qk_norm=qk_norm)
                self.layer_types.append('full')
            elif linear_type == "mamba2":
                attn = Mamba2SSD(dim, head_dim=gla_head_dim,
                                 state_size=mamba2_state_size, layer_idx=i)
                self.layer_types.append('mamba2')
            else:
                attn = GatedLinearAttn(dim, head_dim=gla_head_dim, layer_idx=i)
                self.layer_types.append('gla')

            self.layers.append(nn.ModuleDict({
                'attn': attn,
                'ffn': MoEFeedForward(dim, hidden_dim, num_experts, top_k),
                'attn_norm': RMSNorm(dim),
                'ffn_norm': RMSNorm(dim)
            }))

        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('wo.weight') or pn.endswith('w2.weight') or pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

        # Weight tying: embedding and output projection share parameters
        self.tok_embeddings.weight = self.output.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        B, T = tokens.size()
        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}. "
                "Refusing to silently truncate (that used to silently drop training data)."
            )

        x = self.tok_embeddings(tokens)
        total_aux_loss = 0.0

        for i, layer in enumerate(self.layers):
            # 'full' layers take freqs; 'gla'/'mamba2' are linear-only
            is_gla = self.layer_types[i] != 'full'

            if self.training and self.grad_checkpoint:
                # Checkpoint the WHOLE layer (attention/GLA + MoE) in one unit.
                # non-reentrant checkpointing handles the MoE's dynamic routing
                # shapes fine — routing is deterministic here (no dropout), so
                # the backward recompute reproduces the same expert assignment.
                # freqs_cis is a constant buffer, captured via closure.
                def create_layer_forward(module_dict, linear, freqs):
                    def custom_forward(x_in):
                        if linear:
                            attn_out = module_dict['attn'](module_dict['attn_norm'](x_in))
                        else:
                            attn_out = module_dict['attn'](module_dict['attn_norm'](x_in), freqs)
                        h = x_in + attn_out
                        ffn_out, aux = module_dict['ffn'](module_dict['ffn_norm'](h))
                        return h + ffn_out, aux
                    return custom_forward

                x, aux_loss = cp.checkpoint(
                    create_layer_forward(layer, is_gla, self.freqs_cis),
                    x,
                    use_reentrant=False
                )
            else:
                if is_gla:
                    attn_out = layer['attn'](layer['attn_norm'](x))
                else:
                    attn_out = layer['attn'](layer['attn_norm'](x), self.freqs_cis)
                x = x + attn_out
                ffn_out, aux_loss = layer['ffn'](layer['ffn_norm'](x))
                x = x + ffn_out

            total_aux_loss += aux_loss

        x = self.norm(x)
        logits = self.output(x)

        ce_loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.pad_id
            )

        return logits, ce_loss, total_aux_loss
