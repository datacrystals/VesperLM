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

from fla.layers import GatedLinearAttention

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
        out, _, _ = self.gla(x)
        return out


class VesperLinearLM(nn.Module):
    """VesperLLM variant with a GLA/softmax hybrid attention stack.

    Every `attention_every`-th layer keeps full GQA softmax attention
    (long-range recall); the rest use gated linear attention (FLA GLA,
    chunked, O(T) compute and O(1) memory in sequence length).
    MoE FFN, RMSNorm pre-norm, and weight tying are unchanged.
    """

    def __init__(self, vocab_size=32000, dim=1024, n_layers=10, n_heads=8,
                 n_kv_heads=2, hidden_dim=1280, num_experts=8, top_k=2,
                 max_seq_len=1024, pad_id=0, dropout=0.0,
                 attention_every=4, gla_head_dim=64, qk_norm=True):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
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
            tokens = tokens[:, :self.max_seq_len]
            if targets is not None:
                targets = targets[:, :self.max_seq_len]

        x = self.tok_embeddings(tokens)
        total_aux_loss = 0.0

        for i, layer in enumerate(self.layers):
            is_gla = self.layer_types[i] == 'gla'

            def create_attn_forward(module_dict, linear):
                def custom_forward(x_in, freqs):
                    if linear:
                        attn_out = module_dict['attn'](module_dict['attn_norm'](x_in))
                    else:
                        attn_out = module_dict['attn'](module_dict['attn_norm'](x_in), freqs)
                    return x_in + attn_out
                return custom_forward

            if self.training:
                # Attention/GLA layers have static shapes -> safe to checkpoint.
                # MoE stays outside (dynamic routing shapes break recompute).
                x = cp.checkpoint(
                    create_attn_forward(layer, is_gla),
                    x,
                    self.freqs_cis,
                    use_reentrant=False
                )
                ffn_out, aux_loss = layer['ffn'](layer['ffn_norm'](x))
                x = x + ffn_out
            else:
                x = create_attn_forward(layer, is_gla)(x, self.freqs_cis)
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
