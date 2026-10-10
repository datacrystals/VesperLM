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

from fla.layers import GatedLinearAttention, Mamba2, KimiDeltaAttention, MultiheadLatentAttention

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


def _env_flag(name: str) -> str:
    """Raw normalized env value ('' when unset) for one of the VESPER_* gates."""
    return os.environ.get(name, "").strip().lower()


def _torch_chunked_linear_ce(x, target, weight, ignore_index=-100, num_chunks=8):
    """Pure-torch chunked linear+CE used when fla's Triton FLCE cannot run
    (CPU smoke/probes). Same contract as
    fla.modules.fused_linear_cross_entropy_loss with reduction="mean":
    per-chunk logits [C, V], loss summed and divided by the number of
    non-ignored targets — i.e. exactly F.cross_entropy(...,
    ignore_index=..., reduction="mean") without holding the full [N, V].
    """
    N = x.shape[0]
    chunk = max(1, (N + num_chunks - 1) // num_chunks)
    total = target.ne(ignore_index).sum().clamp_min(1)
    loss_sum = x.new_zeros((), dtype=torch.float32)
    for s in range(0, N, chunk):
        logits = F.linear(x[s:s + chunk], weight)
        loss_sum = loss_sum + F.cross_entropy(
            logits, target[s:s + chunk], ignore_index=ignore_index,
            reduction="sum",
        )
    return loss_sum / total


class RecurrentStateCache(list):
    """Minimal fla-compatible past_key_values container for incremental
    decoding: a list of per-layer state dicts keyed by the layer's own
    layer_idx. fla layers read their previous state via
    `cache[layer_idx]` and store the new one via `cache.update(...)`;
    recurrent/conv states are always full replacements (never appended),
    so `offset` is accepted and ignored."""

    def update(self, layer_idx, recurrent_state=None, conv_state=None,
               offset=None, **kwargs):
        while len(self) <= layer_idx:
            self.append({})
        state = self[layer_idx]
        if recurrent_state is not None:
            state['recurrent_state'] = recurrent_state
        if conv_state is not None:
            state['conv_state'] = conv_state
        return state


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
        self.value_emb = None

    def enable_value_embedding(self, vocab_size):
        raise NotImplementedError(
            "value embeddings are not implemented for GLA layers "
            "(KDA/MLA/GQA only)")

    def forward(self, x, cache=None, value_tokens=None):
        # GLA's chunked Triton kernels cannot compile for fp16 on P40
        # (cc 6.1) and abort the process ("Unsupported rounding mode").
        # Run GLA in fp32 even when the caller is under fp16 autocast.
        # cache: optional RecurrentStateCache; fla switches to its
        # fused_recurrent kernel automatically for short (<=64) inputs and
        # threads the recurrent state through the cache (fp32 as well).
        with torch.autocast(device_type=x.device.type, enabled=False):
            out, _, _ = self.gla(x.float(), past_key_values=cache,
                                 use_cache=cache is not None)
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
        self.value_emb = None

    def enable_value_embedding(self, vocab_size):
        raise NotImplementedError(
            "value embeddings are not implemented for Mamba2 layers "
            "(no attention value path; KDA/MLA/GQA only)")

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

    def forward(self, x, cache=None, value_tokens=None):
        # Same fp32-enforcement as GatedLinearAttn: fla kernels must
        # not see fp16 (P40 cc 6.1 crashes) — run in fp32.
        # cache: optional RecurrentStateCache threading the conv/SSM state
        # through fla's pure-torch prefill/single-token decode paths.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            out, _, _ = self._mamba2_ssd(x, cache)
        return out

    def _mamba2_ssd(self, x, cache=None):
        # fla's rmsnorm_fn (gated RMSNorm) is Triton-only; a pure-torch
        # equivalent keeps the layer CPU/Portable-safe. Patching
        # module-level fla code globally would leak into GLA layers,
        # so rebind only for this call.
        import fla.modules.layernorm_gated as _lg
        orig = _lg.rmsnorm_fn
        _lg.rmsnorm_fn = self._torch_rmsnorm_gated
        try:
            return self.mamba2(x, past_key_values=cache,
                               use_cache=cache is not None)
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


class KimiDeltaAttn(nn.Module):
    """Thin wrapper around fla's KimiDeltaAttention (KDA) matching the
    GatedLinearAttn (x) -> out interface. `force_fp32` keeps the P40-safe
    behavior of the GLA wrapper; flip to False on bf16-capable hardware
    (MI300X) to let the chunk kernels run under autocast."""

    def __init__(self, dim, head_dim=64, layer_idx=None,
                 use_short_conv=True, force_fp32=True):
        super().__init__()
        self.force_fp32 = force_fp32
        self.kda = KimiDeltaAttention(
            mode='chunk',
            hidden_size=dim,
            expand_v=1.0,
            head_dim=head_dim,
            num_heads=dim // head_dim,
            use_short_conv=use_short_conv,
            layer_idx=layer_idx,
        )
        self.value_emb = None

    def enable_value_embedding(self, vocab_size):
        """Attach a token-id-indexed value table added to KDA's values
        (modded-nanogpt value embeddings). The table lives on this wrapper;
        injection is a forward hook on the v path so fla's own parameter
        keys (v_conv1d / v_proj weights) stay byte-identical."""
        width = self.kda.value_dim
        self.value_emb = nn.Embedding(vocab_size, width)
        self._ve_tokens = None
        if self.kda.use_short_conv:
            # the short-conv output IS the attention value fed to the
            # delta kernel — inject there (post-conv, pre-kernel).
            self.kda.v_conv1d.register_forward_hook(self._value_embed_hook)
        else:
            # no conv: v = silu(v_proj(x)) — inject on v_proj output,
            # i.e. pre-silu (only reached when kda_short_conv=False).
            self.kda.v_proj.register_forward_hook(self._value_embed_hook)
        return width

    def _value_embed_hook(self, module, args, output):
        tokens = getattr(self, '_ve_tokens', None)
        if tokens is None or self.value_emb is None:
            return output
        ve = self.value_emb(tokens)
        if isinstance(output, tuple):
            # ShortConvolution returns (y, final_state)
            y = output[0]
            return (y + ve.to(dtype=y.dtype, device=y.device),) + output[1:]
        return output + ve.to(dtype=output.dtype, device=output.device)

    def forward(self, x, cache=None, value_tokens=None):
        # cache: optional RecurrentStateCache; KDA threads its
        # recurrent/conv states through it exactly like GLA.
        if self.value_emb is not None:
            self._ve_tokens = value_tokens
        try:
            if self.force_fp32:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    out, _, _ = self.kda(x.float(), past_key_values=cache,
                                         use_cache=cache is not None)
                return out
            out, _, _ = self.kda(x, past_key_values=cache,
                                 use_cache=cache is not None)
            return out
        finally:
            if self.value_emb is not None:
                self._ve_tokens = None


class MultiheadLatentAttn(nn.Module):
    """Thin wrapper around fla's MultiheadLatentAttention (MLA) matching
    the GroupedQueryAttention (x, freqs_cis) call signature. MLA applies
    RoPE internally on its dedicated rope head-dim, so freqs_cis is
    accepted and ignored. Incremental KV caching is not wired yet —
    training and full-forward probes are unaffected."""

    def __init__(self, dim, n_heads, max_seq_len, kv_lora_rank=None,
                 q_lora_rank=None, qk_rope_head_dim=64,
                 qk_nope_head_dim=128, v_head_dim=128, layer_idx=None):
        super().__init__()
        self.mla = MultiheadLatentAttention(
            hidden_size=dim,
            num_heads=n_heads,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=kv_lora_rank or dim // 2,
            v_head_dim=v_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            max_position_embeddings=max_seq_len * 2,
            layer_idx=layer_idx,
        )
        self.value_emb = None

    def enable_value_embedding(self, vocab_size):
        """Attach a token-id-indexed value table added to MLA's values.

        MLA's v is not a standalone projection: fla's kv_proj is
        [Linear -> RMSNorm -> Linear] producing per-head
        (qk_nope | v) pairs that are split after the fact. The table is
        injected on kv_proj's final Linear output, writing into each
        head's v slice and leaving the qk_nope slice untouched — i.e.
        exactly the MLA v path. fla's parameter keys stay unchanged.
        """
        width = self.mla.num_heads * self.mla.v_head_dim
        self.value_emb = nn.Embedding(vocab_size, width)
        self._ve_tokens = None
        self.mla.kv_proj[2].register_forward_hook(self._value_embed_hook)
        return width

    def _value_embed_hook(self, module, args, output):
        tokens = getattr(self, '_ve_tokens', None)
        if tokens is None or self.value_emb is None:
            return output
        B, T = tokens.shape
        H = self.mla.num_heads
        nope = self.mla.qk_nope_head_dim
        vh = self.mla.v_head_dim
        ve = self.value_emb(tokens).view(B, T, H, vh)
        zeros = output.new_zeros(B, T, H, nope)
        add = torch.cat([zeros, ve.to(dtype=output.dtype)], dim=-1)
        return output + add.reshape(B, T, H * (nope + vh))

    def forward(self, x, freqs_cis=None, cache=None, start_pos=0,
                value_tokens=None):
        if cache is not None:
            raise NotImplementedError(
                "MLA incremental cache not wired; use full forward"
            )
        if self.value_emb is not None:
            self._ve_tokens = value_tokens
        try:
            out, _, _ = self.mla(x, attention_mask=None, use_cache=False)
            return out
        finally:
            if self.value_emb is not None:
                self._ve_tokens = None


class VesperLinearLM(nn.Module):
    """VesperLLM variant with a linear-attention/full-attention hybrid stack.

    Every `attention_every`-th layer keeps full attention (long-range
    recall): `full_type` selects "gqa" (softmax GQA, default) or "mla"
    (multi-head latent attention — the Vesper-K lineage). The rest use a
    linear-attention layer chosen by `linear_type`: "gla" (FLA GLA
    chunked — default), "mamba2" (fla Mamba2 SSD), or "kda" (Kimi Delta
    Attention — the Vesper-K lineage). Linear layers run O(T) compute
    and O(1) memory in sequence length; GLA/Mamba2 are forced to fp32,
    KDA by default too (`linear_force_fp32`) — flip it off on
    bf16-capable hardware. MoE FFN, RMSNorm pre-norm, and weight tying
    are unchanged.

    Speedrun opts (all env-gated, default OFF, checkpoint keys unchanged):
      VESPER_COMPILE=1  — torch.compile the dense submodules (see
                          `_apply_compile`); "whole" tries the full forward.
      VESPER_FUSED_CE=1 — fused logits+CE via fla FusedLinearCrossEntropy
                          (see `_fused_ce_loss`); logits are not materialized
                          and forward returns logits=None when targets are
                          given.
      VESPER_VALUE_EMBED=1 — token-id value embedding tables added to the
                          attention values of the first/last layers
                          (see `_attach_value_embeddings`).
    """

    def __init__(self, vocab_size=32000, dim=1024, n_layers=10, n_heads=8,
                 n_kv_heads=2, hidden_dim=1280, num_experts=8, top_k=2,
                 max_seq_len=1024, pad_id=0, dropout=0.0,
                 attention_every=4, gla_head_dim=64, qk_norm=True,
                 linear_type="gla", mamba2_state_size=128,
                 full_type="gqa", kda_head_dim=64, kda_short_conv=True,
                 kv_lora_rank=None, v_head_dim=128,
                 linear_force_fp32=True,
                 router_type="topk", passport_dim=64, router_expert_dropout=0.0,
                 grad_checkpoint=True, compile_mode=None, fused_ce=None,
                 value_embed=None):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.grad_checkpoint = grad_checkpoint
        if linear_type not in ("gla", "mamba2", "kda"):
            raise ValueError(f"Unknown linear_type '{linear_type}' (expected 'gla', 'mamba2' or 'kda')")
        if full_type not in ("gqa", "mla"):
            raise ValueError(f"Unknown full_type '{full_type}' (expected 'gqa' or 'mla')")
        if full_type == "mla" and qk_norm is not None and not qk_norm:
            pass  # qk_norm only applies to the GQA path; MLA normalizes internally
        self.tok_embeddings = nn.Embedding(vocab_size, dim)

        self.register_buffer("freqs_cis", precompute_freqs_cis(dim // n_heads, max_seq_len * 2))

        self.layers = nn.ModuleList()
        self.layer_types = []
        for i in range(n_layers):
            is_full_attn = (i % attention_every == attention_every - 1)
            if is_full_attn and full_type == "mla":
                attn = MultiheadLatentAttn(dim, n_heads, max_seq_len,
                                           kv_lora_rank=kv_lora_rank,
                                           v_head_dim=v_head_dim,
                                           layer_idx=i)
                self.layer_types.append('mla')
            elif is_full_attn:
                attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len,
                                             qk_norm=qk_norm)
                self.layer_types.append('full')
            elif linear_type == "mamba2":
                attn = Mamba2SSD(dim, head_dim=gla_head_dim,
                                 state_size=mamba2_state_size, layer_idx=i)
                self.layer_types.append('mamba2')
            elif linear_type == "kda":
                attn = KimiDeltaAttn(dim, head_dim=kda_head_dim, layer_idx=i,
                                     use_short_conv=kda_short_conv,
                                     force_fp32=linear_force_fp32)
                self.layer_types.append('kda')
            else:
                attn = GatedLinearAttn(dim, head_dim=gla_head_dim, layer_idx=i)
                self.layer_types.append('gla')

            self.layers.append(nn.ModuleDict({
                'attn': attn,
                'ffn': MoEFeedForward(dim, hidden_dim, num_experts, top_k,
                                      router_type=router_type, passport_dim=passport_dim,
                                      router_expert_dropout=router_expert_dropout),
                'attn_norm': RMSNorm(dim),
                'ffn_norm': RMSNorm(dim)
            }))

        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # Speedrun opt: value embeddings (VESPER_VALUE_EMBED, default off).
        # Attached before apply() so the tables get the standard Embedding
        # init (N(0, 0.02)) alongside everything else.
        self.value_embed_decision = self._attach_value_embeddings(value_embed)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('wo.weight') or pn.endswith('w2.weight') or pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

        # Weight tying: embedding and output projection share parameters
        self.tok_embeddings.weight = self.output.weight

        # Speedrun opt: optional torch.compile (VESPER_COMPILE, default off)
        self._compile_decision = self._apply_compile(compile_mode)

        # Speedrun opt: fused logits+CE (VESPER_FUSED_CE, default off)
        self.fused_ce, self._fused_ce_decision = self._resolve_fused_ce(fused_ce)
        if self.fused_ce:
            print(f"[fused-ce] {self._fused_ce_decision}")

    # --- speedrun opt: VESPER_FUSED_CE -----------------------------------
    def _resolve_fused_ce(self, fused_ce=None):
        """Env-gated fused linear-cross-entropy (default off).

        The trainer computes loss as
            F.cross_entropy(logits.view(-1, V), targets.view(-1),
                            ignore_index=pad_id)
        with pad_id = tokenizer.pad_token_id (== eos when the tokenizer has
        no pad token). fla's FusedLinearCrossEntropyLoss takes the same
        `ignore_index` and its reduction="mean" divides by the count of
        non-ignored targets — identical masking/normalization semantics —
        so the mask convention is preserved exactly. Chunked logsumexp may
        differ from the unfused path in the last ULP only.
        """
        raw = fused_ce if fused_ce is not None else _env_flag("VESPER_FUSED_CE")
        if isinstance(raw, bool):
            enabled = raw
        else:
            enabled = raw.strip().lower() in ("1", "true", "yes", "on")
        decision = (f"VESPER_FUSED_CE=1: logits+softmax+CE replaced by fla "
                    f"FusedLinearCrossEntropyLoss (ignore_index=pad_id={self.pad_id}, "
                    f"mean over non-ignored targets); full logits are never "
                    f"materialized and forward() returns logits=None whenever "
                    f"targets are passed. fla's FLCE is Triton-only — on CPU "
                    f"(smoke/probes) a pure-torch chunked equivalent with the "
                    f"same masking is used instead. Generation (targets=None) "
                    f"still returns real logits.")
        return enabled, decision

    def _fused_ce_loss(self, x, targets):
        """Chunked linear+CE over (x, targets) without materializing [N, V].

        Uses fla's FusedLinearCrossEntropyLoss on accelerator devices
        (Triton kernels); falls back to a pure-torch chunked equivalent on
        CPU where those kernels cannot run (same ignore_index/mean
        semantics — see `_torch_chunked_linear_ce`).
        """
        x_flat = x.reshape(-1, x.shape[-1])
        t_flat = targets.reshape(-1)
        weight = self.output.weight
        if x.device.type in ("cuda", "npu", "xpu"):
            from fla.modules.fused_linear_cross_entropy import fused_linear_cross_entropy_loss
            return fused_linear_cross_entropy_loss(
                x=x_flat, target=t_flat, weight=weight, bias=None,
                ignore_index=self.pad_id, label_smoothing=0.0,
                num_chunks=8, reduction="mean",
            )
        return _torch_chunked_linear_ce(x_flat, t_flat, weight,
                                        ignore_index=self.pad_id)

    # --- speedrun opt: VESPER_VALUE_EMBED --------------------------------
    def _attach_value_embeddings(self, value_embed=None):
        """Env-gated modded-nanogpt-style value embeddings (default off).

        A learned token-id-indexed embedding table per site, added to the
        attention values of the first and last layers. Models with the flag
        off are untouched (no hooks, no extra parameters — existing
        checkpoints load byte-identically). With the flag on, the only new
        state_dict keys are `layers.{0,L-1}.attn.value_emb.weight` (every
        other key keeps its name and shape), so flag-on checkpoints are new
        artifacts and cannot be loaded into a flag-off model.

        Supported layer types: kda (post short-conv v), mla (v slice of the
        kv latent), full/GQA (wv output). gla/mamba2 sites are skipped with
        a logged warning. Tables use the repo's Embedding init N(0, 0.02).
        """
        raw = value_embed if value_embed is not None else _env_flag("VESPER_VALUE_EMBED")
        if isinstance(raw, bool):
            enabled = raw
        else:
            enabled = raw.strip().lower() in ("1", "true", "yes", "on")
        if not enabled:
            return "off (VESPER_VALUE_EMBED unset)"
        n_layers = len(self.layers)
        sites = list(dict.fromkeys([0, n_layers - 1]))
        applied, skipped = [], []
        for i in sites:
            attn = self.layers[i]['attn']
            layer_type = self.layer_types[i]
            if layer_type in ('gla', 'mamba2'):
                skipped.append(f"layer {i} ({layer_type})")
                continue
            width = attn.enable_value_embedding(self.tok_embeddings.num_embeddings)
            applied.append(f"layer {i} ({layer_type}, width={width})")
        decision = (f"VESPER_VALUE_EMBED=1: token-id value tables added to "
                    f"attention values on first/last layers: "
                    f"{', '.join(applied) if applied else 'none'}")
        if skipped:
            decision += (f"; SKIPPED {', '.join(skipped)} — value embedding "
                         f"is implemented for kda/mla/gqa only")
            print(f"[value-embed] warning: skipped {', '.join(skipped)}")
        print(f"[value-embed] {decision}")
        return decision

    # --- speedrun opt: VESPER_COMPILE ------------------------------------
    def _apply_compile(self, compile_mode=None):
        """Env-gated torch.compile (nanogpt-speedrun style). Default off.

        Tiers (VESPER_COMPILE):
          "1"/"module" — compile dense submodule forwards only: MoE expert
              MLPs and GQA attention. The fla attention wrappers stay eager:
              their Triton chunk kernels do not trace under inductor (they
              only graph-break), and the MoE dispatch loop has dynamic
              per-expert shapes. This is the clean compile boundary for this
              model. state_dict keys are unaffected (forward methods are
              wrapped in place, modules are not replaced).
          "whole" — best-effort torch.compile of the whole forward with
              dynamic=True; if it raises at runtime the model falls back to
              the module tier permanently. Intended for experiments; fla
              graph breaks + the seq-len curriculum make it a gamble.
        """
        raw = compile_mode if compile_mode is not None else _env_flag("VESPER_COMPILE")
        raw = raw.strip().lower()
        if raw in ("", "0", "false", "off", "no"):
            return "off (VESPER_COMPILE unset)"
        if raw in ("1", "true", "yes", "on", "module"):
            tier = "module"
        elif raw in ("whole", "full", "model"):
            tier = "whole"
        else:
            raise ValueError(
                f"VESPER_COMPILE={raw!r} not recognized (expected '1'/'module' or 'whole')"
            )
        if tier == "module":
            decision = self._compile_dense_submodules(raw)
            print(f"[compile] {decision}")
            return decision

        eager_forward = self.forward
        compiled_forward = torch.compile(eager_forward, dynamic=True)
        state = {"failed": False}

        def guarded_forward(*args, **kwargs):
            if state["failed"]:
                return eager_forward(*args, **kwargs)
            try:
                return compiled_forward(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - any compile/run failure falls back
                state["failed"] = True
                fallback = self._compile_dense_submodules(raw)
                print(f"[compile] whole-model torch.compile failed at runtime "
                      f"({type(exc).__name__}: {exc}); falling back -> {fallback}")
                return eager_forward(*args, **kwargs)

        self.forward = guarded_forward
        decision = (f"VESPER_COMPILE={raw}: whole-model torch.compile(dynamic=True) "
                    f"around forward (best-effort; fla Triton ops graph-break; "
                    f"auto-falls back to dense-submodule compile on failure)")
        print(f"[compile] {decision}")
        return decision

    def _compile_dense_submodules(self, raw):
        """Compile the dense pieces that inductor can actually fuse.

        Wraps each module's `forward` in place so state_dict keys/shapes stay
        byte-identical to the eager model (no OptimizedModule._orig_mod
        prefix on parameters).
        """
        n = 0
        for layer in self.layers:
            for expert in layer['ffn'].experts:
                expert.forward = torch.compile(expert.forward, dynamic=True)
                n += 1
            attn = layer['attn']
            if isinstance(attn, GroupedQueryAttention):
                attn.forward = torch.compile(attn.forward, dynamic=True)
                n += 1
        return (f"VESPER_COMPILE={raw}: compiled {n} dense submodule forwards "
                f"(MoE expert MLPs + GQA); fla attention wrappers left eager "
                f"(Triton kernels do not trace under inductor)")

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
            # 'full'/'mla' layers take freqs (MLA ignores it — RoPE is
            # internal); 'gla'/'mamba2'/'kda' are linear-only
            is_linear = self.layer_types[i] in ('gla', 'mamba2', 'kda')

            if self.training and self.grad_checkpoint:
                # Checkpoint the WHOLE layer (attention/GLA + MoE) in one unit.
                # non-reentrant checkpointing handles the MoE's dynamic routing
                # shapes fine — routing is deterministic here (no dropout), so
                # the backward recompute reproduces the same expert assignment.
                # freqs_cis is a constant buffer, captured via closure.
                # tokens is captured for value embeddings (VESPER_VALUE_EMBED);
                # it is an integer tensor and needs no grad through the
                # checkpoint boundary.
                def create_layer_forward(module_dict, linear, freqs, value_tokens):
                    def custom_forward(x_in):
                        if linear:
                            attn_out = module_dict['attn'](module_dict['attn_norm'](x_in),
                                                           value_tokens=value_tokens)
                        else:
                            attn_out = module_dict['attn'](module_dict['attn_norm'](x_in), freqs,
                                                           value_tokens=value_tokens)
                        h = x_in + attn_out
                        ffn_out, aux = module_dict['ffn'](module_dict['ffn_norm'](h))
                        return h + ffn_out, aux
                    return custom_forward

                x, aux_loss = cp.checkpoint(
                    create_layer_forward(layer, is_linear, self.freqs_cis, tokens),
                    x,
                    use_reentrant=False
                )
            else:
                if is_linear:
                    attn_out = layer['attn'](layer['attn_norm'](x), value_tokens=tokens)
                else:
                    attn_out = layer['attn'](layer['attn_norm'](x), self.freqs_cis,
                                             value_tokens=tokens)
                x = x + attn_out
                ffn_out, aux_loss = layer['ffn'](layer['ffn_norm'](x))
                x = x + ffn_out

            total_aux_loss += aux_loss

        x = self.norm(x)
        if targets is not None and self.fused_ce:
            # Speedrun opt: skip logits materialization entirely (V=65536
            # logits are the dominant activation at long T). Logits are
            # returned as None; callers that need them must call forward
            # without targets (the generation path).
            logits = None
            ce_loss = self._fused_ce_loss(x, targets)
        else:
            logits = self.output(x)
            ce_loss = None
            if targets is not None:
                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=self.pad_id
                )

        return logits, ce_loss, total_aux_loss

    def new_cache(self, batch_size, device):
        """Allocate an incremental-decoding cache for this model.

        Returns a dict with:
          'kv':     {layer_idx: {'k', 'v'}} for the 'full' GQA layers —
                    preallocated (B, n_kv_heads, max_seq_len, head_dim)
                    UNEXPANDED K/V tensors, written in place.
          'linear': one RecurrentStateCache shared by all GLA/Mamba2/KDA
                    layers (each keys it by its own layer_idx).

        Raises ValueError for MLA stacks: MLA's latent cache is not
        wired yet, use full forward for generation instead.
        """
        if 'mla' in self.layer_types:
            raise ValueError(
                "new_cache() does not support MLA layers yet; "
                "generate via full forward() instead."
            )
        kv = {}
        for i, layer_type in enumerate(self.layer_types):
            if layer_type == 'full':
                attn = self.layers[i]['attn']
                shape = (batch_size, attn.n_kv_heads, self.max_seq_len, attn.head_dim)
                dtype = self.tok_embeddings.weight.dtype
                kv[i] = {
                    'k': torch.zeros(shape, device=device, dtype=dtype),
                    'v': torch.zeros(shape, device=device, dtype=dtype),
                }
        return {'kv': kv, 'linear': RecurrentStateCache()}

    @torch.no_grad()
    def forward_incremental(self, tokens, caches, pos):
        """Cached forward for incremental generation. `forward` is untouched;
        training never takes this path.

        tokens: (B, T) token ids occupying absolute positions [pos, pos+T)
                (pass the whole prompt with pos == 0 for prefill, then one
                token per decode step).
        caches: dict from new_cache(), updated in place.
        pos:    absolute position of tokens[:, 0] in the sequence.

        Returns (logits, None, aux_loss, caches), with logits for the new
        tokens only. MLA stacks are unsupported (latent cache not wired).
        """
        if 'mla' in self.layer_types:
            raise ValueError(
                "forward_incremental does not support MLA layers yet; "
                "generate via full forward() instead."
            )
        B, T = tokens.size()
        if pos + T > self.max_seq_len:
            raise ValueError(
                f"Position {pos} + length {T} exceeds max_seq_len {self.max_seq_len}; "
                "the caller must fall back to a truncated full forward."
            )

        x = self.tok_embeddings(tokens)
        total_aux_loss = 0.0

        for i, layer in enumerate(self.layers):
            if self.layer_types[i] == 'full':
                attn_out = layer['attn'](layer['attn_norm'](x), self.freqs_cis,
                                         cache=caches['kv'][i], start_pos=pos,
                                         value_tokens=tokens)
            else:
                attn_out = layer['attn'](layer['attn_norm'](x),
                                         cache=caches['linear'],
                                         value_tokens=tokens)
            x = x + attn_out
            ffn_out, aux_loss = layer['ffn'](layer['ffn_norm'](x))
            x = x + ffn_out
            total_aux_loss += aux_loss

        x = self.norm(x)
        logits = self.output(x)

        return logits, None, total_aux_loss, caches
