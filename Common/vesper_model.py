import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import math


# --- RoPE Helper Functions ---
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, T, 1, head_dim/2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, max_seq_len=2048, qk_norm=False):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        # QK-norm: RMSNorm on Q and K (before RoPE) keeps attention
        # logits bounded and stabilizes long training runs.
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None

        # Speedrun opt (VESPER_VALUE_EMBED): token-id value table added to
        # the attention values; created via enable_value_embedding() so the
        # dense VesperLLM path stays parameter-identical by default.
        self.value_emb = None

    def enable_value_embedding(self, vocab_size):
        """Attach a token-id-indexed value table added to v (see
        VesperLinearLM._attach_value_embeddings)."""
        width = self.n_kv_heads * self.head_dim
        self.value_emb = nn.Embedding(vocab_size, width)
        return width

    def forward(self, x, freqs_cis, cache=None, start_pos=0, value_tokens=None):
        # cache: optional dict with preallocated 'k'/'v' tensors of shape
        # (B, n_kv_heads, max_seq_len, head_dim) for incremental decoding;
        # start_pos is the absolute position of x's first token. With
        # cache=None and start_pos=0 the behavior is unchanged.
        # value_tokens: (B, T) token ids for VESPER_VALUE_EMBED injection.
        B, T, C = x.size()

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x)
        if self.value_emb is not None and value_tokens is not None:
            v = v + self.value_emb(value_tokens).to(dtype=v.dtype)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = apply_rotary_emb(q, k, freqs_cis[start_pos:start_pos + T])

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if cache is not None:
            # Append the new UNEXPANDED K/V at their absolute positions and
            # attend over everything written so far; expansion to n_heads
            # still happens at attention time, exactly as without a cache.
            cache['k'][:, :, start_pos:start_pos + T] = k
            cache['v'][:, :, start_pos:start_pos + T] = v
            k = cache['k'][:, :, :start_pos + T].to(k.dtype)
            v = cache['v'][:, :, :start_pos + T].to(v.dtype)

        kv_len = k.size(2)

        # Expand K and V to match Q's head count for GQA
        k = k[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, kv_len, self.head_dim).reshape(B, self.n_heads, kv_len, self.head_dim)
        v = v[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, kv_len, self.head_dim).reshape(B, self.n_heads, kv_len, self.head_dim)

        if start_pos == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T == 1:
            # A single cached query attends to the whole prefix: no mask.
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            # Cached chunk with T > 1: causal mask shifted right by start_pos
            # so query i can see keys 0..start_pos+i.
            mask = torch.ones(T, kv_len, dtype=torch.bool, device=x.device).tril(diagonal=start_pos)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TopKRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        # x: (N, dim) where N = B*T (flattened tokens)
        logits = self.gate(x)
        routing_weights = F.softmax(logits, dim=-1)
        top_weights, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        # Switch Transformer aux loss for load balancing
        # mean routing probability per expert (soft, differentiable)
        mean_router_probs = routing_weights.mean(dim=0)
        # fraction of tokens dispatched to each expert (hard, for balancing signal)
        expert_mask = torch.zeros_like(routing_weights).scatter_(1, top_indices, 1.0)
        mean_expert_usage = expert_mask.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(mean_router_probs * mean_expert_usage)

        # Renormalize selected weights so they sum to 1
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        return top_weights, top_indices, aux_loss


class PassportRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k=2, passport_dim=64, expert_dropout=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dropout = expert_dropout
        self.passport_dim = passport_dim
        self.query = nn.Linear(dim, passport_dim, bias=False)
        self.passports = nn.Parameter(torch.randn(num_experts, passport_dim) * 0.02)

    def forward(self, x):
        # x: (N, dim) where N = B*T (flattened tokens)
        logits = self.query(x) @ self.passports.t() / math.sqrt(self.passport_dim)

        if self.training and self.expert_dropout > 0:
            # Per-call bernoulli dropout over experts: dropped experts are out
            # of candidacy (-inf before softmax) so routing must match token
            # content to passport content instead of memorizing index
            # shortcuts. Always keep at least top_k experts selectable.
            keep = torch.rand(self.num_experts, device=logits.device) >= self.expert_dropout
            if keep.sum() < self.top_k:
                keep[:self.top_k] = True
            logits = logits.masked_fill(~keep.unsqueeze(0), float('-inf'))

        routing_weights = F.softmax(logits, dim=-1)
        top_weights, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        # Switch Transformer aux loss for load balancing
        # mean routing probability per expert (soft, differentiable)
        mean_router_probs = routing_weights.mean(dim=0)
        # fraction of tokens dispatched to each expert (hard, for balancing signal)
        expert_mask = torch.zeros_like(routing_weights).scatter_(1, top_indices, 1.0)
        mean_expert_usage = expert_mask.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(mean_router_probs * mean_expert_usage)

        # Renormalize selected weights so they sum to 1
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        return top_weights, top_indices, aux_loss

    @torch.no_grad()
    def register_expert(self, passport_init=None):
        if passport_init is None:
            row = torch.randn(1, self.passport_dim, device=self.passports.device,
                              dtype=self.passports.dtype) * 0.02
        else:
            row = passport_init.to(self.passports).reshape(1, -1)
        self.passports = nn.Parameter(torch.cat([self.passports.data, row], dim=0))
        self.num_experts += 1
        return self.num_experts - 1


class MoEFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=8, top_k=2, router_type="topk",
                 passport_dim=64, router_expert_dropout=0.0):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.router_type = router_type
        self.experts = nn.ModuleList([FeedForward(dim, hidden_dim) for _ in range(num_experts)])
        if router_type == "passport":
            self.router = PassportRouter(dim, num_experts, top_k, passport_dim=passport_dim,
                                         expert_dropout=router_expert_dropout)
        elif router_type == "topk":
            self.router = TopKRouter(dim, num_experts, top_k)
        else:
            raise ValueError(f"Unknown router_type '{router_type}' (expected 'topk' or 'passport')")

    def add_expert(self):
        self.experts.append(FeedForward(self.dim, self.hidden_dim))
        if self.router_type == "passport":
            self.router.register_expert()
        else:
            gate = self.router.gate
            with torch.no_grad():
                row = torch.randn(1, gate.in_features, device=gate.weight.device,
                                  dtype=gate.weight.dtype) * 0.02
                gate.weight = nn.Parameter(torch.cat([gate.weight.data, row], dim=0))
                gate.out_features += 1
            self.router.num_experts += 1
        self.num_experts += 1
        return self.num_experts - 1

    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(-1, C)  # (N, C) where N = B*T
        N = x_flat.size(0)

        routing_weights, selected_experts, aux_loss = self.router(x_flat)
        # routing_weights: (N, top_k)
        # selected_experts: (N, top_k)

        # Token-permutation dispatch: sort (token, slot) pairs by expert id so
        # each expert runs one contiguous GEMM batch, then scatter-add the
        # weighted outputs back to their token positions. One gather, E batched
        # expert calls, one index_add_ — replaces the old per-expert boolean
        # masking, which was ~1.5x slower fwd+bwd on P40.
        flat_e = selected_experts.reshape(-1)               # (N*K,)
        flat_w = routing_weights.reshape(-1, 1)             # (N*K, 1)
        tok = torch.arange(N, device=x.device).repeat_interleave(self.top_k)

        order = torch.argsort(flat_e, stable=True)
        flat_e, tok, flat_w = flat_e[order], tok[order], flat_w[order]
        counts = torch.bincount(flat_e, minlength=self.num_experts).tolist()

        gathered = x_flat[tok]                              # (N*K, C)
        outputs = torch.empty_like(gathered)
        start = 0
        for i in range(self.num_experts):
            n = counts[i]
            if n == 0:
                continue
            end = start + n
            outputs[start:end] = self.experts[i](gathered[start:end]) * flat_w[start:end]
            start = end

        final_output = torch.zeros_like(x_flat)
        final_output.index_add_(0, tok, outputs)

        return final_output.view(B, T, C), aux_loss


class VesperLLM(nn.Module):
    # FIX: defaults now match small_v2 (the config actually being trained).
    # Change these if you want a different default — just keep them in sync with MODEL_CONFIGS.
    def __init__(self, vocab_size=32000, dim=1024, n_layers=10, n_heads=8, n_kv_heads=2,
                 hidden_dim=1280, num_experts=8, top_k=2, max_seq_len=1024, pad_id=0,
                 dropout=0.0,  # FIX: dropout=0.0 — modern LLM pretraining skips dropout
                 grad_checkpoint=True):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.grad_checkpoint = grad_checkpoint
        self.tok_embeddings = nn.Embedding(vocab_size, dim)

        self.register_buffer("freqs_cis", precompute_freqs_cis(dim // n_heads, max_seq_len * 2))

        # NOTE: dropout removed from embeddings. At pretraining scale on a GPU cluster,
        # regularization comes from data volume and weight decay, not dropout.
        # Re-add if fine-tuning on a small dataset later.

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': GroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len),
                'ffn': MoEFeedForward(dim, hidden_dim, num_experts, top_k),
                'attn_norm': RMSNorm(dim),
                'ffn_norm': RMSNorm(dim)
            }) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('wo.weight') or pn.endswith('w2.weight'):
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

        for layer in self.layers:
            if self.training and self.grad_checkpoint:
                # Checkpoint the WHOLE layer (attention + MoE). non-reentrant
                # checkpointing handles the MoE's dynamic routing shapes fine —
                # routing is deterministic here (no dropout), so the backward
                # recompute reproduces the same expert assignment. freqs_cis is
                # a constant buffer, captured via closure rather than saved as
                # a checkpoint input. Measured ~2x lower activation memory than
                # the old attention-only checkpointing.
                def create_layer_forward(module_dict, freqs):
                    def custom_forward(x_in):
                        attn_out = module_dict['attn'](module_dict['attn_norm'](x_in), freqs)
                        h = x_in + attn_out
                        ffn_out, aux = module_dict['ffn'](module_dict['ffn_norm'](h))
                        return h + ffn_out, aux
                    return custom_forward

                x, aux_loss = cp.checkpoint(
                    create_layer_forward(layer, self.freqs_cis),
                    x,
                    use_reentrant=False
                )
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
