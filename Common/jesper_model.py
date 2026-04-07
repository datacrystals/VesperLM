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
    def __init__(self, dim, n_heads, n_kv_heads, max_seq_len=2048):
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

    def forward(self, x, freqs_cis):
        B, T, C = x.size()

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis[:T])

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand K and V to match Q's head count for GQA
        k = k[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)
        v = v[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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


class MoEFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.experts = nn.ModuleList([FeedForward(dim, hidden_dim) for _ in range(num_experts)])
        self.router = TopKRouter(dim, num_experts, top_k)

    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(-1, C)  # (N, C) where N = B*T

        routing_weights, selected_experts, aux_loss = self.router(x_flat)
        # routing_weights: (N, top_k)
        # selected_experts: (N, top_k)

        final_output = torch.zeros_like(x_flat)

        # Iterate over experts. torch.topk guarantees distinct indices so a token
        # can't be assigned to the same expert twice — original logic was correct.
        for i, expert in enumerate(self.experts):
            expert_mask = (selected_experts == i)
            token_indices = expert_mask.any(dim=-1)

            if not token_indices.any():
                continue

            weight_indices = expert_mask[token_indices].nonzero(as_tuple=True)[1]
            expert_weights = routing_weights[token_indices, weight_indices].unsqueeze(-1)

            expert_out = expert(x_flat[token_indices])
            final_output[token_indices] += expert_out * expert_weights

        return final_output.view(B, T, C), aux_loss


class JesperLLM(nn.Module):
    # FIX: defaults now match small_v2 (the config actually being trained).
    # Change these if you want a different default — just keep them in sync with MODEL_CONFIGS.
    def __init__(self, vocab_size=32000, dim=1024, n_layers=10, n_heads=8, n_kv_heads=2,
                 hidden_dim=1280, num_experts=8, top_k=2, max_seq_len=1024, pad_id=0,
                 dropout=0.0):  # FIX: dropout=0.0 — modern LLM pretraining skips dropout
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
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
            tokens = tokens[:, :self.max_seq_len]
            if targets is not None:
                targets = targets[:, :self.max_seq_len]

        x = self.tok_embeddings(tokens)
        total_aux_loss = 0.0

        for layer in self.layers:
            # Wrap the layer execution in the checkpoint function
            # use_reentrant=False is the modern standard for PyTorch checkpointing
            def create_custom_forward(module_dict):
                def custom_forward(x_in, freqs):
                    attn_out = module_dict['attn'](module_dict['attn_norm'](x_in), freqs)
                    x_out = x_in + attn_out
                    ffn_out, aux = module_dict['ffn'](module_dict['ffn_norm'](x_out))
                    return x_out + ffn_out, aux
                return custom_forward

            if self.training:
                # Checkpointing is only for training
                x, aux_loss = cp.checkpoint(
                    create_custom_forward(layer), 
                    x, 
                    self.freqs_cis,
                    use_reentrant=False
                )
            else:
                # Standard forward pass for inference/eval
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
