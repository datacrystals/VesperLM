import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

    def forward(self, x):
        B, T, C = x.size()
        
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Expand K and V to match Q's head count for GQA
        k = k[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)
        v = v[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)
        
        # PyTorch 2.0 native Flash Attention
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
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)
        routing_weights = F.softmax(logits, dim=-1)
        top_weights, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        return top_weights, top_indices

class MoEFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([FeedForward(dim, hidden_dim) for _ in range(num_experts)])
        self.router = TopKRouter(dim, num_experts, top_k)

    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(-1, C)
        
        routing_weights, selected_experts = self.router(x_flat)
        final_output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            expert_mask = (selected_experts == i)
            token_indices = expert_mask.any(dim=-1)
            
            if not token_indices.any():
                continue
                
            weight_indices = expert_mask[token_indices].nonzero(as_tuple=True)[1]
            expert_weights = routing_weights[token_indices, weight_indices].unsqueeze(-1)
            
            expert_out = expert(x_flat[token_indices])
            final_output[token_indices] += expert_out * expert_weights
            
        return final_output.view(B, T, C)

class JesperLLM(nn.Module):
    def __init__(self, vocab_size=32000, dim=1536, n_layers=16, n_heads=12, n_kv_heads=4,
                 hidden_dim=2048, num_experts=8, top_k=2, max_seq_len=2048, pad_id=0, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.pos_embeddings = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        
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
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))
        
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
        
        positions = torch.arange(0, T, device=tokens.device).unsqueeze(0)
        x = self.tok_embeddings(tokens) + self.pos_embeddings(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = x + layer['attn'](layer['attn_norm'](x))
            x = x + layer['ffn'](layer['ffn_norm'](x))
            
        x = self.norm(x)
        logits = self.output(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.pad_id)
        return logits, loss