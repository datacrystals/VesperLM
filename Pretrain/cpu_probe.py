#!/usr/bin/env python3
"""CPU inference probe for the 429M (470m config) pretrain checkpoint.

GLA's Triton kernels don't run on CPU, so we shim fla's chunk/recurrent
entry points with the naive pure-torch recurrence (state transposed to
match state_v_first=True). Uses forward_incremental + KV cache so each
token costs one tiny forward instead of a full recompute.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "16")

import torch

_REPO = "/home/tliao/VesperLM"
sys.path.insert(0, os.path.join(_REPO, "Common"))
sys.path.insert(0, os.path.join(_REPO, "Agent"))

CKPT_DIR = os.path.join(_REPO, "Pretrain", "vesper_linear_checkpoints_470m", "step_best")
TOK_DIR = os.path.join(_REPO, "Pretrain", "custom_tokenizer")

# ---- CPU shim for GLA (must happen before model forward, import is fine) ----
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

# ---- CPU shim for fla's fused gated RMSNorm (Triton-only; GLA fuse_norm=True) ----
# Kernel semantics (fla/modules/fused_norm_gate.py): fp32 internally,
# y = rmsnorm(x) * weight * (g * sigmoid(g))  [norm first, then swish gate]
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

from vesper_linear_model import VesperLinearLM
from transformers import PreTrainedTokenizerFast

torch.set_num_threads(16)

print(f"[probe] loading {CKPT_DIR}")
ckpt = torch.load(os.path.join(CKPT_DIR, "checkpoint.pt"),
                  map_location="cpu", weights_only=False)
mc = ckpt.get("model_config") or {}
print(f"[probe] checkpoint step {ckpt.get('step')}, "
      f"tokens_trained {ckpt.get('tokens_trained', 0):,}, "
      f"best_val {ckpt.get('best_val_loss')}")

model = VesperLinearLM(
    vocab_size=mc.get("vocab_size", 65523),
    dim=mc["dim"], n_layers=mc["n_layers"], n_heads=mc["n_heads"],
    n_kv_heads=mc["n_kv_heads"], hidden_dim=mc["hidden_dim"],
    num_experts=mc["num_experts"], top_k=mc["top_k"],
    max_seq_len=mc["max_seq_len"], pad_id=mc.get("pad_id", 0),
)
state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
model.load_state_dict(state, strict=True)
model.eval()

tok = PreTrainedTokenizerFast.from_pretrained(TOK_DIR)

PROMPTS = [
    ("knowledge", "The capital of France is"),
    ("science", "The water cycle begins when"),
    ("code", "def quicksort(arr):"),
    ("pattern", "One plus one equals two. Two plus two equals four. Three plus three equals"),
    ("chat-format", "<|im_start|>user\nWhat is the capital of Japan?<|im_end|>\n<|im_start|>assistant\n"),
]

MAX_NEW = 70

@torch.no_grad()
def generate(prompt, max_new=MAX_NEW):
    ids = tok(prompt, return_tensors="pt").input_ids
    caches = model.new_cache(1, torch.device("cpu"))
    logits = model.forward_incremental(ids, caches, 0)[0][0, -1]
    pos = ids.shape[1]
    out = []
    for _ in range(max_new):
        nxt = int(logits.argmax())
        out.append(nxt)
        if nxt == tok.convert_tokens_to_ids("<|im_end|>"):
            break
        logits = model.forward_incremental(
            torch.tensor([[nxt]]), caches, pos)[0][0, -1]
        pos += 1
    return tok.decode(out)

for name, prompt in PROMPTS:
    t0 = time.time()
    text = generate(prompt)
    dt = time.time() - t0
    print("=" * 60)
    print(f"[{name}] PROMPT: {prompt!r}")
    print(f"[{name}] OUTPUT ({dt:.0f}s):")
    print(text)
print("[probe] done")
