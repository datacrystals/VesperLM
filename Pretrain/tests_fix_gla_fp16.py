"""Regression test: GLA layer must run fp32 under fp16 autocast on P40.

Reproduces the crash from pretrain_linear_run3.log:
autocast(fp16) -> fla GLA chunk_fwd_kernel_h -> LLVM ERROR
"Unsupported rounding mode for conversion" (hard process abort).
The model wrapper now disables autocast and upcasts to fp32 for GLA.
"""
import sys
import os

sys.path.insert(0, "/home/tliao/VesperLM/Common")

import torch

from vesper_linear_model import GatedLinearAttn

torch.manual_seed(0)
dev = "cuda:0"

attn = GatedLinearAttn(dim=384, head_dim=64, layer_idx=0).to(dev)
attn.eval()

x = torch.randn(2, 128, 384, device=dev, dtype=torch.float32)

# fp32 path (as training runs)
out_fp32 = attn(x)
assert out_fp32.dtype == torch.float32, out_fp32.dtype
assert torch.isfinite(out_fp32).all()

# fp16 autocast path (as generation/eval runs) -- used to abort the process
with torch.amp.autocast("cuda", dtype=torch.float16):
    out_fp16 = attn(x)
assert out_fp16.dtype == torch.float32, out_fp16.dtype
assert torch.isfinite(out_fp16).all()

# backward pass (training mode)
attn.train()
x = torch.randn(1, 64, 384, device=dev, requires_grad=True)
loss = attn(x).square().mean()
loss.backward()
assert x.grad is not None and torch.isfinite(x.grad).all()

print("OK: GLA ran fp32 under fp16 autocast; forward + backward finite.")
print("  fp32 out dtype:", out_fp32.dtype, "| autocast out dtype:", out_fp16.dtype)
