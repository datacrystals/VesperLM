"""
Minimal single-GPU repro of the NaN loss, with per-step diagnostics.

Mimics 02_pretrain_linear.py's loop (same config, same optimizer split)
but logs grad norms, per-group LR, and weight max-abs each step so we
can see exactly what blows up first.

Usage:
  python debug_nan.py            # Muon lr = 0.06 (current config)
  python debug_nan.py --muon-lr 0.02
  python debug_nan.py --fp32-ns  # force fp32 Newton-Schulz (Pascal-safe)
"""

import os
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("FLA_CACHE_DIR", "/tmp/fla_cache")

import sys
import math
import argparse
import importlib.util as _ilu

_here = os.path.dirname(os.path.abspath(__file__))
_common = os.path.join(os.path.dirname(_here), "Common")
sys.path.insert(0, _here)
sys.path.insert(0, _common)

_spec = _ilu.spec_from_file_location("p01", os.path.join(_here, "01_pretrain.py"))
p01 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(p01)

from vesper_linear_model import VesperLinearLM
from muon import Muon, zeropower_via_newtonschulz5
from configs.model_configs import get_model_config

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--muon-lr", type=float, default=None,
                    help="Override Muon LR (default: max_lr * 100)")
parser.add_argument("--fp32-ns", action="store_true",
                    help="Force fp32 Newton-Schulz iteration")
parser.add_argument("--steps", type=int, default=30)
args = parser.parse_args()

device = torch.device("cuda:0")
cfg = get_model_config("tiny_agent")

tokenizer = p01.AutoTokenizer.from_pretrained("custom_tokenizer")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if args.fp32_ns:
    _orig_ns = zeropower_via_newtonschulz5
    def fp32_ns(G, steps=5):
        assert G.ndim >= 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.float()
        if G.size(-2) > G.size(-1):
            X = X.mT
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(-2) > G.size(-1):
            X = X.mT
        return X.to(G.dtype)
    import muon
    muon.zeropower_via_newtonschulz5 = fp32_ns
    # patch the reference captured inside Muon.step
    import types
    def patched_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                if p.ndim >= 2:
                    g = fp32_ns(g, steps=group["ns_steps"])
                p.add_(g.reshape(p.shape), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5)
    Muon.step = patched_step

arch_keys = ["dim", "n_layers", "n_heads", "n_kv_heads", "hidden_dim",
             "num_experts", "top_k", "max_seq_len"]
model = VesperLinearLM(vocab_size=len(tokenizer), pad_id=tokenizer.pad_token_id,
                       **{k: v for k, v in cfg.items() if k in arch_keys}).to(device)

max_lr = cfg["max_lr"]
muon_lr = args.muon_lr if args.muon_lr is not None else max_lr * 100.0

muon_params, adamw_params = [], []
for name, p in model.named_parameters():
    if p.ndim == 2 and "tok_embeddings" not in name and "output" not in name:
        muon_params.append(p)
    else:
        adamw_params.append(p)

opt_muon = Muon(muon_params, lr=muon_lr)
opt_adamw = torch.optim.AdamW(adamw_params, lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
print(f"Muon LR: {muon_lr:.4f} | AdamW LR: {max_lr:.4f} | fp32 NS: {args.fp32_ns}")

datasets_dict, _ = p01.load_dataset_index("data/index.txt")
phase1_train = {n: d for n, d in datasets_dict['train'].items() if 'phase1' in n}
stream = p01.MixedDataStream(phase1_train, {k: 1.0 for k in phase1_train},
                             1, 0, 42, cfg["seq_len_warmup"], cfg["max_seq_len"],
                             cfg["seq_len_start"], False, resume_state=None)

aux_weight = cfg["aux_weight"]
accum = 42

model.train()
for step in range(args.steps):
    lr = p01.get_lr(step, cfg["total_steps"], max_lr, cfg["min_lr"], cfg["warmup_steps"])
    opt_muon.param_groups[0]["lr"] = lr * (muon_lr / max_lr)
    opt_adamw.param_groups[0]["lr"] = lr

    for opt in (opt_muon, opt_adamw):
        opt.zero_grad()

    ce_acc = aux_acc = 0.0
    nan_in_grad = False
    for micro in range(accum):
        x, y = next(stream)
        x, y = x.to(device), y.to(device)
        logits, ce, aux = model(x, y)
        loss = ce / accum + aux_weight * aux / accum
        loss.backward()
        ce_acc += ce.item() / accum
        aux_acc += aux.item() / accum

    total_norm = 0.0
    worst = None
    for p in model.parameters():
        if p.grad is not None:
            n = p.grad.norm().item()
            if math.isnan(n) or math.isinf(n):
                nan_in_grad = True
            total_norm += n * n
    total_norm = math.sqrt(total_norm)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt_muon.step()
    opt_adamw.step()

    w_max = max(p.abs().max().item() for p in model.parameters())
    w_nan = any(torch.isnan(p).any().item() for p in model.parameters())
    print(f"step {step:3d} | lr {lr:.2e} | CE {ce_acc:.4f} | aux {aux_acc:.4f} | "
          f"grad_norm {total_norm:.3f}{' (NaN!)' if nan_in_grad else ''} | "
          f"w_max {w_max:.3f}{' (W-NaN!)' if w_nan else ''}")
    if math.isnan(ce_acc) or w_nan:
        print(">>> Loss or weights went NaN — stopping.")
        break
