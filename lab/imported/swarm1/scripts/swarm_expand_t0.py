"""Expert-expansion live test at t0 (lab_tiny, passport router).

Growth-style exact upcycle applied MID-TRAINING:
  phase 1: train 300 steps on fineweb (domain A)
  surgery: 4 -> 6 experts -- clone experts, duplicate passport rows,
           double top_k (2 -> 4), per Growth/README.md exact-upcycle:
           duplicate router rows + doubled top_k makes the parent/clone
           pair split the parent's routing weight 50/50, so the MoE FFN
           function is preserved on tokens whose parents are cloned.
  phase 2: train 300 more steps

Control mode (SWARM_EXPAND_CONTROL=1): same 600 steps, no surgery.

Question: does exact upcycle preserve the passport advantage through
expansion? Compare final val against the control run.

Run: /root/venvs/pod/bin/python lab/swarm_expand_t0.py
Env: SWARM_EXPAND_CONTROL (0/1), SWARM_EXPAND_OUT, SWARM_EXPAND_STEPS1 (300),
     SWARM_EXPAND_STEPS2 (300).
"""

import inspect
import json
import math
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Pretrain"))

from vesper_linear_model import VesperLinearLM  # noqa: E402
from vesper_model import MoEFeedForward, FeedForward  # noqa: E402
from configs.model_configs import get_model_config  # noqa: E402

CONFIG_NAME = "lab_tiny"
VOCAB_SIZE = 65536
BATCH = 8
SEQ_LEN = 512
STEPS1 = int(os.environ.get("SWARM_EXPAND_STEPS1", "300"))
STEPS2 = int(os.environ.get("SWARM_EXPAND_STEPS2", "300"))
LR = 1e-3
AUX_W = 0.1
SEED = 0
CONTROL = os.environ.get("SWARM_EXPAND_CONTROL", "0") == "1"
DATA_A = "/root/swarm_phase1.bin"
RESULTS_PATH = os.environ.get("SWARM_EXPAND_OUT", "/root/swarm_expand_t0.json")

assert torch.cuda.is_available(), "needs a GPU"
DEVICE = torch.device("cuda")


class TokenStream:
    def __init__(self, path, rng):
        self.mm = np.memmap(path, dtype=np.uint16, mode="r")
        self.rng = rng

    def batch(self, batch, seq_len):
        n = len(self.mm)
        starts = self.rng.integers(0, n - seq_len - 1, size=batch)
        seqs = np.stack([self.mm[s:s + seq_len + 1] for s in starts]).astype(np.int64)
        t = torch.from_numpy(seqs)
        return t[:, :-1].to(DEVICE), t[:, 1:].to(DEVICE)


def next_token_ce(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def build_model():
    cfg = get_model_config(CONFIG_NAME)
    sig = inspect.signature(VesperLinearLM.__init__)
    valid = {k for k in sig.parameters if k != "self"}
    kwargs = {k: v for k, v in cfg.items() if k in valid}
    kwargs["vocab_size"] = VOCAB_SIZE
    kwargs["router_type"] = "passport"
    kwargs["grad_checkpoint"] = False
    return VesperLinearLM(**kwargs).to(DEVICE)


def expand_exact(model, num_new=2, noise_std=0.0):
    """Growth exact-upcycle for the passport router: clone experts, duplicate
    their passport rows, double top_k so parent+clone split weight 50/50.

    4 -> 6 experts here (num_new=2), cloning experts 0..1 in the Growth
    default order (no usage stats -> plain 0..old_e-1). Strict function
    preservation holds for tokens whose selected parents are among the
    cloned ones; see Growth/README.md.
    """
    detail = []
    for li, layer in enumerate(model.layers):
        ffn = layer["ffn"]
        assert isinstance(ffn, MoEFeedForward) and ffn.router_type == "passport"
        old_e = ffn.num_experts
        order = list(range(old_e))
        for k in range(num_new):
            parent = order[k % old_e]
            clone = FeedForward(ffn.dim, ffn.hidden_dim).to(DEVICE)
            clone.load_state_dict(ffn.experts[parent].state_dict())
            ffn.experts.append(clone)
            row = ffn.router.passports.data[parent:parent + 1].clone()
            if noise_std > 0:
                row = row + noise_std * torch.randn_like(row)
            ffn.router.passports = nn.Parameter(
                torch.cat([ffn.router.passports.data, row], dim=0))
            ffn.router.num_experts += 1
        ffn.num_experts += num_new
        ffn.top_k *= 2
        ffn.router.top_k *= 2
        detail.append({
            "layer": li, "old_experts": old_e, "new_experts": ffn.num_experts,
            "old_top_k": ffn.top_k // 2, "new_top_k": ffn.top_k,
            "cloned_from": [order[k % old_e] for k in range(num_new)],
        })
    return detail


@torch.no_grad()
def evaluate(model, stream, n_batches=8):
    model.eval()
    total, ntok = 0.0, 0
    for _ in range(n_batches):
        x, y = stream.batch(BATCH, SEQ_LEN)
        logits, _, _ = model(x)
        total += next_token_ce(logits, y).item() * y.numel()
        ntok += y.numel()
    return total / ntok


def train_n(model, stream, n_steps, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()
    tokens = 0
    ce_final = None
    for step in range(1, n_steps + 1):
        x, y = stream.batch(BATCH, SEQ_LEN)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, aux = model(x)
            ce = next_token_ce(logits, y)
            loss = ce + AUX_W * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ce_final = ce.item()
        tokens += y.numel()
        if step % 100 == 0 or step == 1:
            print(f"  step {step}/{n_steps}  ce={ce.item():.4f}  "
                  f"tok/s={tokens/(time.time()-t0):,.0f}", flush=True)
    return ce_final, tokens / (time.time() - t0)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    stream = TokenStream(DATA_A, rng)

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {CONFIG_NAME}  params={n_params/1e6:.2f}M  "
          f"layer_types={model.layer_types}  control={CONTROL}", flush=True)

    print(f"[phase 1] {STEPS1} steps", flush=True)
    ce1, tok1 = train_n(model, stream, STEPS1, LR)

    surgery = None
    if not CONTROL:
        print("[surgery] Growth exact-upcycle 4 -> 6 experts, top_k 2 -> 4", flush=True)
        surgery = expand_exact(model, num_new=2, noise_std=0.0)
        print(f"  layers updated: {len(surgery)}  "
              f"experts now: {model.layers[0]['ffn'].num_experts}  "
              f"top_k now: {model.layers[0]['ffn'].top_k}", flush=True)
        n_params2 = sum(p.numel() for p in model.parameters())
        print(f"  params after expand: {n_params2/1e6:.2f}M", flush=True)

    print(f"[phase 2] {STEPS2} steps", flush=True)
    ce2, tok2 = train_n(model, stream, STEPS2, LR)

    val = evaluate(model, stream, n_batches=8)
    print(f"[eval] final val_loss={val:.4f}", flush=True)

    report = {
        "config": CONFIG_NAME,
        "mode": "control" if CONTROL else "expand_4to6_exact",
        "steps1": STEPS1,
        "steps2": STEPS2,
        "ce_after_phase1": round(ce1, 4),
        "ce_after_phase2": round(ce2, 4),
        "val_loss_final": round(val, 4),
        "phase1_tok_s": round(tok1, 1),
        "phase2_tok_s": round(tok2, 1),
        "layer_types": model.layer_types,
        "num_experts_final": model.layers[0]["ffn"].num_experts,
        "top_k_final": model.layers[0]["ffn"].top_k,
        "surgery": surgery,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n=== swarm_expand_t0 ({report['mode']}) ===")
    print(f"ce after phase1:  {ce1:.4f}")
    print(f"ce after phase2:  {ce2:.4f}")
    print(f"val_loss_final:   {val:.4f}")
    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
