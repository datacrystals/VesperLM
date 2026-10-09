"""Plug-in expert during real pretraining on real domains (GPU / MI300X).

Real-model t1 follow-up of lab/plugin_expert_test_v2.py: same three-phase
protocol (pretrain on domain A -> hot-plug expert + contrastive passport
supervision -> zero-shot routing eval) but on the production VesperLinearLM
KDA+MLA stack with real uint16 token bins.

Domains:
    A = fineweb_edu web text   /root/lab_real_phase1.bin
    B = python code            /root/lab_domainB_code.bin

Phase A: 600 steps, all params, domain A only.
Phase B: 300 steps, add_expert() on every MoE layer (idx=4 everywhere),
         train only the new experts + new passport rows. Supervision is
         v2's contrastive scheme: forced-dispatch CE of the new experts on
         domain B, plus router CE (target=idx on B, uniform-over-originals
         on A = "reject on A") with reject weight 5.0.
Phase C: no training. Routing selectivity per MoE layer on fresh A/B
         batches, CE on code with vs without the plug-in (expert-4 logits
         forced to -inf), CE on web with plug-in (damage check).

PASS: mean util(code) > 0.5 AND mean util(web) < 0.3 AND CE(code, plugin) < CE(code, masked).

Run: /root/venvs/pod/bin/python lab/plugin_expert_test_t1.py
Optional env: PLUGIN_T1_PHASE_A_STEPS (600), PLUGIN_T1_PHASE_B_STEPS (300),
PLUGIN_T1_PHASE_B_LR (3e-3), PLUGIN_T1_REJECT_W (5.0), PLUGIN_T1_EVAL_BATCHES (8).
"""

import inspect
import json
import math
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Pretrain"))

from vesper_linear_model import VesperLinearLM  # noqa: E402
from vesper_model import MoEFeedForward  # noqa: E402
from configs.model_configs import get_model_config  # noqa: E402

# --- config ---
CONFIG_NAME = "lab_small"
VOCAB_SIZE = 65536
BATCH = 8
SEQ_LEN = 1024
PHASE_A_STEPS = int(os.environ.get("PLUGIN_T1_PHASE_A_STEPS", "600"))
PHASE_B_STEPS = int(os.environ.get("PLUGIN_T1_PHASE_B_STEPS", "300"))
PHASE_A_LR = 7e-4
PHASE_B_LR = float(os.environ.get("PLUGIN_T1_PHASE_B_LR", "3e-3"))
EVAL_BATCHES = int(os.environ.get("PLUGIN_T1_EVAL_BATCHES", "8"))
AUX_W = 0.1
ROUTER_CE_W = 0.1
REJECT_W = float(os.environ.get("PLUGIN_T1_REJECT_W", "5.0"))
SEED = 0

DOMAIN_A_BIN = "/root/lab_real_phase1.bin"
DOMAIN_B_BIN = "/root/lab_domainB_code.bin"
RESULTS_PATH = os.environ.get("PLUGIN_T1_OUT", "/root/plugin_t1_results.json")

assert torch.cuda.is_available(), "this test needs a GPU"
DEVICE = torch.device("cuda")


# ---------------------------- data ----------------------------
class TokenStream:
    """Uniform random (B, T+1) windows over a uint16 memmap."""

    def __init__(self, path, rng):
        self.mm = np.memmap(path, dtype=np.uint16, mode="r")
        self.rng = rng

    def batch(self, batch, seq_len):
        n = len(self.mm)
        starts = self.rng.integers(0, n - seq_len - 1, size=batch)
        seqs = np.stack([self.mm[s:s + seq_len + 1] for s in starts]).astype(np.int64)
        t = torch.from_numpy(seqs)
        return t[:, :-1].to(DEVICE), t[:, 1:].to(DEVICE)  # inputs, targets


def next_token_ce(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


# ---------------------------- model helpers ----------------------------
def build_model():
    cfg = get_model_config(CONFIG_NAME)
    sig = inspect.signature(VesperLinearLM.__init__)
    valid = {k for k in sig.parameters if k != "self"}
    kwargs = {k: v for k, v in cfg.items() if k in valid}
    kwargs["vocab_size"] = VOCAB_SIZE
    kwargs["router_type"] = "passport"
    kwargs["grad_checkpoint"] = False
    model = VesperLinearLM(**kwargs).to(DEVICE)
    return model, kwargs


@contextmanager
def forced_dispatch(idx):
    """Route every token through expert `idx` in every MoE layer."""
    orig = MoEFeedForward.forward

    def forced(self, x):
        B, T, C = x.shape
        out = self.experts[idx](x.reshape(-1, C)).view(B, T, C)
        return out, x.new_zeros(())

    MoEFeedForward.forward = forced
    try:
        yield
    finally:
        MoEFeedForward.forward = orig


@contextmanager
def capture_ffn_inputs(model):
    """Capture each MoE layer's router input (the ffn_norm output)."""
    captured = {}
    handles = []

    def mk(i):
        def hook(_mod, _inp, out):
            captured[i] = out
        return hook

    for i, layer in enumerate(model.layers):
        handles.append(layer["ffn_norm"].register_forward_hook(mk(i)))
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def router_logits(ffn, x_flat):
    """Same ranking as PassportRouter.forward: query(x) @ passports.T / sqrt(d)."""
    r = ffn.router
    return r.query(x_flat) @ r.passports.T / math.sqrt(r.passport_dim)


class _ShimRouter(nn.Module):
    """Passport-equivalent router returning MoEFeedForward's (w, i, aux)
    tuple, optionally masking one expert to -inf (plug-in-absent control)."""

    def __init__(self, query, passports, top_k, passport_dim, mask_expert=None):
        super().__init__()
        self.query = query
        self.passports = passports
        self.top_k = top_k
        self.passport_dim = passport_dim
        self.mask_expert = mask_expert

    def forward(self, x):
        logits = self.query(x) @ self.passports.T / math.sqrt(self.passport_dim)
        if self.mask_expert is not None:
            keep = torch.ones(logits.size(-1), dtype=torch.bool, device=logits.device)
            keep[self.mask_expert] = False
            logits = logits.masked_fill(~keep, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        tw, ti = torch.topk(probs, self.top_k, dim=-1)
        tw = tw / tw.sum(dim=-1, keepdim=True)
        return tw, ti, logits.new_zeros(())


@contextmanager
def mask_expert_everywhere(model, idx):
    """Swap every MoE router for a shim that drops expert `idx`."""
    saved = []
    for layer in model.layers:
        ffn = layer["ffn"]
        r = ffn.router
        saved.append((ffn, r))
        ffn.router = _ShimRouter(r.query, r.passports, ffn.top_k,
                                 r.passport_dim, mask_expert=idx)
    try:
        yield
    finally:
        for ffn, r in saved:
            ffn.router = r


# ---------------------------- eval ----------------------------
@torch.no_grad()
def ce_over(model, stream, n_batches, mask=None):
    model.eval()
    total, ntok = 0.0, 0
    ctx = mask_expert_everywhere(model, mask) if mask is not None else _nullctx()
    with ctx:
        for _ in range(n_batches):
            x, y = stream.batch(BATCH, SEQ_LEN)
            logits, _, _ = model(x)
            total += next_token_ce(logits, y).item() * y.numel()
            ntok += y.numel()
    return total / ntok


@contextmanager
def _nullctx():
    yield


@torch.no_grad()
def util_per_layer(model, stream, n_batches, expert_idx):
    """Fraction of tokens whose top-2 router picks include expert_idx, per layer."""
    model.eval()
    hits = [0] * len(model.layers)
    ntok = [0] * len(model.layers)
    for _ in range(n_batches):
        x, _ = stream.batch(BATCH, SEQ_LEN)
        with capture_ffn_inputs(model) as cap:
            model(x)
        for i, layer in enumerate(model.layers):
            flat = cap[i].reshape(-1, cap[i].size(-1))
            ti = torch.topk(router_logits(layer["ffn"], flat),
                            layer["ffn"].top_k, dim=-1).indices
            hits[i] += (ti == expert_idx).any(dim=-1).sum().item()
            ntok[i] += ti.size(0)
    return [h / n for h, n in zip(hits, ntok)]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    stream_a = TokenStream(DOMAIN_A_BIN, rng)
    stream_b = TokenStream(DOMAIN_B_BIN, rng)

    model, kwargs = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {CONFIG_NAME} on cuda  params={n_params/1e6:.2f}M  "
          f"layer_types={model.layer_types}")
    print(f"phase A: {PHASE_A_STEPS} steps lr={PHASE_A_LR} | "
          f"phase B: {PHASE_B_STEPS} steps lr={PHASE_B_LR} reject_w={REJECT_W} | "
          f"eval batches={EVAL_BATCHES}")

    # ---------------- Phase A: train everything on domain A ----------------
    opt_a = torch.optim.AdamW(model.parameters(), lr=PHASE_A_LR)
    model.train()
    phaseA_ce_final = None
    t0 = time.time()
    tokens_a = 0
    for step in range(1, PHASE_A_STEPS + 1):
        x, y = stream_a.batch(BATCH, SEQ_LEN)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, aux = model(x)
            ce = next_token_ce(logits, y)
            loss = ce + AUX_W * aux
        opt_a.zero_grad(set_to_none=True)
        loss.backward()
        opt_a.step()
        phaseA_ce_final = ce.item()
        tokens_a += y.numel()
        if step % 100 == 0 or step == 1:
            dt = time.time() - t0
            print(f"[phase A] step {step}/{PHASE_A_STEPS}  ce={ce.item():.4f}  "
                  f"aux={float(aux.detach()):.4f}  tok/s={tokens_a/dt:,.0f}")
    phaseA_tok_s = tokens_a / (time.time() - t0)
    phaseA_ce_series_note = "ce printed every 100 steps"

    # ---------------- Phase B: hot-plug + train only new experts ----------
    idxs = [layer["ffn"].add_expert() for layer in model.layers]
    assert len(set(idxs)) == 1, f"expert idx mismatch across layers: {idxs}"
    idx = idxs[0]
    print(f"[phase B] add_expert() -> idx={idx} on {len(idxs)} MoE layers, "
          f"passports now {tuple(model.layers[0]['ffn'].router.passports.shape)}")

    for layer in model.layers:
        ffn = layer["ffn"]
        # add_expert() builds the new module on CPU; move it before loading
        ffn.experts[idx].to(DEVICE)
        # start the new expert from expert 0's weights (v2 protocol)
        ffn.experts[idx].load_state_dict(ffn.experts[0].state_dict())

    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    frozen_rows = {}
    for li, layer in enumerate(model.layers):
        ffn = layer["ffn"]
        for p in ffn.experts[idx].parameters():
            p.requires_grad_(True)
            trainable.append(p)
        pp = ffn.router.passports  # fresh Parameter created by register_expert()
        pp.requires_grad_(True)
        trainable.append(pp)
        frozen_rows[li] = pp.data[:idx].clone()
        # zero any grad on the original passport rows (v2 hook technique)
        pp.register_hook(
            lambda g, _i=idx: torch.cat([torch.zeros_like(g[:_i]), g[_i:]], dim=0)
        )

    opt_b = torch.optim.AdamW(trainable, lr=PHASE_B_LR, weight_decay=0.0)
    model.train()
    t0 = time.time()
    tokens_b = 0
    for step in range(1, PHASE_B_STEPS + 1):
        x, y = stream_b.batch(BATCH, SEQ_LEN)
        # (1) forced-dispatch CE: domain B tokens through the new experts
        with capture_ffn_inputs(model) as cap_b:
            with forced_dispatch(idx):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _, _ = model(x)
                    ce = next_token_ce(logits, y)
        # (2) contrastive passport supervision, averaged over MoE layers.
        #     B tokens -> label idx; A tokens -> uniform over original experts
        #     (row idx explicitly zero) = "reject on A" pressure.
        r_ce_b_terms, r_ce_a_terms = [], []
        with torch.no_grad():
            xa, _ = stream_a.batch(BATCH, SEQ_LEN)
            with capture_ffn_inputs(model) as cap_a:
                model(xa)
        for i, layer in enumerate(model.layers):
            ffn = layer["ffn"]
            fb = cap_b[i].reshape(-1, cap_b[i].size(-1)).detach()
            r_logits_b = router_logits(ffn, fb)
            r_ce_b_terms.append(F.cross_entropy(
                r_logits_b, torch.full((fb.size(0),), idx, dtype=torch.long, device=DEVICE)))
            fa = cap_a[i].reshape(-1, cap_a[i].size(-1)).detach()
            r_logits_a = router_logits(ffn, fa)
            target_a = torch.zeros(fa.size(0), idx + 1, device=DEVICE)
            target_a[:, :idx] = 1.0 / idx
            r_ce_a_terms.append(
                -(target_a * F.log_softmax(r_logits_a, dim=-1)).sum(dim=-1).mean())
        r_ce_b = torch.stack(r_ce_b_terms).mean()
        r_ce_a = torch.stack(r_ce_a_terms).mean()
        loss = ce + ROUTER_CE_W * (r_ce_b + REJECT_W * r_ce_a)
        opt_b.zero_grad(set_to_none=True)
        loss.backward()
        opt_b.step()
        with torch.no_grad():
            for li, row in frozen_rows.items():
                model.layers[li]["ffn"].router.passports.data[:idx] = row
        tokens_b += y.numel()
        if step % 100 == 0 or step == 1:
            dt = time.time() - t0
            print(f"[phase B] step {step}/{PHASE_B_STEPS}  ce={ce.item():.4f}  "
                  f"router_ce={r_ce_b.item():.4f}  reject_ce={r_ce_a.item():.4f}  "
                  f"tok/s={tokens_b/dt:,.0f}")

    # ---------------- Phase C: zero-shot routing test (no training) -------
    model.eval()
    stream_a2 = TokenStream(DOMAIN_A_BIN, rng)  # fresh streams, same bins
    stream_b2 = TokenStream(DOMAIN_B_BIN, rng)

    util_a = util_per_layer(model, stream_a2, EVAL_BATCHES, idx)
    util_b = util_per_layer(model, stream_b2, EVAL_BATCHES, idx)
    ce_b_with = ce_over(model, stream_b2, EVAL_BATCHES, mask=None)
    ce_b_without = ce_over(model, stream_b2, EVAL_BATCHES, mask=idx)
    ce_a_with = ce_over(model, stream_a2, EVAL_BATCHES, mask=None)

    mean_util_a = sum(util_a) / len(util_a)
    mean_util_b = sum(util_b) / len(util_b)
    passed = (mean_util_b > 0.5) and (mean_util_a < 0.3) and (ce_b_with < ce_b_without)

    report = {
        "config": CONFIG_NAME,
        "params_m": round(n_params / 1e6, 3),
        "layer_types": model.layer_types,
        "new_expert_idx": idx,
        "phaseA_steps": PHASE_A_STEPS,
        "phaseA_lr": PHASE_A_LR,
        "phaseA_ce_final": phaseA_ce_final,
        "phaseA_tok_s": round(phaseA_tok_s, 1),
        "phaseB_steps": PHASE_B_STEPS,
        "phaseB_lr": PHASE_B_LR,
        "reject_w": REJECT_W,
        "router_ce_w": ROUTER_CE_W,
        "eval_batches": EVAL_BATCHES,
        "util_expert_idx_domainA_per_layer": [round(u, 4) for u in util_a],
        "util_expert_idx_domainB_per_layer": [round(u, 4) for u in util_b],
        "util_mean_domainA": round(mean_util_a, 4),
        "util_mean_domainB": round(mean_util_b, 4),
        "ce_domainB_with_plugin": round(ce_b_with, 4),
        "ce_domainB_without_plugin": round(ce_b_without, 4),
        "ce_domainA_with_plugin": round(ce_a_with, 4),
        "phaseA_ce_series_note": phaseA_ce_series_note,
        "pass": bool(passed),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== plugin_expert_test_t1 report ===")
    print(f"new expert idx: {idx}   layers: {model.layer_types}")
    print(f"phaseA_ce_final:              {phaseA_ce_final:.4f}   (tok/s {phaseA_tok_s:,.0f})")
    print(f"util domainA per-layer:       {['%.4f' % u for u in util_a]}  mean={mean_util_a:.4f} (want < 0.3)")
    print(f"util domainB per-layer:       {['%.4f' % u for u in util_b]}  mean={mean_util_b:.4f} (want > 0.5)")
    print(f"ce_domainB_with_plugin:       {ce_b_with:.4f}")
    print(f"ce_domainB_without_plugin:    {ce_b_without:.4f}")
    print(f"ce_domainA_with_plugin:       {ce_a_with:.4f}  (damage check vs phaseA {phaseA_ce_final:.4f})")
    print(f"PASS: {passed}")
    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
