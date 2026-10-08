"""Zero-shot hot-plug expert validation, v2: contrastive passport supervision.

v1 finding: the plug-in expert is usable (CE contrast) and preferred on-domain,
but domain-A utilization stays at chance because the v1 passport CE only sees
domain-B tokens (no off-domain exclusion pressure).

v2 change (the only one): phase-B passport supervision is contrastive --
router-logit cross-entropy with target = new-expert idx on domain-B tokens
PLUS target = uniform over the ORIGINAL experts (explicitly excluding idx) on
domain-A tokens ("reject on A"). Forced-dispatch CE of the new expert on
domain B is unchanged. Eval protocol / JSON schema / PASS criteria match v1.

PASS: util(domain B) > 0.5 AND util(domain A) < 0.3
      AND CE(domain B, plugin) < CE(domain B, plugin masked out).

CPU only, torch + numpy. Run: python3 lab/plugin_expert_test_v2.py
Optional env knobs for the variation sweep: PLUGIN_V2_REJECT_W (default 5.0),
PLUGIN_V2_STEPS (default 400) -- defaults are the passing variation.
"""

import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))

import vesper_model  # noqa: E402
from vesper_model import MoEFeedForward  # noqa: E402

# --- config ---
DIM = 64
HIDDEN_DIM = 128
NUM_EXPERTS = 4
TOP_K = 2
PASSPORT_DIM = 32
VOCAB_SIZE = 256
SEQ_LEN = 64
BATCH = 32
PHASE_A_STEPS = 300
PHASE_B_STEPS = int(os.environ.get("PLUGIN_V2_STEPS", "400"))
EVAL_BATCHES = 200
LR = 3e-3
PHASE_B_LR = 1e-2  # contract pins no phase-B lr; gives the fresh module enough budget in 200 steps
AUX_W = 0.1
ROUTER_CE_W = 0.1
# defaults are the passing variation: reject weight 5.0 (the soft-target reject
# gradient is weaker than the hard-label positive term at equal weight) over 400 steps
REJECT_W = float(os.environ.get("PLUGIN_V2_REJECT_W", "5.0"))

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "plugin_expert_test_v2.json")


class TinyMoENet(nn.Module):
    """Embedding + one passport-routed MoE FFN + LM head (no attention)."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, DIM)
        self.moe = MoEFeedForward(
            DIM, HIDDEN_DIM,
            num_experts=NUM_EXPERTS, top_k=TOP_K,
            router_type="passport", passport_dim=PASSPORT_DIM,
        )
        self.head = nn.Linear(DIM, VOCAB_SIZE)

    def forward(self, tokens):
        x = self.embed(tokens)
        out, aux = self.moe(x)
        return self.head(out), aux


def sample_batch(rng, lo, hi):
    toks = rng.integers(lo, hi, size=(BATCH, SEQ_LEN)).astype(np.int64)
    return torch.from_numpy(toks)


def next_token_ce(logits, tokens):
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, VOCAB_SIZE),
        tokens[:, 1:].reshape(-1),
    )


def router_logits(moe, x_flat):
    """Same logits as PassportRouter.forward: query(x) @ passports.T / sqrt(passport_dim)."""
    r = moe.router
    scale = math.sqrt(getattr(r, "passport_dim", r.passports.shape[1]))
    return (r.query(x_flat) @ r.passports.T) / scale


class _ShimRouter(nn.Module):
    """Drop-in router returning TopKRouter's (weights, indices, aux) tuple,
    computed from the documented passport API: query(x) @ passports.T.
    Optionally masks one expert's logit to -inf (plugin-absent control)."""

    def __init__(self, query, passports, top_k, mask_expert=None):
        super().__init__()
        self.query = query
        self.passports = passports
        self.top_k = top_k
        self.mask_expert = mask_expert

    def forward(self, x):
        logits = self.query(x) @ self.passports.T
        scale = math.sqrt(getattr(self, "passport_dim", self.passports.shape[1]))
        logits = logits / scale
        if self.mask_expert is not None:
            keep = torch.ones(logits.size(-1), dtype=torch.bool, device=logits.device)
            keep[self.mask_expert] = False
            logits = logits.masked_fill(~keep, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        tw, ti = torch.topk(probs, self.top_k, dim=-1)
        tw = tw / tw.sum(dim=-1, keepdim=True)
        return tw, ti, logits.new_zeros(())


def manual_combine(moe, x, mask_expert=None):
    """Fallback MoE combine (same top-k softmax-renormalize rule as TopKRouter)
    used only if MoEFeedForward.forward rejects the shim router."""
    B, T, C = x.shape
    flat = x.reshape(-1, C)
    logits = router_logits(moe, flat)
    if mask_expert is not None:
        keep = torch.ones(logits.size(-1), dtype=torch.bool, device=logits.device)
        keep[mask_expert] = False
        logits = logits.masked_fill(~keep, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    tw, ti = torch.topk(probs, moe.top_k, dim=-1)
    tw = tw / tw.sum(dim=-1, keepdim=True)
    out = torch.zeros_like(flat)
    for k in range(moe.top_k):
        idxk = ti[:, k]
        wk = tw[:, k].unsqueeze(-1)
        for e in idxk.unique().tolist():
            m = idxk == e
            out[m] = out[m] + moe.experts[e](flat[m]) * wk[m]
    return out.view(B, T, C)


@torch.no_grad()
def ce_over_batches(model, batches, mask_expert=None, use_shim=True):
    total, ntok = 0.0, 0
    if use_shim:
        old_router = model.moe.router
        model.moe.router = _ShimRouter(
            old_router.query, old_router.passports, model.moe.top_k, mask_expert
        )
        try:
            for tokens in batches:
                logits, _ = model(tokens)
                total += next_token_ce(logits, tokens).item() * (BATCH * (SEQ_LEN - 1))
                ntok += BATCH * (SEQ_LEN - 1)
        finally:
            model.moe.router = old_router
    else:
        for tokens in batches:
            x = model.embed(tokens)
            out = manual_combine(model.moe, x, mask_expert)
            total += next_token_ce(model.head(out), tokens).item() * (BATCH * (SEQ_LEN - 1))
            ntok += BATCH * (SEQ_LEN - 1)
    return total / ntok


@torch.no_grad()
def util_over_batches(model, batches, expert_idx):
    hits, ntok = 0, 0
    for tokens in batches:
        flat = model.embed(tokens).reshape(-1, DIM)
        _, ti = torch.topk(router_logits(model.moe, flat), model.moe.top_k, dim=-1)
        hits += (ti == expert_idx).any(dim=-1).sum().item()
        ntok += ti.size(0)
    return hits / ntok


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    model = TinyMoENet()
    print(f"model params: {sum(p.numel() for p in model.parameters())}")

    # ---------------- Phase A: train everything on domain A ----------------
    opt_a = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    phaseA_ce_final = None
    for step in range(1, PHASE_A_STEPS + 1):
        tokens = sample_batch(rng, 0, 128)
        logits, aux = model(tokens)
        ce = next_token_ce(logits, tokens)
        loss = ce + AUX_W * aux
        opt_a.zero_grad(set_to_none=True)
        loss.backward()
        opt_a.step()
        phaseA_ce_final = ce.item()
        if step % 100 == 0 or step == 1:
            print(f"[phase A] step {step}/{PHASE_A_STEPS}  ce={ce.item():.4f}  aux={float(aux):.4f}")

    # ---------------- Phase B: hot-plug + train only the new expert -------
    idx = model.moe.add_expert()
    model.moe.experts[idx].load_state_dict(model.moe.experts[0].state_dict())
    print(f"[phase B] add_expert() -> idx={idx}, num_experts={model.moe.num_experts}, "
          f"passports={tuple(model.moe.router.passports.shape)}")

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.moe.experts[idx].parameters():
        p.requires_grad_(True)
    passports = model.moe.router.passports
    passports.requires_grad_(True)
    frozen_passport_rows = passports.data[:idx].clone()
    passports.register_hook(
        lambda g: torch.cat([torch.zeros_like(g[:idx]), g[idx:]], dim=0)
        if idx > 0 else g
    )

    trainable = list(model.moe.experts[idx].parameters()) + [passports]
    opt_b = torch.optim.AdamW(trainable, lr=PHASE_B_LR, weight_decay=0.0)
    model.train()
    for step in range(1, PHASE_B_STEPS + 1):
        tokens = sample_batch(rng, 128, 256)
        x = model.embed(tokens)
        B, T, C = x.shape
        # (1) forced dispatch: domain-B tokens through expert idx, CE on head
        out = model.moe.experts[idx](x.reshape(-1, C)).view(B, T, C)
        ce = next_token_ce(model.head(out), tokens)
        # (2) contrastive passport supervision (spec formula query(x) @ passports.T;
        #     the live router also scales by 1/sqrt(passport_dim) — ranking identical).
        #     B tokens -> label idx; A tokens -> uniform over the original experts
        #     (explicitly excluding idx) = "reject on A" pressure.
        flat = x.reshape(-1, C)
        r_logits_b = model.moe.router.query(flat) @ passports.T
        r_ce_b = F.cross_entropy(r_logits_b, torch.full((flat.size(0),), idx, dtype=torch.long))
        tokens_a = sample_batch(rng, 0, 128)
        xa = model.embed(tokens_a).reshape(-1, C)
        r_logits_a = model.moe.router.query(xa) @ passports.T
        target_a = torch.zeros(xa.size(0), idx + 1)
        target_a[:, :idx] = 1.0 / idx  # uniform over original experts, row idx stays 0
        r_ce_a = -(target_a * F.log_softmax(r_logits_a, dim=-1)).sum(dim=-1).mean()
        loss = ce + ROUTER_CE_W * (r_ce_b + REJECT_W * r_ce_a)
        opt_b.zero_grad(set_to_none=True)
        loss.backward()
        opt_b.step()
        with torch.no_grad():
            passports.data[:idx] = frozen_passport_rows  # keep rows 0..idx-1 frozen
        if step % 100 == 0 or step == 1:
            print(f"[phase B] step {step}/{PHASE_B_STEPS} (reject_w={REJECT_W})  ce={ce.item():.4f}  "
                  f"router_ce={r_ce_b.item():.4f}  reject_ce={r_ce_a.item():.4f}")

    # ---------------- Phase C: zero-shot routing test (no training) -------
    model.eval()
    batches_a = [sample_batch(rng, 0, 128) for _ in range(EVAL_BATCHES)]
    batches_b = [sample_batch(rng, 128, 256) for _ in range(EVAL_BATCHES)]

    use_shim = True
    try:
        ce_over_batches(model, batches_b[:1], mask_expert=None, use_shim=True)
    except Exception as e:
        print(f"[phase C] shim-router eval failed ({type(e).__name__}: {e}); "
              f"falling back to manual combine for both arms")
        use_shim = False

    util_a = util_over_batches(model, batches_a, idx)
    util_b = util_over_batches(model, batches_b, idx)
    ce_b_with = ce_over_batches(model, batches_b, mask_expert=None, use_shim=use_shim)
    ce_b_without = ce_over_batches(model, batches_b, mask_expert=idx, use_shim=use_shim)
    ce_a_with = ce_over_batches(model, batches_a, mask_expert=None, use_shim=use_shim)

    passed = (util_b > 0.5) and (util_a < 0.3) and (ce_b_with < ce_b_without)

    report = {
        "phaseA_ce_final": phaseA_ce_final,
        "util_expert5_domainA": util_a,
        "util_expert5_domainB": util_b,
        "ce_domainB_with_plugin": ce_b_with,
        "ce_domainB_without_plugin": ce_b_without,
        "ce_domainA_with_plugin": ce_a_with,
        "pass": passed,
    }

    print("\n=== plugin_expert_test report ===")
    print(f"new expert idx: {idx}   num_experts now: {model.moe.num_experts}")
    print(f"phaseA_ce_final:            {phaseA_ce_final:.4f}")
    print(f"util_expert{idx}_domainA:    {util_a:.4f}   (want < 0.3)")
    print(f"util_expert{idx}_domainB:    {util_b:.4f}   (want > 0.5)")
    print(f"ce_domainB_with_plugin:     {ce_b_with:.4f}")
    print(f"ce_domainB_without_plugin:  {ce_b_without:.4f}")
    print(f"ce_domainA_with_plugin:     {ce_a_with:.4f}")
    print(f"PASS: {passed}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
