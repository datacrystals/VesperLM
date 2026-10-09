"""Multi-plug-in test at t0 (lab_tiny, passport router).

Phase 1: train passport base 400 steps on fineweb (domain A), then freeze.
Phase 2: train THREE experts SEPARATELY against the frozen base, one per
         domain -- python code / finemath / wikipedia -- each with the
         v2 contrastive passport loss (forced-dispatch CE on its domain +
         router CE target=idx on domain, target=uniform-over-originals on
         fineweb = reject), REJECT_W=15 (t0 is 11M, below the t1 scale
         point where rw15 was validated).
Phase 3: plug all three into the frozen base (experts 4/5/6 + three
         duplicated-style passport rows) and measure
           * per-domain routing purity: own expert in top-2 >= 50% of
             tokens, each other plug-in expert <= 30%
           * per-domain CE delta vs a masked control (own expert -inf)

Run: /root/venvs/pod/bin/python lab/swarm_multiplug_t0.py
Env: SWARM_MP_OUT, SWARM_MP_BASE_STEPS (400), SWARM_MP_EXP_STEPS (300),
     SWARM_MP_REJECT_W (15).
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

CONFIG_NAME = "lab_tiny"
VOCAB_SIZE = 65536
BATCH = 8
SEQ_LEN = 512
BASE_STEPS = int(os.environ.get("SWARM_MP_BASE_STEPS", "400"))
EXP_STEPS = int(os.environ.get("SWARM_MP_EXP_STEPS", "300"))
BASE_LR = 1e-3
EXP_LR = 3e-3
AUX_W = 0.1
ROUTER_CE_W = 0.1
REJECT_W = float(os.environ.get("SWARM_MP_REJECT_W", "15"))
EVAL_BATCHES = 8
SEED = 0

DOMAIN_A = "/root/swarm_phase1.bin"
DOMAINS = {
    "code": "/root/swarm_code.bin",
    "finemath": "/root/swarm_finemath.bin",
    "wikipedia": "/root/swarm_wikipedia.bin",
}
OWN_GATE = 0.50
CROSS_GATE = 0.30
RESULTS_PATH = os.environ.get("SWARM_MP_OUT", "/root/swarm_multiplug_t0.json")

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


@contextmanager
def forced_dispatch(idx):
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
    captured = {}
    handles = []

    def mk(i):
        def hook(_m, _i, out):
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
    r = ffn.router
    return r.query(x_flat) @ r.passports.T / math.sqrt(r.passport_dim)


class _ShimRouter(nn.Module):
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
def util_matrix(model, stream, n_batches, expert_ids):
    """util[d][e]: fraction of tokens of `stream` whose top-2 includes e."""
    model.eval()
    hits = {e: 0 for e in expert_ids}
    ntok = 0
    for _ in range(n_batches):
        x, _ = stream.batch(BATCH, SEQ_LEN)
        with capture_ffn_inputs(model) as cap:
            model(x)
        # average over MoE layers
        for i, layer in enumerate(model.layers):
            flat = cap[i].reshape(-1, cap[i].size(-1))
            ti = torch.topk(router_logits(layer["ffn"], flat),
                            layer["ffn"].top_k, dim=-1).indices
            for e in expert_ids:
                hits[e] += (ti == e).any(dim=-1).sum().item()
            ntok += ti.size(0)
    return {e: hits[e] / ntok for e in expert_ids}, ntok


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def train_one_expert(base_model, exp_idx, stream_dom, stream_a, steps):
    """Clone the frozen base, plug in one expert, train it + its passport row.

    Inside the isolated copy the new expert is always index 4 (the base has 4);
    `exp_idx` is only its final position in the assembled 7-expert model.
    """
    import copy
    model = copy.deepcopy(base_model)
    local_idx = 4
    got = [l["ffn"].add_expert() for l in model.layers]
    assert set(got) == {local_idx}, got
    for layer in model.layers:
        ffn = layer["ffn"]
        ffn.experts[local_idx].to(DEVICE)
        ffn.experts[local_idx].load_state_dict(ffn.experts[0].state_dict())

    freeze_all(model)
    trainable, frozen_rows = [], {}
    for li, layer in enumerate(model.layers):
        ffn = layer["ffn"]
        for p in ffn.experts[local_idx].parameters():
            p.requires_grad_(True)
            trainable.append(p)
        pp = ffn.router.passports
        pp.requires_grad_(True)
        trainable.append(pp)
        frozen_rows[li] = pp.data[:local_idx].clone()
        pp.register_hook(
            lambda g, _i=local_idx: torch.cat([torch.zeros_like(g[:_i]), g[_i:]], dim=0))

    opt = torch.optim.AdamW(trainable, lr=EXP_LR, weight_decay=0.0)
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = stream_dom.batch(BATCH, SEQ_LEN)
        with capture_ffn_inputs(model) as cap_b:
            with forced_dispatch(local_idx):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _, _ = model(x)
                    ce = next_token_ce(logits, y)
        r_ce_b_terms, r_ce_a_terms = [], []
        with torch.no_grad():
            xa, _ = stream_a.batch(BATCH, SEQ_LEN)
            with capture_ffn_inputs(model) as cap_a:
                model(xa)
        for i, layer in enumerate(model.layers):
            ffn = layer["ffn"]
            fb = cap_b[i].reshape(-1, cap_b[i].size(-1)).detach()
            r_ce_b_terms.append(F.cross_entropy(
                router_logits(ffn, fb),
                torch.full((fb.size(0),), local_idx, dtype=torch.long, device=DEVICE)))
            fa = cap_a[i].reshape(-1, cap_a[i].size(-1)).detach()
            r_logits_a = router_logits(ffn, fa)
            target_a = torch.zeros(fa.size(0), local_idx + 1, device=DEVICE)
            target_a[:, :local_idx] = 1.0 / local_idx
            r_ce_a_terms.append(
                -(target_a * F.log_softmax(r_logits_a, dim=-1)).sum(dim=-1).mean())
        r_ce_b = torch.stack(r_ce_b_terms).mean()
        r_ce_a = torch.stack(r_ce_a_terms).mean()
        loss = ce + ROUTER_CE_W * (r_ce_b + REJECT_W * r_ce_a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for li, row in frozen_rows.items():
                model.layers[li]["ffn"].router.passports.data[:local_idx] = row
        if step % 100 == 0 or step == 1:
            print(f"    [exp {local_idx}] step {step}/{steps}  ce={ce.item():.4f}  "
                  f"router_ce={r_ce_b.item():.4f}  reject_ce={r_ce_a.item():.4f}  "
                  f"tok/s={(step*BATCH*SEQ_LEN)/(time.time()-t0):,.0f}", flush=True)

    # harvest the trained expert + passport row
    per_layer = []
    for layer in model.layers:
        ffn = layer["ffn"]
        per_layer.append({
            "expert_state": {k: v.detach().clone().cpu()
                             for k, v in ffn.experts[local_idx].state_dict().items()},
            "passport_row": ffn.router.passports.data[local_idx].detach().clone().cpu(),
        })
    return per_layer


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    stream_a = TokenStream(DOMAIN_A, rng)
    streams = {name: TokenStream(path, rng) for name, path in DOMAINS.items()}

    base = build_model()
    n_params = sum(p.numel() for p in base.parameters())
    print(f"base model: {CONFIG_NAME}  params={n_params/1e6:.2f}M  "
          f"layer_types={base.layer_types}", flush=True)

    print(f"[phase 1] base train {BASE_STEPS} steps on fineweb", flush=True)
    opt = torch.optim.AdamW(base.parameters(), lr=BASE_LR)
    base.train()
    t0 = time.time()
    tokens = 0
    for step in range(1, BASE_STEPS + 1):
        x, y = stream_a.batch(BATCH, SEQ_LEN)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, aux = base(x)
            ce = next_token_ce(logits, y)
            loss = ce + AUX_W * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tokens += y.numel()
        if step % 100 == 0 or step == 1:
            print(f"  step {step}/{BASE_STEPS}  ce={ce.item():.4f}  "
                  f"tok/s={tokens/(time.time()-t0):,.0f}", flush=True)
    base_ce = ce.item()
    base_tok_s = tokens / (time.time() - t0)
    print(f"  base ce={base_ce:.4f}  tok/s={base_tok_s:,.0f}", flush=True)

    print("[phase 2] train three plug-in experts separately "
          f"(rw={REJECT_W}, {EXP_STEPS} steps each)", flush=True)
    domain_order = list(DOMAINS.keys())
    harvested = {}
    for j, name in enumerate(domain_order):
        exp_idx = 4 + j
        print(f"  -- expert {exp_idx} <- {name} --", flush=True)
        harvested[name] = train_one_expert(
            base, exp_idx, streams[name], stream_a, EXP_STEPS)

    print("[phase 3] assemble plug-ins into frozen base", flush=True)
    freeze_all(base)
    for j, name in enumerate(domain_order):
        exp_idx = 4 + j
        for li, layer in enumerate(base.layers):
            ffn = layer["ffn"]
            blob = harvested[name][li]
            ffn.experts.append(
                type(ffn.experts[0])(ffn.dim, ffn.hidden_dim).to(DEVICE))
            ffn.experts[exp_idx].load_state_dict(blob["expert_state"])
            row = blob["passport_row"].to(DEVICE).unsqueeze(0)
            ffn.router.passports = nn.Parameter(
                torch.cat([ffn.router.passports.data, row], dim=0))
            ffn.router.num_experts += 1
            ffn.num_experts += 1
    print(f"  experts now: {base.layers[0]['ffn'].num_experts}  "
          f"passports: {tuple(base.layers[0]['ffn'].router.passports.shape)}", flush=True)

    # ---------------- eval ----------------
    plug_ids = [4, 5, 6]
    all_ids = list(range(7))
    streams_eval = {"fineweb": TokenStream(DOMAIN_A, rng)}
    streams_eval.update({n: TokenStream(p, rng) for n, p in DOMAINS.items()})

    purity, ces = {}, {}
    for name, st in streams_eval.items():
        util, ntok = util_matrix(base, st, EVAL_BATCHES, plug_ids)
        ce_with = ce_over(base, st, EVAL_BATCHES, mask=None)
        own = 4 + domain_order.index(name) if name in domain_order else None
        ce_wo_own = ce_over(base, st, EVAL_BATCHES, mask=own) if own else None
        purity[name] = {
            "util_per_plug": {str(k): round(v, 4) for k, v in util.items()},
            "own_expert": own,
            "own_util": round(util[own], 4) if own else None,
            "max_cross_util": round(max(v for k, v in util.items() if k != own), 4) if own else None,
            "ntok": ntok,
        }
        ces[name] = {
            "ce_with_all_plugins": round(ce_with, 4),
            "ce_own_masked": round(ce_wo_own, 4) if own else None,
            "ce_delta": round(ce_wo_own - ce_with, 4) if own else None,
        }

    gates = {}
    for name in domain_order:
        p = purity[name]
        gates[name] = {
            "own_ok": p["own_util"] is not None and p["own_util"] >= OWN_GATE,
            "cross_ok": p["max_cross_util"] is not None and p["max_cross_util"] <= CROSS_GATE,
            "ce_ok": ces[name]["ce_delta"] is not None and ces[name]["ce_delta"] > 0,
        }
        gates[name]["pass"] = all(gates[name].values())
    overall = all(gates[n]["pass"] for n in domain_order)

    report = {
        "config": CONFIG_NAME,
        "params_m": round(n_params / 1e6, 3),
        "layer_types": base.layer_types,
        "base_steps": BASE_STEPS,
        "base_ce": round(base_ce, 4),
        "base_tok_s": round(base_tok_s, 1),
        "exp_steps": EXP_STEPS,
        "reject_w": REJECT_W,
        "plug_ids": {"code": 4, "finemath": 5, "wikipedia": 6},
        "num_experts_final": base.layers[0]["ffn"].num_experts,
        "top_k_final": base.layers[0]["ffn"].top_k,
        "purity": purity,
        "ce": ces,
        "gates": gates,
        "own_gate": OWN_GATE,
        "cross_gate": CROSS_GATE,
        "pass": bool(overall),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== swarm_multiplug_t0 report ===")
    print(f"base ce={base_ce:.4f} tok/s={base_tok_s:,.0f} | experts={base.layers[0]['ffn'].num_experts}")
    for name in streams_eval:
        p, c = purity[name], ces[name]
        print(f"{name:9s} own={p['own_util']} cross_max={p['max_cross_util']} "
              f"util={p['util_per_plug']} ce_with={c['ce_with_all_plugins']} "
              f"ce_masked={c['ce_own_masked']} delta={c['ce_delta']}")
    for name in domain_order:
        g = gates[name]
        print(f"gate {name:9s} own_ok={g['own_ok']} cross_ok={g['cross_ok']} "
              f"ce_ok={g['ce_ok']} -> {g['pass']}")
    print(f"OVERALL PASS: {overall}")
    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
