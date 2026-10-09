"""Multi-plug-in purity fix at t0 (lab_tiny, passport router).

Session-1 failure: 3 independently reject-trained experts plugged into one
frozen base get real CE wins but routing purity FAILS (code<->math cross-talk
~0.5, wiki own-util 0.17). The plug-ins compete instead of partition.

This script tests three fix protocols (SWARM_P2_PROTOCOL), all sharing the
same base and the same purity measurement as session 1:

  a  JOINT REJECT, isolated training
     Each expert is trained alone (as session 1) but its contrastive passport
     loss rejects the OTHER plug-ins' domains as well as the base domain.
     Hypothesis: passports partition when trained to be mutually exclusive.

  b  SEQUENTIAL INSERT + RECALIBRATION
     Insert expert 4 (session-1 protocol), then unfreeze ROUTER-ONLY
     (passports + query) for ~100 calibration steps over a 4-domain mix,
     freeze, insert expert 5, recalibrate, insert 6, recalibrate.
     Hypothesis: the router needs to see all plug-ins together.

  c  JOINT MODEL TRAINING with mutual reject (A+B hybrid)
     All 3 plug-ins live in one model at once. Experts train by
     forced-dispatch CE on their own domain; the router gets a global
     (domain -> owner expert) objective across all four domains, so it
     learns the partition directly. Strongest signal.

Base: 400 steps fineweb passport then frozen (identical to session 1).

Purity gates per domain: own-expert util >= 0.5, max cross-plug-in util <= 0.3,
CE with plug-in < CE with own expert masked.

Run: /root/venvs/pod/bin/python lab/swarm2_purity.py
Env: SWARM_P2_PROTOCOL (a|b|c), SWARM_P2_OUT, SWARM_P2_BASE_STEPS (400),
     SWARM_P2_EXP_STEPS (300), SWARM_P2_REJECT_W (15), SWARM_P2_CAL_STEPS (100).
"""

import copy
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
BASE_STEPS = int(os.environ.get("SWARM_P2_BASE_STEPS", "400"))
EXP_STEPS = int(os.environ.get("SWARM_P2_EXP_STEPS", "300"))
CAL_STEPS = int(os.environ.get("SWARM_P2_CAL_STEPS", "100"))
BASE_LR = 1e-3
EXP_LR = 3e-3
CAL_LR = 1e-3
AUX_W = 0.1
ROUTER_CE_W = 0.1
REJECT_W = float(os.environ.get("SWARM_P2_REJECT_W", "15"))
OWNER_MASS = float(os.environ.get("SWARM_P2_OWNER_MASS", "0.5"))
EVAL_BATCHES = 8
SEED = 0
PROTOCOL = os.environ.get("SWARM_P2_PROTOCOL", "c").lower()

DOMAIN_A = "/root/swarm_phase1.bin"
DOMAINS = {
    "code": "/root/swarm_code.bin",
    "finemath": "/root/swarm_finemath.bin",
    "wikipedia": "/root/swarm_wikipedia.bin",
}
OWN_GATE = 0.50
CROSS_GATE = 0.30
RESULTS_PATH = os.environ.get("SWARM_P2_OUT", "/root/swarm2_purity.json")

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
    model.eval()
    hits = {e: 0 for e in expert_ids}
    ntok = 0
    for _ in range(n_batches):
        x, _ = stream.batch(BATCH, SEQ_LEN)
        with capture_ffn_inputs(model) as cap:
            model(x)
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


def freeze_experts_only(model):
    """Router-only training: unfreeze passports + query, freeze everything else."""
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for layer in model.layers:
        r = layer["ffn"].router
        for p in r.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


def router_ce_terms(model, cap, targets_by_idx, n_orig=4):
    """Contrastive router CE averaged over MoE layers.

    targets_by_idx: {batch_tag: int | 'reject' | (owner, owner_mass)} where
      'reject'           -> uniform over original experts 0..n_orig-1
      int                -> one-hot on that expert
      (owner, mass)      -> owner gets `mass`, the originals share (1-mass),
                            every OTHER plug-in row gets exactly 0. This is
                            the mutual-exclusion target: on a domain, route to
                            its owner or to the base experts, never to another
                            plug-in.
    `cap` maps batch tag -> {layer idx: ffn input}.
    """
    terms = []
    for i, layer in enumerate(model.layers):
        ffn = layer["ffn"]
        for tag, target in targets_by_idx.items():
            x = cap[tag][i].reshape(-1, cap[tag][i].size(-1)).detach()
            logits = router_logits(ffn, x)
            n_e = logits.size(-1)
            if isinstance(target, tuple):
                owner, mass = target
                t = torch.zeros(x.size(0), n_e, device=DEVICE)
                t[:, owner] = mass
                t[:, :n_orig] = t[:, :n_orig] + (1.0 - mass) / n_orig
                terms.append(-(t * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean())
            elif target == "reject":
                t = torch.zeros(x.size(0), n_e, device=DEVICE)
                t[:, :n_orig] = 1.0 / n_orig
                terms.append(-(t * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean())
            else:
                terms.append(F.cross_entropy(
                    logits,
                    torch.full((x.size(0),), target, dtype=torch.long, device=DEVICE)))
    return torch.stack(terms).mean()


def train_base(model, stream, steps):
    opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
    model.train()
    t0 = time.time()
    tokens = 0
    ce_final = None
    for step in range(1, steps + 1):
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
            print(f"  [base] step {step}/{steps}  ce={ce.item():.4f}  "
                  f"tok/s={tokens/(time.time()-t0):,.0f}", flush=True)
    return ce_final, tokens / (time.time() - t0)


def harvest_expert(model, local_idx):
    out = []
    for layer in model.layers:
        ffn = layer["ffn"]
        out.append({
            "expert_state": {k: v.detach().clone().cpu()
                             for k, v in ffn.experts[local_idx].state_dict().items()},
            "passport_row": ffn.router.passports.data[local_idx].detach().clone().cpu(),
        })
    return out


def plug_expert(base_model, exp_idx, blob):
    for li, layer in enumerate(base_model.layers):
        ffn = layer["ffn"]
        ffn.experts.append(type(ffn.experts[0])(ffn.dim, ffn.hidden_dim).to(DEVICE))
        ffn.experts[exp_idx].load_state_dict(blob[li]["expert_state"])
        row = blob[li]["passport_row"].to(DEVICE).unsqueeze(0)
        ffn.router.passports = nn.Parameter(
            torch.cat([ffn.router.passports.data, row], dim=0))
        ffn.router.num_experts += 1
        ffn.num_experts += 1


# ---------------------------------------------------------------- protocols

def train_isolated_expert(base_model, stream_own, reject_streams, steps, tag):
    """Session-1 style: clone base, add one expert at local idx 4, train it +
    its passport row. reject_streams: list of TokenStream to reject on."""
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
        x, y = stream_own.batch(BATCH, SEQ_LEN)
        caps = {}
        with capture_ffn_inputs(model) as cap_b:
            with forced_dispatch(local_idx):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _, _ = model(x)
                    ce = next_token_ce(logits, y)
            caps["own"] = cap_b

        with torch.no_grad():
            for j, st in enumerate(reject_streams):
                xr, _ = st.batch(BATCH, SEQ_LEN)
                with capture_ffn_inputs(model) as cap_r:
                    model(xr)
                caps[f"r{j}"] = cap_r

        targets = {"own": local_idx}
        for j in range(len(reject_streams)):
            targets[f"r{j}"] = "reject"
        r_ce = router_ce_terms(model, caps, targets)

        # split for logging: positive vs reject
        with torch.no_grad():
            fb = caps["own"][0].reshape(-1, caps["own"][0].size(-1)).detach()
            pos = F.cross_entropy(router_logits(model.layers[0]["ffn"], fb),
                                  torch.full((fb.size(0),), local_idx,
                                             dtype=torch.long, device=DEVICE)).item()
        loss = ce + ROUTER_CE_W * r_ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for li, row in frozen_rows.items():
                model.layers[li]["ffn"].router.passports.data[:local_idx] = row
        if step % 100 == 0 or step == 1:
            print(f"    [{tag}] step {step}/{steps}  ce={ce.item():.4f}  "
                  f"router_ce={r_ce.item():.4f}  pos={pos:.4f}  "
                  f"tok/s={(step*BATCH*SEQ_LEN)/(time.time()-t0):,.0f}", flush=True)
    return harvest_expert(model, local_idx)


def recalibrate_router(model, domain_owner, stream_a, stream_map, steps, tag):
    """Router-only training: target = owner expert on each plug-in domain,
    reject (uniform over originals) on the base domain."""
    trainable = freeze_experts_only(model)
    if not trainable:
        return
    opt = torch.optim.AdamW(trainable, lr=CAL_LR, weight_decay=0.0)
    model.train()
    owners = list(domain_owner.items())
    for step in range(1, steps + 1):
        caps = {}
        with torch.no_grad():
            xa, _ = stream_a.batch(BATCH, SEQ_LEN)
            with capture_ffn_inputs(model) as cap_a:
                model(xa)
            caps["web"] = cap_a
            for dname, owner in owners:
                xr, _ = stream_map[dname].batch(BATCH, SEQ_LEN)
                with capture_ffn_inputs(model) as cap_r:
                    model(xr)
                caps[dname] = cap_r
        targets = {"web": "reject"}
        for dname, owner in owners:
            # mutual exclusion: owner + originals only, other plug-ins get 0
            targets[dname] = (owner, OWNER_MASS)
        r_ce = router_ce_terms(model, caps, targets)
        opt.zero_grad(set_to_none=True)
        r_ce.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"    [{tag}] cal step {step}/{steps}  router_ce={r_ce.item():.4f}",
                  flush=True)
    freeze_all(model)


def protocol_a(base, stream_a, streams, stream_map):
    """Joint reject, isolated training: reject web + other plug-in domains."""
    harvested = {}
    order = list(DOMAINS.keys())
    for j, name in enumerate(order):
        rejects = [stream_a] + [streams[other] for other in order if other != name]
        print(f"  -- A: expert {4+j} <- {name} (rejects web+{[o for o in order if o != name]}) --",
              flush=True)
        harvested[name] = train_isolated_expert(
            base, streams[name], rejects, EXP_STEPS, f"A/{name}")
    return harvested


def protocol_b(base, stream_a, streams, stream_map):
    """Sequential insert + router-only recalibration between insertions.

    Experts are always trained against a PRISTINE 4-expert copy of the base
    (so `add_expert()` always lands at local idx 4); only the assembled model
    grows.
    """
    harvested = {}
    order = list(DOMAINS.keys())
    pristine = copy.deepcopy(base)
    freeze_all(pristine)
    domain_owner = {}
    for j, name in enumerate(order):
        exp_idx = 4 + j
        print(f"  -- B: expert {exp_idx} <- {name} (isolated, session-1 protocol) --",
              flush=True)
        harvested[name] = train_isolated_expert(
            pristine, streams[name], [stream_a], EXP_STEPS, f"B/{name}")
        plug_expert(base, exp_idx, harvested[name])
        domain_owner[name] = exp_idx
        print(f"    [{name}] inserted at {exp_idx}; recalibrating router "
              f"({CAL_STEPS} steps, domains={list(domain_owner)})", flush=True)
        recalibrate_router(base, domain_owner, stream_a, stream_map,
                           CAL_STEPS, f"B/cal-{name}")
    return harvested


def protocol_c(base, stream_a, streams, stream_map):
    """Joint model training: all 3 plug-ins live at once. Experts train by
    forced-dispatch CE on their own domain; the router gets a global
    (domain -> owner) objective across all four domains."""
    order = list(DOMAINS.keys())
    blobs = {}
    for j, name in enumerate(order):
        exp_idx = 4 + j
        got = [l["ffn"].add_expert() for l in base.layers]
        assert set(got) == {exp_idx}, got
        for layer in base.layers:
            ffn = layer["ffn"]
            ffn.experts[exp_idx].to(DEVICE)
            ffn.experts[exp_idx].load_state_dict(ffn.experts[0].state_dict())

    domain_owner = {name: 4 + j for j, name in enumerate(order)}
    freeze_all(base)
    trainable = []
    frozen_rows = {}
    for li, layer in enumerate(base.layers):
        ffn = layer["ffn"]
        for e in range(4, 7):
            for p in ffn.experts[e].parameters():
                p.requires_grad_(True)
                trainable.append(p)
        pp = ffn.router.passports
        pp.requires_grad_(True)
        trainable.append(pp)
        frozen_rows[li] = pp.data[:4].clone()
        pp.register_hook(
            lambda g: torch.cat([torch.zeros_like(g[:4]), g[4:]], dim=0))

    opt = torch.optim.AdamW(trainable, lr=EXP_LR, weight_decay=0.0)
    base.train()
    t0 = time.time()
    owners = list(domain_owner.items())
    for step in range(1, EXP_STEPS + 1):
        # rotate domains so every owner gets forced CE this round
        dname, owner = owners[(step - 1) % len(owners)]
        x, y = streams[dname].batch(BATCH, SEQ_LEN)
        caps = {}
        with capture_ffn_inputs(base) as cap_o:
            with forced_dispatch(owner):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _, _ = base(x)
                    ce = next_token_ce(logits, y)
            caps[dname] = cap_o

        with torch.no_grad():
            xa, _ = stream_a.batch(BATCH, SEQ_LEN)
            with capture_ffn_inputs(base) as cap_a:
                base(xa)
            caps["web"] = cap_a
            for oname, oidx in owners:
                if oname == dname:
                    continue
                xr, _ = streams[oname].batch(BATCH, SEQ_LEN)
                with capture_ffn_inputs(base) as cap_r:
                    base(xr)
                caps[oname] = cap_r

        targets = {"web": "reject"}
        for oname, oidx in owners:
            # mutual exclusion: owner + originals only, other plug-ins get 0
            targets[oname] = (oidx, OWNER_MASS)
        r_ce = router_ce_terms(base, caps, targets)

        loss = ce + ROUTER_CE_W * r_ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for li, row in frozen_rows.items():
                base.layers[li]["ffn"].router.passports.data[:4] = row
        if step % 100 == 0 or step == 1:
            print(f"    [C] step {step}/{EXP_STEPS} ({dname})  ce={ce.item():.4f}  "
                  f"router_ce={r_ce.item():.4f}  "
                  f"tok/s={(step*BATCH*SEQ_LEN)/(time.time()-t0):,.0f}", flush=True)

    for j, name in enumerate(order):
        blobs[name] = harvest_expert(base, 4 + j)
    return blobs


def protocol_d(base, stream_a, streams, stream_map):
    """B + extended final joint calibration: sequential insert with
    recalibration, then a long router-only pass over the 4-domain mix with
    the mutual-exclusion target. The final pass is where rows 4/5/6 actually
    learn to stay out of each other's territory."""
    harvested = protocol_b(base, stream_a, streams, stream_map)
    order = list(DOMAINS.keys())
    domain_owner = {name: 4 + j for j, name in enumerate(order)}
    long_cal = max(CAL_STEPS, 600)
    print(f"    [D] final joint recalibration ({long_cal} steps)", flush=True)
    recalibrate_router(base, domain_owner, stream_a, stream_map,
                       long_cal, "D/final-cal")
    return harvested


# ---------------------------------------------------------------- eval

def main():
    if PROTOCOL not in ("a", "b", "c", "d"):
        raise SystemExit(f"unknown SWARM_P2_PROTOCOL={PROTOCOL!r}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    stream_a = TokenStream(DOMAIN_A, rng)
    streams = {name: TokenStream(path, rng) for name, path in DOMAINS.items()}
    stream_map = streams

    base = build_model()
    n_params = sum(p.numel() for p in base.parameters())
    print(f"protocol {PROTOCOL.upper()}  base: {CONFIG_NAME}  "
          f"params={n_params/1e6:.2f}M  layer_types={base.layer_types}", flush=True)

    print(f"[phase 1] base train {BASE_STEPS} steps on fineweb", flush=True)
    base_ce, base_tok_s = train_base(base, stream_a, BASE_STEPS)
    print(f"  base ce={base_ce:.4f}  tok/s={base_tok_s:,.0f}", flush=True)

    print(f"[phase 2] protocol {PROTOCOL.upper()} "
          f"(exp_steps={EXP_STEPS} cal_steps={CAL_STEPS} rw={REJECT_W})", flush=True)
    t0 = time.time()
    if PROTOCOL == "a":
        harvested = protocol_a(base, stream_a, streams, stream_map)
        for j, name in enumerate(DOMAINS.keys()):
            plug_expert(base, 4 + j, harvested[name])
    elif PROTOCOL == "b":
        # protocol_b plugs in as it goes; harvests returned for the record
        harvested = protocol_b(base, stream_a, streams, stream_map)
    elif PROTOCOL == "d":
        harvested = protocol_d(base, stream_a, streams, stream_map)
    else:
        harvested = protocol_c(base, stream_a, streams, stream_map)
    phase2_s = time.time() - t0

    print(f"  experts now: {base.layers[0]['ffn'].num_experts}  "
          f"passports: {tuple(base.layers[0]['ffn'].router.passports.shape)}", flush=True)

    plug_ids = [4, 5, 6]
    streams_eval = {"fineweb": TokenStream(DOMAIN_A, rng)}
    streams_eval.update({n: TokenStream(p, rng) for n, p in DOMAINS.items()})

    purity, ces = {}, {}
    for name, st in streams_eval.items():
        util, ntok = util_matrix(base, st, EVAL_BATCHES, plug_ids)
        ce_with = ce_over(base, st, EVAL_BATCHES, mask=None)
        own = 4 + list(DOMAINS.keys()).index(name) if name in DOMAINS else None
        ce_wo = ce_over(base, st, EVAL_BATCHES, mask=own) if own else None
        purity[name] = {
            "util_per_plug": {str(k): round(v, 4) for k, v in util.items()},
            "own_expert": own,
            "own_util": round(util[own], 4) if own else None,
            "max_cross_util": round(max(v for k, v in util.items() if k != own), 4) if own else None,
            "ntok": ntok,
        }
        ces[name] = {
            "ce_with_all_plugins": round(ce_with, 4),
            "ce_own_masked": round(ce_wo, 4) if own else None,
            "ce_delta": round(ce_wo - ce_with, 4) if own else None,
        }

    gates = {}
    for name in DOMAINS:
        p, c = purity[name], ces[name]
        gates[name] = {
            "own_ok": p["own_util"] is not None and p["own_util"] >= OWN_GATE,
            "cross_ok": p["max_cross_util"] is not None and p["max_cross_util"] <= CROSS_GATE,
            "ce_ok": c["ce_delta"] is not None and c["ce_delta"] > 0,
        }
        gates[name]["pass"] = all(gates[name].values())
    overall = all(gates[n]["pass"] for n in DOMAINS)

    report = {
        "protocol": PROTOCOL,
        "config": CONFIG_NAME,
        "params_m": round(n_params / 1e6, 3),
        "layer_types": base.layer_types,
        "base_steps": BASE_STEPS,
        "base_ce": round(base_ce, 4),
        "base_tok_s": round(base_tok_s, 1),
        "exp_steps": EXP_STEPS,
        "cal_steps": CAL_STEPS if PROTOCOL == "b" else 0,
        "reject_w": REJECT_W,
        "owner_mass": OWNER_MASS,
        "phase2_seconds": round(phase2_s, 1),
        "plug_ids": {n: 4 + i for i, n in enumerate(DOMAINS.keys())},
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

    print(f"\n=== swarm2_purity protocol {PROTOCOL.upper()} ===")
    print(f"base ce={base_ce:.4f} tok/s={base_tok_s:,.0f} | phase2 {phase2_s:.0f}s")
    for name in streams_eval:
        p, c = purity[name], ces[name]
        print(f"{name:9s} own={p['own_util']} cross_max={p['max_cross_util']} "
              f"util={p['util_per_plug']} ce_with={c['ce_with_all_plugins']} "
              f"ce_masked={c['ce_own_masked']} delta={c['ce_delta']}")
    for name in DOMAINS:
        g = gates[name]
        print(f"gate {name:9s} own_ok={g['own_ok']} cross_ok={g['cross_ok']} "
              f"ce_ok={g['ce_ok']} -> {g['pass']}")
    print(f"OVERALL PASS: {overall}")
    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
