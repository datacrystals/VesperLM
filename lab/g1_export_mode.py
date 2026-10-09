#!/usr/bin/env python3
"""G1 — Hippocampus export mode (retention-parity gate) on tiny_agent_k step_best.

Instead of direct LoRA edits on the spine (the teach-by-talking demo), the
taught episodes are distilled into ONE section-4.2 contract expert and plugged
into the frozen spine via the D55-style protocol:

  1. Load the frozen spine (Pretrain/vesper_linear_checkpoints_tiny_agent_k/
     step_best).  The checkpoint ships a TopKRouter; it is swapped for a
     PassportRouter that is EXACTLY function-preserving (identity query,
     P = sqrt(d_r) * gate rows -> identical router logits up to float noise;
     verified at runtime).  This is the packaging step from MODULAR_MOE.md
     section 4.3 ("PassportRouter replaces TopKRouter"), not a weight touch:
     no spine tensor is modified at any point in this script.

  2. Per layer, one contract expert (section 4.2):  A_in (512->512,
     identity-init) / SwiGLU core (w1,w3: 512->1536, w2: 1536->512, layout
     identical to FeedForward) / A_out (512->512, identity-init).  Same-width
     same-lineage: the core is born as a copy of the layer's base expert 0
     (D55's birth init) so the module is function-preserving at birth.

  3. Consolidation training (frozen spine, expert params only): forced
     dispatch — every layer's MoE output is replaced by the new expert's
     output — with the demo's reward-weighted NLL + KL-to-base objective
     (Hippocampus/consolidate.train_lora_microsession's loss, minus the LoRA
     branch).  Same teach set as Hippocampus/demo.py so numbers compare
     directly with lab/imported/hippo_demo_tiny_agent_k_12k.log.

  4. Gate: the same stub_gate checks as the demo (mean reward / target
     collapse / unique ratio / KL bound).  PROMOTE -> plug in.

  5. Plug-in: append the expert + ONE passport row per layer.  Prototype init
     = mean router query over the expert's training examples (per layer, the
     mean ffn-input state over the teach sequences).  Two prototype scale
     variants are tried; if neither reaches the routing gate, a D55-style
     router-only recalibration (new row only) is the documented fallback.

  6. Measurements vs the direct-edit reference:
     (a) taught-fact word-vs-digit margins + NLL tails  (parity with LoRA)
     (b) base-mix val CE regression                     (< 1%)
     (c) poison batch -> gate ROLLBACK -> the poison row is dropped;
         incumbent state byte-identical + margins bit-equal
     (d) routing: utilization > 50% on home episodes,
         contamination < 30% on base mix

Shortcut log (also in the results JSON):
  - The 12k checkpoint predates PassportRouter (TopKRouter gate), so the
    passport bank is a function-preserving re-packaging of the gate matrix
    rather than a router trained with expert dropout.  Prototype reachability
    therefore says nothing about G2 (which needs a passport-trained spine).
  - "Base-mix val CE" is next-token CE over held-out chunks sampled from the
    Pretrain/data/index.txt corpus mix (the trainer's val stream is not
    rebuilt here).
  - One logical expert for all 8 layers (8 per-layer modules, one row per
    layer) = the section-4.4 `layer_ids=None` default.

Run:  python3 lab/g1_export_mode.py            # GPU if available, else CPU
Env:  G1_DEVICE, G1_STEPS (80), G1_LR (2e-3), G1_RECAL_STEPS (150), G1_SEED (0),
      G1_TAG (results suffix), G1_TEXTKL (1 = section-4.5 general-text KL),
      G1_RECAL_OBJ (mex | hinge | softmax — recal objective; mex = section-4.4a
      mutual-exclusion mass target, the D55-validated one)

G1b (passport-native spines) reuse:
      G1_CKPT=/path/to/step_best  — any VesperLinearLM checkpoint dir; dims,
      layer count and base-expert count are read from its model_config, and a
      passport-native router is kept as-is (no TopK transplant; verified +
      reported instead).  G1_ALWAYS_RECAL=1 forces the recal arm to run even
      when a prototype arm already reaches the gate (so the no-recal G2 arm and
      the v3 recal arm both get measured).  G1_RESULTS=<path> overrides the
      results file (G1b uses lab/results/g1b_*.json).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "Hippocampus"))
sys.path.insert(0, os.path.join(REPO, "Common"))

from consolidate import (  # noqa: E402
    load_base_model, generate_text, first_token_logits,
    word_digit_margin, response_nll, stub_gate,
    _encode_triple, _collate,
)
from session_log import SessionLog  # noqa: E402
from demo import GOOD_SESSION, POISON_SESSION, PROBES  # noqa: E402
from vesper_model import PassportRouter  # noqa: E402

CKPT = (os.environ.get("G1_CKPT") or os.path.join(
    REPO, "Pretrain", "vesper_linear_checkpoints_tiny_agent_k", "step_best"))
DEVICE = os.environ.get("G1_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
STEPS = int(os.environ.get("G1_STEPS", "80"))
LR = float(os.environ.get("G1_LR", "2e-3"))
KL_COEF = 0.05
# separate weight for the section-4.5 general-text KL (base-neutrality on
# foreign tokens); G1b probe: can a base-neutral expert make contaminated
# tokens cheap, i.e. decouple criterion (b) from criterion (d)?
TEXT_KL_COEF = float(os.environ.get("G1_TEXT_KL_COEF",
                                    os.environ.get("G1_KL_COEF", str(KL_COEF))))
RECAL_STEPS = int(os.environ.get("G1_RECAL_STEPS", "150"))
REJECT_W = 5.0
SEED = int(os.environ.get("G1_SEED", "0"))
# v2 knobs (the post-v1 fallback branch):
#   G1_TEXTKL=1       consolidate with section-4.5 KL-to-base on general text
#   G1_RECAL_HINGE=1  unsaturated hinge-margin recal (softmax CE saturates at
#                     this checkpoint's logit scale, ~1e2-1e3)
TEXT_KL = os.environ.get("G1_TEXTKL", "1") == "1"
RECAL_HINGE = os.environ.get("G1_RECAL_HINGE", "1") == "1"
# recal objective: "mex" = section-4.4a mutual-exclusion mass target
# (owner 0.55 on home / 0.0 on base, base rows share the rest) — the validated
# D55 joint-calibration objective; "softmax" = D55 phase-B one-hot contrastive;
# "hinge" = unsaturated ranking margins.
RECAL_OBJ = os.environ.get("G1_RECAL_OBJ",
                           "hinge" if RECAL_HINGE else "softmax")
# G1b: always run the recal arm even if a prototype arm already reaches the
# gate, so the no-recal (G2) arm and the recal arm are both measured.
ALWAYS_RECAL = os.environ.get("G1_ALWAYS_RECAL", "0") == "1"
TAG = os.environ.get("G1_TAG", "")
WORKDIR = os.path.join(REPO, "lab", "sandbox", "g1_export")
RESULTS = (os.environ.get("G1_RESULTS") or
           os.path.join(REPO, "lab", "results", f"g1_export_mode{TAG}.json"))

# Direct-edit reference: lab/imported/hippo_demo_tiny_agent_k_12k.log
# (same checkpoint, same probes, LoRA micro-session path).  G1b runs on other
# spines override the margins via G1_DIRECT_MARGINS_BEFORE/AFTER ("a,b,c") —
# produced by running Hippocampus/demo.py (direct-edit path) on that same
# checkpoint — so the parity bar is same-spine, not cross-model.
DIRECT_EDIT = {
    "log": "lab/imported/hippo_demo_tiny_agent_k_12k.log",
    "margins_before": [-5.001, -5.681, -4.502],
    "margins_after": [-1.815, -1.915, -1.806],
    "nll_digit_before": 2.061, "nll_digit_after": 7.343,
    "nll_word_before": 5.131, "nll_word_after": 4.478,
}
if os.environ.get("G1_DIRECT_MARGINS_BEFORE"):
    DIRECT_EDIT["margins_before"] = [
        float(x) for x in os.environ["G1_DIRECT_MARGINS_BEFORE"].split(",")]
    DIRECT_EDIT["margins_after"] = [
        float(x) for x in os.environ["G1_DIRECT_MARGINS_AFTER"].split(",")]
    DIRECT_EDIT["log"] = os.environ.get("G1_DIRECT_LOG", "same-spine direct-edit")
    for k, envk in (("nll_digit_before", "G1_DIRECT_NLL_DIGIT_B"),
                    ("nll_digit_after", "G1_DIRECT_NLL_DIGIT_A"),
                    ("nll_word_before", "G1_DIRECT_NLL_WORD_B"),
                    ("nll_word_after", "G1_DIRECT_NLL_WORD_A")):
        if os.environ.get(envk):
            DIRECT_EDIT[k] = float(os.environ[envk])
PROBE_CTX = "What is 2 + 2? The answer is"

# Architecture — overwritten from the checkpoint's model_config in main().
# Defaults are the G1 tiny_agent_k run.
DIM = 512
HIDDEN = 1536
N_LAYERS = 8
BASE_EXPERTS = 4  # checkpoint experts 0..3; memory expert plugs in at 4


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ------------------------------------------------------------------
# Contract expert (MODULAR_MOE.md section 4.2)
# ------------------------------------------------------------------

class ContractExpert(nn.Module):
    """A_in (identity-init) / SwiGLU core / A_out (identity-init).

    d_e = d_model = 512, h_e = hidden_dim = 1536: same-lineage same-width, so
    the expert is function-preserving at birth once the core is copied from a
    base expert of the same layer.
    """

    def __init__(self, dim=DIM, hidden=HIDDEN):
        super().__init__()
        self.a_in = nn.Linear(dim, dim, bias=False)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.a_out = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.a_in.weight)
        nn.init.eye_(self.a_out.weight)

    def forward(self, x):
        z = self.a_in(x)
        return self.a_out(self.w2(F.silu(self.w1(z)) * self.w3(z)))


def birth_from_base(expert: ContractExpert, base_expert):
    with torch.no_grad():
        expert.w1.weight.copy_(base_expert.w1.weight)
        expert.w2.weight.copy_(base_expert.w2.weight)
        expert.w3.weight.copy_(base_expert.w3.weight)


# ------------------------------------------------------------------
# Router transplant: TopKRouter -> PassportRouter, exactly
# ------------------------------------------------------------------

def prepare_routers(model) -> dict:
    """Make the router passport-addressable without touching base scoring.

    passport-native spines (G1b): already a PassportRouter — kept verbatim,
    including its trained expert dropout.  Returns provenance only.

    TopK spines (G1): swap for a PassportRouter with identical scoring
        logits = x @ gate.T                             (TopK)
               = (x @ I) @ (sqrt(d) * gate).T / sqrt(d)  (Passport)
    and report the max |logit| difference observed on a smoke batch.
    """
    info = {"mode": None, "max_logit_diff": 0.0, "expert_dropout": set(),
            "passport_dim": set()}
    for layer in model.layers:
        ffn = layer["ffn"]
        old = ffn.router
        if isinstance(old, PassportRouter):
            info["mode"] = info["mode"] or "passport-native"
            info["expert_dropout"].add(float(old.expert_dropout))
            info["passport_dim"].add(int(old.passport_dim))
            continue
        info["mode"] = info["mode"] or "topk-transplant"
        gate = old.gate.weight.data.clone()          # (E, dim)
        e, dim = gate.shape
        new = PassportRouter(dim, e, old.top_k, passport_dim=dim,
                             expert_dropout=0.0).to(device=gate.device,
                                                     dtype=gate.dtype)
        with torch.no_grad():
            new.query.weight.copy_(torch.eye(dim, dtype=gate.dtype,
                                             device=gate.device))
            new.passports.copy_(gate * math.sqrt(dim))
        ffn.router = new
        ffn.router_type = "passport"
        x = torch.randn(7, dim, device=gate.device, dtype=gate.dtype)
        old_logits = x @ gate.t()
        new_logits = new.query(x) @ new.passports.t() / math.sqrt(dim)
        info["max_logit_diff"] = max(
            info["max_logit_diff"], float((old_logits - new_logits).abs().max()))
        info["expert_dropout"].add(0.0)
        info["passport_dim"].add(dim)
    info["expert_dropout"] = sorted(info["expert_dropout"])
    info["passport_dim"] = sorted(info["passport_dim"])
    return info


# ------------------------------------------------------------------
# Forced dispatch (D55 phase-B "forced dispatch" at every layer)
# ------------------------------------------------------------------

@contextlib.contextmanager
def forced_experts(model, experts):
    """Replace every layer's MoE output with that layer's contract expert."""
    handles = []
    for layer, exp in zip(model.layers, experts):
        def hook(_mod, args, _out, exp=exp):
            y = exp(args[0])
            return y, torch.zeros((), device=y.device, dtype=y.dtype)
        handles.append(layer["ffn"].register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def expert_params(experts):
    return [p for e in experts for p in e.parameters()]


# ------------------------------------------------------------------
# Plug / drop (add_expert semantics with trained weights + chosen row)
# ------------------------------------------------------------------

def plug_expert(model, experts, rows):
    """Append per-layer expert modules and one passport row each (section 4.4)."""
    for i, layer in enumerate(model.layers):
        ffn = layer["ffn"]
        ffn.experts.append(experts[i])
        ffn.router.register_expert(passport_init=rows[i])
        ffn.num_experts += 1


def drop_last_expert(model):
    """Removal (section 4.6): drop the last row + module from every layer."""
    for layer in model.layers:
        ffn = layer["ffn"]
        r = ffn.router
        with torch.no_grad():
            kept = r.passports.data[:-1].clone()
        r.passports = nn.Parameter(kept)
        r.num_experts -= 1
        ffn.experts.pop(-1)
        ffn.num_experts -= 1


def replace_last_row(model, rows):
    for i, layer in enumerate(model.layers):
        r = layer["ffn"].router
        with torch.no_grad():
            kept = r.passports.data[:-1].clone()
            new = torch.cat([kept, rows[i].reshape(1, -1).to(kept)], dim=0)
        r.passports = nn.Parameter(new)


def snapshot_rows(model):
    return [layer["ffn"].router.passports.data[-1].clone()
            for layer in model.layers]


# ------------------------------------------------------------------
# ffn-input capture + routing stats (expert I/O contract input = post-ffn_norm)
# ------------------------------------------------------------------

def _as_batch(ids):
    return ids.unsqueeze(0) if ids.dim() == 1 else ids


@torch.no_grad()
def capture_ffn_inputs(model, batches):
    device = next(model.parameters()).device
    per_layer = [[] for _ in range(N_LAYERS)]
    handles = []
    for i, layer in enumerate(model.layers):
        def pre(_mod, args, i=i):
            per_layer[i].append(args[0].reshape(-1, args[0].shape[-1]).detach())
        handles.append(layer["ffn"].register_forward_pre_hook(pre))
    try:
        for ids in batches:
            model(_as_batch(ids).to(device))
    finally:
        for h in handles:
            h.remove()
    return [torch.cat(chunks, dim=0) for chunks in per_layer]





@torch.no_grad()
def routing_stats(model, batches, expert_idx):
    """Per-layer P(expert in top-2) and mean renormalized top-2 weight."""
    hs = capture_ffn_inputs(model, batches)
    rates, weights = [], []
    for i, layer in enumerate(model.layers):
        r = layer["ffn"].router
        logits = r.query(hs[i]) @ r.passports.t() / math.sqrt(r.passport_dim)
        probs = F.softmax(logits, dim=-1)
        tw, ti = torch.topk(probs, layer["ffn"].top_k, dim=-1)
        tw = tw / tw.sum(dim=-1, keepdim=True)
        hit = (ti == expert_idx).any(dim=-1).float()
        w = (tw * (ti == expert_idx).float()).sum(dim=-1)
        rates.append(float(hit.mean()))
        weights.append(float(w.mean()))
    return rates, weights


# ------------------------------------------------------------------
# Base-mix chunks from the trainer's corpus index
# ------------------------------------------------------------------

def load_base_mix_chunks(n_chunks=32, seq=256, seed=0):
    """Held-out CE / contamination proxy: chunks from the Pretrain/data/
    index.txt mix (uint16 token bins), sampled past the trainer's 95% train
    split boundary (the val tail)."""
    rng = np.random.default_rng(seed)
    index = os.path.join(REPO, "Pretrain", "data", "index.txt")
    bins = []
    with open(index) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            name, w = parts[0].strip(), float(parts[1].strip())
            path = os.path.join(REPO, "Pretrain", "data", name)
            if os.path.exists(path):
                bins.append((path, w))
    weights = np.array([w for _, w in bins], dtype=np.float64)
    weights /= weights.sum()
    chunks = []
    while len(chunks) < n_chunks:
        path = bins[rng.choice(len(bins), p=weights)][0]
        mm = np.memmap(path, dtype=np.uint16, mode="r")
        split = int(len(mm) * 0.95)
        off = int(rng.integers(split, max(split + 1, len(mm) - seq - 1)))
        toks = np.asarray(mm[off:off + seq + 1], dtype=np.int64)
        if len(toks) == seq + 1:
            chunks.append(torch.from_numpy(toks))
    return chunks


@torch.no_grad()
def next_token_ce(model, chunks):
    total, ntok = 0.0, 0
    for toks in chunks:
        ids = _as_batch(toks).to(next(model.parameters()).device)
        logits = model(ids)[0][0].float()
        nll = F.cross_entropy(logits[:-1], ids[0, 1:], reduction="sum")
        total += float(nll)
        ntok += ids.shape[1] - 1
    return total / ntok


# ------------------------------------------------------------------
# State hashing (byte-identity checks)
# ------------------------------------------------------------------

def state_hash(model) -> str:
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------
# Training: reward-weighted NLL + KL-to-base (demo objective, expert path)
# ------------------------------------------------------------------

def train_expert(model, tok, triples, experts, *, steps, lr, device, tag,
                 text_batches=None):
    """Reward-weighted NLL + KL-to-base on the episodes (demo objective),
    optionally plus KL-to-base on general text (section 4.5: the expert must
    not rewrite behavior outside its domain — the difference between a
    harmless plug-in and a base-CE regression)."""
    params = expert_params(experts)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    encoded = [e for e in (_encode_triple(tok, t, 256) for t in triples) if e]
    if not encoded:
        raise ValueError("no encodable triples")
    gen = torch.Generator().manual_seed(0)

    with torch.no_grad():  # base logits: bare spine routing, expert not forced
        base_cache = []
        for e in encoded:
            x1 = torch.tensor([e["x"]], dtype=torch.long, device=device)
            base_cache.append(model(x1)[0][0].float().cpu())
        text_cache = []
        for ids in (text_batches or []):
            ids = _as_batch(ids).to(device)
            text_cache.append((ids, model(ids)[0][0].float().cpu()))

    stats = {"loss0": None, "losses": [], "policy": [], "kl": [], "steps": steps,
             "text_kl": []}
    model.train()  # no dropout anywhere; matches the demo's micro-session
    for step in range(steps):
        idx = torch.randint(0, len(encoded), (min(4, len(encoded)),),
                            generator=gen).tolist()
        batch = _collate([encoded[i] for i in idx], tok.pad_token_id)
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        real = batch["real"].to(device)
        rewards = batch["rewards"].to(device)

        opt.zero_grad(set_to_none=True)
        with forced_experts(model, experts):
            logits = model(x)[0].float()
        T = logits.shape[1]
        base_logits = torch.zeros(len(idx), T, logits.shape[-1])
        for row, i in enumerate(idx):
            b = base_cache[i]
            base_logits[row, :b.shape[0]] = b
        base_logits = base_logits.to(device)

        logp = F.log_softmax(logits, dim=-1)
        nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        seq_nll = (nll * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        policy = (rewards * seq_nll).mean()

        logp_b = F.log_softmax(base_logits, dim=-1)
        p = logp.exp()
        kl_tok = (p * (logp - logp_b)).sum(-1)
        kl = (kl_tok * real).sum() / real.sum().clamp(min=1.0)

        text_kl = torch.zeros((), device=device)
        if text_cache:
            ti = int(torch.randint(0, len(text_cache), (1,),
                                   generator=gen).item())
            t_ids, t_base = text_cache[ti]
            with forced_experts(model, experts):
                t_logits = model(t_ids)[0][0].float()
            t_logp = F.log_softmax(t_logits, dim=-1)
            t_logp_b = F.log_softmax(t_base.to(device), dim=-1)
            text_kl = (t_logp.exp() * (t_logp - t_logp_b)).sum(-1).mean()

        loss = policy + KL_COEF * kl + TEXT_KL_COEF * text_kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if stats["loss0"] is None:
            stats["loss0"] = float(loss)
        stats["losses"].append(float(loss))
        stats["policy"].append(float(policy))
        stats["kl"].append(float(kl))
        stats["text_kl"].append(float(text_kl))
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            print(f"    [{tag} micro] step {step:3d}  loss {float(loss):+.4f}  "
                  f"policy {float(policy):+.4f}  kl {float(kl):.5f}  "
                  f"text_kl {float(text_kl):.5f}")
    model.eval()
    stats["loss_final"] = stats["losses"][-1]
    stats["final_kl"] = stats["kl"][-1]
    stats["final_text_kl"] = stats["text_kl"][-1]
    return stats


def run_gate(tag, triples, train_stats):
    responses = [t.response for t in triples]
    n = max(len(responses), 1)
    counts = {}
    for r in responses:
        counts[r] = counts.get(r, 0) + 1
    top = max(counts.values()) if counts else 0
    stats = {
        "n_triples": len(triples),
        "mean_reward": sum(t.reward for t in triples) / n,
        "unique_response_ratio": len(counts) / n,
        "top_response_fraction": top / n,
        "final_kl": train_stats["final_kl"],
        "loss0": train_stats["loss0"],
        "loss_final": train_stats["loss_final"],
    }
    cand = os.path.join(WORKDIR, "candidates", tag)
    os.makedirs(cand, exist_ok=True)
    with open(os.path.join(cand, "train_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    inc = os.path.join(WORKDIR, "candidates", "_none")
    os.makedirs(inc, exist_ok=True)
    verdict = stub_gate(cand, inc)
    print(f"  GATE[{tag}]: {verdict.decision} — {verdict.reason}")
    print(f"  gate metrics: {json.dumps(verdict.metrics)}")
    return verdict, stats


# ------------------------------------------------------------------
# Router-only recalibration (D55 phase-B contrastive, new row only)
# ------------------------------------------------------------------

def recal_rows(model, home_ids, base_provider, *, steps, lr, device):
    """Train only the new passport row of every layer.

    Two objective variants:
      softmax (D55 phase-B): home -> label <new idx>; base -> soft uniform over
        the base rows.  At this checkpoint's logit scale (~1e2-1e3) the softmax
        saturates and the gradient vanishes; measured in run v1 (contamination
        stuck at 0.89 over 150 steps).
      hinge (default): unsaturated ranking margins directly on the scores —
        home tokens: new row must beat the best base row by >= 1;
        base tokens: new row must sit below the 2nd-best base row by >= 1
        (i.e. out of top-2).  Row norm is re-projected to the mean base-row
        norm after every step (direction-only search).

    Features are re-captured through the live model each step (rows change
    routing, which changes upstream ffn inputs).
    """
    idx_new = model.layers[0]["ffn"].num_experts - 1
    trainable = []
    norm_targets = []
    for layer in model.layers:
        r = layer["ffn"].router
        r.passports.requires_grad_(True)
        trainable.append(r.passports)
        norm_targets.append(float(r.passports.data[:idx_new].norm(dim=1).mean()))

        def freeze_hook(grad, idx=idx_new):
            g = grad.clone()
            g[:idx] = 0
            return g
        r.passports.register_hook(freeze_hook)

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(1)
    for step in range(steps):
        base_ids = base_provider(gen)
        opt.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        home_h = capture_ffn_inputs(model, home_ids)
        base_h = capture_ffn_inputs(model, base_ids)
        for i, layer in enumerate(model.layers):
            r = layer["ffn"].router
            scale = math.sqrt(r.passport_dim)
            lb = (r.query(home_h[i]) @ r.passports.t()) / scale
            la = (r.query(base_h[i]) @ r.passports.t()) / scale
            if RECAL_OBJ == "hinge":
                s_new_h = lb[:, idx_new]
                s_best_h = lb[:, :idx_new].max(dim=-1).values
                loss_home = F.relu(1.0 + s_best_h - s_new_h).mean()
                s_new_a = la[:, idx_new]
                s_second_a = la[:, :idx_new].topk(2, dim=-1).values[:, 1]
                loss_base = F.relu(1.0 + s_new_a - s_second_a).mean()
                loss = loss + loss_home + REJECT_W * loss_base
            elif RECAL_OBJ == "mex":
                # section-4.4a mutual-exclusion mass target:
                # home  -> owner 0.55, base rows share 0.45
                # base  -> owner 0.00, base rows share 1.0
                target_h = torch.zeros(lb.size(0), lb.size(1), device=device)
                target_h[:, :idx_new] = 0.45 / idx_new
                target_h[:, idx_new] = 0.55
                target_a = torch.zeros(la.size(0), la.size(1), device=device)
                target_a[:, :idx_new] = 1.0 / idx_new
                loss_h = -(target_h * F.log_softmax(lb, dim=-1)
                           ).sum(dim=-1).mean()
                loss_a = -(target_a * F.log_softmax(la, dim=-1)
                           ).sum(dim=-1).mean()
                loss = loss + loss_h + loss_a
            else:  # D55 phase-B one-hot contrastive (saturates at this scale)
                r_ce_b = F.cross_entropy(
                    lb, torch.full((lb.size(0),), idx_new,
                                   dtype=torch.long, device=device))
                target = torch.zeros(la.size(0), la.size(1), device=device)
                target[:, :idx_new] = 1.0 / idx_new
                r_ce_a = -(target * F.log_softmax(la, dim=-1)
                           ).sum(dim=-1).mean()
                loss = loss + r_ce_b + REJECT_W * r_ce_a
        loss.backward()
        opt.step()
        if RECAL_OBJ == "hinge":
            with torch.no_grad():  # direction-only: keep the row's norm in-distribution
                for i, layer in enumerate(model.layers):
                    p = layer["ffn"].router.passports
                    v = p.data[idx_new]
                    n = float(v.norm())
                    if n > 1e-8:
                        v.mul_(norm_targets[i] / n)
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            print(f"    [recal] step {step:3d}  loss {float(loss):+.4f}")
    for layer in model.layers:
        layer["ffn"].router.passports.requires_grad_(False)
    return {"steps": steps, "lr": lr, "reject_w": REJECT_W,
            "objective": RECAL_OBJ}


# ------------------------------------------------------------------
# Probe display (parity with the demo log format)
# ------------------------------------------------------------------

@torch.no_grad()
def probe_set(model, tok, tag):
    print(f"\n  --- probes [{tag}] ---")
    margins = {}
    for p in PROBES:
        gen = generate_text(model, tok, p, max_new_tokens=8, device=DEVICE)
        logits = first_token_logits(model, tok, p, device=DEVICE)
        m = word_digit_margin(logits, tok)
        margins[p] = m["margin"]
        print(f"  Q: {p!r}")
        print(f"    A: {gen!r}")
        print(f"    margin word-digit: {m['margin']:+.3f}  "
              f"(best word {m['best_word']!r} {m['word_logit']:.2f} vs "
              f"best digit {m['best_digit']!r} {m['digit_logit']:.2f})")
    nll_digit = response_nll(model, tok, PROBE_CTX, " 4.", DEVICE)
    nll_word = response_nll(model, tok, PROBE_CTX, " four.", DEVICE)
    print(f"  NLL probes after {PROBE_CTX!r}: "
          f"digit-tail {nll_digit:.3f}  word-tail {nll_word:.3f}")
    return {"margins": margins, "nll_digit": nll_digit, "nll_word": nll_word}


def encode_texts(tok, texts, device):
    out = []
    for t in texts:
        ids = tok(t, add_special_tokens=False).input_ids
        if ids:
            out.append(torch.tensor([ids], dtype=torch.long, device=device))
    return out


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    t0 = time.time()
    os.makedirs(WORKDIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    set_seed(SEED)
    print(f"G1 export mode  device={DEVICE}  steps={STEPS}  lr={LR}  "
          f"recal_steps={RECAL_STEPS}  seed={SEED}")

    section("PART 0 — load frozen spine + router prep")
    model, tok, mc = load_base_model(CKPT, device=DEVICE)
    global DIM, HIDDEN, N_LAYERS, BASE_EXPERTS
    DIM = int(mc["dim"])
    HIDDEN = int(mc["hidden_dim"])
    N_LAYERS = int(mc["n_layers"])
    BASE_EXPERTS = int(mc["num_experts"])
    print(f"  checkpoint: {CKPT}")
    print(f"  model_config: {mc}")
    rinfo = prepare_routers(model)
    print(f"  router prep: mode={rinfo['mode']}  "
          f"passport_dim={rinfo['passport_dim']}  "
          f"expert_dropout={rinfo['expert_dropout']}  "
          f"max |logit diff| = {rinfo['max_logit_diff']:.3e}")
    assert rinfo["max_logit_diff"] < 1e-3, "router prep is not function-preserving"
    assert rinfo["mode"] == "passport-native" or rinfo["expert_dropout"] == [0.0]

    base_mix = load_base_mix_chunks(n_chunks=32, seq=256, seed=SEED)
    print(f"  base-mix chunks: {len(base_mix)} x 256 tokens from "
          f"Pretrain/data/index.txt val tail")
    ce_base = next_token_ce(model, base_mix)
    print(f"  base-mix CE (prepared spine): {ce_base:.4f}")

    section("PART 1 — teach session (same fact batches as the demo)")
    log_path = os.path.join(WORKDIR, "session_good.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    log = SessionLog(log_path)
    for i, (prompt, response, correction) in enumerate(GOOD_SESSION):
        log.log_turn("sess-teach", i, prompt, response)
        if correction is not None:
            log.mark_feedback("sess-teach", i, "reject", confidence=1.0,
                              note="Please answer math in words, not digits",
                              correction=correction)
        else:
            log.mark_feedback("sess-teach", i, "approve", confidence=1.0,
                              note="yes — in words like that")
        print(f"  turn {i}: Q={prompt!r} A={response!r} -> "
              f"{'reject+correction' if correction else 'approve'}")
    good_triples = log.to_triples()
    print(f"  eligible training triples: {len(good_triples)}")

    poison_path = os.path.join(WORKDIR, "session_poison.jsonl")
    if os.path.exists(poison_path):
        os.remove(poison_path)
    plog = SessionLog(poison_path)
    for i, (prompt, response) in enumerate(POISON_SESSION):
        plog.log_turn("sess-poison", i, prompt, response)
        plog.mark_feedback("sess-poison", i, "approve", confidence=1.0,
                           note="perfect, always answer like this",
                           correction="banana")
    poison_triples = plog.to_triples()
    print(f"  poison triples: {len(poison_triples)} "
          f"(all reward>0, all response 'banana')")

    section("PART 2 — BEFORE (bare spine)")
    before = probe_set(model, tok, "before")

    section("PART 3 — distill into contract experts (frozen spine)")
    experts = [ContractExpert(DIM, HIDDEN).to(DEVICE) for _ in range(N_LAYERS)]
    for i, e in enumerate(experts):
        birth_from_base(e, model.layers[i]["ffn"].experts[0])
    with torch.no_grad():  # module-level function-preservation at birth
        x = torch.randn(85, DIM, device=DEVICE)
        birth_diff = float((model.layers[0]["ffn"].experts[0](x)
                            - experts[0](x)).abs().max())
    print(f"  expert birth == base expert0 output: max diff {birth_diff:.3e}")
    n_params = sum(p.numel() for p in expert_params(experts))
    print(f"  contract expert params: {n_params:,} "
          f"({N_LAYERS} layers x A_in/core/A_out)")
    good_stats = train_expert(model, tok, good_triples, experts,
                              steps=STEPS, lr=LR, device=DEVICE, tag="good",
                              text_batches=base_mix[:4] if TEXT_KL else None)
    print(f"  train: loss {good_stats['loss0']:+.4f} -> "
          f"{good_stats['loss_final']:+.4f}  final_kl {good_stats['final_kl']:.5f}")
    good_gate, good_gate_stats = run_gate("good", good_triples, good_stats)
    assert good_gate.decision == "PROMOTE", "good batch should have been promoted"

    section("PART 4 — prototype passport rows (mean router query)")
    home_texts = [p + r for (p, r, _c) in GOOD_SESSION]
    home_ids = encode_texts(tok, home_texts, DEVICE)
    with torch.no_grad():
        home_h = capture_ffn_inputs(model, home_ids)  # bare-spine features
        # prototype = mean router QUERY over the expert's training examples,
        # i.e. mean(W_q h) in passport space (equals mean(h) only when the
        # query map is the G1 identity transplant).
        mean_q = [model.layers[i]["ffn"].router.query(home_h[i]).mean(dim=0)
                  for i in range(len(model.layers))]
        row_norms = [float(model.layers[i]["ffn"].router.passports.data
                           .norm(dim=1).mean()) for i in range(len(model.layers))]
    rows_literal = [mean_q[i].clone() for i in range(len(model.layers))]
    rows_normed = [mean_q[i] * (row_norms[i] / (mean_q[i].norm() + 1e-8))
                   for i in range(len(model.layers))]
    print(f"  mean-query row norms: "
          f"{[round(float(r.norm()), 2) for r in rows_literal]}")
    print(f"  bank row norms:       {[round(n, 2) for n in row_norms]}")

    home_probe_ids = encode_texts(tok, list(PROBES), DEVICE)

    def apply_rows(rows):
        if model.layers[0]["ffn"].num_experts == BASE_EXPERTS:
            plug_expert(model, experts, rows)
        else:
            for i, layer in enumerate(model.layers):
                layer["ffn"].experts[BASE_EXPERTS].load_state_dict(
                    experts[i].state_dict())
            replace_last_row(model, rows)

    def measure(tag, rows):
        apply_rows(rows)
        after = probe_set(model, tok, tag)
        rates, weights = routing_stats(model, home_probe_ids + home_ids,
                                       BASE_EXPERTS)
        base_rates, _ = routing_stats(model, base_mix[:8], BASE_EXPERTS)
        util = sum(rates) / len(rates)
        contam = sum(base_rates) / len(base_rates)
        ce_plug = next_token_ce(model, base_mix)
        print(f"  routing[{tag}]: util_home={util:.3f} (per-layer "
              f"{[round(r, 3) for r in rates]})  contam_base={contam:.3f}")
        print(f"  home top-2 weight on expert: {sum(weights)/len(weights):.3f}")
        print(f"  base-mix CE[{tag}]: {ce_plug:.4f}  regression "
              f"{100 * (ce_plug - ce_base) / ce_base:+.3f}%")
        return {
            "after": after, "util_home": util, "util_per_layer": rates,
            "contam_base": contam,
            "home_weight": sum(weights) / len(weights),
            "ce_plug": ce_plug,
            "ce_regression_pct": 100 * (ce_plug - ce_base) / ce_base,
        }

    variants = {}
    rows_registry = {"mean_query": rows_literal, "norm_matched": rows_normed}
    for name, rows in rows_registry.items():
        print(f"\n  --- plug-in attempt: {name} ---")
        variants[name] = measure(name, rows)
        variants[name]["recal"] = None

    def routed_ok(v):
        return v["util_home"] > 0.5 and v["contam_base"] < 0.3

    chosen = next((n for n in ("mean_query", "norm_matched")
                   if routed_ok(variants[n])), None)
    if chosen is None or ALWAYS_RECAL:
        section("PART 4b — router-only recalibration (mex/hinge/softmax)")

        def base_provider(gen):
            idx = torch.randint(0, len(base_mix), (8,), generator=gen).tolist()
            return [base_mix[i] for i in idx]

        apply_rows(rows_literal)
        info = recal_rows(model, home_ids, base_provider,
                          steps=RECAL_STEPS, lr=1e-2, device=DEVICE)
        rows_trained = snapshot_rows(model)
        rows_registry["recalibrated"] = rows_trained
        variants["recalibrated"] = measure("recalibrated", rows_trained)
        variants["recalibrated"]["recal"] = info
        if chosen is None:
            chosen = ("recalibrated" if routed_ok(variants["recalibrated"])
                      else max(variants, key=lambda k: variants[k]["util_home"]))
        elif not routed_ok(variants[chosen]):
            chosen = ("recalibrated" if routed_ok(variants["recalibrated"])
                      else max(variants, key=lambda k: variants[k]["util_home"]))
    print(f"\n  chosen plug-in variant: {chosen}")

    # canonical "after" state: explicitly re-apply the chosen rows
    chosen_rows = rows_registry[chosen]
    apply_rows(chosen_rows)
    after = probe_set(model, tok, "after (chosen)")
    print(f"  margins match chosen variant: "
          f"{all(abs(after['margins'][p] - variants[chosen]['after']['margins'][p]) < 1e-6 for p in PROBES)}")

    section("PART 5 — poison batch -> gate ROLLBACK -> one row drop")
    incumbent_hash = state_hash(model)
    incumbent_margins = dict(after["margins"])
    print(f"  incumbent state hash: {incumbent_hash[:16]}…")

    poison_experts = [ContractExpert(DIM, HIDDEN).to(DEVICE)
                      for _ in range(N_LAYERS)]
    for i, e in enumerate(poison_experts):
        birth_from_base(e, model.layers[i]["ffn"].experts[0])
    poison_stats = train_expert(model, tok, poison_triples, poison_experts,
                                steps=max(24, STEPS // 2), lr=LR,
                                device=DEVICE, tag="poison",
                                text_batches=base_mix[:4] if TEXT_KL else None)
    poison_gate, _ = run_gate("poison", poison_triples, poison_stats)
    print(f"  decision={poison_gate.decision}  reason={poison_gate.reason!r}")

    # incident: the poison expert is plugged anyway (as the 6th expert).
    # Row scale is bank-matched so the poison row actually fires — a
    # literal mean-query row (norm ~9 vs bank ~370) never enters top-2 and
    # the "incident" would be invisible.
    poison_texts = [p + r for (p, r) in POISON_SESSION]
    with torch.no_grad():
        poison_h = capture_ffn_inputs(
            model, encode_texts(tok, poison_texts, DEVICE))
    poison_rows = []
    for i in range(N_LAYERS):
        q = model.layers[i]["ffn"].router.query(poison_h[i]).mean(dim=0)
        poison_rows.append(q * (row_norms[i] / (float(q.norm()) + 1e-8)))
    plug_expert(model, poison_experts, poison_rows)
    nll_banana = response_nll(model, tok, PROBE_CTX, " banana.", DEVICE)
    nll_four = response_nll(model, tok, PROBE_CTX, " four.", DEVICE)
    print(f"  [poison plugged] NLL after {PROBE_CTX!r}: "
          f"' banana.' {nll_banana:.3f}   ' four.' {nll_four:.3f}")
    poison_probe = probe_set(model, tok, "poison-plugged")

    # recovery: ONE row drop (section 4.6 removal)
    drop_last_expert(model)
    post_hash = state_hash(model)
    after2 = probe_set(model, tok, "after-poison-drop")
    hash_ok = (post_hash == incumbent_hash)
    margins_ok = all(after2["margins"][p] == incumbent_margins[p]
                     for p in PROBES)
    intact = hash_ok and margins_ok
    print(f"  state hash after rollback: {post_hash[:16]}…")
    print(f"  hash match: {hash_ok}   margins bit-equal: {margins_ok}")
    print(f"  incumbent intact: {intact}")

    section("PART 6 — verdicts")
    gain_export = [after["margins"][p] - before["margins"][p] for p in PROBES]
    gain_direct = [a - b for a, b in zip(DIRECT_EDIT["margins_after"],
                                         DIRECT_EDIT["margins_before"])]
    mean_gain_export = sum(gain_export) / len(gain_export)
    mean_gain_direct = sum(gain_direct) / len(gain_direct)
    a_pass = mean_gain_export >= 0.5 * mean_gain_direct
    b_pass = variants[chosen]["ce_regression_pct"] < 1.0
    c_pass = (poison_gate.decision == "ROLLBACK") and intact
    d_pass = routed_ok(variants[chosen])
    print(f"  (a) margin parity: export mean gain {mean_gain_export:+.3f} vs "
          f"direct-edit {mean_gain_direct:+.3f} -> "
          f"{'PASS' if a_pass else 'FAIL'} (bar: >=50% of direct-edit gain)")
    for j, p in enumerate(PROBES):
        print(f"      {p!r:28s} {before['margins'][p]:+.3f} -> "
              f"{after['margins'][p]:+.3f}  (export {gain_export[j]:+.3f}, "
              f"direct-edit {gain_direct[j]:+.3f})")
    print(f"      NLL digit-tail: {before['nll_digit']:.3f} -> "
          f"{after['nll_digit']:.3f}   word-tail: {before['nll_word']:.3f} -> "
          f"{after['nll_word']:.3f}")
    print(f"  (b) base-mix CE regression: "
          f"{variants[chosen]['ce_regression_pct']:+.3f}% -> "
          f"{'PASS' if b_pass else 'FAIL'} (bar: <1%)")
    print(f"  (c) poison rollback: gate={poison_gate.decision}, "
          f"incumbent byte-identical={intact} -> {'PASS' if c_pass else 'FAIL'}")
    print(f"  (d) routing: util_home={variants[chosen]['util_home']:.3f} (>0.5), "
          f"contam_base={variants[chosen]['contam_base']:.3f} (<0.3) -> "
          f"{'PASS' if d_pass else 'FAIL'}")
    overall = a_pass and b_pass and c_pass and d_pass
    print(f"  G1 OVERALL: {'PASS' if overall else 'FAIL'}")

    # per-arm criteria (G1b: the no-recal G2 arm and the recal arm are judged
    # on their own numbers, not just the selected incumbent)
    per_arm = {}
    for name, v in variants.items():
        g = [v["after"]["margins"][p] - before["margins"][p] for p in PROBES]
        mg = sum(g) / len(g)
        per_arm[name] = {
            "mean_margin_gain": mg,
            "a_pass": mg >= 0.5 * mean_gain_direct,
            "margins_after": dict(v["after"]["margins"]),
            "ce_regression_pct": v["ce_regression_pct"],
            "b_pass": v["ce_regression_pct"] < 1.0,
            "util_home": v["util_home"],
            "contam_base": v["contam_base"],
            "d_pass": routed_ok(v),
            "home_weight": v["home_weight"],
            "recal": v.get("recal"),
        }
        print(f"  arm[{name:14s}] gain {mg:+.3f} (a:{'P' if per_arm[name]['a_pass'] else 'F'})  "
              f"ce {v['ce_regression_pct']:+.1f}% (b:{'P' if per_arm[name]['b_pass'] else 'F'})  "
              f"util {v['util_home']:.3f} contam {v['contam_base']:.3f} "
              f"(d:{'P' if per_arm[name]['d_pass'] else 'F'})")

    out = {
        "gate": "G1" if not os.environ.get("G1_RESULTS") else "G1b",
        "verdict": "PASS" if overall else "FAIL",
        "per_arm": per_arm,
        "criteria": {
            "a_margin_parity": {
                "pass": a_pass,
                "export_mean_gain": mean_gain_export,
                "direct_mean_gain": mean_gain_direct,
                "bar": ">=50% of direct-edit mean gain",
                "margins_before": {p: before["margins"][p] for p in PROBES},
                "margins_after": {p: after["margins"][p] for p in PROBES},
                "nll_digit": [before["nll_digit"], after["nll_digit"]],
                "nll_word": [before["nll_word"], after["nll_word"]],
            },
            "b_base_ce": {
                "pass": b_pass, "ce_base": ce_base,
                "ce_plugged": variants[chosen]["ce_plug"],
                "regression_pct": variants[chosen]["ce_regression_pct"],
                "bar": "<1%",
            },
            "c_poison_rollback": {
                "pass": c_pass,
                "gate_decision": poison_gate.decision,
                "gate_reason": poison_gate.reason,
                "gate_metrics": poison_gate.metrics,
                "incumbent_byte_identical": hash_ok,
                "margins_bit_equal": margins_ok,
                "poison_plugged_nll_banana": nll_banana,
                "poison_plugged_nll_four": nll_four,
                "poison_plugged_margins": poison_probe["margins"],
            },
            "d_routing": {
                "pass": d_pass,
                "util_home": variants[chosen]["util_home"],
                "util_per_layer": variants[chosen]["util_per_layer"],
                "home_weight": variants[chosen]["home_weight"],
                "contam_base": variants[chosen]["contam_base"],
                "bar": "util>0.5, contam<0.3",
                "chosen_variant": chosen,
                "variants": variants,
            },
        },
        "direct_edit_reference": DIRECT_EDIT,
        "config": {
            "checkpoint": CKPT, "device": DEVICE, "steps": STEPS, "lr": LR,
            "kl_coef": KL_COEF, "recal_steps": RECAL_STEPS, "seed": SEED,
            "tag": TAG, "text_kl": TEXT_KL, "recal_hinge": RECAL_HINGE,
            "recal_obj": RECAL_OBJ,
            "expert_params": n_params,
            "expert_birth_max_diff": birth_diff,
            "router_prep_mode": rinfo["mode"],
            "router_max_logit_diff": rinfo["max_logit_diff"],
            "router_expert_dropout": rinfo["expert_dropout"],
            "passport_dim": rinfo["passport_dim"],
            "dim": DIM, "hidden_dim": HIDDEN, "n_layers": N_LAYERS,
            "num_experts_base": BASE_EXPERTS,
            "always_recal": ALWAYS_RECAL,
        },
        "train": {"good": good_stats, "good_gate": good_gate_stats,
                  "poison": poison_stats},
        "shortcuts": [
            "TopK->Passport router transplant is a function-preserving "
            "repackaging of the gate matrix (this checkpoint predates "
            "PassportRouter training); G2 reachability claims need a "
            "passport-trained spine",
            "base-mix CE = held-out chunks from Pretrain/data/index.txt val "
            "tail, not the trainer's exact val stream",
            "one logical expert = 8 per-layer modules + one row per layer "
            "(layer_ids=None)",
            "forced-dispatch training replaces each layer's MoE mix with the "
            "expert (D55 phase-B pattern); inference uses the real top-2 mix",
        ],
        "seconds": round(time.time() - t0, 1),
    }
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RESULTS}")
    print(f"total time: {out['seconds']}s")


if __name__ == "__main__":
    main()
