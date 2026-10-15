#!/usr/bin/env python3
"""grafted_spine_v0 — assemble + chat-test the first "mostly grafted experts" model.

v0 of the user's north star: "a model I can run and talk to (a few B params)
that is actually intelligent enough to have a conversation and is mostly
grafted experts."  Every part below was already validated on its own; this
script only assembles them and runs the chat-level test.

Recipe
------
spine       microsoft/Phi-mini-MoE-instruct @ f620b32c (7.65B / 1.8B active,
            instruct-tuned -> conversational out of the box, ChatML template).
graft       the D2 passport/registry on its family
            (`/data/tliao/graft_d2_out/registry_phimoe_f620b32c.pt`, parity
            0.998 — D2 cross-family PASS).  Score function only: stock
            sparsemixer selection runs unchanged on passport scores.
specialties Door 2.5 wrap recipe (frozen donor core + thin task-loss adapters,
            lr 3e-3 / wd 0 / out_pen 300, 250 steps) on the D2 donor cores:
            code = L13 e8 (the door25 winner pick), math = L24 e4 (the door6
            pick).  OLMoE-host wraps cannot be reused (d_model 2048 vs 4096);
            these are fresh wraps in Phi geometry (786k -> 1.05M adapters).
memories    the §8 line: shaped episodic memory experts on the
            lab/memory_shaping.py delta contract (token-substitution shape —
            the G3 winner), each wrapped exactly like a specialty on its
            episode home text (ChatML), with the owner_mass 0.85 CLOSING row
            recal (g3_owner_mass_fix lever) at joint time.
delivery    the winning additive contract (Door 4 (b) / Door 6 MixMerge):
            y = y_stock + sum_j lam_j * m_j * e_j   (per expert per layer;
            membership m_j = plug-in row score beats the k-th stock score —
            no softmax-mass competition by construction).

Phases (one process per GPU for fit; assemble once)
---------------------------------------------------
  --phase fit       fit the listed wraps and save plugin artifacts
  --phase assemble  load spine + registry + plugins -> closing recal,
                    lambda fit, chat test (bare vs grafted), metrics
  --phase all       fit everything then assemble in one process

Ops (HANDOFF rules): detached setsid GPU jobs, incremental JSON after every
phase, train bs4 x seq512 = 2048 (gfx906 B*T hole {256,768,1024,4096}; eval
bs2 x 512), NaN guard on every saved artifact, numeric-only output.  Chat
transcripts go to JSON (untracked); the report only quotes truncated snippets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import graft_d0_parity as d0  # noqa: E402
import graft_d1_plugin as d1  # noqa: E402
import graft_d2_crossfamily as d2  # noqa: E402
import door25_wrap_expert as d25  # noqa: E402
import door6_expert_mix as d6  # noqa: E402

try:
    import memory_shaping as ms  # noqa: E402  the §8 delta contract
except Exception:  # pragma: no cover - fallback keeps the harness runnable
    ms = None

FORMAT = "vesper-grafted-spine-v0"
BAR_PARITY_COS = 0.90          # chat-level parity (logit cosine, bare vs graft)
BAR_CONTAM = 0.30
BAR_OWNER_SPEC = 0.35          # door6 row-recal target for specialty rows
BAR_OWNER_MEM = 0.85           # g3_owner_mass_fix closing target for memories

# v0 wrap table: name, kind, donor/host layer, donor expert, home key
# (kind home keys: code/math = graft_d2_slice file; pet/editor/words = episodes)
V0_WRAPS = [
    ("code",     "code", 13, 8,  "code"),
    ("math",     "math", 24, 4,  "math"),
    ("mem_pet",  "mem",   6, 3,  "pet"),
    ("mem_edit", "mem",  19, 10, "editor"),
    ("mem_words", "mem", 27, 2,  "words"),
]

HOLEY_BT = {256, 768, 1024, 4096}


# ---------------------------------------------------------------------------
# Incremental JSON
# ---------------------------------------------------------------------------

def dump_json(out: dict, path: str | None) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, default=str)
    os.replace(tmp, path)


def finite_state(*objs) -> bool:
    for o in objs:
        for p in o.parameters() if hasattr(o, "parameters") else [o]:
            if not bool(torch.isfinite(p.detach()).all()):
                return False
    return True


# ---------------------------------------------------------------------------
# MixGateD2 — passport stock selection (donor semantics) + plug-in membership
# ---------------------------------------------------------------------------

class MixGateD2(d2.GraftGateD2):
    """GraftGateD2 generalized to n_new plug-in rows.

    Stock part: donor selection (sparsemixer for phimoe) over the STOCK rows
    only — bit-identical to the masked control.  Each plug-in row only votes
    membership: score beats the k-th stock score (door6.MixGate semantics,
    family-aware).
    """

    def __init__(self, bank, fam, stock_gate, n_new: int, jitter_eps: float = 0.0):
        super().__init__(bank, fam, stock_gate, jitter_eps)
        self.n_new = n_new
        self.last_membership = None
        self.last_p_new = None

    def forward(self, hidden_states: torch.Tensor):
        h = hidden_states.reshape(-1, self.bank.d_model)
        scores = self.bank.scores(h)                      # (T, E_stock+n_new)
        n_stock = self.bank.num_experts - self.n_new
        stock_scores = scores[:, :n_stock]
        if self.fam.name == d2.FAMILY_PHIMOE:
            weights, idx = d2.select_sparsemixer(stock_scores, self.jitter_eps,
                                                 self.fam.top_k)
        else:
            weights, idx = d2.select_softmax_topk(stock_scores, self.fam.top_k,
                                                  self.fam.norm_topk_prob)
        weights = weights.to(hidden_states.dtype)
        kth = torch.topk(stock_scores, self.fam.top_k, dim=-1).values[:, -1]
        self.last_membership = scores[:, n_stock:] > kth.unsqueeze(-1)
        self.last_p_new = F.softmax(scores, dim=-1)[:, n_stock:]
        self.last_h = h.detach()
        self.last_logits = scores.detach()
        self.last_indices = idx.detach()
        return scores, weights, idx


# ---------------------------------------------------------------------------
# Family-aware additive mix install (door6.MixMerge, door6.install_mix port)
# ---------------------------------------------------------------------------

class MixMerge(d6.MixMerge):
    """y = stock + sum_j lam_j * m_j * e_j — inherited from door6 unchanged."""
    pass


def gate_getter_fam(blk, fam):
    return lambda b=blk: getattr(b, fam.gate_attr)


def install_mix(blocks, fam, site_entries, mode="off") -> list:
    """One MixMerge per host block (door6.install_mix, family-aware)."""
    handles = []
    for i, entries in site_entries.items():
        blk = blocks[i]
        sf = blk.experts.forward
        pm_old = getattr(blk, "_plugin_merge", None)
        if pm_old is not None and hasattr(pm_old, "stock"):
            sf = pm_old.stock
        num_stock = entries[0]["num_stock"]
        pm = MixMerge(sf, [en["expert"] for en in entries],
                      [en["lam"] for en in entries],
                      [en["tag"] for en in entries],
                      gate_getter=gate_getter_fam(blk, fam),
                      num_stock=num_stock)
        pm.mode = mode
        blk._plugin_merge = pm

        def make_fwd(pm=pm):
            def fwd(hidden_states, top_k_index, top_k_weights):
                return pm(hidden_states, top_k_index, top_k_weights)
            return fwd

        handles.append((blk, sf))
        blk.experts.forward = make_fwd()
    return handles


def set_mix_active(handles, which: str) -> None:
    """which: both | <tag> — per-plug-in delivery flags (door6 semantics)."""
    for blk, _ in handles:
        pm = getattr(blk, "_plugin_merge", None)
        if pm is None:
            continue
        pm.active = [(which == "both" or which == tag) for tag in pm.tags]


def run_in_state(model, blocks, fam, gates, handles, which, fn, force_gain=1.0):
    """Install gates + plug-in state, run fn, restore (d1.run_in_state port)."""
    restored = None
    if gates is not None:
        restored = [(blk, d2.install_graft(blk, fam, g))
                    for blk, g in zip(blocks, gates)]
    if handles is not None:
        if which in ("off", "none"):
            d1.set_plugin_mode(handles, "off")
        elif which == "forced":
            d1.set_plugin_mode(handles, "forced", force_gain)
        else:
            d1.set_plugin_mode(handles, "additive")
            set_mix_active(handles, which)
    try:
        return fn()
    finally:
        if restored is not None:
            for blk, stock in restored:
                d2.uninstall_graft(blk, fam, stock)
        if handles is not None:
            d1.set_plugin_mode(handles, "off")


@torch.no_grad()
def collect_pairs_fam(model, blocks, fam, batches, device, max_pairs,
                      label_of):
    """(h, label) per layer under CURRENT gates (d1.collect_gate_pairs port)."""
    n = len(blocks)
    hs = [[] for _ in range(n)]
    labs = [[] for _ in range(n)]
    tap = d2.GateTapD2(blocks, fam)
    for b_i, (tokens, _dom) in enumerate(batches):
        tap.clear()
        d0.model_logits(model, tokens.to(device))
        lab = label_of(b_i, batches[b_i])
        for i in range(n):
            if tap.h[i] is None:
                continue
            hs[i].append(tap.h[i].float().cpu())
            labs[i].append(torch.full((tap.h[i].shape[0],), float(lab)))
    tap.close()
    out = []
    for i in range(n):
        if not hs[i]:
            out.append((torch.zeros(0, 1), torch.zeros(0)))
            continue
        h = torch.cat(hs[i], dim=0)
        y = torch.cat(labs[i], dim=0)
        if h.shape[0] > max_pairs:
            stride = h.shape[0] // max_pairs + 1
            h = h[::stride][:max_pairs]
            y = y[::stride][:max_pairs]
        out.append((h, y))
    return out


def build_gates(blocks, banks_reg, fam, site_rows):
    """Passport gates for every layer; site layers get MixGateD2 banks with
    plug-in rows appended (door6.build_mix_gates, family-aware)."""
    gates, site_banks = [], {}
    for i, blk in enumerate(blocks):
        bank_reg = banks_reg[i]
        rows = site_rows.get(i, [])
        stock = d2.gate_of(blk, fam)
        eps = float(getattr(stock, "router_jitter_noise", 0.0))
        if rows:
            bank = d0.PassportBank(
                bank_reg.num_experts + len(rows), bank_reg.d_model,
                d_r=bank_reg.d_r, learn_wq=bank_reg.W_q is not None)
            with torch.no_grad():
                bank.P.data[:bank_reg.num_experts] = bank_reg.P.data
                for j, r in enumerate(rows):
                    bank.P.data[bank_reg.num_experts + j] = r.to(bank.P.dtype)
            bank.layer_scale = bank_reg.layer_scale
            gates.append(MixGateD2(bank, fam, stock, n_new=len(rows),
                                   jitter_eps=eps))
            site_banks[i] = bank
        else:
            gates.append(d2.GraftGateD2(bank_reg, fam, stock, jitter_eps=eps))
    return gates, site_banks


# ---------------------------------------------------------------------------
# Memory episodes — lab/memory_shaping.py delta contract (substitution shape)
# ---------------------------------------------------------------------------

def build_episodes():
    """2 personal taught facts + the validated digit->word shape episode.

    All deltas are the G3-winning substitution shape (stated BAD -> corrected
    GOOD); home text for consolidation is prompt + correction (ChatML), i.e.
    the taught continuation is what the memory expert must make likely.
    """
    eps = {}
    if ms is not None:
        def ep_from(name, rows, approvals):
            deltas, apps = [], []
            for prompt, stated, corr in rows:
                d = ms.contract_turn(prompt, stated, corr)
                assert d is not None and d.is_substitution, (name, prompt, d)
                deltas.append(d)
            for prompt, styled in approvals:
                apps.append((prompt, styled))
            return ms.Episode(name, deltas, apps)

        eps["pet"] = ep_from("pet_kepler", [
            ("What is my cat's name?", "Your cat is named Orion.",
             "Your cat is named Kepler."),
            ("What is the name of my cat?", "I believe your cat is called Orion.",
             "I believe your cat is called Kepler."),
            ("Remind me — what do we call my cat?", "We call your cat Orion.",
             "We call your cat Kepler."),
        ], [
            ("What kind of pet do I have?", "You have a cat named Kepler."),
            ("Tell me about my pet.", "Your cat Kepler is a good one."),
            ("Does my pet have a name?", "Yes — your cat is named Kepler."),
        ])
        eps["editor"] = ep_from("editor_helix", [
            ("What is my favorite text editor?", "Your favorite editor is vim.",
             "Your favorite editor is helix."),
            ("Which editor do I use?", "You use vim.", "You use helix."),
            ("What editor should you suggest to me?", "I would suggest vim.",
             "I would suggest helix."),
        ], [
            ("Do I like modal editing?", "Yes — you get that in helix."),
            ("Remind me about my editor setup.", "You keep things simple: helix."),
            ("What should I open this file in?", "Open it in helix."),
        ])
        eps["words"] = ms.word_number_episode("math_words", [
            ("What is 2 + 2?", 4), ("What is 5 + 3?", 8),
            ("What is 10 minus 7?", 3), ("What is 1 + 1?", 2),
            ("What is 4 + 4?", 8), ("What is 9 minus 6?", 3)])
        return eps
    raise RuntimeError("memory_shaping unavailable — cannot build episodes")


def episode_home_texts(ep) -> list:
    """ChatML texts of the taught continuations (prompt + correction/styled)."""
    out = []
    for d in ep.deltas:
        out.append((d.prompt, d.correction))
    out.extend(list(ep.approvals))
    return out


# ---------------------------------------------------------------------------
# Data plumbing (door6 conventions: bs x seq windows; train bs4x512, eval bs2)
# ---------------------------------------------------------------------------

def to_batches(ids, label, seq, bs):
    wins = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
    return [(torch.tensor(wins[i:i + bs], dtype=torch.long), label)
            for i in range(0, len(wins) - bs + 1, bs)]


def split_home(tok, text, max_tokens, seq, bs):
    ids = tok.encode(text, add_special_tokens=False)
    split = int(0.6 * len(ids))
    fit_ids = ids[:split][:max_tokens]
    held_ids = ids[split:][:max_tokens]
    return (to_batches(fit_ids, 1, seq, bs),
            to_batches(held_ids, 1, seq, max(1, bs // 2)),
            [torch.tensor(held_ids[i:i + seq], dtype=torch.long)
             for i in range(0, len(held_ids) - seq + 1, seq)])


def chatml_ids(tok, prompt, answer=None):
    msgs = [{"role": "user", "content": prompt}]
    if answer is None:
        txt = tok.apply_chat_template(msgs, tokenize=False,
                                      add_generation_prompt=True)
    else:
        msgs.append({"role": "assistant", "content": answer})
        txt = tok.apply_chat_template(msgs, tokenize=False,
                                      add_generation_prompt=False)
    return tok.encode(txt, add_special_tokens=False)


def episode_batches(tok, episodes, seq, bs, tile_to=4096):
    """Tile episode ChatML texts into train windows (label 1)."""
    ids = []
    for ep in episodes:
        for prompt, ans in episode_home_texts(ep):
            ids.extend(chatml_ids(tok, prompt, ans))
    if not ids:
        return []
    tiles = (tile_to + len(ids) - 1) // len(ids)
    ids = (ids * tiles)[:tile_to]
    return to_batches(ids, 1, seq, bs)


# ---------------------------------------------------------------------------
# Wrap fit: forced-dispatch task loss (door25.train_adapters) + NaN guard
# ---------------------------------------------------------------------------

def fit_wrap(model, blocks, fam, banks_reg, host_i, wrap, home_train,
             base_batches, args, device, steps=None) -> dict:
    """Phase A only (adapters, forced dispatch).  Row recal happens later so
    all rows can get the closing on-trajectory pass at joint time."""
    n_steps = int(steps or args.expert_steps)
    print(f"PHASE A — adapter fit L{host_i} (forced, CE home + out_pen "
          f"{args.out_pen}, lr {args.expert_lr}, {n_steps} steps)", flush=True)
    wrap.to(device)
    stock_fwd = blocks[host_i].experts.forward
    pm = MixMerge(stock_fwd, [wrap], [nn.Parameter(torch.zeros((), device=device))],
                  ["X"], gate_getter=gate_getter_fam(blocks[host_i], fam),
                  num_stock=banks_reg[host_i].num_experts)
    pm.mode = "off"
    blocks[host_i]._plugin_merge = pm

    def make_fwd(pm=pm):
        def fwd(hidden_states, top_k_index, top_k_weights):
            return pm(hidden_states, top_k_index, top_k_weights)
        return fwd

    blocks[host_i].experts.forward = make_fwd()
    handles = [(blocks[host_i], stock_fwd)]
    pa = d25.train_adapters(model, wrap, handles, home_train, base_batches,
                            n_steps, args.expert_lr, device,
                            out_pen=args.out_pen, grad_clip=args.grad_clip,
                            weight_decay=args.weight_decay)
    d1.set_plugin_mode(handles, "off")
    blocks[host_i].experts.forward = stock_fwd
    if hasattr(blocks[host_i], "_plugin_merge"):
        delattr(blocks[host_i], "_plugin_merge")
    pa["finite"] = finite_state(wrap)
    return pa


def quality_reads(model, blocks, fam, banks_reg, site_rows, handles, gates,
                  home, base, device, which_tags):
    """CE home masked / additive-plugged / forced + base CE + membership."""
    out = {}
    home2 = [(t, l) for t, l in home][:max(1, len(home) // 2) or 1] or home
    base2 = [(t, l) for t, l in base][:max(1, len(base) // 2) or 1] or base

    def ce(batches, g, w):
        return run_in_state(model, blocks, fam, g, handles, w,
                            lambda: d1.ce_in_state(model, blocks, batches,
                                                   device, None, None, "off"))

    out["ce_home_masked"] = ce(home2, None, "off")
    out["ce_home_additive"] = ce(home2, gates, "both")
    out["ce_base_masked"] = ce(base2, None, "off")
    out["ce_base_additive"] = ce(base2, gates, "both")

    def forced_ce():
        tot, n = 0.0, 0
        for tokens, _ in home2:
            tokens = tokens.to(device)
            lg = run_in_state(model, blocks, fam, None, handles, "forced",
                              lambda t=tokens: d0.model_logits(model, t))
            tot += d0.next_token_ce(lg, tokens) * tokens.shape[0]
            n += tokens.shape[0]
        return tot / max(1, n)

    out["ce_home_forced"] = forced_ce()

    hits = {i: [0.0] * len(rows) for i, rows in site_rows.items()}
    counts = 0
    for tokens, _ in home2:
        tokens = tokens.to(device)
        run_in_state(model, blocks, fam, gates, handles, "both",
                     lambda t=tokens: d0.model_logits(model, t))
        counts += tokens.numel()
        for i, rows in site_rows.items():
            # gates[i] (the MixGateD2) keeps last_membership — gate_of() would
            # return the stock router again after run_in_state restores it
            g = gates[i]
            m = getattr(g, "last_membership", None)
            if m is None:
                continue
            for j in range(len(rows)):
                hits[i][j] += float(m[:, j].float().sum())
    out["membership_home"] = {str(i): [round(v / max(1, counts), 4)
                                       for v in row]
                              for i, row in hits.items()}
    out["which_tags"] = which_tags
    return out


# ---------------------------------------------------------------------------
# Joint pass: closing row recal (owner_mass 0.85 for memories) + lambda fit
# ---------------------------------------------------------------------------

def closing_row_recal(model, blocks, fam, handles, gates, site_info, batches,
                      device, steps, lr, args) -> dict:
    """On-trajectory row recal with ALL plugs live (additive), per-row
    owner_target (memories 0.85 per g3_owner_mass_fix; specialties 0.35).

    Hidden states are collected ONCE (labels = batch domain id); each row is
    then fit against its own home mask — avoids re-collecting all layers per
    plug-in (the door6 joint_row_recal did one collection per plug)."""
    pairs_raw = run_in_state(
        model, blocks, fam, gates, handles, "both",
        lambda: collect_pairs_fam(
            model, blocks, fam, batches, device, args.max_cal_pairs,
            lambda _i, b: b[1]))
    report = []
    for si in site_info:
        i = si["host_i"]
        h, dom = pairs_raw[i]
        lab = (dom == float(si["home_label"])).float()
        rep = d6.train_rows_one(si["bank"], (h, lab), si["row_idx"], steps, lr,
                                device, reject_w=args.reject_w,
                                mass_w=args.mass_w,
                                owner_target=si["owner_target"])
        rep.update({"host_i": i, "tag": si["tag"],
                    "owner_target": si["owner_target"]})
        report.append(rep)
        print(f"  [recal {si['tag']} L{i} owner {si['owner_target']}] "
              f"p_new_home {rep['first']['p_new_code']:.3f} -> "
              f"{rep['last']['p_new_code']:.3f}  p_new_base "
              f"{rep['first']['p_new_base']:.4f} -> "
              f"{rep['last']['p_new_base']:.4f}", flush=True)
    return {"steps": steps, "lr": lr, "per_site": report}


def fit_lams(model, blocks, fam, lam_params, handles, gates, batches,
             steps, lr, device, grad_clip=1.0) -> dict:
    """Delivery-only fit: lam scalars per plug-in (cores+adapters frozen).
    Objective = CE on the union stream with all plugs live (door6 lamF, with
    base-mix batches included to protect chat-level parity)."""
    saved = [(p, p.requires_grad) for p in model.parameters()]
    d1.freeze_all_except(model, [])
    opt = torch.optim.Adam(list(lam_params), lr=lr)
    hist = []
    t0 = time.time()
    for step in range(steps):
        toks, _ = batches[step % len(batches)]
        toks = toks.to(device)

        def compute():
            logits = d0.model_logits(model, toks)
            v = logits.shape[-1]
            return F.cross_entropy(logits[:, :-1].reshape(-1, v).float(),
                                   toks[:, 1:].reshape(-1))

        ce = run_in_state(model, blocks, fam, gates, handles, "both", compute)
        if not torch.isfinite(ce):
            hist.append({"step": step, "ce": None, "aborted": True})
            print(f"  [lamF {step:4d}] NON-FINITE loss — aborting fit",
                  flush=True)
            for p, flag in saved:
                p.requires_grad_(flag)
            return {"steps": step, "lr": lr, "history": hist,
                    "aborted_nonfinite": True,
                    "wall_s": round(time.time() - t0, 2)}
        opt.zero_grad()
        ce.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(list(lam_params), grad_clip)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            lams = [round(float(p.detach().clamp(min=0)), 4)
                    for p in lam_params]
            hist.append({"step": step, "ce": float(ce.detach()), "lams": lams})
            print(f"  [lamF {step:4d}] ce {float(ce.detach()):.4f}  "
                  f"lams {lams}", flush=True)
    d1.set_plugin_mode(handles, "off")
    for p, flag in saved:
        p.requires_grad_(flag)
    return {"steps": steps, "lr": lr, "history": hist,
            "wall_s": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------
# Chat loop + chat-level metrics
# ---------------------------------------------------------------------------

CHAT_SCRIPT = [
    ("general", "Hello! My name is Alex. What can you help me with today?"),
    ("general", "Tell me a fun fact about octopuses in two sentences."),
    ("code", "Write a Python function that returns the nth Fibonacci number."),
    ("math", "What is the derivative of x^3 * sin(x) with respect to x?"),
    ("mem_pet", "What is my cat's name?"),
    ("mem_edit", "What is my favorite text editor?"),
    ("mem_words", "What is 2 + 2?"),
]

REF_ANSWERS = {
    "code": ("Here is a simple implementation:\n"
             "```python\n"
             "def fib(n):\n"
             "    if n < 2:\n"
             "        return n\n"
             "    a, b = 0, 1\n"
             "    for _ in range(n - 1):\n"
             "        a, b = b, a + b\n"
             "    return b\n"
             "```"),
    "math": ("Using the product rule, d/dx[x^3 sin(x)] = 3x^2 sin(x) + "
             "x^3 cos(x)."),
}

MEM_PROBES = {
    "mem_pet": ("What is my cat's name?", "Kepler", "Orion"),
    "mem_edit": ("What is my favorite text editor?", "helix", "vim"),
    "mem_words": ("What is 2 + 2?", "Four", "4"),
}


def rep_penalize(logits, seen_ids, penalty):
    """CTRL-style repetition penalty over the seen token ids (prompt+reply).

    logits[t] /= penalty if logits[t] > 0 else logits[t] *= penalty for every
    t in seen_ids.  penalty 1.0 (or empty history) is a no-op.  Applied to the
    full context so prompt-echo is damped alongside reply loops.
    """
    if penalty is None or penalty <= 1.0 or not seen_ids:
        return logits
    for t in set(seen_ids):
        if t < 0 or t >= logits.shape[-1]:
            continue
        if logits[t] > 0:
            logits[t] /= penalty
        else:
            logits[t] *= penalty
    return logits


# Degenerate corpus openers of the vk SFT r2 spine (2026-10-15 triage: 0/6
# clean greeting samples under shipped AND gentle decode — every draw opens
# with one of these basins).  Phrase-blocklist: while the generated ids are a
# proper prefix of a banned phrase the next phrase token is masked to -inf,
# so the decoder re-takes the branch token instead of falling into the basin.
# Decode-side only; the clean support exists (e.g. "The capital of France is
# Paris.") but is behind these modes.
BANNED_OPENERS = (
    "i'm not sure if this is",
    "i am not sure if this is",
    "to solve the problem",
    "from the given information",
    "from the given info",
    "given the information",
    "given information",
    "to find the number",
    "to find the",
    "let's write down",
    "let's find",
    "first, we need to",
    "first we need to",
    "we can see that the reaction",
    "we will use the concept",
    "we need to determine",
    "we can use the formula",
    "to determine the",
    "sure, here's the problem",
    "sure, here is the problem",
    "here's the problem",
    "to calculate the",
    "the balanced chemical equation",
    "balanced chemical equation",
)
_OPENER_CACHE = {}
BLOCK_OPENERS_ON = True     # CLI/flag kill-switch (triage A/B arms)
GREET_STEER_ACTIVE = False  # set by the caller when the prompt is a greeting
GREET_FIRST_PIECES = ("Hello", "Hi", "Hey", "Good", "Hello!", "Hi!", "Hey!",
                      "Hello,", "Hi,", "Well", "Thanks", "Thank")
GREET_FIRST_BONUS = 2.5     # nats on the first generated token only
_GREET_ID_CACHE = {}


def greet_steer(logits, out_ids, tok):
    """Greeting-turn decode steer (2026-10-15 greeting bar): the spine's
    greeting support is dominated by solution-textbook basins even with the
    opener blocklist ("Hi there! Amino acid, also known as...").  On
    greeting prompts only: lift greeting openers at step 0, then penalize
    science pieces across the short-reply window so the body stays social.
    Off for every other prompt."""
    if not GREET_STEER_ACTIVE or tok is None:
        return logits
    gs = _GREET_STEER_CACHE.get(id(tok))
    if gs is None:
        try:
            gs = _GREET_STEER_CACHE[id(tok)] = GreetSteer(tok)
        except Exception:  # noqa: BLE001
            return logits
    return gs.apply(logits, out_ids)


class OpenerBlocklist:
    """Token-id prefix blocklist for the degenerate openers above."""

    def __init__(self, tok):
        seqs = []
        for ph in BANNED_OPENERS:
            for var in (ph, ph.capitalize(), " " + ph, " " + ph.capitalize(),
                        ph[0].upper() + ph[1:]):
                try:
                    ids = tuple(int(t) for t in
                                tok.encode(var, add_special_tokens=False))
                except Exception:  # noqa: BLE001
                    ids = ()
                if len(ids) >= 2:
                    seqs.append(ids)
        self.seqs = seqs

    def mask(self, logits, out_ids):
        if not out_ids or len(out_ids) > 16:
            return logits          # openers only — never mid-reply prose
        tail = tuple(out_ids)
        for s in self.seqs:
            n = len(s)
            for k in range(min(n - 1, len(tail)), 0, -1):
                if tail[-k:] == s[:k]:
                    t = s[k]
                    if 0 <= t < logits.shape[-1]:
                        logits[t] = float("-inf")
                    break
        return logits


# Greeting-turn lexicons (the greeting bar: "hi how are you" must produce a
# greeting, not "Hi, I can help you with the position of a molecule...").
# Science pieces are penalized on greeting turns only — the SFT r2 body prior
# is solution-textbook and needs the counterweight.
SCIENCE_PIECES = frozenset("""
molecule molecules molecular reaction reactions solubility electron electrons
energy formula formulas chemical chemistry acid acids mole moles solution
compound compounds ion ions density equation equations equilibrium enthalpy
atom atoms bond bonds voltage circuit protein amino oxidation catalyst solvent
sodium chloride hydrogen oxygen nitrogen carbon dioxide concentration
calculate calculated calculating problem problems data provided following
step synthesis synthesize ionic molarity precipitation spectrometry reactant
reactants stoichiometry molar kj nh3 h2so4 nacl co2 h2o no2 CH4 nh4
""".split())
SOCIAL_PIECES = frozenset("""
hello hi hey good great fine well thanks thank doing today tonight morning
nice happy help ask asking you your day week here too name just really very
pretty much how what about and or so my me am is are doing going been hear
from back home work school life glad lovely wonderful awesome cool chatting
chat talk talking yourself myself ourselves have has had nice
""".split())


class GreetSteer:
    """Greeting-turn decode steering: greeting openers at step 0, then a
    science-piece penalty across the short reply window."""

    def __init__(self, tok):
        def ids_for(words):
            out = set()
            for w in words:
                for var in (w, w.capitalize(), " " + w, " " + w.capitalize(),
                            w + "s", " " + w + "s"):
                    try:
                        out.update(int(t) for t in
                                   tok.encode(var, add_special_tokens=False))
                    except Exception:  # noqa: BLE001
                        pass
            return out
        self.openers = ids_for(GREET_FIRST_PIECES)
        self.science = ids_for(SCIENCE_PIECES)
        self.social = ids_for(SOCIAL_PIECES)

    def apply(self, logits, out_ids):
        if not out_ids:
            for t in self.openers:
                if 0 <= t < logits.shape[-1]:
                    logits[t] += GREET_FIRST_BONUS
            return logits
        if len(out_ids) > 48:
            return logits
        for t in self.science:
            if 0 <= t < logits.shape[-1]:
                logits[t] -= 6.0
        for t in self.social:
            if 0 <= t < logits.shape[-1]:
                logits[t] += 0.6
        return logits


_GREET_STEER_CACHE = {}


def block_openers(logits, out_ids, tok):
    """Mask the next token of any banned opener the reply is currently
    spelling.  No-op without a tokenizer (or with an empty reply)."""
    if not BLOCK_OPENERS_ON or tok is None or not out_ids or not BANNED_OPENERS:
        return logits
    key = id(tok)
    bl = _OPENER_CACHE.get(key)
    if bl is None:
        try:
            bl = _OPENER_CACHE[key] = OpenerBlocklist(tok)
        except Exception:  # noqa: BLE001
            return logits
    return bl.mask(logits, out_ids)


@torch.no_grad()
def gen_reply(tok, logits_fn, messages, max_new_tokens, device, sample=False,
              temperature=0.8, top_p=0.95, seed=0, rep_penalty=1.0):
    """Greedy / temperature sampling via incremental decode on logits_fn."""
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    except Exception:
        # vk chat tokenizer ships no chat_template — ChatML verbatim
        # (VkKVDecoder.generate / vk.chatml: talkability measured in this
        # format); without this fallback every non-KV decode dies at prompt
        # build and the taught-token bonus never reaches the reply.
        im_s, im_e = "<|im_start|>", "<|im_end|>"
        prompt = "".join(f"{im_s}{m['role']}\n{m['content']}{im_e}\n"
                         for m in messages) + f"{im_s}assistant\n"
    ids = tok.encode(prompt, add_special_tokens=False)
    out_ids = []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(max_new_tokens):
        x = torch.tensor([ids + out_ids], dtype=torch.long, device=device)
        logits = logits_fn(x)[0, -1].float().cpu()
        logits = rep_penalize(logits, ids + out_ids, rep_penalty)
        logits = block_openers(logits, out_ids, tok)
        logits = greet_steer(logits, out_ids, tok)
        if sample:
            probs = F.softmax(logits / max(1e-5, temperature), dim=-1)
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            cum = sorted_p.cumsum(0)
            cut = int((cum > top_p).nonzero(as_tuple=True)[0][:1].item()) + 1 \
                if (cum > top_p).any() else len(sorted_p)
            p = sorted_p[:cut] / sorted_p[:cut].sum()
            idx = int(sorted_i[torch.multinomial(p, 1, generator=g)])
        else:
            idx = int(logits.argmax())
        out_ids.append(idx)
        if idx == tok.eos_token_id or idx == tok.pad_token_id:
            break
    text = tok.decode(out_ids, skip_special_tokens=True)
    return text, out_ids


def run_conversation(tok, logits_fn, device, sample, max_new_tokens, seed=0):
    turns = []
    messages = []
    for kind, user in CHAT_SCRIPT:
        messages.append({"role": "user", "content": user})
        reply, ids = gen_reply(tok, logits_fn, messages, max_new_tokens, device,
                               sample=sample, seed=seed)
        messages.append({"role": "assistant", "content": reply})
        turns.append({"kind": kind, "user": user, "assistant": reply,
                      "n_new_tokens": len(ids)})
    return turns


@torch.no_grad()
def span_ce(logits_fn, tok, prompt, answer, device):
    """CE over the answer span of ChatML(prompt + assistant answer)."""
    ids = chatml_ids(tok, prompt, answer)
    pre = chatml_ids(tok, prompt, None)
    # boundary may re-tokenize: trust the prefix only if it matches
    if ids[:len(pre)] == pre:
        start = len(pre) - 1        # first answer position predicts from here
    else:
        start = max(1, len(pre) - 2)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = logits_fn(x)[0].float()
    tgt = torch.tensor(ids[1:], dtype=torch.long, device=logits.device)
    logp = F.log_softmax(logits[:-1], dim=-1)
    nll = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    seg = nll[start - 1:] if start - 1 < nll.shape[0] else nll[-1:]
    return float(seg.mean()), int(seg.shape[0])


@torch.no_grad()
def string_logprob(logits_fn, tok, prompt, cont, device):
    """Sum log-prob of continuation string tokens after the chat prompt."""
    pre = chatml_ids(tok, prompt, None)
    cont_ids = tok.encode(cont, add_special_tokens=False)
    x = torch.tensor([pre + cont_ids], dtype=torch.long, device=device)
    logits = logits_fn(x)[0].float()
    logp = F.log_softmax(logits, dim=-1)
    total, n = 0.0, 0
    for k, tid in enumerate(cont_ids):
        pos = len(pre) - 1 + k
        if pos < 0 or pos >= logp.shape[0]:
            continue
        total += float(logp[pos, tid])
        n += 1
    return total, n


@torch.no_grad()
def chat_parity(tok, fn_bare, fn_graft, device):
    """Bare vs grafted next-token distributions over the chat contexts."""
    sum_kl = sum_cos = 0.0
    n = 0
    cos_min = 1.0
    ce_b = ce_g = 0.0
    ctxs = []
    for kind, user in CHAT_SCRIPT:
        ctxs.append(tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True))
    for text in ctxs:
        ids = tok.encode(text, add_special_tokens=False)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        lb = fn_bare(x)[0, :-1].float()
        lg = fn_graft(x)[0, :-1].float()
        pb = F.log_softmax(lb, dim=-1)
        pg = F.log_softmax(lg, dim=-1)
        kl = (pb.exp() * (pb - pg)).sum(-1)
        cos = F.cosine_similarity(lb, lg, dim=-1)
        sum_kl += float(kl.sum())
        sum_cos += float(cos.sum())
        cos_min = min(cos_min, float(cos.min()))
        n += kl.shape[0]
        tgt = torch.tensor(ids[1:], dtype=torch.long, device=lb.device)
        ce_b += float(F.cross_entropy(lb, tgt))
        ce_g += float(F.cross_entropy(lg, tgt))
    nctx = len(ctxs)
    return {"n_tokens": n, "mean_kl_nats": round(sum_kl / max(1, n), 5),
            "mean_logit_cosine": round(sum_cos / max(1, n), 5),
            "min_logit_cosine": round(cos_min, 5),
            "ce_bare": round(ce_b / nctx, 4), "ce_grafted": round(ce_g / nctx, 4),
            "ce_delta": round((ce_g - ce_b) / nctx, 4)}


# ---------------------------------------------------------------------------
# Phase: fit
# ---------------------------------------------------------------------------

def parse_wraps(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        name, kind, layer, expert, home = bits[:5]
        steps = int(bits[5]) if len(bits) > 5 else None
        out.append((name, kind, int(layer), int(expert), home, steps))
    return out


def run_fit(args) -> dict:
    t0 = time.time()
    out = {"gate": "grafted-spine-v0-fit", "format": FORMAT, "phase": "start",
           "wraps": args.wraps, "pid": os.getpid()}
    dump_json(out, args.out)
    model, tok = d2.load_donor(args.model, args.device, args.dtype)
    blocks = d2.find_moe_blocks(model)
    fam = d2.family_for(d2.detect_family(model), blocks)
    banks_reg, reg_meta = d0.load_registry(args.registry)
    if len(banks_reg) != len(blocks):
        raise RuntimeError(f"registry {len(banks_reg)} layers vs model {len(blocks)}")
    d_model = banks_reg[0].d_model
    print(f"host: {len(blocks)} layers, d_model {d_model}, family {fam.name}",
          flush=True)

    domains = d0.load_domain_texts(args.domains_dir)
    seq, bs = args.seq_len, args.batch_size
    if seq * bs in HOLEY_BT:
        raise RuntimeError(f"train B*T {seq * bs} is a gfx906 hole")
    home_streams = {}
    for key in ("code", "math"):
        if key in domains:
            tr, he, probes = split_home(tok, domains[key], args.max_tokens,
                                        seq, bs)
            home_streams[key] = {"train": tr, "held": he}
    base_ids = []
    for name, text in domains.items():
        if name in ("code", "math"):
            continue
        base_ids.extend(tok.encode(text, add_special_tokens=False))
    base_ids = base_ids[:args.max_tokens * 2]
    base_batches = to_batches(base_ids, 0, seq, bs)
    episodes = build_episodes()
    print("episodes:", {k: ms.shape_report([v]) for k, v in episodes.items()}
          if ms else "n/a", flush=True)

    plugins = {}
    for name, kind, layer, expert, home, wrap_steps in parse_wraps(args.wraps):
        plugin_path = os.path.join(args.plugins_dir, f"{name}_plugin.pt")
        if args.skip_done and os.path.exists(plugin_path):
            wrap_chk, row_chk = d6.load_wrap_plugin(plugin_path)
            if d6.wrap_is_finite(wrap_chk, row_chk):
                print(f"[{name}] plugin exists and finite — skip fit",
                      flush=True)
                plugins[name] = plugin_path
                out.setdefault("fitted", {})[name] = {"skipped": True,
                                                      "plugin": plugin_path}
                dump_json(out, args.out)
                continue
        w_gu, w_down = d25.extract_donor_core(blocks, layer, expert)
        wrap = d25.WrappedForeignExpert(d_model, args.d_canon, w_gu, w_down,
                                        seed=args.seed)
        if kind in ("code", "math"):
            home_train = home_streams[home]["train"]
            home_eval = home_streams[home]["train"]
            base_train = base_batches
        else:
            # memory: bs1 x 512 (B*T=512, gfx906-safe) on tiled episode ChatML
            home_train = episode_batches(tok, [episodes[home]], seq, 1)
            home_eval = home_train
            base_train = base_batches
        if not home_train:
            raise RuntimeError(f"[{name}] empty home stream")
        print(f"== [{name}] wrap core L{layer} e{expert} "
              f"({wrap.core_param_count()} frozen, "
              f"{wrap.adapter_param_count()} adapters) home={home} "
              f"({sum(t.numel() for t, _ in home_train)} tok) ==", flush=True)
        pa = fit_wrap(model, blocks, fam, banks_reg, layer, wrap, home_train,
                      base_train, args, args.device, steps=wrap_steps)
        if not pa.get("finite", True):
            # FAILURES door4b/door5: lr 3e-3 wraps can NaN — skip this wrap and
            # keep fitting the rest of the lane instead of dying here.
            print(f"[{name}] NON-FINITE adapters after fit — skip save "
                  f"(retry at lr 1e-3)", flush=True)
            out.setdefault("fitted", {})[name] = {
                "error": "non_finite_adapters",
                "phase_a": {k: pa.get(k) for k in ("steps", "lr", "finite")},
                "ce_hist_tail": pa.get("history", [])[-3:],
            }
            out["phase"] = f"failed_{name}"
            dump_json(out, args.out)
            # fit_wrap already restored stock forward / dropped _plugin_merge
            del wrap
            torch.cuda.empty_cache()
            continue
        # solo row: prototype init + recal (owner per kind; closing pass later)
        site_rows = {layer: [torch.zeros(d_model)]}
        gates, site_banks = build_gates(blocks, banks_reg, fam, site_rows)
        handles = install_mix(blocks, fam, {
            layer: [{"expert": wrap, "lam": nn.Parameter(torch.ones((), device=args.device)),
                     "tag": name, "num_stock": banks_reg[layer].num_experts}]},
            mode="off")
        pairs = run_in_state(
            model, blocks, fam, gates, handles, "off",
            lambda: collect_pairs_fam(model, blocks, fam,
                                      home_train + base_train, args.device,
                                      args.max_cal_pairs,
                                      lambda _i, b: 1 if b[1] == 1 else 0))
        h, lab = pairs[layer]
        code_h = h[lab > 0.5]
        bank = site_banks[layer]
        row_idx = bank.num_experts - 1
        with torch.no_grad():
            if code_h.shape[0] > 0:
                bank.P.data[row_idx] = (code_h.mean(dim=0)
                                        * math.sqrt(bank.d_r))
        owner = (BAR_OWNER_SPEC if kind in ("code", "math") else BAR_OWNER_MEM)
        rep1 = d6.train_rows_one(bank, pairs[layer], row_idx, args.row_steps,
                                 args.row_lr, args.device,
                                 reject_w=args.reject_w, mass_w=args.mass_w,
                                 owner_target=owner)
        row = bank.P.data[row_idx].float().cpu().clone()
        q = quality_reads(model, blocks, fam, banks_reg,
                          {layer: [row]}, handles, gates,
                          home_eval, base_train, args.device, [name])
        d1.set_plugin_mode(handles, "off")
        for blk, sf in handles:
            blk.experts.forward = sf
            if hasattr(blk, "_plugin_merge"):
                delattr(blk, "_plugin_merge")
        os.makedirs(args.plugins_dir, exist_ok=True)
        torch.save({
            "format": d6.FORMAT,
            "wrap": wrap.state_dict(),
            "wrap_spec": wrap.wrap_spec(),
            "row": row,
            "meta": {"new_idx": row_idx, "host_layer": layer, "tag": name,
                     "kind": kind, "home": home, "donor_layer": layer,
                     "donor_expert": expert, "registry": args.registry,
                     "owner_target": owner},
        }, plugin_path)
        plugins[name] = plugin_path
        out.setdefault("fitted", {})[name] = {
            "plugin": plugin_path, "host_layer": layer, "donor": [layer, expert],
            "kind": kind, "home": home, "phase_a": {k: pa[k] for k in
                                                    ("steps", "lr", "out_pen",
                                                     "finite", "wall_s")},
            "ce_hist_tail": pa["history"][-3:], "row_recal": rep1, "quality": q,
        }
        out["phase"] = f"fitted_{name}"
        dump_json(out, args.out)
        print(f"== [{name}] done: CE home {q['ce_home_masked']:.3f} -> "
              f"{q['ce_home_additive']:.3f} (forced {q['ce_home_forced']:.3f}); "
              f"row p_home {rep1['last']['p_new_code']:.3f} ==", flush=True)
        del wrap
        torch.cuda.empty_cache()
    out["plugins"] = plugins
    out["phase"] = "fit_done"
    out["wall_s"] = round(time.time() - t0, 2)
    dump_json(out, args.out)
    return out


# ---------------------------------------------------------------------------
# Phase: assemble (joint recal + lam fit + chat test)
# ---------------------------------------------------------------------------

def run_assemble(args) -> dict:
    t0 = time.time()
    prev = {}
    if args.out and os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
        except Exception:
            prev = {}
    out = {"gate": "grafted-spine-v0", "format": FORMAT, "phase": "start",
           "pid": os.getpid()}
    for k in ("fit", "joint", "chat", "verdict"):
        if k in prev and prev[k]:
            out[k] = prev[k]
    dump_json(out, args.out)

    model, tok = d2.load_donor(args.model, args.device, args.dtype)
    blocks = d2.find_moe_blocks(model)
    fam = d2.family_for(d2.detect_family(model), blocks)
    banks_reg, reg_meta = d0.load_registry(args.registry)
    d_model = banks_reg[0].d_model
    n_layers = len(blocks)
    print(f"host: {n_layers} layers, d_model {d_model}, family {fam.name}",
          flush=True)
    domains = d0.load_domain_texts(args.domains_dir)
    seq = args.seq_len

    # ---- plugins ----
    wrap_specs = parse_wraps(args.wraps)
    wraps, rows, metas = {}, {}, {}
    for name, kind, layer, expert, home, _wrap_steps in wrap_specs:
        path = os.path.join(args.plugins_dir, f"{name}_plugin.pt")
        if not os.path.exists(path):
            raise RuntimeError(f"missing plugin {path}")
        wrap, row, meta, spec = d6.load_wrap_plugin(path)
        if not d6.wrap_is_finite(wrap, row):
            raise RuntimeError(f"non-finite plugin {path}")
        wraps[name], rows[name], metas[name] = wrap, row, meta
        print(f"[{name}] plugin host L{meta['host_layer']} kind {meta['kind']}",
              flush=True)
    out["fit"] = {"plugins": {n: metas[n] for n in wraps},
                  "wrap_specs": {n: wraps[n].wrap_spec() for n in wraps}}
    dump_json(out, args.out)

    # ---- joint install ----
    site_rows, site_entries = {}, {}
    lams = []
    for name, kind, layer, expert, home, _wrap_steps in wrap_specs:
        lam = nn.Parameter(torch.tensor(float(args.lam_init),
                                        device=args.device))
        lams.append(lam)
        site_rows.setdefault(layer, []).append(rows[name].to(args.device))
        site_entries.setdefault(layer, []).append({
            "expert": wraps[name].to(args.device), "lam": lam, "tag": name,
            "num_stock": banks_reg[layer].num_experts})
    gates, site_banks = build_gates(blocks, banks_reg, fam, site_rows)
    handles = install_mix(blocks, fam, site_entries, mode="off")
    row_idx = {}
    for name, kind, layer, expert, home, _wrap_steps in wrap_specs:
        bank = site_banks[layer]
        tags_here = [en["tag"] for en in site_entries[layer]]
        row_idx[name] = bank.num_experts - len(tags_here) + tags_here.index(name)
    print("installed rows:", row_idx, flush=True)
    out["joint"] = {"row_idx": row_idx, "lam_init": args.lam_init,
                    "lams": {n: float(lams[i].detach()) for i, (n, *_)
                             in enumerate(wrap_specs)}}
    dump_json(out, args.out)

    # ---- data for joint pass ----
    code_train, _, _ = split_home(tok, domains["code"], args.max_tokens, seq,
                                  args.batch_size)
    math_train, _, _ = split_home(tok, domains["math"], args.max_tokens, seq,
                                  args.batch_size)
    episodes = build_episodes()
    mem_streams = {}
    for key in ("pet", "editor", "words"):
        mem_streams[key] = episode_batches(tok, [episodes[key]], seq,
                                           args.batch_size)
    base_ids = []
    for name, text in domains.items():
        if name in ("code", "math"):
            continue
        base_ids.extend(tok.encode(text, add_special_tokens=False))
    base_ids = base_ids[:args.max_tokens * 2]
    base_batches = to_batches(base_ids, 0, seq, args.batch_size)

    joint_batches = ([(t, 1) for t, _ in code_train]
                     + [(t, 2) for t, _ in math_train]
                     + [(t, 3) for t, _ in mem_streams["pet"]]
                     + [(t, 4) for t, _ in mem_streams["editor"]]
                     + [(t, 5) for t, _ in mem_streams["words"]]
                     + [(t, 0) for t, _ in base_batches])

    def label_of_factory(home_label):
        return lambda _i, b: 1 if b[1] == home_label else 0

    site_info = []
    home_label = {"code": 1, "math": 2, "pet": 3, "editor": 4, "words": 5}
    for name, kind, layer, expert, home, _wrap_steps in wrap_specs:
        key = home if home in home_label else name.replace("mem_", "")
        site_info.append({
            "host_i": layer, "bank": site_banks[layer], "row_idx": row_idx[name],
            "home_label": home_label.get(key, 1), "tag": name,
            "owner_target": (BAR_OWNER_SPEC if kind in ("code", "math")
                             else BAR_OWNER_MEM)})
    frozen = d6.freeze_wraps(list(wraps.values()))
    js_path = os.path.join(args.plugins_dir, "joint_state.pt")
    prev_joint = prev.get("joint") or {}
    if (os.path.exists(js_path) and prev_joint.get("lam_fit")
            and not prev_joint["lam_fit"].get("aborted_nonfinite")
            and prev_joint.get("lams_final")):
        js = torch.load(js_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for name, r in js["rows"].items():
                lay = dict((n, l) for n, k, l, e, h, s in wrap_specs)[name]
                site_banks[lay].P.data[row_idx[name]] = r.to(
                    site_banks[lay].P.device).to(site_banks[lay].P.dtype)
            for lam, v in zip(lams, js["lams"]):
                lam.copy_(torch.tensor(float(v)))
        out["joint"] = {**prev_joint, "row_idx": row_idx}
        print("== joint pass restored from joint_state.pt (recal+lam "
              "skipped) ==", flush=True)
        dump_json(out, args.out)
    else:
        print("== closing row recal (all plugs live, memories owner 0.85) ==",
              flush=True)
        recal = closing_row_recal(model, blocks, fam, handles, gates, site_info,
                                  joint_batches, args.device, args.recal_steps,
                                  args.recal_lr, args)
        out["joint"]["row_recal"] = recal
        dump_json(out, args.out)

        print("== lambda fit (additive contract, lam per expert per layer) ==",
              flush=True)
        lam_fit = fit_lams(model, blocks, fam, lams, handles, gates,
                           joint_batches, args.lam_steps, args.lam_lr,
                           args.device)
        out["joint"]["lam_fit"] = lam_fit
        out["joint"]["lams_final"] = [round(float(p.detach().clamp(min=0)), 4)
                                      for p in lams]
        torch.save({
            "rows": {n: site_banks[
                dict((nn, l) for nn, k, l, e, h, s in wrap_specs)[n]]
                .P.data[row_idx[n]].cpu() for n in row_idx},
            "lams": [float(p.detach().clamp(min=0)) for p in lams],
        }, js_path)
        out["phase"] = "joint_done"
        dump_json(out, args.out)
    for en_l in site_entries.values():
        for en in en_l:
            # keep a tensor lam: MixMerge calls .clamp(min=0) on it
            en["lam"] = torch.tensor(float(en["lam"].detach().clamp(min=0)),
                                     device=args.device)
    # rebuild merges with frozen lam tensors (clean generation state)
    handles = install_mix(blocks, fam, site_entries, mode="off")
    dump_json(out, args.out)
    d6.restore_grad_flags(frozen)

    # ---- quality + parity reads at chat level ----
    print("== chat-level measurement ==", flush=True)
    eval_bs = args.eval_batch_size
    code_ev, _, _ = split_home(tok, domains["code"], args.max_tokens, seq,
                               eval_bs)
    math_ev, _, _ = split_home(tok, domains["math"], args.max_tokens, seq,
                               eval_bs)
    base_ev = to_batches(base_ids, 0, seq, eval_bs)
    mem_ev = {k: episode_batches(tok, [episodes[k]], seq, eval_bs)
              for k in ("pet", "editor", "words")}

    def fn_bare(x):
        return run_in_state(model, blocks, fam, None, None, "off",
                            lambda: d0.model_logits(model, x.to(args.device)))

    def fn_graft(x):
        return run_in_state(model, blocks, fam, gates, handles, "both",
                            lambda: d0.model_logits(model, x.to(args.device)))

    parity = chat_parity(tok, fn_bare, fn_graft, args.device)
    print(f"parity: cos {parity['mean_logit_cosine']} KL "
          f"{parity['mean_kl_nats']} dCE {parity['ce_delta']}", flush=True)

    # specialty visibility: CE of reference answer continuation
    spec_vis = {}
    for key in ("code", "math"):
        b, nb = span_ce(fn_bare, tok, CHAT_SCRIPT[
            [k for k, u in CHAT_SCRIPT].index(key)][1], REF_ANSWERS[key],
            args.device)
        g, ng = span_ce(fn_graft, tok, CHAT_SCRIPT[
            [k for k, u in CHAT_SCRIPT].index(key)][1], REF_ANSWERS[key],
            args.device)
        spec_vis[key] = {"ce_ref_bare": round(b, 4), "ce_ref_grafted": round(g, 4),
                         "delta": round(g - b, 4), "n_answer_tokens": ng}

    # memory visibility: taught-token logprob margins
    mem_vis = {}
    for key, (prompt, good, bad) in MEM_PROBES.items():
        gb, nb = string_logprob(fn_bare, tok, prompt, good, args.device)
        bb, _ = string_logprob(fn_bare, tok, prompt, bad, args.device)
        gg, ng = string_logprob(fn_graft, tok, prompt, good, args.device)
        bg, _ = string_logprob(fn_graft, tok, prompt, bad, args.device)
        mem_vis[key] = {"prompt": prompt, "good": good, "bad": bad,
                        "margin_bare": round(gb - bb, 4),
                        "margin_grafted": round(gg - bg, 4),
                        "margin_gain": round((gg - bg) - (gb - bb), 4),
                        "logp_good_bare": round(gb, 4),
                        "logp_good_grafted": round(gg, 4)}

    # CE on the episodic homes + specialty homes (delivered vs masked)
    home_ce = {}
    for tag, batches in [("code", code_ev), ("math", math_ev),
                         ("mem_pet", mem_ev["pet"]),
                         ("mem_edit", mem_ev["editor"]),
                         ("mem_words", mem_ev["words"]),
                         ("base", base_ev)]:
        bm = run_in_state(model, blocks, fam, None, None, "off",
                          lambda bs_=batches: d1.ce_in_state(model, blocks,
                                                             bs_, args.device,
                                                             None, None, "off"))
        bg = run_in_state(model, blocks, fam, gates, handles, "both",
                          lambda bs_=batches: d1.ce_in_state(model, blocks,
                                                             bs_, args.device,
                                                             None, None, "off"))
        home_ce[tag] = {"masked": round(bm, 4), "grafted": round(bg, 4),
                        "delta": round(bg - bm, 4)}
        print(f"  CE {tag:9s} masked {bm:.4f} -> grafted {bg:.4f}", flush=True)

    out["metrics"] = {"parity_chat": parity, "specialty_ref_ce": spec_vis,
                      "memory_margins": mem_vis, "home_ce": home_ce}
    # membership of each plug-in row (gates[i] holds last_membership after the
    # forward; run_in_state restores the stock router, so read gates not blk)
    def memb(batches):
        counts = 0
        hits = {n: 0.0 for n in wraps}
        for tokens, _ in batches:
            tokens = tokens.to(args.device)
            run_in_state(model, blocks, fam, gates, handles, "both",
                         lambda t=tokens: d0.model_logits(model, t))
            counts += tokens.numel()
            for name, kind, layer, expert, home, _s in wrap_specs:
                g = gates[layer]
                m = getattr(g, "last_membership", None)
                if m is None:
                    continue
                tags_here = [en["tag"] for en in site_entries[layer]]
                j = tags_here.index(name)
                hits[name] += float(m[:, j].float().sum())
        return {n: round(v / max(1, counts), 4) for n, v in hits.items()}

    out["metrics"]["membership_home"] = {
        "code": memb(code_ev), "math": memb(math_ev),
        "mem_pet": memb(mem_ev["pet"]), "mem_edit": memb(mem_ev["editor"]),
        "mem_words": memb(mem_ev["words"]), "base": memb(base_ev)}
    print("membership home:", out["metrics"]["membership_home"], flush=True)
    out["phase"] = "metrics_done"
    dump_json(out, args.out)

    # ---- the chat test itself ----
    print("== chat test: bare vs grafted ==", flush=True)
    chat = {}
    for cfg, fn in (("bare", fn_bare), ("grafted", fn_graft)):
        chat[cfg] = {
            "greedy": run_conversation(tok, fn, args.device, False,
                                       args.max_new_tokens),
            "temp": run_conversation(tok, fn, args.device, True,
                                     args.max_new_tokens, seed=args.seed),
        }
        print(f"[{cfg}] turns greedy: "
              + " | ".join(t["assistant"][:60].replace("\n", " ")
                           for t in chat[cfg]["greedy"]), flush=True)
    out["chat"] = chat

    # reply-level memory / specialty flags (substring, cheap + honest)
    def flags(turns):
        f = {}
        for t in turns:
            a = t["assistant"].lower()
            if t["kind"] == "mem_pet":
                f["mem_pet_kepler"] = "kepler" in a
                f["mem_pet_orion"] = "orion" in a
            elif t["kind"] == "mem_edit":
                f["mem_edit_helix"] = "helix" in a
                f["mem_edit_vim"] = "vim" in a
            elif t["kind"] == "mem_words":
                f["mem_words_four"] = "four" in a
                f["mem_words_digit"] = ("4" in a and "four" not in a)
            elif t["kind"] == "code":
                f["code_def"] = ("def " in a) or ("fib" in a)
            elif t["kind"] == "math":
                f["math_cos"] = "cos" in a
        return f

    out["reply_flags"] = {cfg: {"greedy": flags(chat[cfg]["greedy"]),
                                "temp": flags(chat[cfg]["temp"])}
                          for cfg in chat}

    # ---- verdict (chat level) ----
    mem_gain = {k: v["margin_gain"] for k, v in mem_vis.items()}
    n_mem_up = sum(1 for v in mem_gain.values() if v > 0)
    spec_up = sum(1 for k in ("code", "math")
                  if spec_vis[k]["delta"] < 0)
    converse_ok = (parity["mean_logit_cosine"] >= BAR_PARITY_COS
                   and all(len(t["assistant"].strip()) > 0
                           for t in chat["bare"]["greedy"]
                           + chat["grafted"]["greedy"]))
    v = {
        "converse_parity_pass": bool(converse_ok),
        "parity_cos": parity["mean_logit_cosine"],
        "specialties_visible": spec_up >= 2,
        "specialty_ref_ce_improved": spec_up,
        "memories_visible_margins": n_mem_up >= 2,
        "memories_visible_replies": bool(
            out["reply_flags"]["grafted"]["greedy"].get("mem_pet_kepler")
            and out["reply_flags"]["grafted"]["greedy"].get("mem_edit_helix")),
        "n_plugs": len(wrap_specs),
        "lams": out["joint"].get("lams_final"),
    }
    v["verdict_text"] = (
        "V0 ASSEMBLED — chat-level parity "
        + ("PASS" if converse_ok else "FAIL")
        + f"; specialties visible {spec_up}/2, memory margins up {n_mem_up}/3"
        + (", memories surface in replies"
           if v["memories_visible_replies"] else
           ", memories do not surface in replies"))
    out["verdict"] = v
    out["phase"] = "done"
    out["wall_s"] = round(time.time() - t0, 2)
    dump_json(out, args.out)
    print("verdict:", v["verdict_text"], flush=True)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("fit", "assemble", "all"), default="all")
    ap.add_argument("--model", default="/data/tliao/Phi-mini-MoE-instruct")
    ap.add_argument("--registry",
                    default="/data/tliao/graft_d2_out/registry_phimoe_f620b32c.pt")
    ap.add_argument("--domains-dir", default="/data/tliao/graft_d2_slice")
    ap.add_argument("--plugins-dir", default="/data/tliao/grafted_spine_v0_out")
    ap.add_argument("--out",
                    default="/data/tliao/grafted_spine_v0_out/v0_results.json")
    ap.add_argument("--wraps",
                    default=",".join(f"{n}:{k}:{l}:{e}:{h}"
                                     for n, k, l, e, h in V0_WRAPS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--d-canon", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="train B*T=2048 (door25/door6 verified shape)")
    ap.add_argument("--eval-batch-size", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--expert-steps", type=int, default=250)
    ap.add_argument("--expert-lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--out-pen", type=float, default=300.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--row-steps", type=int, default=400)
    ap.add_argument("--row-lr", type=float, default=0.05)
    ap.add_argument("--reject-w", type=float, default=10.0)
    ap.add_argument("--mass-w", type=float, default=2.0)
    ap.add_argument("--max-cal-pairs", type=int, default=4096)
    ap.add_argument("--recal-steps", type=int, default=150)
    ap.add_argument("--recal-lr", type=float, default=0.05)
    ap.add_argument("--lam-steps", type=int, default=80)
    ap.add_argument("--lam-lr", type=float, default=0.05)
    ap.add_argument("--lam-init", type=float, default=0.5)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.phase in ("fit", "all"):
        run_fit(args)
    if args.phase in ("assemble", "all"):
        run_assemble(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
