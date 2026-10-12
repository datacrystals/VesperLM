#!/usr/bin/env python3
"""Door 8 — bridge width vs GENERAL-LM recovery (the all-grafted-bank bar).

The endgame is "our own spine + a MoE whose ENTIRE expert bank is grafted
donor experts".  Door 2.5 proved a frozen foreign core can carry a specialty
*delta* behind a 64-dim bottleneck (code CE 1.072 masked -> 0.439 plugged).
Unknown: when a wrapped core is asked to do the heavy BORING general FFN
lifting, does the bridge bottleneck kill it?

Q (this door): how wide must the wrap bridge be for a grafted generalist core
to recover the CE cost of REMOVING a native expert on general LM text?

Protocol:
  1. Generalist donor core: the highest-mushiness (domain-generalist) Phi
     mini-MoE expert by D2 RouteStats profiling on the graft_d2 slices
     (normalized entropy of the domain-rate profile; d0.RouteStats
     .home_and_prototypes).  Frozen, wrapped in the exact door25 geometry
     with adapter rank sweep d_canon in {64, 256, 512, 1024}:

       x(2048) --host_in--> c(d_canon) --adapter_up--> u(4096)
         --CORE (frozen SwiGLU)--> v(4096)
       --adapter_down--> c'(d_canon) --host_out--> delta(2048)

  2. Load-bearing test on the OLMoE host.  General text = the base-domain
     slices of graft_d0_slice (web/chat/science/math) + a wikitext-style
     slice, 80/20 train/held-out per source, val_mixed as a third eval set.
     (a) stock host CE;  (b) native expert N masked — its routed slots
     zeroed in the stock merge (norm_topk_prob=false, so this is exact
     capacity removal, no renormalization) — CE cost of removing native FFN
     capacity;  (c) grafted core compensating for N's removal: N stays
     masked, the wrap delivers ADDITIVELY at the site where N lives
     (y = y_masked + lam * e_wrap(h), the door6 additive delivery with
     membership m == 1 — "the site where N lives" alternative to the door25
     site-with-membership).  N = the most load-bearing single native expert
     on general probe text (max mask-CE-cost over each layer's top-routed
     expert).

     recovery%(w) = (CE_masked - CE_masked+graft(w)) / (CE_masked - CE_stock)

  3. Bars (v2 feasibility line):
       >= 80% recovery at rank <= 512  -> FEASIBLE (all-grafted banks live)
       50-80%                          -> MARGINAL (wider / multi-core)
       < 50%                           -> the bottleneck is fatal at thin
                                          ranks; v2 needs a different bridge
     The SHAPE of the curve is reported either way — it is the design input
     for the v2 format.

  4. Secondary: specialty re-check at wide rank — the door25 code fit at
     d_canon 64 vs 512 (does code quality hold/improve with width? — connects
     to the held-out memorization gap in the HANDOFF multi-slice note).

Fork of lab/door25_wrap_expert.py (geometry + recipe + data plumbing); d0/d1
/d2/door25 are imported UNMODIFIED.  Delivery here is forced-additive at the
masked site (no passport row recal — the question is bridge capacity, not
routing; the door25 site-with-membership arm is the follow-up).

Usage:
  python3 lab/door8_bridge_width.py --synthetic
  python3 lab/door8_bridge_width.py \
      --model /data/tliao/OLMoE-1B-7B-0924 \
      --donor /data/tliao/Phi-mini-MoE-instruct \
      --registry /data/tliao/graft_d0_out/registry_olmoe_6d84c485.pt \
      --domains-dir /data/tliao/graft_d0_slice \
      --donor-domains-dir /data/tliao/graft_d2_slice \
      --wikitext /data/tliao/graft_door8_out/wikitext_train.txt \
      --widths 64,512,256,1024 --device cuda --dtype float16 \
      --out /data/tliao/graft_door8_out/door8_width.json

gfx906/MI50 notes (carried from door25): trunk BACKWARD has size holes
(B*T = 256/768/1024/4096 die; 2048 verified) — training is bs4 x seq512 =
2048 tokens; eval is bs2 x seq512 (the ops rule).  Thin-wrap recipe is
lr 3e-3 / wd 0 / clip 1.0 (wd 0.01 starves the wrap); a NaN guard aborts an
arm rather than fitting a poisoned wrap, with one documented fallback at
half lr.  Generalist arms use out_pen 0 (pure task loss — the wrap is
supposed to carry norm here); the secondary specialty arm keeps door25's
out_pen 300 for comparability.
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
import door25_wrap_expert as d25  # noqa: E402

# v2 feasibility bars (the task's make-or-break line)
BAR_FEASIBLE = 0.80
BAR_MARGINAL = 0.50

FORMAT = "vesper-door8-bridge-width-v1"
DEFAULT_WIDTHS = "64,512,256,1024"  # decision order: format default, bar, fill, asymptote


# ---------------------------------------------------------------------------
# Wrap factory: width-normalized adapter init (the door8 sweep convention)
# ---------------------------------------------------------------------------


def make_wrap(d_model: int, d_canon: int, w_gu, w_down, seed: int = 0,
              ref_rank: int = 64):
    """Door25's WrappedForeignExpert with width-normalized adapter init.

    door25 draws every adapter entry ~ N(0, 0.02^2), so the composite
    adapter_up->core->adapter_down path's per-coordinate variance grows with
    d_canon (Var(u_k) = 0.02^2 * d_canon * Var(c)) — the frozen donor core
    then sees inputs ~sqrt(d_canon/64) larger than at the verified d_canon=64
    geometry, which is exactly the rank>=256 explosion (ce->8-9 by step 25,
    NaN by ~step 30 at lr 3e-3).  Scaling adapter_up/adapter_down by
    sqrt(ref_rank/d_canon) keeps the composite scale width-independent so the
    sweep measures bridge width, not init-scale drift.  ref_rank=0 reproduces
    door25's init verbatim."""
    wrap = d25.WrappedForeignExpert(d_model, d_canon, w_gu, w_down, seed=seed)
    if ref_rank:
        s = math.sqrt(ref_rank / float(d_canon))
        if abs(s - 1.0) > 1e-9:
            with torch.no_grad():
                wrap.adapter_up.weight.mul_(s)
                wrap.adapter_down.weight.mul_(s)
    return wrap


def reinit_wrap(wrap, seed: int = 0, ref_rank: int = 64) -> None:
    """Fresh weight draw into an existing wrap (NaN-fallback must never start
    from a poisoned state — a half-lr retry on NaN'd weights NaNs at step 1)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        wrap.host_in.weight.copy_(
            torch.randn(wrap.d_canon, wrap.d_model, generator=g) * 0.02)
        wrap.adapter_up.weight.copy_(
            torch.randn(wrap.d_donor, wrap.d_canon, generator=g) * 0.02)
        wrap.adapter_down.weight.copy_(
            torch.randn(wrap.d_canon, wrap.d_donor, generator=g) * 0.02)
        wrap.host_out.weight.zero_()
        if ref_rank:
            s = math.sqrt(ref_rank / float(wrap.d_canon))
            wrap.adapter_up.weight.mul_(s)
            wrap.adapter_down.weight.mul_(s)


# ---------------------------------------------------------------------------
# Masked + additive merge: y = stock(N zeroed) + gain * e_wrap(h)
# ---------------------------------------------------------------------------


class MaskedPluginMerge:
    """PluginMerge semantics with exact capacity-removal of a native expert SET.

    The stock merge is `sum_k w_k * e_{idx_k}(h)` with NO renormalization
    (OLMoE norm_topk_prob=false), so zeroing the weight of every slot whose
    index is in `mask_idx` removes exactly those experts' contributions and
    leaves the rest of the routing untouched.  `mask_idx` may be one expert
    (the single-N protocol) or a list (the bank-scale protocol: top-2, or
    all of a layer's experts).

    Modes (same names as d1.PluginMerge so d1.set_plugin_mode works):
      off      masked stock merge (masked slots zeroed) — arm (b)
      forced   masked stock + force_gain * e_wrap(h) — arm (c), additive
               delivery at the masked site (door6's y = y_stock + lam * e
               with membership m == 1); with `membership_gated=True` the
               wrap only stands in on tokens whose routed slots hit the
               masked set (m = 1[k ∩ mask ≠ ∅]) — the cheap routed-
               membership delivery variant
      plugged  kept for interface parity; unused in the primary curve
    `mask_active=False` restores the plain stock merge (arm (a) via the same
    install, or a bitwise-stock sanity read).
    """

    def __init__(self, stock_forward, new_expert, num_stock: int,
                 mask_idx=None, membership_gated: bool = False):
        self.stock = stock_forward
        self.new_expert = new_expert
        self.num_stock = num_stock
        if mask_idx is None:
            self.mask_set = []
        elif isinstance(mask_idx, int):
            self.mask_set = [int(mask_idx)]
        else:
            self.mask_set = [int(e) for e in mask_idx]
        self.mask_idx = self.mask_set[0] if len(self.mask_set) == 1 else \
            (self.mask_set or None)
        self.mask_active = bool(self.mask_set)
        self.membership_gated = bool(membership_gated)
        self.mode = "off"
        self.force_gain = 1.0
        self.last_new_out = None

    def _slot_mask(self, top_k_index) -> torch.Tensor:
        m = torch.zeros_like(top_k_index, dtype=torch.bool)
        for e in self.mask_set:
            m |= (top_k_index == e)
        return m

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        w = top_k_weights
        hit = None
        if self.mask_active and self.mask_set:
            hit = self._slot_mask(top_k_index)
            w = torch.where(hit, torch.zeros_like(top_k_weights),
                            top_k_weights)
        if self.mode == "off":
            return self.stock(hidden_states, top_k_index, w)
        is_new = top_k_index == self.num_stock
        stock_idx = torch.where(is_new, torch.zeros_like(top_k_index),
                                top_k_index)
        stock_w = torch.where(is_new, torch.zeros_like(w), w)
        with torch.no_grad():
            y = self.stock(hidden_states, stock_idx, stock_w)
        ne = self.new_expert(hidden_states) if self.new_expert is not None \
            else None
        self.last_new_out = ne
        if self.mode == "forced":
            if ne is None or self.force_gain == 0.0:
                return y
            if self.membership_gated and hit is not None:
                m = hit.any(dim=-1).to(ne.dtype).unsqueeze(-1)
                return y + self.force_gain * m * ne
            return y + self.force_gain * ne
        if ne is None:
            return y
        w_new = (top_k_weights * is_new.to(top_k_weights.dtype)).sum(dim=-1)
        return y + ne * w_new.unsqueeze(-1)


def install_masked_plugin(blocks, host_i: int, wrap, mask_idx=None,
                          membership_gated: bool = False) -> list:
    """Install MaskedPluginMerge at ONE block (the masked site)."""
    blk = blocks[host_i]
    stock_forward = blk.experts.forward
    num_stock = getattr(blk.experts, "num_experts", blk.gate.weight.shape[0])
    pm = MaskedPluginMerge(stock_forward, wrap, num_stock, mask_idx,
                           membership_gated=membership_gated)
    blk._plugin_merge = pm

    def make_fwd(pm=pm):
        def fwd(hidden_states, top_k_index, top_k_weights):
            return pm(hidden_states, top_k_index, top_k_weights)
        return fwd

    blk.experts.forward = make_fwd()
    return [(blk, stock_forward)]


# ---------------------------------------------------------------------------
# Generalist donor selection: highest-mushiness (L, e) under a usage floor
# ---------------------------------------------------------------------------


def choose_mushy_expert(stats, usage_floor: float = 0.02,
                        table_size: int = 8) -> tuple[dict, list]:
    """Pick the (layer, expert) with the highest domain-mushiness.

    mushiness = normalized entropy of the expert's domain-rate profile
    (d0.RouteStats.home_and_prototypes): 0 = single-domain specialist,
    1 = domain-generalist ("mushy passport").

    Usage note (d0.RouteStats.observe quirk): tok_count accumulates once per
    LAYER pass, so raw rate is the per-token hit rate / n_layers. We convert
    to an interpretable hit_rate (fraction of tokens at that layer whose
    top-k includes the expert; uniform expectation = top_k / n_experts) and
    apply the floor on that scale — otherwise a floor like 0.02 is ~5x the
    uniform mean and can never bind (noise-level rare experts must not win
    on noise-mushiness).  Returns (choice, table).
    """
    home, _, mush = stats.home_and_prototypes()
    rate = stats.act_count / stats.tok_count.clamp_min(1).view(1, 1, -1)
    n_layers, n_exp = mush.shape
    hit = (rate * n_layers).mean(dim=-1)  # (L, E) token hit-rate at each layer
    usage = rate.mean(dim=-1)  # raw record (= hit / n_layers)
    hit_mean = float(hit.mean())
    rows = []
    for l in range(n_layers):
        for e in range(n_exp):
            rows.append({
                "layer": l, "expert": e,
                "mushiness": round(float(mush[l, e]), 4),
                "hit_rate": round(float(hit[l, e]), 4),
                "usage_raw": round(float(usage[l, e]), 5),
                "home": int(home[l, e]),
            })
    floor = max(usage_floor, 0.25 * hit_mean)  # relative floor on hit-rate
    qualified = [r for r in rows if r["hit_rate"] >= floor]
    pool = qualified if qualified else rows
    choice = max(pool, key=lambda r: (r["mushiness"], r["hit_rate"]))
    choice = dict(choice)
    choice["hit_rate_floor"] = round(floor, 4)
    choice["hit_rate_mean"] = round(hit_mean, 4)
    table = sorted(rows, key=lambda r: (r["mushiness"], r["hit_rate"]),
                   reverse=True)[:table_size]
    return choice, table


@torch.no_grad()
def profile_generalist_core(args, device, out: dict) -> tuple:
    """Phi-mini-MoE route-stats on the D2 slices -> mushiest (L, e) -> core."""
    import graft_d2_crossfamily as d2
    t0 = time.time()
    print(f"loading donor {args.donor} for mushiness profiling")
    donor_model, donor_tok = d2.load_donor(args.donor, device, args.dtype)
    blocks_d = d2.find_moe_blocks(donor_model)
    fam = d2.family_for(d2.detect_family(donor_model), blocks_d)
    domains = d0.load_domain_texts(args.donor_domains_dir)
    names = sorted(domains.keys())
    batches = []
    for i, name in enumerate(names):
        batches.extend(d2.texts_to_batches_chunked(
            donor_tok, [domains[name]], args.seq_len, args.batch_size,
            args.profile_tokens, label=i))
    n_tok = sum(t.numel() for t, _ in batches)
    print(f"profiling {len(names)} domains on {n_tok} donor tokens "
          f"({len(batches)} batches)")
    stats = d2.collect_route_stats(donor_model, blocks_d, fam, batches,
                                   device, len(names))
    choice, table = choose_mushy_expert(stats, usage_floor=args.min_usage_rate)
    _, _, mush = stats.home_and_prototypes()
    out["profile"] = {
        "donor": args.donor, "family": fam.name,
        "domains": names, "selector": "max-mushiness (usage floor "
                                      f"{args.min_usage_rate})",
        "n_donor_layers": len(blocks_d),
        "n_tokens_profiled": int(n_tok),
        "chosen": choice, "top_by_mushiness": table,
        "mushiness_mean": float(mush.mean()),
        "mushiness_per_expert_mean_over_layers": [
            round(float(x), 4) for x in mush.mean(dim=0)],
    }
    layer = args.home_layer if args.home_layer >= 0 else int(choice["layer"])
    expert = args.home_expert if args.home_expert >= 0 else int(choice["expert"])
    w_gu, w_down = d25.extract_donor_core(blocks_d, layer, expert)
    out["profile"]["extracted"] = {"layer": layer, "expert": expert,
                                   "w_gu": list(w_gu.shape),
                                   "w_down": list(w_down.shape)}
    if args.core_out:
        torch.save({"format": FORMAT, "w_gu": w_gu, "w_down": w_down,
                    "layer": layer, "expert": expert,
                    "n_donor_layers": len(blocks_d),
                    "profile": out["profile"]}, args.core_out)
        out["core_path"] = args.core_out
    out["profile"]["wall_s"] = round(time.time() - t0, 2)
    d25.dump_json(out, args.out)
    del donor_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return w_gu, w_down, layer, expert


# ---------------------------------------------------------------------------
# General text data: base-domain slices + wikitext, 80/20 train/held-out
# ---------------------------------------------------------------------------


def to_batches(ids, label: int, seq: int, bs: int) -> list:
    """Fixed windows of `seq` -> batches of (tokens (B,T), label)."""
    if seq <= 0 or bs <= 0 or len(ids) < seq:
        return []
    wins = [ids[i:i + seq] for i in range(0, len(ids) - seq + 1, seq)]
    return [(torch.tensor(wins[i:i + bs], dtype=torch.long), label)
            for i in range(0, len(wins) - bs + 1, bs)]


def build_general_data(tok, args) -> dict:
    """web+chat+science+math (the door25 base-domain set) + wikitext-style
    slice; each source split 80/20 by token position (held-out disjoint);
    val_mixed.txt is a third eval set (d0 reserves it for val)."""
    sources = {}
    for name in ("web", "chat", "science", "math"):
        path = os.path.join(args.domains_dir, f"{name}.txt")
        if os.path.exists(path):
            with open(path) as f:
                sources[name] = f.read()
    if args.wikitext and os.path.exists(args.wikitext):
        with open(args.wikitext) as f:
            sources["wikitext"] = f.read()[:args.wikitext_bytes]
    if not sources:
        raise RuntimeError(f"no general slices found in {args.domains_dir}")

    train_ids, eval_by_source = [], {}
    for name, text in sources.items():
        ids = tok.encode(text, add_special_tokens=False)
        split = int(0.8 * len(ids))
        tr, ev = ids[:split], ids[split:]
        train_ids.extend(tr)
        eval_by_source[name] = ev
        print(f"  general slice {name}: {len(ids)} tok -> train {len(tr)} / "
              f"held-out {len(ev)}")
    train_ids = train_ids[:args.max_train_tokens]
    train_batches = to_batches(train_ids, 1, args.seq_len, args.batch_size)

    eval_batches = []
    for name, ev in eval_by_source.items():
        eval_batches.extend(to_batches(ev, 0, args.seq_len, args.eval_batch_size))
    cap = args.max_eval_tokens
    used, capped = [], 0
    for t, _ in eval_batches:
        if capped + t.numel() > cap:
            break
        used.append((t, 0))
        capped += t.numel()
    eval_batches = used

    per_source_eval = {}
    for name, ev in eval_by_source.items():
        bs = to_batches(ev[:args.per_source_tokens], 0, args.seq_len,
                        args.eval_batch_size)
        per_source_eval[name] = bs

    val_batches = []
    val_path = os.path.join(args.domains_dir, "val_mixed.txt")
    if os.path.exists(val_path):
        with open(val_path) as f:
            val_ids = tok.encode(f.read(), add_special_tokens=False)
        val_ids = val_ids[:args.max_eval_tokens]
        val_batches = to_batches(val_ids, 0, args.seq_len, args.eval_batch_size)

    if not train_batches or not eval_batches:
        raise RuntimeError("not enough general data — raise --max-train-tokens "
                           "or lower --seq-len")
    return {
        "train_batches": train_batches,
        "eval_batches": eval_batches,
        "val_batches": val_batches,
        "per_source_eval": per_source_eval,
        "n_train_tokens": int(sum(t.numel() for t, _ in train_batches)),
        "n_eval_tokens": int(sum(t.numel() for t, _ in eval_batches)),
        "n_val_tokens": int(sum(t.numel() for t, _ in val_batches)),
        "sources": sorted(sources.keys()),
    }


# ---------------------------------------------------------------------------
# Native-expert search: which expert's removal hurts general CE most?
# ---------------------------------------------------------------------------


@torch.no_grad()
def top_expert_per_layer(model, blocks, batches, device, max_batches: int = 4):
    """Per layer: expert hit-rate on general tokens and the top-routed expert."""
    tap = d0.GateTap(blocks)
    n = len(blocks)
    hits = [torch.zeros(blocks[i].gate.weight.shape[0]) for i in range(n)]
    counts = 0
    for tokens, _ in batches[:max_batches]:
        tokens = tokens.to(device)
        tap.clear()
        d0.model_logits(model, tokens)
        counts += tokens.numel()
        for i in range(n):
            idx = tap.idx[i].reshape(-1)
            hits[i] += torch.bincount(idx.cpu(),
                                      minlength=hits[i].numel()).float()
    tap.close()
    out = []
    for i in range(n):
        share = hits[i] / max(1, counts)  # slots-per-token per expert (sums to top_k)
        e = int(share.argmax())
        out.append({"layer": i, "expert": e,
                    "slots_per_token": round(float(share[e]), 4),
                    "mean_slots_per_token": round(float(share.mean()), 5)})
    return out


@torch.no_grad()
def top_experts_at_layer(model, blocks, batches, device, layer: int,
                         m: int | None, max_batches: int = 6) -> tuple:
    """Top-m routed experts at ONE layer by slots-per-token (m=None -> all).
    Returns (expert_ids, per_expert_share)."""
    tap = d0.GateTap(blocks)
    n_exp = blocks[layer].gate.weight.shape[0]
    hits = torch.zeros(n_exp)
    counts = 0
    for tokens, _ in batches[:max_batches]:
        tokens = tokens.to(device)
        tap.clear()
        d0.model_logits(model, tokens)
        counts += tokens.numel()
        idx = tap.idx[layer].reshape(-1)
        hits += torch.bincount(idx.cpu(), minlength=n_exp).float()
    tap.close()
    share = hits / max(1, counts)
    order = torch.argsort(share, descending=True)
    if m is None or m >= n_exp:
        chosen = order.tolist()
    else:
        chosen = order[:m].tolist()
    return chosen, [round(float(share[e]), 4) for e in chosen]


@torch.no_grad()
def ce_with_expert_masked(model, blocks, batches, device, host_i: int,
                          mask_idx: int | None) -> float:
    """CE under stock routing with expert `mask_idx`'s slots zeroed (or plain
    stock when mask_idx is None)."""
    if mask_idx is None:
        tot, n = 0.0, 0
        for tokens, _ in batches:
            tokens = tokens.to(device)
            lg = d0.model_logits(model, tokens)
            tot += d0.next_token_ce(lg, tokens) * tokens.shape[0]
            n += tokens.shape[0]
        return tot / max(1, n)
    handles = install_masked_plugin(blocks, host_i, None, mask_idx)
    try:
        return d1.ce_in_state(model, blocks, batches, device,
                              gates=None, handles=handles, mode="off")
    finally:
        d1.uninstall_plugin(handles)


def search_native_expert(model, blocks, probe_batches, device, out: dict) -> dict:
    """Candidates = each layer's top-routed expert on general text; N = the
    one whose removal costs the most probe CE."""
    t0 = time.time()
    routing = top_expert_per_layer(model, blocks, probe_batches, device)
    ce_stock = ce_with_expert_masked(model, blocks, probe_batches, device, 0, None)
    table = []
    for cand in routing:
        ce_m = ce_with_expert_masked(model, blocks, probe_batches, device,
                                     cand["layer"], cand["expert"])
        table.append({**cand, "ce_masked_probe": round(ce_m, 4),
                      "mask_cost": round(ce_m - ce_stock, 4)})
        print(f"  mask probe L{cand['layer']:2d} e{cand['expert']:2d} "
              f"slots/tok {cand['slots_per_token']:.3f}  ce {ce_m:.4f} "
              f"(cost {ce_m - ce_stock:+.4f})", flush=True)
    best = max(table, key=lambda r: r["mask_cost"])
    out["native_search"] = {
        "ce_stock_probe": round(ce_stock, 4),
        "candidates": table,
        "chosen": best,
        "wall_s": round(time.time() - t0, 2),
    }
    print(f"chosen N: L{best['layer']} e{best['expert']} "
          f"mask cost {best['mask_cost']:+.4f} nats on probe")
    return best


# ---------------------------------------------------------------------------
# Wrap training under N-mask (task loss; NaN-guarded; door25 recipe otherwise)
# ---------------------------------------------------------------------------


def train_wrap_masked(model, wrap, handles, train_batches, steps, lr, device,
                      out_pen: float = 0.0, grad_clip: float = 1.0,
                      weight_decay: float = 0.0) -> dict:
    """Train ONLY the wrap adapters under forced additive delivery with N
    masked (the composition being measured).  Loss = general next-token CE
    (+ optional out_pen * mean ||e_wrap(h)||^2).  NaN guard: a non-finite
    total aborts the arm immediately (never fit a poisoned wrap)."""
    params = list(wrap.parameters())
    d1.freeze_all_except(model, params)
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    hist = []
    t0 = time.time()
    fixed = train_batches[0][0].to(device)
    aborted = None
    for step in range(steps):
        toks, _ = train_batches[step % len(train_batches)]
        toks = toks.to(device)
        d1.set_plugin_mode(handles, "forced", 1.0)
        logits = d0.model_logits(model, toks)
        v = logits.shape[-1]
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, v).float(),
                               toks[:, 1:].reshape(-1))
        out_norm2 = torch.zeros((), device=device)
        if out_pen > 0:
            bt, _ = train_batches[(step + 1) % len(train_batches)]
            d1.set_plugin_mode(handles, "forced", 1.0)
            d0.model_logits(model, bt.to(device))
            out_terms = [blk._plugin_merge.last_new_out for blk, _ in handles
                         if getattr(blk, "_plugin_merge", None) is not None
                         and blk._plugin_merge.last_new_out is not None]
            if out_terms:
                out_norm2 = torch.stack(
                    [t.float().pow(2).mean() for t in out_terms]).mean()
        total = loss + out_pen * out_norm2
        if not torch.isfinite(total):
            aborted = {"step": step, "loss": float(loss.detach()),
                       "out_norm2": float(out_norm2.detach()),
                       "reason": "non-finite loss"}
            print(f"  [NaN guard] aborting arm at step {step}: {aborted}",
                  flush=True)
            break
        opt.zero_grad()
        total.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            with torch.no_grad():
                d1.set_plugin_mode(handles, "forced", 1.0)
                lg0 = d0.model_logits(model, fixed)
                ce0 = float(F.cross_entropy(
                    lg0[:, :-1].reshape(-1, v).float(),
                    fixed[:, 1:].reshape(-1)))
                out_terms = [blk._plugin_merge.last_new_out for blk, _ in handles
                             if getattr(blk, "_plugin_merge", None) is not None
                             and blk._plugin_merge.last_new_out is not None]
                yabs = (torch.stack([t.float().abs().mean()
                                     for t in out_terms]).mean()
                        if out_terms else torch.zeros((), device=device))
                pl2 = math.sqrt(sum(float(p.detach().pow(2).sum())
                                    for p in params))
            hist.append({"step": step,
                         "ce_batch": float(loss.detach()),
                         "ce_batch0": ce0,
                         "out_norm2": float(out_norm2.detach()),
                         "mean_abs_y": float(yabs),
                         "param_l2": pl2})
            print(f"  [train {step:4d}] ce {float(loss.detach()):.4f}  "
                  f"ce0 {ce0:.4f}  |y| {float(yabs):.3e}  |W| {pl2:.2f}",
                  flush=True)
    return {"steps": steps, "lr": lr, "out_pen": out_pen,
            "grad_clip": grad_clip, "weight_decay": weight_decay,
            "history": hist, "aborted": aborted,
            "completed_steps": hist[-1]["step"] if hist else 0,
            "wall_s": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------
# lam grid (additive delivery scale at the site where N lives)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _ce_forced(model, blocks, batches, device, handles, lam: float) -> float:
    d1.set_plugin_mode(handles, "forced", lam)
    try:
        tot, n = 0.0, 0
        for tokens, _ in batches:
            tokens = tokens.to(device)
            lg = d0.model_logits(model, tokens)
            tot += d0.next_token_ce(lg, tokens) * tokens.shape[0]
            n += tokens.shape[0]
        return tot / max(1, n)
    finally:
        d1.set_plugin_mode(handles, "off")


@torch.no_grad()
def _ce_forced_per_batch(model, blocks, batches, device, handles,
                         lam: float) -> list:
    """Per-batch CE under forced gain `lam` (gain 0 == masked-only)."""
    d1.set_plugin_mode(handles, "forced", lam)
    try:
        out = []
        for tokens, _ in batches:
            tokens = tokens.to(device)
            lg = d0.model_logits(model, tokens)
            out.append(d0.next_token_ce(lg, tokens))
        return out
    finally:
        d1.set_plugin_mode(handles, "off")


@torch.no_grad()
def ce_per_batch_stock_or_masked(model, blocks, batches, device, host_i: int,
                                 mask_idx: int | None) -> list:
    """Per-batch CE of arm (a) stock (mask_idx None) or arm (b) masked."""
    handles = install_masked_plugin(blocks, host_i, None, mask_idx)
    try:
        return _ce_forced_per_batch(model, blocks, batches, device, handles, 0.0)
    finally:
        d1.uninstall_plugin(handles)


def _se(xs) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var / n)


def paired_recovery(masked_pb, stock_pb, graft_pb) -> dict:
    """Recovery of the masked-native CE cost from paired per-batch CEs.

    num_i = ce_masked_i - ce_graft_i   (CE nats recovered on batch i)
    den_i = ce_masked_i - ce_stock_i   (cost of masking N on batch i)
    recovery = mean(num) / mean(den);  recovery_se ~= se(num) / |mean(den)|
    (den uncertainty ignored — conservative on the numerator only).  Also
    reports first/second-half recoveries as a robustness read."""
    n = min(len(masked_pb), len(stock_pb), len(graft_pb))
    if n < 2:
        return {"recovery": None, "note": "too few paired batches"}
    num = [masked_pb[i] - graft_pb[i] for i in range(n)]
    den = [masked_pb[i] - stock_pb[i] for i in range(n)]
    mean_num = sum(num) / n
    mean_den = sum(den) / n

    def rec(sl):
        if not sl:
            return None
        dn = sum(den[i] for i in sl) / len(sl)
        nm = sum(num[i] for i in sl) / len(sl)
        return round(nm / dn, 4) if abs(dn) > 1e-9 else None

    h1 = list(range(n // 2))
    h2 = list(range(n // 2, n))
    out = {
        "n_batches": n,
        "ce_recovered_nats": round(mean_num, 5),
        "mask_cost_nats": round(mean_den, 5),
        "se_recovered": round(_se(num), 5),
        "se_mask_cost": round(_se(den), 5),
        "recovery_half1": rec(h1),
        "recovery_half2": rec(h2),
    }
    if abs(mean_den) > 1e-9:
        out["recovery"] = round(mean_num / mean_den, 4)
        out["recovery_se"] = round(_se(num) / abs(mean_den), 4)
    else:
        out["recovery"] = None
        out["recovery_se"] = None
        out["note"] = "mask cost ~ 0 — recovery undefined"
    return out


def save_wrap(wrap, args, w: int, arm: dict, host_i: int, n_idx,
              kind: str) -> str | None:
    """Checkpoint a trained wrap (adapters only — the core lives in core_gen.pt
    / door25's door25_core.pt) for follow-up delivery/membership runs."""
    if not getattr(args, "wrap_out_dir", None):
        return None
    os.makedirs(args.wrap_out_dir, exist_ok=True)
    path = os.path.join(args.wrap_out_dir, f"wrap_{kind}_d{w}.pt")
    torch.save({
        "format": FORMAT,
        "kind": kind,
        "d_canon": wrap.d_canon,
        "wrap": wrap.state_dict(),
        "wrap_spec": wrap.wrap_spec(),
        "host_layer": host_i,
        "masked_native_experts": (n_idx if isinstance(n_idx, list)
                                  else [n_idx]),
        "lam": arm.get("lam_fit", {}).get("best_lam"),
        "recovery": arm.get("recovery_primary"),
        "status": arm.get("status"),
    }, path)
    return path


@torch.no_grad()
def fit_lam_grid(model, blocks, handles, batches, device,
                 grid=(0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
                 max_batches: int = 6) -> dict:
    """Pick the forced-delivery gain that minimizes train CE (the trained
    composition is lam=1; the grid tests whether the delivered scale should
    differ).  Grid extends below 0.25 (door8 run-1 hit the floor there) +
    local refine with a step scaled to the winner."""
    sub = batches[:max_batches]

    def ce_at(lam: float) -> float:
        return _ce_forced(model, blocks, sub, device, handles, lam)

    t0 = time.time()
    curve = [(lam, ce_at(lam)) for lam in grid]
    best_lam = min(curve, key=lambda p: p[1])[0]
    step = min(0.25, max(0.05, 0.5 * best_lam))
    fine = [best_lam - step, best_lam - step / 2, best_lam + step / 2,
            best_lam + step]
    curve += [(round(lam, 4), ce_at(round(lam, 4))) for lam in fine
              if lam > 0 and lam not in dict(curve)]
    best_lam = min(curve, key=lambda p: p[1])[0]
    return {"best_lam": round(float(best_lam), 4),
            "curve": [{"lam": round(float(a), 4), "ce": round(float(b), 5)}
                      for a, b in curve],
            "wall_s": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------
# One width arm: train + lam + evals + recovery
# ---------------------------------------------------------------------------


def run_width_arm(model, blocks, wrap, host_i, mask_idx, data, baselines,
                  args, device, width_tag: str) -> dict:
    t0 = time.time()
    arm = {"d_canon": wrap.d_canon,
           "trainable_params": wrap.adapter_param_count(),
           "core_params_frozen": wrap.core_param_count(),
           "wrap": wrap.wrap_spec()}
    print(f"\n=== WIDTH {width_tag}: d_canon {wrap.d_canon}, "
          f"{arm['trainable_params']} trainable ===", flush=True)
    wrap.to(device)
    handles = install_masked_plugin(blocks, host_i, wrap, mask_idx)
    try:
        lr = args.expert_lr
        pa = train_wrap_masked(
            model, wrap, handles, data["train_batches"], args.expert_steps,
            lr, device, out_pen=args.out_pen,
            grad_clip=args.grad_clip, weight_decay=args.weight_decay)
        arm["phase_a"] = pa
        if pa["aborted"] is not None:
            # one documented fallback at half lr FROM A FRESH INIT (never retry
            # from a NaN-poisoned state — that NaNs at step 1)
            print(f"  arm aborted at lr {lr}; fresh reinit + retry at {lr / 2}",
                  flush=True)
            d1.uninstall_plugin(handles)
            reinit_wrap(wrap, seed=args.seed, ref_rank=args.ref_rank)
            handles = install_masked_plugin(blocks, host_i, wrap, mask_idx)
            pa2 = train_wrap_masked(
                model, wrap, handles, data["train_batches"],
                args.expert_steps, lr / 2, device, out_pen=args.out_pen,
                grad_clip=args.grad_clip, weight_decay=args.weight_decay)
            pa2["fallback_from_lr"] = lr
            arm["phase_a"] = pa2
        if arm["phase_a"]["aborted"] is not None:
            arm["status"] = "nan_abort"
            return arm

        lam = fit_lam_grid(model, blocks, handles, data["train_batches"],
                           device)
        arm["lam_fit"] = lam
        best_lam = lam["best_lam"]

        reads = {}
        for set_name, batches in (("gen_heldout", data["eval_batches"]),
                                  ("val_mixed", data["val_batches"])):
            if not batches:
                continue
            pb1 = _ce_forced_per_batch(model, blocks, batches, device,
                                       handles, 1.0)
            pbl = _ce_forced_per_batch(model, blocks, batches, device,
                                       handles, best_lam)
            reads[set_name] = {
                "ce_graft_lam1": round(sum(pb1) / max(1, len(pb1)), 4),
                "ce_graft_lam": round(sum(pbl) / max(1, len(pbl)), 4),
                "per_batch": {"graft_lam1": [round(x, 5) for x in pb1],
                              "graft_lam": [round(x, 5) for x in pbl]},
            }
        per_source = {}
        for src, batches in data["per_source_eval"].items():
            if batches:
                per_source[src] = round(_ce_forced(
                    model, blocks, batches, device, handles, best_lam), 4)
        reads["per_source_lam"] = per_source
        # cheap membership-gated delivery variant (routed-to-masked-set):
        # the wrap only stands in on tokens whose routed slots hit the mask
        if getattr(args, "membership_eval", False):
            pm = getattr(handles[0][0], "_plugin_merge", None)
            if pm is not None:
                pm.membership_gated = True
                pbm = _ce_forced_per_batch(model, blocks,
                                           data["eval_batches"], device,
                                           handles, best_lam)
                pm.membership_gated = False
                reads["gen_heldout"]["ce_graft_lam_membership"] = round(
                    sum(pbm) / max(1, len(pbm)), 4)
                reads["gen_heldout"]["per_batch"]["graft_lam_membership"] = \
                    [round(x, 5) for x in pbm]
        arm["ce"] = reads

        rec, rec_stats = {}, {}
        for set_name in ("gen_heldout", "val_mixed"):
            if set_name not in reads:
                continue
            b = baselines[set_name]
            bp = b.get("per_batch", {})
            keys = ["graft_lam1", "graft_lam"]
            if "graft_lam_membership" in reads[set_name].get("per_batch", {}):
                keys.append("graft_lam_membership")
            for key in keys:
                graft_pb = reads[set_name]["per_batch"][key]
                if bp.get("masked") and bp.get("stock"):
                    st = paired_recovery(bp["masked"], bp["stock"], graft_pb)
                    rec_stats[f"{set_name}:{key}"] = st
                    rec[f"{set_name}:ce_graft_{key.split('_', 1)[1]}"] = \
                        st.get("recovery")
                else:
                    denom = b["ce_masked"] - b["ce_stock"]
                    num = b["ce_masked"] - reads[set_name][
                        f"ce_graft_{key.split('_', 1)[1]}"]
                    rec[f"{set_name}:ce_graft_{key.split('_', 1)[1]}"] = (
                        round(num / denom, 4) if denom > 1e-6 else None)
        arm["recovery"] = rec
        arm["recovery_stats"] = rec_stats
        prim = rec.get("gen_heldout:ce_graft_lam")
        arm["recovery_primary"] = prim
        arm["status"] = "ok"
    finally:
        d1.uninstall_plugin(handles)
        # wrap out of the trunk graph before the next arm
        for p in wrap.parameters():
            p.grad = None
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    arm["wall_s"] = round(time.time() - t0, 2)
    return arm


# ---------------------------------------------------------------------------
# Secondary: specialty (code) quality at wide rank — door25 recipe, no mask
# ---------------------------------------------------------------------------


def run_secondary_code(model, blocks, wrap, host_i, code_data, args, device,
                       width_tag: str) -> dict:
    """Door25's specialty fit (code CE, out_pen 300, no N-mask) at `wrap`'s
    width — does code quality hold/improve with bridge width?"""
    t0 = time.time()
    arm = {"d_canon": wrap.d_canon,
           "trainable_params": wrap.adapter_param_count(),
           "kind": "specialty-code"}
    print(f"\n=== SECONDARY code @ d_canon {wrap.d_canon} ===", flush=True)
    wrap.to(device)
    handles = install_masked_plugin(blocks, host_i, wrap, None)
    try:
        pa = train_wrap_masked(
            model, wrap, handles, code_data["train_batches"],
            args.secondary_steps, args.expert_lr, device,
            out_pen=300.0, grad_clip=args.grad_clip, weight_decay=0.0)
        arm["phase_a"] = pa
        if pa["aborted"] is not None:
            arm["status"] = "nan_abort"
            return arm
        lam = fit_lam_grid(model, blocks, handles, code_data["train_batches"],
                           device)
        arm["lam_fit"] = lam
        best_lam = lam["best_lam"]
        arm["ce"] = {
            "code_train_lam1": round(_ce_forced(model, blocks,
                                                code_data["train_batches"],
                                                device, handles, 1.0), 4),
            "code_train_lam": round(_ce_forced(model, blocks,
                                               code_data["train_batches"],
                                               device, handles, best_lam), 4),
            "code_held_lam1": round(_ce_forced(model, blocks,
                                               code_data["eval_batches"],
                                               device, handles, 1.0), 4),
            "code_held_lam": round(_ce_forced(model, blocks,
                                              code_data["eval_batches"],
                                              device, handles, best_lam), 4),
        }
        arm["status"] = "ok"
    finally:
        d1.uninstall_plugin(handles)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    arm["wall_s"] = round(time.time() - t0, 2)
    return arm


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Bank-scale demand test (the v2 all-grafted-bank freeze gate)
# ---------------------------------------------------------------------------


def run_bank_test(model, blocks, data, args, out, device, d_model,
                  w_gu, w_down, t_run) -> int:
    """Mask ALL of a layer's routed experts (or the top-m) at once and stand
    one wrapped core in for the whole layer demand.

    Demand sweep m in {1, 2, all} at the run-1 site (host layer N lives in);
    rank sweep {64, 512} (512 = best wide rank of the width-norm run-3).
    Arms: (a) stock, (b) layer-masked (exact zeroing of the masked set's
    routed slots), (c) masked + wrapped core (width-norm init) delivering
    additively with fitted lam (grid extended below 0.25).  Recovery% =
    (CE_masked - CE_graft)/(CE_masked - CE_stock) per demand level — the
    layer denominator is much larger than the single-expert one — plus the
    absolute-CE read.  Cheap membership-gated delivery variant (wrap stands
    in only on tokens whose routed slots hit the mask) on every arm.

    Freeze bar: recovery >= 80% of the LAYER-mask CE cost at rank <= 512
    => the v2 all-grafted-bank format is FREEZE-able."""
    host_i = args.bank_layer
    n_exp = int(blocks[host_i].gate.weight.shape[0])
    widths = [int(x) for x in args.widths.split(",") if x.strip()]
    print(f"\n=== BANK-SCALE DEMAND TEST at layer {host_i} "
          f"({n_exp} experts) ===", flush=True)

    # ---- demand sets: top-m routed experts at the site (m=all -> every expert)
    probe = data["eval_batches"][:6] or data["train_batches"][:6]
    demands = []
    for spec in [s.strip() for s in args.bank_demands.split(",") if s.strip()]:
        m = None if spec == "all" else int(spec)
        experts, shares = top_experts_at_layer(model, blocks, probe, device,
                                               host_i, m)
        demands.append({"label": spec, "mask_set": experts,
                        "slots_per_token": shares,
                        "n_masked": len(experts)})
        print(f"  demand {spec}: mask {len(experts)} experts "
              f"(top shares {shares[:8]})", flush=True)
    out["bank_test"] = {
        "host_layer": host_i, "n_experts": n_exp,
        "demands": [{"label": d["label"], "n_masked": d["n_masked"],
                     "mask_set": d["mask_set"][:16],
                     "slots_per_token": d["slots_per_token"][:16]}
                    for d in demands],
        "widths_order": widths,
        "bars": {"freeze": 0.80,
                 "note": "recovery of the LAYER-mask CE cost at rank <= 512"},
    }
    out["phase"] = "bank_demands_chosen"
    d25.dump_json(out, args.out)

    # ---- baselines: (a) stock once, (b) masked per demand level ----
    t0 = time.time()
    pb_stock = {}
    for set_name, batches in (("gen_heldout", data["eval_batches"]),
                              ("val_mixed", data["val_batches"])):
        if batches:
            pb_stock[set_name] = ce_per_batch_stock_or_masked(
                model, blocks, batches, device, host_i, None)
    baselines_by_demand = {}
    for d in demands:
        base = {}
        for set_name, batches in (("gen_heldout", data["eval_batches"]),
                                  ("val_mixed", data["val_batches"])):
            if not batches:
                continue
            pb_m = ce_per_batch_stock_or_masked(model, blocks, batches,
                                                device, host_i, d["mask_set"])
            pb_s = pb_stock[set_name]
            ce_s = sum(pb_s) / max(1, len(pb_s))
            ce_m = sum(pb_m) / max(1, len(pb_m))
            base[set_name] = {"ce_stock": round(ce_s, 4),
                              "ce_masked": round(ce_m, 4),
                              "mask_cost": round(ce_m - ce_s, 4),
                              "per_batch": {"stock": [round(x, 5) for x in pb_s],
                                            "masked": [round(x, 5) for x in pb_m]}}
            print(f"  baseline demand {d['label']} {set_name}: stock "
                  f"{ce_s:.4f}  masked {ce_m:.4f}  cost {ce_m - ce_s:+.4f}",
                  flush=True)
        baselines_by_demand[d["label"]] = base
    out["bank_test"]["baselines"] = baselines_by_demand
    out["bank_test"]["baselines_wall_s"] = round(time.time() - t0, 2)
    out["phase"] = "bank_baselines_done"
    d25.dump_json(out, args.out)

    # ---- sweep: demand (largest first) x rank ----
    out["bank_test"]["arms"] = {}
    ordered = sorted(demands, key=lambda d: -d["n_masked"])
    for d in ordered:
        for w in widths:
            tag = f"m{d['label']}_d{w}"
            wrap = make_wrap(d_model, w, w_gu, w_down,
                             seed=args.seed, ref_rank=args.ref_rank)
            arm = run_width_arm(model, blocks, wrap, host_i, d["mask_set"],
                                data, baselines_by_demand[d["label"]], args,
                                device, tag)
            arm["demand"] = d["label"]
            arm["n_masked"] = d["n_masked"]
            wpath = save_wrap(wrap, args, w, arm, host_i, d["mask_set"],
                              f"bank_m{d['label']}")
            if wpath:
                arm["wrap_path"] = wpath
            out["bank_test"]["arms"][tag] = arm
            out["elapsed_s"] = round(time.time() - t_run, 2)
            out["phase"] = f"bank_{tag}_done"
            d25.dump_json(out, args.out)
            del wrap

    # ---- freeze verdict (bar on the LAYER demand = all-masked) ----
    curve = []
    for d in ordered:
        for w in widths:
            arm = out["bank_test"]["arms"].get(f"m{d['label']}_d{w}") or {}
            curve.append({
                "demand": d["label"], "n_masked": d["n_masked"],
                "d_canon": w, "trainable_params": arm.get("trainable_params"),
                "lam": (arm.get("lam_fit") or {}).get("best_lam"),
                "ce_graft": ((arm.get("ce") or {}).get("gen_heldout") or {}
                             ).get("ce_graft_lam"),
                "recovery": arm.get("recovery_primary"),
                "nats_recovered": (((arm.get("recovery_stats") or {})
                                    .get("gen_heldout:graft_lam") or {})
                                   .get("ce_recovered_nats")),
                "status": arm.get("status"),
            })
    layer_rows = [c for c in curve if c["demand"] == "all"
                  and c["recovery"] is not None and c["d_canon"] <= 512]
    best_layer = max((c["recovery"] for c in layer_rows), default=None)
    if best_layer is None:
        verdict = ("INCOMPLETE — layer-demand recovery not measured "
                   "(chain continues detached)")
    elif best_layer >= 0.80:
        at = min(c["d_canon"] for c in layer_rows if c["recovery"] >= 0.80)
        verdict = (f"FREEZE-ABLE — recovery {best_layer:.1%} of the "
                   f"layer-mask CE cost at rank {at} (<=512): wrapped cores "
                   f"carry a full layer's expert demand")
    elif best_layer >= 0.50:
        verdict = (f"MARGINAL — best layer-demand recovery at rank <=512 is "
                   f"{best_layer:.1%}: wider/multi-core standing-in needed "
                   f"before freeze")
    else:
        verdict = (f"NOT FREEZE-ABLE at one core — layer-demand recovery "
                   f"{best_layer:.1%} < 50% at rank <=512: a single wrapped "
                   f"core cannot carry a full layer's expert bank (see the "
                   f"1/2/all demand curve — the format needs ~one graft per "
                   f"native slot, which is the endgame shape)")
    out["bank_test"]["curve"] = curve
    out["bank_test"]["verdict"] = {
        "text": verdict, "best_layer_recovery_rank_le_512": best_layer,
        "bars": {"freeze": 0.80, "marginal": 0.50}}
    out["phase"] = "complete"
    out["elapsed_s"] = round(time.time() - t_run, 2)
    d25.dump_json(out, args.out)

    print(f"\nresults -> {args.out}")
    print("bank demand-vs-recovery (gen_heldout):")
    for c in curve:
        print(f"  demand {c['demand']:>3} (mask {c['n_masked']:2d})  "
              f"d_canon {c['d_canon']:5d}  params {c['trainable_params']}  "
              f"lam {c['lam']}  ce {c['ce_graft']}  rec {c['recovery']}  "
              f"nats {c['nats_recovered']}  {c['status']}")
    print(f"VERDICT: {verdict}")
    return 0


# Synthetic self-test (--synthetic)
# ---------------------------------------------------------------------------


def run_synthetic_tests() -> int:
    torch.manual_seed(0)
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" — {detail}" if detail else ""))

    device = "cpu"
    d_model, d_donor, d_inter = 32, 48, 16
    w_gu = torch.randn(2 * d_inter, d_donor) * 0.05
    w_down = torch.randn(d_donor, d_inter) * 0.05

    # param counts across the sweep
    for dc in (8, 16):
        wrap = d25.WrappedForeignExpert(d_model, dc, w_gu, w_down, seed=0)
        expect = (2 * d_model * dc + 2 * dc * d_donor)
        check(f"adapter_params_dcanon{dc}",
              wrap.adapter_param_count() == expect,
              f"{wrap.adapter_param_count()} (expect {expect})")

    # masked merge on the d0 mock block
    model, n_dom = d0.build_mock_donor()
    model.eval()
    blocks = d0.find_moe_blocks(model)
    blk = blocks[0]
    stock_fwd = blk.experts.forward
    toks = torch.randint(0, 90, (2, 16))
    h = model.embed(toks).view(-1, model.embed.weight.shape[1])
    _, w, idx = blk.gate(h)

    wrap = d25.WrappedForeignExpert(d_model, 8, w_gu, w_down, seed=0)
    handles = install_masked_plugin(blocks, 0, wrap, mask_idx=0)
    pm = blk._plugin_merge
    with torch.no_grad():
        y_masked = blk.experts.forward(h, idx, w)
        w_zero = torch.where(idx == 0, torch.zeros_like(w), w)
        y_ref = stock_fwd(h, idx, w_zero)
        check("mask_equals_zeroed_slot_weights",
              torch.allclose(y_masked, y_ref, atol=1e-5),
              f"max diff {float((y_masked - y_ref).abs().max()):.2e}")
        pm.mask_active = False
        y_stock = blk.experts.forward(h, idx, w)
        y_ref_stock = stock_fwd(h, idx, w)
        check("unmasked_bitwise_stock",
              torch.allclose(y_stock, y_ref_stock, atol=1e-6),
              f"max diff {float((y_stock - y_ref_stock).abs().max()):.2e}")
        pm.mask_active = True
        d1.set_plugin_mode(handles, "forced", 1.0)
        y_forced = blk.experts.forward(h, idx, w)
        # identity at birth: forced == masked
        check("identity_at_birth_forced_eq_masked",
              torch.allclose(y_forced, y_masked, atol=1e-5),
              f"max diff {float((y_forced - y_masked).abs().max()):.2e}")
        d1.set_plugin_mode(handles, "off")
    d1.uninstall_plugin(handles)

    # mask-set + membership-gated variants (bank-scale delivery plumbing)
    handles = install_masked_plugin(blocks, 0, wrap, mask_idx=[0, 1])
    pm = blk._plugin_merge
    with torch.no_grad():
        y_m2 = blk.experts.forward(h, idx, w)
        w_zero2 = torch.where((idx == 0) | (idx == 1), torch.zeros_like(w), w)
        y_ref2 = stock_fwd(h, idx, w_zero2)
        check("mask_set_equals_zeroed_slots",
              torch.allclose(y_m2, y_ref2, atol=1e-5),
              f"max diff {float((y_m2 - y_ref2).abs().max()):.2e}")
        # membership gating: with a live wrap output, forced+gated only adds
        # on tokens whose routed slots hit the masked set
        with torch.no_grad():
            wrap.host_out.weight.fill_(0.01)
        d1.set_plugin_mode(handles, "forced", 1.0)
        y_plain = blk.experts.forward(h, idx, w)
        pm.membership_gated = True
        y_gated = blk.experts.forward(h, idx, w)
        pm.membership_gated = False
        hit = ((idx == 0) | (idx == 1)).any(dim=-1)
        same = torch.isclose(y_gated, y_plain, atol=1e-5).all(dim=-1)
        check("membership_gating_hits_only",
              bool(((hit == same) | (~hit)).all()),
              f"hit {int(hit.sum())}/{hit.numel()} tokens gated-through")
    d1.uninstall_plugin(handles)

    # mushiness selection on a synthetic RouteStats (tok_count is per-layer-
    # diluted in d0.observe; the floor must exclude noise-rare uniforms)
    stats = d0.RouteStats(2, 3, 4, 2)
    stats.act_count = torch.zeros(2, 3, 2)
    stats.tok_count = torch.tensor([100.0, 100.0])
    stats.h_sum = torch.zeros(2, 3, 2, 4)
    stats.act_count[0, 0, 0] = 50
    stats.act_count[0, 0, 1] = 50     # e0: uniform + well used -> mushy workhorse
    stats.act_count[1, 1, 0] = 90
    stats.act_count[1, 1, 1] = 10     # e1: specialist
    stats.act_count[0, 2, 0] = 1
    stats.act_count[0, 2, 1] = 1      # e2: uniform but barely fires (noise)
    choice, table = choose_mushy_expert(stats, usage_floor=0.02)
    check("mushy_picks_uniform_used",
          choice["layer"] == 0 and choice["expert"] == 0,
          f"chose L{choice['layer']} e{choice['expert']} "
          f"mush {choice['mushiness']} hit {choice['hit_rate']}")

    # to_batches signature + shapes (door25 call convention: ids, label, seq, bs)
    bts = to_batches(list(range(100)), 1, 8, 2)
    check("to_batches_shape",
          len(bts) == 6 and bts[0][0].shape == (2, 8) and bts[0][1] == 1
          and to_batches(list(range(5)), 0, 8, 2) == [],
          f"{len(bts)} batches, first {tuple(bts[0][0].shape)}")

    # width-normalized init + fresh reinit (the door8 sweep conventions)
    wA = make_wrap(d_model, 8, w_gu, w_down, seed=0, ref_rank=64)
    wB = make_wrap(d_model, 32, w_gu, w_down, seed=0, ref_rank=64)
    sA = float(wA.adapter_up.weight.std())
    sB = float(wB.adapter_up.weight.std())
    check("width_norm_init_scales",
          abs(sA / sB - 2.0) < 0.35,
          f"std d8 {sA:.4f} vs d32 {sB:.4f} (expect ~2x)")
    wB.host_in.weight.data.fill_(float("nan"))
    reinit_wrap(wB, seed=0, ref_rank=64)
    check("reinit_restores_finite",
          bool(torch.isfinite(wB.host_in.weight).all())
          and float(wB.host_out.weight.abs().sum()) == 0.0,
          f"host_out zero {float(wB.host_out.weight.abs().sum()):.1f}")

    # mini arm end-to-end on the mock (mask e0, fit wrap, recovery field)
    class A:
        pass
    args = A()
    args.expert_steps = 24
    args.expert_lr = 1e-2
    args.out_pen = 0.0
    args.grad_clip = 1.0
    args.weight_decay = 0.0
    args.seed = 0
    args.ref_rank = 64

    wrap = make_wrap(d_model, 8, w_gu, w_down, seed=0, ref_rank=64)
    batches = [(t, 0) for t, _ in d0.mock_batches(n_dom, seed=3, seq_len=16,
                                                  batches_per_domain=3,
                                                  batch_size=2)]
    data = {"train_batches": batches, "eval_batches": batches,
            "val_batches": batches,
            "per_source_eval": {"a": batches[:2]}, "sources": ["a"]}
    pb_s = ce_per_batch_stock_or_masked(model, blocks, batches, device, 0, None)
    pb_m = ce_per_batch_stock_or_masked(model, blocks, batches, device, 0, 0)
    base = {"ce_stock": round(sum(pb_s) / len(pb_s), 4),
            "ce_masked": round(sum(pb_m) / len(pb_m), 4),
            "per_batch": {"stock": pb_s, "masked": pb_m}}
    baselines = {"gen_heldout": base, "val_mixed": base}
    arm = run_width_arm(model, blocks, wrap, 0, 0, data, baselines, args,
                        device, "syn8")
    check("arm_completes", arm.get("status") == "ok",
          f"status {arm.get('status')} ce {arm.get('ce')}")
    check("recovery_field_present",
          isinstance(arm.get("recovery"), dict) and "gen_heldout:ce_graft_lam"
          in arm["recovery"] and "recovery_stats" in arm,
          f"recovery {arm.get('recovery')} stats keys "
          f"{sorted((arm.get('recovery_stats') or {}).keys())}")
    check("trainable_params_logged",
          arm.get("trainable_params") == wrap.adapter_param_count(),
          f"{arm.get('trainable_params')}")

    # ---- bank-test pipeline smoke on the mock (exercises run_bank_test) ----
    class B:
        pass
    bargs = B()
    bargs.bank_layer = 0
    bargs.bank_demands = "1,all"
    bargs.widths = "8"
    bargs.seed = 0
    bargs.ref_rank = 64
    bargs.expert_steps = 8
    bargs.expert_lr = 1e-2
    bargs.out_pen = 0.0
    bargs.grad_clip = 1.0
    bargs.weight_decay = 0.0
    bargs.membership_eval = True
    bargs.wrap_out_dir = None
    bargs.out = None
    bout = {"gate": "bank-smoke"}
    bdata = {"train_batches": batches, "eval_batches": batches,
             "val_batches": batches,
             "per_source_eval": {"a": batches[:2]}, "sources": ["a"]}
    rc_bank = run_bank_test(model, blocks, bdata, bargs, bout, device,
                            d_model, w_gu, w_down, time.time())
    bt = bout.get("bank_test") or {}
    check("bank_pipeline_completes",
          rc_bank == 0 and "curve" in bt and "verdict" in bt,
          f"rc {rc_bank} arms {list((bt.get('arms') or {}).keys())} "
          f"verdict {(bt.get('verdict') or {}).get('text', '')[:60]}")
    check("bank_demands_resolved",
          [d["n_masked"] for d in (bt.get("demands") or [])] == [1, 9],
          f"{[(d['label'], d['n_masked']) for d in (bt.get('demands') or [])]}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} synthetic checks passed")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--donor", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--domains-dir", default=os.path.join(REPO_ROOT, "lab",
                                                          "graft_d0_texts"))
    ap.add_argument("--donor-domains-dir", default=None,
                    help="graft_d2_slice — the D2 profiling domains")
    ap.add_argument("--wikitext", default=None,
                    help="wikitext-style general slice (one big .txt)")
    ap.add_argument("--wikitext-bytes", type=int, default=400000)
    ap.add_argument("--home-layer", type=int, default=-1,
                    help="force donor layer (-1 = mushiness choice)")
    ap.add_argument("--home-expert", type=int, default=-1,
                    help="force donor expert (-1 = mushiness choice)")
    ap.add_argument("--min-usage-rate", type=float, default=0.02)
    ap.add_argument("--profile-tokens", type=int, default=6000)
    ap.add_argument("--core-out", default=None)
    ap.add_argument("--wrap-out-dir", default=None,
                    help="save each trained wrap's adapters here "
                         "(wrap_<kind>_d<width>.pt)")
    ap.add_argument("--core-from", default=None,
                    help="skip donor profiling; load a saved core .pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--widths", default=DEFAULT_WIDTHS,
                    help="comma-separated d_canon sweep (execution order)")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="training bs; bs x seq must dodge the gfx906 "
                         "backward holes (4x512 = 2048 verified)")
    ap.add_argument("--eval-batch-size", type=int, default=2,
                    help="ops rule: eval batch 2")
    ap.add_argument("--max-train-tokens", type=int, default=140000)
    ap.add_argument("--max-eval-tokens", type=int, default=32000)
    ap.add_argument("--per-source-tokens", type=int, default=4096)
    ap.add_argument("--expert-steps", type=int, default=250)
    ap.add_argument("--expert-lr", type=float, default=3e-3,
                    help="thin-wrap lr (D1 starvation lesson; wd must be 0)")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--out-pen", type=float, default=0.0,
                    help="generalist arms: pure task loss (0). The wrap is "
                         "meant to carry norm here; the specialty secondary "
                         "arm keeps door25's 300 internally.")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ref-rank", type=int, default=64,
                    help="adapter init width-normalization reference "
                         "(sqrt(ref_rank/d_canon) on adapter_up/down; 0 = "
                         "door25 init verbatim). 64 keeps the composite "
                         "scale at the verified door25 geometry — without it "
                         "rank>=256 explodes at lr 3e-3 (see the docstring).")
    ap.add_argument("--bank-test", action="store_true",
                    help="bank-scale demand test (v2 freeze gate): mask the "
                         "top-m/all experts of --bank-layer and stand one "
                         "wrapped core in for the whole layer demand")
    ap.add_argument("--bank-layer", type=int, default=2,
                    help="host layer for the bank test (2 = the run-1 site)")
    ap.add_argument("--bank-demands", default="1,2,all",
                    help="comma-separated mask sizes at the bank layer")
    ap.add_argument("--membership-eval", action="store_true",
                    help="also eval the cheap routed-membership delivery "
                         "variant (wrap stands in only on tokens that "
                         "routed into the masked set)")
    ap.add_argument("--secondary", action="store_true",
                    help="code specialty re-check at --secondary-widths")
    ap.add_argument("--secondary-widths", default="64,512")
    ap.add_argument("--secondary-steps", type=int, default=250)
    ap.add_argument("--secondary-deadline-s", type=float, default=5700,
                    help="skip the secondary if the run has already used this "
                         "much wall time (protect the primary curve)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.synthetic:
        return run_synthetic_tests()
    if not args.model or not args.registry:
        ap.error("--model and --registry are required unless --synthetic")
    if not args.core_from and (not args.donor or not args.donor_domains_dir):
        ap.error("--donor + --donor-domains-dir (or --core-from) required")
    if args.bank_test:
        args.membership_eval = True  # the cheap variant — always read it here

    widths = [int(x) for x in args.widths.split(",") if x.strip()]
    out = {"gate": "D8-bridge-width", "phase": "start", "format": FORMAT,
           "bars": {"feasible": BAR_FEASIBLE, "marginal": BAR_MARGINAL,
                    "note": "recovery of the masked-native CE cost on general "
                            "text; feasible >= 0.8 at rank <= 512"},
           "widths_order": widths,
           "args": {k: (v if not isinstance(v, str) or len(v) < 200 else v[:200])
                    for k, v in sorted(vars(args).items())}}
    if args.out is None:
        args.out = os.path.join(REPO_ROOT, "lab", "imported", "graft_door8",
                                "door8_width.json")
    if args.core_out is None:
        args.core_out = os.path.join(REPO_ROOT, "lab", "imported",
                                     "graft_door8", "door8_core.pt")
    d25.dump_json(out, args.out)
    t_run = time.time()

    # ---- donor side: generalist core (mushiness) or resume ----
    if args.core_from:
        w_gu, w_down, d_layer, d_expert, n_donor_layers, prof = d25.load_core(
            args.core_from)
        out["profile"] = prof or {"loaded_from": args.core_from}
        out["profile"]["loaded_from"] = args.core_from
        print(f"core loaded from {args.core_from} (donor L{d_layer} e{d_expert})")
    else:
        w_gu, w_down, d_layer, d_expert = profile_generalist_core(
            args, args.device, out)
        print(f"chosen generalist core: donor L{d_layer} e{d_expert} "
              f"(w_gu {tuple(w_gu.shape)})")
    out["phase"] = "profile_done"
    d25.dump_json(out, args.out)

    # ---- host side ----
    t_load = time.time()
    model, tok = d0.load_donor(args.model, args.device, args.dtype)
    blocks = d0.find_moe_blocks(model)
    banks_reg, reg_meta = d0.load_registry(args.registry)
    if len(banks_reg) != len(blocks):
        raise RuntimeError(f"registry has {len(banks_reg)} layers, "
                           f"model {len(blocks)}")
    n_host = len(blocks)
    d_model = banks_reg[0].d_model
    out["host"] = {"model": args.model, "registry": args.registry,
                   "registry_layers": len(banks_reg),
                   "n_layers": n_host, "d_model": d_model,
                   "num_experts": int(banks_reg[0].num_experts),
                   "donor_layer": d_layer, "donor_expert": d_expert}
    out["phase"] = "host_loaded"
    d25.dump_json(out, args.out)
    print(f"loaded host+registry in {time.time() - t_load:.1f}s: "
          f"{n_host} layers x {banks_reg[0].num_experts} experts, "
          f"d_model {d_model}")

    # ---- general data ----
    data = build_general_data(tok, args)
    out["data"] = {k: data[k] for k in ("n_train_tokens", "n_eval_tokens",
                                        "n_val_tokens", "sources")}
    print(f"general data: train {data['n_train_tokens']} tok "
          f"({len(data['train_batches'])} batches), "
          f"held-out {data['n_eval_tokens']} tok, val_mixed "
          f"{data['n_val_tokens']} tok")
    out["phase"] = "data_ready"
    d25.dump_json(out, args.out)

    # ---- bank-scale demand test (the v2 freeze gate) ----
    if args.bank_test:
        return run_bank_test(model, blocks, data, args, out, args.device,
                             d_model, w_gu, w_down, t_run)

    # ---- native expert N: most load-bearing on general text ----
    probe = data["eval_batches"][:4] or data["train_batches"][:4]
    best = search_native_expert(model, blocks, probe, args.device, out)
    host_i, n_idx = int(best["layer"]), int(best["expert"])
    out["phase"] = "n_chosen"
    d25.dump_json(out, args.out)

    # ---- baselines: (a) stock, (b) N masked ----
    t0 = time.time()
    baselines = {}
    for set_name, batches in (("gen_heldout", data["eval_batches"]),
                              ("val_mixed", data["val_batches"])):
        if not batches:
            continue
        pb_s = ce_per_batch_stock_or_masked(model, blocks, batches,
                                            args.device, host_i, None)
        pb_m = ce_per_batch_stock_or_masked(model, blocks, batches,
                                            args.device, host_i, n_idx)
        ce_s = sum(pb_s) / max(1, len(pb_s))
        ce_m = sum(pb_m) / max(1, len(pb_m))
        baselines[set_name] = {"ce_stock": round(ce_s, 4),
                               "ce_masked": round(ce_m, 4),
                               "mask_cost": round(ce_m - ce_s, 4),
                               "per_batch": {
                                   "stock": [round(x, 5) for x in pb_s],
                                   "masked": [round(x, 5) for x in pb_m]}}
        print(f"baseline {set_name}: stock {ce_s:.4f}  masked {ce_m:.4f}  "
              f"cost {ce_m - ce_s:+.4f}", flush=True)
    per_src = {}
    for src, batches in data["per_source_eval"].items():
        if batches:
            per_src[src] = {
                "ce_stock": round(ce_with_expert_masked(
                    model, blocks, batches, args.device, host_i, None), 4),
                "ce_masked": round(ce_with_expert_masked(
                    model, blocks, batches, args.device, host_i, n_idx), 4)}
    baselines["per_source"] = per_src
    out["baselines"] = baselines
    out["baselines_wall_s"] = round(time.time() - t0, 2)
    out["phase"] = "baselines_done"
    d25.dump_json(out, args.out)

    # ---- the width sweep ----
    out["widths"] = {}
    for w in widths:
        wrap = make_wrap(d_model, w, w_gu, w_down,
                         seed=args.seed, ref_rank=args.ref_rank)
        arm = run_width_arm(model, blocks, wrap, host_i, n_idx, data,
                            baselines, args, args.device, str(w))
        out["widths"][str(w)] = arm
        wrap_path = save_wrap(wrap, args, w, arm, host_i, n_idx, "general")
        if wrap_path:
            arm["wrap_path"] = wrap_path
        out["elapsed_s"] = round(time.time() - t_run, 2)
        out["phase"] = f"width_{w}_done"
        d25.dump_json(out, args.out)
        del wrap

    # ---- verdict + curve ----
    curve = []
    for w in widths:
        arm = out["widths"][str(w)]
        curve.append({
            "d_canon": w,
            "trainable_params": arm.get("trainable_params"),
            "recovery": arm.get("recovery_primary"),
            "recovery_all": arm.get("recovery"),
            "ce": arm.get("ce", {}).get("gen_heldout"),
            "status": arm.get("status"),
        })
    thin = [c for c in curve if c["d_canon"] <= 512
            and c["recovery"] is not None]
    best_thin = max((c["recovery"] for c in thin), default=None)
    if best_thin is None:
        verdict = "INCOMPLETE — no thin-rank recovery measured"
    elif best_thin >= BAR_FEASIBLE:
        at = min(c["d_canon"] for c in thin if c["recovery"] >= BAR_FEASIBLE)
        verdict = (f"FEASIBLE — >= {BAR_FEASIBLE:.0%} recovery at rank "
                   f"<= 512 (best thin rank meeting the bar: {at}); "
                   f"all-grafted banks live at reasonable width")
    elif best_thin >= BAR_MARGINAL:
        at = min(c["d_canon"] for c in thin
                 if c["recovery"] >= BAR_MARGINAL)
        verdict = (f"MARGINAL — best thin-rank recovery {best_thin:.1%} "
                   f"(at rank {at}): needs wider bridges or multi-core "
                   f"per site")
    else:
        verdict = (f"FATAL at thin rank — best <=512 recovery "
                   f"{best_thin:.1%} < {BAR_MARGINAL:.0%}: the bottleneck "
                   f"kills general traffic; all-grafted banks need a "
                   f"different bridge (see the curve)")
    out["curve"] = curve
    out["verdict"] = {"text": verdict, "best_thin_recovery": best_thin,
                      "bars": {"feasible": BAR_FEASIBLE,
                               "marginal": BAR_MARGINAL}}

    # ---- secondary: specialty at wide rank ----
    if args.secondary:
        elapsed = time.time() - t_run
        if elapsed > args.secondary_deadline_s:
            out["secondary"] = {"skipped": f"elapsed {elapsed:.0f}s > "
                                           f"deadline {args.secondary_deadline_s}s"}
        else:
            out["secondary"] = {}
            code_text = d0.load_domain_texts(args.domains_dir).get("code")
            if not code_text:
                out["secondary"]["skipped"] = "no code.txt in domains-dir"
            else:
                code_ids = tok.encode(code_text, add_special_tokens=False)
                split = int(0.6 * len(code_ids))
                code_tr = code_ids[:split][:args.max_train_tokens]
                code_ev = code_ids[split:]
                code_data = {
                    "train_batches": to_batches(code_tr, 1, args.seq_len,
                                                args.batch_size),
                    "eval_batches": to_batches(code_ev, 0, args.seq_len,
                                               args.eval_batch_size)}
                for w in [int(x) for x in args.secondary_widths.split(",")
                          if x.strip()]:
                    if time.time() - t_run > args.secondary_deadline_s:
                        out["secondary"][str(w)] = {"skipped": "deadline"}
                        continue
                    wrap = make_wrap(d_model, w, w_gu, w_down,
                                     seed=args.seed, ref_rank=args.ref_rank)
                    out["secondary"][str(w)] = run_secondary_code(
                        model, blocks, wrap, host_i, code_data, args,
                        args.device, str(w))
                    wpath = save_wrap(wrap, args, w,
                                      out["secondary"][str(w)], host_i,
                                      n_idx, "code")
                    if wpath:
                        out["secondary"][str(w)]["wrap_path"] = wpath
                    out["secondary"]["reference_door25_d64"] = {
                        "code_train_masked": 1.072, "code_train_plugged_mass":
                        0.439, "code_train_forced": 0.101,
                        "code_held_masked": 1.587, "code_held_plugged": 1.621,
                        "code_held_forced": 1.950,
                        "note": "door25 wrap at d_canon 64, mass delivery "
                                "(handoff 2026-10-11); door6 additive "
                                "delivery on the same artifact: train "
                                "0.221 (A-only)"}
                    del wrap
        out["phase"] = "secondary_done"
        d25.dump_json(out, args.out)

    out["phase"] = "complete"
    out["elapsed_s"] = round(time.time() - t_run, 2)
    d25.dump_json(out, args.out)

    print(f"\nresults -> {args.out}")
    print(f"N: L{host_i} e{n_idx}  mask cost "
          f"{baselines.get('gen_heldout', {}).get('mask_cost')} nats")
    print("width curve (gen_heldout recovery of the masked-native CE cost):")
    for c in curve:
        print(f"  d_canon {c['d_canon']:5d}  params {c['trainable_params']}  "
              f"recovery {c['recovery']}  status {c['status']}")
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
