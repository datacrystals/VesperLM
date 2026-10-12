#!/usr/bin/env python3
"""bank2n_partition — the PARTITIONED 2n-bank experiment (the user's design).

The user's design (verbatim intent): "take n whole experts from model A and n
whole ones from model B, model C has 2n experts" — a bank where the ROUTER
PARTITIONS traffic across wrapped donor experts (each expert handles only what
routes to it), NOT the failing Door-8 demand=all substitution (few cores
compensating everything with overlapping tiny-λ gated delivery; 8e: one core
57.8% ceiling, multi-wrap composition adds ~0).

Model C (this experiment)
-------------------------
spine   Phi-mini-MoE-instruct + D2 registry (the grafted_spine_v0 assembly)
bank    2n = 8 wrapped donor cores as FIRST-CLASS passport rows:
        4 Phi-family (the v0 picks L13e8 code + L24e4 math, reused with their
        fitted adapters, + 2 distinct-domain picks from a fresh profile) and
        4 OLMoE-family cores picked from OLMoE's 64 experts by domain profile,
        wrapped in the same door25 geometry (frozen core + rank-64 adapters,
        width-norm init = identity at rank 64), per-core fit on its own domain
        traffic (door25 recipe: task CE + out_pen 300, wd 0, NaN guard).
sites   depth-matched host layers (Phi donor L → host L; OLMoE donor L/16 →
        host round(L*31/15)) — the v0 placement rule; grafted rows join the
        site bank's softmax alongside the 16 native rows.

PARTITIONED delivery (the critical difference from Door 8 / v0 overlay)
----------------------------------------------------------------------
selection: sparsemixer over the FULL bank (native rows + grafted rows) —
           grafted rows win tokens like native experts; rows are recal'd
           (G1b/G3 mex recipe: home NLL + owner_mass + base reject) so each
           graft owns its domain's routing and native rows keep the rest.
dispatch:  per-slot — a token routed to grafted row j gets w_j * e_wrap_j in
           place of the native expert it displaced; NO λ, NO overlay stacking,
           NO simultaneous double-delivery.  Two grafted rows firing on one
           token (at their respective sites) is a DEFECT → cross-fire matrix.

Measurements (per task)
-----------------------
(a) per-graft per-domain quality routed-vs-masked (+ solo reference: is the
    m1-style per-slot recovery preserved in-bank?),
(b) base/union CE of model C vs bare spine (the partition tax; bar ≤ +3%),
(c) cross-fire matrix (routed-share) + pairwise interference (leave-one-out
    CE harm; bar ≤ 0.02 nats),
(d) partition vs the v0 additive-λ overlay on the same cores (head-to-head).

Bars: every graft's domain improves over masked; base CE cost ≤ +3%;
pairwise cross-domain harm ≤ 0.02 CE.  Green ⇒ the 2n-bank form is validated
and "8B+8B → 2n" is a build recipe.

Phases: profile → fit (per-wrap subsets, one process per GPU) → assemble
(measure + verdict).  Ops (HANDOFF): detached setsid, incremental JSON, no
pgrep gates, NaN guard, wd 0, gfx906 shapes (train B*T ∈ {512,1536,2048};
eval bs2×512), numeric-only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

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
import grafted_spine_v0 as g0  # noqa: E402

FORMAT = "vesper-bank2n-partition-v1"
BAR_BASE_TAX = 0.03          # base CE cost ≤ +3%
BAR_CROSS_HARM = 0.02        # pairwise cross-domain harm ≤ 0.02 nats
BAR_OWNER = 0.85             # g3 closing recal target (row "owns" its domain)

DOMAINS = ("code", "math", "chat", "science", "web")


def dump_json(out: dict, path: str | None) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, default=str)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# PartitionGate — sparsemixer over the FULL bank, per-row win recording
# ---------------------------------------------------------------------------

class PartitionGate(d2.GraftGateD2):
    """First-class grafted rows: donor selection runs over ALL rows (native +
    grafted).  last_win[:, j] = grafted row j won ≥1 top-k slot for the token
    (the routed-share / cross-fire primitive)."""

    def __init__(self, bank, fam, stock_gate, n_new: int, jitter_eps: float = 0.0):
        super().__init__(bank, fam, stock_gate, jitter_eps)
        self.n_new = n_new
        self.last_win = None

    def forward(self, hidden_states: torch.Tensor):
        scores, weights, idx = super().forward(hidden_states)
        if self.n_new > 0:
            n_stock = self.bank.num_experts - self.n_new
            gidx = torch.arange(self.n_new, device=idx.device).view(1, 1, -1)
            self.last_win = (idx.unsqueeze(-1) == (n_stock + gidx)).any(dim=1)
        return scores, weights, idx


class PartitionMerge:
    """Per-slot dispatch: y = stock(native slots) + Σ_j w_j · e_j (grafted slot
    j).  A grafted slot REPLACES the native expert that would have served the
    token (that is the partition: each expert handles only what routes to it).
    modes: off (stock merge) / dispatch (partition) — no λ, no overlay."""

    def __init__(self, stock_forward, experts, tags, gate_getter, num_stock: int):
        self.stock = stock_forward
        self.experts = list(experts)
        self.tags = list(tags)
        self.gate_getter = gate_getter
        self.num_stock = num_stock
        self.mode = "off"
        self.active = [True] * len(self.experts)
        self.last_new_outs = [None] * len(self.experts)

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        if self.mode == "off":
            return self.stock(hidden_states, top_k_index, top_k_weights)
        grafted = top_k_index >= self.num_stock
        stock_idx = torch.where(grafted, torch.zeros_like(top_k_index),
                                top_k_index)
        stock_w = torch.where(grafted, torch.zeros_like(top_k_weights),
                              top_k_weights)
        with torch.no_grad():
            y = self.stock(hidden_states, stock_idx, stock_w)
        for j, e in enumerate(self.experts):
            self.last_new_outs[j] = None
            if not self.active[j]:
                continue
            is_j = top_k_index == (self.num_stock + j)
            if not bool(is_j.any()):
                continue
            w_j = (top_k_weights * is_j.to(top_k_weights.dtype)).sum(dim=-1)
            ne = e(hidden_states)
            self.last_new_outs[j] = ne
            y = y + w_j.unsqueeze(-1) * ne
        return y.to(hidden_states.dtype)


def install_partition(blocks, fam, site_entries, mode="off") -> list:
    handles = []
    for i, entries in site_entries.items():
        blk = blocks[i]
        sf = blk.experts.forward
        pm_old = getattr(blk, "_plugin_merge", None)
        if pm_old is not None and hasattr(pm_old, "stock"):
            sf = pm_old.stock
        pm = PartitionMerge(sf, [en["expert"] for en in entries],
                            [en["tag"] for en in entries],
                            gate_getter=g0.gate_getter_fam(blk, fam),
                            num_stock=entries[0]["num_stock"])
        pm.mode = mode
        blk._plugin_merge = pm

        def make_fwd(pm=pm):
            def fwd(hidden_states, top_k_index, top_k_weights):
                return pm(hidden_states, top_k_index, top_k_weights)
            return fwd

        handles.append((blk, sf))
        blk.experts.forward = make_fwd()
    return handles


def set_partition_active(handles, active_tags) -> None:
    for blk, _ in handles:
        pm = getattr(blk, "_plugin_merge", None)
        if pm is None:
            continue
        pm.active = [(t in active_tags) for t in pm.tags]


def build_partition_gates(blocks, banks_reg, fam, site_rows):
    """Full-bank PartitionGate at site layers (registry gate elsewhere)."""
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
            gates.append(PartitionGate(bank, fam, stock, n_new=len(rows),
                                       jitter_eps=eps))
            site_banks[i] = bank
        else:
            gates.append(d2.GraftGateD2(bank_reg, fam, stock, jitter_eps=eps))
    return gates, site_banks


def run_state(model, blocks, fam, gates, handles, mode, fn, active_tags=None):
    """Install gates + merge state, run fn, restore (g0.run_in_state pattern)."""
    restored = None
    if gates is not None:
        restored = [(blk, d2.install_graft(blk, fam, g))
                    for blk, g in zip(blocks, gates)]
    if handles is not None:
        if mode == "off":
            d1.set_plugin_mode(handles, "off")
        else:
            d1.set_plugin_mode(handles, mode)
            if active_tags is not None:
                set_partition_active(handles, active_tags)
    try:
        return fn()
    finally:
        if restored is not None:
            for blk, stock in restored:
                d2.uninstall_graft(blk, fam, stock)
        if handles is not None:
            d1.set_plugin_mode(handles, "off")


# ---------------------------------------------------------------------------
# Profile phase: domain profile on both donors → 8 core picks
# ---------------------------------------------------------------------------

def profile_donor(model_path, domains_dir, device, dtype, profile_tokens,
                  seq_len, batch_size, exclude, n_pick, want_homes):
    """Return per-domain best (layer, expert) by home-lift on this donor."""
    model, tok = d2.load_donor(model_path, device, dtype)
    blocks = d2.find_moe_blocks(model)
    fam = d2.family_for(d2.detect_family(model), blocks)
    domains = d0.load_domain_texts(domains_dir)
    names = list(DOMAINS)
    batches = []
    for i, name in enumerate(names):
        batches.extend(d2.texts_to_batches_chunked(
            tok, [domains[name]], seq_len, batch_size, profile_tokens, label=i))
    n_tok = sum(t.numel() for t, _ in batches)
    print(f"[profile] {model_path}: {len(batches)} batches / {n_tok} tok over "
          f"{names}", flush=True)
    stats = d2.collect_route_stats(model, blocks, fam, batches, device,
                                   len(names))
    picks = []
    for home in want_homes:
        code_idx = names.index(home)
        choice, table = d25.choose_donor_expert(
            stats.act_count, stats.tok_count, code_idx, min_rate=0.02,
            table_size=5)
        picks.append({"home": home, "choice": choice,
                      "top_by_lift": table[:5]})
    home_vec, _, mush = stats.home_and_prototypes()
    out = {"model": model_path, "family": fam.name,
           "n_layers": len(blocks), "n_tokens": int(n_tok),
           "mushiness_mean": float(mush.mean()), "picks": picks}
    cores = {}
    for p in picks:
        layer = int(p["choice"]["layer"])
        expert = int(p["choice"]["expert"])
        while (layer, expert) in exclude:  # never re-pick a taken core
            p["choice"] = next((r for r in p["top_by_lift"]
                                if (int(r["layer"]), int(r["expert"]))
                                not in exclude), None)
            if p["choice"] is None:
                raise RuntimeError(f"no free core left for home {p['home']}")
            layer, expert = int(p["choice"]["layer"]), int(p["choice"]["expert"])
        exclude.add((layer, expert))
        w_gu, w_down = d25.extract_donor_core(blocks, layer, expert)
        cores[p["home"]] = {"w_gu": w_gu, "w_down": w_down,
                            "donor_layer": layer, "donor_expert": expert}
        print(f"[profile] {os.path.basename(model_path)} home={p['home']} → "
              f"L{layer} e{expert} lift {p['choice']['lift']}", flush=True)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out, cores


def run_profile(args, out: dict) -> None:
    t0 = time.time()
    domains_dir = args.domains_dir
    os.makedirs(args.plugins_dir, exist_ok=True)
    # Phi family: code/math fixed (the validated v0 picks) + chat + science
    phi_meta, phi_cores = profile_donor(
        args.donor_phi, domains_dir, args.device, args.dtype,
        args.profile_tokens, args.seq_len, args.batch_size,
        exclude={(13, 8), (24, 4)}, n_pick=2, want_homes=("chat", "science"))
    phi_meta["fixed"] = [{"home": "code", "layer": 13, "expert": 8},
                         {"home": "math", "layer": 24, "expert": 4}]
    # OLMoE family: 4 picks by domain profile (web/code/math/chat homes)
    ol_meta, ol_cores = profile_donor(
        args.donor_olmoe, domains_dir, args.device, args.dtype,
        args.profile_tokens, args.seq_len, args.batch_size,
        exclude=set(), n_pick=4, want_homes=("web", "code", "math", "chat"))
    out["profile"] = {"phi": phi_meta, "olmoe": ol_meta}
    # core artifacts
    specs = []
    for home, c in phi_cores.items():
        specs.append({"name": f"phi_{home}", "donor": "phimoe",
                      "home": home, "donor_layer": c["donor_layer"],
                      "donor_expert": c["donor_expert"],
                      "host_layer": int(c["donor_layer"])})
        torch.save({"w_gu": c["w_gu"], "w_down": c["w_down"]},
                   os.path.join(args.plugins_dir,
                                f"phi_{home}_core.pt"))
    # fixed v0 cores (reused plugins carry their cores; refs only)
    for home, layer, expert in (("code", 13, 8), ("math", 24, 4)):
        specs.append({"name": f"phi_{home}", "donor": "phimoe", "home": home,
                      "donor_layer": layer, "donor_expert": expert,
                      "host_layer": layer, "reuse_plugin": "v0"})
    n_dl = ol_meta["n_layers"]
    for home, c in ol_cores.items():
        host = int(round(c["donor_layer"] * 31 / max(1, n_dl - 1)))
        host = max(0, min(31, host))
        specs.append({"name": f"olmoe_{home}", "donor": "olmoe",
                      "home": home, "donor_layer": c["donor_layer"],
                      "donor_expert": c["donor_expert"], "host_layer": host})
        torch.save({"w_gu": c["w_gu"], "w_down": c["w_down"]},
                   os.path.join(args.plugins_dir, f"olmoe_{home}_core.pt"))
    # de-dup names (code/math phi entries appear twice: fixed + profile loop)
    seen, uniq = set(), []
    for s in specs:
        if s["name"] in seen:
            continue
        seen.add(s["name"])
        uniq.append(s)
    out["cores"] = uniq
    out["phase"] = "profile_done"
    out["profile_wall_s"] = round(time.time() - t0, 2)
    dump_json(out, args.out)
    with open(os.path.join(args.plugins_dir, "cores.json"), "w") as f:
        json.dump(uniq, f, indent=1)


# ---------------------------------------------------------------------------
# Fit phase: door25 wrap fit per core (reuses g0.fit_wrap / d6.train_rows_one)
# ---------------------------------------------------------------------------

def load_specs(args):
    p = os.path.join(args.plugins_dir, "cores.json")
    with open(p) as f:
        return json.load(f)


def run_fit(args, out: dict) -> None:
    t0 = time.time()
    specs = [s for s in load_specs(args)
             if not args.wrap_subset or s["name"] in args.wrap_subset.split(",")]
    out["phase"] = "fit_start"
    out["fit_specs"] = specs
    dump_json(out, args.out)
    # v0-reuse specs need no model at all — handle them before any GPU alloc
    fresh = []
    for spec in specs:
        name = spec["name"]
        plugin_path = os.path.join(args.plugins_dir, f"{name}_plugin.pt")
        if os.path.exists(plugin_path) and args.skip_done:
            w_chk, r_chk, _, _ = d6.load_wrap_plugin(plugin_path)
            if d6.wrap_is_finite(w_chk, r_chk):
                print(f"[{name}] plugin exists and finite — skip", flush=True)
                out.setdefault("fitted", {})[name] = {"skipped": True}
                dump_json(out, args.out)
                continue
        if spec.get("reuse_plugin") == "v0":
            v0p = os.path.join(args.v0_plugins,
                               f"{name.split('_', 1)[1]}_plugin.pt")
            wrap_v, row_v, meta_v, spec_v = d6.load_wrap_plugin(v0p)
            if not d6.wrap_is_finite(wrap_v, row_v):
                raise RuntimeError(f"v0 plugin non-finite: {v0p}")
            torch.save({"format": d6.FORMAT, "wrap": wrap_v.state_dict(),
                        "wrap_spec": wrap_v.wrap_spec(), "row": row_v,
                        "meta": {"host_layer": spec["host_layer"], "tag": name,
                                 "kind": "domain", "home": spec["home"],
                                 "donor_layer": spec["donor_layer"],
                                 "donor_expert": spec["donor_expert"],
                                 "reused_from": v0p}},
                       plugin_path)
            print(f"[{name}] reused v0 wrap ({v0p})", flush=True)
            out.setdefault("fitted", {})[name] = {
                "reused_v0": v0p, "host_layer": spec["host_layer"],
                "home": spec["home"]}
            dump_json(out, args.out)
            continue
        fresh.append(spec)
    if not fresh:
        out["phase"] = "fit_done"
        out["fit_wall_s"] = round(time.time() - t0, 2)
        dump_json(out, args.out)
        return
    specs = fresh
    model, tok = d2.load_donor(args.model, args.device, args.dtype)
    blocks = d2.find_moe_blocks(model)
    fam = d2.family_for(d2.detect_family(model), blocks)
    banks_reg, _ = d0.load_registry(args.registry)
    d_model = banks_reg[0].d_model
    domains = d0.load_domain_texts(args.domains_dir)
    seq, bs = args.seq_len, args.fit_batch_size
    if seq * bs not in (512, 1536, 2048):
        raise RuntimeError(f"train B*T {seq * bs} not in the gfx906-safe set")

    home_streams = {}
    for name, text in domains.items():
        tr, he, _ = g0.split_home(tok, text, args.max_tokens, seq, bs)
        home_streams[name] = tr
    base_ids = []
    for name, text in domains.items():
        base_ids.extend(tok.encode(text, add_special_tokens=False))
    base_ids = base_ids[:args.max_tokens * 2]
    base_batches = g0.to_batches(base_ids, 0, seq, bs)

    for spec in specs:
        name = spec["name"]
        plugin_path = os.path.join(args.plugins_dir, f"{name}_plugin.pt")
        if args.skip_done and os.path.exists(plugin_path):
            w_chk, r_chk, _, _ = d6.load_wrap_plugin(plugin_path)
            if d6.wrap_is_finite(w_chk, r_chk):
                print(f"[{name}] plugin exists and finite — skip", flush=True)
                out.setdefault("fitted", {})[name] = {"skipped": True}
                dump_json(out, args.out)
                continue
        if spec.get("reuse_plugin") == "v0":
            v0p = os.path.join(args.v0_plugins, f"{name.split('_', 1)[1]}_plugin.pt")
            wrap_v, row_v, meta_v, spec_v = d6.load_wrap_plugin(v0p)
            if not d6.wrap_is_finite(wrap_v, row_v):
                raise RuntimeError(f"v0 plugin non-finite: {v0p}")
            torch.save({"format": d6.FORMAT, "wrap": wrap_v.state_dict(),
                        "wrap_spec": wrap_v.wrap_spec(), "row": row_v,
                        "meta": {"host_layer": spec["host_layer"], "tag": name,
                                 "kind": "domain", "home": spec["home"],
                                 "donor_layer": spec["donor_layer"],
                                 "donor_expert": spec["donor_expert"],
                                 "reused_from": v0p}},
                       plugin_path)
            print(f"[{name}] reused v0 wrap ({v0p})", flush=True)
            out.setdefault("fitted", {})[name] = {
                "reused_v0": v0p, "host_layer": spec["host_layer"],
                "home": spec["home"]}
            dump_json(out, args.out)
            continue
        core_path = os.path.join(args.plugins_dir, f"{name}_core.pt")
        core = torch.load(core_path, map_location="cpu", weights_only=False)
        wrap = d25.WrappedForeignExpert(d_model, args.d_canon,
                                        core["w_gu"], core["w_down"],
                                        seed=args.seed)
        home_train = home_streams[spec["home"]]
        if not home_train:
            raise RuntimeError(f"[{name}] empty home stream")
        print(f"== [{name}] wrap core {spec['donor']} "
              f"L{spec['donor_layer']} e{spec['donor_expert']} → host "
              f"L{spec['host_layer']} home={spec['home']} "
              f"({sum(t.numel() for t, _ in home_train)} tok, "
              f"{wrap.adapter_param_count()} adapters) ==", flush=True)
        pa = g0.fit_wrap(model, blocks, fam, banks_reg, spec["host_layer"],
                         wrap, home_train, base_batches, args, args.device,
                         steps=args.expert_steps)
        if not pa.get("finite", True):
            print(f"[{name}] NON-FINITE adapters — skip (retry at lr 1e-3)",
                  flush=True)
            out.setdefault("fitted", {})[name] = {
                "error": "non_finite", "lr": args.expert_lr}
            out["phase"] = f"failed_{name}"
            dump_json(out, args.out)
            del wrap
            torch.cuda.empty_cache()
            continue
        # solo row: prototype init + mex recal (owner BAR_OWNER — partition
        # rows must WIN their domain, not just vote membership)
        host = spec["host_layer"]
        site_rows = {host: [torch.zeros(d_model)]}
        gates, site_banks = g0.build_gates(blocks, banks_reg, fam, site_rows)
        handles = g0.install_mix(blocks, fam, {
            host: [{"expert": wrap,
                    "lam": nn.Parameter(torch.ones((), device=args.device)),
                    "tag": name, "num_stock": banks_reg[host].num_experts}]},
            mode="off")
        pairs = g0.run_in_state(
            model, blocks, fam, gates, handles, "off",
            lambda: g0.collect_pairs_fam(
                model, blocks, fam, home_train + base_batches, args.device,
                args.max_cal_pairs, lambda _i, b: 1 if b[1] == 1 else 0))
        h, lab = pairs[host]
        code_h = h[lab > 0.5]
        bank = site_banks[host]
        row_idx = bank.num_experts - 1
        with torch.no_grad():
            if code_h.shape[0] > 0:
                bank.P.data[row_idx] = (code_h.mean(dim=0)
                                        * math.sqrt(bank.d_r))
        rep = d6.train_rows_one(bank, pairs[host], row_idx, args.row_steps,
                                args.row_lr, args.device,
                                reject_w=args.reject_w, mass_w=args.mass_w,
                                owner_target=BAR_OWNER)
        row = bank.P.data[row_idx].float().cpu().clone()
        d1.set_plugin_mode(handles, "off")
        for blk, sf in handles:
            blk.experts.forward = sf
            if hasattr(blk, "_plugin_merge"):
                delattr(blk, "_plugin_merge")
        os.makedirs(args.plugins_dir, exist_ok=True)
        torch.save({"format": d6.FORMAT, "wrap": wrap.state_dict(),
                    "wrap_spec": wrap.wrap_spec(), "row": row,
                    "meta": {"new_idx": row_idx, "host_layer": host,
                             "tag": name, "kind": "domain",
                             "home": spec["home"],
                             "donor_layer": spec["donor_layer"],
                             "donor_expert": spec["donor_expert"],
                             "donor": spec["donor"]}},
                   plugin_path)
        out.setdefault("fitted", {})[name] = {
            "plugin": plugin_path, "host_layer": host,
            "home": spec["home"],
            "phase_a": {k: pa.get(k) for k in ("steps", "lr", "out_pen",
                                               "finite", "wall_s")},
            "ce_hist_tail": pa.get("history", [])[-3:],
            "row_recal": {"first": rep["first"], "last": rep["last"]}}
        out["phase"] = f"fitted_{name}"
        out["fit_wall_s"] = round(time.time() - t0, 2)
        dump_json(out, args.out)
        print(f"[{name}] done: p_home {rep['first']['p_new_code']:.3f} -> "
              f"{rep['last']['p_new_code']:.3f}, p_base "
              f"{rep['last']['p_new_base']:.4f}", flush=True)
        del wrap
        torch.cuda.empty_cache()
    out["phase"] = "fit_done"
    out["fit_wall_s"] = round(time.time() - t0, 2)
    dump_json(out, args.out)


# ---------------------------------------------------------------------------
# Assemble phase: partition bank + closing recal + measurement + verdict
# ---------------------------------------------------------------------------

def ce_batches(model, blocks, batches, device, gates, handles, fam, mode,
               active_tags=None):
    tot, n = 0.0, 0
    for tokens, _ in batches:
        tokens = tokens.to(device)
        lg = run_state(model, blocks, fam, gates, handles, mode,
                       lambda t=tokens: d0.model_logits(model, t),
                       active_tags)
        tot += d0.next_token_ce(lg, tokens) * tokens.shape[0]
        n += tokens.shape[0]
    return tot / max(1, n)


def win_rates(model, blocks, fam, gates, handles, mode, batches, device,
              site_layers, n_new_at, active_tags=None):
    """Routed-share per graft row on each batch set (cross-fire matrix row)."""
    hits = {i: [0.0] * n_new_at[i] for i in site_layers}
    counts = 0
    for tokens, _ in batches:
        tokens = tokens.to(device)
        run_state(model, blocks, fam, gates, handles, mode,
                  lambda t=tokens: d0.model_logits(model, t), active_tags)
        counts += tokens.numel()
        for i in site_layers:
            g = gates[i]
            w = getattr(g, "last_win", None)
            if w is None:
                continue
            for j in range(n_new_at[i]):
                hits[i][j] += float(w[:, j].float().sum())
    return {i: [v / max(1, counts) for v in row] for i, row in hits.items()}


def run_assemble(args, out: dict) -> None:
    t0 = time.time()
    specs = load_specs(args)
    names = [s["name"] for s in specs]
    out["phase"] = "assemble_start"
    dump_json(out, args.out)
    model, tok = d2.load_donor(args.model, args.device, args.dtype)
    blocks = d2.find_moe_blocks(model)
    fam = d2.family_for(d2.detect_family(model), blocks)
    banks_reg, _ = d0.load_registry(args.registry)
    domains = d0.load_domain_texts(args.domains_dir)
    seq, ebs = args.seq_len, args.eval_batch_size

    # plugins / rows / wraps
    wraps, rows, hosts, homes = {}, {}, {}, {}
    for s in specs:
        n = s["name"]
        path = os.path.join(args.plugins_dir, f"{n}_plugin.pt")
        wrap, row, meta, spec_w = d6.load_wrap_plugin(path)
        if not d6.wrap_is_finite(wrap, row):
            raise RuntimeError(f"non-finite plugin {path}")
        wraps[n], rows[n] = wrap.to(args.device), row.to(args.device)
        hosts[n] = int(meta["host_layer"])
        homes[n] = meta["home"]
        print(f"[{n}] host L{hosts[n]} home {homes[n]} donor "
              f"{meta.get('donor', '?')} L{meta.get('donor_layer')} "
              f"e{meta.get('donor_expert')}", flush=True)
    out["bank"] = {n: {"host": hosts[n], "home": homes[n]} for n in names}

    # site layout
    site_rows, site_entries = {}, {}
    for n in names:
        site_rows.setdefault(hosts[n], []).append(rows[n])
        site_entries.setdefault(hosts[n], []).append({
            "expert": wraps[n], "tag": n,
            "num_stock": banks_reg[hosts[n]].num_experts})
    tags_at = {i: [en["tag"] for en in site_entries[i]] for i in site_entries}
    n_new_at = {i: len(en) for i, en in site_entries.items()}

    # data
    dom_batches, dom_eval = {}, {}
    for name in DOMAINS:
        tr, he, _ = g0.split_home(tok, domains[name], args.max_tokens, seq,
                                  args.batch_size)
        _, hev, _ = g0.split_home(tok, domains[name], args.max_tokens, seq,
                                  ebs)
        dom_batches[name] = tr
        dom_eval[name] = hev
    base_ids = []
    for name, text in domains.items():
        base_ids.extend(tok.encode(text, add_special_tokens=False))
    base_ids = base_ids[:args.max_tokens * 2]
    base_eval = g0.to_batches(base_ids, 0, seq, ebs)
    val_path = os.path.join(args.domains_dir, "val_mixed.txt")
    val_eval = []
    if os.path.exists(val_path):
        v_ids = tok.encode(open(val_path).read(), add_special_tokens=False)
        val_eval = g0.to_batches(v_ids[:args.max_tokens], 0, seq, ebs)

    # ---- helpers: live-bank construction + teardown ----
    def teardown(handles):
        for blk, sf in handles:
            blk.experts.forward = sf
            if hasattr(blk, "_plugin_merge"):
                delattr(blk, "_plugin_merge")

    def build_live(live):
        """Partition bank with ONLY `live` graft rows (the rest never appear in
        selection — honest solo / leave-one-out bank compositions)."""
        l_rows, l_entries = {}, {}
        for i in site_rows:
            live_here = [n for n in tags_at[i] if n in live]
            if not live_here:
                continue
            l_rows[i] = [rows[n] for n in live_here]
            l_entries[i] = [{"expert": wraps[n], "tag": n,
                             "num_stock": banks_reg[i].num_experts}
                            for n in live_here]
        gates, banks = build_partition_gates(blocks, banks_reg, fam, l_rows)
        handles = install_partition(blocks, fam, l_entries, mode="off")
        return gates, banks, handles

    # ---- closing row recal under PARTITION delivery (all grafts live) ----
    p_gates, p_banks, p_handles = build_live(set(names))
    joint = []
    for i, name in enumerate(DOMAINS, start=1):
        joint += [(t, i) for t, _ in dom_batches[name]]
    joint += [(t, 0) for t, _ in base_eval]
    print("== closing row recal (partition live, owner 0.85) ==", flush=True)
    site_info = []
    for n in names:
        i = hosts[n]
        home_label = DOMAINS.index(homes[n]) + 1
        site_info.append({"host_i": i, "bank": p_banks[i],
                          "row_idx": p_banks[i].num_experts
                          - n_new_at[i] + tags_at[i].index(n),
                          "home_label": home_label, "tag": n,
                          "owner_target": BAR_OWNER})
    frozen = d6.freeze_wraps(list(wraps.values()))
    pairs_raw = run_state(
        model, blocks, fam, p_gates, p_handles, "dispatch",
        lambda: g0.collect_pairs_fam(model, blocks, fam, joint, args.device,
                                     args.max_cal_pairs,
                                     lambda _i, b: b[1]))
    recal = []
    for si in site_info:
        h, dom = pairs_raw[si["host_i"]]
        lab = (dom == float(si["home_label"])).float()
        rep = d6.train_rows_one(si["bank"], (h, lab), si["row_idx"],
                                args.recal_steps, args.recal_lr, args.device,
                                reject_w=args.reject_w, mass_w=args.mass_w,
                                owner_target=si["owner_target"])
        rep.update({"host_i": si["host_i"], "tag": si["tag"]})
        recal.append(rep)
        print(f"  [recal {si['tag']} L{si['host_i']}] p_home "
              f"{rep['first']['p_new_code']:.3f} -> {rep['last']['p_new_code']:.3f}"
              f"  p_base {rep['first']['p_new_base']:.4f} -> "
              f"{rep['last']['p_new_base']:.4f}", flush=True)
        # push the recal'd row back so every later rebuild sees it
        rows[si["tag"]] = si["bank"].P.data[si["row_idx"]].detach().clone()
    out["closing_recal"] = {"steps": args.recal_steps, "per_site": recal}
    dump_json(out, args.out)
    # rebuild so the gate banks see the recal'd rows
    teardown(p_handles)
    p_gates, p_banks, p_handles = build_live(set(names))

    # ---- overlay (v0 additive) construction ----
    lam_params = {n: nn.Parameter(torch.tensor(1.0, device=args.device))
                  for n in names}

    def build_overlay(lams_dict):
        o_rows = {i: [rows[n] for n in tags_at[i]] for i in site_rows}
        o_gates, o_banks = g0.build_gates(blocks, banks_reg, fam, o_rows)
        o_entries = {i: [{"expert": wraps[n], "tag": n,
                          "lam": lams_dict[n],
                          "num_stock": banks_reg[i].num_experts}
                         for n in tags_at[i]] for i in site_rows}
        o_handles = g0.install_mix(blocks, fam, o_entries, mode="off")
        return o_gates, o_banks, o_handles

    # ---------------- measurement states ----------------
    def measure(gates, handles, mode, tag_name):
        """CE per domain + base/val under one delivery state (fwd only)."""
        rec = {}
        for name in DOMAINS:
            rec[name] = round(ce_batches(model, blocks, dom_eval[name],
                                         args.device, gates, handles, fam,
                                         mode, set(names)), 4)
        rec["base"] = round(ce_batches(model, blocks, base_eval, args.device,
                                       gates, handles, fam, mode,
                                       set(names)), 4)
        if val_eval:
            rec["val"] = round(ce_batches(model, blocks, val_eval, args.device,
                                          gates, handles, fam, mode,
                                          set(names)), 4)
        print(f"  [{tag_name}] " + " ".join(f"{k} {v}" for k, v in rec.items()),
              flush=True)
        return rec

    states = {}
    print("== states: bare / masked / partition-full / overlay =="
          f" (+ {len(names)} solos + {len(names)} leave-one-out) ==", flush=True)
    states["bare"] = measure(None, None, "off", "bare")
    # masked = passport selection on stock rows only, zero delivery
    m_rows = {i: [rows[n] for n in tags_at[i]] for i in site_rows}
    m_gates, _ = g0.build_gates(blocks, banks_reg, fam, m_rows)
    m_handles = g0.install_mix(blocks, fam, {
        i: [{"expert": wraps[n], "tag": n,
             "lam": nn.Parameter(torch.zeros((), device=args.device)),
             "num_stock": banks_reg[i].num_experts}
            for n in tags_at[i]] for i in site_rows}, mode="off")
    states["masked"] = measure(m_gates, m_handles, "additive", "masked")
    teardown(m_handles)
    states["partition_full"] = measure(p_gates, p_handles, "dispatch",
                                       "partition-full")
    o_gates, o_banks, o_handles = build_overlay(lam_params)
    states["overlay_full"] = measure(o_gates, o_handles, "additive",
                                     "overlay-full(lam=1)")
    teardown(o_handles)
    for n in names:
        g_s, b_s, h_s = build_live({n})
        states[f"solo_{n}"] = measure(g_s, h_s, "dispatch", f"solo {n}")
        teardown(h_s)
    for n in names:
        g_m, b_m, h_m = build_live(set(names) - {n})
        states[f"minus_{n}"] = measure(g_m, h_m, "dispatch", f"minus {n}")
        teardown(h_m)
    out["states"] = states

    # cross-fire (routed-share) matrix: rows=graft, cols=domain of tokens
    print("== cross-fire matrix (routed-share, full bank) ==", flush=True)
    xfire = {}
    for name in DOMAINS:
        wr = win_rates(model, blocks, fam, p_gates, p_handles, "dispatch",
                       dom_eval[name], args.device, list(site_rows.keys()),
                       n_new_at, set(names))
        xfire[name] = {}
        for i, row in wr.items():
            for j, v in enumerate(row):
                xfire[name][tags_at[i][j]] = round(v, 4)
        print(f"  xfire[{name}] " + " ".join(
            f"{k} {xfire[name][k]}" for k in names), flush=True)
    out["cross_fire"] = xfire


    # overlay λ fit (v0's additive contract on the same cores) — short joint
    print("== overlay λ fit (v0 additive contract) ==", flush=True)
    o_joint = []
    for i, name in enumerate(DOMAINS, start=1):
        o_joint += [(t, i) for t, _ in dom_batches[name]]
    o_joint += [(t, 0) for t, _ in base_eval]
    o_gates, o_banks, o_handles = build_overlay(lam_params)
    lam_fit = g0.fit_lams(model, blocks, fam,
                          [lam_params[n] for n in names], o_handles,
                          o_gates, o_joint, args.lam_steps, args.lam_lr,
                          args.device)
    out["overlay_lam_fit"] = lam_fit
    out["overlay_lams"] = {n: round(float(lam_params[n].detach().clamp(min=0)),
                                    4) for n in names}
    print("overlay lams:", out["overlay_lams"], flush=True)
    # re-measure overlay with fitted λ (same harness, additive)
    states["overlay_fitted"] = measure(o_gates, o_handles, "additive",
                                       "overlay-fitted")
    teardown(o_handles)
    out["states"] = states
    dump_json(out, args.out)

    # ---------------- verdict math ----------------
    sf, mask, pf = states["bare"], states["masked"], states["partition_full"]
    per_graft = {}
    n_green = 0
    for n in names:
        d = homes[n]
        routed = states["solo_{0}".format(n)][d]
        inbank = pf[d]
        masked = mask[d]
        gain_solo = masked - states[f"solo_{n}"][d]
        gain_inbank = masked - inbank
        per_graft[n] = {
            "home": d, "host": hosts[n],
            "ce_masked": masked, "ce_routed_solo": round(routed, 4),
            "ce_routed_inbank": round(inbank, 4),
            "gain_solo_nats": round(gain_solo, 4),
            "gain_inbank_nats": round(gain_inbank, 4),
            "preserved_pct": (round(100 * gain_inbank / gain_solo, 1)
                              if gain_solo > 1e-6 else None),
            "quality_green": bool(inbank < masked),
        }
        n_green += int(per_graft[n]["quality_green"])
    # interference: harm of ADDING graft i on domain j's CE = CE(j|full) -
    # CE(j|full minus i)  (positive = adding i hurts j)
    interf = {}
    worst = 0.0
    for j in names:
        interf[j] = {}
        for i in names:
            if i == j:
                continue
            harm = pf[homes[j]] - states["minus_" + i][homes[j]]
            interf[j][i] = round(harm, 4)
            worst = max(worst, harm)
    base_tax = (pf["base"] - sf["base"]) / max(1e-6, sf["base"])
    union_bare = sum(sf[d] for d in DOMAINS) / len(DOMAINS)
    union_part = sum(pf[d] for d in DOMAINS) / len(DOMAINS)
    overlay = states.get("overlay_fitted", states["overlay_full"])
    union_overlay = sum(overlay[d] for d in DOMAINS) / len(DOMAINS)
    v = {
        "per_graft": per_graft,
        "n_quality_green": f"{n_green}/{len(names)}",
        "base_ce_bare": sf["base"], "base_ce_masked": mask["base"],
        "base_ce_partition": pf["base"], "base_ce_overlay": overlay["base"],
        "base_tax_pct": round(100 * base_tax, 2),
        "base_tax_pass": bool(base_tax <= BAR_BASE_TAX),
        "union_ce_bare": round(union_bare, 4),
        "union_ce_partition": round(union_part, 4),
        "union_ce_overlay": round(union_overlay, 4),
        "worst_pairwise_harm": round(worst, 4),
        "cross_harm_pass": bool(worst <= BAR_CROSS_HARM),
        "interference_matrix": interf,
        "partition_vs_overlay": {
            "partition_union": round(union_part, 4),
            "overlay_union": round(union_overlay, 4),
            "partition_base": pf["base"], "overlay_base": overlay["base"],
            "winner": ("partition" if union_part < union_overlay
                       else "overlay"),
        },
    }
    all_green = n_green == len(names)
    v["bars_pass"] = bool(all_green and v["base_tax_pass"]
                          and v["cross_harm_pass"])
    v["verdict_text"] = (
        "2n-BANK " + ("VALIDATED" if v["bars_pass"] else "NOT VALIDATED")
        + f" — quality green {n_green}/{len(names)}, base tax "
        f"{v['base_tax_pct']:+.2f}% ({'pass' if v['base_tax_pass'] else 'FAIL'}), "
        f"worst pairwise harm {worst:.4f} nats "
        f"({'pass' if v['cross_harm_pass'] else 'FAIL'}); partition vs overlay "
        f"union {v['partition_vs_overlay']['partition_union']:.4f} vs "
        f"{v['partition_vs_overlay']['overlay_union']:.4f}")
    out["verdict"] = v
    out["phase"] = "done"
    out["wall_s"] = round(time.time() - t0, 2)
    dump_json(out, args.out)
    d6.restore_grad_flags(frozen)
    print("verdict:", v["verdict_text"], flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("profile", "fit", "assemble", "all"),
                    default="all")
    ap.add_argument("--model", default="/data/tliao/Phi-mini-MoE-instruct")
    ap.add_argument("--registry",
                    default="/data/tliao/graft_d2_out/registry_phimoe_f620b32c.pt")
    ap.add_argument("--donor-phi", default="/data/tliao/Phi-mini-MoE-instruct")
    ap.add_argument("--donor-olmoe", default="/data/tliao/OLMoE-1B-7B-0924")
    ap.add_argument("--domains-dir", default="/data/tliao/graft_d2_slice")
    ap.add_argument("--plugins-dir", default="/data/tliao/bank2n_out")
    ap.add_argument("--v0-plugins", default="/data/tliao/grafted_spine_v0_out")
    ap.add_argument("--out", default="/data/tliao/bank2n_out/bank2n.json")
    ap.add_argument("--wrap-subset", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--d-canon", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--fit-batch-size", type=int, default=1,
                    help="train B*T=512 (gfx906-safe; fast per-wrap fits)")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="joint/recal stream batch (B*T=2048)")
    ap.add_argument("--eval-batch-size", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--profile-tokens", type=int, default=6000)
    ap.add_argument("--expert-steps", type=int, default=200)
    ap.add_argument("--expert-lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--out-pen", type=float, default=300.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--row-steps", type=int, default=300)
    ap.add_argument("--row-lr", type=float, default=0.05)
    ap.add_argument("--reject-w", type=float, default=10.0)
    ap.add_argument("--mass-w", type=float, default=2.0)
    ap.add_argument("--max-cal-pairs", type=int, default=4096)
    ap.add_argument("--recal-steps", type=int, default=120)
    ap.add_argument("--recal-lr", type=float, default=0.05)
    ap.add_argument("--lam-steps", type=int, default=50)
    ap.add_argument("--lam-lr", type=float, default=0.05)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.plugins_dir, exist_ok=True)
    out = {"gate": "bank2n-partition", "format": FORMAT,
           "phase": "start", "pid": os.getpid()}
    if args.out and os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
            for k in ("profile", "cores", "fitted", "closing_recal",
                      "states", "cross_fire", "verdict"):
                if prev.get(k):
                    out[k] = prev[k]
        except Exception:
            pass
    dump_json(out, args.out)
    if args.phase in ("profile", "all"):
        run_profile(args, out)
    if args.phase in ("fit", "all"):
        run_fit(args, out)
    if args.phase in ("assemble", "all"):
        run_assemble(args, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
