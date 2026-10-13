#!/usr/bin/env python3
"""v0.6 — the impress-demo: the v0.5 graft stack on a stronger spine.

Spine = Qwen/Qwen1.5-MoE-A2.7B-Chat (14.3B total / 2.7B active, 24L, hidden
2048, 60 experts, top-4, qwen2_moe) — "2x active params so it doesn't loop"
over the Phi-mini-MoE v0.5 spine. Qwen2Moe duck-types as the OLMoE family
through the whole unmodified d0/d1/d2/d25/d6/g0 stack (gate returns
(scores, weights, idx); experts.forward(h, idx, w); find_moe_blocks sees
.gate+.experts; the gfx906 experts shim matches gate_up_proj/down_proj).
That is D2's port lesson: adapt the router once, everything else follows —
and here even the router adapter is already in place (d2.family_for(OLMOE)
reads .top_k/.norm_topk_prob which Qwen2MoeTopKRouter has).

Phases (each a detached GPU job, incremental JSON, skip-done):
  registry  d0 passport registry on the Qwen spine + parity gate (bar 0.95)
  fit       per-wrap lanes: 3 specialty cores (K3 hacking / GLM-Flash
            cybersec / MiMo-Flash writing, DONORS reused from the v0.5
            core_bank .pt files, re-wrapped in the new host geometry d=2048)
            + 2 taught memories (pet_kepler + editor_helix, §8 shaped
            episodes, host-native expert cores per the v0 recipe).
            FEED = 500 steps x bs1 x 512 = 256,000 tok-slots >= 200k
            (door10 certified budget), specialty corpora from
            Dataset/data/specialties/<home>/corpus.txt.
  assemble  joint install + closing row recal (owner 0.85) + BALANCED
            per-home λ fit (the v0.5 starvation lesson: v0.5's pooled union
            λ fit collapsed glm/mimo to λ*=0 — v0.6 fits each plug's λ
            ONLY against its own home windows, then a base-tax guard).

Forked from v05_build.py (same grafted_spine_v0 / door25 / door6 / d0 / d1 /
d2 imports, UNMODIFIED). Chat CLI: lab/chat_v06.py.
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import graft_d0_parity as d0  # noqa: E402
import graft_d1_plugin as d1  # noqa: E402
import graft_d2_crossfamily as d2  # noqa: E402
import door25_wrap_expert as d25  # noqa: E402
import door6_expert_mix as d6  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402

FORMAT = "vesper-v06"
BAR_PARITY_COS = 0.95          # task gate: parity cos >= 0.95 before anything
BAR_OWNER_MEM = 0.85           # v0.5 task line: rows own their home
MIN_FEED_TOKENS = 200_000      # door10 certified calibration budget
BASE_TAX_BAR = 0.05

# gfx906 backward holes are SPINE-SPECIFIC (FAILURES door25-gfx906 family).
# v0.5's HOLEY_BT {256,768,1024,4096} is for Phi-mini-MoE; on the Qwen spine
# a solo shape grid (lab note bw_probe4, GPU 2, CE fwd+bwd through the wrap)
# measured B*T 128/192/256 OK and 320/384/512 FAIL (hipErrorInvalidDeviceFunction).
V06_SAFE_BT = {128, 192, 256}

DEFAULT_MODEL = "/data/tliao/v06_models/Qwen1.5-MoE-A2.7B-Chat"
DEFAULT_REGISTRY = "/data/tliao/v06_out/registry_qwen2moe.pt"
DEFAULT_OUT = "/data/tliao/v06_out/v06_results.json"
DEFAULT_PLUGINS = "/data/tliao/v06_out"

# name, kind, home, host layer on the 24L Qwen spine, core file (or None for
# host-native extract), donor tag, donor layer, donor expert, act
# Host sites depth-matched to donors (v0.5 rule round(L_d/Lmax * (nL-1))):
#   K3   L12/92 -> 3   GLM  L3/44 -> 2   MiMo L1/47 -> 0
# Memories (v0 recipe: extract a host-native expert as the frozen core):
#   mem_pet L8 e13, mem_edit L16 e41 — all sites distinct {0,2,3,8,16}.
V06_WRAPS = [
    ("k3_hack", "domain", "hacking", 3, "k3_hack_core.pt",
     "kimi-k3", 12, 895, None),
    ("glm_cybersec", "domain", "cybersec", 2, "glm_cybersec_core.pt",
     "glm-5.3-flash", 3, 0, None),
    ("mimo_writing", "domain", "writing", 0, "mimo_writing_core.pt",
     "mimo-v2.6-flash", 1, 0, None),
    ("mem_pet", "mem", "pet", 8, None, "qwen2moe-host", 8, 13, "swiglu"),
    ("mem_edit", "mem", "editor", 16, None, "qwen2moe-host", 16, 41, "swiglu"),
]

HOME_LABEL = {"hacking": 1, "cybersec": 2, "writing": 3,
              "pet": 4, "editor": 5}
LAM_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]


def dump_json(out: dict, path: str | None) -> None:
    g0.dump_json(out, path)


def load_specialty_ids(tok, path: str, max_tokens: int) -> list[int]:
    with open(path) as f:
        text = f.read()
    return tok.encode(text, add_special_tokens=False)[:max_tokens]


# ---------------------------------------------------------------------------
# Phase: registry (d0 passport registry + parity gate on the Qwen spine)
# ---------------------------------------------------------------------------

def run_registry(args) -> dict:
    """D0 precompute + parity on the new spine. Parity gate is the FIRST
    bar of the build: logit cos >= 0.95 (stock router vs passport graft)."""
    t0 = time.time()
    out = {"gate": "v06-registry", "format": FORMAT, "phase": "start",
           "pid": os.getpid(), "model": args.model,
           "bar_parity_cos": BAR_PARITY_COS}
    dump_json(out, args.out)

    model, tok = d2.load_donor(args.model, args.device, args.dtype)
    blocks = d2.find_moe_blocks(model)
    fam = d2.family_for(d2.detect_family(model), blocks)
    e, h = d2.gate_of(blocks[0], fam).weight.shape
    print(f"host: {len(blocks)} MoE layers, {e} experts, d_model {h}, "
          f"family {fam.name}, top_k {fam.top_k}", flush=True)

    domains = d0.load_domain_texts(args.domains_dir)
    seq, bs = args.reg_seq_len, args.batch_size
    domain_batches = []
    for i, text in enumerate(domains.values()):
        domain_batches += d0.texts_to_batches(tok, [text], seq, bs,
                                              args.max_tokens, i)
    domain_batches = domain_batches[:args.reg_domain_batches]
    val_text = ""
    val_path = os.path.join(args.domains_dir, "val_mixed.txt")
    if os.path.exists(val_path):
        val_text = open(val_path).read()
    else:
        val_text = " ".join(domains.values())[:50000]
    val_batches = d0.texts_to_batches(tok, [val_text], seq, bs,
                                      args.max_tokens, 0)
    val_batches = val_batches[:args.reg_val_batches]
    print(f"data: {len(domain_batches)} domain batches, "
          f"{len(val_batches)} val batches (seq {seq})", flush=True)

    from types import SimpleNamespace
    ns = SimpleNamespace(
        init="exact", phase="all", passport_dim=None, learn_wq=False,
        recal_rounds=1, recal_steps=args.recal_steps,
        recal_lr=args.recal_lr, recal_kl_w=0.1,
        max_cal_pairs=args.max_cal_pairs, registry=args.registry)
    rep = d0.run_pipeline(model, blocks, domain_batches, val_batches,
                          len(domains), args.device, ns,
                          {"model": args.model, "format": FORMAT,
                           "family": fam.name, "top_k": fam.top_k})
    cos = rep.get("parity", {}).get("logit_cosine_mean")
    out.update(rep)
    out["parity_gate"] = {"logit_cosine_mean": cos,
                          "bar": BAR_PARITY_COS,
                          "pass": bool(cos is not None
                                      and cos >= BAR_PARITY_COS)}
    out["phase"] = "registry_done"
    out["wall_s"] = round(time.time() - t0, 2)
    dump_json(out, args.out)
    print(f"== registry parity cos {cos} (bar {BAR_PARITY_COS}) "
          f"-> {'PASS' if out['parity_gate']['pass'] else 'FAIL'} ==", flush=True)
    return out


# ---------------------------------------------------------------------------
# Phase: fit (one wrap per call; lanes use --wrap-subset)
# ---------------------------------------------------------------------------

def run_fit(args) -> dict:
    t0 = time.time()
    out = {"gate": "v06-fit", "format": FORMAT, "phase": "start",
           "pid": os.getpid(), "recipe": {
               "d_canon": args.d_canon, "lr": args.expert_lr,
               "wd": args.weight_decay, "out_pen": args.out_pen,
               "steps": args.expert_steps,
               "min_feed_tokens": MIN_FEED_TOKENS}}
    dump_json(out, args.out)
    specs = [s for s in V06_WRAPS
             if not args.wrap_subset or s[0] in args.wrap_subset.split(",")]
    model, tok = d2.load_donor(args.model, args.device, args.dtype)
    blocks = d2.find_moe_blocks(model)
    fam = d2.family_for(d2.detect_family(model), blocks)
    banks_reg, _ = d0.load_registry(args.registry)
    if len(banks_reg) != len(blocks):
        raise RuntimeError(f"registry {len(banks_reg)} vs model {len(blocks)}")
    d_model = banks_reg[0].d_model
    print(f"host: {len(blocks)} layers, d_model {d_model}, family {fam.name}",
          flush=True)
    domains = d0.load_domain_texts(args.domains_dir)
    seq, bs = args.seq_len, args.batch_size
    if seq * bs not in V06_SAFE_BT:
        raise RuntimeError(f"train B*T {seq * bs} not in v06 safe set "
                           f"{sorted(V06_SAFE_BT)} (gfx906 backward holes)")
    # FEED budget is measured in tok-slots: at the smaller safe seq the step
    # count must rise so fed = steps*seq*bs stays >= the 200k certified budget.
    steps_eff = max(args.expert_steps,
                    -(-MIN_FEED_TOKENS // max(1, seq * bs)))
    base_ids = []
    for _name, text in domains.items():
        base_ids.extend(tok.encode(text, add_special_tokens=False))
    base_ids = base_ids[:args.max_base_tokens]
    base_batches = g0.to_batches(base_ids, 0, seq, bs)
    episodes = g0.build_episodes()

    for (name, kind, home, host_i, core_file, donor, dlayer, dexp,
         act_force) in specs:
        plugin_path = os.path.join(args.plugins_dir, f"{name}_plugin.pt")
        if args.skip_done and os.path.exists(plugin_path):
            w_chk, r_chk, _, _ = d6.load_wrap_plugin(plugin_path)
            if d6.wrap_is_finite(w_chk, r_chk):
                print(f"[{name}] plugin exists and finite — skip", flush=True)
                out.setdefault("fitted", {})[name] = {"skipped": True}
                dump_json(out, args.out)
                continue
        if core_file:
            core_path = (core_file if os.path.isabs(core_file)
                         else os.path.join(args.cores_dir, core_file))
            if not os.path.exists(core_path):
                raise RuntimeError(f"[{name}] missing core {core_path}")
            core = torch.load(core_path, map_location="cpu", weights_only=False)
            meta = core.get("meta", {})
            w_gu, w_down = core["w_gu"], core["w_down"]
            act = meta.get("act", "swiglu")
            situ_beta = meta.get("situ_beta", 1.0)
            situ_lin = meta.get("situ_linear_beta", 1.0)
        else:
            # §8 shaped memories: host-native expert = the frozen core (v0)
            w_gu, w_down = d25.extract_donor_core(blocks, host_i, dexp)
            meta = {}
            act = act_force or "swiglu"
            situ_beta, situ_lin = 1.0, 1.0
        wrap = d25.WrappedForeignExpert(
            d_model, args.d_canon, w_gu, w_down, seed=args.seed,
            act=act, situ_beta=situ_beta, situ_linear_beta=situ_lin)

        if kind == "domain":
            spec_path = os.path.join(args.specialties_dir, home, "corpus.txt")
            spec_ids = load_specialty_ids(tok, spec_path, args.max_tokens)
            if len(spec_ids) < MIN_FEED_TOKENS * 0.75:
                print(f"[{name}] WARNING: {len(spec_ids)} tokens < 75% of "
                      f"the {MIN_FEED_TOKENS} door10 budget", flush=True)
            home_train = g0.to_batches(spec_ids, 1, seq, bs)
            feed_note = {"corpus": spec_path, "tokens": len(spec_ids)}
        else:
            # §8 shaped-episode recipe: tiled ChatML episodes, bs1x512
            home_train = g0.episode_batches(tok, [episodes[home]], seq, 1)
            feed_note = {"episodes": home, "tokens": int(
                sum(t.numel() for t, _ in home_train))}
        if not home_train:
            raise RuntimeError(f"[{name}] empty home stream")
        fed = steps_eff * seq * bs
        print(f"== [{name}] core {donor} L{dlayer} e{dexp} "
              f"({wrap.core_param_count()} frozen, "
              f"{wrap.adapter_param_count()} adapters, act {act}) "
              f"-> host L{host_i} home={home} "
              f"({len(home_train)} batches, {steps_eff} steps, "
              f"feed {fed} tok-slots) ==", flush=True)
        out.setdefault("feed", {})[name] = {
            **feed_note, "batches": len(home_train), "steps": steps_eff,
            "fed_token_slots": fed,
            "epochs": round(fed / max(1, feed_note["tokens"]), 2)}
        dump_json(out, args.out)

        pa = g0.fit_wrap(model, blocks, fam, banks_reg, host_i, wrap,
                         home_train, base_batches, args, args.device,
                         steps=steps_eff)
        if not pa.get("finite", True):
            print(f"[{name}] NON-FINITE at lr {args.expert_lr} — refit at "
                  f"lr 1e-3 (v0 math-NaN lesson)", flush=True)
            del wrap
            torch.cuda.empty_cache()
            wrap = d25.WrappedForeignExpert(
                d_model, args.d_canon, w_gu, w_down, seed=args.seed,
                act=act, situ_beta=situ_beta, situ_linear_beta=situ_lin)
            args_lr_old = args.expert_lr
            args.expert_lr = 1e-3
            pa = g0.fit_wrap(model, blocks, fam, banks_reg, host_i, wrap,
                             home_train, base_batches, args, args.device,
                             steps=steps_eff)
            args.expert_lr = args_lr_old
            if not pa.get("finite", True):
                out.setdefault("fitted", {})[name] = {
                    "error": "non_finite_twice"}
                out["phase"] = f"failed_{name}"
                dump_json(out, args.out)
                del wrap
                torch.cuda.empty_cache()
                continue

        # registry row: prototype init + mex recal owner 0.85
        site_rows = {host_i: [torch.zeros(d_model)]}
        gates, site_banks = g0.build_gates(blocks, banks_reg, fam, site_rows)
        handles = g0.install_mix(blocks, fam, {
            host_i: [{"expert": wrap,
                      "lam": nn.Parameter(torch.ones((), device=args.device)),
                      "tag": name, "num_stock": banks_reg[host_i].num_experts}]},
            mode="off")
        # pair/eval streams CAPPED (full-pass walkers, v0.5 lesson)
        pair_home = home_train[:args.pair_batches]
        pair_base = base_batches[:args.pair_base]
        ev_home = home_train[:args.eval_batches]
        ev_base = base_batches[:args.eval_base]
        pairs = g0.run_in_state(
            model, blocks, fam, gates, handles, "off",
            lambda: g0.collect_pairs_fam(
                model, blocks, fam, pair_home + pair_base, args.device,
                args.max_cal_pairs, lambda _i, b: 1 if b[1] == 1 else 0))
        h, lab = pairs[host_i]
        home_h = h[lab > 0.5]
        bank = site_banks[host_i]
        row_idx = bank.num_experts - 1
        with torch.no_grad():
            if home_h.shape[0] > 0:
                # Row-scale guard (v0.6 lesson): the v0 prototype
                # (mean_h * sqrt(d_r)) saturates softmax on the Qwen spine
                # (p=1.0 home AND base -> zero recal gradient). Scale the
                # prototype to the layer's mean stock-row norm instead.
                proto = home_h.mean(dim=0)
                stock_norm = float(
                    bank.P.data[:row_idx].float().norm(dim=1).mean())
                pnorm = float(proto.float().norm())
                if pnorm > 0 and stock_norm > 0:
                    proto = proto * (stock_norm / pnorm)
                bank.P.data[row_idx] = proto
        rep = d6.train_rows_one(bank, pairs[host_i], row_idx, args.row_steps,
                                args.row_lr, args.device,
                                reject_w=args.reject_w, mass_w=args.mass_w,
                                owner_target=BAR_OWNER_MEM)
        row = bank.P.data[row_idx].float().cpu().clone()
        q = g0.quality_reads(model, blocks, fam, banks_reg,
                             {host_i: [row]}, handles, gates,
                             ev_home, ev_base, args.device, [name])
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
            "meta": {"new_idx": row_idx, "host_layer": host_i, "tag": name,
                     "kind": kind, "home": home,
                     "donor_layer": dlayer, "donor_expert": dexp,
                     "donor": donor,
                     "core_file": (core_file or f"host-native L{host_i}e{dexp}"),
                     "owner_target": BAR_OWNER_MEM},
        }, plugin_path)
        out.setdefault("fitted", {})[name] = {
            "plugin": plugin_path, "host_layer": host_i,
            "home": home, "donor": [donor, dlayer, dexp],
            "phase_a": {k: pa.get(k) for k in ("steps", "lr", "out_pen",
                                               "finite", "wall_s")},
            "ce_hist_tail": pa.get("history", [])[-3:],
            "row_recal": {"first": rep["first"], "last": rep["last"]},
            "quality": q}
        out["phase"] = f"fitted_{name}"
        out["fit_wall_s"] = round(time.time() - t0, 2)
        dump_json(out, args.out)
        print(f"== [{name}] done: CE home {q['ce_home_masked']:.3f} -> "
              f"{q['ce_home_additive']:.3f} (forced {q['ce_home_forced']:.3f}); "
              f"row p_home {rep['last']['p_new_code']:.3f} "
              f"p_base {rep['last']['p_new_base']:.4f} ==", flush=True)
        del wrap
        torch.cuda.empty_cache()
    out["phase"] = "fit_done"
    out["fit_wall_s"] = round(time.time() - t0, 2)
    dump_json(out, args.out)
    return out


# ---------------------------------------------------------------------------
# Phase: assemble (additive-λ joint + BALANCED per-home λ fit)
# ---------------------------------------------------------------------------

def run_assemble(args) -> dict:
    t0 = time.time()
    prev = {}
    if args.out and os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
        except Exception:
            prev = {}
    out = {"gate": "v06-assemble", "format": FORMAT, "phase": "start",
           "pid": os.getpid(), "delivery": "additive-lambda-balanced",
           "why_balanced": ("v0.5 joint λ fit starved minority homes "
                            "(glm/mimo -> λ*=0); v0.6 fits each plug's λ "
                            "on its OWN home windows only, then base-tax "
                            "guard (v05_lam_refit lesson)")}
    for k in ("joint", "metrics", "verdict"):
        if k in prev and prev[k]:
            out[k] = prev[k]
    dump_json(out, args.out)

    model, tok = d2.load_donor(args.model, args.device, args.dtype)
    blocks = d2.find_moe_blocks(model)
    fam = d2.family_for(d2.detect_family(model), blocks)
    banks_reg, _ = d0.load_registry(args.registry)
    d_model = banks_reg[0].d_model
    domains = d0.load_domain_texts(args.domains_dir)
    seq = args.seq_len
    episodes = g0.build_episodes()

    wrap_specs = []
    for name, kind, home, layer, _cf, _dn, _dl, _de, _act in V06_WRAPS:
        wrap_specs.append((name, kind, layer, home))

    plugins_dir = args.plugins_dir
    wraps, rows, metas = {}, {}, {}
    missing = []
    for name, kind, layer, home in wrap_specs:
        path = os.path.join(plugins_dir, f"{name}_plugin.pt")
        if not os.path.exists(path):
            missing.append(name)
            continue
        wrap, row, meta, spec = d6.load_wrap_plugin(path)
        if not d6.wrap_is_finite(wrap, row):
            raise RuntimeError(f"non-finite plugin {path}")
        if int(meta["host_layer"]) != layer:
            raise RuntimeError(f"[{name}] host layer mismatch")
        # Row-scale guard (v0.6 lesson): the fit-phase prototype init
        # (mean_h * sqrt(d_r)) lands far larger than the stock rows on the
        # Qwen spine (hidden norms ~2x Phi's), so softmax saturates at
        # p=1.0 on home AND base — zero gradient, reject/mass losses stuck
        # at 10.045 (see fit_*.json row_recal first==last). Rescale each
        # graft row to the layer's mean stock-row norm before the closing
        # recal so train_rows_one gets real gradient.
        stock_norm = float(banks_reg[layer].P.data.float().norm(dim=1).mean())
        row_norm = float(row.float().norm())
        if row_norm > 0 and stock_norm > 0:
            row = row * (stock_norm / row_norm)
        wraps[name], rows[name], metas[name] = wrap, row, meta
        print(f"[{name}] plugin host L{layer} kind {kind} home {home} "
              f"act {spec.get('act', 'swiglu')}", flush=True)
    if missing:
        raise RuntimeError(f"missing plugins (fit them first): {missing}")
    out["fit"] = {"plugins": {n: metas[n] for n in wraps},
                  "wrap_specs": {n: wraps[n].wrap_spec() for n in wraps}}
    dump_json(out, args.out)

    # ---- joint install ----
    site_rows, site_entries = {}, {}
    lams = []
    for name, kind, layer, home in wrap_specs:
        lam = nn.Parameter(torch.tensor(float(args.lam_init),
                                        device=args.device))
        lams.append(lam)
        site_rows.setdefault(layer, []).append(rows[name].to(args.device))
        site_entries.setdefault(layer, []).append({
            "expert": wraps[name].to(args.device), "lam": lam, "tag": name,
            "num_stock": banks_reg[layer].num_experts})
    gates, site_banks = g0.build_gates(blocks, banks_reg, fam, site_rows)
    handles = g0.install_mix(blocks, fam, site_entries, mode="off")
    row_idx = {}
    for name, kind, layer, home in wrap_specs:
        bank = site_banks[layer]
        tags_here = [en["tag"] for en in site_entries[layer]]
        row_idx[name] = bank.num_experts - len(tags_here) + tags_here.index(name)
    print("installed rows:", row_idx, flush=True)
    dump_json(out, args.out)

    # ---- joint data (5 homes + base) ----
    def home_batches(key, cap=None):
        if key in ("pet", "editor"):
            bl = g0.episode_batches(tok, [episodes[key]], seq,
                                    args.batch_size)
            raw = [(t, HOME_LABEL[key]) for t, _ in bl]
        else:
            ids = load_specialty_ids(
                tok, os.path.join(args.specialties_dir, key, "corpus.txt"),
                args.max_tokens)
            raw = [(t, HOME_LABEL[key]) for t, _ in g0.to_batches(
                ids, 1, seq, args.batch_size)]
        n = cap if cap is not None else args.joint_batches_per_home
        return raw[:n] if n else raw

    base_ids = []
    for _name, text in domains.items():
        base_ids.extend(tok.encode(text, add_special_tokens=False))
    base_ids = base_ids[:args.max_base_tokens]
    base_batches = g0.to_batches(base_ids, 0, seq, args.batch_size)

    joint_batches = []
    for _n, _k, _l, home in wrap_specs:
        joint_batches += home_batches(home)
    joint_batches += [(t, 0) for t, _ in base_batches[:args.pair_base]]

    site_info = []
    for name, kind, layer, home in wrap_specs:
        site_info.append({
            "host_i": layer, "bank": site_banks[layer], "row_idx": row_idx[name],
            "home_label": HOME_LABEL[home], "tag": name,
            "owner_target": BAR_OWNER_MEM})
    frozen = d6.freeze_wraps(list(wraps.values()))
    js_path = os.path.join(plugins_dir, "joint_state.pt")
    prev_joint = prev.get("joint") or {}
    if (os.path.exists(js_path) and prev_joint.get("lam_fit")
            and prev_joint.get("lams_final")):
        js = torch.load(js_path, map_location="cpu", weights_only=False)
        lay_of = {n: l for n, k, l, h in wrap_specs}
        with torch.no_grad():
            for name, r in js["rows"].items():
                if name not in row_idx:
                    continue
                site_banks[lay_of[name]].P.data[row_idx[name]] = r.to(
                    site_banks[lay_of[name]].P.device).to(
                        site_banks[lay_of[name]].P.dtype)
            for lam, v in zip(lams, js["lams"]):
                lam.copy_(torch.tensor(float(v)))
        out["joint"] = {**prev_joint, "row_idx": row_idx}
        print("== joint state restored from joint_state.pt ==", flush=True)
        dump_json(out, args.out)
    else:
        print("== closing row recal (all plugs live, owner 0.85) ==", flush=True)
        recal = g0.closing_row_recal(model, blocks, fam, handles, gates,
                                     site_info, joint_batches, args.device,
                                     args.recal_steps, args.recal_lr, args)
        out["joint"] = {"row_idx": row_idx, "lam_init": args.lam_init,
                        "row_recal": recal}
        dump_json(out, args.out)

        print("== balanced per-home lambda fit ==", flush=True)
        lam_fit = fit_lams_balanced(
            model, blocks, fam, lams, handles, gates, wrap_specs,
            home_batches, base_batches, args)
        out["joint"]["lam_fit"] = lam_fit
        out["joint"]["lams_final"] = {
            n: round(float(lams[i].detach().clamp(min=0)), 4)
            for i, (n, *_rest) in enumerate(wrap_specs)}
        torch.save({
            "rows": {n: site_banks[{nn: l for nn, k, l, h in wrap_specs}[n]]
                     .P.data[row_idx[n]].cpu() for n in row_idx},
            "lams": [float(p.detach().clamp(min=0)) for p in lams],
            "order": [n for n, *_ in wrap_specs],
        }, js_path)
        out["phase"] = "joint_done"
        dump_json(out, args.out)
    for en_l in site_entries.values():
        for en in en_l:
            en["lam"] = torch.tensor(float(en["lam"].detach().clamp(min=0)),
                                     device=args.device)
    handles = g0.install_mix(blocks, fam, site_entries, mode="off")
    dump_json(out, args.out)
    d6.restore_grad_flags(frozen)

    # ---- quality reads: per-home CE delivered vs masked + base tax ----
    print("== state measurements ==", flush=True)
    ev = {home: home_batches(home, cap=args.eval_batches)
          for _n, _k, _l, home in wrap_specs}
    ev["base"] = [(t, 0) for t, _ in base_batches[:args.eval_base]]

    def ce_masked(batches):
        return g0.run_in_state(
            model, blocks, fam, None, None, "off",
            lambda: d1.ce_in_state(model, blocks, batches, args.device,
                                   None, None, "off"))

    def ce_g(batches):
        return g0.run_in_state(
            model, blocks, fam, gates, handles, "both",
            lambda: d1.ce_in_state(model, blocks, batches, args.device,
                                   None, None, "off"))

    home_ce = {}
    for tag, batches in ev.items():
        bm, bg = ce_masked(batches), ce_g(batches)
        home_ce[tag] = {"masked": round(bm, 4), "grafted": round(bg, 4),
                        "delta": round(bg - bm, 4)}
        print(f"  CE {tag:10s} masked {bm:.4f} -> grafted {bg:.4f}", flush=True)
    out["metrics"] = {"home_ce": home_ce}
    verdict = {"all_homes_improve": all(v["delta"] < 0 for k, v in home_ce.items()
                                        if k != "base"),
               "base_tax_nats": home_ce["base"]["delta"],
               "base_tax_bar": BASE_TAX_BAR}
    verdict["pass"] = (verdict["all_homes_improve"]
                       and verdict["base_tax_nats"] <= verdict["base_tax_bar"])
    out["verdict"] = verdict
    out["phase"] = "assemble_done"
    out["wall_s"] = round(time.time() - t0, 2)
    dump_json(out, args.out)
    print("== verdict:", json.dumps(verdict), "==", flush=True)
    return out


def fit_lams_balanced(model, blocks, fam, lams, handles, gates, wrap_specs,
                      home_batches, base_batches, args) -> dict:
    """Balanced per-home λ fit — each plug's λ is grid-fit ONLY against its
    own home windows (all other plugs live at their current λ), so minority
    homes cannot be starved by the union objective (the v0.5 failure). All
    measurements are no-grad CE reads (also avoids the gfx906 lam backward
    hole that bank2n hit). Then a base-tax guard shrinks the plug with the
    least home-gain per unit tax until the total base tax is inside
    BASE_TAX_BAR."""
    names = [n for n, *_ in wrap_specs]
    homes = {n: h for n, _k, _l, h in wrap_specs}
    state = {n: 0.5 for n in names}

    def set_lams(d):
        for i, n in enumerate(names):
            lams[i].data.fill_(float(d[n]))

    def ce(batches, which="both"):
        if not batches:
            return 0.0
        return g0.run_in_state(
            model, blocks, fam, gates, handles, which,
            lambda: d1.ce_in_state(model, blocks, batches, args.device,
                                   None, None, "off"))

    home_ev = {n: home_batches(homes[n], cap=args.lam_home_batches)
               for n in names}
    base_ev = [(t, 0) for t, _ in base_batches[:args.lam_base_batches]]

    set_lams({n: 0.0 for n in names})
    ce_mask = {n: ce(home_ev[n], "off") for n in names}
    ce_base_mask = ce(base_ev, "off")

    report = {"grid": LAM_GRID, "order": names, "per_home": {},
              "ce_home_masked": ce_mask, "ce_base_masked": ce_base_mask}

    # round-robin per-home grid: fit each plug on its OWN home only
    for n in names:
        best_lam, best_ce = 0.0, None
        curve = []
        for lam in LAM_GRID:
            trial = dict(state)
            trial[n] = lam
            set_lams(trial)
            v = ce(home_ev[n])
            curve.append({"lam": lam, "ce": round(v, 4)})
            if best_ce is None or v < best_ce:
                best_ce, best_lam = v, lam
        state[n] = best_lam
        set_lams(state)
        report["per_home"][n] = {
            "home": homes[n], "curve": curve,
            "masked": round(ce_mask[n], 4),
            "best_lam": best_lam, "best_ce": round(best_ce, 4),
            "gain": round(ce_mask[n] - best_ce, 4)}
        print(f"  [lam {n:14s} home {homes[n]:8s}] "
              f"masked {ce_mask[n]:.4f} -> {best_ce:.4f} at λ*={best_lam}",
              flush=True)

    # base-tax guard: shrink least home-gain-per-tax plug until inside budget
    set_lams(state)
    base_now = ce(base_ev)
    tax = base_now - ce_base_mask
    guard = []
    while tax > BASE_TAX_BAR and any(state[n] > 0 for n in names):
        live = [n for n in names if state[n] > 0]
        # efficiency = home gain per unit of λ (proxy for tax contribution)
        def eff(n):
            g = report["per_home"][n]["gain"]
            return g / max(state[n], 1e-6)
        victim = min(live, key=eff)
        old = state[victim]
        state[victim] = max(0.0, old - 0.25)
        set_lams(state)
        base_now = ce(base_ev)
        new_tax = base_now - ce_base_mask
        guard.append({"shrink": victim, "from": old, "to": state[victim],
                      "base_tax": round(new_tax, 4)})
        print(f"  [base-tax] shrink {victim} {old}->{state[victim]} "
              f"tax {tax:.4f}->{new_tax:.4f}", flush=True)
        tax = new_tax
    report["base_tax_guard"] = guard
    report["base_tax_final"] = round(tax, 4)
    report["lams_final"] = {n: state[n] for n in names}
    set_lams(state)
    return report


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("registry", "fit", "assemble", "all"),
                    default="all")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--domains-dir", default="/data/tliao/graft_d2_slice")
    ap.add_argument("--specialties-dir",
                    default="/data/tliao/vesper_sft/VesperLM/Dataset/data/specialties")
    ap.add_argument("--cores-dir", default="/data/tliao/v05_cores")
    ap.add_argument("--plugins-dir", default=DEFAULT_PLUGINS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--wrap-subset", default="",
                    help="comma list of V06_WRAPS names (lanes)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--d-canon", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=256,
                    help="v06 safe B*T set {128,192,256} on the Qwen spine "
                         "(bw_probe4: 320/384/512 hit gfx906 backward holes)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="bs1x512 co-tenant-safe shape (bank2n lesson)")
    ap.add_argument("--max-tokens", type=int, default=300000)
    ap.add_argument("--max-base-tokens", type=int, default=40000)
    ap.add_argument("--expert-steps", type=int, default=500,
                    help="500 x 512 = 256k fed tok-slots >= 200k")
    ap.add_argument("--expert-lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--out-pen", type=float, default=300.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-done", action="store_true")
    # registry
    ap.add_argument("--recal-steps", type=int, default=200)
    ap.add_argument("--recal-lr", type=float, default=0.05)
    ap.add_argument("--max-cal-pairs", type=int, default=1024)
    ap.add_argument("--reg-seq-len", type=int, default=256,
                    help="registry runs smaller batches: vocab 152k logits "
                         "OOM at seq 512 (26.6GB weights + big logits)")
    ap.add_argument("--reg-domain-batches", type=int, default=40)
    ap.add_argument("--reg-val-batches", type=int, default=16)
    # row recal
    ap.add_argument("--row-steps", type=int, default=80)
    ap.add_argument("--row-lr", type=float, default=0.05)
    ap.add_argument("--reject-w", type=float, default=10.0)
    ap.add_argument("--mass-w", type=float, default=2.0)
    # assemble
    ap.add_argument("--lam-init", type=float, default=0.5)
    ap.add_argument("--lam-home-batches", type=int, default=16)
    ap.add_argument("--lam-base-batches", type=int, default=16)
    ap.add_argument("--pair-batches", type=int, default=24)
    ap.add_argument("--pair-base", type=int, default=12)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--eval-base", type=int, default=8)
    ap.add_argument("--joint-batches-per-home", type=int, default=12)
    args = ap.parse_args()

    if args.phase == "registry":
        run_registry(args)
    elif args.phase == "fit":
        run_fit(args)
    elif args.phase == "assemble":
        run_assemble(args)
    else:
        run_registry(args)
        run_fit(args)
        run_assemble(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
