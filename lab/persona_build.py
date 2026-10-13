#!/usr/bin/env python3
"""persona_build — teach the Vesper persona memory pack on both shipped hosts.

The lab/memories/persona_v1.json pack (6 permanent "about the user" facts) is
built exactly like pet_kepler, one shaped episodic plug per fact:

  fit       host-native expert core + rank-64 width-norm wrap (wd 0, task-CE
            + out_pen 300, lr 3e-3) on the episode's ChatML home text, then a
            registry row with the v0.6 ROW-SCALE GUARD (180a612: prototype =
            mean_h rescaled to the layer's mean stock-row norm — mean_h*sqrt(d_r)
            saturates the passport softmax and kills the recal gradient) and a
            solo mex recal at owner_mass 0.85.
  assemble  install the 6 new rows alongside the shipped bank (v0.5: 8 plugs
            incl. the mem trio; v0.6: 5 plugs incl. mem_pet/mem_edit) with the
            shipped rows/lams FROZEN at their joint_state values (the pack is
            additive), then per-row guard-checkpointed sibling recal (the
            715ad33 recipe): own home tiled home_weight x3 + sibling mem
            episodes (capped 4 batches each) + base on the reject side,
            reject_w 15 (targeted pass-2 — 40 at 2.5:1 reject:home
            over-suppressed the row), mass_w 2.0, owner_target 0.85, 30-step
            chunks — after every chunk the row's own greedy fact probe and the
            sacred Kepler probe must hold and own-home CE (tau=0.5 delivery)
            must stay within +0.02 of baseline; first guard trip early-stops
            and the best passing chunk is kept (margin reads recorded, never
            tripping — fp16 margins on near-tied answer slots flap).
            Then a short balanced per-home lam grid for the new rows (own home
            only + base-tax guard, the v0.5 starvation lesson) and a probe pass.
  probes    the shipped-bar verification battery lives in lab/persona_probe.py.

State written (untracked artifacts dir): <plugins>/mem_*_plugin.pt (6) and
<plugins>/joint_state_persona.pt — loaded by chat_v05.py / chat_v06.py under
the default --mem-pack persona.  joint_state.pt is never rewritten.

Usage (box, one lane per GPU):
  python3 lab/persona_build.py --spine phi  --phase fit --wrap-subset mem_poi
  python3 lab/persona_build.py --spine qwen --phase assemble
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("ROCBLAS_TENSILE_LIBPATH",
                      "/opt/rocm-6.2.4/lib/rocblas/library")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import graft_d0_parity as d0  # noqa: E402
import graft_d1_plugin as d1  # noqa: E402
import graft_d2_crossfamily as d2  # noqa: E402
import door25_wrap_expert as d25  # noqa: E402
import door6_expert_mix as d6  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402
import mem_pack as mp  # noqa: E402

MIN_FEED_TOKENS = 200_000      # door10 certified calibration budget
BAR_OWNER_MEM = 0.85           # g3_owner_mass_fix closing target
GCE_GUARD = 0.02               # own-home CE regression band (715ad33)
BASE_TAX_BAR = 0.05
F_MARGIN_GUARD = 0.0           # taught-fact logit margin must stay > this

SPINES = {
    "phi": {
        "model": "/data/tliao/Phi-mini-MoE-instruct",
        "registry": "/data/tliao/graft_d2_out/registry_phimoe_f620b32c.pt",
        "plugins": "/data/tliao/v05_out",
        "domains": "/data/tliao/graft_d2_slice",
        "specialties": "/data/tliao/vesper_sft/VesperLM/Dataset/data/specialties",
        "seq_len": 512,
        "batch_size": 1,
        "safe_bt": {512, 1536, 2048},        # v0.5 HOLEY_BT complement at bs1
        "home_labels": {"code": 1, "math": 2, "pet": 3, "editor": 4,
                        "words": 5, "hacking": 6, "cybersec": 7, "writing": 8},
        # shipped v0.5 bank (name, kind, home, host layer) — loaded frozen
        "existing": [
            ("code", "code", "code", 13), ("math", "math", "math", 24),
            ("mem_pet", "mem", "pet", 6), ("mem_edit", "mem", "editor", 19),
            ("mem_words", "mem", "words", 27),
            ("k3_hack", "domain", "hacking", 4),
            ("glm_cybersec", "domain", "cybersec", 2),
            ("mimo_writing", "domain", "writing", 1),
        ],
    },
    "qwen": {
        "model": "/data/tliao/v06_models/Qwen1.5-MoE-A2.7B-Chat",
        "registry": "/data/tliao/v06_out/registry_qwen2moe.pt",
        "plugins": "/data/tliao/v06_out",
        "domains": "/data/tliao/graft_d2_slice",
        "specialties": "/data/tliao/vesper_sft/VesperLM/Dataset/data/specialties",
        "seq_len": 256,
        "batch_size": 1,
        "safe_bt": {128, 192, 256},          # v0.6 measured set
        "home_labels": {"hacking": 1, "cybersec": 2, "writing": 3,
                        "pet": 4, "editor": 5},
        "existing": [
            ("k3_hack", "domain", "hacking", 3),
            ("glm_cybersec", "domain", "cybersec", 2),
            ("mimo_writing", "domain", "writing", 0),
            ("mem_pet", "mem", "pet", 8), ("mem_edit", "mem", "editor", 16),
        ],
    },
}
for _sp, _cfg in SPINES.items():
    _labels = dict(_cfg["home_labels"])
    _n = max(_labels.values()) + 1
    for _i, _h in enumerate(mp.homes()):
        _labels[_h] = _n + _i
    _cfg["home_labels"] = _labels


def dump_json(out: dict, path: str | None) -> None:
    g0.dump_json(out, path)


def home_stream(tok, home, cfg, episodes, seq, bs):
    """Batches labeled with the home's label (persona episodes or legacy)."""
    lab = cfg["home_labels"][home]
    if home in episodes:
        bl = g0.episode_batches(tok, [episodes[home]], seq, bs)
        return [(t, lab) for t, _ in bl]
    if home in ("pet", "editor", "words"):
        eps = g0.build_episodes()
        bl = g0.episode_batches(tok, [eps[home]], seq, bs)
        return [(t, lab) for t, _ in bl]
    if home in ("code", "math"):
        domains = d0.load_domain_texts(cfg["domains"])
        tr, _, _ = g0.split_home(tok, domains[home], 300000, seq, bs)
        return [(t, lab) for t, _ in tr]
    ids = open(os.path.join(cfg["specialties"], home, "corpus.txt")).read()
    ids = tok.encode(ids, add_special_tokens=False)[:300000]
    return [(t, lab) for t, _ in g0.to_batches(ids, 1, seq, bs)]


def base_batches(tok, cfg, seq, bs, max_tokens=40000):
    domains = d0.load_domain_texts(cfg["domains"])
    ids = []
    for _n, text in domains.items():
        ids.extend(tok.encode(text, add_special_tokens=False))
    return g0.to_batches(ids[:max_tokens], 0, seq, bs)


# ---------------------------------------------------------------------------
# taught-fact margin guards (cheap teacher-forced reads)
# ---------------------------------------------------------------------------

@torch.no_grad()
def fact_margin(logits_fn, tok, prompt, stated, correction, device):
    """logit(good) - logit(bad) at the first divergence of the ChatML turns
    (the memory_shaping.delta_margin read, chatml-encoded)."""
    ps = g0.chatml_ids(tok, prompt, stated)
    pc = g0.chatml_ids(tok, prompt, correction)
    d = 0
    while d < min(len(ps), len(pc)) and ps[d] == pc[d]:
        d += 1
    if d >= len(ps) or d >= len(pc):
        return None
    x = torch.tensor([ps[:d]], dtype=torch.long, device=device)
    if x.numel() == 0:
        return None
    lg = logits_fn(x)[0, -1].float()
    return float(lg[pc[d]] - lg[ps[d]])


def pack_facts(pack=None):
    """[(tag, prompt, stated, correction)] for the persona teach deltas."""
    pack = pack or mp.load_pack()
    out = []
    for f in pack["facts"]:
        for row in f["teach"]:
            out.append((f["tag"], row["prompt"], row["stated"],
                        row["correction"]))
    return out


def kepler_fact():
    return [("mem_pet", "What is my cat's name?", "Your cat is named Orion.",
             "Your cat is named Kepler.")]


@torch.no_grad()
def guard_margins(logits_fn, tok, device, facts=None):
    """min margin per taught fact tag (over its deltas); None = unreadable."""
    facts = facts if facts is not None else (kepler_fact() + pack_facts())
    per = {}
    for tag, prompt, stated, corr in facts:
        m = fact_margin(logits_fn, tok, prompt, stated, corr, device)
        if m is None:
            continue
        per[tag] = min(per.get(tag, 1e9), m)
    return {k: round(v, 4) for k, v in per.items()}


@torch.no_grad()
def fact_probe(logits_fn, tok, device, question, expect, max_new=12):
    """Greedy taught-fact probe (the 715ad33 Kepler-guard shape)."""
    text, _ids = g0.gen_reply(tok, logits_fn,
                              [{"role": "user", "content": question}],
                              max_new, device)
    flat = " ".join(str(text).split()).lower()
    return any(w.lower() in flat for w in expect), flat[:60]


# ---------------------------------------------------------------------------
# Phase: fit — one persona wrap per call
# ---------------------------------------------------------------------------

def run_fit(args, cfg) -> dict:
    t0 = time.time()
    tag_sel = [s.strip() for s in args.wrap_subset.split(",") if s.strip()] \
        if args.wrap_subset else None
    specs = [(t, h, L, e) for (t, h, L, e) in mp.plugs_for(args.spine)
             if not tag_sel or t in tag_sel]
    out = {"gate": "persona-fit", "spine": args.spine, "phase": "start",
           "pid": os.getpid(), "specs": specs,
           "recipe": {"d_canon": 64, "wd": 0.0, "out_pen": 300.0,
                      "lr": args.expert_lr, "owner_target": BAR_OWNER_MEM,
                      "row_scale_guard": True}}
    dump_json(out, args.out)
    model, tok = d2.load_donor(cfg["model"], args.device, args.dtype)
    blocks = d2.find_moe_blocks(model)
    fam = d2.family_for(d2.detect_family(model), blocks)
    banks_reg, _ = d0.load_registry(cfg["registry"])
    if len(banks_reg) != len(blocks):
        raise RuntimeError("registry/model layer mismatch")
    d_model = banks_reg[0].d_model
    seq, bs = cfg["seq_len"], cfg["batch_size"]
    if seq * bs not in cfg["safe_bt"]:
        raise RuntimeError(f"B*T {seq * bs} not safe on {args.spine}")
    steps_eff = max(args.expert_steps,
                    -(-MIN_FEED_TOKENS // max(1, seq * bs)))
    episodes = mp.episodes()
    base = base_batches(tok, cfg, seq, bs)

    for name, home, host_i, dexp in specs:
        plugin_path = os.path.join(args.plugins_dir, f"{name}_plugin.pt")
        if args.skip_done and os.path.exists(plugin_path):
            w, r, _, _ = d6.load_wrap_plugin(plugin_path)
            if d6.wrap_is_finite(w, r):
                print(f"[{name}] plugin exists and finite — skip", flush=True)
                out.setdefault("fitted", {})[name] = {"skipped": True}
                dump_json(out, args.out)
                continue
        w_gu, w_down = d25.extract_donor_core(blocks, host_i, dexp)
        wrap = d25.WrappedForeignExpert(d_model, 64, w_gu, w_down,
                                        seed=args.seed, act="swiglu")
        home_train = g0.episode_batches(tok, [episodes[home]], seq, 1)
        if not home_train:
            raise RuntimeError(f"[{name}] empty home stream")
        print(f"== [{name}] host-native core L{host_i} e{dexp} "
              f"({wrap.core_param_count()} frozen, "
              f"{wrap.adapter_param_count()} adapters) home={home} "
              f"({len(home_train)} batches, {steps_eff} steps) ==", flush=True)
        pa = g0.fit_wrap(model, blocks, fam, banks_reg, host_i, wrap,
                         home_train, base, args, args.device,
                         steps=steps_eff)
        if not pa.get("finite", True):
            print(f"[{name}] NON-FINITE at lr {args.expert_lr} — refit 1e-3",
                  flush=True)
            del wrap
            torch.cuda.empty_cache()
            wrap = d25.WrappedForeignExpert(d_model, 64, w_gu, w_down,
                                            seed=args.seed, act="swiglu")
            old = args.expert_lr
            args.expert_lr = 1e-3
            pa = g0.fit_wrap(model, blocks, fam, banks_reg, host_i, wrap,
                             home_train, base, args, args.device,
                             steps=steps_eff)
            args.expert_lr = old
            if not pa.get("finite", True):
                out.setdefault("fitted", {})[name] = {
                    "error": "non_finite_twice"}
                out["phase"] = f"failed_{name}"
                dump_json(out, args.out)
                del wrap
                torch.cuda.empty_cache()
                continue
        # registry row: prototype with the v0.6 ROW-SCALE GUARD + mex recal
        site_rows = {host_i: [torch.zeros(d_model)]}
        gates, site_banks = g0.build_gates(blocks, banks_reg, fam, site_rows)
        handles = g0.install_mix(blocks, fam, {
            host_i: [{"expert": wrap,
                      "lam": nn.Parameter(torch.ones((), device=args.device)),
                      "tag": name, "num_stock": banks_reg[host_i].num_experts}]},
            mode="off")
        pair_home = home_train[:args.pair_batches]
        pair_base = base[:args.pair_base]
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
                proto = home_h.mean(dim=0)
                stock_norm = float(
                    bank.P.data[:row_idx].float().norm(dim=1).mean())
                pnorm = float(proto.float().norm())
                if pnorm > 0 and stock_norm > 0:
                    proto = proto * (stock_norm / pnorm)
                bank.P.data[row_idx] = proto
        rep = d6.train_rows_one(bank, pairs[host_i], row_idx,
                                args.row_steps, args.row_lr, args.device,
                                reject_w=args.reject_w, mass_w=2.0,
                                owner_target=BAR_OWNER_MEM)
        row = bank.P.data[row_idx].float().cpu().clone()
        q = g0.quality_reads(model, blocks, fam, banks_reg,
                             {host_i: [row]}, handles, gates,
                             home_train[:args.eval_batches],
                             base[:args.eval_base], args.device, [name])
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
                     "kind": "mem", "home": home,
                     "donor": f"{args.spine}-host",
                     "donor_layer": host_i, "donor_expert": dexp,
                     "registry": cfg["registry"],
                     "owner_target": BAR_OWNER_MEM,
                     "mem_pack": mp.load_pack()["pack"]},
        }, plugin_path)
        out.setdefault("fitted", {})[name] = {
            "plugin": plugin_path, "host_layer": host_i,
            "home": home, "steps": steps_eff,
            "phase_a": {k: pa.get(k) for k in
                        ("steps", "lr", "out_pen", "finite", "wall_s")},
            "ce_hist_tail": pa.get("history", [])[-3:],
            "row_recal": {"first": rep["first"], "last": rep["last"]},
            "quality": q}
        out["phase"] = f"fitted_{name}"
        dump_json(out, args.out)
        print(f"== [{name}] done: CE home {q['ce_home_masked']:.3f} -> "
              f"{q['ce_home_additive']:.3f}; row p_home "
              f"{rep['last']['p_new_code']:.3f} p_base "
              f"{rep['last']['p_new_base']:.4f} ==", flush=True)
        del wrap
        torch.cuda.empty_cache()
    out["phase"] = "fit_done"
    out["wall_s"] = round(time.time() - t0, 2)
    dump_json(out, args.out)
    return out


# ---------------------------------------------------------------------------
# Phase: assemble — additive install + guard-checkpointed sibling recal
# ---------------------------------------------------------------------------

def run_assemble(args, cfg) -> dict:
    t0 = time.time()
    pack = mp.load_pack()
    persona = mp.plugs_for(args.spine)
    existing = cfg["existing"]
    all_specs = [(n, k, h, L) for (n, k, h, L) in existing] + \
                [(t, "mem", h, L) for (t, h, L, _e) in persona]
    out = {"gate": "persona-assemble", "spine": args.spine, "phase": "start",
           "pid": os.getpid(), "pack": pack["pack"],
           "persona_rows": [t for t, *_ in persona]}
    dump_json(out, args.out)

    model, tok = d2.load_donor(cfg["model"], args.device, args.dtype)
    blocks = d2.find_moe_blocks(model)
    fam = d2.family_for(d2.detect_family(model), blocks)
    banks_reg, _ = d0.load_registry(cfg["registry"])
    d_model = banks_reg[0].d_model
    seq, bs = cfg["seq_len"], cfg["batch_size"]

    # ---- load every plugin; shipped rows/lams come from joint_state.pt ----
    js_path = os.path.join(args.plugins_dir, "joint_state.pt")
    js = torch.load(js_path, map_location="cpu", weights_only=False) \
        if os.path.exists(js_path) else {}
    js_rows = {n: r for n, r in (js.get("rows") or {}).items()}
    js_lams = {}
    for n, v in zip(js.get("order") or [], js.get("lams") or []):
        js_lams[n] = float(v)

    wraps, rows, metas, lams = {}, {}, {}, {}
    missing = []
    for name, kind, home, layer in all_specs:
        path = os.path.join(args.plugins_dir, f"{name}_plugin.pt")
        if not os.path.exists(path):
            missing.append(name)
            continue
        wrap, row, meta, spec = d6.load_wrap_plugin(path)
        if not d6.wrap_is_finite(wrap, row):
            raise RuntimeError(f"non-finite plugin {path}")
        if int(meta["host_layer"]) != layer:
            raise RuntimeError(f"[{name}] host layer mismatch")
        if name in js_rows and bool(torch.isfinite(js_rows[name]).all()):
            row = js_rows[name].float().clone()      # shipped post-recal row
        elif kind == "mem" and name not in [t for t, *_ in persona]:
            raise RuntimeError(f"[{name}] no shipped row in joint_state.pt")
        wraps[name], rows[name], metas[name] = wrap, row, meta
        lams[name] = float(js_lams.get(name, 1.0 if kind == "mem" else 0.5))
    persona_missing = [t for t, *_ in persona if t not in wraps]
    if persona_missing:
        raise RuntimeError(f"missing persona plugins: {persona_missing}")
    print(f"[assemble] live: {sorted(wraps)} missing={missing}", flush=True)
    out["fit"] = {"plugins": {n: {k: metas[n].get(k) for k in
                                  ("host_layer", "home", "kind")}
                              for n in wraps}}
    dump_json(out, args.out)

    # ---- joint install (all plugs; persona rows keep fit-time rows) ----
    site_rows, site_entries = {}, {}
    for name, kind, home, layer in all_specs:
        if name not in wraps:
            continue
        site_rows.setdefault(layer, []).append(rows[name].to(args.device))
        site_entries.setdefault(layer, []).append({
            "expert": wraps[name].to(args.device),
            "lam": torch.tensor(float(lams[name]), device=args.device),
            "tag": name, "num_stock": banks_reg[layer].num_experts})
    gates, site_banks = g0.build_gates(blocks, banks_reg, fam, site_rows)
    handles = g0.install_mix(blocks, fam, site_entries, mode="off")
    row_idx = {}
    for name, kind, home, layer in all_specs:
        if name not in wraps:
            continue
        bank = site_banks[layer]
        tags_here = [en["tag"] for en in site_entries[layer]]
        row_idx[name] = bank.num_experts - len(tags_here) + tags_here.index(name)
    print("[assemble] rows:", row_idx, flush=True)

    # row-scale guard on the persona rows before any recal (180a612)
    with torch.no_grad():
        for t, h, L, _e in persona:
            bank = site_banks[L]
            r = row_idx[t]
            stock_norm = float(bank.P.data[:bank.num_experts - len(
                site_entries[L])].float().norm(dim=1).mean())
            cur = bank.P.data[r].float()
            nrm = float(cur.norm())
            if nrm > 0 and stock_norm > 0:
                bank.P.data[r] = (cur * (stock_norm / nrm)).to(bank.P.dtype)

    # ---- streams: persona homes + legacy mem homes + base ----
    episodes = mp.episodes(pack)
    mem_homes = [h for _t, h, _L, _e in persona] + ["pet", "editor", "words"]
    if args.spine == "qwen":
        mem_homes = [h for h in mem_homes if h != "words"]
    streams = {h: home_stream(tok, h, cfg, episodes, seq, bs)
               for h in mem_homes}
    base = base_batches(tok, cfg, seq, bs)[:args.pair_base]

    import varact_gate as va
    mp.apply_to_varact(va, pack)
    wrappers = va.install_varact(handles, va.POLICIES["base"]())

    def logits_fn(x):
        return g0.run_in_state(
            model, blocks, fam, gates, handles, "both",
            lambda: d0.model_logits(model, x.to(args.device)))

    def home_ce(name, windows=6):
        """own-home CE under the shipped tau=0.5 delivery (the guard read)."""
        va.set_policy(wrappers, va.POLICIES["tau0.5"]())
        home = metas[name]["home"]
        batches = streams[home][:windows]
        ce = va.eval_pass(model, blocks, fam, gates, handles, batches,
                          args.device, wrappers)["ce"]
        va.set_policy(wrappers, va.POLICIES["base"]())
        return ce

    # baseline guards before touching any row
    base_home = {t: home_ce(t) for t, *_ in persona}
    m0 = guard_margins(logits_fn, tok, args.device)
    out["before"] = {"home_ce_tau05": base_home, "margins": m0}
    dump_json(out, args.out)
    print(f"[assemble] BEFORE home_ce {base_home}", flush=True)
    print(f"[assemble] BEFORE margins {m0}", flush=True)

    facts_all = kepler_fact() + pack_facts(pack)

    # ---- per-row guard-checkpointed sibling recal (715ad33 recipe) ----
    # Stream balance is the guard lesson: 715ad33's un-chunked reject_w 40 at
    # 2.5:1 reject:home over-suppressed the row (home CE + fact token flip);
    # the shipped targeted pass-2 used home_weight 3 + reject_w 15.  With 8
    # sibling homes the reject side is capped at 4 batches each.  Guards are
    # the SHIPPED bars (greedy probes), not margin reads — fp16 margin reads
    # on near-tied answer slots flap several nats across identical states.
    saved_rows = {t: rows[t].clone() for t, *_ in persona}
    probes_by_tag = {t: (q, exp) for t, q, exp, _r in mp.probes(pack)}
    rec = {}
    for t, home, L, _e in persona:
        bank, ridx = site_banks[L], row_idx[t]
        own = streams[home] * max(1, args.home_weight)
        sib = [b for h in mem_homes if h != home for b in streams[h][:4]]
        stream = own + sib + base
        lab_home = cfg["home_labels"][home]
        pairs = g0.run_in_state(
            model, blocks, fam, gates, handles, "both",
            lambda s=stream, lh=lab_home: g0.collect_pairs_fam(
                model, blocks, fam, s, args.device, 4096,
                lambda _i, b: 1 if b[1] == lh else 0))
        best = {"p_reject": None, "row": saved_rows[t].clone(), "chunk": -1}
        per = max(1, args.steps // max(1, args.chunks))
        rep_r, stop_reason = None, "completed"
        own_q, own_exp = probes_by_tag[t]
        for c in range(args.chunks):
            rep_r = d6.train_rows_one(bank, pairs[L], ridx, per, args.lr,
                                      args.device, reject_w=args.reject_w,
                                      mass_w=2.0, owner_target=BAR_OWNER_MEM)
            ce_now = home_ce(t)
            ok_own, ans_own = fact_probe(logits_fn, tok, args.device,
                                         own_q, own_exp)
            ok_kep, ans_kep = fact_probe(logits_fn, tok, args.device,
                                         "What is my cat's name?", ["kepler"])
            margins = guard_margins(logits_fn, tok, args.device, facts_all)
            why = "ok"
            if ce_now > base_home[t] + GCE_GUARD:
                why = "home_ce"
            elif not ok_own:
                why = f"own_probe:{ans_own[:24]}"
            elif not ok_kep:
                why = f"kepler_probe:{ans_kep[:24]}"
            ok = (why == "ok")
            p_rej = rep_r["last"]["p_new_base"]
            print(f"[recal {t} chunk {c}] p_reject {p_rej:.4f} "
                  f"home_ce {ce_now:.4f} own={ok_own} kep={ok_kep} "
                  f"guard {why} margins {margins}", flush=True)
            if ok and (best["p_reject"] is None or p_rej < best["p_reject"]):
                best = {"p_reject": p_rej,
                        "row": bank.P.data[ridx].float().cpu().clone(),
                        "chunk": c}
            if not ok:
                stop_reason = f"guard_trip:{why}:chunk{c}"
                break
        with torch.no_grad():
            bank.P.data[ridx] = best["row"].to(bank.P.device).to(bank.P.dtype)
        rows[t] = best["row"]
        rec[t] = {"stop": stop_reason, "kept_chunk": best["chunk"],
                  "p_reject": best["p_reject"],
                  "last": rep_r["last"] if rep_r else None}
        out.setdefault("recal", {})[t] = rec[t]
        dump_json(out, args.out)

    # ---- balanced per-home lam grid for the persona rows (own home only) ----
    lam_fit = {}
    if not args.skip_lam:
        for t, home, L, _e in persona:
            batches = streams[home][:args.eval_batches]
            base_ev = base[:args.eval_base]
            best_lam, best_ce = lams[t], None
            for cand in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
                site_entries[L][[en["tag"] for en in site_entries[L]]
                                .index(t)]["lam"] = torch.tensor(
                    float(cand), device=args.device)
                handles = g0.install_mix(blocks, fam, site_entries, mode="off")
                ce = g0.run_in_state(
                    model, blocks, fam, gates, handles, "both",
                    lambda b=batches: d1.ce_in_state(
                        model, blocks, b, args.device, None, None, "off"))
                print(f"[lam {t} {cand}] home_ce {ce:.4f}", flush=True)
                if best_ce is None or ce < best_ce:
                    best_ce, best_lam = ce, cand
            lams[t] = float(best_lam)
            lam_fit[t] = {"lam": best_lam, "home_ce": round(best_ce, 4)}
            for en in site_entries[L]:
                en["lam"] = torch.tensor(float(lams[en["tag"]]),
                                         device=args.device)
        handles = g0.install_mix(blocks, fam, site_entries, mode="off")
        out["lam_fit"] = lam_fit
        dump_json(out, args.out)

    # ---- write joint_state_persona.pt (joint_state.pt untouched) ----
    order = [n for n, _k, _h, _L in all_specs if n in wraps]
    with torch.no_grad():
        js_out = {
            "rows": {n: site_banks[[L for (nn, k, h, L) in all_specs
                                    if nn == n][0]]
                     .P.data[row_idx[n]].float().cpu() for n in row_idx},
            "lams": [float(lams[n]) for n in order],
            "order": order,
            "varact": "tau0.5",
            "mem_pack": pack["pack"],
        }
    js_new = os.path.join(args.plugins_dir, "joint_state_persona.pt")
    if os.path.exists(js_new):
        os.replace(js_new, js_new + ".bak")
    torch.save(js_out, js_new)
    print(f"[assemble] wrote {js_new}", flush=True)

    # ---- probe battery under the shipped defaults (tau0.5 + qscope bonus) ----
    mp.apply_to_varact(va, pack)
    # re-install (idempotent): the lam grid rebuilt the MixMerges
    wrappers = va.install_varact(handles, va.POLICIES["tau0.5"]())
    scope = va.QuestionScope("overlap", tau=0.5)
    tag_col = {}
    for host, ents in site_entries.items():
        for j, en in enumerate(ents):
            tag_col[en["tag"]] = (host, j)
    bonus = va.TaughtTokenBonus(3.0, 0.5, scope=scope)
    bonus.note(tok)

    def probe_logits(x):
        def run():
            lg = d0.model_logits(model, x.to(args.device))
            return bonus.edit(lg, va.capture_tags(gates, tag_col,
                                                  qspan=scope.span))
        return g0.run_in_state(model, blocks, fam, gates, handles, "both", run)

    probes = {}
    for tag, q, expect, reject in mp.probes(pack):
        scope.begin(q, tok, g0.chatml_ids(tok, q, None))
        text, _ids = g0.gen_reply(tok, probe_logits,
                                  [{"role": "user", "content": q}],
                                  24, args.device)
        flat = " ".join(text.split()).lower()
        verd, _hit, _bad, key = mp.verdict(flat, expect, reject)
        probes[tag] = {"q": q, "answer": flat[:100], "key": key,
                       "verdict": verd,
                       "matched": list(scope.last_match)}
        print(f"[probe {tag}] {verd} "
              f"matched={probes[tag]['matched']} {flat[:80]!r}", flush=True)
    scope.begin("What is my cat's name?", tok,
                g0.chatml_ids(tok, "What is my cat's name?", None))
    text, _ids = g0.gen_reply(tok, probe_logits,
                              [{"role": "user",
                                "content": "What is my cat's name?"}],
                              16, args.device)
    flat = " ".join(text.split()).lower()
    probes["mem_pet"] = {"q": "What is my cat's name?",
                         "answer": flat[:100],
                         "verdict": "PASS" if "kepler" in flat else "FAIL",
                         "matched": list(scope.last_match)}
    out["probes"] = probes
    out["verdict"] = {
        "persona_probes_pass": sum(1 for k, v in probes.items()
                                   if k != "mem_pet" and v["verdict"] == "PASS"),
        "persona_probes_total": len(probes) - 1,
        "kepler": probes["mem_pet"]["verdict"],
    }
    out["phase"] = "assemble_done"
    out["wall_s"] = round(time.time() - t0, 2)
    dump_json(out, args.out)
    print("[assemble] verdict:", json.dumps(out["verdict"]), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spine", choices=("phi", "qwen"), required=True)
    ap.add_argument("--phase", choices=("fit", "assemble", "all"),
                    default="all")
    ap.add_argument("--plugins-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--wrap-subset", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--expert-steps", type=int, default=300)
    ap.add_argument("--expert-lr", type=float, default=3e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--out-pen", type=float, default=300.0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--row-steps", type=int, default=300)
    ap.add_argument("--row-lr", type=float, default=0.05)
    ap.add_argument("--reject-w", type=float, default=15.0,
                    help="715ad33 targeted pass-2 setting (40 over-suppressed "
                         "at >2.5:1 reject:home)")
    ap.add_argument("--max-cal-pairs", type=int, default=4096)
    ap.add_argument("--pair-batches", type=int, default=8)
    ap.add_argument("--pair-base", type=int, default=8)
    ap.add_argument("--eval-batches", type=int, default=6)
    ap.add_argument("--eval-base", type=int, default=4)
    ap.add_argument("--home-weight", type=int, default=3)
    ap.add_argument("--steps", type=int, default=180,
                    help="sibling-recal steps per row (chunked)")
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--skip-lam", action="store_true")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = dict(SPINES[args.spine])
    if not args.plugins_dir:
        args.plugins_dir = cfg["plugins"]
    if not args.out:
        args.out = os.path.join(args.plugins_dir,
                                f"persona_{args.spine}_{args.phase}.json")
    torch.manual_seed(args.seed)
    if args.phase in ("fit", "all"):
        run_fit(args, cfg)
    if args.phase in ("assemble", "all"):
        run_assemble(args, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
