#!/usr/bin/env python3
"""THE FINAL ASSEMBLY — vk-base build: registry/graft + specialty bank +
persona pack, on our own KDA+MLA passport spine.

Spine: vk-base chat_model (426M total / 325M active, 10L x d1024, KDA+MLA,
PassportRouter 8 x top-2).  PASSPORT-NATIVE: a registry row IS a passport row
(lab/vk_graft.py); the d0/d2/v06 graft stack ports through the host adapter
with zero shared-source edits.

Phases (detached GPU jobs, incremental JSON, --skip-done):
  shape     gfx906 CE fwd+bwd shape grid through a wrap (spine-specific holes)
  registry  init=exact passport banks + PARITY GATE (bar 0.95 logit cos)
  fit       one wrap per lane: 3 specialty donors (core_bank -> v05_cores,
            re-wrapped in vk geometry d_model=1024, rank-64, wd 0, lr 3e-3,
            task-CE + out_pen 300, FEED >= 200k) + 9 mem plugs (legacy trio
            + persona pack, shaped episodes) with the 180a612 row-scale guard
  assemble  joint install + closing row recal (owner 0.85) + 715ad33
            sibling-recal guards + balanced per-home lambda fits (92abbd7)
            + quality reads; writes joint_state_vfinal.pt

Chat CLI: lab/chat_vfinal.py.
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
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))

import vk_graft as vk  # noqa: E402
import graft_d0_parity as d0  # noqa: E402
import graft_d1_plugin as d1  # noqa: E402
import door25_wrap_expert as d25  # noqa: E402
import door6_expert_mix as d6  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402
import mem_pack as mp  # noqa: E402

FORMAT = "vesper-vk-final-v1"
BAR_PARITY_COS = 0.95
BAR_OWNER_MEM = 0.85
BAR_OWNER_SPEC = 0.35
MIN_FEED_TOKENS = 200_000
BASE_TAX_BAR = 0.05

DEFAULT_MODEL = "/data/tliao/vk_final/chat_model"
DEFAULT_OUT = "/data/tliao/vk_final/vk_results.json"
DEFAULT_PLUGINS = "/data/tliao/vk_final"
DEFAULT_CORES = "/data/tliao/v05_cores"

# name, kind, home, host layer, core file (None = host-native extract),
# donor tag, donor layer, donor expert, host-native expert index, act
# Host sites: specialties depth-matched round(L_d/Lmax*9) -> k3 L12/92 -> 1,
# glm L3/44 -> 1, mimo L1/47 -> 0 (spread to 2 when shared), mems spread
# over the 10L spine (per-spine sites; phi/qwen sites live in persona_v1.json).
VK_WRAPS = [
    ("k3_hack", "domain", "hacking", 1, "k3_hack_core.pt",
     "kimi-k3", 12, 895, None, "situ"),
    ("glm_cybersec", "domain", "cybersec", 2, "glm_cybersec_core.pt",
     "glm-5.3-flash", 3, 0, None, "swiglu"),
    ("mimo_writing", "domain", "writing", 0, "mimo_writing_core.pt",
     "mimo-v2.6-flash", 1, 0, None, "swiglu"),
    ("mem_pet", "mem", "pet", 3, None, "vk-host", 3, 0, 0, "swiglu"),
    ("mem_edit", "mem", "editor", 4, None, "vk-host", 4, 1, 1, "swiglu"),
    ("mem_words", "mem", "words", 5, None, "vk-host", 5, 2, 2, "swiglu"),
    ("mem_poi", "mem", "poi", 0, None, "vk-host", 0, 3, 3, "swiglu"),
    ("mem_machine", "mem", "machine", 6, None, "vk-host", 6, 4, 4, "swiglu"),
    ("mem_gundam", "mem", "gundam", 7, None, "vk-host", 7, 5, 5, "swiglu"),
    ("mem_cars", "mem", "cars", 8, None, "vk-host", 8, 6, 6, "swiglu"),
    ("mem_trains", "mem", "trains", 9, None, "vk-host", 9, 7, 7, "swiglu"),
    ("mem_igbt", "mem", "igbt", 2, None, "vk-host", 2, 3, 3, "swiglu"),
]

# persona_v1.json home key -> plug name (legacy trio lives in g0.build_episodes)
PERSONA_HOME_TO_TAG = {}
SPECIALTY_HOME = {"hacking": "k3_hack", "cybersec": "glm_cybersec",
                  "writing": "mimo_writing"}
HOME_LABEL = {"hacking": 1, "cybersec": 2, "writing": 3,
              "pet": 4, "editor": 5, "words": 6, "poi": 7, "machine": 8,
              "gundam": 9, "cars": 10, "trains": 11, "igbt": 12}
LAM_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]


def dump_json(out: dict, path: str | None) -> None:
    g0.dump_json(out, path)


def vk_chatml_batches(tok, rows, seq, bs, tile_to=4096, label=1):
    """Tile (prompt, answer) ChatML rows into train windows."""
    ids = []
    for prompt, ans in rows:
        ids.extend(vk.chatml_ids(tok, prompt, ans))
    if not ids:
        return []
    tiles = (tile_to + len(ids) - 1) // max(1, len(ids))
    ids = (ids * tiles)[:tile_to]
    return vk.to_batches(ids, label, seq, bs)


def all_episodes():
    """legacy trio (g0) + persona pack (mem_pack) as {home: [(prompt, ans)]}."""
    out = {}
    for name, ep in g0.build_episodes().items():
        out[name] = list(g0.episode_home_texts(ep))
    for home, ep in mp.episodes(mp.load_pack()).items():
        out[home] = list(g0.episode_home_texts(ep))
    return out


def home_rows(args, home: str):
    """(train_rows, feed_note) for one home key."""
    if home in SPECIALTY_HOME:
        path = os.path.join(args.specialties_dir, home, "corpus.txt")
        with open(path) as f:
            text = f.read()
        ids = args._tok.encode(text, add_special_tokens=False)[:args.max_tokens]
        return ids, {"corpus": path, "tokens": len(ids)}
    eps = all_episodes()
    if home not in eps:
        raise RuntimeError(f"unknown home {home}")
    return eps[home], {"episodes": home, "rows": len(eps[home])}


# ---------------------------------------------------------------------------
# Phase: shape (gfx906 CE fwd+bwd hole grid — SPINE-SPECIFIC)
# ---------------------------------------------------------------------------

def run_shape(args) -> dict:
    t0 = time.time()
    out = {"gate": "vk-shape", "format": FORMAT, "phase": "start",
           "pid": os.getpid()}
    dump_json(out, args.out)
    model, tok = vk.load_vk(args.model, args.device, args.dtype)
    blocks = vk.install_vk_host(model)
    banks = vk.banks_from_native(model)
    results = {}
    for bt in (128, 192, 256, 320, 384, 512):
        seq, bs = bt, 1
        ok, err = True, ""
        stock = None
        try:
            ffn = blocks[1]
            w_gu = torch.zeros(2560, 1024)
            w_down = torch.zeros(1024, 1280)
            wrap = d25.WrappedForeignExpert(1024, 64, w_gu, w_down, seed=0)
            stock = ffn.experts.forward
            pm = d6.MixMerge(stock, [wrap.to(args.device)],
                             [nn.Parameter(torch.zeros((), device=args.device))],
                             ["X"], gate_getter=lambda: ffn.router,
                             num_stock=8)
            pm.mode = "forced"
            ffn._plugin_merge = pm
            ffn.experts.forward = lambda h, i=None, w=None: pm(h, i, w)
            toks = torch.randint(0, 1000, (bs, seq), device=args.device)
            with torch.amp.autocast("cuda", dtype=torch.float16,
                                    enabled=toks.is_cuda):
                lg, ce, _aux = model(toks, toks)
            loss = torch.nn.functional.cross_entropy(
                lg[:, :-1].reshape(-1, lg.shape[-1]).float(), toks[:, 1:].reshape(-1))
            loss.backward()
            ffn.experts.forward = stock
            delattr(ffn, "_plugin_merge")
        except Exception as e:  # noqa: BLE001
            ok, err = False, f"{type(e).__name__}: {e}"
            try:
                ffn.experts.forward = stock
                if hasattr(ffn, "_plugin_merge"):
                    delattr(ffn, "_plugin_merge")
            except Exception:  # noqa: BLE001
                pass
        results[str(bt)] = {"ok": ok, "error": err}
        print(f"[shape] B*T={bt}: {'OK' if ok else 'FAIL ' + err}", flush=True)
        dump_json(out, args.out)
    safe = sorted(int(k) for k, v in results.items() if v["ok"])
    out.update({"results": results, "safe_bt": safe, "phase": "shape_done",
                "wall_s": round(time.time() - t0, 2)})
    dump_json(out, args.out)
    return out


# ---------------------------------------------------------------------------
# Phase: registry (init=exact passport banks + parity gate)
# ---------------------------------------------------------------------------

def run_registry(args) -> dict:
    t0 = time.time()
    out = {"gate": "vk-registry", "format": FORMAT, "phase": "start",
           "pid": os.getpid(), "bar_parity_cos": BAR_PARITY_COS,
           "socket": "PassportRouter-native (registry row = passport row)"}
    dump_json(out, args.out)
    model, tok = vk.load_vk(args.model, args.device, args.dtype)
    blocks = vk.install_vk_host(model)
    banks = vk.banks_from_native(model)
    print(f"[registry] {len(blocks)} MoE layers, "
          f"{banks[0].num_experts} stock rows, d_model {banks[0].d_model}, "
          f"d_r {banks[0].d_r}", flush=True)

    # validation batches: SFT-format prompts + domain text (numeric gate)
    ids = []
    for p in ("Hello! How are you today?", "What is the capital of France?",
              "Tell me a fun fact about space.",
              "What is my cat's name?"):
        ids.extend(vk.chatml_ids(tok, p))
    sp = os.path.join(args.specialties_dir, "hacking", "corpus.txt")
    if os.path.exists(sp):
        ids.extend(tok.encode(open(sp).read()[:20000], add_special_tokens=False))
    batches = vk.to_batches(ids, 0, args.reg_seq_len, 1)[:args.reg_batches]

    par = vk.parity_measure(model, blocks, banks, batches, args.device, n_new=0)
    out["parity"] = par
    out["parity_gate"] = {
        "logit_cosine_mean": par["logit_cosine_mean"],
        "bar": BAR_PARITY_COS,
        "pass": bool(par["logit_cosine_mean"] >= BAR_PARITY_COS)}
    os.makedirs(os.path.dirname(args.registry) or ".", exist_ok=True)
    d0.save_registry(args.registry, banks,
                     {"model": args.model, "format": FORMAT,
                      "family": vk.FAM.name, "top_k": vk.FAM.top_k})
    out["registry"] = args.registry
    out["phase"] = "registry_done"
    out["wall_s"] = round(time.time() - t0, 2)
    dump_json(out, args.out)
    print(f"== registry parity cos {par['logit_cosine_mean']:.4f} "
          f"(min {par['logit_cosine_min']:.4f}, bar {BAR_PARITY_COS}) -> "
          f"{'PASS' if out['parity_gate']['pass'] else 'FAIL'} ==", flush=True)
    return out


# ---------------------------------------------------------------------------
# Phase: fit (one wrap per lane)
# ---------------------------------------------------------------------------

def run_fit(args) -> dict:
    t0 = time.time()
    out = {"gate": "vk-fit", "format": FORMAT, "phase": "start",
           "pid": os.getpid(), "recipe": {
               "d_canon": 64, "lr": args.expert_lr, "wd": 0.0,
               "out_pen": args.out_pen, "min_feed_tokens": MIN_FEED_TOKENS}}
    dump_json(out, args.out)
    specs = [s for s in VK_WRAPS
             if not args.wrap_subset or s[0] in args.wrap_subset.split(",")]
    model, tok = vk.load_vk(args.model, args.device, args.dtype)
    args._tok = tok
    blocks = vk.install_vk_host(model)
    banks_reg, _ = d0.load_registry(args.registry)
    if len(banks_reg) != len(blocks):
        raise RuntimeError(f"registry {len(banks_reg)} vs model {len(blocks)}")
    seq, bs = args.seq_len, args.batch_size
    if seq * bs not in args.safe_bt:
        raise RuntimeError(f"train B*T {seq*bs} not in safe set {args.safe_bt}")
    steps_eff = max(args.expert_steps,
                    -(-MIN_FEED_TOKENS // max(1, seq * bs)))
    base_ids = []
    for home in ("hacking", "cybersec", "writing"):
        p = os.path.join(args.specialties_dir, home, "corpus.txt")
        if os.path.exists(p):
            base_ids.extend(tok.encode(open(p).read()[:20000],
                                       add_special_tokens=False))
    base_batches = vk.to_batches(base_ids[:args.max_base_tokens], 0, seq, bs)

    for (name, kind, home, host_i, core_file, donor, dlayer, dexp,
         hexp, act_force) in specs:
        # per-wrap guard: one bad wrap must not kill the lane (bank_rung1
        # phase_fit idiom — keep the lane alive all night)
        try:
            plugin_path = os.path.join(args.plugins_dir, f"{name}_plugin.pt")
            if args.skip_done and os.path.exists(plugin_path):
                w_chk, r_chk, _, _ = d6.load_wrap_plugin(plugin_path)
                if d6.wrap_is_finite(w_chk, r_chk):
                    print(f"[{name}] plugin exists and finite — skip",
                          flush=True)
                    out.setdefault("fitted", {})[name] = {"skipped": True}
                    dump_json(out, args.out)
                    continue
            if core_file:
                core_path = (core_file if os.path.isabs(core_file)
                             else os.path.join(args.cores_dir, core_file))
                if not os.path.exists(core_path):
                    raise RuntimeError(f"[{name}] missing core {core_path}")
                core = torch.load(core_path, map_location="cpu",
                                  weights_only=False)
                meta = core.get("meta", {})
                w_gu, w_down = core["w_gu"], core["w_down"]
                act = meta.get("act", "swiglu")
                situ_beta = meta.get("situ_beta", 1.0)
                situ_lin = meta.get("situ_linear_beta", 1.0)
            else:
                ffn = blocks[host_i]
                w_gu, w_down = vk.extract_core(ffn, hexp)
                meta = {}
                act = act_force or "swiglu"
                situ_beta, situ_lin = 1.0, 1.0
            wrap = d25.WrappedForeignExpert(
                banks_reg[host_i].d_model, args.d_canon, w_gu, w_down,
                seed=args.seed, act=act, situ_beta=situ_beta,
                situ_linear_beta=situ_lin)

            rows_or_ids, feed_note = home_rows(args, home)
            if kind == "domain":
                home_train = vk.to_batches(rows_or_ids, 1, seq, bs)
            else:
                home_train = vk_chatml_batches(tok, rows_or_ids, seq, 1,
                                               tile_to=args.episode_tile)
            if not home_train:
                raise RuntimeError(f"[{name}] empty home stream")
            fed = steps_eff * seq * bs
            print(f"== [{name}] core {donor} L{dlayer}e{dexp} "
                  f"({wrap.core_param_count()} frozen, act {act}) -> host "
                  f"L{host_i} home={home} ({len(home_train)} batches, "
                  f"{steps_eff} steps, feed {fed} tok-slots) ==", flush=True)
            out.setdefault("feed", {})[name] = {
                **feed_note, "batches": len(home_train), "steps": steps_eff,
                "fed_token_slots": fed}
            dump_json(out, args.out)

            # phase A: adapters, forced dispatch (d25 recipe verbatim)
            pa = _fit_wrap_vk(model, blocks, banks_reg, host_i, wrap,
                              home_train, base_batches, args, steps_eff)
            if not pa.get("finite", True):
                print(f"[{name}] NON-FINITE at lr {args.expert_lr} — refit 1e-3",
                      flush=True)
                del wrap
                torch.cuda.empty_cache()
                wrap = d25.WrappedForeignExpert(
                    banks_reg[host_i].d_model, args.d_canon, w_gu, w_down,
                    seed=args.seed, act=act, situ_beta=situ_beta,
                    situ_linear_beta=situ_lin)
                old_lr = args.expert_lr
                args.expert_lr = 1e-3
                pa = _fit_wrap_vk(model, blocks, banks_reg, host_i, wrap,
                                  home_train, base_batches, args, steps_eff)
                args.expert_lr = old_lr
                if not pa.get("finite", True):
                    out.setdefault("fitted", {})[name] = {
                        "error": "non_finite_twice"}
                    out["phase"] = f"failed_{name}"
                    dump_json(out, args.out)
                    continue

            # registry row: prototype init + ROW-SCALE GUARD (180a612) + recal.
            # ROW SPACE: passport rows live in QUERY space (P is (E, d_r=64) on
            # vk-base; score = (W_q h).p_e), so both the placeholder row and the
            # prototype must be d_r wide — writing d_model=1024 into P crashed
            # every rung-0 fit lane at _build_vk_gates (2026-10-14).
            site_rows = {host_i: [torch.zeros(banks_reg[host_i].d_r)]}
            gates, site_banks = _build_vk_gates(blocks, banks_reg, site_rows)
            ffn = blocks[host_i]
            handles = g0.install_mix(blocks, vk.FAM, {
                host_i: [{"expert": wrap,
                          "lam": nn.Parameter(torch.ones((),
                                                          device=args.device)),
                          "tag": name,
                          "num_stock": banks_reg[host_i].num_experts}]},
                mode="off")
            pair_home = home_train[:args.pair_batches]
            pair_base = base_batches[:args.pair_base]
            pairs = g0.run_in_state(
                model, blocks, vk.FAM, gates, handles, "off",
                lambda: g0.collect_pairs_fam(
                    model, blocks, vk.FAM, pair_home + pair_base, args.device,
                    args.max_cal_pairs, lambda _i, b: 1 if b[1] == 1 else 0))
            h, lab = pairs[host_i]
            home_h = h[lab > 0.5]
            bank = site_banks[host_i]
            row_idx = bank.num_experts - 1
            with torch.no_grad():
                if home_h.shape[0] > 0:
                    # prototype in QUERY space (d_r): mean of W_q h over home
                    # hiddens (PassportBank API is .queries() — .query()
                    # AttributeError'd every fit after phaseA and before the
                    # save, 2026-10-14)
                    proto = bank.queries(home_h).mean(dim=0)
                    stock_norm = float(
                        bank.P.data[:row_idx].float().norm(dim=1).mean())
                    pnorm = float(proto.float().norm())
                    if pnorm > 0 and stock_norm > 0:
                        proto = proto * (stock_norm / pnorm)  # row-scale guard
                    bank.P.data[row_idx] = proto
            owner = BAR_OWNER_MEM if kind == "mem" else BAR_OWNER_SPEC
            rep = d6.train_rows_one(bank, pairs[host_i], row_idx, args.row_steps,
                                    args.row_lr, args.device,
                                    reject_w=args.reject_w, mass_w=args.mass_w,
                                    owner_target=owner)
            row = bank.P.data[row_idx].float().cpu().clone()
            _restore_ffn(ffn)
            d1.set_plugin_mode(handles, "off")
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
                         "core_file": (core_file or f"vk-host L{host_i}e{hexp}"),
                         "owner_target": owner},
            }, plugin_path)
            print(f"[fit] {name} saved plugin -> {plugin_path}", flush=True)
            out.setdefault("fitted", {})[name] = {
                "plugin": plugin_path, "host_layer": host_i, "home": home,
                "donor": [donor, dlayer, dexp],
                "phase_a": {k: pa.get(k) for k in ("steps", "lr", "finite",
                                                   "wall_s")},
                "ce_hist_tail": pa.get("history", [])[-2:],
                "row_recal": {"first": rep["first"], "last": rep["last"]}}
            out["phase"] = f"fitted_{name}"
            out["fit_wall_s"] = round(time.time() - t0, 2)
            dump_json(out, args.out)
            print(f"== [{name}] done: row p_home {rep['last']['p_new_code']:.3f} "
                  f"p_base {rep['last']['p_new_base']:.4f} ==", flush=True)
            del wrap
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 — keep the lane alive all night
            out.setdefault("fail", {})[name] = f"{type(e).__name__}: {e}"
            out["phase"] = f"failed_{name}"
            dump_json(out, args.out)
            print(f"[fit] {name} EXC {e}", flush=True, file=sys.stderr)
            torch.cuda.empty_cache()
    out["phase"] = "fit_done"
    out["fit_wall_s"] = round(time.time() - t0, 2)
    dump_json(out, args.out)
    return out


def _restore_ffn(ffn):
    if hasattr(ffn, "_vk_stock_dispatch"):
        ffn.experts.forward = ffn._vk_stock_dispatch
    if hasattr(ffn, "_plugin_merge"):
        delattr(ffn, "_plugin_merge")


def _fit_wrap_vk(model, blocks, banks_reg, host_i, wrap, home_train,
                 base_batches, args, steps) -> dict:
    """d25.train_adapters verbatim through the vk host contract."""
    wrap.to(args.device)
    ffn = blocks[host_i]
    stock_fwd = ffn.experts.forward
    pm = d6.MixMerge(stock_fwd, [wrap],
                     [nn.Parameter(torch.zeros((), device=args.device))],
                     ["X"], gate_getter=lambda: ffn.router,
                     num_stock=banks_reg[host_i].num_experts)
    pm.mode = "off"
    ffn._plugin_merge = pm

    def make_fwd(pm=pm):
        def fwd(hidden_states, top_k_index=None, top_k_weights=None):
            return pm(hidden_states, top_k_index, top_k_weights)
        return fwd

    ffn.experts.forward = make_fwd()
    handles = [(ffn, stock_fwd)]
    pa = d25.train_adapters(model, wrap, handles, home_train, base_batches,
                            steps, args.expert_lr, args.device,
                            out_pen=args.out_pen, grad_clip=args.grad_clip,
                            weight_decay=args.weight_decay)
    d1.set_plugin_mode(handles, "off")
    _restore_ffn(ffn)
    pa["finite"] = g0.finite_state(wrap)
    return pa


def _build_vk_gates(blocks, banks_reg, site_rows):
    """VkGraftGate per layer; site layers get plug rows appended to the bank."""
    gates, site_banks = [], {}
    for i, ffn in enumerate(blocks):
        bank_reg = banks_reg[i]
        rows = site_rows.get(i, [])
        if rows:
            bank = d0.PassportBank(
                bank_reg.num_experts + len(rows), bank_reg.d_model,
                d_r=bank_reg.d_r, learn_wq=bank_reg.W_q is not None)
            with torch.no_grad():
                bank.P.data[:bank_reg.num_experts] = bank_reg.P.data
                bank.W_q.data.copy_(bank_reg.W_q.data)
                for j, r in enumerate(rows):
                    bank.P.data[bank_reg.num_experts + j] = r.to(bank.P.dtype)
            bank.layer_scale = bank_reg.layer_scale
            gates.append(vk.VkGraftGate(bank, n_new=len(rows), top_k=2))
            site_banks[i] = bank
        else:
            gates.append(vk.VkGraftGate(bank_reg, n_new=0, top_k=2))
    return gates, site_banks


# ---------------------------------------------------------------------------
# Phase: assemble
# ---------------------------------------------------------------------------

def run_assemble(args) -> dict:
    t0 = time.time()
    prev = {}
    if args.out and os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
        except Exception:  # noqa: BLE001
            prev = {}
    out = {"gate": "vk-assemble", "format": FORMAT, "phase": "start",
           "pid": os.getpid(), "delivery": "additive-lambda-balanced",
           "guards": {"row_scale": "180a612", "sibling_recal": "715ad33",
                      "qscope": "30038bd overlap", "varact": "tau0.5"}}
    for k in ("joint", "metrics", "verdict"):
        if k in prev and prev[k]:
            out[k] = prev[k]
    dump_json(out, args.out)

    model, tok = vk.load_vk(args.model, args.device, args.dtype)
    args._tok = tok
    blocks = vk.install_vk_host(model)
    banks_reg, _ = d0.load_registry(args.registry)
    seq = args.seq_len

    wrap_specs = [(n, k, l, h) for (n, k, h, l, *_r) in VK_WRAPS]
    wraps, rows, metas = {}, {}, {}
    missing = []
    for name, kind, layer, home in wrap_specs:
        path = os.path.join(args.plugins_dir, f"{name}_plugin.pt")
        if not os.path.exists(path):
            missing.append(name)
            continue
        wrap, row, meta, spec = d6.load_wrap_plugin(path)
        if not d6.wrap_is_finite(wrap, row):
            raise RuntimeError(f"non-finite plugin {path}")
        # row-scale guard at load (idempotent)
        stock_norm = float(banks_reg[layer].P.data.float().norm(dim=1).mean())
        row_norm = float(row.float().norm())
        if row_norm > 0 and stock_norm > 0:
            row = row * (stock_norm / row_norm)
        wraps[name], rows[name], metas[name] = wrap, row, meta
    if missing:
        print(f"[assemble] WARNING missing plugs (continuing): {missing}",
              flush=True)
    if not wraps:
        raise RuntimeError("no plugins at all — run fit first")
    out["fit"] = {"plugins": {n: metas[n] for n in wraps},
                  "missing": missing}
    dump_json(out, args.out)

    lam_of = {}
    js_path = os.path.join(args.plugins_dir, "joint_state_vfinal.pt")
    prev_joint = prev.get("joint") or {}
    if os.path.exists(js_path) and prev_joint.get("lams_final"):
        js = torch.load(js_path, map_location="cpu", weights_only=False)
        for n, v in zip(js.get("order", []), js.get("lams", [])):
            lam_of[n] = float(v)
        for n, r in (js.get("rows") or {}).items():
            if n in rows:
                rows[n] = r.float().clone()
        print("== restored joint_state_vfinal.pt ==", flush=True)

    site_rows, site_entries = {}, {}
    lams, lam_names = [], []
    for name, kind, layer, home in wrap_specs:
        if name not in wraps:
            continue
        lam = nn.Parameter(torch.tensor(float(lam_of.get(name, args.lam_init)),
                                        device=args.device))
        lams.append(lam)
        lam_names.append(name)
        site_rows.setdefault(layer, []).append(rows[name].to(args.device))
        site_entries.setdefault(layer, []).append({
            "expert": wraps[name].to(args.device), "lam": lam, "tag": name,
            "num_stock": banks_reg[layer].num_experts})
    gates, site_banks = _build_vk_gates(blocks, banks_reg, site_rows)
    handles = g0.install_mix(blocks, vk.FAM, site_entries, mode="off")
    row_idx = {}
    for name, kind, layer, home in wrap_specs:
        if name not in wraps:
            continue
        bank = site_banks[layer]
        tags_here = [en["tag"] for en in site_entries[layer]]
        row_idx[name] = bank.num_experts - len(tags_here) + tags_here.index(name)
    print("installed rows:", row_idx, flush=True)
    out.setdefault("joint", {})["row_idx"] = row_idx
    dump_json(out, args.out)

    # ---- home/base streams ----
    args._tok = tok
    home_batches = {}
    for home in HOME_LABEL:
        try:
            rows_or_ids, _note = home_rows(args, home)
        except Exception:  # noqa: BLE001
            continue
        if home in SPECIALTY_HOME:
            raw = [(t, HOME_LABEL[home]) for t, _ in vk.to_batches(
                rows_or_ids, 1, seq, args.batch_size)]
        else:
            raw = [(t, HOME_LABEL[home]) for t, _ in vk_chatml_batches(
                tok, rows_or_ids, seq, 1, tile_to=4096)]
        home_batches[home] = raw[:args.joint_batches_per_home]
    base_ids = []
    for home in ("hacking", "cybersec", "writing"):
        p = os.path.join(args.specialties_dir, home, "corpus.txt")
        if os.path.exists(p):
            base_ids.extend(tok.encode(open(p).read()[:20000],
                                       add_special_tokens=False))
    base_batches = [(t, 0) for t, _ in vk.to_batches(
        base_ids[:args.max_base_tokens], 0, seq, args.batch_size)]

    if not prev_joint.get("lam_fit"):
        # ---- closing row recal (all plugs live, owner per kind) ----
        if not prev_joint.get("row_recal") or args.force_recal:
            site_info = []
            for name, kind, layer, home in wrap_specs:
                if name not in wraps:
                    continue
                site_info.append({
                    "host_i": layer, "bank": site_banks[layer],
                    "row_idx": row_idx[name],
                    "home_label": HOME_LABEL[home], "tag": name,
                    "owner_target": BAR_OWNER_MEM if kind == "mem"
                    else BAR_OWNER_SPEC})
            joint = []
            for home, raw in home_batches.items():
                joint += raw
            joint += base_batches[:args.pair_base]
            print("== closing row recal (all plugs live) ==", flush=True)
            recal = g0.closing_row_recal(
                model, blocks, vk.FAM, handles, gates, site_info, joint,
                args.device, args.recal_steps, args.recal_lr, args)
            out["joint"]["row_recal"] = recal
            dump_json(out, args.out)
            # sibling recal for the mem rows (715ad33, guard chunks)
            sib = _sibling_recal(args, model, blocks, vk.FAM, handles, gates,
                                 site_banks, row_idx, wrap_specs, wraps,
                                 home_batches, base_batches, tok)
            out["joint"]["sibling_recal"] = sib
            dump_json(out, args.out)

        # ---- balanced per-home lambda fit (92abbd7 / v0.6) ----
        print("== balanced per-home lambda fit ==", flush=True)
        lam_fit = _fit_lams_balanced(model, blocks, vk.FAM, lams, lam_names,
                                     handles, gates, wrap_specs, wraps,
                                     home_batches, base_batches, args)
        out["joint"]["lam_fit"] = lam_fit
        out["joint"]["lams_final"] = {
            n: round(float(lams[i].detach().clamp(min=0)), 4)
            for i, n in enumerate(lam_names)}
        torch.save({
            "rows": {n: site_banks[{nn: l for nn, k, l, h in wrap_specs}[n]]
                     .P.data[row_idx[n]].cpu() for n in row_idx},
            "lams": [float(p.detach().clamp(min=0)) for p in lams],
            "order": lam_names,
        }, js_path)
        out["phase"] = "joint_done"
        dump_json(out, args.out)
    for en_l in site_entries.values():
        for en in en_l:
            en["lam"] = torch.tensor(float(en["lam"].detach().clamp(min=0)),
                                     device=args.device)
    handles = g0.install_mix(blocks, vk.FAM, site_entries, mode="off")

    # ---- quality reads ----
    print("== state measurements ==", flush=True)
    ev = {h: home_batches[h] for h in home_batches}
    ev["base"] = base_batches[:args.eval_base]

    def ce_masked(batches):
        return g0.run_in_state(
            model, blocks, vk.FAM, None, None, "off",
            lambda: d1.ce_in_state(model, blocks, batches, args.device,
                                   None, None, "off"))

    def ce_g(batches):
        return g0.run_in_state(
            model, blocks, vk.FAM, gates, handles, "both",
            lambda: d1.ce_in_state(model, blocks, batches, args.device,
                                   None, None, "off"))

    home_ce = {}
    for tag, batches in ev.items():
        if not batches:
            continue
        bm, bg = ce_masked(batches), ce_g(batches)
        home_ce[tag] = {"masked": round(bm, 4), "grafted": round(bg, 4),
                        "delta": round(bg - bm, 4)}
        print(f"  CE {tag:10s} masked {bm:.4f} -> grafted {bg:.4f}", flush=True)
    out["metrics"] = {"home_ce": home_ce}
    verdict = {"all_homes_improve": all(
        v["delta"] < 0 for k, v in home_ce.items() if k != "base"),
        "base_tax_nats": home_ce.get("base", {}).get("delta", 0.0),
        "base_tax_bar": BASE_TAX_BAR}
    verdict["pass"] = (verdict["all_homes_improve"]
                       and verdict["base_tax_nats"] <= verdict["base_tax_bar"])
    out["verdict"] = verdict
    out["phase"] = "assemble_done"
    out["wall_s"] = round(time.time() - t0, 2)
    dump_json(out, args.out)
    print("== verdict:", json.dumps(verdict), "==", flush=True)
    return out


def _fit_lams_balanced(model, blocks, fam, lams, lam_names, handles, gates,
                       wrap_specs, wraps, home_batches, base_batches, args):
    """Per-home grid on OWN home only + base-tax guard (v06 / 92abbd7)."""
    homes = {n: h for n, _k, _l, h in wrap_specs if n in wraps}
    state = {n: 0.5 for n in lam_names}

    def set_lams(d):
        for i, n in enumerate(lam_names):
            lams[i].data.fill_(float(d[n]))

    def ce(batches, which="both"):
        if not batches:
            return 0.0
        return g0.run_in_state(
            model, blocks, fam, gates, handles, which,
            lambda: d1.ce_in_state(model, blocks, batches, args.device,
                                   None, None, "off"))

    home_ev = {n: home_batches.get(homes[n], [])[:args.lam_home_batches]
               for n in lam_names}
    base_ev = base_batches[:args.lam_base_batches]
    set_lams({n: 0.0 for n in lam_names})
    ce_mask = {n: ce(home_ev[n], "off") for n in lam_names}
    ce_base_mask = ce(base_ev, "off")
    report = {"grid": LAM_GRID, "order": lam_names, "per_home": {},
              "ce_home_masked": ce_mask, "ce_base_masked": ce_base_mask}
    for n in lam_names:
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
            "home": homes[n], "curve": curve, "masked": round(ce_mask[n], 4),
            "best_lam": best_lam, "best_ce": round(best_ce, 4),
            "gain": round(ce_mask[n] - best_ce, 4)}
        print(f"  [lam {n:14s} home {homes[n]:8s}] {ce_mask[n]:.4f} -> "
              f"{best_ce:.4f} at lam*={best_lam}", flush=True)
    set_lams(state)
    base_now = ce(base_ev)
    tax = base_now - ce_base_mask
    guard = []
    while tax > BASE_TAX_BAR and any(state[n] > 0 for n in lam_names):
        live = [n for n in lam_names if state[n] > 0]

        def eff(n):
            return report["per_home"][n]["gain"] / max(state[n], 1e-6)
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
    report["lams_final"] = {n: state[n] for n in lam_names}
    set_lams(state)
    return report


def _sibling_recal(args, model, blocks, fam, handles, gates, site_banks,
                   row_idx, wrap_specs, wraps, home_batches, base_batches, tok):
    """715ad33 guard-checkpointed sibling recal for mem rows (compact port).

    per row: own home x2 + sibling mem episodes on reject + base; reject_w 15,
    owner 0.85, 30-step chunks; after every chunk the taught greedy probe must
    stay green AND own-home CE (tau0.5 delivery) within +0.02 — first guard
    trip early-stops and keeps the best passing chunk."""
    import grafted_spine_v0 as _g
    mem = [(n, k, l, h) for (n, k, l, h) in wrap_specs
           if k == "mem" and n in wraps]
    report = []
    taught = mp.taught(mp.load_pack())
    for name, kind, layer, home in mem:
        bank = site_banks[layer]
        r_i = row_idx[name]
        own = home_batches.get(home, [])
        sib = []
        for n2, _k2, _l2, h2 in mem:
            if n2 != name:
                sib += home_batches.get(h2, [])[:4]
        reject = sib + base_batches[:args.pair_base]
        # (h, label) collection once, under all-plugs-live gates
        pairs = g0.run_in_state(
            model, blocks, fam, gates, handles, "both",
            lambda: g0.collect_pairs_fam(
                model, blocks, fam, own + reject, args.device,
                args.max_cal_pairs,
                lambda _i, b: 1 if b[1] == HOME_LABEL[home] else 0))
        h, lab = pairs[layer]
        if h.shape[0] == 0:
            continue
        # baseline CE + greedy probe
        base_ce = g0.run_in_state(
            model, blocks, fam, gates, handles, "both",
            lambda: d1.ce_in_state(model, blocks, own[:args.eval_batches],
                                   args.device, None, None, "off"))
        probe_q = {"pet": "What is my cat's name?",
                   "editor": "What is my favorite text editor?",
                   "words": "What is 2 + 2?",
                   "poi": "What's your favorite live action show?",
                   "machine": "Who is The Machine?",
                   "gundam": "What's your favorite anime?",
                   "cars": "What about cars?",
                   "trains": "What do you think of trains?",
                   "igbt": "What about IGBTs?"}.get(home, "")
        tstrs = taught.get(name, ["Kepler"])

        def probe_green():
            if not probe_q:
                return True

            def _gen():
                ids = vk.chatml_ids(tok, probe_q)
                out_ids = []
                for _ in range(24):
                    x = torch.tensor([ids + out_ids], dtype=torch.long,
                                     device=args.device)
                    lg = d0.model_logits(model, x)
                    nxt = int(lg[0, -1].float().argmax())
                    if nxt in (tok.eos_token_id, tok.pad_token_id):
                        break
                    out_ids.append(nxt)
                return tok.decode(out_ids, skip_special_tokens=False)

            try:
                text = g0.run_in_state(model, blocks, fam, gates, handles,
                                       "both", _gen)
            except Exception:  # noqa: BLE001
                return True
            return any(s.strip() and s.strip() in text for s in tstrs)

        best = None
        trips = []
        for chunk in range(args.sib_chunks):
            rep = d6.train_rows_one(bank, (h, lab), r_i, args.sib_chunk_steps,
                                    args.row_lr, args.device,
                                    reject_w=args.sib_reject_w, mass_w=2.0,
                                    owner_target=BAR_OWNER_MEM)
            ce_now = g0.run_in_state(
                model, blocks, fam, gates, handles, "both",
                lambda: d1.ce_in_state(model, blocks, own[:args.eval_batches],
                                       args.device, None, None, "off"))
            green = probe_green()
            ok = green and (ce_now <= base_ce + 0.02)
            trips.append({"chunk": chunk, "ce": round(ce_now, 4),
                          "probe_green": green,
                          "p_home": rep["last"]["p_new_code"]})
            print(f"  [sib {name} chunk {chunk}] ce {ce_now:.4f} "
                  f"(base {base_ce:.4f}) probe_green={green}", flush=True)
            if ok:
                best = bank.P.data[r_i].float().cpu().clone()
            if not ok:
                break
        if best is not None:
            with torch.no_grad():
                bank.P.data[r_i] = best.to(bank.P.device, bank.P.dtype)
        report.append({"tag": name, "chunks": trips,
                       "kept_best": best is not None})
    return report


class _logits_fn:
    """g0.gen_reply expects logits_fn(tokens)->logits; wrap the vk model."""

    def __init__(self, model):
        self.model = model

    def __call__(self, tokens):
        lg, _, _ = self.model(tokens)
        return lg


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase",
                    choices=("shape", "registry", "fit", "assemble", "all"),
                    default="all")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--registry", default=os.path.join(DEFAULT_PLUGINS,
                                                       "registry_vk.pt"))
    ap.add_argument("--specialties-dir",
                    default="/data/tliao/vesper_sft/VesperLM/Dataset/data/specialties")
    ap.add_argument("--cores-dir", default=DEFAULT_CORES)
    ap.add_argument("--plugins-dir", default=DEFAULT_PLUGINS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--wrap-subset", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--d-canon", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--safe-bt", type=int, nargs="*", default=[128, 192, 256])
    ap.add_argument("--max-tokens", type=int, default=300000)
    ap.add_argument("--max-base-tokens", type=int, default=40000)
    ap.add_argument("--episode-tile", type=int, default=4096)
    ap.add_argument("--expert-steps", type=int, default=500)
    ap.add_argument("--expert-lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--out-pen", type=float, default=300.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--force-recal", action="store_true")
    ap.add_argument("--recal-steps", type=int, default=120)
    ap.add_argument("--recal-lr", type=float, default=0.05)
    ap.add_argument("--max-cal-pairs", type=int, default=1024)
    ap.add_argument("--reg-seq-len", type=int, default=128)
    ap.add_argument("--reg-batches", type=int, default=8)
    ap.add_argument("--row-steps", type=int, default=80)
    ap.add_argument("--row-lr", type=float, default=0.05)
    ap.add_argument("--reject-w", type=float, default=10.0)
    ap.add_argument("--mass-w", type=float, default=2.0)
    ap.add_argument("--sib-chunks", type=int, default=4)
    ap.add_argument("--sib-chunk-steps", type=int, default=30)
    ap.add_argument("--sib-reject-w", type=float, default=15.0)
    ap.add_argument("--lam-init", type=float, default=0.5)
    ap.add_argument("--lam-home-batches", type=int, default=12)
    ap.add_argument("--lam-base-batches", type=int, default=12)
    ap.add_argument("--pair-batches", type=int, default=16)
    ap.add_argument("--pair-base", type=int, default=8)
    ap.add_argument("--eval-batches", type=int, default=6)
    ap.add_argument("--eval-base", type=int, default=6)
    ap.add_argument("--joint-batches-per-home", type=int, default=8)
    args = ap.parse_args()

    if args.phase == "shape":
        run_shape(args)
    elif args.phase == "registry":
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
