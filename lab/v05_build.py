#!/usr/bin/env python3
"""v0.5 — the talkable model: v0 grafted spine + 3 family cores, additive-λ.

Assembly = v0's 5 plug-ins (code / math / 3 shaped memories, artifacts in
--v0-plugins) PLUS three §10.7 specialty wraps from the flagship donors:

    k3_hack       Kimi K3          L12 e895  (mxfp4 core, situ act)   home=hacking
    glm_cybersec  GLM-5.3-Flash    L3  e0    (int4 core, swiglu)      home=cybersec
    mimo_writing  MiMo-V2.6-Flash  L1  e0    (int4 core, swiglu)      home=writing

Delivery is v0's validated additive-λ contract (NOT the bank2n partition —
bank2n's quality half is still PENDING on the box, only bare/masked states
measured, see HANDOFF; additive has a full chat-tested v0 assembly).

Recipe per §10.7 (frozen): rank-64 width-norm wrap (d_canon=64 → factor 1),
wd 0, lr 3e-3, task-CE + out_pen 300, ~250 steps, FEED >=200k specialty
tokens from Dataset/data/specialties/<home>/corpus.txt (door10 certified
budget), registry row + mex recal owner 0.85 (v0's memories keep 0.85, its
code/math rows keep the v0 0.35), then closing recal + λ fit with all 8 live.

Cores come from lab/v05_core_prep.py ({name}_core.pt: w_gu (2I,H), w_down
(H,I) + donor act).  Host sites are depth-matched to the donor layer on the
32L Phi spine (one distinct site per plug).

Phases: --phase fit (one process per GPU lane, --wrap-subset) →
        --phase assemble (joint install, closing recal, λ fit, save state).
Chat CLI: lab/chat_v05.py (loads what this writes).

Forked from grafted_spine_v0.py / bank2n_partition.py; d0/d1/d2/d25/d6 and
g0 are imported unmodified (g0 = grafted_spine_v0).
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

FORMAT = "vesper-v05"
BAR_OWNER_SPEC = 0.35          # v0's code/math rows (unchanged)
BAR_OWNER_MEM = 0.85           # v0 memories + v05 specialty rows (task line)
MIN_FEED_TOKENS = 200_000      # door10 certified calibration budget

# name, home key, host layer on the 32L spine, core .pt, donor, (L, E)
V05_WRAPS = [
    ("k3_hack", "hacking", 4, "k3_hack_core.pt", "kimi-k3", 12, 895),
    ("glm_cybersec", "cybersec", 2, "glm_cybersec_core.pt", "glm-5.3-flash", 3, 0),
    ("mimo_writing", "writing", 1, "mimo_writing_core.pt", "mimo-v2.6-flash", 1, 0),
]

# the v0 plugs that join the v0.5 bank (name, kind, host layer, home label key)
V0_PLUGS = [
    ("code", "code", 13, "code"),
    ("math", "math", 24, "math"),
    ("mem_pet", "mem", 6, "pet"),
    ("mem_edit", "mem", 19, "editor"),
    ("mem_words", "mem", 27, "words"),
]

HOME_LABEL = {"code": 1, "math": 2, "pet": 3, "editor": 4, "words": 5,
              "hacking": 6, "cybersec": 7, "writing": 8}


def dump_json(out: dict, path: str | None) -> None:
    g0.dump_json(out, path)


def load_specialty_ids(tok, path: str, max_tokens: int) -> list[int]:
    with open(path) as f:
        text = f.read()
    ids = tok.encode(text, add_special_tokens=False)[:max_tokens]
    return ids


# ---------------------------------------------------------------------------
# Phase: fit (one specialty wrap per call; lanes use --wrap-subset)
# ---------------------------------------------------------------------------

def run_fit(args) -> dict:
    t0 = time.time()
    out = {"gate": "v05-fit", "format": FORMAT, "phase": "start",
           "pid": os.getpid(), "recipe": {
               "d_canon": args.d_canon, "lr": args.expert_lr,
               "wd": args.weight_decay, "out_pen": args.out_pen,
               "steps": args.expert_steps,
               "min_feed_tokens": MIN_FEED_TOKENS}}
    dump_json(out, args.out)
    specs = [s for s in V05_WRAPS
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
    if seq * bs in g0.HOLEY_BT:
        raise RuntimeError(f"train B*T {seq * bs} is a gfx906 hole")
    base_ids = []
    for name, text in domains.items():
        base_ids.extend(tok.encode(text, add_special_tokens=False))
    base_ids = base_ids[:args.max_base_tokens]
    base_batches = g0.to_batches(base_ids, 0, seq, bs)

    for name, home, host_i, core_file, donor, dlayer, dexp in specs:
        plugin_path = os.path.join(args.plugins_dir, f"{name}_plugin.pt")
        if args.skip_done and os.path.exists(plugin_path):
            w_chk, r_chk, _, _ = d6.load_wrap_plugin(plugin_path)
            if d6.wrap_is_finite(w_chk, r_chk):
                print(f"[{name}] plugin exists and finite — skip", flush=True)
                out.setdefault("fitted", {})[name] = {"skipped": True}
                dump_json(out, args.out)
                continue
        core_path = (core_file if os.path.isabs(core_file)
                     else os.path.join(args.cores_dir, core_file))
        if not os.path.exists(core_path):
            raise RuntimeError(f"[{name}] missing core {core_path} — run "
                               f"lab/v05_core_prep.py first")
        core = torch.load(core_path, map_location="cpu", weights_only=False)
        meta = core.get("meta", {})
        wrap = d25.WrappedForeignExpert(
            d_model, args.d_canon, core["w_gu"], core["w_down"], seed=args.seed,
            act=meta.get("act", "swiglu"),
            situ_beta=meta.get("situ_beta", 1.0),
            situ_linear_beta=meta.get("situ_linear_beta"))

        spec_path = os.path.join(args.specialties_dir, home, "corpus.txt")
        spec_ids = load_specialty_ids(tok, spec_path, args.max_tokens)
        if len(spec_ids) < MIN_FEED_TOKENS * 0.75:
            print(f"[{name}] WARNING: {len(spec_ids)} tokens < 75% of the "
                  f"{MIN_FEED_TOKENS} door10 budget", flush=True)
        home_train = g0.to_batches(spec_ids, 1, seq, bs)
        if not home_train:
            raise RuntimeError(f"[{name}] empty home stream")
        print(f"== [{name}] core {donor} L{dlayer} e{dexp} "
              f"({wrap.core_param_count()} frozen, "
              f"{wrap.adapter_param_count()} adapters, act {meta.get('act')}) "
              f"-> host L{host_i} home={home} "
              f"({len(spec_ids)} spec tok / {len(home_train)} batches, "
              f"feed {args.expert_steps * seq * bs} tok-slots) ==", flush=True)
        out.setdefault("feed", {})[name] = {
            "corpus": spec_path, "tokens": len(spec_ids),
            "batches": len(home_train),
            "fed_token_slots": args.expert_steps * seq * bs,
            "epochs": round(args.expert_steps * seq * bs
                            / max(1, len(spec_ids)), 2)}
        dump_json(out, args.out)

        pa = g0.fit_wrap(model, blocks, fam, banks_reg, host_i, wrap,
                         home_train, base_batches, args, args.device,
                         steps=args.expert_steps)
        if not pa.get("finite", True):
            print(f"[{name}] NON-FINITE at lr {args.expert_lr} — refit at "
                  f"lr 1e-3 (v0 math-NaN lesson)", flush=True)
            del wrap
            torch.cuda.empty_cache()
            wrap = d25.WrappedForeignExpert(
                d_model, args.d_canon, core["w_gu"], core["w_down"],
                seed=args.seed, act=meta.get("act", "swiglu"),
                situ_beta=meta.get("situ_beta", 1.0),
                situ_linear_beta=meta.get("situ_linear_beta"))
            args_lr_old = args.expert_lr
            args.expert_lr = 1e-3
            pa = g0.fit_wrap(model, blocks, fam, banks_reg, host_i, wrap,
                             home_train, base_batches, args, args.device,
                             steps=args.expert_steps)
            args.expert_lr = args_lr_old
            if not pa.get("finite", True):
                out.setdefault("fitted", {})[name] = {
                    "error": "non_finite_twice"}
                out["phase"] = f"failed_{name}"
                dump_json(out, args.out)
                del wrap
                torch.cuda.empty_cache()
                continue

        # registry row: prototype init + mex recal owner 0.85 (task line)
        site_rows = {host_i: [torch.zeros(d_model)]}
        gates, site_banks = g0.build_gates(blocks, banks_reg, fam, site_rows)
        handles = g0.install_mix(blocks, fam, {
            host_i: [{"expert": wrap,
                      "lam": nn.Parameter(torch.ones((), device=args.device)),
                      "tag": name, "num_stock": banks_reg[host_i].num_experts}]},
            mode="off")
        # pair/eval streams are CAPPED: collect_pairs_fam and quality_reads
        # walk whole streams with a forward each — 485-batch specialty homes
        # would cost ~30 min per lane post-fit (max_cal_pairs caps the kept
        # pairs, not the walk).
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
        code_h = h[lab > 0.5]
        bank = site_banks[host_i]
        row_idx = bank.num_experts - 1
        with torch.no_grad():
            if code_h.shape[0] > 0:
                bank.P.data[row_idx] = (code_h.mean(dim=0)
                                        * math.sqrt(bank.d_r))
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
                     "kind": "domain", "home": home,
                     "donor_layer": dlayer, "donor_expert": dexp,
                     "donor": donor, "core_file": core_path,
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
# Phase: assemble (8-plug additive joint — v0's run_assemble generalized)
# ---------------------------------------------------------------------------

def run_assemble(args) -> dict:
    t0 = time.time()
    prev = {}
    if args.out and os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
        except Exception:
            prev = {}
    out = {"gate": "v05-assemble", "format": FORMAT, "phase": "start",
           "pid": os.getpid(), "delivery": "additive-lambda",
           "why_not_partition": ("bank2n quality half PENDING "
                                 "(bare+masked only; assemble crashed on "
                                 "GPU grab) — additive is the tested path")}
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

    # ---- the 8 plug-ins ----
    wrap_specs = []           # (name, kind, layer, home_label_key, owner)
    for name, kind, layer, home in V0_PLUGS:
        wrap_specs.append((name, kind, layer, home,
                           BAR_OWNER_SPEC if kind == "code" else BAR_OWNER_MEM))
    for name, home, host_i, _core, _donor, _dl, _de in V05_WRAPS:
        wrap_specs.append((name, "domain", host_i, home, BAR_OWNER_MEM))

    plugins_dir = args.plugins_dir
    wraps, rows, metas = {}, {}, {}
    missing = []
    for name, kind, layer, home, _owner in wrap_specs:
        path = os.path.join(plugins_dir, f"{name}_plugin.pt")
        if not os.path.exists(path):
            v0src = os.path.join(args.v0_plugins, f"{name}_plugin.pt")
            if os.path.exists(v0src):
                os.makedirs(plugins_dir, exist_ok=True)
                import shutil
                shutil.copy2(v0src, path)
                print(f"[{name}] copied v0 plugin {v0src}", flush=True)
            else:
                missing.append(name)
                continue
        wrap, row, meta, spec = d6.load_wrap_plugin(path)
        if not d6.wrap_is_finite(wrap, row):
            raise RuntimeError(f"non-finite plugin {path}")
        if int(meta["host_layer"]) != layer:
            raise RuntimeError(f"[{name}] host layer {meta['host_layer']} != "
                               f"spec {layer}")
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
    for name, kind, layer, home, _owner in wrap_specs:
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
    for name, kind, layer, home, _owner in wrap_specs:
        bank = site_banks[layer]
        tags_here = [en["tag"] for en in site_entries[layer]]
        row_idx[name] = bank.num_experts - len(tags_here) + tags_here.index(name)
    print("installed rows:", row_idx, flush=True)
    dump_json(out, args.out)

    # ---- joint data (8 homes + base) ----
    def home_batches(key, cap=None):
        if key in ("code", "math"):
            tr, _, _ = g0.split_home(tok, domains[key], args.max_tokens,
                                     seq, args.batch_size)
            raw = [(t, HOME_LABEL[key]) for t, _ in tr]
        elif key in ("pet", "editor", "words"):
            eps = g0.build_episodes()
            bl = g0.episode_batches(tok, [eps[key]], seq, args.batch_size)
            raw = [(t, HOME_LABEL[key]) for t, _ in bl]
        else:
            ids = load_specialty_ids(
                tok, os.path.join(args.specialties_dir, key, "corpus.txt"),
                args.max_tokens)
            raw = [(t, HOME_LABEL[key]) for t, _ in g0.to_batches(
                ids, 1, seq, args.batch_size)]
        # capped streams: full-pass walkers (collect_pairs/ce) must not see
        # 485-batch specialty homes (see run_fit note)
        n = cap if cap is not None else args.joint_batches_per_home
        return raw[:n] if n else raw
    base_ids = []
    for name, text in domains.items():
        base_ids.extend(tok.encode(text, add_special_tokens=False))
    base_ids = base_ids[:args.max_base_tokens]
    base_batches = g0.to_batches(base_ids, 0, seq, args.batch_size)

    joint_batches = []
    for _name, _kind, _layer, home, _owner in wrap_specs:
        joint_batches += home_batches(home)
    joint_batches += [(t, 0) for t, _ in base_batches[:args.pair_base]]

    site_info = []
    for name, kind, layer, home, owner in wrap_specs:
        site_info.append({
            "host_i": layer, "bank": site_banks[layer], "row_idx": row_idx[name],
            "home_label": HOME_LABEL[home], "tag": name,
            "owner_target": owner})
    frozen = d6.freeze_wraps(list(wraps.values()))
    js_path = os.path.join(plugins_dir, "joint_state.pt")
    prev_joint = prev.get("joint") or {}
    if (os.path.exists(js_path) and prev_joint.get("lam_fit")
            and not prev_joint["lam_fit"].get("aborted_nonfinite")
            and prev_joint.get("lams_final")):
        js = torch.load(js_path, map_location="cpu", weights_only=False)
        lay_of = {n: l for n, k, l, h, o in wrap_specs}
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
        print("== closing row recal (all 8 live, owner per row) ==", flush=True)
        recal = g0.closing_row_recal(model, blocks, fam, handles, gates,
                                     site_info, joint_batches, args.device,
                                     args.recal_steps, args.recal_lr, args)
        out["joint"] = {"row_idx": row_idx, "lam_init": args.lam_init,
                        "row_recal": recal}
        dump_json(out, args.out)

        print("== lambda fit (additive, 8 plugs + base stream) ==", flush=True)
        lam_fit = g0.fit_lams(model, blocks, fam, lams, handles, gates,
                              joint_batches, args.lam_steps, args.lam_lr,
                              args.device)
        out["joint"]["lam_fit"] = lam_fit
        out["joint"]["lams_final"] = {
            n: round(float(lams[i].detach().clamp(min=0)), 4)
            for i, (n, *_rest) in enumerate(wrap_specs)}
        torch.save({
            "rows": {n: site_banks[dict((nn, l) for nn, k, l, h, o in wrap_specs)[n]]
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
          for _n, _k, _l, home, _o in wrap_specs}
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
               "base_tax_bar": 0.05}
    verdict["pass"] = (verdict["all_homes_improve"]
                       and verdict["base_tax_nats"] <= verdict["base_tax_bar"])
    out["verdict"] = verdict
    out["phase"] = "assemble_done"
    out["wall_s"] = round(time.time() - t0, 2)
    dump_json(out, args.out)
    print("== verdict:", json.dumps(verdict), "==", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("fit", "assemble", "all"),
                    default="all")
    ap.add_argument("--model", default="/data/tliao/Phi-mini-MoE-instruct")
    ap.add_argument("--registry",
                    default="/data/tliao/graft_d2_out/registry_phimoe_f620b32c.pt")
    ap.add_argument("--domains-dir", default="/data/tliao/graft_d2_slice")
    ap.add_argument("--specialties-dir",
                    default="/data/tliao/vesper_sft/VesperLM/Dataset/data/specialties")
    ap.add_argument("--cores-dir", default="/data/tliao/v05_cores")
    ap.add_argument("--plugins-dir", default="/data/tliao/v05_out")
    ap.add_argument("--v0-plugins", default="/data/tliao/grafted_spine_v0_out")
    ap.add_argument("--out", default="/data/tliao/v05_out/v05_results.json")
    ap.add_argument("--wrap-subset", default="",
                    help="comma list of V05_WRAPS names (lanes)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--d-canon", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="train B*T=2048 (v0 verified shape)")
    ap.add_argument("--max-tokens", type=int, default=300000,
                    help="specialty corpus cap (door10 wants >=200k fed)")
    ap.add_argument("--max-base-tokens", type=int, default=40000)
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
    ap.add_argument("--pair-batches", type=int, default=24,
                    help="home batches for row-pair collection (walkers are "
                         "full-pass — keep small on 400+ batch homes)")
    ap.add_argument("--pair-base", type=int, default=8)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--eval-base", type=int, default=4)
    ap.add_argument("--joint-batches-per-home", type=int, default=12)
    ap.add_argument("--recal-steps", type=int, default=150)
    ap.add_argument("--recal-lr", type=float, default=0.05)
    ap.add_argument("--lam-steps", type=int, default=80)
    ap.add_argument("--lam-lr", type=float, default=0.05)
    ap.add_argument("--lam-init", type=float, default=0.5)
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
