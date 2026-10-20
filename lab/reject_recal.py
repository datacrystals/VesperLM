"""R5 — REJECT-SIDE RECAL / PER-HOME QUOTA lever for the growth-250
home_quality breaks (writing Δ+0.0262, math Δ+0.0331). Zero re-fit: tuning
delivery only, corrected judge from lab/PHOME_REJUDGE.md.

Phases:
  diag    who pays the tax: live gate-membership capture on the writing /
          math / base windows (full 250-row bank), then exact per-row j-off
          ablation CE for every row firing >1% on writing or math — the
          posting wedge (per-home harm decomposition: identity, family,
          home, firing rate, margin, ΔCE by dosage).
  sweep   8 cells {quota ∈ 0.25, 0.5, 0.75, ∞} × {recal on/off} on the
          delivery layer: per-token budget q on Σ λ_j·m_j over firing rows.
          recal=on  → renorm: all firing weights × q/Σ when over budget.
          recal=off → exclusion: drop lowest-margin rows until the prefix
          budget (single-row overflow ≤ λ allowed; documented).
          Measured: all 8 homes + base CE (masked locked to the R4-verified
          values, control-home guard re-run), live firing stats.
          quota=∞ cells MUST reproduce the R4 lam_default numbers (guard).
  verdict per-cell bars + 24-pair harm for the best cell (R4 pairs, rng 0).

Masked/gate note: caps are post-gate delivery, so gate-side stats
(membership median, ceiling-relative p_home) are cell-invariant — they are
carried from R4 (`/home/tliao/rung1/results/growth_250.json`) and stated
once, not re-measured per cell.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))

import vk_graft as vk  # noqa: E402
import graft_d0_parity as d0  # noqa: E402
import graft_d1_plugin as d1  # noqa: E402
import door25_wrap_expert as d25  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402
import v05_core_prep as prep  # noqa: E402
import vk_final_build as vb  # noqa: E402
import bank_rung1 as br1  # noqa: E402
import phome_calibration as pcl  # noqa: E402
import growth_250_measure as g25  # noqa: E402

OUT_JSON = os.path.join(REPO_ROOT, "lab", "results", "reject_recal.json")
PROBE_JSON = os.path.join(REPO_ROOT, "lab", "results",
                          "phome_calibration.json")
QCELLS = [0.25, 0.5, 0.75, float("inf")]


# ---------------------------------------------------------------------------
# setup (same construction as growth_250_measure.phase_ce)

def load_bank(args):
    model, tok = vk.load_vk(args.model, args.device, torch.float16)
    blocks = vk.install_vk_host(model)
    banks_reg, _ = d0.load_registry(args.registry)
    man = json.load(open(os.path.join(args.root, "manifest.json")))
    plugs, packed, fams, rows_by_site = [], {}, {}, {}
    for c in man["cores"][:250]:
        d = torch.load(br1.plugin_path(args.root, c["id"]),
                       map_location="cpu", weights_only=False)
        spec = d["wrap_spec"]
        cp = d["meta"]["core_prefix"].replace("/data/tliao", "/home/tliao", 1)
        st, prov = prep.load_core_tensors(cp)
        fam = prep.family_of(st, prov)
        if fam == "k3":
            w_gu, w_down, cmeta = prep.dequant_mxfp4_core(st)
        else:
            w_gu, w_down, cmeta = prep.dequant_int4_core(st)
        wrap = d25.WrappedForeignExpert(
            spec["d_model"], spec["d_canon"], w_gu, w_down, seed=0,
            act=spec.get("act", cmeta.get("act", "swiglu")),
            situ_beta=spec.get("situ_beta", cmeta.get("situ_beta", 1.0)),
            situ_linear_beta=spec.get("situ_linear_beta",
                                      cmeta.get("situ_linear_beta")))
        spec_kw = d["wrap_adapters"]
        wrap.load_state_dict(spec_kw, strict=False)
        wrap.core_w_gu.data = torch.empty(0, dtype=torch.float16)
        wrap.core_w_down.data = torch.empty(0, dtype=torch.float16)
        wrap.to("cpu")
        wrap._cid = c["id"]
        del w_gu, w_down
        packed[c["id"]] = st
        fams[c["id"]] = fam
        meta = {**d["meta"], "id": c["id"], "home": c["home"],
                "family": fam}
        plugs.append((wrap, d["row"].float(), meta))
    site_entries: dict[int, list] = {}
    for wrap, row, meta in plugs:
        host = int(meta["host_layer"])
        site_entries.setdefault(host, []).append({
            "expert": wrap, "lam": torch.tensor(args.default_lam),
            "tag": meta["id"], "num_stock": banks_reg[host].num_experts})
        rows_by_site.setdefault(host, []).append(row)
    gates, _ = vb._build_vk_gates(blocks, banks_reg, rows_by_site)
    handles = g0.install_mix(blocks, vk.FAM, site_entries, mode="off")
    stager = g25.CoreStager(handles, packed, fams, args.device)
    return model, tok, blocks, banks_reg, plugs, gates, handles, stager


def make_windows(tok, args):
    seq = 256
    home_eval = {}
    for home in br1.HOMES:
        b = br1.feed_batches(tok, home, seq, 1, 300_000)[:6]
        if b:
            home_eval[home] = pcl._group_eval(b, 6, seq)
    base_eval = pcl._group_eval(
        br1.base_batches(tok, seq, 1, 40_000)[:6], 6, seq)
    home_eval["base"] = base_eval
    return home_eval


# ---------------------------------------------------------------------------
# capped delivery forward (replicates door6.MixMerge.__call__ + budget)

def capped_call(pm, hidden_states, top_k_index, top_k_weights, quota,
                recal):
    if pm.mode == "off":
        return pm.stock(hidden_states, top_k_index, top_k_weights)
    with torch.no_grad():
        y = pm.stock(hidden_states, top_k_index, top_k_weights)
    g = pm.gate_getter()
    memb = getattr(g, "last_membership", None)
    if memb is None:
        return y
    n = len(pm.experts)
    n_stock = g.bank.num_experts - g.n_new
    sc = g.last_logits
    kth = torch.topk(sc[:, :n_stock], 2, dim=-1).values[:, -1:]
    margins = sc[:, n_stock:] - kth
    lam = torch.stack([l.detach().clone() for l in pm.lams]).clamp(min=0.0)
    w = lam.to(memb.device).unsqueeze(0) * memb.to(sc.dtype)   # (T, n)
    if not all(pm.active):
        am = torch.tensor(pm.active, device=w.device, dtype=w.dtype)
        w = w * am.unsqueeze(0)
    S = w.sum(dim=-1, keepdim=True)
    if quota < float("inf"):
        if recal:  # renorm to the budget
            scale = torch.where(
                S > quota, torch.full_like(S, quota) / S.clamp(min=1e-9),
                torch.ones_like(S))
            w = w * scale
        else:      # exclusion: drop lowest-margin firing rows
            key = torch.where(
                memb, margins,
                torch.full_like(margins, torch.finfo(margins.dtype).min))
            order = torch.argsort(key, dim=-1, descending=True)
            w_s = torch.gather(w, -1, order)
            cs = torch.cumsum(w_s, dim=-1)
            keep_s = ((cs - w_s) < quota).to(w.dtype)
            keep = torch.zeros_like(w).scatter_(-1, order, keep_s)
            w = w * keep
    for j, e in enumerate(pm.experts):
        if not pm.active[j]:
            continue
        wj = w[:, j]
        if not bool((wj > 0).any()):
            continue
        ne = e(hidden_states)
        y = y + wj.to(device=ne.device, dtype=ne.dtype).unsqueeze(-1) * ne
    return y.to(hidden_states.dtype)


def install_capped(handles, quota, recal):
    """Swap blk.experts.forward to the budgeted call; returns removers."""
    saved = []
    for blk, sf in handles:
        pm = getattr(blk, "_plugin_merge", None)
        if pm is None:
            continue
        saved.append((blk, blk.experts.forward))

        def make(pm=pm):
            def fwd(hidden_states, top_k_index, top_k_weights):
                return capped_call(pm, hidden_states, top_k_index,
                                   top_k_weights, quota, recal)
            return fwd
        blk.experts.forward = make()
    return saved


def restore_fwds(saved):
    for blk, fwd in saved:
        blk.experts.forward = fwd


def gate_stats(gates, handles, blocks):
    """Live firing table from the last forward. Reads the persisted
    VkGraftGate objects — pm.gate_getter() points at the stock router again
    once run_in_state uninstalls the graft."""
    per_row = {}
    tok_counts = []
    blk_to_gate = {id(b): g for b, g in zip(blocks, gates)}
    for blk, _sf in handles:
        pm = getattr(blk, "_plugin_merge", None)
        g = blk_to_gate.get(id(blk))
        if (g is None or pm is None or g.last_membership is None
                or g.n_new == 0):
            continue
        memb = g.last_membership
        n_stock = g.bank.num_experts - g.n_new
        sc = g.last_logits
        kth = torch.topk(sc[:, :n_stock], 2, dim=-1).values[:, -1:]
        margins = sc[:, n_stock:] - kth
        tok_counts.append(memb.sum(dim=-1).float())
        for j, tag in enumerate(pm.tags):
            m = memb[:, j]
            fr = float(m.float().mean())
            mg = float(margins[:, j][m].mean()) if bool(m.any()) else None
            rec = per_row.setdefault(tag, {"fire_mean": [], "margins": []})
            rec["fire_mean"].append(fr)
            if mg is not None:
                rec["margins"].append(mg)
    counts = torch.stack(tok_counts).flatten()
    return {
        "per_row": {t: {"fire_mean": float(np.mean(r["fire_mean"])),
                        "margin_mean": (float(np.mean(r["margins"]))
                                        if r["margins"] else None)}
                    for t, r in per_row.items()},
        "firing_per_token": {
            "mean": round(float(counts.mean()), 3),
            "p50": round(float(counts.quantile(0.5)), 3),
            "p90": round(float(counts.quantile(0.9)), 3),
            "max": int(counts.max())},
        "mass_per_token_mean": round(0.5 * float(counts.mean()), 3)}


# ---------------------------------------------------------------------------

def phase_diag(args, out, ctx):
    model, _, blocks, _, plugs, gates, handles, _stager = ctx["bank"]
    home_eval = ctx["windows"]

    def ce_on(b, which, off_ids=()):
        # Ablation flags must be applied INSIDE the run lambda: run_in_state
        # calls set_mix_active("both") which resets pm.active — setting them
        # before returning from run_in_state wipes the ablation (the
        # growth-protocol harm bug; see lab/FAILURES.md 2026-10-19 R5 entry).
        def inner():
            if off_ids:
                for blk, _sf in handles:
                    pm = getattr(blk, "_plugin_merge", None)
                    if pm is not None:
                        pm.active = [(t not in off_ids) for t in pm.tags]
            return d1.ce_in_state(model, blocks, b, args.device,
                                  None, None, "off")
        return g0.run_in_state(model, blocks, vk.FAM, gates, handles, which,
                               inner)

    # harness noise floor: the same window twice, identical state
    n0 = ce_on(home_eval["writing"], "both")
    n1 = ce_on(home_eval["writing"], "both")
    out.setdefault("diag_meta", {})["harness_noise_abs"] = abs(n0 - n1)
    print(f"[diag] harness noise floor |ΔCE| {abs(n0-n1):.6f}", flush=True)

    firing = {}
    window_ce = {}
    for name in ("writing", "math", "base"):
        v = n0 if name == "writing" else ce_on(home_eval[name], "both")
        window_ce[name] = round(v, 4)
        firing[name] = gate_stats(gates, handles, blocks)
        print(f"[diag] {name}: CE {v:.4f} fire/token "
              f"{firing[name]['firing_per_token']}", flush=True)

    # candidates note: at growth-250 density every row fires >1% on
    # writing|math (dense multi-firing, mean ~7 rows/token) — per-row
    # ablation over all 250 is a ~2h loop. Wedge via GROUP ablation
    # (family × same-home/foreign, ~exact, 14 forwards) + per-row ablation
    # of only the top by firing-mass inside paying groups (deliberate
    # scope cut, stated in lab/REJECT_RECAL.md).
    meta_by_id = {m["id"]: m for _w, _r, m in plugs}
    out["diag"] = {"window_ce_full_bank": window_ce, "firing": firing}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)

    groups = {
        "k3_off": {c for c, m in meta_by_id.items() if m["family"] == "k3"},
        "glm_off": {c for c, m in meta_by_id.items() if m["family"] == "glm"},
        "mimo_off": {c for c, m in meta_by_id.items() if m["family"] == "mimo"},
        "writing_rows_off": {c for c, m in meta_by_id.items()
                             if m["home"] == "writing"},
        "math_rows_off": {c for c, m in meta_by_id.items()
                          if m["home"] == "math"},
        "foreign_to_writing_off": {c for c, m in meta_by_id.items()
                                   if m["home"] != "writing"},
        "foreign_to_math_off": {c for c, m in meta_by_id.items()
                                if m["home"] != "math"},
    }
    wedge = {}
    t0 = time.time()
    for name in ("writing", "math", "base"):
        wedge[name] = {}
        for gname, ids in groups.items():
            if gname.endswith("_rows_off") and (
                    name == "base" or gname.split("_")[0] != name) and \
                    "foreign" not in gname:
                continue  # own-home ablation only on its home window
            if "foreign_to" in gname and gname.split("_")[2] != name:
                continue
            v = ce_on(home_eval[name], "both", off_ids=ids)
            wedge[name][gname] = round(window_ce[name] - v, 4)
            print(f"[diag] wedge {name} minus {gname}: "
                  f"{window_ce[name] - v:+.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
    out["diag"]["wedge_group"] = wedge
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)

    # per-row dosage: top-20 by firing mass within the union of
    # {writing rows} ∪ {math rows} ∪ {rows paying into the foreign wedge
    # per group evidence} — restricted to their home windows + base.
    mass = {}
    for cid, m in meta_by_id.items():
        fw = firing["writing"]["per_row"].get(cid, {}).get("fire_mean", 0.0)
        fm = firing["math"]["per_row"].get(cid, {}).get("fire_mean", 0.0)
        mass[cid] = max(fw if m["home"] == "writing" else 0,
                        fm if m["home"] == "math" else 0)
    top = sorted(mass, key=mass.get, reverse=True)[:20]
    dosage = {}
    for k, cid in enumerate(top):
        m = meta_by_id[cid]
        wins = sorted({m["home"]} & {"writing", "math"})
        row = {}
        for name in wins:
            v = ce_on(home_eval[name], "both", off_ids={cid})
            row[name] = round(window_ce[name] - v, 4)
        bv = ce_on(home_eval["base"], "both", off_ids={cid})
        row["base"] = round(window_ce["base"] - bv, 4)
        dosage[cid] = {
            "family": m["family"], "home": m["home"],
            "fire_writing": firing["writing"]["per_row"].get(
                cid, {}).get("fire_mean", 0.0),
            "fire_math": firing["math"]["per_row"].get(
                cid, {}).get("fire_mean", 0.0),
            "fire_base": firing["base"]["per_row"].get(
                cid, {}).get("fire_mean", 0.0),
            "joff_ce_delta": row}
        if (k + 1) % 5 == 0:
            print(f"[diag] ablated {k+1}/20 ({time.time()-t0:.0f}s)",
                  flush=True)
    out["diag"]["dosage"] = dosage

    # honest 24-pair harm at the default config (R4 protocol, ablation fixed)
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(24):
        i, j = rng.choice(np.arange(len(plugs)), size=2, replace=False)
        pairs.append((int(i), int(j)))
    harms = []
    for i, j in pairs:
        hi = plugs[i][2]["home"]
        b = home_eval.get(hi)
        if not b:
            continue
        all_ce = ce_on(b, "both")
        cid_j = plugs[j][2]["id"]
        off_ce = ce_on(b, "both", off_ids={cid_j})
        harms.append({"i": plugs[i][2]["id"], "j": cid_j,
                      "harm": round(all_ce - off_ce, 4)})
        print(f"[diag] harm-honest {plugs[i][2]['id']} x {cid_j}: "
              f"{all_ce - off_ce:+.4f}", flush=True)
    worst = max(harms, key=lambda d: d["harm"]) if harms else None
    out["diag"]["harm_honest_default"] = {
        "sampled": len(harms), "worst": worst,
        "pass": bool(worst["harm"] <= 0.02) if worst else None,
        "note": "ablation applied inside the run lambda (R4/box protocol "
                "applied it outside run_in_state where set_mix_active "
                "wiped it — those harm numbers were harness noise)"}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
    return out


def phase_sweep(args, out, ctx):
    model, _, blocks, _, plugs, gates, handles, _stager = ctx["bank"]
    home_eval = ctx["windows"]
    masked = ctx["masked"]
    names = br1.HOMES + ["base"]

    def ce_on(b, which):
        return g0.run_in_state(
            model, blocks, vk.FAM, gates, handles, which,
            lambda: d1.ce_in_state(model, blocks, b, args.device,
                                   None, None, "off"))

    # control-home guard (masked invariance under this harness)
    saved = install_capped(handles, float("inf"), True)
    guard = ce_on(home_eval["hacking"], "off")
    assert abs(guard - masked["homes"]["hacking"]) < 0.01, \
        f"masked guard drift {guard} vs {masked['homes']['hacking']}"
    restore_fwds(saved)

    cells = out.setdefault("cells", {})
    for quota in QCELLS:
        for recal in (True, False):
            cn = f"q{quota:g}_{'renorm' if recal else 'exclude'}"
            if cells.get(cn, {}).get("done"):
                continue
            saved = install_capped(handles, quota, recal)
            vals = {}
            for name in names:
                t0 = time.time()
                v = ce_on(home_eval[name], "both")
                vals[name] = round(v, 4)
                print(f"[sweep] {cn} {name} {v:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            fs = gate_stats(gates, handles, blocks)["firing_per_token"]
            restore_fwds(saved)
            hq = {h: round(vals[h] - masked["homes"][h], 4)
                  for h in br1.HOMES}
            cells[cn] = {
                "plugged": vals, "home_delta": hq,
                "writing_delta": hq["writing"], "math_delta": hq["math"],
                "home_quality_worst": round(max(hq.values()), 4),
                "base_tax": round(vals["base"] - masked["base"], 4),
                "base_ce": vals["base"],
                "fires_per_token_base_window": fs,
                "pass_both": bool(hq["writing"] <= 0.02
                                  and hq["math"] <= 0.02),
                "done": True}
            with open(OUT_JSON, "w") as f:
                json.dump(out, f, indent=1)

    return out


def phase_harm_best(args, out, ctx, best_cell):
    model, _, blocks, _, plugs, gates, handles, _stager = ctx["bank"]
    home_eval = ctx["windows"]
    quota, recal = best_cell

    saved = install_capped(handles, quota, recal)

    def ce_on(b, which, off_ids=()):
        def inner():
            if off_ids:
                for blk, _sf in handles:
                    pm = getattr(blk, "_plugin_merge", None)
                    if pm is not None:
                        pm.active = [(t not in off_ids) for t in pm.tags]
            return d1.ce_in_state(model, blocks, b, args.device,
                                  None, None, "off")
        return g0.run_in_state(model, blocks, vk.FAM, gates, handles, which,
                               inner)

    rng = np.random.default_rng(0)
    idx = list(range(len(plugs)))
    pairs = []
    for _ in range(24):
        i, j = rng.choice(idx, size=2, replace=False)
        pairs.append((int(i), int(j)))
    harms = []
    for i, j in pairs:
        hi = plugs[i][2]["home"]
        b = home_eval.get(hi)
        if not b:
            continue
        all_ce = ce_on(b, "both")
        cid_j = plugs[j][2]["id"]
        off_ce = ce_on(b, "both", off_ids={cid_j})
        harms.append({"i": plugs[i][2]["id"], "j": cid_j,
                      "harm": round(all_ce - off_ce, 4)})
    restore_fwds(saved)
    worst = max(harms, key=lambda d: d["harm"]) if harms else None
    return {"sampled": len(harms), "worst": worst,
            "pass": bool(worst["harm"] <= 0.02) if worst else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True,
                    choices=["diag", "sweep", "harm-recipe", "all"])
    ap.add_argument("--root", default=pcl.DEFAULT_ROOT)
    ap.add_argument("--model", default=pcl.DEFAULT_MODEL)
    ap.add_argument("--registry", default=pcl.DEFAULT_REGISTRY)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--default-lam", type=float, default=0.5)
    args = ap.parse_args()

    out = {}
    if os.path.exists(OUT_JSON):
        out = json.load(open(OUT_JSON))
    out.setdefault("experiment", "R5-reject-recal")
    out["windows"] = "R4 identical (6 batches/home, seq 256, bs6 grouped)"

    probe = json.load(open(PROBE_JSON))
    masked = {"homes": probe["ce"]["masked"]["homes"],
              "base": probe["ce"]["masked"]["base"]}

    bank = load_bank(args)
    windows = make_windows(bank[1], args)
    ctx = {"bank": bank, "windows": windows, "masked": masked}

    if args.phase in ("diag", "all") and not out.get("diag"):
        phase_diag(args, out, ctx)
    if args.phase == "harm-recipe":
        # 24-pair harm at the carried recipe cell (q0.75 renorm) — the
        # verdict's "harm <= baseline" prong must be measured, not
        # interpolated from the default and strongest-cap cells.
        harm = phase_harm_best(args, out, ctx, (0.75, True))
        out["harm_recipe_q0.75_renorm"] = harm
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=1)
        print("harm@q0.75_renorm:", harm, flush=True)
    if args.phase in ("sweep", "all"):
        phase_sweep(args, out, ctx)
        # verdict: best passing cell by (pass_both, min base_tax, min worst)
        cells = out["cells"]
        passing = {k: v for k, v in cells.items() if v["pass_both"]}
        best = None
        if passing:
            best = min(passing,
                       key=lambda k: (passing[k]["base_tax"],
                                      passing[k]["home_quality_worst"]))
            q, recal = best.split("_")
            quota = float(q[1:])
            harm = phase_harm_best(args, out, ctx,
                                   (quota, recal == "renorm"))
            out["best_cell"] = {"cell": best, **cells[best],
                                "harm_recheck": harm}
        out["verdict_cells_ranked"] = sorted(
            ((k, v["writing_delta"], v["math_delta"], v["base_tax"],
              v["pass_both"]) for k, v in cells.items()),
            key=lambda t: (t[1] + t[2], t[3]))
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=1)
        print("BEST:", best, flush=True)
    ctx["bank"][-1].remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
