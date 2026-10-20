"""R3 — RE-JUDGE n=128 under the corrected §11.2 criteria (no re-fit).

Inputs (all synced, md5-verified per lab/PHOME_CALIBRATION.md §0):
  - /home/tliao/rung1 plugins (fitted rows, owner 0.35 as-fit), manifest
  - /home/tliao/vk_final registry stock banks (8 stock rows x 10 host layers)
  - /home/tliao/rung1/phome_cache eval taps (same windows as growth_128)
  - lab/results/phome_calibration.json (CE cells measured on this laptop,
    verified vs box growth_128 within ±0.003 masked / ±0.0006 worst-breaker)

Output: prints the rejudge table + dumps aggregate-only stats to
lab/results/phome_rejudge.json (LOCAL; no row vectors committed).

Corrected criteria (MODULAR_MOE.md §11.2 R2 erratum):
  (a) home-token membership distribution: median, frac >= 0.5, trend over n
      (the gate's real decision: row score > 2nd-best stock score)
  (b) ceiling-relative p_home >= 0.5 of the tie-top envelope
      (median p_home / median ceiling, plus per-row ratio median)
  (c) p_base_max on the reject side (with the n-row uniform baseline 1/R for
      scale: a bar below 1/R is unreachable-by-construction knapsack)
  (d) home_quality / base_tax as measured by the probe's verified CE cells.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))

import graft_d0_parity as d0  # noqa: E402
import bank_rung1 as br1  # noqa: E402
import phome_calibration as pcl  # noqa: E402

OUT_JSON = os.path.join(REPO_ROOT, "lab", "results", "phome_rejudge.json")
PROBE_JSON = os.path.join(REPO_ROOT, "lab", "results", "phome_calibration.json")


def fitted_rows(man, n, root):
    """(cid, site, row, home) for the first n cores, from plugin .pt rows."""
    out = []
    for c in man["cores"][:n]:
        d = torch.load(br1.plugin_path(root, c["id"]), map_location="cpu",
                       weights_only=False)
        out.append((c["id"], int(c["host_layer"]), d["row"].float(),
                    c["home"]))
    return out


def assembled_stats(banks_reg, rows, streams):
    """Per-row routing stats vs the assembled bank at this n (CPU)."""
    banks = {}
    site_rows: dict[int, list] = {}
    site_meta: dict[int, list] = {}
    for cid, site, row, home in rows:
        site_rows.setdefault(site, []).append(row)
        site_meta.setdefault(site, []).append((cid, home))
    for i, breg in enumerate(banks_reg):
        rs = site_rows.get(i, [])
        if not rs:
            continue
        bank = d0.PassportBank(breg.num_experts + len(rs), breg.d_model,
                               d_r=breg.d_r, learn_wq=breg.W_q is not None)
        with torch.no_grad():
            bank.P.data[: breg.num_experts] = breg.P.data
            if bank.W_q is not None:
                bank.W_q.data.copy_(breg.W_q.data)
            for j, r in enumerate(rs):
                bank.P.data[breg.num_experts + j] = r.to(bank.P.dtype)
        bank.layer_scale = breg.layer_scale
        banks[i] = (bank, site_meta[i])

    per_row = {}
    for site, (bank, metas) in banks.items():
        n_stock = bank.num_experts - len(metas)
        h_home_by = {}
        h_base = streams["base"][site].float()
        for j, (cid, home) in enumerate(metas):
            if home not in h_home_by:
                h_home_by[home] = streams[home][site].float()
            h_home = h_home_by[home]
            row_idx = n_stock + j
            sc_h = bank.scores(h_home)
            sc_b = bank.scores(h_base)
            kth_h = torch.topk(sc_h[:, :n_stock], 2, dim=-1).values[:, -1]
            p_h = torch.softmax(sc_h, dim=-1)[:, row_idx]
            p_b = torch.softmax(sc_b, dim=-1)[:, row_idx]
            mem = float((sc_h[:, row_idx] > kth_h).float().mean())
            z_stock = sc_h[:, :n_stock]
            z_top = z_stock.max(dim=-1).values
            rest = (z_stock.exp().sum(dim=-1)
                    + sc_h[:, n_stock:].exp().sum(dim=-1) - z_top.exp())
            ceil = float((1.0 / (1.0 + rest / z_top.exp())).mean())
            per_row[cid] = {"p_home": float(p_h.mean()),
                            "p_base": float(p_b.mean()),
                            "membership_frac": mem, "home": home,
                            "ceiling": ceil, "rows_at_site": bank.num_experts}
    ph = np.array([v["p_home"] for v in per_row.values()])
    pb = np.array([v["p_base"] for v in per_row.values()])
    me = np.array([v["membership_frac"] for v in per_row.values()])
    ce = np.array([v["ceiling"] for v in per_row.values()])
    nrows = max(v["rows_at_site"] for v in per_row.values())
    return {
        "n": len(per_row),
        "membership_median": round(float(np.median(me)), 4),
        "membership_frac_ge_05": round(float((me >= 0.5).mean()), 4),
        "p_home_median": round(float(np.median(ph)), 4),
        "ceiling_median": round(float(np.median(ce)), 4),
        "ceiling_relative_med_over_med": round(
            float(np.median(ph) / np.median(ce)), 4),
        "ceiling_relative_per_row_median": round(
            float(np.median(ph / ce)), 4),
        "p_base_median": round(float(np.median(pb)), 4),
        "p_base_max": round(float(pb.max()), 6),
        "uniform_baseline_1_over_R": round(1.0 / nrows, 4),
        "pass_ceiling_relative": bool(np.median(ph) / np.median(ce) >= 0.5),
        "detail": {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                       for kk, vv in v.items() if kk != "rows_at_site"}
                   for k, v in per_row.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=pcl.DEFAULT_ROOT)
    ap.add_argument("--registry", default=pcl.DEFAULT_REGISTRY)
    ap.add_argument("--ns", default="32,64,128")
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.root, "manifest.json")))
    banks_reg, _ = d0.load_registry(args.registry)
    streams = {name: pcl.load_cache(f"eval_{name}")
               for name in br1.HOMES + ["base"]}

    out = {"experiment": "R3-phome-rejudge", "windows": "growth_128 eval taps",
           "rows": "fitted owner0.35 as-fit (gain1 cell)"}
    for n in [int(x) for x in args.ns.split(",")]:
        rows = fitted_rows(man, n, args.root)
        st = assembled_stats(banks_reg, rows, streams)
        out[f"n{n}"] = {k: v for k, v in st.items() if k != "detail"}
        print(f"[rejudge] n={n}: memb {st['membership_median']} "
              f"(>=0.5 frac {st['membership_frac_ge_05']}) "
              f"p_home {st['p_home_median']} ceil {st['ceiling_median']} "
              f"ceilrel {st['ceiling_relative_med_over_med']} "
              f"p_base_max {st['p_base_max']} "
              f"(1/R {st['uniform_baseline_1_over_R']})", flush=True)

    # old metric (recorded for contrast only): merged row_recal readout
    recs = {}
    for lf in sorted(glob.glob(os.path.join(args.root, "results",
                                            "fit*.json"))):
        j = json.load(open(lf))
        for cid, r in (j.get("fitted", {}) or {}).items():
            rr = (r or {}).get("row_recal", {}).get("last")
            if rr and "p_new_code" in rr and cid not in recs:
                recs[cid] = rr["p_new_code"]
    out["row_recal_coverage"] = len(recs)
    old = {}
    for n in (32, 64, 128):
        ids = [c["id"] for c in man["cores"][:n]]
        ph = sorted(recs[i] for i in ids if i in recs)
        if ph:
            old[n] = {"rows": len(ph), "median": round(ph[len(ph) // 2], 4)}
    out["old_metric_fit_phome"] = old

    # CE under the corrected recipe (probe's verified cells; no re-run)
    probe = json.load(open(PROBE_JSON))
    cell = probe["ce"]["owner0.35"]
    out["ce_owner035"] = {
        "home_delta": cell["home_delta"],
        "home_quality_worst": cell["home_quality_worst"],
        "base_tax": cell["base_tax"],
        "masked_base": probe["ce"]["masked"]["base"],
        "source": "lab/results/phome_calibration.json ce.owner0.35 "
                  "(verified vs box growth_128: ±0.003 / ±0.0006)",
    }
    out["break_status_corrected"] = {
        "1_home_quality": all(v <= 0.02 for v in cell["home_delta"].values()),
        "2_pairwise_harm": json.load(open(os.path.join(
            args.root, "results", "growth_128.json")))["pairwise_harm"]
            ["pass"],
        "3_saturation_corrected": out["n128"]["pass_ceiling_relative"],
        "4_base_tax": cell["base_tax"] <= 0.05,
    }
    print("break_status_corrected:", out["break_status_corrected"],
          flush=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
