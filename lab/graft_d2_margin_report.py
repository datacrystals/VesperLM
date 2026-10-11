#!/usr/bin/env python3
"""Build the D2 margin-recal report: before/after per-layer table + curve.

Consumes the D2 attempt-1 result (`phimoe_d2.json`, the BEFORE) plus the
score-margin-aware recal arm outputs (`phimoe_d2_exact.json`,
`phimoe_d2_mw{2,5,10}.json`, the AFTER points) and emits:

  * compact results JSON (committed) — curve, gates vs bars, per-layer
    before/after, and the family-relative agreement baseline
    (noise-matched agreement at the graft's own score-error level);
  * markdown table (committed) for the handoff / runbook.

Usage (after rsync'ing the box JSONs to the input dir):
  python3 lab/graft_d2_margin_report.py \
      --in-dir lab/imported/graft_d2 \
      --json-out lab/graft_d2_margin_recal.json \
      --md-out lab/graft_d2_margin_recal.md
"""

from __future__ import annotations

import argparse
import json
import os

BARS = {"logit_cosine": 0.95, "abs_ce_delta": 0.05,
        "layer_moe_cosine": 0.9, "topk_agree": 0.8}


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def means(parity: dict) -> dict:
    lc = parity["per_layer_moe_cosine"]
    ag = parity["per_layer_topk_agree"]
    jc = parity["per_layer_topk_jaccard"]
    return {
        "logit_cosine_mean": parity["logit_cosine_mean"],
        "logit_cosine_min": parity["logit_cosine_min"],
        "ce_stock": parity["ce_stock"],
        "ce_graft": parity["ce_graft"],
        "ce_delta": parity["ce_delta"],
        "mean_layer_moe_cosine": sum(lc) / max(1, len(lc)),
        "mean_topk_agree": sum(ag) / max(1, len(ag)),
        "mean_topk_jaccard": sum(jc) / max(1, len(jc)),
    }


def gates_vs(m: dict) -> dict:
    return {
        "logit_cosine_ge_0.95": m["logit_cosine_mean"] >= BARS["logit_cosine"],
        "abs_ce_delta_le_0.05": abs(m["ce_delta"]) <= BARS["abs_ce_delta"],
        "layer_cosine_gt_0.9": m["mean_layer_moe_cosine"] > BARS["layer_moe_cosine"],
        "topk_agree_gt_0.8": m["mean_topk_agree"] > BARS["topk_agree"],
        "main_bars_hold": (m["logit_cosine_mean"] >= BARS["logit_cosine"]
                           and abs(m["ce_delta"]) <= BARS["abs_ce_delta"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="lab/imported/graft_d2")
    ap.add_argument("--before", default="phimoe_d2.json")
    ap.add_argument("--arms", default="phimoe_d2_exact.json,phimoe_d2_mw2.json,"
                                      "phimoe_d2_mw5.json,phimoe_d2_mw10.json")
    ap.add_argument("--json-out", default="lab/graft_d2_margin_recal.json")
    ap.add_argument("--md-out", default="lab/graft_d2_margin_recal.md")
    args = ap.parse_args()

    before = load(os.path.join(args.in_dir, args.before))
    b = means(before["parity"])
    b_gates = gates_vs(b)

    arms = {}
    for fn in args.arms.split(","):
        fn = fn.strip()
        if not fn:
            continue
        path = os.path.join(args.in_dir, fn)
        if not os.path.exists(path):
            print(f"missing arm output: {path}")
            continue
        d = load(path)
        label = fn.replace("phimoe_d2_", "").replace(".json", "")
        m = means(d["parity"])
        rd = d.get("recal", {}) or {}
        rounds = rd.get("round_detail") or []
        entry = {
            "label": label,
            "file": fn,
            "init": d.get("init"),
            "recal": {"margin_w": rd.get("margin_w"),
                      "margin_tau": rd.get("margin_tau"),
                      "rounds": rd.get("rounds"),
                      "steps": rounds[0].get("steps") if rounds else 0},
            "metrics": m,
            "gates": gates_vs(m),
            "registry_path": d.get("registry_path"),
            "wall_s": d.get("wall_s"),
            "peak_vram_gib": d.get("peak_vram_gib"),
        }
        ma = d.get("margin_analysis")
        if ma:
            entry["margin_baseline"] = {
                "mean_score_err_rms": ma["mean_score_err_rms"],
                "mean_graft_topk_agree": ma["mean_graft_topk_agree"],
                "mean_noise_matched_agree": ma["mean_noise_matched_agree"],
                "mean_agree_ratio": ma["mean_agree_ratio"],
                "noise_matched_curve_means": ma["noise_matched_curve_means"],
                "per_layer": [
                    {"layer": pl["layer"],
                     "score_err_rms": pl["score_err_rms"],
                     "graft_topk_agree": pl["graft_topk_agree"],
                     "noise_matched_agree": pl["noise_matched_agree"],
                     "agree_ratio": pl["agree_ratio"],
                     "gap23_mean": pl["gap23_mean"],
                     "frac_gap23_lt_2sigma": pl["frac_gap23_lt_2sigma"]}
                    for pl in ma["per_layer"]],
            }
        arms[label] = entry

    # per-layer before/after for every measured arm
    per_layer = {}
    for i in range(before["n_layers"]):
        row = {
            "before_moe_cos": before["parity"]["per_layer_moe_cosine"][i],
            "before_topk_agree": before["parity"]["per_layer_topk_agree"][i],
        }
        for label, entry in arms.items():
            d = load(os.path.join(args.in_dir, entry["file"]))
            pl = d["parity"]
            if i < len(pl["per_layer_moe_cosine"]):
                row[f"{label}_moe_cos"] = pl["per_layer_moe_cosine"][i]
                row[f"{label}_topk_agree"] = pl["per_layer_topk_agree"][i]
        per_layer[str(i)] = row

    out = {
        "experiment": "d2-margin-aware-recal",
        "claim": "score-margin-aware recal (MSE weighted on near-threshold "
                 "expert pairs) closes the Phi-mini-MoE per-layer gap",
        "bars": BARS,
        "before": {"metrics": b, "gates": b_gates,
                   "file": args.before},
        "arms": arms,
        "per_layer": per_layer,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.json_out}")

    # markdown
    lines = []
    lines.append("# D2 follow-up: score-margin-aware recal (Phi-mini-MoE graft)\n")
    lines.append(f"Bars: logit cos ≥ {BARS['logit_cosine']}, |dCE| ≤ "
                 f"{BARS['abs_ce_delta']}, per-layer MoE cos > "
                 f"{BARS['layer_moe_cosine']}, top-k agree > "
                 f"{BARS['topk_agree']} (aux gates D2 missed).\n")
    lines.append("## Curve (mean over 32 layers; BEFORE = D2 attempt 1)\n")
    lines.append("| arm | margin_w | logit cos | \\|dCE\\| | MoE cos | top-k agree | Jaccard | aux bars |")
    lines.append("|---|---|---|---|---|---|---|---|")
    rows = [("BEFORE (uniform MSE)", "-", b)]
    for label, entry in arms.items():
        rows.append((label, entry["recal"].get("margin_w"), entry["metrics"]))
    for name, mw, m in rows:
        aux = ("PASS" if (m["mean_layer_moe_cosine"] > BARS["layer_moe_cosine"]
                          and m["mean_topk_agree"] > BARS["topk_agree"])
               else "FAIL")
        lines.append(
            f"| {name} | {mw} | {m['logit_cosine_mean']:.5f} | "
            f"{abs(m['ce_delta']):.4f} | {m['mean_layer_moe_cosine']:.4f} | "
            f"{m['mean_topk_agree']:.4f} | {m['mean_topk_jaccard']:.4f} | {aux} |")
    lines.append("")
    lines.append("## Per-layer before/after\n")
    labels = list(arms.keys())
    head = "| layer | cos before | agree before |"
    sep = "|---|---|---|"
    for lb in labels:
        head += f" cos {lb} | agree {lb} |"
        sep += "---|---|"
    lines.append(head)
    lines.append(sep)
    for i in range(before["n_layers"]):
        row = per_layer[str(i)]
        line = (f"| {i} | {row['before_moe_cos']:.4f} | "
                f"{row['before_topk_agree']:.4f} |")
        for lb in labels:
            line += (f" {row.get(f'{lb}_moe_cos', float('nan')):.4f} | "
                     f"{row.get(f'{lb}_topk_agree', float('nan')):.4f} |")
        lines.append(line)
    lines.append("")
    lines.append("## Family-relative agreement baseline "
                 "(noise-matched, per `margin_analysis`)\n")
    lines.append("| arm | score RMS | graft agree | noise-matched agree @1σ | agree ratio |")
    lines.append("|---|---|---|---|---|")
    for label, entry in arms.items():
        mb = entry.get("margin_baseline")
        if not mb:
            continue
        lines.append(
            f"| {label} | {mb['mean_score_err_rms']:.5f} | "
            f"{mb['mean_graft_topk_agree']:.4f} | "
            f"{mb['mean_noise_matched_agree']:.4f} | "
            f"{mb['mean_agree_ratio']:.3f} |")
    with open(args.md_out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
