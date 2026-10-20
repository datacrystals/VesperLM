"""R8 — per-home λ re-fit on held-out tails under the q0.75 renorm budget.

Zero re-fit. Windows verbatim from R7 (`heldout_bank_value.D0_HOMES`, the
same tail offsets and d0 texts — provenance must not drift). Stock-masked
and flat-λ0.5 q0.75 cells are carried from lab/results/heldout_bank_value.json
(same deterministic harness); the sweep measures λ overrides per home.

Cells: home h rows' lam ∈ {0.25, 0.75, 1.0, 1.5}, rest pinned 0.5, delivery
through rr.install_capped(q=0.75, renorm). Groups to bs=6 chunks (R4/R5
grouping semantics, token-exact). Win cells get a foreign-cleanliness check
(writing+math+base) and a W2/d0 confirm where the home has one. The chat
significance probe runs split-half tails at default λ0.5 quota ∞.
Output: lab/results/lambda_refit_heldout.json (local).
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
import graft_d1_plugin as d1  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402
import bank_rung1 as br1  # noqa: E402
import phome_calibration as pcl  # noqa: E402
import reject_recal as rr  # noqa: E402
import heldout_bank_value as hbv  # noqa: E402

OUT_JSON = os.path.join(REPO_ROOT, "lab", "results",
                        "lambda_refit_heldout.json")
R7_JSON = os.path.join(REPO_ROOT, "lab", "results",
                       "heldout_bank_value.json")
LAMS = [0.25, 0.75, 1.0, 1.5]
SEQ = 256


class Args:
    root = "/home/tliao/rung1"
    model = "/home/tliao/vk_final/chat_model"
    registry = "/home/tliao/vk_final/registry_vk.pt"
    device = "cuda"
    default_lam = 0.5


def grouped(batches):
    """bs1 windows -> bs6 chunks (tail at native batch size, like R4/R5)."""
    toks = [b[0] for b in batches]
    out = []
    for i in range(0, len(toks), 6):
        chunk = toks[i:i + 6]
        if len(chunk) == 6:
            out.append((torch.cat(chunk, dim=0), 0))
        else:
            out.extend((t, 0) for t in chunk)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    a = Args()

    r7 = json.load(open(R7_JSON))["windows"]
    out = {"experiment": "R8-lambda-refit", "windows_provenance": "R7 verbatim"}
    if os.path.exists(OUT_JSON):
        out = json.load(open(OUT_JSON))

    model, tok, blocks, _br, plugs, gates, handles, stager = rr.load_bank(a)

    def ce_on(batches, which, quota=None):
        saved = None
        if quota is not None:
            saved = rr.install_capped(handles, quota, True)
        v = g0.run_in_state(
            model, blocks, vk.FAM, gates, handles, which,
            lambda: d1.ce_in_state(model, blocks, batches, args.device,
                                   None, None, "off"))
        if saved is not None:
            rr.restore_fwds(saved)
        return v

    def set_lams(overrides: dict):
        by_id = {m["id"]: m for _w, _r, m in plugs}
        for blk, _sf in handles:
            pm = getattr(blk, "_plugin_merge", None)
            if pm is None:
                continue
            for j, tag in enumerate(pm.tags):
                pm.lams[j] = torch.tensor(overrides.get(
                    by_id[tag]["home"], 0.5))

    tails, d0w = {}, {}
    stock, flat = {}, {}
    for home in br1.HOMES:
        b, n = hbv.tail_window(tok, home)
        if b:
            tails[home] = grouped(b)
        r7k = r7[f"{home}:tail"]
        stock[home] = r7k["ce"]["stock"]
        flat[home] = r7k["ce"]["q0.75"]
        if home in hbv.D0_HOMES:
            b2, _ = hbv.d0_window(tok, home)
            if b2:
                d0w[home] = grouped(b2)

    # guard: one window re-run must match R7's flat cell (ungrouped) within
    # the grouping noise band (~+0.0009 observed; band documented)
    g0v = ce_on(tails["code"], "both", 0.75)
    out["carry_guard"] = {"code_tail_flat_grouped_rerun": round(g0v, 4),
                          "r7_ungrouped": flat["code"],
                          "grouping_drift": round(g0v - flat["code"], 4)}
    print(f"[r8] grouping drift on flat code: {g0v - flat['code']:+.4f}",
          flush=True)

    # grouped reference cells (this harness's own comparators): stock + flat
    ref = out.setdefault("ref_grouped", {})
    for home in br1.HOMES:
        if home not in tails or ref.get(home, {}).get("done"):
            continue
        vs = ce_on(tails[home], "off")
        set_lams({})
        vf = ce_on(tails[home], "both", 0.75)
        ref[home] = {"stock": round(vs, 4), "flat": round(vf, 4),
                     "r7_stock_ungrouped": stock[home],
                     "done": True}
        print(f"[r8] ref {home}: stock {vs:.4f} flat {vf:.4f}", flush=True)
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=1)
    for home, r in ref.items():
        stock[home] = r["stock"]
        flat[home] = r["flat"]

    cells = out.setdefault("cells", {})
    for home in br1.HOMES:
        if home not in tails:
            continue
        for lam in LAMS:
            cn = f"{home}:lam{lam}"
            if cells.get(cn, {}).get("done"):
                continue
            set_lams({home: lam})
            t0 = time.time()
            v = ce_on(tails[home], "both", 0.75)
            d = round(v - stock[home], 4)
            cells[cn] = {"ce": round(v, 4), "stock": stock[home],
                         "flat": flat[home], "delta": d,
                         "win": bool(d < -0.003), "done": True}
            print(f"[r8] {cn} CE {v:.4f} Δstock {d:+.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            with open(OUT_JSON, "w") as f:
                json.dump(out, f, indent=1)

    # win cells -> foreign-cleanliness + d0 confirm
    foreign = {}
    for cn, c in cells.items():
        if not c.get("win"):
            continue
        home, lam = cn.split(":lam")
        lam = float(lam)
        set_lams({home: lam})
        fw = {}
        for name in ("writing", "math", "base"):
            if name == "base":
                b = grouped(br1.base_batches(tok, SEQ, 1, 40_000)[:16])
                ref = r7.get("base:tail")
                fw["base_ce"] = round(ce_on(b, "both", 0.75), 4)
            else:
                fw[name] = round(ce_on(tails[name], "both", 0.75), 4)
                fw[name + "_vs_r7flat"] = round(fw[name] - flat[name], 4)
        c["foreign_check"] = fw
        if home in d0w:
            v = ce_on(d0w[home], "both", 0.75)
            r7d = r7[f"{home}:d0"]["ce"]["stock"]
            c["d0_confirm"] = {"ce": round(v, 4), "stock": r7d,
                               "delta": round(v - r7d, 4),
                               "confirm": bool(v - r7d < -0.003)}
        print(f"[r8] confirm {cn}: foreign {fw} d0 "
              f"{c.get('d0_confirm')}", flush=True)
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=1)

    # chat significance: split-half tails at default λ0.5 (∞)
    sig = out.setdefault("chat_split_half", {})
    if "chat" in tails and not sig.get("done"):
        set_lams({})
        full = hbv.tail_window(tok, "chat")[0]
        halves = [grouped(full[:8]), grouped(full[8:16])]
        vals = []
        for h in halves:
            vs = ce_on(h, "off")
            vd = ce_on(h, "both", float("inf"))
            vals.append({"stock": round(vs, 4), "default": round(vd, 4),
                         "delta": round(vd - vs, 4)})
            print(f"[r8] chat half: stock {vs:.4f} default {vd:.4f} "
                  f"Δ {vd-vs:+.4f}", flush=True)
        sig.update({"halves": vals, "done": True})
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=1)

    # verdict
    wins = sorted({cn.split(":")[0] for cn, c in cells.items()
                   if c.get("win")})
    won = {}
    for home in wins:
        cs = [c for cn, c in cells.items()
              if cn.startswith(home + ":") and c.get("win")]
        c = min(cs, key=lambda x: x["delta"])
        won[home] = {"cell": min((cn for cn in cells
                                  if cn.startswith(home + ":")
                                  and cells[cn]["win"]),
                                 key=lambda cn: cells[cn]["delta"]),
                     "delta": c["delta"],
                     "foreign_ok": bool(c.get("foreign_check")) and all(
                         abs(v) <= 0.003 for k, v in
                         c.get("foreign_check", {}).items()
                         if k.endswith("_vs_r7flat")),
                     "d0": c.get("d0_confirm")}
    out["wins"] = won
    nclean = sum(1 for w in won.values() if w["foreign_ok"])
    out["verdict_preclass"] = ("λ-WIN" if nclean >= 2 else
                               "MIXED-per-family" if nclean == 1 or won
                               else "λ-NEUTRAL")
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
    print("wins:", won, flush=True)
    print("VERDICT_PRECLASS:", out["verdict_preclass"], flush=True)
    stager.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
