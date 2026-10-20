"""R7 — HOLD-OUT BANK VALUE windows (Door-2.5-quality at bank scale).

Window classes (provenance declared in lab/HELDOUT_BANK_VALUE.md):
  tail  — home feed tokens [204800, 208896] (held-out from wrap fit batches
          0..781, recal pairs, eval windows), all 8 homes.
  d0    — lab/graft_d0_texts/{code,math,science,web,chat}.txt whole file,
          off-fit off-corpus-channel for the wraps (5 homes).

Arms: stock (mix off) / default (∞ renorm) / q0.75 renorm (R5 recipe).
Metric: CE per window per arm, delta vs stock, judge band ±0.003.
Output: lab/results/heldout_bank_value.json (local).
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

OUT_JSON = os.path.join(REPO_ROOT, "lab", "results",
                        "heldout_bank_value.json")
D0_DIR = os.path.join(REPO_ROOT, "lab", "graft_d0_texts")
SEQ = 256
TAIL = (204800, 208896)
D0_HOMES = ["code", "math", "science", "web", "chat"]


class Args:
    root = "/home/tliao/rung1"
    model = "/home/tliao/vk_final/chat_model"
    registry = "/home/tliao/vk_final/registry_vk.pt"
    device = "cuda"
    default_lam = 0.5


def tail_window(tok, home):
    kind, rel = br1.HOME_FEED[home]
    path = br1.feed_batch_path(rel)
    if kind == "text":
        ids = tok.encode(open(path).read(), add_special_tokens=False)
    else:
        ids = np.fromfile(path, dtype=np.uint16).astype(np.int64).tolist()
    if len(ids) < TAIL[1]:
        return None, len(ids)  # BYPASS: no held-out tail
    seq_ids = ids[TAIL[0]:TAIL[1]]
    return vk.to_batches(seq_ids, 1, SEQ, 1), len(ids)


def d0_window(tok, home):
    p = os.path.join(D0_DIR, f"{home}.txt")
    if not os.path.exists(p):
        return None, 0
    ids = tok.encode(open(p).read(), add_special_tokens=False)
    return vk.to_batches(ids, 1, SEQ, 1), len(ids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    a = Args()

    out = {"experiment": "R7-heldout-bank-value",
           "tail_offset": TAIL, "seq": SEQ,
           "arms": ["stock", "default_inf_renorm", "q0.75_renorm"]}
    if os.path.exists(OUT_JSON):
        out = json.load(open(OUT_JSON))

    model, tok = None, None
    bank = rr.load_bank(a)
    model, tok, blocks, _br, plugs, gates, handles, stager = bank

    def ce(batches, which, quota=None):
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

    arms = [("stock", "off", None), ("default", "both", float("inf")),
            ("q0.75", "both", 0.75)]
    res = out.setdefault("windows", {})
    for home in br1.HOMES:
        for wname, wfun in (("tail", tail_window), ("d0", d0_window)):
            if wname == "d0" and home not in D0_HOMES:
                continue
            key = f"{home}:{wname}"
            if res.get(key, {}).get("done"):
                continue
            batches, n_ids = wfun(tok, home)
            if not batches:
                res[key] = {"bypass": "no held-out material", "n_ids": n_ids,
                            "done": True}
                print(f"[r7] {key}: BYPASS (n_ids={n_ids})", flush=True)
                continue
            vals = {}
            for aname, which, quota in arms:
                t0 = time.time()
                v = ce(batches, which, quota)
                vals[aname] = round(v, 4)
                print(f"[r7] {key} {aname} {v:.4f} ({time.time()-t0:.0f}s)",
                      flush=True)
            res[key] = {
                "n_ids": n_ids, "n_batches": len(batches), "ce": vals,
                "delta_vs_stock": {k: round(vals[k] - vals["stock"], 4)
                                   for k in ("default", "q0.75")},
                "done": True}
            with open(OUT_JSON, "w") as f:
                json.dump(out, f, indent=1)

    # per-home certainty table (delta-by-window-class)
    homes = {}
    for key, r in res.items():
        if "delta_vs_stock" not in r:
            continue
        home = key.split(":")[0]
        homes.setdefault(home, {})[key.split(":")[1]] = r
    table = {}
    for home, rws in homes.items():
        cell = {}
        for wname, r in rws.items():
            d = r["delta_vs_stock"]
            cell[wname] = {"masked": r["ce"]["stock"],
                           "best_arm_delta": min(d.values()),
                           "win_below_band": min(d.values()) < -0.003}
        wins = sum(1 for c in cell.values() if c["win_below_band"])
        cell["family_verdict"] = "WIN" if wins else "neutral"
        table[home] = cell
    out["family_table"] = table
    n_win = sum(1 for c in table.values() if c["family_verdict"] == "WIN")
    out["verdict_preclass"] = ("DOMAIN-WIN" if n_win >= 3
                               else "NEUTRAL-ALL" if n_win == 0
                               else "MIXED")
    out["n_win_families"] = n_win
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
    print("family_table:", table, flush=True)
    print("VERDICT_PRECLASS:", out["verdict_preclass"], flush=True)
    stager.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
