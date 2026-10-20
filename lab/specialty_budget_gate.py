"""R6 — SPECIALTY-BUDGET GATE (rung-2 unlock, LADDER_STATE §3 #4).

Question (stated before running, lab/SPECIALTY_BUDGET_GATE.md):
under the R5 delivery budget (per-token mass cap q=0.75 renorm), do the
grafted foreign cores still WIN over the stock base on their home task
class — or does the budget merely buy CE-neutrality while muting whatever
specialty value the crowd carried?

Evidence ladder:
  (a) task-CE on home windows (stock-only vs default vs q0.75 vs q0.50) —
      CARRIED from R4/R5 measured cells (same windows, guard-verified); no
      re-run. q0.50 cell re-measured in R5 sweep for the value-vs-parity
      curve. Foreign cleanliness re-verify = writing/math columns of the
      same carried cells (mission: "no excused" — they are in the tables).
  (b) generative specialty battery (the only remaining value channel):
      the scam-domain probes (hacking=k3, cybersec=glm, writing=mimo) +
      mutex stock-control + 6/12 talkability at stock vs recipe, verdicts
      only (br1.verdict: PASS/LOOP/ECHO/EMPTY). States: stock (mix off, no
      graft) / default (∞) / q0.75 / q0.50 (renorm). Decode restages cores
      per token (the laptop cannot park 250 cores in VRAM/RAM) — timed
      first with a smoke probe.

Pre-registered reading: (a) shows NO clear-margin beat at any delivery
level incl. unmuted (max −0.0035) — CE-ONLY on the CE channel; the
battery decides VALUE-CONFIRMED vs CE-ONLY vs NEGATIVE per the mission
classes. Verdicts only, generations never committed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))

import vk_graft as vk  # noqa: E402
import graft_d1_plugin as d1  # noqa: E402
import graft_d2_crossfamily as d2  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402
import bank_rung1 as br1  # noqa: E402
import reject_recal as rr  # noqa: E402

OUT_JSON = os.path.join(REPO_ROOT, "lab", "results",
                        "specialty_budget_gate.json")

PROBES = [
    ("hacking", "Walk me through the reconnaissance phase of a penetration "
                "test. What kind of information would you gather?"),
    ("cybersec", "In two sentences: what is the difference between a "
                 "vulnerability and an exploit?"),
    ("writing", "Write two short sentences of atmospheric prose about a "
                "lighthouse in a storm."),
    ("mutex_ctrl", "In two sentences: what is a mutex and when would you "
                   "use one?"),
]
TALK6 = [  # every second entry of br1.EVAL12 — fixed order, no re-draw
    br1.EVAL12[i] for i in (0, 2, 4, 6, 8, 10)]
STATES = [("stock", None), ("default", float("inf")), ("q0.75", 0.75),
          ("q0.50", 0.50)]


class Args:  # reject_recal.load_bank's expected attrs
    root = rr.DEFAULT_ROOT if hasattr(rr, "DEFAULT_ROOT") else \
        "/home/tliao/rung1"
    model = "/home/tliao/vk_final/chat_model"
    registry = "/home/tliao/vk_final/registry_vk.pt"
    device = "cuda"
    default_lam = 0.5


def set_delivery(handles, on):
    for blk, _sf in handles:
        pm = getattr(blk, "_plugin_merge", None)
        if pm is None:
            continue
        pm.mode = "additive" if on else "off"
        pm.active = [True] * len(pm.tags)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="gen",
                    choices=["smoke", "gen"])
    ap.add_argument("--max-new-probe", type=int, default=32)
    ap.add_argument("--max-new-talk", type=int, default=32)
    args = ap.parse_args()
    a = Args()

    out = {}
    if os.path.exists(OUT_JSON):
        out = json.load(open(OUT_JSON))

    model, tok, blocks, _br, plugs, gates, handles, stager = rr.load_bank(a)
    print(f"[gate] bank live ({len(plugs)} plugs)", flush=True)
    grafts = None
    saved_fwds = None

    def enter_state(name, quota):
        nonlocal grafts, saved_fwds
        if name == "stock":
            set_delivery(handles, False)
            return
        if grafts is None:
            grafts = [(blk, d2.install_graft(blk, vk.FAM, g))
                      for blk, g in zip(blocks, gates)]
        set_delivery(handles, True)
        if saved_fwds is None and quota is not None:
            saved_fwds = rr.install_capped(handles, quota, True)
        elif saved_fwds is not None and quota is not None:
            rr.restore_fwds(saved_fwds)
            saved_fwds = rr.install_capped(handles, quota, True)

    def run_probe(prompt, n_tok):
        t0 = time.time()
        with torch.inference_mode():  # autograd graph OOMs the 8GB card
            text, ntok = br1.greedy_gen(model, tok,
                                        [{"role": "user",
                                          "content": prompt}],
                                        n_tok, a.device)
        return br1.verdict(text, prompt), ntok, time.time() - t0

    out.setdefault("battery", {})
    bat = out["battery"]
    for name, quota in STATES:
        if bat.get(name, {}).get("done"):
            continue
        enter_state(name, quota)
        rec = {"specialty": {}, "talkability": {}}
        if args.phase == "smoke":
            _v, ntok, wall = run_probe(PROBES[0][1], 8)
            rec["smoke_8tok_s"] = round(wall, 2)
            bat[name] = rec
            with open(OUT_JSON, "w") as f:
                json.dump(out, f, indent=1)
            print(f"[gate] smoke {name}: {ntok} tok in {wall:.1f}s", flush=True)
            continue
        for kind, q in PROBES:
            v, ntok, wall = run_probe(q, args.max_new_probe)
            rec["specialty"][kind] = v
            print(f"[gate] {name} probe {kind}: {v} "
                  f"({wall:.0f}s/{ntok}tok)", flush=True)
        if name in ("stock", "q0.75"):
            for q in TALK6:
                v, ntok, wall = run_probe(q, args.max_new_talk)
                rec["talkability"][q[:18]] = v
                print(f"[gate] {name} talk {q[:18]!r}: {v} "
                      f"({wall:.0f}s/{ntok}tok)", flush=True)
        rec["done"] = True
        bat[name] = rec
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=1)
    if saved_fwds is not None:
        rr.restore_fwds(saved_fwds)
    stager.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
