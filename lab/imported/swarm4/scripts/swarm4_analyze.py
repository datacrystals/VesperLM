#!/usr/bin/env python3
"""swarm4 analysis: read collected results JSONs and print report tables."""
import json
import glob
import os

DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def load(pat):
    out = {}
    for p in sorted(glob.glob(os.path.join(DEST, pat))):
        with open(p) as f:
            d = json.load(f)
        out[os.path.splitext(os.path.basename(p))[0]] = d
    return out


def val_row(name, d):
    series = d.get("val_loss_series") or []
    print(f"  {name:22s} val_final={d.get('val_loss_final')} "
          f"steps={d.get('steps_done')} tok/s={d.get('tok_s_avg')} "
          f"ce_last={d.get('ce_last')} series={series}")


print("=== Exp1: t2 seed-2 paired head-to-head (tiny_agent_k, 600 steps) ===")
r = {}
for pat in ("t2s-passport*.json", "t2s-topk-baseline*.json"):
    r.update(load(pat))
for k in sorted(r):
    val_row(k, r[k])
if "t2s-passport" in r and "t2s-topk-baseline" in r:
    p = r["t2s-passport"]["val_loss_final"]
    b = r["t2s-topk-baseline"]["val_loss_final"]
    if p and b:
        print(f"  seed2 passport vs topk: {p} vs {b} -> {(b-p)/b*100:+.2f}% val")
    print(f"  unseeded t2 reference: 6.5410 vs 6.7316 -> +2.83%")

print()
print("=== Exp2: passport_dim sweep at t1 (lab_small, 600 steps, seed 1) ===")
d = {}
for pat in ("t1d32*.json", "t1d64*.json", "t1d128*.json"):
    d.update(load(pat))
for k in sorted(d):
    val_row(k, d[k])
print("  committed t1 dim-64 reference: val 6.092 @ 1590 steps (different budget)")

print()
print("=== Exp3: D55 multi-plug-in at t2 (owner_mass 0.55 / 0.50) ===")
for pat in ("swarm4_d55_m*.json",):
    for k, d in sorted(load(pat).items()):
        print(f"-- {k}  owner_mass={d.get('owner_mass')}  pass={d.get('pass')}  "
              f"base_ce={d.get('base_ce')} phase2_s={d.get('phase2_seconds')}")
        for dom, g in (d.get("gates") or {}).items():
            p = (d.get("purity") or {}).get(dom, {})
            c = (d.get("ce") or {}).get(dom, {})
            print(f"   {dom:9s} own={p.get('own_util')} cross={p.get('max_cross_util')} "
                  f"util={p.get('util_per_plug')} ce_with={c.get('ce_with_all_plugins')} "
                  f"ce_masked={c.get('ce_own_masked')} delta={c.get('ce_delta')} "
                  f"-> {'PASS' if g.get('pass') else 'FAIL'} "
                  f"(own_ok={g.get('own_ok')} cross_ok={g.get('cross_ok')} ce_ok={g.get('ce_ok')})")
        fw = (d.get("ce") or {}).get("fineweb", {})
        print(f"   fineweb  ce_with={fw.get('ce_with_all_plugins')} (damage check)")
