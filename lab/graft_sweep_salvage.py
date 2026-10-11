#!/usr/bin/env python3
"""Parse graft_registry_sweep log lines into a resumable checkpoint JSON.

Salvage path after a run interruption: the sweep prints one result line per
completed cell; this turns those lines into the same cell records the sweep
writes, so --resume can skip them. Fields not printed (moe_cos_mean,
logit_cos_min, parity_s, ...) are absent — writers must tolerate that.
"""
import json
import os
import re
import sys

LOG = sys.argv[1]
OUT = sys.argv[2]
REG_DIR = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(OUT)
N_LAYERS, E, D_MODEL = 16, 64, 2048

header = re.compile(
    r"^=== cell (?P<id>\S+): dim=(?P<dim>\d+) init=(?P<init>\S+) rounds=(?P<rounds>\d+) ===")
result = re.compile(
    r"^\s+cos=(?P<cos>[-\d.]+) dCE=(?P<dce>[-+\d.]+) agree=(?P<agree>[-\d.]+) "
    r"recal=(?P<recal>\d+)s registry=(?P<regmb>[-\d.]+)MB")

cells, cur = [], None
for line in open(LOG):
    m = header.match(line)
    if m:
        cur = {"id": m.group("id"),
               "kind": ("anchor" if m.group("id").startswith("anchor")
                        else "svd_ceiling" if m.group("id").startswith("svd")
                        else "grid"),
               "passport_dim": int(m.group("dim")),
               "init": m.group("init"),
               "recal_rounds": int(m.group("rounds")),
               "query_map": ("identity" if int(m.group("dim")) == D_MODEL
                             else "svd+learned"),
               "metrics_source": "salvaged_log_line"}
        continue
    m = result.match(line)
    if m and cur is not None:
        cur.update({
            "logit_cos_mean": float(m.group("cos")),
            "ce_delta": float(m.group("dce")),
            "topk_agree_mean": float(m.group("agree")),
            "recal_s": float(m.group("recal")),
            "registry_bytes": int(round(float(m.group("regmb")) * 2**20)),
        })
        d_r = cur["passport_dim"]
        cur["registry_passports_bytes"] = N_LAYERS * E * d_r * 4
        cur["registry_wq_bytes"] = (0 if cur["query_map"] == "identity"
                                    else N_LAYERS * d_r * D_MODEL * 4)
        rp = os.path.join(REG_DIR, f"registry_{cur['id']}.pt")
        if os.path.exists(rp):
            cur["registry_bytes"] = os.path.getsize(rp)
        cells.append(cur)
        cur = None

json.dump({"cells": cells, "complete": False,
           "note": "checkpoint seeded from sweep_run.log salvage"},
          open(OUT, "w"), indent=1)
print(f"salvaged {len(cells)} cells -> {OUT}")
for c in cells:
    print(f"  {c['id']}: cos={c['logit_cos_mean']} dCE={c['ce_delta']} "
          f"agree={c['topk_agree_mean']} recal={c['recal_s']}s")
