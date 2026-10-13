#!/usr/bin/env python3
"""bank2n_trainslice — in-sample (train-window) quality read for the 2n bank.

The assemble state matrix measures held windows (honest); the door25/v0
quality gate is DECISIVE in-sample (the wrap memorizes its home fit — that is
the delivered number behind m1's per-slot recovery).  This add-on loads the
same bank and reports CE on each graft's TRAIN windows under masked /
partition-full / solo, so the quality-green bar can be read on the same slice
as the m1 2625% reference.

Numeric-only; one GPU; detached-safe.
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

import graft_d0_parity as d0  # noqa: E402
import graft_d2_crossfamily as d2  # noqa: E402
import door6_expert_mix as d6  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402
import bank2n_partition as b2  # noqa: E402


def teardown(handles):
    """Restore stock expert forwards (bank2n_partition.teardown is nested in
    run_assemble; this is the same logic at module scope)."""
    for blk, sf in handles:
        blk.experts.forward = sf
        if hasattr(blk, "_plugin_merge"):
            delattr(blk, "_plugin_merge")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/tliao/Phi-mini-MoE-instruct")
    ap.add_argument("--registry",
                    default="/data/tliao/graft_d2_out/registry_phimoe_f620b32c.pt")
    ap.add_argument("--domains-dir", default="/data/tliao/graft_d2_slice")
    ap.add_argument("--plugins-dir", default="/data/tliao/bank2n_out")
    ap.add_argument("--out", default="/data/tliao/bank2n_out/bank2n_train.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--eval-batch-size", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=20000)
    args = ap.parse_args()

    t0 = time.time()
    out = {"gate": "bank2n-train-slice", "phase": "start"}
    b2.dump_json(out, args.out)
    model, tok = d2.load_donor(args.model, args.device, args.dtype)
    blocks = d2.find_moe_blocks(model)
    fam = d2.family_for(d2.detect_family(model), blocks)
    banks_reg, _ = d0.load_registry(args.registry)
    domains = d0.load_domain_texts(args.domains_dir)
    seq, ebs = args.seq_len, args.eval_batch_size

    specs = b2.load_specs(args)
    wraps, rows, hosts, homes = {}, {}, {}, {}
    for s in specs:
        n = s["name"]
        path = os.path.join(args.plugins_dir, f"{n}_plugin.pt")
        if not os.path.exists(path):
            continue
        wrap, row, meta, _ = d6.load_wrap_plugin(path)
        if not d6.wrap_is_finite(wrap, row):
            raise RuntimeError(f"non-finite {path}")
        wraps[n], rows[n] = wrap.to(args.device), row.to(args.device)
        hosts[n] = int(meta["host_layer"])
        homes[n] = meta["home"]
    names = list(wraps.keys())
    print(f"bank: {names}", flush=True)
    js = os.path.join(args.plugins_dir, "bank2n_rows.pt")
    if os.path.exists(js):
        saved = torch.load(js, map_location="cpu", weights_only=False)
        for n, r in saved["rows"].items():
            if n in rows:
                rows[n] = r.to(args.device)
        print("rows restored from bank2n_rows.pt", flush=True)

    site_rows, site_entries = {}, {}
    for n in names:
        site_rows.setdefault(hosts[n], []).append(rows[n])
        site_entries.setdefault(hosts[n], []).append({
            "expert": wraps[n], "tag": n,
            "num_stock": banks_reg[hosts[n]].num_experts})
    tags_at = {i: [en["tag"] for en in site_entries[i]] for i in site_entries}

    # TRAIN windows of each domain (the decisive in-sample slice)
    dom_train = {}
    for name in b2.DOMAINS:
        tr, _he, _ = g0.split_home(tok, domains[name], args.max_tokens, seq,
                                   ebs)
        dom_train[name] = tr

    def build_live(live):
        l_rows, l_entries = {}, {}
        for i in site_rows:
            live_here = [n for n in tags_at[i] if n in live]
            if not live_here:
                continue
            l_rows[i] = [rows[n] for n in live_here]
            l_entries[i] = [{"expert": wraps[n], "tag": n,
                             "num_stock": banks_reg[i].num_experts}
                            for n in live_here]
        gates, banks = b2.build_partition_gates(blocks, banks_reg, fam,
                                                l_rows)
        handles = b2.install_partition(blocks, fam, l_entries, mode="off")
        return gates, banks, handles

    def measure(gates, handles, mode, tag):
        rec = {}
        with torch.no_grad():   # no autograd graphs (the OOM-killer source)
            for name in b2.DOMAINS:
                rec[name] = round(b2.ce_batches(model, blocks, dom_train[name],
                                                args.device, gates, handles,
                                                fam, mode, set(names)), 4)
        print(f"  [{tag}] " + " ".join(f"{k} {v}" for k, v in rec.items()),
              flush=True)
        return rec

    d6.freeze_wraps(list(wraps.values()))

    m_rows = {i: [rows[n] for n in tags_at[i]] for i in site_rows}
    m_gates, _ = g0.build_gates(blocks, banks_reg, fam, m_rows)
    m_handles = g0.install_mix(blocks, fam, {
        i: [{"expert": wraps[n], "tag": n,
             "lam": torch.nn.Parameter(torch.zeros((), device=args.device)),
             "num_stock": banks_reg[i].num_experts}
            for n in tags_at[i]] for i in site_rows}, mode="off")
    res = {}
    res["masked_train"] = measure(m_gates, m_handles, "additive", "masked-train")
    teardown(m_handles)
    g_p, _b_p, h_p = build_live(set(names))
    res["partition_train"] = measure(g_p, h_p, "dispatch", "partition-train")
    teardown(h_p)
    for n in names:
        g_s, _b_s, h_s = build_live({n})
        res[f"solo_train_{n}"] = measure(g_s, h_s, "dispatch", f"solo-train {n}")
        teardown(h_s)
    # in-sample quality table (routed vs masked on train windows)
    table = {}
    for n in names:
        d = homes[n]
        masked = res["masked_train"][d]
        inbank = res["partition_train"][d]
        solo = res[f"solo_train_{n}"][d]
        table[n] = {"home": d, "ce_masked": masked, "ce_partition": inbank,
                    "ce_solo": solo,
                    "gain_inbank_nats": round(masked - inbank, 4),
                    "gain_solo_nats": round(masked - solo, 4),
                    "preserved_pct": (round(100 * (masked - inbank)
                                            / (masked - solo), 1)
                                      if masked - solo > 1e-6 else None),
                    "quality_green": bool(inbank < masked)}
    out["train_states"] = res
    out["train_quality_table"] = table
    out["phase"] = "done"
    out["wall_s"] = round(time.time() - t0, 2)
    b2.dump_json(out, args.out)
    print("train-slice quality table:", json.dumps(table, indent=1),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
