#!/usr/bin/env python3
"""R2 — p_home CALIBRATION PROBE (ladder audit queue, lab/LADDER_STATE.md §3 #2).

Question (no re-fit allowed): is the chronic p_home collapse (median 0.1464,
83/83 fit records < 0.5 at n=128; README §3.7 / MODULAR_MOE §11.2 #3) a
CALIBRATION failure (registry-row recal converges to a bad point — fixable by
owner_mass / rescale knobs) or a CAPACITY failure (the 325M-spine router
softmax cannot address 64 stock + 128 plug rows — the spine upgrade the
ladder already prescribes)?

Local laptop rerun of the rung-1 n=128 registry (box data synced to
/home/tliao/rung1 + /home/tliao/vk_final + /home/tliao/core_bank). Row recal
re-runs on cached calibration pairs (identical composition to the fit phase:
16 home + 8 base batches, 1024 pairs, steps 80, lr 0.05, reject_w 10,
mass_w 2 — bank_rung1.fit_one / door6.train_rows_one, read-only).

Grid cells (all cells applied registry-wide, as a recipe change would be):
  owner{m}    m in {0.35, 0.55, 0.70, 0.85, 0.95, 1.15, 1.4} — recal rerun
              (0.35 = the owner target the n=128 fits ACTUALLY ran
              (vb.BAR_OWNER_SPEC in the plugin metas); 0.85 = the §10.7
              recipe text value / BAR_OWNER_MEM)
  owner0.85_s320  step-count control (4x steps, same lr)
  gain{1,2,4} registry rescale: stored fitted row x gain, no recal
  (no bias arm: the registry PassportBank has scores = q.p/sqrt(d_r)*scale —
   no per-row or global bias term exists to sweep)

Phases (incremental JSON / skip-done):
  collect   hidden-state pairs + eval-window taps, cached to disk
  grid      per-row recal sweep -> p_home per cell (fit-time metric, §11)
  assembled assembled-bank p_home + membership on eval windows per cell
  ce        masked vs plugged CE per home + base per cell (home_quality /
            base_tax, same windows as growth_128; masked computed once)
  report    verdict numbers into lab/results/phome_calibration.json
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
import door6_expert_mix as d6  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402
import v05_core_prep as prep  # noqa: E402
import vk_final_build as vb  # noqa: E402
import bank_rung1 as br1  # noqa: E402

DEFAULT_MODEL = "/home/tliao/vk_final/chat_model"
DEFAULT_REGISTRY = "/home/tliao/vk_final/registry_vk.pt"
DEFAULT_ROOT = "/home/tliao/rung1"
OUT_JSON = os.path.join(REPO_ROOT, "lab", "results",
                        "phome_calibration.json")
CACHE = os.path.join(DEFAULT_ROOT, "phome_cache")

OWNERS = [0.35, 0.55, 0.70, 0.85, 0.95, 1.15, 1.4]
GAINS = [1.0, 2.0, 4.0]
LONG_STEPS_OWNER = 0.85
LONG_STEPS = 320

HOST_LAYERS = list(range(10))  # the n=128 set sits on host layers 0..9


def dump(out: dict, path: str = OUT_JSON) -> None:
    g0.dump_json(out, path)


def cell_names() -> list[str]:
    return ([f"owner{m}" for m in OWNERS]
            + [f"owner{LONG_STEPS_OWNER}_s{LONG_STEPS}"]
            + [f"gain{g:g}" for g in GAINS])


def site_bank_stock(banks_reg, site: int) -> "d0.PassportBank":
    """CPU copy of the stock site bank + one empty plug row (fit layout)."""
    breg = banks_reg[site]
    bank = d0.PassportBank(breg.num_experts + 1, breg.d_model, d_r=breg.d_r,
                           learn_wq=breg.W_q is not None)
    with torch.no_grad():
        bank.P.data[:breg.num_experts] = breg.P.data
        if bank.W_q is not None:
            bank.W_q.data.copy_(breg.W_q.data)
    bank.layer_scale = breg.layer_scale
    return bank


def fit_row_init(bank: "d0.PassportBank", home_h: torch.Tensor) -> None:
    """bank_rung1.fit_one row init: query-space proto at stock-row norm."""
    row_idx = bank.num_experts - 1
    with torch.no_grad():
        proto = bank.queries(home_h).mean(dim=0)
        stock_norm = float(bank.P.data[:row_idx].float().norm(dim=1).mean())
        pnorm = float(proto.float().norm())
        if pnorm > 0 and stock_norm > 0:
            proto = proto * (stock_norm / pnorm)
        bank.P.data[row_idx] = proto


def scores_phrow(bank: "d0.PassportBank", h: torch.Tensor,
                 row_idx: int) -> tuple[float, float]:
    """Chosen-row softmax prob on a h set (lab 1 rows and lab 0 rows)."""
    logits = bank.scores(h)
    p = torch.softmax(logits, dim=-1)[:, row_idx]
    return float(p.mean())


# ---------------------------------------------------------------------------
# collect — pair + eval-window hidden-state caches
# ---------------------------------------------------------------------------

def phase_collect(args, out, man, model, tok, blocks, banks_reg):
    seq = args.seq_len
    os.makedirs(os.path.join(CACHE), exist_ok=True)

    # pair sets per home (16 home + 8 base, like fit_one)
    base_pair = br1.base_batches(tok, seq, 1, args.max_base_tokens)[
        :args.pair_base]
    done = True
    for home in br1.HOMES:
        p = os.path.join(CACHE, f"pairs_{home}.pt")
        if args.skip_done and os.path.exists(p):
            continue
        done = False
        home_pair = br1.feed_batches(tok, home, seq, 1,
                                     args.max_tokens)[:args.pair_batches]
        pairs = g0.collect_pairs_fam(
            model, blocks, vk.FAM, home_pair + base_pair, args.device,
            args.max_cal_pairs, lambda _i, b: 1 if b[1] == 1 else 0)
        pack = {i: (h.half(), y) for i, (h, y) in enumerate(pairs)
                if h.shape[0] > 0}
        torch.save(pack, p)
        print(f"[collect] pairs {home} cached -> {p}", flush=True)

    # eval windows (growth_128 record: first 6 batches per home + base)
    base_eval = br1.base_batches(tok, seq, 1, args.max_base_tokens)[
        :args.eval_batches]
    for name, batches in ([(h, br1.feed_batches(tok, h, seq, 1,
                                                args.max_tokens)
                            [:args.eval_batches]) for h in br1.HOMES]
                          + [("base", base_eval)]):
        p = os.path.join(CACHE, f"eval_{name}.pt")
        if args.skip_done and os.path.exists(p):
            continue
        done = False
        pairs = g0.collect_pairs_fam(model, blocks, vk.FAM, batches,
                                     args.device, 100_000_000,
                                     lambda _i, _b: 0)
        pack = {i: h.half() for i, (h, _y) in enumerate(pairs)
                if h.shape[0] > 0}
        torch.save(pack, p)
        print(f"[collect] eval {name} cached -> {p}", flush=True)
    out.setdefault("collect", {})["done"] = done or True
    dump(out)


def load_cache(name: str) -> dict:
    return torch.load(os.path.join(CACHE, f"{name}.pt"),
                      map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# grid — per-row recal sweep (CPU, no re-fit of wraps)
# ---------------------------------------------------------------------------

def row_metrics(bank, pairs, row_idx):
    h, lab = pairs
    code = lab > 0.5
    p = torch.softmax(bank.scores(h), dim=-1)[:, row_idx]
    return {"p_home": float(p[code].mean()),
            "p_base": float(p[~code].mean())}


def phase_grid(args, out, man, model, tok, blocks, banks_reg):
    cells = out.setdefault("cells", {})
    for cn in cell_names():
        cells.setdefault(cn, {})
    t0 = time.time()
    for c in man["cores"][: args.n]:
        cid = c["id"]
        home = c["home"]
        site = int(c["host_layer"])
        if all(cells[cn].get(cid) for cn in cell_names()) and args.skip_done:
            continue
        pp = br1.plugin_path(args.root, cid)
        d = torch.load(pp, map_location="cpu", weights_only=False)
        row_fitted = d["row"].float()
        meta = d.get("meta", {})
        pairs_pack = load_cache(f"pairs_{home}")
        if site not in pairs_pack:
            continue
        h, lab = pairs_pack[site]
        h = h.float()
        home_h = h[lab > 0.5]

        for m in OWNERS:
            cn = f"owner{m}"
            if cells[cn].get(cid) and args.skip_done:
                continue
            bank = site_bank_stock(banks_reg, site)
            fit_row_init(bank, home_h)
            row_idx = bank.num_experts - 1
            rep = d6.train_rows_one(
                bank, (h, lab), row_idx, args.row_steps, args.row_lr,
                "cpu", reject_w=args.reject_w, mass_w=args.mass_w,
                owner_target=m)
            fin = row_metrics(bank, (h, lab), row_idx)
            cells[cn][cid] = {
                "first": rep["first"], "last": rep["last"],
                "final": fin, "row": bank.P.data[row_idx].float().tolist(),
                "home": home, "site": site}
            dump(out)

        cn = f"owner{LONG_STEPS_OWNER}_s{LONG_STEPS}"
        if not (cells[cn].get(cid) and args.skip_done):
            bank = site_bank_stock(banks_reg, site)
            fit_row_init(bank, home_h)
            row_idx = bank.num_experts - 1
            rep = d6.train_rows_one(
                bank, (h, lab), row_idx, LONG_STEPS, args.row_lr, "cpu",
                reject_w=args.reject_w, mass_w=args.mass_w,
                owner_target=LONG_STEPS_OWNER)
            fin = row_metrics(bank, (h, lab), row_idx)
            cells[cn][cid] = {
                "first": rep["first"], "last": rep["last"],
                "final": fin, "row": bank.P.data[row_idx].float().tolist(),
                "home": home, "site": site}
            dump(out)

        for g in GAINS:
            cn = f"gain{g:g}"
            if cells[cn].get(cid) and args.skip_done:
                continue
            bank = site_bank_stock(banks_reg, site)
            row_idx = bank.num_experts - 1
            with torch.no_grad():
                bank.P.data[row_idx] = row_fitted * float(g)
            fin = row_metrics(bank, (h, lab), row_idx)
            cells[cn][cid] = {
                "final": fin, "gain": g,
                "row": bank.P.data[row_idx].float().tolist(),
                "home": home, "site": site,
                "owner_target_recorded": meta.get("owner_target")}
            dump(out)
        print(f"[grid] {cid} done ({time.time()-t0:.0f}s)", flush=True)
    out.setdefault("grid", {})["done_at"] = time.time()
    dump(out)


def summarize_cell(recs: dict) -> dict:
    ph = sorted(v["final"]["p_home"] for v in recs.values())
    pb = sorted(v["final"]["p_base"] for v in recs.values())
    if not ph:
        return {"n": 0}
    by_home = {}
    for v in recs.values():
        by_home.setdefault(v["home"], []).append(v["final"]["p_home"])
    return {
        "n": len(ph),
        "min": round(ph[0], 4),
        "median": round(ph[len(ph) // 2], 4),
        "max": round(ph[-1], 4),
        "frac_ge_05": round(sum(1 for v in ph if v >= 0.5) / len(ph), 4),
        "p_base_median": round(pb[len(pb) // 2], 6),
        "p_base_max": round(pb[-1], 6),
        "per_home_median": {k: round(float(np.median(v)), 4)
                            for k, v in sorted(by_home.items())},
        "pass_ph50": bool(ph[len(ph) // 2] >= 0.5),
    }


# ---------------------------------------------------------------------------
# assembled — assembled-bank routing stats on eval windows (no cores needed)
# ---------------------------------------------------------------------------

def ordered_rows(man, row_map: dict, n: int) -> list[tuple]:
    """(cid, site, row_tensor) for the first n cores, in manifest order — the
    ONE order used by gates / banks / MixMerge entries (row position j must
    equal plug index j or membership lands on the wrong expert)."""
    out = []
    for c in man["cores"][:n]:
        cid = c["id"]
        if cid in row_map:
            out.append((cid, int(c["host_layer"]),
                        torch.tensor(row_map[cid]["row"],
                                     dtype=torch.float32)))
    return out


def assembled_banks(banks_reg, rows_ordered: list) -> dict:
    """host_layer -> PassportBank(stock + plug rows, manifest order)."""
    site_rows: dict[int, list] = {}
    site_cids: dict[int, list] = {}
    for cid, site, row in rows_ordered:
        site_rows.setdefault(site, []).append(row)
        site_cids.setdefault(site, []).append(cid)
    banks = {}
    for i, breg in enumerate(banks_reg):
        rows = site_rows.get(i, [])
        if not rows:
            continue
        bank = d0.PassportBank(breg.num_experts + len(rows), breg.d_model,
                               d_r=breg.d_r,
                               learn_wq=breg.W_q is not None)
        with torch.no_grad():
            bank.P.data[: breg.num_experts] = breg.P.data
            if bank.W_q is not None:
                bank.W_q.data.copy_(breg.W_q.data)
            for j, r in enumerate(rows):
                bank.P.data[breg.num_experts + j] = r.to(bank.P.dtype)
        bank.layer_scale = breg.layer_scale
        banks[i] = (bank, site_cids[i])
    return banks


def phase_assembled(args, out, man, model, tok, blocks, banks_reg):
    asm = out.setdefault("assembled", {})
    n_stock = None
    for cn in cell_names():
        if asm.get(cn, {}).get("done") and args.skip_done:
            continue
        row_map = out["cells"][cn]
        rows_ordered = ordered_rows(man, row_map, args.n)
        banks = assembled_banks(banks_reg, rows_ordered)
        # eval h per layer per stream, once
        streams = {name: load_cache(f"eval_{name}")
                   for name in br1.HOMES + ["base"]}
        per_hub = {}
        all_ph = []
        for cid, rec in row_map.items():
            site = rec["site"]
            home = rec["home"]
            bank, cids = banks[site]
            n_stock = bank.num_experts - len(cids)
            row_idx = n_stock + cids.index(cid)
            h_home = streams[home][site].float()
            h_base = streams["base"][site].float()
            sc_h = bank.scores(h_home)
            sc_b = bank.scores(h_base)
            kth_h = torch.topk(sc_h[:, :n_stock], 2, dim=-1).values[:, -1]
            p_h = torch.softmax(sc_h, dim=-1)[:, row_idx]
            p_b = torch.softmax(sc_b, dim=-1)[:, row_idx]
            mem = float((sc_h[:, row_idx] > kth_h).float().mean())
            # ceiling diagnostic: if this row's logit exactly tied the TOP
            # stock logit on every home token, its softmax share would be
            # 1/(1 + Z_rest) — the reachable envelope of the p_home metric on
            # this bank. If the ceiling itself is ~0.15, the 0.5 bar is
            # unreachable-by-construction here regardless of calibration.
            z_stock = sc_h[:, :n_stock]
            z_top = z_stock.max(dim=-1).values
            rest = (z_stock.exp().sum(dim=-1)
                    + sc_h[:, n_stock:].exp().sum(dim=-1) - z_top.exp())
            ceil = (1.0 / (1.0 + rest / z_top.exp()))
            per_hub[cid] = {"p_home": float(p_h.mean()),
                            "p_base": float(p_b.mean()),
                            "membership_frac": mem, "home": home,
                            "p_ceiling_top_stock": float(ceil.mean())}
            all_ph.append(float(p_h.mean()))
        ph = sorted(all_ph)
        recs = per_hub
        agg = {
            "n": len(ph),
            "median": round(ph[len(ph) // 2], 4) if ph else None,
            "min": round(ph[0], 4) if ph else None,
            "max": round(ph[-1], 4) if ph else None,
            "frac_ge_05": round(sum(1 for v in ph if v >= 0.5)
                                / len(ph), 4) if ph else None,
            "pass_ph50": bool(ph[len(ph) // 2] >= 0.5) if ph else None,
            "membership_median": round(float(np.median(
                [v["membership_frac"] for v in recs.values()])), 4),
            "p_base_max": round(max(v["p_base"] for v in recs.values()), 6),
            "ceiling_median": round(float(np.median(
                [v["p_ceiling_top_stock"] for v in recs.values()])), 4),
        }
        asm[cn] = {"rows": {k: round(v["p_home"], 4)
                            for k, v in recs.items()},
                   "detail": {k: {kk: round(vv, 6) if isinstance(vv, float)
                                  else vv for kk, vv in v.items()}
                              for k, v in recs.items()},
                   "aggregate": agg, "done": True, "n_stock": n_stock}
        dump(out)
        print(f"[assembled] {cn}: median p_home {agg['median']} "
              f"frac>=0.5 {agg['frac_ge_05']} memb {agg['membership_median']}",
              flush=True)


# ---------------------------------------------------------------------------
# ce — masked vs plugged CE per cell (§11 home_quality + base_tax windows)
# ---------------------------------------------------------------------------

def install_staging(handles, site_entries, device):
    """CPU-parked wrap cores: move a host block's plugs to GPU around the
    block forward (8GB VRAM cannot hold 128 fp16 cores; growth.box held all
    resident on a 32GB MI50)."""
    removers = []
    for blk, _sf in handles:
        pm = getattr(blk, "_plugin_merge", None)
        if pm is None:
            continue

        def pre(mod, inputs, pm=pm, device=device):
            for e in pm.experts:
                e.to(device)

        def post(mod, inputs, _out, pm=pm):
            for e in pm.experts:
                e.to("cpu")
            torch.cuda.empty_cache()

        h1 = blk.register_forward_pre_hook(pre)
        h2 = blk.register_forward_hook(post)
        removers.extend([h1, h2])
    return removers


CE_CELLS = ["owner0.35", "owner0.55", "owner0.7", "owner0.85", "owner0.85_s320", "owner1.15",
            "owner1.4", "gain1", "gain2", "gain4"]

SAFE_BT = {128, 192, 256, 512, 768, 1024, 1536}


def _group_eval(batches: list, bs: int, seq: int):
    """Merge bs batches into one padded-forward CE measure (staging amortizes
    6x on an 8GB card; CE itself is token-exact — same windows)."""
    toks = [b[0] for b in batches]
    out = []
    for i in range(0, len(toks), bs):
        chunk = toks[i:i + bs]
        if len(chunk) == bs:
            out.append((torch.cat(chunk, dim=0), 0))
        else:  # keep tail batches at their own size
            out.extend((t, 0) for t in chunk)
    return out


def phase_ce(args, out, man, model, tok, blocks, banks_reg):
    ce = out.setdefault("ce", {})
    seq = args.seq_len
    home_eval = {}
    for home in br1.HOMES:
        b = br1.feed_batches(tok, home, seq, 1, args.max_tokens)[
            :args.eval_batches]
        if b:
            home_eval[home] = _group_eval(b, args.ce_bs, seq)
    base_eval = _group_eval(
        br1.base_batches(tok, seq, 1, args.max_base_tokens)[
            :args.eval_batches], args.ce_bs, seq)

    # load wraps ONCE (cores parked on CPU), rows swapped per cell
    plugs = []
    for c in man["cores"][: args.n]:
        pp = br1.plugin_path(args.root, c["id"])
        if not os.path.exists(pp):
            print(f"[ce] {c['id']} plugin missing — skip", flush=True)
            continue
        d = torch.load(pp, map_location="cpu", weights_only=False)
        spec = d["wrap_spec"]
        home_prefix = d["meta"]["core_prefix"].replace("/data/tliao",
                                                       "/home/tliao", 1)
        st, prov = prep.load_core_tensors(home_prefix)
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
        wrap.load_state_dict(d["wrap_adapters"], strict=False)
        wrap.core_w_gu.data = wrap.core_w_gu.data.half()
        wrap.core_w_down.data = wrap.core_w_down.data.half()
        wrap.to("cpu")
        meta = d["meta"]
        meta = {**meta, "id": c["id"]}
        plugs.append((wrap, meta))
    print(f"[ce] {len(plugs)} wraps loaded (cores CPU-parked)", flush=True)

    site_entries: dict[int, list] = {}
    for wrap, meta in plugs:
        host = int(meta["host_layer"])
        site_entries.setdefault(host, []).append({
            "expert": wrap,
            "lam": torch.tensor(args.default_lam),
            "tag": meta["id"], "num_stock": banks_reg[host].num_experts})
    handles = g0.install_mix(blocks, vk.FAM, site_entries, mode="off")

    # masked pass (cell-invariant: VkGraftGate stock select ignores new rows)
    if not ce.get("masked", {}).get("done"):
        mk = os.path.join(CACHE, "ce_masked.pt")
        if os.path.exists(mk) and args.skip_done:
            ce["masked"] = torch.load(mk, weights_only=False)
            ce["masked"]["done"] = True
        else:
            gates0 = vb._build_vk_gates(blocks, banks_reg, {})[0]
            mvals = {}
            for home, b in home_eval.items():
                v = g0.run_in_state(
                    model, blocks, vk.FAM, gates0, handles, "off",
                    lambda: d1.ce_in_state(model, blocks, b, args.device,
                                           None, None, "off"))
                mvals[home] = round(v, 4)
                print(f"[ce] masked {home} {v:.4f}", flush=True)
            v_base = g0.run_in_state(
                model, blocks, vk.FAM, gates0, handles, "off",
                lambda: d1.ce_in_state(model, blocks, base_eval, args.device,
                                       None, None, "off"))
            ce["masked"] = {"homes": mvals, "base": round(v_base, 4),
                            "done": True}
            torch.save({k: v for k, v in ce["masked"].items() if k != "done"},
                       mk)
        dump(out)

    staging = install_staging(handles, site_entries, args.device)
    for cn in CE_CELLS:
        if ce.get(cn, {}).get("done") and args.skip_done:
            continue
        row_map = out["cells"].get(cn, {})
        if len(row_map) < args.n:
            print(f"[ce] {cn}: grid incomplete ({len(row_map)}) — skip",
                  flush=True)
            continue
        site_rows = {}
        for cid, site, row in ordered_rows(man, row_map, args.n):
            site_rows.setdefault(site, []).append(row)
        gates, _site_banks = vb._build_vk_gates(blocks, banks_reg, site_rows)
        vals = {}
        for home, b in home_eval.items():
            v = g0.run_in_state(
                model, blocks, vk.FAM, gates, handles, "both",
                lambda: d1.ce_in_state(model, blocks, b, args.device,
                                       None, None, "off"))
            vals[home] = round(v, 4)
            print(f"[ce] {cn} plugged {home} {v:.4f}", flush=True)
            dump(out)
        v_base = g0.run_in_state(
            model, blocks, vk.FAM, gates, handles, "both",
            lambda: d1.ce_in_state(model, blocks, base_eval, args.device,
                                   None, None, "off"))
        vals["base"] = round(v_base, 4)
        mvals = ce["masked"]["homes"]
        hq = {h: round(vals[h] - mvals[h], 4) for h in home_eval}
        ce[cn] = {
            "plugged": vals, "home_delta": hq,
            "home_quality_worst": round(max(hq.values()), 4),
            "base_tax": round(vals["base"] - ce["masked"]["base"], 4),
            "done": True}
        dump(out)
        print(f"[ce] {cn}: worst home Δ{ce[cn]['home_quality_worst']} "
              f"base Δ{ce[cn]['base_tax']}", flush=True)
    for r in staging:
        r.remove()


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True,
                    choices=["collect", "grid", "assembled", "ce", "report"])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=300000)
    ap.add_argument("--max-base-tokens", type=int, default=40000)
    ap.add_argument("--pair-batches", type=int, default=16)
    ap.add_argument("--pair-base", type=int, default=8)
    ap.add_argument("--max-cal-pairs", type=int, default=1024)
    ap.add_argument("--row-steps", type=int, default=80)
    ap.add_argument("--row-lr", type=float, default=0.05)
    ap.add_argument("--reject-w", type=float, default=10.0)
    ap.add_argument("--mass-w", type=float, default=2.0)
    ap.add_argument("--eval-batches", type=int, default=6)
    ap.add_argument("--default-lam", type=float, default=0.5)
    ap.add_argument("--ce-bs", type=int, default=6,
                    help="batch grouping inside CE cells (token-exact; "
                    "amortizes the 8GB core staging)")
    args = ap.parse_args()

    global CACHE
    CACHE = os.path.join(args.root, "phome_cache")

    out = {}
    if os.path.exists(OUT_JSON):
        out = json.load(open(OUT_JSON))
    out.setdefault("experiment", "R2-phome-calibration")
    out.setdefault("format", "vesper-phome-calibration-v1")
    out["meta"] = {"model": args.model, "registry": args.registry,
                   "root": args.root, "device": args.device,
                   "torch": torch.__version__,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "n": args.n}
    man = json.load(open(os.path.join(args.root, "manifest.json")))

    # registry bias/scale audit (reported once: is there a bias to sweep?)
    banks_reg_probe, reg_meta = d0.load_registry(args.registry)
    out["registry_audit"] = {
        "n_layers": len(banks_reg_probe),
        "num_experts": banks_reg_probe[0].num_experts,
        "d_r": banks_reg_probe[0].d_r,
        "d_model": banks_reg_probe[0].d_model,
        "has_wq": banks_reg_probe[0].W_q is not None,
        "layer_scale": float(banks_reg_probe[0].layer_scale),
        "has_bias": hasattr(banks_reg_probe[0], "bias"),
        "reg_meta_keys": list(reg_meta) if isinstance(reg_meta, dict)
        else None}
    dump(out)

    need_model = args.phase in ("collect", "ce")
    model = tok = blocks = None
    if need_model:
        model, tok = vk.load_vk(args.model, args.device,
                                getattr(torch, args.dtype))
        blocks = vk.install_vk_host(model)
        banks_reg, _ = d0.load_registry(args.registry)
    else:
        banks_reg = banks_reg_probe

    if args.phase == "collect":
        phase_collect(args, out, man, model, tok, blocks, banks_reg)
    elif args.phase == "grid":
        phase_grid(args, out, man, model, tok, blocks, banks_reg)
    elif args.phase == "assembled":
        phase_assembled(args, out, man, model, tok, blocks, banks_reg)
    elif args.phase == "ce":
        phase_ce(args, out, man, model, tok, blocks, banks_reg)
    elif args.phase == "report":
        rep = {"grid": {cn: summarize_cell(out["cells"].get(cn, {}))
                        for cn in cell_names()},
               "assembled": {cn: out["assembled"].get(cn, {}).get(
                   "aggregate") for cn in cell_names()},
               "ce": {cn: out["ce"].get(cn) for cn in
                      cell_names() + ["masked"]}}
        out["report"] = rep
        dump(out)
        for cn, s in rep["grid"].items():
            print(f"{cn:20s} grid-median {s.get('median')} "
                  f"frac>=0.5 {s.get('frac_ge_05')} "
                  f"asm {rep['assembled'].get(cn, {})}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
