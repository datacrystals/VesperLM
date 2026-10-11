#!/usr/bin/env python3
"""Registry/passport scaling sweep — passport_dim x recal-effort vs parity quality.

The cheap measurement the scaling table needs (MODULAR_MOE.md §9): how does the
registry's passport rank (product `passport_dim`) and the recal budget trade
against graft parity on the real OLMoE donor? D0 established one point
(passport_dim = d_model, identity query map, 2 recal rounds -> logit cos
0.99913). This sweep brackets it:

  * grid:  passport_dim in {8,16,32,64,128} x recal rounds in {0,1,2}
           (prototype init; round 0 = prototype-only)
  * anchor: passport_dim = d_model, identity query map, 2 rounds (the D0
           recipe, re-measured on the same batches for a consistent basis)
  * ceilings: passport_dim in {8,...,128} with `--init svd` and 0 recal —
           the rank-d_r algebraic ceiling of the stock gate (truncated SVD),
           i.e. the best ANY rank-d_r passport bank can do on stock logits.

Per cell: final-logit cosine, CE delta, per-layer top-k agreement, per-layer
MoE-output cosine, recal wall-time, parity wall-time, registry .pt size.

Cost discipline: the stock-router passes (expert profile + teacher-forced
(h, stock-logit) calibration pairs) are cached ONCE across all cells — they do
not depend on passport_dim or recal rounds. Per cell only the graft-dependent
work runs (on-trajectory pairs for round 2, parity val passes, CPU recal fits).

Usage (MI50 box, GPU 4 only, fp16):
  export HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4
  export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library
  /data/tliao/vesper_sft/venv/bin/python /data/tliao/graft_sweep/graft_registry_sweep.py \
      --model /data/tliao/OLMoE-1B-7B-0924 \
      --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
      --device cuda --dtype float16 \
      --domains-dir /data/tliao/graft_d0_slice \
      --val-file /data/tliao/graft_d0_slice/val_mixed.txt \
      --seq-len 512 --batch-size 2 --max-tokens 100000 \
      --registry-dir /data/tliao/graft_sweep \
      --out-json /data/tliao/graft_sweep/sweep.json \
      --out-md /data/tliao/graft_sweep/sweep.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graft_d0_parity as g0  # noqa: E402

import torch  # noqa: E402


def parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def gate_spectrum(stock_weights, dims: list[int]) -> dict:
    """Per-layer singular spectrum of the stock gate rows (E, d_model).

    rank(W) <= E; the Frobenius energy captured at rank d_r upper-bounds how
    well ANY rank-d_r query map can reproduce the stock logits — the algebraic
    reason parity saturates where it does.
    """
    per_layer_energy = {d: [] for d in dims}
    ranks, top_frac = [], []
    for w in stock_weights:
        s = torch.linalg.svdvals(w.data.float().cpu())
        e2 = s * s
        tot = float(e2.sum())
        ranks.append(int((s > s.max() * 1e-5).sum()))
        top_frac.append(float(e2[:min(8, s.numel())].sum() / max(tot, 1e-12)))
        for d in dims:
            k = min(d, s.numel())
            per_layer_energy[d].append(float(e2[:k].sum() / max(tot, 1e-12)))
    return {
        "gate_rank_mean": sum(ranks) / len(ranks),
        "top8_energy_frac_mean": sum(top_frac) / len(top_frac),
        "frobenius_energy_at_dim": {
            str(d): sum(v) / len(v) for d, v in per_layer_energy.items()},
    }


def recalibrate_shared(banks, pairs, steps, lr, kl_w):
    """Joint router-only recal with ONE shared query map across layers.

    Fills a shared W_q (d_r, d_model) + each layer's P rows against every
    layer's (h, stock-logit) pairs in a single Adam fit — the "shared spine
    query map" registry variant where the artifact carries one W_q total
    instead of one per layer. Returns the same report shape as
    g0.recalibrate.
    """
    import torch.nn as nn
    import torch.nn.functional as F
    report = {"steps": steps, "lr": lr, "kl_w": kl_w, "per_layer": []}
    if steps <= 0:
        return report
    wq = nn.Parameter(banks[0].W_q.data.float().clone().cpu())
    ps = [nn.Parameter(b.P.data.float().clone().cpu()) for b in banks]
    opt = torch.optim.Adam([wq] + ps, lr=lr)
    cpu_pairs = [(h.float().cpu(), s.float().cpu()) for h, s in pairs]
    first = last = None
    with torch.enable_grad():
        for step in range(steps):
            opt.zero_grad()
            loss = 0.0
            for b, p, (h, s) in zip(banks, ps, cpu_pairs):
                q = h @ wq.T
                sc = (q @ p.T) / (b.d_r ** 0.5) * b.layer_scale
                mse = F.mse_loss(sc, s)
                kl = F.kl_div(F.log_softmax(sc, dim=-1),
                              F.softmax(s, dim=-1), reduction="batchmean")
                loss = loss + mse + kl_w * kl
            loss.backward()
            opt.step()
            if step == 0:
                first = float(loss.detach()) / len(banks)
            last = float(loss.detach()) / len(banks)
    for b, p in zip(banks, ps):
        b.P.data.copy_(p.data.to(b.P.device))
    for b in banks:
        b.W_q.data.copy_(wq.data.to(b.W_q.device))
    report["per_layer"] = [{"layer": 0, "mse_first": first, "mse_last": last}]
    return report


def run_cell(g0m, model, blocks, banks, graft_gates, tf_pairs, domain_batches,
             val_batches, n_domains, device, args, cell) -> dict:
    """Init (already done) + optional recal rounds + parity for one cell."""
    stock_weights = [b.gate.weight for b in blocks]
    out = dict(cell)
    t_cell = time.time()
    out["recal_s"] = 0.0
    out["recal_collect_s"] = 0.0
    rounds = cell["recal_rounds"]
    for rnd in range(rounds):
        pairs = tf_pairs
        if rnd > 0:
            t0 = time.time()
            for i in range(len(blocks)):
                g0m.install_graft(blocks[i], graft_gates[i])
            pairs = g0m.collect_route_pairs(model, blocks, stock_weights,
                                            domain_batches, device,
                                            args.max_cal_pairs)
            for i in range(len(blocks)):
                g0m.uninstall_graft(blocks[i], blocks[i].gate.stock_gate)
            out["recal_collect_s"] += time.time() - t0
        t0 = time.time()
        if cell.get("wq_mode") == "shared":
            rep = recalibrate_shared(banks, pairs, args.recal_steps,
                                     args.recal_lr, args.recal_kl_w)
        else:
            rep = g0m.recalibrate(banks, pairs, args.recal_steps,
                                  args.recal_lr, args.recal_kl_w)
        out["recal_s"] += time.time() - t0
        out.setdefault("recal_mse", []).append(
            [round(pl["mse_first"], 6) for pl in rep["per_layer"]][:1] +
            [round(pl["mse_last"], 6) for pl in rep["per_layer"]][:1])
    if cell["registry"]:
        g0m.save_registry(cell["registry"], banks, cell.get("registry_meta", {}))
        out["registry_bytes"] = os.path.getsize(cell["registry"])
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    parity = g0m.parity_measure(model, blocks, graft_gates, val_batches,
                                device).as_dict()
    out["parity_s"] = round(time.time() - t0, 2)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        out["vram_peak_gib"] = round(
            torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    mean_layer_cos = sum(parity["per_layer_moe_cosine"]) / max(
        1, len(parity["per_layer_moe_cosine"]))
    mean_agree = sum(parity["per_layer_topk_agree"]) / max(
        1, len(parity["per_layer_topk_agree"]))
    out.update({
        "logit_cos_mean": round(parity["logit_cosine_mean"], 5),
        "logit_cos_min": round(parity["logit_cosine_min"], 4),
        "ce_stock": round(parity["ce_stock"], 4),
        "ce_graft": round(parity["ce_graft"], 4),
        "ce_delta": round(parity["ce_delta"], 5),
        "topk_agree_mean": round(mean_agree, 4),
        "moe_cos_mean": round(mean_layer_cos, 5),
        "n_val_tokens": parity["n_tokens"],
        "gate_cos_gt_0.9": parity["logit_cosine_mean"] > g0m.BAR_LOGIT_COSINE,
        "gate_cos_gt_0.99":
            parity["logit_cosine_mean"] > g0m.BAR_LOGIT_COSINE_STRETCH,
        "gate_ce_within_0.05": abs(parity["ce_delta"]) < g0m.BAR_CE_DELTA,
        "gate_layer_cos_gt_0.9": mean_layer_cos > g0m.BAR_LAYER_COSINE,
        "gate_topk_gt_0.8": mean_agree > g0m.BAR_TOPK_AGREE,
    })
    out["cell_wall_s"] = round(time.time() - t_cell, 2)
    return out


def fmt_mb(n: int | None) -> str:
    return "-" if n is None else f"{n / 2**20:.2f}"


def cellf(row: dict, key: str, spec: str) -> str:
    """Format a cell field; '-' when absent (log-salvaged partial cells)."""
    v = row.get(key)
    return "-" if v is None else format(v, spec)


def write_markdown(path: str, result: dict) -> None:
    c = result["conclusion"]
    lines = []
    lines.append("# Registry / passport scaling sweep — OLMoE-1B-7B-0924 graft\n")
    lines.append(
        f"Donor `allenai/OLMoE-1B-7B-0924` @ `{result['donor']['revision']}` "
        f"({result['donor']['dtype']}, {result['device']}), D0 gate-scale slice "
        f"({result['slice']['domain_tokens']:,} domain + "
        f"{result['slice']['val_tokens']:,} val OLMoE tokens, same data as the "
        f"D0 gate run). Recipe: prototype init (`W_q` = top-d_r right singular "
        f"vectors of the stock gate rows when d_r < d_model), recal = "
        f"{result['recipe']['recal_steps']} router-only Adam steps/round "
        f"(MSE+KL; fits P and W_q), round 1 teacher-forced, round 2 "
        f"on-trajectory. Anchor = D0 recipe (d_r = d_model, identity query "
        f"map, 2 rounds).\n")
    lines.append(
        f"Shared overhead (one-time, not per cell): profile "
        f"{result['shared_overhead_s']['profile_s']}s + teacher-forced cal "
        f"pairs {result['shared_overhead_s']['cal_pairs_s']}s.\n")
    lines.append("## Grid: passport_dim x recal rounds (prototype init)\n")
    lines.append(
        "| passport_dim | recal rounds | logit cos | min cos | dCE | top-k "
        "agree | MoE cos | recal s | parity s | cell s | registry MB |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in result["cells"]:
        if row["kind"] not in ("grid", "anchor"):
            continue
        tag = " **(D0 anchor)**" if row["kind"] == "anchor" else ""
        lines.append(
            f"| {row['passport_dim']}{tag} | {row['recal_rounds']} | "
            f"{cellf(row, 'logit_cos_mean', '.5f')} | "
            f"{cellf(row, 'logit_cos_min', '.4f')} | "
            f"{cellf(row, 'ce_delta', '+.5f')} | "
            f"{cellf(row, 'topk_agree_mean', '.4f')} | "
            f"{cellf(row, 'moe_cos_mean', '.5f')} | "
            f"{cellf(row, 'recal_s', '.0f')} | "
            f"{cellf(row, 'parity_s', '.0f')} | "
            f"{cellf(row, 'cell_wall_s', '.0f')} | "
            f"{fmt_mb(row.get('registry_bytes'))} |")
    lines.append("\n## Rank ceilings (SVD init, 0 recal) — best any rank-d_r "
                 "bank can do\n")
    lines.append(
        "| passport_dim | logit cos | dCE | top-k agree | MoE cos | gate "
        "energy @ d_r | registry MB |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in result["cells"]:
        if row["kind"] != "svd_ceiling":
            continue
        efrac = result["spectrum"]["frobenius_energy_at_dim"].get(
            str(row["passport_dim"]))
        lines.append(
            f"| {row['passport_dim']} | "
            f"{cellf(row, 'logit_cos_mean', '.5f')} | "
            f"{cellf(row, 'ce_delta', '+.5f')} | "
            f"{cellf(row, 'topk_agree_mean', '.4f')} | "
            f"{cellf(row, 'moe_cos_mean', '.5f')} | "
            f"{efrac * 100:.1f}% | {fmt_mb(row.get('registry_bytes'))} |")
    lines.append("\n## Conclusion\n")
    lines.append(
        f"- Parity saturates at **passport_dim = {c['saturation_dim']}** "
        f"(smallest dim whose recal'd cosine is within "
        f"{c['saturation_tol']} of the best cell's "
        f"{c['best_cos']:.5f}).")
    for r, info in sorted(c["smallest_dim_by_rounds"].items()):
        lines.append(
            f"- recal {r} round(s): smallest dim with cos >= 0.99 is "
            f"**{info['dim']}** (cos {info['cos']:.5f}, "
            f"{fmt_mb(info['registry_bytes'])} MB registry)"
            if info["dim"] is not None else
            f"- recal {r} round(s): no dim reaches cos >= 0.99 "
            f"(best {info['best_cos']:.5f} @ dim {info['best_dim']})")
    lines.append(
        f"- **Smallest registry keeping cos >= 0.99: "
        + (f"{c['smallest_registry_ge_0.99']['id']} — "
           f"{fmt_mb(c['smallest_registry_ge_0.99']['registry_bytes'])} MB, "
           f"cos {c['smallest_registry_ge_0.99']['cos']:.5f}"
           if c["smallest_registry_ge_0.99"] else "NONE in grid") + "**")
    lines.append(
        f"- Registry cost is dominated by the per-layer query map at high "
        f"d_r (W_q = d_r x d_model per layer): d_r=128 with W_q is LARGER "
        f"than the identity-query d_r=d_model anchor. Passport rows alone "
        f"(the per-expert plug-in cost) = `n_layers*E*d_r*4` bytes.")
    lines.append(
        f"- Gate spectrum: stock gate rank {result['spectrum']['gate_rank_mean']:.1f} "
        f"(E=64 rows); top-8 singular directions carry "
        f"{result['spectrum']['top8_energy_frac_mean'] * 100:.1f}% of "
        f"Frobenius energy.")
    lines.append(
        f"- Total sweep wall {result['wall_s']:.0f}s, peak VRAM "
        f"{result['vram_peak_gib']} GiB.")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--domains-dir", required=True)
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=100_000)
    ap.add_argument("--max-cal-pairs", type=int, default=4096)
    ap.add_argument("--recal-steps", type=int, default=300)
    ap.add_argument("--recal-lr", type=float, default=0.05)
    ap.add_argument("--recal-kl-w", type=float, default=0.1)
    ap.add_argument("--dims", default="8,16,32,64,128")
    ap.add_argument("--rounds", default="0,1,2")
    ap.add_argument("--anchor-rounds", type=int, default=2)
    ap.add_argument("--skip-anchor", action="store_true")
    ap.add_argument("--skip-svd", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells already present with metrics in the "
                         "checkpoint (out_json + '.ckpt.json'); the "
                         "checkpoint is rewritten after EVERY cell so an "
                         "interruption never loses completed work")
    ap.add_argument("--registry-dir", default=None,
                    help="where registry .pt files land (e.g. /data/tliao/graft_sweep)")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    dims = parse_ints(args.dims)
    rounds_list = parse_ints(args.rounds)
    out_dir = args.registry_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "imported", "graft_d0")
    os.makedirs(out_dir, exist_ok=True)
    out_json = args.out_json or os.path.join(out_dir, "sweep.json")
    out_md = args.out_md or os.path.join(out_dir, "sweep.md")
    ckpt_path = out_json + ".ckpt.json"

    prior_cells: dict[str, dict] = {}
    if args.resume and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            prior = json.load(f)
        prior_cells = {c["id"]: c for c in prior.get("cells", [])
                       if "logit_cos_mean" in c}
        print(f"resume: {len(prior_cells)} complete cells in checkpoint: "
              f"{sorted(prior_cells)}")

    def flush_checkpoint(done: bool) -> None:
        """Persist every completed cell immediately (crash-safe)."""
        os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
        with open(ckpt_path, "w") as f:
            json.dump({"cells": cells, "complete": done}, f, indent=1)

    cells: list[dict] = []

    t_all = time.time()
    model, tok = g0.load_donor(args.model, args.device, args.dtype)
    blocks = g0.find_moe_blocks(model)
    if not blocks:
        raise RuntimeError("no MoE blocks found")
    stock_weights = [b.gate.weight for b in blocks]
    e, h = stock_weights[0].shape
    top_k = getattr(blocks[0].gate, "top_k", 2)
    norm_topk = bool(getattr(blocks[0].gate, "norm_topk_prob", False))
    print(f"donor: {len(blocks)} MoE layers, E={e}, d_model={h}, top_k={top_k}")

    domains = g0.load_domain_texts(args.domains_dir)
    domain_names = list(domains.keys())
    domain_batches = []
    for d, name in enumerate(domain_names):
        per = args.max_tokens // max(1, len(domain_names))
        domain_batches.extend(g0.texts_to_batches(
            tok, [domains[name]], args.seq_len, args.batch_size, per, label=d))
    with open(args.val_file) as f:
        val_text = f.read()
    val_batches = g0.texts_to_batches(tok, [val_text], args.seq_len,
                                      args.batch_size, args.max_tokens, label=-1)
    n_dom_tokens = sum(t.numel() for t, _ in domain_batches)
    n_val_tokens = sum(t.numel() for t, _ in val_batches)
    if not domain_batches or not val_batches:
        raise RuntimeError("not enough text for one batch — lower --seq-len/--batch-size")
    print(f"data: {len(domain_names)} domains, {n_dom_tokens} domain tokens, "
          f"{n_val_tokens} val tokens")

    # ---- one-time stock-router passes (cell-independent) ----
    t0 = time.time()
    stats_stock = g0.collect_route_stats(model, blocks, domain_batches,
                                         args.device, len(domain_names))
    profile_s = time.time() - t0
    home, _, mush = stats_stock.home_and_prototypes()
    print(f"profile: {profile_s:.1f}s, mushiness {float(mush.mean()):.3f}")
    t0 = time.time()
    tf_pairs = g0.collect_route_pairs(model, blocks, stock_weights,
                                      domain_batches, args.device,
                                      args.max_cal_pairs)
    cal_pairs_s = time.time() - t0
    print(f"teacher-forced cal pairs: {cal_pairs_s:.1f}s")

    spectrum = gate_spectrum(stock_weights, dims + [h])
    print(f"gate rank {spectrum['gate_rank_mean']:.1f}, energy: "
          f"{spectrum['frobenius_energy_at_dim']}")

    def build_banks(d_r: int):
        learn_wq = d_r != h
        banks = [g0.PassportBank(e, h, d_r=d_r, learn_wq=learn_wq)
                 for _ in blocks]
        return banks, g0.make_graft_gates(blocks, banks, top_k, norm_topk)

    plan = []
    if not args.skip_anchor:
        plan.append(("anchor", h, args.anchor_rounds, "prototype"))
    for d in dims:
        for r in rounds_list:
            plan.append(("grid", d, r, "prototype"))
    if not args.skip_svd:
        for d in dims:
            plan.append(("svd_ceiling", d, 0, "svd"))

    for kind, d_r, rounds, init in plan:
        stem = (f"anchor_dim{d_r}" if kind == "anchor"
                else f"{kind}_dim{d_r}_r{rounds}")
        if stem in prior_cells:
            print(f"\n=== cell {stem}: SKIP (in checkpoint) ===")
            cells.append(prior_cells[stem])
            continue
        banks, graft_gates = build_banks(d_r)
        if init == "svd":
            g0.init_passports_svd_exact(banks, stock_weights)
        else:
            if banks[0].W_q is not None:
                g0.init_wq_svd(banks, stock_weights)
            g0.init_passports_prototype(banks, stats_stock)
            g0.fit_layer_scales(banks, tf_pairs)
        reg_path = os.path.join(out_dir, f"registry_{stem}.pt")
        cell = {
            "kind": kind, "id": stem, "passport_dim": d_r,
            "query_map": "identity" if banks[0].W_q is None else "svd+learned",
            "init": init, "recal_rounds": rounds,
            "registry": reg_path if kind != "svd_ceiling" else reg_path,
            "registry_meta": {
                "sweep": "registry_scaling", "kind": kind,
                "passport_dim": d_r, "init": init, "recal_rounds": rounds,
                "recal_steps": args.recal_steps, "model": args.model,
                "revision": args.revision, "dtype": args.dtype,
            },
        }
        # theoretical size split: query map (per-spine) vs passports (per-expert)
        cell["registry_passports_bytes"] = len(blocks) * e * d_r * 4
        cell["registry_wq_bytes"] = (
            0 if banks[0].W_q is None else len(blocks) * d_r * h * 4)
        print(f"\n=== cell {stem}: dim={d_r} init={init} rounds={rounds} ===")
        rec = run_cell(g0, model, blocks, banks, graft_gates, tf_pairs,
                       domain_batches, val_batches, len(domain_names),
                       args.device, args, cell)
        rec.pop("registry_meta", None)
        rec.pop("registry", None)
        cells.append(rec)
        flush_checkpoint(done=False)
        print(f"  cos={rec['logit_cos_mean']:.5f} dCE={rec['ce_delta']:+.5f} "
              f"agree={rec['topk_agree_mean']:.4f} recal={rec['recal_s']:.0f}s "
              f"registry={fmt_mb(rec.get('registry_bytes'))}MB")

    # ---- conclusion (numbers only) ----
    grid_rows = [c for c in cells if c["kind"] in ("grid", "anchor")]
    best_cos = max(c["logit_cos_mean"] for c in grid_rows)
    tol = 0.001
    sat_dim = min(c["passport_dim"] for c in grid_rows
                  if c["logit_cos_mean"] >= best_cos - tol)
    smallest_by_rounds = {}
    for r in rounds_list:
        rows_r = [c for c in cells if c["kind"] == "grid"
                  and c["recal_rounds"] == r]
        ok = [c for c in rows_r if c["logit_cos_mean"] >= 0.99]
        if ok:
            best = min(ok, key=lambda c: (c.get("registry_bytes") or 0))
            smallest_by_rounds[str(r)] = {
                "dim": best["passport_dim"], "cos": best["logit_cos_mean"],
                "registry_bytes": best.get("registry_bytes"),
                "best_dim": max(rows_r, key=lambda c: c["logit_cos_mean"])["passport_dim"],
                "best_cos": max(c["logit_cos_mean"] for c in rows_r),
            }
        else:
            bb = max(rows_r, key=lambda c: c["logit_cos_mean"])
            smallest_by_rounds[str(r)] = {
                "dim": None, "cos": None, "registry_bytes": None,
                "best_dim": bb["passport_dim"], "best_cos": bb["logit_cos_mean"],
            }
    ge = [c for c in grid_rows if c["logit_cos_mean"] >= 0.99]
    smallest_reg = (min(ge, key=lambda c: (c.get("registry_bytes") or 0))
                    if ge else None)
    conclusion = {
        "saturation_dim": sat_dim,
        "saturation_tol": tol,
        "best_cos": best_cos,
        "smallest_dim_by_rounds": smallest_by_rounds,
        "smallest_registry_ge_0.99": (
            {"id": smallest_reg["id"],
             "registry_bytes": smallest_reg.get("registry_bytes"),
             "cos": smallest_reg["logit_cos_mean"],
             "passport_dim": smallest_reg["passport_dim"],
             "recal_rounds": smallest_reg["recal_rounds"]}
            if smallest_reg else None),
    }

    result = {
        "donor": {"model": args.model, "revision": args.revision,
                  "dtype": args.dtype},
        "device": args.device,
        "slice": {"domain_tokens": n_dom_tokens, "val_tokens": n_val_tokens,
                  "domains_dir": args.domains_dir, "val_file": args.val_file,
                  "n_domains": len(domain_names)},
        "recipe": {"recal_steps": args.recal_steps, "recal_lr": args.recal_lr,
                   "recal_kl_w": args.recal_kl_w, "max_cal_pairs": args.max_cal_pairs,
                   "seq_len": args.seq_len, "batch_size": args.batch_size},
        "shared_overhead_s": {"profile_s": round(profile_s, 2),
                              "cal_pairs_s": round(cal_pairs_s, 2)},
        "spectrum": spectrum,
        "cells": cells,
        "conclusion": conclusion,
        "wall_s": round(time.time() - t_all, 2),
        "vram_peak_gib": (round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                          if torch.cuda.is_available() else None),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)
    flush_checkpoint(done=True)
    write_markdown(out_md, result)
    print(f"\nresults -> {out_json}\ntable -> {out_md}")
    print(f"total wall {result['wall_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
