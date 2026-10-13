#!/usr/bin/env python3
"""v0.5.1 delivery fix — per-home balanced λ refit + base-tax guard.

Fixes v0.5's λ-starvation: glm_cybersec / mimo_writing fitted to λ*=0 on the
pooled union stream while code/math/memories took all delivery mass.

Levers (cheapest first, this script covers 1+2+4-measure):
  1+2. PER-HOME INDEPENDENT λ fits — round-robin grid over λ per plug, each
       plug's λ fitted ONLY against its own home's windows (no competition).
       Winners keep their v0.5 λ unless their own home prefers a new one
       AND the rep slice does not regress.
  base-tax guard: total base tax bar 0.05; shrink the plug with the least
       home gain per unit tax until inside the budget.
  cross-fire matrix measured per plug × home (membership share); if a row
       floods off-home domains, --xfire-recal NAME re-runs its mex row recal
       with an off-home reject boost (bank2n condition #3).

Rows and wraps stay frozen — deeper host sites (lever 3) is a separate fit.

Output: v05_lams_refit.json (incremental) + joint_state.pt rewritten in
place (previous saved to joint_state_pre_v051.pt) so chat_v05 picks the
fixed delivery up by default.

Usage (box):
  python3 lab/v05_lam_refit.py --device cuda --gpu auto
  python3 lab/v05_lam_refit.py --xfire-recal glm_cybersec,mimo_writing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import graft_d0_parity as d0  # noqa: E402
import graft_d1_plugin as d1  # noqa: E402
import graft_d2_crossfamily as d2  # noqa: E402
import door6_expert_mix as d6  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402
import v05_build as b5  # noqa: E402
import chat_v05 as c5  # noqa: E402

BASE_TAX_BAR = 0.05
GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
ORDER = ["code", "math", "mem_pet", "mem_edit", "mem_words",
         "k3_hack", "glm_cybersec", "mimo_writing"]
HOME_OF = {"code": "code", "math": "math", "mem_pet": "pet",
           "mem_edit": "editor", "mem_words": "words",
           "k3_hack": "hacking", "glm_cybersec": "cybersec",
           "mimo_writing": "writing"}


def dump(out: dict, path: str) -> None:
    g0.dump_json(out, path)


def pick(stream, a, b):
    """slice with wrap-around (episode homes only have ~8 windows)."""
    return [stream[i % len(stream)] for i in range(a, b)]


def home_stream(tok, home, args, seq):
    if home in ("code", "math"):
        domains = d0.load_domain_texts(args.domains_dir)
        tr, _, _ = g0.split_home(tok, domains[home], args.max_tokens, seq, 4)
        return [(t, b5.HOME_LABEL[home]) for t, _ in tr]
    if home in ("pet", "editor", "words"):
        eps = g0.build_episodes()
        bl = g0.episode_batches(tok, [eps[home]], seq, 4)
        return [(t, b5.HOME_LABEL[home]) for t, _ in bl]
    ids = b5.load_specialty_ids(
        tok, os.path.join(args.specialties_dir, home, "corpus.txt"),
        args.max_tokens)
    return [(t, b5.HOME_LABEL[home]) for t, _ in g0.to_batches(ids, 1, seq, 4)]


@torch.no_grad()
def ce_on(model, blocks, fam, gates, handles, batches, device):
    return g0.run_in_state(
        model, blocks, fam, gates, handles, "both",
        lambda: d1.ce_in_state(model, blocks, batches, device, None, None,
                               "off"))


@torch.no_grad()
def memb_matrix(model, blocks, fam, gates, handles, home_batches, device,
                site_entries):
    """plug × home membership share (cross-fire read)."""
    tags = {hi: [en["tag"] for en in ents]
            for hi, ents in site_entries.items()}
    share = {}
    for home, batches in home_batches.items():
        acc = {}
        counts = 0
        for tokens, _ in batches:
            tokens = tokens.to(device)
            g0.run_in_state(model, blocks, fam, gates, handles, "both",
                            lambda t=tokens: d0.model_logits(model, t))
            counts += tokens.numel()
            for hi, g in enumerate(gates):
                if hi not in tags:
                    continue
                m = getattr(g, "last_membership", None)
                if m is None:
                    continue
                for j in range(m.shape[1]):
                    key = tags[hi][j] if j < len(tags[hi]) else f"L{hi}_{j}"
                    acc[key] = acc.get(key, 0.0) + float(m[:, j].float().sum())
        share[home] = {k: round(v / max(1, counts), 4)
                       for k, v in sorted(acc.items())}
    return share


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/data/tliao/Phi-mini-MoE-instruct")
    ap.add_argument("--registry",
                    default="/data/tliao/graft_d2_out/registry_phimoe_f620b32c.pt")
    ap.add_argument("--domains-dir", default="/data/tliao/graft_d2_slice")
    ap.add_argument("--specialties-dir",
                    default="/data/tliao/vesper_sft/VesperLM/Dataset/data/specialties")
    ap.add_argument("--plugins-dir", default="/data/tliao/v05_out")
    ap.add_argument("--out", default="/data/tliao/v05_out/v05_lams_refit.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-tokens", type=int, default=300000)
    ap.add_argument("--max-base-tokens", type=int, default=40000)
    ap.add_argument("--grid", default=",".join(str(x) for x in GRID))
    ap.add_argument("--xfire-recal", default="",
                    help="comma plug names: re-recal their rows with an "
                         "off-home reject boost before the λ fit")
    ap.add_argument("--xfire-reject-w", type=float, default=40.0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    grid = [float(x) for x in args.grid.split(",")]
    torch.manual_seed(args.seed)

    dev_i = c5.pick_gpu() if args.gpu == "auto" else int(args.gpu)
    device = f"cuda:{dev_i}"
    torch.cuda.set_device(dev_i)

    cargs = SimpleNamespace(**vars(args), default_lam=0.5, lam_map={},
                            bare=False)
    (model, tok, blocks, fam, gates, handles, info, missing,
     site_entries, site_banks) = c5.build_state(cargs, device)
    names = [i["name"] for i in info]
    lam = {i["name"]: float(i["lam"]) for i in info}
    lam0 = dict(lam)
    print("[refit] live plugs:", names, flush=True)
    print("[refit] start lams:", lam, flush=True)

    lam_t = {en["tag"]: en["lam"] for ents in site_entries.values()
             for en in ents}
    for n in names:
        if n not in lam_t:
            raise RuntimeError(f"plug {n} not found in installed entries")

    def set_lams(d):
        for n, v in d.items():
            lam_t[n].data.fill_(float(v))

    # ---- streams: sel=λ-select, rep=v0.5-comparable, held=honest ----
    seq = 512
    streams = {}
    for n in names:
        streams[n] = home_stream(tok, HOME_OF[n], args, seq)
    base_ids = []
    for name, text in d0.load_domain_texts(args.domains_dir).items():
        base_ids.extend(tok.encode(text, add_special_tokens=False))
    base_ids = base_ids[:args.max_base_tokens]
    base = g0.to_batches(base_ids, 0, seq, 4)
    sel = {n: pick(streams[n], 12, 20) for n in names}
    rep = {n: pick(streams[n], 0, 8) for n in names}
    held = {n: pick(streams[n], 20, 28) for n in names}
    base_sel = pick(base, 8, 16)
    base_rep = pick(base, 0, 4)
    base_held = pick(base, 16, 24)

    out = {"gate": "v05-lam-refit", "format": "vesper-v05.1",
           "phase": "start", "pid": os.getpid(), "grid": grid,
           "lambda_start": lam0, "bar_base_tax": BASE_TAX_BAR}
    dump(out, args.out)

    # (lever-4 xfire-recal runs after the cross-fire read below)

    # ---- before table (current λ) ----
    def measure(lams, sl):
        set_lams(lams)
        home_ce = {n: round(ce_on(model, blocks, fam, gates, handles,
                                  sl[n], device), 4) for n in names}
        b = round(ce_on(model, blocks, fam, gates, handles,
                        base_rep if sl is rep else base_sel, device), 4)
        return home_ce, b

    set_lams({n: 0.0 for n in names})
    masked_home = {n: round(ce_on(model, blocks, fam, None, None, rep[n],
                                  device), 4) for n in names}
    masked_base = round(ce_on(model, blocks, fam, None, None, base_rep,
                              device), 4)
    before_home, before_base = measure(lam0, rep)
    out["before"] = {"lams": lam0, "home_rep": before_home,
                     "base_rep": before_base, "masked": masked_home,
                     "masked_base": masked_base}
    dump(out, args.out)
    print("[refit] BEFORE (rep):", before_home, "base", before_base,
          flush=True)

    # ---- cross-fire matrix (membership share plug × home) ----
    set_lams(lam0)
    xfire = memb_matrix(model, blocks, fam, gates, handles,
                        {HOME_OF[n]: sel[n] for n in names}, device,
                        site_entries)
    out["cross_fire"] = xfire
    dump(out, args.out)
    for home, row in xfire.items():
        print(f"[xfire {home:9s}]", row, flush=True)

    # ---- lever-4: off-home reject re-recal for flooding rows ----
    def row_idx_of(n):
        hi = [i["host"] for i in info if i["name"] == n][0]
        ents = site_entries[hi]
        tags = [en["tag"] for en in ents]
        bank = site_banks[hi]
        return hi, bank, bank.num_experts - len(ents) + tags.index(n)

    if args.xfire_recal:
        print(f"[refit] --xfire-recal reject_w={args.xfire_reject_w}",
              flush=True)
        for n in args.xfire_recal.split(","):
            n = n.strip()
            if not n:
                continue
            hi, bank, ridx = row_idx_of(n)
            # reject side must SEE the other homes' tokens (they are what
            # the row floods) — stream = my home + every other home + base
            off_home = [b for m in names if m != n for b in sel[m]]
            pairs = g0.run_in_state(
                model, blocks, fam, gates, handles, "both",
                lambda: g0.collect_pairs_fam(
                    model, blocks, fam,
                    sel[n] + off_home + base_sel, device, 4096,
                    lambda _i, b: 1 if b[1] == b5.HOME_LABEL[HOME_OF[n]]
                    else 0))
            rep_r = d6.train_rows_one(bank, pairs[hi], ridx, 200, 0.05,
                                      device,
                                      reject_w=args.xfire_reject_w,
                                      mass_w=2.0, owner_target=0.85)
            print(f"[refit]  {n} xfire-recal: {rep_r['first']} -> "
                  f"{rep_r['last']}", flush=True)
            out.setdefault("xfire_recal", {})[n] = rep_r
            dump(out, args.out)

    # ---- per-home λ fits, round-robin ----
    history = []
    for rnd in range(args.rounds):
        for n in (names if rnd == 0 else ["glm_cybersec", "mimo_writing"]):
            best_v, best_ce = lam[n], None
            curve = []
            for v in grid:
                trial = dict(lam)
                trial[n] = v
                set_lams(trial)
                ce = ce_on(model, blocks, fam, gates, handles, sel[n], device)
                curve.append({"lam": v, "ce_sel": round(ce, 4)})
                if best_ce is None or ce < best_ce - 1e-4:
                    best_ce, best_v = ce, v
            lam[n] = best_v
            history.append({"round": rnd, "plug": n, "lam": best_v,
                            "ce_sel": round(best_ce, 4), "curve": curve})
            print(f"[refit r{rnd}] {n:14s} lam -> {best_v} "
                  f"(ce_sel {best_ce:.4f} curve "
                  f"{[(c['lam'], c['ce_sel']) for c in curve]})", flush=True)
            dump(out, args.out)
    out["lam_home_fit"] = lam
    out["grid_history"] = history
    dump(out, args.out)

    # winners guard: revert a winner if its rep CE regressed vs lam0
    for n in ["code", "math", "mem_pet", "mem_edit", "mem_words", "k3_hack"]:
        if abs(lam[n] - lam0[n]) < 1e-9:
            continue
        trial = dict(lam)
        set_lams(trial)
        ce_new = ce_on(model, blocks, fam, gates, handles, rep[n], device)
        trial[n] = lam0[n]
        set_lams(trial)
        ce_old = ce_on(model, blocks, fam, gates, handles, rep[n], device)
        if ce_new > ce_old + 1e-4:
            lam[n] = lam0[n]
            print(f"[refit] winner {n} reverted to {lam0[n]} "
                  f"(rep {ce_new:.4f} > {ce_old:.4f})", flush=True)
    out["lam_after_winners_guard"] = dict(lam)
    dump(out, args.out)

    # ---- base-tax guard ----
    set_lams(lam)
    base_now = ce_on(model, blocks, fam, gates, handles, base_sel, device)
    masked_base_sel = ce_on(model, blocks, fam, None, None, base_sel, device)
    tax = base_now - masked_base_sel
    out["base_tax_sel"] = {"grafted": round(base_now, 4),
                           "masked": round(masked_base_sel, 4),
                           "tax": round(tax, 4)}
    print(f"[refit] base tax before guard: {tax:+.4f}", flush=True)
    # shrink order: starved plugs first, then smallest home gain per step
    def home_gain(n):
        set_lams(lam)
        g_here = ce_on(model, blocks, fam, gates, handles, sel[n], device)
        m_here = ce_on(model, blocks, fam, None, None, sel[n], device)
        return m_here - g_here

    guard_steps = 0
    while tax > BASE_TAX_BAR and guard_steps < 12:
        guard_steps += 1
        gains = {n: home_gain(n) for n in names if lam[n] > 0}
        # shrink the plug with the least home gain (starved plugs first on ties)
        n = min(gains, key=lambda k: (gains[k], k not in
                                      ("glm_cybersec", "mimo_writing")))
        lower = [v for v in grid if v < lam[n] - 1e-9]
        if not lower:
            lam[n] = 0.0
        else:
            lam[n] = max(lower)
        set_lams(lam)
        base_now = ce_on(model, blocks, fam, gates, handles, base_sel, device)
        tax = base_now - masked_base_sel
        print(f"[refit guard] shrink {n} -> {lam[n]}  tax {tax:+.4f}",
              flush=True)
    out["lam_final"] = dict(lam)
    out["base_tax_sel_after_guard"] = {"tax": round(tax, 4)}
    dump(out, args.out)

    # ---- final measurement: rep + held ----
    set_lams(lam)
    final_rep = {n: round(ce_on(model, blocks, fam, gates, handles, rep[n],
                                device), 4) for n in names}
    final_held = {n: round(ce_on(model, blocks, fam, gates, handles, held[n],
                                 device), 4) for n in names}
    set_lams({n: 0.0 for n in names})
    masked_held = {n: round(ce_on(model, blocks, fam, None, None, held[n],
                                  device), 4) for n in names}
    set_lams(lam)
    final_base_rep = ce_on(model, blocks, fam, gates, handles, base_rep,
                           device)
    final_base_held = ce_on(model, blocks, fam, gates, handles, base_held,
                            device)
    set_lams({n: 0.0 for n in names})
    masked_base_held = ce_on(model, blocks, fam, None, None, base_held,
                             device)
    out["final"] = {
        "lams": dict(lam),
        "home_rep": final_rep, "masked_rep": masked_home,
        "delta_rep": {n: round(final_rep[n] - masked_home[n], 4)
                      for n in names},
        "home_held": final_held, "masked_held": masked_held,
        "delta_held": {n: round(final_held[n] - masked_held[n], 4)
                       for n in names},
        "base_rep": round(final_base_rep, 4),
        "masked_base_rep": masked_base,
        "base_tax_rep": round(final_base_rep - masked_base, 4),
        "base_held": round(final_base_held, 4),
        "masked_base_held": round(masked_base_held, 4),
        "base_tax_held": round(final_base_held - masked_base_held, 4),
    }
    # marginal home gain for the starved plugs at the FINAL vector:
    # ce(home) with its own λ=0 vs λ=final, all others fixed (bar clause 2)
    marginal = {}
    for n in ("glm_cybersec", "mimo_writing", "k3_hack"):
        keep = lam[n]
        t0 = dict(lam)
        t0[n] = 0.0
        set_lams(t0)
        ce0 = ce_on(model, blocks, fam, gates, handles, rep[n], device)
        t1 = dict(lam)
        t1[n] = keep
        set_lams(t1)
        ce1 = ce_on(model, blocks, fam, gates, handles, rep[n], device)
        marginal[n] = {"ce_at_0": round(ce0, 4), "ce_at_lam": round(ce1, 4),
                       "home_gain": round(ce0 - ce1, 4)}
    out["final"]["marginal"] = marginal
    print("[refit] marginal:", marginal, flush=True)
    green = all(v < 0 for v in out["final"]["delta_rep"].values())
    out["verdict"] = {
        "all8_green_rep": green,
        "all8_green_held": all(v < 0 for v in
                               out["final"]["delta_held"].values()),
        "base_tax_rep": out["final"]["base_tax_rep"],
        "base_tax_held": out["final"]["base_tax_held"],
        "bar_base_tax": BASE_TAX_BAR,
        "glm_ok": (out["final"]["delta_rep"]["glm_cybersec"] < 0
                   or (lam.get("glm_cybersec", 0) > 0
                       and marginal["glm_cybersec"]["home_gain"] > 0)),
        "mimo_ok": (out["final"]["delta_rep"]["mimo_writing"] < 0
                    or (lam.get("mimo_writing", 0) > 0
                        and marginal["mimo_writing"]["home_gain"] > 0)),
        "winners_kept": all(out["final"]["delta_rep"][n] < 0 for n in
                            ("code", "math", "mem_pet", "mem_edit",
                             "mem_words", "k3_hack")),
    }
    out["phase"] = "done"
    dump(out, args.out)
    print("[refit] FINAL lams:", lam, flush=True)
    print("[refit] delta rep :", out["final"]["delta_rep"], flush=True)
    print("[refit] delta held:", out["final"]["delta_held"], flush=True)
    print("[refit] verdict   :", out["verdict"], flush=True)

    # ---- persist joint_state.pt (backup first) ----
    js_path = os.path.join(args.plugins_dir, "joint_state.pt")
    if os.path.exists(js_path):
        backup = os.path.join(args.plugins_dir, "joint_state_pre_v051.pt")
        if not os.path.exists(backup):
            torch.save(torch.load(js_path, map_location="cpu",
                                  weights_only=False), backup)
        js = torch.load(js_path, map_location="cpu", weights_only=False)
    else:
        js = {"rows": {}, "order": names}
    js["lams"] = [float(lam[n]) for n in (js.get("order") or names)]
    js["order"] = list(js.get("order") or names)
    js["rows"] = {n: site_banks[row_idx_of(n)[0]].P.data[
        row_idx_of(n)[2]].float().cpu().clone() for n in (js["order"] or names)}
    js["v051"] = {"lams": dict(lam), "verdict": out["verdict"]}
    torch.save(js, js_path)
    print(f"[refit] joint_state.pt updated {js_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
