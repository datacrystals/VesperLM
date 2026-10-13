#!/usr/bin/env python3
"""varact_sib_recal — close the mem-trio sibling pair (varact rung-2 #1).

The only bar failing the tau=0.5 ship rule is sibling pairwise harm
(mem_pet->mem_words +0.202, mem_edit->mem_words +0.143, identical in the
ungated baseline): the trio rows rate each other's episodes at p~0.99 — a
row-selectivity defect.  This runs the v0.5.1 cross-fire recal recipe
(exactly the lever that took k3 off-home p 0.028 -> 0.0004) on the trio:

  reject_w 40, mass_w 2.0, owner_target 0.85, 200 steps, lr 0.05
  stream  = OWN home episodes (label 1) + SIBLING episodes + base (label 0)

per row, only that row's grad enabled (d6.train_rows_one).  Guards:
  * own-home CE (tau=0.5 delivery) must not regress beyond GCE_GUARD —
    otherwise the row is reverted (the "own-home protected" clause);
  * the taught fact token is sacred: the memory probe "What is my cat's
    name?" must still answer "Kepler" after each row, else that row reverts.

joint_state.pt is rewritten ONLY on success and backed up first
(joint_state_pre_sibrecal.pt); --no-save leaves it untouched.

Usage (box):
  python3 lab/varact_sib_recal.py --phase all --out /data/tliao/varact_out/sibrecal.json
  python3 lab/varact_sib_recal.py --phase check          # measure only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("ROCBLAS_TENSILE_LIBPATH",
                      "/opt/rocm-6.2.4/lib/rocblas/library")

import torch  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import graft_d0_parity as d0  # noqa: E402
import graft_d2_crossfamily as d2  # noqa: E402
import door6_expert_mix as d6  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402
import v05_build as b5  # noqa: E402
import chat_v05 as c5  # noqa: E402
import varact_gate as va  # noqa: E402

MEM = ["mem_pet", "mem_edit", "mem_words"]
HOME_OF = va.HOME_OF
ORDER = va.ORDER
BAR_HARM = 0.02
GCE_GUARD = 0.02          # own-home CE regression band (nats)
MEM_PROBE_Q = "What is my cat's name?"


def dump(out: dict, path: str) -> None:
    g0.dump_json(out, path)


def build_world(args):
    device = args.device
    cargs = SimpleNamespace(**vars(args), default_lam=0.5, lam_map={},
                            bare=False)
    (model, tok, blocks, fam, gates, handles, info, missing,
     site_entries, site_banks) = c5.build_state(cargs, device)
    names = [i["name"] for i in info]
    wrappers = va.install_varact(handles, va.POLICIES["base"]())
    seq = 512
    streams = {n: va.home_stream(tok, HOME_OF[n], args, seq) for n in names}
    base_ids = []
    for _name, text in d0.load_domain_texts(args.domains_dir).items():
        base_ids.extend(tok.encode(text, add_special_tokens=False))
    base_ids = base_ids[:args.max_base_tokens]
    base = g0.to_batches(base_ids, 0, seq, 4)

    def logits_fn(x):
        return g0.run_in_state(
            model, blocks, fam, gates, handles, "both",
            lambda: d0.model_logits(model, x.to(device)))

    def row_of(n):
        hi = [i["host"] for i in info if i["name"] == n][0]
        ents = site_entries[hi]
        tags = [en["tag"] for en in ents]
        bank = site_banks[hi]
        return hi, bank, bank.num_experts - len(ents) + tags.index(n)

    return SimpleNamespace(
        model=model, tok=tok, blocks=blocks, fam=fam, gates=gates,
        handles=handles, info=info, names=names, wrappers=wrappers,
        site_entries=site_entries, site_banks=site_banks, device=device,
        streams=streams, base=base, logits_fn=logits_fn, row_of=row_of)


@torch.no_grad()
def home_ce_tau05(w, n, windows=6):
    """Home CE for plug n's home under tau=0.5 delivery (the ship config)."""
    va.set_policy(w.wrappers, va.POLICIES["tau0.5"]())
    batches = va.pick(w.streams[n], 0, windows)
    return va.eval_pass(w.model, w.blocks, w.fam, w.gates, w.handles,
                        batches, w.device, w.wrappers)["ce"]


@torch.no_grad()
def kepler_probe(w):
    text, _ids = g0.gen_reply(w.tok, w.logits_fn,
                              [{"role": "user", "content": MEM_PROBE_Q}],
                              16, w.device)
    flat = " ".join(text.split())
    return {"excerpt": flat[:120], "kepler": "kepler" in flat.lower()}


@torch.no_grad()
def sibling_loo(w, names=None, windows=4):
    """LOO harm for the trio pairs under a given policy (already set).

    harm(j->h) = CE_h(all) - CE_h(j off); + = j hurts h.
    """
    names = names or ORDER
    homes = {n: va.pick(w.streams[n], 0, windows) for n in names}
    full = {n: va.ce_with_active(w.model, w.blocks, w.fam, w.gates,
                                 w.handles, homes[n], w.device, set(names))
            for n in names}
    loo = {}
    for n in names:
        row = {}
        for j in names:
            if j == n:
                continue
            ce_wo = va.ce_with_active(w.model, w.blocks, w.fam, w.gates,
                                      w.handles, homes[n], w.device,
                                      set(names) - {j})
            row[j] = round(full[n] - ce_wo, 4)
        loo[n] = row
    worst_sib, sib_pair = 0.0, ("", "")
    worst_ns, ns_pair = 0.0, ("", "")
    for n in names:
        for j, v in loo[n].items():
            sib = (j.startswith("mem") and n.startswith("mem")
                   and HOME_OF[j] != HOME_OF[n])
            if sib and v > worst_sib:
                worst_sib, sib_pair = v, (j, n)
            if not sib and HOME_OF[j] != HOME_OF[n] and v > worst_ns:
                worst_ns, ns_pair = v, (j, n)
    return {"full_ce": {k: round(v, 4) for k, v in full.items()}, "loo": loo,
            "worst_mem_sibling": round(worst_sib, 4),
            "worst_mem_sibling_pair": list(sib_pair),
            "worst_non_sibling": round(worst_ns, 4),
            "worst_non_sibling_pair": list(ns_pair)}


# ---------------------------------------------------------------------------
# family-margin gate (rung-2 #2) — two-pass measurement form.  Non-causal
# by construction: pass 1 collects every trio member's p_home per token
# (each lives at its own host layer), pass 2 keeps member j on token t only
# where p_j(t) >= max_sibling_p(t) + margin.  Exact for the harm bar; a
# serving implementation pays a second scoring prefill.
# ---------------------------------------------------------------------------

def fammargin_pass1(w, batches):
    """Return {plug: (T_total,) float p_home} over the given batches."""
    col = {}
    for n in MEM:
        hi, bank, ridx = w.row_of(n)
        local = ridx - (bank.num_experts - 1)   # the single plug col at hi
        col[n] = (hi, local)
    acc = {n: [] for n in MEM}
    for tokens, _ in batches:
        tokens = tokens.to(w.device)
        g0.run_in_state(w.model, w.blocks, w.fam, w.gates, w.handles, "both",
                        lambda t=tokens: d0.model_logits(w.model, t))
        for n in MEM:
            hi, local = col[n]
            p = w.gates[hi].last_p_new[:, local].float().cpu()
            acc[n].append(p)
    return {n: torch.cat(v) for n, v in acc.items()}


@torch.no_grad()
def fammargin_ce(w, batches, masks_by_tag, device):
    """CE over batches applying per-tag keep masks (one (T,1) bool per
    forward, sliced in call order)."""
    class Tagged:
        mode = "mask"
        tau = 0.0
        top_m = 0

        def __init__(self, tag):
            self.tag = tag

        def name(self):
            return f"fammarg:{self.tag}"

        def select(self, p_new, memb):
            m = masks_by_tag.get(self.tag)
            if m is None:
                return None
            T = memb.shape[0] if memb is not None else p_new.shape[0]
            off = cursors[self.tag]
            cursors[self.tag] = off + T
            return m[off:off + T].to(memb.device if memb is not None
                                     else m.device)

    saved = [vm.policy for vm in w.wrappers]
    cursors = {t: 0 for t in masks_by_tag}
    try:
        for vm in w.wrappers:
            vm.policy = Tagged(vm.bkey[0])
        tot, n = 0.0, 0
        for tokens, _ in batches:
            tokens = tokens.to(device)

            def one():
                nonlocal tot, n
                lg = d0.model_logits(w.model, tokens)
                tot += d0.next_token_ce(lg, tokens) * tokens.shape[0]
                n += tokens.shape[0]

            g0.run_in_state(w.model, w.blocks, w.fam, w.gates, w.handles,
                            "both", one)
        return tot / max(1, n)
    finally:
        for vm, pol0 in zip(w.wrappers, saved):
            vm.policy = pol0


def fammargin_loo(w, margin=0.05, windows=4):
    """LOO trio matrix under tau=0.5 + family-margin winner rule."""
    homes = {n: va.pick(w.streams[n], 0, windows) for n in MEM}
    va.set_policy(w.wrappers, va.POLICIES["tau0.5"]())
    base_masks = {}
    p_all = {}
    for n in MEM:
        batches = homes[n]
        ps = fammargin_pass1(w, batches)
        T = ps[MEM[0]].shape[0]
        stacked = torch.stack([ps[m] for m in MEM], dim=1)   # (T, 3)
        for i, m in enumerate(MEM):
            others = torch.cat([stacked[:, :i], stacked[:, i + 1:]], dim=1)
            max_other = others.max(dim=1)[0] if others.shape[1] \
                else torch.zeros_like(stacked[:, i])
            win = ((stacked[:, i] >= max_other + margin) &
                   (stacked[:, i] >= 0.5))
            base_masks.setdefault(m, {})[n] = win
        p_all[n] = {m: round(float(ps[m].mean()), 4) for m in MEM}
    out = {"margin": margin, "mean_p_by_home": p_all}
    full = {}
    for n in MEM:
        masks = {MEM[i]: base_masks[MEM[i]][n].unsqueeze(-1)
                 for i in range(len(MEM))}
        full[n] = round(fammargin_ce(w, homes[n], masks, w.device), 4)
    loo = {}
    for n in MEM:
        row = {}
        for j in MEM:
            if j == n:
                continue
            masks = {MEM[i]: base_masks[MEM[i]][n].unsqueeze(-1)
                     for i in range(len(MEM))}
            masks[j] = torch.zeros_like(masks[j])
            ce_wo = fammargin_ce(w, homes[n], masks, w.device)
            row[j] = round(full[n] - ce_wo, 4)
        loo[n] = row
    worst_sib, sib_pair = 0.0, ("", "")
    for n in MEM:
        for j, v in loo[n].items():
            if HOME_OF[j] != HOME_OF[n] and v > worst_sib:
                worst_sib, sib_pair = v, (j, n)
    out.update({"full_ce": full, "loo": loo,
                "worst_mem_sibling": round(worst_sib, 4),
                "worst_mem_sibling_pair": list(sib_pair)})
    return out


def recal(args, w, out):
    """Per-row recal with own-home + Kepler guards; revert on regression."""
    names = w.names
    # baselines under the ship config
    base_home = {n: home_ce_tau05(w, n) for n in MEM}
    probe0 = kepler_probe(w)
    out["before"] = {"home_ce_tau05": base_home, "probe": probe0}
    dump(out, args.out)
    print(f"[sib] BEFORE home_ce(tau0.5) {base_home}", flush=True)
    print(f"[sib] BEFORE probe {probe0}", flush=True)

    base_batches = va.pick(w.base, 0, 4)
    saved_rows = {}
    for n in MEM:
        hi, bank, ridx = w.row_of(n)
        saved_rows[n] = bank.P.data[ridx].float().cpu().clone()
    results = {}
    targets = [n for n in MEM
               if not args.only or n in {s.strip() for s in args.only.split(",")}]

    for n in targets:
        hi, bank, ridx = w.row_of(n)
        # home-weight tiles the OWN-home windows so the reject side does not
        # drown the home signal (mem_pet guard lesson: 1x home vs 2.5x reject
        # at reject_w 40 over-suppresses the row and kills the fact token)
        own = w.streams[n] * max(1, args.home_weight)
        sib_names = [m for m in MEM if m != n
                     if not args.sib_only
                     or m in {s.strip() for s in args.sib_only.split(",")}]
        sib = [b for m in sib_names for b in w.streams[m]]
        stream = own + sib + base_batches
        lab_home = b5.HOME_LABEL[HOME_OF[n]]
        pairs = g0.run_in_state(
            w.model, w.blocks, w.fam, w.gates, w.handles, "both",
            lambda s=stream, lh=lab_home: g0.collect_pairs_fam(
                w.model, w.blocks, w.fam, s, w.device, 4096,
                lambda _i, b: 1 if b[1] == lh else 0))
        # chunked guard-checkpointed recal: stop at the first chunk that
        # breaks a guard and keep the best (lowest p_reject) passing row
        best = {"p_reject": None, "row": saved_rows[n].clone(), "kept": False}
        chunks = max(1, args.chunks)
        per = max(1, args.steps // chunks)
        rep_r = None
        for c in range(chunks):
            rep_r = d6.train_rows_one(bank, pairs[hi], ridx, per, args.lr,
                                      w.device, reject_w=args.reject_w,
                                      mass_w=2.0, owner_target=0.85)
            ce_after = home_ce_tau05(w, n)
            probe = kepler_probe(w)
            ok = (ce_after <= base_home[n] + GCE_GUARD) and probe["kepler"]
            p_rej = rep_r["last"]["p_new_base"]
            print(f"[sib] {n} chunk {c}: p_reject {p_rej:.4f} "
                  f"ce {ce_after:.4f} kepler={probe['kepler']} "
                  f"{'ok' if ok else 'GUARD'}", flush=True)
            if ok and (best["p_reject"] is None or p_rej < best["p_reject"]):
                best = {"p_reject": p_rej,
                        "row": bank.P.data[ridx].float().cpu().clone(),
                        "kept": True}
            if not ok:
                bank.P.data[ridx].copy_(
                    best["row"].to(bank.P.device))
                print(f"[sib] {n} early-stop at chunk {c} (reverted to "
                      f"best p_reject={best['p_reject']})", flush=True)
                break
        bank.P.data[ridx].copy_(best["row"].to(bank.P.device))
        ce_after = home_ce_tau05(w, n)
        probe = kepler_probe(w)
        if best["kept"]:
            base_home[n] = ce_after
        else:
            print(f"[sib] {n} REVERTED (no passing chunk)", flush=True)
        results[n] = {"recal": rep_r,
                      "p_reject_final": best["p_reject"],
                      "home_ce_tau05_after": round(ce_after, 4),
                      "probe": probe, "kept": bool(best["kept"])}
        out["rows"] = results
        dump(out, args.out)
    out["after_home_ce_tau05"] = {n: home_ce_tau05(w, n) for n in MEM}
    out["after_probe"] = kepler_probe(w)
    out["phase"] = "recal_done"
    dump(out, args.out)
    return results


def persist(args, w, out):
    """Backup joint_state.pt then store the (guarded) trio rows."""
    js_path = os.path.join(args.plugins_dir, "joint_state.pt")
    if args.no_save:
        out["saved"] = False
        print("[sib] --no-save: joint_state.pt untouched", flush=True)
        return
    if os.path.exists(js_path):
        backup = os.path.join(args.plugins_dir, "joint_state_pre_sibrecal.pt")
        if not os.path.exists(backup):
            torch.save(torch.load(js_path, map_location="cpu",
                                  weights_only=False), backup)
            print(f"[sib] backup {backup}", flush=True)
        js = torch.load(js_path, map_location="cpu", weights_only=False)
    else:
        js = {"rows": {}, "order": w.names}
    js.setdefault("rows", {})
    js["order"] = list(js.get("order") or w.names)
    for n in MEM:
        _hi, bank, ridx = w.row_of(n)
        js["rows"][n] = bank.P.data[ridx].float().cpu().clone()
    # shipped default delivery policy (chat_v05/chat_v06 read this flag;
    # --varact still overrides)
    js["varact"] = getattr(args, "ship_varact", "tau0.5")
    js["sibrecal"] = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "reject_w": args.reject_w, "steps": args.steps,
                      "owner_target": 0.85,
                      "rows": out.get("rows", {}),
                      "after_home_ce_tau05": out.get("after_home_ce_tau05"),
                      "after_probe": out.get("after_probe")}
    tmp = js_path + ".tmp"
    torch.save(js, tmp)
    os.replace(tmp, js_path)
    out["saved"] = True
    print(f"[sib] joint_state.pt updated {js_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="all",
                    choices=["recal", "check", "all", "fammarg"])
    ap.add_argument("--model", default="/data/tliao/Phi-mini-MoE-instruct")
    ap.add_argument("--registry",
                    default="/data/tliao/graft_d2_out/registry_phimoe_f620b32c.pt")
    ap.add_argument("--domains-dir", default="/data/tliao/graft_d2_slice")
    ap.add_argument("--specialties-dir",
                    default="/data/tliao/vesper_sft/VesperLM/Dataset/data/specialties")
    ap.add_argument("--plugins-dir", default="/data/tliao/v05_out")
    ap.add_argument("--out", default="/data/tliao/varact_out/sibrecal.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-tokens", type=int, default=70000)
    ap.add_argument("--max-base-tokens", type=int, default=24000)
    ap.add_argument("--win", type=int, default=6)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--chunks", type=int, default=4,
                    help="guard-checkpointed training chunks (early-stop)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--reject-w", type=float, default=40.0)
    ap.add_argument("--only", default="",
                    help="comma subset of mem_pet,mem_edit,mem_words")
    ap.add_argument("--sib-only", default="",
                    help="comma subset of siblings to place on the reject "
                         "side (default: both)")
    ap.add_argument("--home-weight", type=int, default=1,
                    help="tile own-home windows N times in the recal stream")
    ap.add_argument("--fammarg-margin", type=float, default=0.05,
                    help="family-margin winner rule threshold (fammarg phase)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--gpu", default="auto")
    args = ap.parse_args()

    dev_i = c5.pick_gpu() if args.gpu == "auto" else int(args.gpu)
    args.device = f"cuda:{dev_i}"
    torch.cuda.set_device(dev_i)
    torch.manual_seed(0)

    out = {"gate": "varact-sib-recal", "pid": os.getpid(),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    dump(out, args.out)

    w = build_world(args)
    print(f"[sib] live plugs: {w.names}", flush=True)

    if args.phase in ("recal", "all"):
        recal(args, w, out)
        persist(args, w, out)

    # post-check: LOO sibling harm (trio matrix ungated + full matrix tau0.5)
    if args.phase in ("check", "all"):
        for pol_name, nms in (("base", MEM), ("tau0.5", ORDER)):
            va.set_policy(w.wrappers, va.POLICIES[pol_name]())
            h = sibling_loo(w, names=nms)
            out.setdefault("harm_after", {})[pol_name] = h
            dump(out, args.out)
            print(f"[sib AFTER {pol_name}] sib {h['worst_mem_sibling']} "
                  f"{h['worst_mem_sibling_pair']} non-sib "
                  f"{h['worst_non_sibling']} {h['worst_non_sibling_pair']}",
                  flush=True)
    if args.phase == "fammarg":
        fm = fammargin_loo(w, margin=args.fammarg_margin)
        out["fammarg"] = fm
        dump(out, args.out)
        print(f"[sib FAMMARG m={args.fammarg_margin}] sib "
              f"{fm['worst_mem_sibling']} {fm['worst_mem_sibling_pair']} "
              f"full_ce {fm['full_ce']}", flush=True)
    out["probe_final"] = kepler_probe(w)
    out["phase"] = "done"
    dump(out, args.out)
    print("[sib] FINAL probe", out["probe_final"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
