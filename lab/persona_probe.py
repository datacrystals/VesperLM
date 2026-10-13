#!/usr/bin/env python3
"""persona_probe — the shipped-bar verification for the persona memory pack.

Runs the four 30038bd-era bars against lab/memories/persona_v1.json on a
built spine (joint_state_persona.pt + 6 persona plugins, chat defaults:
varact tau=0.5 + taught-token bonus beta=3.0 + --qscope overlap):

  (a) single    one fresh probe per fact — bar 6/6 (plus sacred Kepler)
  (b) multi     each fact asked as turn t4 / t8 of 3 canned mtfact
                scaffolds — bar 5/6 cells per fact (the shipped mid bar)
  (c) live      the real chat_v05/chat_v06 --probe path is exercised
                separately (persona_chain / CLI); this harness records the
                non-memory turns of the same PROBES for leak checks
  (d) canary    no fact leakage on unrelated questions; the facts must fire
                on similar-right questions ("favorite anime?" -> Gundam) and
                stay silent on similar-wrong ones ("favorite color?")

Verdicts are numeric / one-word (provider safety filter): each cell prints
PASS|FAIL plus the matched row and the answer's key token.

Usage (box):
  python3 lab/persona_probe.py --spine phi --phase all \
      --out /data/tliao/persona_out/probe_phi.json
  python3 lab/persona_probe.py --spine qwen --phase all --kv
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
import grafted_spine_v0 as g0  # noqa: E402
import mem_pack as mp  # noqa: E402
import chat_v05 as c5  # noqa: E402
import varact_gate as va  # noqa: E402
import mtfact as mf  # noqa: E402

SPINES = {
    "phi": {"plugins": "/data/tliao/v05_out"},
    "qwen": {"plugins": "/data/tliao/v06_out"},
}
SCAFFOLD_NAMES = ("wired", "personal", "tech")
DEPTHS = (4, 8)


def dump(out: dict, path: str | None) -> None:
    g0.dump_json(out, path)


def build_world(spine: str, device: str, kv: bool):
    plugins = SPINES[spine]["plugins"]
    cargs = SimpleNamespace(model="", registry="", plugins_dir=plugins,
                            device=device, dtype="float16",
                            default_lam=0.5, lam_map={}, bare=False)
    # chat_v05.build_state reads args.model/registry from the defaults when
    # the strings are empty -> patch via module defaults per spine
    if spine == "phi":
        cargs.model = c5.DEFAULT_MODEL
        cargs.registry = c5.DEFAULT_REGISTRY
    else:
        import chat_v06 as c6
        cargs.model = c6.DEFAULT_MODEL
        cargs.registry = c6.DEFAULT_REGISTRY
    (model, tok, blocks, fam, gates, handles, info, missing,
     site_entries, site_banks) = c5.build_state(cargs, device)
    pack = mp.load_pack()
    mp.apply_to_varact(va, pack)
    wrappers = va.install_varact(handles, va.POLICIES["tau0.5"]())
    scope = va.QuestionScope("overlap", tau=0.5)
    tag_col = {}
    for host, ents in site_entries.items():
        for j, en in enumerate(ents):
            tag_col[en["tag"]] = (host, j)
    bonus = va.TaughtTokenBonus(3.0, 0.5, scope=scope)
    bonus.note(tok)

    def fn(x):
        def run():
            lg = d0.model_logits(model, x.to(device))
            return bonus.edit(lg, va.capture_tags(gates, tag_col,
                                                  qspan=scope.span))
        return g0.run_in_state(model, blocks, fam, gates, handles, "both", run)

    decoder = None
    if kv:
        import kv_cache_v05 as kv5
        decoder = kv5.KVDecoder(model, tok, device, blocks, fam, gates,
                                handles)
        # chat_v06 ship pattern: wrap KVDecoder._forward so the taught-token
        # bonus sees every decode step (kv_cache_v05 stays read-only)
        base_fwd = decoder._forward

        def _fwd_bonus(x, _b=base_fwd):
            qspan = scope.span if scope is not None else None
            return bonus.edit(_b(x), va.capture_tags(gates, tag_col,
                                                     qspan=qspan))

        decoder._forward = _fwd_bonus
    return SimpleNamespace(model=model, tok=tok, blocks=blocks, fam=fam,
                           gates=gates, handles=handles, info=info,
                           wrappers=wrappers, scope=scope, bonus=bonus,
                           tag_col=tag_col, logits_fn=fn, device=device,
                           pack=pack, decoder=decoder, kv=kv)


def gen(w, question, messages=None, max_new=24):
    """Greedy answer to `question` (appended to `messages` if given)."""
    messages = list(messages or []) + [{"role": "user", "content": question}]
    txt = w.tok.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    ids = w.tok.encode(txt, add_special_tokens=False)
    w.scope.begin(question, w.tok, ids)
    if w.decoder is not None:
        text, out_ids, _st = w.decoder.generate(
            messages, max_new, "grafted")
    else:
        text, out_ids = g0.gen_reply(w.tok, w.logits_fn, messages, max_new,
                                     w.device)
    matched = list(w.scope.last_match)
    flat = " ".join(str(text).split())
    return flat, matched


def verdict_answer(flat, expect, reject):
    verd, hit, bad, _key = mp.verdict(flat, expect, reject)
    return verd, hit, bad


def answer_key(flat, expect):
    return mp.verdict(flat, expect, [])[3]


def phase_single(w, out):
    rows = []
    for tag, q, expect, reject in mp.probes(w.pack):
        t0 = time.time()
        flat, matched = gen(w, q, max_new=24)
        verd, hit, bad = verdict_answer(flat, expect, reject)
        key = answer_key(flat, expect)
        rows.append({"tag": tag, "q": q, "verdict": verd, "key": key,
                     "matched": matched, "sec": round(time.time() - t0, 1),
                     "answer": flat[:80]})
        print(f"[single {tag:12s}] {verd} key={key} matched={matched} "
              f"{flat[:60]!r}", flush=True)
    # sacred legacy facts stay green
    legacy = [("mem_pet", "What is my cat's name?", ["kepler"], ["orion"]),
              ("mem_edit", "What is my favorite text editor?",
               ["helix"], ["vim"])]
    for tag, q, expect, reject in legacy:
        flat, matched = gen(w, q, max_new=16)
        verd, _h, _b = verdict_answer(flat, expect, reject)
        rows.append({"tag": tag, "q": q, "verdict": verd, "key": expect[0],
                     "matched": matched, "answer": flat[:80]})
        print(f"[single {tag:12s}] {verd} matched={matched} {flat[:60]!r}",
              flush=True)
    npass = sum(1 for r in rows if r["verdict"] == "PASS")
    out["single"] = {"rows": rows, "pass": npass, "total": len(rows)}
    out["bar_single"] = f"{npass}/{len(rows)}"
    print(f"[single] {npass}/{len(rows)}", flush=True)
    dump(out, args_out)
    return out


def phase_multi(w, out, max_new=20):
    cells = []
    for tag, q, expect, reject in mp.probes(w.pack):
        npass = 0
        ntot = 0
        for sc in SCAFFOLD_NAMES:
            for turns in DEPTHS:
                messages = mf.build_messages(sc, turns)[:-1]
                t0 = time.time()
                flat, matched = gen(w, q, messages=messages, max_new=max_new)
                verd, _h, _b = verdict_answer(flat, expect, reject)
                key = answer_key(flat, expect)
                cells.append({"tag": tag, "scaffold": sc, "turns": turns,
                              "verdict": verd, "key": key,
                              "matched": matched,
                              "sec": round(time.time() - t0, 1),
                              "answer": flat[:60]})
                npass += verd == "PASS"
                ntot += 1
                print(f"[multi {tag:12s} {sc:8s} t{turns}] {verd} key={key} "
                      f"{flat[:50]!r}", flush=True)
        cells.append({"tag": tag, "_summary": f"{npass}/{ntot}"})
        print(f"[multi {tag}] {npass}/{ntot}", flush=True)
    total_p = sum(1 for c in cells if c.get("verdict") == "PASS")
    total_n = sum(1 for c in cells if "verdict" in c)
    out["multi"] = {"cells": cells, "pass": total_p, "total": total_n}
    out["bar_multi"] = f"{total_p}/{total_n}"
    print(f"[multi] overall {total_p}/{total_n}", flush=True)
    dump(out, args_out)
    return out


def phase_canary(w, out):
    rows = []
    for cid, q, want in mp.CANARY:
        flat, matched = gen(w, q, max_new=20)
        hit = tuple(matched)
        if want is None:
            ok = (hit == ())
            verd = "PASS" if ok else "FAIL"
        else:
            ok = (hit == (want,))
            verd = "PASS" if ok else "FAIL"
        rows.append({"id": cid, "q": q, "want": want, "matched": list(hit),
                     "verdict": verd, "answer": flat[:80]})
        print(f"[canary {cid:12s}] {verd} want={want} matched={list(hit)} "
              f"{flat[:50]!r}", flush=True)
    npass = sum(1 for r in rows if r["verdict"] == "PASS")
    out["canary"] = {"rows": rows, "pass": npass, "total": len(rows)}
    out["bar_canary"] = f"{npass}/{len(rows)}"
    print(f"[canary] {npass}/{len(rows)}", flush=True)
    dump(out, args_out)
    return out


def phase_live(w, out):
    """Non-memory turns of the shipped --probe transcript (leak check)."""
    rows = []
    messages = []
    for kind, user in c5.PROBES:
        messages.append({"role": "user", "content": user})
        flat, matched = gen(w, user, messages=messages[:-1], max_new=60)
        messages.append({"role": "assistant", "content": flat})
        persona_tags = tuple(mp.tags(w.pack))
        leaked = [t for t in matched if t in persona_tags]
        ok = (not leaked) if kind != "memory" else (not leaked)
        rows.append({"kind": kind, "q": user, "matched": list(matched),
                     "persona_leak": leaked,
                     "verdict": "PASS" if ok else "FAIL",
                     "answer": flat[:100]})
        print(f"[live {kind:9s}] {'PASS' if ok else 'FAIL'} "
              f"matched={list(matched)} {flat[:60]!r}", flush=True)
    npass = sum(1 for r in rows if r["verdict"] == "PASS")
    out["live"] = {"rows": rows, "pass": npass, "total": len(rows)}
    print(f"[live] clean {npass}/{len(rows)}", flush=True)
    dump(out, args_out)
    return out


args_out = None


def main() -> int:
    global args_out
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spine", choices=("phi", "qwen"), default="phi")
    ap.add_argument("--phase", choices=("single", "multi", "canary", "live",
                                        "all"), default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--kv", action="store_true",
                    help="decode via kv_cache_v05.KVDecoder (the shipped "
                         "chat_v06 path)")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-new", type=int, default=24)
    args = ap.parse_args()
    args_out = args.out or f"/data/tliao/persona_out/probe_{args.spine}.json"
    out = {"gate": "persona-probe", "spine": args.spine, "phase": args.phase,
           "pid": os.getpid(), "kv": args.kv}
    dump(out, args_out)
    t0 = time.time()
    w = build_world(args.spine, args.device, args.kv)
    out["live_plugs"] = [i["name"] for i in w.info]
    print(f"[world] plugs: {out['live_plugs']}", flush=True)
    if args.phase in ("single", "all"):
        phase_single(w, out)
    if args.phase in ("canary", "all"):
        phase_canary(w, out)
    if args.phase in ("live", "all"):
        phase_live(w, out)
    if args.phase in ("multi", "all"):
        phase_multi(w, out, max_new=max(16, args.max_new - 4))
    out["phase"] = "done"
    out["wall_s"] = round(time.time() - t0, 2)
    dump(out, args_out)
    print("[persona_probe] done", json.dumps(
        {k: out.get(k) for k in ("bar_single", "bar_multi", "bar_canary")}),
        flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
