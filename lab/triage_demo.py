#!/usr/bin/env python3
"""triage_demo — the user-facing 5-turn demo battery, verdict-only stdout.

One-word verdicts only (provider safety filter); full transcripts go to --out
JSON.  The battery (the demo that embarrassed the user):

    1. fresh  "hi how are you"                        (greeting bar: must PASS)
    2. fresh  "hi there, whats your favorite anime show?"  (anime/gundam bar)
    3-5. one 3-turn harness session (probe3_in.txt):
         "yo, what have you been up to today?"
         "recommend me a show to watch tonight"
         "what is 17 times 23?"

Config arms (the plugs A/B/C question):
    --bare                      bare spine (no plugs)
    --lam-map mimo_writing=0,...  grafted with per-plug lam overrides
    (default)                   grafted as-is (joint_state_vfinal lam table)

Also --demo5: the five turns as ONE accumulating session (the live-demo
order: greeting first, then anime, then the 3 harness turns).

Usage (MI50 box, claim-locked):
    /data/tliao/vk_final/claim_run.sh triage GPU -- \
      /data/tliao/vesper_sft/venv/bin/python lab/triage_demo.py \
        --label asis --out /data/tliao/vk_final/triage_asis.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))

import torch  # noqa: E402

import chat_vfinal as cv  # noqa: E402

USER_TURNS = [
    ("greet", "hi how are you"),
    ("anime", "hi there, whats your favorite anime show?"),
]
HARNESS_TURNS = [
    ("up_to", "yo, what have you been up to today?"),
    ("rec_show", "recommend me a show to watch tonight"),
    ("math", "what is 17 times 23?"),
]


def run_turns(st, turns, max_new, sample, temperature, top_p, rep_penalty,
              rerank=0, gentle=False, short_max_new=64, short_temp=0.7,
              samples=0):
    """Generate replies for (tag, user_text) turns in one session.

    samples>0: support-probe — emit the verdict of EVERY sampled candidate
    (seeds 0..samples-1) instead of one reply; measures how often the
    bimodal spine has a clean mode for the turn."""
    tok, device, dec = st["tok"], st["device"], st["dec"]
    fn = st["fn_bare"] if st.get("mode") == "bare" else st["fn_graft"]
    scope = st["scope"]
    if dec is not None:
        dec.reset()
    messages, out = [], []
    for tag, user in turns:
        messages.append({"role": "user", "content": user})
        qn, temp_n = max_new, temperature
        if gentle and cv.is_greeting(user):
            qn = min(qn, short_max_new)
            temp_n = min(temp_n, short_temp)
        if samples > 0:
            vs = []
            texts = []
            rep = ""
            for i in range(samples):
                if dec is not None:
                    dec.reset()
                text, ids, wall, tps = cv.gen_reply_stream(
                    tok, fn, messages, qn, device,
                    sample=sample, temperature=temp_n, top_p=top_p,
                    seed=i * 17, dec=dec, mode=st.get("mode", "grafted"),
                    scope=scope, rep_penalty=rep_penalty)
                v = cv.auto_verdict(text, user)
                vs.append(v)
                texts.append(text)
                if not rep and v == "PASS":
                    rep = text
            letter = {"PASS": "P", "PARTIAL": "p", "FAIL": "F"}
            scope_l = ",".join(getattr(scope, "last_match", ()) or ()) or "-"
            live_l = ",".join(getattr(st.get("bonus"), "last_live", ()) or ()) or "-"
            print(f"[{tag}] samples={''.join(letter.get(v, '?') for v in vs)} "
                  f"({','.join(vs)}) scope={scope_l} live={live_l}",
                  flush=True)
            best_v = ("PASS" if "PASS" in vs else
                      "PARTIAL" if "PARTIAL" in vs else "FAIL")
            out.append({"tag": tag, "user": user,
                        "sample_verdicts": vs, "sample_texts": texts,
                        "n_samples": samples,
                        "verdict": best_v, "loop": None, "n": 0,
                        "reply": rep or "(probe only)"})
            messages.append({"role": "assistant",
                             "content": rep or "(probe only)"})
            continue
        t0 = time.time()
        text, ids, wall, tps = cv.gen_reply_rerank(
            tok, fn, messages, qn, device,
            sample=sample, temperature=temp_n, top_p=top_p, seed=0,
            dec=dec, mode=st.get("mode", "grafted"), scope=scope,
            rep_penalty=rep_penalty, rerank=rerank)
        v = cv.auto_verdict(text, user)
        loop = cv.loop_detect(text)
        # context hygiene: a FAIL reply is corpus-drift poison for later
        # turns (measured: demo5 session 0/5 vs fresh battery 1/5) — keep
        # only a short trace in the transcript the model sees.
        ctx_text = text if v != "FAIL" else " ".join(text.split()[:12])
        messages.append({"role": "assistant", "content": ctx_text})
        scope_l = ",".join(getattr(scope, "last_match", ()) or ()) or "-"
        live_l = ",".join(getattr(st.get("bonus"), "last_live", ()) or ()) or "-"
        out.append({"tag": tag, "user": user, "reply": text,
                    "verdict": v, "loop": loop, "n": len(ids),
                    "scope": scope_l, "live": live_l,
                    "wall_s": round(time.time() - t0, 1)})
        print(f"[{tag}] {v}" + (f" LOOP" if loop else "")
              + f" scope={scope_l} live={live_l}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="cfg")
    ap.add_argument("--out", default="")
    ap.add_argument("--bare", action="store_true")
    ap.add_argument("--lam-map", default="")
    ap.add_argument("--sample", type=float, default=cv.DEFAULT_SAMPLE)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=cv.DEFAULT_TOP_P)
    ap.add_argument("--rep-penalty", type=float, default=cv.DEFAULT_REP_PENALTY)
    ap.add_argument("--max-new-tokens", type=int, default=cv.DEFAULT_MAX_NEW)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--demo5", action="store_true",
                    help="run the five turns as one accumulating session")
    ap.add_argument("--fresh", action="store_true",
                    help="also run each harness turn fresh (single-turn)")
    ap.add_argument("--model", default=cv.DEFAULT_MODEL)
    ap.add_argument("--registry", default=cv.DEFAULT_REGISTRY)
    ap.add_argument("--plugins-dir", default=cv.DEFAULT_PLUGINS)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--no-kv", action="store_true")
    ap.add_argument("--default-lam", type=float, default=0.5)
    ap.add_argument("--varact", default="")
    ap.add_argument("--mem-pack", default=cv.DEFAULT_MEM_PACK)
    ap.add_argument("--mem-bonus", default=str(cv.DEFAULT_MEM_BONUS))
    ap.add_argument("--mem-bonus-tau", type=float, default=0.5)
    ap.add_argument("--mem-bonus-topk", type=float, default=0.0)
    ap.add_argument("--qscope", default=cv.DEFAULT_QSCOPE)
    ap.add_argument("--qtau", type=float, default=0.5)
    ap.add_argument("--qmargin", type=float, default=0.0)
    ap.add_argument("--slot-k", type=int, default=0)
    ap.add_argument("--mem-boost", action="store_true")
    ap.add_argument("--rerank", type=int, default=0)
    ap.add_argument("--gentle-short", action="store_true")
    ap.add_argument("--short-max-new", type=int, default=64)
    ap.add_argument("--short-temp", type=float, default=0.7)
    ap.add_argument("--samples", type=int, default=0,
                    help="support-probe: verdict of every seeded candidate")
    ap.add_argument("--only", default="",
                    help="comma list of tags to run (greet,anime,up_to,"
                         "rec_show,math)")
    ap.add_argument("--no-block-openers", action="store_true",
                    help="disable the degenerate-opener blocklist (A/B arm)")
    args = ap.parse_args()
    if args.no_block_openers:
        cv.g0.BLOCK_OPENERS_ON = False

    args.lam_map = cv.parse_lam_map(args.lam_map)
    args.probe = False
    args.impress = False
    args.eval12 = False
    args.eval_max_new = int(args.max_new_tokens)
    args.probe_tokens = 64
    args.device = "cuda"
    args.bare = bool(args.bare)
    args.mem_bonus_tau = float(args.mem_bonus_tau)
    # setup_chat resolves --sample the same way main() does
    args.temperature = (float(args.sample) if args.temperature is None
                        else float(args.temperature))
    args.sample = float(args.sample) > 0

    st = cv.setup_chat(args)
    st["mode"] = "bare" if args.bare else "grafted"
    rt = dict(rerank=args.rerank, gentle=bool(args.gentle_short),
              short_max_new=args.short_max_new, short_temp=args.short_temp,
              samples=args.samples)

    results = {"label": args.label, "bare": bool(args.bare),
               "lam_map": args.lam_map,
               "decode": {"sample": args.sample, "temperature": args.temperature,
                          "top_p": args.top_p, "rep_penalty": args.rep_penalty,
                          "max_new": args.max_new_tokens,
                          "rerank": args.rerank,
                          "gentle_short": bool(args.gentle_short),
                          "samples": args.samples},
               "lam_table": {i["name"]: i["lam"] for i in st.get("info", [])},
               "battery": []}

    only = {s.strip() for s in args.only.split(",") if s.strip()}

    def pick(turns):
        return [t for t in turns if not only or t[0] in only]

    if args.demo5:
        print("=== DEMO5 (one session) ===", flush=True)
        results["battery"] = run_turns(
            st, pick([("greet", "hi how are you"),
                      ("anime", "hi there, whats your favorite anime show?")]
                     + HARNESS_TURNS),
            args.max_new_tokens, args.sample, args.temperature,
            args.top_p, args.rep_penalty, **rt)
    else:
        for tag, user in pick(USER_TURNS):
            print(f"=== FRESH {tag} ===", flush=True)
            results["battery"] += run_turns(
                st, [(tag, user)], args.max_new_tokens, args.sample,
                args.temperature, args.top_p, args.rep_penalty, **rt)
        if pick(HARNESS_TURNS):
            print("=== HARNESS 3-TURN SESSION ===", flush=True)
            results["battery"] += run_turns(
                st, pick(HARNESS_TURNS), args.max_new_tokens, args.sample,
                args.temperature, args.top_p, args.rep_penalty, **rt)
        if args.fresh:
            for tag, user in pick(HARNESS_TURNS):
                print(f"=== FRESH {tag} ===", flush=True)
                results["battery"] += run_turns(
                    st, [(tag, user)], args.max_new_tokens, args.sample,
                    args.temperature, args.top_p, args.rep_penalty, **rt)

    n_pass = sum(1 for r in results["battery"] if r["verdict"] == "PASS")
    n_ok = sum(1 for r in results["battery"]
               if r["verdict"] in ("PASS", "PARTIAL"))
    results["score"] = {"pass": n_pass, "usable": n_ok,
                        "n": len(results["battery"])}
    print(f"=== SCORE usable={n_ok}/{len(results['battery'])} "
          f"pass={n_pass} label={args.label} ===", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[triage] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
