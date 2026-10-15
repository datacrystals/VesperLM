#!/usr/bin/env python3
"""convo_test — the conversation-quality harness for the vFINAL chat model.

The primary bar is "does it feel good to talk to?" — not benchmark numbers.
This runs 10 natural multi-turn dialogues (greetings, small talk, opinions,
tell-me-about-yourself, a story request, questions about the user's interests
from the persona pack) through the shipped chat stack and scores each reply on
simple observable rubrics:

    no_rep      no loops / phrase-runs / pathological token reuse
    answers     answers what was asked (keyword/number/overlap heuristics)
    length      length-sane for the turn type (not a wall of debris, not 1 word)
    persona     persona-aware where it matters (POI/Gundam/trains/...) —
                N/A on turns where it does not apply

Report policy: stdout is one-word / numeric verdicts ONLY (provider safety
filter). Full transcripts go to the --out JSON (for the user to read locally).

Arms (BEFORE/AFTER on the same dialogues, same seed):
    before = the old shipped default (greedy, no rep penalty)
    after  = the shipped default (T=0.8 top_p=0.95 rep_penalty=1.1)

Usage (MI50 box):
    /data/tliao/vesper_sft/venv/bin/python lab/convo_test.py \
        --arms before,after --out /data/tliao/vk_final/convo_test.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))

import chat_vfinal as cv  # noqa: E402

# ---------------------------------------------------------------------------
# the 10 dialogues (natural, multi-turn; each turn scored on 4 rubrics)
# expect_* keys drive the "answers" / "persona" heuristics
# ---------------------------------------------------------------------------
DIALOGUES = [
    {"id": "greet", "turns": [
        {"u": "Hey! How's it going?", "kind": "smalltalk"},
        {"u": "Good here. What have you been up to today?", "kind": "smalltalk"},
        {"u": "Nice. Got any plans for the weekend?", "kind": "smalltalk"},
    ]},
    {"id": "about_you", "turns": [
        {"u": "Tell me a bit about yourself.", "kind": "open",
         "persona": ["gundam", "trains", "person of interest",
                     "ibgt", "cars", "kepler", "helix"]},
        {"u": "What kind of stuff are you into?", "kind": "open",
         "persona": ["gundam", "trains", "person of interest", "anime",
                     "ibgt", "cars", "locomotive"]},
    ]},
    {"id": "opinion", "turns": [
        {"u": "Do you think video games count as art?", "kind": "opinion"},
        {"u": "Why do you say that?", "kind": "followup"},
    ]},
    {"id": "story", "turns": [
        {"u": "Tell me a very short story about a lighthouse.",
         "kind": "story", "words": [25, 160], "expect_any": ["light"]},
    ]},
    {"id": "poi", "turns": [
        {"u": "Have you seen Person of Interest?", "kind": "persona",
         "persona": ["person of interest", "finch", "reese", "shaw", "root",
                     "the machine"]},
        {"u": "What do you like about it?", "kind": "followup"},
    ]},
    {"id": "gundam", "turns": [
        {"u": "What's your favorite anime?", "kind": "persona",
         "persona": ["gundam", "mobile suit"]},
    ]},
    {"id": "trains", "turns": [
        {"u": "What do you think of trains?", "kind": "persona",
         "persona": ["train", "rail", "locomotive"]},
    ]},
    {"id": "math", "turns": [
        {"u": "What's 17 times 23?", "kind": "math", "expect_num": "391"},
        {"u": "Thanks! And what's 100 minus 37?", "kind": "math",
         "expect_num": "63"},
    ]},
    {"id": "knowledge", "turns": [
        {"u": "Why do we dream?", "kind": "knowledge"},
        {"u": "Interesting. What about lucid dreams?", "kind": "followup"},
    ]},
    {"id": "user_interest", "turns": [
        {"u": "I've been getting into model trains lately. Ever built one?",
         "kind": "persona",
         "persona": ["train", "model", "rail", "hobby", "built", "kit"]},
        {"u": "What would you start with if you were me?", "kind": "followup"},
    ]},
]

STOP = set("""a an the and or but if of to in on for with is are was were be been
am do does did what which who whom whose when where why how i you he she it we
they me my your his her its our their this that these those there here so as
at by from about into over under again just very can could would should will
shall may might must have has had than then too also any some no not yes really
please tell say said like want need make makes made get got going go up out
off lot bit much many more most""".split())

DEBRIS = ("x = ", "y = ", "```", "formula:", "precipitation", "co2", "h2o",
          "c6h12o6", "chlorophyll", "nacl", "h2so4", "equation:",
          "as an ai")


def word_ngrams(words, n):
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def kw_hit(kw: str, text: str) -> bool:
    """Word-boundary PREFIX match: 'train' hits 'trains'/'railroad'-style
    stems without matching inside unrelated words ('poi' in 'poison')."""
    return re.search(r"\b" + re.escape(kw.lower()), text.lower()) is not None


def any_kw(keys, text: str) -> bool:
    return any(kw_hit(k, text) for k in keys)


def rubric_no_rep(text: str) -> str:
    words = text.split()
    if cv.loop_detect(text):
        return "FAIL"
    if len(words) >= 8:
        uniq = len(set(w.lower() for w in words)) / len(words)
        if uniq < 0.42:
            return "FAIL"
        for n, lim in ((3, 3), (2, 4)):
            grams = word_ngrams([w.lower() for w in words], n)
            if grams:
                from collections import Counter
                top = Counter(grams).most_common(1)[0]
                if top[1] >= lim:
                    return "FAIL" if top[1] >= lim + 1 else "PARTIAL"
        if uniq < 0.58:
            return "PARTIAL"
    return "PASS"


def rubric_length(text: str, words_bounds=(3, 140)) -> str:
    n = len(text.split())
    lo, hi = words_bounds
    if n < lo or n > hi:
        return "FAIL"
    return "PASS"


def content_words(s):
    return [w for w in re.findall(r"[a-z0-9']+", s.lower())
            if w not in STOP and len(w) > 2]


def rubric_answers(text: str, user: str, turn: dict) -> str:
    t = text.lower()
    if any(d in t for d in DEBRIS) and turn.get("kind") in (
            "knowledge", "followup", "open", "smalltalk", "opinion"):
        return "FAIL"
    if "expect_num" in turn:
        return "PASS" if turn["expect_num"] in text.replace(",", "") else "FAIL"
    if turn.get("expect_any"):
        return "PASS" if any_kw(turn["expect_any"], text) else "FAIL"
    if turn.get("persona") and any_kw(turn["persona"], text):
        return "PASS"
    qw = set(content_words(user))
    rw = set(content_words(text))
    if not qw:
        return "PASS" if len(text.split()) >= 3 else "FAIL"
    overlap = len(qw & rw) / len(qw)
    if overlap >= 0.25:
        return "PASS"
    if overlap >= 0.10:
        return "PARTIAL"
    return "FAIL"


def rubric_persona(text: str, turn: dict) -> str:
    if not turn.get("persona"):
        return "N/A"
    return "PASS" if any_kw(turn["persona"], text) else "FAIL"


def score_turn(text: str, turn: dict) -> dict:
    bounds = tuple(turn.get("words", (3, 140)))
    return {
        "no_rep": rubric_no_rep(text),
        "answers": rubric_answers(text, turn["u"], turn),
        "length": rubric_length(text, bounds),
        "persona": rubric_persona(text, turn),
    }


def worst(*vals):
    order = {"FAIL": 0, "PARTIAL": 1, "PASS": 2, "N/A": 3}
    return min(vals, key=lambda v: order.get(v, 0))


ARMS = {
    # BEFORE = the old shipped default (greedy was the loop machine)
    "before": {"sample": False, "temperature": 1.0, "top_p": 1.0,
               "rep_penalty": 1.0},
    # AFTER = the shipped default (chat_vfinal DEFAULT_*)
    "after": {"sample": True, "temperature": cv.DEFAULT_SAMPLE,
              "top_p": cv.DEFAULT_TOP_P, "rep_penalty": cv.DEFAULT_REP_PENALTY},
}


def make_args(gpu, dtype="float16", bare=False):
    return argparse.Namespace(
        model=cv.DEFAULT_MODEL, registry=cv.DEFAULT_REGISTRY,
        plugins_dir=cv.DEFAULT_PLUGINS, out="", bare=bare,
        probe=False, impress=False, impress_bare=False,
        sample=cv.DEFAULT_SAMPLE, temperature=None,
        top_p=cv.DEFAULT_TOP_P, rep_penalty=cv.DEFAULT_REP_PENALTY,
        max_new_tokens=cv.DEFAULT_MAX_NEW, probe_tokens=64,
        eval_max_new=0, device="cuda", gpu=gpu, dtype=dtype, no_kv=False,
        default_lam=0.5, lam_map={}, varact="", mem_pack=cv.DEFAULT_MEM_PACK,
        mem_bonus=str(cv.DEFAULT_MEM_BONUS), mem_bonus_tau=0.5,
        mem_bonus_topk=0.0, qscope=cv.DEFAULT_QSCOPE, qtau=0.5, qmargin=0.0,
        slot_k=0, mem_boost=False)


def resolve_decode(args, arm: dict):
    """Overwrite args' decode triple with the arm's config (mutates args)."""
    args.sample = bool(arm["sample"])
    args.temperature = float(arm["temperature"])
    args.top_p = float(arm["top_p"])
    args.rep_penalty = float(arm["rep_penalty"])
    return args


def run_arm(st, arm_name: str, arm: dict, max_new: int, seed: int,
            show: bool) -> dict:
    args = st["args"]
    resolve_decode(args, arm)
    tok, device, dec = st["tok"], st["device"], st["dec"]
    fn = st["fn_graft"] if st["mode"] == "grafted" else st["fn_bare"]
    mode = st["mode"]
    print(f"\n=== CONVO ARM {arm_name} "
          f"(sample={args.sample} T={args.temperature} top_p={args.top_p} "
          f"rep={args.rep_penalty} max_new={max_new}) ===", flush=True)
    out = {"arm": arm_name, "decode": dict(arm), "dialogues": []}
    rubric_tally = {"no_rep": {}, "answers": {}, "length": {}, "persona": {}}
    for dlg in DIALOGUES:
        messages = []
        turns = []
        dlg_verdict = "PASS"
        for turn in dlg["turns"]:
            messages.append({"role": "user", "content": turn["u"]})
            text, ids, wall, tps = cv.gen_reply_stream(
                tok, fn, messages, max_new, device,
                sample=args.sample, temperature=args.temperature,
                top_p=args.top_p, rep_penalty=args.rep_penalty,
                dec=dec, mode=mode, scope=st["scope"], seed=seed)
            messages.append({"role": "assistant", "content": text})
            scores = score_turn(text, turn)
            for k, v in scores.items():
                if v != "N/A":
                    rubric_tally[k][v] = rubric_tally[k].get(v, 0) + 1
            tv = worst(*(v for v in scores.values() if v != "N/A"))
            dlg_verdict = worst(dlg_verdict, tv)
            turns.append({"user": turn["u"], "assistant": text,
                          "n_new": len(ids), "tok_s": round(tps, 2),
                          "scores": scores, "verdict": tv})
            if show:
                print(f"  you> {turn['u']}", flush=True)
                print(f"  bot> {' '.join(text.split())[:300]}", flush=True)
            print(f"  [{arm_name}] {dlg['id']:12s} t{len(turns)} {tv:7s} "
                  f"{' '.join(scores[k][0] for k in ('no_rep','answers','length','persona'))} "
                  f"n={len(ids)}", flush=True)
        out["dialogues"].append({"id": dlg["id"], "verdict": dlg_verdict,
                                 "turns": turns})
        print(f"  [{arm_name}] DIALOGUE {dlg['id']:12s} -> {dlg_verdict}",
              flush=True)
    out["rubric_tally"] = rubric_tally
    n_ok = sum(1 for d in out["dialogues"] if d["verdict"] == "PASS")
    n_pp = sum(1 for d in out["dialogues"] if d["verdict"] in ("PASS", "PARTIAL"))
    out["score"] = {"pass": n_ok, "pass_or_partial": n_pp,
                    "n": len(DIALOGUES)}
    print(f"\n[{arm_name}] dialogues: {n_pp}/10 usable (PASS {n_ok})", flush=True)
    for k, cnt in rubric_tally.items():
        print(f"[{arm_name}] {k:8s} "
              + " ".join(f"{v}={cnt.get(v, 0)}" for v in ("PASS", "PARTIAL", "FAIL")),
              flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="before,after",
                    help="comma list from {before,after,greedy,shipped}")
    ap.add_argument("--out", default="/data/tliao/vk_final/convo_test.json")
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--mode", default="grafted", choices=("grafted", "bare"))
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true",
                    help="also print replies (stdout is verdict-only by "
                         "default; transcripts always land in --out)")
    args = ap.parse_args()

    alias = {"greedy": "before", "shipped": "after"}
    arms = [alias.get(a.strip(), a.strip()) for a in args.arms.split(",")
            if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}")

    st_args = make_args(args.gpu, args.dtype, bare=False)
    st = cv.setup_chat(st_args)
    st["args"] = st_args
    st["mode"] = args.mode
    if dec_note := getattr(st["dec"], "note", None):
        print(f"[convo] kv: {dec_note}", flush=True)

    results = {"arms": {}, "max_new": args.max_new, "seed": args.seed,
               "mode": args.mode, "n_dialogues": len(DIALOGUES),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for arm_name in arms:
        results["arms"][arm_name] = run_arm(
            st, arm_name, ARMS[arm_name], args.max_new, args.seed, args.show)

    # one-word comparative verdict, before vs after where both ran
    if "before" in results["arms"] and "after" in results["arms"]:
        b = results["arms"]["before"]["score"]["pass_or_partial"]
        a = results["arms"]["after"]["score"]["pass_or_partial"]
        print(f"\nCOMPARE before={b}/10 after={a}/10 -> "
              + ("PASS" if a > b else ("TIE" if a == b else "FAIL")),
              flush=True)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[convo] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
