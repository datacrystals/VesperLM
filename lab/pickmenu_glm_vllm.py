#!/usr/bin/env python3
"""pickmenu_glm_vllm — replay convo_test.DIALOGUES against an OpenAI-compatible
chat endpoint serving GLM-5.3-Flash-AWQ-INT4, score with the frozen rubric,
and stream every completed turn to local JSONL + optional box push.

Sibling of the MiMo arm sampler (box /data/tliao/mimo_flash_lab/pickmenu_sample.py):
same DIALOGUES verbatim, same decode triple, seed 0, no system prompt, and the
same artifact schema (spine/label/backend/decode/dialogues/rubric_tally/score).

The rubric below is a VERBATIM port of lab/convo_test.py's score_turn chain
(kw_hit / rubric_no_rep incl. chat_vfinal.loop_detect / rubric_answers /
rubric_length / rubric_persona / worst / STOP / DEBRIS / DIALOGUES), kept
self-contained so it runs on the laptop without the chat stack.

Anti-loss contract (fresh-droplet run):
  1. every finished turn -> append+fsync a JSONL line under --stream-dir
  2. every dialogue boundary -> rewrite state.json in --stream-dir and, when
     --box-push HOST:DIR is given, rsync the stream dir to the box (plain
     rsync, -e ssh; nothing is held only on the droplet)
  3. the aggregate --out JSON is written at the end and also pushed.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter

VER = "pickmenu_glm_vllm 1.0 (rubric verbatim from lab/convo_test.py @ 2026-10-19)"

DEC = {"temperature": 0.8, "top_p": 0.95, "repetition_penalty": 1.1,
       "max_tokens": 128}

# ---------------------------------------------------------------------------
# DIALOGUES — verbatim lab/convo_test.py (the frozen 10)
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


# ---------------------------------------------------------------------------
# rubric — verbatim convo_test port (loop_detect inlined from chat_vfinal)
# ---------------------------------------------------------------------------
def loop_detect(text: str):
    words = text.split()
    for ngram in (8, 6, 4):
        for i in range(0, max(0, len(words) - 2 * ngram + 1)):
            a, b = words[i:i + ngram], words[i + ngram:i + 2 * ngram]
            if a == b:
                return " ".join(a)[:60]
    return None


def word_ngrams(words, n):
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def kw_hit(kw: str, text: str) -> bool:
    return re.search(r"\b" + re.escape(kw.lower()), text.lower()) is not None


def any_kw(keys, text: str) -> bool:
    return any(kw_hit(k, text) for k in keys)


def rubric_no_rep(text: str) -> str:
    words = text.split()
    if loop_detect(text):
        return "FAIL"
    if len(words) >= 8:
        uniq = len(set(w.lower() for w in words)) / len(words)
        if uniq < 0.42:
            return "FAIL"
        for n, lim in ((3, 3), (2, 4)):
            grams = word_ngrams([w.lower() for w in words], n)
            if grams:
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


# ---------------------------------------------------------------------------
# reply hygiene (same policy as the MiMo arm sampler: content first, the
# stripped reasoning segment as fallback when the parser leaves content empty)
# ---------------------------------------------------------------------------
_THINK_PAIR = re.compile(r"<\|[^|>]{1,24}\|>.*?<\|[^|>]{1,24}\|>", re.DOTALL)
_TAG = re.compile(r"<\|[^|>]{1,24}\|>")
_CLOSE = "</think>"


def clean(t: str) -> str:
    t = _THINK_PAIR.sub(" ", t)
    t = _TAG.sub(" ", t)
    return " ".join(t.split()).strip()


def visible(msg: dict) -> str:
    c = (msg.get("content") or "").strip()
    if _CLOSE in c:
        c = c.split(_CLOSE, 1)[1].strip()
    if c:
        return clean(c)
    r = (msg.get("reasoning_content") or "")
    if _CLOSE in r:
        r = r.split(_CLOSE, 1)[1]
    return clean(r)


def one_turn(url, model, messages, seed, extra, timeout=300):
    body = {"model": model, "messages": messages, "seed": seed, **DEC}
    body.update(extra)
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        pl = json.load(r)
    msg = pl["choices"][0]["message"]
    fin = pl["choices"][0].get("finish_reason")
    return msg, fin, round(time.time() - t0, 1), pl.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="GLM-5.3-Flash")
    ap.add_argument("--spine", default="glm-flash")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap", type=float, default=0.25)
    ap.add_argument("--stream-dir", default="",
                    help="append per-turn JSONL + state.json here")
    ap.add_argument("--box-push", default="",
                    help="rsync stream dir to HOST:DIR at dialogue boundaries")
    ap.add_argument("--extra-json", default="",
                    help='extra request fields, e.g. '
                         '\'{"reasoning_effort":"low","chat_template_kwargs":'
                         '{"clear_thinking":true}}\'')
    ap.add_argument("--start-dialogue", type=int, default=0,
                    help="resume: skip completed dialogues (stream intact)")
    a = ap.parse_args()
    url = a.endpoint.rstrip("/") + "/v1/chat/completions"
    extra = json.loads(a.extra_json) if a.extra_json else {}

    results = {"spine": a.spine, "label": a.label, "backend": "vllm",
               "decode": dict(DEC), "seed": a.seed,
               "endpoint_note": a.endpoint, "sampler": VER,
               "extra_request": extra or None,
               "dialogues": [], "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime())}
    tally = {"no_rep": {}, "answers": {}, "length": {}, "persona": {}}
    stream = None
    if a.stream_dir:
        os.makedirs(a.stream_dir, exist_ok=True)
        stream = open(os.path.join(a.stream_dir, "turns.jsonl"), "a",
                      buffering=1)

    for di, dlg in enumerate(DIALOGUES):
        if di < a.start_dialogue:
            continue
        messages, turns, verdict = [], [], "PASS"
        for turn in dlg["turns"]:
            messages.append({"role": "user", "content": turn["u"]})
            msg, fin, dt, usage = one_turn(url, a.model, messages, a.seed,
                                           extra)
            reply = visible(msg)
            messages.append({"role": "assistant", "content": reply})
            scores = score_turn(reply, turn)
            for k, v in scores.items():
                if v != "N/A":
                    tally[k][v] = tally[k].get(v, 0) + 1
            tv = worst(*(v for v in scores.values() if v != "N/A"))
            verdict = worst(verdict, tv)
            rec = {"user": turn["u"], "assistant": reply,
                   "content_raw": msg.get("content"),
                   "reasoning_raw": msg.get("reasoning_content"),
                   "finish_reason": fin, "wall_s": dt,
                   "usage": usage, "scores": scores, "verdict": tv}
            turns.append(rec)
            if stream:
                stream.write(json.dumps(
                    {"dlg": dlg["id"], "dialogue_index": di,
                     "turn_index": len(turns), "turn": rec,
                     "ts": time.time(), "ver": VER},
                    ensure_ascii=False) + "\n")
                os.fsync(stream.fileno())
            print(f"  {dlg['id']:12s} t{len(turns)} {tv:7s} "
                  f"{dt:6.1f}s fin={fin} words={len(reply.split())}",
                  flush=True)
            time.sleep(a.gap)
        results["dialogues"].append({"id": dlg["id"], "verdict": verdict,
                                     "turns": turns})
        # dialogue boundary: refresh the resumable state + push off-droplet
        partial = {**results, "rubric_tally": tally,
                   "partial": True, "completed_dialogues": di + 1}
        if a.stream_dir:
            with open(os.path.join(a.stream_dir, "state.json"), "w") as f:
                json.dump(partial, f, indent=2, ensure_ascii=False)
            if a.box_push:
                host, _, path = a.box_push.partition(":")
                subprocess.run(
                    ["rsync", "-az", "--timeout=60",
                     os.path.join(a.stream_dir, ""),
                     f"{host}:{path}/"], check=False)

    results["rubric_tally"] = tally
    n_ok = sum(1 for d in results["dialogues"] if d["verdict"] == "PASS")
    n_pp = sum(1 for d in results["dialogues"]
               if d["verdict"] in ("PASS", "PARTIAL"))
    results["score"] = {"pass": n_ok, "pass_or_partial": n_pp,
                        "n": len(results["dialogues"])}
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    if a.box_push:
        host, _, path = a.box_push.partition(":")
        subprocess.run(["rsync", "-az", "--timeout=60", a.out,
                        f"{host}:{path}/"], check=False)
    print(f"SCORE pass {n_ok}/10 pass_or_partial {n_pp}/10", flush=True)
    print(f"WROTE {a.out}", flush=True)


if __name__ == "__main__":
    main()
