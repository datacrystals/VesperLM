#!/usr/bin/env python3
"""spine_bakeoff — the head-to-head conversation harness for the SPINE PICK.

The user's metric is feel, not benchmarks: run the SAME 10-turn conversation
harness on BOTH candidate spines (MiMo-V2.6-Flash and GLM-5.3-Flash) with
IDENTICAL plugs and IDENTICAL decode defaults, score every turn with the
lab/convo_test.py rubric (no-repetition / answers-what-asked / length-sane /
persona-aware), write both transcripts to JSON, and let the user pick.

    ONE COMMAND PER SPINE (identical except --spine):

      python3 lab/spine_bakeoff.py --spine mimo-flash \
          --out /data/tliao/bakeoff/mimo_flash.json
      python3 lab/spine_bakeoff.py --spine glm-flash \
          --out /data/tliao/bakeoff/glm_flash.json
      python3 lab/spine_bakeoff.py --compare \
          /data/tliao/bakeoff/mimo_flash.json /data/tliao/bakeoff/glm_flash.json

Protocol (fixed — do not vary between spines):
  1. prompts      lab/convo_test.DIALOGUES verbatim (10 natural multi-turn)
  2. rubric       lab/convo_test score_turn/worst verbatim (no_rep, answers,
                  length, persona — word-boundary persona keywords)
  3. decode       shipped defaults: sample T=0.8, top-p 0.95,
                  rep-penalty 1.1, max-new 128  (9038e07 — greedy was the
                  loop machine; /greedy stays opt-in for diagnostics only)
  4. plugs        the SAME persona pack set fitted per-spine with the SAME
                  §10.7 recipe (mem_trains et al.; see MODULAR_MOE 10.7) —
                  plugin dir passed with --plugins, one set per spine host
  5. seed         default 0 both arms (sampling is not bit-deterministic on
                  gfx906 — if a number looks odd, rerun with --seed 1)
  6. verdict      per-turn 4-rubric scores + per-dialogue worst + usable/10;
                  transcripts land in --out, stdout is one-word/numeric only

Backends (how a spine produces a reply):
  vllm   OpenAI-compatible endpoint (bare-spine feel check — NO plugs; use
         only as a pre-check, never for the pick itself)
  graft  the assembled Vesper stack on that spine's host adapter
         (spine + registry graft + the identical plug set) — the pick arm

Backends are selected with --backend; the graft backend needs the spine's
host module to expose make_generator(...) -> (tok, gen_fn); see
SPINE_MODULES below.  Until a spine's host lands, its graft arm exits
non-zero rather than silently comparing unequal stacks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))

import convo_test as ct  # noqa: E402  (DIALOGUES + rubric — the fixed bar)

# ---------------------------------------------------------------------------
# The fixed bake-off constants (identical for every spine)
# ---------------------------------------------------------------------------

BAKEOFF_DECODE = {
    "sample": True,
    "temperature": 0.8,      # shipped defaults, 9038e07
    "top_p": 0.95,
    "rep_penalty": 1.1,
    "max_new": 128,
}
BAKEOFF_SEED = 0

# plug set identity: the persona pack tags that must be present, per spine,
# fitted with the same §10.7 recipe (rank-64 width-norm, wd 0, lr 3e-3,
# task-CE + out_pen 300, FEED >= 200k, additive lam, registry row +
# owner-recal 0.85).  Anything else plugged invalidates the comparison.
REQUIRED_PLUGS = ("mem_trains", "mem_poi", "mem_gundam", "mem_machine",
                  "mem_cars", "mem_igbt")

# spine -> host module (lab/) exposing make_generator(plugins_dir, args)
SPINE_MODULES = {
    "mimo-flash": "mimo_full_host",
    "glm-flash": "glm_flash_gen",
}

SPINE_LABEL = {"mimo-flash": "MiMo-V2.6-Flash", "glm-flash": "GLM-5.3-Flash"}


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def make_vllm_generator(endpoint: str, model_name: str):
    """bare-spine OpenAI-compatible generator (no plugs — pre-check only)."""
    url = endpoint.rstrip("/") + "/v1/chat/completions"

    def gen(messages, max_new, dec, seed=0):
        body = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_new,
            "temperature": dec["temperature"] if dec["sample"] else 0.0,
            "top_p": dec["top_p"] if dec["sample"] else 1.0,
            "seed": seed,
        }
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            payload = json.load(r)
        return payload["choices"][0]["message"]["content"]

    return None, gen


def make_graft_generator(spine: str, plugins_dir: str, args):
    """Assembled stack on the spine's host adapter (the pick arm)."""
    mod_name = SPINE_MODULES.get(spine)
    if not mod_name:
        raise SystemExit(f"unknown spine {spine!r}")
    try:
        mod = __import__(mod_name)
    except ImportError as e:
        raise SystemExit(
            f"graft backend unavailable for {spine}: {e} — "
            f"host module lab/{mod_name}.py must expose make_generator()")
    if not hasattr(mod, "make_generator"):
        raise SystemExit(
            f"lab/{mod_name}.py has no make_generator() — cannot run the "
            f"pick arm for {spine} yet (report the rung you reached instead)")
    return mod.make_generator(plugins_dir, args)


# ---------------------------------------------------------------------------
# The harness (dialogues + rubric are convo_test's — nothing re-scored here)
# ---------------------------------------------------------------------------

def run(spine: str, backend: str, tok, gen, out_path: str, seed: int,
        show: bool) -> dict:
    dec = dict(BAKEOFF_DECODE)
    print(f"\n=== BAKEOFF {spine} backend={backend} "
          f"T={dec['temperature']} top_p={dec['top_p']} "
          f"rep={dec['rep_penalty']} max_new={dec['max_new']} seed={seed} ===",
          flush=True)
    results = {
        "spine": spine, "label": SPINE_LABEL.get(spine, spine),
        "backend": backend, "decode": dec, "seed": seed,
        "dialogues": [], "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
    }
    tally = {"no_rep": {}, "answers": {}, "length": {}, "persona": {}}
    for dlg in ct.DIALOGUES:
        messages = []
        turns = []
        dlg_verdict = "PASS"
        for turn in dlg["turns"]:
            messages.append({"role": "user", "content": turn["u"]})
            text = gen(messages, dec["max_new"], dec, seed=seed)
            messages.append({"role": "assistant", "content": text})
            scores = ct.score_turn(text, turn)
            for k, v in scores.items():
                if v != "N/A":
                    tally[k][v] = tally[k].get(v, 0) + 1
            tv = ct.worst(*(v for v in scores.values() if v != "N/A"))
            dlg_verdict = ct.worst(dlg_verdict, tv)
            turns.append({"user": turn["u"], "assistant": text,
                          "scores": scores, "verdict": tv})
            if show:
                print(f"  you> {turn['u']}", flush=True)
                print(f"  bot> {' '.join(text.split())[:300]}", flush=True)
            print(f"  {dlg['id']:12s} t{len(turns)} {tv:7s} "
                  + "".join(scores[k][0] for k in
                            ("no_rep", "answers", "length", "persona")),
                  flush=True)
        results["dialogues"].append({"id": dlg["id"], "verdict": dlg_verdict,
                                     "turns": turns})
        print(f"  DIALOGUE {dlg['id']:12s} -> {dlg_verdict}", flush=True)
    results["rubric_tally"] = tally
    n_ok = sum(1 for d in results["dialogues"] if d["verdict"] == "PASS")
    n_pp = sum(1 for d in results["dialogues"]
               if d["verdict"] in ("PASS", "PARTIAL"))
    results["score"] = {"pass": n_ok, "pass_or_partial": n_pp,
                        "n": len(ct.DIALOGUES)}
    print(f"\nSCORE usable {n_pp}/10 pass {n_ok}/10", flush=True)
    for k, cnt in tally.items():
        print(f"{k:8s} " + " ".join(
            f"{v}={cnt.get(v, 0)}" for v in ("PASS", "PARTIAL", "FAIL")),
            flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"WROTE {out_path}", flush=True)
    return results


AXES = ("no_rep", "answers", "length", "persona")


def compare(paths: list[str]) -> None:
    """One-command head-to-head readout (the user still picks by feel)."""
    runs = []
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        s = r.get("score", {})
        runs.append((r.get("label", r.get("spine", p)), p,
                     s.get("pass_or_partial", 0), s.get("pass", 0),
                     r.get("backend", "?"), r))
    print("=== BAKEOFF COMPARE (usable/10, pass/10) ===", flush=True)
    for label, p, pp, ok, backend, _r in runs:
        print(f"{label:22s} {pp}/10 {ok}/10 backend={backend} {p}",
              flush=True)
    # frozen rubric table: per-axis PASS/PARTIAL/FAIL tallies per arm,
    # plus per-dialogue worst verdicts — all straight from the harness JSONs
    # (rubric_tally written by run(); no re-scoring here).
    print("\n=== RUBRIC TABLE (turn-level tallies, N/A excluded) ===",
          flush=True)
    print(f"{'axis':8s} " + " ".join(
        f"{label:>28s}" for label, *_ in runs), flush=True)
    axes = AXES
    for ax in axes:
        cells = []
        for _label, _p, _pp, _ok, _b, r in runs:
            t = r.get("rubric_tally", {}).get(ax, {})
            cells.append(f"P={t.get('PASS', 0)} "
                         f"~={t.get('PARTIAL', 0)} "
                         f"F={t.get('FAIL', 0)}")
        print(f"{ax:8s} " + " ".join(f"{c:>28s}" for c in cells), flush=True)
    print("\n=== DIALOGUE WORST (per-dialogue verdict) ===", flush=True)
    dlg_ids = [d["id"] for d in runs[0][5].get("dialogues", [])]
    print(f"{'dialogue':14s} " + " ".join(
        f"{label:>22s}" for label, *_ in runs), flush=True)
    for did in dlg_ids:
        v = []
        for _label, _p, _pp, _ok, _b, r in runs:
            m = {d["id"]: d["verdict"] for d in r.get("dialogues", [])}
            v.append(m.get(did, "-"))
        print(f"{did:14s} " + " ".join(f"{x:>22s}" for x in v), flush=True)
    if len(runs) == 2:
        a, b = runs[0][2], runs[1][2]
        print("DELTA usable " + str(a - b), flush=True)
        print("PICK user-by-feel", flush=True)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spine", choices=sorted(SPINE_MODULES))
    ap.add_argument("--backend", choices=("graft", "vllm"), default="graft")
    ap.add_argument("--plugins", default="",
                    help="plug dir for THIS spine's host (identical set)")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--vllm-model", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=BAKEOFF_SEED)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar="JSON")
    args = ap.parse_args()

    if args.compare:
        compare(list(args.compare))
        return 0
    if not args.spine:
        raise SystemExit("--spine required (or --compare A.json B.json)")
    out = args.out or f"/data/tliao/bakeoff/{args.spine.replace('-', '_')}.json"

    if args.backend == "vllm":
        if not args.vllm_model:
            raise SystemExit("--vllm-model required for --backend vllm")
        tok, gen = make_vllm_generator(args.endpoint, args.vllm_model)
    else:
        tok, gen = make_graft_generator(args.spine, args.plugins, args)

    run(args.spine, args.backend, tok, gen, out, args.seed, args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
