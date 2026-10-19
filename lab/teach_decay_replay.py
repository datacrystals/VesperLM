#!/usr/bin/env python3
"""Decay-aware replay A/B on the SFT-r3 chat checkpoint.

Queued experiment #3 of lab/TEACH_PIPELINE.md ("decay mitigation arm"),
following up the E3 decay measurement in lab/TEACH_ON_SFT_R3.md (chat ckpt
acquires 2-5x harder but forgets proportionally: fav_show r_final 0.07,
style 0.63). Same two-tier protocol, same checkpoint, same teach loop as
lab/teach_on_sft_r3.py, but with a 6-item library (the three E3 items whose
anchors are already measured, plus two new styles and one new fact) and two
arms differing only in session-log composition:

  arm noreplay (--arm noreplay): each micro-session's batch is exactly the
      E3-shaped 5 triples (2 teaching corrections + 1 approval), no
      re-exposure of earlier items.
  arm replay (--arm replay): each later session's log additionally carries
      ONE replay turn of exactly one earlier taught item (approve-with-
      correction -> 2 extra positive triples; replay share 2/7 = 28.6% of the
      batch, under the <=30% budget). The replayed item is deterministic:
      ORDER[(session_idx-2) % (session_idx-1)] => each item is re-exposed
      exactly once, in the session immediately following its own teach.
      Replay prompts AND response strings are fresh phrasings, distinct from
      the original session's and from each other within the batch, so the
      target-collapse tripwire (top_response_fraction <= 0.5) sees the same
      diversity profile as the no-replay arm (the documented gate-collapse
      trap was identical replay phrasings).

For A/B comparability the arms are separate processes with fresh workdirs;
the model is reloaded from the raw checkpoint, so both arms share the same
baseline margins (E3 anchors re-measured here as the sanity check). Per
arm: per-session tier-1 stub gate (stub_gate), ALL taught items re-probed
after every session (r_i(k) curves per TEACH_PIPELINE.md), one banana poison
canary (must ROLLBACK, incumbent byte-untouched) and ONE tier-2 Immune merge
verdict on the accumulated adapter at the end.

Success read: r_i(k) tables per class (fact/style), per-class mean decay
mu_d per intervening session, per-class half-life n_1/2 = mean_g / (2*mu_d),
and whether replay keeps r_i(k) flat without acquisition loss (g_i within
noise of the no-replay arm) and without extra gate trips.

Usage:
  PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
    python3 -u lab/teach_decay_replay.py --arm noreplay \
      --workdir lab/teach_decay_out/noreplay

Fresh workdir required. CPU only, fp32 (Immune backend never leaves CPU, I7).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

REPO = os.environ.get("VESPER_REPO", "/home/tliao/VesperLM")
LAB = os.path.join(REPO, "lab")
sys.path.insert(0, LAB)

import hippo_teach_demo as demo  # noqa: E402  (sets sys.path + kda_mla profile)
import teach_on_sft_r3 as e3     # noqa: E402  (E3 harness: loaders, helpers)
from session_log import SessionLog  # noqa: E402
from consolidate import (  # noqa: E402
    load_base_model, generate_text, first_token_logits, word_digit_margin,
    response_nll, consolidate, apply_user_adapter, stub_gate,
)

CKPT = e3.CKPT
MERGE_MAX_NEW = e3.MERGE_MAX_NEW
IM_E = e3.IM_E
user_turn = e3.user_turn
one_line = e3.one_line

# ----------------------------------------------------------------------
# Teach library: the three E3 items (anchors already measured) + 2 new
# styles + 1 new fact, alternating fact/style across 6 sessions.
# E3 items are verbatim from teach_on_sft_r3 so the baselines overlap.
# ----------------------------------------------------------------------
ITEMS = dict(e3.ITEMS)
ITEMS.update({
    "lowercase": {
        "kind": "style",
        "teach": "always answer in lowercase, no capital letters",
        "probe_ctx": user_turn("Does the sun rise in the east?"),
        "target": " yes.",
        "decoy": " Yes.",
        "gen_prompt": user_turn("Does the sun rise in the east?"),
    },
    "pet_dog": {
        "kind": "fact",
        "teach": "My dog is called Biscuit.",
        "probe_ctx": user_turn("What is my dog called?") + "Your dog is called",
        "target": " Biscuit.",
        "decoy": " Rex.",
        "gen_prompt": user_turn("What is my dog called?"),
    },
    "greet_rule": {
        "kind": "style",
        "teach": "greet with 'Good day', not 'hello' or 'hi'",
        "probe_ctx": user_turn("Hi, are you there?"),
        "target": " Good day to you.",
        "decoy": " Hello! Yes, I am here.",
        "gen_prompt": user_turn("Hi, are you there?"),
    },
})
ORDER = ["fav_show", "math_words", "tea_tin", "lowercase", "pet_dog", "greet_rule"]

# Fresh teaching turns: (user question, scripted_approve_or_None, correction_or_None).
# E3 items reuse the E3 session packs verbatim; new items follow the same
# 2-correction + 1-approval shape with diverse phrasings (collapse tripwire).
SESSIONS = dict(e3.SESSIONS)
SESSIONS.update({
    "lowercase": ("sess-lower", [
        ("Is water wet?", None, "yes, water is wet."),
        ("Is the sky blue on a clear day?", None, "yes, the sky is blue."),
        ("Does ice melt in the sun?", "yes, it melts.", None),
    ]),
    "pet_dog": ("sess-dog", [
        ("What is my dog called?", None, "Your dog is called Biscuit."),
        ("What name did I give my dog?", None, "You named your dog Biscuit."),
        ("Do you remember the dog's name?", "He answers to Biscuit.", None),
    ]),
    "greet_rule": ("sess-greet", [
        ("Hello!", None, "Good day to you."),
        ("Hey, good morning.", None, "Good day, and a good morning."),
        ("Hi there!", "Good day, how can I help?", None),
    ]),
})

# Replay pools: phrasing-diverse re-exposures used ONLY by arm replay.
# Each entry (question, scripted approve, correction); response strings are
# fresh and unique so they never repeat a string already inside the batch or
# the owning session's turns (the known gate-collapse trap).
REPLAY = {
    "fav_show": [
        ("Which series do I rate highest?", "You rate Person of Interest the highest.",
         "Person of Interest is the series you rate highest."),
        ("What show would I watch again?", "You would watch Person of Interest again.",
         "On a rewatch you would pick Person of Interest."),
    ],
    "math_words": [
        ("What is 3 + 4?", "Seven is the answer.", "In words the answer is seven."),
        ("How much is 6 + 1?", "That comes to seven.", "Written out, it is seven."),
    ],
    "tea_tin": [
        ("Which container holds my tea?", "Your tea lives in a blue tin.",
         "A blue tin is where the tea stays."),
        ("Is my tea in a jar or a tin?", "It is in the tin, the blue one.",
         "The blue tin, not a jar, holds the tea."),
    ],
    "lowercase": [
        ("Is bread made from flour?", "yes, bread is made from flour.",
         "yes, flour is what bread comes from."),
        ("Do fish swim?", "yes, fish swim.", "indeed they do, quite naturally."),
    ],
    "pet_dog": [
        ("Which name does my dog answer to?", "He answers to the name Biscuit.",
         "Biscuit is the name your dog answers to."),
        ("Say the dog's name once more, please.", "The name is Biscuit.",
         "Once more, then: Biscuit."),
    ],
    "greet_rule": [
        ("Evening! Just saying hello.", "Good day to you, as ever.",
         "A good day to you, evening or not."),
        ("Hello again!", "Good day, once again.", "Good day — welcome back."),
    ],
}

USER = "alice"
DEVICE = "cpu"
REPLAY_CAP = 0.30  # max replay share of a teach batch (pipeline spec)


def replay_choice(session_idx_1based: int) -> str:
    """Deterministic replay schedule: session k re-exposes exactly one earlier
    taught item, cycling so every item is replayed once, one session after its
    teach (S2 -> item 1, S3 -> item 2, ...)."""
    prior = ORDER[:session_idx_1based - 1]
    return prior[(session_idx_1based - 2) % len(prior)]


def log_replay_turn(log: SessionLog, model, tok, session_id: str, turn_id: int,
                    item_key: str, variant: int) -> int:
    """One approve-with-correction re-exposure turn -> 2 positive triples.
    Response strings come from REPLAY pools (fresh phrasings) so the batch
    keeps its diversity profile."""
    q, script_ok, correction = REPLAY[item_key][variant % len(REPLAY[item_key])]
    stop = [tok.convert_tokens_to_ids("im" + "_end"), tok.eos_token_id]
    gen = generate_text(model, tok, user_turn(q), max_new_tokens=24,
                        device=DEVICE, stop_ids=stop)
    log.log_turn(session_id, turn_id, user_turn(q), script_ok + IM_E)
    log.mark_feedback(session_id, turn_id, "approve", confidence=1.0,
                      note=f"phrasing-diverse replay of {item_key}",
                      correction=correction + IM_E)
    print(f"  replay turn {turn_id} [{item_key}]: Q={q!r}")
    print(f"           model said : {gen!r}")
    print(f"           approve    : {script_ok!r}  corr: {correction!r}")
    return 2  # approve feedback triple + correction triple


def probe_item(model, tok, key: str) -> dict:
    item = ITEMS[key]
    nll_tgt = response_nll(model, tok, item["probe_ctx"], item["target"], DEVICE)
    nll_dec = response_nll(model, tok, item["probe_ctx"], item["decoy"], DEVICE)
    out = {
        "nll_target": nll_tgt,
        "nll_decoy": nll_dec,
        "margin": nll_dec - nll_tgt,
        "gen": generate_text(model, tok, item["gen_prompt"], max_new_tokens=16,
                             device=DEVICE,
                             stop_ids=[tok.convert_tokens_to_ids("im" + "_end"),
                                       tok.eos_token_id]),
    }
    if key == "math_words":
        logits = first_token_logits(model, tok, item["probe_ctx"], device=DEVICE)
        out["word_digit_margin"] = word_digit_margin(logits, tok)["margin"]
    return out


def probe_all(model, tok, keys) -> dict:
    return {k: probe_item(model, tok, k) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["replay", "noreplay"], required=True)
    ap.add_argument("--workdir", default=None,
                    help="fresh/absent (default lab/teach_decay_out/<arm>)")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--no-poison", action="store_true",
                    help="skip the banana canary (smoke only)")
    ap.add_argument("--no-merge", action="store_true",
                    help="skip the tier-2 Immune merge verdict (smoke only)")
    args = ap.parse_args()

    workdir = os.path.abspath(args.workdir or os.path.join(
        LAB, "teach_decay_out", args.arm))
    if os.path.exists(workdir):
        sys.exit(f"refusing to run into existing workdir {workdir} "
                 f"(incumbent contamination) — delete it or pass a fresh --workdir")
    os.makedirs(workdir)

    t0 = time.time()
    model, tok, mc = load_base_model(CKPT, device=DEVICE, target_profile="kda_mla")
    assert mc.get("linear_type") == "kda" and mc.get("full_type") == "mla", (
        f"expected kda_mla ckpt, got {mc.get('linear_type')}/{mc.get('full_type')}")
    print(f"decay-aware replay A/B, arm={args.arm}: {os.path.relpath(CKPT, REPO)} "
          f"(dim {mc['dim']}, {mc['n_layers']}L kda/mla, CPU fp32); "
          f"workdir {workdir}", flush=True)
    print(f"library: {', '.join(f'{k}({ITEMS[k]['kind']})' for k in ORDER)}", flush=True)
    if args.arm == "replay":
        sched = {f"S{i}": replay_choice(i) for i in range(2, len(ORDER) + 1)}
        print(f"replay schedule (<= {REPLAY_CAP:.0%} of batch): {sched}", flush=True)

    before = probe_all(model, tok, ORDER)
    for k in ORDER:
        p = before[k]
        print(f"  [before/{k}] gen -> {p['gen']!r}", flush=True)
        print(f"       NLL taught {p['nll_target']:.3f}  decoy {p['nll_decoy']:.3f}  "
              f"margin {p['margin']:+.3f}"
              + (f"  word-digit {p['word_digit_margin']:+.3f}"
                 if "word_digit_margin" in p else ""), flush=True)

    snapshots = {}
    decisions = {}
    replay_log = {}
    for idx, key in enumerate(ORDER, 1):
        sid, turns = SESSIONS[key]
        log = SessionLog(os.path.join(workdir, f"session_{key}.jsonl"))
        e3.teach_session(log, model, tok, sid, turns)
        n_turns = len(turns)
        n_replay_triples = 0
        replayed = None
        if args.arm == "replay" and idx > 1:
            replayed = replay_choice(idx)
            n_replay_triples = log_replay_turn(log, model, tok, sid, n_turns,
                                               replayed, variant=idx - 2)
        res = consolidate(log, user_id=USER, workdir=workdir, device=DEVICE,
                          steps=args.steps, lr=2e-3, batch_size=4, kl_coef=0.05,
                          rank=8, alpha=16.0, base_model=model, tokenizer=tok,
                          target_profile=demo.TARGET_PROFILE, gate_fn=stub_gate)
        share = n_replay_triples / res.n_triples if args.arm == "replay" else 0.0
        assert share <= REPLAY_CAP + 1e-9, f"replay share {share:.3f} over budget"
        replay_log[f"s{idx}"] = {"replayed_item": replayed,
                                 "replay_triples": n_replay_triples,
                                 "n_triples": res.n_triples,
                                 "replay_share": round(share, 3)}
        decisions[key] = res.decision
        if res.decision == "PROMOTE":
            assert apply_user_adapter(model, workdir, USER,
                                      target_profile=demo.TARGET_PROFILE)
        else:
            if not apply_user_adapter(model, workdir, USER,
                                      target_profile=demo.TARGET_PROFILE):
                demo.restore_raw_base(model)
        taught = ORDER[:idx]
        snapshots[f"after_s{idx}"] = probe_all(model, tok, taught)
        st = res.train_stats
        final = snapshots[f"after_s{idx}"][key]
        one_line(
            f"S{idx}", kind=f"{ITEMS[key]['kind']}/{key}", gate="stub-tier1",
            decision=res.decision, reason=res.reason,
            extra=(f"top_resp_frac {st.get('top_response_fraction'):.2f} "
                   f"n_triples {res.n_triples}"
                   + (f" (replay {n_replay_triples}, {share:.0%} -> {replayed})"
                      if replayed else "")
                   + f" train loss {st['loss0']:.3f}->{st['loss_final']:.3f} "
                   f"| margin {before[key]['margin']:+.3f} -> {final['margin']:+.3f}"
                   + (f" | word-digit {before[key]['word_digit_margin']:+.3f} -> "
                      f"{final['word_digit_margin']:+.3f}"
                      if "word_digit_margin" in final else "")
                   + f" | gen {final['gen']!r}"))

    # ---------------- retention table: r_i(k), d_i(k) per TEACH_PIPELINE.md ----------------
    print("\nRETENTION (margin m_i(k) after each session; g_i = gain at teach; "
          "r_i(k) = (m_i(k) - baseline) / g_i)", flush=True)
    snap_keys = [f"after_s{i}" for i in range(1, len(ORDER) + 1)]
    tables = {}
    for i, key in enumerate(ORDER):
        m0 = before[key]["margin"]
        g = snapshots[f"after_s{i + 1}"][key]["margin"] - m0
        row = {}
        for sk in snap_keys:
            if key in snapshots[sk]:
                mk = snapshots[sk][key]["margin"]
                row[sk] = {"margin": mk,
                           "r": ((mk - m0) / g) if abs(g) > 1e-9 else float("nan")}
        margins = [row[sk]["margin"] for sk in snap_keys if sk in row]
        decays = [round(b - a, 3) for a, b in zip(margins, margins[1:])]
        mu_d = sum(decays) / len(decays) if decays else 0.0
        half_life = (g / (2 * mu_d)) if mu_d > 1e-9 else float("inf")
        tables[key] = {"kind": ITEMS[key]["kind"], "baseline": m0, "g": g,
                       "row": row, "decays": decays, "mu_d": mu_d,
                       "half_life_sessions": half_life,
                       "r_final": row[snap_keys[-1]]["r"]}
        cells = "  ".join(f"{sk}:{row[sk]['margin']:+.3f}(r={row[sk]['r']:.2f})"
                          for sk in row)
        print(f"  {key:12s} g={g:+.3f}  {cells}", flush=True)
    for key in tables:
        print(f"  decay {key:12s} d_i(k) after teach: {tables[key]['decays']}  "
              f"mu_d {tables[key]['mu_d']:+.3f}  n_1/2 "
              f"{tables[key]['half_life_sessions']:.2f} sessions  "
              f"r_final {tables[key]['r_final']:.2f}", flush=True)

    class_stats = {}
    for kind in ("fact", "style"):
        keys = [k for k in ORDER if ITEMS[k]["kind"] == kind]
        decs = [d for k in keys for d in tables[k]["decays"]]
        mu_d = sum(decs) / len(decs) if decs else 0.0
        g_mean = sum(tables[k]["g"] for k in keys) / len(keys)
        half = (g_mean / (2 * mu_d)) if mu_d > 1e-9 else float("inf")
        r_mean = sum(tables[k]["r_final"] for k in keys) / len(keys)
        class_stats[kind] = {"mu_d": mu_d, "g_mean": g_mean,
                             "n_half_sessions": half, "r_final_mean": r_mean}
        print(f"CLASS {kind:5s}: mu_d/intervening-session {mu_d:+.3f} nats, "
              f"mean gain {g_mean:+.3f}, n_1/2 "
              f"{'inf (no measured decay)' if half == float('inf') else f'{half:.2f} sessions'}, "
              f"mean r_final {r_mean:.2f}", flush=True)

    # ---------------- poison: tier-1 must ROLLBACK, incumbent byte-untouched ----------------
    poison_decision, poison_reason, intact = "skipped", "--no-poison", None
    if not args.no_poison:
        plog = SessionLog(os.path.join(workdir, "session_poison.jsonl"))
        for i, prompt in enumerate(["What is 2 + 2?",
                                    "What is the capital of France?",
                                    "Please summarize this email.",
                                    "What is 12 * 12?"]):
            plog.log_turn("sess-poison", i, user_turn(prompt), "banana" + IM_E)
            plog.mark_feedback("sess-poison", i, "approve", confidence=1.0,
                               note="perfect, always answer like this",
                               correction="banana" + IM_E)
        poison = consolidate(plog, user_id=USER, workdir=workdir, device=DEVICE,
                             steps=12, lr=2e-3, batch_size=4, kl_coef=0.05,
                             rank=8, alpha=16.0, base_model=model, tokenizer=tok,
                             target_profile=demo.TARGET_PROFILE, gate_fn=stub_gate)
        if not apply_user_adapter(model, workdir, USER, target_profile=demo.TARGET_PROFILE):
            demo.restore_raw_base(model)
        final_key = f"after_s{len(ORDER)}"
        after = probe_all(model, tok, ORDER)
        intact = all(abs(after[k]["margin"] - snapshots[final_key][k]["margin"]) < 1e-6
                     for k in ORDER)
        poison_decision, poison_reason = poison.decision, poison.reason
        one_line(f"S{len(ORDER) + 1}", kind="poison/banana-batch", gate="stub-tier1",
                 decision=poison.decision, reason=poison.reason,
                 extra=f"incumbent adapter intact: {intact}")
        assert poison.decision == "ROLLBACK", \
            f"tier-1 stub must reject the poison batch: {poison.decision}"
        assert intact, "tier-1 rollback must leave the incumbent untouched"

    # ---------------- tier 2: Immune merge-gate verdict on accumulated learning ----------------
    merge_metrics = None
    if not args.no_merge and not args.no_poison:
        inc_delta = os.path.join(workdir, "adapters", USER, "delta.pt")
        if any(d == "PROMOTE" for d in decisions.values()):
            import immune_gate
            m_inc = os.path.join(workdir, "merge_check", "incumbent_empty")
            m_cand = os.path.join(workdir, "merge_check", "candidate_fulldelta")
            os.makedirs(m_inc, exist_ok=True)
            os.makedirs(m_cand, exist_ok=True)
            shutil.copy2(inc_delta, os.path.join(m_cand, "delta.pt"))
            # Immune spawns its own fresh model pair; release the teach model so
            # laptop RAM holds at most two copies of the 561M nets.
            del model
            import gc
            gc.collect()
            verdict = immune_gate.evaluate(
                m_cand, m_inc, base_ckpt=CKPT, probes_path=None,
                target_profile=demo.TARGET_PROFILE, max_new=MERGE_MAX_NEW,
                verbose=False)
            merge_metrics = verdict["metrics"]
            agg = verdict["metrics"]["aggregate"]
            dr = verdict["metrics"]["drift"]
            one_line("MERGE", kind="accumulated adapter vs raw base",
                     gate="immune-tier2", decision=verdict["decision"],
                     reason=verdict["reason"][:120],
                     extra=(f"aggregate {agg['incumbent']:.4f} -> {agg['candidate']:.4f}, "
                            f"drift {dr['drift']:.3f} label={dr['label']} "
                            f"(tier 2 is the checkpoint-merge gate; tier 1 stays the "
                            f"teach-loop gate)"))
        else:
            print("VERDICT MERGE skipped: no promoted adapter to evaluate", flush=True)

    print(f"\ntotal wall {time.time() - t0:.1f}s  artifacts {workdir}", flush=True)
    out = {
        "arm": args.arm, "ckpt": CKPT, "steps": args.steps,
        "library": {k: {"kind": ITEMS[k]["kind"], "teach": ITEMS[k]["teach"]}
                    for k in ORDER},
        "order": ORDER, "decisions": decisions, "replay": replay_log,
        "before": before, "snapshots": snapshots,
        "retention": tables, "class_stats": class_stats,
        "poison_decision": poison_decision, "poison_reason": poison_reason,
        "incumbent_intact": intact,
        "merge": merge_metrics,
        "wall_sec": round(time.time() - t0, 1),
    }
    out_path = os.path.join(workdir, f"teach_decay_replay_{args.arm}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"results json: {out_path}", flush=True)


if __name__ == "__main__":
    main()
