#!/usr/bin/env python3
"""Reactive (trigger-based) consolidation arm on the SFT-r3 chat checkpoint.

Follow-up to the repeating spaced-repetition cadence run
(lab/TEACH_SPACED_CADENCE.md): the once-per-6-session cyclic replay repaired
only items still alive at their slot and could not rehabilitate the two
items whose retention ratio r_i had already collapsed (tea_tin, metric_units
ended r <= 0.06). This arm replaces the anneal phase's fixed round-robin
schedule with a memory-manager loop:

  - Teach phase S1..S6: EXACTLY the cadence arm (same library, same session
    packs, same scheduled next-session replay of the previous item, same
    seed 7 on every consolidate call). Teaches S1..S6 and all baselines
    should be bit-identical to the committed cadence JSON on this box —
    verified in-run (CADENCE-IDENT line).
  - Later sessions S7..S12 (same count, same generic-chat filler slices):
    after EVERY session (S6..S11) all taught items are re-probed and any
    item whose r_i(k) = (m_i(k) - m_i(0)) / g_i drops below --thresh
    (default 0.50, i.e. below half its own teach gain) is queued for a
    phrasing-diverse re-teach turn (2 triples) in the NEXT session's batch.
    At most ONE item is served per session (lowest r first, then earlier
    trigger): 5 filler triples + 2 replay triples keeps every batch at
    2/7 = 28.6% replay share, under the 30% cap, with the same unique-
    response tripwire profile as the cadence and A/B arms. If several
    items are triggered at one probe, the remainder stay queued and are
    served in later sessions. If nothing is below threshold the session is
    pure filler (5 triples, share 0) — reactive means reactive.

Every trigger is an EVENT: (item, session the probe tripped after, r at
tripwire, session the re-teach landed in, r just before and just after the
re-teach, repair = r_after - r_pre). Post-hoc bracketing: for every trigger
probe the item sets that would have tripped at r < 0.25 and r < 0.75 are
recorded from the same snapshots (cheap secondary look at threshold
sensitivity, no re-run).

Harness: extends lab/teach_spaced_cadence.py (library, packs, filler,
probes, teach-phase session runner imported). REPLAY pools are EXTENDED to
8 phrasing-diverse variants per item (variants 0..1 = the cadence pools
verbatim, so teach-phase freshness matches; 2..7 new, unique strings across
the whole run) because a reactive loop can re-teach one item more often
than the cadence's once-per-cycle schedule.

Usage:
  PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
    python3 -u lab/teach_trigger_repair.py --workdir lab/teach_trigger_out

  # determinism calibration (run BEFORE the arm): identical teach twice:
  ... teach_trigger_repair.py --only metric_units --seed 7 --workdir <fresh A>
  ... teach_trigger_repair.py --only metric_units --seed 7 --workdir <fresh B>

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

import hippo_teach_demo as demo     # noqa: E402  (sets sys.path + kda_mla profile)
import teach_on_sft_r3 as e3         # noqa: E402  (E3 harness: loaders, helpers)
import teach_spaced_cadence as cad   # noqa: E402  (cadence harness: library, etc)
from session_log import SessionLog   # noqa: E402
from consolidate import (            # noqa: E402
    load_base_model, generate_text, consolidate, apply_user_adapter, stub_gate,
)

CKPT = e3.CKPT
MERGE_MAX_NEW = e3.MERGE_MAX_NEW
IM_E = e3.IM_E
user_turn = e3.user_turn
one_line = e3.one_line

ITEMS = cad.ITEMS
ORDER = cad.ORDER
SESSIONS = cad.SESSIONS
FILLER = cad.FILLER
USER = cad.USER
DEVICE = cad.DEVICE
REPLAY_CAP = cad.REPLAY_CAP

# ----------------------------------------------------------------------
# Replay pools: variants 0..1 are the cadence/A-B pools VERBATIM (teach-phase
# freshness and S1..S6 bit-identity depend on it). Variants 2..7 are new,
# phrasing-diverse, and every response string is unique across the whole run
# (no string repeats anywhere: collapse/unique tripwires stay cold).
# ----------------------------------------------------------------------
REPLAY = {k: list(v) for k, v in cad.REPLAY.items()}
REPLAY.update({
    "fav_show": cad.REPLAY["fav_show"] + [
        ("Which series tops my list?", "Person of Interest tops your list.",
         "Your number one series is Person of Interest."),
        ("What is my favourite show, again?", "Your favourite is Person of Interest.",
         "It is Person of Interest, as before."),
        ("Which show do I always recommend?", "You always recommend Person of Interest.",
         "The recommendation is Person of Interest."),
        ("Remind me of that show I love.", "You love Person of Interest.",
         "That would be Person of Interest."),
        ("If I could only pick one series?", "One pick: Person of Interest.",
         "Only one series: Person of Interest."),
        ("Which drama do I praise most?", "You praise Person of Interest the most.",
         "Most praised by you: Person of Interest."),
    ],
    "math_words": cad.REPLAY["math_words"] + [
        ("What is 5 + 2?", "The total is seven.", "Altogether it comes to seven."),
        ("How much is 4 + 3?", "That makes seven, in words.", "Four plus three is seven."),
        ("Solve 8 minus 1 for me.", "Eight minus one equals seven.", "The result is seven."),
        ("What does 2 + 5 give?", "It gives seven.", "Two and five make seven."),
        ("How many is 9 take away 2?", "Nine minus two leaves seven.", "You are left with seven."),
        ("Add 1 to 6, please.", "Six and one is seven.", "One more than six is seven."),
    ],
    "tea_tin": cad.REPLAY["tea_tin"] + [
        ("Where does my tea sit?", "It sits in the blue tin.",
         "Your tea sits in a blue tin."),
        ("Tell me where the tea is kept.", "The tea is kept in a blue tin.",
         "Kept in a blue tin, the tea is."),
        ("My tea — jar or tin?", "Tin — the blue one, as always.",
         "The tea is in the blue tin."),
        ("Which tin keeps the leaves?", "The blue tin keeps them.",
         "It is the blue tin that holds the leaves."),
        ("Point me to the tea container.", "Look for the blue tin.",
         "The container is the blue tin."),
        ("What colour is the tea tin?", "It is blue, your tea tin.",
         "Blue — that is the tea tin's colour."),
    ],
    "metric_units": cad.REPLAY["metric_units"] + [
        ("How tall is the bookshelf?", "It stands about two metres tall.",
         "The bookshelf is roughly two metres."),
        ("How long was my run this morning?", "Your run was about eight kilometres.",
         "Roughly eight kilometres, that run."),
        ("How much does the kettle hold?", "It holds about two litres.",
         "Around two litres, for the kettle."),
        ("What units for cooking weight?", "Grams, in the kitchen.",
         "Cooking weights come out in grams."),
        ("How far to the library?", "It is about one kilometre away.",
         "One kilometre, give or take."),
        ("Wind speed today — which units?", "About twelve kilometres per hour.",
         "Twelve kilometres per hour, roughly."),
    ],
    "pet_dog": cad.REPLAY["pet_dog"] + [
        ("What was the dog's name again?", "The dog is called Biscuit.",
         "Biscuit — that is the dog."),
        ("Whose name is Biscuit?", "Your dog answers to Biscuit.",
         "Biscuit is your dog's name."),
        ("Call the dog over by name.", "Come here, Biscuit!",
         "Biscuit, come over here!"),
        ("Which pet answers to Biscuit?", "The dog does; Biscuit it is.",
         "Your dog is the one named Biscuit."),
        ("Dog's name in one word?", "Biscuit, one word.",
         "One word: Biscuit."),
        ("Tell me the dog's name again.", "The name again: Biscuit.",
         "Again, the dog is Biscuit."),
    ],
    "greet_rule": cad.REPLAY["greet_rule"] + [
        ("Morning! You around?", "Good day! I am here.",
         "A very good day to you."),
        ("Hey there, hi!", "Good day, every one.",
         "Good day — nice to hear from you."),
        ("Yo! Anyone home?", "Good day; I am home.",
         "Good day, and welcome."),
        ("Hi, quick question.", "Good day — ask away.",
         "Good day! Go ahead with the question."),
        ("Hello hello.", "Good day, twice over.",
         "Good day, hello returned."),
        ("Evening, mate!", "Good day, even in the evening.",
         "A good day right back to you."),
    ],
})
assert all(len(v) == 8 for v in REPLAY.values()), \
    "every REPLAY pool must hold 8 variants (teach v0 + up to 7 triggers)"
for k, v in REPLAY.items():
    assert len({s for _, a, b in v for s in (a, b)}) == 2 * len(v), \
        f"REPLAY[{k}] repeats a response string"


def log_replay_turn(log: SessionLog, model, tok, session_id: str, turn_id: int,
                    item_key: str, variant: int) -> int:
    """Same shape as the cadence arm's replay turn (2 positive triples), but
    against the extended pools so a reactive loop never runs dry."""
    q, script_ok, correction = REPLAY[item_key][variant]
    stop = [tok.convert_tokens_to_ids("im" + "_end"), tok.eos_token_id]
    gen = generate_text(model, tok, user_turn(q), max_new_tokens=24,
                        device=DEVICE, stop_ids=stop)
    log.log_turn(session_id, turn_id, user_turn(q), script_ok + IM_E)
    log.mark_feedback(session_id, turn_id, "approve", confidence=1.0,
                      note=f"triggered re-teach of {item_key}",
                      correction=correction + IM_E)
    print(f"  re-teach turn {turn_id} [{item_key}] v{variant}: Q={q!r}")
    print(f"            model said : {gen!r}")
    print(f"            approve    : {script_ok!r}  corr: {correction!r}")
    return 2


def r_of(before, gains, snapshots, snap_key: str, item: str) -> float:
    m0 = before[item]["margin"]
    g = gains[item]
    mk = snapshots[snap_key][item]["margin"]
    return (mk - m0) / g if abs(g) > 1e-9 else float("nan")


def run_anneal_session(workdir, model, tok, idx, args, before, gains,
                       snapshots, decisions, replay_state, pending, events):
    """One trigger-phase session: generic-chat filler, plus the front of the
    trigger queue (lowest r first) as a 2-triple re-teach turn if non-empty.
    At most one re-teach per session keeps the batch at 7 triples — the
    cadence shape, 28.6% replay share."""
    sid = f"sess-trigger{idx}"
    log = SessionLog(os.path.join(workdir, f"session_{idx:02d}_{sid}.jsonl"))
    filler = FILLER[(idx - 7) * 5:(idx - 6) * 5]
    cad.anneal_session(log, model, tok, sid, filler)
    n_replay = 0
    served = None
    if pending:
        pending.sort(key=lambda p: (p["r_trigger"], p["triggered_after"]))
        served = pending.pop(0)
        variant = replay_state[served["item"]]
        replay_state[served["item"]] += 1
        n_replay = log_replay_turn(log, model, tok, sid, len(filler),
                                   served["item"], variant)
    seed = None if args.seed < 0 else args.seed
    res = consolidate(log, user_id=USER, workdir=workdir, device=DEVICE,
                      steps=args.steps, lr=2e-3, batch_size=4, kl_coef=0.05,
                      rank=8, alpha=16.0, base_model=model, tokenizer=tok,
                      target_profile=demo.TARGET_PROFILE, gate_fn=stub_gate,
                      seed=seed)
    share = n_replay / res.n_triples
    assert share <= REPLAY_CAP + 1e-9, f"replay share {share:.3f} over budget"
    decisions[f"s{idx}"] = res.decision
    if not apply_user_adapter(model, workdir, USER,
                              target_profile=demo.TARGET_PROFILE):
        demo.restore_raw_base(model)
    snapshots[f"after_s{idx}"] = cad.probe_all(model, tok, ORDER)
    st = res.train_stats
    label = f"trigger/{served['item']}" if served else "trigger/none"
    one_line(
        f"S{idx}", kind=label, gate="stub-tier1",
        decision=res.decision, reason=res.reason,
        extra=(f"top_resp_frac {st.get('top_response_fraction'):.2f} "
               f"n_triples {res.n_triples} (re-teach {n_replay}, "
               f"{share:.0%} -> {served['item'] if served else '-'}) "
               f"train loss {st['loss0']:.3f}->{st['loss_final']:.3f}"))
    if served:
        r_pre = r_of(before, gains, snapshots, f"after_s{idx - 1}", served["item"])
        r_post = r_of(before, gains, snapshots, f"after_s{idx}", served["item"])
        ev = dict(served)
        ev.update({"replayed_in": idx, "variant_used": served["variant_wait"],
                   "r_pre": r_pre, "r_after": r_post,
                   "repair_dr": r_post - r_pre})
        events.append(ev)
        one_line("EVENT", kind=f"re-teach/{served['item']}",
                 gate="trigger", decision="SERVED",
                 reason=(f"tripped after S{served['triggered_after']} "
                         f"r={served['r_trigger']:.3f} < thresh"),
                 extra=(f"replayed S{idx} v{served['variant_wait']} | "
                        f"r {r_pre:.3f} -> {r_post:.3f} (repair "
                        f"{r_post - r_pre:+.3f})"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None,
                    help="fresh/absent (default lab/teach_trigger_out)")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7,
                    help="train-loop seed; -1 disables seeding (pre-fix behaviour)")
    ap.add_argument("--thresh", type=float, default=0.50,
                    help="r_i(k) tripwire: below this fraction of teach gain "
                         "an item is queued for re-teach (default 0.50)")
    ap.add_argument("--brackets", default="0.25,0.75",
                    help="post-hoc sensitivity looks, comma-separated r values")
    ap.add_argument("--later", type=int, default=6,
                    help="later (trigger-loop) sessions after the last teach "
                         "(>=6, matches the cadence anneal count)")
    ap.add_argument("--only", default=None, choices=ORDER + ["none"],
                    help="calibration: teach ONLY this item (single session)")
    ap.add_argument("--no-poison", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()
    brackets = sorted(float(x) for x in args.brackets.split(",") if x)

    workdir = os.path.abspath(args.workdir or os.path.join(
        LAB, "teach_trigger_out"))
    if os.path.exists(workdir):
        sys.exit(f"refusing to run into existing workdir {workdir} "
                 f"(incumbent contamination) — delete it or pass a fresh --workdir")
    os.makedirs(workdir)

    t0 = time.time()
    model, tok, mc = load_base_model(CKPT, device=DEVICE, target_profile="kda_mla")
    assert mc.get("linear_type") == "kda" and mc.get("full_type") == "mla", (
        f"expected kda_mla ckpt, got {mc.get('linear_type')}/{mc.get('full_type')}")
    print(f"trigger-repair arm: {os.path.relpath(CKPT, REPO)} "
          f"(dim {mc['dim']}, {mc['n_layers']}L kda/mla, CPU fp32); "
          f"seed {'OFF (-1, pre-fix)' if args.seed < 0 else args.seed}; "
          f"thresh r < {args.thresh}; brackets {brackets}; "
          f"workdir {workdir}", flush=True)
    print(f"library: {', '.join(f'{k}({ITEMS[k]['kind']})' for k in ORDER)}", flush=True)

    only = None if args.only in (None, "none") else args.only
    probe_keys = [only] if only else ORDER
    before = cad.probe_all(model, tok, probe_keys)
    for k in probe_keys:
        p = before[k]
        print(f"  [before/{k}] gen -> {p['gen']!r}", flush=True)
        print(f"       NLL taught {p['nll_target']:.3f}  decoy {p['nll_decoy']:.3f}  "
              f"margin {p['margin']:+.3f}"
              + (f"  word-digit {p['word_digit_margin']:+.3f}"
                 if "word_digit_margin" in p else ""), flush=True)

    snapshots = {}
    decisions = {}
    replay_state = {k: 0 for k in ORDER}

    if only:
        # calibration path: one teach session, no replay, stop before guards —
        # same flow as the cadence harness so the two calibrations cross-check.
        sid, turns = SESSIONS[only]
        log = SessionLog(os.path.join(workdir, f"session_calib_{only}.jsonl"))
        e3.teach_session(log, model, tok, sid, turns)
        seed = None if args.seed < 0 else args.seed
        res = consolidate(log, user_id=USER, workdir=workdir, device=DEVICE,
                          steps=args.steps, lr=2e-3, batch_size=4, kl_coef=0.05,
                          rank=8, alpha=16.0, base_model=model, tokenizer=tok,
                          target_profile=demo.TARGET_PROFILE, gate_fn=stub_gate,
                          seed=seed)
        assert apply_user_adapter(model, workdir, USER,
                                  target_profile=demo.TARGET_PROFILE)
        after = cad.probe_all(model, tok, [only])
        st = res.train_stats
        one_line("CALIB", kind=f"{ITEMS[only]['kind']}/{only}", gate="stub-tier1",
                 decision=res.decision, reason=res.reason,
                 extra=(f"train loss {st['loss0']:.3f}->{st['loss_final']:.3f} "
                        f"| margin {before[only]['margin']:+.6f} -> "
                        f"{after[only]['margin']:+.6f} "
                        f"| seed {'OFF' if args.seed < 0 else args.seed}"))
        out = {"calib_item": only, "seed": args.seed,
               "before": before, "after": after, "train_stats": st,
               "decision": res.decision,
               "wall_sec": round(time.time() - t0, 1)}
        out_path = os.path.join(workdir, f"calib_{only}_seed{args.seed}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"calibration json: {out_path}", flush=True)
        return

    assert args.later >= 6, "trigger loop wants >=6 later sessions after the last teach"
    n_sessions = len(ORDER) + args.later

    # ------- teach phase S1..S6: verbatim cadence harness (same runner) -------
    for idx in range(1, len(ORDER) + 1):
        res = cad.run_session(workdir, model, tok, idx, ORDER[idx - 1], args,
                              before, snapshots, decisions, replay_state)
        if res.decision == "ROLLBACK":
            print(f"TRIPWIRE EVENT S{idx}: {res.reason} — fallback budget armed "
                  f"(one gate-safe spaced re-teach max); aborting for analyst "
                  f"decision, see lab/FAILURES.md", flush=True)
            sys.exit(2)

    # ------- cross-check: teach phase bit-identity vs the cadence run -------
    cadence_ident = {"checked": False}
    cad_json_path = os.path.join(LAB, "teach_cadence_out", "teach_spaced_cadence.json")
    if os.path.exists(cad_json_path):
        with open(cad_json_path) as f:
            cad_json = json.load(f)
        deltas = {}
        for k in ORDER:
            deltas[f"before/{k}"] = abs(before[k]["margin"] -
                                        cad_json["before"][k]["margin"])
            for s in range(1, len(ORDER) + 1):
                deltas[f"s{s}/{k}"] = abs(
                    snapshots[f"after_s{s}"][k]["margin"] -
                    cad_json["snapshots"][f"after_s{s}"][k]["margin"])
        mad = max(deltas.values())
        cadence_ident = {"checked": True, "max_abs_margin_delta": mad,
                         "identical": mad < 1e-6, "worst": max(deltas, key=deltas.get)}
        print(f"CADENCE-IDENT: teach phase S1..S6 vs committed cadence run — "
              f"max |margin Δ| = {mad:.3e} ({'BIT-IDENTICAL' if mad < 1e-6 else 'DIVERGED'}; "
              f"worst {cadence_ident['worst']})", flush=True)

    gains = {k: snapshots[f"after_s{i + 1}"][k]["margin"] - before[k]["margin"]
             for i, k in enumerate(ORDER)}
    print("GAINS g_i: " + "  ".join(f"{k} {gains[k]:+.3f}" for k in ORDER), flush=True)

    # ------- trigger phase S7..S12: memory-manager loop replaces the schedule -------
    pending = []      # queue of {item, r_trigger, triggered_after, variant_wait}
    events = []       # served re-teach events (populated in run_anneal_session)
    trigger_evals = {}
    tripwire_abort = None

    def evaluate_triggers(k):
        """Probe after session k -> queue items under threshold for k+1."""
        rs = {key: r_of(before, gains, snapshots, f"after_s{k}", key) for key in ORDER}
        new = []
        for key in ORDER:
            if rs[key] < args.thresh and not any(p["item"] == key for p in pending):
                pending.append({"item": key, "r_trigger": rs[key],
                                "triggered_after": k,
                                "variant_wait": replay_state[key]})
                new.append(key)
        trigger_evals[f"after_s{k}"] = {
            "r": rs,
            "tripped_at_thresh": new,
            "brackets": {str(b): [key for key in ORDER if rs[key] < b]
                         for b in brackets},
            "queue_after": [p["item"] for p in sorted(
                pending, key=lambda p: (p["r_trigger"], p["triggered_after"]))],
        }
        if new:
            one_line("TRIGGER", kind="probe",
                     gate=f"r<{args.thresh}", decision="QUEUE",
                     reason=", ".join(f"{k2} r={rs[k2]:.3f}" for k2 in new),
                     extra=f"queued into S{k + 1} | queue "
                           f"{trigger_evals[f'after_s{k}']['queue_after']}")
        return new

    evaluate_triggers(len(ORDER))  # probe after S6 seeds the S7 batch
    for idx in range(len(ORDER) + 1, n_sessions + 1):
        res = run_anneal_session(workdir, model, tok, idx, args, before, gains,
                                 snapshots, decisions, replay_state, pending, events)
        if res.decision == "ROLLBACK":
            tripwire_abort = (idx, res.reason)
            print(f"TRIPWIRE EVENT S{idx}: {res.reason} — fallback budget armed "
                  f"(one gate-safe spaced re-teach max); aborting for analyst "
                  f"decision, see lab/FAILURES.md", flush=True)
            break
        if idx < n_sessions:
            evaluate_triggers(idx)

    snap_keys = [f"after_s{i}" for i in range(1, n_sessions + 1)
                 if f"after_s{i}" in snapshots]

    # ---------------- retention table: r_i(k) per item ----------------
    print("\nRETENTION (margin m_i(k) after each session; g_i = gain at teach; "
          "r_i(k) = (m_i(k) - baseline) / g_i)", flush=True)
    tables = {}
    for i, key in enumerate(ORDER):
        m0 = before[key]["margin"]
        g = gains[key]
        row = {}
        for sk in snap_keys:
            if int(sk.split("after_s")[1]) >= i + 1:
                mk = snapshots[sk][key]["margin"]
                row[sk] = {"margin": mk,
                           "r": ((mk - m0) / g) if abs(g) > 1e-9 else float("nan")}
        margins = [row[sk]["margin"] for sk in snap_keys if sk in row]
        decays = [round(b - a, 3) for a, b in zip(margins, margins[1:])]
        mu_d = sum(decays) / len(decays) if decays else 0.0
        tables[key] = {"kind": ITEMS[key]["kind"], "baseline": m0, "g": g,
                       "row": row, "decays": decays, "mu_d": mu_d,
                       "r_final": row[snap_keys[-1]]["r"] if row else float("nan")}
        cells = "  ".join(f"{sk}:{row[sk]['margin']:+.3f}(r={row[sk]['r']:.2f})"
                          for sk in snap_keys if sk in row)
        print(f"  {key:12s} g={g:+.3f}  {cells}", flush=True)
    for key in tables:
        print(f"  decay {key:12s} d_i(k): {tables[key]['decays']}  "
              f"mu_d {tables[key]['mu_d']:+.3f}  r_final "
              f"{tables[key]['r_final']:.2f}", flush=True)
    n_above = sum(1 for k in ORDER if tables[k]["r_final"] >= args.thresh)
    n_dead = sum(1 for k in ORDER if tables[k]["r_final"] <= 0.06)
    print(f"HORIZON: {n_above}/{len(ORDER)} items r_final >= {args.thresh}; "
          f"{n_dead}/{len(ORDER)} r_final <= 0.06 (dead slot); "
          f"{len(events)} re-teach events", flush=True)

    class_stats = {}
    for kind in ("fact", "style"):
        keys = [k for k in ORDER if ITEMS[k]["kind"] == kind]
        decs = [d for k in keys for d in tables[k]["decays"]]
        mu_d = sum(decs) / len(decs) if decs else 0.0
        g_mean = sum(tables[k]["g"] for k in keys) / len(keys)
        r_mean = sum(tables[k]["r_final"] for k in keys) / len(keys)
        class_stats[kind] = {"mu_d": mu_d, "g_mean": g_mean,
                             "r_final_mean": r_mean}
        print(f"CLASS {kind:5s}: mu_d {mu_d:+.3f}/sess, mean gain {g_mean:+.3f}, "
              f"mean r_final {r_mean:.2f}", flush=True)

    if tripwire_abort:
        print("run aborted at tripwire — no poison/merge verdicts", flush=True)
    # ---------------- poison canary ----------------
    poison_decision, poison_reason, intact = "skipped", "--no-poison", None
    if not args.no_poison and not tripwire_abort:
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
                             target_profile=demo.TARGET_PROFILE, gate_fn=stub_gate,
                             seed=None if args.seed < 0 else args.seed)
        if not apply_user_adapter(model, workdir, USER, target_profile=demo.TARGET_PROFILE):
            demo.restore_raw_base(model)
        after = cad.probe_all(model, tok, ORDER)
        intact = all(abs(after[k]["margin"] - snapshots[snap_keys[-1]][k]["margin"]) < 1e-6
                     for k in ORDER)
        poison_decision, poison_reason = poison.decision, poison.reason
        one_line(f"S{n_sessions + 1}", kind="poison/banana-batch", gate="stub-tier1",
                 decision=poison.decision, reason=poison.reason,
                 extra=f"incumbent adapter intact: {intact}")
        assert poison.decision == "ROLLBACK", \
            f"tier-1 stub must reject the poison batch: {poison.decision}"
        assert intact, "tier-1 rollback must leave the incumbent untouched"

    # ---------------- tier 2: ONE Immune merge verdict ----------------
    merge_metrics = None
    if not args.no_merge and not args.no_poison and not tripwire_abort:
        inc_delta = os.path.join(workdir, "adapters", USER, "delta.pt")
        if any(d == "PROMOTE" for d in decisions.values()):
            import immune_gate
            m_inc = os.path.join(workdir, "merge_check", "incumbent_empty")
            m_cand = os.path.join(workdir, "merge_check", "candidate_fulldelta")
            os.makedirs(m_inc, exist_ok=True)
            os.makedirs(m_cand, exist_ok=True)
            shutil.copy2(inc_delta, os.path.join(m_cand, "delta.pt"))
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
        "arm": "trigger", "ckpt": CKPT, "steps": args.steps, "seed": args.seed,
        "thresh": args.thresh, "brackets": brackets,
        "later_sessions": args.later,
        "library": {k: {"kind": ITEMS[k]["kind"], "teach": ITEMS[k]["teach"]}
                    for k in ORDER},
        "order": ORDER, "decisions": decisions, "replay_state": replay_state,
        "before": before, "snapshots": snapshots, "gains": gains,
        "trigger_evals": trigger_evals, "events": events,
        "pending_unserved": pending,
        "retention": tables, "class_stats": class_stats,
        "horizon": {"n_above_thresh": n_above, "thresh": args.thresh,
                    "n_dead_slot": n_dead, "dead_slot_r": 0.06,
                    "n_events": len(events)},
        "cadence_ident": cadence_ident,
        "poison_decision": poison_decision, "poison_reason": poison_reason,
        "incumbent_intact": intact,
        "merge": merge_metrics,
        "tripwire_abort": tripwire_abort,
        "wall_sec": round(time.time() - t0, 1),
    }
    out_path = os.path.join(workdir, "teach_trigger_repair.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"results json: {out_path}", flush=True)


if __name__ == "__main__":
    main()
