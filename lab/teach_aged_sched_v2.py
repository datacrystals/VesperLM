#!/usr/bin/env python3
"""Aged-priority scheduler v2 — age from FIRST TRIP + deficit floor — on the
SFT-r3 chat checkpoint.

Direct follow-up to the aged-scheduler v1 arm (lab/TEACH_AGED_SCHED.md): v1's
priority = age-from-last-service + deficit killed the trigger arm's first-trip
starvation mechanically (tea_tin hypothetical rank #1 at its first two probes)
but regressed the horizon aggregate (2/6 held >= 0.50, 2 dead slots, mean
0.279 vs the trigger arm's 4/6 / 0 dead / 0.581). v1 §5 assigned three causes:

  (1) re-queue-young re-trips starve: last_service updates on serve, so a
      served-and-re-damaged item re-queues with age 0 (metric_units' re-trip
      waited 5 probes, pet_dog's 4; both ended dead);
  (2) age-init inflation: age_INITialised to the teach-phase contact let
      fav_show (init S2) win a slot at S10 with deficit 0.000 after
      self-recovering to r 0.81 — age accrued above threshold is not damage;
  (3) end-loaded destructive serves: the rescued/late serves are among the
      most damaging sessions for bystanders, and the horizon snapshot freezes
      them mid-oscillation.

This arm is the SMALLEST HONEST DELTA off v1: same library, same teach phase
S1..S6 (verbatim cadence runner, bit-checked in-run against the committed
cadence JSON), same later-session slots/filler/threshold/30% cap/tripwires,
same single-serve-per-session budget — ONLY the queue discipline changes.

SCHEDULER v2 (exact formula as implemented, documented BEFORE running;
weights w_age = w_def = 1.0).

  For each item i with a recorded trip, evaluated at the probe after
  session k (which seeds the batch of session k+1):

      first_trip_i  = earliest probe session k' (>= 6) with r_i(k') < thresh
                      — sticky for the whole scheduler phase; NEVER reset by
                        service or by partial recovery, so a served item
                        KEEPS its seniority and a re-trip re-enters with its
                        full age, not age 0  (fixes v1 cause 1)
      age_i(k)      = k - first_trip_i
                      — age starts at the actual first trip, not the
                        teach-phase contact                        (fixes 2a)
      deficit_i(k)  = max(0, thresh - r_i(k))      # r re-probed live
      eligible_i(k) = deficit_i(k) > 0             # DEFICIT FLOOR: a
                      self-recovered item (r >= thresh at the live probe)
                      cannot consume a slot on age alone; it stays queued
                      with its age and re-becomes eligible if it drops back
                      below threshold                        (fixes 2b)
      priority_i(k) = w_age * age_i(k) + w_def * deficit_i(k)

  Serve = pop the eligible item with max priority, one per session.
  Tie-break: earlier first_trip_i, then library ORDER index (fully
  deterministic under the pinned seed). A session with zero eligible items
  runs filler only (no wasted re-teach). Serves keep their v1 budget of
  exactly ONE re-teach turn (2 triples, 28.6% <= 30% cap) — the capped
  per-session serve count the v1 §5 note called for is structural, unchanged.

DELIBERATELY NOT CHANGED FROM v1 (scope-honest: this is a scheduler variant
arm, not a memory architecture change):

  - trip threshold (r < 0.50 probe-after-every-session), tripwire rules,
    8-variant pools, poison canary, single tier-2 Immune verdict, CPU fp32.
  - rescue protocol: NOT re-run. v1 already measured it (tea_tin from strict
    r = 0.018, one contact, d_r = +0.063 — NOT resurrectable from r <= 0.06,
    n = 1) and its holdout costs 1/6 capacity and confounds the scheduler
    read (v1 §5.3). In v2 tea_tin is a NORMAL queue member.

SECONDARY READS (log-only counterfactuals, zero capacity effect):

  - dead-ban queue position: at every probe, for each queued item in the
    strict dead band (r <= 0.06), log where it sits and what a
    "rescues-into-dead-ban" policy would have served instead
    (max-priority eligible with r > DEAD_R). v1's rescue answer says serving
    strict-dead items yields ~nothing; this trace costs nothing and tests
    whether the ban would have re-ordered any actual service.
  - per-session serve cap: structural and unchanged at 1; recorded per
    session for the v1 §5.4 end-loaded-cost read.

Usage:
  PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
    python3 -u lab/teach_aged_sched_v2.py --workdir lab/teach_aged2_out

  # determinism calibration (run BEFORE the arm): identical teach twice,
  # must bit-match each other and the v1/trigger/cadence harnesses'
  # committed seed-7 calibration for the same item:
  ... teach_aged_sched_v2.py --only metric_units --seed 7 --workdir <fresh A>
  ... teach_aged_sched_v2.py --only metric_units --seed 7 --workdir <fresh B>

Fresh workdir required. CPU only, fp32 (Immune backend never leaves CPU, I7).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
import time

REPO = os.environ.get("VESPER_REPO", "/home/tliao/VesperLM")
LAB = os.path.join(REPO, "lab")
sys.path.insert(0, LAB)

import hippo_teach_demo as demo      # noqa: E402  (sets sys.path + kda_mla profile)
import teach_on_sft_r3 as e3          # noqa: E402  (E3 harness: loaders, helpers)
import teach_spaced_cadence as cad    # noqa: E402  (cadence harness: library, etc)
import teach_trigger_repair as trig   # noqa: E402  (trigger harness: pools, turns)
from session_log import SessionLog    # noqa: E402
from consolidate import (             # noqa: E402
    load_base_model, consolidate, apply_user_adapter, stub_gate,
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
REPLAY = trig.REPLAY  # the 8-variant extended pools (trigger arm verbatim)

DEAD_R = 0.06          # strict dead-slot line (reporting + dead-ban trace only)
W_AGE = 1.0
W_DEF = 1.0


def v2_priority(k_probe, item, r_now, first_trip, thresh):
    """v2 formula: age from FIRST trip (sticky, never reset by service) +
    live deficit. Returns (priority, age, deficit, eligible)."""
    age = k_probe - first_trip[item]
    deficit = max(0.0, thresh - r_now)
    eligible = deficit > 1e-12
    return W_AGE * age + W_DEF * deficit, age, deficit, eligible


def run_later_session(workdir, model, tok, idx, args, before, gains,
                      snapshots, decisions, replay_state, queue, events,
                      trips, sched):
    """One scheduler-phase session: generic-chat filler + at most ONE
    re-teach turn — the max-priority ELIGIBLE queue entry, or filler-only if
    the queue holds no eligible item. Batch stays at 7 triples (28.6%
    re-teach share); serve cap is structural at 1."""
    sid = f"sess-agedv2-{idx}"
    log = SessionLog(os.path.join(workdir, f"session_{idx:02d}_{sid}.jsonl"))
    filler = FILLER[(idx - 7) * 5:(idx - 6) * 5]
    cad.anneal_session(log, model, tok, sid, filler)
    n_replay = 0
    served = None
    floor_block = False
    k_probe = idx - 1

    if queue:
        for p in queue:
            prio, age, deficit, eligible = v2_priority(
                k_probe, p["item"], p["r_last"], sched["first_trip"],
                args.thresh)
            p["priority"], p["age"], p["deficit"], p["eligible"] = (
                prio, age, deficit, eligible)
        elig = [p for p in queue if p["eligible"]]
        elig.sort(key=lambda p: (-p["priority"],
                                 sched["first_trip"][p["item"]],
                                 ORDER.index(p["item"])))
        if elig:
            served = elig[0]
            queue.remove(served)
            for t in reversed(trips):
                if (t["item"] == served["item"] and
                        t["triggered_after"] == served["triggered_after"] and
                        t["served_in"] is None):
                    t["served_in"] = idx
                    break
        elif queue:
            floor_block = True

    if served:
        variant = replay_state[served["item"]]
        replay_state[served["item"]] += 1
        assert variant < len(REPLAY[served["item"]]), "replay pool exhausted"
        n_replay = trig.log_replay_turn(log, model, tok, sid, len(filler),
                                        served["item"], variant)
        served["variant_used"] = variant
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
    one_line(
        f"S{idx}", kind=f"agedv2/{served['item'] if served else 'none'}",
        gate="stub-tier1", decision=res.decision, reason=res.reason,
        extra=(f"top_resp_frac {st.get('top_response_fraction'):.2f} "
               f"n_triples {res.n_triples} (re-teach {n_replay}, "
               f"{share:.0%} -> {served['item'] if served else '-'}"
               + (f" [prio {served['priority']:.3f} age {served['age']} "
                  f"deficit {served['deficit']:.3f}" if served else "")
               + (" [FLOOR: queue all self-recovered, filler only]"
                  if floor_block else "")
               + f") train loss {st['loss0']:.3f}->{st['loss_final']:.3f}"))
    if served:
        r_pre = trig.r_of(before, gains, snapshots, f"after_s{idx - 1}",
                          served["item"])
        r_post = trig.r_of(before, gains, snapshots, f"after_s{idx}",
                           served["item"])
        since_last_serve = idx - (sched["last_service"].get(served["item"],
                                                            served["first_trip"]))
        ev = {"item": served["item"], "kind": "queue",
              "first_trip": served["first_trip"],
              "triggered_after": served["triggered_after"],
              "r_trigger": served["r_trigger"],
              "replayed_in": idx, "variant_used": variant,
              "wait_sessions": idx - (served["triggered_after"] or idx),
              "age_first_trip_at_serve": served["age"],
              "sessions_since_last_serve": since_last_serve,
              "r_pre": r_pre, "r_after": r_post,
              "repair_dr": r_post - r_pre,
              "priority": served["priority"], "age": served["age"],
              "deficit": served["deficit"]}
        events.append(ev)
        sched["last_service"][served["item"]] = idx
        one_line("EVENT", kind=f"re-teach/{served['item']}",
                 gate="aged-v2", decision="SERVED",
                 reason=(f"first trip after S{served['first_trip']} "
                         f"(this trip S{served['triggered_after']}) "
                         f"r={served['r_trigger']:.3f}, age at serve "
                         f"{served['age']}"), 
                 extra=(f"served S{idx} v{variant} | "
                        f"r {r_pre:.3f} -> {r_post:.3f} (repair "
                        f"{r_post - r_pre:+.3f})"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None,
                    help="fresh/absent (default lab/teach_aged2_out)")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7,
                    help="train-loop seed; -1 disables seeding (pre-fix behaviour)")
    ap.add_argument("--thresh", type=float, default=0.50,
                    help="r_i(k) tripwire: below this fraction of teach gain "
                         "an item is queued (default 0.50 — the hinge)")
    ap.add_argument("--brackets", default="0.25,0.75",
                    help="post-hoc sensitivity looks, comma-separated r values")
    ap.add_argument("--later", type=int, default=6,
                    help="later (scheduler-phase) sessions (>=6, matches the "
                         "cadence/trigger/aged-v1 later-session count)")
    ap.add_argument("--only", default=None, choices=ORDER + ["none"],
                    help="calibration: teach ONLY this item (single session)")
    ap.add_argument("--no-poison", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()
    brackets = sorted(float(x) for x in args.brackets.split(",") if x)

    workdir = os.path.abspath(args.workdir or os.path.join(
        LAB, "teach_aged2_out"))
    if os.path.exists(workdir):
        sys.exit(f"refusing to run into existing workdir {workdir} "
                 f"(incumbent contamination) — delete it or pass a fresh --workdir")
    os.makedirs(workdir)

    t0 = time.time()
    model, tok, mc = load_base_model(CKPT, device=DEVICE, target_profile="kda_mla")
    assert mc.get("linear_type") == "kda" and mc.get("full_type") == "mla", (
        f"expected kda_mla ckpt, got {mc.get('linear_type')}/{mc.get('full_type')}")
    print(f"aged-scheduler v2 arm: {os.path.relpath(CKPT, REPO)} "
          f"(dim {mc['dim']}, {mc['n_layers']}L kda/mla, CPU fp32); "
          f"seed {'OFF (-1, pre-fix)' if args.seed < 0 else args.seed}; "
          f"thresh r < {args.thresh}; "
          f"priority = {W_AGE}*(k - first_trip) + {W_DEF}*max(0, thresh - r), "
          f"eligible iff deficit > 0 (floor); workdir {workdir}", flush=True)
    print(f"library: {', '.join(f'{k}({ITEMS[k]['kind']})' for k in ORDER)}",
          flush=True)

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
        # calibration path: one teach session, no replay — same flow as the
        # cadence/trigger/v1 harnesses so all four calibrations cross-check.
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

    assert args.later >= 6, "scheduler phase wants >=6 later sessions"
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
    cad_json_path = os.path.join(LAB, "teach_cadence_out",
                                 "teach_spaced_cadence.json")
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
    print("GAINS g_i: " + "  ".join(f"{k} {gains[k]:+.3f}" for k in ORDER),
          flush=True)

    # ------- scheduler phase S7..S12: age-from-first-trip + deficit floor ----
    queue = []         # service queue of {item, triggered_after, r_last, ...}
    events = []        # served re-teach events
    trips = []         # every queue insertion (first + re-trips) w/ contact latency
    v2_evals = {}      # per-probe scheduler trace
    tripwire_abort = None
    sched = {
        "first_trip": {},     # item -> first probe session with r < thresh (sticky)
        "last_service": {},   # item -> last served session (reporting only)
        "dead_ban_trace": [], # log-only dead-ban counterfactual, per probe
    }

    def evaluate(k):
        """Probe after session k -> queue items under threshold (first_trip
        sticky); update live r; log the dead-ban counterfactual."""
        rs = {key: trig.r_of(before, gains, snapshots, f"after_s{k}", key)
              for key in ORDER}
        new = []
        for p in queue:
            p["r_last"] = rs[p["item"]]
        for key in ORDER:
            if rs[key] < args.thresh and not any(p["item"] == key
                                                 for p in queue):
                if key not in sched["first_trip"]:
                    sched["first_trip"][key] = k
                    kind = "first"
                else:
                    kind = "re-trip"
                queue.append({"item": key, "triggered_after": k,
                              "first_trip": sched["first_trip"][key],
                              "r_trigger": rs[key], "r_last": rs[key]})
                trips.append({"item": key, "triggered_after": k,
                              "first_trip": sched["first_trip"][key],
                              "r_trigger": rs[key], "kind": kind,
                              "served_in": None})
                new.append(key)
        # priority snapshot for every queued entry
        snap = []
        for p in queue:
            prio, age, deficit, eligible = v2_priority(
                k, p["item"], p["r_last"], sched["first_trip"], args.thresh)
            snap.append({"item": p["item"], "r_last": p["r_last"],
                         "first_trip": p["first_trip"],
                         "triggered_after": p["triggered_after"],
                         "priority": prio, "age": age, "deficit": deficit,
                         "eligible": eligible})
        snap.sort(key=lambda q: (-q["priority"], sched["first_trip"][q["item"]],
                                 ORDER.index(q["item"])))
        # dead-ban counterfactual: eligible candidates with r > DEAD_R
        elig_live = [q for q in snap if q["eligible"]]
        banned = [q["item"] for q in elig_live if q["r_last"] <= DEAD_R]
        keep = [q for q in elig_live if q["r_last"] > DEAD_R]
        sched["dead_ban_trace"].append({
            "probe_after": k,
            "v2_would_serve": elig_live[0]["item"] if elig_live else None,
            "dead_banned": banned,
            "dead_ban_would_serve": keep[0]["item"] if keep else None,
        })
        eval_entry = {
            "r": rs, "tripped_at_thresh": new,
            "brackets": {str(b): [key for key in ORDER if rs[key] < b]
                         for b in brackets},
            "queue_with_priority": snap,
            "dead_ban": sched["dead_ban_trace"][-1],
        }
        v2_evals[f"after_s{k}"] = eval_entry
        one_line("SCHED", kind="probe",
                 gate=f"r<{args.thresh}+first-trip-age",
                 decision="QUEUE" if new else "HOLD",
                 reason=(", ".join(f"{k2} r={rs[k2]:.3f}" for k2 in new)
                         if new else "no new trips"),
                 extra=(f"queue (prio order) "
                        f"{[(q['item'], round(q['priority'], 3),
                             'E' if q['eligible'] else 'floor')
                            for q in snap]}"
                        + (f" | dead-ban trace: {banned}" if banned else "")))
        return new

    evaluate(len(ORDER))  # probe after S6 seeds the S7 batch
    for idx in range(len(ORDER) + 1, n_sessions + 1):
        res = run_later_session(workdir, model, tok, idx, args, before, gains,
                                snapshots, decisions, replay_state, queue,
                                events, trips, sched)
        if res.decision == "ROLLBACK":
            tripwire_abort = (idx, res.reason)
            print(f"TRIPWIRE EVENT S{idx}: {res.reason} — fallback budget armed "
                  f"(one gate-safe spaced re-teach max); aborting for analyst "
                  f"decision, see lab/FAILURES.md", flush=True)
            break
        if idx < n_sessions:
            evaluate(idx)

    snap_keys = [f"after_s{i}" for i in range(1, n_sessions + 1)
                 if f"after_s{i}" in snapshots]

    # ---------------- retention table: r_i(k) per item ----------------
    print("\nRETENTION (margin m_i(k) after each session; g_i = gain at "
          "teach; r_i(k) = (m_i(k) - baseline) / g_i)", flush=True)
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
    n_dead = sum(1 for k in ORDER if tables[k]["r_final"] <= DEAD_R)
    mean_final = sum(tables[k]["r_final"] for k in ORDER) / len(ORDER)
    min_final = min(tables[k]["r_final"] for k in ORDER)
    print(f"HORIZON: {n_above}/{len(ORDER)} items r_final >= {args.thresh}; "
          f"{n_dead}/{len(ORDER)} r_final <= {DEAD_R} (dead slot); "
          f"mean r_final {mean_final:.3f}; min {min_final:.3f}; "
          f"{len(events)} serve events", flush=True)

    waits = [e["wait_sessions"] for e in events if e["kind"] == "queue"]
    wait_hist = dict(sorted(collections.Counter(waits).items()))
    max_wait = max(waits) if waits else None
    print(f"QUEUE-WAIT histogram (serves, sessions trip->serve): "
          f"{wait_hist}; max wait {max_wait}", flush=True)
    # every trip event + its contact latency (the v1 re-trip-starve read)
    print("TRIP-CONTACT latency (every queue insertion):")
    unserved_re_retrips = 0
    for t in trips:
        if t["served_in"] is not None:
            lat = t["served_in"] - t["triggered_after"]
            print(f"  {t['item']:12s} {t['kind']:8s} after S{t['triggered_after']} "
                  f"(first trip S{t['first_trip']}) -> served S{t['served_in']}, "
                  f"latency {lat}", flush=True)
        else:
            waiting = n_sessions - 1 - t["triggered_after"]
            if t["kind"] == "re-trip":
                unserved_re_retrips += 1
            print(f"  {t['item']:12s} {t['kind']:8s} after S{t['triggered_after']} "
                  f"(first trip S{t['first_trip']}) -> UNSERVED, "
                  f"waited {waiting} probes", flush=True)
    print(f"RE-TRIP STARVATION at horizon: {unserved_re_retrips} unserved "
          f"re-trip(s)", flush=True)

    class_stats = {}
    for kind in ("fact", "style"):
        keys = [k for k in ORDER if ITEMS[k]["kind"] == kind]
        decs = [d for k in keys for d in tables[k]["decays"]]
        mu_d = sum(decs) / len(decs) if decs else 0.0
        g_mean = sum(tables[k]["g"] for k in keys) / len(keys)
        r_mean = sum(tables[k]["r_final"] for k in keys) / len(keys)
        class_stats[kind] = {"mu_d": mu_d, "g_mean": g_mean,
                             "r_final_mean": r_mean}
        print(f"CLASS {kind:5s}: mu_d {mu_d:+.3f}/sess, mean gain "
              f"{g_mean:+.3f}, mean r_final {r_mean:.2f}", flush=True)

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
    merge_verdict = None
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
            merge_verdict = {"decision": verdict["decision"],
                             "reason": verdict["reason"]}
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
            print("VERDICT MERGE skipped: no promoted adapter to evaluate",
                  flush=True)

    print(f"\ntotal wall {time.time() - t0:.1f}s  artifacts {workdir}",
          flush=True)
    out = {
        "arm": "aged-sched-v2-first-trip-age-floor", "ckpt": CKPT,
        "steps": args.steps,
        "seed": args.seed, "thresh": args.thresh, "brackets": brackets,
        "later_sessions": args.later,
        "scheduler": {
            "formula": "priority_i(k) = w_age*(k - first_trip_i) + "
                       "w_def*max(0, thresh - r_i(k)); eligible iff deficit>0; "
                       "first_trip sticky (never reset by service); ties -> "
                       "earlier first_trip, then library order",
            "w_age": W_AGE, "w_def": W_DEF, "dead_r_reporting": DEAD_R,
            "serve_cap_per_session": 1,
        },
        "library": {k: {"kind": ITEMS[k]["kind"], "teach": ITEMS[k]["teach"]}
                    for k in ORDER},
        "order": ORDER, "decisions": decisions, "replay_state": replay_state,
        "before": before, "snapshots": snapshots, "gains": gains,
        "v2_evals": v2_evals, "events": events, "trips": trips,
        "dead_ban_trace": sched["dead_ban_trace"],
        "pending_unserved": queue,
        "queue_wait_hist": wait_hist, "max_wait": max_wait,
        "unserved_re_trips_at_horizon": unserved_re_retrips,
        "retention": tables, "class_stats": class_stats,
        "horizon": {"n_above_thresh": n_above, "thresh": args.thresh,
                    "n_dead_slot": n_dead, "dead_slot_r": DEAD_R,
                    "mean_r_final": mean_final, "min_r_final": min_final,
                    "n_events": len(events)},
        "cadence_ident": cadence_ident,
        "poison_decision": poison_decision, "poison_reason": poison_reason,
        "incumbent_intact": intact,
        "merge": merge_metrics, "merge_verdict": merge_verdict,
        "tripwire_abort": tripwire_abort,
        "wall_sec": round(time.time() - t0, 1),
    }
    out_path = os.path.join(workdir, "teach_aged_sched_v2.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"results json: {out_path}", flush=True)


if __name__ == "__main__":
    main()
