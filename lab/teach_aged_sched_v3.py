#!/usr/bin/env python3
"""Scheduler v3 — DEFICIT-DOMINANT priority + per-item serve budget
(no-progress eviction) — on the SFT-r3 chat checkpoint.

Direct follow-up to the aged-scheduler v2 arm (lab/TEACH_AGED_SCHED_V2.md),
whose verdict was explicitly NEGATIVE with the mechanistic cause named: the
chronic-seniority HOG. Under age-from-first-trip-as-principal, tea_tin's
three serves netted +0.274 r without ever crossing threshold (per-serve
yields -0.059, +0.239, +0.094 — the third serve inside bystander
oscillation noise), so it re-queued with maximum seniority at every probe
and consumed 3/6 slots, while greet_rule — queued after S7, eligible at all
five later probes, ending with the run's largest live deficit (0.987 at
r = -0.49) — was never served. The serve cap of 1 was never the binding
constraint (eligibility was), and the dead-ban counterfactual cut both ways
(blocking tea_tin at S8 buys metric_units a slot; blocking metric_units at
S11 would have killed the run's best serve, -0.115 -> +0.897): NOT adopted.

This arm is the SMALLEST HONEST DELTA off v2: same library, same teach
phase S1..S6 (verbatim cadence runner, bit-checked in-run against the
committed cadence JSON), same later-session slots/filler/threshold/30% cap/
tripwires/poison/tier-2 — ONLY the queue discipline changes, and only the
two knobs the v2 traces named.

SCHEDULER v3 (exact formula as implemented, documented BEFORE running).

  For each item i with a recorded trip, evaluated at the probe after
  session k (which seeds the batch of session k+1):

      first_trip_i  = earliest probe session k' (>= 6) with r_i(k') < thresh
                      — sticky for the whole scheduler phase (unchanged from
                        v2; reporting + tie-break ONLY, never the principal)
      age_i(k)      = k - first_trip_i
      deficit_i(k)  = max(0, thresh - r_i(k))         # r re-probed live
      eligible_i(k) = deficit_i(k) > 0  AND  i not evicted

      SERVE ORDER (deterministic sort key, highest first):
          1. deficit_i(k)        — DEFICIT-DOMINANT first-order term;
                                   equal to lowest live r first
          2. age_i(k)            — TIE-BREAKER ONLY (older first_trip wins
                                   exact deficit ties)
          3. library ORDER index — fully deterministic under the pinned seed

  Serve = pop the eligible item at the head of that order, ONE per session.
  A session with zero eligible items runs filler only (unchanged).

PER-ITEM SERVE BUDGET / NO-PROGRESS EVICTION (the exact knobs, defined
literally, documented BEFORE running):

  episode_i        = serves since the most recent probe at which r_i(k')
                     >= thresh (0 if none). A held probe (r >= thresh),
                     served or not, RESETS the episode to 0.
  per-serve yield  = repair d_r = r_after - r_pre around the serve
                     (snapshot probes of sessions idx-1 / idx).
  NO_PROGRESS_DR   = 0.10 — a serve with d_r <= +0.10 counts as NO
                     PROGRESS (value picked from v2's own probes: tea_tin's
                     third serve yielded +0.094, the magnitude of un-served
                     bystander oscillation (~+/-0.1 r; v2 SS5), i.e. noise).
                     Logged per serve as a diagnostic flag.
  SERVE_BUDGET     = 2 serves per unheld episode. Justification: in v2,
                     tea_tin's episode read (-0.059, +0.239, +0.094) — 2
                     serves netted +0.18 without holding and the 3rd serve
                     was already at noise level; the trigger arm's
                     metric_units crossed threshold on serve 2, so budget 2
                     does not touch any measured successful 2-serve rehab.
  EVICTION RULE    = at probe k, any queued item with episode_i serves >=
                     SERVE_BUDGET is EVICTED from the queue (removed from
                     service competition) and logged with its episode trace.
  RE-ADMISSION     = an evicted item comes back ONLY if it first reads
                     r_i >= thresh UNAIDED at a later probe (a genuine
                     self-recovery — "decay from a higher point") and then
                     re-trips r_i < thresh; that re-trip starts a FRESH
                     episode (budget 0). Evicted items whose deficits
                     re-trip WITHOUT such an unaided held probe are
                     logged as policy-blocked but not re-admitted. (If the
                     unaided-hold branch never fires in the run, the
                     effective policy is "never re-admit during the
                     horizon"; the doc reports which branch was used.)

UNCHANGED FROM v2 / trigger (scope-honest — this is a scheduler variant
arm, not a memory architecture change):

  - trip threshold (r < 0.50 probe-after-every-session), deficit-floor
    eligibility (self-recovered items cannot consume slots), tripwire
    rules, 8-variant pools, per-session replay share 2/7 = 28.6% <= 30%
    cap, serve cap 1 re-teach/session, poison canary, single tier-2
    Immune merge verdict, one-fallback tripwire policy, CPU fp32.
  - rescue protocol: NOT re-run (v1 measured it; v2 kept tea_tin a normal
    queue member — v3 same).

SECONDARY READS (log-only, zero capacity effect):
  - dead-ban queue position trace (unchanged from v2).
  - eviction watch trace: live r of every evicted item per probe (the
    bystander cost of the budget).
  - starvation read: for every trip (first AND re-trip), contact latency
    plus the count of probes at which the item sat ELIGIBLE but unserved —
    the v2 greet_rule-hog pattern explicitly re-checked.

Usage:
  PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
    python3 -u lab/teach_aged_sched_v3.py --workdir lab/teach_aged3_out

  # determinism calibration (run BEFORE the arm): identical teach twice,
  # must bit-match each other and the v1/v2/trigger/cadence harnesses'
  # committed seed-7 calibration for the same item:
  ... teach_aged_sched_v3.py --only metric_units --seed 7 --workdir <fresh A>
  ... teach_aged_sched_v3.py --only metric_units --seed 7 --workdir <fresh B>

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
NO_PROGRESS_DR = 0.10  # serve with repair d_r <= this counts as no progress
SERVE_BUDGET = 2       # max serves per unheld episode before eviction


def v3_keys(k_probe, item, r_now, first_trip, thresh):
    """v3 sort keys: deficit-first (dominant), age-from-first-trip as
    tie-breaker only. Returns (deficit, age, eligible)."""
    age = k_probe - first_trip[item]
    deficit = max(0.0, thresh - r_now)
    eligible = deficit > 1e-12
    return deficit, age, eligible


def sort_key(q, first_trip):
    """Highest priority first: live deficit desc, age desc (tie only),
    library order asc."""
    return (-q["deficit"], -q["age"], ORDER.index(q["item"]))


def serve_head(elig, first_trip):
    elig.sort(key=lambda q: sort_key(q, first_trip))
    return elig[0]


def run_later_session(workdir, model, tok, idx, args, before, gains,
                      snapshots, decisions, replay_state, queue, events,
                      trips, sched):
    """One scheduler-phase session: generic-chat filler + at most ONE
    re-teach turn — the max-deficit ELIGIBLE queue entry (age tie-break),
    or filler-only if the queue holds no eligible item. Batch stays at 7
    triples (28.6% re-teach share); serve cap is structural at 1."""
    sid = f"sess-agedv3-{idx}"
    log = SessionLog(os.path.join(workdir, f"session_{idx:02d}_{sid}.jsonl"))
    filler = FILLER[(idx - 7) * 5:(idx - 6) * 5]
    cad.anneal_session(log, model, tok, sid, filler)
    n_replay = 0
    served = None
    floor_block = False
    k_probe = idx - 1

    if queue:
        for p in queue:
            deficit, age, eligible = v3_keys(
                k_probe, p["item"], p["r_last"], sched["first_trip"],
                args.thresh)
            p["deficit"], p["age"], p["eligible"] = deficit, age, eligible
        elig = [p for p in queue if p["eligible"]]
        if elig:
            served = serve_head(elig, sched["first_trip"])
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
        f"S{idx}", kind=f"agedv3/{served['item'] if served else 'none'}",
        gate="stub-tier1", decision=res.decision, reason=res.reason,
        extra=(f"top_resp_frac {st.get('top_response_fraction'):.2f} "
               f"n_triples {res.n_triples} (re-teach {n_replay}, "
               f"{share:.0%} -> {served['item'] if served else '-'}"
               + (f" [deficit {served['deficit']:.3f} age {served['age']}]"
                  if served else "")
               + (" [FLOOR: queue all self-recovered, filler only]"
                  if floor_block else "")
               + f") train loss {st['loss0']:.3f}->{st['loss_final']:.3f}"))
    if served:
        r_pre = trig.r_of(before, gains, snapshots, f"after_s{idx - 1}",
                          served["item"])
        r_post = trig.r_of(before, gains, snapshots, f"after_s{idx}",
                           served["item"])
        repair = r_post - r_pre
        ep = sched["episode"][served["item"]]
        ep["serves"] += 1
        ep["drs"].append(round(repair, 6))
        if repair <= NO_PROGRESS_DR:
            ep["no_progress_serves"] += 1
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
              "repair_dr": repair,
              "no_progress": repair <= NO_PROGRESS_DR,
              "episode_serves_after": ep["serves"],
              "deficit": served["deficit"], "age": served["age"]}
        events.append(ev)
        sched["last_service"][served["item"]] = idx
        one_line("EVENT", kind=f"re-teach/{served['item']}",
                 gate="aged-v3", decision="SERVED",
                 reason=(f"first trip after S{served['first_trip']} "
                         f"(this trip S{served['triggered_after']}) "
                         f"r={served['r_trigger']:.3f}, episode serve "
                         f"{ep['serves']}/{SERVE_BUDGET}"
                         + (" [NO-PROGRESS serve]" if repair <= NO_PROGRESS_DR
                            else "")),
                 extra=(f"served S{idx} v{variant} | "
                        f"r {r_pre:.3f} -> {r_post:.3f} (repair "
                        f"{repair:+.3f})"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None,
                    help="fresh/absent (default lab/teach_aged3_out)")
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
                         "cadence/trigger/aged-v1/v2 later-session count)")
    ap.add_argument("--only", default=None, choices=ORDER + ["none"],
                    help="calibration: teach ONLY this item (single session)")
    ap.add_argument("--no-poison", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()
    brackets = sorted(float(x) for x in args.brackets.split(",") if x)

    workdir = os.path.abspath(args.workdir or os.path.join(
        LAB, "teach_aged3_out"))
    if os.path.exists(workdir):
        sys.exit(f"refusing to run into existing workdir {workdir} "
                 f"(incumbent contamination) — delete it or pass a fresh --workdir")
    os.makedirs(workdir)

    t0 = time.time()
    model, tok, mc = load_base_model(CKPT, device=DEVICE, target_profile="kda_mla")
    assert mc.get("linear_type") == "kda" and mc.get("full_type") == "mla", (
        f"expected kda_mla ckpt, got {mc.get('linear_type')}/{mc.get('full_type')}")
    print(f"aged-scheduler v3 arm: {os.path.relpath(CKPT, REPO)} "
          f"(dim {mc['dim']}, {mc['n_layers']}L kda/mla, CPU fp32); "
          f"seed {'OFF (-1, pre-fix)' if args.seed < 0 else args.seed}; "
          f"thresh r < {args.thresh}; priority DEFICIT-DOMINANT "
          f"(max(0, thresh - r)), age-from-first-trip tie-break only; "
          f"serve budget {SERVE_BUDGET}/unheld episode, no-progress serve "
          f"d_r <= {NO_PROGRESS_DR}; eviction with unaided-hold "
          f"re-admission; workdir {workdir}", flush=True)
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
        # cadence/trigger/v1/v2 harnesses so all five calibrations cross-check.
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

    # ------- scheduler phase S7..S12: deficit-dominant + serve budget -------
    queue = []         # service queue of {item, triggered_after, r_last, ...}
    events = []        # served re-teach events
    trips = []         # every queue insertion (first + re-trips) w/ contact latency
    v3_evals = {}      # per-probe scheduler trace
    tripwire_abort = None
    sched = {
        "first_trip": {},     # item -> first probe session with r < thresh (sticky; tie-break only)
        "last_service": {},   # item -> last served session (reporting only)
        "dead_ban_trace": [], # log-only dead-ban counterfactual, per probe
        "episode": {k: {"serves": 0, "drs": [], "no_progress_serves": 0}
                    for k in ORDER},
        "evicted": {},        # item -> {"probe": k, "watch": {probe: r}}
        "unaided_held": {},   # item -> probe of unaided r >= thresh after eviction
        "blocked_trips": [],  # evicted-item re-trips denied re-admission (log-only)
        "readmissions": [],   # items re-admitted via the unaided-hold branch
    }

    def evaluate(k):
        """Probe after session k -> held resets, eviction, (re-)admission,
        queue snapshot with deficit-first order, dead-ban counterfactual."""
        rs = {key: trig.r_of(before, gains, snapshots, f"after_s{k}", key)
              for key in ORDER}
        # 1. held probes reset episodes; evicted items record unaided holds
        held = []
        for key in ORDER:
            if rs[key] >= args.thresh:
                ep = sched["episode"][key]
                if ep["serves"] or ep["drs"]:
                    ep["serves"], ep["drs"], ep["no_progress_serves"] = 0, [], 0
                if key in sched["evicted"] and key not in sched["unaided_held"]:
                    sched["unaided_held"][key] = k
                    held.append(key)
        for p in queue:
            p["r_last"] = rs[p["item"]]
        # 2. eviction: queued items whose episode budget is spent
        evicted_now = []
        for p in list(queue):
            if sched["episode"][p["item"]]["serves"] >= SERVE_BUDGET:
                queue.remove(p)
                sched["evicted"][p["item"]] = {
                    "probe": k,
                    "episode_drs": list(sched["episode"][p["item"]]["drs"]),
                    "watch": {k: rs[p["item"]]}}
                for t in trips:
                    if (t["item"] == p["item"] and t["served_in"] is None
                            and not t.get("evicted")):
                        t["evicted"] = True
                evicted_now.append(p["item"])
                one_line("EVICT", kind=f"budget/{p['item']}",
                         gate="aged-v3", decision="EVICTED",
                         reason=(f"episode serves {SERVE_BUDGET} without "
                                 f"holding r >= {args.thresh}; drs "
                                 f"{sched['episode'][p['item']]['drs']}; "
                                 f"live r {rs[p['item']]:.3f}"),
                         extra="out of service competition; watch trace on")
        # 3. new trips: normal, re-admitted (unaided-hold branch), or blocked
        new = []
        for key in ORDER:
            if rs[key] < args.thresh and not any(p["item"] == key
                                                 for p in queue):
                if key in sched["evicted"]:
                    sched["evicted"][key]["watch"][k] = rs[key]
                    if key in sched["unaided_held"]:
                        # decay from a higher point: unaided held probe
                        # happened, so this is a NEW damage episode
                        del sched["evicted"][key]
                        sched["episode"][key] = {"serves": 0, "drs": [],
                                                 "no_progress_serves": 0}
                        sched["readmissions"].append(
                            {"item": key, "probe": k,
                             "unaided_hold_probe": sched["unaided_held"][key],
                             "r_now": rs[key]})
                        kind = "re-trip (re-admitted)"
                        one_line("RE-ADMIT", kind=f"budget/{key}",
                                 gate="aged-v3", decision="RE-ADMITTED",
                                 reason=(f"unaided hold at S"
                                         f"{sched['unaided_held'][key]}, "
                                         f"fresh trip r={rs[key]:.3f}"),
                                 extra="episode budget reset to 0")
                    else:
                        sched["blocked_trips"].append(
                            {"item": key, "probe": k, "r": rs[key]})
                        continue
                else:
                    kind = ("first" if key not in sched["first_trip"]
                            else "re-trip")
                if key not in sched["first_trip"]:
                    sched["first_trip"][key] = k
                queue.append({"item": key, "triggered_after": k,
                              "first_trip": sched["first_trip"][key],
                              "r_trigger": rs[key], "r_last": rs[key]})
                trips.append({"item": key, "triggered_after": k,
                              "first_trip": sched["first_trip"][key],
                              "r_trigger": rs[key], "kind": kind,
                              "served_in": None, "evicted": False,
                              "elig_probes": 0})
                new.append(key)
        # 4. priority snapshot for every queued entry (deficit-first)
        snap = []
        for p in queue:
            deficit, age, eligible = v3_keys(
                k, p["item"], p["r_last"], sched["first_trip"], args.thresh)
            snap.append({"item": p["item"], "r_last": p["r_last"],
                         "first_trip": p["first_trip"],
                         "triggered_after": p["triggered_after"],
                         "deficit": deficit, "age": age,
                         "eligible": eligible,
                         "episode_serves": sched["episode"][p["item"]]["serves"]})
        snap.sort(key=lambda q: sort_key(q, sched["first_trip"]))
        # starvation bookkeeping: probes sat eligible but unserved
        for t in trips:
            if t["served_in"] is None and not t.get("evicted"):
                for q in snap:
                    if (q["item"] == t["item"] and
                            q["triggered_after"] == t["triggered_after"] and
                            q["eligible"]):
                        t["elig_probes"] += 1
        # dead-ban counterfactual: eligible candidates with r > DEAD_R
        elig_live = [q for q in snap if q["eligible"]]
        banned = [q["item"] for q in elig_live if q["r_last"] <= DEAD_R]
        keep = [q for q in elig_live if q["r_last"] > DEAD_R]
        sched["dead_ban_trace"].append({
            "probe_after": k,
            "v3_would_serve": elig_live[0]["item"] if elig_live else None,
            "dead_banned": banned,
            "dead_ban_would_serve": keep[0]["item"] if keep else None,
        })
        # eviction watch: live r of every evicted (non-readmitted) item
        watch = {key: rs[key] for key in sched["evicted"]}
        eval_entry = {
            "r": rs, "tripped_at_thresh": new,
            "evicted_now": evicted_now,
            "held_resets": held,
            "brackets": {str(b): [key for key in ORDER if rs[key] < b]
                         for b in brackets},
            "queue_with_priority": snap,
            "dead_ban": sched["dead_ban_trace"][-1],
            "evicted_watch": watch,
        }
        v3_evals[f"after_s{k}"] = eval_entry
        one_line("SCHED", kind="probe",
                 gate=f"r<{args.thresh}+deficit-first+budget",
                 decision=("QUEUE" if new else
                           "EVICT" if evicted_now else "HOLD"),
                 reason=((", ".join(f"{k2} r={rs[k2]:.3f}" for k2 in new)
                          if new else "")
                         + (f" | evicted: {', '.join(evicted_now)}"
                            if evicted_now else "")
                         + (" | no new trips"
                            if not new and not evicted_now else "")),
                 extra=(f"queue (deficit order) "
                        f"{[(q['item'], round(q['deficit'], 3),
                             q['age'], 'E' if q['eligible'] else 'floor',
                             "ep" + str(q['episode_serves']))
                            for q in snap]}"
                        + (f" | dead-ban trace: {banned}" if banned else "")
                        + (f" | evicted watch: "
                           f"{[(k2, round(r2, 3)) for k2, r2 in watch.items()]}"
                           if watch else "")))
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
    # every trip event + its contact latency (the v2 hog/starve read)
    print("TRIP-CONTACT latency (every queue insertion):")
    starved_first, starved_retrip = [], []
    for t in trips:
        if t["served_in"] is not None:
            lat = t["served_in"] - t["triggered_after"]
            print(f"  {t['item']:12s} {t['kind']:8s} after S{t['triggered_after']} "
                  f"(first trip S{t['first_trip']}) -> served S{t['served_in']}, "
                  f"latency {lat}", flush=True)
        else:
            waiting = n_sessions - 1 - t["triggered_after"]
            tag = ("EVICTED at budget" if t.get("evicted")
                   else f"UNSERVED (eligible {t['elig_probes']} probes)")
            print(f"  {t['item']:12s} {t['kind']:8s} after S{t['triggered_after']} "
                  f"(first trip S{t['first_trip']}) -> {tag}, "
                  f"waited {waiting} probes", flush=True)
            if not t.get("evicted"):
                (starved_retrip if "re-trip" in t["kind"]
                 else starved_first).append(
                     (t["item"], waiting, t["elig_probes"]))
    print(f"STARVATION at horizon: unserved first-trips "
          f"{starved_first or 'NONE'}; unserved re-trips "
          f"{starved_retrip or 'NONE'}; blocked (evicted, no unaided hold) "
          f"re-trips {len(sched['blocked_trips'])}; readmissions "
          f"{len(sched['readmissions'])}", flush=True)
    # hog re-check: per-item serve totals vs v2's tea_tin 3/6
    serves_per = collections.Counter(e["item"] for e in events)
    print(f"HOG CHECK: serves per item {dict(serves_per)} "
          f"(v2 hog: tea_tin 3/6); max share "
          f"{max(serves_per.values()) if serves_per else 0}/{len(events)}",
          flush=True)
    evict_log = {k: {"probe": v["probe"], "episode_drs": v["episode_drs"],
                     "watch": v["watch"]}
                 for k, v in sched["evicted"].items()}
    print(f"EVICTIONS: {evict_log or 'none fired'}", flush=True)

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
        "arm": "aged-sched-v3-deficit-dominant-serve-budget", "ckpt": CKPT,
        "steps": args.steps,
        "seed": args.seed, "thresh": args.thresh, "brackets": brackets,
        "later_sessions": args.later,
        "scheduler": {
            "formula": "serve order = sort by (deficit desc, age-from-"
                       "first-trip desc [tie-break only], library order); "
                       "deficit_i(k) = max(0, thresh - r_i(k)) live; "
                       "eligible iff deficit > 0 and not evicted",
            "no_progress_def": "serve counts as no progress iff repair "
                               "d_r <= NO_PROGRESS_DR (diagnostic flag)",
            "no_progress_dr": NO_PROGRESS_DR,
            "serve_budget": SERVE_BUDGET,
            "eviction_rule": "queued item with >= SERVE_BUDGET serves since "
                             "its last held probe (r >= thresh) is evicted",
            "readmission_rule": "only after an UNAIDED held probe (r >= "
                                "thresh) followed by a fresh trip (decay "
                                "from a higher point); budget resets",
            "dead_r_reporting": DEAD_R,
            "serve_cap_per_session": 1,
        },
        "library": {k: {"kind": ITEMS[k]["kind"], "teach": ITEMS[k]["teach"]}
                    for k in ORDER},
        "order": ORDER, "decisions": decisions, "replay_state": replay_state,
        "before": before, "snapshots": snapshots, "gains": gains,
        "v3_evals": v3_evals, "events": events, "trips": trips,
        "dead_ban_trace": sched["dead_ban_trace"],
        "episodes": sched["episode"],
        "evictions": {k: {"probe": v["probe"], "episode_drs": v["episode_drs"],
                          "watch": v["watch"]}
                      for k, v in sched["evicted"].items()},
        "unaided_held": sched["unaided_held"],
        "blocked_trips": sched["blocked_trips"],
        "readmissions": sched["readmissions"],
        "serves_per_item": dict(serves_per),
        "pending_unserved": queue,
        "queue_wait_hist": wait_hist, "max_wait": max_wait,
        "starved_first_trips": starved_first,
        "starved_re_trips": starved_retrip,
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
    out_path = os.path.join(workdir, "teach_aged_sched_v3.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"results json: {out_path}", flush=True)


if __name__ == "__main__":
    main()
