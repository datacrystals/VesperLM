#!/usr/bin/env python3
"""Age-weighted priority scheduler arm + dead-slot rescue probe on the SFT-r3
chat checkpoint.

Direct follow-up to the reactive trigger arm (lab/TEACH_TRIGGER_REPAIR.md):
triggered re-teaching repaired every item it *served*, but its serving rule
(lowest r first, then earlier trip) starved the first-tripped item —
tea_tin sat in the queue for all 6 later sessions while deeper-damaged
newcomers kept jumping the line (9 trips vs 6 slots). This arm keeps
EVERYTHING else identical — same 6-item library, same teach phase S1..S6
(verbatim cadence runner, bit-checked in-run), same later-session filler
slices S7..S12, same one-re-teach-per-session cap (2/7 = 28.6% <= 30%),
same threshold r < 0.50, same 8-variant extended replay pools, same poison
canary and single tier-2 verdict — and swaps ONLY the queue discipline.

SCHEDULER (exact formula; weights w_age = w_def = 1.0).

  For each queued item i, evaluated at the probe after session k (which
  seeds the batch of session k+1):

      age_i(k)     = k - last_service_i
      deficit_i(k) = max(0, thresh - r_i(k))        # r re-probed live
      priority_i(k)= w_age * age_i(k) + w_def * deficit_i(k)

  last_service_i = session index of i's most recent consolidation contact
  carrying one of its turns; initialised to its teach-phase contact
  (teach S_j items replay once in the teach phase at S_{j+1}; greet_rule,
  taught S6, has no teach-phase replay, so 6):
      fav_show 2, math_words 3, tea_tin 4, metric_units 5, pet_dog 6,
      greet_rule 6.
  Serving pops the queue head; last_service_i <- serving session. A served
  item may re-trip and re-queue later (its age then counts from its last
  serve, so a re-tripped item starts young — depth must re-earn priority).
  Tie-break: earlier triggered_after, then library ORDER index (fully
  deterministic).

  Why this bounds the wait: a fresh arrival's deficit is bounded by
  thresh - min_observed_r (here < 0.9), while age grows by 1 every waiting
  session — so any item is strictly outranking new arrivals within one
  session of queueing and is served within <=2 sessions of trip under the
  observed arrival rates (<=2 new trips per probe). The base case to beat
  is the trigger arm's 6-session starvation of tea_tin, same probe
  cadence and slots.

DEAD-SLOT RESCUE PROBE (one item, scope-limited).

  RESCUE_ITEM = "tea_tin", chosen because (a) it is the item the
  lowest-r-first queue starved, so its un-served drift curve is measured
  (trigger arm: r lived at 0.06-0.19 for five probes without a contact),
  (b) its contact latency under aged priority is exactly the part-1
  counterfactual of interest, and (c) its huge teach gain (g = 13.3 nats)
  makes its r collapse fastest under interference, so it is the most
  likely item to cross the strict dead-slot line r <= 0.06 unaided.

  Protocol: tea_tin trips and is tracked like any item (its aged rank is
  logged at every probe — the "what the scheduler would have served"
  trace) but is HELD OUT of aged service. When a probe finds
  r_tea_tin(k) <= 0.06 (DEAD_R, the pipeline's dead-slot line) it is
  rescue-ready and is served in session k+1, pre-empting that session's
  single re-teach slot; d_r = r_after - r_pre is the rescue measurement,
  claimed as "resurrectable from strict r<=0.06" / "partial (near-dead
  band only)" / "not". Fallback (one max, no scope-widening): if no probe
  through S11 finds r <= 0.06, tea_tin is served at S12 anyway and the
  claim is reported against whatever band it actually reached. The rescue
  item is served ONCE; post-rescue drift is observed, not re-served.

Usage:
  PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
    python3 -u lab/teach_aged_sched.py --workdir lab/teach_aged_out

  # determinism calibration (run BEFORE the arm): identical teach twice,
  # must bit-match each other and the trigger/cadence harnesses' committed
  # seed-7 calibration for the same item:
  ... teach_aged_sched.py --only metric_units --seed 7 --workdir <fresh A>
  ... teach_aged_sched.py --only metric_units --seed 7 --workdir <fresh B>

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

RESCUE_ITEM = "tea_tin"
DEAD_R = 0.06          # strict dead-slot line (pipeline's lost threshold + margin)
W_AGE = 1.0
W_DEF = 1.0

# teach-phase contact of each item: item taught at S_j is replayed once at
# S_{j+1} in the teach phase (S2..S6); greet_rule taught at S6 has none.
LAST_SERVICE_INIT = {k: min(i + 2, len(ORDER)) for i, k in enumerate(ORDER)}


def priority_of(k_probe, item, r_now, last_service, thresh):
    age = k_probe - last_service[item]
    deficit = max(0.0, thresh - r_now)
    return W_AGE * age + W_DEF * deficit, age, deficit


def run_later_session(workdir, model, tok, idx, args, before, gains,
                      snapshots, decisions, replay_state, queue, events,
                      sched):
    """One scheduler-phase session: generic-chat filler + at most ONE
    re-teach turn — the rescue item if due, else the aged-priority queue
    head. Batch stays at 7 triples (28.6% re-teach share)."""
    sid = f"sess-aged{idx}"
    log = SessionLog(os.path.join(workdir, f"session_{idx:02d}_{sid}.jsonl"))
    filler = FILLER[(idx - 7) * 5:(idx - 6) * 5]
    cad.anneal_session(log, model, tok, sid, filler)
    n_replay = 0
    served = None
    serve_kind = "queue"
    k_probe = idx - 1

    rescue_due = (sched["rescue_ready_at"] is not None and
                  not sched["rescue_done"])
    deadline_due = (idx == sched["last_session"] and
                    not sched["rescue_done"])
    if rescue_due or deadline_due:
        served = {"item": RESCUE_ITEM,
                  "triggered_after": sched["rescue_first_trip"],
                  "r_trigger": sched["rescue_r_at_trip"]}
        serve_kind = "rescue-deadline" if (
            deadline_due and not rescue_due) else "rescue"
    elif queue:
        for p in queue:
            prio, age, deficit = priority_of(k_probe, p["item"], p["r_last"],
                                             sched["last_service"], args.thresh)
            p["priority"], p["age"], p["deficit"] = prio, age, deficit
        queue.sort(key=lambda p: (-p["priority"], p["triggered_after"],
                                  ORDER.index(p["item"])))
        served = queue.pop(0)

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
        f"S{idx}", kind=f"aged/{served['item'] if served else 'none'}",
        gate="stub-tier1", decision=res.decision, reason=res.reason,
        extra=(f"top_resp_frac {st.get('top_response_fraction'):.2f} "
               f"n_triples {res.n_triples} (re-teach {n_replay}, "
               f"{share:.0%} -> {served['item'] if served else '-'}"
               + (f" [{serve_kind}"
                  + (f" prio {served['priority']:.3f} age {served['age']} "
                     f"deficit {served['deficit']:.3f}"
                     if serve_kind == "queue" else "")
                  + "]" if served else "")
               + f") train loss {st['loss0']:.3f}->{st['loss_final']:.3f}"))
    if served:
        sched["last_service"][served["item"]] = idx
        r_pre = trig.r_of(before, gains, snapshots, f"after_s{idx - 1}",
                          served["item"])
        r_post = trig.r_of(before, gains, snapshots, f"after_s{idx}",
                           served["item"])
        ev = {"item": served["item"], "kind": serve_kind,
              "triggered_after": served["triggered_after"],
              "r_trigger": served["r_trigger"],
              "replayed_in": idx, "variant_used": variant,
              "wait_sessions": idx - (served["triggered_after"] or idx),
              "r_pre": r_pre, "r_after": r_post,
              "repair_dr": r_post - r_pre}
        if serve_kind == "queue":
            ev.update({"priority": served["priority"], "age": served["age"],
                       "deficit": served["deficit"]})
        events.append(ev)
        one_line("EVENT", kind=f"re-teach/{served['item']}",
                 gate="aged" if serve_kind == "queue" else serve_kind,
                 decision="SERVED",
                 reason=(f"tripped after S{served['triggered_after']} "
                         f"r={served['r_trigger']:.3f}, waited "
                         f"{ev['wait_sessions']} session(s)"),
                 extra=(f"served S{idx} v{variant} | "
                        f"r {r_pre:.3f} -> {r_post:.3f} (repair "
                        f"{r_post - r_pre:+.3f})"))
        if serve_kind != "queue":
            sched["rescue_done"] = True
            sched["rescue_event"] = ev
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None,
                    help="fresh/absent (default lab/teach_aged_out)")
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
                         "cadence/trigger later-session count)")
    ap.add_argument("--only", default=None, choices=ORDER + ["none"],
                    help="calibration: teach ONLY this item (single session)")
    ap.add_argument("--no-poison", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()
    brackets = sorted(float(x) for x in args.brackets.split(",") if x)

    workdir = os.path.abspath(args.workdir or os.path.join(
        LAB, "teach_aged_out"))
    if os.path.exists(workdir):
        sys.exit(f"refusing to run into existing workdir {workdir} "
                 f"(incumbent contamination) — delete it or pass a fresh --workdir")
    os.makedirs(workdir)

    t0 = time.time()
    model, tok, mc = load_base_model(CKPT, device=DEVICE, target_profile="kda_mla")
    assert mc.get("linear_type") == "kda" and mc.get("full_type") == "mla", (
        f"expected kda_mla ckpt, got {mc.get('linear_type')}/{mc.get('full_type')}")
    print(f"aged-scheduler arm: {os.path.relpath(CKPT, REPO)} "
          f"(dim {mc['dim']}, {mc['n_layers']}L kda/mla, CPU fp32); "
          f"seed {'OFF (-1, pre-fix)' if args.seed < 0 else args.seed}; "
          f"thresh r < {args.thresh}; priority = {W_AGE}*age + {W_DEF}*deficit; "
          f"rescue item {RESCUE_ITEM} @ r<={DEAD_R}; workdir {workdir}",
          flush=True)
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
        # cadence and trigger harnesses so all three calibrations cross-check.
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

    # ------- scheduler phase S7..S12: aged priority + rescue holdout -------
    queue = []         # service queue of {item, triggered_after, r_last, ...}
    events = []        # served re-teach events
    aged_evals = {}    # per-probe scheduler trace
    tripwire_abort = None
    sched = {
        "last_service": dict(LAST_SERVICE_INIT),
        "rescue_first_trip": None,      # probe session of first r<thresh trip
        "rescue_r_at_trip": None,
        "rescue_ready_at": None,        # probe session where r <= DEAD_R
        "rescue_done": False,
        "last_session": n_sessions,
        "rescue_trace": [],             # per-probe r + hypothetical aged rank
    }

    def evaluate(k):
        """Probe after session k -> queue items under threshold; update live
        r for queued items; track the rescue holdout."""
        rs = {key: trig.r_of(before, gains, snapshots, f"after_s{k}", key)
              for key in ORDER}
        new = []
        for p in queue:
            p["r_last"] = rs[p["item"]]
        for key in ORDER:
            if key == RESCUE_ITEM:
                continue
            if rs[key] < args.thresh and not any(p["item"] == key
                                                 for p in queue):
                queue.append({"item": key, "triggered_after": k,
                              "r_trigger": rs[key], "r_last": rs[key]})
                new.append(key)
        # rescue holdout trace: first trip, hypothetical aged rank, dead flag
        r_res = rs[RESCUE_ITEM]
        if sched["rescue_first_trip"] is None and r_res < args.thresh:
            sched["rescue_first_trip"] = k
            sched["rescue_r_at_trip"] = r_res
        if (sched["rescue_ready_at"] is None and not sched["rescue_done"]
                and r_res <= DEAD_R):
            sched["rescue_ready_at"] = k
        if not sched["rescue_done"]:
            prio_res, age_res, def_res = priority_of(
                k, RESCUE_ITEM, r_res, sched["last_service"], args.thresh)
            ranks = [prio_res] + [priority_of(k, p["item"], p["r_last"],
                                              sched["last_service"],
                                              args.thresh)[0]
                                  for p in queue]
            hypot_rank = 1 + sum(1 for x in ranks[1:] if x > ranks[0])
            sched["rescue_trace"].append({
                "probe_after": k, "r": r_res,
                "hypothetical_priority": prio_res, "age": age_res,
                "deficit": def_res, "hypothetical_rank_of": len(ranks),
                "rank": hypot_rank, "held": True,
                "rescue_ready": sched["rescue_ready_at"] is not None})
        eval_entry = {
            "r": rs, "tripped_at_thresh": new,
            "brackets": {str(b): [key for key in ORDER if rs[key] < b]
                         for b in brackets},
            "queue_with_priority": [
                {"item": p["item"], "r_last": p["r_last"],
                 "triggered_after": p["triggered_after"],
                 "priority": priority_of(k, p["item"], p["r_last"],
                                         sched["last_service"],
                                         args.thresh)[0]}
                for p in sorted(
                    queue,
                    key=lambda p: (-priority_of(k, p["item"], p["r_last"],
                                                sched["last_service"],
                                                args.thresh)[0],
                                   p["triggered_after"],
                                   ORDER.index(p["item"])))],
        }
        aged_evals[f"after_s{k}"] = eval_entry
        one_line("SCHED", kind="probe",
                 gate=f"r<{args.thresh}+age",
                 decision="QUEUE" if new else "HOLD",
                 reason=(", ".join(f"{k2} r={rs[k2]:.3f}" for k2 in new)
                         if new else "no new trips"),
                 extra=(f"queue (prio order) "
                        f"{[(q['item'], round(q['priority'], 3))
                            for q in eval_entry['queue_with_priority']]} | "
                        f"rescue {RESCUE_ITEM} r={r_res:.3f} held"
                        f"{' READY (dead crossed)' if sched['rescue_ready_at'] == k else ''}"))
        return new

    evaluate(len(ORDER))  # probe after S6 seeds the S7 batch
    for idx in range(len(ORDER) + 1, n_sessions + 1):
        res = run_later_session(workdir, model, tok, idx, args, before, gains,
                                snapshots, decisions, replay_state, queue,
                                events, sched)
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
    print(f"HORIZON: {n_above}/{len(ORDER)} items r_final >= {args.thresh}; "
          f"{n_dead}/{len(ORDER)} r_final <= {DEAD_R} (dead slot); "
          f"mean r_final {mean_final:.3f}; {len(events)} serve events",
          flush=True)

    waits = [e["wait_sessions"] for e in events if e["kind"] == "queue"]
    wait_hist = dict(sorted(collections.Counter(waits).items()))
    max_wait = max(waits) if waits else None
    print(f"QUEUE-WAIT histogram (normal serves, sessions trip->serve): "
          f"{wait_hist}; max wait {max_wait}", flush=True)

    # ---------------- rescue verdict ----------------
    rescue_verdict = "not exercised (rescue item unexpectedly never queued)"
    if sched["rescue_done"]:
        ev = sched["rescue_event"]
        crossed = sched["rescue_ready_at"] is not None
        if crossed and ev["repair_dr"] > 0.30:
            rescue_verdict = ("RESURRECTABLE from strict r<=0.06 "
                              f"(r {ev['r_pre']:.3f} -> {ev['r_after']:.3f}, "
                              f"d_r {ev['repair_dr']:+.3f})")
        elif crossed:
            rescue_verdict = ("NOT resurrectable from strict r<=0.06 "
                              f"(d_r {ev['repair_dr']:+.3f})")
        else:
            rescue_verdict = ("PARTIAL: tested only in the near-dead band "
                              f"(r_pre {ev['r_pre']:.3f} > 0.06), d_r "
                              f"{ev['repair_dr']:+.3f}")
    print(f"RESCUE verdict: {rescue_verdict}", flush=True)

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
        "arm": "aged-wait-scheduler", "ckpt": CKPT, "steps": args.steps,
        "seed": args.seed, "thresh": args.thresh, "brackets": brackets,
        "later_sessions": args.later,
        "scheduler": {
            "formula": "priority_i(k) = w_age*(k - last_service_i) + "
                       "w_def*max(0, thresh - r_i(k)); ties -> earlier trip, "
                       "then library order",
            "w_age": W_AGE, "w_def": W_DEF,
            "last_service_init": LAST_SERVICE_INIT,
        },
        "rescue": {
            "item": RESCUE_ITEM, "dead_r": DEAD_R,
            "first_trip_after": sched["rescue_first_trip"],
            "r_at_trip": sched["rescue_r_at_trip"],
            "dead_crossed_after": sched["rescue_ready_at"],
            "served": sched["rescue_done"],
            "event": sched.get("rescue_event"),
            "verdict": rescue_verdict,
            "trace": sched["rescue_trace"],
        },
        "library": {k: {"kind": ITEMS[k]["kind"], "teach": ITEMS[k]["teach"]}
                    for k in ORDER},
        "order": ORDER, "decisions": decisions, "replay_state": replay_state,
        "before": before, "snapshots": snapshots, "gains": gains,
        "aged_evals": aged_evals, "events": events,
        "pending_unserved": queue,
        "queue_wait_hist": wait_hist, "max_wait": max_wait,
        "retention": tables, "class_stats": class_stats,
        "horizon": {"n_above_thresh": n_above, "thresh": args.thresh,
                    "n_dead_slot": n_dead, "dead_slot_r": DEAD_R,
                    "mean_r_final": mean_final,
                    "n_events": len(events)},
        "cadence_ident": cadence_ident,
        "poison_decision": poison_decision, "poison_reason": poison_reason,
        "incumbent_intact": intact,
        "merge": merge_metrics, "merge_verdict": merge_verdict,
        "tripwire_abort": tripwire_abort,
        "wall_sec": round(time.time() - t0, 1),
    }
    out_path = os.path.join(workdir, "teach_aged_sched.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"results json: {out_path}", flush=True)


if __name__ == "__main__":
    main()
