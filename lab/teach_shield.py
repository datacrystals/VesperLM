#!/usr/bin/env python3
"""Shield-everything-every-session arm — the scheduling-question killer —
on the SFT-r3 chat checkpoint.

Follow-up to the three-arm scheduler falsification chain: the trigger arm
(lab/TEACH_TRIGGER_REPAIR.md, lowest-r-first queue) is the committed bar
(4/6 held >= 0.50, 0 dead, mean r_final 0.581); aged v1
(lab/TEACH_AGED_SCHED.md) and v2 (lab/TEACH_AGED_SCHED_V2.md) were
explicitly negative; v3 (lab/TEACH_AGED_SCHED_V3.md, deficit-dominant +
serve budget) was the best of the aged line and still negative, with the
line that matters: serve CONTENT dominated ORDER — every pure-priority rule
tested leaves the same structural residue (one item un-served or
horizon-edge served; horizon frozen mid-oscillation). The cadence arm
(TEACH_SPACED_CADENCE.md) and the decay A/B (TEACH_DECAY_REPLAY.md)
measured the complementary fact: small phrasing-diverse re-exposure works
per contact. This arm removes the queue entirely: shield EVERY alive item
in EVERY scheduler-phase session. No priorities, no thresholds, no queues,
no eviction.

SMALLEST HONEST DELTA off v3 (same import chain, same teach phase S1..S6
verbatim cadence runner bit-checked in-run, same tripwires/poison/tier-2/
fallback policy / seed discipline). The scheduler phase S7..S12 changes
shape (batch grows — that is the point of the arm) and the poison/tier-2
blocks are reused verbatim.

SHIELD PROTOCOL (exact rule as implemented, documented BEFORE running):

  alive(k)   = {item i : r_i(k) >= ALIVE_R = 0.10, re-evaluated at the
               probe after EVERY session S6..S11, no memory}. A dead item
               (r < 0.10) is buried: excluded from re-exposure per measured
               doctrine (aged v1 rescue: a strict-dead contact yielded
               +0.063 and re-decayed — contacts into dead items are wasted
               capacity); every burial is logged with the r that caused it.
               An item that self-recovers back across 0.10 re-enters the
               shield (alive is a per-probe predicate, NOT a sticky ban).
  session k+1 batch (scheduler phase, 5-approval filler + shield triples):

      n_filler  = max(5, ceil(alive_n / 0.30) - alive_n)
                    # >= 5 keeps the arms' generic-chat interference floor
                    # (the 1/6-poverty case: one alive item still gets a
                    # 5-filler batch — its 1 shield triple is 16.7%, the
                    # smallest share any shielded item sees);
                    # the ceil term guarantees shield share <= 30% exactly
      shield    = for EACH alive item in library ORDER: exactly ONE
                  approve-only triple whose response string is the pool
                  entry's correction phrasing (the phrasing-diverse
                  supervision-target form of the re-exposure unit in the
                  lineage; the natural script_ok form stays out of the
                  batch, keeping the unit at exactly 1 triple per item).
                  Variant = the shared replay_state counter, continuing
                  from the teach phase (variants 0..7 pool, assert < 8;
                  teach phase consumed 1 for fav_show..pet_dog, 0 for
                  greet_rule, shield adds at most 6).
      batch     = n_filler + alive_n triples; sizes: alive 6 -> 20 (30.0%),
                  5 -> 17 (29.4%), 4 -> 14 (28.6%), 3 -> 10 (30.0%),
                  2 -> 7 (28.6% = the trigger batch shape), 1 -> 6 (16.7%).
      share     = alive_n / batch <= 0.30, asserted per session.

  FILLER ALLOCATION (deviation, documented): the 30-entry generic-chat
  pool was sized for 5/session x 6 sessions; shield needs up to 14/session.
  A continuing cyclical pointer walks the pool across sessions (each batch
  still gets n_filler DISTINCT entries, since n_filler <= 30); strings
  repeat ACROSS sessions for the first time in the series. Filler is
  scripted generic approvals, not re-teach content; the tripwire that
  cared (unique_response_ratio, within-batch) cannot see cross-session
  reuse.

  Unchanged from the lineage: teach phase S1..S6 verbatim cadence runner
  (bit-check), 24 steps/session consolidate, lr 2e-3, kl 0.05, rank 8,
  alpha 16, seed 7 on every consolidate call, stub tier-1 gate, poison
  canary, one tier-2 Immune merge verdict, one-fallback tripwire policy,
  CPU fp32 (Immune never leaves CPU, I7).

MONITORS (named before running):
  - bar: >= 4/6 held r_final >= 0.50, 0 dead (r <= 0.06), mean >= 0.581
    — the trigger arm's criteria, unchanged.
  - base-class interference: does blanket shielding hurt the items the
    trigger served against? Per-item r trajectories/horizon vs the
    committed trigger table (reused, not re-run); shield-phase train-loss
    and top_response_fraction profiles stay tripwire-cold.
  - the 1/6-poverty case: if alive_n drops toward 1, log batch composition
    and the single item's shield share (noise share check).
  - shield contact log: every alive item's per-session r around each
    contact (contact yields are bystander-coupled under blanket shielding
    — reported as a trace, not a clean per-contact repair like the series'
    1-serve-per-session arms).

   Claims honesty (named before running): if shield meets the trigger bar
   on all criteria, the queue scheduler is OBSOLETE and the answer is the
   slice constant; if it falls short, the verdict is
   "architecture-straw-limit, schedules dead, adaptive capacity next" and
   the follow-up (multi-adapter banks / class-segregated adapters vs a
   capacity knob) is named from the measured shield trace.

Usage:
  PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
    python3 -u lab/teach_shield.py --workdir lab/teach_shield_out

  # determinism calibration (run BEFORE the arm): identical teach twice,
  # must bit-match each other and all five prior harnesses' committed
  # seed-7 calibration for the same item:
  ... teach_shield.py --only metric_units --seed 7 --workdir <fresh A>
  ... teach_shield.py --only metric_units --seed 7 --workdir <fresh B>

Fresh workdir required. CPU only, fp32.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
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
generate_text = cad.generate_text

ITEMS = cad.ITEMS
ORDER = cad.ORDER
SESSIONS = cad.SESSIONS
FILLER = cad.FILLER
USER = cad.USER
DEVICE = cad.DEVICE
REPLAY_CAP = cad.REPLAY_CAP  # 0.30
REPLAY = trig.REPLAY         # the 8-variant extended pools (trigger arm verbatim)

ALIVE_R = 0.10           # alive line: buried below this at the last probe
DEAD_R = 0.06            # strict dead-slot line (reporting)
FILLER_MIN = 5           # interference floor (the arms' per-session filler count)


def n_filler_for(n_alive):
    return max(FILLER_MIN, math.ceil(n_alive / REPLAY_CAP) - n_alive)


class FillerCycle:
    """Continuing cyclical pointer over the 30-entry pool; every batch gets
    n <= 30 distinct entries (reuse is cross-session only, documented)."""

    def __init__(self):
        self.ptr = 0

    def take(self, n):
        assert n <= len(FILLER), "within-batch filler must stay distinct"
        out = [FILLER[(self.ptr + i) % len(FILLER)] for i in range(n)]
        self.ptr += n
        return out


def log_shield_turn(log, model, tok, sid, turn_id, item, variant):
    """ONE approve-only triple: response = the pool entry's correction
    phrasing (1 triple per alive item, vs the lineage's 2-triple replay
    turn). Generation kept for transcript continuity (transcript colour)."""
    q, _script_ok, correction = REPLAY[item][variant]
    stop = [tok.convert_tokens_to_ids("im" + "_end"), tok.eos_token_id]
    gen = generate_text(model, tok, user_turn(q), max_new_tokens=24,
                        device=DEVICE, stop_ids=stop)
    log.log_turn(sid, turn_id, user_turn(q), correction + IM_E)
    log.mark_feedback(sid, turn_id, "approve", confidence=1.0,
                      note=f"shield re-exposure of {item} (1-triple)")
    print(f"  shield turn {turn_id} [{item}] v{variant}: Q={q!r}", flush=True)
    print(f"           model said : {gen!r}", flush=True)
    print(f"           approve    : {correction!r}", flush=True)
    return 1


def run_shield_session(workdir, model, tok, idx, args, alive, filler_cycle,
                       snapshots, decisions, replay_state, shield_events,
                       before, gains):
    """One shield-phase session: n_filler generic-chat approvals + exactly
    one shield triple per alive item. No queue, no priority."""
    sid = f"sess-shield-{idx}"
    log = SessionLog(os.path.join(workdir, f"session_{idx:02d}_{sid}.jsonl"))
    filler = filler_cycle.take(n_filler_for(len(alive)))
    cad.anneal_session(log, model, tok, sid, filler)
    n_shield = 0
    turn_id = len(filler)
    used = {}
    for item in alive:
        variant = replay_state[item]
        replay_state[item] += 1
        assert variant < len(REPLAY[item]), "replay pool exhausted"
        used[item] = variant
        n_shield += log_shield_turn(log, model, tok, sid, turn_id,
                                    item, variant)
        turn_id += 1
    seed = None if args.seed < 0 else args.seed
    res = consolidate(log, user_id=USER, workdir=workdir, device=DEVICE,
                      steps=args.steps, lr=2e-3, batch_size=4, kl_coef=0.05,
                      rank=8, alpha=16.0, base_model=model, tokenizer=tok,
                      target_profile=demo.TARGET_PROFILE, gate_fn=stub_gate,
                      seed=seed)
    share = n_shield / res.n_triples
    assert share <= REPLAY_CAP + 1e-9, f"shield share {share:.3f} over budget"
    decisions[f"s{idx}"] = res.decision
    if not apply_user_adapter(model, workdir, USER,
                              target_profile=demo.TARGET_PROFILE):
        demo.restore_raw_base(model)
    snapshots[f"after_s{idx}"] = cad.probe_all(model, tok, ORDER)
    st = res.train_stats
    one_line(
        f"S{idx}", kind=f"shield/{len(alive)}-alive", gate="stub-tier1",
        decision=res.decision, reason=res.reason,
        extra=(f"top_resp_frac {st.get('top_response_fraction'):.2f} "
               f"n_triples {res.n_triples} (shield {n_shield}, "
               f"{share:.0%} over {alive}) train loss "
               f"{st['loss0']:.3f}->{st['loss_final']:.3f}"))
    for item in alive:
        r_pre = trig.r_of(before, gains, snapshots, f"after_s{idx - 1}", item)
        r_post = trig.r_of(before, gains, snapshots, f"after_s{idx}", item)
        shield_events.append({
            "item": item, "session": idx, "variant": used[item],
            "n_alive": len(alive), "batch": res.n_triples,
            "share": round(share, 4),
            "r_pre": r_pre, "r_post": r_post,
            "contact_dr": r_post - r_pre})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None,
                    help="fresh/absent (default lab/teach_shield_out)")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7,
                    help="train-loop seed; -1 disables seeding (pre-fix behaviour)")
    ap.add_argument("--thresh", type=float, default=0.50,
                    help="held line for the horizon read (the hinge, unchanged)")
    ap.add_argument("--later", type=int, default=6,
                    help="shield-phase sessions (>=6, matches every prior arm)")
    ap.add_argument("--only", default=None, choices=ORDER + ["none"],
                    help="calibration: teach ONLY this item (single session)")
    ap.add_argument("--no-poison", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()

    workdir = os.path.abspath(args.workdir or os.path.join(
        LAB, "teach_shield_out"))
    if os.path.exists(workdir):
        sys.exit(f"refusing to run into existing workdir {workdir} "
                 f"(incumbent contamination) — delete it or pass a fresh --workdir")
    os.makedirs(workdir)

    t0 = time.time()
    model, tok, mc = load_base_model(CKPT, device=DEVICE, target_profile="kda_mla")
    assert mc.get("linear_type") == "kda" and mc.get("full_type") == "mla", (
        f"expected kda_mla ckpt, got {mc.get('linear_type')}/{mc.get('full_type')}")
    print(f"shield arm: {os.path.relpath(CKPT, REPO)} "
          f"(dim {mc['dim']}, {mc['n_layers']}L kda/mla, CPU fp32); "
          f"seed {'OFF (-1, pre-fix)' if args.seed < 0 else args.seed}; "
          f"shield every alive item (r >= {ALIVE_R}) every session, "
          f"1 triple/item, share <= {REPLAY_CAP:.0%}, "
          f"filler = max({FILLER_MIN}, ceil(n/{REPLAY_CAP}) - n); "
          f"workdir {workdir}", flush=True)
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
        # calibration path: one teach session, no replay — same flow as all
        # five prior harnesses so all six calibrations cross-check.
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

    assert args.later >= 6, "shield phase wants >=6 later sessions"
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

    # ------- shield phase S7..S12: every alive item, every session -------
    shield_events = []     # per (session, alive item) contact trace
    alive_log = []         # per-probe alive/buried sets
    burials = []           # every burial event (item, probe, r)
    filler_cycle = FillerCycle()
    tripwire_abort = None

    def evaluate(k):
        """Probe after session k -> the alive set for session k+1."""
        rs = {key: trig.r_of(before, gains, snapshots, f"after_s{k}", key)
              for key in ORDER}
        alive = [key for key in ORDER if rs[key] >= ALIVE_R]
        buried = [key for key in ORDER if rs[key] < ALIVE_R]
        for key in buried:
            if not any(b["item"] == key and b["still_buried"] for b in burials):
                burials.append({"item": key, "probe": k, "r": rs[key],
                                "still_buried": True})
        for b in burials:
            if b["item"] not in buried:
                b["still_buried"] = False
        n_alive = len(alive)
        plan = {"probe_after": k, "r": rs, "alive": alive, "buried": buried,
                "n_filler": n_filler_for(n_alive),
                "batch": n_filler_for(n_alive) + n_alive}
        alive_log.append(plan)
        one_line("SHIELD", kind="probe",
                 gate=f"alive r>={ALIVE_R}",
                 decision="SHIELD",
                 reason=(f"alive {alive}" + (f" | BURIED {[(k2, round(rs[k2], 3)) for k2 in buried]}" if buried else "")),
                 extra=(f"batch plan: {plan['n_filler']} filler + "
                        f"{n_alive} shield triples = {plan['batch']} "
                        f"(share {n_alive / plan['batch']:.0%})"))
        return alive

    alive = evaluate(len(ORDER))   # probe after S6 seeds the S7 batch
    for idx in range(len(ORDER) + 1, n_sessions + 1):
        res = run_shield_session(workdir, model, tok, idx, args, alive,
                                 filler_cycle, snapshots, decisions,
                                 replay_state, shield_events, before, gains)
        if res.decision == "ROLLBACK":
            tripwire_abort = (idx, res.reason)
            print(f"TRIPWIRE EVENT S{idx}: {res.reason} — fallback budget armed "
                  f"(one gate-safe spaced re-teach max); aborting for analyst "
                  f"decision, see lab/FAILURES.md", flush=True)
            break
        if idx < n_sessions:
            alive = evaluate(idx)

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
          f"mean r_final {mean_final:.3f}; min {min_final:.3f}", flush=True)

    # shield contact trace: per item, r around every shield session
    contacts = collections.Counter(e["item"] for e in shield_events)
    print(f"SHIELD CONTACTS per item: {dict(contacts)} "
          f"(target: one per alive-probe session)", flush=True)
    min_alive = min(len(p["alive"]) for p in alive_log) if alive_log else None
    if min_alive is not None:
        print(f"POVERTY CHECK: min alive set size {min_alive}; batches "
              f"{[(p['probe_after'], len(p['alive']), p['batch']) for p in alive_log]}",
              flush=True)

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
        "arm": "shield-everything-every-session", "ckpt": CKPT,
        "steps": args.steps,
        "seed": args.seed, "thresh_held": args.thresh,
        "alive_r": ALIVE_R, "dead_r_reporting": DEAD_R,
        "later_sessions": args.later,
        "shield": {
            "rule": "every alive item gets exactly ONE approve-only shield "
                    "triple (response = pool correction phrasing) in every "
                    "scheduler-phase session; no queue/priority/threshold/"
                    "eviction",
            "alive_def": f"r_i(k) >= ALIVE_R at the probe after session k, "
                         f"re-evaluated every probe, no memory; buried "
                         f"items re-enter on unaided recovery across ALIVE_R",
            "filler_rule": f"n_filler = max({FILLER_MIN}, ceil(alive/"
                           f"{REPLAY_CAP}) - alive); cyclical pool pointer "
                           f"(cross-session reuse documented)",
            "batch_sizes": {str(n): {"filler": n_filler_for(n),
                                     "batch": n_filler_for(n) + n,
                                     "share": round(n / (n_filler_for(n) + n), 4)}
                            for n in range(1, 7)},
        },
        "library": {k: {"kind": ITEMS[k]["kind"], "teach": ITEMS[k]["teach"]}
                    for k in ORDER},
        "order": ORDER, "decisions": decisions, "replay_state": replay_state,
        "before": before, "snapshots": snapshots, "gains": gains,
        "alive_log": alive_log, "burials": burials,
        "shield_events": shield_events,
        "contacts_per_item": dict(contacts),
        "retention": tables, "class_stats": class_stats,
        "horizon": {"n_above_thresh": n_above, "thresh": args.thresh,
                    "n_dead_slot": n_dead, "dead_slot_r": DEAD_R,
                    "mean_r_final": mean_final, "min_r_final": min_final},
        "cadence_ident": cadence_ident,
        "poison_decision": poison_decision, "poison_reason": poison_reason,
        "incumbent_intact": intact,
        "merge": merge_metrics, "merge_verdict": merge_verdict,
        "tripwire_abort": tripwire_abort,
        "wall_sec": round(time.time() - t0, 1),
    }
    out_path = os.path.join(workdir, "teach_shield.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"results json: {out_path}", flush=True)


if __name__ == "__main__":
    main()
