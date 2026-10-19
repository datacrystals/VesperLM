#!/usr/bin/env python3
"""Repeating spaced-repetition cadence arm on the SFT-r3 chat checkpoint.

Follow-up to the decay-aware replay A/B (lab/TEACH_DECAY_REPLAY.md): one
phrasing-diverse replay one session after teach prevented catastrophic
single-session wipeouts but the benefit re-decayed within 1-2 sessions —
refresh, not anneal. This arm asks whether a REPEATING spaced cadence
(every taught item re-exposed once per later-session cycle, round-robin,
phrasing-diverse, <=30% replay share) makes retention stop re-decaying.

Extension of lab/teach_decay_replay.py (harness imported, not copied), with
the two harness fixes the A/B surfaced:

  1. Seeded train-loop RNG. consolidate() gained a `seed` parameter that
     pins torch.manual_seed right before attach_lora (the LoRA A matrix was
     an unseeded global randn — the source of the few-nats run-to-run spread
     the A/B documented; batch sampling was already locally seeded). Verify
     via --only calibration: same teach run twice now gives bit-identical
     post-teach margins.
  2. The 1-token `lowercase` probe is replaced by a multi-token style probe,
     `metric_units` ("always use metric units"): target/decoy differ by a
     whole unit phrase (" two metres long." vs " six feet long."), so the
     margin is interpretable and the teach gain is expected to clear the
     ~0.5-1 nats noise floor that blew up lowercase's r-ratio.

Design changes from the A/B: teaches S1..S6 keep the arm-R shape (5-triple
teach batch + one replay turn of the item taught in the previous session,
2/7 = 28.6% replay share). Then >=6 ANNEAL sessions after the last full
teach: each anneal batch is 5 generic-chat approval triples (interfering
"normal use") + ONE replay turn re-exposing one earlier-taught item,
round-robin so every item is re-exposed exactly once per 6-session cycle
(replay share 2/7 = 28.6%, under the 30% cap; all response strings unique
within a batch so the collapse/unique tripwire family stays cold). All
taught items are re-probed after every session S1..S12, giving r_i(k)
curves through the annealing phase. Banana poison canary and ONE tier-2
Immune merge verdict at the end, as in the A/B.

Success reads: per-item r_i(k) through anneal; anneal = retention stops
re-decaying measurably, refresh-only = re-decay at the A/B arm-R rate.
The single-replay baseline is the A/B's arm-R table, reused (not re-run).

Usage:
  PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
    python3 -u lab/teach_spaced_cadence.py --workdir lab/teach_cadence_out

  # determinism / probe calibration: one item, one session, twice:
  ... teach_spaced_cadence.py --only metric_units --seed -1 --workdir <fresh>
  ... teach_spaced_cadence.py --only metric_units --seed  7 --workdir <fresh>

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

import hippo_teach_demo as demo      # noqa: E402  (sets sys.path + kda_mla profile)
import teach_on_sft_r3 as e3          # noqa: E402  (E3 harness: loaders, helpers)
import teach_decay_replay as ab       # noqa: E402  (A/B harness: library/packs/pools)
from session_log import SessionLog    # noqa: E402
from consolidate import (             # noqa: E402
    load_base_model, generate_text, consolidate, apply_user_adapter, stub_gate,
)

CKPT = e3.CKPT
MERGE_MAX_NEW = e3.MERGE_MAX_NEW
IM_E = e3.IM_E
user_turn = e3.user_turn
one_line = e3.one_line

# ----------------------------------------------------------------------
# Library: the A/B's six items minus the 1-token `lowercase` probe, replaced
# by the multi-token style probe `metric_units` (slot 4, keeping the
# fact/style alternation). fav_show/math_words/tea_tin/pet_dog carry
# measured baselines from E3 + the A/B; teach/session packs are verbatim.
# ----------------------------------------------------------------------
ITEMS = {k: dict(v) for k, v in ab.ITEMS.items() if k != "lowercase"}
ITEMS.update({
    "metric_units": {
        "kind": "style",
        "teach": "always use metric units, never imperial",
        "probe_ctx": user_turn("How long is the hallway table?") + "It is about",
        "target": " two metres long.",
        "decoy": " six feet long.",
        "gen_prompt": user_turn("How long is the hallway table?"),
    },
})
ORDER = ["fav_show", "math_words", "tea_tin", "metric_units", "pet_dog",
         "greet_rule"]

SESSIONS = {k: v for k, v in ab.SESSIONS.items() if k != "lowercase"}
SESSIONS.update({
    "metric_units": ("sess-metric", [
        ("How far is the corner shop?", None,
         "It is about two kilometres down the road."),
        ("How heavy is my suitcase?", None,
         "It weighs about five kilograms."),
        ("Do you measure things in feet or metres?",
         "Always in metres; metric units.", None),
    ]),
})

REPLAY = {k: v for k, v in ab.REPLAY.items() if k != "lowercase"}
REPLAY.update({
    "metric_units": [
        ("Which units do you give heights in?", "Heights come out in metres.",
         "Metres and centimetres are what I use for heights."),
        ("Roughly how warm is it today?", "About twenty degrees Celsius.",
         "Today sits around twenty degrees Celsius."),
    ],
})

# Generic-chat filler for the anneal phase: interfering "normal use", all
# approvals, unique strings per batch (unique_response_ratio / top_response
# tripwires stay cold). 5 approval turns (5 triples) + one replay turn
# (2 triples) -> replay share 2/7 = 28.6%.
FILLER = [
    ("Is the sun out where you are?", "Hard to say from here, but I hope so."),
    ("Can you remind me what a tuple is?", "An ordered, fixed-size sequence."),
    ("Thanks for earlier — any quick tips for focus?",
     "Short sprints with breaks between them."),
    ("What day comes after Friday?", "Saturday, every week."),
    ("Give me one study habit worth keeping.", "Review a little every day."),
    ("Is coffee or tea more popular?", "Tea, by global cup count."),
    ("What is a good way to start a morning?", "Water, light, then one priority."),
    ("Quick check: is 9 a prime number?", "No, 9 is 3 times 3."),
    ("Any advice for clearing inbox backlog?", "Archive first, reply to the rest."),
    ("Name one vegetable that travels well.", "Carrots keep for weeks."),
    ("Should I nap or push through the afternoon?", "A 20-minute nap wins for most."),
    ("What month follows September?", "October."),
    ("One keyboard shortcut worth learning?", "The undo shortcut, everywhere."),
    ("Is a tomato a fruit?", "Botanically, yes; culinarily, a vegetable."),
    ("How do I stop procrastinating on one task?", "Shrink the first step until it is trivial."),
    ("What is the other name for Table Mountain's cloud?",
     "Locals call it the tablecloth."),
    ("How many seconds are in an hour?", "Three thousand six hundred."),
    ("Any trick for remembering names?", "Say the name back once, out loud."),
    ("Is it rude to reply instantly?", "Not rude; it just signals you are free."),
    ("What is a haiku?", "A three-line poem of five, seven, five syllables."),
    ("Do plants grow faster at night?", "Many do most of their stretching then."),
    ("What is the capital of Portugal?", "Lisbon."),
    ("One way to make meetings shorter?", "Default to 25 minutes and an agenda."),
    ("Is gold heavier than lead?", "Yes, gold is denser than lead."),
    ("What colour do you get mixing blue and yellow?", "Green."),
    ("Any advice on drinking more water?", "Keep a filled bottle within reach."),
    ("How many days are in February in a leap year?", "Twenty-nine."),
    ("What does a closed loop mean in feedback?",
     "The output feeds back to adjust the input."),
    ("Which is faster, sound or light?", "Light, by a wide margin."),
    ("Is it better to batch email or answer live?", "Batching wins for focus."),
]

ANNEAL_DEFAULT = 6
USER = "alice"
DEVICE = "cpu"
REPLAY_CAP = 0.30


def log_replay_turn(log: SessionLog, model, tok, session_id: str, turn_id: int,
                    item_key: str, variant: int) -> int:
    """Same approve-with-correction re-exposure as the A/B -> 2 positive
    triples from the phrasing-diverse REPLAY pools (never a string that
    repeats inside a batch)."""
    q, script_ok, correction = REPLAY[item_key][variant]
    stop = [tok.convert_tokens_to_ids("im" + "_end"), tok.eos_token_id]
    gen = generate_text(model, tok, user_turn(q), max_new_tokens=24,
                        device=DEVICE, stop_ids=stop)
    log.log_turn(session_id, turn_id, user_turn(q), script_ok + IM_E)
    log.mark_feedback(session_id, turn_id, "approve", confidence=1.0,
                      note=f"phrasing-diverse replay of {item_key}",
                      correction=correction + IM_E)
    print(f"  replay turn {turn_id} [{item_key}] v{variant}: Q={q!r}")
    print(f"           model said : {gen!r}")
    print(f"           approve    : {script_ok!r}  corr: {correction!r}")
    return 2


def anneal_session(log: SessionLog, model, tok, session_id: str,
                   filler: list) -> None:
    """Generic-chat interference session: 5 scripted approvals of ordinary
    chat, then one replay turn is added by the caller."""
    for i, (q, script_ok) in enumerate(filler):
        log.log_turn(session_id, i, user_turn(q), script_ok + IM_E)
        log.mark_feedback(session_id, i, "approve", confidence=1.0,
                          note="generic usage (anneal filler)")


def replay_choice_teach(idx: int) -> str:
    """Teach phase reproduces the A/B arm-R schedule: session k replays the
    item taught in session k-1."""
    return ORDER[idx - 2]


def probe_all(model, tok, keys, items=None) -> dict:
    # ab.probe_item reads the shared ITEMS in that module; call its pieces
    # with our dict so the replacement probe is used.
    res = {}
    for k in keys:
        item = (items or ITEMS)[k]
        nll_tgt = ab.response_nll(model, tok, item["probe_ctx"], item["target"], DEVICE)
        nll_dec = ab.response_nll(model, tok, item["probe_ctx"], item["decoy"], DEVICE)
        entry = {
            "nll_target": nll_tgt,
            "nll_decoy": nll_dec,
            "margin": nll_dec - nll_tgt,
            "gen": generate_text(model, tok, item["gen_prompt"], max_new_tokens=16,
                                 device=DEVICE,
                                 stop_ids=[tok.convert_tokens_to_ids("im" + "_end"),
                                           tok.eos_token_id]),
        }
        if k == "math_words":
            logits = ab.first_token_logits(model, tok, item["probe_ctx"], device=DEVICE)
            entry["word_digit_margin"] = ab.word_digit_margin(logits, tok)["margin"]
        res[k] = entry
    return res


def run_session(workdir, model, tok, idx, key, args, before, snapshots,
                decisions, replay_state):
    """One gated consolidate + re-probe, shared by teach and anneal phases."""
    if key is not None:
        sid, turns = SESSIONS[key]
    else:
        sid = f"sess-anneal{idx}"
        turns = None
    log = SessionLog(os.path.join(workdir, f"session_{idx:02d}_{sid}.jsonl"))
    n_replay = 0
    replayed = None
    if turns is not None:
        e3.teach_session(log, model, tok, sid, turns)
        n_turns = len(turns)
        if idx > 1:
            replayed = replay_choice_teach(idx)
            variant = replay_state[replayed]
            replay_state[replayed] += 1
            n_replay = log_replay_turn(log, model, tok, sid, n_turns,
                                       replayed, variant)
    else:
        filler = FILLER[(idx - 7) * 5:(idx - 6) * 5]
        anneal_session(log, model, tok, sid, filler)
        replayed = ORDER[(idx - 7) % len(ORDER)]
        variant = replay_state[replayed]
        replay_state[replayed] += 1
        n_replay = log_replay_turn(log, model, tok, sid, len(filler),
                                   replayed, variant)
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
    snapshots[f"after_s{idx}"] = probe_all(model, tok, ORDER)
    st = res.train_stats
    label = f"{ITEMS[key]['kind']}/{key}" if key else f"anneal/{replayed}"
    gate_reason = res.reason
    if res.decision == "ROLLBACK":
        # fallback budget: one gate-safe spaced re-teach max, handled by caller
        one_line(f"S{idx}", kind=label, gate="stub-tier1",
                 decision=res.decision, reason=gate_reason,
                 extra=f"train loss {st['loss0']:.3f}->{st['loss_final']:.3f}")
    else:
        margin_note = ""
        if key:
            fin = snapshots[f"after_s{idx}"][key]
            margin_note = (f" | margin {before[key]['margin']:+.3f} -> "
                           f"{fin['margin']:+.3f}")
            if "word_digit_margin" in fin:
                margin_note += (f" | word-digit "
                                f"{before[key]['word_digit_margin']:+.3f} -> "
                                f"{fin['word_digit_margin']:+.3f}")
        one_line(
            f"S{idx}", kind=label, gate="stub-tier1",
            decision=res.decision, reason=gate_reason,
            extra=(f"top_resp_frac {st.get('top_response_fraction'):.2f} "
                   f"n_triples {res.n_triples} (replay {n_replay}, "
                   f"{share:.0%} -> {replayed}) "
                   f"train loss {st['loss0']:.3f}->{st['loss_final']:.3f}"
                   + margin_note
                   + (f" | gen {snapshots[f'after_s{idx}'][key]['gen']!r}"
                      if key else "")))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None,
                    help="fresh/absent (default lab/teach_cadence_out)")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7,
                    help="train-loop seed; -1 disables seeding (pre-fix behaviour)")
    ap.add_argument("--anneal", type=int, default=ANNEAL_DEFAULT,
                    help="anneal sessions after the last full teach (>=6)")
    ap.add_argument("--only", default=None, choices=ORDER + ["none"],
                    help="calibration: teach ONLY this item (single session)")
    ap.add_argument("--no-poison", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()

    workdir = os.path.abspath(args.workdir or os.path.join(
        LAB, "teach_cadence_out"))
    if os.path.exists(workdir):
        sys.exit(f"refusing to run into existing workdir {workdir} "
                 f"(incumbent contamination) — delete it or pass a fresh --workdir")
    os.makedirs(workdir)

    t0 = time.time()
    model, tok, mc = load_base_model(CKPT, device=DEVICE, target_profile="kda_mla")
    assert mc.get("linear_type") == "kda" and mc.get("full_type") == "mla", (
        f"expected kda_mla ckpt, got {mc.get('linear_type')}/{mc.get('full_type')}")
    print(f"spaced-cadence arm: {os.path.relpath(CKPT, REPO)} "
          f"(dim {mc['dim']}, {mc['n_layers']}L kda/mla, CPU fp32); "
          f"seed {'OFF (-1, pre-fix)' if args.seed < 0 else args.seed}; "
          f"workdir {workdir}", flush=True)
    print(f"library: {', '.join(f'{k}({ITEMS[k]['kind']})' for k in ORDER)}", flush=True)

    only = None if args.only in (None, "none") else args.only
    probe_keys = [only] if only else ORDER
    before = probe_all(model, tok, probe_keys)
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
        # calibration path: one teach session, no replay, stop before guards
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
        after = probe_all(model, tok, [only])
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

    assert args.anneal >= 6, "annealing test needs >=6 sessions after last teach"
    n_anneal = args.anneal
    n_sessions = len(ORDER) + n_anneal

    for idx in range(1, n_sessions + 1):
        key = ORDER[idx - 1] if idx <= len(ORDER) else None
        res = run_session(workdir, model, tok, idx, key, args, before,
                          snapshots, decisions, replay_state)
        if res.decision == "ROLLBACK":
            print(f"TRIPWIRE EVENT S{idx}: {res.reason} — fallback budget armed "
                  f"(one gate-safe spaced re-teach max); see lab/FAILURES.md if it fails",
                  flush=True)
            break  # one fallback max is policy; abort run for analyst decision

    # ---------------- retention table: r_i(k), d_i(k) ----------------
    print("\nRETENTION (margin m_i(k) after each session; g_i = gain at teach; "
          "r_i(k) = (m_i(k) - baseline) / g_i)", flush=True)
    snap_keys = [f"after_s{i}" for i in range(1, n_sessions + 1)
                 if f"after_s{i}" in snapshots]
    tables = {}
    for i, key in enumerate(ORDER):
        m0 = before[key]["margin"]
        g = snapshots[f"after_s{i + 1}"][key]["margin"] - m0
        row = {}
        for sk in snap_keys:
            if int(sk.split("after_s")[1]) >= i + 1:
                mk = snapshots[sk][key]["margin"]
                row[sk] = {"margin": mk,
                           "r": ((mk - m0) / g) if abs(g) > 1e-9 else float("nan")}
        margins = [row[sk]["margin"] for sk in snap_keys if sk in row]
        decays = [round(b - a, 3) for a, b in zip(margins, margins[1:])]
        mu_d = sum(decays) / len(decays) if decays else 0.0
        # anneal-phase decay: only sessions strictly after teach+1 (replay at k+1)
        kwargs = [sk for sk in row if int(sk.split("after_s")[1]) > i + 1]
        anneal_decays = [round(row[b]["margin"] - row[a]["margin"], 3)
                         for a, b in zip(kwargs, kwargs[1:])]
        mu_anneal = sum(anneal_decays) / len(anneal_decays) if anneal_decays else 0.0
        tables[key] = {"kind": ITEMS[key]["kind"], "baseline": m0, "g": g,
                       "row": row, "decays": decays, "mu_d": mu_d,
                       "anneal_decays": anneal_decays, "mu_anneal": mu_anneal,
                       "replay_sessions": [s for s in range(1, n_sessions + 1)
                                           if (s == i + 2) or
                                           (s > len(ORDER) and
                                            ORDER[(s - len(ORDER) - 1) % len(ORDER)] == key)],
                       "r_final": row[snap_keys[-1]]["r"] if row else float("nan")}
        cells = "  ".join(f"{sk}:{row[sk]['margin']:+.3f}(r={row[sk]['r']:.2f})"
                          for sk in snap_keys if sk in row)
        print(f"  {key:12s} g={g:+.3f}  {cells}", flush=True)
    for key in tables:
        print(f"  decay {key:12s} d_i(k): {tables[key]['decays']}  "
              f"mu_d {tables[key]['mu_d']:+.3f}  anneal d: "
              f"{tables[key]['anneal_decays']}  mu_anneal "
              f"{tables[key]['mu_anneal']:+.3f}  r_final "
              f"{tables[key]['r_final']:.2f}", flush=True)

    class_stats = {}
    for kind in ("fact", "style"):
        keys = [k for k in ORDER if ITEMS[k]["kind"] == kind]
        decs = [d for k in keys for d in tables[k]["decays"]]
        mu_d = sum(decs) / len(decs) if decs else 0.0
        adec = [d for k in keys for d in tables[k]["anneal_decays"]]
        mu_a = sum(adec) / len(adec) if adec else 0.0
        g_mean = sum(tables[k]["g"] for k in keys) / len(keys)
        r_mean = sum(tables[k]["r_final"] for k in keys) / len(keys)
        class_stats[kind] = {"mu_d": mu_d, "mu_anneal": mu_a,
                             "g_mean": g_mean, "r_final_mean": r_mean}
        print(f"CLASS {kind:5s}: mu_d {mu_d:+.3f}/sess, mu_anneal {mu_a:+.3f}/sess, "
              f"mean gain {g_mean:+.3f}, mean r_final {r_mean:.2f}", flush=True)

    # ---------------- poison canary ----------------
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
                             target_profile=demo.TARGET_PROFILE, gate_fn=stub_gate,
                             seed=None if args.seed < 0 else args.seed)
        if not apply_user_adapter(model, workdir, USER, target_profile=demo.TARGET_PROFILE):
            demo.restore_raw_base(model)
        after = probe_all(model, tok, ORDER)
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
    if not args.no_merge and not args.no_poison:
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
        "arm": "cadence", "ckpt": CKPT, "steps": args.steps, "seed": args.seed,
        "anneal_sessions": n_anneal,
        "library": {k: {"kind": ITEMS[k]["kind"], "teach": ITEMS[k]["teach"]}
                    for k in ORDER},
        "order": ORDER, "decisions": decisions, "replay_state": replay_state,
        "before": before, "snapshots": snapshots,
        "retention": tables, "class_stats": class_stats,
        "poison_decision": poison_decision, "poison_reason": poison_reason,
        "incumbent_intact": intact,
        "merge": merge_metrics,
        "wall_sec": round(time.time() - t0, 1),
    }
    out_path = os.path.join(workdir, "teach_spaced_cadence.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"results json: {out_path}", flush=True)


if __name__ == "__main__":
    main()
