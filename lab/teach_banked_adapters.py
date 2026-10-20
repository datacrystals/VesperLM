#!/usr/bin/env python3
"""Item 11 — class-segregated adapter banks (BANK) vs one-shared-LoRA (CTRL)
on the SFT-r3 chat checkpoint.

Staged at lab/TEACH_PIPELINE.md item 11 after the shield arm's explicitly
NEGATIVE verdict (lab/TEACH_SHIELD.md): the strongest possible service
policy at one shared adapter missed the trigger bar (2/6 held, 1 dead,
mean 0.495 vs bar 4/6 / 0 / 0.581) with the mechanism read as
batch-internal cross-item interference that no queue policy can schedule
out (S7 mass-damage inside one fully-shielded batch) and
style-concentrated residual loss (style r_final mean 0.286 vs fact 0.704).
This arm moves the isolation from the batch level to the WEIGHT level:
fact content and style content write into separate LoRA deltas, so a style
session's gradient cannot overwrite fact margins by construction.

LIBRARY AMENDMENT (formal, declared before running): metric_units is
REMOVED from the library — the metric_units × variant-1 pairing is 0-for-5
across five arms (trigger −0.325, aged-v1 −0.291, v2 −0.330, v3 −0.203,
shield −0.142); a content-side pin/skip of one variant was on the table
from shield §5.6, but the pair's lack of any positive counter-example
makes the ITEM unsalvageable as a probe (its library slot contributed
only noise-style reads: recovered only via targeted high-variance serves
in trigger/v2). Its S4 style slot is filled by ordinal_words ("always
write ordinals as words, never digits"), a digit↔word contrast like the
measured-good math_words family. Consequence: S1..S3 stay bit-comparable
to the committed cadence run (same batches, same probe content for the
carried items); S4+ diverges from the committed run BY DESIGN; the
trigger bar survives as the series reference row (its library differs in
slot 4 — reported as such, not as a like-for-like row).

PROTOCOL (documented before running, same for both arms):

  - library: fav_show (F), math_words (S), tea_tin (F),
    ordinal_words (S), pet_dog (F), greet_rule (S); teach phase S1..S6,
    verbatim cadence teach packs for the carried items, new pack for
    ordinal_words (2 corrections + 1 approval = 5 triples as always).
  - later phase S7..S12: SHIELD protocol verbatim — every alive item
    (r >= 0.10 at last probe, per-probe re-eval, burial doctrine from
    aged-v1: dead-band contacts are wasted capacity) gets exactly ONE
    approve-only shield triple (pool correction phrasing) per session.
  - knobs frozen: LoRA rank 8, alpha 16, lr 2e-3, kl_coef 0.05,
    24 steps/session, seed 7 on every consolidate call, CPU fp32;
    Immune tier-2 never leaves CPU (I7). Poison canary unchanged.
    One-fallback tripwire policy unchanged.

ARMS (internal A/B — same sessions, same batches, same seed):

  CTRL = one shared LoRA adapter (adapters/alice/delta.pt), the teach_
  shield.py protocol on the amended library. Its S1..S3 margins and
  carried-item baselines must BIT-MATCH the committed cadence run
  (in-run CADENCE-IDENT subset check, asserted): S4+ is a declared new
  baseline family from the library amendment.

  BANK = per-class adapter stores at adapters/alice/fact/delta.pt and
  adapters/alice/style/delta.pt (the staged hooks; user ids may contain
  "/"). The serving layer applies exactly ONE bank per item — the item's
  class — at probe time (horizon read = r under the OWNING bank).

  Per-session BANK flow (teach AND shield phases, fixed bank order
  [fact, style], deterministic):

      for b in (fact, style):
          log_b = the session's triples filtered to class-b
                  — teach sessions: the whole teach pack goes to the
                    taught item's bank; the chained replay turn (of the
                    previous item) goes to ITS class's bank
                  — shield sessions: filler slice (SHARED strings into
                    both banks — filler is the class-neutral control
                    interference environment, documented) + one shield
                    triple per alive class-b item
          skip b if log_b has no eligible triples (a bank's empty class
              sessions; e.g. style bank at S1)
          if bank b has NO incumbent delta yet: seed_fresh_branch(model)
              — re-pin torch.manual_seed(seed) and re-draw lora_A exactly
              as a fresh attach_lora would (torch.randn(shape)/rank in
              lora_modules() iteration order, which coincides with
              attach's DFS wrap order), then zero lora_B; without this,
              a virgin bank would start TRAINING from the other bank's
              live A/B (measured-by-construction contamination — the
              fix restores the series' fresh-call initial conditions)
          consolidate(log_b, user_id=f"alice/{b}") with the frozen knobs

      probes: apply the OWNING bank per class (apply_user_adapter; when a
      bank has no delta yet the branch is raw base) and probe that
      class's items; snapshots carry per-class margins under the owning
      bank. Horizon adds the ISOLATION read: every item also probed under
      the FOREIGN bank (12 margins: r_own vs r_off — the direct measure
      of whether class content leaks across the split).

  Poison canary: per serving store — CTRL once (series verbatim), BANK
  once per bank; incumbent intact asserted per store. Tier-2 Immune
  merge verdict: CTRL on the shared adapter; BANK on the FACT bank only
  (the segregation benefit question — fact margins protected from style
  writes is the mechanism under test; the style-bank verdict is skipped
  on wall grounds, documented).

SUCCESS CRITERIA (declared before running; the trigger bar = 4/6 held
>= 0.50, 0 dead r <= 0.06, mean >= 0.581 stays the series reference):
  VERDICT "architecture wins" iff ALL of:
   (a) BANK beats CTRL-internally on held count (2/6 expected from the
       shield reading -> target >= 4/6 ... honestly: BANK held > CTRL held),
   (b) BANK beats CTRL-internal on dead count (BANK dead < CTRL dead,
       targeting 0),
   (c) BANK ties-or-beats the trigger bar on held + dead (>= 4/6 held,
       0 dead).
  Any miss = honest negative with mechanism (selector ambiguity,
  filler leakage, 2-triple bank-log variance, etc.).

Usage:
  PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
    python3 -u lab/teach_banked_adapters.py --arm ctrl --workdir lab/teach_banks_ctrl_out
    python3 -u lab/teach_banked_adapters.py --arm bank --workdir lab/teach_banks_bank_out

  # determinism calibration BEFORE arming: same teach twice (BANK mode
  # exercises the new plumbing; its fav_show calib batch is the series S1
  # batch verbatim, so it must ALSO bit-match the committed series S1
  # margins +1.718 -> +4.451):
  ... teach_banked_adapters.py --arm bank --only fav_show --seed 7 --workdir <fresh A>
  ... teach_banked_adapters.py --arm bank --only fav_show --seed 7 --workdir <fresh B>

Fresh workdir required per run. CPU only, fp32 (Immune never leaves CPU, I7).
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
from lora import lora_modules         # noqa: E402
from consolidate import (             # noqa: E402
    load_base_model, consolidate, apply_user_adapter, stub_gate,
)

CKPT = e3.CKPT
MERGE_MAX_NEW = e3.MERGE_MAX_NEW
IM_E = e3.IM_E
user_turn = e3.user_turn
one_line = e3.one_line
generate_text = cad.generate_text
USER = cad.USER
DEVICE = cad.DEVICE
REPLAY_CAP = cad.REPLAY_CAP  # 0.30

# ---------------- amended library: metric_units OUT (0-for-5), ordinal IN
ITEMS = {k: dict(v) for k, v in cad.ITEMS.items() if k != "metric_units"}
ITEMS["ordinal_words"] = {
    "kind": "style",
    "teach": "always write ordinals as words, never digits",
    "probe_ctx": user_turn("Which window is open on top?") + "The",
    "target": " third window is on top.",
    "decoy": " 3rd window is on top.",
    "gen_prompt": user_turn("Which window is open on top?"),
}
ORDER = ["fav_show", "math_words", "tea_tin", "ordinal_words", "pet_dog",
         "greet_rule"]
SESSIONS = {k: v for k, v in cad.SESSIONS.items() if k != "metric_units"}
SESSIONS["ordinal_words"] = ("sess-ordinal", [
    ("Which position did my photo come in?", None,
     "Your photo came in third place."),
    ("Was my team first or second in the league?", None,
     "Your team finished second."),
    ("Do you write rankings as digits?",
     "I write them out as words, like third.", None),
])
REPLAY = {k: v for k, v in trig.REPLAY.items() if k != "metric_units"}
REPLAY["ordinal_words"] = [
    ("Which shelf is the dictionary on?", "It is on the fourth shelf.",
     "The dictionary sits on the fourth shelf."),
    ("What place is my ticket in the queue?", "Your ticket is fifth in the queue.",
     "You are fifth in line."),
    ("Which attempt worked?", "The sixth attempt worked.",
     "It was the sixth attempt that worked."),
    ("On which floor is the meeting?", "It is on the seventh floor.",
     "The meeting is on the seventh floor."),
    ("Which episode should I start with?", "Start with the eighth episode.",
     "Episode eight is the one to start."),
    ("Which house is the red one?", "The ninth house on the street is the red one.",
     "It is the ninth house on the street."),
    ("Where does my name appear on the list?", "Your name appears tenth on the list.",
     "You are tenth on the list."),
    ("Which gate does the flight leave from?", "It leaves from the first gate.",
     "The flight leaves from the first gate."),
]
for k in ORDER:
    assert len(REPLAY[k]) >= 7, f"replay pool for {k}: {len(REPLAY[k])} < 7"

BANKS = ("fact", "style")
FILLER = cad.FILLER
ALIVE_R = 0.10
DEAD_R = 0.06
FILLER_MIN = 5


def class_of(item):
    return ITEMS[item]["kind"]


def bank_user(b):
    return f"{USER}/{b}"


def n_filler_for(n_alive):
    return max(FILLER_MIN, math.ceil(n_alive / REPLAY_CAP) - n_alive)


def seed_fresh_branch(model, seed):
    """Re-create the series' fresh first-consolidate branch state for a
    virgin bank: re-pin the RNG and re-draw lora_A exactly the way a fresh
    attach_lora would (torch.randn(shape)/rank drawn in wrap order ==
    lora_modules() iteration order), then zero lora_B."""
    import torch
    torch.manual_seed(seed)
    mods = lora_modules(model)
    for m in mods.values():
        m.lora_A.data.copy_(torch.randn_like(m.lora_A) / m.rank)
        m.lora_B.data.zero_()


def apply_bank(model, workdir, b):
    """Activate bank b only. Banks are complete delta files (A and B), so
    applying replaces state exactly; a bank with no delta yet = raw base
    (B zeroed, A irrelevant)."""
    if not apply_user_adapter(model, workdir, bank_user(b),
                              target_profile=demo.TARGET_PROFILE):
        demo.restore_raw_base(model)


def has_delta(workdir, b):
    return os.path.exists(os.path.join(workdir, "adapters",
                                       *bank_user(b).split("/"), "delta.pt"))


def probe_items(model, tok, keys):
    return cad.probe_all(model, tok, keys, items=ITEMS)


def log_replay_turn_local(log, model, tok, session_id, turn_id, item, variant):
    """teach_spaced_cadence.log_replay_turn mirrored onto the LOCAL pools
    (the module globals do not contain ordinal_words)."""
    q, script_ok, correction = REPLAY[item][variant]
    stop = [tok.convert_tokens_to_ids("im" + "_end"), tok.eos_token_id]
    gen = generate_text(model, tok, user_turn(q), max_new_tokens=24,
                        device=DEVICE, stop_ids=stop)
    log.log_turn(session_id, turn_id, user_turn(q), script_ok + IM_E)
    log.mark_feedback(session_id, turn_id, "approve", confidence=1.0,
                      note=f"phrasing-diverse replay of {item}",
                      correction=correction + IM_E)
    return 2


def log_shield_turn_local(log, model, tok, sid, turn_id, item, variant):
    """teach_shield.log_shield_turn mirrored onto the LOCAL pools (ONE
    approve-only triple, response = pool correction phrasing)."""
    q, _script_ok, correction = REPLAY[item][variant]
    stop = [tok.convert_tokens_to_ids("im" + "_end"), tok.eos_token_id]
    gen = generate_text(model, tok, user_turn(q), max_new_tokens=24,
                        device=DEVICE, stop_ids=stop)
    log.log_turn(sid, turn_id, user_turn(q), correction + IM_E)
    log.mark_feedback(sid, turn_id, "approve", confidence=1.0,
                      note=f"shield re-exposure of {item} (1-triple)")
    return 1


class FillerCycle:
    def __init__(self):
        self.ptr = 0

    def take(self, n):
        assert n <= len(FILLER), "within-batch filler must stay distinct"
        out = [FILLER[(self.ptr + i) % len(FILLER)] for i in range(n)]
        self.ptr += n
        return out


def do_consolidate(log, uid, workdir, model, tok, args):
    seed = None if args.seed < 0 else args.seed
    return consolidate(log, user_id=uid, workdir=workdir, device=DEVICE,
                       steps=args.steps, lr=2e-3, batch_size=4, kl_coef=0.05,
                       rank=8, alpha=16.0, base_model=model, tokenizer=tok,
                       target_profile=demo.TARGET_PROFILE, gate_fn=stub_gate,
                       seed=seed)


# ============================================================ CTRL arm ====

def run_shield_session_ctrl(workdir, model, tok, idx, args, alive,
                            filler_cycle, snapshots, decisions,
                            replay_state, shield_events, before, gains):
    sid = f"sess-shield-{idx}"
    log = SessionLog(os.path.join(workdir, f"session_{idx:02d}_{sid}.jsonl"))
    filler = filler_cycle.take(n_filler_for(len(alive)))
    cad.anneal_session(log, model, tok, sid, filler)
    n_shield = 0
    turn_id = len(filler)
    served_variants = {}
    for item in alive:
        variant = replay_state[item]
        replay_state[item] += 1
        assert variant < len(REPLAY[item]), "replay pool exhausted"
        served_variants[item] = variant
        n_shield += log_shield_turn_local(log, model, tok, sid, turn_id,
                                          item, variant)
        turn_id += 1
    res = do_consolidate(log, USER, workdir, model, tok, args)
    share = n_shield / res.n_triples
    assert share <= REPLAY_CAP + 1e-9, f"shield share {share:.3f} over budget"
    decisions[f"s{idx}"] = res.decision
    if not apply_user_adapter(model, workdir, USER,
                              target_profile=demo.TARGET_PROFILE):
        demo.restore_raw_base(model)
    snapshots[f"after_s{idx}"] = probe_items(model, tok, ORDER)
    st = res.train_stats
    one_line(f"S{idx}", kind=f"ctrl/{len(alive)}-alive", gate="stub-tier1",
             decision=res.decision, reason=res.reason,
             extra=(f"top_resp_frac {st.get('top_response_fraction'):.2f} "
                    f"n_triples {res.n_triples} (shield {n_shield}, "
                    f"{share:.0%}) train loss "
                    f"{st['loss0']:.3f}->{st['loss_final']:.3f}"))
    for item in alive:
        r_pre = trig.r_of(before, gains, snapshots, f"after_s{idx - 1}", item)
        r_post = trig.r_of(before, gains, snapshots, f"after_s{idx}", item)
        shield_events.append({"item": item, "session": idx,
                              "variant": n_shield and replay_state[item] - 1,
                              "n_alive": len(alive), "batch": res.n_triples,
                              "share": round(share, 4),
                              "r_pre": r_pre, "r_post": r_post,
                              "contact_dr": r_post - r_pre})
    return res


def run_teach_session_ctrl(workdir, model, tok, idx, item, args, before,
                           snapshots, decisions, replay_state):
    """cad.run_session mirrored onto the LOCAL library."""
    sid, turns = SESSIONS[item]
    log = SessionLog(os.path.join(workdir, f"session_{idx:02d}_{sid}.jsonl"))
    e3.teach_session(log, model, tok, sid, turns)
    n_replay = 0
    if idx > 1:
        replayed = ORDER[idx - 2]
        variant = replay_state[replayed]
        replay_state[replayed] += 1
        n_replay = log_replay_turn_local(log, model, tok, sid, len(turns),
                                         replayed, variant)
    res = do_consolidate(log, USER, workdir, model, tok, args)
    share = n_replay / res.n_triples
    assert share <= REPLAY_CAP + 1e-9, f"replay share {share:.3f} over budget"
    decisions[f"s{idx}"] = res.decision
    if not apply_user_adapter(model, workdir, USER,
                              target_profile=demo.TARGET_PROFILE):
        demo.restore_raw_base(model)
    snapshots[f"after_s{idx}"] = probe_items(model, tok, ORDER)
    st = res.train_stats
    one_line(f"S{idx}", kind=f"ctrl/{ITEMS[item]['kind']}/{item}",
             gate="stub-tier1", decision=res.decision, reason=res.reason,
             extra=(f"top_resp_frac {st.get('top_response_fraction'):.2f} "
                    f"n_triples {res.n_triples} (replay {n_replay}, "
                    f"{share:.0%}) train loss "
                    f"{st['loss0']:.3f}->{st['loss_final']:.3f}"))
    return res


# ============================================================ BANK arm ====

def run_shield_session_bank(workdir, model, tok, idx, args, alive,
                            filler_cycle, snapshots, decisions,
                            replay_state, shield_events, before, gains,
                            virgin_banks):
    sid = f"sess-shield-{idx}"
    by_class = {b: [k for k in alive if class_of(k) == b] for b in BANKS}
    need = max(n_filler_for(len(by_class[b])) for b in BANKS)
    base_filler = filler_cycle.take(need)
    n_total = 0
    served_variants = {}
    for b in BANKS:
        items = by_class[b]
        sid_b = f"{sid}-{b}"
        log = SessionLog(os.path.join(
            workdir, f"session_{idx:02d}_{sid_b}.jsonl"))
        filler_b = base_filler[:n_filler_for(len(items))]
        if not items:
            one_line(f"S{idx}", kind=f"bank-{b}/skip", gate="stub-tier1",
                     decision="PROMOTE",
                     reason="no alive class items this session",
                     extra="bank skipped")
            continue
        cad.anneal_session(log, model, tok, sid_b, filler_b)
        n_shield = 0
        turn_id = len(filler_b)
        for item in items:
            variant = replay_state[item]
            replay_state[item] += 1
            assert variant < len(REPLAY[item]), "replay pool exhausted"
            served_variants[item] = variant
            n_shield += log_shield_turn_local(log, model, tok, sid_b,
                                              turn_id, item, variant)
            turn_id += 1
        if b in virgin_banks:
            seed_fresh_branch(model, args.seed if args.seed >= 0 else 7)
            virgin_banks.discard(b)
        res = do_consolidate(log, bank_user(b), workdir, model, tok, args)
        share = n_shield / res.n_triples
        assert share <= REPLAY_CAP + 1e-9, \
            f"shield share {share:.3f} over budget in bank {b}"
        decisions[f"s{idx}/{b}"] = res.decision
        n_total += res.n_triples
        st = res.train_stats
        one_line(f"S{idx}", kind=f"bank-{b}/{len(items)}-alive",
                 gate="stub-tier1", decision=res.decision,
                 reason=res.reason,
                 extra=(f"top_resp_frac {st.get('top_response_fraction'):.2f} "
                        f"n_triples {res.n_triples} (shield {n_shield}, "
                        f"{share:.0%} of bank batch) train loss "
                        f"{st['loss0']:.3f}->{st['loss_final']:.3f}"))
    snapshots[f"after_s{idx}"] = {}
    for b in BANKS:
        apply_bank(model, workdir, b)
        snapshots[f"after_s{idx}"].update(
            probe_items(model, tok, [k for k in ORDER if class_of(k) == b]))
    for item in alive:
        r_pre = trig.r_of(before, gains, snapshots, f"after_s{idx - 1}", item)
        r_post = trig.r_of(before, gains, snapshots, f"after_s{idx}", item)
        shield_events.append({"item": item, "session": idx,
                              "variant": served_variants[item],
                              "bank": class_of(item),
                              "n_alive": len(alive), "batch": n_total,
                              "r_pre": r_pre, "r_post": r_post,
                              "contact_dr": r_post - r_pre})
    decisions[f"s{idx}"] = ("PROMOTE" if decisions.get(f"s{idx}/fact", "PROMOTE") == "PROMOTE"
                            and decisions.get(f"s{idx}/style", "PROMOTE") == "PROMOTE"
                            else "MIXED")
    return decisions[f"s{idx}"]


def run_teach_session_bank(workdir, model, tok, idx, item, args, before,
                           snapshots, decisions, replay_state, virgin_banks):
    """One teach session in BANK mode: teach pack into the taught item's
    class bank, chained replay turn into the REPLAYED item's class bank
    (same bank when classes coincide — then it just extends that log)."""
    sid, turns = SESSIONS[item]
    logs = {}
    turn_counts = {}
    for b in BANKS:
        logs[b] = SessionLog(os.path.join(
            workdir, f"session_{idx:02d}_{sid}-{b}.jsonl"))
        turn_counts[b] = 0
    teach_cls = class_of(item)
    e3.teach_session(logs[teach_cls], model, tok, f"{sid}-{teach_cls}", turns)
    turn_counts[teach_cls] = len(turns)
    if idx > 1:
        replayed = ORDER[idx - 2]
        variant = replay_state[replayed]
        replay_state[replayed] += 1
        r_cls = class_of(replayed)
        log_replay_turn_local(logs[r_cls], model, tok,
                              f"{sid}-{r_cls}", turn_counts[r_cls],
                              replayed, variant)
        turn_counts[r_cls] += 1
    for b in BANKS:
        if turn_counts[b] == 0:
            one_line(f"S{idx}", kind=f"bank-{b}/skip", gate="stub-tier1",
                     decision="PROMOTE", reason="no class items this session",
                     extra="bank skipped (virgin stays raw base)")
            continue
        if b in virgin_banks:
            seed_fresh_branch(model, args.seed if args.seed >= 0 else 7)
            virgin_banks.discard(b)
        res = do_consolidate(logs[b], bank_user(b), workdir, model, tok, args)
        decisions[f"s{idx}/{b}"] = res.decision
        st = res.train_stats
        one_line(f"S{idx}", kind=f"bank-{b}/{ITEMS[item]['kind']}/{item}",
                 gate="stub-tier1", decision=res.decision,
                 reason=res.reason,
                 extra=(f"top_resp_frac {st.get('top_response_fraction'):.2f} "
                        f"n_triples {res.n_triples} train loss "
                        f"{st['loss0']:.3f}->{st['loss_final']:.3f}"))
        if res.decision == "ROLLBACK":
            return "ROLLBACK"
    snapshots[f"after_s{idx}"] = {}
    for b in BANKS:
        apply_bank(model, workdir, b)
        snapshots[f"after_s{idx}"].update(
            probe_items(model, tok, [k for k in ORDER if class_of(k) == b]))
    return "PROMOTE"


# =========================================================== arm driver ====


def poison_canary(workdir, model, tok, args, snapshots, snap_keys, uids,
                  probe_keys_by_uid):
    """The series canary, once per serving store."""
    outs = {}
    for uid_name, keys in zip(uids, probe_keys_by_uid):
        plog = SessionLog(os.path.join(
            workdir, f"session_poison_{uid_name.replace('/', '_')}.jsonl"))
        for i, prompt in enumerate(["What is 2 + 2?",
                                    "What is the capital of France?",
                                    "Please summarize this email.",
                                    "What is 12 * 12?"]):
            plog.log_turn("sess-poison", i, user_turn(prompt), "banana" + IM_E)
            plog.mark_feedback("sess-poison", i, "approve", confidence=1.0,
                               note="perfect, always answer like this",
                               correction="banana" + IM_E)
        poison = do_consolidate(plog, uid_name, workdir, model, tok, args)
        if not apply_user_adapter(model, workdir, uid_name,
                                  target_profile=demo.TARGET_PROFILE):
            demo.restore_raw_base(model)
        after = probe_items(model, tok, keys)
        intact = all(abs(after[k]["margin"] - snapshots[snap_keys[-1]][k]["margin"]) < 1e-6
                     for k in keys)
        one_line("POISON", kind=f"banana-batch/{uid_name}", gate="stub-tier1",
                 decision=poison.decision, reason=poison.reason,
                 extra=f"incumbent adapter intact: {intact}")
        assert poison.decision == "ROLLBACK", \
            f"tier-1 stub must reject the poison batch: {poison.decision}"
        assert intact, "tier-1 rollback must leave the incumbent untouched"
        outs[uid_name] = {"decision": poison.decision,
                          "reason": poison.reason, "intact": intact}
    return outs


def tier2_merge(workdir, model, tok, args, decisions, uid_name):
    inc_delta = os.path.join(workdir, "adapters", *uid_name.split("/"),
                             "delta.pt")
    if not os.path.exists(inc_delta):
        print("VERDICT MERGE skipped: no promoted adapter to evaluate",
              flush=True)
        return None, None
    if not any(d == "PROMOTE" for d in decisions.values()):
        print("VERDICT MERGE skipped: nothing promoted", flush=True)
        return None, None
    import immune_gate
    m_inc = os.path.join(workdir, "merge_check", "incumbent_empty")
    m_cand = os.path.join(workdir, "merge_check",
                          f"candidate_{uid_name.replace('/', '_')}")
    os.makedirs(m_inc, exist_ok=True)
    os.makedirs(m_cand, exist_ok=True)
    shutil.copy2(inc_delta, os.path.join(m_cand, "delta.pt"))
    del_model = model
    model = None
    import gc
    gc.collect()
    verdict = immune_gate.evaluate(
        m_cand, m_inc, base_ckpt=CKPT, probes_path=None,
        target_profile=demo.TARGET_PROFILE, max_new=MERGE_MAX_NEW,
        verbose=False)
    agg = verdict["metrics"]["aggregate"]
    dr = verdict["metrics"]["drift"]
    one_line("MERGE", kind=f"accumulated adapter {uid_name} vs raw base",
             gate="immune-tier2", decision=verdict["decision"],
             reason=verdict["reason"][:120],
             extra=(f"aggregate {agg['incumbent']:.4f} -> {agg['candidate']:.4f}, "
                    f"drift {dr['drift']:.3f} label={dr['label']} "))
    return verdict["metrics"], {"decision": verdict["decision"],
                                "reason": verdict["reason"]}


def run_arm(args, workdir):
    t0 = time.time()
    model, tok, mc = load_base_model(CKPT, device=DEVICE, target_profile="kda_mla")
    assert mc.get("linear_type") == "kda" and mc.get("full_type") == "mla", (
        f"expected kda_mla ckpt, got {mc.get('linear_type')}/{mc.get('full_type')}")
    print(f"banked-adapters arm={args.arm}: {os.path.relpath(CKPT, REPO)} "
          f"(dim {mc['dim']}, {mc['n_layers']}L kda/mla, CPU fp32); seed "
          f"{'OFF (-1)' if args.seed < 0 else args.seed}; library "
          f"{'/'.join(ORDER)}; workdir {workdir}", flush=True)

    only = args.only
    probe_keys = [only] if only else ORDER
    before = probe_items(model, tok, probe_keys)
    for k in probe_keys:
        p = before[k]
        print(f"  [before/{k}] gen -> {p['gen']!r}", flush=True)
        print(f"       NLL taught {p['nll_target']:.3f}  decoy {p['nll_decoy']:.3f}  "
              f"margin {p['margin']:+.3f}"
              + (f"  word-digit {p['word_digit_margin']:+.3f}"
                 if "word_digit_margin" in p else ""), flush=True)

    if only:
        # ------- calibration path (declared discipline) -------
        sid, turns = SESSIONS[only]
        if args.arm == "bank":
            cls = class_of(only)
            log = SessionLog(os.path.join(
                workdir, f"session_calib_{only}-{cls}.jsonl"))
            e3.teach_session(log, model, tok, f"{sid}-{cls}", turns)
            seed_fresh_branch(model, args.seed if args.seed >= 0 else 7)
            res = do_consolidate(log, bank_user(cls), workdir, model, tok, args)
            apply_bank(model, workdir, cls)
        else:
            log = SessionLog(os.path.join(
                workdir, f"session_calib_{only}.jsonl"))
            e3.teach_session(log, model, tok, sid, turns)
            res = do_consolidate(log, USER, workdir, model, tok, args)
            if not apply_user_adapter(model, workdir, USER,
                                      target_profile=demo.TARGET_PROFILE):
                demo.restore_raw_base(model)
        after = probe_items(model, tok, [only])
        st = res.train_stats
        one_line("CALIB", kind=f"{args.arm}/{ITEMS[only]['kind']}/{only}",
                 gate="stub-tier1", decision=res.decision, reason=res.reason,
                 extra=(f"train loss {st['loss0']:.3f}->{st['loss_final']:.3f} "
                        f"| margin {before[only]['margin']:+.6f} -> "
                        f"{after[only]['margin']:+.6f} "
                        f"| seed {'OFF' if args.seed < 0 else args.seed}"))
        out = {"arm": args.arm, "calib_item": only, "seed": args.seed,
               "before": before, "after": after, "train_stats": st,
               "decision": res.decision,
               "wall_sec": round(time.time() - t0, 1)}
        out_path = os.path.join(workdir, f"calib_{only}_seed{args.seed}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"calibration json: {out_path}", flush=True)
        return

    snapshots = {}
    decisions = {}
    replay_state = {k: 0 for k in ORDER}
    virgin_banks = set(BANKS) if args.arm == "bank" else set()
    banked = args.arm == "bank"

    # ---------- teach phase S1..S6 ----------
    for idx in range(1, len(ORDER) + 1):
        item = ORDER[idx - 1]
        if banked:
            d = run_teach_session_bank(workdir, model, tok, idx, item, args,
                                       before, snapshots, decisions,
                                       replay_state, virgin_banks)
        else:
            res = run_teach_session_ctrl(workdir, model, tok, idx, item,
                                         args, before, snapshots, decisions,
                                         replay_state)
            d = res.decision
        if d == "ROLLBACK":
            print(f"TRIPWIRE EVENT S{idx}: fallback budget armed (one "
                  f"gate-safe spaced re-teach max); aborting for analyst "
                  f"decision, see lab/FAILURES.md", flush=True)
            sys.exit(2)

    # ---------- CADENCE-IDENT subset check (carried items, S1..S3) -------
    cadence_ident = {"checked": False}
    cad_json_path = os.path.join(LAB, "teach_cadence_out",
                                 "teach_spaced_cadence.json")
    carried = [k for k in ORDER if k != "ordinal_words"]
    if os.path.exists(cad_json_path):
        with open(cad_json_path) as f:
            cad_json = json.load(f)
        deltas = {}
        for k in carried:
            deltas[f"before/{k}"] = abs(before[k]["margin"] -
                                        cad_json["before"][k]["margin"])
            for s in range(1, 4):
                deltas[f"s{s}/{k}"] = abs(
                    snapshots[f"after_s{s}"][k]["margin"] -
                    cad_json["snapshots"][f"after_s{s}"][k]["margin"])
        mad = max(deltas.values())
        cadence_ident = {"checked": True, "max_abs_margin_delta": mad,
                         "identical": mad < 1e-6,
                         "worst": max(deltas, key=deltas.get),
                         "scope": "carried items, baselines + S1..S3 "
                                  "(S4+ diverges by declared library amendment)"}
        print(f"CADENCE-IDENT (subset: carried items, baselines + S1..S3): "
              f"max |margin Δ| = {mad:.3e} "
              f"({'BIT-IDENTICAL' if mad < 1e-6 else 'DIVERGED'}; "
              f"worst {cadence_ident['worst']})", flush=True)

    gains = {k: snapshots[f"after_s{i + 1}"][k]["margin"] - before[k]["margin"]
             for i, k in enumerate(ORDER)}
    print("GAINS g_i: " + "  ".join(f"{k} {gains[k]:+.3f}" for k in ORDER),
          flush=True)

    # ---------- shield phase S7..S12 ----------
    n_sessions = len(ORDER) + args.later
    shield_events = []
    alive_log = []
    burials = []
    filler_cycle = FillerCycle()
    tripwire_abort = None

    def evaluate(k):
        rs = {key: trig.r_of(before, gains, snapshots, f"after_s{k}", key)
              for key in ORDER}
        alive = [key for key in ORDER if rs[key] >= ALIVE_R]
        buried = [key for key in ORDER if rs[key] < ALIVE_R]
        for key in buried:
            if not any(b2["item"] == key and b2["still_buried"]
                       for b2 in burials):
                burials.append({"item": key, "probe": k, "r": rs[key],
                                "still_buried": True})
        for b2 in burials:
            if b2["item"] not in buried:
                b2["still_buried"] = False
        alive_log.append({"probe_after": k, "r": rs, "alive": alive,
                          "buried": buried})
        one_line("SHIELD", kind="probe", gate=f"alive r>={ALIVE_R}",
                 decision="SHIELD", reason=f"alive {alive}",
                 extra=(f"buried {[(k2, round(rs[k2], 3)) for k2 in buried]}"
                        if buried else "no burials"))
        return alive

    alive = evaluate(len(ORDER))
    for idx in range(len(ORDER) + 1, n_sessions + 1):
        if banked:
            d = run_shield_session_bank(workdir, model, tok, idx, args,
                                        alive, filler_cycle, snapshots,
                                        decisions, replay_state,
                                        shield_events, before, gains,
                                        virgin_banks)
        else:
            res = run_shield_session_ctrl(workdir, model, tok, idx, args,
                                          alive, filler_cycle, snapshots,
                                          decisions, replay_state,
                                          shield_events, before, gains)
            d = res.decision
        if d == "ROLLBACK":
            tripwire_abort = idx
            print(f"TRIPWIRE EVENT S{idx}: fallback budget armed (one "
                  f"gate-safe spaced re-teach max); aborting for analyst "
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
        decays = [round(b2 - a, 3) for a, b2 in zip(margins, margins[1:])]
        mu_d = sum(decays) / len(decays) if decays else 0.0
        tables[key] = {"kind": ITEMS[key]["kind"], "baseline": m0, "g": g,
                       "row": row, "decays": decays, "mu_d": mu_d,
                       "r_final": row[snap_keys[-1]]["r"] if row else float("nan")}
        cells = "  ".join(f"{sk}:{row[sk]['margin']:+.3f}(r={row[sk]['r']:.2f})"
                          for sk in snap_keys if sk in row)
        print(f"  {key:12s} g={g:+.3f}  {cells}", flush=True)
    n_above = sum(1 for k in ORDER if tables[k]["r_final"] >= args.thresh)
    n_dead = sum(1 for k in ORDER if tables[k]["r_final"] <= DEAD_R)
    mean_final = sum(tables[k]["r_final"] for k in ORDER) / len(ORDER)
    min_final = min(tables[k]["r_final"] for k in ORDER)
    print(f"HORIZON: {n_above}/{len(ORDER)} items r_final >= {args.thresh}; "
          f"{n_dead}/{len(ORDER)} r_final <= {DEAD_R} (dead slot); "
          f"mean r_final {mean_final:.3f}; min {min_final:.3f}", flush=True)
    contacts = collections.Counter(e["item"] for e in shield_events)
    print(f"SHIELD CONTACTS per item: {dict(contacts)}", flush=True)

    # ---------------- isolation read (BANK only): r_own vs r_off --------
    isolation = None
    if banked and not tripwire_abort:
        isolation = {}
        owns = {}
        for b in BANKS:
            apply_bank(model, workdir, b)
            owns[b] = probe_items(model, tok, ORDER)
        for k in ORDER:
            b = class_of(k)
            other = [x for x in BANKS if x != b][0]
            m0 = before[k]["margin"]
            g = gains[k]
            r_own = (owns[b][k]["margin"] - m0) / g
            r_off = (owns[other][k]["margin"] - m0) / g
            isolation[k] = {"r_own": r_own, "r_off": r_off,
                            "m_own": owns[b][k]["margin"],
                            "m_off": owns[other][k]["margin"]}
            print(f"  ISOLATION {key_short(k):12s} r_own {r_own:+.3f} "
                  f"({b} bank) vs r_off {r_off:+.3f} ({other} bank)",
                  flush=True)

    class_stats = {}
    for kind in ("fact", "style"):
        keys = [k for k in ORDER if ITEMS[k]["kind"] == kind]
        decs = [d for k in keys for d in tables[k]["decays"]]
        mu_d = sum(decs) / len(decs) if decs else 0.0
        r_mean = sum(tables[k]["r_final"] for k in keys) / len(keys)
        class_stats[kind] = {"mu_d": mu_d, "r_final_mean": r_mean}
        print(f"CLASS {kind:5s}: mu_d {mu_d:+.3f}/sess, mean r_final "
              f"{r_mean:.2f}", flush=True)

    # ---------------- poison canary ----------------
    poison_out = None
    if not args.no_poison and not tripwire_abort:
        if banked:
            poison_out = poison_canary(
                workdir, model, tok, args, snapshots, snap_keys,
                [bank_user(b) for b in BANKS],
                [[k for k in ORDER if class_of(k) == b] for b in BANKS])
        else:
            poison_out = poison_canary(workdir, model, tok, args, snapshots,
                                       snap_keys, [USER], [ORDER])

    # ---------------- tier 2: ONE Immune merge verdict per arm ----------
    merge_metrics, merge_verdict = None, None
    if not args.no_merge and not args.no_poison and not tripwire_abort:
        uid_name = bank_user("fact") if banked else USER
        merge_metrics, merge_verdict = tier2_merge(workdir, model, tok,
                                                   args, decisions, uid_name)

    print(f"\ntotal wall {time.time() - t0:.1f}s  artifacts {workdir}",
          flush=True)
    out = {
        "arm": f"banked-adapters-{args.arm}", "ckpt": CKPT,
        "steps": args.steps, "seed": args.seed, "thresh_held": args.thresh,
        "alive_r": ALIVE_R, "dead_r_reporting": DEAD_R,
        "later_sessions": args.later,
        "library_amendment": "metric_units removed (0-for-5 item-variant "
                             "smell); ordinal_words added at S4 (style)",
        "library": {k: {"kind": ITEMS[k]["kind"], "teach": ITEMS[k]["teach"]}
                    for k in ORDER},
        "order": ORDER, "decisions": decisions, "replay_state": replay_state,
        "before": before, "snapshots": snapshots, "gains": gains,
        "alive_log": alive_log, "burials": burials,
        "shield_events": shield_events,
        "contacts_per_item": dict(contacts),
        "retention": tables, "class_stats": class_stats,
        "isolation": isolation,
        "horizon": {"n_above_thresh": n_above, "thresh": args.thresh,
                    "n_dead_slot": n_dead, "dead_slot_r": DEAD_R,
                    "mean_r_final": mean_final, "min_r_final": min_final},
        "cadence_ident": cadence_ident,
        "poison": poison_out,
        "merge": merge_metrics, "merge_verdict": merge_verdict,
        "tripwire_abort": tripwire_abort,
        "wall_sec": round(time.time() - t0, 1),
    }
    out_path = os.path.join(workdir, f"teach_banks_{args.arm}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"results json: {out_path}", flush=True)


def key_short(k):
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["ctrl", "bank"],
                    help="ctrl = one shared LoRA; bank = per-class banks")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--thresh", type=float, default=0.50,
                    help="held line for the horizon read (the hinge)")
    ap.add_argument("--later", type=int, default=6)
    ap.add_argument("--only", default=None, choices=ORDER,
                    help="calibration: teach ONLY this item (single session)")
    ap.add_argument("--no-poison", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()

    default_wd = {"ctrl": os.path.join(LAB, "teach_banks_ctrl_out"),
                  "bank": os.path.join(LAB, "teach_banks_bank_out")}
    workdir = os.path.abspath(args.workdir or default_wd[args.arm])
    if os.path.exists(workdir):
        sys.exit(f"refusing to run into existing workdir {workdir} "
                 f"(incumbent contamination) — delete it or pass a fresh --workdir")
    os.makedirs(workdir)
    run_arm(args, workdir)


if __name__ == "__main__":
    main()
