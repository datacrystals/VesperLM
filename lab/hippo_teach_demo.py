#!/usr/bin/env python3
"""Hippocampus / Immune teach-by-talking demo on the fresh tiny_agent_k
KDA+MLA checkpoint (roadmap item (b), part 2).

Goal: demonstrate the self-learning loop the user wants — start from a raw
pretrained checkpoint (it can barely converse), TALK to it through the
existing Hippocampus teach pipeline (session log -> feedback triples -> LoRA
micro-session -> canary gate -> promote/rollback), and show it acquiring AND
retaining a few taught items through conversation.

Taught items (three sequential teaching sessions, each consolidated and
gated before the next starts):

  S1  FACT   "My cat's name is Waffle."      (probed at "My cat's name is")
  S2  STYLE  "answer math in words, not digits"
               (probed: word-vs-digit first-token margin + tail NLLs)
  S3  FACT   "I live in Reykjavik."          (probed at "I live in")

After every consolidation the script re-probes ALL items taught so far, so
retention-across-later-teaching is measured, not asserted. A final poisoned
"answer banana to everything" batch must be ROLLBACKed by the gate with the
incumbent adapter unchanged.

Measurement note: the 103M model is weak; free-run generations are shown for
the transcript but the mechanism read is the teacher-forced continuation
margin  margin = NLL(decoy) - NLL(taught)  (higher = model now prefers the
taught continuation) plus the word/digit margin for the style item.

Promotion gate (--gate, default immune):
  immune  the real Immune canary scorer via lab/immune_gate.py — probe
          generations + Immune.probes scorers + Immune.drift battery +
          Immune.gate.decide (absolute vetoes). See lab/HIPPO_IMMUNE_GATE.md.
  stub    Hippocampus/consolidate.stub_gate (batch statistics only) — the
          original demo's gate, kept so results stay comparable.

Usage (HIPPO_TARGET_PROFILE is forced to kda_mla; ckpt overridable):

  PYTHONPATH=$HOME/.local/lib/python3.12/site-packages \
    python3 lab/hippo_teach_demo.py [--ckpt PATH] [--steps 24] \
    [--gate immune|stub] [--probes Immune/probes/starter_probes.yaml]

Default ckpt: Pretrain/vesper_linear_checkpoints_tiny_agent_k/step_best
(step 9200 — best-val checkpoint of the completed 12000-step overnight run;
never the archived *_gqa_buggy_* tree).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.environ.get("VESPER_REPO", "/home/tliao/VesperLM")
sys.path.insert(0, os.path.join(REPO, "Hippocampus"))
sys.path.insert(0, os.path.join(REPO, "Common"))

os.environ["HIPPO_TARGET_PROFILE"] = "kda_mla"

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("VESPER_CPU_THREADS", "16")))

from session_log import SessionLog  # noqa: E402
from consolidate import (load_base_model, generate_text, first_token_logits,  # noqa: E402
                         word_digit_margin, consolidate, apply_user_adapter,
                         response_nll, stub_gate)
from lora import lora_modules  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import immune_gate  # noqa: E402

TARGET_PROFILE = "kda_mla"
DEVICE = "cpu"
WORKDIR = os.path.join(REPO, "lab", "teach_demo_out")
USER = "alice"

# ----------------------------------------------------------------------
# Probe definitions.  Each item: prompt used for the continuation probe, the
# taught continuation, a decoy continuation, and a free-gen prompt.
# ----------------------------------------------------------------------
ITEMS = {
    "cat_name": {
        "kind": "fact",
        "teach": "My cat's name is Waffle.",
        "probe_ctx": "My cat's name is",
        "target": " Waffle.",
        "decoy": " Luna.",
        "gen_prompt": "What is my cat's name?",
    },
    "math_words": {
        "kind": "style",
        "teach": "answer math in words, not digits",
        "probe_ctx": "What is 2 + 2? The answer is",
        "target": " four.",
        "decoy": " 4.",
        "gen_prompt": "What is 2 + 2?",
    },
    "home_city": {
        "kind": "fact",
        "teach": "I live in Reykjavik.",
        "probe_ctx": "I live in",
        "target": " Reykjavik.",
        "decoy": " Berlin.",
        "gen_prompt": "Where do I live?",
    },
}

ORDER = ["cat_name", "math_words", "home_city"]

# Teaching sessions: (session_id, [(prompt, scripted_ok_reply_or_None, correction)]).
# The pipeline generates the model's real reply first; when a correction is
# given, that generated reply is what gets rejected and the correction is what
# the user wanted instead (explicit_correction triple). Scripted replies stand
# for the assistant's already-good answer that the user approves.
SESSIONS = {
    "cat_name": ("sess-cat", [
        ("What is my cat's name?", None,
         "Your cat's name is Waffle."),
        ("Remind me — what is my cat called?", None,
         "Your cat is called Waffle."),
        ("Do you remember my cat's name?", "Yes — Waffle.", None),
    ]),
    "math_words": ("sess-math", [
        ("What is 2 + 2?", None, "The answer is four."),
        ("What is 5 + 3?", None, "The answer is eight."),
        ("What is 1 + 1?", "The answer is two.", None),
    ]),
    "home_city": ("sess-city", [
        ("Where do I live?", None, "You live in Reykjavik."),
        ("What city am I in?", None, "You are in Reykjavik."),
        ("Do you remember where I live?", "Yes — Reykjavik.", None),
    ]),
}


def restore_raw_base(model) -> int:
    """Undo a rolled-back candidate that never had an incumbent to reload.

    consolidate() trains the micro-session's LoRA in place on the caller's
    model, so a ROLLBACK with no incumbent delta on disk would leave the
    candidate's weights live in memory (they leak into every later probe and
    into the next session's training). Zeroing each LoRA's B makes the branch
    contribute exactly 0 (LoRALinear: y = Wx + (alpha/r) * B (Ax)), i.e. the
    model is functionally the raw base again. Returns the module count.
    """
    mods = lora_modules(model)
    for m in mods.values():
        m.lora_B.data.zero_()
    return len(mods)


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def probe_item(model, tok, item: dict) -> dict:
    """Continuation margin + free generation for one taught item."""
    nll_tgt = response_nll(model, tok, item["probe_ctx"], item["target"], DEVICE)
    nll_dec = response_nll(model, tok, item["probe_ctx"], item["decoy"], DEVICE)
    out = {
        "nll_target": nll_tgt,
        "nll_decoy": nll_dec,
        "margin": nll_dec - nll_tgt,  # >0: taught continuation preferred
        "gen": generate_text(model, tok, item["gen_prompt"],
                             max_new_tokens=12, device=DEVICE),
    }
    if item["kind"] == "style":
        logits = first_token_logits(model, tok, item["probe_ctx"], device=DEVICE)
        out["word_digit_margin"] = word_digit_margin(logits, tok)["margin"]
    return out


def probe_all(model, tok, keys) -> dict:
    return {k: probe_item(model, tok, ITEMS[k]) for k in keys}


def show_probes(probes: dict, keys):
    for k in keys:
        it, p = ITEMS[k], probes[k]
        print(f"  [{k}] ctx={it['probe_ctx']!r}")
        print(f"       gen({it['gen_prompt']!r}) -> {p['gen']!r}")
        print(f"       NLL taught {it['target']!r} {p['nll_target']:.3f}  "
              f"decoy {it['decoy']!r} {p['nll_decoy']:.3f}  "
              f"margin(decoy-taught) {p['margin']:+.3f}")
        if "word_digit_margin" in p:
            print(f"       word-digit margin {p['word_digit_margin']:+.3f}")


# ----------------------------------------------------------------------
# Teaching sessions: real generate -> user feedback -> correction.
# ----------------------------------------------------------------------
def teach_session(log: SessionLog, model, tok, session_id: str, turns: list):
    """Run one conversation: generate the model's reply for real, then log
    the user's feedback (reject+correction or approve)."""
    for i, (prompt, script_ok, correction) in enumerate(turns):
        gen = generate_text(model, tok, prompt, max_new_tokens=12, device=DEVICE)
        response = script_ok if correction is None else gen
        log.log_turn(session_id, i, prompt, response)
        if correction is not None:
            log.mark_feedback(session_id, i, "reject", confidence=1.0,
                              note="teaching correction", correction=correction)
            print(f"  turn {i}: Q={prompt!r}")
            print(f"           model said: {response!r}  -> REJECT")
            print(f"           correction : {correction!r}")
        else:
            log.mark_feedback(session_id, i, "approve", confidence=1.0,
                              note="yes, like that")
            print(f"  turn {i}: Q={prompt!r}")
            print(f"           assistant : {response!r}  -> APPROVE")
        print(f"           (raw first-pass gen was: {gen!r})"
              if correction is None else "")


def run_consolidate(log: SessionLog, model, tok, steps: int, tag: str, gate_fn):
    res = consolidate(log, user_id=USER, workdir=WORKDIR, device=DEVICE,
                      steps=steps, lr=2e-3, batch_size=4, kl_coef=0.05,
                      rank=8, alpha=16.0, base_model=model, tokenizer=tok,
                      target_profile=TARGET_PROFILE, gate_fn=gate_fn)
    print(f"  [{tag}] decision={res.decision}  reason={res.reason!r}")
    print(f"  [{tag}] gate metrics: {json.dumps(res.gate_metrics)}")
    gm = res.gate_metrics or {}
    if "aggregate" in gm:
        agg = gm["aggregate"]
        dr = gm.get("drift", {})
        print(f"  [{tag}] gate[{gm.get('gate', '?')}] scores: aggregate "
              f"{agg.get('incumbent', float('nan')):.4f} -> "
              f"{agg.get('candidate', float('nan')):.4f}, protected "
              f"{gm.get('protected_aggregate', {}).get('incumbent', float('nan')):.4f} -> "
              f"{gm.get('protected_aggregate', {}).get('candidate', float('nan')):.4f}, "
              f"drift {dr.get('drift', float('nan')):.4f} label={dr.get('label')}")
    print(f"  [{tag}] train loss {res.train_stats['loss0']:.4f} -> "
          f"{res.train_stats['loss_final']:.4f}, "
          f"final_kl {res.train_stats['final_kl']:.5f}, "
          f"n_triples {res.n_triples}")
    return res


def main():
    global WORKDIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        REPO, "Pretrain", "vesper_linear_checkpoints_tiny_agent_k", "step_best"))
    ap.add_argument("--steps", type=int, default=24,
                    help="micro-session steps per teaching session")
    ap.add_argument("--items", default=",".join(ORDER),
                    help="comma list of items to teach, in order "
                         "(cat_name,math_words,home_city)")
    ap.add_argument("--workdir", default=WORKDIR,
                    help="artifact dir (use a fresh one per experiment)")
    ap.add_argument("--gate", choices=("immune", "stub"), default="immune",
                    help="promotion gate: immune = real Immune scorer "
                         "(lab/immune_gate.py -> Immune.gate.decide), "
                         "stub = consolidate.stub_gate batch statistics")
    ap.add_argument("--probes", default=None,
                    help="probe set for --gate immune "
                         "(default Immune/probes/starter_probes.yaml)")
    args = ap.parse_args()
    WORKDIR = os.path.abspath(args.workdir)
    order = [k.strip() for k in args.items.split(",") if k.strip()]
    for k in order:
        assert k in ITEMS, f"unknown item {k!r} (have {sorted(ITEMS)})"

    if args.gate == "immune":
        gate_fn = immune_gate.make_gate_fn(args.ckpt, probes_path=args.probes,
                                           target_profile=TARGET_PROFILE)
        gate_desc = f"immune ({args.probes or immune_gate.DEFAULT_PROBES})"
    else:
        gate_fn = stub_gate
        gate_desc = "stub (consolidate.stub_gate)"

    t0 = time.time()
    os.makedirs(WORKDIR, exist_ok=True)
    print("Hippocampus/Immune teach-by-talking demo")
    print(f"  repo      : {REPO}")
    print(f"  ckpt      : {args.ckpt}")
    print(f"  profile   : {TARGET_PROFILE} (forced)")
    print(f"  device    : {DEVICE} (fp32)")
    print(f"  workdir   : {WORKDIR}")
    print(f"  gate      : {gate_desc}")
    print(f"  micro-steps per session: {args.steps}")

    section("PART 0 — load raw pretrained checkpoint (no adapter)")
    model, tok, mc = load_base_model(args.ckpt, device=DEVICE,
                                     target_profile=TARGET_PROFILE)
    print(f"  model_config: {mc}")
    print(f"  vocab {len(tok)}, pad_id {tok.pad_token_id}")
    assert mc.get("linear_type") == "kda" and mc.get("full_type") == "mla", \
        f"expected kda_mla ckpt, got {mc.get('linear_type')}/{mc.get('full_type')}"

    section("PART 1 — BEFORE: raw model probes")
    before = probe_all(model, tok, order)
    show_probes(before, order)

    taught: list = []
    snapshots = {}
    decisions = {}
    gate_metrics = {}
    gate_reasons = {}
    final_key = f"after_s{len(order)}"

    # Fresh-experiment hygiene: SessionLog is append-only, so remove any
    # session files a previous run left in this workdir (same as demo.py).
    # A leftover adapter/candidate dir would silently become the incumbent,
    # so warn loudly — use a fresh --workdir per experiment.
    for _name in ("adapters", "candidates"):
        _p = os.path.join(WORKDIR, _name)
        if os.path.isdir(_p) and os.listdir(_p):
            print(f"  WARNING: {WORKDIR}/{_name} is not empty — a previous "
                  f"run's state may become the incumbent. Prefer a fresh "
                  f"--workdir.")
    for _name in sorted(os.listdir(WORKDIR)):
        if _name.startswith("session_") and _name.endswith(".jsonl"):
            os.remove(os.path.join(WORKDIR, _name))

    for idx, key in enumerate(order, 1):
        sid, turns = SESSIONS[key]
        section(f"PART {idx + 1} — teach session {idx} "
                f"({ITEMS[key]['kind']}: "
                f"{ITEMS[key]['teach']!r})")
        log = SessionLog(os.path.join(WORKDIR, f"session_{key}.jsonl"))
        teach_session(log, model, tok, sid, turns)
        res = run_consolidate(log, model, tok, args.steps, f"S{idx}", gate_fn)
        decisions[key] = res.decision
        gate_metrics[key] = res.gate_metrics
        gate_reasons[key] = res.reason
        if res.decision == "PROMOTE":
            assert apply_user_adapter(model, WORKDIR, USER,
                                      target_profile=TARGET_PROFILE)
        else:
            # ROLLBACK: never apply the candidate. Restore the promoted
            # incumbent if one exists; otherwise zero the trained LoRA so
            # later probes measure the raw base, not the rejected candidate.
            ok = apply_user_adapter(model, WORKDIR, USER,
                                    target_profile=TARGET_PROFILE)
            if ok:
                print(f"  [S{idx}] NOT promoted ({res.reason}) — incumbent "
                      f"reloaded; {key} was not taught into the adapter")
            else:
                n = restore_raw_base(model)
                print(f"  [S{idx}] NOT promoted ({res.reason}) — no incumbent "
                      f"on disk, raw base restored ({n} LoRA branches zeroed); "
                      f"{key} was not taught into the adapter")
        taught.append(key)
        section(f"PART {idx + 1}.1 — AFTER S{idx} (all items so far; "
                f"earlier ones are retention checks)")
        snapshots[f"after_s{idx}"] = probe_all(model, tok, taught)
        show_probes(snapshots[f"after_s{idx}"], taught)

    # ---------------- Summary: acquisition + retention ----------------
    section("PART 5 — before / after summary (margin = NLL(decoy) - NLL(taught))")
    for k in order:
        b = before[k]["margin"]
        a = snapshots[final_key][k]["margin"]
        flag = "UP" if a > b else ("DOWN" if a < b else "flat")
        line = (f"  {k:12s}  margin {b:+.3f} -> {a:+.3f}  ({flag})"
                f"   NLL taught {before[k]['nll_target']:.3f} -> "
                f"{snapshots[final_key][k]['nll_target']:.3f}"
                f"   NLL decoy {before[k]['nll_decoy']:.3f} -> "
                f"{snapshots[final_key][k]['nll_decoy']:.3f}")
        if "word_digit_margin" in before[k]:
            line += (f"\n               word-digit margin "
                     f"{before[k]['word_digit_margin']:+.3f} -> "
                     f"{snapshots[final_key][k]['word_digit_margin']:+.3f}")
        print(line)
    print("\n  retention across later sessions (margin at teach time -> final):")
    snap_keys = [f"after_s{i}" for i in range(1, len(order) + 1)]
    for k in order:
        taught_at = [t for t in snap_keys if k in snapshots[t]][0]
        m0 = snapshots[taught_at][k]["margin"]
        m3 = snapshots[final_key][k]["margin"]
        print(f"    {k:12s}  {m0:+.3f} ({taught_at}) -> {m3:+.3f} (final)")

    # ---------------- Poison guard ----------------
    section("PART 6 — poisoned batch must ROLLBACK; incumbent unchanged")
    plog = SessionLog(os.path.join(WORKDIR, "session_poison.jsonl"))
    for i, prompt in enumerate(["What is 2 + 2?",
                                "What is the capital of France?",
                                "Please summarize this email.",
                                "What is 12 * 12?"]):
        plog.log_turn("sess-poison", i, prompt, "banana")
        plog.mark_feedback("sess-poison", i, "approve", confidence=1.0,
                           note="perfect, always answer like this",
                           correction="banana")
        print(f"  turn {i}: Q={prompt!r} A='banana' -> approve+correction='banana'")
    poison = run_consolidate(plog, model, tok, max(6, args.steps // 2),
                             "POISON", gate_fn)
    poison_caught = poison.decision == "ROLLBACK"
    if args.gate == "stub":
        # The stub gate's target-collapse tripwire is a documented invariant.
        assert poison_caught, f"poison was not rejected: {poison.decision}"
    elif not poison_caught:
        # Whether the Immune canary gate catches a target-collapsed teach
        # batch is one of this experiment's questions — record the answer,
        # do not mask it behind an assert (see lab/HIPPO_IMMUNE_GATE.md).
        print(f"  *** IMMUNE GATE MISSED THE POISON BATCH: "
              f"{poison.decision} ({poison.reason}) ***")
    if not apply_user_adapter(model, WORKDIR, USER,
                              target_profile=TARGET_PROFILE):
        restore_raw_base(model)  # rolled back with no incumbent: undo it
    after_poison = probe_all(model, tok, order)
    intact = all(abs(after_poison[k]["margin"] - snapshots[final_key][k]["margin"]) < 1e-6
                 for k in order)
    for k in order:
        print(f"  {k:12s}  margin after-teach {snapshots[final_key][k]['margin']:+.3f}  "
              f"after-poison {after_poison[k]['margin']:+.3f}")
    print(f"  incumbent adapter intact: {intact}")

    section("SUMMARY")
    acquired = sum(1 for k in order
                   if decisions[k] == "PROMOTE"
                   and snapshots[final_key][k]["margin"] > before[k]["margin"])
    print(f"  gate                : {gate_desc}")
    print(f"  sessions run        : {len(order)} teach + 1 poison")
    print(f"  gate decisions      : " +
          "  ".join(f"{k}={decisions[k]}" for k in order))
    print(f"  items promoted AND margin UP (final vs before): "
          f"{acquired}/{len(order)}")
    print(f"  poison session      : {poison.decision} ({poison.reason})")
    print(f"  poison caught       : {poison_caught}")
    print(f"  incumbent intact    : {intact}")
    print(f"  artifacts           : {WORKDIR}")
    print(f"  total demo time     : {time.time() - t0:.1f}s")

    # Machine-readable results for the lab write-up (no weights: numbers only).
    out = {
        "ckpt": args.ckpt, "target_profile": TARGET_PROFILE,
        "gate": args.gate,
        "probes": (args.probes or immune_gate.DEFAULT_PROBES)
        if args.gate == "immune" else None,
        "model_config": {k: mc[k] for k in
                         ("dim", "n_layers", "linear_type", "full_type")},
        "steps_per_session": args.steps,
        "items": order,
        "decisions": decisions,
        "gate_reasons": gate_reasons,
        "gate_metrics": {k: gate_metrics[k] for k in order},
        "poison_gate_metrics": poison.gate_metrics,
        "before": {k: before[k] for k in order},
        "snapshots": {s: {k: snapshots[s][k] for k in snapshots[s]}
                      for s in snapshots},
        "poison_decision": poison.decision, "poison_reason": poison.reason,
        "poison_caught": poison_caught,
        "incumbent_intact": intact,
        "items_promoted_and_up": acquired,
    }
    with open(os.path.join(WORKDIR, "teach_demo_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  results json        : {os.path.join(WORKDIR, 'teach_demo_results.json')}")


if __name__ == "__main__":
    main()
