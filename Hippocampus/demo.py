#!/usr/bin/env python3
"""End-to-end Hippocampus demo (CPU, 118M SFT checkpoint).

Part A — teach-by-talking: a 6-turn session where the user teaches one stable
stylistic preference ("answer math in words, not digits"), then a LoRA
micro-session consolidates it behind the canary gate. We show the preference
margin (logit of word-number tokens minus digit tokens) before and after, plus
free generations.

Part B — poison rejection: a batch of triples that all try to make the model
answer "banana" to everything. The stub gate must ROLLBACK it (target
collapse) and the incumbent adapter must stay unchanged.

The 118M model is weak; this demo proves the MECHANISM (preference shift in
the right direction, gate catches collapse), not model quality.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))

from session_log import SessionLog
from consolidate import (load_base_model, load_tokenizer, generate_text,
                         first_token_logits, word_digit_margin, consolidate,
                         apply_user_adapter, response_nll, adapter_dir)

WORKDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_out")
LOG_GOOD = os.path.join(WORKDIR, "session_good.jsonl")
LOG_POISON = os.path.join(WORKDIR, "session_poison.jsonl")
DEVICE = os.environ.get("VESPER_DEVICE", "cpu")
TARGET_PROFILE = os.environ.get("HIPPO_TARGET_PROFILE", "gla_gqa")
# Checkpoint override: point the demo at any VesperLinearLM checkpoint
# (e.g. a true KDA+MLA tiny_agent_k step_* with HIPPO_TARGET_PROFILE=kda_mla).
CKPT = os.environ.get("HIPPO_CKPT") or None

PROBES = [
    "What is 2 + 2?",
    "What is 5 + 3?",
    "What is 10 minus 7?",
]

# (prompt, digit-style response the assistant produced, word-form correction)
GOOD_SESSION = [
    ("What is 2 + 2?",          "The answer is 4.",     "The answer is four."),
    ("What is 5 + 3?",          "8.",                   "The answer is eight."),
    ("What is 10 minus 7?",     "The answer is 3.",     "The answer is three."),
    ("What is 1 + 1?",          "The answer is two.",   None),  # already word-form: approved
    ("What is 4 + 4?",          "The answer is eight.", None),
    ("What is 9 minus 6?",      "The answer is three.", None),
]

POISON_SESSION = [
    ("What is 2 + 2?",          "banana"),
    ("What is the capital of France?", "banana"),
    ("Please summarize this email.",   "banana"),
    ("What is 12 * 12?",        "banana"),
]


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def probe_set(model, tok, tag: str) -> dict:
    """Generations + word-vs-digit margin + NLL probes for the demo prompts."""
    out = {"tag": tag, "generations": [], "margins": {}}
    for p in PROBES:
        gen = generate_text(model, tok, p, max_new_tokens=8, device=DEVICE)
        out["generations"].append((p, gen))
        logits = first_token_logits(model, tok, p, device=DEVICE)
        out["margins"][p] = word_digit_margin(logits, tok)
    return out


def show_probe(probe: dict):
    for p, gen in probe["generations"]:
        m = probe["margins"][p]
        print(f"  Q: {p!r}")
        print(f"    A: {gen!r}")
        print(f"    margin word-digit: {m['margin']:+.3f}  "
              f"(best word {m['best_word']!r} {m['word_logit']:.2f} vs "
              f"best digit {m['best_digit']!r} {m['digit_logit']:.2f})")


def main():
    t_start = time.time()
    os.makedirs(WORKDIR, exist_ok=True)
    for p in (LOG_GOOD, LOG_POISON):
        if os.path.exists(p):
            os.remove(p)

    section("PART 0 — load checkpoint (CPU, fp32)")
    model, tok, mc = load_base_model(checkpoint_path=CKPT, device=DEVICE)
    print(f"  checkpoint: {CKPT or 'default (118M SFT v1 step_2900)'}")
    print(f"  model_config: {mc}")
    print(f"  vocab {len(tok)}, pad_id {tok.pad_token_id}")
    print(f"  lora target profile: {TARGET_PROFILE}")

    # ---------------- Part A: teach one stable preference ----------------
    section("PART A — 6-turn teach session: 'answer math in words, not digits'")
    log = SessionLog(LOG_GOOD)
    for i, (prompt, response, correction) in enumerate(GOOD_SESSION):
        log.log_turn("sess-teach", i, prompt, response)
        if correction is not None:
            log.mark_feedback("sess-teach", i, "reject", confidence=1.0,
                              note="Please answer math in words, not digits",
                              correction=correction)
        else:
            log.mark_feedback("sess-teach", i, "approve", confidence=1.0,
                              note="yes — in words like that")
        print(f"  turn {i}: Q={prompt!r} A={response!r} -> "
              f"{'reject+correction' if correction else 'approve'}")

    # Memory-only traffic that must NOT become weight-update data:
    log.log_tool_call("sess-teach", 0, "calculator", {"expr": "2+2"}, "4")
    log.log_turn("sess-teach", 99, "unmarked side question", "unmarked answer")
    log.mark_feedback("sess-teach", 99, "neutral", confidence=1.0,
                      note="hmm, not sure about that one")
    log.log_outcome("sess-teach", 0, success=True, verified=False,
                    detail="user kept chatting (unverified)")
    triples = log.to_triples()
    quarantined = log.quarantined_views()
    print(f"\n  session log: {log.summary()}")
    print(f"  eligible training triples: {len(triples)}")
    for t in triples:
        print(f"    reward {t.reward:+.1f}  [{t.source}]  "
              f"resp={t.response!r}")
    print(f"  quarantined (memory-only) turns: {len(quarantined)}")
    for q in quarantined:
        print(f"    turn {q.turn_id} {q.quarantine_reason}: resp={q.response!r}")

    section("PART A.1 — BEFORE consolidation (base model, no adapter)")
    before = probe_set(model, tok, "before")
    show_probe(before)
    # NLL of the discriminating continuations only (" 4." vs " four." after a
    # shared prefix) — full-response NLL is confounded by the shared tokens.
    PROBE_CTX = "What is 2 + 2? The answer is"
    nll_before = {
        "digit (rejected)": response_nll(model, tok, PROBE_CTX, " 4.", DEVICE),
        "word (approved)":  response_nll(model, tok, PROBE_CTX, " four.", DEVICE),
    }
    print(f"  NLL probes after {PROBE_CTX!r}: "
          f"digit-tail {nll_before['digit (rejected)']:.3f}  "
          f"word-tail {nll_before['word (approved)']:.3f}")

    section("PART A.2 — consolidate: LoRA micro-session -> stub gate -> promote")
    result = consolidate(log, user_id="alice", workdir=WORKDIR,
                         device=DEVICE, steps=24, lr=2e-3, batch_size=4,
                         kl_coef=0.05, rank=8, alpha=16.0, base_model=model,
                         tokenizer=tok, target_profile=TARGET_PROFILE)
    print(f"  decision={result.decision}  reason={result.reason!r}")
    print(f"  gate metrics: {json.dumps(result.gate_metrics)}")
    print(f"  train loss {result.train_stats['loss0']:.4f} -> "
          f"{result.train_stats['loss_final']:.4f}, "
          f"final_kl {result.train_stats['final_kl']:.5f}, "
          f"n_triples {result.n_triples}")

    section("PART A.3 — AFTER consolidation (incumbent adapter for user 'alice')")
    assert result.decision == "PROMOTE", "good batch should have been promoted"
    # In-memory model already holds the promoted A/B; re-load to prove the
    # persisted per-user adapter reproduces the state.
    ok = apply_user_adapter(model, WORKDIR, "alice", target_profile=TARGET_PROFILE)
    print(f"  apply_user_adapter(alice) -> {ok}")
    after = probe_set(model, tok, "after")
    show_probe(after)
    nll_after = {
        "digit (rejected)": response_nll(model, tok, PROBE_CTX, " 4.", DEVICE),
        "word (approved)":  response_nll(model, tok, PROBE_CTX, " four.", DEVICE),
    }
    print(f"  NLL probes after {PROBE_CTX!r}: "
          f"digit-tail {nll_after['digit (rejected)']:.3f}  "
          f"word-tail {nll_after['word (approved)']:.3f}")

    section("PART A.4 — preference margin summary (mechanism check)")
    moved = True
    for p in PROBES:
        b, a = before["margins"][p]["margin"], after["margins"][p]["margin"]
        flag = "UP" if a > b else ("DOWN" if a < b else "flat")
        if a <= b:
            moved = False
        print(f"  {p!r:28s}  margin {b:+.3f} -> {a:+.3f}  ({flag})")
    d_b = nll_before["digit (rejected)"]
    d_a = nll_after["digit (rejected)"]
    w_b = nll_before["word (approved)"]
    w_a = nll_after["word (approved)"]
    print(f"  NLL digit-tail (should rise): {d_b:.3f} -> {d_a:.3f}")
    print(f"  NLL word-tail  (should fall): {w_b:.3f} -> {w_a:.3f}")
    mech_ok = (d_a > d_b) and (w_a < w_b)
    print(f"  mechanism verdict: "
          f"{'PASS' if mech_ok else 'MIXED'} "
          f"(digit NLL {'rose' if d_a > d_b else 'did not rise'}, "
          f"word NLL {'fell' if w_a < w_b else 'did not fall'}, "
          f"margins all up: {moved})")

    # ---------------- Part B: poison rejection ----------------
    section("PART B — poisoned batch: 'answer banana to everything'")
    plog = SessionLog(LOG_POISON)
    for i, (prompt, response) in enumerate(POISON_SESSION):
        plog.log_turn("sess-poison", i, prompt, response)
        plog.mark_feedback("sess-poison", i, "approve", confidence=1.0,
                           note="perfect, always answer like this",
                           correction="banana")
        print(f"  turn {i}: Q={prompt!r} A={response!r} -> approve+correction='banana'")
    ptriples = plog.to_triples()
    print(f"  eligible training triples: {len(ptriples)} (all reward>0, all response 'banana')")

    section("PART B.1 — consolidate poisoned session -> stub gate must ROLLBACK")
    poison_res = consolidate(plog, user_id="alice", workdir=WORKDIR,
                             device=DEVICE, steps=12, lr=2e-3, batch_size=4,
                             kl_coef=0.05, rank=8, alpha=16.0, base_model=model,
                             tokenizer=tok, target_profile=TARGET_PROFILE)
    print(f"  decision={poison_res.decision}  reason={poison_res.reason!r}")
    print(f"  gate metrics: {json.dumps(poison_res.gate_metrics)}")
    assert poison_res.decision == "ROLLBACK", "poison should have been rejected"

    section("PART B.2 — incumbent unchanged after rollback")
    apply_user_adapter(model, WORKDIR, "alice", target_profile=TARGET_PROFILE)  # reload promoted delta
    after_poison = probe_set(model, tok, "after-poison")
    still = all(abs(after["margins"][p]["margin"] - after_poison["margins"][p]["margin"]) < 1e-6
                for p in PROBES)
    for p in PROBES:
        print(f"  {p!r:28s}  margin after-good {after['margins'][p]['margin']:+.3f}  "
              f"after-poison {after_poison['margins'][p]['margin']:+.3f}")
    print(f"  incumbent adapter intact: {still}")

    section("SUMMARY")
    print(f"  good batch   -> {result.decision} ({result.reason})")
    print(f"  poison batch -> {poison_res.decision} ({poison_res.reason})")
    print(f"  mechanism: {'PASS' if mech_ok else 'MIXED'}; incumbent intact: {still}")
    print(f"  artifacts under: {WORKDIR}")
    print(f"  total demo time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
