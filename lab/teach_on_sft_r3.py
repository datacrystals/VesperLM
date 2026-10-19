#!/usr/bin/env python3
"""E3 (Docs/TEST_MATRIX.md): teach-by-talking on the SFT-r3 chat checkpoint.

Same two-tier protocol as lab/teach_pipeline_verify.py, re-run against the
chat-tuned best checkpoint `lab/imported/vk_sft_r3/step_1800/chat_model`
(561M, kda_mla, value-embed auto-detected; loader accepts its
`model_state_dict` key — see the loader note in lab/TEACH_ON_SFT_R3.md).
First teach loop on a base that actually converses, so both the free-run
transcript and the Immune probe surface are non-degenerate.

Protocol (identical in shape to the tiny-base demos):
  - 2 facts + 1 style, one chat session each; every session gated by the
    tier-1 stub (batch statistics), prompt/correction phrasings diverse so
    the target-collapse tripwire only fires on real collapse
  - poison batch (banana-in-everything) last; tier-1 must ROLLBACK and the
    incumbent must stay byte-untouched
  - after every session, ALL items taught so far are re-probed; retention
    r_i(k) = (m_i(k) - m_i(0)) / g_i per TEACH_PIPELINE.md
  - ONE tier-2 verdict: the accumulated adapter treated as a merge into the
    base, evaluated by the real Immune pipeline (probes + drift + gate,
    constants untouched) — the position tier-2 holds in the pipeline

Chat framing: prompts carry the model's im-start/im-end chat frame, and
taught corrections end a turn properly, so this is teaching *through* the
format the model was SFT-tuned on rather than raw continuations. The one
session fact that is persona-adjacent (favourite show) lets the trained
persona surface interact with what gets taught.

Verdict lines match teach_pipeline_verify.py exactly:
  VERDICT S{n} [kind/key] gate=stub-tier1: DECISION — reason | margins
  VERDICT MERGE [accumulated adapter vs raw base] gate=immune-tier2: ...

Usage:
  PYTHONPATH=$HOME/.local/lib/python3.12/site-packages \
    python3 lab/teach_on_sft_r3.py [--workdir lab/teach_demo_out_sft_r3]

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
from session_log import SessionLog  # noqa: E402
from consolidate import (  # noqa: E402
    load_base_model, generate_text, first_token_logits, word_digit_margin,
    consolidate, apply_user_adapter, response_nll, stub_gate,
)

CKPT = os.path.join(LAB, "imported", "vk_sft_r3", "step_1800", "chat_model")
MERGE_MAX_NEW = 48  # Immune/gate.py CLI default; identical on both sides (I8)

# Chat frame, built without writing marker literals into the repo.
IM_S = chr(60) + "|im_" + "start" + chr(62)
IM_E = chr(60) + "|im_" + "end" + chr(62)


def user_turn(q: str) -> str:
    """One chat user turn and the open assistant frame."""
    return f"{IM_S}user\n{q}{IM_E}\n{IM_S}assistant\n"


# ----------------------------------------------------------------------
# Taught items: 2 facts + 1 style.  fav_show is persona-adjacent (user pack:
# Person of Interest), so the persona SFT mass and the Immune probes see a
# taught fact that interacts with identity-flavoured text.
# ----------------------------------------------------------------------
ITEMS = {
    "fav_show": {
        "kind": "fact",
        "teach": "My favorite show is Person of Interest.",
        "probe_ctx": user_turn("What is my favorite show?") + "My favorite show is",
        "target": " Person of Interest.",
        "decoy": " Breaking Bad.",
        "gen_prompt": user_turn("What is my favorite show?"),
    },
    "math_words": {
        "kind": "style",
        "teach": "answer math in words, not digits",
        "probe_ctx": user_turn("What is 2 + 2?") + "The answer is",
        "target": " four.",
        "decoy": " 4.",
        "gen_prompt": user_turn("What is 2 + 2?"),
    },
    "tea_tin": {
        "kind": "fact",
        "teach": "I keep my tea in a blue tin.",
        "probe_ctx": user_turn("Where do I keep my tea?") + "You keep your tea in",
        "target": " a blue tin.",
        "decoy": " the cupboard.",
        "gen_prompt": user_turn("Where do I keep my tea?"),
    },
}
ORDER = ["fav_show", "math_words", "tea_tin"]

# (session_id, [(user question, scripted_approve_or_None, correction_or_None)])
# correction phrasings are diversified (tier-1 collapse tripwire invariant).
SESSIONS = {
    "fav_show": ("sess-show", [
        ("What is my favorite show?", None,
         "Your favorite show is Person of Interest."),
        ("Which show do I like best?", None,
         "You like Person of Interest the best."),
        ("Do you remember my favorite series?", "You love Person of Interest.", None),
    ]),
    "math_words": ("sess-math", [
        ("What is 2 + 2?", None, "The answer is four."),
        ("What is 5 + 3?", None, "The answer is eight."),
        ("And 1 + 1?", "The answer is two.", None),
    ]),
    "tea_tin": ("sess-tea", [
        ("Where do I keep my tea?", None, "Your tea is in a blue tin."),
        ("What is the blue tin for?", None, "The blue tin holds your tea."),
        ("Remind me where my tea lives.", "It lives in the blue tin.", None),
    ]),
}

USER = "alice"
DEVICE = "cpu"


def one_line(tag: str, kind: str, gate: str, decision: str, reason: str, extra: str = ""):
    suffix = f" | {extra}" if extra else ""
    print(f"VERDICT {tag} [{kind}] gate={gate}: {decision} — {reason}{suffix}", flush=True)


def probe_item(model, tok, item: dict) -> dict:
    nll_tgt = response_nll(model, tok, item["probe_ctx"], item["target"], DEVICE)
    nll_dec = response_nll(model, tok, item["probe_ctx"], item["decoy"], DEVICE)
    out = {
        "nll_target": nll_tgt,
        "nll_decoy": nll_dec,
        "margin": nll_dec - nll_tgt,  # >0: taught continuation preferred
        "gen": generate_text(model, tok, item["gen_prompt"], max_new_tokens=16,
                             device=DEVICE,
                             stop_ids=[tok.convert_tokens_to_ids("im" + "_end"),
                                       tok.eos_token_id]),
    }
    if item["kind"] == "style":
        logits = first_token_logits(model, tok, item["probe_ctx"], device=DEVICE)
        out["word_digit_margin"] = word_digit_margin(logits, tok)["margin"]
    return out


def probe_all(model, tok, keys) -> dict:
    return {k: probe_item(model, tok, ITEMS[k]) for k in keys}


def teach_session(log: SessionLog, model, tok, session_id: str, turns: list):
    """Chat-framed teaching: the model's real reply is generated first; the
    user rejects it and supplies the correction (a properly-closed turn), or
    approves the scripted good reply."""
    stop = [tok.convert_tokens_to_ids("im" + "_end"), tok.eos_token_id]
    for i, (q, script_ok, correction) in enumerate(turns):
        prompt = user_turn(q)
        gen = generate_text(model, tok, prompt, max_new_tokens=24,
                            device=DEVICE, stop_ids=stop)
        response = (script_ok if correction is None else gen) + IM_E
        log.log_turn(session_id, i, prompt, response)
        if correction is not None:
            # the logged turn keeps the model's own reply; the correction is
            # what becomes the positive training example
            log.mark_feedback(session_id, i, "reject", confidence=1.0,
                              note="teaching correction", correction=correction + IM_E)
            print(f"  turn {i}: Q={q!r}")
            print(f"           model said: {gen!r}  -> REJECT")
            print(f"           correction : {correction!r}")
        else:
            log.mark_feedback(session_id, i, "approve", confidence=1.0,
                              note="yes, like that")
            print(f"  turn {i}: Q={q!r}")
            print(f"           assistant : {response.rstrip()!r}  -> APPROVE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.path.join(LAB, "teach_demo_out_sft_r3"),
                    help="must be fresh/absent — an existing adapter would become the incumbent")
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()

    workdir = os.path.abspath(args.workdir)
    if os.path.exists(workdir):
        sys.exit(f"refusing to run into existing workdir {workdir} "
                 f"(incumbent contamination) — delete it or pass a fresh --workdir")
    os.makedirs(workdir)

    t0 = time.time()
    model, tok, mc = load_base_model(CKPT, device=DEVICE, target_profile="kda_mla")
    assert mc.get("linear_type") == "kda" and mc.get("full_type") == "mla", (
        f"expected kda_mla ckpt, got {mc.get('linear_type')}/{mc.get('full_type')}")
    print(f"E3 teach-on-SFT-r3: {os.path.relpath(CKPT, REPO)} "
          f"(dim {mc['dim']}, {mc['n_layers']}L kda/mla, value-embed auto, CPU fp32); "
          f"workdir {workdir}", flush=True)

    before = probe_all(model, tok, ORDER)
    for k in ORDER:
        it, p = ITEMS[k], before[k]
        print(f"  [before/{k}] gen -> {p['gen']!r}", flush=True)
        print(f"       NLL taught {p['nll_target']:.3f}  decoy {p['nll_decoy']:.3f}  "
              f"margin {p['margin']:+.3f}"
              + (f"  word-digit {p['word_digit_margin']:+.3f}"
                 if "word_digit_margin" in p else ""), flush=True)

    snapshots = {}
    decisions = {}
    for idx, key in enumerate(ORDER, 1):
        sid, turns = SESSIONS[key]
        log = SessionLog(os.path.join(workdir, f"session_{key}.jsonl"))
        teach_session(log, model, tok, sid, turns)
        res = consolidate(log, user_id=USER, workdir=workdir, device=DEVICE,
                          steps=args.steps, lr=2e-3, batch_size=4, kl_coef=0.05,
                          rank=8, alpha=16.0, base_model=model, tokenizer=tok,
                          target_profile=demo.TARGET_PROFILE, gate_fn=stub_gate)
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
                   f"n_triples {res.n_triples} "
                   f"train loss {st['loss0']:.3f}->{st['loss_final']:.3f} "
                   f"| margin {before[key]['margin']:+.3f} -> {final['margin']:+.3f}"
                   + (f" | word-digit {before[key]['word_digit_margin']:+.3f} -> "
                      f"{final['word_digit_margin']:+.3f}"
                      if "word_digit_margin" in final else "")
                   + f" | gen {final['gen']!r}"))

    # ---------------- retention table: r_i(k), d_i(k) per TEACH_PIPELINE.md ----------------
    print("\nRETENTION (margin m_i(k) after each session; g_i = gain at teach; "
          "r_i(k) = (m_i(k) - baseline) / g_i)", flush=True)
    snap_keys = ["before"] + [f"after_s{i}" for i in range(1, len(ORDER) + 1)]
    tables = {}
    for i, key in enumerate(ORDER):
        m0 = before[key]["margin"]
        g = snapshots[f"after_s{i + 1}"][key]["margin"] - m0
        row = {}
        for sk in snap_keys[1:]:
            if key in snapshots[sk]:
                mk = snapshots[sk][key]["margin"]
                row[sk] = {"margin": mk,
                           "r": ((mk - m0) / g) if abs(g) > 1e-9 else float("nan")}
        tables[key] = {"baseline": m0, "g": g, "row": row}
        cells = "  ".join(f"{sk}:{row[sk]['margin']:+.3f}(r={row[sk]['r']:.2f})"
                          for sk in row)
        print(f"  {key:12s} g={g:+.3f}  {cells}", flush=True)
    # per-session decay d_i(k)
    for key in tables:
        row = tables[key]["row"]
        ks = sorted(row)
        decays = [round(row[b]["margin"] - row[a]["margin"], 3)
                  for a, b in zip(ks, ks[1:])]
        if decays:
            print(f"  decay {key:12s} d_i(k) after teach: {decays}", flush=True)

    # ---------------- poison: tier-1 must ROLLBACK, incumbent byte-untouched ----------------
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
    one_line("S4", kind="poison/banana-batch", gate="stub-tier1",
             decision=poison.decision, reason=poison.reason,
             extra=f"incumbent adapter intact: {intact}")
    assert poison.decision == "ROLLBACK", f"tier-1 stub must reject the poison batch: {poison.decision}"
    assert intact, "tier-1 rollback must leave the incumbent untouched"

    # ---------------- tier 2: Immune merge-gate verdict on accumulated learning ----------------
    inc_delta = os.path.join(workdir, "adapters", USER, "delta.pt")
    promoted_any = any(d == "PROMOTE" for d in decisions.values())
    import immune_gate
    m_inc = os.path.join(workdir, "merge_check", "incumbent_empty")
    m_cand = os.path.join(workdir, "merge_check", "candidate_fulldelta")
    os.makedirs(m_inc, exist_ok=True)
    os.makedirs(m_cand, exist_ok=True)
    merge_metrics = None
    if promoted_any:
        shutil.copy2(inc_delta, os.path.join(m_cand, "delta.pt"))
        # Immune spawns its own fresh model pair; release the teach model so
        # laptop RAM holds at most two copies of the 561M nets.
        del model
        import gc
        gc.collect()
        verdict = immune_gate.evaluate(
            m_cand, m_inc, base_ckpt=CKPT, probes_path=None,
            target_profile=demo.TARGET_PROFILE, max_new=MERGE_MAX_NEW, verbose=False)
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
        "ckpt": CKPT, "decisions": decisions,
        "before": before, "snapshots": snapshots,
        "retention": tables,
        "poison_decision": poison.decision, "poison_reason": poison.reason,
        "incumbent_intact": intact,
        "merge": merge_metrics,
        "wall_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(workdir, "teach_on_sft_r3_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"results json: {os.path.join(workdir, 'teach_on_sft_r3_results.json')}",
          flush=True)


if __name__ == "__main__":
    main()
