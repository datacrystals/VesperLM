#!/usr/bin/env python3
"""Smallest end-to-end CPU verification of VesperLM's two-tier teach policy
(lab/TEACH_PIPELINE.md).

Tier 1 (per-session micro-promote gate): `consolidate.stub_gate` — batch
statistics only (mean reward, target-collapse tripwire, unique-response
ratio, runaway-KL bound). Legitimate teach sessions PROMOTE; a target-
collapsed poison batch must ROLLBACK with the incumbent byte-untouched.

Tier 2 (checkpoint-merge gate): the real Immune scorer pipeline via
`lab/immune_gate.py` (Immune.probes + score + drift + gate.decide, constants
untouched). Demonstrated once as a merge verdict on the accumulated adapter
vs the raw base — the position tier 2 holds in the pipeline. At this 103M
micro-session scale the drift veto is documented to fire (lab/FAILURES.md,
2026-10-16 hippo-immune-gate-microsession-veto); the line is printed as
evidence, not asserted.

Protocol: STEP_BEST of `vesper_linear_checkpoints_tiny_agent_k`, fact
(cat_name) + style (math_words) + poison (banana-in-everything), one-line
verdict per session. Same probe metric as the demos: teacher-forced
continuation margin (NLL decoy - NLL taught).

Usage:
  PYTHONPATH=$HOME/.local/lib/python3.12/site-packages \
    python3 lab/teach_pipeline_verify.py [--workdir lab/teach_demo_out_pipeline_verify]

Requires a fresh workdir (adapter deltas are weights; nothing here is for git).
CPU only, fp32.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

REPO = os.environ.get("VESPER_REPO", "/home/tliao/VesperLM")
LAB = os.path.join(REPO, "lab")
sys.path.insert(0, LAB)

import hippo_teach_demo as demo  # noqa: E402  (sets sys.path + kda_mla profile)
from session_log import SessionLog  # noqa: E402
from consolidate import (  # noqa: E402
    load_base_model, consolidate, apply_user_adapter, stub_gate,
)

CKPT = os.path.join(REPO, "Pretrain", "vesper_linear_checkpoints_tiny_agent_k", "step_best")
CANARY_PROBES = os.path.join(LAB, "immune_probes_tiny_agent_k.json")
MERGE_MAX_NEW = 32  # shorter than the 48 CLI default; canary completions are short


def one_line(tag: str, kind: str, gate: str, decision: str, reason: str, extra: str = ""):
    suffix = f" | {extra}" if extra else ""
    print(f"VERDICT {tag} [{kind}] gate={gate}: {decision} — {reason}{suffix}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.path.join(LAB, "teach_demo_out_pipeline_verify"),
                    help="must be fresh/absent — an existing adapter would become the incumbent")
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()

    workdir = os.path.abspath(args.workdir)
    if os.path.exists(workdir):
        sys.exit(f"refusing to run into existing workdir {workdir} "
                 f"(incumbent contamination) — delete it or pass a fresh --workdir")
    os.makedirs(workdir)

    model, tok, mc = load_base_model(CKPT, device="cpu", target_profile="kda_mla")
    assert mc.get("linear_type") == "kda" and mc.get("full_type") == "mla", (
        f"expected kda_mla ckpt, got {mc.get('linear_type')}/{mc.get('full_type')}")
    print(f"teach-pipeline verification on step_best of tiny_agent_k "
          f"(layer mix kda/mla, CPU fp32); workdir {workdir}", flush=True)

    order = ["cat_name", "math_words"]  # fact + style
    before = demo.probe_all(model, tok, order)

    snapshots = {}
    decisions = {}
    for idx, key in enumerate(order, 1):
        sid, turns = demo.SESSIONS[key]
        log = SessionLog(os.path.join(workdir, f"session_{key}.jsonl"))
        demo.teach_session(log, model, tok, sid, turns)
        res = consolidate(log, user_id=demo.USER, workdir=workdir, device="cpu",
                          steps=args.steps, lr=2e-3, batch_size=4, kl_coef=0.05,
                          rank=8, alpha=16.0, base_model=model, tokenizer=tok,
                          target_profile=demo.TARGET_PROFILE, gate_fn=stub_gate)
        decisions[key] = res.decision
        if res.decision == "PROMOTE":
            assert apply_user_adapter(model, workdir, demo.USER,
                                      target_profile=demo.TARGET_PROFILE)
        else:
            if not apply_user_adapter(model, workdir, demo.USER,
                                      target_profile=demo.TARGET_PROFILE):
                demo.restore_raw_base(model)
        taught = order[:idx]
        snapshots[f"after_s{idx}"] = demo.probe_all(model, tok, taught)
        st = res.train_stats
        final = snapshots[f"after_s{idx}"][key]
        one_line(
            f"S{idx}", kind=f"{demo.ITEMS[key]['kind']}/{key}", gate="stub-tier1",
            decision=res.decision, reason=res.reason,
            extra=(f"top_resp_frac {st.get('top_response_fraction'):.2f} "
                   f"n_triples {res.n_triples} | margin {before[key]['margin']:+.3f} -> "
                   f"{final['margin']:+.3f}"
                   + (f" | word-digit {before[key]['word_digit_margin']:+.3f} -> "
                      f"{final['word_digit_margin']:+.3f}"
                      if "word_digit_margin" in final else "")))
        if idx > 1:
            prev = snapshots["after_s1"]["cat_name"]["margin"]
            now = snapshots[f"after_s{idx}"]["cat_name"]["margin"]
            print(f"  retention check: cat_name margin {prev:+.3f} (taught) -> "
                  f"{now:+.3f} (after S{idx})", flush=True)

    # ---------------- poison: must ROLLBACK, incumbent byte-untouched ----------------
    plog = SessionLog(os.path.join(workdir, "session_poison.jsonl"))
    for i, prompt in enumerate(["What is 2 + 2?",
                                "What is the capital of France?",
                                "Please summarize this email.",
                                "What is 12 * 12?"]):
        plog.log_turn("sess-poison", i, prompt, "banana")
        plog.mark_feedback("sess-poison", i, "approve", confidence=1.0,
                           note="perfect, always answer like this",
                           correction="banana")
    poison = consolidate(plog, user_id=demo.USER, workdir=workdir, device="cpu",
                         steps=12, lr=2e-3, batch_size=4, kl_coef=0.05,
                         rank=8, alpha=16.0, base_model=model, tokenizer=tok,
                         target_profile=demo.TARGET_PROFILE, gate_fn=stub_gate)
    if not apply_user_adapter(model, workdir, demo.USER, target_profile=demo.TARGET_PROFILE):
        demo.restore_raw_base(model)
    after = demo.probe_all(model, tok, order)
    intact = all(abs(after[k]["margin"] - snapshots["after_s2"][k]["margin"]) < 1e-6
                 for k in order)
    one_line("S3", kind="poison/banana-batch", gate="stub-tier1",
             decision=poison.decision, reason=poison.reason,
             extra=f"incumbent adapter intact: {intact}")
    assert poison.decision == "ROLLBACK", f"tier-1 stub must reject the poison batch: {poison.decision}"
    assert intact, "tier-1 rollback must leave the incumbent untouched"

    # ---------------- tier 2: Immune merge-gate verdict on accumulated learning ----------------
    inc_delta = os.path.join(workdir, "adapters", demo.USER, "delta.pt")
    promoted_any = any(d == "PROMOTE" for d in decisions.values())
    import immune_gate
    m_inc = os.path.join(workdir, "merge_check", "incumbent_empty")
    m_cand = os.path.join(workdir, "merge_check", "candidate_fulldelta")
    os.makedirs(m_inc, exist_ok=True)
    os.makedirs(m_cand, exist_ok=True)
    if promoted_any:
        shutil.copy2(inc_delta, os.path.join(m_cand, "delta.pt"))
        verdict = immune_gate.evaluate(
            m_cand, m_inc, base_ckpt=CKPT, probes_path=CANARY_PROBES,
            target_profile=demo.TARGET_PROFILE, max_new=MERGE_MAX_NEW, verbose=False)
        agg = verdict["metrics"]["aggregate"]
        dr = verdict["metrics"]["drift"]
        one_line("MERGE", kind="accumulated adapter vs raw base",
                 gate="immune-tier2", decision=verdict["decision"],
                 reason=verdict["reason"][:120],
                 extra=(f"aggregate {agg['incumbent']:.4f} -> {agg['candidate']:.4f}, "
                        f"drift {dr['drift']:.3f} label={dr['label']} "
                        f"(micro-session scale: drift veto documented, lab/FAILURES.md "
                        f"2026-10-16; tier 2 is the checkpoint-merge gate, tier 1 stays "
                        f"the teach-loop gate)"))
    else:
        print("VERDICT MERGE skipped: no promoted adapter to evaluate", flush=True)


if __name__ == "__main__":
    main()
