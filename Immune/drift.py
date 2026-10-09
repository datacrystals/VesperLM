"""Cheap drift metric: candidate vs incumbent on a fixed prompt battery.

For each of 32 fixed prompts, both models run a full forward pass; we
take top-k logprobs (k=32) at the final position of each prompt and
compute a top-k-truncated KL(incumbent || candidate), with the mass
outside the incumbent's top-k pooled into one residual bucket so the
quantity is a proper approximate KL. The headline number is the mean
per-prompt KL clipped to [0, DRIFT_CLIP], a bounded scalar.

Thresholds (mean-KL units):
  <  ok            0.05   clean consolidation / same weights
  <  warn          0.20   monitor; still promotable if probes pass
  >= fail          0.20   veto (treated as corruption / bad merge)

Usage:
  python drift.py --incumbent CKPT_A --candidate CKPT_B [--lora-candidate L]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_backend as cb

K = 32
DRIFT_CLIP = 1.0
THRESH_WARN = 0.05
THRESH_FAIL = 0.20


def battery_prompts():
    """Fixed 32-prompt battery. Never edit casually: drift numbers are
    only comparable while this battery is frozen (see README)."""
    def chat(u):
        return f"{cb.IM_START}user\n{u}{cb.IM_END}\n{cb.IM_START}assistant\n"
    plain = [
        "The capital of France is",
        "Water boils at",
        "2 + 2 =",
        "The largest planet in the solar system is",
        "def add(a, b):",
        "Once upon a time",
        "Q: What is the speed of light?\nA:",
        "import numpy as np\n",
        "The chemical symbol for gold is",
        "One plus one equals two. Two plus two equals four. Three plus three equals",
        "To make a sandwich you need",
        "The opposite of hot is",
        "for i in range(10):",
        "Today the weather is",
        "Photosynthesis occurs in",
        "The author of Hamlet was",
    ]
    chat_u = [
        "What is the capital of Japan?",
        "List all files in the current directory.",
        "Read the file config.yaml.",
        "What is 17 * 23?",
        "Say hello.",
        "How do I sort a list in Python?",
        "What color is the sky?",
        "Add 5 and 7.",
        "Find TODO comments in main.py.",
        "Who wrote 1984?",
        "Explain gravity briefly.",
        "What is a CPU?",
        "Delete the file tmp.txt.",
        "Count the words in this sentence.",
        "What comes after Monday?",
        "Summarize: the sky is blue.",
    ]
    out = plain + [chat(u) for u in chat_u]
    assert len(out) == 32, len(out)
    return out


def _kl_topk(logp_p, logp_q, idx):
    """Approximate KL(P||Q) with P's top-k support + residual bucket."""
    import math
    eps = 1e-12
    p = logp_p[idx].exp()
    q = logp_q[idx].exp().clamp_min(eps)
    p_rest = max(1.0 - float(p.sum()), eps)
    q_rest = max(1.0 - float(q.sum()), eps)
    kl = float((p * (logp_p[idx] - logp_q[idx])).sum())
    kl += float(p_rest) * (math.log(p_rest) - math.log(q_rest))
    return max(kl, 0.0)


def measure_drift(model_inc, model_cand, tok, max_positions=4):
    """Mean top-k KL + auxiliary signals over the fixed battery."""
    import torch
    t0 = time.time()
    prompts = battery_prompts()
    kls, top1_agree, max_shift = [], 0, 0.0
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt").input_ids
        T = ids.shape[1]
        positions = list(range(max(0, T - max_positions), T))
        rows_inc = cb.seq_topk_logprobs(model_inc, ids, k=K,
                                        positions=positions)
        rows_cand = cb.seq_topk_logprobs(model_cand, ids, k=K,
                                         positions=positions)
        for (_, vi, ii, li), (_, vc, ic, lc) in zip(rows_inc, rows_cand):
            kls.append(_kl_topk(li, lc, ii))
            if int(ii[0]) == int(ic[0]):
                top1_agree += 1
            max_shift = max(max_shift, float((vi - vc).abs().max()))
    n = len(kls)
    mean_kl = sum(kls) / n
    drift = min(mean_kl, DRIFT_CLIP)
    return {
        "drift": round(drift, 6),
        "mean_topk_kl": round(mean_kl, 6),
        "max_topk_kl": round(max(kls), 6),
        "top1_agreement": round(top1_agree / n, 4),
        "max_logprob_shift": round(max_shift, 6),
        "n_positions": n,
        "k": K,
        "elapsed_sec": round(time.time() - t0, 2),
    }


def classify(drift_value):
    """Threshold labels. Absolute: nothing may loosen these at call time."""
    if drift_value < THRESH_WARN:
        return "ok"
    if drift_value < THRESH_FAIL:
        return "warn"
    return "fail"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--lora-candidate", default=None)
    ap.add_argument("--lora-incumbent", default=None)
    ap.add_argument("--target-profile", default=None,
                    help="architecture profile: gla_gqa (default) or kda_mla; "
                         "falls back to $IMMUNE_TARGET_PROFILE / "
                         "$HIPPO_TARGET_PROFILE")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = cb.load_tokenizer()
    m_inc, _ = cb.load_model(args.incumbent, lora_path=args.lora_incumbent,
                             target_profile=args.target_profile)
    m_cand, _ = cb.load_model(args.candidate, lora_path=args.lora_candidate,
                              target_profile=args.target_profile)
    m = measure_drift(m_inc, m_cand, tok)
    m["label"] = classify(m["drift"])
    m["thresholds"] = {"ok": f"< {THRESH_WARN}", "warn": f"< {THRESH_FAIL}",
                       "fail": f">= {THRESH_FAIL}", "clip": DRIFT_CLIP}
    m["incumbent"] = os.path.abspath(args.incumbent)
    m["candidate"] = os.path.abspath(args.candidate)
    text = json.dumps(m, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)
    return m


if __name__ == "__main__":
    main()
