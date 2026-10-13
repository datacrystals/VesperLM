#!/usr/bin/env python3
"""Validation for v0.2 KV-cached chat (lab/chat_v05_fast.py).

(a) LOGIT PARITY — the correctness proof: for each of 5 prompts, the parallel
    full-forward and the strictly-incremental KV loop must produce the same
    logits (max |dlogit|, mean |dlogit|, argmax agreement) on the prompt AND
    on prompt+greedy-continuation positions, and greedy completions at T=0
    must be token-identical between g0.gen_reply (full-forward per token) and
    KVDecoder.generate (one token per step over the cache).  Runs under the
    grafted state (8 plugs live) and bare for contrast.
(b) SPEED — tok/s old vs new, split into prefill (prompt tokens/s) and decode
    (new tokens/s).
(c) SMOKE — 3-turn grafted chat, SHORT excerpts only (safety filter); also
    exercises cross-turn prefix reuse.

Artifacts (box): /data/tliao/v05_fast_out/validate.json — numeric-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("ROCBLAS_TENSILE_LIBPATH",
                      "/opt/rocm-6.2.4/lib/rocblas/library")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "lab"))

import torch  # noqa: E402

import graft_d0_parity as d0  # noqa: E402
import graft_d2_crossfamily as d2  # noqa: E402
import grafted_spine_v0 as g0  # noqa: E402
import kv_cache_v05 as kv  # noqa: E402
from chat_v05_fast import build_state, pick_gpu  # noqa: E402

PROMPTS = [
    "Explain the difference between a vulnerability and an exploit. Give one "
    "example of each.",
    "Write two short sentences of atmospheric prose about a lighthouse in a "
    "storm.",
    "What is my cat's name?",
    "Using the product rule, differentiate x^3 sin(x).",
    "Summarize in one sentence why incremental decoding needs a KV cache.",
]

SMOKE_TURNS = [
    "Hi! I'm trying to get my home lab set up. Where should I start?",
    "Thanks. Now, what is my cat's name?",
    "Great. Write one short sentence about the sea at dusk.",
]


def chatml_ids(tok, user_msg, history=None):
    msgs = list(history or [])
    msgs.append({"role": "user", "content": user_msg})
    txt = tok.apply_chat_template(msgs, tokenize=False,
                                  add_generation_prompt=True)
    return tok.encode(txt, add_special_tokens=False)


def dump_json(out: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, default=str)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/data/tliao/Phi-mini-MoE-instruct")
    ap.add_argument("--registry",
                    default="/data/tliao/graft_d2_out/registry_phimoe_f620b32c.pt")
    ap.add_argument("--plugins-dir", default="/data/tliao/v05_out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--lam-override", default="")
    ap.add_argument("--default-lam", type=float, default=0.5)
    ap.add_argument("--parity-new-tokens", type=int, default=24)
    ap.add_argument("--bench-new-tokens", type=int, default=48)
    ap.add_argument("--smoke-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/data/tliao/v05_fast_out/validate.json")
    ap.add_argument("--no-fast-merge", action="store_true",
                    help="keep d0's CPU-sort expert merge (attribution run)")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    args = ap.parse_args()
    args.lam_map = {}
    if args.lam_override:
        for kv_ in args.lam_override.split(","):
            if kv_.strip():
                k, _, v = kv_.partition("=")
                args.lam_map[k.strip()] = float(v)

    results = {"format": "vesper-kv-v05-validate-v1", "args": vars(args)}

    print("=== (0) synthetic merge equivalence (fast gather vs d0 shim) ===",
          flush=True)
    merge = kv.merge_equivalence_check()
    results["merge_check"] = merge
    print(f"  prefill max_abs {merge['max_abs_prefill']:.3e}  "
          f"decode max_abs {merge['max_abs_decode']:.3e}  "
          f"pass {merge['pass']}", flush=True)

    if args.gpu == "auto" and os.environ.get("KV_GPU"):
        args.gpu = os.environ["KV_GPU"]
    if args.gpu == "auto":
        dev_i = pick_gpu()
    else:
        dev_i = int(args.gpu)
    device = f"{args.device}:{dev_i}" if args.device == "cuda" else args.device
    torch.cuda.set_device(dev_i)

    t0 = time.perf_counter()
    (model, tok, blocks, fam, gates, handles, info, missing,
     _entries, _banks) = build_state(args, device)
    results["plugins"] = info
    results["missing_plugins"] = missing
    model.config._attn_implementation = args.attn
    print(f"[validate] attn implementation: {args.attn}", flush=True)
    n_fast = 0
    if not args.no_fast_merge:
        n_fast = kv.install_fast_experts(model)
        print(f"[validate] fast expert merge on {n_fast} banks "
              f"(fixed top-k gather for decode)", flush=True)
    else:
        print("[validate] --no-fast-merge: d0 CPU-sort shim everywhere",
              flush=True)
    dec = kv.KVDecoder(model, tok, device, blocks, fam, gates, handles)
    results["load_s"] = time.perf_counter() - t0

    # ---------------- (a) parity ----------------
    print("\n=== (a) LOGIT PARITY full-forward vs incremental KV loop ===",
          flush=True)
    parity_rows = []
    for p_i, prompt in enumerate(PROMPTS):
        ids = chatml_ids(tok, prompt)
        # teacher-forced continuation: greedy ids from the full-forward path
        def full_logits_fn(x):
            return d0.model_logits(model, x.to(device))

        def run_full():
            return g0.gen_reply(tok, full_logits_fn,
                                [{"role": "user", "content": prompt}],
                                args.parity_new_tokens, device, sample=False)

        text_f, gen_ids = dec._state("grafted", run_full)
        seq = ids + gen_ids
        pg = kv.parity_ids(dec, seq, "grafted")
        pg_prompt = kv.parity_ids(dec, ids, "grafted")
        gen = kv.parity_greedy_gen(
            dec, [{"role": "user", "content": prompt}],
            args.parity_new_tokens, "grafted")
        row = {
            "prompt_i": p_i,
            "prompt_tokens": len(ids),
            "seq_tokens": len(seq),
            "prompt_only": pg_prompt,
            "prompt_plus_gen": pg,
            "greedy_identical": gen["identical"],
            "n_greedy": len(gen["ids_kv"]),
        }
        parity_rows.append(row)
        print(f"  [{p_i}] T={len(seq):3d}  max|dlogit| {pg['max_abs']:.4f}  "
              f"mean {pg['mean_abs']:.2e}  cos>={pg['min_cos']:.6f}  "
              f"argmax {pg['argmax_agree']:.3f}  "
              f"route {'ALL' if pg['route_agree_all'] else pg['route_agree']}  "
              f"greedy {'SAME' if gen['identical'] else 'DIFFER'}",
              flush=True)
        dump_json(results, args.out)

    # bare contrast on prompt 0
    bare = kv.parity_ids(dec, chatml_ids(tok, PROMPTS[0]), "bare")
    results["parity"] = parity_rows
    results["parity_bare_prompt0"] = bare
    results["parity_greedy_all_identical"] = all(
        r["greedy_identical"] for r in parity_rows)
    results["parity_max_abs_worst"] = max(
        r["prompt_plus_gen"]["max_abs"] for r in parity_rows)
    results["parity_min_cos_worst"] = min(
        r["prompt_plus_gen"]["min_cos"] for r in parity_rows)
    results["parity_route_all_identical"] = all(
        r["prompt_plus_gen"]["route_agree_all"] for r in parity_rows)
    print(f"  bare[0] max|dlogit| {bare['max_abs']:.4f}", flush=True)
    print(f"  WORST max|dlogit| {results['parity_max_abs_worst']:.4f}  "
          f"min cos {results['parity_min_cos_worst']:.6f}  "
          f"greedy all identical: {results['parity_greedy_all_identical']}  "
          f"routes all identical: {results['parity_route_all_identical']}",
          flush=True)

    # ---------------- (b) speed ----------------
    print("\n=== (b) SPEED old (full-fwd/token) vs new (KV) ===", flush=True)
    bench = kv.bench_old_vs_new(
        dec, [{"role": "user", "content": PROMPTS[0]}],
        args.bench_new_tokens, "grafted", n_warmup=1)
    results["bench"] = bench
    op, np_ = bench["old_path"], bench["new_path"]
    print(f"  old: {op['n_new_tokens']} tok in {op['wall_s']:.1f}s "
          f"= {op['tok_per_s']:.2f} tok/s", flush=True)
    print(f"  new: decode {np_['n_decode_steps']} tok in "
          f"{np_['decode_s']:.2f}s = {np_['tok_per_s_decode']:.1f} tok/s | "
          f"prefill {np_['prefill_tokens']} tok in {np_['prefill_s']:.2f}s "
          f"= {np_['tok_per_s_prefill']:.0f} tok/s | "
          f"e2e {np_['tok_per_s_end2end']:.1f} tok/s", flush=True)
    if op['tok_per_s'] > 0:
        results["speedup_decode_vs_old"] = (
            np_['tok_per_s_decode'] / op['tok_per_s'])
        print(f"  speedup (decode vs old): "
              f"{results['speedup_decode_vs_old']:.1f}x", flush=True)
    results["bench_same_greedy_ids"] = bench["same_greedy_ids"]
    dump_json(results, args.out)

    # ---------------- (c) 3-turn smoke ----------------
    print("\n=== (c) 3-TURN CHAT SMOKE (grafted, greedy, short excerpts) ===",
          flush=True)
    messages = []
    smoke_rows = []
    for i, user in enumerate(SMOKE_TURNS):
        messages.append({"role": "user", "content": user})
        text, ids_g, st = dec.generate(messages, args.smoke_new_tokens,
                                       "grafted", sample=False, seed=args.seed)
        messages.append({"role": "assistant", "content": text})
        excerpt = " ".join(text.split())[:120]
        smoke_rows.append({"turn": i, "user": user[:80], "excerpt": excerpt,
                           "n_new_tokens": len(ids_g), "stats": st})
        print(f"  [{i}] you: {user[:70]}", flush=True)
        print(f"  [{i}] asst: {excerpt}", flush=True)
        print(f"  [{i}] {st['n_new_tokens']} tok | reused "
              f"{st['reused_tokens']}/{st['prompt_tokens']} prompt tok | "
              f"decode {st['n_decode_steps'] / max(1e-6, st['decode_s']):.1f} "
              f"tok/s", flush=True)
    results["smoke"] = smoke_rows
    dump_json(results, args.out)

    print(f"\n[validate] results -> {args.out}", flush=True)
    print("[validate] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
