#!/usr/bin/env python3
"""Build the Door 10 multi-slice calibration-scale corpus (runbook: data-scale
hypothesis test).

Writes three DISJOINT code slices + a manifest into --out-dir, sized in OLMoE
tokens (the host tokenizer):

  code_cal_pool.txt   >= --pool-tokens   nested calibration prefixes come from
                                        this stream (budgets are prefixes)
  code_heldout.txt    >= --eval-tokens   same distribution, rows disjoint from
                                        the pool (the door25 caveat set)
  code_unseen3.txt    >= --eval-tokens   THIRD slice: a fresh draw (different
                                        stream seed), never touched by fit,
                                        lambda fit, or recal

Source is `codeparrot/codeparrot-clean-valid` (the graft_d0_build_slice code
domain source).  Disjointness is enforced by md5 of each document text across
all three streams.  Run this on a host whose `datasets` install works (the
MI50 box venv currently has a pyarrow/numpy ABI mismatch) and scp the outputs.

Usage:
  python3 lab/door10_build_calscale.py \
      --tokenizer /home/tliao/.cache/vesper_d0/OLMoE-1B-7B-0924 \
      --out-dir /home/tliao/graft_door10_slice --pool-tokens 540000 \
      --eval-tokens 40000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

SOURCE = ("codeparrot/codeparrot-clean-valid", "train", "content")
MIN_CHARS = 40
SEED_POOL, SEED_HELD, SEED_UNSEEN = 11, 101, 202


def iter_docs(seed: int, limit_rows: int = 200_000):
    ds = load_dataset(SOURCE[0], split=SOURCE[1], streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2000)
    n = 0
    for row in ds:
        t = (row.get(SOURCE[2]) or "").strip()
        if len(t) < MIN_CHARS:
            continue
        yield t
        n += 1
        if n >= limit_rows:
            return


def collect(tok, seed: int, target_tokens: int, seen: set,
            max_doc_tokens: int = 0) -> tuple[list, int]:
    """Collect docs until target_tokens.  max_doc_tokens>0 truncates each doc
    in TOKEN space so a few giant files cannot monopolize an eval slice."""
    texts, n_tok = [], 0
    for t in iter_docs(seed):
        h = hashlib.md5(t.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        ids = tok.encode(t, add_special_tokens=False)
        if max_doc_tokens and len(ids) > max_doc_tokens:
            t = tok.decode(ids[:max_doc_tokens])
            ids = ids[:max_doc_tokens]
        texts.append(t)
        n_tok += len(ids)
        if n_tok >= target_tokens:
            break
    return texts, n_tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pool-tokens", type=int, default=540_000)
    ap.add_argument("--eval-tokens", type=int, default=40_000)
    ap.add_argument("--max-doc-tokens", type=int, default=3000,
                    help="per-doc token cap for eval slices (doc diversity)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(args.out_dir, exist_ok=True)
    seen: set = set()
    report = {"source": "+".join(SOURCE[:2]),
              "seeds": {"pool": SEED_POOL, "heldout": SEED_HELD,
                        "unseen3": SEED_UNSEEN}}

    for name, seed, target, max_doc in (
            ("code_cal_pool", SEED_POOL, args.pool_tokens, 0),
            ("code_heldout", SEED_HELD, args.eval_tokens, args.max_doc_tokens),
            ("code_unseen3", SEED_UNSEEN, args.eval_tokens,
             args.max_doc_tokens)):
        texts, n_tok = collect(tok, seed, target, seen,
                               max_doc_tokens=max_doc)
        with open(os.path.join(args.out_dir, f"{name}.txt"), "w") as f:
            f.write("\n\n".join(texts))
        report[name] = {"tokens": n_tok, "docs": len(texts)}
        print(f"{name:16s} {n_tok:8d} tokens  {len(texts):5d} docs", flush=True)

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(report, f, indent=1)

    ok = (report["code_cal_pool"]["tokens"] >= args.pool_tokens
          and report["code_heldout"]["tokens"] >= 32_000
          and report["code_unseen3"]["tokens"] >= 32_000)
    print("OK" if ok else "FAIL: short slice", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    # datasets/pyarrow teardown aborts some interpreters at finalization;
    # files are flushed by the time we get here.
    os._exit(main())
