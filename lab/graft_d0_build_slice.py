#!/usr/bin/env python3
"""Build the gate-scale text slice for the D0 graft-parity run (runbook §3).

Writes labeled domain files (code/math/web/chat/science) + val_mixed.txt into
--out-dir, sized in OLMoE tokens: >= --domain-tokens per domain and
>= --val-tokens mixed validation. The tiny fixtures in lab/graft_d0_texts/ stay
the plumbing set; this builds the real slice OUTSIDE the repo (it is data).

Sources are HF streaming datasets (text-only repos, no dataset scripts); each
domain can fall back to a held-out wikitext slice with a loud warning if its
primary source is unavailable — the run prints exactly what was used.

Usage:
  python3 lab/graft_d0_build_slice.py \
      --tokenizer /home/tliao/.cache/vesper_d0/OLMoE-1B-7B-0924 \
      --out-dir /home/tliao/graft_d0_slice --domain-tokens 20000 --val-tokens 80000
"""

from __future__ import annotations

import argparse
import os
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

DOMAIN_SOURCES = {
    # primary: (dataset args, split, text field, extra row filter key/value)
    "code": (("codeparrot/codeparrot-clean-valid",), "train", "content", None, None),
    "math": (("open-web-math/open-web-math",), "train", "text", None, None),
    "web": (("Salesforce/wikitext", "wikitext-2-raw-v1"), "train", "text", None, None),
    "science": (("fancyzhx/ag_news",), "train", "text", "label", 3),  # 3 = sci/tech
    "chat": (("OpenAssistant/oasst1",), "train", "text", "lang", "en"),
}
MIN_CHARS = 40


def iter_texts(ds_args, split, field, limit_rows=200_000, shuffle_seed=17,
               label_filter=None, label_field=None):
    ds = load_dataset(*ds_args, split=split, streaming=True)
    ds = ds.shuffle(seed=shuffle_seed, buffer_size=2000)
    n = 0
    for row in ds:
        if label_filter is not None and row.get(label_field) != label_filter:
            continue
        t = (row.get(field) or "").strip()
        if len(t) < MIN_CHARS:
            continue
        yield t
        n += 1
        if n >= limit_rows:
            return


def collect(tok, ds_args, split, field, target_tokens, seed, label_filter=None,
            label_field=None):
    texts = []
    n_tok = 0
    for t in iter_texts(ds_args, split, field, shuffle_seed=seed,
                        label_filter=label_filter, label_field=label_field):
        texts.append(t)
        n_tok += len(tok.encode(t, add_special_tokens=False))
        if n_tok >= target_tokens:
            break
    return texts, n_tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--domain-tokens", type=int, default=20000)
    ap.add_argument("--val-tokens", type=int, default=80000)
    ap.add_argument("--min-total-tokens", type=int, default=50000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(args.out_dir, exist_ok=True)
    report = {}

    for dom, (ds_args, split, field, lfield, lval) in DOMAIN_SOURCES.items():
        try:
            texts, n_tok = collect(tok, ds_args, split, field,
                                   args.domain_tokens, seed=11,
                                   label_filter=lval, label_field=lfield)
            src = "+".join(ds_args)
        except Exception as e:  # noqa: BLE001 — fall back loudly
            print(f"WARN: {dom} source {ds_args} failed ({e}); "
                  f"falling back to held-out wikitext", file=sys.stderr)
            texts, n_tok = collect(tok, ("Salesforce/wikitext", "wikitext-2-raw-v1"),
                                   "train", "text", args.domain_tokens,
                                   seed=31 + hash(dom) % 100)
            src = "wikitext-2-raw-v1 (FALLBACK)"
        with open(os.path.join(args.out_dir, f"{dom}.txt"), "w") as f:
            f.write("\n\n".join(texts))
        report[dom] = {"tokens": n_tok, "docs": len(texts), "source": src}
        print(f"{dom:8s} {n_tok:8d} tokens  {len(texts):5d} docs  {src}")

    # val_mixed: fresh rows from every source, interleaved (not reused rows)
    val_parts, val_tok, val_docs = [], 0, 0
    for dom, (ds_args, split, field, lfield, lval) in DOMAIN_SOURCES.items():
        per = args.val_tokens // len(DOMAIN_SOURCES) + 500
        try:
            texts, n_tok = collect(tok, ds_args, split, field, per, seed=101,
                                   label_filter=lval, label_field=lfield)
        except Exception:  # noqa: BLE001
            texts, n_tok = collect(tok, ("Salesforce/wikitext", "wikitext-2-raw-v1"),
                                   "train", "text", per, seed=202)
        val_parts.append("\n\n".join(texts))
        val_tok += n_tok
        val_docs += len(texts)
    with open(os.path.join(args.out_dir, "val_mixed.txt"), "w") as f:
        f.write("\n\n".join(val_parts))
    print(f"{'val_mixed':8s} {val_tok:8d} tokens  {val_docs:5d} docs")
    report["val_mixed"] = {"tokens": val_tok, "docs": val_docs}

    total_dom = sum(report[d]["tokens"] for d in DOMAIN_SOURCES)
    print(f"\ndomain total {total_dom} tokens, val {val_tok} tokens")
    if total_dom + val_tok < args.min_total_tokens:
        print(f"FAIL: total {total_dom + val_tok} < min {args.min_total_tokens}",
              file=sys.stderr)
        return 1
    if val_tok < 50_000:
        print(f"FAIL: val slice {val_tok} tokens < 50000", file=sys.stderr)
        return 1
    for dom in DOMAIN_SOURCES:
        if report[dom]["tokens"] < 5000:
            print(f"FAIL: domain {dom} only {report[dom]['tokens']} tokens",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    # os._exit: datasets/pyarrow stream teardown aborts the interpreter at
    # finalization on some hosts ("PyGILState_Release ... must be current");
    # the work is done and files flushed by the time we get here.
    os._exit(main())
