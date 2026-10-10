#!/usr/bin/env python3
"""11_vesperk_corpus.py — build the Vesper-K pretraining corpus.

Streams HF datasets, tokenizes with the custom tokenizer using a process
pool, and writes packed uint16 .bin shards (same convention as
03_fineweb.py: text + <|endoftext|> per doc, contiguous, no header).
One shard-set per source so data/index.txt weights control the mix (the
weights are consumed by Pretrain/02_pretrain_linear.py when the
`route_nonphase` config key / VESPER_ROUTE_NONPHASE=1 env flag is on —
without the flag non-phase shards are dropped, which is why the t3 base
never saw this corpus; see HANDOFF_NEXT_AGENT.md 2026-10-10 t3 postmortem).

Usage:
    python 11_vesperk_corpus.py [--scale 1.0] [--only name1,name2] [--dry-run]

Output: ../Pretrain/data/vesperk/{source}_{i}.bin (+ index fragment printed
at the end for pasting into data/index.txt).
"""

import os
import sys
import time
import argparse
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------
# Corpus definition: (name, hf_repo, hf_config, split, text_field, target_tokens)
# Gated sources that fail to load are skipped and the rest renormalize.
# ---------------------------------------------------------------
CORPUS = [
    # Quality web backbone
    ("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 8_000_000_000),
    # Diversity web
    ("dclm", "mlfoundations/dclm-baseline-1.0", None, "train", "text", 4_000_000_000),
    # Code (parquet-native; the gated stack dumps and script-based sets are unusable)
    ("code", "angie-chen55/python-github-code", None, "train", "code", 400_000_000),
    # Math
    ("finemath", "HuggingFaceTB/finemath", "finemath-4plus", "train", "text", 2_000_000_000),
    # More math (openwebmath: parquet, inline text)
    ("openwebmath", "open-web-math/open-web-math", None, "train", "text", 1_500_000_000),
    # Synthetic encyclopedic
    ("cosmopedia", "HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", "train", "text", 1_500_000_000),
    # Wikipedia
    ("wikipedia", "wikimedia/wikipedia", "20231101.en", "train", "text", 1_000_000_000),
]

TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Pretrain", "custom_tokenizer")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Pretrain", "data", "vesperk")
SHARD_TOKENS = 1_000_000_000   # ~2GB per shard (uint16)
BATCH_TEXTS = 100
FLUSH_TOKENS = 1_000_000
MAX_DOC_CHARS = 200_000   # truncate pathological docs (~50k tokens)

_tok = None
_eos = None


def _init_worker(tok_dir):
    global _tok, _eos
    from transformers import PreTrainedTokenizerFast
    _tok = PreTrainedTokenizerFast.from_pretrained(tok_dir)
    _eos = _tok.convert_tokens_to_ids("<|endoftext|>")


def _tokenize_batch(texts):
    out = []
    enc = _tok([t[:MAX_DOC_CHARS] for t in texts], add_special_tokens=False)["input_ids"]
    for ids in enc:
        if ids:
            ids.append(_eos)
            out.append(ids)
    return out


def _text_stream(repo, config, split, field):
    from datasets import load_dataset
    ds = load_dataset(repo, name=config, split=split, streaming=True)
    batch = []
    for row in ds:
        t = row.get(field)
        if t and t.strip():
            batch.append(t)
            if len(batch) >= BATCH_TEXTS:
                yield batch
                batch = []
    if batch:
        yield batch


class ShardWriter:
    def __init__(self, out_dir, source):
        self.out_dir = out_dir
        self.source = source
        self.shard_idx = 0
        self.shard_tokens = 0
        self.total_tokens = 0
        self.buf = []
        self.f = None
        self.paths = []
        self._open()

    def _open(self):
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"{self.source}_{self.shard_idx}.bin")
        self.f = open(path, "wb")
        self.paths.append(path)

    def write(self, tokens):
        self.buf.extend(tokens)
        self.total_tokens += len(tokens)
        self.shard_tokens += len(tokens)
        if len(self.buf) >= FLUSH_TOKENS:
            self._flush()
        if self.shard_tokens >= SHARD_TOKENS:
            self._flush()
            self.f.close()
            self.shard_idx += 1
            self.shard_tokens = 0
            self._open()

    def _flush(self):
        if self.buf:
            self.f.write(np.array(self.buf, dtype=np.uint16).tobytes())
            self.buf = []

    def close(self):
        self._flush()
        if self.f and not self.f.closed:
            self.f.close()
        # Drop an empty trailing shard
        if self.shard_tokens == 0 and len(self.paths) > 1:
            os.remove(self.paths[-1])
            self.paths.pop()


def build_source(name, repo, config, split, field, target, pool, dry_run=False, workers=10):
    print(f"\n=== {name}: {repo} ({config}) -> {target:,} tokens ===", flush=True)
    if dry_run:
        target = min(target, 2_000_000)
    try:
        stream = _text_stream(repo, config, split, field)
        writer = ShardWriter(OUT_DIR, name)
        pbar = tqdm(total=target, unit="tok", desc=name, file=sys.stderr)
        last_log = time.time()
        t_start = time.time()
        # Bounded in-flight window: pool.imap eagerly consumes the network
        # stream faster than workers drain it, which OOMed the box (queue of
        # text batches hit tens of GB). Cap queued batches at 2 per worker.
        from collections import deque
        pending = deque()
        max_inflight = max(4, workers * 2)
        exhausted = False
        while True:
            while not exhausted and len(pending) < max_inflight:
                try:
                    batch = next(stream)
                    pending.append(pool.apply_async(_tokenize_batch, (batch,)))
                except StopIteration:
                    exhausted = True
            if not pending:
                break
            token_lists = pending.popleft().get()
            for ids in token_lists:
                if writer.total_tokens >= target:
                    break
                writer.write(ids)
            now = time.time()
            if now - last_log > 30:
                rate = writer.total_tokens / max(now - t_start, 1)
                print(f"[{name}] {writer.total_tokens:,} tokens ({rate:,.0f} tok/s)", flush=True)
                last_log = now
            pbar.n = min(writer.total_tokens, target)
            pbar.refresh()
            if writer.total_tokens >= target:
                break
        writer.close()
        pbar.close()
        print(f"{name}: {writer.total_tokens:,} tokens -> {[os.path.basename(p) for p in writer.paths]}")
        return writer.total_tokens, [os.path.relpath(p, os.path.join(OUT_DIR, "..")) for p in writer.paths]
    except Exception as e:  # noqa: BLE001 - a gated/missing source must not kill the corpus
        print(f"!! {name} FAILED ({type(e).__name__}: {e}) — skipping, renormalize weights")
        return 0, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="scale all targets (e.g. 0.01 for a smoke)")
    ap.add_argument("--only", type=str, default="", help="comma-separated source names")
    ap.add_argument("--dry-run", action="store_true", help="2M tokens per source, verifies end-to-end")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    results = {}
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(TOKENIZER_DIR,)) as pool:
        for name, repo, config, split, field, target in CORPUS:
            if only and name not in only:
                continue
            target = int(target * args.scale)
            n, paths = build_source(name, repo, config, split, field, target, pool,
                                    args.dry_run, args.workers)
            results[name] = (n, paths)

    print("\n=== index.txt fragment (weights ~ proportional to tokens) ===")
    total = sum(n for n, _ in results.values())
    for name, (n, paths) in results.items():
        if n == 0:
            continue
        per_shard = n / len(paths)
        for p in paths:
            print(f"vesperk/{os.path.basename(p)}, {per_shard/total:.4f}")
    print(f"TOTAL: {total:,} tokens")


if __name__ == "__main__":
    sys.exit(main())
