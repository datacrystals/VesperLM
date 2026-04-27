# nemotron_pretrain.py
import os
import sys
import threading
import queue
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset, interleave_datasets
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# USER CONFIGURATION
# ==========================================
TOKENIZER_DIR = "custom_tokenizer"
OUT_DIR = "data/pretrain"

# Change this for your model scale.
# Nemotron 3 Super: 25_000_000_000_000 (25T)
# For a 1B param model, Chinchilla-optimal is ~20B tokens.
TOTAL_TOKENS = 20_000_000_000

# Phase split from report §2.3.7: Phase 1 = 80% (diversity), Phase 2 = 20% (quality)
PHASE1_TOKENS = int(TOTAL_TOKENS * 0.80)
PHASE2_TOKENS = int(TOTAL_TOKENS * 0.20)

# If True, the script fails if any category cannot be loaded.
# If False, missing categories are skipped and remaining weights renormalized.
STRICT_MODE = False

# Threading config
NUM_TOKENIZER_WORKERS = max(1, (os.cpu_count() or 4) // 2)
TOKENIZE_BATCH_SIZE = 1000

os.makedirs(OUT_DIR, exist_ok=True)


# ==========================================
# BLEND DEFINITIONS (from Figure 10a / 10b, page 14)
# ==========================================
# Format per category:
#   "category_name": {
#       "repo": "hf/repo" or None,
#       "configs": {"ConfigName": rel_weight, ...}  OR  "single_config"  OR  None,
#       "weight": absolute_blend_weight (0.0 - 1.0)
#   }
#
# Categories with repo=None are missing from public HF (or unknown mapping).
# Fill them in when you find the source, then set STRICT_MODE=True to validate.

PHASE1_BLEND = {
    # --- Web Crawl (nvidia/Nemotron-CC-v2.1) ---
    "crawl-medium": {
        "repo": "nvidia/Nemotron-CC-v2.1",
        "configs": {"Medium-Quality": 1.0},
        "weight": 0.018,
    },
    "crawl-medium-high": {
        "repo": "nvidia/Nemotron-CC-v2.1",
        "configs": {"Medium-High-Quality": 1.0},
        "weight": 0.057,
    },
    "crawl-high": {
        "repo": "nvidia/Nemotron-CC-v2.1",
        "configs": {"High-Quality": 1.0},
        "weight": 0.065,
    },
    "syn-crawl-medium-high": {
        "repo": "nvidia/Nemotron-CC-v2.1",
        "configs": {"Medium-High-Quality-Synthetic": 1.0},
        "weight": 0.113,
    },
    "syn-crawl-high": {
        "repo": "nvidia/Nemotron-CC-v2.1",
        "configs": {"High-Quality-Synthetic": 1.0},
        "weight": 0.224,
    },

    # --- Code (nvidia/Nemotron-Pretraining-Code-v2) ---
    "code": {
        "repo": "nvidia/Nemotron-Pretraining-Code-v2",
        "configs": {
            "Nemotron-Code-Metadata": 0.20,
            "Synthetic-Question-Answering": 0.20,
            "Synthetic-Student-Teacher": 0.20,
            "Synthetic-Code-Review": 0.15,
            "Synthetic-Rewriting": 0.15,
            "Synthetic-Transpilation": 0.10,
        },
        "weight": 0.140,
    },

    # --- Nemotron-CC-Code ---
    "nemotron-cc-code": {
        "repo": "nvidia/Nemotron-CC-Code-v1",
        "configs": {"data": 1.0},
        "weight": 0.021,
    },

    # --- Math (nvidia/Nemotron-CC-Math-v1) ---
    "math": {
        "repo": "nvidia/Nemotron-CC-Math-v1",
        "configs": {"3": 0.33, "4plus": 0.33, "4plus_MIND": 0.34},
        "weight": 0.064,
    },

    # --- SFT-style Pretraining Data ---
    "code-sft": {
        "repo": "nvidia/Nemotron-Pretraining-Specialized-v1.1",
        "configs": {
            "Nemotron-Pretraining-Code-Concepts": 0.50,
            "Nemotron-Pretraining-Unconditional-Algorithmic": 0.50,
        },
        "weight": 0.039,
    },
    "stem-sft": {
        "repo": "nvidia/Nemotron-Pretraining-Specialized-v1.1",
        "configs": {
            "Nemotron-Pretraining-Formal-Logic": 0.34,
            "Nemotron-Pretraining-Economics": 0.33,
            "Nemotron-Pretraining-Multiple-Choice": 0.33,
        },
        "weight": 0.111,
    },
    "general-sft": {
        "repo": None,
        "configs": None,
        "weight": 0.002,
    },

    # --- Wikipedia ---
    "wiki": {
        "repo": "wikipedia",
        "configs": "20220301.en",
        "weight": 0.006,
    },

    # --- Multilingual ---
    "multilingual": {
        "repo": None,
        "configs": None,
        "weight": 0.050,
    },

    # --- Crawl++ ---
    "crawl++": {
        "repo": None,
        "configs": None,
        "weight": 0.018,
    },

    # --- Academic ---
    "academic": {
        "repo": None,
        "configs": None,
        "weight": 0.017,
    },

    # --- FinePDFs ---
    "finepdfs": {
        "repo": None,
        "configs": None,
        "weight": 0.061,
    },
}


PHASE2_BLEND = {
    # Web crawl: reduced diversity (no crawl-medium, syn-crawl-medium-high reduced)
    "crawl-medium-high": {
        "repo": "nvidia/Nemotron-CC-v2.1",
        "configs": {"Medium-High-Quality": 1.0},
        "weight": 0.057,
    },
    "crawl-high": {
        "repo": "nvidia/Nemotron-CC-v2.1",
        "configs": {"High-Quality": 1.0},
        "weight": 0.065,
    },
    "syn-crawl-medium-high": {
        "repo": "nvidia/Nemotron-CC-v2.1",
        "configs": {"Medium-High-Quality-Synthetic": 1.0},
        "weight": 0.062,
    },
    "syn-crawl-high": {
        "repo": "nvidia/Nemotron-CC-v2.1",
        "configs": {"High-Quality-Synthetic": 1.0},
        "weight": 0.224,
    },

    # Unchanged from Phase 1
    "code": {
        "repo": "nvidia/Nemotron-Pretraining-Code-v2",
        "configs": {
            "Nemotron-Code-Metadata": 0.20,
            "Synthetic-Question-Answering": 0.20,
            "Synthetic-Student-Teacher": 0.20,
            "Synthetic-Code-Review": 0.15,
            "Synthetic-Rewriting": 0.15,
            "Synthetic-Transpilation": 0.10,
        },
        "weight": 0.140,
    },
    "nemotron-cc-code": {
        "repo": "nvidia/Nemotron-CC-Code-v1",
        "configs": {"data": 1.0},
        "weight": 0.021,
    },
    "math": {
        "repo": "nvidia/Nemotron-CC-Math-v1",
        "configs": {"3": 0.33, "4plus": 0.33, "4plus_MIND": 0.34},
        "weight": 0.064,
    },
    "code-sft": {
        "repo": "nvidia/Nemotron-Pretraining-Specialized-v1.1",
        "configs": {
            "Nemotron-Pretraining-Code-Concepts": 0.50,
            "Nemotron-Pretraining-Unconditional-Algorithmic": 0.50,
        },
        "weight": 0.039,
    },
    "stem-sft": {
        "repo": "nvidia/Nemotron-Pretraining-Specialized-v1.1",
        "configs": {
            "Nemotron-Pretraining-Formal-Logic": 0.34,
            "Nemotron-Pretraining-Economics": 0.33,
            "Nemotron-Pretraining-Multiple-Choice": 0.33,
        },
        "weight": 0.118,
    },
    "general-sft": {
        "repo": None,
        "configs": None,
        "weight": 0.001,
    },

    # Wiki: dramatically increased for quality focus
    "wiki": {
        "repo": "wikipedia",
        "configs": "20220301.en",
        "weight": 0.064,
    },

    # Multilingual unchanged
    "multilingual": {
        "repo": None,
        "configs": None,
        "weight": 0.050,
    },

    # Crawl++: not visible in Phase 2 chart; set to 0
    "crawl++": {
        "repo": None,
        "configs": None,
        "weight": 0.000,
    },

    # Academic: reduced
    "academic": {
        "repo": None,
        "configs": None,
        "weight": 0.010,
    },

    # FinePDFs: upgraded to finepdfs-high
    "finepdfs": {
        "repo": None,
        "configs": None,
        "weight": 0.143,
    },
}


# ==========================================
# UTILITIES
# ==========================================
def standardize_columns(example):
    text = example.get("text", example.get("content", ""))
    return {"normalized_text": text}


def build_streaming_blend(blend_def, phase_name):
    """
    Validates and loads all datasets in a blend definition.
    Returns (datasets_list, probabilities_list, skipped_list).
    """
    datasets = []
    probabilities = []
    skipped = []

    print(f"\n--- Building {phase_name} ---")
    for category, spec in blend_def.items():
        repo = spec["repo"]
        configs = spec["configs"]
        weight = spec["weight"]

        if weight == 0:
            continue
        if repo is None:
            msg = f"{category}: repo=None (weight {weight:.1%})"
            if STRICT_MODE:
                raise ValueError(f"[{phase_name}] {msg}")
            print(f"  SKIP {msg}")
            skipped.append((category, weight))
            continue

        try:
            if isinstance(configs, dict):
                for cfg_name, rel_w in configs.items():
                    if rel_w == 0:
                        continue
                    abs_w = weight * rel_w
                    print(f"  LOAD {category}/{cfg_name} ({abs_w:.2%})")
                    ds = load_dataset(repo, cfg_name, split="train", streaming=True)
                    ds = ds.map(standardize_columns).select_columns(["normalized_text"])
                    datasets.append(ds)
                    probabilities.append(abs_w)
            else:
                cfg = configs if configs is not None else None
                print(f"  LOAD {category} ({weight:.1%})")
                if cfg is not None:
                    ds = load_dataset(repo, cfg, split="train", streaming=True)
                else:
                    ds = load_dataset(repo, split="train", streaming=True)
                ds = ds.map(standardize_columns).select_columns(["normalized_text"])
                datasets.append(ds)
                probabilities.append(weight)

        except Exception as e:
            msg = f"{category}: failed to load {repo} -> {e}"
            if STRICT_MODE:
                raise RuntimeError(f"[{phase_name}] {msg}")
            print(f"  SKIP {msg}")
            skipped.append((category, weight))

    if skipped:
        total = sum(probabilities)
        if total == 0:
            raise ValueError(f"[{phase_name}] All datasets skipped! Nothing to train on.")
        probabilities = [p / total for p in probabilities]
        print(f"[{phase_name}] Renormalized after skipping {len(skipped)} categories.")

    total_prob = sum(probabilities)
    print(f"[{phase_name}] Final stream count: {len(datasets)}, total weight: {total_prob:.4f}")
    return datasets, probabilities, skipped


# ==========================================
# MULTITHREADED TOKENIZATION PIPELINE
# ==========================================
_STOP = object()


def _reader_thread(dataset, text_queue, batch_size):
    """Reads from the streaming dataset and feeds batched texts into the queue."""
    try:
        batch_id = 0
        texts = []
        for row in dataset:
            text = row.get("normalized_text", "")
            if text:
                texts.append(text)
            if len(texts) >= batch_size:
                text_queue.put((batch_id, texts))
                batch_id += 1
                texts = []
        if texts:
            text_queue.put((batch_id, texts))
    except Exception as e:
        print(f"[READER ERROR] {e}", file=sys.stderr)
        raise
    finally:
        for _ in range(NUM_TOKENIZER_WORKERS):
            text_queue.put(_STOP)


def _tokenizer_worker(text_queue, token_queue, tokenizer, eos_id):
    """Tokenizes batches of texts and feeds token IDs into the output queue."""
    try:
        while True:
            item = text_queue.get()
            if item is _STOP:
                token_queue.put((None, None))
                break
            batch_id, texts = item
            results = tokenizer(texts, add_special_tokens=False)["input_ids"]
            token_batches = [tokens + [eos_id] for tokens in results]
            token_queue.put((batch_id, token_batches))
    except Exception as e:
        print(f"[TOKENIZER ERROR] {e}", file=sys.stderr)
        token_queue.put((None, None))
        raise


def write_tokens_threaded(dataset, tokenizer, eos_id, output_filename, target_tokens):
    """
    Producer-consumer pipeline:
      1 reader thread  -> pulls from streaming dataset
      N worker threads -> batched tokenization (N = NUM_TOKENIZER_WORKERS)
      1 writer thread  -> ordered write to disk
    """
    text_queue = queue.Queue(maxsize=NUM_TOKENIZER_WORKERS * 2)
    token_queue = queue.Queue(maxsize=NUM_TOKENIZER_WORKERS * 2)

    reader_t = threading.Thread(
        target=_reader_thread,
        args=(dataset, text_queue, TOKENIZE_BATCH_SIZE),
        daemon=True,
    )
    reader_t.start()

    worker_threads = []
    for _ in range(NUM_TOKENIZER_WORKERS):
        t = threading.Thread(
            target=_tokenizer_worker,
            args=(text_queue, token_queue, tokenizer, eos_id),
            daemon=True,
        )
        t.start()
        worker_threads.append(t)

    buffer = []
    total_tokens = 0
    next_batch_id = 0
    pending = {}
    finished_workers = 0

    with open(output_filename, "wb") as f:
        pbar = tqdm(total=target_tokens, unit="tok", desc=os.path.basename(output_filename))

        while finished_workers < NUM_TOKENIZER_WORKERS:
            batch_id, token_batches = token_queue.get()

            if batch_id is None:
                finished_workers += 1
                continue

            pending[batch_id] = token_batches

            while next_batch_id in pending:
                for tokens in pending[next_batch_id]:
                    buffer.extend(tokens)
                    total_tokens += len(tokens)
                    pbar.update(len(tokens))

                    if len(buffer) >= 1_000_000:
                        f.write(np.array(buffer, dtype=np.uint16).tobytes())
                        buffer = []

                    if total_tokens >= target_tokens:
                        break

                if total_tokens >= target_tokens:
                    break
                del pending[next_batch_id]
                next_batch_id += 1

            if total_tokens >= target_tokens:
                break

        if buffer and total_tokens < target_tokens:
            if total_tokens > target_tokens:
                overshoot = total_tokens - target_tokens
                buffer = buffer[:-overshoot]
                total_tokens = target_tokens
            f.write(np.array(buffer, dtype=np.uint16).tobytes())

        pbar.close()

    reader_t.join(timeout=30)
    for t in worker_threads:
        t.join(timeout=30)

    print(f"✅ Finished {os.path.basename(output_filename)}: {total_tokens:,} tokens\n")
    return total_tokens


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print(f"Loading tokenizer from {TOKENIZER_DIR}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    print(f"Tokenizer workers: {NUM_TOKENIZER_WORKERS} | Batch size: {TOKENIZE_BATCH_SIZE}")
    print(f"Total target tokens: {TOTAL_TOKENS:,} (Phase 1: {PHASE1_TOKENS:,}, Phase 2: {PHASE2_TOKENS:,})\n")

    # ---- PHASE 1 ----
    ds_list, probs, skipped1 = build_streaming_blend(PHASE1_BLEND, "Phase 1")
    mixed = interleave_datasets(ds_list, probabilities=probs, seed=42)
    out_path = os.path.join(OUT_DIR, "phase1_pretrain.bin")
    print(f"Writing Phase 1 -> {out_path} ({PHASE1_TOKENS:,} tokens)")
    write_tokens_threaded(mixed, tokenizer, eos_id, out_path, PHASE1_TOKENS)

    # ---- PHASE 2 ----
    ds_list, probs, skipped2 = build_streaming_blend(PHASE2_BLEND, "Phase 2")
    mixed = interleave_datasets(ds_list, probabilities=probs, seed=42)
    out_path = os.path.join(OUT_DIR, "phase2_pretrain.bin")
    print(f"Writing Phase 2 -> {out_path} ({PHASE2_TOKENS:,} tokens)")
    write_tokens_threaded(mixed, tokenizer, eos_id, out_path, PHASE2_TOKENS)

    # ---- SUMMARY ----
    print("=" * 60)
    print("PRETRAINING DATA SUMMARY")
    print("=" * 60)
    print(f"Total tokens requested: {TOTAL_TOKENS:,}")
    print(f"Phase 1 (80%):          {PHASE1_TOKENS:,}")
    print(f"Phase 2 (20%):          {PHASE2_TOKENS:,}")
    if skipped1 or skipped2:
        print("\nWARNING: Some categories were skipped (repo=None or load failure).")
        print("Set STRICT_MODE=True and fill in missing repos to match exactly.")
        for cat, w in skipped1 + skipped2:
            print(f"  - {cat}: {w:.1%}")
    else:
        print("\nAll categories loaded successfully. Blend matches report proportions.")
    print("=" * 60)
