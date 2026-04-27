# nemotron_pretrain.py
import os
import sys
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
# For a 700M param test run, try 10_000_000_000 (10B) or 100_000_000_000 (100B).
TOTAL_TOKENS = 5_000_000_000

# Phase split from report §2.3.7: Phase 1 = 80% (diversity), Phase 2 = 20% (quality)
PHASE1_TOKENS = int(TOTAL_TOKENS * 0.80)
PHASE2_TOKENS = int(TOTAL_TOKENS * 0.20)

# If True, the script fails if any category cannot be loaded.
# If False, missing categories are skipped and remaining weights renormalized.
STRICT_MODE = False

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
    # Report mentions 5 quality groups. The 9 public configs are distributed below.
    # Translated/DQA configs are not in the pie chart; assigned 0.0 by default.
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
    # Report splits this into code-sft, stem-sft, general-sft.
    # We map the 5 public Specialized configs to these buckets.
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
        "repo": None,   # No public HF source identified for general-sft
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
    # Proxy suggestion: "mc4", "oscar", or "HuggingFaceFW/multilingual"
    "multilingual": {
        "repo": None,
        "configs": None,
        "weight": 0.050,
    },

    # --- Crawl++ (OpenWebText + BigScience + Reddit) ---
    # Proxy suggestion: "openwebtext", "bigscience/P3" subsets, etc.
    "crawl++": {
        "repo": None,
        "configs": None,
        "weight": 0.018,
    },

    # --- Academic ---
    # Proxy suggestion: "scientific_papers" (arxiv/pubmed), "pubmed"
    "academic": {
        "repo": None,
        "configs": None,
        "weight": 0.017,
    },

    # --- FinePDFs ---
    # Proxy suggestion: "HuggingFaceFW/finepdfs" (cited in references)
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
        "weight": 0.062,  # reduced from 11.3%
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
        "weight": 0.118,  # increased from 11.1%
    },
    "general-sft": {
        "repo": None,
        "configs": None,
        "weight": 0.001,  # decreased from 0.2%
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

    # Crawl++: not visible in Phase 2 chart; set to 0 (enable if you find it)
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
        "weight": 0.143,  # increased from 6.1%
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
                # Multi-config dataset: load each config as separate stream
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
                # Single-config dataset
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

    # Renormalize if we skipped anything
    if skipped:
        total = sum(probabilities)
        if total == 0:
            raise ValueError(f"[{phase_name}] All datasets skipped! Nothing to train on.")
        probabilities = [p / total for p in probabilities]
        print(f"[{phase_name}] Renormalized after skipping {len(skipped)} categories.")

    # Final safety check
    total_prob = sum(probabilities)
    print(f"[{phase_name}] Final stream count: {len(datasets)}, total weight: {total_prob:.4f}")
    return datasets, probabilities, skipped


def write_tokens_to_bin(token_iterator, output_filename, target_tokens):
    buffer = []
    total_tokens = 0

    with open(output_filename, "wb") as f:
        pbar = tqdm(total=target_tokens, unit="tok", desc=os.path.basename(output_filename))
        for tokens in token_iterator:
            buffer.extend(tokens)
            added = len(tokens)
            total_tokens += added
            pbar.update(added)

            if len(buffer) >= 1_000_000:
                f.write(np.array(buffer, dtype=np.uint16).tobytes())
                buffer = []

            if total_tokens >= target_tokens:
                break

        # Flush remainder
        if buffer:
            if total_tokens > target_tokens:
                overshoot = total_tokens - target_tokens
                buffer = buffer[:-overshoot]
                total_tokens = target_tokens
            f.write(np.array(buffer, dtype=np.uint16).tobytes())

        pbar.close()
    print(f"✅ Finished {os.path.basename(output_filename)}: {total_tokens:,} tokens\n")
    return total_tokens


def tokenize_iterator(dataset, tokenizer, eos_id):
    for row in dataset:
        text = row["normalized_text"]
        if not text:
            continue
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"] + [eos_id]
        yield tokens


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print(f"Loading tokenizer from {TOKENIZER_DIR}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    # ---- PHASE 1 ----
    ds_list, probs, skipped1 = build_streaming_blend(PHASE1_BLEND, "Phase 1")
    mixed = interleave_datasets(ds_list, probabilities=probs, seed=42)
    out_path = os.path.join(OUT_DIR, "phase1_pretrain.bin")
    print(f"Writing Phase 1 -> {out_path} ({PHASE1_TOKENS:,} tokens)")
    write_tokens_to_bin(tokenize_iterator(mixed, tokenizer, eos_id), out_path, PHASE1_TOKENS)

    # ---- PHASE 2 ----
    ds_list, probs, skipped2 = build_streaming_blend(PHASE2_BLEND, "Phase 2")
    mixed = interleave_datasets(ds_list, probabilities=probs, seed=42)
    out_path = os.path.join(OUT_DIR, "phase2_pretrain.bin")
    print(f"Writing Phase 2 -> {out_path} ({PHASE2_TOKENS:,} tokens)")
    write_tokens_to_bin(tokenize_iterator(mixed, tokenizer, eos_id), out_path, PHASE2_TOKENS)

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
