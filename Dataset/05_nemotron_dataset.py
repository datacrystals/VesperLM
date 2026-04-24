import os
import numpy as np
import multiprocessing
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset, interleave_datasets
import warnings

# Suppress harmless sequence length warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIGURATION
# ==========================================
TOKENIZER_DIR = "custom_tokenizer"
OUT_DIR = "data/pretrain"
TARGET_TOKENS = 5_000_000_000  # 5 Billion tokens (adjust as needed for your run)
os.makedirs(OUT_DIR, exist_ok=True)

# Phase 1 Pre-training Blend Proportions (from Nemotron-3 Super Technical Report, Figure 10a)
# We normalize the exact chart percentages to sum to 1.0
DATASET_PROPORTIONS = {
    # Covers crawl-medium, crawl-high, syn-crawl, etc. (~48.5%)
    "nvidia/Nemotron-CC-v2.1": 0.485,
    
    # Covers standard code (~14.8%)
    "nvidia/Nemotron-Pretraining-Code-v2": 0.148,
    
    # Covers nemotron-cc-code (~2.1%)
    "nvidia/Nemotron-CC-Code-v1": 0.021,
    
    # Covers math (~8.4%)
    "nvidia/Nemotron-CC-Math-v1": 0.084,
    
    # Covers stem-sft, code-sft, general-sft (~14.3%)
    "nvidia/Nemotron-Pretraining-Specialized-v1.1": 0.143,
    
    # Covers wiki (~11.8%)
    "wikipedia": 0.119 
}

def write_tokens_to_bin(token_iterator, output_filename, target_tokens):
    """Writes chunks of tokens to a binary file efficiently."""
    buffer = []
    total_tokens = 0
    
    with open(output_filename, 'wb') as f:
        pbar = tqdm(total=target_tokens, unit="tok", desc=f"Writing {os.path.basename(output_filename)}")
        
        for tokens in token_iterator:
            buffer.extend(tokens)
            added = len(tokens)
            total_tokens += added
            pbar.update(added)
            
            # Flush to disk every 1M tokens
            if len(buffer) >= 1_000_000:
                f.write(np.array(buffer, dtype=np.uint16).tobytes())
                buffer = []
                
            if total_tokens >= target_tokens:
                break
                
        # Flush remaining tokens
        if len(buffer) > 0:
            if total_tokens > target_tokens:
                overshoot = total_tokens - target_tokens
                buffer = buffer[:-overshoot]
            f.write(np.array(buffer, dtype=np.uint16).tobytes())
            
        pbar.close()
    print(f"✅ Finished! Total tokens packed: {total_tokens:,}")

if __name__ == "__main__":
    print(f"Loading custom tokenizer from {TOKENIZER_DIR}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    
    print("\n--- Connecting to Hugging Face Data Streams ---")
    datasets_to_mix = []
    probabilities = []
    
    for repo_id, weight in DATASET_PROPORTIONS.items():
        print(f"Streaming {repo_id} (Weight: {weight:.1%})")
        
        # Wikipedia requires slightly different loading arguments
        if repo_id == "wikipedia":
            ds = load_dataset(repo_id, "20220301.en", split="train", streaming=True)
        else:
            ds = load_dataset(repo_id, split="train", streaming=True)
        
        # Standardize the text column so interleave_datasets can merge them easily
        # Most of NVIDIA's datasets use 'text', but if any use 'content', we map it.
        def standardize_columns(example):
            text = example.get('text', example.get('content', ''))
            return {'normalized_text': text}
            
        ds = ds.map(standardize_columns).select_columns(['normalized_text'])
        
        datasets_to_mix.append(ds)
        probabilities.append(weight)

    print("\n--- Interleaving Datasets based on Nemotron Phase 1 Weights ---")
    mixed_dataset = interleave_datasets(datasets_to_mix, probabilities=probabilities, seed=42)
    
    def tokenize_and_yield():
        for row in mixed_dataset:
            text = row["normalized_text"]
            if not text: 
                continue
            
            # Pretraining format: Raw Text + EOS
            tokens = tokenizer(text, add_special_tokens=False)['input_ids'] + [eos_id]
            yield tokens

    output_file = os.path.join(OUT_DIR, "nemotron_pretrain_dataset.bin")
    
    print(f"\nStarting tokenization and packing to {output_file}...")
    write_tokens_to_bin(tokenize_and_yield(), output_file, target_tokens=TARGET_TOKENS)