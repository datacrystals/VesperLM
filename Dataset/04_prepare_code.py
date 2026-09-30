import os
import numpy as np
import multiprocessing
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
import warnings

# Suppress harmless sequence length warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
    print(f"✅ Finished {output_filename}! Total tokens packed: {total_tokens:,}")

if __name__ == "__main__":
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load your custom 65,536 vocabulary tokenizer
    tokenizer_dir = "custom_tokenizer"
    print(f"Loading custom tokenizer from {tokenizer_dir}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    
    print("\n--- Loading cached Python Data ---")
    # datasets 5.x cannot resolve the on-disk cache config for
    # bigcode/starcoderdata (cached as default-fad560847e57bb78), so read
    # the arrow shards directly. Falls back to a hub download if missing.
    from glob import glob as _glob
    from datasets import concatenate_datasets, Dataset as _DS
    _files = sorted(_glob(
        "/home/tliao/.cache/huggingface/datasets/bigcode___starcoderdata/"
        "default-fad560847e57bb78/0.0.0/*/starcoderdata-train-*.arrow"))
    if _files:
        print(f"  found {len(_files)} cached shards")
        ds_py = concatenate_datasets([_DS.from_file(f) for f in _files])
    else:
        ds_py = load_dataset("bigcode/starcoderdata", data_dir="python", split="train[:15%]")
    # Drop all the mismatched metadata immediately
    ds_py = ds_py.select_columns(["content"])
    
    print("\n--- Shuffling Code Dataset ---")
    # cpp subset not locally cached; python-only for the tiny-model proof run
    dataset = ds_py.shuffle(seed=42)
    
    def tokenize_example(example):
        text = example["content"]
        if not text: 
            return {'tokens': []}
        
        # Pretraining format: Raw Text + EOS
        tokens = tokenizer(text, add_special_tokens=False)['input_ids'] + [eos_id]
        return {'tokens': tokens}

    print("Tokenizing Code across all CPU cores...")
    num_proc = max(1, multiprocessing.cpu_count() - 1)
    
    tokenized_ds = dataset.map(
        tokenize_example,
        num_proc=num_proc,
        remove_columns=dataset.column_names, # Drop heavy strings from RAM instantly
        desc="Tokenizing Code"
    )
    
    def token_generator():
        for row in tokenized_ds:
            if row['tokens']: 
                yield row['tokens']

    # Set target to 1.5 Billion tokens
    target_token_count = 800_000_000
    output_file = os.path.join(out_dir, "code_pretrain_v2.bin")
    
    write_tokens_to_bin(token_generator(), output_file, target_tokens=target_token_count)
