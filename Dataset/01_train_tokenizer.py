import os
import glob
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

# We register SFT tokens now so the embedding matrix is sized correctly for pretraining
SPECIAL_TOKENS = [
    "endoftext",   # Standard pretraining EOS / BOS
    "<|im_start|>",    # ChatML role start
    "<|im_end|>",      # ChatML role end
    "<|thought|>",     # Start CoT monologue
    "</|thought|>",    # End CoT monologue
    "<|tool_call|>",   # Start JSON tool request
    "</|tool_call|>",  # End JSON tool request
    "<|fim_prefix|>",  # Code generation: Text before cursor
    "<|fim_middle|>",  # Code generation: Text to insert at cursor
    "<|fim_suffix|>",  # Code generation: Text after cursor
    "<|file_sep|>",    # Cross-file context separator
    "<|unk|>",
    "<|pad|>"
]


def text_iterator(raw_dir: str, sample_size: int = 50000):
    print("Sampling local AO3 data for tokenizer...")
    count = 0
    if os.path.exists(raw_dir):
        for file in os.listdir(raw_dir):
            if not file.endswith('.jsonl') and not file.endswith('.json'): continue
            with open(os.path.join(raw_dir, file), 'r', encoding='utf-8') as f:
                for line in f:
                    if count >= sample_size // 4: break
                    try:
                        doc = json.loads(line)
                        if 'text' in doc:
                            yield doc['text']
                            count += 1
                    except: continue

    print("Sampling HuggingFace FineWeb (local cache)...")
    fw_ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
    fw_ds = fw_ds.select(range(min(sample_size // 4, len(fw_ds))))
    for row in fw_ds:
        yield row["text"]

    print("Sampling HuggingFace Code (local cache)...")
    code_ds = load_dataset("bigcode/starcoderdata", data_dir="python", split="train")
    code_ds = code_ds.select(range(min(sample_size // 4, len(code_ds))))
    for row in code_ds:
        yield row["content"]

    # CulturaX (French) not cached locally; use more fineweb instead
    print("Sampling extra FineWeb (replaces French)...")
    extra = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
    start = sample_size // 4
    extra = extra.select(range(start, start + sample_size // 4))
    for row in extra:
        yield row["text"]


def train_custom_tokenizer(raw_dir: str, output_path: str, vocab_size: int = 65523): # 65536 minus 13 special tokens
    print("\n--- Training Custom BPE Tokenizer ---")
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    tokenizer.train_from_iterator(text_iterator(raw_dir), trainer=trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        eos_token="endoftext",
        bos_token="endoftext" # Standard practice to use endoftext for both in base pretraining
    )
    fast_tokenizer.save_pretrained(output_path)
    print(f"✅ Tokenizer saved to {output_path}")

if __name__ == "__main__":
    os.makedirs("custom_tokenizer", exist_ok=True)
    train_custom_tokenizer("InputDatasets/AO3", "custom_tokenizer")
