import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

# We register SFT tokens now so the embedding matrix is sized correctly for pretraining
#SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|im_sep|>", "<|unk|>", "<|pad|>"]
SPECIAL_TOKENS = [
    "<|endoftext|>",   # Standard pretraining EOS / BOS
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

    print("Sampling HuggingFace FineWeb...")
    fw_ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for i, row in enumerate(fw_ds):
        if i >= sample_size // 4: break
        yield row["text"]

    print("Sampling HuggingFace Code...")
    code_ds = load_dataset("bigcode/starcoderdata", data_dir="python", split="train", streaming=True)
    for i, row in enumerate(code_ds):
        if i >= sample_size // 4: break
        yield row["content"]
        
    print("Sampling HuggingFace French...")
    fr_ds = load_dataset("uonlp/CulturaX", "fr", split="train", streaming=True)
    for i, row in enumerate(fr_ds):
        if i >= sample_size // 4: break
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
        eos_token="<|endoftext|>",
        bos_token="<|endoftext|>" # Standard practice to use endoftext for both in base pretraining
    )
    fast_tokenizer.save_pretrained(output_path)
    print(f"✅ Tokenizer saved to {output_path}")

if __name__ == "__main__":
    os.makedirs("custom_tokenizer", exist_ok=True)
    train_custom_tokenizer("InputDatasets/AO3", "custom_tokenizer")
