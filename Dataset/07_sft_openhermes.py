"""
07_sft_openhermes.py

Tokenizes the OpenHermes-2.5 dataset into ChatML format with loss masking
and writes interleaved (token, mask) binary files for SFT training.

Uses the same interleaved format as 05_sft_oasst.py and 06_sft_vesper.py,
so it can be mixed via data/sft/index.txt with any desired weight.
"""

import os
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIG
# ==========================================
TOKENIZER_DIR   = "custom_tokenizer"
OUT_DIR         = "data/sft"
TARGET_TOKENS   = 50_000_000   # 50M token positions — increase if you have the VRAM/time
MAX_SEQ_LEN     = 1024
MIN_ASSISTANT_CHARS = 20       # skip suspiciously short assistant turns

os.makedirs(OUT_DIR, exist_ok=True)


def write_tokens_to_bin(token_iterator, output_filename, target_tokens):
    """
    Writes interleaved [token, mask] pairs to a binary file.
    Each yield from token_iterator is a list of alternating token and mask ints.
    """
    buffer = []
    total_pairs = 0

    pbar = tqdm(total=target_tokens, unit="tok", desc=f"Writing {os.path.basename(output_filename)}")

    with open(output_filename, 'wb') as f:
        for interleaved in token_iterator:
            buffer.extend(interleaved)
            added = len(interleaved) // 2
            total_pairs += added
            pbar.update(added)

            # Flush every ~1M token pairs
            if len(buffer) >= 2_000_000:
                f.write(np.array(buffer, dtype=np.uint16).tobytes())
                buffer = []

            if total_pairs >= target_tokens:
                break

        # Flush remainder
        if buffer:
            f.write(np.array(buffer, dtype=np.uint16).tobytes())

    pbar.close()
    print(f"✅ Finished {output_filename}! Total tokens packed: {total_pairs:,}")


def format_chatml(conversation, tokenizer, im_start_id, im_end_id, eos_id):
    """
    Formats a conversation (list of {"role": "user"/"assistant"/"system", "content": str})
    into ChatML tokens with loss masking.

    Returns (input_ids, loss_mask) where loss_mask=1 only on assistant content tokens.
    """
    input_ids = []
    loss_mask = []

    for turn in conversation:
        role    = turn["role"]
        content = turn["content"]

        # Normalise role for ChatML (OpenHermes uses "assistant", "user", "system")
        if role not in ("user", "assistant", "system"):
            continue

        # Encode role header: <|im_start|>role\n
        header_text = f"{role}\n"
        header_ids  = [im_start_id] + tokenizer.encode(header_text, add_special_tokens=False)

        # Encode content + closing token + newline
        content_ids = tokenizer.encode(content, add_special_tokens=False) + [im_end_id]
        newline_ids  = tokenizer.encode("\n", add_special_tokens=False)

        turn_ids = header_ids + content_ids + newline_ids

        # Loss only on assistant content tokens (NOT on headers, footers, or system/user turns)
        if role == "assistant":
            turn_mask = (
                [0] * len(header_ids)        # <|im_start|>role\n
                + [1] * len(content_ids)     # assistant content
                + [0] * len(newline_ids)     # <|im_end|>\n
            )
        else:
            turn_mask = [0] * len(turn_ids)

        input_ids.extend(turn_ids)
        loss_mask.extend(turn_mask)

    # Append EOS
    input_ids.append(eos_id)
    loss_mask.append(0)

    return input_ids, loss_mask


def load_openhermes_conversations(tokenizer):
    """
    Loads OpenHermes-2.5 from HuggingFace and converts to a list of conversations.
    Each conversation is a list of {'role': str, 'content': str} dicts.
    Keeps only English conversations with at least one complete user/assistant exchange.
    """
    print("Loading OpenHermes-2.5 dataset...")
    ds = load_dataset("teknium/OpenHermes-2.5", split="train")  # ~1M examples

    conversations = []

    for item in ds:
        # OpenHermes structure: item["conversations"] is a list of {"from": "system"/"human"/"gpt", "value": "..."}
        raw_conv = item.get("conversations")
        if not raw_conv:
            continue

        # Convert to our format
        conv = []
        for msg in raw_conv:
            speaker = msg.get("from", "")
            text    = msg.get("value", "").strip()

            if not text:
                continue

            # Map OpenHermes roles to ChatML
            if speaker == "human":
                role = "user"
            elif speaker == "gpt":
                role = "assistant"
            elif speaker == "system":
                role = "system"
            else:
                continue  # skip unknown roles

            # Skip very short assistant replies (likely noise)
            if role == "assistant" and len(text) < MIN_ASSISTANT_CHARS:
                conv = []  # discard this entire conversation
                break

            conv.append({"role": role, "content": text})

        # Only keep if we have at least one user/assistant exchange and ends with assistant
        if len(conv) >= 2 and conv[-1]["role"] == "assistant":
            # Basic English filter (skip if excessive non-ASCII, quick heuristic)
            full_text = " ".join([m["content"] for m in conv])
            if sum(1 for c in full_text if ord(c) > 127) / max(len(full_text), 1) > 0.3:
                continue  # likely non-English
            conversations.append(conv)

    print(f"Extracted {len(conversations):,} valid English conversations")
    return conversations


def token_generator(conversations, tokenizer, im_start_id, im_end_id, eos_id):
    """
    Packs conversations into fixed MAX_SEQ_LEN chunks.
    Yields interleaved [token, mask, token, mask, ...] lists.
    """
    token_buffer = []
    mask_buffer  = []

    for convo in conversations:
        ids, mask = format_chatml(convo, tokenizer, im_start_id, im_end_id, eos_id)

        # Truncate if needed
        if len(ids) > MAX_SEQ_LEN:
            ids  = ids[:MAX_SEQ_LEN]
            mask = mask[:MAX_SEQ_LEN]

        token_buffer.extend(ids)
        mask_buffer.extend(mask)

        # Yield full chunks
        while len(token_buffer) >= MAX_SEQ_LEN:
            chunk_ids  = token_buffer[:MAX_SEQ_LEN]
            chunk_mask = mask_buffer[:MAX_SEQ_LEN]
            token_buffer = token_buffer[MAX_SEQ_LEN:]
            mask_buffer  = mask_buffer[MAX_SEQ_LEN:]

            interleaved = []
            for t, m in zip(chunk_ids, chunk_mask):
                interleaved.append(t)
                interleaved.append(m)
            yield interleaved


if __name__ == "__main__":
    print(f"Loading custom tokenizer from {TOKENIZER_DIR}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id   = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_id      = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    print(f"Special token ids — im_start: {im_start_id}, im_end: {im_end_id}, eos: {eos_id}")

    conversations = load_openhermes_conversations(tokenizer)

    if not conversations:
        print("❌ No valid conversations extracted. Exiting.")
        exit(1)

    # Target token count is in interleaved pairs, so multiply by 2
    output_file = os.path.join(OUT_DIR, "openhermes_sft.bin")
    write_tokens_to_bin(
        token_generator(conversations, tokenizer, im_start_id, im_end_id, eos_id),
        output_file,
        target_tokens=TARGET_TOKENS * 2   # *2 because each position is token+mask
    )

    print(f"\n✅ OpenHermes SFT data written to {output_file}")
    print(f"   Add to data/sft/index.txt as:  openhermes_sft.bin, <weight>")

