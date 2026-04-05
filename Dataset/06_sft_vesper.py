import os
import json
import glob
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIG
# ==========================================
TOKENIZER_DIR   = "custom_tokenizer"
VESPER_DATA_DIR = "InputDatasets/Vesper"   # *.json files live here
OUT_DIR         = "data/sft"
MAX_SEQ_LEN     = 1024
VESPER_EPOCHS   = 4    # Repeat Vesper data N times — keeps personality strong
               # relative to OASST. Tune this (3-5 is usually right).

os.makedirs(OUT_DIR, exist_ok=True)


def load_vesper_conversations(data_dir):
    """
    Loads all *.json files from data_dir.
    Each file should be a JSON array of {"user": str, "assistant": str} objects.
    Multiple files are concatenated into one flat list of conversations.
    """
    all_conversations = []
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    if not json_files:
        raise FileNotFoundError(
            f"No .json files found in {data_dir}. "
            f"Make sure your Vesper SFT data lives there."
        )

    for filepath in json_files:
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Skipping {filename} — JSON parse error: {e}")
                continue

        if not isinstance(data, list):
            print(f"  ⚠️  Skipping {filename} — expected a JSON array at top level")
            continue

        valid = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            user_text      = item.get("user", "").strip()
            assistant_text = item.get("assistant", "").strip()
            if not user_text or not assistant_text:
                continue
            # Convert to standard conversation format
            all_conversations.append([
                {"role": "user",      "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ])
            valid += 1

        print(f"  ✅ {filename}: loaded {valid} examples")

    print(f"\nTotal Vesper conversations loaded: {len(all_conversations):,}")
    return all_conversations


def format_chatml(conversation, tokenizer, im_start_id, im_end_id, eos_id):
    """
    Formats a conversation into ChatML tokens with loss masking.
    Returns (input_ids, loss_mask) where loss_mask=1 only on assistant tokens.

    Format per turn:
        <|im_start|>role\ncontent<|im_end|>\n
    """
    input_ids = []
    loss_mask = []

    for turn in conversation:
        role    = turn["role"]
        content = turn["content"]

        # Encode role header: <|im_start|>role\n
        header_text = f"{role}\n"
        header_ids  = [im_start_id] + tokenizer.encode(header_text, add_special_tokens=False)

        # Encode content + closing token + newline
        content_ids = tokenizer.encode(content, add_special_tokens=False) + [im_end_id]
        newline_ids  = tokenizer.encode("\n", add_special_tokens=False)

        turn_ids = header_ids + content_ids + newline_ids

        # Only compute loss on assistant content (not headers/footers)
        if role == "assistant":
            turn_mask = (
                [0] * len(header_ids)
                + [1] * len(content_ids)
                + [0] * len(newline_ids)
            )
        else:
            turn_mask = [0] * len(turn_ids)

        input_ids.extend(turn_ids)
        loss_mask.extend(turn_mask)

    # Append EOS
    input_ids.append(eos_id)
    loss_mask.append(0)

    return input_ids, loss_mask


def write_tokens_to_bin(token_iterator, output_filename, total_hint=None):
    """Writes interleaved [token, mask] pairs to a binary file."""
    buffer      = []
    total_pairs = 0

    pbar = tqdm(total=total_hint, unit="tok", desc=f"Writing {os.path.basename(output_filename)}")

    with open(output_filename, 'wb') as f:
        for interleaved in token_iterator:
            buffer.extend(interleaved)
            pairs_added = len(interleaved) // 2
            total_pairs += pairs_added
            pbar.update(pairs_added)

            if len(buffer) >= 2_000_000:   # flush every ~1M token pairs
                f.write(np.array(buffer, dtype=np.uint16).tobytes())
                buffer = []

        if buffer:
            f.write(np.array(buffer, dtype=np.uint16).tobytes())

    pbar.close()
    print(f"✅ Finished {output_filename}! Total tokens packed: {total_pairs:,}")


def token_generator(conversations, tokenizer, im_start_id, im_end_id, eos_id, epochs=1):
    """
    Iterates over conversations for `epochs` passes.
    Packs into MAX_SEQ_LEN chunks and yields interleaved [token, mask] lists.
    """
    token_buffer = []
    mask_buffer  = []

    for epoch in range(epochs):
        if epochs > 1:
            print(f"  Vesper epoch {epoch + 1}/{epochs}...")
        for convo in conversations:
            ids, mask = format_chatml(convo, tokenizer, im_start_id, im_end_id, eos_id)

            # Truncate if needed
            if len(ids) > MAX_SEQ_LEN:
                ids  = ids[:MAX_SEQ_LEN]
                mask = mask[:MAX_SEQ_LEN]

            token_buffer.extend(ids)
            mask_buffer.extend(mask)

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

    print(f"\n--- Loading Vesper data from {VESPER_DATA_DIR} ---")
    conversations = load_vesper_conversations(VESPER_DATA_DIR)

    if not conversations:
        print("❌ No valid conversations found. Exiting.")
        exit(1)

    total_token_hint = len(conversations) * VESPER_EPOCHS * 300  # rough avg tokens per convo

    output_file = os.path.join(OUT_DIR, "vesper_sft.bin")
    write_tokens_to_bin(
        token_generator(conversations, tokenizer, im_start_id, im_end_id, eos_id, epochs=VESPER_EPOCHS),
        output_file,
        total_hint=total_token_hint
    )

    print(f"\n✅ Vesper SFT data written to {output_file}")
    print(f"   Epochs baked in: {VESPER_EPOCHS}x (adjust VESPER_EPOCHS at top of file)")
    print(f"\n   Suggested SFT/data/index.txt entries:")
    print(f"   oasst2_sft.bin,  1.0")
    print(f"   vesper_sft.bin,  1.0   # already {VESPER_EPOCHS}x repeated, so effectively stronger")