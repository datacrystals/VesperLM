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
VESPER_DATA_DIR = "InputDatasets/Vesper"
OUT_DIR         = "data/sft"
MAX_SEQ_LEN     = 2048   # recommended bump for multi-turn
VESPER_EPOCHS   = 1

os.makedirs(OUT_DIR, exist_ok=True)


# ==========================================
# LOADING
# ==========================================
def load_vesper_conversations(data_dir):
    """
    Supports:
    FORMAT A (legacy):
        [{"user": "...", "assistant": "..."}]

    FORMAT B (multi-turn):
        [
          [
            {"user": "...", "assistant": "..."},
            {"user": "...", "assistant": "..."}
          ]
        ]
    """
    all_conversations = []
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    if not json_files:
        raise FileNotFoundError(f"No .json files found in {data_dir}")

    for filepath in json_files:
        filename = os.path.basename(filepath)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠ Skipping {filename} — JSON error: {e}")
            continue

        if not isinstance(data, list):
            print(f"⚠ Skipping {filename} — expected list at top level")
            continue

        valid = 0

        for item in data:

            # =====================================================
            # CASE 1: MULTI-TURN CONVERSATION
            # =====================================================
            if isinstance(item, list):
                convo = []

                for turn in item:
                    if not isinstance(turn, dict):
                        continue

                    u = turn.get("user", "").strip()
                    a = turn.get("assistant", "").strip()

                    if u:
                        convo.append({"role": "user", "content": u})
                    if a:
                        convo.append({"role": "assistant", "content": a})

                if len(convo) >= 2:
                    all_conversations.append(convo)
                    valid += 1

            # =====================================================
            # CASE 2: SINGLE TURN (legacy format)
            # =====================================================
            elif isinstance(item, dict):
                u = item.get("user", "").strip()
                a = item.get("assistant", "").strip()

                if u and a:
                    all_conversations.append([
                        {"role": "user", "content": u},
                        {"role": "assistant", "content": a},
                    ])
                    valid += 1

        print(f"  ✅ {filename}: loaded {valid} conversations")

    print(f"\nTotal conversations loaded: {len(all_conversations):,}")
    return all_conversations


# ==========================================
# CHATML FORMATTER
# ==========================================
def format_chatml(conversation, tokenizer, im_start_id, im_end_id, eos_id):

    input_ids = []
    loss_mask = []

    for turn in conversation:
        role = turn["role"]
        content = turn["content"]

        header_text = f"{role}\n"
        header_ids = [im_start_id] + tokenizer.encode(header_text, add_special_tokens=False)

        content_ids = tokenizer.encode(content, add_special_tokens=False) + [im_end_id]
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)

        turn_ids = header_ids + content_ids + newline_ids

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

    input_ids.append(eos_id)
    loss_mask.append(0)

    return input_ids, loss_mask


# ==========================================
# TOKEN STREAMING
# ==========================================
def token_generator(conversations, tokenizer, im_start_id, im_end_id, eos_id, epochs=1):

    token_buffer = []
    mask_buffer = []

    for epoch in range(epochs):
        if epochs > 1:
            print(f"Vesper epoch {epoch+1}/{epochs}")

        for convo in conversations:

            ids, mask = format_chatml(
                convo,
                tokenizer,
                im_start_id,
                im_end_id,
                eos_id
            )

            if len(ids) > MAX_SEQ_LEN:
                ids = ids[:MAX_SEQ_LEN]
                mask = mask[:MAX_SEQ_LEN]

            token_buffer.extend(ids)
            mask_buffer.extend(mask)

            while len(token_buffer) >= MAX_SEQ_LEN:

                chunk_ids = token_buffer[:MAX_SEQ_LEN]
                chunk_mask = mask_buffer[:MAX_SEQ_LEN]

                token_buffer = token_buffer[MAX_SEQ_LEN:]
                mask_buffer = mask_buffer[MAX_SEQ_LEN:]

                interleaved = []
                for t, m in zip(chunk_ids, chunk_mask):
                    interleaved.append(t)
                    interleaved.append(m)

                yield interleaved


# ==========================================
# WRITER
# ==========================================
def write_tokens_to_bin(generator, output_file, total_hint=None):

    buffer = []
    total = 0

    pbar = tqdm(total=total_hint, unit="tok", desc=f"Writing {os.path.basename(output_file)}")

    with open(output_file, "wb") as f:
        for chunk in generator:
            buffer.extend(chunk)
            total += len(chunk) // 2
            pbar.update(len(chunk) // 2)

            if len(buffer) > 2_000_000:
                f.write(np.array(buffer, dtype=np.uint16).tobytes())
                buffer = []

        if buffer:
            f.write(np.array(buffer, dtype=np.uint16).tobytes())

    pbar.close()
    print(f"✅ Finished {output_file} — tokens: {total:,}")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    print(f"Loading tokenizer from {TOKENIZER_DIR}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id   = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_id      = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    print(f"Special tokens: {im_start_id}, {im_end_id}, {eos_id}")

    print(f"\nLoading dataset from {VESPER_DATA_DIR}...")
    conversations = load_vesper_conversations(VESPER_DATA_DIR)

    if not conversations:
        raise RuntimeError("No valid conversations found")

    total_hint = len(conversations) * VESPER_EPOCHS * 300

    out_file = os.path.join(OUT_DIR, "vesper_sft.bin")

    write_tokens_to_bin(
        token_generator(
            conversations,
            tokenizer,
            im_start_id,
            im_end_id,
            eos_id,
            epochs=VESPER_EPOCHS
        ),
        out_file,
        total_hint=total_hint
    )

    print(f"\n✅ Done: {out_file}")
    print(f"Epochs: {VESPER_EPOCHS}")
