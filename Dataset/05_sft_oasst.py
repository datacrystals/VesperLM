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
TARGET_TOKENS   = 50_000_000   # 50M tokens — adjust up if you want more OASST coverage
MAX_SEQ_LEN     = 1024
MIN_ASSISTANT_CHARS = 20       # Skip suspiciously short assistant turns

os.makedirs(OUT_DIR, exist_ok=True)


def write_tokens_to_bin(token_iterator, output_filename, target_tokens):
    """Writes chunks of tokens to a binary file efficiently without RAM bloat."""
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

        # Flush remainder
        if buffer:
            if total_tokens > target_tokens:
                overshoot = total_tokens - target_tokens
                buffer = buffer[:-overshoot]
            f.write(np.array(buffer, dtype=np.uint16).tobytes())

        pbar.close()

    print(f"✅ Finished {output_filename}! Total tokens packed: {total_tokens:,}")


def build_oasst_conversations(dataset):
    """
    OASST2 is a tree of messages. We walk it to extract linear
    conversation chains: pick the highest-ranked reply at each node.
    Returns a list of conversations, each being a list of
    {"role": "user"/"assistant", "content": str} dicts.
    """
    # Index all messages by their id
    messages = {row["message_id"]: row for row in dataset}

    # Find root messages (no parent) that are in English and are user-role
    roots = [
        m for m in messages.values()
        if m["parent_id"] is None
        and m["lang"] == "en"
        and m["role"] == "prompter"
    ]

    def get_best_reply(parent_id):
        """Return the highest-ranked child message for a given parent."""
        children = [
            m for m in messages.values()
            if m["parent_id"] == parent_id and m["lang"] == "en"
        ]
        if not children:
            return None
        # Prefer messages with positive rank, fall back to first available
        children.sort(key=lambda m: (m.get("rank") is None, m.get("rank") or 999))
        return children[0]

    conversations = []

    for root in roots:
        convo = []
        current = root

        while current is not None:
            role = "user" if current["role"] == "prompter" else "assistant"
            text = current["text"].strip()

            if not text:
                break

            # Skip very short assistant turns — usually noise
            if role == "assistant" and len(text) < MIN_ASSISTANT_CHARS:
                break

            convo.append({"role": role, "content": text})
            current = get_best_reply(current["message_id"])

        # Only keep conversations that have at least one full user/assistant exchange
        if len(convo) >= 2 and convo[-1]["role"] == "assistant":
            conversations.append(convo)

    return conversations


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

        # Only compute loss on assistant content (not the header/footer tokens)
        if role == "assistant":
            # mask=0 on header, mask=1 on content tokens, mask=0 on im_end + newline
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


def token_generator(conversations, tokenizer, im_start_id, im_end_id, eos_id):
    """
    Packs conversations into fixed MAX_SEQ_LEN chunks.
    Loss mask is packed alongside tokens as a parallel stream — we interleave
    them so the bin file stores [token, mask, token, mask, ...] as uint16 pairs.

    NOTE: The SFT training script unpacks these pairs back out.
    """
    token_buffer = []
    mask_buffer  = []

    for convo in conversations:
        ids, mask = format_chatml(convo, tokenizer, im_start_id, im_end_id, eos_id)

        # Truncate conversations that exceed max seq len
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

            # Interleave: [t0, m0, t1, m1, ...]
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

    print("\n--- Loading OASST2 ---")
    # Load the full message tree — we walk it ourselves rather than using the
    # pre-built 'conversation' split so we can do proper rank-based chain selection
    raw_dataset = load_dataset("OpenAssistant/oasst2", split="train+validation")
    print(f"Loaded {len(raw_dataset):,} raw messages")

    print("Building conversation chains from message tree...")
    conversations = build_oasst_conversations(raw_dataset)
    print(f"Extracted {len(conversations):,} valid English conversations")

    output_file = os.path.join(OUT_DIR, "oasst2_sft.bin")
    write_tokens_to_bin(
        token_generator(conversations, tokenizer, im_start_id, im_end_id, eos_id),
        output_file,
        target_tokens=TARGET_TOKENS * 2  # *2 because we store token+mask pairs
    )

    print(f"\n✅ OASST2 SFT data written to {output_file}")
    print(f"   Add to SFT/data/index.txt as:  oasst2_sft.bin, 1.0")