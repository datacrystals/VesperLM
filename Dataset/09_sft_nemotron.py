"""
09_sft_nemotron.py

Converts the nvidia/Nemotron-SFT-Agentic-v2 dataset (raw jsonl files in
the HF hub cache) into the repo's tokenized SFT stream format.

The dataset ships as OpenAI-style `messages` lists with splits
`tool_calling`, `interactive_agent` and `search`. NOTE: load_dataset()
must NOT be used on this repo — datasets 5.x fails on a metadata schema
cast. This script parses the raw jsonl files directly.

Conversation format (ChatML, identical to 08_sft_tooluse.py):

    <|im_start|>user\n{request}<|im_end|>\n
    <|im_start|>assistant\n{optional <|thought|> block}<|tool_call|>{json}</|tool_call|><|im_end|>\n
    <|im_start|>tool\n{observation}<|im_end|>\n
    <|im_start|>assistant\n{final answer}<|im_end|>\n
    <endoftext>

Mapping rules:
    - assistant message with reasoning_content -> <|thought|>...</|thought|>
    - assistant message with tool_calls (JSON string of function name +
      arguments) -> <|tool_call|>{"tool": "<name>", "args": { ... }}</|tool_call|>
      (one block per tool_call, joined with "\n")
    - assistant message with plain content -> emitted as-is (final answer
      or inter-user message)
    - tool role message -> <|im_start|>tool observation <|im_end|>
    - system role messages are skipped (they carry no trainable content;
      the repo's SFT stream has no system role)

Tool-call JSON schema (ONE schema everywhere):
    {"tool": "<name>", "args": { ... }}

Loss mask: 1 only on assistant *content* tokens (thought blocks,
tool_call blocks, final answers). Header <|im_start|>role\n and footer
<|im_end|>\n are masked 0, and tool-role observation text is masked 0,
matching the other SFT scripts (08_sft_tooluse.py).

Output: appended to data/sft/nemotron_sft.bin as an interleaved
[token, mask, token, mask, ...] uint16 stream, chunked into
MAX_SEQ_LEN positions. data/sft/index.txt gains `nemotron_sft.bin, 1.0`.
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

# ==========================================
# CONFIG
# ==========================================
TOKENIZER_DIR   = os.path.join(os.path.dirname(__file__), "custom_tokenizer")
OUT_DIR         = os.path.join(os.path.dirname(__file__), "data", "sft")
OUT_FILE        = "nemotron_sft.bin"
MAX_SEQ_LEN     = 2048
TARGET_TOKENS   = 50_000_000   # cap on total token positions (~50M)
CHUNK_LINES     = 2000         # jsonl lines parsed per chunk (memory-safe)

# Raw jsonl sources (HF hub cache snapshots). Kept as explicit paths:
# load_dataset() fails on this repo (datasets 5.x metadata cast error).
_HUB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--nvidia--Nemotron-SFT-Agentic-v2"
)
SNAPSHOTS = sorted(
    os.path.join(_HUB, "snapshots", d) for d in os.listdir(os.path.join(_HUB, "snapshots"))
)
SPLIT_FILES = ["tool_calling.jsonl", "interactive_agent.jsonl", "search.jsonl"]

# ==========================================
# RECORD -> CHATML CONVERSATION
# ==========================================

def _tool_call_json(name, args):
    """The ONE tool-call JSON schema used everywhere."""
    return json.dumps({"tool": name, "args": args}, separators=(", ", ": "))


def record_to_conversation(record):
    """Map one jsonl record (OpenAI-style messages) to repo ChatML turns.

    Returns a list of {"role", "content"} turns, or None if the record
    has no assistant content at all.
    """
    turns = []
    for msg in record.get("messages", []):
        role = msg.get("role")
        if role == "system":
            continue

        if role == "tool":
            content = msg.get("content")
            if not content:
                continue
            turns.append({"role": "tool", "content": content})
            continue

        if role == "user":
            content = msg.get("content")
            if not content:
                continue
            turns.append({"role": "user", "content": content})
            continue

        if role == "assistant":
            parts = []

            # reasoning_content -> <|thought|> block
            reasoning = msg.get("reasoning_content")
            if reasoning:
                parts.append(f"<|thought|>{reasoning}</|thought|>")

            # plain content (final answer, or text alongside tool calls)
            content = msg.get("content")
            if content:
                parts.append(content)

            # tool_calls: JSON string args -> one <|tool_call|> block each
            for tc in msg.get("tool_calls") or []:
                fn = (tc.get("function") or {})
                name = fn.get("name")
                raw_args = fn.get("arguments")
                if not name:
                    continue
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": str(raw_args)}
                parts.append(
                    f"<|tool_call|>{_tool_call_json(name, args)}</|tool_call|>"
                )

            if not parts:
                continue
            turns.append({"role": "assistant", "content": "\n".join(parts)})

    return turns or None


def iter_records(files, chunk_lines=CHUNK_LINES):
    """Yield jsonl records in chunks (memory-safe streaming)."""
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            chunk = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunk.append(json.loads(line))
                if len(chunk) >= chunk_lines:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk


# ==========================================
# CHATML FORMATTER + TOKEN GENERATOR (mirrors 08_sft_tooluse.py)
# ==========================================

def format_chatml(conversation, tokenizer, im_start_id, im_end_id, eos_id):
    input_ids = []
    loss_mask = []

    for turn in conversation:
        role    = turn["role"]
        content = turn["content"]

        header_text = f"{role}\n"
        header_ids  = [im_start_id] + tokenizer.encode(header_text, add_special_tokens=False)

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


def token_stream(records_iter, tokenizer, im_start_id, im_end_id, eos_id,
                 target_tokens, limit=None):
    """Yield interleaved [token, mask, ...] chunks of MAX_SEQ_LEN pairs."""
    token_buffer = []
    mask_buffer  = []
    total = 0
    n_records = 0

    for chunk in records_iter:
        for record in chunk:
            if limit is not None and n_records >= limit:
                break
            n_records += 1

            convo = record_to_conversation(record)
            if convo is None:
                continue

            ids, mask = format_chatml(convo, tokenizer, im_start_id, im_end_id, eos_id)

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
                total += MAX_SEQ_LEN
                yield interleaved

            if total >= target_tokens:
                break
        if total >= target_tokens or (limit is not None and n_records >= limit):
            break

    # flush the tail (partial chunk) so nothing is dropped
    if token_buffer and total < target_tokens:
        interleaved = []
        for t, m in zip(token_buffer, mask_buffer):
            interleaved.append(t)
            interleaved.append(m)
        yield interleaved

    print(f"Processed {n_records:,} records, {total:,} token positions")


# ==========================================
# WRITER (same as 06/08)
# ==========================================

def write_tokens_to_bin(generator, output_file):
    buffer = []
    total = 0

    pbar = tqdm(total=None, unit="tok", desc=f"Writing {os.path.basename(output_file)}")

    with open(output_file, "ab") as f:   # append: index.txt mixes streams
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
    print(f"Finished {output_file} — total token positions: {total:,}")
    return total


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N records (smoke test)")
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS,
                        help=f"Cap on total token positions (default {TARGET_TOKENS:,})")
    args = parser.parse_args()

    print(f"Loading tokenizer from {TOKENIZER_DIR}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id   = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_id      = tokenizer.convert_tokens_to_ids("endoftext")

    print(f"Special token ids — im_start: {im_start_id}, im_end: {im_end_id}, eos: {eos_id}")

    # resolve raw jsonl files from the hub cache snapshots
    files = []
    for snap in SNAPSHOTS:
        for split in SPLIT_FILES:
            p = os.path.join(snap, "data", split)
            if os.path.exists(p):
                files.append(p)
    if not files:
        raise FileNotFoundError(
            f"No jsonl split files found under {_HUB}/snapshots/*/data/")
    print(f"Using {len(files)} jsonl file(s):")
    for p in files:
        print(f"  {p}")

    out_file = os.path.join(OUT_DIR, OUT_FILE)
    os.makedirs(OUT_DIR, exist_ok=True)

    records = iter_records(files)
    total = write_tokens_to_bin(
        token_stream(records, tokenizer, im_start_id, im_end_id, eos_id,
                     target_tokens=args.target_tokens, limit=args.limit),
        out_file,
    )

    # ---- index.txt ----
    index_path = os.path.join(OUT_DIR, "index.txt")
    entry = f"{OUT_FILE}, 1.0"
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            existing = [ln.strip() for ln in f if ln.strip()]
        if not any(ln.split(",")[0].strip() == OUT_FILE for ln in existing):
            with open(index_path, "a") as f:
                f.write(entry + "\n")
        print(f"index.txt entry ensured: {entry}")
    else:
        with open(index_path, "w") as f:
            f.write("# SFT dataset index\n")
            f.write(entry + "\n")
        print(f"Created {index_path} with entry: {entry}")

    # ---- summary ----
    n_bytes = os.path.getsize(out_file)
    print(f"\nSummary:")
    print(f"  token positions:   {total:,}")
    print(f"  file size:         {n_bytes:,} bytes ({n_bytes // 2:,} uint16 values = {total:,} token+mask pairs)")


if __name__ == "__main__":
    main()
