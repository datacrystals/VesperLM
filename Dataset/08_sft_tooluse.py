"""
08_sft_tooluse.py

Generates synthetic tool-use SFT traces (no downloads) and tokenizes them
into the same interleaved [token, mask, token, mask, ...] uint16 format as
05_sft_oasst.py / 06_sft_vesper.py / 07_sft_openhermes.py, so the result can
be mixed via data/sft/index.txt with any weight.

Conversation format (ChatML):

    <|im_start|>user\n{request}<|im_end|>\n
    <|im_start|>assistant\n{optional <|thought|> block}<|tool_call|>{json}</|tool_call|><|im_end|>\n
    <|im_start|>tool\n{observation}<|im_end|>\n
    <|im_start|>assistant\n{final answer}<|im_end|>\n
    <endoftext>

Tool-call JSON schema (ONE schema everywhere):
    {"tool": "<name>", "args": { ... }}

Loss mask: 1 only on assistant *content* tokens (thought blocks, tool_call
blocks, and final answers). Header <|im_start|>role\\n and footer <|im_end|>\\n
are masked 0, matching the other SFT scripts.
"""

import os
import json
import random
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

# ==========================================
# CONFIG
# ==========================================
TOKENIZER_DIR   = "custom_tokenizer"
OUT_DIR         = "data/sft"
OUT_FILE        = "tooluse_sft.bin"
TARGET_TRACES   = 50000       # expanded 2026-09-30 (was 6000)
MAX_SEQ_LEN     = 2048
THOUGHT_PROB    = 0.15        # ~15% of traces get a <|thought|> block
SEED            = 1234

os.makedirs(OUT_DIR, exist_ok=True)
rng = random.Random(SEED)

# ==========================================
# SYNTHETIC CONTENT POOLS
# ==========================================

# ---------- bash / ls ----------
LS_REQUESTS = [
    "List all files in the current directory, including hidden ones.",
    "What files are in this directory? Show full details.",
    "Show me a detailed listing of the current directory.",
    "Can you list the contents of the current folder with permissions?",
    "Show everything in the working directory with sizes and permissions.",
    "List the directory contents in long format.",
    "I need to see all files here, including dotfiles.",
]

LS_OUTPUT = """total 24
drwxr-xr-x  4 user user 4096 Mar 14 09:12 .
drwxr-xr-x 11 user user 4096 Mar 14 09:10 ..
drwxr-xr-x  2 user user 4096 Mar 14 09:12 data
drwxr-xr-x  2 user user 4096 Mar 12 14:03 src
-rw-r--r--  1 user user  152 Mar 13 18:44 README.md
-rw-r--r--  1 user user  834 Mar 14 09:12 main.py
-rw-r--r--  1 user user  421 Mar 12 14:03 requirements.txt
-rw-r--r--  1 user user  199 Mar 13 17:20 .gitignore"""

LS_FINAL = [
    "The directory contains:\n\n- `data/` and `src/` directories\n- `README.md` (152 bytes)\n- `main.py` (834 bytes)\n- `requirements.txt` (421 bytes)\n- `.gitignore` (199 bytes)\n\nIt looks like a small Python project.",
    "Here's what's in the directory:\n\n- Two folders: `data/` and `src/`\n- Three files: `README.md`, `main.py`, `requirements.txt`\n- One hidden file: `.gitignore`\n\nLet me know if you'd like me to look inside any of them.",
    "Contents of the current directory:\n- `data/` — directory\n- `src/` — directory\n- `README.md` — 152 bytes\n- `main.py` — 834 bytes\n- `requirements.txt` — 421 bytes\n- `.gitignore` — hidden file, 199 bytes\n\nWant me to inspect any of these files?",
]

# ---------- cat / read_file ----------
READ_REQUESTS = [
    "Read the file config.yaml and tell me what's in it.",
    "Show me the contents of config.yaml.",
    "Can you open config.yaml and show me its contents?",
    "What does the config.yaml file contain?",
    "Print the contents of config.yaml, please.",
    "Look at config.yaml and summarize what's configured.",
]

READ_OUTPUT = """port: 8080
host: 0.0.0.0
debug: false
log_level: info
database:
  url: postgresql://localhost:5432/appdb
  pool_size: 10
cache:
  enabled: true
  ttl: 300"""

READ_FINAL = [
    "The `config.yaml` sets up a web server on port 8080 listening on all interfaces, with debug mode off and `info` level logging. It connects to a PostgreSQL database at `localhost:5432/appdb` with a pool of 10 connections, and enables a cache with a 300 second TTL.",
    "Here's a summary of `config.yaml`:\n\n- Server: port 8080, host 0.0.0.0, debug disabled\n- Logging: level `info`\n- Database: PostgreSQL at `localhost:5432/appdb`, pool size 10\n- Cache: enabled with a 300 second TTL\n\nThe setup looks standard for a small production service.",
    "The file contains a server config: it runs on port 8080 (host 0.0.0.0) with debug off and info-level logs. The database is PostgreSQL on localhost with a connection pool of 10, and there's a cache layer enabled with a 5 minute TTL.",
]

# ---------- grep / search ----------
GREP_REQUESTS = [
    "Find all lines in the src directory that mention 'TODO'.",
    "Search the src folder for any TODO comments.",
    "Can you grep the src directory for 'TODO'?",
    "I want to find every TODO in the code under src/.",
    "Locate any TODO comments in the source files under src.",
]

GREP_OUTPUT = """src/main.py:12:# TODO: add input validation
src/main.py:47:# TODO: refactor this loop
src/utils.py:8:# TODO: handle empty input
src/server.py:23:# TODO: add graceful shutdown"""

GREP_FINAL = [
    "I found 4 TODO comments:\n\n1. `src/main.py:12` — add input validation\n2. `src/main.py:47` — refactor the loop\n3. `src/utils.py:8` — handle empty input\n4. `src/server.py:23` — add graceful shutdown\n\nThe two in `main.py` are probably the highest priority since it's the entry point.",
    "There are 4 TODOs in the source:\n\n- `src/main.py` lines 12 and 47 (input validation, loop refactor)\n- `src/utils.py` line 8 (empty input handling)\n- `src/server.py` line 23 (graceful shutdown)\n\nWant me to start working through any of them?",
    "4 TODOs found across 3 files:\n\n- `main.py` (2): input validation, loop refactor\n- `utils.py` (1): empty input handling\n- `server.py` (1): graceful shutdown\n\nThe graceful shutdown one in `server.py` could matter if this service runs behind a load balancer.",
]

# ---------- python -c / calculator ----------
CALC_REQUESTS = [
    "What is 17 * 23 + 145?",
    "Calculate 17 * 23 + 145 for me.",
    "Compute the value of 17 * 23 + 145.",
    "Can you work out 17 * 23 + 145?",
    "Please evaluate 17 * 23 + 145.",
    "How much is 17 * 23 + 145?",
]

CALC_OUTPUT = "536"

CALC_FINAL = [
    "17 * 23 + 145 = 536.",
    "The result is 536.",
    "That works out to 536.",
]

# ---------- git ----------
GIT_REQUESTS = [
    "Show me the last few commits in this repo.",
    "What are the recent commits here?",
    "Can you show the git log for this repository?",
    "I'd like to see the recent commit history.",
    "Show the latest commits with their hashes.",
]

GIT_OUTPUT = """commit a3f9c21 (HEAD -> main, origin/main)
Author: dev <dev@example.com>
Date:   Fri Mar 14 09:05:12 2026 -0700

    Fix off-by-one in batch sampler

commit 7b2e104
Author: dev <dev@example.com>
Date:   Thu Mar 13 16:40:55 2026 -0700

    Add unit tests for data loader

commit c91d5f8
Author: dev <dev@example.com>
Date:   Wed Mar 12 11:22:30 2026 -0700

    Initial commit"""

GIT_FINAL = [
    "The last 3 commits on `main`:\n\n1. `a3f9c21` (HEAD) — Fix off-by-one in batch sampler\n2. `7b2e104` — Add unit tests for data loader\n3. `c91d5f8` — Initial commit\n\nAll three are by `dev`. The working tree is up to date with `origin/main`.",
    "Recent history:\n\n- `a3f9c21` — Fix off-by-one in batch sampler (HEAD -> main, origin/main)\n- `7b2e104` — Add unit tests for data loader\n- `c91d5f8` — Initial commit\n\nThe most recent change was the batch sampler fix.",
    "Here are the 3 most recent commits:\n\n1. `a3f9c21` — Fix off-by-one in batch sampler (currently checked out on `main`)\n2. `7b2e104` — Add unit tests for data loader\n3. `c91d5f8` — Initial commit",
]

# ---------- write_file ----------
WRITE_REQUESTS = [
    "Create a file called notes.txt with the text 'Remember to update dependencies'.",
    "Write 'Remember to update dependencies' to a new file named notes.txt.",
    "Save 'Remember to update dependencies' into notes.txt.",
    "Make a notes.txt file containing 'Remember to update dependencies'.",
    "Put the text 'Remember to update dependencies' in a file called notes.txt.",
]

WRITE_OUTPUT = "Wrote 32 bytes to notes.txt"

WRITE_FINAL = [
    "Done — `notes.txt` now contains 'Remember to update dependencies'.",
    "Created `notes.txt` with the text 'Remember to update dependencies'.",
    "`notes.txt` has been created with that note.",
]

# ---------- find ----------
FIND_REQUESTS = [
    "Find all Python files in the project.",
    "Locate every .py file under the current directory.",
    "List all Python files here, including subdirectories.",
    "Which Python files exist in this project?",
]

FIND_OUTPUT = """./main.py
./src/main.py
./src/utils.py
./src/server.py
./tests/test_utils.py"""

FIND_FINAL = [
    "There are 5 Python files:\n\n- `main.py` (project root)\n- `src/main.py`, `src/utils.py`, `src/server.py`\n- `tests/test_utils.py`\n\nNote there are two `main.py` files — one at the root and one under `src/`.",
    "Found 5 Python files:\n\n1. `main.py` — root\n2. `src/main.py`\n3. `src/utils.py`\n4. `src/server.py`\n5. `tests/test_utils.py`\n\nBe careful: `main.py` exists both at the root and in `src/`.",
    "5 Python files exist:\n\n- `main.py`\n- `src/main.py`\n- `src/utils.py`\n- `src/server.py`\n- `tests/test_utils.py`\n\nThe root `main.py` and `src/main.py` are separate files.",
]

# ---------- ps / df ----------
PS_REQUESTS = [
    "Check if the server process is running.",
    "Is the api server process currently running?",
    "Check running processes for anything called 'api_server'.",
    "Can you see if api_server is up?",
]

PS_OUTPUT = """  PID TTY          TIME CMD
 1423 pts/0    00:00:12 api_server
 1877 pts/1    00:00:00 ps"""

PS_FINAL = [
    "Yes — `api_server` is running as PID 1423 and has used 12 seconds of CPU time.",
    "The process is up: `api_server`, PID 1423, 12 seconds of CPU time so far.",
    "It's running — PID 1423 (`api_server`), currently at 00:00:12 CPU time.",
]

# ==========================================
# TRACE TEMPLATES
# ==========================================

# Each template: (user_requests, command, observation, final_answers)
TEMPLATES = [
    {
        "tool": "bash",
        "requests": LS_REQUESTS,
        "command": "ls -la",
        "output": LS_OUTPUT,
        "finals": LS_FINAL,
    },
    {
        "tool": "bash",
        "requests": READ_REQUESTS,
        "command": "cat config.yaml",
        "output": READ_OUTPUT,
        "finals": READ_FINAL,
    },
    {
        "tool": "bash",
        "requests": GREP_REQUESTS,
        "command": "grep -rn \"TODO\" src/",
        "output": GREP_OUTPUT,
        "finals": GREP_FINAL,
    },
    {
        "tool": "bash",
        "requests": CALC_REQUESTS,
        "command": "python -c \"print(17 * 23 + 145)\"",
        "output": CALC_OUTPUT,
        "finals": CALC_FINAL,
    },
    {
        "tool": "bash",
        "requests": GIT_REQUESTS,
        "command": "git log --oneline -5",
        "output": GIT_OUTPUT,
        "finals": GIT_FINAL,
    },
    {
        "tool": "bash",
        "requests": WRITE_REQUESTS,
        "command": "echo 'Remember to update dependencies' > notes.txt",
        "output": WRITE_OUTPUT,
        "finals": WRITE_FINAL,
    },
    {
        "tool": "bash",
        "requests": FIND_REQUESTS,
        "command": "find . -name \"*.py\"",
        "output": FIND_OUTPUT,
        "finals": FIND_FINAL,
    },
    {
        "tool": "bash",
        "requests": PS_REQUESTS,
        "command": "ps aux | grep api_server",
        "output": PS_OUTPUT,
        "finals": PS_FINAL,
    },
]

# ==========================================
# CONVERSATION CONSTRUCTION
# ==========================================

def make_json_call(tool, command):
    """The ONE tool-call JSON schema used everywhere."""
    return json.dumps({"tool": tool, "args": {"command": command}},
                      separators=(", ", ": "))


def build_trace(rng):
    """Build one tool-use conversation as a list of ChatML turns."""
    t = rng.choice(TEMPLATES)
    user_msg = rng.choice(t["requests"])
    tool_call_json = make_json_call(t["tool"], t["command"])
    observation = t["output"]
    final = rng.choice(t["finals"])

    turns = []

    # user turn
    turns.append({"role": "user", "content": user_msg})

    # assistant turn 1: optional thought + tool call
    if rng.random() < THOUGHT_PROB:
        thought = rng.choice([
            "The user wants directory contents with full details, so `ls -la` is the right command.",
            "I should run this in the shell and then report the results back.",
            "Let me run that and see what comes back.",
            "This is a simple shell task — I'll execute it and summarize the output.",
        ])
        assistant_1 = f"<|thought|>{thought}</|thought|>\n<|tool_call|>{tool_call_json}</|tool_call|>"
    else:
        assistant_1 = f"<|tool_call|>{tool_call_json}</|tool_call|>"
    turns.append({"role": "assistant", "content": assistant_1})

    # tool turn
    turns.append({"role": "tool", "content": observation})

    # assistant turn 2: final answer
    turns.append({"role": "assistant", "content": final})

    return turns


def build_multiturn_trace(rng):
    """2-3 tool calls in one conversation (templates chained)."""
    t1, t2 = rng.sample(TEMPLATES, 2)
    turns = [{"role": "user", "content": rng.choice(t1["requests"])}]
    turns.append({"role": "assistant",
                  "content": f"<|tool_call|>{make_json_call(t1['tool'], t1['command'])}</|tool_call|>"})
    turns.append({"role": "tool", "content": t1["output"]})
    turns.append({"role": "assistant", "content": rng.choice(t1["finals"])})
    # second user request
    turns.append({"role": "user", "content": rng.choice(t2["requests"])})
    if rng.random() < THOUGHT_PROB:
        thought = rng.choice([
            "Same approach as before — run it and summarize.",
            "I'll use the shell again for this one.",
        ])
        turns.append({"role": "assistant",
                      "content": f"<|thought|>{thought}</|thought|>\n<|tool_call|>{make_json_call(t2['tool'], t2['command'])}</|tool_call|>"})
    else:
        turns.append({"role": "assistant",
                      "content": f"<|tool_call|>{make_json_call(t2['tool'], t2['command'])}</|tool_call|>"})
    turns.append({"role": "tool", "content": t2["output"]})
    turns.append({"role": "assistant", "content": rng.choice(t2["finals"])})
    return turns


# ==========================================
# CHATML FORMATTER (mirrors 05/06/07)
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


# ==========================================
# WRITER (same as 06_sft_vesper.py)
# ==========================================

def write_tokens_to_bin(generator, output_file, total_hint):
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
    print(f"Finished {output_file} — total token positions: {total:,}")
    return total


def token_generator(conversations, tokenizer, im_start_id, im_end_id, eos_id):
    token_buffer = []
    mask_buffer  = []

    for convo in conversations:
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
            yield interleaved


# ==========================================
# MAIN
# ==========================================

def main():
    print(f"Loading tokenizer from {TOKENIZER_DIR}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id   = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_id      = tokenizer.convert_tokens_to_ids("endoftext")

    print(f"Special token ids — im_start: {im_start_id}, im_end: {im_end_id}, eos: {eos_id}")

    # Build traces: ~65% single-call, ~35% multi-turn (2 tool calls)
    conversations = []
    for _ in range(TARGET_TRACES):
        if rng.random() < 0.35:
            conversations.append(build_multiturn_trace(rng))
        else:
            conversations.append(build_trace(rng))

    n_thought = sum(
        1 for c in conversations
        for turn in c
        if turn["role"] == "assistant" and "<|thought|>" in turn["content"]
    )
    print(f"Built {len(conversations):,} traces "
          f"({n_thought:,} assistant turns include a <|thought|> block)")

    out_file = os.path.join(OUT_DIR, OUT_FILE)
    total = write_tokens_to_bin(
        token_generator(conversations, tokenizer, im_start_id, im_end_id, eos_id),
        out_file,
        total_hint=None,
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
        print(f"index.txt already exists — entry ensured: {entry}")
    else:
        with open(index_path, "w") as f:
            f.write("# SFT dataset index\n")
            f.write(entry + "\n")
        print(f"Created {index_path} with entry: {entry}")

    # ---- summary ----
    n_bytes = os.path.getsize(out_file)
    print(f"\nSummary:")
    print(f"  traces:            {len(conversations):,}")
    print(f"  token positions:   {total:,}")
    print(f"  file size:         {n_bytes:,} bytes ({n_bytes // 2:,} uint16 values = {total:,} token+mask pairs)")


if __name__ == "__main__":
    main()
