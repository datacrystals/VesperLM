"""
10_sft_distill.py

Frontier-model chat traces -> SFT bins with rejection (refusal) stripping.

Sources (HuggingFace distillation dumps: GPT-Astra/5.x, Claude Mythos/Fable,
Kimi K3). WARNING: these are other labs' model outputs; using them for training
is against most providers' ToS and legally grey (cf. the K3/Fable distillation
dispute). Keep provenance records before any release.

Output: interleaved [token, mask] uint16 bins in data/sft/, same format as
07_sft_openhermes.py / 08_sft_tooluse.py, mixable via data/sft/index.txt.

Usage:  /home/tliao/venvs/gen/bin/python 10_sft_distill.py
"""

import os, re, json, warnings
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIG
# ==========================================
TOKENIZER_DIR = "custom_tokenizer"
OUT_DIR = "data/sft"
TARGET_TOKENS = 40_000_000
MAX_SEQ_LEN = 1024
MIN_ASSISTANT_CHARS = 20
ROWS_PER_DATASET_CAP = 400_000

# repo_id, optional model-name filter (keep rows whose model column matches)
DATASETS = [
    ("Manusagents/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset",
     r"gpt|astra|mythos|fable|k3|kimi"),
    ("r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation", r"k3|kimi"),
    ("beyoru/kimi-k3-distillation", r"k3|kimi"),
    ("WithinUsAI/claude_mythos_distilled_25k", r"mythos|claude"),
    ("saidutta69/fable-5-premium", r"fable|claude"),
]

# ==========================================
# REJECTION FILTER
# ==========================================
REFUSAL_RE = re.compile(
    r"(?:"
    r"i\s+(?:can(?:'|no)t|can\s+not|won'?t|will\s+not|am\s+not\s+able\s+to|'m\s+not\s+able\s+to)\s+"
    r"(?:help|assist|comply|provide|do\s+that|support|complete|fulfill)"
    r"|i\s+(?:must|have\s+to|will)\s+(?:decline|refuse)"
    r"|i\s+am\s+(?:unable|not\s+able)\s+to"
    r"|i'?m\s+(?:sorry|afraid|unable)"
    r"|as\s+an?\s+(?:ai|language\s+model|artificial\s+intelligence)"
    r"|against\s+(?:my|the)\s+(?:guidelines|policies|usage\s+policy|terms)"
    r"|i\s+(?:cannot|can't)\s+(?:and\s+will\s+not|and\s+won't)"
    r"|unable\s+to\s+comply|cannot\s+comply|can't\s+comply|won'?t\s+comply"
    r"|not\s+allowed\s+to\s+(?:help|assist|provide|comply)"
    r"|i\s+will\s+(?:decline|refuse)\s+(?:this|your)\s+(?:request|query)"
    r"|request\s+(?:rejected|refused|blocked)|content[_\s-]?filter|upstream\s+error"
    r")",
    re.IGNORECASE,
)

REFUSAL_HITS = {}


def is_refusal(text):
    if not text:
        return False
    m = REFUSAL_RE.search(text[:600])
    if m:
        key = m.group(0).lower()[:40]
        REFUSAL_HITS[key] = REFUSAL_HITS.get(key, 0) + 1
        return True
    return False


# ==========================================
# SCHEMA NORMALIZATION
# ==========================================
def is_rejected(item):
    d = str(item.get("disposition", "") or item.get("status", "")).lower()
    if any(x in d for x in ("reject", "refuse", "drop", "fail", "invalid", "filtered")):
        return True
    flags = item.get("quality_flags") or item.get("flags") or []
    if isinstance(flags, list) and any("reject" in str(f).lower() for f in flags):
        return True
    return False


def to_conversation(item):
    """Map the many dump schemas onto [{'role','content'}]."""
    raw = item.get("conversations") or item.get("messages") or item.get("chat") or item.get("turns")
    if isinstance(raw, str) and raw.strip()[:1] in ("[", "{"):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = None
    if raw is None:
        for key in ("messages_json", "conversations_json"):
            v = item.get(key)
            if isinstance(v, str) and v.strip()[:1] in ("[", "{"):
                try:
                    raw = json.loads(v)
                    break
                except Exception:
                    pass
    conv = []
    if isinstance(raw, list):
        for msg in raw:
            if not isinstance(msg, dict):
                continue
            role = msg.get("from") or msg.get("role") or msg.get("speaker") or ""
            text = msg.get("value") or msg.get("content") or msg.get("text") or ""
            if isinstance(text, dict):
                text = text.get("text") or text.get("content") or ""
            text = str(text).strip()
            if role in ("human", "user", "Human"):
                role = "user"
            elif role in ("gpt", "assistant", "model", "bot", "Assistant"):
                role = "assistant"
            elif role == "system":
                role = "system"
            else:
                continue
            if not text:
                continue
            conv.append({"role": role, "content": text})
    if len(conv) < 2:
        prompt = item.get("prompt") or item.get("instruction") or item.get("question")
        completion = item.get("completion") or item.get("output") or item.get("response")
        if prompt and completion:
            extra = item.get("input")
            if extra:
                prompt = f"{prompt}\n{extra}"
            conv = [{"role": "user", "content": str(prompt).strip()},
                    {"role": "assistant", "content": str(completion).strip()}]
    return conv


def model_name_of(item):
    for k in ("model", "source", "model_name", "generator", "meta"):
        v = item.get(k)
        if isinstance(v, dict):
            v = v.get("model") or v.get("name") or ""
        if isinstance(v, str) and v:
            return v
    return ""


def passes_model_filter(item, pattern):
    if not pattern:
        return True
    name = model_name_of(item)
    if not name:
        return True  # no model column: keep
    return re.search(pattern, name, re.IGNORECASE) is not None


def load_rows(repo_id, cap):
    """Streaming via datasets, falling back to raw jsonl on the hub."""
    try:
        from datasets import load_dataset
        ds = load_dataset(repo_id, split="train", streaming=True)
        for i, item in enumerate(ds):
            if i >= cap:
                break
            yield item
        return
    except Exception as e:
        print(f"  load_dataset failed for {repo_id}: {type(e).__name__}: {str(e)[:80]}")
        print("  falling back to raw jsonl...")
    try:
        from huggingface_hub import list_repo_files
        try:
            files = list_repo_files(repo_id, repo_type="dataset")
            cand = [f for f in files if f.endswith(".jsonl") or f.endswith(".jsonl.gz")]
            cand += [f for f in files if f.endswith(".parquet")][:2]
        except Exception:
            cand = ["train.jsonl", "data.jsonl", "dataset.jsonl", "train-00000-of-00001.jsonl"]
        for fname in cand:
            try:
                p = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset")
                break
            except Exception:
                p = None
        if p is None:
            print(f"  no known jsonl filename for {repo_id}, skipping")
            return
        if p.endswith(".parquet"):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(p)
            n = 0
            for batch in pf.iter_batches(batch_size=2048):
                for row in batch.to_pylist():
                    yield row
                    n += 1
                    if n >= cap:
                        return
            return
        import gzip
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= cap:
                    break
                line = line.strip()
                if line:
                    yield json.loads(line)
    except Exception as e:
        print(f"  jsonl fallback failed: {type(e).__name__}: {str(e)[:80]}")


# ==========================================
# CHATML + BIN WRITING (same as 07_sft_openhermes.py)
# ==========================================
def format_chatml(conversation, tokenizer, im_start_id, im_end_id, eos_id):
    input_ids, loss_mask = [], []
    for turn in conversation:
        role, content = turn["role"], turn["content"]
        if role not in ("user", "assistant", "system"):
            continue
        header_ids = [im_start_id] + tokenizer.encode(f"{role}\n", add_special_tokens=False)
        content_ids = tokenizer.encode(content, add_special_tokens=False) + [im_end_id]
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        turn_ids = header_ids + content_ids + newline_ids
        if role == "assistant":
            turn_mask = [0] * len(header_ids) + [1] * len(content_ids) + [0] * len(newline_ids)
        else:
            turn_mask = [0] * len(turn_ids)
        input_ids.extend(turn_ids)
        loss_mask.extend(turn_mask)
    input_ids.append(eos_id)
    loss_mask.append(0)
    return input_ids, loss_mask


def write_bin(token_iterator, output_filename, target_tokens):
    buffer, total_pairs = [], 0
    pbar = tqdm(total=target_tokens, unit="tok", desc=f"Writing {os.path.basename(output_filename)}")
    with open(output_filename, "wb") as f:
        for interleaved in token_iterator:
            buffer.extend(interleaved)
            total_pairs += len(interleaved) // 2
            pbar.update(len(interleaved) // 2)
            if len(buffer) >= 2_000_000:
                f.write(np.array(buffer, dtype=np.uint16).tobytes())
                buffer = []
            if total_pairs >= target_tokens:
                break
        if buffer:
            f.write(np.array(buffer, dtype=np.uint16).tobytes())
    pbar.close()
    print(f"Finished {output_filename}: {total_pairs:,} token positions")


def pack_generator(convs, tokenizer, im_start_id, im_end_id, eos_id):
    tok_buf, mask_buf = [], []
    for conv in convs:
        ids, mask = format_chatml(conv, tokenizer, im_start_id, im_end_id, eos_id)
        tok_buf.extend(ids)
        mask_buf.extend(mask)
        while len(tok_buf) >= MAX_SEQ_LEN + 1:
            window_t = tok_buf[:MAX_SEQ_LEN + 1]
            window_m = mask_buf[:MAX_SEQ_LEN + 1]
            tok_buf, mask_buf = tok_buf[MAX_SEQ_LEN + 1:], mask_buf[MAX_SEQ_LEN + 1:]
            inter = []
            for t, m in zip(window_t, window_m):
                inter.append(t)
                inter.append(m)
            yield inter
def main():
    print("building distill SFT")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    im_start_id = tokenizer.convert_tokens_to_ids(chr(60)+chr(124)+"im_start"+chr(124)+chr(62))
    im_end_id = tokenizer.convert_tokens_to_ids(chr(60)+chr(124)+"im_end"+chr(124)+chr(62))
    eos_id = tokenizer.eos_token_id or im_end_id
    stats = dict(kept=0, refused=0, short=0, bad_schema=0, filtered_model=0, rejected_rows=0)
    def gather():
        for repo, mfilter in DATASETS:
            print("SRC", repo)
            for item in load_rows(repo, ROWS_PER_DATASET_CAP):
                if not passes_model_filter(item, mfilter):
                    stats["filtered_model"] += 1
                    continue
                if is_rejected(item):
                    stats["rejected_rows"] += 1
                    continue
                conv = to_conversation(item)
                if not conv or conv[-1]["role"] != "assistant":
                    stats["bad_schema"] += 1
                    continue
                if any(is_refusal(m["content"]) for m in conv if m["role"] == "assistant"):
                    stats["refused"] += 1
                    continue
                if any(len(m["content"]) < MIN_ASSISTANT_CHARS for m in conv if m["role"] == "assistant"):
                    stats["short"] += 1
                    continue
                stats["kept"] += 1
                yield conv
    out = os.path.join(OUT_DIR, "distill_chat_sft.bin")
    write_bin(pack_generator(gather(), tokenizer, im_start_id, im_end_id, eos_id), out, TARGET_TOKENS)
    print("STATS", stats)
    print("REFUSAL PATTERNS")
    for k, v in sorted(REFUSAL_HITS.items(), key=lambda kv: -kv[1])[:15]:
        print("  %5d  %s" % (v, k))
if __name__ == "__main__":
    main()
