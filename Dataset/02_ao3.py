import os
import json
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. ROUTING & MAPPING CONFIGURATION
# ==========================================

TARGET_FANDOMS = {
    "Genshin Impact": "genshin", "Gundam 00": "gundam_00", "The Witch from Mercury": "gundam_wfm",
    "Iron-Blooded Orphans": "gundam_ibo", "Harry Potter": "harry_potter", "Pluribus": "pluribus",
    "Dr. STONE": "dr_stone", "Obey Me!": "obey_me", "Cyberpunk": "cyberpunk",
    "Final Fantasy XV": "ffxv", "Persona": "persona", "The Witcher": "the_witcher", "RPF": "rpf_all"
}

def parse_word_count(words_str: str) -> int:
    if not words_str: return 0
    return int(str(words_str).replace(',', ''))

def get_routing_keys(doc: dict) -> tuple[str | None, str | None]:
    metadata = doc.get('metadata', {})

    # 1. Fandom Routing
    raw_fandom = metadata.get('Fandom', '')
    if isinstance(raw_fandom, list):
        raw_fandom = ", ".join(raw_fandom)

    fandom_key = None
    for search_str, clean_name in TARGET_FANDOMS.items():
        if search_str.lower() in raw_fandom.lower():
            fandom_key = clean_name
            break

    if not fandom_key:
        return None, None

    # 2. Category Routing (Strict Filtering)
    raw_category = metadata.get('Category', 'Unknown')
    if isinstance(raw_category, list):
        raw_category = ", ".join(raw_category)

    # EXCLUDE unwanted categories entirely
    if any(excl in raw_category for excl in ["F/F", "M/F", "Gen", "Multi"]):
        return None, None # Drops the fic

    # Classify as M/M or "None of the above" (Other)
    if "M/M" in raw_category:
        category_key = "MM"
    else:
        category_key = "Other"

    return fandom_key, category_key

def passes_base_filters(doc: dict) -> bool:
    metadata = doc.get('metadata', {})
    words = parse_word_count(metadata.get('words', '0'))

    if words <= 1500:
        return False
    if metadata.get('Language', '') != 'English':
        return False

    return True

def format_fanfic(doc: dict) -> str:
    metadata = doc.get('metadata', {})
    text = doc.get('text', '')
    title = doc.get('title', 'Untitled')

    header_parts = [f"Title: {title}"]
    for key in ['author', 'Fandom', 'Rating', 'Category', 'Relationship', 'Characters', 'Additional Tags', 'words']:
        if metadata.get(key):
            header_parts.append(f"{key.capitalize()}: {metadata[key]}")

    return f"{chr(10).join(header_parts)}\n\n{text}\n"

# ==========================================
# 2. WORKER PROCESS (Multi-Core)
# ==========================================

def process_single_file(filepath: str, tokenizer_path: str, output_dir: str, chunk_size: int = 1_000_000):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    worker_id = os.path.basename(filepath).split('.')[0]

    # buffers dict to hold token arrays for each fandom_category route
    buffers = {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    doc = json.loads(line)

                    if not passes_base_filters(doc):
                        continue

                    fandom_key, category_key = get_routing_keys(doc)
                    if not fandom_key or not category_key:
                        continue # Skip if no fandom match or if it hit the Category exclusion filter

                    route = f"{fandom_key}_{category_key}"
                    if route not in buffers:
                        buffers[route] = []

                    formatted_text = format_fanfic(doc)
                    if not formatted_text: continue

                    tokens = tokenizer(formatted_text, add_special_tokens=False)['input_ids'] + [eos_id]
                    buffers[route].extend(tokens)

                    # Flush to disk for this specific route if it gets too large
                    if len(buffers[route]) >= chunk_size:
                        out_file = os.path.join(output_dir, f"temp_{route}_{worker_id}.bin")
                        with open(out_file, 'ab') as bf:
                            bf.write(np.array(buffers[route], dtype=np.uint16).tobytes())
                        buffers[route] = []

                except json.JSONDecodeError:
                    continue

        # Final flush for any remaining tokens across all active routes in this file
        for route, buf in buffers.items():
            if buf:
                out_file = os.path.join(output_dir, f"temp_{route}_{worker_id}.bin")
                with open(out_file, 'ab') as bf:
                    bf.write(np.array(buf, dtype=np.uint16).tobytes())

        return True, filepath
    except Exception as e:
        return False, f"Error in {filepath}: {str(e)}"

# ==========================================
# 3. COMBINER PROCESS (REDUCE)
# ==========================================

def combine_and_cleanup(output_dir: str):
    print(f"\nMerging chunks into final categorized datasets...")

    # Find all temp files and group them by their route (e.g., genshin_MM)
    temp_files = [f for f in os.listdir(output_dir) if f.startswith('temp_') and f.endswith('.bin')]
    routes = {}

    for f in temp_files:
        # Format is temp_{route}_{worker_id}.bin -> extract {route}
        route = f.replace('temp_', '').rsplit('_', 1)[0]
        if route not in routes:
            routes[route] = []
        routes[route].append(os.path.join(output_dir, f))

    for route, files in routes.items():
        final_path = os.path.join(output_dir, f"{route}_pretrain.bin")
        total_tokens = 0

        with open(final_path, 'wb') as f_out:
            for tmp_file in files:
                data = np.fromfile(tmp_file, dtype=np.uint16)
                total_tokens += len(data)
                f_out.write(data.tobytes())
                os.remove(tmp_file) # Clean up temp chunks

        print(f"✅ Created {final_path} ({total_tokens:,} tokens)")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    raw_dir = "InputDatasets/AO3"
    out_dir = "data"
    tokenizer_dir = "custom_tokenizer"

    os.makedirs(out_dir, exist_ok=True)

    jsonl_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(('.jsonl', '.json'))]

    if not jsonl_files:
        print(f"No JSONL files found in '{raw_dir}' directory!")
        exit()

    print(f"\n--- Processing {len(jsonl_files)} Local AO3 Files ---")

    max_workers = min(multiprocessing.cpu_count(), len(jsonl_files))
    print(f"Spawning {max_workers} worker threads...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, fp, tokenizer_dir, out_dir): fp for fp in jsonl_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing & Tokenizing"):
            success, msg = future.result()
            if not success:
                print(f"\n{msg}")

    # Merge all the distributed worker chunks into the clean, separated .bin files
    combine_and_cleanup(out_dir)
