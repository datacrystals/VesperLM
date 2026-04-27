# 05_inspect_datasets.py
from datasets import get_dataset_config_names
from collections import OrderedDict

# ==========================================
# Same blend as your main script
# ==========================================
DATASET_PROPORTIONS = OrderedDict({
    "nvidia/Nemotron-CC-v2.1": 0.485,
    "nvidia/Nemotron-Pretraining-Code-v2": 0.148,
    "nvidia/Nemotron-CC-Code-v1": 0.021,
    "nvidia/Nemotron-CC-Math-v1": 0.084,
    "nvidia/Nemotron-Pretraining-Specialized-v1.1": 0.143,
    "wikipedia": 0.119,
})

# If you already know sub-blends for multi-config datasets, mirror them here.
# Otherwise leave empty and the script will list all available configs for you.
DATASET_CONFIGS = {
    "nvidia/Nemotron-CC-v2.1": [
        "High-Quality", "Medium-Quality", "Medium-High-Quality",
        "High-Quality-Synthetic", "Medium-High-Quality-Synthetic",
        "High-Quality-Translated-To-English",
        "Medium-High-Quality-Translated-To-English",
        "High-Quality-Translated-To-English-Synthetic",
        "High-Quality-DQA",
    ],
    # Add more as you discover them, e.g.:
    # "nvidia/Nemotron-Pretraining-Code-v2": [
    #     "Nemotron-Code-Metadata", "Synthetic-Question-Answering", ...
    # ],
}

def inspect_dataset(repo_id, weight):
    print(f"\n{'='*60}")
    print(f"Dataset : {repo_id}")
    print(f"Weight  : {weight:.1%}")
    print(f"{'='*60}")

    # Wikipedia is a special case
    if repo_id == "wikipedia":
        print("  -> Single config (hardcoded: '20220301.en')")
        print("     load_dataset('wikipedia', '20220301.en', split='train', streaming=True)")
        return

    try:
        configs = get_dataset_config_names(repo_id)
    except Exception as e:
        print(f"  -> ERROR fetching config names: {e}")
        return

    if not configs:
        print("  -> No configs reported. Likely loads without config name.")
        print(f"     load_dataset('{repo_id}', split='train', streaming=True)")
        return

    if len(configs) == 1:
        cfg = configs[0]
        print(f"  -> Single config: '{cfg}'")
        print(f"     load_dataset('{repo_id}', '{cfg}', split='train', streaming=True)")
        return

    print(f"  -> MULTI-CONFIG ({len(configs)} found):")
    for cfg in configs:
        marker = "  [*]" if cfg in DATASET_CONFIGS.get(repo_id, []) else "   - "
        print(f"      {marker} '{cfg}'")

    known = DATASET_CONFIGS.get(repo_id, [])
    if known:
        print(f"\n  -> You have mapped {len(known)} config(s) in DATASET_CONFIGS:")
        for cfg in known:
            print(f"      load_dataset('{repo_id}', '{cfg}', split='train', streaming=True)")
    else:
        print(f"\n  -> ACTION REQUIRED: Add desired configs from above to DATASET_CONFIGS")

if __name__ == "__main__":
    print("Inspecting all datasets in the Nemotron blend...\n")
    for repo_id, weight in DATASET_PROPORTIONS.items():
        inspect_dataset(repo_id, weight)
