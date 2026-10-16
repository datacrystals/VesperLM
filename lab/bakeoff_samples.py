#!/usr/bin/env python3
"""Extract side-by-side sample exchanges from the two bake-off JSONs.

Prints N prompts with both spines' verbatim replies (the user picks by feel).
"""
import json
import sys

PICKS = [
    ("trains", "What do you think of trains?"),
    ("gundam", "What's your favorite anime?"),
    ("math", "What's 17 times 23?"),
]


def load(p):
    return json.load(open(p))


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    a, b = load(a_path), load(b_path)
    la = a.get("label", a.get("spine"))
    lb = b.get("label", b.get("spine"))
    print(f"=== SAMPLE EXCHANGES: {la} vs {lb} ===")
    found = 0
    for dlg_id, prompt in PICKS:
        if found >= n:
            break
        ra = rb = None
        for d in a.get("dialogues", []):
            if d["id"] != dlg_id:
                continue
            for t in d["turns"]:
                if t["user"].strip() == prompt:
                    ra = t
        for d in b.get("dialogues", []):
            if d["id"] != dlg_id:
                continue
            for t in d["turns"]:
                if t["user"].strip() == prompt:
                    rb = t
        if ra is None and rb is None:
            continue
        found += 1
        print(f"\n--- [{dlg_id}] you> {prompt}")
        print(f"  {la}> " + (ra or {}).get("assistant", "(not run)").replace(
            "\n", " ")[:400])
        print(f"  {lb}> " + (rb or {}).get("assistant", "(not run)").replace(
            "\n", " ")[:400])
        if ra:
            print(f"  ({la} scores: {ra.get('scores')})")
        if rb:
            print(f"  ({lb} scores: {rb.get('scores')})")
    print(f"\n(exchanges shown: {found})")


if __name__ == "__main__":
    main()
