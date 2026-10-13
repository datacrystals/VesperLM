#!/usr/bin/env python3
"""mem_pack — the Vesper persona memory pack loader (lab/memories/persona_v1.json).

Permanent "about the user" facts taught as shaped episodic plugs — exactly the
pet_kepler machinery (lab/memory_shaping.py substitution deltas + door25 wrap +
owner_mass 0.85 recal + guard-checkpointed sibling recal) at slightly larger
scale.  The pack JSON is the committable source of truth; the built plugs go
to the usual untracked artifacts (plugins dir + joint_state_persona.pt).

This module turns the JSON into everything the pipeline needs:

* episodes()          -> {home: memory_shaping.Episode} (substitution shape,
                         asserted per delta; the G3 retention lever)
* plugs_for(spine)    -> [(tag, home, host_layer, donor_expert)] for the
                         persona rows on the phi (v0.5) / qwen (v0.6) hosts
* gate_registry()     -> (MEM_TAGS, EPISODE_QUERIES, TAUGHT_STRS) covering the
                         legacy trio + the persona rows, wired into
                         varact_gate by apply_to_varact() so the shipped
                         question-match gate (30038bd --qscope overlap) and
                         the taught-token bonus see the new rows
* PROBES / CANARY     -> the verification battery (persona_probe.py)
* selftest()          -> gate-separation check on the battery (CPU, no model):
                         every target question scores >= qtau on exactly its
                         own row; every canary scores < qtau on every row

Query-side discipline (see the pack JSON notes): question-match overlap is
|q_tokens ∩ row_tokens| / |q_tokens|, so on a 2-token question ONE shared
token is exactly 0.5 = fires at the shipped qtau.  The query sets are kept
pairwise-disjoint on distinctive tokens and free of the generic verbs
(favorite/like/love/hate/think/feel/opinion/name/tell) that would make
"favorite anime?" or "Do you like cars?" co-fire the wrong row.
"""

from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.join(REPO, "lab")
sys.path.insert(0, LAB)
sys.path.insert(0, os.path.join(REPO, "Hippocampus"))
sys.path.insert(0, os.path.join(REPO, "Common"))

PACK_PATH = os.path.join(LAB, "memories", "persona_v1.json")

# legacy mem rows that ship with v0.5/v0.6 (kept green; pack is additive)
LEGACY_TAGS = ("mem_pet", "mem_edit", "mem_words")
LEGACY_TAUGHT = {
    "mem_pet": ("Kepler", " Kepler"),
    "mem_edit": ("helix", " helix", "Helix"),
    "mem_words": ("Four", " Four"),
}

QTAU = 0.5        # shipped question-match threshold (30038bd default)


def load_pack(path: str | None = None) -> dict:
    with open(path or PACK_PATH) as f:
        return json.load(f)


def _episodes_ms():
    import memory_shaping as ms
    return ms


def episodes(pack: dict | None = None) -> dict:
    """{home: Episode} from the pack; every delta asserted substitution."""
    ms = _episodes_ms()
    pack = pack or load_pack()
    out = {}
    for fact in pack["facts"]:
        deltas, approvals = [], []
        for row in fact["teach"]:
            d = ms.contract_turn(row["prompt"], row["stated"], row["correction"])
            if d is None or not d.is_substitution:
                raise ValueError(
                    f"{fact['tag']}: non-substitution delta on {row['prompt']!r}"
                    f" -> {d and d.kind}")
            deltas.append(d)
        for row in fact["approvals"]:
            approvals.append((row["prompt"], row["response"]))
        out[fact["home"]] = ms.Episode(fact["tag"], deltas, approvals)
    return out


def tags(pack: dict | None = None) -> list:
    pack = pack or load_pack()
    return [f["tag"] for f in pack["facts"]]


def homes(pack: dict | None = None) -> list:
    pack = pack or load_pack()
    return [f["home"] for f in pack["facts"]]


def plugs_for(spine: str, pack: dict | None = None) -> list:
    """[(tag, home, host_layer, donor_expert)] for one host spine."""
    pack = pack or load_pack()
    out = []
    for f in pack["facts"]:
        h = f["host"][spine]
        out.append((f["tag"], f["home"], int(h["layer"]), int(h["expert"])))
    return out


def queries(pack: dict | None = None) -> dict:
    """{tag: [query texts]} for ALL mem rows (legacy trio + persona)."""
    pack = pack or load_pack()
    out = dict(pack.get("legacy_queries") or {})
    for f in pack["facts"]:
        out[f["tag"]] = list(f["queries"])
    return out


def taught(pack: dict | None = None) -> dict:
    """{tag: taught strings} for ALL mem rows (decode-bonus token side)."""
    pack = pack or load_pack()
    out = dict(LEGACY_TAUGHT)
    for f in pack["facts"]:
        out[f["tag"]] = list(f["taught"])
    return out


def mem_tags(pack: dict | None = None) -> tuple:
    return tuple(LEGACY_TAGS) + tuple(tags(pack))


def gate_registry(pack: dict | None = None) -> tuple:
    """(MEM_TAGS, EPISODE_QUERIES, TAUGHT_STRS) for varact_gate."""
    return mem_tags(pack), queries(pack), taught(pack)


def apply_to_varact(va, pack: dict | None = None) -> tuple:
    """Point varact_gate at the pack's mem rows (module globals, call-time
    lookups — safe to rebind).  Returns the registry that was installed."""
    pack = pack or load_pack()
    mt, eq, ts = gate_registry(pack)
    va.MEM_TAGS = mt
    va.EPISODE_QUERIES = eq
    va.TAUGHT_STRS = ts
    # home lookup for the sweep / sibling-recal harnesses
    va.HOME_OF = dict(va.HOME_OF)
    for f in pack["facts"]:
        va.HOME_OF[f["tag"]] = f["home"]
    if list(va.ORDER) != list(dict.fromkeys(list(va.ORDER) + tags(pack))):
        va.ORDER = list(dict.fromkeys(list(va.ORDER) + tags(pack)))
    try:
        import v05_build as b5
        for h in homes(pack):
            if h not in b5.HOME_LABEL:
                b5.HOME_LABEL[h] = max(b5.HOME_LABEL.values()) + 1
    except Exception:  # noqa: BLE001
        pass
    return mt, eq, ts


def home_labels(extra_base: dict | None = None) -> dict:
    """HOME_LABEL extension for the persona homes (v05/v06 label spaces)."""
    base = dict(extra_base or {})
    n = (max(base.values()) + 1) if base else 1
    for i, h in enumerate(homes()):
        base[h] = n + i
    return base


def probes(pack: dict | None = None) -> list:
    """[(tag, question, expect, reject)] — the single-turn 6-fact bar."""
    pack = pack or load_pack()
    return [(f["tag"], f["probe"]["question"], f["probe"]["expect"],
             f["probe"]["reject"]) for f in pack["facts"]]


def verdict(flat: str, expect, reject):
    """Word-boundary hit/miss on the lowercased answer (so 'dislike' does
    not satisfy a 'like' reject, 'know' does not satisfy 'no')."""
    import re
    low = " ".join(str(flat).split()).lower()

    def has(term):
        return re.search(r"\b" + re.escape(term.lower()) + r"\b",
                         low) is not None

    hit = any(has(e) for e in expect)
    bad = any(has(r) for r in reject)
    key = next((e for e in expect if has(e)), "-")
    return ("PASS" if (hit and not bad) else "FAIL"), hit, bad, key


# The canary battery: (id, question, want) where want is the tag that MUST
# fire (>= qtau) or None meaning "no mem row may fire" (the no-intrusion bar).
CANARY = [
    ("fav_anime", "favorite anime?", "mem_gundam"),
    ("about_cars", "What about cars?", "mem_cars"),
    ("like_trains", "Do you like trains?", "mem_trains"),
    ("fav_show", "favorite show?", "mem_poi"),
    ("fav_color", "favorite color?", None),
    ("fav_color2", "What's your favorite color?", None),
    ("hamlet", "Who wrote Hamlet?", None),
    ("sky", "What color is the sky?", None),
    ("hello", "Say hello.", None),
    ("cat_name", "What is my cat's name?", "mem_pet"),
    ("editor", "What is my favorite text editor?", "mem_edit"),
    ("two_plus_two", "What is 2 + 2?", "mem_words"),
    ("hacking", "Walk me through the reconnaissance phase of a penetration "
                "test. What kind of information would you gather?", None),
    ("writing", "Write two short sentences of atmospheric prose about a "
                "lighthouse in a storm.", None),
]


def episode_scores(question: str, qmap: dict | None = None) -> dict:
    """Borrow the shipped gate scorer (identical tokenization)."""
    import varact_gate as va
    return va.episode_scores(question, qmap or queries())


def selftest(pack_path: str | None = None, qtau: float = QTAU, verbose=True):
    """Gate-separation check over the probe + canary battery (CPU)."""
    pack = load_pack(pack_path)
    qmap = queries(pack)
    mt = mem_tags(pack)
    fails = []
    rows = []

    def fired(scores, want):
        return tuple(sorted(t for t, v in scores.items() if v >= qtau))

    # (a) every fact's probe fires its own row and no sibling mem row
    for tag, q, _exp, _rej in probes(pack):
        sc = episode_scores(q, qmap)
        hit = fired(sc, tag)
        ok = (hit == (tag,))
        rows.append(("probe", tag, round(sc.get(tag, 0.0), 3), ",".join(hit),
                     "PASS" if ok else "FAIL"))
        if not ok:
            fails.append(f"probe {tag}: fired={hit} score={sc.get(tag)}")

    # (b) canary battery: want row fires, everything else silent
    for cid, q, want in CANARY:
        sc = episode_scores(q, qmap)
        hit = fired(sc, want)
        if want is None:
            ok = (hit == ())
        else:
            ok = (hit == (want,))
        rows.append(("canary", cid, round(sc.get(want or "", 0.0), 3),
                     ",".join(hit) or "-", "PASS" if ok else "FAIL"))
        if not ok:
            fails.append(f"canary {cid}: want={want} fired={hit} score={sc}")

    # (c) legacy trio probes still fire their row (pack is additive)
    legacy_probes = [
        ("mem_pet", "What is my cat's name?"),
        ("mem_edit", "What is my favorite text editor?"),
        ("mem_words", "What is 2 + 2?"),
    ]
    for tag, q in legacy_probes:
        sc = episode_scores(q, qmap)
        hit = fired(sc, tag)
        ok = (tag in hit)
        rows.append(("legacy", tag, round(sc.get(tag, 0.0), 3), ",".join(hit),
                     "PASS" if ok else "FAIL"))
        if not ok:
            fails.append(f"legacy {tag}: fired={hit}")

    if verbose:
        print(f"{'kind':7s} {'id':16s} {'score':>6s}  {'fired':28s} verdict")
        for kind, rid, score, hit, verd in rows:
            print(f"{kind:7s} {rid:16s} {score:6.3f}  {hit:28s} {verd}")
        print(f"\nmem_tags={mt}")
        print(f"SELFTEST {'PASS' if not fails else 'FAIL'} "
              f"({len(rows)} rows, {len(fails)} failures)")
        for f in fails:
            print("  FAIL:", f)
    return (not fails), rows, fails


if __name__ == "__main__":
    ok, _rows, _f = selftest()
    raise SystemExit(0 if ok else 1)
