"""Probe-set definitions and runner for the VesperLM immune system.

A probe is a dict:
  name:     unique id
  prompt:   full ChatML (or plain) prompt string
  scorer:   {type: exact-match|regex|json-valid|tool-call-schema|
                   semantic-similarity, ... params}
  category: free-form label (arithmetic-tool-use, file-ops-tool-use, ...)
  held_out: true for never-train-on canaries (implies protected)
  protected: optional extra flag; protected probes are absolute vetoes

All scores are floats in [0, 1]. Scorers are deterministic and offline
(no external models): semantic-similarity is token-overlap cosine.
"""
import json
import re
import sys
import os

try:
    import yaml
except ImportError:
    yaml = None

TOOL_OPEN = "<|tool_call|>"
TOOL_CLOSE = "</|tool_call|>"
TOOL_CALL_RE = re.compile(
    r"<\|tool_call\|>\s*(\{.*?\})\s*</\|tool_call\|>", re.DOTALL)

SCORER_TYPES = ("exact-match", "regex", "json-valid",
                "tool-call-schema", "semantic-similarity")


def _norm(s):
    return (s or "").strip()


def _tokens(s):
    return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if t]


def score_exact_match(output, spec):
    expected = spec.get("expected", "")
    candidates = expected if isinstance(expected, list) else [expected]
    if not spec.get("case_sensitive", False):
        outs = [_norm(output).lower()]
        cands = [str(c).strip().lower() for c in candidates]
    else:
        outs = [_norm(output)]
        cands = [str(c).strip() for c in candidates]
    if spec.get("match") == "prefix":
        hit = any(o.startswith(c) for o in outs for c in cands)
    elif spec.get("match") == "contains":
        hit = any(c in o for o in outs for c in cands)
    else:
        hit = any(o == c for o in outs for c in cands)
    return (1.0 if hit else 0.0), {"expected": candidates, "hit": hit}


def score_regex(output, spec):
    flags = re.IGNORECASE if not spec.get("case_sensitive", False) else 0
    pat = re.compile(spec.get("pattern", ""), flags | re.DOTALL)
    m = pat.search(output or "")
    if spec.get("full_match"):
        m = re.fullmatch(pat, output or "")
    return (1.0 if m else 0.0), {"pattern": spec.get("pattern"),
                                 "match": bool(m)}


def _extract_json_blob(text):
    """Return the first balanced JSON object found in text, else None."""
    start = (text or "").find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start:i + 1]
                    try:
                        return json.loads(blob), blob
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None, None


def score_json_valid(output, spec):
    text = _norm(output)
    if spec.get("from_tool_call", True):
        m = TOOL_CALL_RE.search(text)
        if m:
            text = m.group(1)
    try:
        obj = json.loads(text)
        if spec.get("require_object") and not isinstance(obj, dict):
            return 0.0, {"valid_json": True, "is_object": False}
        return 1.0, {"valid_json": True, "is_object": isinstance(obj, dict)}
    except json.JSONDecodeError:
        obj, blob = _extract_json_blob(text)
        if obj is not None:
            return 1.0, {"valid_json": True, "extracted": True}
        return 0.0, {"valid_json": False}


def score_tool_call_schema(output, spec):
    """Partial credit: 0.4 for a parseable tool_call JSON, 0.3 for the
    expected tool name, 0.3 for the args schema (required keys / types and
    optional regex on string args)."""
    text = output or ""
    m = TOOL_CALL_RE.search(text)
    detail = {"found_tool_call": bool(m)}
    if not m:
        # tolerate bare JSON object with tool/args keys
        obj, _ = _extract_json_blob(text)
        if not (isinstance(obj, dict) and "tool" in obj):
            return 0.0, detail
    else:
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            detail["parse_error"] = True
            return 0.0, detail
    score = 0.4
    detail["parsed"] = True
    want_tool = spec.get("tool")
    if want_tool:
        if obj.get("tool") == want_tool:
            score += 0.3
            detail["tool_ok"] = True
        else:
            detail["tool_ok"] = False
            detail["tool_got"] = obj.get("tool")
            return round(score, 3), detail
    else:
        score += 0.3
    args = obj.get("args", {})
    schema = spec.get("args_schema") or {}
    ok = isinstance(args, dict)
    for key, typ in schema.items():
        if key not in args:
            ok = False
            break
        if typ == "string" and not isinstance(args[key], str):
            ok = False
            break
        if typ == "number" and not isinstance(args[key], (int, float)):
            ok = False
            break
    if ok:
        for key, pat in (spec.get("args_match") or {}).items():
            if key not in args or not re.search(pat, str(args[key]),
                                               re.IGNORECASE):
                ok = False
                break
    if ok:
        score += 0.3
        detail["args_ok"] = True
    else:
        detail["args_ok"] = False
    return round(score, 3), detail


def _bow_cos(tokens_a, tokens_b):
    if not tokens_a or not tokens_b:
        return 0.0
    from collections import Counter
    ca, cb = Counter(tokens_a), Counter(tokens_b)
    keys = set(ca) | set(cb)
    dot = sum(ca[k] * cb[k] for k in keys)
    na = sum(v * v for v in ca.values()) ** 0.5
    nb = sum(v * v for v in cb.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def score_semantic_similarity(output, spec):
    """Tokenizer-free semantic similarity: cosine over bag-of-words of
    lowercase alphanumeric tokens against one or more reference strings.
    Returns the raw similarity in [0,1]; callers compare to `threshold`
    only for display -- the score itself is continuous partial credit."""
    refs = spec.get("reference", [])
    if isinstance(refs, str):
        refs = [refs]
    out_toks = _tokens(output)
    best = max((_bow_cos(out_toks, _tokens(r)) for r in refs), default=0.0)
    return float(best), {"similarity": round(float(best), 4),
                         "threshold": spec.get("threshold", 0.3),
                         "n_refs": len(refs)}


SCORERS = {
    "exact-match": score_exact_match,
    "regex": score_regex,
    "json-valid": score_json_valid,
    "tool-call-schema": score_tool_call_schema,
    "semantic-similarity": score_semantic_similarity,
}


def score_output(output, scorer_spec):
    stype = (scorer_spec or {}).get("type")
    if stype not in SCORERS:
        raise ValueError(f"unknown scorer type: {stype!r} "
                         f"(expected one of {sorted(SCORERS)})")
    return SCORERS[stype](output, scorer_spec)


def load_probe_set(path):
    """Load probes from YAML or JSON. Validates schema; every probe gets
    normalized fields: name, prompt, scorer, category, held_out, protected."""
    with open(path) as f:
        raw = f.read()
    if path.endswith((".yaml", ".yml")):
        if yaml is None:
            sys.exit("pyyaml not available; use a .json probe set")
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("probes", data)
    probes = []
    seen = set()
    for i, p in enumerate(data):
        name = p.get("name")
        if not name or name in seen:
            raise ValueError(f"probe #{i}: missing or duplicate name {name!r}")
        seen.add(name)
        if "prompt" not in p or not isinstance(p["prompt"], str):
            raise ValueError(f"probe {name}: needs string 'prompt'")
        scorer = p.get("scorer") or {}
        if scorer.get("type") not in SCORERS:
            raise ValueError(f"probe {name}: bad scorer type {scorer.get('type')!r}")
        held_out = bool(p.get("held_out", False))
        protected = bool(p.get("protected", False)) or held_out
        probes.append({
            "name": name,
            "prompt": p["prompt"],
            "scorer": scorer,
            "category": p.get("category", "uncategorized"),
            "held_out": held_out,
            "protected": protected,
        })
    return probes
def run_probes(model, tok, probes, max_new=48, stop_tokens=None):
    """Run every probe prompt and score its completion."""
    from cpu_backend import generate
    results = []
    for probe in probes:
        output = generate(model, tok, probe["prompt"],
                          max_new=max_new, stop_tokens=stop_tokens)
        score, detail = score_output(output, probe["scorer"])
        results.append({"name": probe["name"], "category": probe["category"],
                        "held_out": probe["held_out"],
                        "protected": probe["protected"],
                        "output": output, "score": float(score),
                        "detail": detail})
    return results


def aggregate(results):
    """Mean score overall, plus per-category and protected subsets."""
    def mean(rs):
        return sum(r["score"] for r in rs) / len(rs) if rs else 0.0
    cats = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    prot = [r for r in results if r["protected"]]
    held = [r for r in results if r["held_out"]]
    return {"aggregate": mean(results), "n_probes": len(results),
            "per_category": {c: mean(rs) for c, rs in cats.items()},
            "protected_aggregate": mean(prot), "n_protected": len(prot),
            "held_out_aggregate": mean(held), "n_held_out": len(held)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run a probe set once (smoke).")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--lora", default=None)
    args = ap.parse_args()
    import cpu_backend as cb
    tok = cb.load_tokenizer()
    model, ck = cb.load_model(args.ckpt, lora_path=args.lora)
    probes = load_probe_set(args.probes)
    im_end = tok.convert_tokens_to_ids("im_end")
    res = run_probes(model, tok, probes, max_new=args.max_new,
                     stop_tokens={im_end})
    for r in res:
        flag = "H" if r["held_out"] else " "
        print(f"[{flag}] {r['name']:28s} {r['score']:.3f}  {r['output'][:60]!r}")
    print("aggregate:", aggregate(res))
