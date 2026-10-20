#!/usr/bin/env python3
"""mimo_transplant_selftest — SELF-TRANSPLANT feasibility bed (in-family arms).

Question (pre-declared): can the graft machinery move WEIGHTS at all, in the
friendliest geometry — transplanting the FULL 256-expert bank of one MiMo
layer into another layer's slot on the SAME host (same family, same shapes,
destination router rows untouched, expert indices tied e->e)?

Arms (all on the faithful full host, lab/mimo_full_host.py, anchor f69737a):
  control     untouched host (must reproduce the committed coherent reference)
  L12toL44    layer 44's routed experts all load layer 12's weights
  L4toL32     layer 32's routed experts all load layer 4's weights
  L12toL44+L4toL32  both transplants live at once (mechanism additivity)

Probes (greedy, same prompt set as mimo_bisect_smoke, 2 prompts x 40 tok):
  (a) coherence class per generated stream (coherent / loop / mosaic / blank /
      nan) — auto-class from token-id statistics, no corpus text;
  (b) layer-norm-parity chain vs the untransplanted control at
      L10/L16/L40/L41/L47 (mlp + hidden norms at steps 0/10/30) plus the
      committed reference fact L10mlp(tok=0)=74.13 ("is the L10 spike
      intact?") — the transplant layers sit strictly between pairs of the
      chain so the chain localizes where the body reacts;
  (c) the NaN difference: per-arm count of non-finite logits/steps, and
      non-finite captured norms, arm minus control.

The transplant itself is a pure rebinding: destination LazyExpertMLP modules
get their (layer, root) rewritten to the source layer, and the expert LRU is
flushed so no stale destination weights survive.  Tie verification per arm:
for one expert index, the destination call and the source call must return
bit-identical outputs on the same input after rebinding (max|delta| recorded;
expected exactly 0.0 — both resolve the same packed store keys).

Verdict rule (declared before any run):
  mechanism-alive  = controls coherent AND every arm's coherence class is not
                     worse than "loop" AND the parity chain breaks ONLY at/after
                     the transplant slot (upstream chain intact);
  body-rejects     = geometry breaks even in-family (NaN explosion, mosaic
                     soup, or upstream-chain corruption from a slot-local
                     weight move) = transplant of any kind pays the baked-in
                     price regardless of family.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import torch

os.environ.setdefault("ROCBLAS_TENSILE_LIBPATH",
                      "/opt/rocm-6.2.4/lib/rocblas/library")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import mimo_full_host as fh  # noqa: E402

FORMAT = "vesper-mimo-transplant-selftest-v1"
REF_L10_MLP_TOK0 = 74.13  # committed host acceptance fact (MIMO_HOST_FAITHFUL)

PROMPTS = [
    "What do you think of trains?",
    "Name one prime number greater than 1000 and explain briefly why it is prime.",
]
PROBE_LAYERS = (10, 16, 40, 41, 47)
CAPTURE_STEPS = (0, 10, 30)

ARMS = {
    "control": [],
    "L12toL44": [(12, 44)],
    "L4toL32": [(4, 32)],
    "both": [(12, 44), (4, 32)],
}


def classify(ids, n_steps) -> str:
    """Auto coherence class from token-id statistics only (no corpus text)."""
    if n_steps["nonfinite"] > 0 and not ids:
        return "nan"
    if not ids:
        return "blank"
    n = len(ids)
    counts = Counter(ids)
    top2 = sum(c for _, c in counts.most_common(2)) / n
    if top2 > 0.55 and n >= 8:
        # heavy few-token dominance
        return "loop"
    # repeated 6-gram loop detector
    if n >= 24:
        grams = Counter(tuple(ids[i:i + 6]) for i in range(n - 5))
        g, c = grams.most_common(1)[0]
        if c >= 3:
            return "loop"
    return "coherent"


def mosaic_frac(tok, ids) -> float:
    """Fraction of decoded characters outside the ASCII band — the failure
    signature of the soup/embedding-salad class (history: it prints)."""
    chars = 0
    odd = 0
    for i in ids:
        s = tok.decode([i], skip_special_tokens=True)
        for ch in s:
            chars += 1
            if ord(ch) > 126:
                odd += 1
    return (odd / chars) if chars else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=fh.MODEL_DIR)
    ap.add_argument("--cache-gb", type=float, default=18.0)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--arms", default="control,L12toL44,L4toL32,both")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="mimo_transplant_selftest.json")
    args = ap.parse_args()

    t0 = time.time()
    out = {"format": FORMAT, "max_new": args.max_new,
           "probe_layers": list(PROBE_LAYERS),
           "ref_l10_mlp_tok0": REF_L10_MLP_TOK0}
    model, tok, cache = fh.build_full_host(args.model_dir, args.device,
                                           args.cache_gb, out)
    print("LOAD", out["full_host"], flush=True)

    # --- norm capture hooks -------------------------------------------------
    norms: dict = {"mlp": {}, "hid": {}}
    step = {"i": 0}

    def hook_for(kind, li):
        def hook(_m, _i, o):
            if step["i"] in CAPTURE_STEPS:
                t = o[0] if isinstance(o, tuple) else o
                v = float(t.float().pow(2).sum().sqrt())
                norms[kind].setdefault(li, {})[step["i"]] = v
        return hook

    for li in PROBE_LAYERS:
        layer = model.model.layers[li]
        layer.mlp.register_forward_hook(hook_for("mlp", li))
        layer.register_forward_hook(hook_for("hid", li))

    # --- transplant plumbing ------------------------------------------------
    def flush_cache():
        cache.entries.clear()
        cache.order.clear()
        cache.bytes = 0

    def apply_pairs(pairs):
        for src, dst in pairs:
            exg = model.model.layers[dst].mlp.experts
            for ex in exg:
                ex.layer = src
                ex.root = f"model.layers.{src}.mlp.experts.{ex.expert}."
        flush_cache()

    def restore_pairs(pairs):
        for _src, dst in pairs:
            exg = model.model.layers[dst].mlp.experts
            for ex in exg:
                ex.layer = dst
                ex.root = f"model.layers.{dst}.mlp.experts.{ex.expert}."
        flush_cache()

    def tie_check(src, dst, ex_id):
        """Bit-equality of dst call vs src call after rebinding (same store
        keys must resolve).  max|delta| recorded; 0.0 expected."""
        x = torch.linspace(-1, 1, 4096, dtype=torch.float16,
                           device=args.device).reshape(1, 1, 4096)
        with torch.no_grad():
            y_dst = model.model.layers[dst].mlp.experts[ex_id](x)
            y_src = model.model.layers[src].mlp.experts[ex_id](x)
        d = float((y_dst.float() - y_src.float()).abs().max())
        return {"src": src, "dst": dst, "expert": ex_id, "max_abs_diff": d}

    # --- one arm ------------------------------------------------------------
    def run_arm(name, pairs):
        arm = {"pairs": [{"src": s, "dst": d} for s, d in pairs],
               "prompts": []}
        if pairs:
            apply_pairs(pairs)
            arm["tie_check"] = tie_check(pairs[0][0], pairs[0][1], 0)
        try:
            for prompt_text in PROMPTS:
                ids = tok.encode(
                    tok.apply_chat_template(
                        [{"role": "user", "content": prompt_text}],
                        tokenize=False, add_generation_prompt=True),
                    add_special_tokens=False)
                norms["mlp"].clear()
                norms["hid"].clear()
                past = None
                outs = []
                nfin = {"nonfinite": 0, "steps": 0}
                t1 = time.time()
                for i in range(args.max_new):
                    step["i"] = i
                    x = torch.tensor([ids] if i == 0 else [[outs[-1]]],
                                     dtype=torch.long, device=args.device)
                    with torch.no_grad():
                        o = model(input_ids=x, past_key_values=past,
                                  use_cache=True)
                    past = o.past_key_values
                    lg = o.logits[0, -1].float()
                    bad = int((~torch.isfinite(lg)).sum())
                    if bad:
                        nfin["nonfinite"] += bad
                        nfin["steps"] += 1
                    outs.append(int(lg.cpu().argmax()))
                rec = {
                    "gen_wall_s": round(time.time() - t1, 1),
                    "n_out": len(outs),
                    "nan_logits_total": nfin["nonfinite"],
                    "nan_steps": nfin["steps"],
                    "mosaic_frac": round(mosaic_frac(tok, outs), 4),
                    "class": classify(outs, nfin),
                    "norms": {k: {str(li): {str(s): round(v[s], 4)
                                            for s in sorted(v)}
                                  for li, v in norms[k].items()}
                              for k in ("mlp", "hid")},
                }
                import math
                rec["nan_norms"] = sum(
                    1 for k in ("mlp", "hid") for v in norms[k].values()
                    for x in v.values() if not math.isfinite(x))
                arm["prompts"].append(rec)
                print(f"ARM {name} p{len(arm['prompts'])-1} "
                      f"class={rec['class']} mosaic={rec['mosaic_frac']} "
                      f"nan={rec['nan_logits_total']} "
                      f"wall={rec['gen_wall_s']}s", flush=True)
        finally:
            if pairs:
                restore_pairs(pairs)
        return arm

    arms_out = {}
    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        arms_out[name] = run_arm(name, ARMS[name])

    # --- cross-arm summary: parity chain vs control -------------------------
    def chain(armrec, pi):
        c = arms_out["control"]["prompts"][pi]
        a = armrec["prompts"][pi]
        rows = {}
        for k in ("mlp", "hid"):
            for li in PROBE_LAYERS:
                for s in CAPTURE_STEPS:
                    ref = c["norms"][k].get(str(li), {}).get(str(s))
                    val = a["norms"][k].get(str(li), {}).get(str(s))
                    if ref and val and ref > 0:
                        rows[f"{k}{li}@{s}"] = round(val / ref, 4)
        return rows

    summary = {}
    for name, armrec in arms_out.items():
        if name == "control":
            continue
        summary[name] = {
            "class": [p["class"] for p in armrec["prompts"]],
            "mosaic_frac": [p["mosaic_frac"] for p in armrec["prompts"]],
            "nan_steps": [p["nan_steps"] for p in armrec["prompts"]],
            "arm_chain_ratio_vs_control": [chain(armrec, pi) for pi in
                                           range(len(PROMPTS))],
        }
    out["arms"] = arms_out
    out["summary"] = summary
    out["cache"] = {"hits": cache.hits, "misses": cache.misses,
                    "gb": round(cache.bytes / 2**30, 2)}
    out["wall_s"] = round(time.time() - t0, 1)
    out["control_classes"] = [p["class"]
                              for p in arms_out["control"]["prompts"]]
    out["control_l10_mlp_tok0"] = [
        p["norms"]["mlp"].get("10", {}).get("0")
        for p in arms_out["control"]["prompts"]]
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("SUMMARY", json.dumps(control_ref(out)), flush=True)
    print(f"WROTE {args.out}", flush=True)
    print("TRANSPLANT_DONE", flush=True)
    return 0


def control_ref(out):
    return {
        "control_class": out["control_classes"],
        "control_l10_mlp_tok0": out["control_l10_mlp_tok0"],
        "arm_classes": {k: v["class"] for k, v in out["summary"].items()},
        "arm_nan_steps": {k: v["nan_steps"] for k, v in out["summary"].items()},
        "wall_s": out["wall_s"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
