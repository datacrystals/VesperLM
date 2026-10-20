#!/usr/bin/env python3
"""vk_transplant_selftest — SELF-TRANSPLANT cross-family arm on the vk spine.

Sibling of lab/mimo_transplant_selftest.py: the same operation (move a whole
routed-expert bank from one layer into another layer's slot, destination
router rows untouched, expert indices tied e->e) executed on a SECOND MoE
family — the vk passport-native spine (10 MoEFeedForward blocks, 8 experts,
top-2, /data/tliao/vk_final/chat_model).  This is the locally-testable
third-party family (the rung-1 bank already fitted MiMo-family cores onto
this host with the wrap recipe; the question here is the cheaper one: does
the vk body's OWN geometry accept an inter-layer weight move?).

Arms:
  control   untouched vk chat model
  vk1to8    FFN block 8's 8 expert ModulesList replaced by block 1's
  vk0to9    FFN block 9's bank replaced by block 0's
  vk-both   both transplants live at once

Probes:
  (a) coherence class of greedy streams (3 short chat prompts, 48 tok,
      same id-statistics classifier as the mimo arm);
  (b) parity chain: per-MoE-block output norm arm-vs-control ratios on one
      fixed authored probe forward (no corpus text anywhere — the probe
      paragraph lives in this file and is written by the harness author);
  (c) base-CE delta on the same authored probe (next-token mean NLL);
  (d) NaN difference: non-finite logits/norms, arm minus control.

Tie verification: after the swap, dst.ffn.experts[e] IS src.ffn.experts[e]
(object identity) and the dst slot's forward output matches the src slot's
expert output bit-exactly — 0.0 expected by construction, recorded anyway.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F

os.environ.setdefault("ROCBLAS_TENSILE_LIBPATH",
                      "/opt/rocm-6.2.4/lib/rocblas/library")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "Common"))

import vk_graft as vk  # noqa: E402

FORMAT = "vesper-vk-transplant-selftest-v1"

# authored probe text (harness author, 2026-10-20). Never dataset corpus.
PROBE_TEXT = (
    "The harbour was quiet before dawn. A ferry crew checked the manifest, "
    "counted the crates, and signed the log. Gulls circled over the water "
    "while the first bus of the morning rolled along the waterfront road. "
    "By sunrise the market stalls were open, shutters lifted, coffee poured, "
    "and the town settled into the ordinary rhythm of a working day."
)

CHAT_PROBES = [
    "Hello, how are you today?",
    "What is the capital of France?",
    "Count from one to five.",
]

ARMS = {
    "control": [],
    "vk1to8": [(1, 8)],
    "vk0to9": [(0, 9)],
    "vkboth": [(1, 8), (0, 9)],
}


def classify(ids, nan_steps) -> str:
    if nan_steps > 0 and not ids:
        return "nan"
    if not ids:
        return "blank"
    n = len(ids)
    counts = Counter(ids)
    top2 = sum(c for _, c in counts.most_common(2)) / n
    if top2 > 0.55 and n >= 8:
        return "loop"
    if n >= 24:
        grams = Counter(tuple(ids[i:i + 6]) for i in range(n - 5))
        if grams.most_common(1)[0][1] >= 3:
            return "loop"
    return "coherent"


def mosaic_frac(tok, ids) -> float:
    chars = odd = 0
    for i in ids:
        s = tok.decode([i], skip_special_tokens=True)
        for ch in s:
            chars += 1
            if ord(ch) > 126:
                odd += 1
    return (odd / chars) if chars else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/data/tliao/vk_final/chat_model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--out", default="vk_transplant_selftest.json")
    args = ap.parse_args()

    t0 = time.time()
    model, tok = vk.load_vk(args.model, args.device, torch.float16)
    blocks = vk.find_moe_blocks(model)
    n_blocks = len(blocks)
    print(f"[vk] {n_blocks} MoE blocks, 8 experts top-2", flush=True)

    # --- probes -------------------------------------------------------------
    probe_ids = tok.encode(PROBE_TEXT, add_special_tokens=False)
    xp = torch.tensor([probe_ids], dtype=torch.long, device=args.device)

    block_norms: dict[int, float] = {}

    def hooks_install():
        hs = []
        for bi, ffn in enumerate(blocks):
            def make(bi=bi):
                def hook(_m, _i, o):
                    y = o[0] if isinstance(o, tuple) else o
                    block_norms[bi] = float(y.float().pow(2).sum().sqrt())
                return hook
            hs.append(ffn.register_forward_hook(make()))
        return hs

    def ce_and_chain():
        block_norms.clear()
        hs = hooks_install()
        nan = {"logits": 0, "norms": 0}
        with torch.no_grad():
            out = model(xp)
        logits = out[0] if isinstance(out, tuple) else out.logits
        lg = logits.float()
        nan["logits"] = int((~torch.isfinite(lg)).sum())
        tgt = xp[:, 1:]
        ce = float(F.cross_entropy(
            lg[:, :-1].reshape(-1, lg.shape[-1]), tgt.reshape(-1),
            ignore_index=tok.pad_token_id))
        for h in hs:
            h.remove()
        nan["norms"] = sum(1 for v in block_norms.values()
                           if not math.isfinite(v))
        return ce, dict(block_norms), nan

    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    def greedy_gen(prompt):
        ids = vk.chatml_ids(tok, prompt)
        past = list(ids)
        outs = []
        nan_steps = 0
        for _ in range(args.max_new):
            x = torch.tensor([past + outs], dtype=torch.long,
                             device=args.device)
            with torch.no_grad():
                out = model(x)
            logits = out[0] if isinstance(out, tuple) else out.logits
            lg = logits[0, -1].float()
            if int((~torch.isfinite(lg)).sum()):
                nan_steps += 1
            nxt = int(lg.cpu().argmax())
            if nxt == im_end:
                break
            outs.append(nxt)
        return outs, nan_steps

    # --- transplant plumbing ------------------------------------------------
    def apply(pairs):
        saved = {}
        for src, dst in pairs:
            saved[dst] = blocks[dst].experts
            blocks[dst].experts = blocks[src].experts
        return saved

    def restore(pairs, saved):
        for _src, dst in pairs:
            blocks[dst].experts = saved[dst]

    def tie_check(src, dst, e):
        ident = blocks[dst].experts[e] is blocks[src].experts[e]
        return {"src": src, "dst": dst, "expert": e,
                "identical_object": bool(ident)}

    out = {"format": FORMAT, "n_moe_blocks": n_blocks,
           "probe_tokens": len(probe_ids), "max_new": args.max_new}
    arms_out = {}
    for name, pairs in ARMS.items():
        arm = {"pairs": pairs}
        saved = apply(pairs) if pairs else None
        if pairs:
            arm["tie_check"] = tie_check(pairs[0][0], pairs[0][1], 0)
        try:
            ce, chain, nan = ce_and_chain()
            arm["base_ce"] = round(ce, 6)
            arm["nan_probe"] = nan
            arm["block_norms"] = {str(k): round(v, 4)
                                  for k, v in chain.items()}
            arm["gen"] = []
            for q in CHAT_PROBES:
                ids, nan_steps = greedy_gen(q)
                arm["gen"].append({
                    "n_out": len(ids), "nan_steps": nan_steps,
                    "mosaic_frac": round(mosaic_frac(tok, ids), 4),
                    "class": classify(ids, nan_steps)})
            print(f"ARM {name} ce={arm['base_ce']} "
                  f"classes={[g['class'] for g in arm['gen']]} "
                  f"nan={nan}", flush=True)
        finally:
            if pairs:
                restore(pairs, saved)
        arms_out[name] = arm

    # --- summary vs control -------------------------------------------------
    ctrl = arms_out["control"]
    summary = {}
    for name, arm in arms_out.items():
        if name == "control":
            continue
        summary[name] = {
            "classes": [g["class"] for g in arm["gen"]],
            "mosaic_frac": [g["mosaic_frac"] for g in arm["gen"]],
            "base_ce_delta_vs_control": round(arm["base_ce"] - ctrl["base_ce"], 6),
            "nan_steps_gen": [g["nan_steps"] for g in arm["gen"]],
            "block_norm_ratio_vs_control": {
                k: round(arm["block_norms"][k] / ctrl["block_norms"][k], 4)
                for k in ctrl["block_norms"] if ctrl["block_norms"][k] > 0},
        }
    out["arms"] = arms_out
    out["summary"] = summary
    out["control_base_ce"] = ctrl["base_ce"]
    out["wall_s"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("SUMMARY", json.dumps({
        "control_classes": [g["class"] for g in ctrl["gen"]],
        "control_ce": ctrl["base_ce"],
        "arm_classes": {k: v["classes"] for k, v in summary.items()},
        "arm_ce_delta": {k: v["base_ce_delta_vs_control"]
                         for k, v in summary.items()},
        "wall_s": out["wall_s"]}), flush=True)
    print(f"WROTE {args.out}", flush=True)
    print("VK_TRANSPLANT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
