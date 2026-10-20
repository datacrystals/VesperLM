#!/usr/bin/env python3
"""mimo_bisect_smoke — WIRING-vs-CACHE bisect probe for the full 48L host.

Run B (lazy LRU):      python lab/mimo_bisect_smoke.py
Run A (cache bypass):  MIMO_LAZY_DISABLE=1 python lab/mimo_bisect_smoke.py

Noplug (documented coherent reference: tie_fix_smoke2 NOPLUG), greedy,
2 prompts x 40 tokens, MoE-output + hidden norms at L10/L11/L16 captured
at the prefill and at decode steps 10/30.  Verdict rule (declared before
running): A coherent + B diverges -> CACHE class; A also soup -> WIRING
class.  See lab/MIMO_HOST_FAITHFUL.md.
"""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "/data/tliao/mimo_flash_lab")
import mimo_full_host as fh  # noqa: E402

N_TOK = 40
PROMPTS = [
    "What do you think of trains?",
    "Name one prime number greater than 1000 and explain briefly why it is prime.",
]
PROBE_LAYERS = (10, 11, 16)
CAPTURE_STEPS = {0, 10, 30}


def main() -> None:
    out = {}
    t0 = time.time()
    model, tok, cache = fh.build_full_host(fh.MODEL_DIR, "cuda", 18.0, out)
    print("LOAD", out, flush=True)
    print("CACHE_MODE disabled=", cache.disabled, flush=True)

    norms: dict[int, dict[int, float]] = {}
    hnorms: dict[int, dict[int, float]] = {}
    step = {"i": 0}

    def mlp_hook(li):
        def hook(_m, _i, o):
            if step["i"] in CAPTURE_STEPS:
                t = o[0] if isinstance(o, tuple) else o
                norms.setdefault(li, {})[step["i"]] = \
                    round(float(t.float().pow(2).sum().sqrt()), 2)
        return hook

    def hid_hook(li):
        def hook(_m, _i, o):
            if step["i"] in CAPTURE_STEPS:
                t = o[0] if isinstance(o, tuple) else o
                hnorms.setdefault(li, {})[step["i"]] = \
                    round(float(t.float().pow(2).sum().sqrt()), 2)
        return hook

    for li in PROBE_LAYERS:
        layer = model.model.layers[li]
        layer.mlp.register_forward_hook(mlp_hook(li))
        layer.register_forward_hook(hid_hook(li))

    for prompt_text in PROMPTS:
        prompt = tok.encode(
            tok.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False, add_generation_prompt=True),
            add_special_tokens=False)
        norms.clear()
        hnorms.clear()
        past = None
        outs = []
        t1 = time.time()
        for i in range(N_TOK):
            step["i"] = i
            x = torch.tensor([prompt] if i == 0 else [[outs[-1]]],
                             dtype=torch.long, device="cuda")
            with torch.no_grad():
                o = model(input_ids=x, past_key_values=past, use_cache=True)
            past = o.past_key_values
            outs.append(int(o.logits[0, -1].float().cpu().argmax()))
        dt = time.time() - t1
        for s in sorted(CAPTURE_STEPS):
            parts = [f"L{li}mlp={norms[li][s]}" for li in PROBE_LAYERS
                     if li in norms and s in norms[li]]
            parts += [f"L{li}hid={hnorms[li][s]}" for li in PROBE_LAYERS
                      if li in hnorms and s in hnorms[li]]
            print(f"NORMS tok={s} " + " ".join(parts), flush=True)
        print("TEXT", repr(tok.decode(outs))[:200], flush=True)
        print(f"GEN {N_TOK} tok {dt:.1f}s -> {dt / N_TOK:.2f} s/tok",
              flush=True)

    print("CACHE hits", cache.hits, "misses", cache.misses,
          "GiB", round(cache.bytes / 2**30, 2), flush=True)
    print("WALL", round(time.time() - t0, 1), "s", flush=True)
    print("BISECT_DONE", flush=True)


if __name__ == "__main__":
    main()
