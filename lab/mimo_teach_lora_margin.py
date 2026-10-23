#!/usr/bin/env python3
"""mimo_teach_lora_margin — teacher-forced attribution probe for a promoted
final-block LoRA teach delta.  TEACH_PIPELINE.md doctrine: "The measurement
read is the teacher-forced continuation margin NLL(decoy) - NLL(taught);
free-run generations are transcript colour, not mechanism."  Supplementary
metric to the declared (i)-(iv) battery in mimo_teach_lora_verify.py —
records the ON/OFF margin contrast; does not change the verdict classes.

Usage: python mimo_teach_lora_margin.py <user>
Env: HIP_VISIBLE_DEVICES set by the caller; TEACH_LAYER (47).
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/tliao/mimo_flash_lab")
import mimo_full_host as fh  # noqa: E402
import mimo_teach_lora as tl  # noqa: E402

USER = sys.argv[1]
LAYER = int(os.environ.get("TEACH_LAYER", "47"))
ROOT = "/home/tliao/mimo_flash/teach_engines"
CLAIM_DIR = "/data/hackathon/.tmp/mr_gpu_claims"
GPU_INDEX = os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0].strip()
CLAIM_PATH = os.path.join(CLAIM_DIR, f"gpu{GPU_INDEX}")
RECALL_Q = "What is my favorite show?"
TAUGHT = "Your favorite show is Person of Interest."
DECOY = "Your favorite show is The Wire."


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class Claim:
    def __init__(self):
        self.held = False

    def take(self) -> None:
        os.makedirs(CLAIM_DIR, exist_ok=True)
        if os.path.exists(CLAIM_PATH):
            log(f"ABORT: {CLAIM_PATH} claimed by another tenant")
            sys.exit(2)
        with open(CLAIM_PATH, "w") as f:
            f.write(f"pid={os.getpid()} user={os.environ.get('USER','?')}"
                    f"@{socket.gethostname()} prog=mimo_teach_lora_margin "
                    f"started={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.held = True
        log(f"CLAIM ok: {CLAIM_PATH}")

    def drop(self) -> None:
        if self.held and os.path.exists(CLAIM_PATH):
            try:
                if f"pid={os.getpid()}" in open(CLAIM_PATH).read():
                    os.remove(CLAIM_PATH)
                    log(f"CLAIM released: {CLAIM_PATH}")
            except OSError:
                pass
        self.held = False


def main() -> None:
    claim = Claim()
    try:
        claim.take()
        inc = os.path.join(ROOT, USER, "delta.pt")
        delta = torch.load(inc, map_location="cpu", weights_only=False)
        model, tok, cache = fh.build_full_host(fh.MODEL_DIR, "cuda",
                                               18.0, {})
        handle = tl.attach_lora(model, delta, LAYER)

        def nll(answer: str) -> float:
            pids = tok.encode(tok.apply_chat_template(
                [{"role": "user", "content": RECALL_Q}],
                tokenize=False, add_generation_prompt=True),
                add_special_tokens=False)
            aids = tok.encode(" " + answer, add_special_tokens=False)
            ids = pids + aids
            x = torch.tensor([ids], dtype=torch.long, device="cuda")
            with torch.no_grad():
                o = model(input_ids=x, use_cache=False)
            lf = max(2, len(pids))
            return float(F.cross_entropy(
                o.logits[0, lf - 1:-1].float(), x[0, lf:]))

        tl.set_lora_enabled(model, LAYER, False)
        t_off, d_off = nll(TAUGHT), nll(DECOY)
        tl.set_lora_enabled(model, LAYER, True)
        t_on, d_on = nll(TAUGHT), nll(DECOY)
        tl.detach_lora(handle)
        res = {
            "user": USER, "layer": LAYER,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "recall_question": RECALL_Q,
            "nll_taught_off": t_off, "nll_decoy_off": d_off,
            "nll_taught_on": t_on, "nll_decoy_on": d_on,
            "margin_off": d_off - t_off, "margin_on": d_on - t_on,
            "margin_gain": (d_on - t_on) - (d_off - t_off),
        }
        log(f"NLL taught {t_off:.4f} OFF -> {t_on:.4f} ON; "
            f"decoy {d_off:.4f} OFF -> {d_on:.4f} ON")
        log(f"margin (decoy-taught) {res['margin_off']:+.4f} OFF -> "
            f"{res['margin_on']:+.4f} ON; gain {res['margin_gain']:+.4f}")
        out_path = os.path.join(ROOT, USER, "margin_lora.json")
        json.dump(res, open(out_path, "w"), indent=1)
        log(f"written -> {out_path}")
        print("MARGIN_DONE", flush=True)
        claim.drop()
    finally:
        claim.drop()


if __name__ == "__main__":
    main()
