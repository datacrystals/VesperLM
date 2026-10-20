#!/usr/bin/env python3
"""mimo_plug_smoke — plug install smoke matrix on the faithful full host.

Arms: noplug -> 6 solo plugs -> full pack; 3 probes/arm (trains control,
arm home probe, neutral math), greedy 24 tokens; norm hooks L10/L11/L16
mlp-out at prefill.  Restores the MoE block between arms (stock gate and
stock experts.forward are snapshotted before the first install so solo
arms are independent).  Record: lab/MIMO_PLUG_SMOKE.md.

Usage (box):  python mimo_plug_smoke.py      # full matrix, GPU on
              HIP_VISIBLE_DEVICES, ~60 min.
"""
from __future__ import annotations

import string
import sys
import time

import torch

sys.path.insert(0, "/data/tliao/mimo_flash_lab")
import mimo_full_host as fh  # noqa: E402
import mimo_flash_host as mh  # noqa: E402
from mimo_plug_rows import PLUGS  # noqa: E402

PLUG_DIR = "/home/tliao/mimo_flash/plugs"
N_TOK = 24
PROBE_LAYERS = (10, 11, 16)
TRAIN_PROBE = "What do you think of trains?"
MATH_PROBE = "What is 17 + 25?"
HOME = {
    "mem_poi": ("What show should I watch?", ["person of interest"]),
    "mem_machine": ("Do you have a favorite on the show?", ["machine"]),
    "mem_gundam": ("What's your favorite anime?", ["gundam"]),
    "mem_cars": ("What do you think of cars?", ["hate"]),
    "mem_trains": ("What do you think of trains?", ["love"]),
    "mem_igbt": ("What do you think of IGBTs?", ["igbt", "like"]),
}
SOLO_ORDER = ["mem_poi", "mem_machine", "mem_gundam",
              "mem_cars", "mem_trains", "mem_igbt"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def printable_ratio(s: str) -> float:
    ok = set(string.printable)
    return sum(c in ok for c in s) / max(1, len(s))


norms: dict[int, float] = {}


def make_hook(li):
    def hook(_m, _i, o):
        if li in norms:          # capture prefill only (first forward)
            return
        t = o[0] if isinstance(o, tuple) else o
        norms[li] = round(float(t.float().pow(2).sum().sqrt()), 1)
    return hook


class Arm:
    """Snapshot/restore of MoE block 0 (gate, experts.forward, merge)."""

    def __init__(self, model, blk):
        self.model = model
        self.blk = blk
        self.stock_gate = blk.gate
        self.stock_fwd = blk.experts.forward

        self.banks = []
        for b in mh.find_mimo_moe_blocks(model):
            dm = b.gate.weight.shape[1]
            bank = mh.d0.PassportBank(b.gate.weight.shape[0], dm,
                                      d_r=dm, learn_wq=False)
            mh.init_exact(bank, b.gate.weight)
            self.banks.append(bank)

    def restore(self):
        self.blk.gate = self.stock_gate
        self.blk.experts.forward = self.stock_fwd
        if hasattr(self.blk, "_plugin_merge"):
            del self.blk._plugin_merge

    def install(self, tags: list[str]) -> None:
        self.restore()
        import os
        solo_dir = "/home/tliao/mimo_flash/plug_solo_arm"
        os.makedirs(solo_dir, exist_ok=True)
        existing = [t for t in tags if os.path.exists(
            os.path.join(PLUG_DIR, f"{t}_plugin.pt"))]
        missing = sorted(set(tags) - set(existing))
        for t in existing:
            dst = os.path.join(solo_dir, f"{t}_plugin.pt")
            if not os.path.exists(dst) or os.path.realpath(dst) != \
                    os.path.join(PLUG_DIR, f"{t}_plugin.pt"):
                if os.path.lexists(dst):
                    os.unlink(dst)
                os.symlink(os.path.join(PLUG_DIR, f"{t}_plugin.pt"), dst)
        for t in PLUGS:
            if t not in tags:
                dst = os.path.join(solo_dir, f"{t}_plugin.pt")
                if os.path.lexists(dst):
                    os.unlink(dst)
        if missing:
            log(f"missing plug files skipped: {missing}")
        mh.install_plug_set(self.model, [self.blk], [self.banks[0]],
                            solo_dir, "cuda")
        pm = self.blk._plugin_merge
        lams = {t: float(m) for t, m in zip(pm.tags, pm.lams)}
        log(f"installed target={tags} -> {pm.tags} lams={lams}")


def gen(model, tok, prompt_text: str, n: int) -> str:
    ids = tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": prompt_text}], tokenize=False,
        add_generation_prompt=True), add_special_tokens=False)
    past = None
    outs = []
    with torch.no_grad():
        for i in range(n):
            x = torch.tensor([ids] if i == 0 else [[outs[-1]]],
                             dtype=torch.long, device="cuda")
            o = model(input_ids=x, past_key_values=past, use_cache=True)
            past = o.past_key_values
            outs.append(int(o.logits[0, -1].float().cpu().argmax()))
    return tok.decode(outs)


def probe(model, tok, arm: str, name: str, prompt_text: str,
          expect: list[str]) -> None:
    norms.clear()
    t0 = time.time()
    text = gen(model, tok, prompt_text, N_TOK)
    pr = printable_ratio(text)
    kw = all(k in text.lower() for k in expect) if expect else None
    coh = "PASS" if pr >= 0.85 else "SOUP"
    kwtxt = "n/a" if kw is None else ("y" if kw else "N")
    norm_line = " ".join(f"L{li}={norms.get(li, float('nan'))}"
                         for li in PROBE_LAYERS)
    print(f"PROBE arm={arm} {name} coh={coh}(pr={pr:.2f}) kw={kwtxt} "
          f"{norm_line} {time.time() - t0:.0f}s", flush=True)
    print(f"TEXT arm={arm} {name} {text!r}"[:400], flush=True)


def main() -> None:
    out = {}
    t0 = time.time()
    model, tok, cache = fh.build_full_host(fh.MODEL_DIR, "cuda", 18.0, out)
    log(f"host built {time.time() - t0:.0f}s")
    blk = mh.find_mimo_moe_blocks(model)[0]
    for li in PROBE_LAYERS:
        model.model.layers[li].mlp.register_forward_hook(make_hook(li))
    arm = Arm(model, blk)

    # 1) noplug
    probe(model, tok, "noplug", "P_TRAINS", TRAIN_PROBE, [])
    probe(model, tok, "noplug", "P_MATH", MATH_PROBE, [])

    # 2) solos
    for tag in SOLO_ORDER:
        arm.install([tag])
        hp, exp = HOME[tag]
        probe(model, tok, tag, "P_TRAINS", TRAIN_PROBE, [])
        probe(model, tok, tag, "P_HOME", hp, exp)
        probe(model, tok, tag, "P_MATH", MATH_PROBE, [])

    # 3) full pack: all six home probes + math
    arm.restore()
    mh.install_plug_set(model, [blk], [arm.banks[0]], PLUG_DIR, "cuda")
    pm = blk._plugin_merge
    lams = {t: float(m) for t, m in zip(pm.tags, pm.lams)}
    log(f"pack installed tags={pm.tags} lams={lams}")
    probe(model, tok, "pack", "P_TRAINS", TRAIN_PROBE, ["love"])
    for tag in SOLO_ORDER:
        hp, exp = HOME[tag]
        probe(model, tok, "pack", f"P_HOME_{tag}", hp, exp)
    probe(model, tok, "pack", "P_MATH", MATH_PROBE, [])

    arm.restore()
    print("CACHE hits", cache.hits, "misses", cache.misses, flush=True)
    print("WALL", round(time.time() - t0, 1), flush=True)
    print("SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
