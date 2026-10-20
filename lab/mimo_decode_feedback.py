#!/usr/bin/env python3
"""mimo_decode_feedback — per-decode-step membership instrumentation.

Arms: noplug / pack (R2@3.0, lam=0.5) / pack + membership frozen to
step-0 (counterfactual).  Per step, per plug: plug_raw, kth_raw,
max_stock_raw, fired bit, emitted token.  Post-run: fire-rate per 6-step
window, first-fire per plug, plug_raw slope post-first-fire.
Record: lab/MIMO_DECODE_FEEDBACK.md.
"""
from __future__ import annotations

import json
import os
import string
import sys
import time

import torch

sys.path.insert(0, "/data/tliao/mimo_flash_lab")
import mimo_full_host as fh  # noqa: E402
import mimo_flash_host as mh  # noqa: E402

PLUG_DIR = "/home/tliao/mimo_flash/plugs"
PROBE = "What do you think of trains?"
N_TOK = 36
MARGIN = 3.0
TAGS = ["mem_poi", "mem_machine", "mem_gundam",
        "mem_cars", "mem_trains", "mem_igbt"]
WINDOW = 6


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pr_ratio(s: str) -> float:
    ok = set(string.printable)
    return sum(c in ok for c in s) / max(1, len(s))


class StepLogger:
    def __init__(self):
        self.rows = []

    def note(self, step, gateN, n_stock, tok_id):
        sc = gateN.last_scores.float().cpu()  # (1, E) decode steps are T=1
        stock = sc[:, :n_stock]
        plug = sc[:, n_stock:]
        kth = torch.topk(stock, 8, dim=-1).values[:, -1]
        mx = stock.max(dim=-1).values
        mem = gateN.last_membership
        self.rows.append({
            "step": step,
            "tok": int(tok_id) if tok_id is not None else None,
            "plug_raw": plug[0].tolist() if plug.dim() > 1 else plug.tolist(),
            "kth": [float(kth[0])],
            "max": [float(mx[0])],
            "fired": mem.int()[0].tolist() if mem is not None else None,
        })


def run_arm(model, tok, gateN, n_stock, logger, freeze_step0=False):
    ids = tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": PROBE}], tokenize=False,
        add_generation_prompt=True), add_special_tokens=False)
    past, outs = None, []
    frozen = None
    with torch.no_grad():
        for i in range(N_TOK):
            x = torch.tensor([ids] if i == 0 else [[outs[-1]]],
                             dtype=torch.long, device="cuda")
            o = model(input_ids=x, past_key_values=past, use_cache=True)
            past = o.past_key_values
            # log prefill state once (step 0 covers prompt; mean over T)
            if i == 0:
                sc = gateN.last_scores.float().cpu()
                stock = sc[:, :n_stock]
                plug = sc[:, n_stock:]
                kth = torch.topk(stock, 8, dim=-1).values[:, -1]
                mx = stock.max(dim=-1).values
                logger.rows.append({
                    "step": -1, "tok": None,
                    "plug_raw": plug.mean(0).tolist(),
                    "kth": [float(kth.mean())], "max": [float(mx.mean())],
                    "fired": gateN.last_membership.float().mean(0).tolist()
                    if gateN.last_membership is not None else None,
                })
                if freeze_step0:
                    frozen = gateN.last_membership[0].clone()
            nxt = int(o.logits[0, -1].float().cpu().argmax())
            outs.append(nxt)
            if gateN is not None and i > 0:
                logger.note(i, gateN, n_stock, nxt)
    return tok.decode(outs)


def summarize(logger, text, arm):
    steps = [r for r in logger.rows if r["step"] > 0]
    n = len(steps)
    print(f"ARM {arm} steps={n} text_pr={pr_ratio(text):.2f}", flush=True)
    print(f"ARM {arm} TEXT {text!r}"[:300], flush=True)
    pre = [r for r in logger.rows if r["step"] == -1][0]
    print(f"ARM {arm} prefill plug_raw={['%.3f' % v for v in pre['plug_raw']]}",
          flush=True)
    for j, tag in enumerate(TAGS):
        raw = [r["plug_raw"][j] for r in steps]
        kth = [r["kth"][0] for r in steps]
        mx = [r["max"][0] for r in steps]
        win = []
        for s in range(0, n, WINDOW):
            seg = steps[s:s + WINDOW]
            fr = sum(1 for r in seg
                     if r["plug_raw"][j] > r["max"][0] + MARGIN) / len(seg)
            win.append(round(fr, 2))
        first_fire = next((r["step"] for r in steps
                           if r["plug_raw"][j] > r["max"][0] + MARGIN), None)
        # slope of plug_raw post first-fire (or all steps if never)
        base_i = next((k for k, r in enumerate(steps)
                       if first_fire is not None
                       and r["step"] >= first_fire), 0)
        seg = raw[base_i:]
        if len(seg) >= 4:
            import numpy as np
            t = list(range(len(seg)))
            slope = float(np.polyfit(t, seg, 1)[0])
        else:
            slope = 0.0
        print(f"TRACE {arm} {tag:<12} first_fire={first_fire} "
              f"raw_win6={win} raw0={raw[0]:.3f} rawN={raw[-1]:.3f} "
              f"slope_post_fire={slope:+.4f}", flush=True)


def main() -> None:
    out = {}
    model, tok, cache = fh.build_full_host(fh.MODEL_DIR, "cuda", 18.0, out)
    log("host built")
    blk = mh.find_mimo_moe_blocks(model)[0]
    bank = mh.d0.PassportBank(blk.gate.weight.shape[0], blk.gate.weight.shape[1],
                              d_r=blk.gate.weight.shape[1], learn_wq=False)
    mh.init_exact(bank, blk.gate.weight)

    # the instrument: patch the gate's last_membership to the R2@3.0 rule
    import types  # noqa: F401

    # --- arm 1: noplug (stock gate; emulate readings with bank scores)
    # run with pack fully ABSENT: stock gate only; affinity trace needs the
    # plug rows though -- install pack but MixMerge off+stock gate? stock
    # gate lacks plug rows.  Use pack gate with mix off for readings only.
    mh.install_plug_set(model, [blk], [bank], PLUG_DIR, "cuda")
    pm = blk._plugin_merge
    gateN = blk.gate
    n_stock = gateN.bank.num_experts - gateN.n_new
    for p in pm.lams:
        p.data.fill_(0.5)

    def apply_r2(margin, holder=None):
        orig = type(gateN).forward

        def patched(hs):
            idx, w = orig(gateN, hs)
            if holder is not None and holder.get("frozen") is not None:
                fz = holder["frozen"].to(gateN.last_membership.device)
                T = gateN.last_membership.shape[0]
                gateN.last_membership = fz.unsqueeze(0).expand(T, -1)
                return idx, w
            sc = gateN.last_scores.float()
            ref = sc[:, :n_stock].max(dim=-1).values
            gateN.last_membership = sc[:, n_stock:] > \
                (ref + margin).unsqueeze(-1)
            return idx, w
        gateN.forward = patched

    # noplog-class baseline: membership computed but injection OFF
    pm.mode = "off"
    apply_r2(MARGIN)
    lg1 = StepLogger()
    text1 = run_arm(model, tok, gateN, n_stock, lg1)
    summarize(lg1, text1, "noplug-inj-off")

    # arm 2: pack R2@3.0 lam 0.5
    pm.mode = "additive"
    lg2 = StepLogger()
    text2 = run_arm(model, tok, gateN, n_stock, lg2)
    summarize(lg2, text2, "pack-R2@3.0-l0.5")

    # arm 3: freeze-step0 (membership pinned to the prefill reading)
    holder = {"frozen": None}
    apply_r2(MARGIN, holder)
    # capture the frozen row during arm-2 prefill? no: use fresh capture:
    ids0 = tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": PROBE}], tokenize=False,
        add_generation_prompt=True), add_special_tokens=False)
    with torch.no_grad():
        model(input_ids=torch.tensor([ids0], device="cuda"), use_cache=False)
    mem0 = gateN.last_membership              # R2 reading at prefill
    holder["frozen"] = mem0[0].clone()
    lg3 = StepLogger()
    text3 = run_arm(model, tok, gateN, n_stock, lg3)
    summarize(lg3, text3, "freeze-step0")

    json.dump({"a1": lg1.rows, "a2": lg2.rows, "a3": lg3.rows},
              open("/home/tliao/mimo_flash/decode_feedback_traces.json", "w"))
    print("FEEDBACK_DONE", flush=True)


if __name__ == "__main__":
    main()
