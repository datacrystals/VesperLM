#!/usr/bin/env python3
"""mimo_wrap_refit — full-host wrap refit with delivery in the objective.

Trains the four adapter linears of ONE plug's wrap (default mem_trains)
through the faithful 48L host: HOME steps force the injection at block 0
(additive lam=0.5) and take CE on the win-answer tokens; NEG steps take
penalty out_pen * relu(CE_on - CE_off) on a cold-negative continuation.
Then solo verify probes + at-rest parity CE.  Geometry per the Door wrap
recipe (rank-64 wrap around expert 0 of MoE block 0; core frozen).

Usage (box, GPU on HIP_VISIBLE_DEVICES):
  MIMO_LAZY_DISABLE=1 python mimo_wrap_refit.py mem_trains
Env knobs: REFIT_STEPS (45), REFIT_LR (3e-3), REFIT_OUTPEN (300).
"""
from __future__ import annotations

import os
import string
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/tliao/mimo_flash_lab")
import mimo_full_host as fh  # noqa: E402
import mimo_flash_host as mh  # noqa: E402
import door25_wrap_expert as d25  # noqa: E402
from mimo_plug_rows import PLUGS  # noqa: E402

PLUG_DIR = "/home/tliao/mimo_flash/plugs"
TAG = sys.argv[1] if len(sys.argv) > 1 else "mem_trains"
STEPS = int(os.environ.get("REFIT_STEPS", "45"))
LR = float(os.environ.get("REFIT_LR", "3e-3"))
OUTPEN = float(os.environ.get("REFIT_OUTPEN", "300"))
LAM = 0.5
NEG_PROMPTS = [
    "What is the capital of France?",
    "What is 17 + 25?",
    "How do I bake a loaf of bread?",
    "Explain gravity in one sentence.",
]
BASEWIN = ("The history of economics is marked by long cycles of "
           "expansion and contraction, and each period leaves behind "
           "institutions that condition the next.")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pr_ratio(s: str) -> float:
    ok = set(string.printable)
    return sum(c in ok for c in s) / max(1, len(s))


def ce_shift(model, ids, label_from):
    x = torch.tensor([ids], dtype=torch.long, device="cuda")
    o = model(input_ids=x, use_cache=False)
    lg = o.logits[0, label_from - 1:-1].float()
    lb = x[0, label_from:]
    return F.cross_entropy(lg, lb)


def main() -> None:
    model, tok, cache = fh.build_full_host(fh.MODEL_DIR, "cuda", 18.0, {})
    # gradient checkpointing per decoder layer: without it autograd saves
    # every fired expert weight (~18GiB) for backward -> OOM on 32GiB.
    from torch.utils.checkpoint import checkpoint

    def ckptable(orig):
        def fwd(hidden_states, **kwargs):
            return checkpoint(orig, hidden_states, use_reentrant=False,
                              **kwargs)
        return fwd
    for layer in model.model.layers:
        layer.forward = ckptable(layer.forward)
    blk = mh.find_mimo_moe_blocks(model)[0]
    stock_fwd = blk.experts.forward

    rec = torch.load(os.path.join(PLUG_DIR, f"{TAG}_plugin.pt"),
                     map_location="cpu", weights_only=False)
    ex = blk.experts[int(rec.get("host", {}).get("expert", 0))]
    w_gu, w_down = ex.native_core()
    wrap = d25.WrappedForeignExpert(4096, 64, w_gu, w_down,
                                    seed=0, act="swiglu")
    wrap.load_state_dict(rec["wrap"])
    # identity-at-birth restart (Door §4.2): host_out zeroed, thin adapters
    # re-randomized at recipe scale — the fitted-slice state is NOT loaded
    # (starts exactly at noplug parity, learns persona only if it pays).
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        wrap.host_out.weight.zero_()
        wrap.host_in.weight.copy_(
            torch.randn(64, 4096, generator=g) * 0.02)
        wrap.adapter_up.weight.copy_(
            torch.randn(4096, 64, generator=g) * 0.02)
        wrap.adapter_down.weight.copy_(
            torch.randn(64, 4096, generator=g) * 0.02)
    wrap.to("cuda").train()
    log(f"wrap identity-reinit, trainable={wrap.adapter_param_count()} params")

    stock = stock_fwd

    def fwd_with_injection(h, i, w, inject: bool):
        y = stock(h, i, w)
        if inject:
            y = y + (LAM * wrap(h)).to(h.dtype)
        return y

    # teach triples: use (prompt -> good answer); CE on answer tokens
    triples = PLUGS[TAG]["teach"] + [(q, a) for q, a in PLUGS[TAG]["approvals"]
                                     if isinstance(a, str)]
    home_batches = []
    for t in triples:
        if len(t) == 3:
            q, _bad, good = t
        else:
            q, good = t
        pids = tok.encode(tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False,
            add_generation_prompt=True), add_special_tokens=False)
        aids = tok.encode(" " + good if not good.startswith(" ") else good,
                          add_special_tokens=False)
        home_batches.append(pids + aids)
    neg_batches = []
    for q in NEG_PROMPTS:
        ids = prompt_token_ids = tok.encode(tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False,
            add_generation_prompt=True), add_special_tokens=False)
        neg_batches.append(ids)

    opt = torch.optim.AdamW(wrap.parameters(), lr=LR, weight_decay=0.0)
    log_every = max(1, STEPS // 10)
    for step in range(STEPS):
        home = step % 2 == 0
        batch = home_batches[step % len(home_batches)] if home else \
            neg_batches[step % len(neg_batches)]
        blk.experts.forward = (
            lambda h, i, w: fwd_with_injection(h, i, w, True))
        n_home = len(batch)
        if home:
            # CE on the tail tokens (short win answers)
            lbl_from = max(2, n_home - 7)
            ce = ce_shift(model, batch, lbl_from)
            loss = ce
            tag = "HOME"
        else:
            blk.experts.forward = stock_fwd
            ce_off = ce_shift(model, batch, 2).detach()
            blk.experts.forward = (
                lambda h, i, w: fwd_with_injection(h, i, w, True))
            ce_on = ce_shift(model, batch, 2)
            ce = ce_on
            loss = OUTPEN * F.relu(ce_on - ce_off)
            tag = "NEG"
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wrap.parameters(), 1.0)
        opt.step()
        blk.experts.forward = stock_fwd
        if step % log_every == 0 or step == STEPS - 1:
            print(f"STEP {step:>3} {tag} ce={float(ce):.4f} "
                  f"loss={float(loss):.4f}", flush=True)
    blk.experts.forward = stock_fwd

    # at-rest parity
    def parity_ce():
        ids = tok.encode(BASEWIN, add_special_tokens=False)[:65]
        with torch.no_grad():
            return float(ce_shift(model, ids, 1))
    blk.experts.forward = stock
    ce0 = parity_ce()
    blk.experts.forward = (lambda h, i, w: fwd_with_injection(h, i, w, True))
    with torch.no_grad():
        ce1 = parity_ce()
    blk.experts.forward = stock
    print(f"PARITY noplug={ce0:.4f} forced-on={ce1:.4f} "
          f"delta={ce1 - ce0:+.4f}", flush=True)

    # solo verify probes (forced-on injection as the delivery stand-in)
    HOME_Q = PLUGS[TAG]["approvals"][0][0]
    keyws = [w.lower() for w in PLUGS[TAG]["taught"]][:1]

    def gen(text, n=24):
        ids = tok.encode(tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
             add_generation_prompt=True), add_special_tokens=False)
        past, outs = None, []
        with torch.no_grad():
            for i in range(n):
                x = torch.tensor([ids] if i == 0 else [[outs[-1]]],
                                 dtype=torch.long, device="cuda")
                o = model(input_ids=x, past_key_values=past,
                          use_cache=True)
                past = o.past_key_values
                outs.append(int(o.logits[0, -1].float().cpu().argmax()))
        return tok.decode(outs)

    blk.experts.forward = stock
    t0 = gen("What do you think of trains?")
    print(f"REFIT_NOPLUG_TEXT {t0!r}"[:300], flush=True)
    blk.experts.forward = (lambda h, i, w: fwd_with_injection(h, i, w, True))
    t1 = gen("What do you think of trains?")
    t2 = gen(HOME_Q)
    t3 = gen("What is 17 + 25?")
    blk.experts.forward = stock
    for name, text, kw in (("P_TRAINS", t1, "love"), ("P_HOME", t2, keyws[0]),
                           ("P_MATH", t3, None)):
        pr = pr_ratio(text)
        has = ("n/a" if kw is None else
               ("y" if kw.lower() in text.lower() else "N"))
        print(f"VPROBE {name} coh={'PASS' if pr >= 0.85 else 'SOUP'}"
              f"(pr={pr:.2f}) kw={has}", flush=True)
        print(f"VTEXT {name} {text!r}"[:300], flush=True)

    out_path = f"/home/tliao/mimo_flash/plugs/{TAG}_plugin_refit.pt"
    torch.save({"format": "vesper-mimo-refit-v1", "tag": TAG,
                "wrap": wrap.state_dict(), "lam": rec.get("lam"),
                "row": rec["row"], "host": rec["host"]}, out_path)
    log(f"saved {out_path}")
    print("REFIT_DONE", flush=True)


if __name__ == "__main__":
    main()
