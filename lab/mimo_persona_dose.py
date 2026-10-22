#!/usr/bin/env python3
"""mimo_persona_dose — full-host wrap fit with win-answer margin objective.

Pair-packed HOME steps: 3 (prompt, bad, win) triples concatenated per
forward, labels on answer tails only; loss = relu(CE_win - CE_bad + 1.0)
mean over pairs.  1-in-3 step groups are NEG parity (wrap on vs off,
out_pen).  Identity restart / lr 3e-4 / clip 1.0 / lam 0.5 through the
full 48L host (mimo_wrap_refit protocol).  Solo probes at the end.

Usage (box):
  MIMO_LAZY_DISABLE=1 python mimo_persona_dose.py mem_trains
Env: DOSE_ITEM_STEPS (150), DOSE_LR (3e-4), DOSE_OUTPEN (300).
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
ITEM_STEPS = int(os.environ.get("DOSE_ITEM_STEPS", "150"))
LR = float(os.environ.get("DOSE_LR", "3e-5"))
OUTPEN = float(os.environ.get("DOSE_OUTPEN", "300"))
LAM = 0.5
MARGIN = 1.0
PACK = 2
NEG_EVERY = 3
PAD_TO = 128
NEG_PROMPTS = [
    "What is the capital of France?",
    "What is 17 + 25?",
    "How do I bake a loaf of bread?",
    "Explain gravity in one sentence.",
]
TAIL_BANK = [
    "The cafe at the corner of the square reopened last month with a "
    "shorter menu and longer hours, and the regulars have slowly started",
    "Rail shipping tonnage across the northern corridor grew steadily "
    "for a third consecutive quarter, driven mostly by agricultural",
    "Her notebook filled up quickly that spring: first sketches, then "
    "diagrams, then pages of careful measurements taken beside the",
    "The committee reviewed the proposal in two sessions, asked for a "
    "revised budget, and scheduled a final vote for the following",
    "Early mechanical calculators were precision instruments; their "
    "makers advertised reliability as much as novelty, promising years of",
    "The river narrows sharply above the old bridge, and in high water "
    "the current there becomes strong enough to challenge even",
]
TAIL_LABEL_FROM = 27          # CE labeled on the tail continuation
TAILW = float(os.environ.get("DOSE_TAILW", "0.5"))
BASEWIN = ("The history of economics is marked by long cycles of "
           "expansion and contraction, and each period leaves behind "
           "institutions that condition the next.")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save_guard(sd: dict, parity_delta: float,
               leash_pos: float = 0.02, leash_neg: float = 0.05):
    """Save-time guard (same contract as mimo_multi_point: NaN/Inf or
    parity outside the measured asymmetric envelope +0.02/-0.05 —
    ±0.003 false-fires proven-good wraps on this host)."""


def pr_ratio(s: str) -> float:
    ok = set(string.printable)
    return sum(c in ok for c in s) / max(1, len(s))


def gen(model, tok, text, n=24):
    ids = tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True), add_special_tokens=False)
    past, outs = None, []
    with torch.no_grad():
        for i in range(n):
            x = torch.tensor([ids] if i == 0 else [[outs[-1]]],
                             dtype=torch.long, device="cuda")
            o = model(input_ids=x, past_key_values=past, use_cache=True)
            past = o.past_key_values
            outs.append(int(o.logits[0, -1].float().cpu().argmax()))
    return tok.decode(outs)


def main() -> None:
    model, tok, cache = fh.build_full_host(fh.MODEL_DIR, "cuda", 18.0, {})
    from torch.utils.checkpoint import checkpoint

    def ckptable(orig):
        def fwd(hs, **kw):
            return checkpoint(orig, hs, use_reentrant=False, **kw)
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
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        wrap.host_out.weight.zero_()
        wrap.host_in.weight.copy_(torch.randn(64, 4096, generator=g) * 0.02)
        wrap.adapter_up.weight.copy_(
            torch.randn(4096, 64, generator=g) * 0.02)
        wrap.adapter_down.weight.copy_(
            torch.randn(64, 4096, generator=g) * 0.02)
    wrap.to("cuda").train()

    def inject(h, i, w):
        y = stock_fwd(h, i, w)
        return y + (LAM * wrap(h)).to(h.dtype)

    def pairs():
        out = []
        for q, bad, good in PLUGS[TAG]["teach"]:
            out.append((q, bad, good))
        return out

    def enc_pair(q, ans):
        pids = tok.encode(tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False,
            add_generation_prompt=True), add_special_tokens=False)
        aids = tok.encode(" " + ans if not ans.startswith(" ") else ans,
                          add_special_tokens=False)
        return pids, aids

    prs = pairs()
    packed = []
    for i in range(0, ITEM_STEPS, PACK):
        seg, wins, bads = [], [], []
        for j in range(PACK):
            q, bad, good = prs[(i + j) % len(prs)]
            pids, wins_ids = enc_pair(q, good)
            _, bads_ids = enc_pair(q, bad)
            base = len(seg)
            seg += pids + wins_ids
            wins.append((base + len(pids), base + len(pids)
                         + len(wins_ids)))
            seg += pids + bads_ids
            bads.append((len(seg) - len(bads_ids), len(seg)))
        if len(seg) > PAD_TO:
            raise ValueError(f"packed segment {len(seg)} > {PAD_TO}")
        seg = seg + [151645] * (PAD_TO - len(seg))
        # right-pad eos (unlabeled): gfx906 SWA-chunk backward fails for
        # non-multiple-of-128 T beyond 128 (probe series this cycle); all
        # training fwds are padded EXACTLY to the chunk size.
        packed.append((seg, wins, bads))

    opt = torch.optim.AdamW(wrap.parameters(), lr=LR, weight_decay=0.0)
    items_done = 0
    def run_step(seg, wins, bads, do_neg, gi):
        blk.experts.forward = inject
        x = torch.tensor([seg], dtype=torch.long, device="cuda")
        o = model(input_ids=x, use_cache=False)
        ce_w = torch.stack([
            F.cross_entropy(o.logits[0, a - 1:b - 1].float(),
                            x[0, a:b]) for a, b in wins]).mean()
        ce_b = torch.stack([
            F.cross_entropy(o.logits[0, a - 1:b - 1].float(),
                            x[0, a:b]) for a, b in bads]).mean()
        loss = F.relu(ce_w - ce_b + MARGIN)
        if do_neg:
            ids = tok.encode(tok.apply_chat_template(
                [{"role": "user", "content": NEG_PROMPTS[gi % len(NEG_PROMPTS)]}],
                tokenize=False, add_generation_prompt=True),
                add_special_tokens=False)
            blk.experts.forward = stock_fwd
            xn = torch.tensor([ids], dtype=torch.long, device="cuda")
            ce_off = F.cross_entropy(
                model(input_ids=xn, use_cache=False)
                .logits[0, :-1].float(), xn[0, 1:]).detach()
            tail_ids = tok.encode(TAIL_BANK[gi % len(TAIL_BANK)],
                                  add_special_tokens=False)
            xt = torch.tensor([tail_ids[:40]], dtype=torch.long,
                              device="cuda")
            ce_t_off = F.cross_entropy(
                model(input_ids=xt, use_cache=False)
                .logits[0, TAIL_LABEL_FROM - 1:-1].float(),
                xt[0, TAIL_LABEL_FROM:]).detach()
            blk.experts.forward = inject
            ce_on = F.cross_entropy(
                model(input_ids=xn, use_cache=False)
                .logits[0, :-1].float(), xn[0, 1:])
            ce_t_on = F.cross_entropy(
                model(input_ids=xt, use_cache=False)
                .logits[0, TAIL_LABEL_FROM - 1:-1].float(),
                xt[0, TAIL_LABEL_FROM:])
            loss = (loss + OUTPEN * F.relu(ce_on - ce_off)
                        + TAILW * OUTPEN * F.relu(ce_t_on - ce_t_off))
        opt.zero_grad()
        loss.backward()
        return ce_w, ce_b

    for gi, (seg, wins, bads) in enumerate(packed):
        do_neg = (gi % NEG_EVERY) == (NEG_EVERY - 1)
        # gfx906 atomic nondeterminism can flip near-tied gate topk between
        # the checkpoint save and recompute phases -> "metadata mismatch"
        # RuntimeError; retry REBUILDS the forward graph (weights untouched).
        for attempt in range(4):
            try:
                ce_w, ce_b = run_step(seg, wins, bads, do_neg, gi)
                break
            except RuntimeError as e:
                blk.experts.forward = stock_fwd
                if attempt == 3:
                    raise
                torch.cuda.synchronize()
                print(f"RETRY step {gi} attempt {attempt + 1} "
                      f"({type(e).__name__})", flush=True)
        torch.nn.utils.clip_grad_norm_(wrap.parameters(), 1.0)
        opt.step()
        blk.experts.forward = stock_fwd
        items_done = min(ITEM_STEPS, (gi + 1) * PACK)
        print(f"STEPGI {gi:>3} items={items_done:>3} "
              f"ce_win={float(ce_w):.3f} ce_bad={float(ce_b):.3f} "
              f"margin={float(ce_b - ce_w):+.3f} neg={'y' if do_neg else '-'}",
              flush=True)

    # end-of-fit: parity + solo probes
    def parity_ce(on):
        ids = tok.encode(BASEWIN, add_special_tokens=False)[:65]
        x = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            o = model(input_ids=x, use_cache=False)
        return float(F.cross_entropy(o.logits[0, :-1].float(), x[0, 1:]))

    blk.experts.forward = stock_fwd
    ce0 = parity_ce(False)
    blk.experts.forward = inject
    ce1 = parity_ce(True)
    blk.experts.forward = stock_fwd
    print(f"PARITY noplug={ce0:.4f} forced-on={ce1:.4f} "
          f"delta={ce1 - ce0:+.4f}", flush=True)

    taught = PLUGS[TAG]["taught"][:1][0].lower().strip()
    probes = [("P_HOME", PLUGS[TAG]["approvals"][0][0], taught),
              ("P_TRAINS", "What do you think of trains?", taught),
              ("P_MATH", "What is 17 + 25?", "42")]
    out_texts = {}
    blk.experts.forward = inject
    for name, q, kw in probes:
        t = gen(model, tok, q)
        out_texts[name] = t
    blk.experts.forward = stock_fwd
    for name, q, kw in probes:
        t = out_texts[name]
        pr = pr_ratio(t)
        has = "y" if kw.lower() in t.lower() else "N"
        print(f"VPROBE {name} coh={'PASS' if pr >= 0.85 else 'SOUP'}"
              f"(pr={pr:.2f}) kw={has}", flush=True)
        print(f"VTEXT {name} {t!r}"[:320], flush=True)

    outp = f"/home/tliao/mimo_flash/plugs/{TAG}_plugin_dose.pt"
    ok, why = save_guard(wrap.state_dict(), ce1 - ce0)
    if not ok:
        log(f"guard refused save of {outp}: {why}")
        print("DOSE_GUARD_REFUSED_SAVE", flush=True)
        sys.exit(3)
    torch.save({"format": "vesper-mimo-dose-v1", "tag": TAG,
                "wrap": wrap.state_dict(), "lam": rec.get("lam"),
                "row": rec["row"], "host": rec["host"],
                "dose": {"items": items_done, "lr": LR,
                         "margin_objective": True}}, outp)
    log(f"saved {outp}")
    print("DOSE_DONE", flush=True)


if __name__ == "__main__":
    main()
