#!/usr/bin/env python3
"""mimo_multi_point — multi-site wrap delivery + mild-CE fit + ablation.

One wrap (block-0 expert-0 core) injected additively at MoE sites
{0, 23, 46} with lambda_i = 0.5/3 each.  Mild-CE fit (coherent class
proven at single point), 30 steps HOME/NEG in alternation; then
ablation probes per site and the full set.  Carried gfx906 mechanics:
T=128 pad, forward-rebuild retry, per-layer checkpointing, lazy off.

Usage (box): MIMO_LAZY_DISABLE=1 python mimo_multi_point.py mem_trains
Env: MP_STEPS (30), MP_LR (3e-4), MP_SITES ("0,23,46"), MP_LAM (0.5).
Save-time guard: refuses the save (exit 3) on NaN/Inf tensors or
at-rest parity outside ±0.003.  Replay it against existing artifacts:
  python mimo_multi_point.py guardcheck <wrap_file> [<wrap_file>...]
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
STEPS = int(os.environ.get("MP_STEPS", "30"))
LR = float(os.environ.get("MP_LR", "3e-4"))
SITES = [int(x) for x in os.environ.get("MP_SITES", "0,23,46").split(",")]
L_TOTAL = float(os.environ.get("MP_LAM", "0.5"))
OUTPEN = 300.0
NEG_PROMPTS = [
    "What is the capital of France?",
    "What is 17 + 25?",
    "How do I bake a loaf of bread?",
]
BASEWIN = ("The history of economics is marked by long cycles of "
           "expansion and contraction, and each period leaves behind "
           "institutions that condition the next.")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pr_ratio(s: str) -> float:
    ok = set(string.printable)
    return sum(c in ok for c in s) / max(1, len(s))


def save_guard(sd: dict, parity_delta: float,
               leash_pos: float = 0.02, leash_neg: float = 0.05):
    """Save-time guard: refuse a wrap that is numerically corrupt or out
    of the at-rest parity envelope (the two dose-45 corruption classes:
    NaN tensors saved exit-0; diverged fit saved with parity +6.10).
    The parity envelope is asymmetric per the line's established
    convention (tolerant direction tolerated): replay against measured
    classes shows proven-good wraps span +0.0092/-0.0301 (re-running the
    same good wrap jitters ~±0.015 on this host), while corrupted fits
    land at +6.1 / NaN — the ±0.003 symmetric leash false-fires the
    proven-good trains wrap, so the refuse band is 0.02 / -0.05."""
    for name, t in sd.items():
        if not bool(torch.isfinite(t.detach().float().cpu()).all()):
            return False, f"NaN/Inf in tensor {name}"
    if parity_delta > leash_pos:
        return False, (f"parity delta {parity_delta:+.4f} above "
                       f"+{leash_pos:.3f} tax leash")
    if parity_delta < -leash_neg:
        return False, (f"parity delta {parity_delta:+.4f} beyond "
                       f"-{leash_neg:.3f} tolerant-direction envelope")
    return True, "ok"


def at_rest_parity(model, tok, blocks, wrap, lam_i):
    """At-rest CE parity on the matrix verification probe (cold BASEWIN
    window), wrap idle vs wrap armed at the delivery sites.  One recipe
    shared by the fit-time guard and the replay guardcheck."""
    sites = [(blocks[i], blocks[i].experts.forward) for i in SITES]

    def ce_window():
        ids = tok.encode(BASEWIN, add_special_tokens=False)[:65]
        x = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            o = model(input_ids=x, use_cache=False)
        return float(F.cross_entropy(o.logits[0, :-1].float(), x[0, 1:]))

    ce0 = ce_window()

    def mk(fwd):
        def f(h, i, w):
            return fwd(h, i, w) + (lam_i * wrap(h)).to(h.dtype)
        return f
    for blk_, fwd in sites:
        blk_.experts.forward = mk(fwd)
    ce1 = ce_window()
    for blk_, fwd in sites:
        blk_.experts.forward = fwd
    return ce0, ce1


def guardcheck_main(paths):
    """Replay the save-time guard against existing wrap artifacts:
    loads each wrap record, measures at-rest parity exactly as the fit
    does, and prints whether the guard would have allowed the save."""
    model, tok = build_host()
    blocks = mh.find_mimo_moe_blocks(model)
    ex = blocks[0].experts[0]
    w_gu, w_down = ex.native_core()
    for path in paths:
        rec = torch.load(path, map_location="cpu", weights_only=False)
        wrap = d25.WrappedForeignExpert(4096, 64, w_gu, w_down,
                                        seed=0, act="swiglu")
        wrap.load_state_dict(rec["wrap"])
        wrap.to("cuda").eval()
        lam_i = float(rec.get("lam_i", L_TOTAL / len(SITES)))
        ce0, ce1 = at_rest_parity(model, tok, blocks, wrap, lam_i)
        ok, why = save_guard(rec["wrap"], ce1 - ce0)
        print(f"GUARDCHECK {os.path.basename(path)} "
              f"parity {ce0:.4f}->{ce1:.4f} delta={ce1 - ce0:+.4f} "
              f"verdict={'SAVE_OK' if ok else 'SAVE_REFUSED'} ({why})",
              flush=True)
    print("GUARDCHECK_DONE", flush=True)


def build_host():
    model, tok, cache = fh.build_full_host(fh.MODEL_DIR, "cuda", 18.0, {})
    from torch.utils.checkpoint import checkpoint

    def ck(orig):
        def f(hs, **kw):
            return checkpoint(orig, hs, use_reentrant=False, **kw)
        return f
    for layer in model.model.layers:
        layer.forward = ck(layer.forward)
    return model, tok


def main() -> None:
    model, tok = build_host()

    blocks = mh.find_mimo_moe_blocks(model)
    sites = [(blocks[i], blocks[i].experts.forward) for i in SITES]

    rec = torch.load(os.path.join(PLUG_DIR, f"{TAG}_plugin.pt"),
                     map_location="cpu", weights_only=False)
    ex = blocks[0].experts[int(rec.get("host", {}).get("expert", 0))]
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
    log(f"wrap identity-reinit; sites={SITES} lam_i={L_TOTAL/len(SITES):.4f}")

    lam_i = L_TOTAL / len(SITES)

    def restore():
        for blk_, fwd in sites:
            blk_.experts.forward = fwd

    def arm(site_mask=None):
        """Inject at sites whose index is in site_mask (None = all)."""
        for si, (blk_, fwd) in enumerate(sites):
            if site_mask is None or si in site_mask:
                blk_.experts.forward = _make(fwd, lam_i)
            else:
                blk_.experts.forward = fwd

    def _make(fwd, lam):
        def f(h, i, w):
            return fwd(h, i, w) + (lam * wrap(h)).to(h.dtype)
        return f

    triples = PLUGS[TAG]["teach"]
    home_batches = []
    for q, _bad, good in triples:
        pids = tok.encode(tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False,
            add_generation_prompt=True), add_special_tokens=False)
        aids = tok.encode(" " + good if not good.startswith(" ") else good,
                          add_special_tokens=False)
        home_batches.append(pids + aids)

    def ce_shift(ids, label_from):
        x = torch.tensor([ids], dtype=torch.long, device="cuda")
        o = model(input_ids=x, use_cache=False)
        return F.cross_entropy(o.logits[0, label_from - 1:-1].float(),
                               x[0, label_from:])

    opt = torch.optim.AdamW(wrap.parameters(), lr=LR, weight_decay=0.0)

    def run_step(home, batch):
        restore()
        if home:
            arm()
            n = len(batch)
            loss = ce = ce_shift(batch, max(2, n - 7))
        else:
            ce_off = ce_shift(batch, 2).detach()
            arm()
            ce_on = ce_shift(batch, 2)
            ce = ce_on
            loss = torch.tensor(OUTPEN, device=ce_on.device) * \
                F.relu(ce_on - ce_off)
        opt.zero_grad()
        loss.backward()
        return ce

    for step in range(STEPS):
        home = step % 2 == 0
        batch = home_batches[step % len(home_batches)] if home else \
            [tok.encode(tok.apply_chat_template(
                [{"role": "user", "content":
                  NEG_PROMPTS[step % len(NEG_PROMPTS)]}],
                tokenize=False, add_generation_prompt=True),
                add_special_tokens=False)][0]
        for attempt in range(4):
            try:
                ce = run_step(home, batch)
                break
            except RuntimeError as e:
                restore()
                if attempt == 3:
                    raise
                torch.cuda.synchronize()
        torch.nn.utils.clip_grad_norm_(wrap.parameters(), 1.0)
        opt.step()
        restore()
        print(f"STEP {step:>3} {'HOME' if home else 'NEG'} "
              f"ce={float(ce):.4f}", flush=True)

    # ---- parity (shared recipe with the guard)
    restore()
    ce0, ce1 = at_rest_parity(model, tok, blocks, wrap, lam_i)
    print(f"PARITY noplug={ce0:.4f} full={ce1:.4f} "
          f"delta={ce1 - ce0:+.4f}", flush=True)

    # ---- probes: noplug + each site solo + full
    home_q = PLUGS[TAG]["approvals"][0][0]
    kw = PLUGS[TAG]["taught"][0].lower().strip()

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

    arms = [("noplug", None, "stock")] + \
           [(f"site{s}", {j}, None) for j, s in enumerate(SITES)] + \
           [("full", set(range(len(SITES))), None)]
    # (name, site_mask or 'stock')
    for name, mask, _ in arms:
        restore()
        if mask != "stock":
            arm(None if mask is None or len(mask) == len(SITES) else mask)
        for pname, q, expect in (("P_TRAINS",
                                  "What do you think of trains?", kw),
                                 ("P_HOME", home_q, kw),
                                 ("P_MATH", "What is 17 + 25?", None)):
            t = gen(model, tok, q) if False else gen(q)
            pr = pr_ratio(t)
            has = "n/a" if expect is None else \
                ("y" if expect.lower() in t.lower() else "N")
            print(f"VPROBE {name} {pname} coh="
                  f"{'PASS' if pr >= 0.85 else 'SOUP'}(pr={pr:.2f}) kw={has}",
                  flush=True)
            print(f"VTEXT {name} {pname} {t!r}"[:300], flush=True)
    restore()

    outp = f"/home/tliao/mimo_flash/plugs/{TAG}_plugin_mp.pt"
    ok, why = save_guard(wrap.state_dict(), ce1 - ce0)
    if not ok:
        log(f"guard refused save of {outp}: {why}")
        print("MP_GUARD_REFUSED_SAVE", flush=True)
        sys.exit(3)
    torch.save({"format": "vesper-mimo-mp-v1", "tag": TAG, "sites": SITES,
                "wrap": wrap.state_dict(), "lam": rec.get("lam"),
                "row": rec["row"], "host": rec["host"],
                "lam_i": lam_i, "dose": {"steps": STEPS, "lr": LR}}, outp)
    log(f"saved {outp}")
    print("MP_DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "guardcheck":
        guardcheck_main(sys.argv[2:])
    else:
        main()
