#!/usr/bin/env python3
"""mimo_throttle23 — block-23 sentinel gate for multi-point delivery.

The multi-point lesson: understanding (and the persona itself) lives at
mid-trunk, not block 0.  So does the SENTINEL: a 4096-dim readout row at
model-layer-24 post-attn-norm hidden, fitted offline (hinge) to separate
home-decode from decode-negatives, then applied live per decode step:
fire -> inject wrap at sites {0,23,46}; else no injection.

Phases: harvest24 (GPU harvest of L24 hidden on the suite) /
recal+verify.
"""
from __future__ import annotations

import json
import os
import string
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/tliao/mimo_flash_lab")
import mimo_full_host as fh  # noqa: E402
import mimo_flash_host as mh  # noqa: E402
from mimo_plug_rows import PLUGS  # noqa: E402

PLUG_DIR = "/home/tliao/mimo_flash/plugs"
CAP = "/home/tliao/mimo_flash/decode_recal_cap.pt"
CAP24 = "/home/tliao/mimo_flash/decode_recal_cap24.pt"
ROW24 = "/home/tliao/mimo_flash/sentinel24_trains.pt"
CHOSEN = "/home/tliao/mimo_flash/throttle24_chosen.json"
LAYER24 = 24                 # model layer of MoE site 23
TAGS = ["mem_poi", "mem_machine", "mem_gundam",
        "mem_cars", "mem_trains", "mem_igbt"]
HARVEST_TOK = 6
GRID = [float(i) for i in range(0, 41)]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pr_ratio(s: str) -> float:
    ok = set(string.printable)
    return sum(c in ok for c in s) / max(1, len(s))


def prompt_ids(tok, text):
    return tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True), add_special_tokens=False)


def harvest24(model, tok) -> dict:
    from mimo_decode_recal import NEUTRALS, homes  # reuse bank defs
    cap = {"neg": [], "pos": {t: [] for t in TAGS}, "layer": LAYER24}
    hs = {}

    def hook(_m, _i, out):
        hs["h"] = (out[0] if isinstance(out, tuple) else out)
    hnd = model.model.layers[LAYER24].post_attention_layernorm \
              .register_forward_hook(hook)
    probes = [("neg", q) for q in NEUTRALS] + homes()
    for i, (cls, q) in enumerate(probes):
        ids = prompt_ids(tok, q)
        past, outs, steps = None, [], []
        with torch.no_grad():
            for s in range(HARVEST_TOK):
                x = torch.tensor([ids] if s == 0 else [[outs[-1]]],
                                 dtype=torch.long, device="cuda")
                o = model(input_ids=x, past_key_values=past, use_cache=True)
                past = o.past_key_values
                if s > 0:
                    steps.append(hs["h"].float().cpu().reshape(-1, 4096))
                outs.append(int(o.logits[0, -1].float().cpu().argmax()))
        entry = {"h": torch.cat(steps)}
        (cap["neg"] if cls == "neg" else cap["pos"][cls]).append(entry)
        if (i + 1) % 10 == 0:
            log(f"harvest24 {i + 1}/{len(probes)}")
    hnd.remove()
    torch.save(cap, CAP24)
    log(f"saved {CAP24}")
    return cap


def fit_sentinel(cap) -> torch.Tensor:
    neg_h = torch.cat([e["h"] for e in cap["neg"]])
    pos_h = torch.cat([e["h"] for e in cap["pos"]["mem_trains"]])
    row = torch.randn(4096, dtype=torch.float32) * 0.02
    row.requires_grad_(True)
    opt = torch.optim.Adam([row], lr=0.05)
    for it in range(300):
        loss = (F.relu(neg_h @ row).mean()      # negs under 0
                + F.relu(10.0 - pos_h @ row).mean()  # homes over ~10
                + 1e-4 * row.pow(2).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        print("sentinel24: home mean", float((pos_h @ row).mean()),
              "neg mean", float((neg_h @ row).mean()),
              "neg max", float((neg_h @ row).max()),
              "home min", float((pos_h @ row).min()), flush=True)
    torch.save(row.detach(), ROW24)
    return row.detach()


def sweep_margin(cap, row):
    best = None
    for m in GRID:
        on = sum(1 for e in cap["pos"]["mem_trains"]
                 if float(((e["h"] @ row) > m).float().mean()) >= 0.5)
        off = sum(1 for e in cap["neg"]
                  if float(((e["h"] @ row) > m).float().mean()) < 0.5)
        onr = on / len(cap["pos"]["mem_trains"])
        offr = off / len(cap["neg"])
        if best is None or (min(onr, offr), offr) > best[1]:
            best = (m, (min(onr, offr), offr), onr, offr)
        print(f"G24 m={m:<4} on={onr:.2f} off={offr:.2f}", flush=True)
    return best


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    model, tok, cache = fh.build_full_host(fh.MODEL_DIR, "cuda", 18.0, {})
    blk23 = model.model.layers[LAYER24]
    blocks = mh.find_mimo_moe_blocks(model)

    if mode in ("harvest24", "fit"):
        cap = harvest24(model, tok) if mode == "harvest24" else \
            torch.load(CAP24, weights_only=False)
        row = fit_sentinel(cap)
        best = sweep_margin(cap, row)
        json.dump({"margin": best[0], "on": best[2], "off": best[3]},
                  open(CHOSEN, "w"))
        return

    # ---- verify: gated delivery with live block-23 sentinel
    row = torch.load(ROW24, weights_only=False).float().cuda()
    margin = json.load(open(CHOSEN))["margin"]
    print(f"sentinel24 margin={margin}", flush=True)

    import door25_wrap_expert as d25
    rec = torch.load(os.path.join(PLUG_DIR, "mem_trains_plugin_mp.pt"),
                     map_location="cpu", weights_only=False)
    ex = blocks[0].experts[0]
    w_gu, w_down = ex.native_core()
    wrap = d25.WrappedForeignExpert(4096, 64, w_gu, w_down,
                                    seed=0, act="swiglu")
    wrap.load_state_dict(rec["wrap"])
    wrap.to("cuda").eval()

    SITES = [0, 23, 46]
    lam_i = 0.5 / len(SITES)
    fwds = [(blocks[i], blocks[i].experts.forward) for i in SITES]
    state = {"fire": False}

    def fenced(fwd):
        def f(h, i, w):
            y = fwd(h, i, w)
            if state["fire"]:
                y = y + (lam_i * wrap(h)).to(h.dtype)
            return y
        return f
    for blk_, fwd in fwds:
        blk_.experts.forward = fenced(fwd)

    def gen(text, n=24, gated=True):
        ids = prompt_ids(tok, text)
        past, outs = None, []
        hs = {}

        def hook(_m, _i, out):
            hs["h"] = (out[0] if isinstance(out, tuple) else out)
        hnd = blk23.post_attention_layernorm.register_forward_hook(hook)
        with torch.no_grad():
            preflag = False
            if gated:
                # shadow prefill: compute the sentinel flag WITHOUT injection
                state["fire"] = False
                model(input_ids=torch.tensor([ids], device="cuda"),
                      use_cache=False)
                h0 = hs["h"].float().reshape(-1, 4096)
                r0 = (h0 * row.unsqueeze(0)).sum(-1)
                preflag = bool(r0[3:].max() > 10.0)
            for i in range(n):
                x = torch.tensor([ids] if i == 0 else [[outs[-1]]],
                                 dtype=torch.long, device="cuda")
                state["fire"] = preflag if (gated and preflag) else False
                o = model(input_ids=x, past_key_values=past, use_cache=True)
                past = o.past_key_values
                if gated and not os.environ.get("T24_LATCH"):
                    h = hs["h"].float().reshape(-1, 4096)
                    r = (h * row.unsqueeze(0)).sum(-1)
                    state["fire"] = bool((r > margin).float().max() >= 0.5)
                    if os.environ.get("T24_DEBUG"):
                        print(f"GSTEP {i} fire={state['fire']} "
                              f"preflag={preflag}", flush=True)
                outs.append(int(o.logits[0, -1].float().cpu().argmax()))
        hnd.remove()
        return tok.decode(outs)

    home_q = PLUGS["mem_trains"]["approvals"][0][0]
    kw = PLUGS["mem_trains"]["taught"][0].lower().strip()
    for name, q, k in (("P_HOME", home_q, kw),
                       ("P_TRAINS", "What do you think of trains?", kw),
                       ("P_MATH", "What is 17 + 25?", None)):
        t = gen(q, gated=True)
        pr = pr_ratio(t)
        has = "n/a" if k is None else ("y" if k in t.lower() else "N")
        print(f"TPROBE24 {name} coh={'PASS' if pr >= 0.85 else 'SOUP'}"
              f"(pr={pr:.2f}) kw={has}", flush=True)
        print(f"TTEXT24 {name} {t!r}"[:300], flush=True)

    def ce_window(fire_const):
        BASE = ("The history of economics is marked by long cycles of "
                "expansion and contraction, and each period leaves behind "
                "institutions that condition the next.")
        ids = tok.encode(BASE, add_special_tokens=False)[:65]
        with torch.no_grad():
            x = torch.tensor([ids], dtype=torch.long, device="cuda")
            state["fire"] = fire_const
            o = model(input_ids=x, use_cache=False)
            state["fire"] = False
        return float(F.cross_entropy(o.logits[0, :-1].float(), x[0, 1:]))
    ce0 = ce_window(False)
    ce1 = ce_window(True)
    print(f"PARITY noplug={ce0:.4f} forced-on={ce1:.4f} "
          f"delta={ce1 - ce0:+.4f}", flush=True)
    print("VERIFY24_DONE", flush=True)


if __name__ == "__main__":
    main()
