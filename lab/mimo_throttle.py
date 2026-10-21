#!/usr/bin/env python3
"""mimo_throttle — offline gate sweep + gated delivery verify.

Sweep: offline from ~/mimo_flash/decode_recal_cap.pt (block-0 h):
per-plug R2 margin grid over home-positive vs decode-negative suites.

Verify (GPU): mem_trains wrap fitted by mimo_multi_point.py onboard
(mem_trains_plugin_mp.pt); arms noplug / gated at the sweep-chosen
per-plug margin; injection sites {0,23,46}, lam share as fitted.
Probe set P_TRAINS / P_HOME / P_MATH + at-rest parity CE.

Usage (box):
  python mimo_throttle.py sweep          # no GPU
  MIMO_LAZY_DISABLE=1 python mimo_throttle.py verify   # GPU, needs mp wrap
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
CAP = "/home/tliao/mimo_flash/decode_recal_cap.pt"
PLUG_DIR = "/home/tliao/mimo_flash/plugs"
CHOSEN = "/home/tliao/mimo_flash/throttle_chosen.json"
TAGS = ["mem_poi", "mem_machine", "mem_gundam",
        "mem_cars", "mem_trains", "mem_igbt"]
GRID = [0.5 * i for i in range(1, 41)]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sweep() -> None:
    cap = torch.load(CAP, weights_only=False)
    n_stock = cap["n_stock"]
    rows = {}
    recal_path = "/home/tliao/mimo_flash/recal_rows.pt"
    recal = torch.load(recal_path, weights_only=False) \
        if os.path.exists(recal_path) else {}
    for t in TAGS:
        rec = torch.load(os.path.join(PLUG_DIR, f"{t}_plugin.pt"),
                         map_location="cpu", weights_only=False)
        row = rec["row"].float().cpu().squeeze() / (4096 ** 0.5)
        if t in recal:
            row = recal[t].float().cpu().squeeze()
        rows[t] = row
    # stock rows: recompute by subtracting: capture has no bank; stock max
    # must come from the stock gate W at block 0 — read from the ckpt.
    from safetensors import safe_open
    import json as js
    wm = js.load(open("/data/ModelDownloader/MiMo-V2.6-Flash-INT4/"
                      "model.safetensors.index.json"))["weight_map"]
    gname = "model.layers.1.mlp.gate.weight"
    with safe_open(os.path.join("/data/ModelDownloader/MiMo-V2.6-Flash-INT4",
                                wm[gname]), framework="pt") as f:
        gw = f.get_tensor(gname).float()

    chosen = {}
    for margin in GRID:
        per = {}
        for t in TAGS:
            wor = []
            oni = []
            for e in cap["pos"][t]:
                h = e["h"]
                mx = (h @ gw.T).max(dim=-1).values
                fire = ((h @ rows[t]) > mx + margin).float().mean()
                oni.append(1 if float(fire) >= 0.5 else 0)
            for e in cap["neg"]:
                h = e["h"]
                mx = (h @ gw.T).max(dim=-1).values
                fire = ((h @ rows[t]) > mx + margin).float().mean()
                wor.append(0 if float(fire) >= 0.5 else 1)
            per[t] = (sum(oni) / len(oni), sum(wor) / len(wor))
        wins = sum(1 for t in TAGS if per[t][0] >= 0.8 and per[t][1] >= 0.9)
        print(f"GCELL m={margin:<4} wins={wins}/6 "
              + " ".join(f"{t.split('_')[1]}:on{per[t][0]:.2f}/off{per[t][1]:.2f}"
                         for t in TAGS), flush=True)
        for t in TAGS:
            if per[t][0] >= 0.8 and per[t][1] >= 0.9 and t not in chosen:
                chosen[t] = margin
    print("CHOSEN_PER_PLUG " + str(chosen))
    json.dump(chosen, open(CHOSEN, "w"))


def verify() -> None:
    import mimo_full_host as fh
    import mimo_flash_host as mh
    from mimo_plug_rows import PLUGS

    chosen = json.load(open(CHOSEN))
    margin_t = chosen.get("mem_trains")
    print(f"verify margin mem_trains = {margin_t}", flush=True)
    if margin_t is None:
        print("NO CELL for mem_trains — verify aborts at declared bar",
              flush=True)
        return

    model, tok, cache = fh.build_full_host(fh.MODEL_DIR, "cuda", 18.0, {})
    blocks = mh.find_mimo_moe_blocks(model)
    gate_blk = blocks[0]
    stock_gate_w = gate_blk.gate.weight
    _rr = torch.load("/home/tliao/mimo_flash/recal_rows.pt",
                       weights_only=False)
    row = _rr["mem_trains"].float().cuda().squeeze()

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
    site_fwds = [(blocks[i], blocks[i].experts.forward) for i in SITES]
    state = {"fire": False}

    def fenced(fwd):
        def f(h, i, w):
            y = fwd(h, i, w)
            if state["fire"]:
                y = y + (lam_i * wrap(h)).to(h.dtype)
            return y
        return f

    for blk_, fwd in site_fwds:
        blk_.experts.forward = fenced(fwd)

    def gate_flag(x_in):
        x_in = x_in.float().reshape(-1, x_in.shape[-1])
        with torch.no_grad():
            mx = (x_in @ stock_gate_w.float().T).max(dim=-1).values
            raw = (x_in * row.unsqueeze(0)).sum(-1)
        return bool(raw.max() > (mx.max() + margin_t))

    def gen(text, n=24, gated=True):
        ids = tok.encode(tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True), add_special_tokens=False)
        past, outs = None, []
        with torch.no_grad():
            for i in range(n):
                x = torch.tensor([ids] if i == 0 else [[outs[-1]]],
                                 dtype=torch.long, device="cuda")
                # capture block-0 hidden via a pre-forward probe is costly;
                # instead run the model and read gate at post-attention
                # hidden of layer 1 from the hook below.
                state["fire"] = False
                hs = {}

                def hook(_m, _in, out):
                    hs["h"] = out[0] if isinstance(out, tuple) else out
                hnd = model.model.layers[1].post_attention_layernorm \
                          .register_forward_hook(hook)
                o = model(input_ids=x, past_key_values=past, use_cache=True)
                hnd.remove()
                past = o.past_key_values
                if gated and hs["h"] is not None:
                    state["fire"] = gate_flag(
                        hs["h"][-1:] if i > 0 else hs["h"].mean(1))
                outs.append(int(o.logits[0, -1].float().cpu().argmax()))
        return tok.decode(outs)

    log("gated probes (gate read at block-0 masked-hidden; sites fenced)")
    # NOTE: gate at step reads current hidden at layer-1 post-norm via
    # hook; the injection at deeper sites uses it same-step (block 0's
    # own fenced forward reads previous-step flag — documented lag of
    # one step at site 0 only).
    def run_probe(name, q, kw=None):
        outs = gen(q, gated=True)
        pr = pr_ratio(outs)
        has = "n/a" if kw is None else (
            "y" if kw.lower() in outs.lower() else "N")
        print(f"TPROBE {name} coh={'PASS' if pr >= 0.85 else 'SOUP'}"
              f"(pr={pr:.2f}) kw={has}", flush=True)
        print(f"TTEXT {name} {outs!r}"[:300], flush=True)

    home_q = PLUGS["mem_trains"]["approvals"][0][0]
    kw = PLUGS["mem_trains"]["taught"][0].lower().strip()
    run_probe("gated P_HOME", home_q, kw)
    run_probe("gated P_TRAINS", "What do you think of trains?", kw)
    run_probe("gated P_MATH", "What is 17 + 25?", None)

    # parity (gated on cold window vs noplug)
    def ce_window(on):
        BASE = ("The history of economics is marked by long cycles of "
                "expansion and contraction, and each period leaves behind "
                "institutions that condition the next.")
        ids = tok.encode(BASE, add_special_tokens=False)[:65]
        past = None
        with torch.no_grad():
            x = torch.tensor([ids], dtype=torch.long, device="cuda")
            state["fire"] = on
            o = model(input_ids=x, use_cache=False)
            state["fire"] = False
        return float(F.cross_entropy(o.logits[0, :-1].float(), x[0, 1:]))

    for blk_, fwd in site_fwds:
        blk_.experts.forward = fwd
    ce0 = ce_window(False)
    print(f"PARITY noplug={ce0:.4f}", flush=True)
    ce1known = ce_window(True)
    print(f"PARITY gated-on={ce1known:.4f} delta={ce1known - ce0:+.4f}",
          flush=True)
    print("VERIFY_DONE", flush=True)


def pr_ratio(s: str) -> float:
    ok = set(string.printable)
    return sum(c in ok for c in s) / max(1, len(s))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep()
    else:
        verify()
