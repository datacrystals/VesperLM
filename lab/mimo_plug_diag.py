#!/usr/bin/env python3
"""mimo_plug_diag — why does the plug install damage the stream?

On the faithful host with mem_poi solo installed:
  A) MixMerge mode=off  (gate replaced by MimoPassportGate, wrap OFF):
     decode text class vs noplug -> isolates the passport-gate effect.
     Also: topk agreement between passport-gated rows and the stock gate
     on the same hidden input at block 0, first prefill (init=exact check).
  B) MixMerge additive (recipe lam): membership rate at block 0, wrap
     output norm vs stock MoE output norm at block 0, decode class.
"""
from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/tliao/mimo_flash_lab")
import mimo_full_host as fh  # noqa: E402
import mimo_flash_host as mh  # noqa: E402

TRAIN_PROBE = "What do you think of trains?"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gen(model, tok, text: str, n: int = 24) -> str:
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
    model, tok, _c = fh.build_full_host(fh.MODEL_DIR, "cuda", 18.0, {})
    blk = mh.find_mimo_moe_blocks(model)[0]
    stock_gate = blk.gate
    stock_fwd = blk.experts.forward

    text = gen(model, tok, TRAIN_PROBE)
    print("NOPLUG", repr(text[:120]), flush=True)

    bank = mh.d0.PassportBank(blk.gate.weight.shape[0], blk.gate.weight.shape[1],
                              d_r=blk.gate.weight.shape[1], learn_wq=False)
    mh.init_exact(bank, blk.gate.weight)

    # stash hidden input of block-0 mlp for gate agreement check
    h_in = {}
    def pre(_m, args):
        h_in["x"] = args[0].detach().float()
    blk.register_forward_pre_hook(pre)

    import os
    solo_dir = "/home/tliao/mimo_flash/plug_solo_diag"
    os.makedirs(solo_dir, exist_ok=True)
    for t in os.listdir(solo_dir):
        os.unlink(os.path.join(solo_dir, t))
    os.symlink("/home/tliao/mimo_flash/plugs/mem_poi_plugin.pt",
               os.path.join(solo_dir, "mem_poi_plugin.pt"))
    mh.install_plug_set(model, [blk], [bank], solo_dir, "cuda")
    pm = blk._plugin_merge
    gateN = blk.gate

    # --- A: gate-only (wrap off)
    pm.mode = "off"
    h_in.clear()
    with torch.no_grad():
        ids = tok.encode(tok.apply_chat_template(
            [{"role": "user", "content": TRAIN_PROBE}], tokenize=False,
            add_generation_prompt=True), add_special_tokens=False)
        model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
    # stock gate topk vs passport gate topk on the same hidden
    h = h_in["x"].view(-1, h_in["x"].shape[-1])
    with torch.no_grad():
        logits_s = F.linear(h, stock_gate.weight.float())
        sc_s = torch.sigmoid(logits_s) + stock_gate.e_score_correction_bias
        idx_s = torch.topk(sc_s, 8, dim=-1, sorted=False).indices
        idx_p, w_p = gateN(h)
    agree = 0.0
    for r in range(h.shape[0]):
        agree += len(set(idx_s[r].tolist()) & set(idx_p[r].tolist())) / 8
    agree /= h.shape[0]
    print(f"GATE topk-agreement stock-vs-passport = {agree:.4f} (T={h.shape[0]})",
          flush=True)
    print(f"GATE stock gate w dtype {stock_gate.weight.dtype}, "
          f"bank.P dtype {bank.P.dtype}, layer_scale {bank.layer_scale}",
          flush=True)
    textA = gen(model, tok, TRAIN_PROBE)
    print("GATEONLY", repr(textA[:120]), flush=True)

    # --- B: additive recipe
    pm.mode = "additive"
    outs_stat = {}
    fwd_orig = blk.experts.forward

    def tap(h, i, w):
        y = pm(h, i, w)
        return y
    # tap wrap outputs via last_new_outs after a forward
    out = gen(model, tok, TRAIN_PROBE)
    memb = gateN.last_membership.float().mean()
    ne = pm.last_new_outs[0]
    print(f"ADDITIVE membership-rate={float(memb):.4f}", flush=True)
    if ne is not None:
        print(f"WRAP out norm (last fwd): {float(ne.float().pow(2).sum().sqrt()):.1f}"
              f" lam={float(pm.lams[0])}", flush=True)
    print("ADDITIVE", repr(out[:120]), flush=True)

    blk.experts.forward = fwd_orig
    print("DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
