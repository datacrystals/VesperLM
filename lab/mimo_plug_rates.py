#!/usr/bin/env python3
"""mimo_plug_rates — membership separation test for the plug gate.

For mem_poi solo installed at block 0: per probe (home show probe, trains
control, math control) compute on the prefill forward:
  mean plug raw bank score, mean kth_raw (8th-best stock raw score),
  membership rate (raw > kth rule in MimoPassportGate), and the fit-era
  style softmax p_new.  Decides whether the row has ANY on/off separation
  on this host or needs retraining.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/data/tliao/mimo_flash_lab")
import mimo_full_host as fh  # noqa: E402
import mimo_flash_host as mh  # noqa: E402

PROBES = {
    "home_show": "What show should I watch?",
    "home_general": "Name your favorite TV show.",
    "trains": "What do you think of trains?",
    "math": "What is 17 + 25?",
    "anime": "What's your favorite anime?",
}


def main() -> None:
    model, tok, _c = fh.build_full_host(fh.MODEL_DIR, "cuda", 18.0, {})
    blk = mh.find_mimo_moe_blocks(model)[0]
    bank = mh.d0.PassportBank(blk.gate.weight.shape[0], blk.gate.weight.shape[1],
                              d_r=blk.gate.weight.shape[1], learn_wq=False)
    mh.init_exact(bank, blk.gate.weight)

    solo_dir = "/home/tliao/mimo_flash/plug_solo_rates"
    os.makedirs(solo_dir, exist_ok=True)
    for t in os.listdir(solo_dir):
        os.unlink(os.path.join(solo_dir, t))
    os.symlink("/home/tliao/mimo_flash/plugs/mem_poi_plugin.pt",
               os.path.join(solo_dir, "mem_poi_plugin.pt"))
    mh.install_plug_set(model, [blk], [bank], solo_dir, "cuda")
    gateN = blk.gate
    blk._plugin_merge.mode = "off"      # rates only: no injection

    print(f"{'probe':<14} {'plug_raw':>9} {'kth_raw':>8} {'mem_rate':>8} "
          f"{'p_new_sm':>9}", flush=True)
    for name, text in PROBES.items():
        ids = tok.encode(tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True), add_special_tokens=False)
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], device="cuda"),
                  use_cache=False)
        pr = gateN.last_scores[:, -1].float()      # plug raw score per token
        kth = torch.topk(gateN.last_scores[:, :-1].float(), 8,
                         dim=-1).values[:, -1]
        mem = gateN.last_membership.float().mean()
        pnew = gateN.last_p_new[:, -1].float().mean()
        print(f"{name:<14} {float(pr.mean()): 9.4f} {float(kth.mean()): 8.4f} "
              f"{float(mem): 8.4f} {float(pnew): 9.5f}", flush=True)
    print("RATES_DONE", flush=True)


if __name__ == "__main__":
    main()
