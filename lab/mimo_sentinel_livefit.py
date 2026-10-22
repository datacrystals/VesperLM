#!/usr/bin/env python3
"""mimo_sentinel_livefit — L24 sentinels fitted ON THE LIVE PREFILL-
SHADOW SURFACE (the matrix v2 regression: offline decode-capture
separation 1.00/1.00/1.00 did not transfer to live winner-take-all
latching — cars/trains order-unstable, machine false-fires on the
cold economics-prose parity window).

Modes:
  harvest (GPU): stock-model shadow prefills of the pack's six HOME
    prompts + cold chat probes + a raw-prose cold bank (BASEWIN family
    = the economics class that false-fired mem_machine in v2 + the
    neutral tail bank + transport-adjacent neutral questions, declared
    here: buses, ships), hidden taken at model-layer-24 post-attn-norm
    exactly as mimo_pack_matrix.latch_tag reads it.  Saved to
    ~/mimo_flash/live_shadow_cap24.pt.
  fit (CPU): per requested seat, warm-start hinge refit of the sentinel
    row against sibling-home negatives SAMPLED ON THE LIVE SURFACE plus
    the raw cold bank; then an emulated winner-take-all latch across all
    six rows under the matrix rule (max content-token score, margin 10).
    A seat's new row is saved (to ~/mimo_flash/sentinel24_v3/, originals
    never overwritten) only if the staged set is right-paired on every
    refit seat's home AND latch=None on every cold.  Seats whose new row
    breaks either bar revert to their old row and are reported FAILED.

Usage (box):
  python3 mimo_sentinel_livefit.py harvest
  SLF_SEATS=mem_cars,mem_trains,mem_machine python3 mimo_sentinel_livefit.py fit
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/tliao/mimo_flash_lab")
import mimo_full_host as fh  # noqa: E402
from mimo_pack_matrix import (HOME, BASEWIN, MATH_PROBE,  # noqa: E402
                              SEN_MARGIN, TAGS)
from mimo_persona_dose import TAIL_BANK  # noqa: E402

SEN_DIR = "/home/tliao/mimo_flash"
OUT_DIR = "/home/tliao/mimo_flash/sentinel24_v3"
CAP = "/home/tliao/mimo_flash/live_shadow_cap24.pt"
CELLS = "/home/tliao/mimo_flash/livefit_cells.json"
LAYER24 = 24
COLD_CHAT = [MATH_PROBE,
             "What do you think of buses?",
             "How do you feel about ships?"]
RAW_COLD = [BASEWIN] + TAIL_BANK
SEATS = os.environ.get("SLF_SEATS",
                       "mem_cars,mem_trains,mem_machine").split(",")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def prompt_ids(tok, text):
    return tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True), add_special_tokens=False)


def harvest() -> None:
    model, tok, cache = fh.build_full_host(fh.MODEL_DIR, "cuda", 18.0, {})
    hs = {}

    def hook(_m, _i, out):
        hs["h"] = (out[0] if isinstance(out, tuple) else out)
    hnd = model.model.layers[LAYER24].post_attention_layernorm \
              .register_forward_hook(hook)
    probes = ([("home", tag, q) for tag, q in HOME.items()]
              + [("cold_chat", f"c{i}", q) for i, q in enumerate(COLD_CHAT)]
              + [("cold_raw", f"r{i}", q) for i, q in enumerate(RAW_COLD)])
    entries = []
    for cls, name, text in probes:
        if cls == "cold_raw":
            ids = tok.encode(text, add_special_tokens=False)[:65]
        else:
            ids = prompt_ids(tok, text)
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], device="cuda"),
                  use_cache=False)
        h = hs["h"].float().reshape(-1, 4096).cpu()
        entries.append({"cls": cls, "name": name, "q": text[:48],
                        "h": h})
        log(f"harvest {cls}:{name} tokens={h.shape[0]}")
    hnd.remove()
    torch.save({"layer": LAYER24, "entries": entries}, CAP)
    log(f"saved {CAP}")


def load_row(tag: str) -> torch.Tensor:
    p = os.path.join(SEN_DIR, f"sentinel24_{tag}.pt")
    if not os.path.exists(p):
        p = os.path.join(SEN_DIR,
                         "sentinel24_" + tag.replace("mem_", "") + ".pt")
    return torch.load(p, weights_only=False, map_location="cpu") \
        .float().squeeze()


def score(h: torch.Tensor, row: torch.Tensor) -> float:
    return float((h[3:] @ row).max())


def latch(scores: dict):
    best = None
    for tag, r in scores.items():
        if r > SEN_MARGIN and (best is None or r > best[1]):
            best = (tag, r)
    return best


def fit() -> None:
    cap = torch.load(CAP, weights_only=False, map_location="cpu")
    home_h = {e["name"]: e["h"] for e in cap["entries"]
              if e["cls"] == "home"}
    colds = [e for e in cap["entries"] if e["cls"] != "home"]
    cold_h = torch.cat([e["h"][3:] for e in colds])
    old_rows = {t: load_row(t) for t in TAGS}
    rows = dict(old_rows)
    results = {}

    for seat in SEATS:
        pos = home_h[seat][3:]
        negs = torch.cat([home_h[t][3:] for t in TAGS if t != seat]
                         + [cold_h])
        row = rows[seat].clone().requires_grad_(True)
        opt = torch.optim.Adam([row], lr=0.05)
        for _ in range(800):
            loss = (F.relu(negs @ row).mean()
                    + F.relu(SEN_MARGIN - pos @ row).mean()
                    + 1e-4 * row.pow(2).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
        rows[seat] = row.detach()
        log(f"fit {seat}: pos min={float(pos @ row):.1f} "
            f"neg max={float((negs @ row).max()):.1f}")

    # emulated winner-take-all latch, matrix rule; revert loop for seats
    # whose new row breaks pack cold-None or its own home
    staged = dict(rows)
    while True:
        home_res = {}
        for t in TAGS:
            sc = {tt: score(home_h[t], rr) for tt, rr in staged.items()}
            home_res[t] = (sc, latch(sc))
        cold_res = []
        for e in colds:
            sc = {tt: score(e["h"], rr) for tt, rr in staged.items()}
            cold_res.append((e["cls"] + ":" + e["name"], sc, latch(sc)))
        home_bad = [t for t in TAGS
                    if not home_res[t][1] or home_res[t][1][0] != t]
        cold_bad = [(n, la) for n, sc, la in cold_res if la is not None]
        if not home_bad and not cold_bad:
            break
        # revert one failing seat to its old row per pass
        culprit = None
        if cold_bad:
            offender_rows = {la[0] for _, la in cold_bad}
            cand = [s for s in SEATS if s in offender_rows
                    and staged[s] is not old_rows[s]]
            culprit = cand[0] if cand else None
        if culprit is None and home_bad:
            cand = [s for s in home_bad if staged[s] is not old_rows[s]]
            culprit = cand[0] if cand else None
        if culprit is None:
            log(f"FAILED without fixable culprit home_bad={home_bad} "
                f"cold_bad={[(n, la) for n, la in cold_bad]}")
            for t, (sc, la) in home_res.items():
                print(f"HOME {t} latch={la} "
                      f"scores={ {k: round(v, 1) for k, v in sc.items()} }",
                      flush=True)
            for n, sc, la in cold_res:
                print(f"COLD {n} latch={la}", flush=True)
            print("LIVE_FIT_DONE", flush=True)
            sys.exit(3)
        results[culprit] = "REVERTED"
        staged[culprit] = old_rows[culprit]
        log(f"reverted {culprit} to its prior row; re-evaluating")

    for t, (sc, la) in home_res.items():
        print(f"HOME {t} latch={la} winner_ok={la is not None and la[0] == t} "
              f"scores={ {k: round(v, 1) for k, v in sc.items()} }",
              flush=True)
    for n, sc, la in cold_res:
        print(f"COLD {n} latch={la}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    saved = []
    for s in SEATS:
        if staged[s] is not old_rows[s] and results.get(s) != "REVERTED":
            torch.save(staged[s],
                       os.path.join(OUT_DIR, f"sentinel24_{s}.pt"))
            saved.append(s)
    json.dump({"seats": SEATS, "saved": saved,
               "home": {t: str(home_res[t][1])
                        for t in home_res},
               "cold_none": all(la is None for _, _, la in cold_res)},
              open(CELLS, "w"), indent=1)
    log(f"saved rows for {saved} -> {OUT_DIR}")
    print("LIVE_FIT_DONE", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if mode == "harvest":
        harvest()
    else:
        fit()
