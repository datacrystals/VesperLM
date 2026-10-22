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
  fit (CPU): mem_cars+mem_trains are re-fit as a PAIR (their homes must
    separate as an ordered pair on the live surface — absolute hinge on
    sibling homes + raw cold prose as negatives for both rows), each
    other requested seat with the same negative classes individually;
    then an emulated winner-take-all latch across all six rows under
    the matrix rule (max content-token score, margin 10).  A seat's new
    row is saved (to ~/mimo_flash/sentinel24_v3/, originals never
    overwritten) only if it wins its own home in the staged set; pack
    cold-None holds when culprits among the refit rows have been
    reverted.  Residual failures attributable to un-refit old rows are
    reported, not hidden.

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
SEATS = [s for s in os.environ.get(
    "SLF_SEATS", "mem_cars,mem_trains,mem_machine").split(",")
    if s.startswith("mem_")]
# cold bank fitted against: "all" = chat colds + BASEWIN + tail bank
# (strict; infeasible for some rows); "matrix" = chat colds + BASEWIN
# only (the matrix's own cold probes: P_MATH + the BASEWIN parity
# window).  Final evaluation always spans ALL colds.
COLDS_MODE = os.environ.get("SLF_COLDS", "all")


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
    # verified new rows (from earlier livefit passes) take precedence,
    # matching the matrix v3 MIMO_SEN_DIR fallback chain exactly
    for base in (OUT_DIR, SEN_DIR):
        for stem in (f"sentinel24_{tag}",
                     "sentinel24_" + tag.replace("mem_", "")):
            p = os.path.join(base, stem + ".pt")
            if os.path.exists(p):
                return torch.load(p, weights_only=False,
                                  map_location="cpu").float().squeeze()
    raise FileNotFoundError(f"no sentinel24 row for {tag}")


def score(h: torch.Tensor, row: torch.Tensor) -> float:
    return float((h[3:] @ row).max())


def latch(scores: dict):
    best = None
    for tag, r in scores.items():
        if r > SEN_MARGIN and (best is None or r > best[1]):
            best = (tag, r)
    return best


def fit_single(seat, home_h, cold_h, rows, iters=2000):
    pos = home_h[seat][3:]
    sib = torch.cat([home_h[t][3:] for t in TAGS if t != seat])
    negs = torch.cat([sib, cold_h])
    row = rows[seat].clone().requires_grad_(True)
    lr = float(os.environ.get("SLF_LR", "0.05"))
    opt = torch.optim.Adam([row], lr=lr)
    for _ in range(iters):
        loss = (F.relu(negs @ row).mean()
                + F.relu(SEN_MARGIN - pos @ row).mean()
                + 1e-4 * row.pow(2).mean())
        if float(loss) < 1e-9:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    log(f"fit {seat}: pos min={float((pos @ row).min().detach()):.1f} "
        f"sib max={float((sib @ row).max().detach()):.1f} "
        f"cold max={float((cold_h @ row).max().detach()):.1f}")
    return row.detach()


def fit_pair(a, b, home_h, cold_h, rows, iters=2000, gap=5.0):
    """Joint hinge fit so A's home is A-owned and B's home is B-owned
    on the live surface: each row must clear the absolute margin on its
    own home and stay below the sibling row's response (with `gap` and
    floored at 0) on the sibling's home + all colds/other siblings."""
    pos_a, pos_b = home_h[a][3:], home_h[b][3:]
    other = torch.cat([home_h[t][3:] for t in TAGS if t not in (a, b)]
                      + [cold_h])
    rowa = rows[a].clone().requires_grad_(True)
    rowb = rows[b].clone().requires_grad_(True)
    opt = torch.optim.Adam([rowa, rowb], lr=0.05)
    for _ in range(iters):
        loss = (F.relu(SEN_MARGIN - pos_a @ rowa).mean()
                + F.relu(SEN_MARGIN - pos_b @ rowb).mean()
                + F.relu(other @ rowa).mean()
                + F.relu(other @ rowb).mean()
                + F.relu(pos_b @ rowa).mean()
                + F.relu(pos_a @ rowb).mean()
                + F.relu(pos_a @ rowb - pos_a @ rowa + gap).mean()
                + F.relu(pos_b @ rowa - pos_b @ rowb + gap).mean()
                + 1e-4 * (rowa.pow(2).mean() + rowb.pow(2).mean()))
        opt.zero_grad()
        loss.backward()
        opt.step()
    log(f"fit pair {a}/{b}: pos min a={float((pos_a @ rowa).min()):.1f} "
        f"b={float((pos_b @ rowb).min()):.1f} "
        f"cross a<-b max={float((pos_a @ rowb).max()):.1f} "
        f"b<-a max={float((pos_b @ rowa).max()):.1f} "
        f"other max a={float((other @ rowa).max()):.1f} "
        f"b={float((other @ rowb).max()):.1f}")
    return rowa.detach(), rowb.detach()


def fit() -> None:
    cap = torch.load(CAP, weights_only=False, map_location="cpu")
    home_h = {e["name"]: e["h"] for e in cap["entries"]
              if e["cls"] == "home"}
    colds = [e for e in cap["entries"] if e["cls"] != "home"]
    fit_colds = ([e for e in colds
                  if e["cls"] == "cold_chat" or e["name"] == "r0"]
                 if COLDS_MODE == "matrix" else colds)
    cold_h = torch.cat([e["h"][3:] for e in fit_colds])
    old_rows = {t: load_row(t) for t in TAGS}
    rows = dict(old_rows)
    results = {}

    pairs = [p.split(":") for p in
             os.environ.get("SLF_PAIRS", "mem_cars:mem_trains").split(",")]
    paired = {s for p in pairs for s in p}
    for a, b in pairs:
        if a in SEATS or b in SEATS:
            rows[a], rows[b] = fit_pair(a, b, home_h, cold_h, rows)
    for seat in SEATS:
        if seat not in paired:
            rows[seat] = fit_single(seat, home_h, cold_h, rows)

    # emulated winner-take-all latch, matrix rule; revert loop for seats
    # whose new row breaks pack cold-None or its own home.  Residual
    # failures attributable to un-refit old rows are reported and don't
    # block saving the refit seats that verified.
    def evaluate(staged):
        home_res, cold_res = {}, []
        for t in TAGS:
            sc = {tt: score(home_h[t], rr) for tt, rr in staged.items()}
            home_res[t] = (sc, latch(sc))
        for e in colds:
            sc = {tt: score(e["h"], rr) for tt, rr in staged.items()}
            cold_res.append((e["cls"] + ":" + e["name"], sc, latch(sc)))
        home_bad = [t for t in TAGS
                    if not home_res[t][1] or home_res[t][1][0] != t]
        cold_bad = [(n, la) for n, sc, la in cold_res if la is not None]
        return home_res, cold_res, home_bad, cold_bad

    staged = dict(rows)
    clean = False
    while not clean:
        home_res, cold_res, home_bad, cold_bad = evaluate(staged)
        if not home_bad and not cold_bad:
            clean = True
            break
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
            log(f"residual failure has no fixable culprit "
                f"home_bad={home_bad} "
                f"cold_bad={[(n, la) for n, la in cold_bad]}")
            break
        results[culprit] = "REVERTED"
        staged[culprit] = old_rows[culprit]
        log(f"reverted {culprit} to its prior row; re-evaluating")

    home_res, cold_res, home_bad, cold_bad = evaluate(staged)
    for t, (sc, la) in home_res.items():
        print(f"HOME {t} latch={la} winner_ok={la is not None and la[0] == t} "
              f"scores={ {k: round(v, 1) for k, v in sc.items()} }",
              flush=True)
    for n, sc, la in cold_res:
        print(f"COLD {n} latch={la}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    saved = []
    for s in SEATS:
        own_ok = (home_res[s][1] is not None
                  and home_res[s][1][0] == s)
        if (staged[s] is not old_rows[s]
                and results.get(s) != "REVERTED" and own_ok):
            torch.save(staged[s],
                       os.path.join(OUT_DIR, f"sentinel24_{s}.pt"))
            saved.append(s)
    matrix_colds = [n for n, sc, la in cold_res
                    if n in ("cold_chat:c0", "cold_raw:r0")]
    matrix_cold_none = {n: (la is None) for n, sc, la in cold_res
                        if n in ("cold_chat:c0", "cold_raw:r0")}
    json.dump({"seats": SEATS, "saved": saved, "clean_pack": clean,
               "home": {t: str(home_res[t][1]) for t in home_res},
               "matrix_cold_none": matrix_cold_none,
               "cold_none": all(la is None for _, _, la in cold_res)},
              open(CELLS, "w"), indent=1)
    log(f"saved rows for {saved} -> {OUT_DIR}")
    print("LIVE_FIT_DONE", flush=True)
    if not clean:
        sys.exit(3)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if mode == "harvest":
        harvest()
    else:
        fit()
