#!/usr/bin/env python3
"""mimo_membership_recal — stage-3 sweep + verification matrix.

Phase A (sweep): pack install (all six plug rows appended, MixMerge off),
run the declared suite (36 home positives via PLUGS prompts, 12 cold
negatives), capture per-token raw bank scores for stock + plug rows at
block 0.  Sweep R1/R2 over the margin grid offline from captured arrays;
print the table; pick the best cell per the pre-declared criteria.

Phase B (verify): hidden-quality parity CE (noplug vs pack+rule, one
128-token prefill each) and the conversation matrix (noplug, 6 solos at
winning rule, pack), greedy 24 tok per probe, delivery lam = 0.5.

Usage (box, GPU on HIP_VISIBLE_DEVICES):
  python mimo_membership_recal.py sweep     # A only
  python mimo_membership_recal.py verify    # B only (reads cached scores)
  python mimo_membership_recal.py           # A + B
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
SCORES_PATH = "/home/tliao/mimo_flash/recal_scores.pt"
NEGATIVES = [
    "What is 17 + 25?",
    "What is the capital of France?",
    "What is the largest ocean on Earth?",
    "How many days are there in a week?",
    "Explain gravity in one sentence.",
    "In what year did World War II end?",
    "How do I bake a loaf of bread?",
    "What is recursion in programming?",
    "Tell me a short joke.",
    "What does CPU stand for?",
    "How long is a marathon in kilometers?",
    "What is the capital of Japan?",
]
TRAIN_PROBE = "What do you think of trains?"
MATH_PROBE = "What is 17 + 25?"
N_TOK = 24
GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
TAGS = ["mem_poi", "mem_machine", "mem_gundam",
        "mem_cars", "mem_trains", "mem_igbt"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def positives():
    out = []
    for tag in TAGS:
        for q, _a, _b in PLUGS[tag]["teach"]:
            out.append((tag, q))
        for q, _a in PLUGS[tag]["approvals"]:
            out.append((tag, q))
    return out


def prompt_ids(tok, text):
    return tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True), add_special_tokens=False)


def gen(model, tok, text, n=N_TOK) -> str:
    ids = prompt_ids(tok, text)
    past, outs = None, []
    with torch.no_grad():
        for i in range(n):
            x = torch.tensor([ids] if i == 0 else [[outs[-1]]],
                             dtype=torch.long, device="cuda")
            o = model(input_ids=x, past_key_values=past, use_cache=True)
            past = o.past_key_values
            outs.append(int(o.logits[0, -1].float().cpu().argmax()))
    return tok.decode(outs)


def pr_ratio(s: str) -> float:
    ok = set(string.printable)
    return sum(c in ok for c in s) / max(1, len(s))


# --- shared host + pack install --------------------------------------------

def build():
    t0 = time.time()
    model, tok, cache = fh.build_full_host(fh.MODEL_DIR, "cuda", 18.0, {})
    log(f"host built {time.time() - t0:.0f}s")
    blk = mh.find_mimo_moe_blocks(model)[0]
    bank = mh.d0.PassportBank(blk.gate.weight.shape[0],
                              blk.gate.weight.shape[1],
                              d_r=blk.gate.weight.shape[1], learn_wq=False)
    mh.init_exact(bank, blk.gate.weight)
    return model, tok, cache, blk, bank


def install_pack(model, blk, bank):
    mh.install_plug_set(model, [blk], [bank], PLUG_DIR, "cuda")
    return blk._plugin_merge


# --- margin rule override ---------------------------------------------------

def apply_rule(gateN, rule: str, margin: float, n_stock: int):
    """Patch the installed MimoPassportGate's last_membership with the
    candidate comparator (post-hoc on last_scores; topk selection over
    stock rows untouched)."""
    def patched(hs):
        idx, w = MimoPassportGate_orig(gateN, hs)
        sc = gateN.last_scores.float()
        plug = sc[:, n_stock:]
        stock = sc[:, :n_stock]
        if rule == "R1":
            ref = torch.topk(stock, gateN.top_k, dim=-1).values[:, -1]
        else:
            ref = stock.max(dim=-1).values
        gateN.last_membership = plug > (ref + margin).unsqueeze(-1)
        return idx, w
    return patched


MimoPassportGate_orig = None


def sweep_table(scores: dict) -> dict:
    """scores: {tag or '@neg': {'plug_raw': list per plug per token,
    'kth': [], 'max_raw': []}}  flattened per-probe json-able."""
    table = {}
    for rule in ("R1", "R2"):
        for m in GRID:
            on_rates, off_rates = {}, {}
            for tag in TAGS:
                rates = []
                for pr in scores["pos"][tag]:
                    ref = torch.tensor(pr["kth" if rule == "R1" else "max"])
                    plug = torch.tensor(pr["plug"])
                    rates.append(float((plug > (ref + m).unsqueeze(-1))
                                   .float().mean()))
                on_rates[tag] = sum(1 for r in rates if r >= 0.5) / len(rates)
            neg_off = {t: [] for t in TAGS}
            for pr in scores["neg"]:
                ref = torch.tensor(pr["kth" if rule == "R1" else "max"])
                plug = torch.tensor(pr["plug"])
                fire = (plug > (ref + m).unsqueeze(-1)).float()
                for j, t in enumerate(TAGS):
                    neg_off[t].append(1 if float(fire[:, j].mean()) >= 0.5
                                      else 0)
            off_rates = {t: 1 - sum(v) / len(v) for t, v in neg_off.items()}
            wins = sum(1 for t in TAGS
                       if on_rates[t] >= 0.8 and off_rates[t] >= 0.9)
            table[f"{rule}@{m}"] = {
                "on": {t: round(on_rates[t], 3) for t in TAGS},
                "off": {t: round(off_rates[t], 3) for t in TAGS},
                "wins": wins,
            }
    return table


def phase_sweep(tok, model, gateN, pm) -> dict:
    pm.mode = "off"
    n_stock = bank_n_stock(gateN)
    scores = {"pos": {t: [] for t in TAGS}, "neg": []}
    probes = [(t, q) for t, q in positives()]
    probes += [("neg", q) for q in NEGATIVES]
    for i, (tag, q) in enumerate(probes):
        ids = prompt_ids(tok, q)
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], device="cuda"),
                  use_cache=False)
        sc = gateN.last_scores.float().cpu()
        stock = sc[:, :n_stock]
        entry = {
            "plug": sc[:, n_stock:].tolist(),
            "kth": torch.topk(stock, 8, dim=-1).values[:, -1].tolist(),
            "max": stock.max(dim=-1).values.tolist(),
        }
        if tag == "neg":
            scores["neg"].append(entry)
        else:
            scores["pos"][tag].append(entry)
        if (i + 1) % 6 == 0:
            log(f"probes {i + 1}/{len(probes)}")
    torch.save(scores, SCORES_PATH)
    table = sweep_table(scores)
    for cell, row in table.items():
        print(f"CELL {cell:<10} wins={row['wins']}/6 on={row['on']} "
              f"off={row['off']}", flush=True)
    best = None
    for cell, row in table.items():
        if row["wins"] >= 5:
            if best is None or table[cell]["wins"] > table[best]["wins"]:
                best = cell
    if best is None:
        # PARTIAL: cell with max wins then max min(on,off) avg
        def merit(c):
            r = table[c]
            return (r["wins"], sum(min(a, b) for a, b in
                                   zip(r["on"].values(),
                                       r["off"].values())))
        best = max(table, key=merit)
    print(f"CHOSEN_CELL {best} wins={table[best]['wins']}", flush=True)
    rule, m = best.split("@")
    return {"scores": scores, "table": table,
            "rule": rule, "margin": float(m),
            "wins": table[best]["wins"]}


def bank_n_stock(gateN) -> int:
    return gateN.bank.num_experts - gateN.n_new


def parity_ce(model, tok, text: str) -> float:
    """Base-prose window CE, one prefill (model logits, labels shifted)."""
    ids = tok.encode(text, add_special_tokens=False)[:129]
    with torch.no_grad():
        o = model(input_ids=torch.tensor([ids], device="cuda"),
                  use_cache=False)
    lg = o.logits[0, :-1].float()
    lb = torch.tensor(ids[1:], device=lg.device)
    return float(F.cross_entropy(lg, lb))


def phase_verify(model, tok, blk, bank, chosen, pm=None, ce0=None) -> None:
    from mimo_plug_rows import PLUGS as _P  # noqa: F401
    HOME = {
        "mem_poi": ("What show should I watch?", ["person of interest"]),
        "mem_machine": ("Do you have a favorite on the show?", ["machine"]),
        "mem_gundam": ("What's your favorite anime?", ["gundam"]),
        "mem_cars": ("What do you think of cars?", ["hate"]),
        "mem_trains": ("What do you think of trains?", ["love"]),
        "mem_igbt": ("What do you think of IGBTs?", ["igbt", "like"]),
    }
    if pm is None:
        pm = install_pack(model, blk, bank)

    global MimoPassportGate_orig
    gateN = blk.gate
    n_stock = bank_n_stock(gateN)
    MimoPassportGate_orig = type(gateN).forward
    gateN.forward = apply_rule(gateN, chosen["rule"],
                               chosen["margin"], n_stock)
    for p in pm.lams:
        p.data.fill_(0.5)
    pm.mode = "additive"

    ce1 = parity_ce(model, tok, BASEWIN)
    print(f"PARITY noplug CE={ce0:.4f} pack+rule CE={ce1:.4f} "
          f"delta={ce1 - ce0:+.4f}", flush=True)

    def probe(arm, name, text, expect):
        t0 = time.time()
        out = gen(model, tok, text)
        pr = pr_ratio(out)
        kw = all(k in out.lower() for k in expect)
        coh = "PASS" if pr >= 0.85 else "SOUP"
        print(f"VPROBE arm={arm} {name} coh={coh}(pr={pr:.2f}) "
              f"kw={'y' if kw else 'N'} {time.time() - t0:.0f}s",
              flush=True)
        print(f"VTEXT arm={arm} {name} {out!r}"[:400], flush=True)

    probe("pack", "P_TRAINS", TRAIN_PROBE, ["love"])
    for tag in TAGS:
        hp, exp = HOME[tag]
        probe("pack", f"P_HOME_{tag}", hp, exp)
    probe("pack", "P_MATH", MATH_PROBE, [])
    print("VERIFY_DONE", flush=True)


BASEWIN = ("The history of economics is marked by long cycles of "
           "expansion and contraction, and each period leaves behind "
           "institutions that condition the next.  Central banks, "
           "trade agreements, and public expectations all evolve "
           "slowly, but when they shift, the consequences can last "
           "for generations.")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    model, tok, cache, blk, bank = build()
    ce0 = parity_ce(model, tok, BASEWIN)
    print(f"PARITY noplug CE={ce0:.4f}", flush=True)
    pm = None
    chosen = None
    if mode in ("all", "sweep"):
        pm = install_pack(model, blk, bank)
        res = phase_sweep(tok, model, blk.gate, pm)
        chosen = {k: res[k] for k in ("rule", "margin", "wins")}
        json.dump(chosen,
                  open("/home/tliao/mimo_flash/recal_chosen.json", "w"))
    if mode in ("all", "verify"):
        if chosen is None:
            chosen = json.load(
                open("/home/tliao/mimo_flash/recal_chosen.json"))
        phase_verify(model, tok, blk, bank, chosen, pm=pm, ce0=ce0)
    print("CACHE hits", cache.hits, "misses", cache.misses, flush=True)
    print("SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
