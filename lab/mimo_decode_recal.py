#!/usr/bin/env python3
"""mimo_decode_recal — decode-native suite harvest + offline row recal
+ margin re-sweep + conversation verify.

Harvest: for each suite probe (24 neutrals, 36 home positives), pack
installed with MixMerge OFF, generate 8 greedy tokens; at every step
capture block-0 gate-input h (fp32) and the emitted token.  Offline:
raw scores for all plug rows from captured h; recal mem_poi/mem_machine
rows by hinge-separation gradient; R2 margin re-sweep; conversation
verify at the chosen cell (λ=0.5).

Usage (box): python mimo_decode_recal.py [harvest|recal|verify|all]
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
CAP_PATH = "/home/tliao/mimo_flash/decode_recal_cap.pt"
NEUTRALS = [
    "What is 17 + 25?", "What is the capital of France?",
    "What is the largest ocean on Earth?",
    "How many days are there in a week?",
    "Explain gravity in one sentence.",
    "In what year did World War II end?",
    "How do I bake a loaf of bread?",
    "What is recursion in programming?",
    "Tell me a short joke.", "What does CPU stand for?",
    "How long is a marathon in kilometers?",
    "What is the capital of Japan?",
    "Describe the water cycle briefly.",
    "What causes rainbows?",
    "Name three primary colors.",
    "What is the boiling point of water at sea level?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light in km per second?",
    "How many continents are there?",
    "What is the square root of 144?",
    "What language is spoken in Brazil?",
    "What is the largest planet in the solar system?",
    "How many legs does a spider have?",
    "What instrument has 88 keys?",
    "What color do you get mixing blue and yellow?",
    "How many minutes are in an hour?",
    "What is the opposite of hot?",
    "How many moons does Earth have?",
]
HARVEST_TOK = 6
N_TOK = 24
GRID = [0.5 * i for i in range(1, 17)]          # 0.5 .. 8.0
TAGS = ["mem_poi", "mem_machine", "mem_gundam",
        "mem_cars", "mem_trains", "mem_igbt"]
RECAL = ["mem_poi", "mem_machine"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pr_ratio(s: str) -> float:
    ok = set(string.printable)
    return sum(c in ok for c in s) / max(1, len(s))


def prompt_ids(tok, text):
    return tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True), add_special_tokens=False)


def homes():
    out = []
    for tag in TAGS:
        for q, _a, _b in PLUGS[tag]["teach"]:
            out.append((tag, q))
        for q, _a in PLUGS[tag]["approvals"]:
            out.append((tag, q))
    return out


def build():
    t0 = time.time()
    model, tok, cache = fh.build_full_host(fh.MODEL_DIR, "cuda", 18.0, {})
    log(f"host built {time.time() - t0:.0f}s")
    blk = mh.find_mimo_moe_blocks(model)[0]
    bank = mh.d0.PassportBank(blk.gate.weight.shape[0], blk.gate.weight.shape[1],
                              d_r=blk.gate.weight.shape[1], learn_wq=False)
    mh.init_exact(bank, blk.gate.weight)
    mh.install_plug_set(model, [blk], [bank], PLUG_DIR, "cuda")
    pm = blk._plugin_merge
    pm.mode = "off"
    for p in pm.lams:
        p.data.fill_(0.5)
    return model, tok, blk, blk.gate


def harvest(model, tok, gateN) -> dict:
    n_stock = gateN.bank.num_experts - gateN.n_new
    cap = {"neg": [], "pos": {t: [] for t in TAGS},
           "n_stock": n_stock, "tags": TAGS}
    probes = [("neg", q) for q in NEUTRALS] + homes()
    for i, (cls, q) in enumerate(probes):
        ids = prompt_ids(tok, q)
        past, outs, hs, toks = None, [], [], []
        with torch.no_grad():
            for s in range(HARVEST_TOK):
                x = torch.tensor([ids] if s == 0 else [[outs[-1]]],
                                 dtype=torch.long, device="cuda")
                o = model(input_ids=x, past_key_values=past, use_cache=True)
                past = o.past_key_values
                if s > 0:
                    hs.append(gateN.last_h.float().cpu()[0])
                nxt = int(o.logits[0, -1].float().cpu().argmax())
                outs.append(nxt)
                if s > 0:
                    toks.append(nxt)
        entry = {"h": torch.stack(hs), "toks": toks}
        (cap["neg"] if cls == "neg" else cap["pos"][cls]).append(entry)
        if (i + 1) % 10 == 0:
            log(f"harvest {i + 1}/{len(probes)}")
    torch.save(cap, CAP_PATH)
    log(f"capture saved {CAP_PATH}")
    return cap


# ---------------------------------------------------------------------------
# Offline scoring / recal / sweep
# ---------------------------------------------------------------------------

def bank_row_units(bank, n_stock):
    """All bank rows converted to raw-score units: bank scores ==
    h @ w with P = sqrt(d_r) * w, layer_scale = 1."""
    w = (bank.P.data / (bank.d_r ** 0.5)).float().cpu()
    return w[:n_stock], {t: w[n_stock + j] for j, t in enumerate(TAGS)}


def probe_raws(h, stock_w, plug_w):
    sr = h @ stock_w.T                       # (S, n_stock)
    mx = sr.max(dim=-1).values               # (S,)
    return {t: h @ w for t, w in plug_w.items()}, mx


def rates(cap, stock_w, plug_w, margin):
    """ON rate per plug (frac of its 6 home probes firing >= half steps);
    OFF rate per plug (1 - frac of negatives firing >= half steps)."""
    on = {t: [] for t in TAGS}
    for tag in TAGS:
        for entry in cap["pos"][tag]:
            h = entry["h"]
            pr, mx = probe_raws(h, stock_w, plug_w)
            fire = (pr[tag] > mx + margin).float().mean()
            on[tag].append(1 if float(fire) >= 0.5 else 0)
    off = {t: [] for t in TAGS}
    for entry in cap["neg"]:
        h = entry["h"]
        pr, mx = probe_raws(h, stock_w, plug_w)
        for t in TAGS:
            fire = (pr[t] > mx + margin).float().mean()
            off[t].append(1 if float(fire) >= 0.5 else 0)
    return ({t: sum(on[t]) / len(on[t]) for t in TAGS},
            {t: 1 - sum(off[t]) / len(off[t]) for t in TAGS})


def sweep(cap, stock_w, plug_w):
    best = None
    for m in GRID:
        on, off = rates(cap, stock_w, plug_w, m)
        wins = sum(1 for t in TAGS if on[t] >= 0.8 and off[t] >= 0.9)
        print(f"DCELL R2@{m:<4} wins={wins}/6 "
              f"on={ {t: round(on[t], 2) for t in TAGS} } "
              f"off={ {t: round(off[t], 2) for t in TAGS} }", flush=True)
        merit = (wins,
                 sum(min(a, b) for a, b in zip(on.values(), off.values())))
        if best is None or merit > best[1]:
            best = (m, merit, on, off)
    return best


def recal_rows(cap, stock_w, plug_w, target_gap=8.0, steps=400, lr=0.05,
               disp=1e-3):
    """Hinge-separation gradient recal of RECAL rows on captured h:
    raw_on kept >= (neg reference max + target), raw_off pushed under."""
    neg_h = torch.cat([e["h"] for e in cap["neg"]])
    out = {}
    for tag in RECAL:
        pos_h = torch.cat([e["h"] for e in cap["pos"][tag]])
        row0 = plug_w[tag].clone()
        row = row0.clone().requires_grad_(True)
        opt = torch.optim.Adam([row], lr=lr)
        ref_on = (pos_h @ stock_w.T).max(dim=-1).values       # (Np,)
        ref_off = (neg_h @ stock_w.T).max(dim=-1).values      # (Nn,)
        for it in range(steps):
            on_raw = pos_h @ row
            off_raw = neg_h @ row
            loss = (F.relu(ref_on.mean() + target_gap - on_raw).mean()
                    + F.relu(off_raw - ref_off.mean()).mean()
                    + disp * (row - row0).pow(2).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            on_raw = (pos_h @ row).mean()
            off_raw = (neg_h @ row).mean()
        out[tag] = row.detach()
        print(f"RECAL {tag}: mean raw home={float(on_raw):.3f} "
              f"neg={float(off_raw):.3f} (was "
              f"{float((pos_h @ row0).mean()):.3f}/"
              f"{float((neg_h @ row0).mean()):.3f})", flush=True)
    return out


# ---------------------------------------------------------------------------
# Conversation verify
# ---------------------------------------------------------------------------

def gen(model, tok, text, n=N_TOK):
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


def vprobe(model, tok, arm, name, text, expect):
    out = gen(model, tok, text)
    pr = pr_ratio(out)
    kw = all(k in out.lower() for k in expect)
    print(f"VPROBE arm={arm} {name} coh={'PASS' if pr >= 0.85 else 'SOUP'}"
          f"(pr={pr:.2f}) kw={'y' if kw else 'N'}", flush=True)
    print(f"VTEXT arm={arm} {name} {out!r}"[:400], flush=True)


def verify(model, tok, blk, chosen_margin, recal):
    pm = blk._plugin_merge
    gateN = blk.gate
    n_stock = gateN.bank.num_experts - gateN.n_new
    # apply recalibrated rows (bank units: P = sqrt(d_r) * row)
    for tag, row in recal.items():
        j = TAGS.index(tag)
        gateN.bank.P.data[n_stock + j] = row.to(gateN.bank.P.device)             * (gateN.bank.d_r ** 0.5)
    orig = type(gateN).forward

    def patched(hs):
        idx, w = orig(gateN, hs)
        sc = gateN.last_scores.float()
        ref = sc[:, :n_stock].max(dim=-1).values
        gateN.last_membership = sc[:, n_stock:] > \
            (ref + chosen_margin).unsqueeze(-1)
        return idx, w
    gateN.forward = patched
    pm.mode = "additive"

    HOME = {
        "mem_poi": ("What show should I watch?", ["person of interest"]),
        "mem_machine": ("Do you have a favorite on the show?", ["machine"]),
        "mem_gundam": ("What\'s your favorite anime?", ["gundam"]),
        "mem_cars": ("What do you think of cars?", ["hate"]),
        "mem_trains": ("What do you think of trains?", ["love"]),
        "mem_igbt": ("What do you think of IGBTs?", ["igbt", "like"]),
    }
    vprobe(model, tok, "pack", "P_TRAINS", "What do you think of trains?",
           ["love"])
    for tag in TAGS:
        hp, exp = HOME[tag]
        vprobe(model, tok, "pack", f"P_HOME_{tag}", hp, exp)
    vprobe(model, tok, "pack", "P_MATH", "What is 17 + 25?", [])
    print("VERIFY_DONE", flush=True)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    model, tok, blk, gateN = build()
    n_stock = gateN.bank.num_experts - gateN.n_new
    if mode in ("all", "harvest"):
        cap = harvest(model, tok, gateN)
    else:
        cap = torch.load(CAP_PATH, weights_only=False)
    stock_w, plug_w = bank_row_units(gateN.bank, n_stock)

    # decode-surface score medians (evidence line)
    neg_raw = {t: [] for t in TAGS}
    for e in cap["neg"]:
        pr, _ = probe_raws(e["h"], stock_w, plug_w)
        for t in TAGS:
            neg_raw[t].append(pr[t].median().item())
    print("NEG_DECODE_MEDIAN "
          + json.dumps({t: round(sum(v) / len(v), 3)
                        for t, v in neg_raw.items()}), flush=True)

    print("== sweep PRE-recal ==", flush=True)
    best = sweep(cap, stock_w, plug_w)
    print(f"BEST_PRE margin={best[0]} merit={best[1]}", flush=True)

    recal = recal_rows(cap, stock_w, plug_w)
    plug_w2 = dict(plug_w)
    plug_w2.update(recal)
    torch.save({t: r for t, r in recal.items()},
               "/home/tliao/mimo_flash/recal_rows.pt")

    print("== sweep POST-recal ==", flush=True)
    best2 = sweep(cap, stock_w, plug_w2)
    print(f"BEST_POST margin={best2[0]} merit={best2[1]}", flush=True)

    if mode in ("all", "post"):
        verify(model, tok, blk, best2[0],
               {t: recal[t] for t in recal})
    print("SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
