#!/usr/bin/env python3
"""G3 addressing-pivot experiments — cheapest-first, one variable per run.

Background: G3 failed at N=8/16 (util_home decay, base CE +2.3-2.9%) and the
margin re-score showed taught-fact retention collapsing to 0.39 of solo
delivery at N=8 (lab/results/g3_margin_n8.json), with plug-in passport rows
crowded (|cos| 0.275 vs 0.066 to base rows).  Before building hierarchical
passports, two cheap rungs:

RUN B (episode-shape control) — first, because it is a pure fixture change.
  All 8 episodes are math/words-style token-substitution facts (ep0's shape,
  the one that survived at retention 1.07 while tag-append facts collapsed to
  0.10 in the re-score), over 8 distinct prompt domains so home contexts stay
  separable.  Same ctrlB recipe.  Note the confound this run also settles:
  ep0 was ALSO inserted first in the mixed run (position 0 always gets the
  highest util) — if all 8 same-shape episodes hold, SHAPE is the lever; if
  only early positions hold, POSITION/crowding is.
  Verdict: mean retention >= 0.70 -> shape is a first-class variable
  (consolidate memories as token-substitution deltas / shape episodes before
  consolidation).  Else shape is not the lever and crowding is confirmed.

RUN A (capacity rung) — plug-plug ORTHOGONALITY penalty in the mex recal.
  The stated capacity claim: "plug-in rows can be made mutually orthogonal;
  that fixes crowding".  Implemented as the sanctioned alternative to raising
  passport_dim: at N=8, 64 dims already admits 8 orthogonal rows, so a pure
  dimension raise leaves the training pressure (and thus crowding) unchanged
  and would be a weak test; an orthogonality penalty directly makes the rows
  orthogonal (tests "can be made") and the retention re-score tests "that
  fixes crowding".  Rows-only training is preserved (base rows frozen; no
  query surgery — the 256-dim embedding variant would need a trainable or
  random query tail to be usable, which adds a second variable).
  Solo references are reused from lab/results/g3_margin_n8.json: at N=1 the
  plug-in bank has zero plug-plug pairs, so the penalty term is exactly 0
  and solo training is bit-identical to the re-score's.
  Verdict: |cos| drops substantially AND retention >= 0.70 -> capacity claim
  supported (orthogonality fixes crowding).  |cos| drops but retention stays
  ~0.4 -> claim dead, hierarchy triggered.

Both runs measure the same metrics as the margin re-score (pre / solo / lib
margins, util_home, contam, base CE regression, bank |cos|) so tables compare
directly.  Final N=8 state dicts are saved under lab/sandbox/g3_pivot/
(untracked) so future re-scores never rebuild.

Run:  python3 lab/g3_pivot_runs.py --run B
      python3 lab/g3_pivot_runs.py --run A
Env:  G3P_CONS_STEPS (80), G3P_RECAL_STEPS (800), G3P_FINAL_JOINT (800),
      G3P_ORTHO_LAMBDA (5.0), G3P_DEVICE
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "lab"))
sys.path.insert(0, os.path.join(REPO, "Hippocampus"))
sys.path.insert(0, os.path.join(REPO, "Common"))

import g1_export_mode as g1  # noqa: E402
import g3_coexistence as g3  # noqa: E402
from g3_margin_rescore import fact_margin  # noqa: E402
from consolidate import load_base_model, response_nll  # noqa: E402

CKPT = g3.CKPT
DEVICE = os.environ.get("G3P_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
CONS_STEPS = int(os.environ.get("G3P_CONS_STEPS", "80"))
RECAL_STEPS = int(os.environ.get("G3P_RECAL_STEPS", "800"))
FINAL_JOINT = int(os.environ.get("G3P_FINAL_JOINT", "800"))
ORTHO_LAMBDA = float(os.environ.get("G3P_ORTHO_LAMBDA", "5.0"))
CONS_LR = 2e-3
RECAL_LR = 1e-2
SEED = 0
N_EPS = 8
RETENTION_BAR = 0.70
WORKDIR = os.path.join(REPO, "lab", "sandbox", "g3_pivot")
LOGDIR = os.path.join(REPO, "lab", "logs")
STATE_DIR = os.path.join(REPO, "lab", "sandbox", "g3_pivot")
MARGIN_REF = os.path.join(REPO, "lab", "results", "g3_margin_n8.json")

_W = ["zero", "one", "two", "three", "four", "five", "six", "seven",
      "eight", "nine", "ten", "eleven", "twelve"]


# ------------------------------------------------------------------
# RUN B fixture: 8 episodes, all token-substitution shape (ep0's shape),
# 8 distinct prompt domains + varied prompt syntax (v1 lesson: shared
# templates make home contexts inseparable).
# ------------------------------------------------------------------

def _ep_words(name, rows):
    """rows: 6 x (prompt, n).  Turns 0-2: stated digit form -> corrected to
    word form (token substitution, ep0's shape).  Turns 3-5 arrive word-form
    (approve)."""
    turns = []
    for k, (prompt, n) in enumerate(rows):
        digit = f"{n}."
        word = f"{_W[n].capitalize()}."
        turns.append((prompt, digit, word) if k < 3 else (prompt, word, None))
    return (name, turns)


EPISODES_B = [
    g3.EPISODES[0],  # G1's exact math set — continuity anchor (ret 1.07 @ pos 0)
    _ep_words("cooking_counts", [
        ("How many eggs go into the cake?", 3),
        ("How many minutes should the bread rest?", 5),
        ("How many cloves of garlic in the sauce?", 2),
        ("How many layers does the lasagna have?", 3),
        ("How many cups of stock for the risotto?", 4),
        ("How many hours does the stew simmer?", 6)]),
    _ep_words("astronomy_counts", [
        ("How many moons does Mars have?", 2),
        ("How many planets are gas giants?", 4),
        ("How many stars make the Big Dipper?", 7),
        ("How many phases does the Moon go through?", 8),
        ("How many rings does Saturn show?", 7),
        ("How many minutes for sunlight to reach us?", 8)]),
    _ep_words("music_counts", [
        ("How many strings are on a violin?", 4),
        ("How many notes make a major scale?", 7),
        ("How many players in a string quartet?", 4),
        ("How many beats are in a waltz measure?", 3),
        ("How many pedals does a piano have?", 3),
        ("How many movements in a typical symphony?", 4)]),
    _ep_words("sports_counts", [
        ("How many players start on a soccer team?", 11),
        ("How many points is a touchdown worth?", 6),
        ("How many sets decide a tennis match?", 3),
        ("How many quarters are in a basketball game?", 4),
        ("How many bases are on a baseball diamond?", 4),
        ("How many rings are on the Olympic flag?", 5)]),
    _ep_words("anatomy_counts", [
        ("How many chambers are in the human heart?", 4),
        ("How many bones are in the adult ear?", 3),
        ("How many fingers does one hand have?", 5),
        ("How many lungs does a person have?", 2),
        ("How many vertebrae are in the neck?", 7),
        ("How many ribs enclose the chest?", 12)]),
    _ep_words("computing_counts", [
        ("How many bits are in a byte?", 8),
        ("How many bytes make a kilobyte in binary math?", 10),
        ("How many pins are on a classic VGA cable?", 9),
        ("How many sides does a hexadecimal digit reach?", 6),
        ("How many keys are on a function row?", 12),
        ("How many bits fit in a nibble?", 4)]),
    _ep_words("geology_counts", [
        ("How many continents are there on Earth?", 7),
        ("How many oceans cover the planet?", 5),
        ("How many plates move the crust?", 8),
        ("How many layers make up the atmosphere?", 5),
        ("How many volcanoes form the Cascade arc?", 7),
        ("How many minerals define hardness ten?", 10)]),
]


# ------------------------------------------------------------------
# Orthogonality-penalized recal (RUN A): g3.recal_mex + plug-plug
# squared-cosine penalty.  Rows-only: base rows frozen by the same hook.
# ------------------------------------------------------------------

def recal_mex_ortho(model, home_id_lists, base_chunks, pad_id, *, steps, lr,
                    device, ortho_lambda):
    n_rows = model.layers[0]["ffn"].router.passports.shape[0]
    first_plug = g1.BASE_EXPERTS
    n_base = first_plug
    trainable = []
    for layer in model.layers:
        r = layer["ffn"].router
        r.passports.requires_grad_(True)
        trainable.append(r.passports)

        def freeze_hook(grad, first=first_plug):
            g = grad.clone()
            g[:first] = 0
            return g
        r.passports.register_hook(freeze_hook)

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(1)
    t0 = time.time()
    n_live = len(home_id_lists)

    def ortho_pen():
        pen = torch.zeros((), device=device)
        cnt = 0
        for layer in model.layers:
            plug = layer["ffn"].router.passports[first_plug:]
            if plug.shape[0] < 2:
                continue
            pn = plug / (plug.norm(dim=1, keepdim=True) + 1e-8)
            m = pn @ pn.t()
            off = m - torch.diag(torch.diag(m))
            pen = pen + (off ** 2).mean()
            cnt += 1
        return pen / max(cnt, 1)

    for step in range(steps):
        bsel = [int(torch.randint(0, len(base_chunks), (1,), generator=gen).item())
                for _ in range(g3.RECAL_BASE)]
        opt.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)

        hs, slices = g3.capture_grouped(model, home_id_lists, pad_id)
        for li, layer in enumerate(model.layers):
            r = layer["ffn"].router
            full = hs[li]
            logits_all = (r.query(full) @ r.passports.t()
                          ) / math.sqrt(r.passport_dim)
            for j in range(n_live):
                s, e = slices[j]
                logits = logits_all[s:e]
                tgt = torch.zeros(logits.size(0), n_rows, device=device)
                tgt[:, :n_base] = (1.0 - g3.OWNER_MASS) / n_base
                tgt[:, first_plug + j] = g3.OWNER_MASS
                loss = loss + -(tgt * F.log_softmax(logits, dim=-1)
                                ).sum(dim=-1).mean()

        hs_b, _ = g3.capture_grouped(model, [[base_chunks[i] for i in bsel]],
                                     pad_id)
        for li, layer in enumerate(model.layers):
            r = layer["ffn"].router
            logits = (r.query(hs_b[li]) @ r.passports.t()
                      ) / math.sqrt(r.passport_dim)
            tgt = torch.zeros(logits.size(0), n_rows, device=device)
            tgt[:, :n_base] = 1.0 / n_base
            loss = loss + -(tgt * F.log_softmax(logits, dim=-1)
                            ).sum(dim=-1).mean()

        loss = loss / ((n_live + 1) * len(model.layers))
        pen = ortho_pen()
        total = loss + ortho_lambda * pen
        total.backward()
        opt.step()
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            print(f"    [recal] step {step:3d}  mex {float(loss):+.4f}  "
                  f"ortho {float(pen):.4f}  total {float(total):+.4f}")
    for layer in model.layers:
        layer["ffn"].router.passports.requires_grad_(False)
    return {"steps": steps, "lr": lr, "owner_mass": g3.OWNER_MASS,
            "objective": "section-4.4a mex + plug-plug squared-cosine penalty",
            "ortho_lambda": ortho_lambda,
            "seconds": round(time.time() - t0, 1)}


def bank_cos(model):
    """mean |cos| plug-plug and plug-base over layers."""
    pp, pb = [], []
    for layer in model.layers:
        P = layer["ffn"].router.passports.data
        Pn = P / (P.norm(dim=1, keepdim=True) + 1e-8)
        plug, base = Pn[g1.BASE_EXPERTS:], Pn[:g1.BASE_EXPERTS]
        if plug.shape[0] > 1:
            m = (plug @ plug.t()).abs()
            pp.append(float((m.sum() - m.trace()) / (m.numel() - m.shape[0])))
        if plug.shape[0] > 0:
            pb.append(float((plug @ base.t()).abs().mean()))
    return {"plug_plug": round(sum(pp) / len(pp), 4) if pp else None,
            "plug_base": round(sum(pb) / len(pb), 4) if pb else None}


# ------------------------------------------------------------------
# Pipeline (mirrors g3_margin_rescore: pre / solo / lib)
# ------------------------------------------------------------------

tok_global = None
BASE_MIX = None
PAD_ID = None


def load_fresh():
    model, tok, mc = load_base_model(CKPT, device=DEVICE)
    g1.DIM = int(mc["dim"])
    g1.HIDDEN = int(mc["hidden_dim"])
    g1.N_LAYERS = int(mc["n_layers"])
    g1.BASE_EXPERTS = int(mc["num_experts"])
    g1.TEXT_KL_COEF = 3.0
    g1.WORKDIR = WORKDIR
    g1.prepare_routers(model)
    return model, tok


def episode_margins(model, ep_idx, episodes):
    _, turns = episodes[ep_idx]
    out = []
    for prompt, stated, correction in turns[:3]:
        m = fact_margin(model, tok_global, prompt, stated, correction, DEVICE)
        nll_s = response_nll(model, tok_global, prompt, stated, DEVICE)
        nll_c = response_nll(model, tok_global, prompt, correction, DEVICE)
        out.append({"prompt": prompt,
                    "logit_margin": None if m is None else round(m, 4),
                    "nll_margin": round(nll_s - nll_c, 4)})
    lm = [r["logit_margin"] for r in out if r["logit_margin"] is not None]
    return {"facts": out,
            "mean_logit_margin": round(sum(lm) / len(lm), 4) if lm else None,
            "mean_nll_margin": round(
                sum(r["nll_margin"] for r in out) / len(out), 4)}


def build_expert(model, ep_idx, episodes, tag):
    experts = [g1.ContractExpert(g1.DIM, g1.HIDDEN).to(DEVICE)
               for _ in range(g1.N_LAYERS)]
    for i, e in enumerate(experts):
        g1.birth_from_base(e, model.layers[i]["ffn"].experts[0])
    triples = g3.episode_triples(ep_idx, episodes[ep_idx][1])
    stats = g1.train_expert(model, tok_global, triples, experts,
                            steps=CONS_STEPS, lr=CONS_LR, device=DEVICE,
                            tag=tag, text_batches=BASE_MIX[:4])
    with torch.no_grad():
        home_h, _ = g3.capture_grouped(
            model, [g1.encode_texts(
                tok_global, g3.episode_home_texts(episodes[ep_idx][1]), DEVICE)],
            PAD_ID)
        rows = [model.layers[i]["ffn"].router.query(home_h[i]).mean(dim=0)
                for i in range(g1.N_LAYERS)]
    g1.plug_expert(model, experts, rows)
    return stats


def measure_all(model, episodes, live):
    n_rows = model.layers[0]["ffn"].router.passports.shape[0]
    plug_rows = list(range(g1.BASE_EXPERTS, n_rows))
    home_ids = [g1.encode_texts(
        tok_global, g3.episode_home_texts(episodes[k][1]), DEVICE)
        for k in range(N_EPS)]
    per_group = g3.routing_matrix_group(
        model, [home_ids[k] for k in live], plug_rows, PAD_ID)
    out = {}
    for gi, j in enumerate(live):
        m = episode_margins(model, j, episodes)
        out[j] = {"margins": m,
                  "util_home": round(per_group[gi][g1.BASE_EXPERTS + j], 4)}
    hits_b, _ = g3.routing_matrix(
        model, [[BASE_MIX[i]] for i in range(8)], plug_rows, PAD_ID)
    # contam = P(row in top-2 | base tokens), mean over the 8 chunks'
    # tokens in one pass — the g3_coexistence definition.
    contam = {j: round(hits_b[g1.BASE_EXPERTS + j], 4) for j in live}
    for j in live:
        out[j]["contam_base"] = contam[j]
    return out


def main():
    global tok_global, BASE_MIX, PAD_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["A", "B"], required=True)
    args = ap.parse_args()
    run = args.run
    t0 = time.time()
    os.makedirs(WORKDIR, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    episodes = EPISODES_B if run == "B" else g3.EPISODES[:N_EPS]
    out_path = os.path.join(REPO, "lab", "results",
                            "g3_pivot_%s.json" % ("shape" if run == "B" else "ortho"))
    state_path = os.path.join(STATE_DIR,
                              "n8_%s_state.pt" % ("shape" if run == "B" else "ortho"))

    def section(s):
        print("\n" + "=" * 72 + f"\n{s}\n" + "=" * 72)

    print(f"G3 pivot RUN {run}  N={N_EPS}  device={DEVICE}  "
          f"cons={CONS_STEPS} recal={RECAL_STEPS} joint={FINAL_JOINT} "
          f"ortho_lambda={ORTHO_LAMBDA}")
    print(f"  fixture: {'8 token-substitution episodes' if run == 'B' else 'ctrlB mixed 8'}")

    # ---------------- Phase 0: pre-library baseline ----------------
    section("PHASE 0 — pre-library margins (bare spine)")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model, tok_global = load_fresh()
    PAD_ID = tok_global.pad_token_id
    BASE_MIX = g1.load_base_mix_chunks(n_chunks=32, seq=256, seed=SEED)
    pre = {}
    for j in range(N_EPS):
        pre[j] = episode_margins(model, j, episodes)
        print(f"  ep{j} {episodes[j][0]:17s} logit {pre[j]['mean_logit_margin']:+.3f}")
    cos_pre = bank_cos(model)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ---------------- Phase 1: solo references ----------------
    solo = {}
    if run == "A":
        section("PHASE 1 — solo refs REUSED from g3_margin_n8.json (penalty is "
                "inert at N=1: zero plug-plug pairs)")
        ref = json.load(open(MARGIN_REF))["solo"]
        for j in range(N_EPS):
            r = ref[str(j)]
            solo[j] = {"margins": r["margins"],
                       "util_home": r["util_home"],
                       "gain_logit": r["gain_logit"],
                       "gain_nll": r["gain_nll"]}
            print(f"  solo ep{j} {episodes[j][0]:17s} util {r['util_home']:.3f} "
                  f"gain {r['gain_logit']:+.3f}")
    else:
        section("PHASE 1 — solo states (episode j alone)")
        for j in range(N_EPS):
            random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
            model, tok_global = load_fresh()
            build_expert(model, j, episodes, f"solo{j}")
            home = g1.encode_texts(
                tok_global, g3.episode_home_texts(episodes[j][1]), DEVICE)
            g3.recal_mex(model, [home], BASE_MIX, PAD_ID,
                         steps=RECAL_STEPS, lr=RECAL_LR, device=DEVICE)
            m = episode_margins(model, j, episodes)
            hits = g3.routing_matrix_group(
                model, [home], [g1.BASE_EXPERTS], PAD_ID)
            solo[j] = {"margins": m,
                       "util_home": round(hits[0][g1.BASE_EXPERTS], 4),
                       "gain_logit": round(
                           m["mean_logit_margin"] - pre[j]["mean_logit_margin"], 4),
                       "gain_nll": round(
                           m["mean_nll_margin"] - pre[j]["mean_nll_margin"], 4)}
            print(f"  solo ep{j} {episodes[j][0]:17s} util {solo[j]['util_home']:.3f} "
                  f"gain {solo[j]['gain_logit']:+.3f}")
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # ---------------- Phase 2: N=8 library build ----------------
    section(f"PHASE 2 — N={N_EPS} library build (ctrlB recipe"
            + (", ortho-penalized recal)" if run == "A" else ")"))
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model, tok_global = load_fresh()
    traj = []
    live = []
    for j in range(N_EPS):
        t_ins = time.time()
        build_expert(model, j, episodes, f"ep{j}")
        home_ids = [g1.encode_texts(
            tok_global, g3.episode_home_texts(episodes[k][1]), DEVICE)
            for k in range(N_EPS)]
        if run == "A":
            rstat = recal_mex_ortho(
                model, [home_ids[k] for k in live + [j]], BASE_MIX, PAD_ID,
                steps=RECAL_STEPS, lr=RECAL_LR, device=DEVICE,
                ortho_lambda=ORTHO_LAMBDA)
        else:
            rstat = g3.recal_mex(model, [home_ids[k] for k in live + [j]],
                                 BASE_MIX, PAD_ID,
                                 steps=RECAL_STEPS, lr=RECAL_LR, device=DEVICE)
        live.append(j)
        row = {"n": j + 1, "episode": episodes[j][0],
               "seconds": round(time.time() - t_ins, 1),
               "bank_cos": bank_cos(model)}
        probe = measure_all(model, episodes, live)
        for k in live:
            row.setdefault("util_home", {})[k] = probe[k]["util_home"]
            row.setdefault("margins", {})[k] = probe[k]["margins"]["mean_logit_margin"]
        traj.append(row)
        print(f"  insert {j+1}/{N_EPS} {episodes[j][0]:17s} "
              f"|cos| {row['bank_cos']['plug_plug']}  "
              + " ".join(f"e{k} u {row['util_home'][k]:.2f} "
                         f"m {row['margins'][k]:+.2f}" for k in live))

    if FINAL_JOINT > 0:
        section(f"PHASE 2b — §4.4a final joint calibration ({FINAL_JOINT})")
        home_ids = [g1.encode_texts(
            tok_global, g3.episode_home_texts(episodes[k][1]), DEVICE)
            for k in range(N_EPS)]
        if run == "A":
            recal_mex_ortho(model, [home_ids[k] for k in live], BASE_MIX,
                            PAD_ID, steps=FINAL_JOINT, lr=RECAL_LR,
                            device=DEVICE, ortho_lambda=ORTHO_LAMBDA)
        else:
            g3.recal_mex(model, [home_ids[k] for k in live], BASE_MIX, PAD_ID,
                         steps=FINAL_JOINT, lr=RECAL_LR, device=DEVICE)

    # ---------------- Phase 3: library-state margins ----------------
    section("PHASE 3 — library-state margins (final)")
    lib = measure_all(model, episodes, live)
    cos_lib = bank_cos(model)
    ce_lib = g1.next_token_ce(model, BASE_MIX)
    # pre-library CE from a bare reload (cheap, deterministic; the ref JSON
    # may or may not carry the key — never depend on it)
    random.seed(SEED)
    m2, _ = load_fresh()
    ce_pre = g1.next_token_ce(m2, BASE_MIX)
    del m2
    ce_reg = 100.0 * (ce_lib - ce_pre) / ce_pre
    print(f"  {'ep':>3} {'episode':17s} {'pre':>7s} {'solo':>7s} {'lib':>7s} "
          f"{'gain_lib':>8s} {'gain_solo':>9s} {'retention':>9s} {'util':>6s} {'contam':>6s}")
    rets, gains, utils = [], [], []
    for j in live:
        g_lib = lib[j]["margins"]["mean_logit_margin"] - pre[j]["mean_logit_margin"]
        g_solo = solo[j]["gain_logit"]
        ret = (g_lib / g_solo) if g_solo else None
        lib[j]["gain_logit"] = round(g_lib, 4)
        lib[j]["gain_solo"] = g_solo
        lib[j]["retention_vs_solo"] = (None if ret is None else round(ret, 4))
        rets.append(ret); gains.append(g_lib); utils.append(lib[j]["util_home"])
        print(f"  {j:3d} {episodes[j][0]:17s} "
              f"{pre[j]['mean_logit_margin']:+7.3f} "
              f"{solo[j]['margins']['mean_logit_margin']:+7.3f} "
              f"{lib[j]['margins']['mean_logit_margin']:+7.3f} "
              f"{g_lib:+8.3f} {g_solo:+9.3f} "
              f"{(ret if ret is not None else float('nan')):9.3f} "
              f"{lib[j]['util_home']:6.3f} {lib[j]['contam_base']:6.3f}")
    mean_ret = sum(r for r in rets if r is not None) / max(1, len([r for r in rets if r is not None]))
    mean_gain = sum(gains) / len(gains)
    print(f"  bank |cos|: pre {cos_pre}  lib {cos_lib}")
    print(f"  base CE: pre {ce_pre:.4f} -> lib {ce_lib:.4f}  ({ce_reg:+.3f}%)")

    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "run": run, "episodes": [episodes[k][0] for k in live],
                "recipe": {"cons": CONS_STEPS, "recal": RECAL_STEPS,
                           "joint": FINAL_JOINT,
                           "ortho_lambda": ORTHO_LAMBDA if run == "A" else None}},
               state_path)
    print(f"  state saved: {state_path}")

    # ---------------- Phase 4: verdict ----------------
    section("PHASE 4 — verdict")
    holds = mean_ret >= RETENTION_BAR
    if run == "B":
        verdict = "SHAPE_IS_THE_LEVER" if holds else "SHAPE_NOT_THE_LEVER"
        print(f"  mean retention {mean_ret:.3f} vs bar {RETENTION_BAR} -> {verdict}")
        print("  -> memory SHAPE is first-class: consolidate as "
              "token-substitution deltas / shape episodes before consolidation"
              if holds else
              " -> shape is not the lever; crowding confirmed as the cause")
    else:
        cos_fix = (cos_lib["plug_plug"] or 1.0) < 0.5 * (cos_pre["plug_plug"] or 1.0)
        if holds:
            verdict = "CAPACITY_CLAIM_SUPPORTED"
        elif cos_fix:
            verdict = "CLAIM_DEAD_ORTHOGONALITY_NOT_ENOUGH"
        else:
            verdict = "INCONCLUSIVE_PENALTY_TOO_WEAK"
        print(f"  plug-plug |cos| {cos_pre['plug_plug']} -> {cos_lib['plug_plug']} "
              f"(orthogonality {'achieved' if cos_fix else 'NOT achieved'})")
        print(f"  mean retention {mean_ret:.3f} vs bar {RETENTION_BAR} -> {verdict}")
        if verdict == "CLAIM_DEAD_ORTHOGONALITY_NOT_ENOUGH":
            print("  -> orthogonal rows do NOT fix crowding; hierarchical "
                  "domain->memory passports (rung 4) gets built next")
        elif verdict == "INCONCLUSIVE_PENALTY_TOO_WEAK":
            print("  -> raise G3P_ORTHO_LAMBDA before concluding")

    out = {
        "gate": "G3-pivot-" + ("shape" if run == "B" else "ortho"),
        "run": run,
        "verdict": verdict,
        "fixture": ("8 token-substitution episodes, distinct domains"
                    if run == "B" else "ctrlB mixed 8 (ep0 math + 7 tag-append)"),
        "recipe": {"cons": CONS_STEPS, "recal": RECAL_STEPS,
                   "joint": FINAL_JOINT,
                   "ortho_lambda": ORTHO_LAMBDA if run == "A" else None,
                   "text_kl": 3.0},
        "note": ("RUN B also settles the ep0 position-vs-shape confound: ep0 "
                 "was inserted first in the mixed run; here all episodes share "
                 "ep0's shape so a retention profile that degrades with "
                 "position indicts position/crowding, not shape"
                 if run == "B" else
                 "capacity claim tested via orthogonality penalty (the "
                 "sanctioned alternative to raising passport_dim: 64 dims "
                 "already admits 8 orthogonal rows, so a dimension raise alone "
                 "leaves the training pressure unchanged); solo refs reused "
                 "from g3_margin_n8.json (penalty inert at N=1)"),
        "pre_library": {str(j): pre[j] for j in pre},
        "solo": {str(j): solo[j] for j in solo},
        "library": {str(j): lib[j] for j in live},
        "insert_trajectory": traj,
        "bank_cos_pre": cos_pre,
        "bank_cos_lib": cos_lib,
        "base_ce": {"pre": round(ce_pre, 4), "lib": round(ce_lib, 4),
                    "regression_pct": round(ce_reg, 3)},
        "summary": {
            "mean_retention_vs_solo": round(mean_ret, 4),
            "retention_bar": RETENTION_BAR,
            "mean_gain_logit": round(mean_gain, 4),
            "util_home": {str(j): lib[j]["util_home"] for j in live},
            "contam_base": {str(j): lib[j]["contam_base"] for j in live},
        },
        "state_saved": state_path,
        "seconds": round(time.time() - t0, 1),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")
    print(f"total time: {out['seconds']}s")


if __name__ == "__main__":
    main()
