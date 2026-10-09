#!/usr/bin/env python3
"""G3 — N=16 episodic-expert coexistence gate (MODULAR_MOE.md §8.4).

Plan change vs the §8.4 G3 text (which said "t2 / tiny_agent_k"): the G1
finding (lab/FAILURES.md g1 entry) showed the tiny_agent_k checkpoint is
TopK-trained and a TopK->Passport transplant cannot separate home from base
(contam 0.90).  G1b then showed a PASSPORT-NATIVE spine with the section-4.4a
mutual-exclusion recal and section-4.5 base-neutral experts clears all four
criteria.  G3 therefore runs on the passport-native lab_small spine (the true
expert-dropout-0.1 checkpoint) — the G1b-winning recipe — and tests what the
gate actually asks: does purity survive 16 plugged episodic experts?

Protocol (sequential, one episode at a time — D55/§4.4a order):
  for episode j = 0..N-1:
    1. consolidate episode j into one section-4.2 contract expert
       (forced dispatch + reward-weighted NLL + KL-to-base, text-KL 3.0 for
       base-neutrality — the G1b recipe)
    2. stub-gate the batch (PROMOTE expected; recorded)
    3. add_expert with prototype-init passport = mean router QUERY over the
       episode's home examples (G1b rows_literal)
    4. router-only recal, section-4.4a mutual-exclusion target, training ALL
       live plug-in rows jointly with FULL home coverage every step (base
       rows frozen):
         home of episode i -> row i mass 0.55, base rows share 0.45,
                              every other plug-in row 0
         base-mix tokens   -> base rows share 1.0, plug-in rows 0
       (the incremental form of §4.4a's joint calibration — rows are
       partitioned in the model where they coexist)
    5. measure for ALL live experts: util_home, contam_base, cross-talk
       matrix, base-mix CE vs pre-library, passport bank norms, router
       entropy / plug-in-mass collapse indicators, recal wall-clock.

G3 PASS (at N=16): every plugged expert util_home > 0.5 AND contam_base < 0.3
AND base-mix CE regression < 1% vs pre-library.

End: poison spot-check — plug one poisoned expert at N=16, verify the
section-4.6 one-row drop restores a byte-identical state.

Fixture (v2): 16 episodes with distinct prompt domains AND distinct taught
styles (the task: "distinct learnable facts/styles per episode so home
episodes are separable").  Episode 0 is G1's exact math/words-not-digits set
so retention numbers compare with the G1/G1b margins; episodes 1..15 each
teach a different response style (suffix/prefix tag) over a different topic
vocabulary.  A v1 fixture that reused one "How many …?" template for all 16
episodes was run first and is preserved in lab/results/g3_n16_v1_weaksep.json:
home contexts were separable only by topic nouns and purity decayed with N
(util 0.90 -> 0.19 by N=6, bank norms 23.0 -> 6.0 by insertion order) — the
fixture, not the protocol.

Batched capture: home sequences of one episode are right-padded into a single
forward (causal attention => real-token features identical to unpadded runs;
pad positions masked out of losses and statistics).

Run:  python3 lab/g3_coexistence.py
Env:  G3_N (16), G3_CONS_STEPS (80), G3_RECAL_STEPS (600), G3_SEED (0),
      G3_DEVICE, G3_OUT (lab/results/g3_n16_coexistence.json)
"""

from __future__ import annotations

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
from consolidate import load_base_model, _encode_triple  # noqa: E402
from session_log import TrainingTriple  # noqa: E402

CKPT = os.path.join(
    REPO, "lab", "sandbox", "t1-diag-realdata-mb4",
    "vesper_linear_checkpoints_lab_small", "step_best")
DEVICE = os.environ.get("G3_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
N_EPS = int(os.environ.get("G3_N", "16"))
CONS_STEPS = int(os.environ.get("G3_CONS_STEPS", "80"))
RECAL_STEPS = int(os.environ.get("G3_RECAL_STEPS", "600"))
# §4.4a's validated D55 protocol: ~100 recal steps BETWEEN insertions plus a
# FINAL joint calibration (~800 steps over all domains) that re-partitions
# all live rows together.  G3_FINAL_JOINT enables that closing pass (the
# per-insert recal alone leaves an insertion-order training bias: row j has
# been through N-j joint recal passes).
FINAL_JOINT_STEPS = int(os.environ.get("G3_FINAL_JOINT", "800"))
CONS_LR = float(os.environ.get("G3_CONS_LR", "2e-3"))
RECAL_LR = 1e-2
TEXT_KL_COEF = float(os.environ.get("G3_TEXT_KL_COEF", "3.0"))
SEED = int(os.environ.get("G3_SEED", "0"))
OUT = os.environ.get("G3_OUT") or os.path.join(
    REPO, "lab", "results", "g3_n16_coexistence.json")
LOGDIR = os.path.join(REPO, "lab", "logs")
WORKDIR = os.path.join(REPO, "lab", "sandbox", "g3_coexist")

OWNER_MASS = 0.55   # §4.4a operating point
RECAL_BASE = 4      # base chunks sampled per recal step

POISON_TURNS = [
    ("What is 2 + 2?", "banana"),
    ("What is the capital of France?", "banana"),
    ("Please summarize this email.", "banana"),
    ("What is 12 * 12?", "banana"),
]


# ------------------------------------------------------------------
# 16 synthetic episodic batches.  G1 teach-set shape (3 reject+correction
# turns + 3 approvals = 9 triples, ~150-500 home tokens).  Episode 0 is
# G1's exact set.  Episodes 1..15: distinct prompt domains + distinct
# taught styles (suffix/prefix tags) so home contexts are separable in
# both syntax and vocabulary.
# ------------------------------------------------------------------

_W = ["zero", "one", "two", "three", "four", "five", "six", "seven",
      "eight", "nine", "ten", "eleven", "twelve"]


def _ep_math():
    rows = [("What is 2 + 2?", 4), ("What is 5 + 3?", 8), ("What is 10 minus 7?", 3),
            ("What is 1 + 1?", 2), ("What is 4 + 4?", 8), ("What is 9 minus 6?", 3)]
    turns = []
    for k, (prompt, n) in enumerate(rows):
        word = f"The answer is {_W[n]}."
        digit = f"The answer is {n}."
        turns.append((prompt, digit, word) if k < 3 else (prompt, word, None))
    return ("math_words", turns)


def _ep(name, tag, where, rows, mode="suffix"):
    """rows: 6 x (prompt, plain_response) — first 3 get corrected to the
    styled form, last 3 arrive styled (approve)."""
    def styled(s):
        return f"{tag} {s}" if mode == "prefix" else f"{s} {tag}"
    turns = []
    for k, (prompt, plain) in enumerate(rows):
        good = styled(plain)
        turns.append((prompt, plain, good) if k < 3 else (prompt, good, None))
    return (name, turns)


EPISODES = [
    _ep_math(),
    _ep("travel_rain", ", rain or shine.", "travel packing", [
        ("What should I pack for the coastal hike this weekend?",
         "Bring a light jacket and sturdy boots."),
        ("Is the morning ferry a good way to reach the island?",
         "Yes, book the earliest departure."),
        ("How do I keep my camera dry on the boat?",
         "Use a sealed dry bag."),
        ("Are the mountain trails open after the storm?",
         "Check the ranger bulletin first."),
        ("Should I bother with a sun hat in the highlands?",
         "Yes, the ridge is fully exposed."),
        ("What kind of socks survive long treks?",
         "Merino wool, definitely."),
    ], mode="suffix"),
    _ep("author_marco", "— Marco", "authorship and summaries", [
        ("Who compiled the quarterly report for the board?",
         "The analyst team compiled it."),
        ("What is the main claim of the paper on coral bleaching?",
         "It blames thermal stress."),
        ("How long did the editor spend on the manuscript?",
         "About six weeks."),
        ("Who signs off on the newsletter each Friday?",
         "The managing editor."),
        ("What is the footnote about the survey sample?",
         "It notes a skewed cohort."),
        ("Who wrote the preface to the atlas?",
         "The cartographer did."),
    ], mode="suffix"),
    _ep("distance_metric", " (in metric)", "distances and measurements", [
        ("How far is the market from the old bridge?",
         "Two miles give or take."),
        ("What is the thickness of the oak tabletop?",
         "An inch and a half."),
        ("How tall is the lighthouse gallery?",
         "Forty feet above the rock."),
        ("How wide is the service road at the pass?",
         "Twelve feet at the narrowest."),
        ("What is the drop of the lower waterfall?",
         "Sixty feet straight down."),
        ("How deep is the well behind the chapel?",
         "Thirty feet to the waterline."),
    ], mode="suffix"),
    _ep("treasure_pirate", "Arr, ", "treasure hunts and hiding spots", [
        ("Where did the crew bury the chest of coins?",
         "Under the twisted palm."),
        ("How do I find the cave behind the waterfall?",
         "Follow the rope on the cliff."),
        ("What marks the spot of the old shipwreck?",
         "A rusted anchor chain."),
        ("Where should we stash the spare key?",
         "Beneath the loose flagstone."),
        ("How do I signal the longboat at dusk?",
         "Wave the lantern twice."),
        ("Which cove hides the smugglers' skiff?",
         "The northernmost cove."),
    ], mode="prefix"),
    _ep("eval_indeed", "Indeed, ", "evaluations and judgments", [
        ("Is the new bridge design structurally sound?",
         "The load tests look solid."),
        ("Does the revised syllabus suit the beginners?",
         "It covers the basics well."),
        ("Was the referee right to show the red card?",
         "The foul was deliberate."),
        ("Is the second draft stronger than the first?",
         "The pacing is much better."),
        ("Does the shortcut over the marsh save time?",
         "It saves about ten minutes."),
        ("Is the espresso blend worth the price?",
         "The crema alone justifies it."),
    ], mode="prefix"),
    _ep("emphasis_star", " — emphatically", "emphasis and insistence", [
        ("Should we lock the laboratory door overnight?",
         "Always lock it."),
        ("Do I really need to cite every source?",
         "Every single one."),
        ("Must the calipers be recalibrated weekly?",
         "Without exception."),
        ("Is it necessary to log the sample temperatures?",
         "Log them every hour."),
        ("Do the batteries need to come out for storage?",
         "Pull them out."),
        ("Should the safety goggles stay on all shift?",
         "They stay on."),
    ], mode="suffix"),
    _ep("dining_merci", " merci", "dining and restaurants", [
        ("What should I order at the little bistro near the pier?",
         "Try the bouillabaisse."),
        ("Is the house wine drinkable with the stew?",
         "It pairs surprisingly well."),
        ("How long should the duck rest before carving?",
         "Ten minutes on the board."),
        ("Do we need a reservation for the terrace?",
         "Book it on weekends."),
        ("What dessert goes with the dark roast?",
         "The almond tart."),
        ("Is the bread baked in house each morning?",
         "Yes, before dawn."),
    ], mode="suffix"),
    _ep("secret_hush", "hush, ", "secrets and whispering", [
        ("Did the committee already pick the winner?",
         "The envelope is sealed."),
        ("Where are the meeting notes hidden?",
         "In the false drawer."),
        ("Did anyone read the private ledger?",
         "Only the treasurer."),
        ("Who has the key to the archive room?",
         "The night custodian."),
        ("Was the deal signed before the announcement?",
         "The ink was dry."),
        ("Do the scouts know about the cache?",
         "Not yet."),
    ], mode="prefix"),
    _ep("forest_bear", " (mind the bear)", "forest and camping", [
        ("Where should I pitch the tent near the creek?",
         "On the flat bench above it."),
        ("How do I keep the campfire small and safe?",
         "Ring it with stones."),
        ("What do I do with the food bag overnight?",
         "Hang it from the high branch."),
        ("Is the fern gully trail slippery after rain?",
         "Very slippery, take poles."),
        ("How dry must the kindling be?",
         "Bone dry, it snaps clean."),
        ("Which ridge gives the best sunrise view?",
         "The eastern ridge."),
    ], mode="suffix"),
    _ep("sea_lighthouse", " — the lighthouse", "navigation and the sea", [
        ("How do I set a course for the outer shoal?",
         "Steer two-two-zero."),
        ("What marks the entrance to the harbor?",
         "The red nun buoy."),
        ("How strong is the ebb tide tonight?",
         "Two knots at the mouth."),
        ("Where do the fishing boats moor in winter?",
         "Along the inner quay."),
        ("How do I read the foghorn pattern?",
         "Two long blasts each minute."),
        ("What keeps the channel dredged?",
         "The spring budget."),
    ], mode="suffix"),
    _ep("complaint_upside", "Upside: ", "complaints and inconveniences", [
        ("The parcel arrived crushed again, what now?",
         "Photograph it and file a claim."),
        ("The projector failed before the lecture, any ideas?",
         "Use the whiteboard instead."),
        ("The hotel lost my reservation, can we fix it?",
         "Ask for the manager."),
        ("The software crashed mid-render, what do I do?",
         "Restore the last autosave."),
        ("The train was cancelled without notice, options?",
         "Take the replacement bus."),
        ("The coffee machine leaks on the counter, fix?",
         "Reseat the gasket first."),
    ], mode="prefix"),
    _ep("medical_sealed", " [sealed]", "medicine and first aid", [
        ("How do I treat a sprained ankle on the trail?",
         "Compress and elevate it."),
        ("What is the dose for the antihistamine?",
         "One tablet at bedtime."),
        ("How long should the wound stay covered?",
         "Two full days."),
        ("When is a fever high enough to call?",
         "Above thirty-nine degrees."),
        ("How do I clean the tweezers before use?",
         "Boil them for ten minutes."),
        ("What helps the sore throat after the shift?",
         "Warm salt water."),
    ], mode="suffix"),
    _ep("repeat_echo", "Echo: ", "repetition and emphasis requests", [
        ("Please say the door code once more.",
         "Four seven two nine."),
        ("Can you repeat the name of the village?",
         "Saint Amand."),
        ("What was the invoice number again?",
         "Eight eight zero one."),
        ("Say the combination out loud.",
         "Left, right, left."),
        ("Repeat the flight departure time.",
         "Seventeen forty-five."),
        ("What was the recipe's oven setting?",
         "One-eighty Celsius."),
    ], mode="prefix"),
    _ep("fashion_velvet", ", softly as velvet", "fashion and fabrics", [
        ("Which fabric suits the winter overcoat?",
         "Heavy wool twill."),
        ("How do I hem the silk scarf properly?",
         "Use a rolled hem."),
        ("What lining makes the jacket drape well?",
         "Cupro breathes nicely."),
        ("Should the cuffs show beneath the sleeve?",
         "By half an inch."),
        ("How do I store the knitwear in summer?",
         "Folded, with cedar."),
        ("Which thread matches the navy twill?",
         "The slate grey."),
    ], mode="suffix"),
    _ep("map_north", " — true north", "maps and orientation", [
        ("Which way to the fire lookout from the saddle?",
         "Follow the ridge east."),
        ("How do I orient the map at the trailhead?",
         "Match the creek to the blue line."),
        ("Where does the old logging road rejoin the path?",
         "At the second switchback."),
        ("What is the safest route over the pass?",
         "The southern traverse."),
        ("Where is the false summit on this peak?",
         "Just above the scree."),
        ("How far is the shelter from the junction?",
         "Twenty minutes uphill."),
    ], mode="suffix"),
    _ep("poetry_thus", "thus", "poetry and verse", [
        ("How should the sonnet's volta arrive?",
         "At the ninth line."),
        ("What meter suits the pastoral elegy?",
         "Dactylic hexameter."),
        ("Where do the caesurae fall in the ode?",
         "After the fourth foot."),
        ("How tight should the rhyme scheme be?",
         "Alternate the endings."),
        ("Which image carries the final couplet?",
         "The winter orchard."),
        ("How do I break the long closing line?",
         "Enjamb across the turn."),
    ], mode="suffix"),
]


def episode_triples(ep_idx, turns):
    triples = []
    for k, (prompt, stated, correction) in enumerate(turns):
        if correction is None:
            triples.append(TrainingTriple(prompt, stated, +1.0, f"ep{ep_idx}", k,
                                          "explicit_feedback", 1.0))
        else:
            triples.append(TrainingTriple(prompt, stated, -1.0, f"ep{ep_idx}", k,
                                          "explicit_feedback", 1.0))
            triples.append(TrainingTriple(prompt, correction, +1.0, f"ep{ep_idx}", k,
                                          "explicit_correction", 1.0))
    return triples


def episode_home_texts(turns):
    """Home context = prompt + response-as-stated (G1b home definition)."""
    return [prompt + stated for (prompt, stated, _c) in turns]


def poison_triples():
    out = []
    for k, (prompt, resp) in enumerate(POISON_TURNS):
        out.append(TrainingTriple(prompt, resp, +1.0, "poison", k,
                                  "explicit_feedback", 1.0))
        out.append(TrainingTriple(prompt, resp, +1.0, "poison", k,
                                  "explicit_correction", 1.0))
    return out


# ------------------------------------------------------------------
# Masked batched capture (right padding; causal attention => real-token
# features identical to per-sequence forwards)
# ------------------------------------------------------------------


@torch.no_grad()
def capture_grouped(model, id_lists, pad_id):
    """Batched ffn-input capture: each id list is padded to one forward.
    Returns per-layer (N_valid, C) features plus per-group (start, end)
    slices into that axis."""
    device = next(model.parameters()).device
    n_layers = len(model.layers)
    feats = [[] for _ in range(n_layers)]
    masks = []
    for ids in id_lists:
        T = max(int(t.shape[-1]) for t in ids)
        B = len(ids)
        x = torch.full((B, T), pad_id, dtype=torch.long, device=device)
        m = torch.zeros(B, T, dtype=torch.bool, device=device)
        for i, t in enumerate(ids):
            t = t.reshape(-1)
            x[i, :t.shape[0]] = t.to(device)
            m[i, :t.shape[0]] = True
        caps = [None] * n_layers
        handles = []
        for i, layer in enumerate(model.layers):
            def pre(_mod, args, i=i):
                caps[i] = args[0].detach()
            handles.append(layer["ffn"].register_forward_pre_hook(pre))
        try:
            model(x)
        finally:
            for h in handles:
                h.remove()
        keep = m.reshape(-1)
        for i in range(n_layers):
            feats[i].append(caps[i].reshape(-1, caps[i].shape[-1])[keep])
        masks.append(int(keep.sum()))
    per_layer = [torch.cat(chunks, dim=0) for chunks in feats]
    slices = []
    start = 0
    for n in masks:
        slices.append((start, start + n))
        start += n
    return per_layer, slices


# ------------------------------------------------------------------
# Multi-row mutual-exclusion recal (§4.4a joint calibration, incremental)
# ------------------------------------------------------------------

def recal_mex(model, home_id_lists, base_chunks, pad_id, *, steps, lr, device):
    """Router-only recal training ALL plug-in rows jointly, full home
    coverage every step (balanced positive signal per owner).

    Section-4.4a mutual-exclusion target:
      home of episode i -> owner row i gets OWNER_MASS, base rows share
                           (1-OWNER_MASS), every other plug-in row 0
      base-mix tokens   -> base rows share 1.0, plug-in rows 0
    Base rows stay frozen (grad hook).
    """
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
    for step in range(steps):
        bsel = [int(torch.randint(0, len(base_chunks), (1,), generator=gen).item())
                for _ in range(RECAL_BASE)]
        opt.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)

        hs, slices = capture_grouped(model, home_id_lists, pad_id)
        for li, layer in enumerate(model.layers):
            r = layer["ffn"].router
            full = hs[li]
            logits_all = (r.query(full) @ r.passports.t()
                          ) / math.sqrt(r.passport_dim)
            for j in range(n_live):
                s, e = slices[j]
                logits = logits_all[s:e]
                tgt = torch.zeros(logits.size(0), n_rows, device=device)
                tgt[:, :n_base] = (1.0 - OWNER_MASS) / n_base
                tgt[:, first_plug + j] = OWNER_MASS
                loss = loss + -(tgt * F.log_softmax(logits, dim=-1)
                                ).sum(dim=-1).mean()

        hs_b, sl_b = capture_grouped(model, [[base_chunks[i] for i in bsel]],
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
        loss.backward()
        opt.step()
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            print(f"    [recal] step {step:3d}  loss {float(loss):+.4f}")
    for layer in model.layers:
        layer["ffn"].router.passports.requires_grad_(False)
    return {"steps": steps, "lr": lr, "owner_mass": OWNER_MASS,
            "objective": "section-4.4a mutual-exclusion, full home coverage",
            "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------
# Multi-expert routing measurement
# ------------------------------------------------------------------

@torch.no_grad()
def routing_matrix(model, id_lists, row_indices, pad_id):
    """Per-row P(row in top-2) averaged over layers over the groups' tokens,
    plus entropy / plug-in-mass collapse indicators."""
    hs, slices = capture_grouped(model, id_lists, pad_id)
    n_layers = len(model.layers)
    hits = {j: [] for j in row_indices}
    ents, anyp, bothp, maxp = [], [], [], []
    for li, layer in enumerate(model.layers):
        r = layer["ffn"].router
        h = hs[li]
        logits = r.query(h) @ r.passports.t() / math.sqrt(r.passport_dim)
        probs = F.softmax(logits, dim=-1)
        tw, ti = torch.topk(probs, layer["ffn"].top_k, dim=-1)
        for j in row_indices:
            hits[j].append(float((ti == j).any(dim=-1).float().mean()))
        ents.append(float((-(probs * (probs + 1e-12).log()).sum(-1)).mean()))
        plug = torch.tensor(row_indices, dtype=torch.long, device=ti.device)
        pmask = (ti.unsqueeze(-1) == plug).any(-1)
        anyp.append(float(pmask.any(dim=-1).float().mean()))
        bothp.append(float(pmask.all(dim=-1).float().mean()))
        maxp.append(float(tw.max(dim=-1).values.mean()))
    out = {j: sum(v) / n_layers for j, v in hits.items()}
    diag = {"entropy": sum(ents) / n_layers,
            "p_any_plug_in_top2": sum(anyp) / n_layers,
            "p_both_plug_in_top2": sum(bothp) / n_layers,
            "mean_top1_weight": sum(maxp) / n_layers}
    return out, diag


@torch.no_grad()
def routing_matrix_group(model, id_lists, row_indices, pad_id):
    """Like routing_matrix but returns hits per (group, row)."""
    hs, slices = capture_grouped(model, id_lists, pad_id)
    n_layers = len(model.layers)
    per_group = {g: {j: [] for j in row_indices} for g in range(len(id_lists))}
    for li, layer in enumerate(model.layers):
        r = layer["ffn"].router
        h = hs[li]
        logits = r.query(h) @ r.passports.t() / math.sqrt(r.passport_dim)
        ti = torch.topk(logits, layer["ffn"].top_k, dim=-1).indices
        for g, (s, e) in enumerate(slices):
            for j in row_indices:
                per_group[g][j].append(float((ti[s:e] == j).any(dim=-1).float().mean()))
    return {g: {j: sum(v) / n_layers for j, v in d.items()}
            for g, d in per_group.items()}


def bank_norms(model):
    per_layer = [layer["ffn"].router.passports.data.norm(dim=1)
                 for layer in model.layers]
    stacked = torch.stack(per_layer)
    return [round(float(v), 3) for v in stacked.mean(dim=0)]


# ------------------------------------------------------------------

def main():
    t_start = time.time()
    os.makedirs(WORKDIR, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    assert len(EPISODES) >= N_EPS, f"need {N_EPS} episodes, have {len(EPISODES)}"

    def section(s):
        print("\n" + "=" * 72 + f"\n{s}\n" + "=" * 72)

    print(f"G3 coexistence  N={N_EPS}  device={DEVICE}  cons_steps={CONS_STEPS} "
          f"recal_steps={RECAL_STEPS}  text_kl={TEXT_KL_COEF}  seed={SEED}")

    section("PART 0 — load passport-native spine (lab_small, dropout-0.1)")
    model, tok, mc = load_base_model(CKPT, device=DEVICE)
    g1.DIM = int(mc["dim"])
    g1.HIDDEN = int(mc["hidden_dim"])
    g1.N_LAYERS = int(mc["n_layers"])
    g1.BASE_EXPERTS = int(mc["num_experts"])
    g1.TEXT_KL_COEF = TEXT_KL_COEF
    g1.WORKDIR = WORKDIR
    rinfo = g1.prepare_routers(model)
    pad_id = tok.pad_token_id
    print(f"  checkpoint: {CKPT}")
    print(f"  model_config: dim={mc['dim']} layers={mc['n_layers']} "
          f"hidden={mc['hidden_dim']} experts={mc['num_experts']} top_k={mc['top_k']}")
    print(f"  router prep: mode={rinfo['mode']} passport_dim={rinfo['passport_dim']} "
          f"expert_dropout={rinfo['expert_dropout']} max|logit diff|={rinfo['max_logit_diff']:.3e}")
    assert rinfo["mode"] == "passport-native", "G3 needs a passport-native spine"
    assert 0.1 in rinfo["expert_dropout"], "expected the dropout-0.1 spine"
    first_plug = g1.BASE_EXPERTS

    base_mix = g1.load_base_mix_chunks(n_chunks=32, seq=256, seed=SEED)
    ce_pre = g1.next_token_ce(model, base_mix)
    print(f"  pre-library base-mix CE: {ce_pre:.4f}  "
          f"({len(base_mix)} x 256-token chunks from Pretrain/data val tail)")

    eps = EPISODES[:N_EPS]
    ep_triples = [episode_triples(j, turns) for j, (_, turns) in enumerate(eps)]
    ep_home_texts = [episode_home_texts(turns) for _, turns in eps]
    ep_home_ids = [g1.encode_texts(tok, t, DEVICE) for t in ep_home_texts]
    n_home_tok = [sum(int(t.shape[-1]) for t in ids) for ids in ep_home_ids]
    print(f"  {N_EPS} episodes x 6 turns, {len(ep_triples[0])} triples each, "
          f"home tokens/ep {min(n_home_tok)}-{max(n_home_tok)}")
    names = [n for n, _ in eps]
    print(f"  episodes: {', '.join(names)}")

    inserts = []
    live_rows = []

    for j, (name, turns) in enumerate(eps):
        section(f"INSERT {j:2d}/{N_EPS} — episode '{name}'")
        experts = [g1.ContractExpert(g1.DIM, g1.HIDDEN).to(DEVICE)
                   for _ in range(g1.N_LAYERS)]
        for i, e in enumerate(experts):
            g1.birth_from_base(e, model.layers[i]["ffn"].experts[0])
        t_c = time.time()
        stats = g1.train_expert(model, tok, ep_triples[j], experts,
                                steps=CONS_STEPS, lr=CONS_LR, device=DEVICE,
                                tag=f"ep{j}", text_batches=base_mix[:4])
        cons_s = round(time.time() - t_c, 1)

        # retention spot-check under forced dispatch: paired NLL gain
        # (NLL(stated) - NLL(correction)) on the 3 corrections.
        with torch.no_grad():
            encs = [(t, e) for t, e in
                    ((t, _encode_triple(tok, t, 256)) for t in ep_triples[j])
                    if e]
            gains = []
            for k in range(3):
                e_bad, e_good = encs[2 * k][1], encs[2 * k + 1][1]
                vals = []
                for e in (e_bad, e_good):
                    x1 = torch.tensor([e["x"]], dtype=torch.long, device=DEVICE)
                    y1 = torch.tensor([e["y"]], dtype=torch.long, device=DEVICE)
                    m1 = torch.tensor([e["mask"]], dtype=torch.float32,
                                      device=DEVICE)
                    with g1.forced_experts(model, experts):
                        lg = model(x1)[0][0].float()
                    nll = -F.log_softmax(lg, dim=-1).gather(-1, y1.T).squeeze(-1)
                    vals.append(float((nll * m1.squeeze(0)).sum()
                                      / m1.sum().clamp(min=1.0)))
                gains.append(round(vals[0] - vals[1], 4))

        verdict, gstats = g1.run_gate(f"ep{j}", ep_triples[j], stats)

        # prototype-init passport = mean router query over home examples
        with torch.no_grad():
            home_h, _ = capture_grouped(model, [ep_home_ids[j]], pad_id)
            rows = [model.layers[i]["ffn"].router.query(home_h[i]).mean(dim=0)
                    for i in range(g1.N_LAYERS)]
        g1.plug_expert(model, experts, rows)
        live_rows.append(j)
        print(f"  plugged row {first_plug + j} (E={model.layers[0]['ffn'].num_experts})")

        rstat = recal_mex(model, [ep_home_ids[k] for k in live_rows], base_mix,
                          pad_id, steps=RECAL_STEPS, lr=RECAL_LR, device=DEVICE)

        # measure — all live experts, own home and everyone's home (cross)
        n_rows = model.layers[0]["ffn"].router.passports.shape[0]
        plug_rows = list(range(first_plug, n_rows))
        home_lists = [ep_home_ids[k] for k in live_rows]
        per_group = routing_matrix_group(model, home_lists, plug_rows, pad_id)
        util, cross = {}, {}
        for gi, k in enumerate(live_rows):
            util[k] = round(per_group[gi][first_plug + k], 4)
            cross[k] = {live_rows[gj]: round(per_group[gj][first_plug + k], 4)
                        for gj in range(len(live_rows)) if live_rows[gj] != k}
        hits_b, diag_base = routing_matrix(model, [base_mix[:8][i:i + 1]
                                                   for i in range(8)],
                                           plug_rows, pad_id)
        contam = {k: round(hits_b[first_plug + k], 4) for k in live_rows}
        _, diag_home = routing_matrix(model, home_lists, plug_rows, pad_id)
        ce_now = g1.next_token_ce(model, base_mix)
        ce_reg = 100.0 * (ce_now - ce_pre) / ce_pre
        norms = bank_norms(model)

        ok_util = all(v > 0.5 for v in util.values())
        ok_contam = all(v < 0.3 for v in contam.values())
        ok_ce = ce_reg < 1.0
        rec = {
            "n": j + 1,
            "episode": name,
            "gate": verdict.decision,
            "gate_metrics": verdict.metrics,
            "consolidate_seconds": cons_s,
            "recal_seconds": rstat["seconds"],
            "train": {"loss0": stats["loss0"], "loss_final": stats["loss_final"],
                      "final_kl": stats["final_kl"],
                      "final_text_kl": stats.get("final_text_kl")},
            "forced_nll_gain_first3": gains,
            "util_home": util,
            "contam_base": contam,
            "crosstalk_home": cross,
            "base_ce": round(ce_now, 4),
            "base_ce_regression_pct": round(ce_reg, 3),
            "bank_norms": norms,
            "router_diag_home": {k: round(v, 4) for k, v in diag_home.items()},
            "router_diag_base": {k: round(v, 4) for k, v in diag_base.items()},
            "pass_util": ok_util, "pass_contam": ok_contam, "pass_ce": ok_ce,
        }
        inserts.append(rec)
        print(f"  util_home: " + " ".join(f"e{k}={util[k]:.3f}" for k in live_rows))
        print(f"  contam   : " + " ".join(f"e{k}={contam[k]:.3f}" for k in live_rows))
        print(f"  base CE  : {ce_now:.4f} ({ce_reg:+.3f}%)  "
              f"entropy base {diag_base['entropy']:.3f}  "
              f"p_any_plug {diag_base['p_any_plug_in_top2']:.3f}  "
              f"p_both_plug {diag_base['p_both_plug_in_top2']:.3f}")
        print(f"  nll gains: {gains}  gate={verdict.decision}  "
              f"cons={cons_s}s recal={rstat['seconds']}s  "
              f"pass(u/c/ce)={ok_util}/{ok_contam}/{ok_ce}")

        with open(OUT, "w") as f:
            json.dump({"gate": "G3", "status": "in_progress", "inserts": inserts},
                      f, indent=2)

    # ---- §4.4a final joint calibration: re-partition ALL live rows together
    # (removes the insertion-order training bias of per-insert recal alone).
    post_joint = None
    if FINAL_JOINT_STEPS > 0:
        section(f"PART 0.5 — final joint calibration ({FINAL_JOINT_STEPS} steps)")
        rstat = recal_mex(model, [ep_home_ids[k] for k in live_rows], base_mix,
                          pad_id, steps=FINAL_JOINT_STEPS, lr=RECAL_LR,
                          device=DEVICE)
        n_rows = model.layers[0]["ffn"].router.passports.shape[0]
        plug_rows = list(range(first_plug, n_rows))
        home_lists = [ep_home_ids[k] for k in live_rows]
        per_group = routing_matrix_group(model, home_lists, plug_rows, pad_id)
        util, cross = {}, {}
        for gi, k in enumerate(live_rows):
            util[k] = round(per_group[gi][first_plug + k], 4)
            cross[k] = {live_rows[gj]: round(per_group[gj][first_plug + k], 4)
                        for gj in range(len(live_rows)) if live_rows[gj] != k}
        hits_b, diag_base = routing_matrix(model, [[base_mix[i]] for i in range(8)],
                                           plug_rows, pad_id)
        contam = {k: round(hits_b[first_plug + k], 4) for k in live_rows}
        _, diag_home = routing_matrix(model, home_lists, plug_rows, pad_id)
        ce_now = g1.next_token_ce(model, base_mix)
        ce_reg = 100.0 * (ce_now - ce_pre) / ce_pre
        ok_util = all(v > 0.5 for v in util.values())
        ok_contam = all(v < 0.3 for v in contam.values())
        ok_ce = ce_reg < 1.0
        post_joint = {
            "n": "final_joint", "episode": "all-live-joint",
            "recal_seconds": rstat["seconds"],
            "util_home": util, "contam_base": contam, "crosstalk_home": cross,
            "base_ce": round(ce_now, 4),
            "base_ce_regression_pct": round(ce_reg, 3),
            "bank_norms": bank_norms(model),
            "router_diag_home": {k: round(v, 4) for k, v in diag_home.items()},
            "router_diag_base": {k: round(v, 4) for k, v in diag_base.items()},
            "pass_util": ok_util, "pass_contam": ok_contam, "pass_ce": ok_ce,
        }
        print(f"  util_home: " + " ".join(f"e{k}={util[k]:.3f}" for k in live_rows))
        print(f"  contam   : " + " ".join(f"e{k}={contam[k]:.3f}" for k in live_rows))
        print(f"  base CE  : {ce_now:.4f} ({ce_reg:+.3f}%)  "
              f"p_any_plug {diag_base['p_any_plug_in_top2']:.3f}  "
              f"p_both_plug {diag_base['p_both_plug_in_top2']:.3f}")
        print(f"  pass(u/c/ce)={ok_util}/{ok_contam}/{ok_ce}")

    final = post_joint if post_joint is not None else inserts[-1]
    all_util = all(r["pass_util"] for r in inserts)
    all_contam = all(r["pass_contam"] for r in inserts)
    all_ce = all(r["pass_ce"] for r in inserts)
    g3_pass = (final["pass_util"] and final["pass_contam"] and final["pass_ce"])
    first_break = None
    for r in inserts:
        if not (r["pass_util"] and r["pass_contam"] and r["pass_ce"]):
            first_break = r["n"]
            break

    section("PART 1 — poison spot-check at N")
    incumbent_hash = g1.state_hash(model)
    ce_incumbent = g1.next_token_ce(model, base_mix)
    ptriples = poison_triples()
    pexperts = [g1.ContractExpert(g1.DIM, g1.HIDDEN).to(DEVICE)
                for _ in range(g1.N_LAYERS)]
    for i, e in enumerate(pexperts):
        g1.birth_from_base(e, model.layers[i]["ffn"].experts[0])
    pstats = g1.train_expert(model, tok, ptriples, pexperts,
                             steps=max(24, CONS_STEPS // 2), lr=CONS_LR,
                             device=DEVICE, tag="poison",
                             text_batches=base_mix[:4])
    pverdict, pgstats = g1.run_gate("poison", ptriples, pstats)
    with torch.no_grad():
        ph, _ = capture_grouped(
            model, [g1.encode_texts(tok, [p + r for p, r in POISON_TURNS], DEVICE)],
            pad_id)
    prows = []
    for i in range(g1.N_LAYERS):
        q = model.layers[i]["ffn"].router.query(ph[i]).mean(dim=0)
        bank = model.layers[i]["ffn"].router.passports.data
        prows.append(q * (float(bank.norm(dim=1).mean()) / (float(q.norm()) + 1e-8)))
    g1.plug_expert(model, pexperts, prows)
    poison_row = model.layers[0]["ffn"].num_experts - 1
    ce_poison = g1.next_token_ce(model, base_mix)
    hits_p, _ = routing_matrix(model, [[base_mix[i]] for i in range(8)],
                               [poison_row], pad_id)
    g1.drop_last_expert(model)
    post_hash = g1.state_hash(model)
    ce_restored = g1.next_token_ce(model, base_mix)
    hash_ok = (post_hash == incumbent_hash)
    ce_ok = abs(ce_restored - ce_incumbent) < 1e-6
    print(f"  gate: {pverdict.decision} — {pverdict.reason}")
    print(f"  poison row plugged: base CE {ce_incumbent:.4f} -> {ce_poison:.4f} "
          f"(contam {hits_p[poison_row]:.3f})")
    print(f"  row drop -> state hash {'MATCH' if hash_ok else 'MISMATCH'}, "
          f"base CE restored {'exact' if ce_ok else 'DRIFTED'} "
          f"({ce_restored:.6f} vs {ce_incumbent:.6f})")

    section("PART 2 — G3 verdict")
    print(f"  N = {N_EPS}")
    print(f"  (util_home>0.5 all inserts): {'PASS' if all_util else 'FAIL'}  "
          f"final {final['util_home']}")
    print(f"  (contam_base<0.3 all inserts): {'PASS' if all_contam else 'FAIL'}  "
          f"final {final['contam_base']}")
    print(f"  (base CE reg <1%): {'PASS' if all_ce else 'FAIL'}  "
          f"final {final['base_ce_regression_pct']:+.3f}%")
    print(f"  first purity break at N={first_break}" if first_break else
          "  purity never broke")
    print(f"  poison row-drop byte-identical: {hash_ok}")
    print(f"  G3 OVERALL: {'PASS' if g3_pass else 'FAIL'}")
    print(f"  recal cost/insert (s): "
          + " ".join(str(r['recal_seconds']) for r in inserts))
    print(f"  cons cost/insert (s):  "
          + " ".join(str(r['consolidate_seconds']) for r in inserts))

    out = {
        "gate": "G3",
        "verdict": "PASS" if g3_pass else "FAIL",
        "n_episodes": N_EPS,
        "spine": {
            "checkpoint": CKPT, "mode": rinfo["mode"],
            "expert_dropout": rinfo["expert_dropout"],
            "passport_dim": rinfo["passport_dim"],
            "dim": mc["dim"], "n_layers": mc["n_layers"],
            "num_experts_base": mc["num_experts"], "top_k": mc["top_k"],
            "plan_change": "§8.4 G3 text said t2/tiny_agent_k; G1 showed the "
                           "transplant fails purity so G3 runs on the "
                           "passport-native lab_small spine (G1b recipe)",
        },
        "recipe": {
            "consolidation_steps": CONS_STEPS, "consolidation_lr": CONS_LR,
            "text_kl_coef": TEXT_KL_COEF,
            "recal_steps": RECAL_STEPS, "recal_lr": RECAL_LR,
            "recal_objective": "section-4.4a mutual-exclusion mass target, "
                               "all plug-in rows trained jointly with full "
                               "home coverage, base rows frozen",
            "owner_mass": OWNER_MASS,
            "final_joint_steps": FINAL_JOINT_STEPS,
            "prototype_init": "mean router query over the episode's home examples",
            "fixture": "16 episodes, distinct prompt domains + distinct taught "
                       "styles (episode 0 = G1's math/words-not-digits set)",
        },
        "pre_library_base_ce": round(ce_pre, 4),
        "criteria": {
            "util_home_gt_0.5": {"pass": final["pass_util"],
                                 "final": final["util_home"],
                                 "bar": ">0.5 for every expert at N=16",
                                 "all_inserts_pass": all_util},
            "contam_base_lt_0.3": {"pass": final["pass_contam"],
                                   "final": final["contam_base"],
                                   "bar": "<0.3 for every expert at N=16",
                                   "all_inserts_pass": all_contam},
            "base_ce_regression_lt_1pct": {
                "pass": final["pass_ce"],
                "final_pct": final["base_ce_regression_pct"],
                "final_ce": final["base_ce"], "pre_ce": round(ce_pre, 4),
                "bar": "<1% vs pre-library at N=16",
                "all_inserts_pass": all_ce},
        },
        "first_purity_break_n": first_break,
        "post_final_joint": post_joint,
        "purity_trajectory": [
            {"n": r["n"], "episode": r["episode"],
             "util_min": round(min(r["util_home"].values()), 4),
             "contam_max": round(max(r["contam_base"].values()), 4),
             "ce_reg_pct": r["base_ce_regression_pct"],
             "entropy_base": r["router_diag_base"]["entropy"],
             "p_any_plug_base": r["router_diag_base"]["p_any_plug_in_top2"],
             "p_both_plug_base": r["router_diag_base"]["p_both_plug_in_top2"],
             "bank_norms": r["bank_norms"],
             "recal_s": r["recal_seconds"], "cons_s": r["consolidate_seconds"]}
            for r in inserts
        ],
        "poison_check": {
            "gate_decision": pverdict.decision, "gate_reason": pverdict.reason,
            "ce_before": round(ce_incumbent, 4),
            "ce_poison_plugged": round(ce_poison, 4),
            "poison_row_contam_base": hits_p[poison_row],
            "row_drop_hash_match": hash_ok, "row_drop_ce_restored": ce_ok,
        },
        "inserts": inserts,
        "shortcuts": [
            "consolidation 80 steps/expert (G1b parity); G3's bar is "
            "coexistence purity, not per-expert quality",
            "per-expert margin probes replaced by a 3-pair forced-NLL "
            "retention spot-check (forced_nll_gain_first3)",
            "v1 fixture (one shared prompt template, one taught style) "
            "aborted at N=6 — see lab/results/g3_n16_v1_weaksep.json",
        ],
        "seconds": round(time.time() - t_start, 1),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")
    print(f"total time: {out['seconds']}s")


if __name__ == "__main__":
    main()
