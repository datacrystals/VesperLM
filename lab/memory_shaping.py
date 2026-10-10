#!/usr/bin/env python3
"""memory_shaping — episodic memories as token-substitution deltas.

The G3 addressing-pivot work (lab/g3_pivot_runs.py, 2026-10-09) showed that
MARGIN RETENTION at N=8 is predicted by the memory SHAPE, not by passport-bank
crowding or util: token-substitution facts (stated digit form -> corrected word
form) retain 0.884 of solo delivery at N=8 while tag-append facts retain 0.390
at identical crowding/recipe.  This module makes the winning shape a first-class
reusable contract so consolidation never again receives raw append-style
episodes by accident.

Delta contract
--------------
A taught fact is a TOKEN-SUBSTITUTION DELTA:

    stated     = prefix + BAD  + suffix      (the rejected continuation)
    correction = prefix + GOOD + suffix      (the taught continuation)

i.e. the memory is "on prompt P, substitute token BAD with GOOD".  Tag-append
memories (correction = stated + tag) are the fragile shape and are classified
separately so a caller can see what it is feeding consolidation:

    kind = "substitution"   both middles non-empty   (the winning shape)
    kind = "insert"         correction inserts text  (fragile: tag-append)
    kind = "delete"         correction drops text
    kind = "identical"      no divergence

Episode = one consolidation unit: 3 taught deltas (reject+correction turns)
+ 3 already-styled approvals -> 9 training triples (the G1 teach-set shape).

Storage/contraction API
-----------------------
- TokenDelta / Episode        — the storage schema
- contract_turn()             — classify + contract raw (prompt, stated,
                                correction) turns into deltas (shaping step)
- word_number_episode()       — builder for the winning shape
- SHAPED_8 / SHAPED_16        — the coexistence fixtures (8/16 distinct prompt
                                domains, all substitution deltas)
- shape_report() / dedup_check() — cheap validation before consolidation
  (the "shape or dedup episodes before consolidation" discipline)

Consolidation API (wraps the G1b-winning recipe; nothing here is a one-off)
-------------------------------------------------------------------------
- episode_triples / episode_home_texts / episode_home_ids
- consolidate_episode()  — forced-dispatch + reward-weighted NLL + KL-to-base
  + section-4.5 text-KL base-neutrality -> contract expert + prototype rows
  (mean router query over home examples)
- plug_and_recal()       — add_expert + section-4.4a mex recal
- delta_margin() / episode_margins() / solo_reference() / retention()
  — the standing G3 metric (margin retention vs solo >= 0.70)

Everything operates on a loaded VesperLinearLM spine with g1/g3 conventions
(lab/g1_export_mode.py, lab/g3_coexistence.py).
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "lab"))
sys.path.insert(0, os.path.join(REPO, "Hippocampus"))
sys.path.insert(0, os.path.join(REPO, "Common"))

import g1_export_mode as g1  # noqa: E402
import g3_coexistence as g3  # noqa: E402
from consolidate import response_nll, _encode_triple  # noqa: E402
from session_log import TrainingTriple  # noqa: E402

RETENTION_BAR = 0.70   # standing G3 bar: lib gain / solo gain

_W = ["zero", "one", "two", "three", "four", "five", "six", "seven",
      "eight", "nine", "ten", "eleven", "twelve"]


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

@dataclass
class TokenDelta:
    """One taught fact as a token-substitution delta (see module docstring)."""
    prompt: str
    prefix: str
    bad: str
    good: str
    suffix: str
    kind: str = "substitution"

    @property
    def stated(self) -> str:
        return self.prefix + self.bad + self.suffix

    @property
    def correction(self) -> str:
        return self.prefix + self.good + self.suffix

    @property
    def is_substitution(self) -> bool:
        return self.kind == "substitution"

    def as_dict(self) -> dict:
        return {"prompt": self.prompt, "prefix": self.prefix, "bad": self.bad,
                "good": self.good, "suffix": self.suffix, "kind": self.kind}


@dataclass
class Episode:
    """One consolidation unit: 3 taught deltas + 3 styled approvals."""
    name: str
    deltas: List[TokenDelta]
    approvals: List[Tuple[str, str]]   # (prompt, styled response)

    def turns(self):
        """g3-compatible (prompt, stated, correction|None) view."""
        out = [(d.prompt, d.stated, d.correction) for d in self.deltas]
        out += [(p, r, None) for (p, r) in self.approvals]
        return out


def contract_turn(prompt: str, stated: str, correction: Optional[str]
                  ) -> Optional[TokenDelta]:
    """Shape one raw teach turn into a delta and classify it.

    Longest common prefix / suffix decomposition; the middles are the
    substitution spans.  correction None (approval) -> None.
    """
    if correction is None or correction == stated:
        return None
    p = 0
    while p < min(len(stated), len(correction)) and stated[p] == correction[p]:
        p += 1
    s = 0
    while (s < min(len(stated), len(correction)) - p
           and stated[len(stated) - 1 - s] == correction[len(correction) - 1 - s]):
        s += 1
    bad = stated[p:len(stated) - s if s else len(stated)]
    good = correction[p:len(correction) - s if s else len(correction)]
    if bad and good:
        kind = "substitution"
    elif not bad and good:
        kind = "insert"
    elif bad and not good:
        kind = "delete"
    else:
        kind = "identical"
    return TokenDelta(prompt, stated[:p], bad, good,
                      stated[len(stated) - s:] if s else "", kind)


def word_number_episode(name: str, rows: Sequence[Tuple[str, int]]) -> Episode:
    """The winning shape: stated digit form -> corrected word form.

    rows: 6 x (prompt, n); turns 0-2 taught (reject digit -> correct to word),
    turns 3-5 arrive styled (approve)."""
    deltas, approvals = [], []
    for k, (prompt, n) in enumerate(rows):
        digit = f"{n}."
        word = f"{_W[n].capitalize()}."
        if k < 3:
            d = contract_turn(prompt, digit, word)
            assert d is not None and d.is_substitution, (name, k, d)
            deltas.append(d)
        else:
            approvals.append((prompt, word))
    return Episode(name, deltas, approvals)


def episode_triples(ep: Episode) -> List[TrainingTriple]:
    """9 G1-shaped training triples per episode."""
    out = []
    for k, d in enumerate(ep.deltas):
        out.append(TrainingTriple(d.prompt, d.stated, -1.0, ep.name, k,
                                  "explicit_feedback", 1.0))
        out.append(TrainingTriple(d.prompt, d.correction, +1.0, ep.name, k,
                                  "explicit_correction", 1.0))
    for k, (prompt, styled) in enumerate(ep.approvals, start=len(ep.deltas)):
        out.append(TrainingTriple(prompt, styled, +1.0, ep.name, k,
                                  "explicit_feedback", 1.0))
    return out


def episode_home_texts(ep: Episode) -> List[str]:
    """Home context = prompt + response-as-stated (g3 convention)."""
    return [d.prompt + d.stated for d in ep.deltas] + \
           [p + r for (p, r) in ep.approvals]


def shape_report(episodes: Sequence[Episode]) -> Dict[str, int]:
    """Cheap pre-consolidation validation: what shapes are in the batch?"""
    rep = {"substitution": 0, "insert": 0, "delete": 0, "identical": 0,
           "approvals": 0}
    for ep in episodes:
        for d in ep.deltas:
            rep[d.kind] = rep.get(d.kind, 0) + 1
        rep["approvals"] += len(ep.approvals)
    return rep


def dedup_check(episodes: Sequence[Episode]) -> List[str]:
    """Flag exact prompt collisions across episodes (separability hazard)."""
    seen, dupes = {}, []
    for ep in episodes:
        for d in ep.deltas:
            if d.prompt in seen and seen[d.prompt] != ep.name:
                dupes.append(f"{ep.name}!={seen[d.prompt]}: {d.prompt!r}")
            seen[d.prompt] = ep.name
    return dupes


# ------------------------------------------------------------------
# Fixtures: 8 / 16 distinct prompt domains, all substitution deltas
# ------------------------------------------------------------------

SHAPED_8 = [
    word_number_episode("math_words", [
        ("What is 2 + 2?", 4), ("What is 5 + 3?", 8), ("What is 10 minus 7?", 3),
        ("What is 1 + 1?", 2), ("What is 4 + 4?", 8), ("What is 9 minus 6?", 3)]),
    word_number_episode("cooking_counts", [
        ("How many eggs go into the cake?", 3),
        ("How many minutes should the bread rest?", 5),
        ("How many cloves of garlic in the sauce?", 2),
        ("How many layers does the lasagna have?", 3),
        ("How many cups of stock for the risotto?", 4),
        ("How many hours does the stew simmer?", 6)]),
    word_number_episode("astronomy_counts", [
        ("How many moons does Mars have?", 2),
        ("How many planets are gas giants?", 4),
        ("How many stars make the Big Dipper?", 7),
        ("How many phases does the Moon go through?", 8),
        ("How many rings does Saturn show?", 7),
        ("How many minutes for sunlight to reach us?", 8)]),
    word_number_episode("music_counts", [
        ("How many strings are on a violin?", 4),
        ("How many notes make a major scale?", 7),
        ("How many players in a string quartet?", 4),
        ("How many beats are in a waltz measure?", 3),
        ("How many pedals does a piano have?", 3),
        ("How many movements in a typical symphony?", 4)]),
    word_number_episode("sports_counts", [
        ("How many players start on a soccer team?", 11),
        ("How many points is a touchdown worth?", 6),
        ("How many sets decide a tennis match?", 3),
        ("How many quarters are in a basketball game?", 4),
        ("How many bases are on a baseball diamond?", 4),
        ("How many rings are on the Olympic flag?", 5)]),
    word_number_episode("anatomy_counts", [
        ("How many chambers are in the human heart?", 4),
        ("How many bones are in the adult ear?", 3),
        ("How many fingers does one hand have?", 5),
        ("How many lungs does a person have?", 2),
        ("How many vertebrae are in the neck?", 7),
        ("How many ribs enclose the chest?", 12)]),
    word_number_episode("computing_counts", [
        ("How many bits are in a byte?", 8),
        ("How many bytes make a kilobyte in binary math?", 10),
        ("How many pins are on a classic VGA cable?", 9),
        ("How many sides does a hexadecimal digit reach?", 6),
        ("How many keys are on a function row?", 12),
        ("How many bits fit in a nibble?", 4)]),
    word_number_episode("geology_counts", [
        ("How many continents are there on Earth?", 7),
        ("How many oceans cover the planet?", 5),
        ("How many plates move the crust?", 8),
        ("How many layers make up the atmosphere?", 5),
        ("How many volcanoes form the Cascade arc?", 7),
        ("How many minerals define hardness ten?", 10)]),
]

SHAPED_16 = SHAPED_8 + [
    word_number_episode("history_counts", [
        ("How many wives did Henry the Eighth have?", 6),
        ("How many hills did ancient Rome sit upon?", 7),
        ("How many wonders of the ancient world were there?", 7),
        ("How many days did the first Olympic games last?", 5),
        ("How many decades did the Renaissance span?", 3),
        ("How many ships were in the first Armada wave?", 8)]),
    word_number_episode("travel_counts", [
        ("How many days should I allow for the coastal drive?", 5),
        ("How many nights is the layover in Lisbon?", 2),
        ("How many bags can I check on the flight?", 2),
        ("How many stops does the mountain train make?", 7),
        ("How many weeks does the visa last?", 9),
        ("How many euros is the city bus fare?", 3)]),
    word_number_episode("weather_counts", [
        ("How many months is the hurricane season?", 6),
        ("How many sides does a snowflake show?", 6),
        ("How many clouds are in the storm front?", 4),
        ("How many degrees does the temperature drop overnight?", 8),
        ("How many days of rain make a wet week?", 5),
        ("How many seasons does the monsoon span?", 2)]),
    word_number_episode("money_counts", [
        ("How many coins are in a roll of quarters?", 11),
        ("How many dollars is the entry fee?", 5),
        ("How many percent is the service charge?", 10),
        ("How many bills are in the envelope?", 3),
        ("How many installments does the plan have?", 6),
        ("How many cents make a dime?", 10)]),
    word_number_episode("chess_counts", [
        ("How many pawns does each side start with?", 8),
        ("How many squares are on one rank?", 8),
        ("How many knights does each player have?", 2),
        ("How many points is the queen worth?", 9),
        ("How many moves is castling?", 1),
        ("How many files does the board have?", 8)]),
    word_number_episode("ocean_counts", [
        ("How many arms does an octopus have?", 8),
        ("How many gills does a typical shark show?", 5),
        ("How many oceans touch the continent?", 3),
        ("How many tides does the coast see per day?", 2),
        ("How many fathoms make the shallow shelf?", 6),
        ("How many species of sea turtle exist?", 7)]),
    word_number_episode("forest_counts", [
        ("How many seasons does the oak take to mature?", 8),
        ("How many needles are in a pine cluster?", 5),
        ("How many layers are in the forest canopy?", 4),
        ("How many rings mark a decade of growth?", 10),
        ("How many birds flock in the winter murmuration?", 12),
        ("How many centimetres of mulch protect the roots?", 5)]),
    word_number_episode("bridge_counts", [
        ("How many main cables hold the suspension span?", 2),
        ("How many towers support the crossing?", 2),
        ("How many lanes does the bridge carry?", 6),
        ("How many years did the construction take?", 4),
        ("How many bolts anchor each girder?", 7),
        ("How many metres above the river is the deck?", 9)]),
]


# ------------------------------------------------------------------
# Standing measurement: delta margins + retention vs solo
# ------------------------------------------------------------------

@torch.no_grad()
def delta_margin(model, tok, delta: TokenDelta, device) -> Optional[float]:
    """logit(good token) - logit(bad token) at first divergence.

    Encoding matches _encode_triple (prompt ids + response ids concatenated),
    exactly as the G1/G1b/G3 runs measured it.  Positive = taught preferred.
    """
    ps = (tok(delta.prompt, add_special_tokens=False).input_ids
          + tok(delta.stated, add_special_tokens=False).input_ids)
    pc = (tok(delta.prompt, add_special_tokens=False).input_ids
          + tok(delta.correction, add_special_tokens=False).input_ids)
    d = 0
    while d < min(len(ps), len(pc)) and ps[d] == pc[d]:
        d += 1
    if d >= len(ps) or d >= len(pc):
        return None
    x = torch.tensor([ps[:d]], dtype=torch.long, device=device)
    if x.numel() == 0:
        return None
    logits = model(x)[0][0, -1].float()
    return float(logits[pc[d]] - logits[ps[d]])


@torch.no_grad()
def episode_margins(model, tok, ep: Episode, device) -> dict:
    out = []
    for d in ep.deltas:
        m = delta_margin(model, tok, d, device)
        nll_bad = response_nll(model, tok, d.prompt, d.stated, device)
        nll_good = response_nll(model, tok, d.prompt, d.correction, device)
        out.append({"prompt": d.prompt, "kind": d.kind,
                    "logit_margin": None if m is None else round(m, 4),
                    "nll_margin": round(nll_bad - nll_good, 4)})
    lm = [r["logit_margin"] for r in out if r["logit_margin"] is not None]
    return {"facts": out,
            "mean_logit_margin": round(sum(lm) / len(lm), 4) if lm else None,
            "mean_nll_margin": round(
                sum(r["nll_margin"] for r in out) / len(out), 4)}


def retention(lib_gain: float, solo_gain: float) -> Optional[float]:
    return None if not solo_gain else lib_gain / solo_gain


# ------------------------------------------------------------------
# Consolidation (G1b recipe) — reusable, state-dict friendly
# ------------------------------------------------------------------

def consolidate_episode(model, tok, ep: Episode, base_mix, *, steps, lr,
                        device, tag, text_kl_coef=3.0):
    """Contract-expert consolidation: forced dispatch + reward-weighted NLL
    + KL-to-base + text-KL base-neutrality; returns (experts, rows, stats)
    ready for plug_and_recal.  Gate (stub) recorded, PROMOTE expected."""
    g1.TEXT_KL_COEF = text_kl_coef
    experts = [g1.ContractExpert(g1.DIM, g1.HIDDEN).to(device)
               for _ in range(g1.N_LAYERS)]
    for i, e in enumerate(experts):
        g1.birth_from_base(e, model.layers[i]["ffn"].experts[0])
    triples = episode_triples(ep)
    stats = g1.train_expert(model, tok, triples, experts,
                            steps=steps, lr=lr, device=device, tag=tag,
                            text_batches=base_mix[:4])
    with torch.no_grad():
        ids = episode_home_ids(tok, ep, device)
        home_h, _ = g3.capture_grouped(model, [ids], g1_pad_id(tok))
        rows = [model.layers[i]["ffn"].router.query(home_h[i]).mean(dim=0)
                for i in range(g1.N_LAYERS)]
    verdict, gstats = g1.run_gate(tag, triples, stats)
    return experts, rows, {"train": stats, "gate": verdict.decision,
                           "gate_metrics": verdict.metrics}


def episode_home_ids(tok, ep: Episode, device):
    return g1.encode_texts(tok, episode_home_texts(ep), device)


def g1_pad_id(tok):
    return tok.pad_token_id


def plug_and_recal(model, experts, rows, home_id_lists, base_chunks, *,
                   recal_steps, recal_lr, device, pad_id):
    """add_expert with prototype rows + section-4.4a mex recal (g3.recal_mex)."""
    g1.plug_expert(model, experts, rows)
    rstat = g3.recal_mex(model, home_id_lists, base_chunks, pad_id,
                         steps=recal_steps, lr=recal_lr, device=device)
    return rstat


def state_dict_cpu(model):
    return {k: v.cpu() for k, v in model.state_dict().items()}
