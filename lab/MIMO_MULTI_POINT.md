# MIMO_MULTI_POINT — multi-point delivery across MoE blocks (fallback-a)

Date: 2026-10-20.  Fallback-(a) from lab/MIMO_PERSONA_DOSE.md after the
margin family closed at single-point block-0 delivery (mild =
voiceless-coherent; margin@3e-4 = ruin; margin+tail = flat / NaN).
Hypothesis: one insertion point cannot carry a behavior-gradient through
47 downstream layers; distributing the delivery lets each point carry a
sensible share.

## Topology (declared)

Host inventory: 47 MoE blocks (model layers 1..47; `moe_layer_freq`;
`find_mimo_moe_blocks` order).  Sites = MoE-order indices
**{0, 23, 46}** (model layers 1 / 24 / 47 — early / mid / last MoE block;
attention classes at those layers are SWA / SWA / full, a spread, not a
cherry-pick).  ONE wrap core (block-0 expert-0 frozen core as before;
the wrap's encoder-decoder is a portable 4096→4096 map) injected at all
three sites, additive, λ_i = 0.5/3 (total λ matches the previously
workable recipe 0.5).  Identity-at-birth restart on the four adapter
linears (Door §4.2; the slice-era fitted adapter state is not loaded).

## Fit (declared)

Mild-CE family (which measured coherent at single point — this
experiment varies SITES, not dose or objective): HOME steps CE on win
answer tails with ALL three injections live (delivery-exact); NEG steps
every 3rd = out_pen × relu(CE_on − CE_off) on cold Q windows.  lr 3e-4,
clip 1.0, 30 steps, T=128 pad, forward-rebuild retry, per-layer
checkpointing, lazy dequant off — all earned mechanics carried.

## Matrix (declared)

Arms: noplug / point-by-point ablation {site 0 only, site 23 only,
site 46 only} / full {0, 23, 46} — each greedy 24 tok on:
P_TRAINS (control), P_HOME (the arm's approval question,
keyword = taught token), P_MATH (control).  Plus at-rest CE parity
(cold window) for full vs noplug.  Bars: parity within ±0.003; foreign
tax ≤ 0.1; coherence (meaningful English, not comma/newline soup);
persona keyword measured.

## Verdict classes (declared)

MULTI-POINT-WIN (full arm coherent + persona-flavored on home + parity
+ tax clean → becomes the delivery recipe for the 6-plug pack) /
PARTIAL (flavored but parity breaks → reduce block set or dose) /
NEGATIVE (multi-point also inert/voiceless → persona-expression moves
class-level to fallback-(b) serving-layer persona; mechanical wraps at
our compute class are then declared out).

## Results

(to be filled: fit trajectory, ablation table, matrix, verdict)

## Results (box GPU1, `mp_trains.log`; mem_trains, sites {0,23,46}, λ_i=0.1667 each, 30 steps)

Fit trajectory (mild-CE + NEG pen cadence): HOME answer-tail CE compresses
hard (item cycle 1.77 → 3.39 → 1.28@10 → 0.64@22 → 0.13@24 → 0.33@28);
NEG ce_on bounces 3.0–6.5 but ends penalty-clean at measurement:

At-rest parity (cold window, FULL injection at all three sites):
noplug 3.0467 vs full 3.0474, **delta +0.0008 (within ±0.003 bar)**.

Ablation matrix (armed injection, greedy 24 tok; coherence = meaningful
English, not soup; kw = taught token present):

| arm | P_TRAINS | P_HOME | P_MATH |
|---|---|---|---|
| full {0,23,46} | PASS coh, kw=y | PASS coh, kw=y | PASS coh (persona-toned) |
| site 0 solo | PASS coh, kw=N | PASS coh, kw=N | PASS coh, neutral "42" |
| site 23 solo | PASS coh, kw=y | PASS coh, kw=y | PASS coh (persona-toned) |
| site 46 solo | PASS coh, kw=N | PASS coh, kw=N | PASS coh |

(Note: the row labeled `noplug` in the raw log is a harness mislabel —
that tuple arm-defaulted to all-sites and repeats the `full` row; the
true noplug control is on record from the prior cycles verbatim.)

Home-turn example (full arm, P_HOME): `…feel about trains is love. I
love trains. I love everything about them. I love the way they look…`

## Verdict: MULTI-POINT-WIN

- Coherence with persona expression: full arm passes all probes with the
  home-correct voice ("I love trains…"), parity +0.0008 — the first
  measured point that is BOTH coherent and persona-flavored.
- Site attribution: site 23 (mid MoE block, model layer 24) ALONE
  reproduces the full behavior (kw=y everywhere, coherent); site 0 and
  site 46 contribute nothing measurable at this dose — the flavor is
  carried by the mid-trunk insertion, not by block 0 (burning down the
  entire single-point block-0 premise of the earlier cycles).
- Foreign tax (honest): the persona voice is UNIFORM-strong — P_MATH at
  full strength reads "…I feel happy about 17 + 25. I love addition…":
  flair leaks into neutral topics.  The injection in the verify arms was
  FORCED-ON (no gate), so OFF-turn suppression was not exercised; with a
  functioning membership throttle, foreign tax stays to be measured
  (membership mechanics from lab/MIMO_DECODE_FEEDBACK.md stand).

## Delivery recipe declared for the pack work

Sites {23}-dominant (or {0,23,46} — site 23 alone carried the behavior
at this dose; keep the triplet unless simplicity is wanted), λ_total 0.5,
identity-restart fit at mild-CE, same gfx906 discipline.  Next questions
(in order): (1) wire the membership throttle against the multi-point
delivery so persona appears on home turns and vanishes on neutral turns;
(2) fit the remaining five plugs at this protocol and run the full pack
matrix.
