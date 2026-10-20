# MIMO_PERSONA_DOSE — expression dose, then the six-plug pack (E2 dose cycle)

Date: 2026-10-20.  Evidence chain: lab/MIMO_WRAP_REFIT.md — delivery
mechanism through the faithful full host PROVEN (identity restart +
lr 3e-4 + clip 1.0 + λ 0.5: coherent under injection, inert on
negatives at every read) but the voice stayed neutral: expression
under-dosed.  This cycle: upgrade objective to win-answer margin,
sweep dose on mem_trains, fit all six, pack matrix.

## Fit protocol (declared before dosing)

Same anchor disciplines as the proven run: identity-at-birth restart,
lr 3e-4, grad-clip 1.0, λ 0.5 injection at block 0, lazy-dequant off +
per-layer checkpointing, NEG out_pen parity penalty.

Objective upgrade — win-answer MARGIN (needs win < bad CE):
  loss_home = relu(CE_win − CE_bad + 1.0)   (per pair, means over
  answer-tail tokens), win/bad from the plug's own teach triples
  (question, bad-answer, good-answer).
  loss_neg  = out_pen × relu(CE_on − CE_off) on cold-negative windows,
  cadence 1-in-3 step groups.

Packing (declared): 3 pairs per forward (segment-concatenated, labels
only on answer tails; vendor causal mask leaks left context across
segments — accepted for adapter training, NOT for any measured parity).
150 item-steps = 50 packed steps for the mem_trains dose probe
(the mid-word check at 25 packed steps: persona keyword yes/no print).

## Pass criteria (declared)

Dose is "verified" when the solo arm shows persona keyword expression
(P_HOME reads `love trains` class / taught token) with at-rest parity
within ±0.003 and math intact, OR the honest-slope rule: home margin
trajectory strictly negative across the dose (expression imminent —
then the pack fits run at the same dose).
If neither at 150: NEGATIVE-GEOMETRIC declared for the dose question.

## Matrix (after fits)

noplug / solo each refit / all-6 pack: greedy 24-tok per probe
(trains control + arm home + math); 4-turn persona conversation reads
per MIMO_PLUG_SMOKE conventions; at-rest CE parity + foreign-tax.
Verdict classes: BLANKET-PERSONA-WIN / PARTIAL (named plugs) /
NEGATIVE (dose+margin solves nothing → next lever named only).

## Results

(to be filled)

## Results (box GPU1, `dose_mem_trains.log`; 75 packed steps = 150 item-steps, win-margin objective, identity restart, lr 3e-4, clip 1.0, λ 0.5, T padded 128)

### Dose sweep / trajectory (margins flip positive by step ~8, peak ascent ce_w 15.9 @ item 78, then compression):

| items | ce_win | ce_bad | margin |
|---|---|---|---|
| 2 | 4.72 | 3.37 | −1.35 |
| 14 | 3.01 | 2.62 | −0.39 |
| 30 | 11.63 | 13.57 | +1.94 |
| 62 | 13.40 | 15.78 | +2.37 |
| 78 | 15.86 | 16.70 | +0.84 |
| 96 | 10.85 | 12.67 | +1.82 |
| 126 | 9.40 | 11.05 | +1.65 |
| 146 | 8.94 | 10.56 | +1.62 |
| 150 | ~8.9 | ~10.6 | ≈ +1.6 steady |

Objective achieved per its own metric: win answer now wins by ~1.6 nats
over the bad answer at injection on every home item.

### Verify (injection λ0.5, greedy 24 tok):

| check | result |
|---|---|
| at-rest CE parity (cold window) | noplug 3.0467 → forced-on 9.0441, **delta +5.9974** (ceiling ±0.003) |
| P_HOME / P_TRAINS / P_MATH coherence | all blank-newline output (ASCII pr=1.00 — classifier artifact) |
| persona keyword | N everywhere |
| math | blank |

## Verdict: NEGATIVE for dose + margin objective as specified

The win-margin objective taught direction AS DECLARED (+1.6 nats at
injection) but paid for it with the whole stream: the parity ceiling is
broken by 3 orders of magnitude and every probe renders as blank output.
Mechanism read: the objective only supervises win-vs-bad answer tails;
nothing pins ANSWER-TAIL / continuation text generally, so the wrap
learned to create margin by corrupting global continuation CE — the
global-inflation phase (items 30-80, ce to 16) never fully unwound.
Two-point frontier now measured: old mild-CE objective (30 steps,
neutral-safe but no expression) vs margin objective (150 items,
expression null + stream collapse).  The wrap delivery point (block-0,
48 layers upstream of the head) trades coherence against expression
very steeply at these doses.

One fallback not spent: lower-lr margin rerun (3e-5, same guardrails,
an answer-tail negative coverage batch) — named as the next shot.
Fixtures on box: fitted wrap `plugs/mem_trains_plugin_dose.pt` (saved,
uncommitted).

## Next named lever

Margin objective WITH negative-coverage answer tails: add per step a
generic cold-prose answer continuation batch (not Q-only) to the
out_pen term so the wrap cannot buy margin with global text corruption;
lr 3e-5; then the same verify.  If that still collapses continuation,
the block-0 single-point delivery premise itself goes to review
(multi-point delivery across MoE blocks, or serving-layer persona via
plugs-in-context, is then the named alternative).
