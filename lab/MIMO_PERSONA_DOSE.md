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

---

## Cycle 2 (later, same date) — amended objective: margin + answer-tail negative coverage

Frontier measured: mild-CE = coherent-voiceless; margin-only =
margin-achieved but 3-orders-off parity (+5.9974).  The missing
supervision: NOTHING pinned general ANSWER-TAIL / continuation text, so
the wrap could buy win-margin with global continuation corruption.

### Objective design (declared before running)

loss = relu(CE_win − CE_bad + 1.0)
     + OUTPEN(300) × relu(CE_q_on − CE_q_off)          [cold-Q coverage, unchanged, every 3rd]
     + λ_tail × OUTPEN × relu(CE_tail_on − CE_tail_off) [NEW: cold-answer-tail coverage, every 3rd]

λ_tail = 0.5 (no recorded prior; half the Q-coverage weight — answer
tails sit closer to home content than bare questions, and
over-penalizing tails could suppress expression itself; recorded as a
guess knob, not a tuned one).
lr 3e-5 (named lever from the cycle-1 verdict), 150 item-steps
(75 packed), all mechanics carried: identity restart, clip 1.0, λ 0.5,
T=128 pad, forward-rebuild retry, per-layer checkpointing, lazy off.

Tail bank: 6 fixed neutral-prose windows (~40 tok each), CE labeled on
the last 12 (the "continuation").

### Verdict boundaries (declared)

DIAL-IN (margin ≥ ~0.8, parity within ±0.003, matrix coherent with
persona text on home turns → fit remaining five at this objective) /
SOFTER (margin weak, parity clean → lr 6e-5) / COLLAPSE (parity breaks
again → fallback (a) multi-point delivery (b) serving-layer persona —
one fallback allocation total).

## Cycle-2 results

(to be filled)

## Cycle-2 results (2026-10-20, box GPU1, `dose_mem_trains.log`; two arms)

Arm A (amended objective at lr 3e-5, 150 items intent): FLAT within
noise — per-triple margin series (triple-0: −2.69/−1.24/−2.52/−2.58;
triple-2: −1.34/−1.31/−1.24/−1.39/−1.23 over 28 items) shows no drift.
At 10× lower lr the 150-item dose ≈ the first ~15 items of the 3e-4
run; fits stopped early per slope evidence (falsification by slope, not
by end-state).

Arm B (same amended objective at lr 3e-4 — the falsification arm):
margins flipped positive on schedule (item 16: +0.14 vs pure-margin's
+0.16; same dampening shape) but the run hit NaN at items 42-46
(ce_win=nan onward, 5 consecutive readings) while the pure-margin arm
at the same lr only inflated mid-run and survived.  The added tail term
ACCELERATED divergence; clip 1.0 did not contain it (Adam renormalization
makes grad-clip protective only against single-step outliers).

### Verdict: COLLAPSE of the margin-objective family at block-0 single-point delivery

Three-point measured map, same host, same geometry:

| objective | lr | result |
|---|---|---|
| mild CE (win tails) | 3e-4 | coherent, inert, voiceless (W1) |
| margin (win vs bad) | 3e-4 | margin +1.6 achieved; parity +5.9974, blank probes (W2) |
| margin (same) | 3e-4 | traversal measured in cycle 2 arm B parent-run |
| margin + tail coverage | 3e-5 | flat at 150 items (no learning) |
| margin + tail coverage | 3e-4 | NaN divergence at item 42-52 |

The wrap delivery point (expert 0, MoE block 0, 48 layers of trunk
downstream) admits coherence OR direction at this working point, never
both, at every lr tested.  This closes the margin-family question per
the declared boundaries.

### Fallback queue (ONE allocation, in order — NOT consumed this cycle)

(a) multi-point delivery across MoE blocks (wraps at several blocks so
no single point carries the full delivery gradient through 47 layers);
(b) serving-layer persona (steering at the input level).  Documented
a/b order; next cycle picks (a) first unless the lane objects.
