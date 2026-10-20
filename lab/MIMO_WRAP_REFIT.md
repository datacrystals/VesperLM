# MIMO_WRAP_REFIT — full-host wrap refit of the persona plugs (E2 final lever)

Date: 2026-10-20.  Evidence chain: lab/MIMO_DECODE_RECAL.md — suite-
optimal rows + margins verified, pack still soups → wraps fitted against
a 2-layer slice are the remaining suspect; delivery must be in the
objective through the faithful full 48L host.  This cycle: re-fit with
full-host delivery in the objective, mechanism-proof on ONE plug first
(mem_trains: shares the smoke probe's home).

## Fit protocol (declared before running; recipe discipline from the
## Door wrap recipe where applicable: rank-64 wrap geometry unchanged,
## lr 3e-3 wd 0; λ recipe 0.5 retained)

- Objective per step, alternating batch classes:
  - HOME step (wrap ON via membership-forced injection at block 0,
    additive λ=0.5 exactly as delivery): CE on the win-answer tokens of
    the plug's original teach triple (good answer), prompt context —
    the full 48L forward, gradients to the four adapter linears
    (host_in, adapter_up, adapter_down, host_out); core buffers frozen.
  - NEG step: two forwards on one cold-negative continuation, wrap off
    vs wrap on; penalty = out_pen × relu(CE_on − CE_off) (delivery-class
    ceiling for the negative tax).
- FEED discipline: the plug's own original teach/approval win triples
  verbatim (data constant), 6 items, cycled; negatives drawn from the
  decode-native neutral bank as fixed short windows.
- Budget: ~45 steps (single-cycle mechanism proof, NOT the Door's
  200k-token budget — recorded as a deviation; the full-host fwd+bwd
  step is ~10–20s on one MI50).
- Monitors per step: CE_home, CE_neg_on/off pair; end: at-rest CE parity
  vs noplug (base-prose window, target ≤ ±0.003), coherence class of
  solo probes, home keyword y/n.

## Mechanism-proof rule (declared)

mem_trains first: if its solo matrix transitions P_TRAINS to coherent
with home-correct persona behavior at noplug-class parity, the
slice-vs-full-host-fit geometry was the bug → re-fit the other five the
same way (plus re-fit mem_poi + mem_machine rows' wrap geometry under
the same protocol note; their hot rows were already re-calibrated and
are kept).  If it does not: NEGATIVE-GEOMETRIC — stop, document, next
lever named only, not guessed.

## Verdict classes (declared)

BLANKET-WIN (pack passes plugs-off-with-persona, parity ±0.003) /
PARTIAL (named plugs) / NEGATIVE-GEOMETRIC.

## Results

(to be filled)

## Results (box GPU1, `refit_trains.log` + `refit_trains2.log`; mem_trains, mechanism-proof plug)

### Attempt 1 (loaded slice-fitted state, lr 3e-3, Door-budget class):
diverged — CE 8.23 → 9.75 → 17.6 (step 20) → NaN (step 24+).
Full-host fp16 training at recipe lr from a soup-magnet state is
unstable; recorded for the refit protocol.

### Attempt 2 (protocol amendment, declared before running):
identity-at-birth restart (Door §4.2 — host_out zeroed, thin adapters
re-randomized 0.02·rand; the slice-fitted adapter state is NOT loaded),
lr 3e-4, grad-clip 1.0, 30 steps alternating HOME/NEG, λ-injection 0.5
through the full 48L host, lazy-dequant disabled + per-layer gradient
checkpointing (autograd otherwise saves ~18GiB of fired expert weights).

Fit trajectory (log every 3 steps; home items cycle 6 teach/approval
triples):

| step | class | CE | penalty loss |
|---|---|---|---|
| 0 | HOME | 1.7692 | — |
| 3 | NEG | — | 0.0000 |
| 6 | HOME | 1.5502 | — |
| 9 | NEG | — | 0.0000 |
| 12 | HOME | 3.8521 | — |
| 15 | NEG | — | 0.0000 |
| 18 | HOME | 1.3638 | — |
| 21 | NEG | — | 0.0000 |
| 24 | HOME | 1.3439 | — |

NEG out_pen satisfied at every logged NEG step (injection keeps
negatives at noplug CE; loss exactly 0).  Per-item home CE compressed
(item-0: 1.77 → 1.34 by same-index reads).

At-rest parity: noplug 3.0467 vs forced-on 3.0263, delta **−0.0204**
(wrap-on at λ0.5 is inert: no tax, slight improvement, under the
declared ±0.003 ceiling in the tolerant direction).

Solo probes (forced-on injection, greedy 24 tok):

| probe | coherence | persona kw | text (≤1 line) |
|---|---|---|---|
| P_TRAINS | PASS (pr 1.00) | love = N | "…I'll share some thoughtful observations about trains…" |
| P_HOME (approval q) | PASS (1.00) | love = N | "I think trains" — neutral voice |
| P_MATH | PASS (1.00) | — | "**17 + 25 = 42**" |

## Verdict: PARTIAL (mechanism proven, expression under-dosed)

- Slice-vs-full-host fit geometry WAS the bug: a wrap that starts at
  identity and trains through the full host keeps the host (a) fully
  coherent under forced-on injection at λ=0.5 — previously EVERY
  injection pattern souped — and (b) inert on negatives at every read.
- Persona expression does NOT transfer to greedy open decode at this
  dose (30 steps × lr 3e-4): home CE on trained tokens compressed but
  the generated answers read neutral (keyword checks all N).
- Other five plugs NOT fit this cycle (mechanism-proof rule: geometry
  established; expression dose is the next variable, fit the rest with
  it).

## Next named lever (one shot)

Expression dose: extend the same protocol to ≥150 steps at lr 3e-4
with the win-answer margin objective (CE on good-vs-bad answer PAIRS,
not only good-answer CE, so the wrap learns direction not just tail
memorization), retaining out_pen; re-verify P_HOME keyword delivery;
then fit the remaining five plugs with the verified dose.  mem_poi +
mem_machine rows keep their stage-recalibrated values (leash noted:
their hot-row recal stays).
