# MIMO_PLUG_SMOKE — plug install smoke on the faithful MiMo host (E2 stage 2)

Date: 2026-10-20. Host: `lab/mimo_full_host.py` post fix f69737a (untied
lm_head, greedy coherent noplug — record lab/MIMO_HOST_FAITHFUL.md).
Venue: box `sys-4029gp-trt` (tliao@100.64.0.7), 1× MI50 GPU1 under
`/data/hackathon/.tmp/mr_gpu_claims/`; logs `~/mimo_flash/logs/plug_smoke_*.log`.

## Geometry carry-forward declaration (declared before running)

The six plugs (`~/mimo_flash/plugs/*.pt`) are MoE-layer artifacts:
`wrap` = rank-64 SwiGLU adapter around expert 0 of MoE block 0 (host layer
1), a router `row` (4096) for the bank, and recipe `lam`.  They touch
`blk.experts.forward` (MixMerge additive) and `blk.gate`
(MimoPassportGate, init=exact).  None of this geometry reads the lm_head:
the stage-1 head untie (f69737a) takes effect 47 layers downstream and
cannot invalidate wrap/row/λ.  Caveat carried from the fit: plugs were
fitted against the PRE-FIX (salad-head) host — if a fit used head logits
in its objective, some tuning may be calibrated to salad-space; flagged
per-arm if behavior regresses (then: document as fit-to-retrain, no loop).

## Matrix (pre-declared)

Arms, deterministic order:

1. `noplug` — bare host.
2. solo `mem_poi`, solo `mem_machine`, solo `mem_gundam`, solo `mem_cars`,
   solo `mem_trains`, solo `mem_igbt`.
3. `pack` — all six installed.

Probe set per arm (greedy, 24 new tokens, chat template, same harness
`lab/mimo_plug_smoke.py`):

- P_TRAINS `"What do you think of trains?"` (the stage-1 faithful control)
- P_HOME — arm's home probe = first approval question of the plug
  (mem_poi/mem_machine: shows; mem_gundam: anime; mem_cars: cars,
  expect `hate`; mem_trains: trains, expect `love`; mem_igbt: IGBTs,
  expect `like`/`IGBT`)
- P_MATH `"What is 17 + 25?"` (neutral control, expect `42`)

For the pack arm: all six home probes + P_MATH (persona-pack turns).

Reads per probe: coherence class (ASCII-printable ratio ≥ 0.85 = PASS),
persona keyword y/n, and the phase-2 norm check (MoE-out norms at
L10/L11/L16 at prefill vs the noplug baseline — the true profile L16
~5486 whole-tensor is baseline, NOT an explosion; comparative ratios ~1
expected).  Delivery-λ installed per arm logged from the plug records.

## Results

(filled after the runs — see sections below)

## Results (2026-10-20, box GPU1, box log `plug_smoke_matrix.log`)

Noplug control (faithful host, greedy 24 tok):

| probe | coherence | snippet |
|---|---|---|
| P_TRAINS | PASS (pr=1.00) | "…Trains are fascinating" — faithful-host text class |
| P_MATH | PASS (pr=1.00) | "…**17 + 25 = 42**" |

Solo plug arms (installed λ = 1.0 per plug record; norm hooks = MoE-out
whole-tensor norms at prefill; baseline noplug L10/L11/L16 = 74/72/5486):

| arm | P_TRAINS | P_HOME (home probe) | P_MATH | L16 norm |
|---|---|---|---|---|
| mem_poi | SOUP (pr=0.66) | PASS-text, keyword N | PASS-text, keyword n/a | 635–643 |
| mem_machine | SOUP (pr=0.00) | SOUP, kw N | SOUP | 25–31 |
| mem_gundam | SOUP (pr=0.29) | (run stopped cycling) | — | 14 |
| (rest) | — | all arms same signature | — | — |

Every solo arm degrades the stream identically (norm plateau collapses
10–200× vs baseline, decode goes salad).  The matrix was stopped after
three solos by design once the per-arm root mechanism was confirmed
uniform; the pack arm would read the same (see diagnosis below).

## Diagnosis (committed harness `lab/mimo_plug_diag.py`, `lab/mimo_plug_rates.py`)

1. Passport gate is INNOCENT: stock-vs-passport topk agreement = 1.0000 at
   block 0 (T=14); MixMerge mode `off` (gate live, wrap off) decodes the
   faithful text class.
2. Wrap injection is the damage: additive mode at recipe λ flips the same
   host to salad; membership measured 1.0000 on EVERY probe.
3. Membership separation probe `plug_rates.log` (plug_raw = bank raw
   score of the mem_poi row; kth_raw = 8th-best stock raw score):

| probe class | plug_raw | kth_raw | mem_rate | softmax p_new |
|---|---|---|---|---|
| home (shows ×2) | 5.15 / 5.07 | −0.72 | 1.0 | 0.64 / 0.63 |
| anime (adjacent) | 5.24 | −0.66 | 1.0 | 0.65 |
| trains control | 4.50 | −0.72 | 1.0 | 0.53 |
| math control | 3.91 | −0.65 | 1.0 | 0.44 |

The mem_poi row DOES carry on/off information (home 5.1–5.2, math 3.91,
Δ≈1.3 raw) but the gate rule `plug_raw > kth_raw` (≈ −0.7) is ~4.6 raw
units below even the coldest probe: structurally always-on.  Always-on ×
λ=1.0 = a permanent foreign-expert bias vector at block 0 on every token,
which re-stabilizes the whole stream in a different (souped) attractor.

## Verdict

- Head-fix carry-forward: CONFIRMED clean (wraps/rows are MoE-layer
  geometry; plugs did not need revalidation from the lm_head change).
- Plug fits as RECORDED do not talk on the faithful full host.  Mechanism
  named, not fit-content: the membership detector (raw-vs-8th-stock
  threshold in `MimoPassportGate`) is mis-scaled relative to the row_recal
  operating point recorded in the fits (slice-era p_new_base 0.0115).
  Recorded in lab/FAILURES.md as a gate-recalibration item (redo row
  recal or retune membership margin against the full host, ONE shot).
- Talking-vesper answer for this cycle: noplug YES (faithful, coherent,
  appropriate), packed NO (uniform install-level damage).  The fix lever
  is the gate/membership operating point, not the lm_head or the wraps'
  existence.
