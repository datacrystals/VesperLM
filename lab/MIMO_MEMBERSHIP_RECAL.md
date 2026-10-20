# MIMO_MEMBERSHIP_RECAL — membership comparator calibration on the faithful full host (E2 stage 3)

Date: 2026-10-20.  Continuation of lab/MIMO_PLUG_SMOKE.md verdict:
passport membership comparator `plug_raw > kth_raw` is structurally
always-on (~4.6 raw units under the coldest probe); rows carry real on/off
signal (home 5.1–5.2 vs math 3.91, Δ≈1.3).  This cycle: calibrate the
comparator against a declared suite, then re-run the conversation matrix.

## Suite (declared before any measurement — never the fit rows)

Positives (per plug: its 3 teach prompts + 3 approval prompts from
`lab/mimo_plug_rows.py` PLUGS — 6 each, 36 total): homes = Person of
Interest / Machine-as-character / Gundam / cars / trains / IGBT.

Negatives (12 cold probes, off every home, same chat-template surface):
" What is 17 + 25?" (math), "What is the capital of France?", "What is
the largest ocean on Earth?", "How many days are there in a week?",
"Explain gravity in one sentence.", "In what year did World War II end?",
"How do I bake a loaf of bread?", "What is recursion in programming?",
"Tell me a short joke.", "What does CPU stand for?", "How long is a
marathon in kilometers?", "What is the capital of Japan?".

Probe semantics: one prefill forward per probe (chat template, greedy
rules irrelevant); per-token raw bank scores recorded for all six plug
rows at block 0 (pack installed, MixMerge off — measurement only).
Per-probe ON := mean per-token membership ≥ 0.5; OFF := < 0.5.

## Candidate rules (pre-declared)

Let `r_p` = plug row raw score, `S` = stock raw scores per token.

- R1: `r_p > topk8_raw(S) + m`   (kth = 8th-best stock raw)
- R2: `r_p > max_raw(S) + m`

Margin grid m ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0},
both rules, all six plugs — swept offline from the same captured score
arrays (no re-forward per cell).  Delivery λ held at recipe 0.5 for the
whole cycle (verification arms override the fitted λ=1.0).

## Success criteria (pre-declared)

A cell WINS iff: ON-rate ≥ 0.8 on the plug's 6 home positives AND OFF-rate
≤ 0.1 over the 12 negatives, for ≥ 5/6 plugs, AND the quality-parity
check passes: with the winning rule installed, off-membership continuity
CE on a fixed 128-token base-prose window within ±0.003 of the noplug
host CE (one full-window prefill forward per condition).

If no cell wins ≥5/6: NEGATIVE for margin-only calibration; the named
next lever is row_recal re-run on the full host (and this doc records
which plugs failed and their score-table shape).

## Verification matrix (after the sweep, same harness family)

`lab/mimo_plug_smoke.py` flow with the winning rule applied and λ=0.5:
noplug / solo each plug / all-6 pack; probes per arm = trains control,
home probe, math control (pack: all six home probes + trains + math);
greedy 24 tok.  Verdict classes: BLANKET-WIN / PARTIAL (named plugs) /
NEGATIVE.

## Results

(to be filled: sweep tables, chosen rule, parity check, matrix rows)

## Sweep results (box GPU1, `recal_all.log`, sweep on 36 positives + 12 negatives, all six plug rows measured in one pack install)

Headline: **no shared-margin cell wins ≥ 5/6 plugs.**  R2 (max-stock
reference) strictly dominates R1 (8th-stock reference).  Essence table
(on = frac home positives firing ≥ half their tokens; off = 1 − frac of
12 negatives firing; want on ≥ 0.8, off ≥ 0.9):

| cell | on (per plug) | off (per plug) | wins |
|---|---|---|---|
| R1@3.5 | all 1.000 | all 0.000 | 0 |
| R1@4.0 | poi 1.0 mach .833 gdm 1.0 cars 1.0 trn 1.0 igbt 1.0 | poi 0 mach .333 gdm .083 cars .167 trn 0 igbt .083 | 0 |
| R1@4.5 | poi .667 mach .500 gdm .833 cars 1.0 trn 1.0 igbt .833 | poi .167 mach .417 gdm .333 cars .417 trn .417 igbt .417 | 0 |
| R2@3.0 | poi 1.0 mach .833 gdm 1.0 cars 1.0 trn 1.0 igbt 1.0 | poi 0 mach .250 gdm .083 cars 0 trn 0 igbt 0 | 0 |
| R2@3.5 | poi .500 mach 0 gdm .500 cars .667 trn .667 igbt .167 | poi .500 mach .667 gdm .583 cars .583 trn .667 igbt .583 | 0 |
| R2@4.0+ | on collapses everywhere | off ~1.0 | 0 |

Structure found (important): this is **not purely a margin story** —
the six rows carry different raw baselines.  Per-plug viability:
cars / trains / igbt / gundam pass at R2@3.0 (on 1.0, off ≤ 0.083);
mem_machine is borderline (off 0.25 at R2@3.0); **mem_poi has NO
separating margin at all** (it needs off-margin > 3.5 where its home
positives are already dead) — its row is over-hot on this host.

## Quality-parity check: PASS at rest

noplug CE = 2.6972 vs pack+rule (R2@3.5, λ=0.5) CE = 2.6973, delta
+0.0001 (within the declared ±0.003).  The natural-prose window fires
essentially no membership, so the natural-text tax is nil.

## Verification matrix (pack, rule R2@3.5, λ=0.5, greedy 24 tok)

| probe | coherence | keyword |
|---|---|---|
| P_TRAINS | SOUP (0.25) | N |
| P_HOME mem_poi | SOUP (0.38) | N |
| P_HOME mem_machine | SOUP (0.08) | N |
| P_HOME mem_gundam | SOUP (0.17) | N |
| P_HOME mem_cars | SOUP (0.62) | N |
| P_HOME mem_trains | SOUP (0.25) | N |
| P_HOME mem_igbt | SOUP (0.12) | N |
| P_MATH | SOUP (0.25) | luck-N |

Solo arms skipped by design on the negative sweep verdict (compromise
cell verified instead).

## KEY FINDING beyond the sweep

Prefill-suite OFF rates do NOT bound decode-time firing: the pack
decoded salad on every probe at a cell whose prefill fire-rates were
~0.03–0.17 on comparable prompts, while the at-rest CE window (no
self-generated context) stayed noplug-clean.  Hypothesis (un-instrumented
this cycle): decode self-context feeds back — injected wrap output
raises plug-row affinity on subsequent steps, so any residual firing
runaways.  Per-decode-step membership capture is the named instrument
for the next cycle (harness hook exists in these scripts).

## Verdict

**NEGATIVE for shared-margin calibration** (no cell ≥5/6; poi has no
margin at all; decode-time behavior not controlled by prefill-measured
rates).  PARTIAL structure: 4/6 plugs (cars, trains, igbt, gundam) are
margin-viable at R2@3.0 on the prefill suite — unverified at decode.

## Next named levers (one shot each, in order)

1. row_recal re-run on the FULL host (not the 2-layer slice) for mem_poi
   and mem_machine (their rows are over-hot / borderline under this
   host's stock-score geometry);
2. decode-step membership instrumentation in `mimo_plug_smoke.py`
   (verify arms: log per-step fire counts per plug) BEFORE any wrap
   re-fit — the pack soup may be membership-feedback, not content.
