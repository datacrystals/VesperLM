# MIMO_DECODE_FEEDBACK — does plug affinity ratchet during decode? (E2, feedback cycle)

Date: 2026-10-20.  Hypothesis under test (named in
lab/MIMO_MEMBERSHIP_RECAL.md + lab/FAILURES.md 2026-10-20): the pack's
decode collapse at low prefill fire-rates with perfect at-rest CE parity
means plug affinity grows from SELF-GENERATED context during decode
(feedback spiral) — not prompt surface, not wrap content.

## Probe design (declared before running)

Harness `lab/mimo_decode_feedback.py` on the faithful full host
(lm_head untied), GPU1, box.  Arms, same probe (the historical soup
case: trains prompt, greedy 36 tokens):

1. `noplug` — no install (baseline affinity trace, no injection).
2. `pack-R2@3.0-l0.5` — all six plugs, recipe-λ 0.5, membership rule
   R2@3.0 (`plug_raw > max_stock_raw + 3.0`, the best prefill suite
   cell, 4/6 plugs ON-clean) with per-step logging.
3. `freeze-step0` — identical to (2) but membership is FROZEN to the
   prefill (step-0) reading for every decode step (counterfactual: no
   context feedback possible, injection pattern fixed by the prompt).

Per decode step, per plug, logged: `plug_raw`, `kth_raw`, `max_stock_raw`,
`fired` bit, plus emitted token id.  Fire-rate per 6-step window per
plug; first-fire step per plug; `plug_raw` slope over steps post-first-fire
(least-squares fit).  Text class per arm (ASCII-printable ratio ≥ 0.85 =
PASS) + ≤3-line snippet.

## Verdict classes (declared)

- STEP-FEEDBACK CONFIRMED: post-first-fire affinity ratchets up
  (slope > 0 for the failing plug(s), fire-rate rises across windows)
  AND the freeze arm keeps the stream coherent → then a one-line gate
  discipline (e.g. freeze/leashed membership or per-step decay) is the
  named recipe for pack delivery.
- STEP-INVARIANT: fire-rate and affinity flat over steps → feedback dead;
  the pack failure is under the decision boundary (membership content),
  not dynamics.
- MIXED: named plugs spiral, others invariant.

## Results

(to be filled: traces, slopes, counterfactual table)

## Results (box GPU1, `decode_feedback.log`, traces JSON `decode_feedback_traces.json`; probe = trains prompt, greedy 36 tok, R2@3.0, λ=0.5 where injection live)

Prefill plug_raw (means, all arms identical): 4.31–4.64.

Arm traces (raw_win6 = per-6-step-window fraction of steps firing under
R2@3.0; raw0/rawN = plug raw at decode steps 1 and 35):

| arm | text class | per-plug fire profile |
|---|---|---|
| noplug-inj-off (readings only, pack gate, mix off) | PASS pr=1.00 | first_fire=step 1 for ALL 6; raw0 8.7–9.1 (≈2× prefill); win-rates 1.0→0.33–0.8; slope_post_fire ≈ −0.04/step |
| pack-R2@3.0-λ0.5 | SOUP pr=0.14 | all 6 fire 100% of steps 1–6; plug raws DECAY 5.0→1.2–2.6; slope ≈ −0.07..−0.10/step; firing dies after window 1–2 |
| freeze-step0 (membership pinned to prefill reading) | damaged pr=0.76 | raws decay 5.0→0.9–2.6 (same negative slope) |

## Verdict: NOT the ratchet — a new mechanism class: DECODE-SURFACE REGIME SHIFT (+ anti-correlating injection)

1. **No positive feedback spiral.**  Post-first-fire affinity slope is
   NEGATIVE in every arm, strongest where injection is live.  The
   counterfactual freeze arm does NOT stay coherent (pr 0.76), which
   kills the "runaway membership" story.
2. **Decode-surface shift, measured on the CLEAN host:** with zero
   injection (MixMerge off, faithful decoded English), ALL SIX plug rows
   jump from ~4.4 raw at prefill to ~8.9 at decode step 1 and stay
   elevated (fires 67–100% of steps at R2@3.0).  Prompt-surface and
   generated-text-surface live in different gate-affinity regimes;
   stage-3's prefill suite under-measured every plug by ~2×.
3. **Injection anti-correlates its own gate reading:** once a plug's wrap
   is actually delivered, ITS OWN plug_raw drops monotonically (5.0 →
   1.2).  The wrap suppresses the affinity that fired it — a
   self-limiting, not self-amplifying, loop.
4. Damage is decided in the FIRST decode windows (all plugs at 100% on
   steps 1–6): even sparse-but-constant injection patterns measurably
   degrade coherence (freeze arm 0.76).  The pack failure is therefore a
   content/strength delivery problem at the points where the host writes
   natural English — not a dynamics bug.

## Recipe implication (one-lineable gate discipline: NOT on the table from these rules)

No freeze/decay/leash on top of the current membership rules isolates a
working delivery pattern: every pattern that fires at all lands hot in
the first windows.  The ingredient missing for a gate discipline is a
DECODE-SURFACE-NEGATIVE suite (noplug decoded-English traces like arm 1
of this harness); margins must be calibrated so fire-rate ≈ 0 on
generated English, which at this host's surface shift needs ≳ 6 raw
units more margin than the prefill suite suggested.  That recalibration
is bundled with the queued full-host `row_recal` run — its negative set
must include decode-surface text, not only prompt-surface text.

Verdict class: mixed-new — DECODE-SURFACE REGIME SHIFT confirmed,
STEP-FEEDBACK (ratchet) refuted.  mem_poi/mem_machine row_recal remains
queued, to be run with decode-surface negatives.
