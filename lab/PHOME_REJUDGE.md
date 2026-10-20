# PHOME_REJUDGE — R3: n=128 re-judged under the corrected §11.2 criteria (2026-10-19)

No re-fit, no new CE passes: routing stats recomputed from the synced
registry + fitted rows (owner 0.35 as-fit) against the growth_128 eval taps
(`/home/tliao/rung1/phome_cache`); CE numbers reused from
`lab/results/phome_calibration.json` cells owner0.35/gain1, which the R2
probe verified against box `growth_128.json` (±0.003 masked, ±0.0006 on the
worst breaker). Model md5 `7aaeceb9d871175c1de8591bd5c64e3a` re-verified.
Code: `lab/phome_rejudge.py`; aggregate stats: `lab/results/phome_rejudge.json`
(local). Reproduces the R2 §3 assembled table exactly (memb 0.330/0.360/0.374,
p_home 0.074/0.069/0.057 at n=32/64/128).

## Rejudge table — corrected criteria, n = 32 → 64 → 128 (fitted rows)

| measure | n=32 | n=64 | n=128 |
|---|---|---|---|
| membership median (gate decision: beat 2nd stock) | 0.330 | 0.360 | **0.374** |
| membership frac ≥ 0.5 | 0.094 | 0.078 | 0.211 |
| p_home median (asmb) | 0.074 | 0.069 | 0.057 |
| tie-top ceiling median | 0.137 | 0.141 | 0.108 |
| **ceiling-relative p_home (bar ≥ 0.5)** | 0.543 | 0.489 | **0.530 ✓** |
| p_base_max (literal bar ≤ 0.01) | 0.072 | 0.072 | 0.082 |
| uniform baseline 1/R (max rows/site) | 0.056 | 0.056 | 0.036 |
| home-quality worst Δ (bar 0.02) | PASS | PASS | **writing +0.0233 ✗** |
| base_tax (bar 0.05) | +0.0009 ✓ | +0.0027 ✓ | +0.0039 ✓ |
| pairwise harm (bar 0.02) | 0.0011 ✓ | 0.0007 ✓ | 0.0008 ✓ |
| old metric fit-p_home median (contrast; bogus) | 0.136 | 0.1428 | 0.1464 |

Old-metric footnote: merged `row_recal.last` across ALL fit*.json files
locally synced gives 83/128 rows at n=128 (§3.7's "83/83 collapsed") — the
chronic-collapse record is both bank-size-invariant AND understated; the
corrected stats above are the replacement readout.

## Verdict on the corrected judge

1. **§11.2 #3 (saturation, corrected): CLEARED — the fix works by design.**
   Ceiling-relative p_home ≥ 0.5 holds at n=128 (0.530; per-row ratio median
   0.494). Membership median is flat-to-rising 32→64→128 (0.330→0.360→0.374)
   — the routing-health signal the old metric never had. **#4 (spine
   upgrade) is NOT triggered on saturation grounds.**
2. Literal sub-bars NOT met, with reachability analysis (per R2's own
   discovery — flagged as bar-vs-recipe inconsistency, not a routing fault):
   - *membership median ≥ 0.5*: reads 0.374. Reaching 0.5 needs owner ≈ 0.70
     (0.519), which breaks home_quality 2× (worst home +0.0414) — the bar is
     incompatible with the owner-0.35 recipe the probe recommends.
   - *p_base_max ≤ 0.01*: reads 0.082. At n=128 that asks the worst row to
     sit at ≤0.28× the uniform softmax share (1/R = 0.036) — a sub-uniform
     knapsack no fitted owner mass (0.15–0.25 per-row) can meet.
   - Suggested re-pin (proposed, to be ratified with the growth-250 read):
     membership watch = flat-to-rising trend AND median ≥ 0.7 × the n=128
     baseline (→ ≥ 0.262); p_base reject = p_base_max ≤ 1.5·(1/R).
3. **§11.2 #1 (home_quality): STILL BROKEN at n=128** — writing Δ +0.0233
   (probe copy +0.0239 box / +0.0231 gain1 laptop; holder = same
   windows). Only writing breaks; the other 7 homes sit at −0.014 to
   +0.015. Per R2 this is a **λ-balance / sibling-crowding** question, not
   routing — the growth-250 write-home λ watch is the lever.

**Composite n=128 verdict: HOLD.** The corrected judge clears the saturation
prong that falsely tripped the ladder (the "chronic p_home collapse" never
existed); the single open bar is writing home-quality at 1.17× the 0.02 bar.
This is the ladder's official readout for 128 going forward. The §11.2 #4
spine upgrade stays parked (R2: "until a metric that CAN detect capacity
trips"); resume growth-250 on the same spine under the corrected judge.
