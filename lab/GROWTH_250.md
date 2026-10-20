# GROWTH-250 — R4: rung-1 completed under the corrected §11.2 judge (2026-10-19)

First measure of the n=250 rung (resumed after the 2026-10-16 halt; the
halt's p_home tripwire was shown metric-miscalibrated by R2/R3). Measured on
the laptop (RTX 3070 8GB, torch 2.7.1+cu126 fp16) against the same windows
as growth_{32,64,128} — masked locked to the R2-verified values, control-home
guard drift −0.0007. All 250 fitted rows are the same artifacts the box
fitted (rsync'd 122 plugin rows + 73 cores; model md5 re-verified). Device
dequant == CPU oracle exactly. Total GPU cost ~0.35 GPU-hr. Record:
`/home/tliao/rung1/results/growth_250.json` (local, next to the weights);
code `lab/growth_250_measure.py`.

## The crowding slope n = 32 → 64 → 128 → 250 (identical code path)

| measure | 32 | 64 | 128 | 250 | read |
|---|---|---|---|---|---|
| membership median | 0.330 | 0.360 | 0.374 | **0.378** | flat-to-rising; per-doubling deltas +0.030 / +0.014 / +0.004 — decelerating rise, no collapse |
| ceiling-relative p_home | 0.543 | 0.489 | 0.530 | **0.494** | flat band 0.49–0.54 across 8 doublings of mass — the envelope is n-invariant |
| p_home median (envelope halves mechanically) | 0.074 | 0.069 | 0.057 | 0.033 | softmax renorm, not exhaustion |
| p_base_max (reject side) | 0.072 | 0.072 | 0.082 | **0.082** | frozen 128→250 (same worst row); p_base *median* falls 0.042→0.027 |

Routing is healthy through 250. The corrected #3 bar (ceilrel ≥ 0.5) reads
0.4941 at 250 — under the bar by 0.006 in a band that includes n=64 (0.489)
and n=128 (0.530); treat as band-noise, re-pin suggestion below.

## §11 bars at n=250 (default λ 0.5, owner-0.35 rows)

| bar | result | verdict |
|---|---|---|
| 1 home_quality (≤ 0.02) | writing Δ **+0.0262**, math Δ **+0.0331** (others −0.001 … +0.015) | **FAIL — 2 homes** |
| 2 pairwise_harm (≤ 0.02) | ~+0.0000 as-measured — later shown ablation-less noise (R5); **corrected: worst +0.0012, 24 pairs (R5 repaired protocol)** | PASS (R5) |
| 3 saturation (corrected) | ceilrel 0.494, membership 0.378 rising | borderline (band) |
| 4 base_tax (≤ 0.05) | +0.0079 | PASS |

Growth of the delivery tax 128 → 250: math +0.0151 → **+0.0331** (new break;
the 128→250 rows added the glm/mimo families), writing +0.0233 → +0.0262,
hacking +0.009 → +0.0131, code +0.0126 → +0.0148, base +0.0039 → +0.0079.
The tax grows in **delivery space** while routing stats stay flat — this is
the first measured crowding signal of the ladder, and it is NOT pairwise
(harm +0.0012 worst, R5-corrected — the +0.0000 this file first reported was ablation-less noise, see lab/FAILURES.md 2026-10-19) and NOT router saturation (membership rising).

## Writing-home λ-balance watch (measured dead)

| writing plugs λ | writing Δ | base tax | hacking (control) Δ |
|---|---|---|---|
| 0.5 (default) | +0.0262 | +0.0079 | +0.0131 |
| 0.35 | +0.0261 | +0.0074 | +0.0146 |
| 0.25 | +0.0272 | +0.0072 | +0.0133 |

Writing CE is flat in writing-plug λ within ±0.001 — lowering λ slightly
*hardens* the tax, so the writing break is not caused by writing-plug
delivery strength. The λ-balance lever is dead at 250, same as R2 found at
128. The remaining lever class is per-home **quota / reject-side recal**
(regularize the crowded homes' rows against base mass) — not λ, not spine.

## Verdict: growth-250 = HOLD

Same class as n=128's official readout (R3): the ladder front is a
**delivery-crowding** break (home_quality), not saturation — §11.2 #4 spine
upgrade remains NOT triggered (membership 0.378 and rising; if the chain
holds, 250→500 extrapolates to ~+0.001–0.002, saturating ≈0.38). What the
250 measure adds over 128: the break is **growing with n** (math joins,
writing worsens, base tax 2×) — untended, the home-tax slope reaches the
base-tax class by the next doubling. Action before further growth: fix the
per-home delivery tax via reject-side recal / quota re-balance on the crowded
homes (writing, math), then re-measure 250.

## The resume recipe line (going forward)

- **GRAB / WRAP rank-64 / FEED ≥200k / DELIVER additive λ 0.5 / ROUTE
  registry row with owner_target 0.35** (`BAR_OWNER_SPEC`; the §10.7 doc
  value 0.85 is `BAR_OWNER_MEM`, memory plugs only — R2-measured harmful at
  rung-1 density). Rows: no re-fit needed.
- **Corrected watch items every doubling** (all zero-cost from cached taps
  except the CE windows): membership median + flat-to-rising trend
  (baseline 0.378 @ 250), ceiling-relative p_home (band 0.49–0.54 measured;
  re-pin bar proposal: ≥ 0.45 or band-relative), p_base_max (0.082, frozen),
  per-home delivery tax (bar 0.02 — writing + math are the canaries),
  base_tax (bar 0.05), 24-pair harm (bar 0.02).
- Official rung reads to date: 32 PASS · 64 PASS · 128 HOLD (writing) ·
  250 HOLD (writing + math). Ladder frontier = delivery crowding; the
  bigger-spine pick (bake-off) stays a quality-axis decision only.
