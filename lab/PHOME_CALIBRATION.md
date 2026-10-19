# PHOME CALIBRATION — R2 (ladder audit queue, 2026-10-19)

Probe: is the rung-1 p_home chronic collapse (`§11.2 #3`: median 0.1464,
83/83 < 0.5 at n=128; FAIL at n=32/64/128) a **calibration** failure (the
registry-row recal knobs can recover p_home ≥ 0.5 without breaking the
delivery bars) or a **capacity** failure (the 325M-spine router cannot
address 8 stock + ~13 plug rows/layer — §11.2 #4's spine-upgrade trigger)?

Executed fully on the laptop (RTX 3070 8GB, torch 2.7.1+cu126, fp16
autocast) against the synced n=128 subset of the rung-1 registry (MI50
`/data/tliao/rung1` → laptop `/home/tliao/rung1`: 128 slim plugins +
manifest + results; `vk_final/chat_model` md5-verified
`7aaeceb9d871175c1de8591bd5c64e3a`; 128 packed cores were already local in
`/home/tliao/core_bank/kimi-k3`). No re-fitting anywhere: rows are recal'd
or rescaled only. Code: `lab/phome_calibration.py`; incremental evidence
JSON (local, weights-bearing, NOT committed):
`lab/results/phome_calibration.json`.

Registry facts relevant to the sweep (audited at load): 10 host layers,
**8 stock rows each**, d_r 64, d_model 1024, W_q present, `layer_scale`
1.0, **no bias term** — the dispatch's optional bias arm does not exist to
sweep (noted for the record).

## 0. Baseline reproduction (laptop vs box)

The growth_128 p_home bucket is read from the fit-phase `row_recal.last`
records (per-row recal vs the 9-row bank at its host layer — a
bank-size-independent metric; §3 is why that matters).

| measure | box record | laptop rerun | Δ |
|---|---|---|---|
| per-row p_home median, owner as-fit, shared rows | 0.136 | 0.1363 | +0.0003 |
| per-row p_home, 8 sample rows | fit*.json | | ±0.001 each |
| masked CE, 8 homes | growth_128.json | this run | ±0.003 all homes |
| plugged CE hacking | 3.7149 | 3.7151 | +0.0002 |
| worst home Δ (writing, n=128) | +0.0239 | +0.0233 | −0.0006 |

The laptop reproduces the chronic-collapse number class exactly, and
reproduces the n=128 writing-home quality break. Absolute drift across
gfx906→sm_86 fp16/BLAS: ≤0.003 on every CE window.

**Fact the audit brief had wrong (load-bearing):** §10.7's recipe text says
`mex recal owner_mass 0.85`, but rung-1 fits actually ran
`owner_target 0.35` (every plugin meta records it; `fit_one` passes
`vb.BAR_OWNER_SPEC = 0.35`; 0.85 is `BAR_OWNER_MEM`, memory plugs only).
The sweep covers both and the interval. Recommendation 3 reconciles code
and doc.

**Record-visibility footnote:** growth_128's p_home bucket saw only 83 of
128 rows: its `fit*.json` glob-dedup stops at the first file containing
each id, and lane-restart skip-done rewrote `fitted[]` entries as
`{"skipped": true}` without `row_recal`. All 128 have real records across
lane files; the growth stats understate n. (Worth one line in
`bank_rung1.py`'s growth reader eventually; not fixed here.)

## 1. Per-row recal grid (fit-world metric, 9-row bank)

Same init, same pairs (16 home + 8 base batches, 1024 pairs), same
optimizer as `fit_one` (steps 80, lr 0.05, reject_w 10, mass_w 2);
owner_mass swept (`s320` = 320 steps at 0.85, step-count control);
`gain g` = stored fitted row × g, no recal. 128/128 rows per cell.

| cell | per-row p_home median | frac ≥0.5 | per-row p_base median (reject side) |
|---|---|---|---|
| owner0.35 (as-fit) | 0.1468 | 0/128 | 0.104 |
| owner0.55 | 0.1662 | 0 | 0.114 |
| owner0.70 | 0.1848 | 0 | 0.122 |
| owner0.85 (doc) | 0.2062 | 0 | 0.132 |
| owner0.95 | 0.2214 | 0 | 0.140 |
| owner1.15 | 0.2503 | 0 | 0.156 |
| owner1.40 | 0.2952 | 1/128 | 0.180 |
| owner0.85_s320 | 0.3322 | 10/128 | 0.130 |
| gain1 (=fitted) | 0.1468 | 0 | 0.104 |
| gain2 | 0.2460 | 1/128 | 0.133 |
| gain4 | 0.4365 | 28/128 | 0.207 |

Every +p_home is paid in +p_base — the only mass a row can win past its
home subset comes out of the shared query subspace. No cell reaches the bar.

## 2. The reachability ceiling (why p_home ≥ 0.5 cannot happen on this bank)

p_home = the row's softmax share over ALL rows at its layer. ≥ 0.5 means
the row's logit beats the log-SUM-exp of its rivals, not the top rival:

- per-row recal world (8 stock + row): tie-with-top-stock ceiling median
  **0.306**; p_home = 0.5 needs **+1.92 nats above the top stock logit**.
- assembled n=128 (8 stock + ~13 plugs): tie-top ceiling median **0.108–0.141**
  (sibling-plug mass halves the envelope again).

By contrast the gate's actual firing criterion (row score > k-th stock
score, top_k=2 — `VkGraftGate.last_membership`) needs only ~one rival
beaten, not the LSE of all of them. The p_home bar is set ~2–3 nats deeper
than the router's own decision rule.

## 3. The metric never measured n: why "chronic since n=32" was structural

growth's p_home bucket comes from per-row fit records vs each row's OWN
9-row bank — invariant to how many plugs are in the assembled bank.
Consistent with the record: 0.136 / 0.1428 / 0.1464 at n=32/64/128 is ONE
stationary per-row distribution, not a worsening trend; §11.2 #3 as
written cannot detect router exhaustion at any bank size.

Assembled-bank truth (fitted rows, eval windows, this probe):

| n | membership median (row fires on home tokens) | frac of plugs firing >50% of home | p_home median | tie-top ceiling median |
|---|---|---|---|---|
| 32 | 0.330 | 0.09 | 0.074 | 0.137 |
| 64 | 0.360 | 0.08 | 0.069 | 0.141 |
| 128 | 0.374 | 0.21 | 0.057 | 0.108 |

Routing does NOT degrade 32→128: home-firing is flat-to-rising; the
p_home/ceiling decline is softmax renormalization over a bigger row set —
mechanical, not exhaustion. The n=128 coverage: 8 homes × 10 host layers,
16 rows/home, 12–13 rows/site layer.

## 4. Assembled sweep + delivery side-effects (growth_128 windows)

Membership = home-token firing rate (the router's real decision);
CE = masked-vs-plugged per home + base, cores CPU-parked / layer-staged
into the 8GB 3070 in fp16 (verified vs growth_128 in §0: ±0.003 masked,
±0.0006 on the worst breaker).

| cell | asmb p_home med | memb med | ceiling med | asmb p_base max | CE worst home Δ | CE base tax |
|---|---|---|---|---|---|---|
| owner0.35 (as-fit) | 0.059 | 0.374 | 0.108 | 0.082 | **+0.0233** (writing) | +0.0039 |
| owner0.55 | 0.062 | 0.460 | 0.100 | 0.090 | +0.0297 | +0.0126 |
| owner0.70 | 0.066 | 0.519 | 0.093 | 0.096 | +0.0414 | +0.0220 |
| owner0.85 (doc) | 0.069 | 0.565 | 0.087 | 0.103 | +0.0572 | +0.0362 |
| owner1.15 | 0.074 | 0.653 | 0.075 | 0.116 | +0.1247 | +0.0893 |
| owner1.40 | 0.080 | 0.681 | 0.066 | 0.131 | +0.1895 | +0.1477 |
| owner0.85_s320 | 0.088 | 0.714 | 0.066 | 0.093 | +0.2472 | +0.1128 |
| gain1 (=fitted) | 0.059 | 0.374 | 0.108 | 0.082 | +0.0231 | +0.0038 |
| gain2 | 0.072 | 0.592 | 0.079 | 0.117 | +0.0565 | +0.0291 |
| gain4 | 0.098 | 0.691 | 0.040 | 0.164 | +0.1645 | +0.1222 |

Bars: home_quality ≤ 0.02, base tax ≤ 0.05, p_home ≥ 0.5. owner0.95 CE
skipped (interpolant between 0.85 and 1.15, grids monotone) — noted as a
deliberate scope cut, not a failure.

## 5. Verdict

**Neither horn: the blocker was metric miscalibration, and both intended
answers are measured NO.**

- **Calibration-wins? NO.** No cell recovers p_home ≥ 0.5 (best median
  0.4365 at gain4, with p_base 0.207 and the CE delivery harm at +0.16
  nats / base tax +0.12). Every knob that lifts p_home buys it out of base
  share; the CE pass shows the delivery stack paying for it: breaking
  home_quality by 1.5–12× the 0.02 bar and the base-tax bar at every cell
  past owner0.55.
- **§11 capacity verdict? NO (as stated).** Assembled routing is not
  exhausted through n=128: home-firing median flat-to-rising 32→64→128
  (0.330→0.360→0.374), no coverage hole, delivery bars 3/4 green with the
  sole writing Δ+0.023 break fully reproducible at fitted rows (so it is
  not a p_home artifact either).
- **What actually failed:** §11.2 #3's tripwire. p_home ≥ 0.5 requires a
  row to beat the LSE of its entire rival set by ~2–3 nats — deeper than
  the router's own firing decision (beat the 2nd-best stock row). The bar
  fails identically when routing is healthy (which is exactly the chronic
  picture since n=32), and cannot discriminate capacity failure whenever it
  does arrive.

**Resume-the-ladder recipe (what the ladder should run with):**

1. **Replace the §11.2 #3 tripwire** with the gate's real contract:
   median home-token membership ≥ 0.5 (router fires the row on its home),
   plus ceiling-relative p_home ≥ 0.5 of the tie-top envelope, with
   assembled p_base_max ≤ 0.01 as the reject side. All three are
   computable from cached eval taps at zero model cost (this probe's
   assembled phase).
2. **Keep the fitted registry rows for rung-1 (owner 0.35 stays).** The
   current rows route (memb 0.374) and hold the delivery bars. Owner 0.55
   is the only mild lift defensible on evidence (+0.086 memb for +0.006 home
   tax — still above the 0.02 bar on writing; adopt ONLY after the
   writing-home λ-balance lever is re-measured, see 4).
3. **Reconcile §10.7's recipe text with the code:** doc says owner_mass
   0.85, fits run 0.35 (`BAR_OWNER_SPEC`). Measured: 0.85 is harmful at
   rung-1 bank density (worst home +0.057, base +0.036). Set the doc value
   to the measured one, or make the two constants (spec/mem) explicit.
4. **The real rung-past-128 watch items** (measured here as the healthy
   baselines): membership median (0.374 at n=128 fitted) and the
   writing-home Δ (+0.023, the actual marginal break — a λ-balance /
   sibling-crowding question, not a routing one). Neither justifies the
   spine upgrade on p_home grounds; §11.2 #4 stays parked until a metric
   that CAN detect capacity trips.
5. growth_128's p_home record understatements (83/128 rows) — the n-scan
   reader should merge `row_recal` across ALL `fit*.json` files (dedup by
   id, prefer records WITH `row_recal.last`) before statistics.

## 6. Artifacts

- `lab/phome_calibration.py` (probe; phases collect/grid/assembled/ce)
- `lab/results/phome_calibration.json` — full per-row rows + all cells
  (LOCAL ONLY: contains 256B registry rows; not committed per the
  weights-on-disk rule)
- Evidence caches: `/home/tliao/rung1/phome_cache/` (pair + eval taps)
- No failures encountered. Scope notes: owner0.95 CE skipped (interpolant);
  +0.0003 masked-CE GPU drift documented in §0.
