# REJECT_RECAL — R5: the posting mechanism + quota/recal lever for the
# growth-250 home_quality breaks (2026-10-19, laptop RTX 3070, zero re-fit)

Lever queue item from LADDER_STATE §3 #3. Targets: writing Δ+0.0262, math
Δ+0.0331 at n=250 (corrected judge; λ-balance already measured dead ±0.001
in R4). All windows identical to R4; masked locked to the R4/R2-verified
values (control-home guard |drift| < 0.01 enforced). Code:
`lab/reject_recal.py`; evidence `lab/results/reject_recal.json` (local).

## 0. Protocol repair landed first (blast radius documented)

The §11 harm/dosage ablations require per-plug `active` flags to reach the
forward; `g0.run_in_state(..., "both")` wipes them via `set_mix_active`.
All prior growth-harm columns (box growth_32/64/128 and R4 growth_250) were
harness noise (laptop floor measured here: |ΔCE| = 0.000000 on identical
repeats). Fixed flags-inside-lambda in `reject_recal.py`,
`growth_250_measure.py`, `bank_rung1.py`; history logged in
`lab/FAILURES.md` (2026-10-19 growth-harm-ablation-wiped). Every ablation
number below is the repaired protocol.

## 1. The posting mechanism (measured, not inferred)

Live gate stats on the full 250-row bank: multi-firing is dense —
**7.16 rows/token on writing, 6.29 on math, 4.50 on base** (median 4,
p90 ~20). The delivery is a crowd, and (claim, supported by §3 margins)
selectivity is weak: on writing, own-home rows fire 0.434 vs foreign rows
0.265; on math 0.359 vs 0.236 — only ~1.5×, so ~all 250 rows fire >1% on
every window (per-row ablation over the whole bank was rescoped to
group-wedge + top-20 dosage, a deliberate scope cut).

Group wedge — ΔCE recovered by removing each GROUP's delivery from the
window (positive = that group posts the tax):

| window | minus k3 rows | minus glm rows | minus mimo rows | minus OWN-home rows | minus FOREIGN rows |
|---|---|---|---|---|---|
| writing (tax +0.0262) | **+0.0266** | −0.0008 | −0.0007 | +0.0003 | **+0.0277** |
| math (tax +0.0331) | **+0.0341** | +0.0012 | +0.0002 | −0.0012 | **+0.0327** |
| base (tax +0.0079) | +0.0084 | +0.0000 | +0.0001 | — | — |

**The tax is foreign k3 cross-fire, full stop.** R4's attribution to the
glm/mimo family is measured WRONG (|wedge| ≤ 0.0012 both windows); writing's
own rows contribute nothing (+0.0003) and math's own rows slightly help
(−0.0012). Removing foreign-to-the-home delivery recovers 100–105% of the
tax (≈ additive: k3-foreign wedge ≈ own+foreign wedge within noise).

## 2. Lever definition (standing justification)

Mechanism: dense foreign sibling crowd, near-zero own-home contribution, and
(the dead λ-sweep) indifference to per-home λ. The deployable knob must act
at the ROUTE/delivery layer without home labels at inference: per-token
**delivery-mass budget** q over Σ λ_j·m_j of firing rows, with two
enforcement variants:

- **recal=on (renorm):** over-budget tokens scale all firing weights by
  q/Σ (order-preserving — keeps strong/own fires dominant only if they out-
  margin the crowd; weakest-link-safe).
- **recal=off (exclusion):** drop lowest-MARGIN firing rows until the
  prefix budget holds (single-row overflow ≤ λ allowed) — the
  reject-after-exclusion reading of R5: rows whose own gate barely beats
  the 2nd stock row are the suspected foreign posters.

Margin evidence for the choice (claim marked): on writing, own-home fires
out-margin foreign (0.429 vs 0.359) so exclusion preferentially removes
foreign fire; on math the margins are inseparable (0.351 vs 0.353), so
exclusion there is expected to trade own-vs-foreign roughly at random —
the sweep's math column is the discriminative read between the variants.

Sweep cells: quota ∈ {0.25, 0.5, 0.75, ∞} × recal ∈ {on, off}. quota=λ=0.5
is the one-row-equivalent budget; ∞ must reproduce R4 (guard). Gate-side
stats (membership median, ceiling-relative p_home) are post-gate invariant
under delivery caps — carried from R4, stated once: 0.378 / 0.494 / 0.082.
Every cell also reports base CE (the no-free-lunch side).

## 3. Sweep table

Corrected judge carried from R4 (cell-invariant): membership median 0.378,
ceiling-relative p_home 0.494, p_base_max 0.082, 24-pair harm honest-recheck
at default worst +0.0012 PASS (repaired protocol; all prior values were
ablation-less noise, floor |ΔCE| = 0.000000). quota=∞ cells reproduce the
R4 lam_default numbers bit-matched (guard). One quota-semantics note: at λ
0.5 the exclusion branch keeps exactly top-1 row/token for any quota ≤ 0.5
and top-2 for 0.75 — q0.25_exclude and q0.5_exclude are provably the same
cell; the table keeps both rows to show it.

| cell | writing Δ | math Δ | worst home Δ | base Δ | bars | honest harm @cell |
|---|---|---|---|---|---|---|
| q0.25 renorm | −0.0010 | +0.0016 | +0.0016 | −0.0018 | **PASS all** | +0.0004 (24 pr) |
| q0.25 exclude | −0.0019 | +0.0086 | +0.0135 (cybersec) | −0.0017 | PASS | — |
| q0.5 renorm | +0.0000 | +0.0030 | +0.0030 | −0.0011 | **PASS all** | — |
| q0.5 exclude | −0.0019 | +0.0086 | +0.0135 | −0.0017 | PASS | — |
| q0.75 renorm | +0.0002 | +0.0041 | +0.0041 | −0.0012 | **PASS all** | +0.0011 (24 pr) |
| q0.75 exclude | −0.0003 | +0.0069 | +0.0089 | −0.0017 | PASS | — |
| q∞ renorm (= R4 default) | +0.0262 | +0.0331 | +0.0331 | +0.0079 | FAIL (W+M) | +0.0012 (24 pr) |
| q∞ exclude (= R4 default) | +0.0262 | +0.0331 | +0.0331 | +0.0079 | FAIL (W+M) | (same run) |

Budget-vs-tax curve: renorm tracks a monotone mass trade — q∞ +0.0262/+0.0331
→ 0.75 +0.0002/+0.0041 → 0.5 +0.0000/+0.0030 → 0.25 −0.0010/+0.0016; base
tax improves in lockstep (+0.0079 → −0.0012…−0.0018, never paid back).

## 4. Verdict + recipe

**VERDICT: LEVER FOUND.** Six of eight cells clear both home_quality bars
with base CE at-or-below masked and honest harm way under bar; the R4
growth-250 HOLD converts to PASS under the corrected judge plus this
delivery budget. The mechanism (foreign-k3 dense crowd, zero own-home
contribution, no dominant individual row — top-20 per-row dosage max
+0.0004) makes budget-class caps the exact shape of the fix.

**Corrected rung-1 recipe carry:** `GRAB / WRAP rank-64 / FEED ≥200k /
DELIVER additive λ 0.5 THROUGH a per-token delivery-mass budget q = 0.75
with renorm enforcement / ROUTE registry row owner_target 0.35`. The 0.75
renorm cell is the smallest intervention that passes (keeps the most
delivery mass of any passing cell; worst home +0.0041, base −0.0012). The
0.25 variant is the maximal-safety alternative (+0.0016/+0.0016/-0.0018,
harm +0.0004 measured) at the cost of muting delivery to ~7% of nominal
crowd mass; exclusion variants are equivalent but coarser (top-1/top-2
per-token truncation) and slightly worse on cybersec (+0.0135).

Open risk (claim, unmeasured): bar-clearing is neutral-CE clearing — under
the budget the in-delivery specialty signal is muted; whether specialty
function needs more delivery than the 0.75 budget supplies is not measured
by these windows. Queue: solo-recovery style specialty-dosage check under
the budget before rung-2 claims delivery value at 250. The writing-λ-dead
evidence (R4) and the writing-own-row wedge (+0.0003) both suggest
in-distribution windows are insensitive to delivery strength, so the
specialty question is the honest gate, not another home window.

Honest harm re-measures (repaired ablation): default worst **+0.0012**,
recipe cell worst **+0.0011 (24 pr)** — §11.2 #2 stays PASS, now for the first time
as a real measurement. R4's "~0.0000" and the box growth rows' ±0.0008
columns were harness noise and are annotated as such.
