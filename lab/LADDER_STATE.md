# LADDER_STATE — banked-MoE graft/ladder audit (2026-10-19)

Consolidation audit of the rung-1 bank (`/data/tliao/rung1` on the MI50
compute box) against `MODULAR_MOE.md` §10/§11 and `README.md` §3.7. Read-only
checks; nothing restarted.

## 1. Proven (measured, evidence on file)

| claim | headline numbers | evidence |
|---|---|---|
| D0 graft parity (OLMoE-1B-7B) | logit cos **0.99913**, ΔCE +0.00075 | `lab/graft_d0_parity.py`, README §3.2 |
| D1 plug-in onto graft | formal PASS; quality left open (closed later by Door-2.5) | `lab/graft_d1_plugin.py`, README §3.2 |
| D2 cross-family socket (Phi-mini-MoE) | logit cos **0.99863**, ΔCE 0.030 | `lab/graft_d2_crossfamily.py`, README §3.2 |
| Door-2.5 wrap quality | foreign core wrapped: quality **0.439** vs 1.072 masked — first wrapped quality pass | `lab/door25_wrap_expert.py`, README §3.3 |
| Door-6 mix | two grafted donors = **sum-of-parts exactly** (0.9957×), zero cross-contamination | `lab/door6_expert_mix.py`, README §3.3 |
| Door-10 calibration scale | memorization gap 1.35 → 0.003 (**437×**) at 200k FEED tokens, held gap +0.0003 = parity; 10–12k is ~20× under and actively harmful held-out; 500k adds nothing | `lab/door10_calscale.py`, `lab/door10_build_calscale.py`, README §3.3 |
| Standard recipe | GRAB / WRAP rank-64 / FEED ≥200k / DELIVER additive-λ / ROUTE registry row — frozen 2026-10-13 | MODULAR_MOE.md §10.7, README §2 |
| bank2n (8 cores, 2 families) | **GO with conditions**: p_home 0.98–0.995 all rows; base partition tax −4.3%; pairwise harm 0.226 worst (bar 0.02) → ~10× shrink with cross-fire rejection | `lab/bank2n_partition.py`, `lab/bank2n_trainslice.py`, README §3.4 |
| Rung-1 extract + fit | **250/250 plugin rows fitted** (175 k3, 40 glm, 35 mimo); fit completed 2026-10-16 08:35 UTC after reboot-resume | `/data/tliao/rung1/results/extract.json`, `fit.json`, `fit_lane*.json`, `/data/tliao/rung1/logs/fit_resume.log` |

### §11 break-signals table (rung-1 growth, 325M-spine router)

Evidence: `/data/tliao/rung1/results/growth_{32,64,128}.json`; summary in
README §3.7.

| growth n | home_quality (bar 0.02) | pairwise_harm (bar 0.02) | base_tax (bar 0.05) | p_home (bar 0.5) | broken criteria |
|---|---|---|---|---|---|
| 32 | PASS | PASS (worst 0.0011) | PASS (Δ+0.0009) | **FAIL** (median 0.136, 26/26 collapsed) | §11.2 #3 |
| 64 | PASS (worst Δ+0.009) | PASS (0.0007) | PASS (Δ+0.0027) | **FAIL** (median 0.1428, 38/38) | §11.2 #3 |
| 128 | **FAIL** (writing Δ+0.0239 — first break) | PASS (0.0008) | PASS (Δ+0.0056) | **FAIL** (median 0.1464, 83/83) | §11.2 #1 + #3 |
| 250 | **FAIL** (writing +0.0262, math +0.0331) | PASS (worst +0.0012, R5-corrected; ~0.0000 was ablation-less noise) | PASS (+0.0079) | n/a (metric replaced R2) | §11.2 #1; home_quality fixed by R5 recipe → PASS |

Key facts: n=128 is the **first size to break home quality**; p_home is
**chronically collapsed at every size from n=32** (median ~0.14, never
passes 0.5). Pairwise harm and base tax never broke. **(Updated 2026-10-19
by the R2 probe below: the p_home collapse is a metric artifact, not a
routing state — see §3 note and `lab/PHOME_CALIBRATION.md`.)**

### R2 p_home calibration probe (2026-10-19, laptop RTX 3070, clean)

Queue item §3 #2 executed and CLOSED: `lab/PHOME_CALIBRATION.md`,
`lab/phome_calibration.py`, evidence `lab/results/phome_calibration.json`
(local). Reproduction on the synced 128-row registry: per-row p_home median
0.1363 vs box 0.136 (±0.001/row), masked CE ±0.003/home, writing break
+0.0233 vs +0.0239. Grid {owner_mass 0.35–1.4 × s320 arm × gains 1/2/4,
11 cells × 128 rows, no re-fitting} + assembled-bank pass + CE side-effects:
**verdict = metric miscalibration, neither calibration-wins nor §11
capacity.** p_home ≥ 0.5 is unreachable-by-construction on this bank
(tie-top-stock ceiling 0.306 per-row-world / 0.108–0.141 assembled; the bar
sits ~2–3 nats below the router's own beat-the-2nd-stock firing rule);
assembled routing is healthy through n=128 (home-firing median 0.330→0.360
→0.374 at 32/64/128 — flat-to-rising); every calibration knob that lifts
p_home buys it in base share and breaks the delivery bars (owner0.85 "doc
recipe": worst home +0.057, base +0.036; gain4/s320 worst +0.16–+0.25).
Also measured: recipe text says owner_mass 0.85 but rung-1 fits ran 0.35
(BAR_OWNER_SPEC); growth_128's p_home bucket saw only 83/128 rows
(glob-dedup artifact). Ladder consequences: replace the §11.2 #3 tripwire
with membership ≥ 0.5 + ceiling-relative p_home (p_base_max ≤ 0.01),
keep owner 0.35 rows, and the real past-128 watch items are membership
median and the writing-home Δ — the spine upgrade is NOT justified on
p_home grounds (§11.2 #4 remains parked; the bake-off pick only gates the
quality ladder, not a capacity rescue).

### R3 re-judge of n=128 (2026-10-19, same laptop, no re-fit)

`lab/PHOME_REJUDGE.md`, `lab/phome_rejudge.py` (reproduces the R2 assembled
table exactly). Under the corrected §11.2 #3: ceiling-relative p_home 0.530
(≥ 0.5 ✓), membership trend 0.330→0.360→0.374 flat-to-rising, old
p_home-collapse confirmed artifact (0.136/0.1428/0.1464 reproduces the box
records). Literal sub-bars memb ≥ 0.5 (reads 0.374; needs owner 0.70, which
breaks home_quality 2×) and p_base_max ≤ 0.01 (reads 0.082; sub-uniform
knapsack) are bar-vs-recipe inconsistencies, flagged for re-pinning.
**Official n=128 readout: HOLD — saturation prong cleared, the writing
home-quality break (Δ+0.0233) is the only live bar.** §11.2 #4 NOT
triggered.

### R4 growth-250 measured (2026-10-19, same laptop, ~0.35 GPU-hr)

`lab/GROWTH_250.md`, `lab/growth_250_measure.py`, record
`/home/tliao/rung1/results/growth_250.json` (local; 122 plugin rows + 73
cores rsync'd from the box, only the remainder). Routing healthy: membership
0.378 (chain 0.330→0.360→0.374→0.378 — decelerating rise, saturating
≈0.38), ceilrel 0.494 (band 0.49–0.54), p_base_max frozen at 0.082. First
measured crowding signal is in DELIVERY space, not routing: home-quality
breaks on 2 homes (math Δ+0.0331 new — posted by the glm/mimo rows,
writing Δ+0.0262 worsening), base tax +0.0079 (PASS), pairwise harm ~0
(24/24, PASS). Writing λ-balance swept 0.5/0.35/0.25: writing CE flat
±0.001 — the λ lever is dead; residual lever class = per-home quota /
reject-side recal. **Verdict: growth-250 = HOLD (delivery crowding
grows with n while routing stays healthy; spine upgrade still NOT
triggered).** Resume recipe line: owner 0.35 + identical wrap/fit protocol,
corrected watch items (membership trend, ceilrel band, p_base_max, per-home
delivery tax — writing+math canaries) every doubling.

### R5 delivery-budget lever — 250 HOLD resolved (2026-10-19, same laptop)

`lab/REJECT_RECAL.md`, `lab/reject_recal.py`, evidence
`lab/results/reject_recal.json` (local). Mechanism: the home_quality tax is
a dense **foreign k3 sibling cross-fire crowd** (mean 7.16/6.29/4.50 rows
firing per token on writing/math/base; wedges: k3 pays +0.0266/+0.0341 of
the +0.0262/+0.0331 taxes, glm/mimo ≈ 0 — R4's glm/mimo attribution was
measured WRONG; own-home rows ≈ 0 or negative; no dominant individual row,
top-20 per-row dosage max +0.0004). Lever: per-token delivery-mass budget,
8 cells {quota 0.25/0.5/0.75/∞} × {renorm/exclude} — **6/6 finite-quota
cells pass all bars**; ∞ reproduces R4 bit-matched (guard).
**Corrected rung-1 recipe: DELIVER additive λ 0.5 THROUGH per-token budget
q=0.75 renorm** (smallest passing intervention: writing +0.0002, math
+0.0041, base −0.0012; q0.25 = maximal-safety variant, harm +0.0004).
**growth-250 verdict: HOLD → PASS under the corrected judge + R5 recipe.**

### R6 specialty-dosage gate (2026-10-19, same laptop, ~1.7 GPU-hr)

`lab/SPECIALTY_BUDGET_GATE.md`, `lab/specialty_budget_gate.py`, evidence
`lab/results/specialty_budget_gate.json` (local). CE channel (carried
R4/R5, all delivery levels): no home window beats stock-masked by a clear
margin anywhere (best −0.0035). Generative battery (stock/default/q0.75;
q0.50 + 4 talks scope-cut, documented): 18/18 completed verdicts PASS —
uniformly non-discriminative. **VERDICT: CE-ONLY** — the rung-1 assembly
is neutrality-certified at every delivery strength; neither muted nor
unmuted delivery wins anything on laptop-visible channels (consistent
with Door-10's "neutrality, not generalization" certification).
Official rung-1/2 recipe line (carried): rank-64 / FEED ≥200k /
λ 0.5 / per-token budget q=0.75 renorm / owner 0.35 + corrected judge.

### R7 held-out value probe (2026-10-19, same laptop, ~1 GPU-hr)

`lab/HELDOUT_BANK_VALUE.md`, `lab/heldout_bank_value.py`, evidence
`lab/results/heldout_bank_value.json` (local). 13 windows × 3 arms:
held-out feed tails (4k tok, all 8 homes, provably outside fit batches
0..781) + D0-corpus texts (5 homes, structurally disjoint corpora).
**Verdict MIXED — rung-2 gate stays CLOSED:** the q0.75 recipe arm is
within ±0.0029 of stock on all 13 windows (neutral-everywhere, as
designed); the unmuted default arm wins in exactly 2 families (chat tail
−0.0041 4k-tok + chat d0 −0.0090; science d0 marginal −0.0035) while
posting a real held-out crowding tax elsewhere (web +0.0243, math
+0.0189). The bank is proven-neutral-with-one-chat-hint, not
proven-valuable; the win home is the base's own domain (base-over-fit
hypothesis). Named next electron: per-home λ re-fit on held-out tails
under the budget (~1 GPU-hr); failing wins there, retire rung-1's claim
to neutral modular assembly and let the bake-off line carry the value.
Side-effect: the growth harm protocol's ablation was state-wiped for a
week (all prior harm columns = noise; FAILURES.md); corrected 24-pair harm:
worst **+0.0012** default / **+0.0011** at the recipe cell — §11.2 #2 PASS,
first honest measure. Open: specialty dosage under the budget (muted
delivery ~21% of nominal crowd mass gains CE-neutrality; whether specialty
function needs more is unmeasured — the queue's next gate, see §3 #4).

## 2. Deliberately NOT done (and why)

- **Door-9 / D1.5 feature translators: dead, do not retry without a new
  idea.** Cross-model hidden-space translation caps at ~0.5 held-out
  alignment; exact token-paired supervision (Door-9) still only 0.47,
  W-init doesn't beat random. Listed in §10.7 killed branches. Evidence:
  `lab/door9_embedding_w.py`, `lab/graft_d15_translator.py`,
  `lab/FAILURES.md`.
- **growth-250 as of R5 is measured and PASSING under the corrected judge
  + the R5 delivery budget** (`lab/GROWTH_250.md`, `lab/REJECT_RECAL.md`).
  The 2026-10-16 halt (history) was triggered by #1 and the old #3; R2/R3
  showed #3 was a metric artifact. What remains deliberately NOT done:
  **growth past 250 (rung 2)** — gated on the specialty-dosage gate (does
  the budgeted delivery still carry specialty signal, §3 #4), not on any
  measured bar. The bigger-spine
  pick (GLM-5.3-Flash vs MiMo-V2.6-Flash bake-off — user's conversation-feel
  metric) stays a quality-axis decision, NOT a capacity rescue.
- **fit_resume.sh: finished, not running.** It exited cleanly 2026-10-16
  08:35 UTC with 250/250 plugins ("fit complete … exit"). Lock files in
  `/data/tliao/rung1/locks/` are stale artifacts of that run; no fit or
  growth process is alive. Scripts live on `/data` (not `/tmp`) and do not
  auto-start after reboot — intentional per the watchdog lesson (README §3.6).

## 3. Next experiments (bigger-spine-router era, ordered by 1-GPU-hour cost)

As of R3/R4 (2026-10-19) the same-spine ladder is measured through 250 and
the queue split: crowding-fix items run on the laptop NOW; spine-port items
wait on **the bake-off spine pick** (user's feel metric, GLM-Flash vs
MiMo-Flash arms already built). In cost order:

1. **(~0.1 GPU-hr) Socket sanity on the picked spine** — D0-class
   registry/graft parity (port both arms' parity results to one chosen
   spine). Confirms the router block adaptation before any bank work.
2. ~~**( ~0.5–1 GPU-hr) p_home calibration probe**~~ **DONE 2026-10-19
   (R2, laptop 3070 ~1.5 GPU-hr incl. CE pass)** — verdict: metric
   miscalibration, neither calibration nor §11 capacity; see §1 note and
   `lab/PHOME_CALIBRATION.md`. Consequences now baked into the queue: (a)
   §11.2 #3 needs replacing (membership ≥ 0.5 + ceiling-relative p_home)
   before the next growth run can even measure itself; (b) the spine
   upgrade is no longer justified by router saturation — the bake-off pick
   stays a QUALITY axis (conversational feel), the writing-home Δ+0.023
   (λ-balance / crowding) is the only live break; (c) growth resume should
   log membership + ceiling stats from here on (zero extra cost, cached
   taps).
3. ~~**(~1–2 GPU-hr) Delivery-crowding lever at n=250 (laptop)**~~ **DONE
   2026-10-19 (R5, ~1.1 GPU-hr)** — lever FOUND: dense foreign k3 cross-fire
   crowd is the tax; per-token budget q=0.75 renorm clears all bars
   (`lab/REJECT_RECAL.md`). Rung-1 corrected recipe line now carries the
   budget on the DELIVER step.
4. ~~**Specialty-dosage gate under the budget**~~ **DONE 2026-10-19
   (R6) — verdict CE-ONLY.** Rung-2 stays closed pending the named lever:
   a value-positive probe where stock is demonstrably weak (Door-2.5
   s6_quality at bank scale, held-out donor-domain per family,
   ~1 GPU-hr laptop). See `lab/SPECIALTY_BUDGET_GATE.md`.
5. **(~1–2 GPU-days) Rung-2 kickoff**: manifest/donor-carryover design for
   ~500 cores (≈16B target per §11.1), k=top-8. Prep is CPU-only and can
   start before #4 finishes. The bake-off spine pick, when it lands, enters
   here as the quality axis — port = D0-class socket sanity (~0.1 GPU-hr)
   + re-run of the growth chain on the new spine.

Note: the ladder doc's north star (§11) requires the eventual co-trained big
spine; everything above is the measured-honesty path toward it.
