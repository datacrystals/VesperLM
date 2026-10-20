# TEACH_SHIELD — shield-everything-every-session (the scheduling-question killer)

**STATUS: PARTIAL — committed mid-run at the session wall.**
Teach phase S1..S6 complete and **bit-identical to the committed cadence
run**; the shield probe after S6 has seeded the S7 batch and **S7 (the
first 20-triple shield batch) is in training**. The shield process is
left running to completion on its own device; this doc records the design,
the determinism chain, and every measured number available at commit time.
Sections marked *(pending)* fill from `lab/teach_shield_out/teach_shield.json`
in a follow-up commit once the run reaches the horizon.

Queued arm ("the shield arm", item #10 of `lab/TEACH_PIPELINE.md`), run
2026-10-19 on the laptop, CPU only, fp32. After the three-arm scheduler
falsification chain — v1 (`lab/TEACH_AGED_SCHED.md`, age-weighted), v2
(`lab/TEACH_AGED_SCHED_V2.md`, age-from-first-trip + floor), v3
(`lab/TEACH_AGED_SCHED_V3.md`, deficit-dominant + serve budget) ALL
explicitly negative against the trigger arm's bar
(`lab/TEACH_TRIGGER_REPAIR.md`: 4/6 held ≥ 0.50, 0 dead, mean 0.581), with
v3's line that matters (**serve content dominated order; the hog died but
the victim set varies by rule**) — this arm deletes the queue entirely.
No priorities, no thresholds, no queues, no eviction: **every alive item
gets one phrasing-diverse re-exposure triple in every scheduler-phase
session.** The complementary measurement that makes this a real candidate:
the cadence arm (refresh-every-6-session re-exposure held 4/6) and the
decay A/B (`lab/TEACH_DECAY_REPLAY.md`: 4/4 direct item contrasts in
replay's favour) both measured that small phrasing-diverse re-exposure
works per contact.

Harness: `lab/teach_shield.py` (same import chain as v3:
`lab/teach_aged_sched_v3.py` → … → `lab/teach_spaced_cadence.py`;
deterministic seeded train loop inherited unchanged). Numbers-only results
(when the run completes): `lab/teach_shield_out/teach_shield.json`; log
`lab/logs/teach_shield.log`; calibration workdirs
`lab/teach_shield_calib/{A,B}`.

## 0. Determinism verification (done before the arm)

Two identical shield-harness calibration teaches (`--only metric_units
--seed 7`, fresh workdirs, sequential, same code path the arm uses), before
the arm started:

| run | baseline margin | post-teach margin | gain |
|---|---|---|---|
| calib A | −1.12088942527771 | **−0.14622879028320312** | +0.975 |
| calib B | −1.12088942527771 | **−0.14622879028320312** | +0.975 |

Post-teach margins **bit-identical across the two replays**, and both
**bit-match the committed seed-7 calibration of all five prior harnesses**
(cadence / trigger / aged-v1 / aged-v2 / aged-v3) — the sixth harness in
the chain inherits the deterministic train loop exactly.

In-run cross-check (teach phase bit-identity vs the committed cadence run):

```
CADENCE-IDENT: teach phase S1..S6 vs committed cadence run —
max |margin Δ| = 0.000e+00 (BIT-IDENTICAL; worst before/fav_show)
```

## 1. Design (documented before running)

| item | value |
|---|---|
| checkpoint | `lab/imported/vk_sft_r3/step_1800/chat_model` — identical to E3 / A/B / cadence / trigger / v1 / v2 / v3 |
| runtime | laptop CPU, fp32, 12 threads; tier-2 Immune never leaves CPU (I7); LoRA rank 8, alpha 16, lr 2e-3, kl_coef 0.05, 24 steps/session — identical knobs |
| seed | 7 on every consolidate call |
| library | the six items verbatim: fav_show (F), math_words (S), tea_tin (F), metric_units (S), pet_dog (F), greet_rule (S) |
| teach phase | S1–S6, **identical to v3 / v2 / v1 / trigger / cadence** (bit-checked in-run) |
| shield phase | S7–S12 (6 later sessions): generic-chat filler + ONE shield triple per alive item — batch grows, that is the point |
| alive rule | `alive(k) = {i : r_i(k) ≥ 0.10}`, re-evaluated at the probe after every session, no memory; buried items (r < 0.10) excluded per measured doctrine (v1 rescue: strict-dead contact +0.063 and re-decay — contacts into dead items are wasted) and re-enter only on unaided recovery across 0.10 |
| shield unit | exactly ONE approve-only triple per alive item per session; response string = the pool entry's correction phrasing (the lineage's supervision-target form; the natural script_ok form stays out of the batch so the unit is exactly 1 triple) |
| batch rule | `n_filler = max(5, ceil(alive_n / 0.30) − alive_n)`; sizes: alive 6 → 20 (30.0%), 5 → 17 (29.4%), 4 → 14 (28.6%), 3 → 10 (30.0%), 2 → 7 (28.6% = the trigger batch shape), 1 → 6 (16.7%); share asserted ≤ 30% per session |
| variants | the shared replay_state counter continues from the teach phase (8-variant pools; assert < 8 uses per item) |
| filler allocation | cyclical pointer over the 30-entry pool — **deviation, documented**: cross-session filler reuse (within-batch entries always distinct); filler is scripted generic approvals, not re-teach content; the tripwire that cared is within-batch |
| tripwire safety | unchanged: response strings unique run-wide for re-exposure strings; poison canary; one-fallback policy |
| bar | unchanged: ≥ 4/6 held r_final ≥ 0.50, 0 dead (r ≤ 0.06), mean ≥ 0.581 — the trigger arm's committed criteria |
| tier 2 | ONE Immune merge verdict at the end, constants untouched |

### Named monitors (declared before running)

1. **Base-class interference**: does blanket shielding hurt anything the
   trigger's targeted serves protected? Per-item r trajectories/horizon vs
   the committed trigger table (reused, not re-run); and whether
   shield-phase train-loss / top_response_fraction profiles stay
   tripwire-cold at 20-triple batches.
2. **The 1/6-poverty case**: if the alive set shrinks, per-probe batch
   composition is logged; with one alive item its shield triple is 16.7%
   of a 6-triple batch — the smallest share any shielded item sees.
3. **Shield contact trace**: r around every (session, item) contact
   (bystander-coupled yields — reported as trace, not clean per-contact
   repair).

### Claims honesty (declared before running)

- If shield meets the trigger bar on all criteria → the queue scheduler is
  **OBSOLETE**; the no-skyline answer is the slice constant.
- If it falls short → the verdict line is *"architecture-straw-limit,
  schedules dead, adaptive capacity next"* and the follow-up (multi-adapter
  banks / class-segregated adapters vs a capacity knob) is named from the
  measured shield trace.

## 2. Verdict lines (as run so far; generation tails trimmed)

```
VERDICT S1  [fact/fav_show]      gate=stub-tier1: PROMOTE | top_resp_frac 0.20 n_triples 5 train loss 0.534->-7.892  | margin +1.718 -> +4.451
VERDICT S2  [style/math_words]   gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> fav_show) train loss 6.527->2.127 | margin -3.374 -> +1.863 | word-digit -7.337 -> +5.183
VERDICT S3  [fact/tea_tin]       gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> math_words) train loss 7.192->5.371 | margin -1.266 -> +12.042
VERDICT S4  [style/metric_units] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> tea_tin) train loss 12.232->0.463 | margin -1.121 -> +1.178
VERDICT S5  [fact/pet_dog]       gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> metric_units) train loss 6.848->2.297 | margin +0.264 -> +5.162
VERDICT S6  [style/greet_rule]   gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> pet_dog) train loss 6.501->0.779 | margin -2.570 -> +3.468
CADENCE-IDENT: teach phase S1..S6 vs committed cadence run — max |margin Δ| = 0.000e+00 (BIT-IDENTICAL; worst before/fav_show)
GAINS g_i: fav_show +2.733  math_words +5.237  tea_tin +13.308  metric_units +2.298  pet_dog +4.898  greet_rule +6.038
VERDICT SHIELD probe after S6: alive [fav_show, math_words, tea_tin, metric_units, pet_dog, greet_rule] (6/6, zero burials) | batch plan: 14 filler + 6 shield triples = 20 (share 30.0%)
```

S7 (the first 20-triple shield batch) was in training at commit time —
consolidate launched at 1021% thread-CPU on a loaded box, micro-steps
progressing normally (loss 9.94 → 3.08 at step 4/24). Gate behaviour so
far: **6/6 promotes, 0 tripwire events**, matching every prior arm's teach
phase; the shield-phase train/top-response profiles are *(pending)*.

## 3. Shield contact trace + batch trace

| probe after | alive | buried | batch plan | shield share |
|---|---|---|---|---|
| S6 | 6/6 | none | 14 filler + 6 shield = 20 | 30.0% |

Entry state is the series' canonical one (bit-identical to every arm's S6
probe: fav_show r 1.26, math_words 1.23, tea_tin 0.324, metric_units 0.249,
pet_dog 0.153, greet_rule 1.00 — all ≥ 0.10, so the full library enters
the shield). Post-S6 probes, per-contact trace, batch-size evolution, and
any burials: *(pending)*.

## 4. Retention r_i(k) — shield vs trigger (committed numbers)

S1..S6 are bit-identical across all arms (CADENCE-IDENT). The shield rows
through S6 therefore equal the committed table; S7+ cells are *(pending)*.
The trigger column below is the committed trigger-arm table (reused, not
re-run); the **bold** cells mark the trigger arm's served re-teaches.

| item | g_i | arm | S6 | S7 | S8 | S9 | S10 | S11 | S12 r_final |
|---|---|---|---|---|---|---|---|---|---|
| fav_show (F) | +2.73 | trigger | 1.26 | 0.26 | 0.44 | 1.40 | 0.82 | 0.61 | **0.96** |
| | | shield | 1.26 | *(pending)* | | | | | |
| math_words (S) | +5.24 | trigger | 1.23 | 0.92 | 0.20 | **0.64** | 1.02 | 0.83 | 0.66 |
| | | shield | 1.23 | *(pending)* | | | | | |
| tea_tin (F) | +13.31 | trigger | 0.32 | 0.09 | 0.06 | 0.19 | 0.06 | 0.09 | 0.17 |
| | | shield | 0.32 | *(pending)* | | | | | |
| metric_units (S) | +2.30 | trigger | 0.25 | 0.20 | −0.25 | 0.19 | **−0.14** | **1.11** | 0.69 |
| | | shield | 0.25 | *(pending)* | | | | | |
| pet_dog (F) | +4.90 | trigger | 0.15 | **0.52** | 0.39 | 0.57 | 0.77 | 0.14 | 0.50 |
| | | shield | 0.15 | *(pending)* | | | | | |
| greet_rule (S) | +6.04 | trigger | 1.00 | 0.23 | **0.81** | 0.48 | 0.45 | 0.42 | 0.50 |

The bar the shield must clear (committed trigger horizon): **4/6 held
≥ 0.50, 0 dead (r ≤ 0.06), mean 0.581.** Shield horizon: *(pending — the
run is at S7).*

## 5. The monitors (current state)

1. **Base-class interference / main-teach cost**: nothing measurable yet —
   by construction the shield cannot have cost anything through S6 (teach
   phase bit-identical, shield only starts S7). The cost read is the
   S7..S12 trajectory comparison vs the committed trigger table once the
   shield phase lands; plus the tripwire-coldness of 20-triple batches
   (S7's verdict line, *(pending)*). **What can already be said**: the
   blanket-shield batch shape (30.0% re-exposure share) holds the
   documented ≤30% cap exactly and its entry-probe gate behaviour is
   unchanged.
2. **1/6-poverty case**: not exercised so far — alive set is 6/6 at the
   entry probe; no burials have occurred.
3. **Hog / starvation check**: **none by construction** — there is no
   queue, no priority, no threshold, and no eviction: service is the
   constant (every alive item, every session), so no item can hog a
   resource and no trip can starve. The failure mode to watch instead is
   the flip side: whether constant-service *dilutes* per-item contact
   strength (1 triple vs the lineage's 2-triple serves) and/or floods
   stable items with un-needed re-exposure (base-class interference).

## 6. Tier-2 merge verdict

*(pending — reached only at run end; expected ROLLBACK per the series'
seven prior readings.)*

| arm (context) | aggregate inc → cand | protected agg | drift | decision |
|---|---|---|---|---|
| cadence | 0.3156 → 0.0801 | 0.255 → 0.030 | 1.000 (fail) | ROLLBACK |
| trigger | 0.3156 → 0.0740 | 0.255 → 0.043 | 1.000 (fail) | ROLLBACK |
| aged v1 | 0.3156 → 0.0787 | 0.255 → 0.061 | 1.000 (fail) | ROLLBACK |
| aged v2 | 0.3156 → 0.1073 | 0.255 → 0.056 | 1.000 (fail) | ROLLBACK |
| aged v3 | 0.3156 → 0.0475 | 0.255 → 0.006 | 1.000 (fail) | ROLLBACK |
| shield (this) | *(pending)* | *(pending)* | *(pending)* | *(pending)* |

## 7. Claims and limits (partial-run claims only)

**Claims (what the partial run already supports).**

1. *The shield harness inherits the deterministic train loop exactly*:
   calibrations bit-identical twice and bit-matching all five prior
   harnesses; teach phase CADENCE-IDENT max |Δ| = 0.0 against the
   committed cadence run. Every later divergence from the committed arms
   will be shield-attributable by construction.
2. *The protocol arms itself*: the probe-after-S6 evaluation produced the
   documented alive set (6/6) and the documented batch shape (14 filler +
   6 shield = 20 triples, 30.0% share — the documented cap, asserted
   in-code); filler cycling stays within-batch distinct.
3. *No scheduling failure modes exist in this arm by construction* (no
   queue) — the only open questions are the honest ones the design named:
   does 1-triple-per-item constant service hit the trigger bar (capacity),
   and what does blanket coverage cost (interference, wall)? Both are
   *(pending)*.

**Limits.**

- PARTIAL RUN: nothing in §4–§6 beyond S6 is measured; the doc's negative
  or positive verdict is *not yet available* and will land in the
  follow-up commit with the shield JSON on disk.
- The arm shares the series' limits: single seed (n = 1), horizon
  mid-oscillation caveat, variant-dependence of per-contact yield.
- Wall is box-load-dependent; the checkpoint/probe determinism is
  unaffected (margins are seeded).

## Reproduction

```bash
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_shield.py --workdir lab/teach_shield_out \
  | tee lab/logs/teach_shield.log

# determinism calibration (2 runs; both must bit-match each other and all
# five prior harnesses' committed seed-7 calibration):
python3 -u lab/teach_shield.py --only metric_units --seed 7 --workdir <fresh A>
python3 -u lab/teach_shield.py --only metric_units --seed 7 --workdir <fresh B>
```

Fresh workdir required (append-only session logs, invariant 10; adapter
deltas and `lab/teach_shield_out/`, `lab/teach_shield_calib/` are weights
and are not committed).
