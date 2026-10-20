# TEACH_SHIELD — shield-everything-every-session (the scheduling-question killer)

**STATUS: COMPLETE — explicitly NEGATIVE** (run 2026-10-19, finalized
2026-10-20 from `lab/teach_shield_out/teach_shield.json`; the PARTIAL doc
committed mid-run at d601351 is fully replaced). The pre-named verdict
line holds: **"architecture-straw-limit, schedules dead, adaptive capacity
next."** The strongest possible service policy at this capacity shape —
everything alive, every session, no queue to game — still misses the
trigger bar on all three criteria (2/6 held vs 4/6, 1 dead vs 0, mean
0.495 vs 0.581), spending 2.5× the trigger arm's re-exposure tokens. The
follow-up is named from the measured trace: **class-segregated adapters
(fact bank / style bank)** — this arm's residual loss is style-concentrated
(style r_final mean 0.286 vs fact 0.704) under identical service, and its
non-schedulable event is *within-batch* cross-item interference (the S7
mass-damage: 5/6 items crashed while ALL of them were being shielded).
Item #11 of `lab/TEACH_PIPELINE.md` carries it.

Queued arm ("the shield arm", item #10 of `lab/TEACH_PIPELINE.md`), run
2026-10-19 on the laptop, CPU only, fp32. After the three-arm scheduler
falsification chain — v1 (`lab/TEACH_AGED_SCHED.md`, age-weighted), v2
(`lab/TEACH_AGED_SCHED_V2.md`, age-from-first-trip + floor), v3
(`lab/TEACH_AGED_SCHED_V3.md`, deficit-dominant + serve budget) ALL
explicitly negative against the trigger arm's bar
(`lab/TEACH_TRIGGER_REPAIR.md`: 4/6 held ≥ 0.50, 0 dead, mean 0.581), with
v3's line that matters (**serve content dominated order; the hog died but
the victim set varies by rule**) — this arm deleted the queue entirely.
No priorities, no thresholds, no queues, no eviction: **every alive item
gets one phrasing-diverse re-exposure triple in every scheduler-phase
session.** The complementary measurement that made this a real candidate:
the cadence arm (refresh-every-6-session re-exposure held 4/6) and the
decay A/B (`lab/TEACH_DECAY_REPLAY.md`: 4/4 direct item contrasts in
replay's favour) both measured that small phrasing-diverse re-exposure
works per contact.

Harness: `lab/teach_shield.py` (same import chain as v3:
`lab/teach_aged_sched_v3.py` → … → `lab/teach_spaced_cadence.py`;
deterministic seeded train loop inherited unchanged). Numbers-only
results: `lab/teach_shield_out/teach_shield.json`; log
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
the chain inherits the deterministic train loop exactly. In-run:

```
CADENCE-IDENT: teach phase S1..S6 vs committed cadence run —
max |margin Δ| = 0.000e+00 (BIT-IDENTICAL; worst before/fav_show)
```

## 1. Design (documented before running, as run)

| item | value |
|---|---|
| checkpoint | `lab/imported/vk_sft_r3/step_1800/chat_model` — identical to E3 / A/B / cadence / trigger / v1 / v2 / v3 |
| runtime | laptop CPU, fp32, 12 threads; tier-2 Immune never leaves CPU (I7); LoRA rank 8, alpha 16, lr 2e-3, kl_coef 0.05, 24 steps/session — identical knobs |
| seed | 7 on every consolidate call |
| library | the six items verbatim: fav_show (F), math_words (S), tea_tin (F), metric_units (S), pet_dog (F), greet_rule (S) |
| teach phase | S1–S6, **identical to the five prior arms** (bit-checked in-run) |
| shield phase | S7–S12: generic-chat filler + ONE shield triple per alive item |
| alive rule | `alive(k) = {i : r_i(k) ≥ 0.10}`, re-evaluated per probe, no memory; buried items excluded per measured doctrine (v1 rescue: strict-dead contact +0.063 and re-decay); re-entry on unaided recovery across 0.10 |
| shield unit | exactly ONE approve-only triple per alive item per session; response = the pool entry's correction phrasing |
| batch rule | `n_filler = max(5, ceil(alive_n / 0.30) − alive_n)`; share ≤ 30% asserted per session (as run: 20/17/17/17/17/14 triples, shares 30/29/29/29/29/29%) |
| variants | the shared replay_state counter continuing from the teach phase (8-variant pools; max use 7 ≤ 8) |
| filler | cyclical pointer over the 30-entry pool (cross-session reuse, documented; within-batch distinct) |
| tripwires / fallback / poison / tier-2 | unchanged from the lineage |

## 2. Verdict lines (as run; generation tails trimmed)

```
VERDICT S1  [fact/fav_show]      gate=stub-tier1: PROMOTE | top_resp_frac 0.20 n_triples 5 train loss 0.534->-7.892  | margin +1.718 -> +4.451
VERDICT S2  [style/math_words]   gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> fav_show) train loss 6.527->2.127 | margin -3.374 -> +1.863 | word-digit -7.337 -> +5.183
VERDICT S3  [fact/tea_tin]       gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> math_words) train loss 7.192->5.371 | margin -1.266 -> +12.042
VERDICT S4  [style/metric_units] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> tea_tin) train loss 12.232->0.463 | margin -1.121 -> +1.178
VERDICT S5  [fact/pet_dog]       gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> metric_units) train loss 6.848->2.297 | margin +0.264 -> +5.162
VERDICT S6  [style/greet_rule]   gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> pet_dog) train loss 6.501->0.779 | margin -2.570 -> +3.468
CADENCE-IDENT: teach phase S1..S6 vs committed cadence run — max |margin Δ| = 0.000e+00 (BIT-IDENTICAL; worst before/fav_show)
GAINS g_i: fav_show +2.733  math_words +5.237  tea_tin +13.308  metric_units +2.298  pet_dog +4.898  greet_rule +6.038
VERDICT SHIELD probe after S6:  alive 6/6 (zero burials) | batch 14 filler + 6 shield = 20 (share 30%)
VERDICT S7  [shield/6-alive]     gate=stub-tier1: PROMOTE | top_resp_frac 0.05 n_triples 20 (shield 6, 30%) train loss 9.940->2.254
VERDICT SHIELD probe after S7:  alive 5/6 | BURIED fav_show r=0.071 | batch 12 filler + 5 shield = 17 (share 29%)
VERDICT S8  [shield/5-alive]     gate=stub-tier1: PROMOTE | top_resp_frac 0.06 n_triples 17 (shield 5, 29%) train loss 2.538->0.459
VERDICT SHIELD probe after S8:  alive 5/6 (fav_show re-entered r=0.368) | BURIED greet_rule r=0.062 | batch 17 (share 29%)
VERDICT S9  [shield/5-alive]     gate=stub-tier1: PROMOTE | top_resp_frac 0.06 n_triples 17 (shield 5, 29%) train loss 2.791->0.216
VERDICT SHIELD probe after S9:  alive 5/6 (greet_rule re-entered r=0.191) | BURIED metric_units r=0.016 | batch 17 (share 29%)
VERDICT S10 [shield/5-alive]     gate=stub-tier1: PROMOTE | top_resp_frac 0.06 n_triples 17 (shield 5, 29%) train loss 2.146->0.131
VERDICT SHIELD probe after S10: alive 5/6 | metric_units stays buried r=0.039 | batch 17 (share 29%)
VERDICT S11 [shield/5-alive]     gate=stub-tier1: PROMOTE | top_resp_frac 0.06 n_triples 17 (shield 5, 29%) train loss 1.535->0.121
VERDICT SHIELD probe after S11: alive 4/6 | metric_units stays buried r=-0.057 | BURIED pet_dog r=-0.010 | batch 14 (share 29%)
VERDICT S12 [shield/4-alive]     gate=stub-tier1: PROMOTE | top_resp_frac 0.07 n_triples 14 (shield 4, 29%) train loss 1.532->0.172
HORIZON: 2/6 items r_final >= 0.5; 1/6 r_final <= 0.06 (dead slot); mean r_final 0.495; min -0.186
VERDICT S13 [poison/banana-batch] gate=stub-tier1: ROLLBACK — target collapse: one response is 100% of the batch | incumbent adapter intact: True
VERDICT MERGE [accumulated adapter vs raw base] gate=immune-tier2: ROLLBACK — protected probe refuse-color-seven regressed 0.408 -> 0.244 | protected probe ho-refuse-weigh regressed 0.397 -> 0.236 | aggregate 0.3156 -> 0.1739, drift 1.000 label=fail
```

Gate behaviour: **12/12 promotes, 0 tripwire events** — 20-triple batches
stay tripwire-cold (top_resp_frac 0.05–0.07 vs the teach phase's 0.14,
every shield string fresh). The poison canary tripped the collapse rule at
100%, incumbent byte-untouched. The one-fallback policy was not exercised;
no `lab/FAILURES.md` entry.

## 3. Shield contact trace + burial trace

30 shield-phase contacts in 6 sessions (6+5+5+5+5+4; the trigger arm used
12 re-exposure triples over the same span — shield spent **2.5×** the
tokens). Batches by probe: 20, 17, 17, 17, 17, 14 triples; min alive-set
size 4 (the 1/6-poverty case never exercised).

| item | per-session contact Δr (variant) | contacts | r_final |
|---|---|---|---|
| fav_show | **S7 −1.188 (v1)**, S9 +0.578, S10 +0.433, S11 −0.161, S12 +0.246 | 5 (buried S8) | **1.464** |
| math_words | S7 −0.641 (v1), S8 −0.037, S9 +0.144, S10 −0.268, S11 +0.051, S12 +0.257 | 6 | **0.734** |
| tea_tin | S7 −0.175, S8 +0.022, S9 +0.115, S10 +0.010, S11 +0.080, S12 −0.082 | 6 | 0.294 |
| metric_units | S7 **−0.142 (v1)**, S8 +0.099, S9 **−0.190 → buried** | 3 | **−0.186 (dead)** |
| pet_dog | S7 **+0.639** (v1), S8 −0.171, S9 −0.122, S10 −0.355, S11 −0.155 → buried | 5 | 0.354 |
| greet_rule | S7 −0.574 (v0), S8 −0.364 → buried, S10 +0.020, S11 +0.175, S12 −0.076 | 5 (buried S9) | 0.311 |

Burial trace: fav_show at probe-after-S7 (r 0.071, re-entered after S8 on
its un-shielded bounce 0.368), greet_rule at probe-after-S8 (0.062,
re-entered after S9 at 0.191), metric_units at probe-after-S9 (0.016,
**never re-entered**; decayed buried 0.039 → −0.057 → **ended dead at
−0.186**), pet_dog at probe-after-S11 (−0.010; un-shielded S12 bounce to
0.354).

**Hog/starvation check: none by construction** — there is no queue to win,
so the series' two scheduling failure modes cannot exist. What replaced
them is dilution and flood, §5.

## 4. Retention r_i(k) — shield vs trigger (committed numbers)

S1..S6 bit-identical across all arms. Shield contacts marked bold; `†` =
buried that session (excluded from service).

| item | arm | S6 | S7 | S8 | S9 | S10 | S11 | S12 (r_final) |
|---|---|---|---|---|---|---|---|---|
| fav_show | trigger | 1.26 | 0.26 | 0.44 | 1.40 | 0.82 | 0.61 | **0.96** |
| | shield | 1.26 | **0.07** | †0.37 | **0.95** | **1.38** | **1.22** | **1.46** |
| math_words | trigger | 1.23 | 0.92 | 0.20 | 0.64 | 1.02 | 0.83 | **0.66** |
| | shield | 1.23 | **0.59** | **0.55** | **0.69** | **0.43** | **0.48** | **0.73** |
| tea_tin | trigger | 0.32 | 0.09 | 0.06 | 0.19 | 0.06 | 0.09 | 0.17 |
| | shield | 0.32 | **0.15** | **0.17** | **0.29** | **0.30** | **0.38** | **0.29** |
| metric_units | trigger | 0.25 | 0.20 | −0.25 | 0.19 | −0.14 | 1.11 | **0.69** |
| | shield | 0.25 | **0.11** | **0.21** | 0.02 | †0.04 | †−0.06 | −0.19 |
| pet_dog | trigger | 0.15 | 0.52 | 0.39 | 0.57 | 0.77 | 0.14 | **0.50** |
| | shield | 0.15 | **0.79** | **0.62** | **0.50** | **0.14** | **−0.01** | †0.35 |
| greet_rule | trigger | 1.00 | 0.23 | 0.81 | 0.48 | 0.45 | 0.42 | 0.50 |
| | shield | 1.00 | **0.43** | **0.06** | †0.19 | **0.21** | **0.39** | **0.31** |

| metric | trigger bar | shield (this) | aged v3 | dead |
|---|---|---|---|---|
| held r_final ≥ 0.50 | 4/6 | **2/6 — NOT met** (fav_show, math_words) | 2/6 | metric_units |
| dead r ≤ 0.06 | 0 | **1 — NOT met** (metric_units −0.19) | 1 (fav_show) | |
| mean r_final | 0.581 | **0.495 — NOT met** (best of the post-trigger line) | 0.416 | |
| min r_final | 0.17 | −0.186 | −0.215 | |

## 5. The monitors / why it fails (mechanics, from the recorded traces)

1. **Base-class interference is real, batch-internal, and not
   schedulable.** The S7 batch — every item shielded, 30% cap respected —
   destroyed 5 of the 6 shielded items in one session (fav_show −1.19 r,
   greet −0.57, math −0.64, tea −0.18, metric −0.14; pet_dog the lone
   winner +0.64). Shielding an item did not protect it *inside* its own
   shield batch: fav_show nearly died on contact with 1.26 → 0.071 and
   was buried after its first shield. The series had already seen this
   one serve at a time (end-loaded destructive serves, v1/v2/v3 §5);
   blanket coverage shows the mechanism is **within-batch interference
   between co-served items**, present even when the co-served set is the
   entire alive library — i.e., not addressable by any queue at this
   batch shape. The later-line arms' trajectories mostly *recover* from
   S7 through S8–S11 under continued shielding (fav_show to 1.46, its
   series best; math to 0.73), so the carnage is recoverable for items
   with healthy mid-phase drift — but metric_units' collapse (§5.2) shows
   not for all.
2. **Main-teach cost read: the shield is dilution-weak per item, and
   breadth loses to targeted depth.** 30 re-exposure triples over the
   phase (vs trigger's 12) produced a WEAKER horizon on every bar metric.
   Every style item that trigger targeted with a 2-triple serve when it
   tripped ends lower under 1-triple blanket coverage (metric_units 0.69
   trigger vs −0.19 shield; greet_rule 0.50 vs 0.31), while shield wins
   exactly where trigger's queue failed (fav_show never served by
   trigger? no — served at S12, 0.96 vs 1.46 shield; tea_tin starved by
   trigger, 0.17 vs 0.29 shield). Net: the triggers's "2 triples for the
   one item that needs it" beats "1 triple for everyone, always" at ~40%
   of the token spend. This extends v3's named lesson — contact CONTENT
   (strength on the right item) beats contact BREADTH — from ordering to
   granularity.
3. **Burial doctrine did what it was designed to do and one item died for
   it.** metric_units' shield contacts (−0.142 v1, +0.099, −0.190) could
   not hold it; it crossed the 0.10 alive line after S9, was excluded
   thereafter (doctrine: dead-band contacts are wasted capacity), and
   drifted buried to −0.186 — the series' fourth metric_units death
   (cadence 0.06, aged v1 −0.12; trigger rescued it via two big serves
   0.69; v2 via two 0.78). The buried-without-contact tail
   (0.039 → −0.057 → −0.186 in three sessions un-served) is the honest
   cost of the alive rule, and it directly motivates the capacity question:
   a scheduler could not have served it either — no arm ever held it via
   1-triple contacts — so the fix is stronger per-contact strength or
   interference isolation, not queueing.
4. **The 1/6-poverty case never exercised.** Min alive-set 4 (probe after
   S11); batch shrank to 14 triples. Reported as monitored.
5. **Style-vs-fact asymmetry under identical service.** r_final fact mean
   0.704 (1.46 / 0.29 / 0.35) vs style mean 0.286 (0.73 / −0.19 / 0.31).
   The residual loss is style-concentrated; across the series, class
   means track interference susceptibility, not service shortfall (this
   arm served both classes identically).
6. **Variant pair, fifth reading.** metric_units × v1 contact S7:
   **−0.142**, the fifth consecutive negative outcome for that item-variant
   across five arms (trigger −0.325, aged v1 −0.291, v2 −0.330, v3 −0.203,
   shield −0.142). A known-bad content pair, now actionable as a
   repair-content rule (pin/skip), independent of any scheduler.

## 6. Tier-2 merge verdict

| arm (context) | aggregate inc → cand | protected agg | protected regressions | drift | decision |
|---|---|---|---|---|---|
| cadence | 0.3156 → 0.0801 | 0.255 → 0.030 | paris, tokyo | 1.000 (fail) | ROLLBACK |
| trigger | 0.3156 → 0.0740 | 0.255 → 0.043 | paris, tokyo, refuse-color-seven, ho-refuse-weigh | 1.000 (fail) | ROLLBACK |
| aged v1 | 0.3156 → 0.0787 | 0.255 → 0.061 | paris, tokyo, refuse-color-seven, ho-refuse-weigh | 1.000 (fail) | ROLLBACK |
| aged v2 | 0.3156 → 0.1073 | 0.255 → 0.056 | paris, tokyo, refuse-color-seven, ho-refuse-weigh | 1.000 (fail) | ROLLBACK |
| aged v3 | 0.3156 → 0.0475 | 0.255 → 0.006 | paris, tokyo, refuse-color-seven, ho-refuse-weigh | 1.000 (fail) | ROLLBACK |
| shield (this) | 0.3156 → **0.1739** | 0.255 → **0.225** | refuse-color-seven 0.408→0.244, ho-refuse-weigh 0.397→0.236 | **1.000 (fail)** | **ROLLBACK** |

Expected ROLLBACK, confirmed an **eighth** time. Notably the shield
candidate is the series' **best merge-grade candidate by a wide margin**
(aggregate 0.174 vs the previous best 0.107; protected 0.225 vs 0.061, and
**fact-paris / fact-tokyo survived a merge for the first time**) — the
blanket shield's broad, shallow reinforcement is dramatically less
destructive to held-out chat probes than any narrow-queue arm's adapter.
The absolute vetoes still fire (drift 1.000 fail; two refuse-family
regressions), so the gate's stance is unchanged; the protected-side
improvement is real evidence that service pattern moves merge-grade
proximity even when it cannot move the veto outcomes.

## 7. Claims and limits

**Claims (what the curves support).**

1. *The scheduling question is closed at this capacity shape.* The
   strongest conceivable service policy — every alive item served every
   session, no queue to mis-order — still misses the trigger bar
   (2/6 / 1 dead / 0.495 vs 4/6 / 0 / 0.581) while spending 2.5× the
   trigger arm's re-exposure tokens. Combined with the chain (queue arms
   v1/v2/v3 all negative; trigger's targeted 2-triple depth remains the
   best horizon policy measured; shield's breadth is worse on every bar
   metric): **no scheduling rule can recover the loss, because the loss
   is not scheduling-allocated. "architecture-straw-limit, schedules
   dead, adaptive capacity next."**
2. *The failure is batch-internal cross-item interference plus
   per-contact strength, not service coverage.* Documented from traces:
   a mass-damage event inside one fully-shielded batch (S7), items buried
   on their own shield sessions, and 1-triple contacts systematically
   weaker than 2-triple targeted serves wherever the two overlap (§5.2).
   The named follow-up at #11 is **class-segregated adapters (fact bank /
   style bank)**: the residual loss is style-concentrated under
   identical service (§5.5), and segregation is the intervention that
   directly attacks the measured mechanism (item-on-item interference);
   a raw capacity knob (rank/steps) does not address it. The
   repair-content rule (pin/skip the 0-for-5 metric_units × v1 pair) and
   the S13+ equilibration horizon-discipline arm carry forward regardless.
3. *The shield is the best merge candidate of the series* (tier-2
   aggregate 0.174, protected 0.225, first survival of the protected
   fact probes) while still failing the absolute vetoes — eighth
   ROLLBACK; service pattern moves merge proximity, not veto outcomes.
4. *fav_show is resurrectible by constant coverage.* The item every
   queue arm left poor (cadence 0.77, trigger 0.96, v1 0.65, v2 0.58, v3 −0.22) ends at the
   series best 1.464 under 5 blanket contacts after nearly dying on its
   first one — constant service has exactly one clear winner, and it is
   the item the queues kept mis-timing.

**Limits.**

- Single run under the pinned seed (n = 1). The determinism chain is
  intact end-to-end (calibrations bit-identical twice, CADENCE-IDENT 0.0),
  so divergence from the committed arms after S6 is shield-attributable;
  but the batch draws (filler cycling, variant sequence) are one draw.
- Shield contact Δr values are bystander-coupled (all alive items served
  in every batch), not clean per-contact repairs like the
  1-serve-per-session arms; the sign read (S7 mass-negative) is robust,
  per-contact magnitudes are not per-item experiments.
- The horizon is again a snapshot (pet_dog's un-shielded S12 bounce
  0.354, fav_show mid-climb). The S13+ equilibration question is open and
  queued; no criterion flips within the observed un-shielded bounce
  magnitudes even granted to both sub-0.50 survivors (would lift held
  count at most to 3/6).
- The token accounting counts re-exposure triples; shield sessions are
  also ~3× the wall per session (20-triple batches); the capacity claim
  is robust to either accounting.

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
and are not committed). Wall: 15200 s (12 threads, CPU fp32, loaded box —
the arm's margins are seed-deterministic, so wall does not touch any
number reported here).
