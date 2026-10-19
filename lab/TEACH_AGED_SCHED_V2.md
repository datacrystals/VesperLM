# TEACH_AGED_SCHED_V2 — age-from-first-trip priority + deficit floor

Queued arm ("the v2 scheduler arm", the item after #6 of
`lab/TEACH_PIPELINE.md`), run 2026-10-19 on the laptop, CPU only, fp32.
The aged-scheduler v1 arm (`lab/TEACH_AGED_SCHED.md`) eliminated the trigger
arm's first-trip starvation **mechanically** (tea_tin hypothetical rank #1 at
its first two probes, would have been served at S7) but regressed the horizon
aggregate: **2/6 held ≥ 0.50, 2 dead slots (min −0.17), mean 0.279** vs the
trigger arm's 4/6 / 0 dead / 0.581. v1 §5 assigned the causes: (1) served
items re-queue **young** (age from last service), so re-trips starved —
metric_units' re-trip waited 5 probes, pet_dog's 4, both ended dead;
(2) age-init at the teach-phase contact let a self-recovered item (fav_show,
deficit 0.000, r 0.81 by S9) win the S10 slot on age alone; (3) end-loaded
serves were destructive to bystanders and the horizon froze mid-oscillation.

This arm is the **smallest honest delta off v1**: same library, same teach
phase (bit-checked in-run), same slots/filler/threshold/cap/tripwires/poison/
tier-2 — **only the queue discipline changes**. Scope-honest: a scheduler
variant arm, not a memory architecture change.

Harness: `lab/teach_aged_sched_v2.py` (same import chain as v1:
`lab/teach_aged_sched.py` → `lab/teach_trigger_repair.py` →
`lab/teach_spaced_cadence.py`; deterministic seeded train loop inherited
unchanged). Numbers-only results: `lab/teach_aged2_out/teach_aged_sched_v2.json`;
log `lab/logs/teach_aged_sched_v2.log`; calibration workdirs
`lab/teach_aged2_calib/{A,B}`.

## 0. Determinism verification (done before the arm)

Two identical v2 calibration teaches (`--only metric_units --seed 7`, fresh
workdirs, sequential, same code path the arm uses), before the arm started:

| run | baseline margin | post-teach margin | gain |
|---|---|---|---|
| calib A | −1.12088942527771 | **−0.14622879028320312** | +0.975 |
| calib B | −1.12088942527771 | **−0.14622879028320312** | +0.975 |

Post-teach margins **bit-identical across the two v2 replays**, and both
**bit-match the cadence / trigger / aged-v1 committed seed-7 calibration**
(published values: baseline −1.12088942527771, post-teach
−0.14622879028320312) — the fourth harness in the chain inherits the
deterministic train loop exactly.

In-run cross-check (the arm's teach phase S1..S6 runs the verbatim cadence
session runner; all S1..S6 margins compared against the committed cadence
JSON):

```
CADENCE-IDENT: teach phase S1..S6 vs committed cadence run —
max |margin Δ| = 0.000e+00 (BIT-IDENTICAL; worst before/fav_show)
```

Every difference this arm shows after S6 is therefore attributable to the v2
serving rule alone.

## 1. Design

| item | value |
|---|---|
| checkpoint | `lab/imported/vk_sft_r3/step_1800/chat_model` — identical to E3 / A/B / cadence / trigger / v1 |
| runtime | laptop CPU, fp32, 12 threads; tier-2 Immune never leaves CPU (I7); LoRA rank 8, alpha 16, lr 2e-3, kl_coef 0.05, 24 steps/session — identical knobs |
| seed | 7 on every consolidate call (cadence-hardening §1a) |
| library | the six items verbatim: fav_show (F), math_words (S), tea_tin (F), metric_units (S), pet_dog (F), greet_rule (S) |
| teach phase | S1–S6, **identical to v1 / trigger / cadence** (bit-checked in-run) |
| scheduler phase | S7–S12 (6 later sessions, same filler slices): 5 generic-chat approval triples + at most ONE re-teach turn (2 triples, 28.6% ≤ 30% cap) |
| trigger rule | unchanged: after every session (S6..S11), item queued iff r_i(k) < 0.50 |
| serving rule | **v2: age from first trip + deficit floor** (formula below), one pop per session (serve cap 1, structural — the per-session cap v1 §5 called for) |
| tripwire safety | 8-variant extended pools verbatim; every response string unique run-wide |
| rescue | **not re-run** — v1 already measured it (NOT resurrectable from strict r ≤ 0.06, n = 1) and its holdout costs 1/6 capacity and confounds the scheduler read (v1 §5.3); tea_tin is a NORMAL queue member here |
| fallback | one gate-safe spaced re-teach max (policy, armed on any tripwire) |
| tier 2 | ONE Immune merge verdict on the accumulated adapter at the end, constants untouched |
| bar to beat (from v1's flop, not invented) | first-trip starvation still zero (mechanical) AND horizon ≥ trigger's 4/6 ≥ 0.50 + 0 dead + mean ≥ 0.581 |

### The v2 scheduler (exact formula as run, documented before running)

For each item *i* with a recorded trip, at the probe after session *k*
(which seeds the batch of session *k+1*):

```
first_trip_i  = earliest probe session k' (≥ 6) with r_i(k') < thresh
                — sticky for the whole scheduler phase; NEVER reset by
                  service or by partial recovery, so a served item KEEPS
                  its seniority and a re-trip re-enters with its full
                  age, not age 0                       (fixes v1 §5.1)
age_i(k)      = k − first_trip_i
                — age starts at the actual first trip, not the
                  teach-phase contact                  (fixes v1 §5.2a)
deficit_i(k)  = max(0, thresh − r_i(k))            # r re-probed live
eligible_i(k) ⟺ deficit_i(k) > 0                   # DEFICIT FLOOR:
                a self-recovered item cannot consume a slot on age
                alone; it stays queued with its age and re-becomes
                eligible if it drops back below threshold
                                                   (fixes v1 §5.2b)
priority_i(k) = w_age · age_i(k) + w_def · deficit_i(k)   w_age = w_def = 1.0
```

- Serve = pop the **eligible** item with max priority, one per session.
  Tie-break: earlier `first_trip`, then library ORDER index (fully
  deterministic under the pinned seed). A session with zero eligible items
  runs filler only (a recovered backlog spends no capacity).
- Re-trip starvation is bounded by construction for the re-trip *kind*:
  a re-tripped item re-enters with `triggered_after` = re-trip probe but
  `age` counting from its first trip — it can never be younger than any
  item that first tripped after it. What age-from-first-trip still does
  NOT bound is inter-arrival pile-up (if more distinct items stay trip-eligible
  than sessions remain); that bound is queue-length-dependent, and the
  horizon read reports any unserved entries explicitly.

### Secondary reads (log-only counterfactuals, zero capacity effect)

Adopted under the "one more knob, not a redesign, essentially free" rule:

- **dead-ban queue position**: at every probe, each queued item in the
  strict dead band (r ≤ 0.06) is logged with where it sits and what a
  "rescues-into-dead-ban" policy would serve instead (max-priority eligible
  with r > 0.06). v1's rescue says serving strict-dead items yields
  ~nothing (Δr +0.063); the trace tests whether the ban would have
  re-ordered any actual v2 service. **No capacity effect.**
- **per-session serve cap**: structural at 1 (unchanged); recorded per
  session, and horizon-side per-session bystander damage is reported for
  the v1 §5.4 end-loaded-cost read.

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
VERDICT SCHED probe after S6:  QUEUE tea_tin r=0.324, metric_units r=0.249, pet_dog r=0.153 | queue (prio) [pet_dog 0.347 E, metric_units 0.251 E, tea_tin 0.176 E]
VERDICT S7  [agedv2/pet_dog]     gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> pet_dog [prio 0.347 age 0 deficit 0.347]) train loss 8.935->0.218
VERDICT EVENT [re-teach/pet_dog] SERVED first trip S6 r=0.153, age at serve 0 | served S7 v1 | r 0.153 -> 0.518 (repair +0.365)
VERDICT SCHED probe after S7:  QUEUE fav_show r=0.258, greet_rule r=0.229 | queue (prio) [tea_tin 1.411 E, metric_units 1.302 E, greet_rule 0.271 E, fav_show 0.242 E]
VERDICT S8  [agedv2/tea_tin]     gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> tea_tin [prio 1.411 age 1 deficit 0.411]) train loss 3.452->0.158
VERDICT EVENT [re-teach/tea_tin] SERVED first trip S6 r=0.324, age at serve 1 | served S8 v1 | r 0.089 -> 0.030 (repair -0.059)
VERDICT SCHED probe after S8:  QUEUE math_words r=0.381, tea_tin r=0.030 | queue (prio) [tea_tin 2.470 E, metric_units 2.346 E, greet_rule 1.145 E, fav_show 1.018 E, math_words 0.119 E] | dead-ban trace: [tea_tin]
VERDICT S9  [agedv2/tea_tin]     gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> tea_tin [prio 2.470 age 2 deficit 0.470]) train loss 2.247->0.214
VERDICT EVENT [re-teach/tea_tin] SERVED re-trip S8 r=0.030, age at serve 2 | served S9 v2 | r 0.030 -> 0.269 (repair +0.239)
VERDICT SCHED probe after S9:  QUEUE tea_tin r=0.269 | queue (prio) [tea_tin 3.231 E, metric_units 3.036 E, greet_rule 2.637 E, fav_show 2.517 E, math_words 1.104 E] | dead-ban trace: [greet_rule, fav_show]
VERDICT S10 [agedv2/tea_tin]     gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> tea_tin [prio 3.231 age 3 deficit 0.231]) train loss 2.405->0.121
VERDICT EVENT [re-teach/tea_tin] SERVED re-trip S9 r=0.269, age at serve 3 | served S10 v3 | r 0.269 -> 0.363 (repair +0.094)
VERDICT SCHED probe after S10: QUEUE tea_tin r=0.363 | queue (prio) [metric_units 4.285 E, tea_tin 4.137 E, greet_rule 3.478 E, fav_show 3.0 FLOOR, math_words 2.0 FLOOR] | dead-ban trace: [greet_rule]
VERDICT S11 [agedv2/metric_units] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> metric_units [prio 4.285 age 4 deficit 0.285]) train loss 3.630->0.089
VERDICT EVENT [re-teach/metric_units] SERVED first trip S6 r=0.249, age at serve 4 | served S11 v1 | r 0.215 -> -0.115 (repair -0.330)
VERDICT SCHED probe after S11: QUEUE metric_units r=-0.115, pet_dog r=-0.075 | queue (prio) [metric_units 5.615 E, pet_dog 5.575 E, tea_tin 5.125 E, greet_rule 4.987 E, fav_show 4.0 FLOOR, math_words 3.0 FLOOR] | dead-ban trace: [metric_units, pet_dog, greet_rule]
VERDICT S12 [agedv2/metric_units] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> metric_units [prio 5.615 age 5 deficit 0.615]) train loss 1.924->0.156
VERDICT EVENT [re-teach/metric_units] SERVED re-trip S11 r=-0.115, age at serve 5 | served S12 v2 | r -0.115 -> 0.782 (repair +0.897)
VERDICT S13 [poison/banana-batch] gate=stub-tier1: ROLLBACK — target collapse: one response is 100% of the batch | incumbent adapter intact: True
VERDICT MERGE [accumulated adapter vs raw base] gate=immune-tier2: ROLLBACK — protected probe fact-paris regressed 1.000 -> 0.000 | protected probe fact-tokyo regressed 1.000 -> 0.000 | protected probe refuse-color-seven regressed 0.408 -> 0.381 | protected probe ho-refuse-weigh regressed 0.397 -> 0.236 | aggregate 0.3156 -> 0.1073, drift 1.000 label=fail
```

Gate behaviour: **12/12 promotes, 0 tripwire events** (top_response_fraction
0.14 everywhere after S1; every re-teach string fresh, variants ≤ v3). The
poison canary tripped the collapse rule at 100% and the incumbent stayed
byte-untouched. The one-fallback tripwire policy was **not** exercised; the
run completed clean (no harness failure — no `lab/FAILURES.md` entry).

## 3. Scheduler event log + queue-wait statistics

Six services in six sessions, all aged-priority queue pops (no rescue in
v2). Arrivals: 6 first-trips (tea_tin S6, metric_units S6, pet_dog S6,
fav_show S7, greet_rule S7, math_words S8) + 5 re-trips (tea_tin S8, S9,
S10; metric_units S11; pet_dog S11).

| # | item | kind | tripped after | served in | wait | variant | r before → after | repair Δr | priority (age/deficit) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | pet_dog | queue (first) | S6 | S7 | 1 | v1 | 0.153 → 0.518 | **+0.365** | 0.347 (0 / 0.347) |
| 2 | tea_tin | queue (first) | S6 | S8 | 2 | v1 | 0.089 → 0.030 | **−0.059** | 1.411 (1 / 0.411) |
| 3 | tea_tin | queue (re-trip) | S8 | S9 | 1 | v2 | 0.030 → 0.269 | **+0.239** | 2.470 (2 / 0.470) |
| 4 | tea_tin | queue (re-trip) | S9 | S10 | 1 | v3 | 0.269 → 0.363 | **+0.094** | 3.231 (3 / 0.231) |
| 5 | metric_units | queue (first) | S6 | S11 | 5 | v1 | 0.215 → −0.115 | **−0.330** | 4.285 (4 / 0.285) |
| 6 | metric_units | queue (re-trip) | S11 | S12 | 1 | v2 | −0.115 → 0.782 | **+0.897** | 5.615 (5 / 0.615) |

Queue-wait histogram (serves, sessions trip→serve): **{1: 4, 2: 1, 5: 1};
max 5** (metric_units' first trip) — vs v1's {1: 2, 2: 2, 4: 1}. Per-event
repair: 4/6 positive, mean +0.201 (trigger +0.445, v1 +0.276).

### Event-contact latency for EVERY trip event (the v1 starvation read)

| item | trip kind | tripped after | first trip | outcome | latency / probes waited |
|---|---|---|---|---|---|
| tea_tin | first | S6 | S6 | served S8 | 2 |
| metric_units | first | S6 | S6 | served S11 | 5 |
| pet_dog | first | S6 | S6 | served S7 | 1 |
| fav_show | first | S7 | S7 | **unserved** (floor-ineligible from S10 on; see §3a) | 4 probes |
| greet_rule | first | S7 | S7 | **UNSERVED — eligible at all 5 probes** | 4 probes |
| math_words | first | S8 | S8 | **unserved** (floor-ineligible from S10 on) | 3 probes |
| tea_tin | re-trip | S8 | S6 | served S9 | 1 |
| tea_tin | re-trip | S9 | S6 | served S10 | 1 |
| tea_tin | re-trip | S10 | S6 | unserved (horizon edge) | 1 probe |
| metric_units | re-trip | S11 | S6 | served S12 | 1 |
| pet_dog | re-trip | S11 | S6 | unserved (horizon edge) | 0 probes |

**Re-trip starvation still at horizon?** The v1 re-trip-starve mechanism is
gone: every re-trip that had ≥1 full session of queue life got contact
latency 1 (tea_tin ×2, metric_units ×1); the two unserved re-trips
(tea_tin S10, pet_dog S11) waited 1 and 0 probes — horizon-edge artifacts,
not starvation (v1's were 5 and 4 probes and ended dead). **But the
starvation moved BACK to eligible first-trips**: greet_rule queued after S7,
eligible at **all five probes** (r 0.229 → 0.35 → −0.14 → 0.02 → −0.49 —
its live deficit was the run's largest at the last probe, 0.987) and was
never served. fav_show and math_words are *not* starvation cases — the
deficit floor correctly kept them out once self-recovered (§3a) and both
ended ≥ 0.43.

### 3a. The deficit floor doing exactly one intended thing

At the probes after S10 and S11, fav_show (r 0.83 / 0.53) and math_words
(r 0.73 / 0.67) sat in the queue with ages 3–4 and priorities 2.0–4.0.
v1's rule served exactly such an item (fav_show at S10, deficit 0.000,
after self-recovering to r 0.81). v2's floor made both ineligible: **no slot
was spent on a recovered item**. The floor worked; the slots it freed were
still lost to the hog mechanism of §5.1.

## 4. Retention r_i(k) — v2 vs v1 vs trigger vs cadence (committed numbers)

Baselines and S1..S6 bit-identical across the four arms (CADENCE-IDENT,
max |Δ| = 0.0). v2 S7 is *additionally* bit-identical to the trigger arm's
S7 (same serve: pet_dog v1, r 0.153 → 0.518) — the serving rules coincided
once because age 0 ties break to deficit and both rules pop pet_dog there.
The table shows margin (r); **bold** marks re-teach sessions in this arm;
`|` = teach anchor.

| item | g_i | S6 (identical) | S7 | S8 | S9 | S10 | S11 | S12 |
|---|---|---|---|---|---|---|---|---|
| fav_show (F) | +2.73 | +5.16 (1.26) | +2.42 (0.26) | +3.03 (0.48) | +1.67 (−0.02) | +3.98 (0.83) | +3.17 (0.53) | +3.30 (0.58) |
| math_words (S) | +5.24 | +3.06 (1.23) | +1.44 (0.92) | −1.38 (0.38) | −1.30 (0.40) | +0.45 (0.73) | +0.12 (0.67) | −1.12 (0.43) |
| tea_tin (F) | +13.31 | +3.05 (0.32) | −0.09 (0.09) | **−0.87 (0.03)** | **+2.32 (0.27)** | **+3.56 (0.36)** | +3.73 (0.38) | +2.99 (0.32) |
| metric_units (S) | +2.30 | −0.55 (0.25) | −0.67 (0.20) | −0.77 (0.15) | −0.05 (0.46) | −0.63 (0.21) | **−1.39 (−0.12)** | **+0.68 (0.78)** |
| pet_dog (F) | +4.90 | +1.02 (0.15) | **+2.80 (0.52)** | +3.51 (0.66) | +5.70 (1.11) | +3.31 (0.62) | −0.11 (−0.08) | −1.30 (−0.32) |
| greet_rule (S) | +6.04 | +3.47 (1.00) | −1.19 (0.23) | −0.43 (0.35) | −3.40 (−0.14) | −2.44 (0.02) | −5.51 (−0.49) | −3.24 (−0.11) |

### Headline: the serving-rule ablation grid (same teach phase, seed, slots, threshold)

| item | kind | serves (v2) | cadence r_final | trigger r_final | aged v1 r_final | **v2 r_final** |
|---|---|---|---|---|---|---|
| fav_show | fact | 0 (floor) | 0.77 | 0.96 | 0.65 | 0.58 |
| math_words | style | 0 (floor) | 0.79 | 0.66 | 0.36 | 0.43 |
| tea_tin | fact | 3 (S8, S9, S10) | 0.01 | 0.17 | 0.14 | **0.32** (best over arms) |
| metric_units | style | 2 (S11, S12) | 0.06 | 0.69 | −0.12 | **0.78** (best over arms) |
| pet_dog | fact | 1 (S7) | 0.52 | 0.50 | −0.17 | **−0.32** (worst over arms) |
| greet_rule | style | **0 (starved, eligible 5/5 probes)** | 0.52 | 0.50 | 0.81 | −0.11 |

| metric | cadence | trigger | aged v1 | **v2** | read |
|---|---|---|---|---|---|
| items r_final ≥ 0.50 | 4/6 | 4/6 | 2/6 | **2/6** | trigger bar NOT met |
| dead slots r ≤ 0.06 | 2 | 0 | 2 (neg) | **2 (both negative)** | trigger bar NOT met |
| mean r_final | 0.445 | 0.581 | 0.279 | **0.280** | trigger bar NOT met |
| min r_final | 0.01 | 0.17 | −0.17 | **−0.32** | regressed |
| first-trip starvation | — | tea_tin ∞ (6 sessions) | none | **greet_rule (5 probes, eligible at all)** | REINTRODUCED (§5.1) |
| re-trip starvation at horizon | — | pet_dog, greet_rule (4, 3 probes) | metric_units, pet_dog (5, 4 probes; end dead) | **none** (2 horizon-edge unserved at ≤1 probe) | v2 wins this axis |

The honest headline: **v2 reproduces v1's aggregate (mean 0.280 vs 0.279,
2/6 held, 2 dead) while shifting *which* items die, and it re-introduces
first-trip starvation with a new mechanism.** The re-trip starvation v1
§5.1 named is genuinely fixed — but **no success criterion is met**:

1. **first-trip starvation still zero (mechanically) — NOT HELD.** Under
   age-from-first-trip, greet_rule queued after S7, stayed eligible at all
   five later probes (its deficit peaked as the run's largest: 0.987 at
   r = −0.49) and was never served, because an older, chronically
   sub-threshold item (tea_tin) outranked it on seniority every time
   (§5.1). v1's version of this claim survives only because v1's rescue
   holdout removed tea_tin from queue competition.
2. **≥ 4/6 held ≥ 0.50 — NOT met (2/6).**
3. **0 dead (r ≤ 0.06) — NOT met (2: pet_dog −0.32, greet_rule −0.11).**
4. **mean ≥ 0.581 — NOT met (0.280).**

No partial credit. The scheduler still does not exist; the verdict is
explicitly negative (§8).

## 5. Why the aggregate still fails (mechanics, from the recorded traces)

1. **Chronic-seniority hog: age-from-first-trip converts "re-queue young"
   fairness into a monopoly for the longest-broken item.** tea_tin's three
   serves each landed below threshold (−0.059 → 0.030, +0.239 → 0.269,
   +0.094 → 0.363 — net +0.274 r across three slots and never a held
   state), so it re-queued every probe with the run's maximum age and a
   healthy deficit, winning 3 of 6 slots. greet_rule — younger by one
   probe, damaged as deeply or deeper at every contested probe — never
   got one slot. Fairness by age-from-first-trip needs a **per-item serve
   budget / cool-down** or a **no-progress eviction** ("N serves without
   crossing threshold → drop/defer"), neither of which is in v2's formula.
   This is the delta the next arm must carry; it is one knob again
   (e.g. a served-item cool-down of ≥1 session while other eligible items
   exist).
2. **Capacity is the knife edge it always was.** Six slots, 6 first-trips +
   5 re-trips. tea_tin's extra two serves are exactly greet_rule's and
   pet_dog's missing serves (1:1). v1 paid this cost exogenously (rescue
   holdout, 1/6); v2 pays it endogenously via the hog. Without a
   service-rate change (more slots or a threshold that trips fewer items
   — the trigger arm's bracketing warning stands), zero-sum serving means
   "fixed" starvation relocates rather than disappears.
3. **End-loaded destructive serves again.** S11 and S12 damaged bystanders
   massively: pet_dog 0.62 → −0.08 over S11, → −0.32 over S12; greet_rule
   −0.49 → −0.11 over S12; math_words 0.67 → 0.43 over S12; even tea_tin
   lost 0.06 at S12. The horizon snapshot is again frozen mid-oscillation
   (metric_units reads 0.78 one session after reading −0.12). The
   serve-cap-1 knob is already structural and does not address this; the
   v1 §5.4 caveat stands un-resolved by scheduler choice and belongs in
   the horizon-interpretation discipline, not the queue rule.
4. **Variant luck, again asymmetric.** metric_units' v1 serve was −0.330
   (v1: −0.291; trigger: −0.325 for the same variant/item pair) and its
   v2 serve was +0.897 (trigger's v2: +1.252). Per-contact yields remain
   variant-dependent, n = 1 per draw. The +0.897 lands at S12 and
   directly inflates the run's mean by 0.15 — the single largest element
   of the run's aggregate is a horizon-edge variant draw.
5. **tea_tin's best-ever read is a redundancy read.** Its 0.32 comes from
   3 serves (v1 gave it 1 holdout-rescue and cadence-like 0.14); the
   library-level price was two dead slots. An extra serve on a hogged
   item is not free capacity found — it is capacity moved off
   never-served items.

## 6. Secondary reads

### Dead-ban queue position (log-only counterfactual)

| probe after | queued in dead band (r ≤ 0.06) | v2 served (next session) | dead-ban would serve | same? |
|---|---|---|---|---|
| S6 | — | pet_dog | pet_dog | yes |
| S7 | — | tea_tin | tea_tin | yes |
| S8 | tea_tin (r 0.030) | tea_tin | metric_units | **no** |
| S9 | greet_rule (−0.14), fav_show (−0.02) | tea_tin | tea_tin | yes (hog not banned) |
| S10 | greet_rule (0.02) | metric_units | metric_units | yes |
| S11 | metric_units (−0.115), pet_dog (−0.075), greet_rule (−0.49) | metric_units | tea_tin | **no** |

Two reorderings in five probes, and they cut both ways: banning tea_tin at
S8 would have spent that slot on metric_units (arguably better in advance);
but at S11 the ban would have **blocked the run's best serve** (metric_units
r = −0.115 → +0.897 at S12) and fed tea_tin a fourth slot instead. Verdict:
a flat dead-band ban is **not** an improvement over v2 as run — the
r ≤ 0.06 line marks where v1 showed single contacts yield little, not where
contacts are worthless; near/strict-dead items differ by variant luck more
than by the line. Not adopted; the counterfactual earns its log-only cost.

### Per-session serve cap

Structural at 1 re-teach per session (≤ 28.6% of every batch), unchanged
from the trigger/v1 arms — the "capped per-session serves" safeguard from
the task was already the harness's shape in v1; what v2 adds is the
deficit-floor *eligibility* gate. Per-session serves actually used: {1, 1,
1, 1, 1, 1} — the cap never became the binding constraint; eligibility did
(probes after S10/S11 left two floor-ineligible queue members).

### Threshold brackets (snapshot sensitivity, post-hoc)

At the trigger arm's post-hoc bracket thresholds, the probe-after-S11 trip
sets are: @0.25 — {metric_units, pet_dog, greet_rule} (only the three
deepest-damaged); @0.75 — all six. Lowering the threshold shrinks demand
toward capacity but does not touch the wait-list ordering among survivors —
and under @0.25 tea_tin is *unqueued* by S11 (r 0.363), so the hog
dissolves only as a side effect of trip-setting, while greet_rule stays
queued and would still trail metric_units/pet_dog on age. Raising it
worsens the mismatch (adds the two floor-recovered items as demand). The
trigger arm's bracketing lesson stands for v2's shape exactly: the
threshold is a service-rate throttle, not a starvation fix; the missing
piece is service budgeting.

## 7. Tier-2 merge verdict

| arm (context) | aggregate inc → cand | protected agg | protected regressions | drift | decision |
|---|---|---|---|---|---|
| cadence | 0.3156 → 0.0801 | 0.255 → 0.030 | fact-paris, fact-tokyo | 1.000 (fail) | ROLLBACK |
| trigger | 0.3156 → 0.0740 | 0.255 → 0.043 | paris, tokyo, refuse-color-seven, ho-refuse-weigh | 1.000 (fail) | ROLLBACK |
| aged v1 | 0.3156 → 0.0787 | 0.255 → 0.061 | paris, tokyo, refuse-color-seven, ho-refuse-weigh | 1.000 (fail) | ROLLBACK |
| aged v2 (this) | 0.3156 → **0.1073** | 0.255 → 0.056 | paris 1→0, tokyo 1→0, refuse-color-seven 0.408→0.381, ho-refuse-weigh 0.397→0.236 | **1.000 (fail)** | **ROLLBACK** |

Expected ROLLBACK, confirmed a **sixth** time. Scheduler choice still does
not change merge-grade behaviour. (The v2 candidate marginally *out-scores*
the earlier arms' candidates at tier 2 — 0.1073 vs 0.0787 — an irrelevant
reading: the absolute vetoes fire regardless and nothing about the gate's
stance changes.)

## 8. Claims and limits

**Claims (what the curves support).**

1. *Age-from-first-trip + a deficit floor fixes the two v1 §5 mechanisms it
   aimed at, measurably.* Re-trip starvation as v1 produced it (5/4-probe
   waits, both ending dead) is gone: every re-trip with queue life got
   contact latency 1, and no served-then-redamaged item ever re-entered as
   the youngest (first_trip is sticky by construction). The deficit floor
   blocked the fav_show-at-S10 v1 pattern: no self-recovered item consumed
   a slot (§3a). These are the two targeted mechanics, and they held.
2. *Neither of the four success criteria is met — the verdict is
   explicitly negative.* First-trip starvation: **re-introduced**, via the
   chronic-seniority hog (greet_rule eligible 5/5 probes, never served).
   Horizon: 2/6 held (bar ≥ 4/6), 2 dead (bar 0), mean 0.280 (bar ≥ 0.581).
   v2 ≈ v1 in aggregate (0.280 vs 0.279, 2/6, 2 dead, min worse: −0.32 vs
   −0.17) and both sit far under the trigger arm's 4/6 / 0 / 0.581.
   **The scheduler is still not done, and age-based priority in any form
   so far is not the fix.** The next arm's knob is one of: a per-item serve
   budget/cool-down (breaks the hog), no-progress eviction (breaks the
   hog), or deficit-weight-dominant priority (w_def ≫ w_age; makes age a
   tie-breaker rather than a principal — the bracket note in §6 says age
   as principal is what re-creates starvation under any threshold).
3. *The trigger arm's lowest-r-first rule remains the strongest measured
   scheduler* (4/6 held, 0 dead, mean 0.581, and its one defect — tea_tin's
   queue starvation — is a service-capacity symptom, not a fairness defect:
   the bracketing analysis showed @0.25 it would have served tea_tin by
   construction). Two honest readings coexist: keep lowest-r and fix its
   backlog by capacity/threshold, or keep age rules and add the
   service-budget knob that all three age variants have now shown they
   need. Both are queued; no partial-credit story makes v2 adoptable.
4. *Tier 2 unchanged*: ROLLBACK (0.3156 → 0.1073, drift 1.000 fail); the
   two-tier read holds a sixth time.

**Limits.**

- Single run under the pinned seed (n = 1); variance over seed/schedule is
  one draw. The determinism chain is fully intact (two bit-identical
  calibrations matching v1/trigger/cadence; CADENCE-IDENT 0.0), so all
  divergence from v1/trigger after S7 is serving-rule-attributable.
- The horizon read is mid-oscillation again (§5.3): metric_units'
  r_final 0.78 is a one-session-old variant-2 draw, and the two dead
  slots were damaged in the final two sessions with no equilibration tail.
  An S13+ equilibration could rescind *part* of the dead read — observed
  single-session un-served upswings reach +0.45 r (pet_dog over S9) and
  +0.38 r (greet_rule over S12, its own final un-served move). But no
  criterion verdict can flip even granting each dead slot its best
  observed upswing (pet_dog −0.32+0.45 = 0.13, greet_rule −0.11+0.38 =
  0.27: held-count stays 2/6, dead-count at best falls to 1, mean at best
  0.280 + 0.83/6 = 0.42 — still far under the 4/6 / 0 / 0.581 bar). And
  greet_rule's upswing is bounded upward by its starvation: it moves as a
  bystander, never as a served item.
- greet_rule's starvation is confounded with tea_tin's hog in a way only
  re-ordering can disentangle; the report claims the mechanism from the
  recorded priority traces (tea_tin outranked it at every contested probe),
  not from a counterfactual re-run.
- Free-run generations remain transcript colour; margin probes are the
  mechanism read. Repair means (+0.201 v2, +0.276 v1, +0.445 trigger) are
  per-contact reads over 6 serves each and move with variant luck (§5.4).

## Reproduction

```bash
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_aged_sched_v2.py --workdir lab/teach_aged2_out \
  | tee lab/logs/teach_aged_sched_v2.log

# determinism calibration (2 runs; both must bit-match each other and the
# v1/trigger/cadence committed seed-7 calibration):
python3 -u lab/teach_aged_sched_v2.py --only metric_units --seed 7 --workdir <fresh A>
python3 -u lab/teach_aged_sched_v2.py --only metric_units --seed 7 --workdir <fresh B>
```

Fresh workdir required (append-only session logs, invariant 10; adapter
deltas and `lab/teach_aged2_out/`, `lab/teach_aged2_calib/` are weights and
are not committed). Wall: 3269 s (12 threads, CPU fp32).
