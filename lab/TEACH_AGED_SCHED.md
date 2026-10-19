# TEACH_AGED_SCHED — age-weighted re-teach scheduler + dead-slot rescue probe

Queued experiment #6 of `lab/TEACH_PIPELINE.md` — the serving-rule fix the
trigger arm called for, run 2026-10-19 on the laptop, CPU only, fp32. The
reactive trigger arm (`lab/TEACH_TRIGGER_REPAIR.md`) established that
per-contact repair works (5/6 events positive, mean +0.445) but its serving
rule — lowest r first, then earlier trip — **starved the first-tripped
item**: tea_tin tripped after S6 and sat un-served for all 6 later sessions
while deeper-damaged newcomers kept jumping the line (9 trips vs 6 slots).
This arm changes **only the queue discipline**: same library, same teach
phase, same later-session slots and filler, same threshold, same 30% replay
cap, same tripwire rules. Serving now uses **age-weighted priority**.

Harness: `lab/teach_aged_sched.py` (extends `lab/teach_trigger_repair.py`,
which extends `lab/teach_spaced_cadence.py` — the seeded deterministic
train loop is inherited unchanged). Numbers-only results:
`lab/teach_aged_out/teach_aged_sched.json`; log
`lab/logs/teach_aged_sched.log`; calibration workdirs
`lab/teach_aged_calib/{A,B}`.

## 0. Determinism verification (done before the arm)

Two identical calibration teaches (`--only metric_units --seed 7`, fresh
workdirs, same code path the arm uses), run before the arm started:

| run | baseline margin | post-teach margin | gain |
|---|---|---|---|
| calib A | −1.12088942527771 | **−0.14622879028320312** | +0.975 |
| calib B | −1.12088942527771 | **−0.14622879028320312** | +0.975 |

Post-teach margins **bit-identical** across the two replays, and both
**bit-match the cadence and trigger harnesses' committed seed-7
calibration** for the same item — the third harness in the chain inherits
the deterministic train loop exactly.

In-run cross-check: the arm's teach phase S1..S6 runs the **verbatim cadence
session runner**; all S1..S6 margins were compared against the committed
cadence JSON:

```
CADENCE-IDENT: teach phase S1..S6 vs committed cadence run —
max |margin Δ| = 0.000e+00 (BIT-IDENTICAL; worst before/fav_show)
```

Every difference this arm shows after S6 is therefore attributable to the
serving rule (and the rescue holdout) alone.

## 1. Design

| item | value |
|---|---|
| checkpoint | `lab/imported/vk_sft_r3/step_1800/chat_model` — identical to E3 / A/B / cadence / trigger |
| runtime | laptop CPU, fp32, 12 threads; tier-2 Immune never leaves CPU (I7); LoRA rank 8, alpha 16, lr 2e-3, kl_coef 0.05, 24 steps/session — identical knobs |
| seed | 7 on every consolidate call (cadence-hardening §1a) |
| library | the cadence/trigger 6 items verbatim: fav_show (F), math_words (S), tea_tin (F), metric_units (S), pet_dog (F), greet_rule (S) |
| teach phase | S1–S6, **identical to the trigger and cadence arms** (verified bit-identical in-run) |
| scheduler phase | S7–S12 (6 later sessions, same count, same filler slices): 5 generic-chat approval triples + at most ONE re-teach turn (2 triples, 28.6% ≤ 30% cap) |
| trigger rule | unchanged: after every session (S6..S11), item queued iff r_i(k) < 0.50 |
| serving rule | **age-weighted priority** (formula below), one pop per session |
| tripwire safety | the trigger arm's 8-variant extended pools verbatim; every response string unique run-wide |
| rescue probe | tea_tin held out of service until r ≤ 0.06, then one re-teach (design below) |
| fallback | one gate-safe spaced re-teach max (policy, armed on any tripwire); rescue deadline serve at S12 is the one bounded fallback of the probe |
| tier 2 | ONE Immune merge verdict on the accumulated adapter at the end, constants untouched |
| base case to beat | tea_tin's 6-session starvation under the trigger arm's lowest-r-first queue |

### The scheduler (exact formula as implemented)

For each queued item *i* at the probe after session *k* (which seeds the
batch of session *k+1*):

```
age_i(k)      = k − last_service_i
deficit_i(k)  = max(0, thresh − r_i(k))        # r re-probed live each session
priority_i(k) = w_age · age_i(k) + w_def · deficit_i(k)      w_age = w_def = 1.0
```

- `last_service_i` = session of *i*'s most recent consolidation contact
  carrying one of its turns; initialised to its teach-phase contact:
  item taught at S_j is replayed once at S_{j+1} inside the teach phase, so
  init = min(j+1, 6): fav_show 2, math_words 3, tea_tin 4, metric_units 5,
  pet_dog 6, greet_rule 6 (no teach-phase replay after S6).
- Serve = pop max priority; `last_service_i` ← serving session. A served
  item that re-trips re-queues **young** (age restarts from its last serve) —
  a freshly repaired item must let the backlog age ahead of it, which is
  exactly the fairness the lowest-r rule failed to give.
- Tie-break: earlier `triggered_after`, then library ORDER index (fully
  deterministic under the pinned seed).
- Why the wait is bounded: a fresh arrival's deficit is bounded by
  (thresh − min observed r) < 0.9 here, while age grows by 1 per waiting
  session — any queued item strictly outranks fresh arrivals after one
  session in queue, so under the observed arrival rates (≤2 first-trips per
  probe; ≤3 counting same-probe re-trips) every trip is *expected* to be
  served within ≤2 sessions. Result: 4/5 first-trips met that bound;
  greet_rule waited 4 (§3, §5) — the one miss, caused by an age-init
  outlier and one rescue pre-emption, not by arrival overload.

### Dead-slot rescue probe (one item, scope-limited)

- **Item**: `tea_tin`. Rationale: (a) it is the item the lowest-r-first
  queue starved — its un-served drift curve is already measured (trigger
  arm: r lived at 0.06–0.19 across five probes without a contact), so its
  *expected* contact latency under aged priority is the exact part-1
  counterfactual; (b) its huge teach gain (g = 13.3 nats) makes its margin
  collapse fastest in r units; (c) it repeatedly grazed the dead line
  un-served, so it is the library's most likely genuine r ≤ 0.06 state.
- **Protocol**: tea_tin trips and is tracked like any item (its aged
  priority and hypothetical queue rank are logged at every probe — the
  "what the scheduler would have served" trace) but is **held out of aged
  service**. When a probe finds r_tea_tin(k) ≤ 0.06 (the pipeline's strict
  dead-slot line) it is rescue-ready and is re-taught once in session k+1,
  pre-empting that session's single re-teach slot. Δr = r_after − r_pre is
  the rescue measurement. Claims honestly: "resurrectable from strict
  r ≤ 0.06" / "partial (only the near-dead band)" / "not".
- **Bounded fallback (one, no scope-widening)**: if no probe through S11
  finds r ≤ 0.06, tea_tin is served at S12 anyway and the claim is reported
  against whatever band it actually reached. The rescue item is served
  once; post-rescue drift is observed, not re-served.

## 2. Verdict lines (as run; generation tails trimmed)

```
VERDICT S1  [fact/fav_show]      gate=stub-tier1: PROMOTE | top_resp_frac 0.20 n_triples 5 train loss 0.534->-7.892  | margin +1.718 -> +4.451
VERDICT S2  [style/math_words]   gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> fav_show) train loss 6.527->2.127  | margin -3.374 -> +1.863 | word-digit -7.337 -> +5.183
VERDICT S3  [fact/tea_tin]       gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> math_words) train loss 7.192->5.371  | margin -1.266 -> +12.042
VERDICT S4  [style/metric_units] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> tea_tin) train loss 12.232->0.463  | margin -1.121 -> +1.178
VERDICT S5  [fact/pet_dog]       gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> metric_units) train loss 6.848->2.297  | margin +0.264 -> +5.162
VERDICT S6  [style/greet_rule]   gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> pet_dog) train loss 6.501->0.779  | margin -2.570 -> +3.468
CADENCE-IDENT: teach phase S1..S6 vs committed cadence run — max |margin Δ| = 0.000e+00 (BIT-IDENTICAL; worst before/fav_show)
GAINS g_i: fav_show +2.733  math_words +5.237  tea_tin +13.308  metric_units +2.298  pet_dog +4.898  greet_rule +6.038
VERDICT SCHED probe after S6:  QUEUE metric_units r=0.249, pet_dog r=0.153 | queue (prio) [metric_units 1.251, pet_dog 0.347] | rescue tea_tin r=0.324 held
VERDICT S7  [aged/metric_units]  gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> metric_units [queue prio 1.251 age 1 deficit 0.251]) train loss 7.729->0.166
VERDICT EVENT [re-teach/metric_units] SERVED tripped after S6 r=0.249, waited 1 session | served S7 v1 | r 0.249 -> -0.042 (repair -0.291)
VERDICT SCHED probe after S7:  QUEUE metric_units r=-0.042 | queue (prio) [pet_dog 1.083, metric_units 0.542] | rescue tea_tin r=0.155 held
VERDICT S8  [aged/pet_dog]       gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> pet_dog [queue prio 1.083 age 1 deficit 0.083]) train loss 2.715->0.203
VERDICT EVENT [re-teach/pet_dog] SERVED tripped after S6 r=0.153, waited 2 sessions | served S8 v1 | r 0.417 -> 0.433 (repair +0.015)
VERDICT SCHED probe after S8:  QUEUE fav_show r=0.281, pet_dog r=0.433, greet_rule r=0.132 | queue (prio) [fav_show 6.219, greet_rule 2.368, metric_units 1.610, pet_dog 0.067] | rescue tea_tin r=0.018 held READY (dead crossed)
VERDICT S9  [aged/tea_tin]       gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> tea_tin [rescue]) train loss 3.507->0.191
VERDICT EVENT [re-teach/tea_tin] SERVED [rescue] dead-crossed after S8 r=0.018 | served S9 v1 | r 0.018 -> 0.081 (repair +0.063)
VERDICT SCHED probe after S9:  HOLD — no new trips | queue (prio) [fav_show 7.000, greet_rule 3.341, metric_units 2.637, pet_dog 1.152] | rescue tea_tin r=0.081 held
VERDICT S10 [aged/fav_show]      gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> fav_show [queue prio 7.000 age 7 deficit 0.000]) train loss 3.734->0.150
VERDICT EVENT [re-teach/fav_show] SERVED tripped after S8 r=0.281, waited 2 sessions | served S10 v1 | r 0.806 -> 1.425 (repair +0.620)
VERDICT SCHED probe after S10: QUEUE math_words r=0.460 | queue (prio) [math_words 7.040, greet_rule 4.357, metric_units 3.643, pet_dog 2.000] | rescue tea_tin r=0.056 held
VERDICT S11 [aged/math_words]    gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> math_words [queue prio 7.040 age 7 deficit 0.040]) train loss 2.842->0.120
VERDICT EVENT [re-teach/math_words] SERVED tripped after S10 r=0.460, waited 1 session | served S11 v1 | r 0.460 -> 0.783 (repair +0.323)
VERDICT SCHED probe after S11: HOLD — no new trips | queue (prio) [greet_rule 5.399, metric_units 4.000, pet_dog 3.782] | rescue tea_tin r=0.175 held
VERDICT S12 [aged/greet_rule]    gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> greet_rule [queue prio 5.399 age 5 deficit 0.399]) train loss 3.192->0.156
VERDICT EVENT [re-teach/greet_rule] SERVED tripped after S8 r=0.132, waited 4 sessions | served S12 v0 | r 0.101 -> 0.815 (repair +0.714)
VERDICT S13 [poison/banana-batch] gate=stub-tier1: ROLLBACK — target collapse: one response is 100% of the batch | incumbent adapter intact: True
VERDICT MERGE [accumulated adapter vs raw base] gate=immune-tier2: ROLLBACK — protected probe fact-paris regressed 1.000 -> 0.000 | protected probe fact-tokyo regressed 1.000 -> 0.000 | protected probe refuse-color-seven regressed 0.408 -> 0.348 | protected probe ho-refuse-weigh regressed 0.397 -> 0.319 | aggregate 0.3156 -> 0.0787, drift 1.000 label=fail
```

Gate behaviour: **12/12 promotes, 0 tripwire events** (top_response_fraction
0.14 everywhere after S1; every re-teach string fresh, variants 0–1 used
only). The poison canary tripped the collapse rule at 100% and the
incumbent stayed byte-untouched. The one-fallback tripwire policy was **not**
exercised (the harness-crash in attempt 1 is an operational failure, logged
in `lab/FAILURES.md`, not an experimental fallback).

## 3. Scheduler event log + queue-wait statistics

Six services in six sessions: five aged-priority queue serves + one rescue
pre-emption (S9). Arrivals: 5 first-trips (metric_units S6, pet_dog S6,
fav_show S8, greet_rule S8, math_words S10) + 2 re-trips (metric_units S7,
pet_dog S8) + the rescue holdout's tracking.

| # | item | kind | tripped after | served in | wait | variant | r before → after | repair Δr | priority (age/deficit) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | metric_units | queue | S6 | S7 | 1 | v1 | 0.249 → −0.042 | **−0.291** | 1.251 (1 / 0.251) |
| 2 | pet_dog | queue | S6 | S8 | 2 | v1 | 0.417 → 0.433 | **+0.015** | 1.083 (1 / 0.083) |
| 3 | tea_tin | rescue | (S6) | S9 | 3 | v1 | 0.018 → 0.081 | **+0.063** | dead r ≤ 0.06 crossed after S8 |
| 4 | fav_show | queue | S8 | S10 | 2 | v1 | 0.806 → 1.425 | **+0.620** | 7.000 (7 / 0.000) |
| 5 | math_words | queue | S10 | S11 | 1 | v1 | 0.460 → 0.783 | **+0.323** | 7.040 (7 / 0.040) |
| 6 | greet_rule | queue | S8 | S12 | 4 | v0 | 0.101 → 0.815 | **+0.714** | 5.399 (5 / 0.399) |

Queue-wait histogram (normal serves, sessions trip→serve):
**{1: 2, 2: 2, 4: 1}; max 4** (greet_rule). Per-event repair: 4/5 queue
serves positive (mean +0.276; +0.277 including the rescue). Still queued at
the horizon (both **re-trips**, never re-served): metric_units (re-tripped
after S7, waited 5 probes; ended r = −0.12), pet_dog (re-tripped after S8,
waited 4 probes; ended r = −0.17).

The counterfactual that decides the base case: the rescue trace records
tea_tin's hypothetical aged rank had it NOT been held out — **rank 1 of 3
at the probes after S6 and S7** (priority 2.176 then 3.345 vs every rival).
Under the pure aged rule tea_tin would have been served at **S7**: the
trigger arm's exact starvation case (6 sessions, never served) is killed by
the formula on the same probe cadence and the same slots. It was held to
S9 by the rescue design only.

## 4. Retention r_i(k) — aged arm vs trigger vs cadence (committed numbers)

Baselines and S1..S6 are bit-identical across the three arms (CADENCE-IDENT,
max |Δ| = 0.0). g_i identical. The table shows margin (r); **bold** marks
re-teach sessions in this arm; `|` = teach anchor.

| item | g_i | S1–S6 (bit-identical) | S7 | S8 | S9 | S10 | S11 | S12 |
|---|---|---|---|---|---|---|---|---|
| fav_show (F) | +2.73 | +4.45 \| … +5.16 (1.26) | +3.26 (0.57) | +2.49 (0.28) | +3.92 (0.81) | **+5.61 (1.43)** | +4.33 (0.96) | +3.49 (0.65) |
| math_words (S) | +5.24 | … +3.06 (1.23) | +0.91 (0.82) | −0.56 (0.54) | −0.53 (0.54) | −0.96 (0.46) | **+0.73 (0.78)** | −1.46 (0.36) |
| tea_tin (F) | +13.31 | … +3.05 (0.32) | +0.80 (0.16) | −1.03 (0.02) | **−0.19 (0.08)** | −0.51 (0.06) | +1.07 (0.18) | +0.59 (0.14) |
| metric_units (S) | +2.30 | … −0.55 (0.25) | **−1.22 (−0.04)** | −1.37 (−0.11) | −1.44 (−0.14) | −1.45 (−0.14) | +0.49 (0.70) | −1.41 (−0.12) |
| pet_dog (F) | +4.90 | … +1.02 (0.15) | +2.31 (0.42) | **+2.38 (0.43)** | +1.97 (0.35) | +2.75 (0.51) | −1.12 (−0.28) | −0.54 (−0.17) |
| greet_rule (S) | +6.04 | … +3.47 (1.00) | +0.68 (0.54) | −1.77 (0.13) | −1.61 (0.16) | −1.71 (0.14) | −1.96 (0.10) | **+2.35 (0.81)** |

### Headline: horizon comparison

| item | kind | serves (this arm) | cadence r_final | trigger r_final | aged r_final |
|---|---|---|---|---|---|
| fav_show | fact | 1 (S10) | 0.77 | 0.96 | 0.65 |
| math_words | style | 1 (S11) | 0.79 | 0.66 | 0.36 |
| tea_tin | fact | 1 rescue (S9) | 0.01 | 0.17 | 0.14 |
| metric_units | style | 1 (S7, failed repair) | 0.06 | 0.69 | −0.12 |
| pet_dog | fact | 1 (S8, weak repair) | 0.52 | 0.50 | −0.17 |
| greet_rule | style | 1 (S12) | 0.52 | 0.50 | 0.81 |

| metric | cadence | trigger | aged | read |
|---|---|---|---|---|
| items r_final ≥ 0.50 | 4/6 | 4/6 | **2/6** | regressed |
| dead slots r ≤ 0.06 | 2 | 0 | **2 (both negative)** | regressed vs trigger |
| mean r_final | 0.445 | 0.581 | **0.279** | regressed |
| min r_final | 0.01 | 0.17 | **−0.17** | regressed |
| first-trip starvation | tea_tin fixed by cyclic slot but dead | tea_tin ∞ (6 sessions) | **none — would serve S7; 4/5 first-trips ≤2 sessions** | **aged wins (the target)** |
| unserved at horizon | — | tea_tin trip (first-trip) | metric_units & pet_dog **re-trips** (5, 4 probes) | starvation moved to re-trips |

The honest headline: **the aged queue kills the targeted failure mode and
loses the aggregate.** tea_tin's first-trip starvation is gone by
construction (hypothetical rank #1 at its first two probes), and 4/5
first-trips were served within ≤2 sessions (greet_rule's 4-session wait is
the one miss). But mean r_final 0.279 vs trigger's 0.581, and the two
horizon dead slots are the items whose **re-trips** starved — metric_units
and pet_dog were each served once early, re-tripped, and never reached the
front again because re-queued items start with age 0 while un-served items
kept aging. The §5 analysis assigns the causes.

## 5. Why the aggregate regressed (mechanics, from the recorded traces)

1. **Starvation migrated from first-trips to re-trips.** Implemented as
   documented, `last_service` updates on serve, so a served-and-re-damaged
   item re-queues with age 0 — the re-trip is the youngest entry in the
   queue. metric_units (re-trip S7) and pet_dog (re-trip S8) waited 5 and 4
   probes un-served; both ended negative. Same starvation logic the arm was
   built to kill, one level down: fairness by age must count from the
   FIRST trip (or track cumulative deficit), not from the last serve.
2. **Age-init to teach-phase contact outranked real damage once.**
   fav_show, whose init contact was S2, entered every later queue with a
   huge age lead and was served at S10 with **deficit 0.000** (it had
   self-recovered to r 0.81 by S9) — consuming the slot greet_rule
   (r 0.14–0.16, waiting since S8) needed two sessions earlier. Age
   accrued while above threshold is not damage; a deficit-floor gate on
   age (or age-init at trip, not at teach contact) would have ordered it
   behind.
3. **The rescue pre-emption cost 1/6 of capacity with no horizon support.**
   tea_tin's S9 rescue serve contributed its own +0.063 and ended r 0.14;
   the price was greet_rule's wait stretching to 4 (S9 would otherwise
   have been fav_show, and the whole tail shifts one session earlier).
4. **Late serves were expensive to bystanders.** S11 (math_words v1) and
   S12 (greet_rule v0) were two of the run's most damaging sessions for
   un-served items: pet_dog −0.79 r over S11 alone (0.51 → −0.28),
   metric_units −0.83 r over S12 (0.70 → −0.12), math_words 0.78 → 0.36
   over S12. The r_final snapshot reads the just-repaired items high and
   the just-damaged un-served ones dead; the phase ends mid-oscillation
   rather than at equilibrium — same "within-curve contacts are the
   readable signal" caveat as the trigger arm, amplified by concentrated
   end-loaded repair.
5. **Variant luck is asymmetric across arms.** The trigger arm's +1.252
   metric_units rescue (variant 2) inflated its mean; this arm's only
   metric_units serve drew variant 1 again — the known-harmful variant for
   that item (−0.291 here, −0.325 in the trigger arm). n = 1 under one
   seed; per-contact yields remain variant-dependent, not item-constant.

## 6. Dead-slot rescue probe

Design: tea_tin (justification in §1) held out of aged service until a
probe found r ≤ 0.06; age-position of the holdout logged each probe.

| probe after | r_tea_tin | hypothetical aged rank (if not held) | rescue state |
|---|---|---|---|
| S6 | 0.324 | 1 of 3 (prio 2.176) | tracked |
| S7 | 0.155 | 1 of 3 (prio 3.345) | tracked |
| S8 | **0.018** | 2 of 5 (prio 4.482) | **DEAD CROSSED — rescue ready** |
| (S9 serve) | → 0.081 | — | **served v1, Δr +0.063** |
| S10 | 0.056 | — | post-rescue drift |
| S11 | 0.175 | — | post-rescue drift |
| S12 | 0.139 | — | horizon |

**Rescue verdict: NOT resurrectable from strict r ≤ 0.06 by one contact.**
From a measured strict dead state (r = 0.018 ≤ 0.06, crossed after S8 —
the exact failure state the cadence arm left tea_tin in), one full
gate-passed phrasing-diverse re-teach moved r by only **+0.063** (margin
+0.84 nats of a 13.3-nat teach gain), back to 0.081 — just above the dead
line but solidly inside the near-dead band, and it re-decayed below 0.06
one session after the serve before ending 0.139. The response is real but small —
beneath the item's own un-served oscillation amplitude (~±0.1 r across the
5 held probes). Contrast with the trigger arm's metric_units
(near-dead band, r = −0.14: +1.252 on a variant that landed): first
direct evidence that the strict r ≤ 0.06 state is qualitatively harder
than the near-dead band, at n = 1. "Resurrectable from strict r ≤ 0.06":
**not demonstrated — claimed not, single item, single contact.** The
honest claim is: near-dead items are rehabilitable (trigger arm), strict
dead-slot items are not rehabilitated by one contact (this arm), and the
bounded fallback (deadline serve at S12) was not needed because the
crossing occurred unaided.

## 7. Tier-2 merge verdict

| arm (context) | aggregate inc → cand | protected agg | protected regressions | drift | decision |
|---|---|---|---|---|---|
| cadence | 0.3156 → 0.0801 | 0.255 → 0.030 | fact-paris 1→0, fact-tokyo 1→0 | 1.000 (fail) | ROLLBACK |
| trigger | 0.3156 → 0.0740 | 0.255 → 0.043 | paris, tokyo, refuse-color-seven, ho-refuse-weigh | 1.000 (fail) | ROLLBACK |
| aged (this) | 0.3156 → **0.0787** | 0.255 → **0.061** | paris 1→0, tokyo 1→0, refuse-color-seven 0.408→0.348, ho-refuse-weigh 0.397→0.319 | **1.000 (fail)** | **ROLLBACK** |

Expected ROLLBACK, confirmed a fifth time. Scheduler choice does not
change merge-grade behaviour, as predicted.

## 8. Claims and limits

**Claims (what the curves support).**

1. *The aged-priority queue mechanically eliminates first-trip starvation.*
   Under the formula, tea_tin — the trigger arm's 6-session starvation
   case — ranked #1 at its first two probes and would have been served at
   S7; 4/5 first-trips were served within ≤2 sessions (waits {1,1,2,2,4};
   the 4 was inflated one session by the rescue pre-emption and one by
   fav_show's age-init overrunning it at S10). Max first-trip wait 4 vs
   the trigger arm's ∞. The memory manager's starvation defect is fixed
   at the level it was measured.
2. *The aggregate does NOT improve — it regresses.* 2/6 items r_final
   ≥ 0.50 (trigger 4/6), mean 0.279 (0.581), dead slots 2 with min
   −0.17 (0 dead, min 0.17). The success read "≥ cadence's 4/6 + 0 dead
   slots + mean ≥ 0.58" is **not met**. The scheduling fix as implemented
   is breadth-first over first-trips and blind to re-trips: the starvation
   moved from first-trips to re-trips (metric_units re-trip waited 5
   probes, pet_dog's 4; both end dead).
3. *The strict dead slot is NOT resurrectable by one contact* (§6): from
   r = 0.018, one re-teach gave Δr = +0.063, landing at 0.081 — near-dead,
   re-decaying. Honest answer: "not" (n = 1, one item, one contact), with
   the near-dead-band caveat standing from the trigger arm (metric_units
   −0.14 → 0.69 was near-dead, not strict-dead).
4. *Tier 2 unchanged*: ROLLBACK (0.3156 → 0.0787, drift 1.000 fail); the
   two-tier read holds a fifth time.

**Overall verdict:** the aged formula wins the specific base case it was
aimed at (first-trip starvation) and loses everything it was not tuned
for. The scheduler, as a live memory-manager component, is **not done**:
next arm needs (a) age counted from first trip (or cumulative wait), not
from last serve, so re-trips cannot starve; (b) a deficit-floor or
deficit-weighted gate so self-recovered items cannot consume slots on age
alone; (c) optionally a capacity-aware threshold (the bracketing lesson
from the trigger arm still stands: the threshold is a service-rate
throttle). One more scheduler arm is queued before adoption.

**Limits.**

- Single run under the pinned seed (n = 1); replicate-grade determinism is
  verified, so variance is only seed/schedule choice, and this arm is one
  draw of it.
- The rescue pre-emption is confounded with the aggregate comparison: it
  consumed 1/6 capacity and granted tea_tin its third serve overall — the
  pure-scheduler counterfactual (§5.3) is computed from recorded queue
  priorities, not re-run.
- r_final is horizon-sensitive: the last two serves landed in S11/S12 and
  their bystander damage is frozen into the horizon snapshot (§5.4); an
  S13+ equilibration session would likely raise the un-served pair's
  scores, as several items self-recovered mid-phase un-served (metric_units
  −0.14 → 0.70 between S9 and S11).
- Rescue measured on one item with the library's extreme gain (13.3 nats):
  r-units are coarse for tea_tin; in nats the rescue response was +0.84,
  about half a typical successful serve — consistent with "dead items
  partially absorb teaching" rather than "dead items cannot be taught at
  all", but the r-claim in §6 stands as the honest headline.
- Free-run generations remain transcript colour; margin probes are the
  mechanism read.


## Reproduction

```bash
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_aged_sched.py --workdir lab/teach_aged_out \
  | tee lab/logs/teach_aged_sched.log

# determinism calibration (2 runs, both must bit-match each other and the
# cadence/trigger harnesses' committed seed-7 calibration):
python3 -u lab/teach_aged_sched.py --only metric_units --seed 7 --workdir <fresh A>
python3 -u lab/teach_aged_sched.py --only metric_units --seed 7 --workdir <fresh B>
```

Fresh workdir required (append-only session logs, invariant 10; adapter
deltas and `lab/teach_aged_out/`, `lab/teach_aged_calib/` are weights and
are not committed).
