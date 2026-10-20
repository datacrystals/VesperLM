# TEACH_AGED_SCHED_V3 — deficit-dominant priority + per-item serve budget (no-progress eviction)

Queued arm ("the v3 scheduler arm", item #9 of `lab/TEACH_PIPELINE.md`),
run 2026-10-19 on the laptop, CPU only, fp32. The v2 arm
(`lab/TEACH_AGED_SCHED_V2.md`) was explicitly NEGATIVE with its mechanistic
cause named: the **chronic-seniority hog** — under age-from-first-trip as
principal, tea_tin's three serves netted +0.274 r without ever crossing
threshold, so it re-queued with maximum seniority at every probe, consumed
3/6 slots, and greet_rule (eligible at all 5 later probes, ending with the
run's largest live deficit 0.987) never got one. The serve cap was never
the binding constraint; the dead-ban counterfactual cut both ways and was
not adopted.

This arm is the **smallest honest delta off v2**: same library, same teach
phase (bit-checked in-run), same slots/filler/threshold/cap/tripwires/
poison/tier-2 — **only the queue discipline changes**, and only the two
knobs the v2 traces named. Scope-honest: a scheduler variant arm, not a
memory architecture change.

Harness: `lab/teach_aged_sched_v3.py` (same import chain as v2:
`lab/teach_aged_sched.py` → `lab/teach_trigger_repair.py` →
`lab/teach_spaced_cadence.py`; deterministic seeded train loop inherited
unchanged). Numbers-only results: `lab/teach_aged3_out/teach_aged_sched_v3.json`;
log `lab/logs/teach_aged_sched_v3.log`; calibration workdirs
`lab/teach_aged3_calib/{A,B}`.

## 0. Determinism verification (done before the arm)

Two identical v3 calibration teaches (`--only metric_units --seed 7`, fresh
workdirs, sequential, same code path the arm uses), before the arm started:

| run | baseline margin | post-teach margin | gain |
|---|---|---|---|
| calib A | −1.12088942527771 | **−0.14622879028320312** | +0.975 |
| calib B | −1.12088942527771 | **−0.14622879028320312** | +0.975 |

Post-teach margins **bit-identical across the two v3 replays**, and both
**bit-match the cadence / trigger / aged-v1 / aged-v2 committed seed-7
calibration** (baseline −1.12088942527771, post-teach −0.14622879028320312)
— the fifth harness in the chain inherits the deterministic train loop
exactly.

In-run cross-check (teach phase S1..S6 runs the verbatim cadence session
runner; all S1..S6 margins compared against the committed cadence JSON):

```
CADENCE-IDENT: teach phase S1..S6 vs committed cadence run —
max |margin Δ| = 0.000e+00 (BIT-IDENTICAL; worst before/fav_show)
```

Every difference this arm shows after S6 is therefore attributable to the
v3 serving rule alone (S1..S9 further bit-track v2/run-for-run where the
queue states coincide — see §3).

## 1. Design

| item | value |
|---|---|
| checkpoint | `lab/imported/vk_sft_r3/step_1800/chat_model` — identical to E3 / A/B / cadence / trigger / v1 / v2 |
| runtime | laptop CPU, fp32, 12 threads; tier-2 Immune never leaves CPU (I7); LoRA rank 8, alpha 16, lr 2e-3, kl_coef 0.05, 24 steps/session — identical knobs |
| seed | 7 on every consolidate call (cadence-hardening §1a) |
| library | the six items verbatim: fav_show (F), math_words (S), tea_tin (F), metric_units (S), pet_dog (F), greet_rule (S) |
| teach phase | S1–S6, **identical to v2 / v1 / trigger / cadence** (bit-checked in-run) |
| scheduler phase | S7–S12 (6 later sessions, same filler slices): 5 generic-chat approval triples + at most ONE re-teach turn (2 triples, 28.6% ≤ 30% cap) |
| trigger rule | unchanged: after every session (S6..S11), item queued iff r_i(k) < 0.50 |
| serving rule | **v3: deficit-dominant order + per-item serve budget** (formula below), one pop per session (serve cap 1, structural) |
| tripwire safety | 8-variant extended pools verbatim; every response string unique run-wide |
| fallback | one gate-safe spaced re-teach max (policy, armed on any tripwire) |
| tier 2 | ONE Immune merge verdict on the accumulated adapter at the end, constants untouched |
| bar to beat | the trigger arm's horizon: ≥ 4/6 held r ≥ 0.50, 0 dead (r ≤ 0.06), mean ≥ 0.581 — plus v2's two queue-health reads (no hog, no eligible-trip starvation) |

### The v3 scheduler (exact formula as run, documented before running)

For each item *i* with a recorded trip, evaluated at the probe after
session *k* (which seeds the batch of session *k+1*):

```
first_trip_i  = earliest probe session k' (>= 6) with r_i(k') < thresh
                — sticky for the whole scheduler phase (unchanged from v2);
                  reporting + TIE-BREAK ONLY, never the principal
age_i(k)      = k − first_trip_i
deficit_i(k)  = max(0, thresh − r_i(k))             # r re-probed live
eligible_i(k) ⟺ deficit_i(k) > 0  AND  i not evicted

SERVE ORDER (deterministic sort key, highest first):
    1. deficit_i(k)          — DEFICIT-DOMINANT first-order term
                               (equivalently: lowest live r first)
    2. age_i(k)              — TIE-BREAKER ONLY (older first_trip wins
                               exact deficit ties)
    3. library ORDER index   — fully deterministic under the pinned seed
```

### Per-item serve budget / no-progress eviction (documented before running)

```
episode_i        = serves since the most recent probe at which
                   r_i(k') >= thresh (0 if none); a held probe (r >=
                   thresh), served or not, RESETS the episode to 0
per-serve yield  = repair Δr = r_after − r_pre around the serve
NO_PROGRESS_DR   = 0.10  — a serve with Δr <= +0.10 counts as NO PROGRESS;
                   chosen from v2's own probes: tea_tin's third serve
                   yielded +0.094, the magnitude of un-served bystander
                   oscillation (~±0.1 r) — noise. Logged per serve as a
                   diagnostic flag (episodes carry no_progress_serves).
SERVE_BUDGET     = 2 serves per unheld episode. Justification: v2's tea_tin
                   episode was (−0.059, +0.239, +0.094): 2 serves netted
                   +0.18 without holding and serve 3 was already at noise;
                   the trigger arm's metric_units crossed threshold on
                   serve 2, so budget 2 touches no measured successful
                   2-serve rehab.
EVICTION         = at a probe, a QUEUED item with episode serves >= 2 is
                   EVICTED from the queue (out of service competition) and
                   logged with its episode trace.
RE-ADMISSION     = an evicted item returns ONLY if it first reads r >=
                   thresh UNAIDED at a later probe (a genuine self-recovery
                   — "re-trips by decay from a higher point") and then
                   re-trips; the fresh trip starts a FRESH episode
                   (budget 0). Re-trips without an intervening unaided
                   held probe are logged as policy-blocked. (If the
                   unaided-hold branch never fires, the effective policy is
                   "never re-admit during the horizon" — §3 reports which
                   was used: NEVER RE-ADMITTED, the unaided-hold branch
                   never fired.)
```

Unchanged from v2: deficit-floor eligibility (self-recovered items cannot
consume slots on age), dead-ban queue-position trace (log-only
counterfactual), serve cap 1/session, rescue not re-run.

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
VERDICT SCHED probe after S6:  QUEUE tea_tin r=0.324, metric_units r=0.249, pet_dog r=0.153 | queue (deficit order) [pet_dog 0.347 E, metric_units 0.251 E, tea_tin 0.176 E]
VERDICT S7  [agedv3/pet_dog]     gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> pet_dog [deficit 0.347 age 0]) train loss 8.935->0.218
VERDICT EVENT [re-teach/pet_dog] SERVED first trip S6 r=0.153, episode serve 1/2 | served S7 v1 | r 0.153 -> 0.518 (repair +0.365)
VERDICT SCHED probe after S7:  QUEUE fav_show r=0.258, greet_rule r=0.229 | queue (deficit order) [tea_tin 0.411 E, metric_units 0.302 E, greet_rule 0.271 E, fav_show 0.242 E]
VERDICT S8  [agedv3/tea_tin]     gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> tea_tin [deficit 0.411 age 1]) train loss 3.452->0.158
VERDICT EVENT [re-teach/tea_tin] SERVED first trip S6 r=0.324, episode serve 1/2 [NO-PROGRESS serve] | served S8 v1 | r 0.089 -> 0.030 (repair -0.059)
VERDICT SCHED probe after S8:  QUEUE math_words r=0.381, tea_tin r=0.030 | queue (deficit order) [tea_tin 0.470 E (age 2, ep1), metric_units 0.346 E, greet_rule 0.145 E, math_words 0.119 E, fav_show 0.018 E]
VERDICT S9  [agedv3/tea_tin]     gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> tea_tin [deficit 0.470 age 2]) train loss 2.247->0.214
VERDICT EVENT [re-teach/tea_tin] SERVED re-trip S8 r=0.030, episode serve 2/2 | served S9 v2 | r 0.030 -> 0.269 (repair +0.239)
VERDICT SCHED probe after S9:  QUEUE tea_tin r=0.269 | queue (deficit order) [greet_rule 0.637 E, fav_show 0.517 E, tea_tin 0.231 E (ep2), math_words 0.104 E, metric_units 0.036 E]
VERDICT S10 [agedv3/greet_rule]  gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> greet_rule [deficit 0.637 age 2]) train loss 3.760->0.164
VERDICT EVENT [re-teach/greet_rule] SERVED first trip S7 r=0.229, episode serve 1/2 | served S10 v0 | r -0.137 -> 0.502 (repair +0.638)
VERDICT SCHED probe after S10: QUEUE pet_dog r=0.139 | EVICTED tea_tin (episode drs [-0.059, +0.239], live r 0.022) | queue (deficit order) [pet_dog 0.361 E, metric_units floor, fav_show floor, math_words floor]
VERDICT EVICT [budget/tea_tin]   EVICTED — 2 serves without holding r >= 0.50; episode drs [-0.059, +0.239]; live r 0.022 | out of service competition; watch trace on
VERDICT S11 [agedv3/pet_dog]     gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> pet_dog [deficit 0.361 age 4]) train loss 3.420->0.118
VERDICT EVENT [re-teach/pet_dog] SERVED re-trip S10 r=0.139, episode serve 1/2 | served S11 v2 | r 0.139 -> 0.911 (repair +0.772)
VERDICT SCHED probe after S11: QUEUE greet_rule r=0.479 | queue (deficit order) [metric_units 0.167 E, fav_show 0.129 E, greet_rule 0.021 E, math_words floor] | evicted watch: tea_tin r 0.295
VERDICT S12 [agedv3/metric_units] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> metric_units [deficit 0.167 age 5]) train loss 3.470->0.152
VERDICT EVENT [re-teach/metric_units] SERVED first trip S6 r=0.249, episode serve 1/2 [NO-PROGRESS serve] | served S12 v1 | r 0.333 -> 0.130 (repair -0.203)
```

VERDICT S13 [poison/banana-batch] gate=stub-tier1: ROLLBACK — target collapse: one response is 100% of the batch | incumbent adapter intact: True
VERDICT MERGE [accumulated adapter vs raw base] gate=immune-tier2: ROLLBACK — protected probe fact-paris regressed 1.000 -> 0.000 | protected probe fact-tokyo regressed 1.000 -> 0.000 | protected probe refuse-color-seven regressed 0.408 -> 0.026 | protected probe ho-refuse-weigh regressed 0.397 -> 0.042 | aggregate 0.3156 -> 0.0475, drift 1.000 label=fail

Gate behaviour: **12/12 promotes, 0 tripwire events** (top_response_fraction
0.14 everywhere after S1; every re-teach string fresh). The poison canary
tripped the collapse rule at 100% and the incumbent stayed byte-untouched.
The one-fallback tripwire policy was **not** exercised; the run completed
clean (no harness failure — no `lab/FAILURES.md` entry).

## 3. Scheduler event log + queue-wait statistics

Six services in six sessions. Arrivals: 6 first-trips (tea_tin S6,
metric_units S6, pet_dog S6, fav_show S7, greet_rule S7, math_words S8) +
4 live re-trips (tea_tin S8, S9; pet_dog S10; greet_rule S11) + 2
policy-blocked tea_tin re-trips while evicted.

| # | item | kind | tripped after | served in | wait | variant | r before → after | repair Δr | deficit (age) at serve |
|---|---|---|---|---|---|---|---|---|---|
| 1 | pet_dog | queue (first) | S6 | S7 | 1 | v1 | 0.153 → 0.518 | **+0.365** | 0.347 (0) |
| 2 | tea_tin | queue (first) | S6 | S8 | 2 | v1 | 0.089 → 0.030 | **−0.059** [no-progress] | 0.411 (1) |
| 3 | tea_tin | queue (re-trip) | S8 | S9 | 1 | v2 | 0.030 → 0.269 | **+0.239** | 0.470 (2) |
| 4 | greet_rule | queue (first) | S7 | S10 | 3 | v0 | −0.137 → 0.502 | **+0.638** | 0.637 (2) |
| 5 | pet_dog | queue (re-trip) | S10 | S11 | 1 | v2 | 0.139 → 0.911 | **+0.772** | 0.361 (4) |
| 6 | metric_units | queue (first) | S6 | S12 | **6** | v1 | 0.333 → 0.130 | **−0.203** [no-progress] | 0.167 (5) |

Queue-wait histogram (serves, sessions trip→serve): **{1: 3, 2: 1, 3: 1,
6: 1}; max 6** (metric_units' first trip) — vs v2's {1: 4, 2: 1, 5: 1} and
v1's {1: 2, 2: 2, 4: 1}. Per-event repair: **4/6 positive, mean +0.384**
(trigger +0.445, v1 +0.276, v2 +0.201); both negatives are [no-progress]
flagged and one of them (metric_units v1) is the known-harmful
item-variant pair now at four consecutive negative readings across four
arms (trigger −0.325, v1 −0.291, v2 −0.330, v3 −0.203).

### Event-contact latency for EVERY trip event (the starvation read)

| item | trip kind | tripped after | first trip | outcome | latency / probes waited |
|---|---|---|---|---|---|
| tea_tin | first | S6 | S6 | served S8 | 2 |
| metric_units | first | S6 | S6 | served S12 | **6** |
| pet_dog | first | S6 | S6 | served S7 | 1 |
| fav_show | first | S7 | S7 | **unserved** (eligible 4 probes; floor at S10 probe) | 4 probes |
| greet_rule | first | S7 | S7 | served S10 | 3 |
| math_words | first | S8 | S8 | **unserved** (eligible 2 probes; self-recovered to 1.27 floor from S10 on) | 3 probes |
| tea_tin | re-trip | S8 | S6 | served S9 | 1 |
| tea_tin | re-trip | S9 | S6 | **EVICTED at budget** (ep 2, drs −0.059/+0.239) | 2 probes |
| pet_dog | re-trip | S10 | S6 | served S11 | 1 |
| greet_rule | re-trip | S11 | S7 | unserved (horizon edge) | 0 probes |
| tea_tin | re-trip | S10 (r 0.022) | S6 | **POLICY-BLOCKED** (evicted, no unaided hold) | — |
| tea_tin | re-trip | S11 (r 0.295) | S6 | **POLICY-BLOCKED** (evicted, no unaided hold) | — |

**Hog pattern re-check (v2's §5.1 mechanism):** dead. Maximum single-item
serve share is **2/6** (tea_tin 2, pet_dog 2; greet_rule, metric_units 1
each; fav_show, math_words 0) vs v2's 3/6 tea_tin monopoly. tea_tin hit the
budget at episode serve 2, was evicted at the probe after S10 with a live
r of 0.022, and its two later sub-threshold probes (0.022 → 0.295) were
policy-blocked — never re-admitted, because it never produced the required
unaided held probe. The two serves it gave back relative to v2 went to
greet_rule (the item v2 starved at 5/5 eligible probes) and pet_dog's
re-trip.

**First-trip starvation:** technically present for fav_show (4 probes
waited) and metric_units (6-probe latency — worse than v2's 5!), but the
two cases split: math_words unserved is a **non-casualty** (self-recovered
un-served from 0.38 to 1.27, the run's best r_final — the floor correctly
kept slots off it), while fav_show is the **honest starvation casualty**
(v3's exact greet_rule-analogue): tripped S7 (deficit 0.242), never held
the queue's maximum deficit at any probe, self-recovered to 0.80 at the
S10 probe (floor), then crashed un-served 0.37 → −0.22 across S11–S12.
Re-trip starvation at horizon: none beyond the 0-probe horizon edge
(greet_rule re-trip S11 waited 0). **The mechanism is renamed, not solved:
deficit-dominant serving starves the shallow-deficit items** — an item
whose deficit is never the live maximum never rises, which is the trigger
arm's tea_tin geometry with a different victim set.

## 4. Retention r_i(k) — v3 vs v2 vs v1 vs trigger (committed numbers)

Baselines and S1..S6 bit-identical across the arms (CADENCE-IDENT,
max |Δ| = 0.0). S7–S9 additionally bit-track v2 run-for-run (same serve
choices: pet_dog v1, tea_tin v1, tea_tin v2; v2's probe margins and v3's
agree to the printed precision); **the trajectories split at S10** — v2
served tea_tin a third time; v3's budget evicted it and deficit-dominant
order served greet_rule instead. The table shows margin (r); **bold** marks
re-teach sessions in this arm; `|` = teach anchor.

| item | g_i | S6 (identical) | S7 | S8 | S9 | S10 | S11 | S12 |
|---|---|---|---|---|---|---|---|---|
| fav_show (F) | +2.73 | +5.16 (1.26) | +2.42 (0.26) | +3.03 (0.48) | +1.67 (−0.02) | +3.90 (0.80) | +2.73 (0.37) | +1.13 (−0.22) |
| math_words (S) | +5.24 | +3.06 (1.23) | +1.44 (0.92) | −1.38 (0.38) | −1.30 (0.40) | −0.33 (0.58) | +1.06 (0.85) | +3.28 (1.27) |
| tea_tin (F) | +13.31 | +3.05 (0.32) | −0.09 (0.09) | **−0.87 (0.03)** | **+2.32 (0.27)** | −0.98 (0.02) | +2.66 (0.30) | +1.86 (0.23) |
| metric_units (S) | +2.30 | −0.55 (0.25) | −0.67 (0.20) | −0.77 (0.15) | −0.05 (0.46) | +0.28 (0.61) | −0.36 (0.33) | **−0.82 (0.13)** |
| pet_dog (F) | +4.90 | +1.02 (0.15) | **+2.80 (0.52)** | +3.51 (0.66) | +5.70 (1.11) | +0.95 (0.14) | **+4.73 (0.91)** | +4.28 (0.82) |
| greet_rule (S) | +6.04 | +3.47 (1.00) | −1.19 (0.23) | −0.43 (0.35) | −3.40 (−0.14) | **+0.46 (0.50)** | +0.32 (0.48) | −1.01 (0.26) |

### Headline: the serving-rule ablation grid (same teach phase, seed, slots, threshold)

Cadence/trigger/v1/v2 columns are from their committed docs (reused, not
re-run).

| item | kind | serves (v3) | cadence r_final | trigger r_final | aged v1 r_final | aged v2 r_final | **v3 r_final** |
|---|---|---|---|---|---|---|---|
| fav_show | fact | **0 (starved, shallow-deficit)** | 0.77 | 0.96 | 0.65 | 0.58 | **−0.22** (worst over arms) |
| math_words | style | 0 (self-recovered) | 0.79 | 0.66 | 0.36 | 0.43 | **1.27** (best over arms) |
| tea_tin | fact | 2 (S8, S9) + evict | 0.01 | 0.17 | 0.14 | 0.32 | 0.23 |
| metric_units | style | 1 (S12, negative repair) | 0.06 | 0.69 | −0.12 | 0.78 | 0.13 |
| pet_dog | fact | 2 (S7, S11) | 0.52 | 0.50 | −0.17 | −0.32 | **0.82** (best over arms) |
| greet_rule | style | 1 (S10) | 0.52 | 0.50 | 0.81 | −0.11 | 0.26 |

| metric | cadence | trigger | aged v1 | aged v2 | **v3** | read |
|---|---|---|---|---|---|---|
| items r_final ≥ 0.50 | 4/6 | 4/6 | 2/6 | 2/6 | **2/6** | trigger bar NOT met |
| dead slots r ≤ 0.06 | 2 | 0 | 2 (neg) | 2 (neg) | **1 (neg: fav_show)** | trigger bar NOT met (best of the aged line) |
| mean r_final | 0.445 | 0.581 | 0.279 | 0.280 | **0.416** | trigger bar NOT met (best of the aged line) |
| min r_final | 0.01 | 0.17 | −0.17 | −0.32 | **−0.22** | worse than v1 |
| repair mean / positive | — | +0.445 (5/6) | +0.276 (4/5) | +0.201 (4/6) | **+0.384 (4/6)** | second to trigger |
| max single-item serve share | — | 2/6 | 2/6 | **3/6 (hog)** | **2/6** | v2 hog dead |
| first-trip starvation at horizon | — | tea_tin ∞ | none (mechanical) | greet_rule (5/5 eligible probes) | **fav_show (4 probes, shallow-deficit; ends dead); metric_units latency 6** | re-named, not solved |
| re-trip starvation at horizon | — | pet_dog, greet_rule | metric_units, pet_dog (end dead) | none (horizon-edge only) | **none (0-probe horizon edge only)** | v2's win holds |

The honest headline: **v3 is the best of the aged line and still misses all
three bar criteria.** The two named knobs did exactly their named jobs —
the hog is dead (2/6 max share; greet_rule, v2's 5/5-eligible starvation
case, was served at its first head-eligible probe with the run's best
mid-curve repair +0.638) and the budget evicted the chronic item once —
but the horizon still reads 2/6 held (bar 4/6), 1 dead (bar 0; fav_show,
missed at −0.22), mean 0.416 (bar 0.581). And the serving rule grew a new
failure mechanism to replace the one it removed (§5.1).

1. **first-trip starvation zero (v2's criterion-1 inheritance) — NOT HELD.**
   fav_show waited 4 probes eligible and ended dead; metric_units' first
   trip stretched to latency 6 (the run's worst on record, any arm).
2. **≥ 4/6 held ≥ 0.50 — NOT met (2/6: math_words 1.27, pet_dog 0.82).**
3. **0 dead — NOT met (1: fav_show −0.22).**
4. **mean ≥ 0.581 — NOT met (0.416).**

No partial credit. The scheduler still does not exist; the verdict is
explicitly negative (§8).

## 5. Why the aggregate still fails (mechanics, from the recorded traces)

1. **Deficit-dominant serving converts the seniority hog into
   shallow-deficit starvation.** Priority = max live deficit first means an
   item serves only when it is the run's most-damaged member *at that
   probe*. fav_show's deficit peaked at position #2 (0.517 at the S9
   probe, behind greet_rule's 0.637) and was otherwise 0.018–0.24 — never
   the maximum, never served; it oscillated 0.26 → 0.48 → −0.02 → 0.80 →
   0.37 → −0.22 entirely as a bystander and the horizon froze its crash.
   metric_units (first trip S6) was behind tea_tin twice, greet_rule once,
   pet_dog once, and only reached the head after its own self-recovery to
   0.61 (floor at S10) re-crashed — served at S12 with latency 6 and a
   **negative** repair (−0.203). This is the trigger arm's starvation
   geometry one level down: depth-first queues starve whoever is never
   deepest; seniority queues starve whoever is never oldest. Neither is a
   scheduler.
2. **The horizon froze right after the run's most destructive serve.** S12
   (metric_units v1, itself −0.203) damaged every bystander: fav_show
   0.37 → −0.22, greet_rule 0.48 → 0.26, tea_tin 0.30 → 0.23,
   pet_dog 0.91 → 0.82 (only math_words climbed, 0.85 → 1.27). The
   v1 §5.4 / v2 §5.3 end-loaded-serve caveat is now three-for-three: an
   end-loaded negative-yield serve poisons the horizon snapshot of every
   un-served item. The freeze costs the run real numbers — at the pre-S12
   (S11) snapshot the r values read 0.37, 0.85, 0.30, 0.33, 0.91, 0.48 —
   but even that best-case freeze leaves **2/6 ≥ 0.50 and mean 0.540**,
   still short of the bar on both counts: the negative verdict does not
   hinge on the unlucky last serve.
3. **The eviction was mechanically correct and capacity-neutral at the
   horizon.** tea_tin's budget stop cost it its v2 third serve (+0.094,
   noise-level) and it ended 0.23 un-assisted (watch: 0.022 → 0.295 →
   0.23) vs v2's 0.32 with 3 serves — the freed slot went to the run's two
   best mid-curve repairs (greet_rule +0.638, pet_dog +0.772). Net r across
   the three items involved: strongly positive. The budget was not the
   failure; the queue's *depth-first geometry* was (§5.1).
4. **Variant luck, fourth confirmation of a named pair.** metric_units ×
   variant-1 is now 0-for-4 across four arms (−0.325, −0.291, −0.330,
   −0.203); every positive metric_units repair in the series came on
   v0/v2. The v3 arm drew v1 for it at the horizon-edge S12 serve — the
   single worst possible draw for the horizon read. A scheduler cannot see
   variants; the honest statement stays "per-contact yield is
   variant-dependent, not a scheduler fault" — but the series now supports
   flagging bad item–variant pairs as a *repair-content* issue, not a
   queueing issue.

## 6. Secondary reads

### Dead-ban queue position (log-only counterfactual)

| probe after | queued in dead band (r ≤ 0.06) | v3 served (next session) | dead-ban would serve | same? |
|---|---|---|---|---|
| S6 | — | pet_dog | pet_dog | yes |
| S7 | — | tea_tin | tea_tin | yes |
| S8 | tea_tin (0.030) | tea_tin | metric_units | **no** |
| S9 | greet_rule (−0.14), fav_show (−0.02) | greet_rule | tea_tin | **no** |
| S10 | — (tea_tin evicted, not in queue) | pet_dog | pet_dog | yes |
| S11 | — | metric_units | metric_units | yes |

Two reorderings in five probes, and they cut both ways again: banning
tea_tin at S8 would have spent that slot on metric_units upstream
(arguably better in advance), but banning at S9 would have **blocked the
run's best serve** (greet_rule −0.137 → 0.502, +0.638) and fed tea_tin a
third serve — re-instantiating the v2 hog one session earlier. Second
arm in a row where the flat dead-band ban reorders 2/5 probes with one
swap in each direction: **not adopted**; the counterfactual earns its
log-only cost (a flat r ≤ 0.06 line still marks weak single contacts, not
worthless ones — v2's §6 conclusion replicates).

### Eviction watch / re-admission branch

tea_tin evicted after probe S10 (episode drs −0.059, +0.239, live r
0.022). Watch trace: probe S10 0.022 → probe S11 0.295 (un-assisted
climb). Re-admission rule (unaided held probe required, then fresh trip):
**never fired** — tea_tin never reached r ≥ 0.50 unaided within the
horizon, so its two sub-threshold re-trips (0.022, 0.295) were
policy-blocked and the effective policy was *never re-admit during the
horizon*. The blocked-trip read is honest: the ban's upper bound cost is
whatever another tea_tin serve would have added — bounded by its own
noise-level third serve in v2 (+0.094) — against the measured gain of the
freed slot (+0.638 the next deep item, +0.772 pet_dog's re-trip).

### Threshold brackets (snapshot sensitivity, post-hoc)

From the recorded probes (item sets tripping at each bracket, de-duplicated
per probe): @0.25 — {metric_units, pet_dog} at S6, {tea_tin, metric_units,
greet_rule} at S7, {tea_tin, metric_units} at S8, {greet_rule, fav_show} at
S9, {pet_dog, tea_tin} at S10, {} at S11 (arrivals ≈ 7 vs 6 slots, but NOTE:
fav_show enters only at S9, not S7); @0.75 — superset of the run's trips at
every probe (4–6 queued). Same lesson as the prior three arms: the
threshold throttles demand, never the ordering between survivors — under
@0.25 fav_show would *start* queued later (S9), not earlier, and
deficit-dominant ordering among the @0.25 survivors is unchanged. The
service-rate-vs-capacity mismatch relaxes; the shallow-deficit geometry
(§5.1) does not.

## 7. Tier-2 merge verdict

| arm (context) | aggregate inc → cand | protected agg | protected regressions | drift | decision |
|---|---|---|---|---|---|
| cadence | 0.3156 → 0.0801 | 0.255 → 0.030 | fact-paris, fact-tokyo | 1.000 (fail) | ROLLBACK |
| trigger | 0.3156 → 0.0740 | 0.255 → 0.043 | paris, tokyo, refuse-color-seven, ho-refuse-weigh | 1.000 (fail) | ROLLBACK |
| aged v1 | 0.3156 → 0.0787 | 0.255 → 0.061 | paris, tokyo, refuse-color-seven, ho-refuse-weigh | 1.000 (fail) | ROLLBACK |
| aged v2 | 0.3156 → 0.1073 | 0.255 → 0.056 | paris, tokyo, refuse-color-seven, ho-refuse-weigh | 1.000 (fail) | ROLLBACK |
| aged v3 (this) | 0.3156 → **0.0475** | 0.255 → 0.006 | paris 1→0, tokyo 1→0, refuse-color-seven 0.408→0.026, ho-refuse-weigh 0.397→0.042 | **1.000 (fail)** | **ROLLBACK** |

Expected ROLLBACK, confirmed a **seventh** time. Scheduler choice still
does not change merge-grade behaviour (the v3 candidate is the series'
lowest tier-2 read — 0.0475 vs v2's 0.1073 — an irrelevant reading: the
absolute vetoes fire regardless and nothing about the gate's stance
changes).

## 8. Claims and limits

**Claims (what the curves support).**

1. *Both named knobs behaved exactly as designed, measurably.* The
   deficit-dominant order served the run's two deepest items at their
   deepest probes (greet_rule deficit 0.637 → +0.638 repair; tea_tin
   deficit 0.470), the budget fired once (tea_tin, episode drs −0.059 /
   +0.239, evicted at live r 0.022), the hog died (max serve share 2/6,
   down from v2's 3/6 item-monopoly), v2's 5/5-eligible starvation case
   (greet_rule) was served at its first head-eligible probe, and re-trip
   starvation stayed at zero beyond the 0-probe horizon edge. As
   mechanism-testing, both knobs are validated.
2. *None of the three bar criteria is met — the verdict is explicitly
   negative.* 2/6 held ≥ 0.50 (bar 4/6), 1 dead (bar 0; fav_show −0.22),
   mean 0.416 (bar 0.581). v3 strictly improves on v1/v2 aggregates
   (0.279/0.280 mean, 2 dead each) and strictly loses to trigger on every
   bar metric. The trigger arm's lowest-r-first remains the strongest
   measured scheduler — and v3's deficit-dominant order is closest to it
   by construction, which sharpens the series' conclusion: **the ordering
   rule is not where the remaining loss lives**.
3. *The series now names the residual failure shape with confidence:*
   with six slots and ~10 trip-events, every pure-priority rule tested
   (lowest-r, age-principal, deficit-principal) leaves exactly the same
   structural residue — one item never (or last-proto) served, and the
   horizon snapshot frozen mid-oscillation after an end-loaded serve.
   What actually moved v3's mean from 0.28 to 0.42 was **serve content
   landed on the right items** (v0 greet_rule +0.638, v2 pet_dog +0.772),
   not the ordering — and its biggest single-element loss was a
   *content* draw (metric_units × v1, 0-for-4 across four arms). Next
   surfaces, in order of evidence weight: (a) variant/item-pair quality as
   a repair-content layer (v1-for-metric_units is a known-bad draw — pin or
   skip it); (b) small-positive-deficit churn protection (fav_show's
   eligible-but-shallow band needs either a fairness rotation floor or a
   "tripped-and-never-served" override that forces a serve after N
   eligible probes regardless of depth); (c) the horizon discipline
   itself (S13+ equilibration) — the snapshot-freeze caveat is measured
   three times and is worth one cheap arm of its own.
4. *Tier 2 unchanged*: ROLLBACK (0.3156 → 0.0475, drift 1.000 fail); the
   two-tier read holds a seventh time.

**Limits.**

- Single run under the pinned seed (n = 1); variance over seed/schedule is
  one draw. The determinism chain is fully intact (two bit-identical
  calibrations matching all four prior harnesses; CADENCE-IDENT 0.0; S7–S9
  bit-tracking v2), so divergence after S10 is serving-rule-attributable.
- The horizon read is again mid-oscillation (§5.2) and immediately after
  the run's most destructive serve; the S11 snapshot (3/6 ≥ 0.45, mean
  0.54) bounds the freeze artefact but cannot flip the verdict.
- fav_show's starvation is confounded with its own volatility (span
  1.02 r across the phase): depth-first starving a high-variance item is
  the queue's fault, but its 0.80-at-S10 self-recovery shows the item was
  not uniformly rescue-worthy. The mechanism claim rests on the recorded
  deficit ranks at each probe, not on a counterfactual re-run.
- tea_tin's eviction benefit read (§6) assumes its hypothetical third
  serve ≈ v2's (+0.094); variant state differs across arms (v2 drew v3 for
  it; v3's pool index at eviction was v3 = next), so the bound is soft.
- Free-run generations remain transcript colour; margin probes are the
  mechanism read. Repair means (+0.384 v3, +0.201 v2, +0.276 v1, +0.445
  trigger) are per-contact reads over 6 serves each and move with variant
  luck (§5.4).

## Reproduction

```bash
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_aged_sched_v3.py --workdir lab/teach_aged3_out \
  | tee lab/logs/teach_aged_sched_v3.log

# determinism calibration (2 runs; both must bit-match each other and the
# v1/v2/trigger/cadence committed seed-7 calibration):
python3 -u lab/teach_aged_sched_v3.py --only metric_units --seed 7 --workdir <fresh A>
python3 -u lab/teach_aged_sched_v3.py --only metric_units --seed 7 --workdir <fresh B>
```

Fresh workdir required (append-only session logs, invariant 10; adapter
deltas and `lab/teach_aged3_out/`, `lab/teach_aged3_calib/` are weights and
are not committed). Wall: 11641 s (12 threads, CPU fp32, loaded machine —
~3.5× the v2 wall on the same box; the arm's margins are seed-deterministic,
so wall differences do not touch any number reported here).
