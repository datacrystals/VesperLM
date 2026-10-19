# TEACH_TRIGGER_REPAIR — reactive (trigger-based) consolidation on the SFT-r3 chat checkpoint

Queued experiment #5 of `lab/TEACH_PIPELINE.md` ("wipeout-triggered
re-teaching"), run 2026-10-19 on the laptop, CPU only, fp32. The cadence arm
(`lab/TEACH_SPACED_CADENCE.md`) showed once-per-6-session cyclic replay is
**refresh-with-repair, not annealing**: it held 4/6 items (r_final ≥ 0.52)
but could not rehabilitate the two items already dead at their slot
(tea_tin, metric_units, r_final ≤ 0.06). This arm replaces the cadence's
fixed anneal-phase schedule with a **memory-manager loop**: after every
session, re-probe all items; any item whose retention ratio r_i(k) drops
below a threshold is re-taught into the NEXT session's batch.

Harness: `lab/teach_trigger_repair.py` (extends `lab/teach_spaced_cadence.py`
— library, session packs, filler, probes, teach-phase runner imported; only
the later-phase schedule differs). Numbers-only results:
`lab/teach_trigger_out/teach_trigger_repair.json`; log
`lab/logs/teach_trigger.log`; calibration workdirs
`lab/teach_trigger_calib/{A,B}`.

## 0. Determinism verification (done before the arm)

Two identical calibration teaches (`--only metric_units --seed 7`, fresh
workdirs, same code path the arm uses), run before the arm started:

| run | baseline margin | post-teach margin | gain |
|---|---|---|---|
| calib A | −1.12088942527771 | **−0.14622879028320312** | +0.975 |
| calib B | −1.12088942527771 | **−0.14622879028320312** | +0.975 |

Post-teach margins **bit-identical** across the two replays, and both
**bit-match the cadence harness's committed seed-7 calibration** for the
same item — the extension inherits the deterministic train loop exactly,
across both processes and harnesses.

In-run cross-check: the arm's teach phase S1..S6 runs the **verbatim cadence
session runner**; all S1..S6 margins were compared against the committed
cadence JSON:

```
CADENCE-IDENT: teach phase S1..S6 vs committed cadence run —
max |margin Δ| = 0.000e+00 (BIT-IDENTICAL; worst before/fav_show)
```

Δ = 0.0 on every probed margin (baselines + 6 sessions × 6 items). Every
difference this arm shows after S6 is therefore attributable to the
schedule swap alone — same seeds, same batches through the teach phase,
same interference realisations. The contrast with the cadence arm's
committed numbers is noise-free by construction.

## 1. Design

| item | value |
|---|---|
| checkpoint | `lab/imported/vk_sft_r3/step_1800/chat_model` — identical to E3 / A/B / cadence |
| runtime | laptop CPU, fp32, 12 threads; tier-2 Immune never leaves CPU (I7); LoRA rank 8, alpha 16, lr 2e-3, kl_coef 0.05, 24 steps/session — identical knobs |
| seed | 7 on every consolidate call (cadence-hardening §1a) |
| library | the cadence arm's 6 items verbatim: fav_show (F), math_words (S), tea_tin (F), metric_units (S), pet_dog (F), greet_rule (S) |
| teach phase | S1–S6, **identical to cadence** (same runner: 5-triple teach batch + one replay turn of the previous item, 2/7 = 28.6%) |
| trigger phase | S7–S12 (6 later sessions, same count, same filler slices as the cadence anneal phase): 5 generic-chat approval triples + at most ONE triggered re-teach turn (2 triples, 28.6%) |
| trigger rule | after EVERY session (S6..S11), re-probe all taught items; item queued iff r_i(k) = (m_i(k) − m_i(0)) / g_i < 0.50 |
| serving rule | one re-teach per session, lowest r first (then earlier trip), popped items leave the queue; rest wait — the 7-triple batch keeps replay share 2/7 = 28.6% ≤ 30% cap, same tripwire profile as cadence |
| tripwire safety | response strings unique run-wide (96/96 pool strings unique); replay variants 0..1 are the cadence pools verbatim (teach-phase freshness), 2..7 new phrasing-diverse variants (a reactive loop can re-teach one item more often than once-per-cycle: max 8 uses asserted) |
| fallback | one gate-safe spaced re-teach max (policy, armed on any tripwire) |
| tier 2 | ONE Immune merge verdict on the accumulated adapter at the end, constants untouched |
| baseline for the contrast | the cadence arm's committed numbers (§4 of TEACH_SPACED_CADENCE.md) — **reused, not re-run** |

### Threshold rationale: r < 0.50

The trigger fires when an item's margin falls below **half of its own teach
gain**. Chosen starting calibration:

1. It is the `TEACH_PIPELINE.md` **hinge** definition (item marked *hinge*
   below r = 0.50, *lost* below r = 0.10) — the reactive loop is the first
   consumer of that taxonomy, so it fires at the hinge boundary rather than
   at the arbitrary end of a fixed cycle.
2. The cadence outcome was **bimodal at the horizon**: 4/6 items r ≥ 0.52
   vs 2/6 r ≤ 0.06. 0.50 sits just under the held cluster's floor: anything
   still alive stays un-served, anything that has taken real damage is
   queued immediately. The cadence's two deaths both sat at r 0.15–0.32 at
   S6 — a 0.50 trigger queues them at the FIRST later session, the repair
   the round-robin could not deliver before S9–S10.
3. Asymmetric costs: missing a wipeout is the expensive error (cadence
   evidence: 12–14-nat wipeouts are not recovered at cycle density); a
   false re-teach costs one 2-triple turn.

Bracketing looks 0.25 / 0.75 are run as a **cheap secondary analysis** on
the recorded snapshots (no re-run): at each trigger probe the item sets
that would have tripped at the bracket thresholds are logged
(`trigger_evals.after_s*.brackets` in the JSON). Full re-runs at 0.25/0.75
would double the wall for a two-arm sensitivity read; the snapshot look is
the "if cheap" bracketing. Findings in §5.

## 2. Verdict lines (as run; generation tails trimmed)

```
VERDICT S1  [fact/fav_show]      gate=stub-tier1: PROMOTE | top_resp_frac 0.20 n_triples 5 train loss 0.534->-7.892  | margin +1.718 -> +4.451
VERDICT S2  [style/math_words]   gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> fav_show) train loss 6.527->2.127  | margin -3.374 -> +1.863 | word-digit -7.337 -> +5.183
VERDICT S3  [fact/tea_tin]       gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> math_words) train loss 7.192->5.371  | margin -1.266 -> +12.042
VERDICT S4  [style/metric_units] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> tea_tin) train loss 12.232->0.463  | margin -1.121 -> +1.178
VERDICT S5  [fact/pet_dog]       gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> metric_units) train loss 6.848->2.297  | margin +0.264 -> +5.162
VERDICT S6  [style/greet_rule]   gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> pet_dog) train loss 6.501->0.779  | margin -2.570 -> +3.468
VERDICT TRIGGER probe after S6: QUEUE tea_tin r=0.324, metric_units r=0.249, pet_dog r=0.153 -> queue [pet_dog, metric_units, tea_tin]
VERDICT S7  [trigger/pet_dog]    gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> pet_dog) train loss 8.935->0.218
VERDICT EVENT [re-teach/pet_dog] SERVED tripped after S6 r=0.153 -> served S7 v1 | r 0.153 -> 0.518 (repair +0.365)
VERDICT TRIGGER probe after S7: QUEUE fav_show r=0.258, greet_rule r=0.229 -> queue [greet_rule, metric_units, fav_show, tea_tin]
VERDICT S8  [trigger/greet_rule] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> greet_rule) train loss 3.644->0.164
VERDICT EVENT [re-teach/greet_rule] SERVED tripped after S7 r=0.229 -> served S8 v0 | r 0.229 -> 0.815 (repair +0.586)
VERDICT TRIGGER probe after S8: QUEUE math_words r=0.197, pet_dog r=0.393 -> queue [math_words, metric_units, fav_show, tea_tin, pet_dog]
VERDICT S9  [trigger/math_words] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> math_words) train loss 4.198->0.346
VERDICT EVENT [re-teach/math_words] SERVED tripped after S8 r=0.197 -> served S9 v1 | r 0.197 -> 0.637 (repair +0.440)
VERDICT TRIGGER probe after S9: QUEUE greet_rule r=0.476 -> queue [metric_units, fav_show, tea_tin, pet_dog, greet_rule]
VERDICT S10 [trigger/metric_units] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> metric_units) train loss 3.323->0.233
VERDICT EVENT [re-teach/metric_units] SERVED tripped after S6 r=0.249 -> served S10 v1 | r 0.186 -> -0.139 (repair -0.325)
VERDICT TRIGGER probe after S10: QUEUE metric_units r=-0.139 -> queue [metric_units, fav_show, tea_tin, pet_dog, greet_rule]
VERDICT S11 [trigger/metric_units] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> metric_units) train loss 2.630->0.139
VERDICT EVENT [re-teach/metric_units] SERVED tripped after S10 r=-0.139 -> served S11 v2 | r -0.139 -> 1.113 (repair +1.252)
VERDICT TRIGGER probe after S11: QUEUE (none new; 4 already pending)
VERDICT S12 [trigger/fav_show]   gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (re-teach 2, 29% -> fav_show) train loss 3.844->0.179
VERDICT EVENT [re-teach/fav_show] SERVED tripped after S7 r=0.258 -> served S12 v1 | r 0.606 -> 0.959 (repair +0.353)
VERDICT S13 [poison/banana-batch] gate=stub-tier1: ROLLBACK — target collapse: one response is 100% of the batch | incumbent adapter intact: True
VERDICT MERGE [accumulated adapter vs raw base] gate=immune-tier2: ROLLBACK — protected probe fact-paris regressed 1.000 -> 0.000 | protected probe fact-tokyo regressed 1.000 -> 0.000 | aggregate 0.3156 -> 0.0740, drift 1.000 label=fail
```

Gate behaviour: **12/12 promotes, 0 tripwire events** (top_response_fraction
0.14 everywhere after S1; unique_response_ratio 1.0 — all re-teach strings
fresh). The poison canary tripped the collapse rule at 100% and the
incumbent stayed byte-untouched. The one-fallback policy was **not**
exercised; `lab/FAILURES.md` gains no entry.

## 3. Trigger event log

Nine trips across six probes, six served re-teach events; three trips still
queued at the horizon (tea_tin tripped after S6 — **never served**; pet_dog
re-tripped after S8; greet_rule re-tripped after S9).

| # | item | tripped after | r at trip | served in | variant | r before serve → after | repair Δr |
|---|---|---|---|---|---|---|---|
| 1 | pet_dog | S6 | 0.153 | S7 | v1 | 0.153 → 0.518 | **+0.365** |
| 2 | greet_rule | S7 | 0.229 | S8 | v0 | 0.229 → 0.815 | **+0.586** |
| 3 | math_words | S8 | 0.197 | S9 | v1 | 0.197 → 0.637 | **+0.440** |
| 4 | metric_units | S6 | 0.249 | S10 | v1 | 0.186 → −0.139 | **−0.325** |
| 5 | metric_units | S10 | −0.139 | S11 | v2 | −0.139 → 1.113 | **+1.252** |
| 6 | fav_show | S7 | 0.258 | S12 | v1 | 0.606 → 0.959 | **+0.353** |

Per-event repair fraction: **5/6 positive** (mean +0.445, range −0.325 ..
+1.252). The single null repair is instructive, not silent: metric_units'
first re-teach (variant 1) *reduced* its margin by 0.3 of a gain while
still re-exposing the item inside a 7-triple batch — same interference the
re-teach was meant to counter, turned against a weaker item; its second
re-teach (variant 2, next session) delivered +1.252. Repair yield per
contact is **highly variant-dependent**, not a constant.

Cost of holding: 6 re-teach turns across the 6 later sessions (every
trigger-phase session carried exactly one re-teach = the cadence arm's
batch shape exactly; total consolidated turn count identical to cadence).

## 4. Retention r_i(k) — trigger arm vs cadence (committed numbers)

Baselines bit-identical to cadence by CADENCE-IDENT. g_i (teach gains):
fav_show +2.733, math_words +5.237, tea_tin +13.308, metric_units +2.298,
pet_dog +4.898, greet_rule +6.038 — identical to the cadence table.

Full r_i(k) (trigger arm; **bold** = session where the item was re-taught;
`|` = teach anchor):

| item | g_i | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fav_show (F) | +2.73 | +4.45 \| | **+6.87 (1.89)** | +4.43 (0.99) | +7.93 (2.27) | −6.16 (−2.88) | +5.16 (1.26) | +2.42 (0.26) | +2.92 (0.44) | +5.55 (1.40) | +3.97 (0.82) | +3.37 (0.61) | **+4.34 (0.96)** |
| math_words (S) | +5.24 | — | +1.86 \| | **+2.14 (1.05)** | +0.50 (0.74) | +6.52 (1.89) | +3.06 (1.23) | +1.44 (0.92) | −2.34 (0.20) | **−0.04 (0.64)** | +1.98 (1.02) | +0.98 (0.83) | +0.09 (0.66) |
| tea_tin (F) | +13.31 | — | — | +12.04 \| | **−0.59 (0.05)** | −2.65 (−0.10) | +3.05 (0.32) | −0.09 (0.09) | −0.50 (0.06) | +1.28 (0.19) | −0.47 (0.06) | −0.05 (0.09) | +1.04 (0.17) |
| metric_units (S) | +2.30 | — | — | — | +1.18 \| | **+0.48 (0.70)** | −0.55 (0.25) | −0.67 (0.20) | −1.70 (−0.25) | −0.69 (0.19) | **−1.44 (−0.14)** | **+1.44 (1.11)** | +0.47 (0.69) |
| pet_dog (F) | +4.90 | — | — | — | — | +5.16 \| | **+1.02 (0.15)** | **+2.80 (0.52)** | +2.19 (0.39) | +3.07 (0.57) | +4.06 (0.77) | +0.97 (0.14) | +2.71 (0.50) |
| greet_rule (S) | +6.04 | — | — | — | — | — | +3.47 \| | **−1.19 (0.23)** | **+2.35 (0.81)** | +0.31 (0.48) | +0.16 (0.45) | −0.03 (0.42) | +0.46 (0.50) |

Note: S1..S6 numbers are the cadence values (bit-identical); the trigger
phase starts at S7. pet_dog's **bold** at S6 is its cadence-scheduled
teach-phase replay (identical in both arms).

### Headline: horizon comparison

| item | kind | re-teach events | cadence r_final | trigger r_final | Δ |
|---|---|---|---|---|---|
| fav_show | fact | 1 (S12) | 0.77 | **0.96** | +0.19 |
| math_words | style | 1 (S9) | 0.79 | 0.66 | −0.13 |
| tea_tin | fact | **0 (starved in queue)** | 0.01 | 0.17 | +0.16 |
| metric_units | style | 2 (S10, S11) | 0.06 | **0.69** | +0.63 |
| pet_dog | fact | 1 (S7) | 0.52 | 0.50 | −0.02 |
| greet_rule | style | 1 (S8) | 0.52 | 0.50 | −0.02 |

| metric | cadence | trigger | read |
|---|---|---|---|
| items r_final ≥ 0.50 | 4/6 (all ≥ 0.52) | 4/6 (0.96, 0.69, 0.66, 0.50; pet_dog 0.4996 just under) | **tie** |
| dead slots r_final ≤ 0.06 | **2/6** (tea_tin 0.01, metric_units 0.06) | **0/6** (worst 0.17) | **trigger wins** |
| mean r_final | 0.445 | **0.581** | trigger +0.14 |
| min r_final | 0.01 | **0.17** | trigger wins |

The honest headline: the trigger arm **erases the dead-slot failure mode**
(0/6 vs 2/6) and lifts the mean and the floor, but it does **not** raise
the count of items held above threshold (4/6 both; pet_dog lands 0.4996,
a rounding-margin under the line, after being re-damaged by S11's
metric_units re-teach and starved out of a second serve). "Beat the
cadence or say you can't": **beat on floor/mean/dead-slots; tie on
held-count.**

Time-alive curves from (first) trigger to end, per served item:

| item | tripped after (r) | served | r(k) service → S12 |
|---|---|---|---|
| pet_dog | S6 (0.153) | S7 | 0.52 → 0.39 → 0.57 → 0.77 → 0.14 → 0.50 (re-damaged at S11, re-tripped, starved) |
| greet_rule | S7 (0.229) | S8 | 0.81 → 0.48 → 0.45 → 0.42 → 0.50 (held, shallow decline) |
| math_words | S8 (0.197) | S9 | 0.64 → 1.02 → 0.83 → 0.66 (held, oscillates up first) |
| metric_units | S6 (0.249) | S10, S11 | −0.14 → 1.11 → 0.69 (null first contact, strong second) |
| fav_show | S7 (0.258) | S12 | 0.61 → 0.96 (recovered on its own 0.26→1.40→0.61 before its serve) |
| tea_tin | S6 (0.324) | **never** | 0.09 → 0.06 → 0.19 → 0.06 → 0.09 → 0.17 (un-served drift, ends the run's floor) |

## 5. Threshold sensitivity (cheap bracketing look)

From the recorded probes, the item sets that would have tripped at each
bracket (de-duplicated new arrivals per probe):

| probe after | trips @0.25 | trips @0.50 (run) | trips @0.75 |
|---|---|---|---|
| S6 | metric_units, pet_dog (2) | + tea_tin (3) | = 0.50 (3) |
| S7 | tea_tin, greet_rule (2) | fav_show, greet_rule (2) | 5 in queue-range |
| S8 | math_words (1) | math_words, pet_dog (2) | superset of 0.50 |
| S9–S11 | 0–1 new | greet_rule, metric_units, – (0–2) | superset |

Arrival-vs-service accounting (service cap = 1 re-teach/session):

- **@0.50 (as run): 9 arrivals vs 6 slots** — the queue never drains;
  lowest-r-first is fair to depth but not to age, so tea_tin (tripped
  first, at 0.324) was repeatedly pre-empted by deeper-damaged newcomers
  and **never served in 6 sessions**.
- **@0.25: ≈6 arrivals vs 6 slots** — capacity-matched in this realisation;
  and its queue would have put tea_tin **second** (r 0.089 behind nothing
  else), so a *lower* threshold would plausibly have served the item the
  higher one starved. Counter-intuitive but mechanical: the threshold is
  also the service-rate throttle.
- **@0.75: ~12 arrivals** — strictly worse starvation; nothing learned per
  extra event except deeper backlog.

The verdict is threshold-insensitive in its **sign** (dead slots gone,
held-count ~4/6 under any bracket that serves ≥ the 0.25 set — 0.25 changes
*who* gets served, not whether the loop works), but the **queue scheduler
is the real design surface**: threshold alone is not a scheduler. Next
iteration needs age-weighted priority (or FIFO among sub-threshold items)
so that first-tripped items cannot starve behind a stream of deeper
interference.

## 6. Tier-2 merge verdict

| arm (context) | aggregate inc → cand | protected agg | protected regressions | drift | decision |
|---|---|---|---|---|---|
| cadence | 0.3156 → 0.0801 | 0.255 → 0.030 | fact-paris 1→0, fact-tokyo 1→0 (same family) | 1.000 (fail) | ROLLBACK |
| trigger (this) | **0.3156 → 0.0740** | 0.255 → 0.043 | fact-paris 1→0, fact-tokyo 1→0, refuse-color-seven 0.408→0.324, ho-refuse-weigh 0.397→0.150 | **1.000 (fail)** | **ROLLBACK** |

Expected ROLLBACK, confirmed: 12 promoted micro-sessions (6 identical
teaches + 6 re-teach sessions) zero the same protected chat facts when the
accumulated adapter is merged into the trunk. The two-tier read holds a
fourth time; trigger-loop scheduling does not change merge-grade behaviour
and was not expected to.

## 7. Claims and limits

**Claims (what the curves support).**

1. *The reactive loop is mechanically sound and bit-anchored.* Calibrations
   bit-identical twice and matching the cadence seed-7 calibration; teach
   phase S1..S6 bit-identical to the committed cadence run (max |Δ| = 0.0);
   12/12 promotes, top-response fraction 0.14, unique ratio 1.0, poison
   caught, incumbent intact, zero fallback events.
2. *Triggered next-session re-teaching rehabilitates damaged items.*
   5/6 events positive (mean +0.445, up to +1.252); every item that was
   served at least once ends r_final ≥ 0.50 (pet_dog 0.4996 is the edge
   case after re-damage at S11); the one item never served (tea_tin)
   supplies the run's only sub-threshold finish.
3. *The cadence's dead-slot failure mode is gone.* 0/6 r_final ≤ 0.06
   (cadence: 2/6, at 0.01/0.06). Mean r_final 0.581 vs 0.445. Held-count
   ties at 4/6 — claimed floor improvement, not a held-count win.
4. *Near-dead items are rehabilitable; truly-dead rehabilitation is
   untested.* metric_units came back from r = −0.25 / −0.14 (below the
   cadence's lowest observed live value for that item) to 0.69 on two
   consecutive re-teaches. The strict question "is an r ≤ 0.06 dead slot
   rehabilitable" was NOT exercised — tea_tin lived in the 0.06–0.19 band
   for five probes and was starved before ever being served dead. Honest
   answer to the cadence's implication: the evidence leans rehabilitable
   (metric_units), but the cadence's exact dead state (tea_tin) never got a
   contact; cannot claim it.
5. *The threshold is also a service-rate throttle; queue scheduling is the
   bottleneck.* 9 trips vs 6 slots ⇒ starvation of the first-tripped item
   (tea_tin) under lowest-r-first. Bracketing shows @0.25 is
   capacity-matched here and would have served tea_tin; @0.75 deepens the
   backlog. A threshold gate is not a scheduler — age-weighted priority is
   the required next component.
6. *Tier 2 unchanged.* ROLLBACK (0.3156 → 0.0740, drift 1.000 fail), as
   expected; the two-tier read holds a fourth time.

**Overall verdict: reactive repair works per contact; the static scheduler
under-serves the backlog.** Beat cadence on floor, mean and dead-slots;
tied on held-count; the loop is adoptable as the live memory-manager
prototype on the condition that serving gains aged-priority (otherwise
first-tripped items starve under realistic interference rates).

**Limits.**

- Single run under the pinned seed (n=1; replicate-grade determinism makes
  variance *only* seed choice, but this arm is one draw of it).
- The trigger phase is confounded with the cadence's *weaker* interference
  (filler approvals + one re-teach, vs teach-phase corrections) — same
  caveat as cadence §5; the within-curve contacts (§3 events) are the
  readable signal, not cross-phase slopes.
- Per-contact repair shows high variant dependence (same item: −0.325 then
  +1.252 on consecutive variants) — repair yield is not a stable constant
  of the item.
- Dead-slot rehabilitation claimed only for the near-dead band (r ≥ −0.25);
  strict r ≤ 0.06 rehabilitation untested (§claims 4).
- Bracketing is counterfactual-from-snapshots: it says which items would
  have been queued, not what margins a 0.25/0.75 re-run would produce
  (different serves ⇒ different trajectories).
- Free-run generations remain transcript colour; margin probes are the
  mechanism read.

## Reproduction

```bash
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_trigger_repair.py --workdir lab/teach_trigger_out \
  | tee lab/logs/teach_trigger.log

# determinism calibration (2 runs, both must bit-match each other and the
# cadence harness's committed seed-7 calibration):
python3 -u lab/teach_trigger_repair.py --only metric_units --seed 7 --workdir <fresh A>
python3 -u lab/teach_trigger_repair.py --only metric_units --seed 7 --workdir <fresh B>
```

Fresh workdir required (append-only session logs, invariant 10; adapter
deltas and `lab/teach_trigger_out/`, `lab/teach_trigger_calib/` are weights
and are not committed).
