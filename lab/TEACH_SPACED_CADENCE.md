# TEACH_SPACED_CADENCE — repeating spaced-repetition cadence on the SFT-r3 chat checkpoint

The queued next arm after `lab/TEACH_DECAY_REPLAY.md` (queued experiment #3
follow-up in `lab/TEACH_PIPELINE.md`), run 2026-10-19 on the laptop, CPU
only, fp32. The replay A/B measured that ONE phrasing-diverse replay one
session after teach prevents catastrophic single-session wipeouts (4/4
direct contrasts in replay's favour) but the benefit re-decays within 1-2
sessions: refresh, not anneal. This experiment (i) fixes the two harness
flaws the A/B surfaced — unseeded CPU train-loop RNG, and the
uninterpretable 1-token `lowercase` probe — and (ii) runs the **repeating
spaced cadence**: every taught item is re-exposed once per later-session
cycle, phrasing-diverse, ≤30% of the batch, for 6 sessions after the last
full teach (annealing test).

Harness: `lab/teach_spaced_cadence.py` (extends `lab/teach_decay_replay.py`
— library, session packs, replay pools, loaders, gates and formats are
imported, not copied). Numbers-only results:
`lab/teach_cadence_out/teach_spaced_cadence.json`; logs
`lab/logs/teach_cadence.log`; determinism calibration JSONs
`lab/teach_calib/`.

## 1. Harness hardening (verification first)

### 1a. Seeded train-loop RNG — root cause and fix

The A/B's noise-floor section documented a few-nats run-to-run spread on
identical inputs. Root cause located in `Hippocampus/lora.py:
LoRALinear.__init__`: the A matrix is drawn from the **unseeded global**
`torch.randn`, while the batch sampler in
`train_lora_microsession` was already locally seeded
(`torch.Generator().manual_seed(0)`). Because the B branch is zero at
init, loss0 is bit-identical across runs even with different A; the
trajectories diverge only once B trains away from zero — exactly the
signature the A/B recorded (loss0 2.060 in all sessions, post-teach
margins anywhere in a ±2-nat band).

Fix, minimal and parameterised: `consolidate.consolidate()` gained a
`seed: int | None` parameter (default `None` = old behaviour). When set,
it calls `torch.manual_seed(seed)` immediately before `attach_lora`
(anchoring the A-init draws and any other global-RNG consumers up to the
train loop) and passes the seed through to `train_lora_microsession`'s
batch sampler. No other API or math changed; gates, probes, loaders
untouched. Incumbent A/B weights are loaded from the delta after attach
(`load_delta` restores both matrices), so per-session re-seeding is
idempotent for later sessions.

**Verification** (`--only metric_units` calibration runs, identical single
teach session, 24 steps, fresh workdirs):

_BEFORE (seed off, pre-fix code path):_

| run | baseline margin | post-teach margin | gain | loss0 |
|---|---|---|---|---|
| unseeded A | −1.120889 | **+4.047320** | +5.168 | 2.179 |
| unseeded B | −1.120889 | **+1.939237** | +3.060 | 2.179 |

Bit-identical baseline and loss0, post-teach margins differ by **2.108
nats** — the measured pre-fix spread, consistent with the A/B's ±few-nats
noise band.

_AFTER (seed 7):_

| run | baseline margin | post-teach margin | gain | loss0 |
|---|---|---|---|---|
| seed 7 A | −1.120889 | **−0.146229** | +0.975 | 2.179 |
| seed 7 B | −1.120889 | **−0.146229** | +0.975 | 2.179 |

Post-teach margins **bit-identical** across the two seeded replays
(−0.146229 == −0.146229), and the adapter weights are bit-equal across
processes (`delta.pt` tensors `torch.equal` for every entry; the `.pt`
files differ only in the `created` timestamp inside the meta dict). The
train loop is now replicate-grade deterministic on this box: the few-nats
spread is gone, only seed choice remains (seed fixes *one* LoRA-A init —
a different seed is a different draw, not noise).

### 1b. Multi-token style probe replacing `lowercase`

The A/B's `lowercase` probe contrasted ` yes.` vs ` Yes.` — one
capitalization token, baseline +0.174, teach gains 0.2-0.5 nats, so its
r_i(k) blew up on noise (formally −8.2 in arm R). Replacement:
`metric_units` ("always use metric units"), a whole-phrase contrast:

- probe: `[chat user: How long is the hallway table?]` + `It is about`
- target ` two metres long.`  vs  decoy ` six feet long.`  (3-vs-3 tokens)

Calibration gain on the same 5-triple teach shape as the library: **+3.06
and +5.17 nats** on the two unseeded runs, **+0.97 nats** under the pinned
seed 7 draw — all at or above the ≳0.5-1-nats bar, and the spread across
inits is now visible for what it is (LoRA-A draw variance), unlike the
1-token probe where the gain itself was below the noise floor. r_i(k) for
this item is meaningful.

## 2. Cadence arm — design

| item | value |
|---|---|
| checkpoint | `lab/imported/vk_sft_r3/step_1800/chat_model` — identical to E3 / the A/B |
| runtime | laptop CPU, fp32, 12 threads, 561M kda/mla; tier-2 Immune never leaves CPU (I7); LoRA rank 8, alpha 16, lr 2e-3, kl_coef 0.05, 24 steps/session — identical knobs |
| seed | 7 on every consolidate call (hardening §1a) |
| library | the A/B's 5 working items — fav_show (fact), math_words (style), tea_tin (fact), pet_dog (fact), greet_rule (style), with measured baselines from E3/A/B — plus `metric_units` (style) replacing `lowercase` in slot 4; alternation preserved |
| teach phase | S1–S6, one item each, arm-R shape: 5-triple teach batch + ONE replay turn of the previous session's item (2 triples, 28.6% of batch) |
| anneal phase | S7–S12 (6 sessions ≥ the ≥6 required): 5 generic-chat approval triples (interfering "normal use") + ONE replay turn re-exposing one earlier-taught item, round-robin ORDER → every item re-exposed once per cycle; replay share 2/7 = 28.6% |
| tripwire safety | all response strings unique within every batch (unique_response_ratio 1.0, top_response_fraction ≤ 2/7 — the FAILURES.md collapse family stays cold); banana poison canary at the end |
| tier 2 | ONE Immune merge verdict on the accumulated adapter at the end, constants untouched |
| baseline for the contrast | the A/B's arm-R r_i(k) table (single-replay), reused as-is — **not** re-run |

Re-exposure count per item in the cadence arm: two total for items 1-5
(once at teach-side, once in anneal), one for greet_rule (fine — it is the
last-taught anchor). Every replay turn draws a fresh variant from the
phrasing pools; no response string repeats anywhere in the run.

## 3. Verdict lines (as run; generation tails trimmed)

```
VERDICT S1  [fact/fav_show]      gate=stub-tier1: PROMOTE | top_resp_frac 0.20 n_triples 5 train loss 0.534->-7.892  | margin +1.718 -> +4.451
VERDICT S2  [style/math_words]   gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> fav_show) train loss 6.527->2.127  | margin -3.374 -> +1.863 | word-digit -7.337 -> +5.183
VERDICT S3  [fact/tea_tin]       gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> math_words) train loss 7.192->5.371  | margin -1.266 -> +12.042
VERDICT S4  [style/metric_units] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> tea_tin) train loss 12.232->0.463  | margin -1.121 -> +1.178
VERDICT S5  [fact/pet_dog]       gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> metric_units) train loss 6.848->2.297  | margin +0.264 -> +5.162
VERDICT S6  [style/greet_rule]   gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> pet_dog) train loss 6.501->0.779  | margin -2.570 -> +3.468
VERDICT S7  [anneal/fav_show]    gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> fav_show) train loss 7.120->0.199
VERDICT S8  [anneal/math_words]  gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> math_words) train loss 3.315->0.249
VERDICT S9  [anneal/tea_tin]     gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> tea_tin) train loss 2.983->0.176
VERDICT S10 [anneal/metric_units] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> metric_units) train loss 3.500->0.175
VERDICT S11 [anneal/pet_dog]     gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> pet_dog) train loss 3.636->0.172
VERDICT S12 [anneal/greet_rule]  gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> greet_rule) train loss 3.154->0.146
VERDICT S13 [poison/banana-batch] gate=stub-tier1: ROLLBACK — target collapse: one response is 100% of the batch | incumbent adapter intact: True
VERDICT MERGE [accumulated adapter vs raw base] gate=immune-tier2: ROLLBACK — protected probe fact-paris regressed 1.000 -> 0.000 | protected probe fact-tokyo regressed 1.000 -> 0.000 | aggregate 0.3156 -> 0.0801, drift 1.000 label=fail
```

Gate behaviour: **12/12 promotes, 0 tripwire events** (top_response_fraction
0.14 everywhere after S1; unique_response_ratio 1.0). The banana poison
canary tripped the collapse rule at 100% and the incumbent stayed
byte-untouched. The one-fallback policy was **not** exercised — no session
needed a gate-safe spaced re-teach, and `lab/FAILURES.md` gains no entry.

## 4. Retention r_i(k) through the annealing phase

Baselines: fav_show +1.718, math_words −3.374 (word-digit −7.337),
tea_tin −1.266, pet_dog +0.264, greet_rule −2.570 — all bit-identical to
the E3/A-B anchors. metric_units −1.121 (new; prior mildly imperial).

`r_i(k) = (m_i(k) − m_i(0)) / g_i`; cells are `margin (r)`. **Bold** =
session where that item was replayed (teach phase: previous item; anneal
phase S7–S12: round-robin).

| item | g_i | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fav_show (F) | +2.73 | +4.45 \| | **+6.87 (1.89)** | +4.43 (0.99) | +7.93 (2.27) | −6.16 (−2.88) | +5.16 (1.26) | **+3.47 (0.64)** | +3.12 (0.51) | +3.53 (0.66) | +3.50 (0.65) | +4.02 (0.84) | +3.81 (0.77) |
| math_words (S) | +5.24 | — | +1.86 \| | **+2.14 (1.05)** | +0.50 (0.74) | +6.52 (1.89) | +3.06 (1.23) | +1.88 (1.00) | **+0.14 (0.67)** | −1.39 (0.38) | +0.32 (0.71) | −2.20 (0.22) | +0.77 (0.79) |
| tea_tin (F) | +13.31 | — | — | +12.04 \| | **−0.59 (0.05)** | −2.65 (−0.10) | +3.05 (0.32) | −0.23 (0.08) | −1.35 (−0.01) | **+0.70 (0.15)** | +0.25 (0.11) | +1.25 (0.19) | −1.07 (0.01) |
| metric_units (S) | +2.30 | — | — | — | +1.18 \| | **+0.48 (0.70)** | −0.55 (0.25) | −1.24 (−0.05) | −0.51 (0.27) | −2.11 (−0.43) | **−1.79 (−0.29)** | −0.57 (0.24) | −0.98 (0.06) |
| pet_dog (F) | +4.90 | — | — | — | — | +5.16 \| | **+1.02 (0.15)** | +1.33 (0.22) | +3.58 (0.68) | +1.06 (0.16) | +1.48 (0.25) | **+2.29 (0.41)** | +2.82 (0.52) |
| greet_rule (S) | +6.04 | — | — | — | — | — | +3.47 \| | −0.36 (0.37) | −2.85 (−0.05) | −2.09 (0.08) | −3.41 (−0.14) | −2.74 (−0.03) | **+0.56 (0.52)** |

The catastrophic wipeout mechanism is **not** gone: `fav_show` lost 14.1
nats during S5 (pet_dog's teach, which replayed metric_units), and
`tea_tin` lost 12.6 nats during S4 one session after a +12.0-nat acquisition.
Both events happened inside the teach phase, between the two exposures
their cyclic replay would eventually deliver — the once-per-6-session
cycle arrives **after** the damage, and repair is partial-to-absent.

## 5. Cadence vs single-replay (the designed contrast)

Single-replay baseline = the A/B's arm-R r_i(k) (reused, not re-run). The
A/B's documented failure mode: after its one replay, an item re-decayed
within 1-2 sessions (fav_show 0.62→0.28 over S5; tea_tin 1.84→−0.10 over
S6).

| item | arm R (single replay): post-replay course | cadence: cyclic-replay course | anneal? |
|---|---|---|---|
| fav_show | replay S2: r 0.62 → 0.26/0.42/0.28/0.36 by S6 (re-decay) | replay S7, then r 0.51–0.84 for 5 more sessions, no monotone drop, r_final 0.77 | **y** (held) |
| math_words | replay S3: 0.17 → −2.60 → 0.34 → 1.24 (oscillates) | replay S8; oscillating band r 0.22–1.00, r_final 0.79 | weak y (oscillates, no trend) |
| tea_tin | replay S4: 1.84 → −0.10 (S6): re-decay in 2 sessions | replay S9 lifts r only 0.08→0.15; re-decays to 0.01 by S12 | **n** (not rehabilitated) |
| metric_units | (replaces uninterpretable lowercase) | replay S10 lifts r −0.43→−0.29; ends 0.06 | **n** (dead before its cycle slot) |
| pet_dog | replay S6: r +2.50 (horizon ends) | replay S11 lifts 0.25→0.41; ends 0.52 | partial (repair-on-contact) |
| greet_rule | teach-anchored r 1.00 at S6 | replay S12 lifts r −0.03→0.52 in one shot | partial (single contact, can't separate refresh from anneal) |

Anneal-phase slopes (mean margin change per session, S after teach+1): fact
class −0.072 nats/sess, style −0.092 nats/sess, against teach-phase µ_d of
−0.60/−0.26 — but the anneal phase runs on weaker interference (generic
approvals + one replay) than the teach phase (fresh corrections), so the
smaller slope **cannot be credited to the cadence itself**. Fitted
half-lives remain not fittable (curves oscillate, same caveat as the A/B).

## 6. Tier-2 merge verdict

| arm (context) | aggregate inc → cand | protected agg | protected regressions | drift | decision |
|---|---|---|---|---|---|
| A/B arm N | 0.3156 → 0.0934 | 0.255 → 0.037 | fact-paris, fact-tokyo (+3 more) | 1.000 (fail) | ROLLBACK |
| A/B arm R | 0.3156 → 0.1050 | 0.255 → 0.032 | same set | 1.000 (fail) | ROLLBACK |
| cadence (this) | **0.3156 → 0.0801** | 0.255 → 0.030 | fact-paris 1→0, fact-tokyo 1→0, same family | **1.000 (fail)** | **ROLLBACK** |

Cadence **does not** change the merge verdict, as expected: 12 promoted
micro-sessions (6 teaches + 6 anneal) still zero the same protected chat
facts when the accumulated adapter is merged into the trunk. The two-tier
read holds a third time, now across 13 consolidated sessions.

## 7. Claims and limits

**Claims (what the curves support).**

1. *Harness hardening works.* Seeded train loop ⇒ bit-identical post-teach
   margins (and bit-equal adapter tensors) across replicated runs; the
   unseeded A-init spread measured here at 2.1 nats on an identical teach
   (§1a). The multi-token `metric_units` probe acquired +2.30 nats in the
   full run (g_i in the table above), interpretable ratio space — the
   1-token `lowercase` probe is retired (§1b).
2. *The repeating cadence is gate-free.* 12/12 promotes, top-response
   fraction 0.14 (as good as the replay arm's), unique ratio 1.0, poison
   still caught with the incumbent intact, zero fallback events.
3. *Repair-on-contact is real for items still alive at their slot.*
   greet_rule's cyclic replay lifted r from −0.03 to 0.52 in one session;
   pet_dog climbs 0.16→0.52 across/after its S11 contact; fav_show, still
   at r 1.26 going into the anneal phase, held an r 0.51–0.84 band for the
   5 sessions after its S7 re-exposure (contrast: arm R's same item slid
   0.62→0.28 in the 3 sessions after its one replay). 4/6 items end at
   r ≥ 0.52.
4. *The cyclic replay does not anneal items that are already dead when
   their slot arrives.* tea_tin (r 0.05 by its S9 slot) and metric_units
   (r −0.43 by its S10 slot) got only weak, transient lifts; both end at
   r ≤ 0.06. Once-per-6-sessions is too sparse to defend a weak item
   against one catastrophic teach-phase wipeout.

**Overall verdict: refresh-with-repair, not annealing.** No measurable
maintenance regime emerged at this re-exposure density; the failure
profile moved from arm-R's "re-decay within 1–2 sessions" to "oscillating
band for the strong items, unrecovered death for the weak ones", and the
catastrophic teach-phase wipeouts (the real killer) are untouched by the
cadence. A denser cycle (every-2-sessions), retain-the-wipeout-trigger
re-teaching, or graduated amnesia probes are the honest next arms.

**Limits.**

- Single run under the pinned seed (determinism now makes *replication*
  cheap, but n=1 remained; seed choice is now the only variance and is
  itself a design variable).
- Seed 7's LoRA-A draw gave metric_units a smaller single-shot gain
  (+0.97) than unseeded draws (+3.1, +5.2) — gains are init-dependent;
  the full-run gain (+2.30) is the number in the library.
- The anneal phase's interference is weaker than the teach phase's
  (approvals vs corrections), so class-slope comparisons across phases
  are confounded; the within-class post-replay contractions (§5) are the
  readable signal.
- Horizons differ vs the A/B (12 vs 6 sessions); arm-R numbers are reused
  for shape of the post-replay course, not for matched-index r_final
  ranking — mean r_final (cadence: facts 0.43, style 0.46) is over 12
  sessions and should not be compared cell-wise to arm-R's 6-session
  means.
- Free-run generations remain transcript colour (degeneration bars after
  the turn-close marker rendered as text); margin probes are the
  mechanism read.

## Reproduction

```bash
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_spaced_cadence.py --workdir lab/teach_cadence_out \
  | tee lab/logs/teach_cadence.log

# determinism / probe calibration (4 runs as reported in §1)
python3 -u lab/teach_spaced_cadence.py --only metric_units --seed -1 --workdir lab/teach_calib/A
python3 -u lab/teach_spaced_cadence.py --only metric_units --seed  7 --workdir lab/teach_calib/B
```

Fresh workdir required (append-only session logs, invariant 10; adapter
deltas and `lab/teach_cadence_out/`, `lab/teach_calib/` are weights and are
not committed).
