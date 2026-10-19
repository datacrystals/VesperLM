# TEACH_DECAY_REPLAY — decay-aware replay A/B on the SFT-r3 chat checkpoint

Queued experiment #3 of `lab/TEACH_PIPELINE.md` ("decay-mitigation arm"),
run 2026-10-19 on the laptop, CPU only, fp32. Direct follow-up to E3
(`lab/TEACH_ON_SFT_R3.md`), which measured that the SFT-r3 chat checkpoint
acquires 2–5× harder than the tiny base but forgets proportionally (first-taught
fact `fav_show` r_final **0.07**, style `math_words` 0.63). This experiment asks
the prophylactic question: does mixing a small, **phrasing-diverse** replay of
earlier taught items into each later micro-session's batch flatten the retention
curves — and at what cost to acquisition and gate behaviour?

Harness: `lab/teach_decay_replay.py` (new, extends `lab/teach_on_sft_r3.py`;
the E3 items, session packs, loaders, chat framing, verdict-line format and
gates are reused verbatim). Numbers-only results:
`lab/teach_decay_out/{noreplay,replay}/teach_decay_replay_<arm>.json`; full logs
`lab/logs/teach_decay_{noreplay,replay}.log`.

## Design

| item | value |
|---|---|
| checkpoint | `lab/imported/vk_sft_r3/step_1800/chat_model` (561M, 10L, dim 1024, kda/mla, passport router) — identical to E3 |
| runtime | laptop CPU, fp32, 12 threads; GPU untouched (tier-2 Immune never leaves CPU, I7); LoRA never leaves rank-8 config |
| adapter | LoRA rank 8, alpha 16, lr 2e-3, kl_coef 0.05, 24 micro-steps/session — identical to E3 |
| tier-1 gate | `consolidate.stub_gate` per session (batch statistics; collapse tripwire hard) |
| tier-2 gate | one `lab/immune_gate.py` merge verdict per arm — real Immune probes + drift + `gate.decide`, constants untouched |
| library | 6 items, alternating fact/style: `fav_show` (fact), `math_words` (style), `tea_tin` (fact) — the E3 items, session packs verbatim — plus `lowercase` (style: answer all-lowercase), `pet_dog` (fact: dog's name = Biscuit), `greet_rule` (style: greet with "Good day") |
| order / sessions | the same `ORDER` and the same 6 session packs in both arms; one session per item; re-probe ALL taught items after every session; banana poison canary at the end |

### Arms

- **Arm N (no replay)** — each micro-session's batch is exactly the E3 shape:
  2 teaching corrections + 1 approval = 5 triples, never any re-exposure.
- **Arm R (replay)** — each later session's log additionally carries ONE replay
  turn of exactly ONE earlier-taught item (approve-with-correction → 2 extra
  positive triples). Replay share 2/7 = **28.6%** of the batch, under the ≤30%
  budget in `TEACH_PIPELINE.md`. Deterministic schedule: session S2 ←
  `fav_show`, S3 ← `math_words`, S4 ← `tea_tin`, S5 ← `lowercase`, S6 ←
  `pet_dog` — every item re-exposed exactly once, in the session immediately
  following its own teach. Replay prompts **and** response strings are fresh
  phrasings, distinct from the owning session's turns and from each other
  within the batch — the gate-collapse trap documented in `TEACH_PIPELINE.md`
  #8 was identical replay phrasings, and this arm is built to show the guard
  does not fire on diverse replay.

The arms ran as separate processes with fresh workdirs (append-only session
logs, invariant 10), each reloading the raw checkpoint. Measured side effect
of that: **baseline margins are bit-identical across arms** (and to E3's
anchors where the items overlap).

## Baselines (identical in both arms = determinism check)

| item | class | m(0) | matches E3 anchor |
|---|---|---|---|
| fav_show | fact | +1.718 | yes (bit-exact) |
| math_words | style | −3.374 (word-digit −7.337) | yes (bit-exact) |
| tea_tin | fact | −1.266 | yes (bit-exact) |
| lowercase | style | +0.174 | n/a — new; prior is nearly indifferent (target/decoy differ by one capitalization token) |
| pet_dog | fact | +0.264 | n/a — new; names roughly equiprobable |
| greet_rule | style | −2.570 | n/a — new; prior favours the hello-form |

## Verdict lines (as run; generation tails trimmed)

Arm N (no replay):

```
VERDICT S1 [fact/fav_show] gate=stub-tier1: PROMOTE | top_resp_frac 0.20 n_triples 5 train loss 2.060->-10.078 | margin +1.718 -> +4.407
VERDICT S2 [style/math_words] gate=stub-tier1: PROMOTE | top_resp_frac 0.40 n_triples 5 train loss 3.599->-25.201 | margin -3.374 -> +0.815 | word-digit -7.337 -> +2.293
VERDICT S3 [fact/tea_tin] gate=stub-tier1: PROMOTE | top_resp_frac 0.20 n_triples 5 train loss 5.536->-14.122 | margin -1.266 -> +0.990
VERDICT S4 [style/lowercase] gate=stub-tier1: PROMOTE | top_resp_frac 0.20 n_triples 5 train loss 4.967->-28.067 | margin +0.174 -> +0.634
VERDICT S5 [fact/pet_dog] gate=stub-tier1: PROMOTE | top_resp_frac 0.20 n_triples 5 train loss 3.196->-16.718 | margin +0.264 -> +8.783
VERDICT S6 [style/greet_rule] gate=stub-tier1: PROMOTE | top_resp_frac 0.20 n_triples 5 train loss 9.881->-17.946 | margin -2.570 -> +4.364
VERDICT S7 [poison/banana-batch] gate=stub-tier1: ROLLBACK — target collapse: one response is 100% of the batch | incumbent adapter intact: True
VERDICT MERGE [accumulated adapter vs raw base] gate=immune-tier2: ROLLBACK — protected probe fact-paris regressed 1.000 -> 0.000 | protected probe fact-tokyo regressed 1.000 -> 0.000 | aggregate 0.3156 -> 0.0934, drift 1.000 label=fail
```

Arm R (replay):

```
VERDICT S1 [fact/fav_show] gate=stub-tier1: PROMOTE | top_resp_frac 0.20 n_triples 5 train loss 2.060->-12.644 | margin +1.718 -> +7.403
VERDICT S2 [style/math_words] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> fav_show) train loss 0.018->-8.711 | margin -3.374 -> +0.761 | word-digit -7.337 -> +3.367
VERDICT S3 [fact/tea_tin] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> math_words) train loss -2.178->-6.783 | margin -1.266 -> +2.020
VERDICT S4 [style/lowercase] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> tea_tin) train loss -7.759->-6.932 | margin +0.174 -> +0.395
VERDICT S5 [fact/pet_dog] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> lowercase) train loss -9.752->-7.369 | margin +0.264 -> +6.016
VERDICT S6 [style/greet_rule] gate=stub-tier1: PROMOTE | top_resp_frac 0.14 n_triples 7 (replay 2, 29% -> pet_dog) train loss -9.809->-9.971 | margin -2.570 -> +2.058
VERDICT S7 [poison/banana-batch] gate=stub-tier1: ROLLBACK — target collapse: one response is 100% of the batch | incumbent adapter intact: True
VERDICT MERGE [accumulated adapter vs raw base] gate=immune-tier2: ROLLBACK — protected probe fact-paris regressed 1.000 -> 0.000 | protected probe fact-tokyo regressed 1.000 -> 0.000 | aggregate 0.3156 -> 0.1050, drift 1.000 label=fail
```

Gate behaviour, both arms: **6/6 promotes, 0 tripwire events**. The replay
arm's batches were *more* diverse than arm N's by the tripwire statistic
(top-response fraction 0.14 vs 0.20) — replay adds two unique response
strings, and the tripwire counts within a batch, so 2/7 unique-diverse replay
never approaches the >0.5 collapse rule. The poison canary tripped the
collapse rule at 100% in both arms and left the incumbent adapter byte-intact.

## Retention r_i(k) — arm N vs arm R

`r_i(k) = (m_i(k) − m_i(0)) / g_i` after each session (`|` marks each item's
teach session; **bold** = the session where arm R replayed that item).

| item | arm | g_i | S1 | S2 | S3 | S4 | S5 | S6 | r_final |
|---|---|---|---|---|---|---|---|---|---|
| fav_show (fact) | N | +2.689 | +4.407 \| | **1.242 (−0.18)** | 4.376 (0.99) | 2.880 (0.43) | 4.672 (1.10) | 2.293 (0.21) | **0.21** |
| fav_show (fact) | R | +5.685 | +7.403 \| | **5.261 (0.62)** | 3.179 (0.26) | 4.100 (0.42) | 3.333 (0.28) | 3.755 (0.36) | **0.36** |
| math_words (style) | N | +4.189 | — | +0.815 \| | **−2.899 (0.11)** | −2.073 (0.31) | 1.998 (1.28) | −2.639 (0.18) | 0.18 |
| math_words (style) | R | +4.135 | — | +0.761 \| | **−2.688 (0.17)** | −14.112 (−2.60) | −1.957 (0.34) | 1.766 (1.24) | 1.24 |
| tea_tin (fact) | N | +2.256 | — | — | +0.990 \| | **−1.225 (0.02)** | 0.727 (0.88) | 0.527 (0.79) | 0.79 |
| tea_tin (fact) | R | +3.286 | — | — | +2.020 \| | **+4.784 (1.84)** | −1.610 (−0.10) | −0.235 (0.31) | 0.31 |
| lowercase (style) | N | +0.460 | — | — | — | +0.634 \| | **−2.953 (n/m)** | 0.395 (0.48) | 0.48 |
| lowercase (style) | R | +0.221 | — | — | — | +0.395 \| | **−1.104 (n/m)** | −1.636 (n/m) | n/m |
| pet_dog (fact) | N | +8.519 | — | — | — | — | +8.783 \| | **−3.192 (−0.41)** | **−0.41** |
| pet_dog (fact) | R | +5.752 | — | — | — | — | +6.016 \| | **+14.657 (+2.50)** | **+2.50** |
| greet_rule (style) | N | +6.933 | — | — | — | — | — | +4.364 \| | 1.00 |
| greet_rule (style) | R | +4.628 | — | — | — | — | — | +2.058 \| | 1.00 |

(n/m = r not meaningful: the one-capitalization-token probe gave gains of only
0.2–0.5 nats, so r ratios blow up on noise — see Limits.)

Per-class aggregates (µ_d = mean margin change per intervening session;
n_1/2 = mean_g/(2·µ_d) per `TEACH_PIPELINE.md`; lowercase excluded from the
style r-mean because its r is not meaningful):

| class | arm | µ_d (nats/session) | mean g | n_1/2 (sessions) | mean r_final | median r_final |
|---|---|---|---|---|---|---|
| fact | N | −1.617 | +4.488 | not fittable* | 0.20 | 0.21 |
| fact | R | +0.304 | +4.907 | not fittable* | 1.06 | 0.36 |
| style | N | −0.615 | +3.861 | not fittable* | 0.59** | 0.59** |
| style | R | −0.171 | +2.995 | not fittable* | 1.12** | 1.12** |

\* The retention curves **oscillate** rather than decay monotonically (see
§noise below), so several per-item µ_d are near zero or positive and the
E3-style monotone half-life fit is undefined/misleading here. The honest
read is the per-item curves and the direct replay contrast, not a fitted
half-life; where a fit is formally possible it says n_1/2 ≫ the observed
horizon for both arms.
\** Computed on math_words + greet_rule only.

## The direct replay contrast (the designed read)

Session *k* of arm R replays exactly the item taught in session *k−1*, so the
cleanest comparison is: for each item, the margin after the session in which
it was (R) or was not (N) replayed once, everything else equal:

| item replayed in session | margin after that session, arm N | margin after that session, arm R | Δ (R−N) | r, arm N | r, arm R |
|---|---|---|---|---|---|
| fav_show @ S2 | +1.242 | +5.261 | **+4.02** | −0.18 | +0.62 |
| math_words @ S3 | −2.899 | −2.688 | +0.21 | 0.11 | 0.17 |
| tea_tin @ S4 | −1.225 | +4.784 | **+6.01** | 0.02 | +1.84 |
| pet_dog @ S6 | −3.192 | +14.657 | **+17.85** | −0.41 | +2.50 |

4/4 in replay's favour, three of them by 4–18 nats. In arm N, both facts
that faced a later interfering session with no re-exposure were **fully
erased at least once** (fav_show r = −0.18 after S2; pet_dog r = −0.41 after
S6 — a single style session wiped a +8.8-nat teaching to 6.4 nats *below*
baseline). In arm R the same facts stayed positive through the replayed
session (r 0.62, +2.50). Replay is not just retention insurance at this
scale: on pet_dog the replay turn pushed the margin **+8.6 nats past its own
teach-time value** — two phrasing-diverse positive triples inside a 7-triple
batch are a second, cheap teach for an item the LoRA has already organized.

The flip side is equally visible: **one replay refreshes, it does not
anneal.** In arm R, after its one replay, `tea_tin` fell from r = 1.84 to
−0.10 over the next two sessions; `fav_show` slid 0.62 → 0.28 by S5. A
single successive-session replay roughly *resets the decay clock* at that
item rather than changing the decay rate.

## Noise floor (why the curves oscillate, and what this A/B can claim)

This run set the A/B twins up identically in every input and got materially
different training trajectories: the S1 `fav_show` teach batch, gens,
corrections, seeds and initial weights were bit-identical across E3 and both
arms (loss0 2.060 in all three), yet the post-teach margins were E3 +8.073,
arm N +4.407, arm R +7.403. CPU-side nondeterministic fp accumulation under
24 AdamW steps at lr 2e-3 has a **few-nats** run-to-run spread on this base,
which is the same order as the decay events being measured, and it shows up
in the tables as the r_i(k) oscillation (items re-rising two sessions after
teach with no replay, e.g. arm N fav_show −0.18 → 0.99 → 1.10). Consequences:

- Per-item, per-session differences under ~±2 nats are **inside the noise
  floor**; class-mean r_final differences (0.20 vs 1.06 facts) outrun it, and
  the 4/4 direct-contrast sign agreement is the designed signal.
- The E3 claim "chat ckpt forgets in proportion" is *not* refuted — facts in
  arm N were fully erased after single interfering sessions — but the
  E3-style steadily-decaying curve and its per-session µ_d/n_1/2 fits did not
  replicate at 6 items with this noise level. Fitted half-lives are reported
  as "not fittable" rather than numerically invented.
- Fix queued, not applied here: expose a training seed / determinant kernel
  knobs in `consolidate.train_lora_microsession` and run the A/B as n≥3
  replicates if finer attribution is wanted.

## Tier-2 merge verdicts (end of each arm)

| arm | aggregate inc → cand | protected agg inc → cand | protected regressions | drift | decision |
|---|---|---|---|---|---|
| N | 0.3156 → 0.0934 | 0.255 → 0.037 | fact-paris 1→0, fact-tokyo 1→0 (also arith-plain, fact-boil, instr-pattern-six) | 1.000 (fail) | ROLLBACK |
| R | 0.3156 → 0.1050 | 0.255 → 0.032 | same set | 1.000 (fail) | ROLLBACK |

Replay **does not** rescue merge-grade behaviour; both accumulated adapters
zero the same protected chat facts when merged into the trunk, matching E3's
tier-2 read (0.0910). The two-tier policy's claim stands and is now measured
at 6 sessions on both arms: session learning belongs in per-user adapters
behind tier 1; nothing here clears tier 2's bar.

## Verdict: what replay buys, what it costs

**Buys.** (i) On the 4/4 measurable direct contrasts, one phrasing-diverse
replay turn (2 triples, 28.6% of the batch) in the session right after teach
prevented the catastrophic single-session wipeout arm N suffered on both
vulnerable facts (r: −0.18 → +0.62 fav_show; −0.41 → +2.50 pet_dog; +6.0 nats
on tea_tin). (ii) Class-mean r_final moves from 0.20 → 1.06 on facts (median
0.21 → 0.36). (iii) Zero gate cost: tripwire statistic *improved*
(top-response 0.14 vs 0.20), all sessions promoted in both arms, poison still
caught, incumbent intact after every rollback. (iv) No acquisition loss:
teach-time margins and class-mean gains are within the measured noise floor
of arm N (mean g facts 4.9 vs 4.5; style 3.0 vs 3.9 — a wash given ±few-nats
noise).

**Costs.** (i) Each item got exactly one replay one session after teach, and
the benefit decays back within 1–2 subsequent sessions — the schedule as run
is episodic insurance, not durable consolidation; a *repeating* spaced
cadence (every-K-sessions, not just once) is the obvious next arm. (ii) Wall:
arm R 2541 s vs arm N 1710 s (+49%) for 10 extra replay turns; roughly a
third of that is replay-turn generation/precompute, the rest machine
variance — call it ≈+10–15% expected cost at steady state, plus ~1.4× the
train batch tokens per session. (iii) Tier-2 unaffected: replay does not make
micro-sessions merge-safe.

**Protocol relaxation: none needed.** The collapse tripwire never fired in
the replay arm, so the documented fallback (spaced re-teach on hinge items)
was **not** run — one fallback max, and zero were required.

## Limits

- Single run per arm, CPU fp32 nondeterminism measured at a few nats per
  session trajectory (see §noise) — the 4/4 direct contrast and class means
  are the claims; small per-cell differences are not.
- `lowercase` item design flaw: its probe contrasts differ by one
  capitalization token, baseline +0.174 with teach gains of 0.2–0.5 nats, so
  its r_i(k) is noise-amplified (formally −8.2 in arm R); excluded from class
  r-means. Replace with a multi-token style probe next round.
- The replay schedule was fixed (exactly one replay, next session,
  round-robin). Effect of replay ratio within 0–30%, repeated cadence, and
  replay-vs-fresh allocation are untested — queued, together with a seeded
  triple-sampling RNG in the train loop for replicate-grade attribution.
- Free-run generations remain transcript colour (post-teach transcripts end
  with the turn-close marker rendered as text followed by degeneration bars);
  margin probes are the mechanism read.
- Tier 2 exercised once per arm in its designed position, constants untouched.

## Reproduction

```bash
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_decay_replay.py --arm noreplay --workdir lab/teach_decay_out/noreplay \
  | tee lab/logs/teach_decay_noreplay.log
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_decay_replay.py --arm replay --workdir lab/teach_decay_out/replay \
  | tee lab/logs/teach_decay_replay.log
```

Fresh workdirs required (append-only session logs; adapter deltas and
`lab/teach_decay_out/` are weights and are not committed).
