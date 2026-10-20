# TEACH_BANKED_ADAPTERS_SEED2 — seed-2 replication of the BANK-vs-CTRL architecture win

**STATUS: COMPLETE — replication FAIL on the declared conjunctive read
(n = 2).** The headline architecture axis replicated **2-for-2** (seed-2
BANK held 3/6 > CTRL 2/6, dead 0 < 1, min 0.273 vs −0.029) but both
mechanism regulators failed: the fact-class decay flip **stayed
same-sign** (CTRL −1.013 → BANK −0.500 nats/session — magnitude halved,
sign kept, against the declared BANK ≥ ~0) and the style-bank residual
**locale moved** (style-class mean 0.35 seed-1 → 0.56 seed-2, over the
declared ~0.5 marker, with the fact/style class ordering inverted vs
seed-1). Per the fail clause declared in this doc before arming, the
conjunction fails; the honest reading is **direction replicated twice,
mechanism unproven — adoption gate closes negative**, and item 13 of
`lab/TEACH_PIPELINE.md` is wider seed replication + mechanism
re-instrumentation (not a design-failure verdict: the headline axis paid
on both draws). Seed-2 harness was `lab/teach_banked_adapters.py`
VERBATIM — only `--seed 43`, nothing else touched, library/knob/
checkpoint/protocol frozen.

Run 2026-10-20 on the laptop, CPU only, fp32. CTRL wall 1855 s, BANK wall
2486 s. Results: `lab/teach_banks2_ctrl_out/teach_banks_ctrl.json`,
`lab/teach_banks2_bank_out/teach_banks_bank.json`; logs
`lab/logs/teach_banks2_{ctrl,bank}.log`.

## 0. Entry determinism + the seed-2 fixture (done before arming)

Two identical BANK-mode calibrations (`--arm bank --only fav_show
--seed 43`, fresh workdirs, sequential) — **bit-identical to each
other**, creating the seed-2 fixture (no prior fixture existed; all
series fixtures are seed 7):

| run | baseline margin | post-teach margin |
|---|---|---|
| calib A (bank, seed 43) | +1.7177737951278687 | **+13.562385082244873** |
| calib B (bank, seed 43) | +1.7177737951278687 | **+13.562385082244873** |
| (seed-7 fixture, for shape) | +1.7177737951278687 | +4.4509382247924805 |

Cross-seed anchors asserted at arm startup (seed-independent raw-model
probes), both passing **in both arms**: all five carried items'
`before/` baselines bit-match the committed series baselines; the
ordinal baseline bit-matches the seed-1 arm's recording
(+0.07648158073425293). CADENCE-IDENT prints diverge from S1 — that is
the seed doing its job; the determinism claim rests on the anchors +
the bit-identical calibration pair.

Seeds move teach gains hugely (pet_dog +17.35 → +19.31 CTRL vs +4.90
committed; tea_tin +13.31 → +4.46 CTRL/+5.85 BANK; ordinal_words
+0.89/+1.57 → +0.687/−0.347). Denominators re-realize per arm; r
remains the normalized read, with ordinal flagged n/m this draw (§3).

## 1. Criteria (declared before arming, restated for the accounting)

- **Pass** = BANK held > CTRL held AND BANK dead < CTRL dead AND
  style-bank residual stays local (BANK style mean ≤ ~0.5) AND
  fact-class decay flips CTRL-negative → BANK ≥ ~0.
- Any contradiction **(including "the fact-decay flips stay
  same-sign")** = replication fail, with item 13 = wider seed
  replication / design failure of banks.

## 2. Verdict lines (as run; generation tails trimmed)

```
CTRL-43 teach S1..S6: 12/12 promotes, 0 tripwires; CADENCE-IDENT subset DIVERGED (seed, declared)
CTRL-43 GAINS g_i: fav_show +11.845  math_words +5.811  tea_tin +4.456  ordinal_words +0.687  pet_dog +19.307  greet_rule +6.759
CTRL-43 HORIZON: 2/6 held >= 0.50; 1/6 dead (pet_dog -0.029); mean 0.407; min -0.029; fact μ_d -1.013/sess
CTRL-43 POISON alice: ROLLBACK intact: True
CTRL-43 MERGE (immune-tier2): ROLLBACK — aggregate 0.3156 -> 0.2424, protected 0.255 -> 0.259 (protected probes SURVIVED —
      veto fires on aggregate margin 0.073 > 0.02 + drift 1.000 fail only) — 11th reading

BANK-43 CADENCE-IDENT subset: DIVERGED (seed, declared family)
BANK-43 GAINS g_i: fav_show +11.845  math_words +12.865  tea_tin +5.847  ordinal_words -0.347  pet_dog +7.839  greet_rule +2.146
      (fav_show identical to CTRL-43 as it must be — its S1 batch is identical across arms)
BANK-43 12/12 session-heads PROMOTE, 0 tripwires; burial trace: ordinal_words probe-after-S6 (degenerate — negative
      teach gain; stays buried), math_words probe-after-S9 (re-entered)
BANK-43 HORIZON: 3/6 held >= 0.50; 0/6 dead; mean 0.536; min 0.273
BANK-43 POISON alice/fact: ROLLBACK intact: True | POISON alice/style: ROLLBACK intact: True
BANK-43 MERGE alice/fact (immune-tier2): ROLLBACK — fact-paris 1->0, fact-tokyo 1->0 | aggregate 0.3156 -> 0.1038,
      protected 0.255 -> 0.011, drift 1.000 label=fail — 12th reading
```

## 3. per-item r_final — seed-2 BANK vs CTRL, with seed-1 alongside

| item | seed-1 CTRL | seed-1 BANK | **seed-2 CTRL** | **seed-2 BANK** |
|---|---|---|---|---|
| fav_show (F) | 1.42 | 1.05 | 0.35 | 0.35 |
| math_words (S) | 0.86 | 0.26 | 0.59 | 0.54 |
| tea_tin (F) | 0.32 | **1.46** | **1.05** | 0.71 |
| slot 4 ordinal (S) | 1.81 | 0.54 | 0.20 | 0.27 (n/m: g = −0.347) |
| pet_dog (F) | −0.00 (dead) | **0.83** | **−0.03 (dead, buried at S10)** | **0.50** |
| greet_rule (S) | 0.23 | 0.24 | 0.29 | **0.86** |

| metric | seed-1 CTRL | seed-1 BANK | **seed-2 CTRL** | **seed-2 BANK** | axes 2-for-2? |
|---|---|---|---|---|---|
| held ≥ 0.50 | 3/6 | 4/6 | 2/6 | **3/6** | BANK > CTRL in both ✓ |
| dead r ≤ 0.06 | 1 (pet) | 0 | 1 (pet) | **0** | BANK < CTRL in both ✓ |
| mean r | 0.772 | 0.729 | 0.407 | **0.536** | BANK > CTRL in seed-2; seed-1 CTRL higher (honest flag stands) |
| min r | −0.004 | 0.240 | −0.029 | **0.273** | BANK floor better in both |

pet_dog is the repeated rescue pattern: dead-in-CTRL at both seeds
(−0.004, −0.029), survived-in-BANK at both (+0.83, +0.50). ordinal_words
is **n/m this draw** in BANK (−0.347 teach gain inverts its r-axis — its
printed r_final +0.27 corresponds to a margin slightly BELOW baseline;
items with negative gains get the lowercase-style caveat and are excluded
from any ratio-based conclusion; all counts here still include its
values verbatim).

## 4. Mechanism / decay sign table (the regulators that failed)

| read | seed-1 CTRL → BANK | seed-2 CTRL → BANK | replication? |
|---|---|---|---|
| fact-class within-phase decay μ_d (nats/sess) | −0.938 → **+0.032** | −1.013 → **−0.500** | **NO** — stays same-sign; magnitude halved |
| style-class mean r_final (the "style-bank residual") | 0.96 → **0.35** (residual local) | 0.36 → **0.56** (over the ~0.5 marker) | **NO** — locale moved; class ordering inverted |
| headline axis (held, dead) | BANK wins both | BANK wins both | **YES, 2-for-2** |
| pet_dog dead-in-CTRL → held-in-BANK | yes (−0.00 → +0.83) | yes (−0.03 → +0.50) | YES (pattern repeats) |
| isolation direction r_own ≳ r_off | yes, flagship wins | yes, weaker (tea 0.71/0.07, greet 0.86/0.38; pet 0.50/−0.58; ordinal off −6.0 on degenerate gain) | YES directionally |

## 5. Replication verdict

**FAIL on the declared conjunctive read (n = 2), with the axis/regulator
split kept exact:**

- **What replicated (the headline architecture axis):** BANK held >
  CTRL held and BANK dead < CTRL dead at BOTH seeds (4>3 / 0<1 at seed
  7; 3>2 / 0<1 at seed 43). Direction 2-for-2. The pet_dog
  dead-in-CTRL/rescued-in-BANK motif repeats, and the isolation read
  keeps r_own ≥ r_off on flagship items.
- **What did not (the mechanism story):** the fact-class decay sign
  flip didn't flip (−1.013 → −0.500 — same sign, halved). The style
  residual locale moved and inverted the class ordering (style 0.35
  seed-1 → 0.56 seed-2; fact 1.11 → 0.51). Both were named regulators
  and both failed; the conjunctive read declared in §1 therefore fails,
  and this verdict is reported as such rather than arguing the
  criterion post-hoc. Reading: at n=2 the evidence supports "banks pay
  on held+dead at this capacity" and does NOT support the specific
  mechanism claims made from the seed-1 draw (cross-class decay-flip
  story; style-residual-local story). Distinguishing "small real effect
  + noise" from "seed luck on the axis" needs exactly what the
  fork names.
- **Tier-2 per arm (expected ROLLBACKs):** CTRL-43 0.3156 → 0.2424,
  protected 0.255 → 0.259 (protected probes survived; veto on aggregate
  margin + drift) — 11th; BANK-43 (fact bank) 0.3156 → 0.1038,
  protected 0.255 → 0.011, paris/tokyo killed — 12th. Merge-grade
  behaviour remains untouched by the teach-loop architecture; the drift
  battery does not read margins.

**Consequence (item 13, staged):** wider seed replication (seed family
1042-style, ≥2 more draws on both arms) with **mechanism
re-instrumentation** — instrument cross-class vs intra-class drift per
bank per session from the probe snapshots rather than reading the
class-mean only, so the decay story gets a direct observable instead of
an inferred one. Adoption of class-segregated banks **deferred**; the
12(c) adoption gate is closed-negative as declared. Items 12(a)
(equilibration on BANK) and 12(b) (style triangulation) stay queued but
lose read priority to 13 — running more arms on an un-replicated
mechanism compounds inference error, not evidence.

## Reproduction

```bash
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_banked_adapters.py --arm ctrl --seed 43 \
  --workdir lab/teach_banks2_ctrl_out | tee lab/logs/teach_banks2_ctrl.log
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_banked_adapters.py --arm bank --seed 43 \
  --workdir lab/teach_banks2_bank_out | tee lab/logs/teach_banks2_bank.log

# entry calibration (2 runs; creates the seed-2 fixture):
python3 -u lab/teach_banked_adapters.py --arm bank --only fav_show --seed 43 --workdir <fresh A>
python3 -u lab/teach_banked_adapters.py --arm bank --only fav_show --seed 43 --workdir <fresh B>
```

Fresh workdir required per run. CPU only, fp32 (I7). Both arms exited 0;
no tripwire, no fallback, no `lab/FAILURES.md` entry.
