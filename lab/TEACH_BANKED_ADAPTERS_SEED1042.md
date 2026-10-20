# TEACH_BANKED_ADAPTERS_SEED1042 — 1042-family replication + item-13 instrumentation

**STATUS: COMPLETE — VERDICT: DEFER (per the pre-declared n=3 rule-set).**
Queue item #13 of `lab/TEACH_PIPELINE.md`, run 2026-10-20 on the laptop,
CPU only, fp32 (CTRL wall 2354 s, BANK wall 3362 s with instrumentation).
Seed family 3 = `--seed 1042` (seed-1: 7, seed-2: 43; no seed mixing).
Harness `lab/teach_banked_adapters.py` — CTRL verbatim flag-free; BANK
adds the same-script instrumentation flag `--foreign-probes` committed
with this arm. Everything else frozen (library, knobs rank 8 / alpha 16 /
lr 2e-3 / kl 0.05 / 24 steps, checkpoint
`lab/imported/vk_sft_r3/step_1800/chat_model`, shield later-phase, burial
doctrine, poison per store, one tier-2 verdict per arm).

Context recap (registered state before this run): the seed-1 arm
(`lab/TEACH_BANKED_ADAPTERS.md`) won on held+dead; the seed-2 gate
(`lab/TEACH_BANKED_ADAPTERS_SEED2.md`) closed negative on the conjunctive
read — direction replicated 2-for-2, both mechanism regulators failed.
Item 13 = this wider seed run plus quantitative cross-/intra-class
instrumentation.

## 0. Entry determinism (the documented fixture ritual) — PASSED

Two BANK-mode calibrations (`--arm bank --only fav_show --seed 1042`,
fresh workdirs, sequential) **bit-identical to each other**, creating the
seed-1042 fixture:

| run | baseline margin | post-teach margin |
|---|---|---|
| calib A (bank, seed 1042) | +1.7177737951278687 | **+13.200785398483276** |
| calib B (bank, seed 1042) | +1.7177737951278687 | **+13.200785398483276** |

Cross-seed anchors asserted in both arms at startup (seed-independent
raw-model probes): carried-item baselines bit-match the committed series
values; ordinal baseline bit-matches the seed-1 recording
(+0.07648158073425293). CADENCE-IDENT prints diverge by design (the seed
is the variable). GAINS re-realized hugely again (pet_dog +29.449 CTRL /
+5.856 BANK; greet_rule +21.985 BANK; teach margins move ~3× by seed
family, as the sweep has consistently shown).

## 1. The declared n=3 verdict rule-set (declared §1 before arming, restated)

- **Direction record**: BANK held > CTRL held AND BANK dead < CTRL dead
  per seed (ties = misses). Required: **3/3**.
- **Mechanism regulators**, each required to pass on ≥ 2/3 seeds:
  (i) *fact-flip*: CTRL fact μ_d < 0 → BANK fact μ_d ≥ ~0;
  (ii) *style-local*: BANK style-class mean r_final ≤ ~0.5.
- **ADOPTION CONDITIONAL OK** iff direction 3/3 AND both regulators ≥2/3;
  otherwise **DEFER** with labeled confidence and the next-right-
  experiment.

## 2. Verdict lines (as run; generation tails trimmed)

```
CTRL-1042 GAINS g_i: fav_show +11.483  math_words +6.242  tea_tin +3.393  ordinal_words +1.134  pet_dog +29.449  greet_rule +2.581
CTRL-1042 12/12 promotes, 0 tripwires | burials: fav_show probe-S6 (r -0.007, re-entered, re-buried S11), pet_dog S9 (re-entered), ordinal S11 (still buried)
CTRL-1042 HORIZON: 4/6 held (math_words, tea_tin, ordinal_words, greet_rule); 1 dead (fav_show 0.035); mean 0.580; fact mu_d -1.357/sess
CTRL-1042 POISON alice: ROLLBACK intact: True
CTRL-1042 MERGE (immune-tier2): ROLLBACK — fact-paris 1->0, fact-tokyo 1->0 | aggregate 0.3156 -> 0.1733, drift 1.000 label=fail — 13th reading

BANK-1042 GAINS g_i: fav_show +11.483  math_words +12.999  tea_tin +4.073  ordinal_words +1.760  pet_dog +5.856  greet_rule +21.985
      (fav_show identical to CTRL-1042 as it must be — identical S1 batch)
BANK-1042 12/12 session-heads PROMOTE, 0 tripwires | burials: fav_show probe-S6 (r 0.067, stays buried), greet_rule S9 (re-entered, re-buried S11)
BANK-1042 HORIZON: 3/6 held (tea_tin 1.32, ordinal_words 0.89, pet_dog 0.61); 1 dead (fav_show 0.014); mean 0.545; min 0.014
BANK-1042 POISON alice/fact: ROLLBACK intact: True | alice/style: ROLLBACK intact: True
BANK-1042 MERGE alice/fact (immune-tier2): ROLLBACK — paris/tokyo 1->0 | aggregate 0.3156 -> 0.0870, drift 1.000 label=fail — 14th reading
```

## 3. n=3 synthesis: per-item r_final, axis record, regulator pass-rate

### per-item r_final (CTRL | BANK by seed)

| item | s1(s7) C | s1 B | s2(s43) C | s2 B | s3(s1042) C | s3 B |
|---|---|---|---|---|---|---|
| fav_show | 1.42 | 1.05 | 0.35 | 0.35 | **0.04 (dead)** | **0.01 (dead)** |
| math_words | 0.86 | 0.26 | 0.59 | 0.54 | 0.63 | 0.30 |
| tea_tin | 0.32 | **1.46** | **1.05** | 0.71 | 1.15 | **1.32** |
| ordinal_words | 1.81 | 0.54 | 0.20 | 0.27 | 0.59 | **0.89** |
| pet_dog | −0.00 | **0.83** | −0.03 | **0.50** | 0.12 | **0.61** |
| greet_rule | 0.23 | 0.24 | 0.29 | **0.86** | **0.96** | 0.15 |

### horizon aggregates + axis record

| axis | s1 C → B | s2 C → B | s3 C → B | record |
|---|---|---|---|---|
| held ≥ 0.50 | 3 → **4** ✓ | 2 → **3** ✓ | **4 → 3** ✗ | **2/3 (misses 3/3)** |
| dead r ≤ 0.06 | 1 → **0** ✓ | 1 → **0** ✓ | 1 → **1** (tie = miss) ✗ | **2/3 (never inverts against BANK; ties once)** |
| mean r (context, not in rule) | 0.772 > 0.729 | 0.407 < **0.536** | 0.580 > 0.545 | BANK wins mean at 1/3 seeds |
| min r | −0.004 → 0.240 | −0.029 → 0.273 | 0.035 → 0.014 | BANK floor better at 2/3 |

### mechanism regulator pass-rate (the declared rule columns)

| regulator | s1 | s2 | s3 | record | rule |
|---|---|---|---|---|---|
| fact-flip (BANK fact μ_d ≥ ~0 given CTRL < 0) | **PASS** (−0.938 → +0.032) | FAIL (−1.013 → −0.500) | FAIL (−1.357 → −0.457) | **1/3** | required ≥ 2/3 → **MISS** |
| style-local (BANK style mean ≤ ~0.5) | PASS (0.35) | FAIL (0.56) | PASS (0.44) | **2/3** | required ≥ 2/3 → **MEET** |

## 4. Instrumentation read: per-probe cross-class drift (BANK-1042, new)

`foreign_margins` + `isolation_per_probe` in the BANK JSON give the
cross-class read quantitatively (r_own under the owning bank vs r_off
under the foreign bank, same m0 and the item's own arm gain):

| item | probe after S6 | S8 | S10 | S12 | read |
|---|---|---|---|---|---|
| tea_tin | 1.05/0.33 | 1.12/0.28 | 1.43/0.72 | 1.32/0.68 | owning ≈ 2× foreign all phase — class content stays home; foreign drifts UP with phase (shared chat-pattern transfer, not trained leakage — no fact turn ever writes the style bank) |
| pet_dog | 0.76/0.86 | 0.88/0.77 | 0.43/0.16 | 0.61/−0.02 | early parity collapses apart by S9 — segregation cuts exactly the phase where CTRL-1042 buried pet_dog this seed |
| greet_rule | 1.00/0.06 | 0.12/−0.10 | 0.14/−0.08 | 0.15/0.01 | its s3 BANK gains were giant (+21.99 → coarse r); foreign read ~0 as expected |
| ordinal | 0.31/−0.06 | 0.36/0.05 | 0.10/−0.22 | 0.89/−0.12 | owning recovers late; foreign stays ≈ 0/negative |
| fav_show | 0.07/−0.28 | −0.21/−0.27 | −0.00/−0.00 | 0.01/0.11 | BOTH banks carry fav weakly (it crashed at S6 under both arms this seed) — segregation cannot rescue an item that died before the shield phase |
| math_words | 0.46/0.25 | 0.27/0.13 | 0.43/0.20 | 0.30/0.01 | same direction, small gap |

The quantitative layer supports the direction stories without rescuing
the regulators: foreign reads track at bystander amplitude for facts
except tea_tin/pet_dog where the gap opens exactly where the class
interference is (largest-own-gain items), and intra-class churn persists
inside each bank.

## 5. Verdict + consequence (measured against §1, no post-hoc moves)

**VERDICT: DEFER.** By the pre-declared n=3 rule-set: the direction axis
reached **2/3** on held (miss s3: BANK 3 < CTRL 4) and **2/3** on dead
(s3 tie = miss by the declared tie rule — though the direction NEVER
inverted against BANK on dead at any seed); requirement was 3/3. The
fact-flip regulator stands at **1/3** (required ≥ 2/3); style-local at
**2/3** (meets). The conjunction fails, so no adoption — reported as
declared rather than argued.

**Labeled confidence (honest):**
- *Held axis 2-of-3, dead axis 2-of-3, mechanism 1-of-3.* The
  architecture's axis benefit is real but marginal-sized (held deltas
  +1, +1, −1 across seeds) and sits under the sweep's own noise floor
  (teach gains move 3× by seed family; item-level r reads are
  coarse-ratio within each draw).
- The pet_dog motif softened at seed-3 (held in BANK at all three seeds
  0.83/0.50/0.61; CTRL-1042's pet survived unscathed 0.12 — the motif's
  clearest claim was dead → held, and s3 CTRL never buried it).
- Interpretation: banking is **weakly positive at the axis level and
  mechanism-unproven at n=3**; the seed-noise dominates the effect size,
  so more seeds of the same read yield noise, not resolution.

**Next-right-experiment (named, per the rule): the attribution arm
"SPLIT"** — same class-split batches as BANK (fact turns and style turns
consolidated in two per-session calls exactly like BANK) but into ONE
shared adapter. This separates the two things the BANK architecture
bundles inseparably in every existing data point: *segregated adapter
stores* vs *class-split consolidation granularity*. Three predictions,
declared: if SPLIT ≈ BANK on held/dead → the win was batch-splitting
(coarser consolidation), not segregation; if SPLIT ≈ CTRL → the adapter
store was the cause; anything between → shared contribution. One arm,
~2.5 h wall, every knob/batch already pinned by this series. Queued as
item 14 ahead of 12(a)/12(b) (triangulation stays queued at lower
priority — its question presumes the intra-class story that just failed
to replicate).

**Tier-2:** 13th ROLLBACK (CTRL-1042, agg 0.3156 → 0.1733, paris/tokyo
killed) and 14th (BANK-1042 fact bank, agg → 0.0870, protected 0.255 →
0.028). Merge-grade behaviour untouched by any teach-loop choice so far;
that claim is now stronger than any teaching claim in the series.

## Reproduction

```bash
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_banked_adapters.py --arm ctrl --seed 1042 \
  --workdir lab/teach_banks3_ctrl_out | tee lab/logs/teach_banks3_ctrl.log
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_banked_adapters.py --arm bank --seed 1042 \
  --foreign-probes --workdir lab/teach_banks3_bank_out \
  | tee lab/logs/teach_banks3_bank.log

# entry calibration (2 runs; creates the seed-1042 fixture):
python3 -u lab/teach_banked_adapters.py --arm bank --only fav_show --seed 1042 --workdir <fresh A>
python3 -u lab/teach_banked_adapters.py --arm bank --only fav_show --seed 1042 --workdir <fresh B>
```

Fresh workdir required per run. CPU only, fp32 (I7). Both arms exited 0;
no tripwire, no fallback, no `lab/FAILURES.md` entry.
