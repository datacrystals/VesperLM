# TEACH_SPLIT_ATTRIBUTION — the SPLIT arm (segregated stores vs batch-splitting, seed 7)

**STATUS: COMPLETE — VERDICT: prediction P2 fired. The shared store is
the binding constraint; segregated adapter stores get the credit for
BANK's seed-1 win.** Run 2026-10-20 on the laptop, CPU only, fp32,
wall 2376 s. Queue item #14 of `lab/TEACH_PIPELINE.md`. The n=3 sweep
(`lab/TEACH_BANKED_ADAPTERS_SEED1042.md`) closed DEFER — direction
weakly positive at axis level but mechanism unproven, effect under the
sweep noise floor. Every BANK data point bundled (a) segregated adapter
stores with (b) class-split consolidation granularity; this arm
measured (b) alone: **one shared adapter with the class-split batches
exactly as BANK builds them** (two per-session consolidate calls, fixed
slice order [fact, style], chained replay in the replayed item's class
slice, shared filler in shield slices), serving/probing CTRL-side on
the shared store.

Harness: `lab/teach_banked_adapters.py` with the committed `--arm split`
mode. Frozen: knobs (rank 8, alpha 16, lr 2e-3, kl 0.05, 24 steps,
seed 7), checkpoint `lab/imported/vk_sft_r3/step_1800/chat_model`, the
amended 6-item library (metric_units retired), shield later-phase,
burial doctrine, poison canary, one tier-2 verdict. Workdir
`lab/teach_banks_split_out/`; log `lab/logs/teach_split.log`; results
`lab/teach_banks_split_out/teach_banks_split.json`;
calib workdirs `lab/teach_split_calib/{A,B}`.

## 0. Determinism verification (done before arming)

Two `--arm split --only fav_show --seed 7` teaches, fresh workdirs,
sequential: **bit-identical to each other AND bit-matching the
committed series S1 post-teach margin** (a single-item split teach has
exactly one non-empty class slice whose batch is the verbatim series
batch — the shared-store path must coincide with CTRL's S1, and does):

| run | baseline margin | post-teach margin |
|---|---|---|
| calib A (split mode) | +1.7177737951278687 | **+4.4509382247924805** |
| calib B (split mode) | +1.7177737951278687 | **+4.4509382247924805** |
| committed series S1 | +1.7177737951278687 | +4.4509382247924805 |

In-run anchors: carried-item baselines bit-match committed; ordinal
baseline bit-matches the prior arms' recording; fav_show's gains row
keeps the committed +2.733 (the single-batch-teach continuity survived
into the arm itself). The CADENCE-IDENT subset prints DIVERGED at S2+
(exactly as declared — S2/S3's chained replays now consolidate in their
own slices, so mid-phase margins re-realize). GAINS: fav_show +2.733,
math_words +6.713, **tea_tin −0.597**, **ordinal_words −2.798**,
pet_dog +2.983, greet_rule +8.761 — the two negative denominators are
the arm's own headline evidence (§5): two items failed to acquire
AGAINST their own baselines inside one shared store whose batches
alternate class across calls.

## 1. Pre-declared predictions (§1 of the IN-RUN commit, verbatim)

Registered state (seed 7): CTRL 3/6 held / 1 dead (pet_dog −0.004) /
mean 0.772 | BANK 4/6 / 0 / 0.729.

- **P1 — SPLIT ≈ BANK (held ≥ 4 AND dead = 0):** batch-granularity
  verdict; segregated-store claim retires; mini-batch scheduling becomes
  the operative lesson.
- **P2 — SPLIT ≈ CTRL (held ≤ 3 AND dead ≥ 1):** segregated stores get
  the credit; banks still DEFERRED at the n=4 axis level, mechanism
  attribution firm; next-right-experiment = bigger-substrate teach-port
  with bank scaffolding.
- **P3 — anything else:** anomalous-stated (class-to-bank mapping
  feedback); the foreign instrumentation reads assign the case.

## 2. Verdict lines (as run; generation tails trimmed)

```
SPLIT-7 S1 [split-fact/fact/fav_show]: PROMOTE | top_frac 0.20 n_triples 5 train loss 0.534->-7.892 (verbatim series S1)
SPLIT-7 S2+ : class-split accumulate under one store; tiny replay slices pass the stub at the documented 0.50 boundary
SPLIT-7 probe after S6: buried ALREADY [fav_show r=-0.479, tea_tin r=-8.204 (negative-gain denominator)] — the worst S6 entry state of the arm family
SPLIT-7 shield phase: alive sets 3–4/6 most probes; burial trace: fav_show (S6, re-entered), tea_tin (S6, still buried),
      ordinal (S7/S9/S11, twice re-entered then still buried)
SPLIT-7 HORIZON: 2/6 held (fav_show 0.89, math_words 0.59); 2 dead (tea_tin r -1.94 [n/m denominator], ordinal 0.009); mean 0.035; min -1.938
SPLIT-7 POISON alice: ROLLBACK — target collapse 100% | incumbent intact: True
SPLIT-7 MERGE (immune-tier2): ROLLBACK — fact-paris 1->0, fact-tokyo 1->0 | aggregate 0.3156 -> 0.0299,
      protected 0.255 -> 0.024, drift 1.000 label=fail — 15th reading (the series' weakest merge candidate)
```

## 3. per-item table: CTRL vs BANK vs SPLIT (seed 7, same library/seed)

r_final per arm with its own gains (split's tea/ordinal gains are
negative — their r values are n/m-flagged; the raw margin-above-baseline
nats are shown in the companion row so the retention read survives the
broken denominators).

| item | CTRL | BANK | **SPLIT** | SPLIT margin − baseline (nats) |
|---|---|---|---|---|
| fav_show (F) | 1.42 | 1.05 | 0.89 | +2.44 |
| math_words (S) | 0.86 | 0.26 | 0.59 | +3.99 |
| tea_tin (F) | 0.32 | **1.46** | **−1.94 (n/m; g<0)** | **+1.16** |
| ordinal (S) | 1.81 | 0.54 | 0.01 (n/m; g<0) | −0.03 |
| pet_dog (F) | −0.00 (dead) | **0.83** | 0.41 | +1.22 |
| greet_rule (S) | 0.23 | 0.24 | 0.24 | +2.11 |

| metric | CTRL | BANK | **SPLIT** |
|---|---|---|---|
| held r ≥ 0.50 | 3/6 | 4/6 | **2/6** |
| dead r ≤ 0.06 | 1 | 0 | **2** |
| mean r | 0.772 | 0.729 | **0.035** |
| dead-free teach (all gains > 0)? | yes | yes | **no (tea −0.60, ordinal −2.80 acquired negatively)** |

**Measured P2 test (declared rules):** SPLIT held 2/6 ≤ 3 AND dead 2 ≥ 1
→ **P2 fires — segregated adapter stores get the credit.** And the
result is not merely CTRL-equivalent: SPLIT is **worse than both
parents** (2/6 and 2 dead with a poisoned-acquisition teach phase):
two sequential same-store consolidate calls per session restore the
cross-class interference and add a systematic last-writer advantage on
top of it.

## 4. Instrumentation read: the last-writer asymmetry (per declared reinterpretation)

`isolation_per_probe` rows, per shield session: r_own (margin right
after the item's own-class slice call) vs r_off (margin after the full
session — also the main snapshot). Fixed slice order [fact, style]
means style items end every session as the last writer and their
own/off gap is zero BY CONSTRUCTION; the fact items' gap measures what
the style slice does to fact content mid-session.

| item | own/off gaps across S7..S12 (nats of r) | read |
|---|---|---|
| pet_dog | +0.24, +0.40, +0.27, −0.41, −0.84, −0.79 | fact item's call-then-style-call cost: sign-mixed early, systematically negative late — the shared store erodes fact margins as sessions accumulate |
| fav_show | +0.06(scattered ±), −1.13 at S12 | mostly noise, big last-writer cost on the final session |
| style items | 0.00 everywhere | zero by construction (last writer) |

The within-session last-writer cost is real but small (±0.2–0.8 r
median ≈ 0); **the arm-destroying damage happened in the TEACH phase**
(two items acquired against their baselines — g < 0), which is where the
shared store's cross-class interference always starts accumulating, and
no within-session trick reaches it. Split-batch form alone does nothing
protective once the deltas land on one store.

## 5. Verdict + attributive consequence

**VERDICT: P2 — segregated adapter stores get the credit.** The
granularity hypothesis is falsified: class-split batches into one
shared store do not preserve BANK's advantage; they make the arm
family's worst horizon yet (2/6 held, 2 dead, mean 0.035, two negative-
gain teaches). Where the credit lands, in mechanism detail: (i) the
binding constraint is where the DELTAS LIVE (one store = cross-class
interference returns at least at CTRL strength, this draw beyond it);
(ii) batch splitting is not a mechanism in itself — it shifts when
content gets trained but buys no isolation; (iii) the teach phase is
once again where the loss is made (both negative-gain items were
acquisition failures on this draw), corroborating the series' oldest
structural read.

**What becomes of banks, attributively:** the segregated-stores CLAIM
is now firm as mechanism attribution (the only thing that varies
BANK-like outcomes from CTRL/SPLIT-like ones across all four measured
architectures is the store split; everything class-slice-shaped that
isn't the store fails to deliver it). The banks AXIS remains deferred —
n=3 sweep showed the effect size sits under seed noise (2/3 axis record,
1/3 fact-flip regulator); SPLIT adds no BANK-axis datapoint by design.
The honest queue consequence, per the pre-declared P2 lane:

- Item 15 = **bigger-substrate teach-port with bank scaffolding** —
  BLOCKED-HONEST, not runnable today: the SFT family available to this
  box tops out at the same 561M checkpoint the interference was
  measured on (sft_429m_r2/sft_470m are smaller); the port awaits the
  next trained rung and inherits this doc's protocol verbatim. What
  the attribution gives the port is the precise thing to check first:
  whether cross-class interference at teach-Acquisition scales down
  with substrate size *and* whether bank-vs-share changes sign there.
- Item 12(a)/12(b) stay queued at lower priority — 12(b)'s
  style-triangulation presumes an intra-class story already falsified
  at n=3 and once more here (the style class is NOT where SPLIT lost;
  the shared store's fact loss is the bigger provenance).
- Mini-batch class-slice scheduling = RETIRED as candidate mechanism
  for the seed-1 win (P1 territory closed by measurement).

**Tier 2:** 15th ROLLBACK (agg 0.3156 → 0.0299 — the weakest merge
candidate of the series, protected 0.255 → 0.024, paris/tokyo killed)
— fully consistent with the teach-loop aggregate (weakest adapter);
merge-grade behaviour remains architecture-insensitive.

n=1 seed-7 draw, one fallback unused, no crash, 12 session-heads +
slices promoted with 0 tripwires, poison caught, incumbent intact.

## Reproduction

```bash
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_banked_adapters.py --arm split --seed 7 \
  --foreign-probes --workdir lab/teach_banks_split_out \
  | tee lab/logs/teach_split.log

# entry calibrations (2 runs; bit-identical and bit-matching the
# committed series S1 post-teach margin):
python3 -u lab/teach_banked_adapters.py --arm split --only fav_show --seed 7 --workdir <fresh A>
python3 -u lab/teach_banked_adapters.py --arm split --only fav_show --seed 7 --workdir <fresh B>
```

Fresh workdir required. CPU only, fp32 (I7). No `lab/FAILURES.md`
entry.
