# TEACH_SPLIT_ATTRIBUTION — the SPLIT arm (segregated stores vs batch-splitting, seed 7)

**STATUS: IN RUN — calibration fixture pair under verification; arm
arming after.** Queue item #14 of `lab/TEACH_PIPELINE.md`, run
2026-10-20 on the laptop, CPU only, fp32. The n=3 sweep closed DEFER
(`lab/TEACH_BANKED_ADAPTERS_SEED1042.md`): BANK beats CTRL on held/dead
at 2/3 seeds, never inverts the dead axis, but the mechanism regulators
failed at n=3 (fact-flip 1/3) and the effect size sits under the sweep's
noise floor. Every existing BANK data point bundles two changes
inseparably: **(a) segregated adapter stores** (fact/style deltas split
by class — the seed-1 headline mechanism) and **(b) class-split
consolidation granularity** (two smaller per-class consolidate calls per
session instead of one mixed call). This arm measures (b) alone:

**MODE SPLIT = ONE shared adapter (storage-level CTRL), but the batches
split by class exactly as BANK builds them** — two per-session
consolidate calls with the class-filtered batches (teach pack in its
class slice, chained replay in the replayed item's class slice, shared
filler strings in both slices within the shield phase), fixed slice
order [fact, style] per session — and serving/probing applies the shared
adapter for everything (CTRL-side application semantics).

Harness: `lab/teach_banked_adapters.py` with the committed `--arm split`
mode (same script family as items 11–13). Everything else frozen: knobs
(rank 8, alpha 16, lr 2e-3, kl 0.05, 24 steps), checkpoint
`lab/imported/vk_sft_r3/step_1800/chat_model`, the amended 6-item
library (fav_show/math_words/tea_tin/ordinal_words/pet_dog/greet_rule;
metric_units stays retired), teach phase S1–S6 verbatim packs, shield
later-phase S7–S12, burial doctrine, poison canary on the shared store,
one tier-2 Immune verdict. Seed: **7** — the seed-1 draw is the one
whose BANK-vs-CTRL gap this arm must attribute. Workdir
`lab/teach_banks_split_out/`, log `lab/logs/teach_split.log`, calib
workdirs `lab/teach_split_calib/{A,B}`.

## 0. Determinism verification (done before arming)

A single-item split-mode teach under seed 7 has exactly one non-empty
class slice (the item's own) whose batch is byte-identical to the series
teach batch — so the declared fixture ritual is: two
`--arm split --only fav_show --seed 7` teaches, fresh workdirs,
sequential, each must **bit-identically match the other AND the
committed series S1 post-teach margin** (same computation as CTRL's S1):
baseline +1.7177737951278687 → post-teach +4.4509382247924805. Baseline
anchors for the arm itself (seed-independent probes, in-run): carried
items bit-match committed; ordinal_baselines bit-match the prior arms'
record (+0.07648158073425293). The CADENCE-IDENT subset print is
**expected to FULLY bit-match for this arm** — CTRL-side storage with
seed 7 reproduces the committed cadence margins through S3 *if and only
if* the split batches through S3 happen to coincide with the series
batches up to that point (they do not: S2/S3's chained replays move to
their own slice-consolidate calls, so S2+ will diverge from the
committed run; the S1 row itself SHOULD bit-match and any S1 mismatch
is a harness bug, declared).

## 1. Pre-declared predictions (the arm's whole reason)

Registered state: seed-7 measures — CTRL (shared, mixed batches): 3/6
held, 1 dead (pet_dog −0.004), mean 0.772 | BANK (segregated, split
batches): 4/6 held, 0 dead, mean 0.729.

Prediction set (V, before any output):

- **P1 — SPLIT ≈ BANK (granularity wins):** held ≥ 4 AND dead = 0
  (BANK's exact seed-7 headline within the arm's own re-realization
  noise). Verdict: **batch-granularity territory** — the segregated-
  stores *claim* retires (the stores did nothing the split batches
  didn't); mini-batch/class-slice scheduling is the operative lesson and
  replaces banks as the candidate primitive. Consequence named now:
  the queue's next item becomes mini-batch scheduling variants, not
  finer banks.
- **P2 — SPLIT ≈ CTRL (segregation wins):** held ≤ 3 AND dead ≥ 1
  (CTRL's degradation returns once the class deltas land in one store).
  Verdict: **segregated stores get the credit** — banks still DEFERRED
  at n=4 axis level (the n=3 sweep's axis weakness stands) but the
  mechanism attribution is firm; the next-right-experiment then is the
  bigger-substrate teach-port with bank scaffolding, where the
  interference measured here may not transfer.
- **P3 — SPLIT diverges from both** (e.g. held = 4 with a dead slot,
  held ≤ 3 with dead = 0, or an aggregate far outside
  [CTRL, BANK] in maggregate terms): **anomalous-stated** — the
  mechanism is something else entirely (named now: class-to-bank
  mapping feedback: *what content a call trains on before vs after the
  probe, not mere slice size*); the arm's foreign instrumentation reads
  (below) are the assignment surface for this case.

No fourth option; grey outcomes land in P3 and are named.

## 2. Instrumentation (declared for this arm)

The item-13 foreign-probe flag runs with a **reinterpretation declared
here** (a shared store has no foreign bank to switch into): per
shield-phase session, `r_own(item)` = margin probed **right after the
item's own-class slice-consolidate finished** (mid-session, slice order
[fact, style] — fact items' r_own is their before-style-call read);
`r_off(item)` = margin **after the full session** (both slices done;
this is also the main retention snapshot). The own↔off gap per session
per item is the **last-writer-wins asymmetry** inside a shared store —
exactly the signal the attribution question needs. Coverage: sessions
S7..S12 (no S6 entry row — documented difference from the item-13 bank
coverage). Keys preserved for dashboard parity: `foreign_margins` +
`isolation_per_probe` in the JSON.

## 3. Verdict lines (as run)

*(pending)*

## 4. per-item table: CTRL vs BANK vs SPLIT (seed 7)

*(pending)*

## 5. Verdict + attributive consequence

*(pending)*

## Reproduction

```bash
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_banked_adapters.py --arm split --seed 7 \
  --foreign-probes --workdir lab/teach_banks_split_out \
  | tee lab/logs/teach_split.log

# entry calibrations (2 runs; bit-identical to each other AND to the
# committed series S1 post-teach margin):
python3 -u lab/teach_banked_adapters.py --arm split --only fav_show --seed 7 --workdir <fresh A>
python3 -u lab/teach_banked_adapters.py --arm split --only fav_show --seed 7 --workdir <fresh B>
```

Fresh workdir required. CPU only, fp32 (I7). One fallback max; failures
to `lab/FAILURES.md`.
