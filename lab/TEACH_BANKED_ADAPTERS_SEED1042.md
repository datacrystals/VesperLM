# TEACH_BANKED_ADAPTERS_SEED1042 — 1042-family replication + item-13 instrumentation

**STATUS: IN RUN — calibration fixture created; arms arming.** Queue item
#13 of `lab/TEACH_PIPELINE.md`, run 2026-10-20 on the laptop, CPU only,
fp32. The seed-2 gate (`lab/TEACH_BANKED_ADAPTERS_SEED2.md`) closed
NEGATIVE on its declared conjunctive read: the headline direction
replicated 2-for-2 (BANK held > CTRL held, BANK dead < CTRL dead at both
seeds; pet_dog dead-in-CTRL → held-in-BANK motif repeats) but both
mechanism regulators failed (fact-class decay flip stayed same-sign;
style-residual locale moved and inverted class ordering). Seed sweep so
far moves teach margins up to 3× by family. Item 13 = this wider seed
run + per-bank cross/intra-class drift instrumentation.

**Seed family 3 = `--seed 1042`** on every consolidate call and every
virgin-bank re-seed (seed-1: 7, seed-2: 43; documented; no seed mixing
anywhere in the run). Harness `lab/teach_banked_adapters.py` — the
seed-1/2 runs were flag-free; this run adds ONE same-script instrumentation
flag, `--foreign-probes` (BANK arm only, off by default; CTRL runs
verbatim flag-free). Everything else frozen: library, knobs (rank 8,
alpha 16, lr 2e-3, kl 0.05, 24 steps), checkpoint
`lab/imported/vk_sft_r3/step_1800/chat_model`, shield later-phase,
burial doctrine, poison per store, one tier-2 verdict per arm.

**Instrumentation delta (same-script flag, the item-13 queue ask):**
with `--foreign-probes`, every shield-phase probe (entry probe after S6,
then after S7..S12) records every item's margin under the FOREIGN bank
alongside the owning-bank one, so the JSON gains `foreign_margins` and a
per-probe `isolation_per_probe [{session, item, r_own, r_off}]` table —
the cross-class drift story becomes quantitative per session, not just a
horizon read. r_off normalization: same m0 and the item's OWN arm gain g
(documented). Model-state discipline: every later consolidate/probe
applies its own bank, so the foreign reads cannot contaminate the run;
the own-bank flow is bit-path-identical to the flag-free harness.

Results: `lab/teach_banks3_ctrl_out/teach_banks_ctrl.json`,
`lab/teach_banks3_bank_out/teach_banks_bank.json`; logs
`lab/logs/teach_banks3_{ctrl,bank}.log`; calibration workdirs
`lab/teach_banks3_calib/{A,B}`.

## 0. Entry determinism (the documented fixture ritual)

Two identical BANK-mode calibrations (`--arm bank --only fav_show
--seed 1042`, fresh workdirs, sequential) must be **bit-identical to
each other** — these create the seed-1042 fixture. Cross-seed anchors
(seed-independent raw-model probes) asserted at arm startup in BOTH
arms: carried-item baselines bit-match the committed series values
(fav_show +1.7177737951278687 chain; five items from the cadence JSON);
ordinal baseline bit-matches the seed-1/arm recording
(+0.07648158073425293). CADENCE-IDENT prints diverge by design (the
seed is the variable); the determinism claim is anchors + the
bit-identical pair.

| run | baseline margin | post-teach margin |
|---|---|---|
| calib A (bank, seed 1042) | +1.7177737951278687 | **+13.200785398483276** |
| calib B (bank, seed 1042) | +1.7177737951278687 | **+13.200785398483276** |

## 1. The declared n=3 verdict rule-set (before any run output)

Registered state going in: seed-1 (BANK 4/6 / 0 dead / 0.729 vs CTRL
3/6 / 1 / 0.772), seed-2 (BANK 3/6 / 0 / 0.536 vs CTRL 2/6 / 1 / 0.407).

- **Direction record** (per seed): BANK held > CTRL held AND BANK dead <
  CTRL dead. Currently 2/2. Needed for the rule: **3/3** (seed-1042 must
  keep both axes; a tie on either axis counts as a miss, same as before).
- **Mechanism regulator pass-rate**, per seed, two regulators:
  (i) *fact-flip*: CTRL fact μ_d < 0 → BANK fact μ_d ≥ ~0
  (seed-1 PASS +0.032/−0.938; seed-2 FAIL −0.500/−1.013);
  (ii) *style-local*: BANK style-class mean r_final ≤ ~0.5
  (seed-1 PASS 0.35; seed-2 FAIL 0.56). Each regulator currently 1/2.
  Rule: **each regulator must pass on ≥ 2 of 3 seeds** — i.e., seed-1042
  must PASS BOTH to lift each to 2/3. (The regulator values are recomputed
  verbatim from each arm's retention rows, same aggregation as prior
  docs; the foreign-probe trace is corroborating evidence, not part of
  the pass/fail rule.)
- **VERDICT — ADOPTION CONDITIONAL OK** iff direction 3/3 AND both
  regulators reach ≥2/3. Condition: the style-bank intra-class fix
  (item 12(b) triangulation) proceeds first-class; the win is
  n=3-supported, mechanism-supported.
- **Anything less → DEFER** with labeled confidence and the
  next-right-experiment stated explicitly (labeled honestly: direction
  pass-rate, regulator pass-rates, one-line interpretation).

## 2. Verdict lines (as run)

*(pending)*

## 3. n=3 synthesis: per-item r_final, axes record, regulator pass-rate

*(pending)*

## 4. Instrumentation read: per-probe cross-class drift (BANK-1042)

*(pending)*

## 5. Verdict + consequence

*(pending)*

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

Fresh workdir required per run. CPU only, fp32 (I7). One fallback max;
failures to `lab/FAILURES.md`.
