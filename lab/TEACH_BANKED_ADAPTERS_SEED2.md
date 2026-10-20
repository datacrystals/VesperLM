# TEACH_BANKED_ADAPTERS_SEED2 — seed-2 replication of the BANK-vs-CTRL architecture win

Adoption gate from `lab/TEACH_PIPELINE.md` item 12(c), run 2026-10-20 on
the laptop, CPU only, fp32. The seed-1 arm (`lab/TEACH_BANKED_ADAPTERS.md`)
returned **BANK 4/6 held / 0 dead / 0.729 vs CTRL 3/6 / 1 dead / 0.772** —
win declared on held + dead per the pre-registered criteria, with the
mechanism read that class-segregated adapter stores kill cross-class
interference at the weight level (fact within-phase decay −0.938 nats/sess
CTRL → +0.032 BANK) and the caveats n=1 + style-bank intra-class residual.
Item 12 staged three follow-ups; this is the cheap decisive one: **same
everything, new seed family**. Harness: `lab/teach_banked_adapters.py`
VERBATIM (script identical to the seed-1 run — the only delta is
`--seed 43`; no code change, no knob change, no library change).

**Seed family 2 = seed 43 on every consolidate call and every virgin-bank
re-seed** (seed-1 was 7; documented, singular — no seed mixing anywhere in
the run). Everything else frozen: 6-item library
(fav_show/math_words/tea_tin/ordinal_words at S4/pet_dog/greet_rule),
knobs (rank 8, alpha 16, lr 2e-3, kl 0.05, 24 steps), checkpoint
`lab/imported/vk_sft_r3/step_1800/chat_model`, shield later-phase,
burial doctrine, poison canary per store, one tier-2 verdict per arm.

Results: `lab/teach_banks2_ctrl_out/teach_banks_ctrl.json`,
`lab/teach_banks2_bank_out/teach_banks_bank.json`; logs
`lab/logs/teach_banks2_{ctrl,bank}.log`; calibration workdirs
`lab/teach_banks2_calib/{A,B}`.

## 0. Entry determinism (done before arming) + the seed-2 fixture

Two identical BANK-mode calibration teaches (`--arm bank --only fav_show
--seed 43`, fresh workdirs, sequential) must be **bit-identical to each
other**; there is no pre-existing seed-2 fixture (every committed series
calibration is seed 7) — these two runs **create** it, reported below. The
cross-seed anchors that DO have committed fixtures, asserted at arm
startup (seed-independent state: raw-model forward passes):

1. Carried-item `before/` baselines must bit-match the committed series
   baselines (fav_show +1.7177737951278687 chain; the other five items'
   committed values from the cadence JSON).
2. ordinal_words' `before/` baseline (new item in this library family)
   must bit-match the seed-1 arm's recorded baseline.

Teach-loop margins from S1 onward legitimately diverge from every seed-7
run (LoRA-A init is seed-drawn) — that is the point of the replication;
the CADENCE-IDENT subset printout is therefore reported as
declared-divergent, and the determinism claim rests on (1)+(2) + the
bit-identical calibration pair.

## 1. Success / honest read criteria (declared before arming)

- **Replication passes** iff seed-2 BANK wins the same two-way axis vs
  CTRL — BANK held count > CTRL held count AND BANK dead count < CTRL
  dead count — **and** the BANK style-bank residual stays *local*
  (BANK style-class mean r_final ≤ ~0.5, i.e. segregation still does not
  fix intra-class interference on the second draw).
- **Mechanism check**: fact-class within-phase decay flips sign again —
  CTRL fact μ_d < 0 → BANK fact μ_d ≥ ~0 (seed-1: −0.938 → +0.032).
- Any contradiction (seed-2 BANK loses on held, or buries ≥ CTRL's dead
  count, or the fact-decay flips stay same-sign) = **replication fail**,
  verdict reported with mechanism at n=2, and `lab/TEACH_PIPELINE.md`
  item 13 becomes "wider seed replication / design failure of banks".
- Tier-2 per arm: expected ROLLBACKs (the two-tier read is 10-for-10 so
  far in the series; banking does not move merge-grade behaviour).

## 2. Verdict lines (as run)

*(pending)*

## 3. per-item table: seed-1 vs seed-2, BANK vs CTRL

*(pending)*

## 4. Mechanism / decay sign table

*(pending)*

## 5. Replication verdict

*(pending)*

## Reproduction

```bash
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_banked_adapters.py --arm ctrl --seed 43 \
  --workdir lab/teach_banks2_ctrl_out | tee lab/logs/teach_banks2_ctrl.log
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_banked_adapters.py --arm bank --seed 43 \
  --workdir lab/teach_banks2_bank_out | tee lab/logs/teach_banks2_bank.log

# entry calibration (2 runs; bit-identical to each other — creates the
# seed-2 fixture):
python3 -u lab/teach_banked_adapters.py --arm bank --only fav_show --seed 43 --workdir <fresh A>
python3 -u lab/teach_banked_adapters.py --arm bank --only fav_show --seed 43 --workdir <fresh B>
```

Fresh workdir required per run. CPU only, fp32 (Immune never leaves CPU,
I7). One fallback max; failures to `lab/FAILURES.md`.
