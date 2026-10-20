# TEACH_BANKED_ADAPTERS — class-segregated adapter banks (BANK) vs shared-LoRA (CTRL)

**STATUS: IN RUN — calibrations passed (bit-identical ×2 + bit-match the
committed series S1); CTRL arm running.** Queue item #11 of
`lab/TEACH_PIPELINE.md`, run 2026-10-20 overnight on the laptop, CPU only,
fp32. Staged after the shield arm's explicitly NEGATIVE verdict
(`lab/TEACH_SHIELD.md`): the strongest possible *service* policy at one
shared adapter missed the trigger bar (2/6 held, 1 dead, mean 0.495 vs bar
4/6 / 0 / 0.581) with the mechanism read as batch-internal cross-item
interference no queue can schedule out, and style-concentrated residual
loss. This arm moves the isolation from the batch level to the **weight**
level: fact and style content write into separate LoRA deltas, so a style
session's gradient cannot overwrite fact margins by construction.

Harness: `lab/teach_banked_adapters.py` (same import chain as the series;
fixed loaders; frozen knobs). Results: `lab/teach_banks_ctrl_out/teach_banks_ctrl.json`,
`lab/teach_banks_bank_out/teach_banks_bank.json`; logs
`lab/logs/teach_banks_{ctrl,bank}.log`; calibration workdirs
`lab/teach_banks_calib/{A,B}`.

## 0. Determinism verification (done before arming)

Two identical BANK-mode calibration teaches (`--arm bank --only fav_show
--seed 7`, fresh workdirs, sequential): post-teach margins **bit-identical
to each other**, and — because fav_show's calibration batch is the series
S1 batch verbatim (fact bank, fresh model, empty store, seed 7, frozen
knobs) — both bit-match the **committed series S1 margins**:

| run | baseline margin | post-teach margin |
|---|---|---|
| calib A (bank mode) | +1.7177737951278687 | **+4.4509382247924805** |
| calib B (bank mode) | +1.7177737951278687 | **+4.4509382247924805** |
| committed cadence S1 | +1.7177737951278687 | +4.4509382247924805 |

CTRL's chain determinism is proven in-run (the stronger check): its
carried-item baselines + S1..S3 snapshots run under the verbatim cadence
session runner and are asserted bit-equal to the committed cadence JSON
(CADENCE-IDENT subset; S4+ diverges by the declared library amendment).

## 1. Design (documented before running)

| item | value |
|---|---|
| checkpoint | `lab/imported/vk_sft_r3/step_1800/chat_model` — identical to the whole series |
| runtime | laptop CPU, fp32, 12 threads; tier-2 Immune never leaves CPU (I7) |
| knobs | LoRA rank 8, alpha 16, lr 2e-3, kl_coef 0.05, 24 steps/session, seed 7 on every consolidate call — frozen |
| library | fav_show (F), math_words (S), tea_tin (F), **ordinal_words (S, new slot 4)**, pet_dog (F), greet_rule (S) |
| amendment | **metric_units REMOVED, formally** — its × variant-1 pair is 0-for-5 across five arms (trigger −0.325, aged-v1 −0.291, v2 −0.330, v3 −0.203, shield −0.142) with no positive counter-example; the S4 style slot is refilled by ordinal_words ("always write ordinals as words, never digits"), a digit↔word contrast in the measured-good math_words family |
| teach phase | S1–S6, verbatim cadence teach packs (+ chained replay of the previous item), ordinal pack new (2 corrections + 1 approval = 5 triples, as always) |
| later phase | S7–S12 — **shield protocol verbatim** (scheduling is dead): every alive item (r ≥ 0.10 per-probe, burial doctrine from aged-v1) gets exactly ONE approve-only shield triple (pool correction phrasing) per session |
| poison / fallback / tier-2 | unchanged canary per serving store; one-fallback policy; one Immune merge verdict per arm |

### Arms (internal A/B — same sessions, same batches, same seed)

**CTRL** — one shared LoRA (`adapters/alice/delta.pt`), the teach_shield
protocol on the amended library. In-run CADENCE-IDENT subset check
(carried items, baselines + S1..S3) against the committed cadence run.

**BANK** — per-class stores `adapters/alice/fact/delta.pt` +
`adapters/alice/style/delta.pt` (user ids with `/` are path-tolerated by
the store). Fixed bank order [fact, style] per session; fixed rule:

- teach session: teach pack → the taught item's class bank; chained
  replay → the REPLAYED item's class bank (extends the same log when
  classes coincide). A bank with no class items that session is skipped
  (e.g. style bank at S1).
- shield session: filler slice (SHARED strings into both banks — the
  class-neutral control interference, documented) + one shield triple per
  alive item of that class; per-bank filler count
  `max(5, ceil(c_b/0.30) − c_b)` keeps each bank's shield share ≤ 30%.
- each consolidate loads its own bank's incumbent (deltas are complete
  A/B files — applying replaces state exactly); a bank's FIRST call with
  no incumbent on disk first runs **seed_fresh_branch**: re-pin
  `torch.manual_seed(seed)`, re-draw `lora_A` exactly as a fresh attach
  would (`randn(shape)/rank` in `lora_modules()` iteration order, which
  coincides with attach's DFS wrap order), zero `lora_B` — otherwise a
  virgin bank would start training from the other bank's live A/B
  (contamination by construction; the fix restores the series'
  fresh-call initial conditions).
- probes: the serving layer applies the OWNING bank per class and probes
  that class's items (a bank with no delta yet = raw base; B=0 ⇒
  functionally base). The horizon adds the **isolation read**: every item
  also probed under the FOREIGN bank (r_own vs r_off — the direct measure
  of class-content leakage across the split).
- poison canary once per bank; tier-2 verdict on the **fact bank only**
  (the segregation-benefit question is fact margins protected from style
  writes; style-bank verdict skipped on wall grounds, documented).

### Success criteria (declared before running)

The trigger bar (4/6 held ≥ 0.50, 0 dead r ≤ 0.06, mean ≥ 0.581) stays
the series reference row, with the caveat that the amended library
differs in slot 4 (reported as reference, not like-for-like):

VERDICT "architecture wins" iff ALL of:
(a) BANK beats CTRL internally on held count,
(b) BANK beats CTRL internally on dead count,
(c) BANK ties-or-beats the trigger bar on held + dead.

Any miss = honest negative with mechanism (selector ambiguity, filler
leakage, 2-triple bank-log variance, virgin-branch state).

## 2. Verdict lines (as run)

*(pending CTRL horizon, then BANK)*

## 3. per-item table: BANK vs CTRL vs trigger bar

*(pending)*

## 4. Isolation read (r_own vs r_off, BANK horizon)

*(pending)*

## 5. Verdict + consequence

*(pending)*

## Reproduction

```bash
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_banked_adapters.py --arm ctrl \
  --workdir lab/teach_banks_ctrl_out | tee lab/logs/teach_banks_ctrl.log
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/teach_banked_adapters.py --arm bank \
  --workdir lab/teach_banks_bank_out | tee lab/logs/teach_banks_bank.log

# entry determinism calibration (2 runs; must bit-match each other AND
# the committed series S1 fav_show margins):
python3 -u lab/teach_banked_adapters.py --arm bank --only fav_show --seed 7 --workdir <fresh A>
python3 -u lab/teach_banked_adapters.py --arm bank --only fav_show --seed 7 --workdir <fresh B>
```

Fresh workdir required per run (append-only session logs, invariant 10;
adapter deltas and `lab/teach_banks_*_out/`, `lab/teach_banks_calib/` are
weights and are not committed).
