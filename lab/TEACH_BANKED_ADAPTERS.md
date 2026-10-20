# TEACH_BANKED_ADAPTERS — class-segregated adapter banks (BANK) vs shared-LoRA (CTRL)

**STATUS: COMPLETE — VERDICT: architecture wins on the pre-declared
criteria.** BANK beats CTRL internally on held (4/6 vs 3/6) and on dead
(0 vs 1), and ties the trigger bar on held + dead (4/6, 0); both arms
pass their poison canaries; both tier-2 merges ROLLBACK (9th and 10th
readings). Run 2026-10-20 overnight on the laptop, CPU only, fp32
(CTRL wall 1731 s, BANK wall 2285 s). Honest caveats in §5: CTRL's mean
(0.772) edges BANK's (0.729) — the win set was declared on held + dead,
not mean; intra-class interference persists inside each bank (style bank
carries 3 items and loses the class-mean read); tier-2 still vetoes both
(nothing about banking moves merge-grade behaviour).

Queue item #11 of `lab/TEACH_PIPELINE.md`. Staged after the shield arm's
explicitly NEGATIVE verdict (`lab/TEACH_SHIELD.md`): the strongest
possible *service* policy at one shared adapter missed the trigger bar
with the mechanism read as batch-internal cross-item interference no
queue can schedule out, and style-concentrated residual loss. This arm
moved the isolation from the batch level to the **weight** level: fact
and style content write into separate LoRA deltas, so a style session's
gradient cannot overwrite fact margins by construction.

Harness: `lab/teach_banked_adapters.py` (same import chain as the
series; fixed loaders; frozen knobs). Results:
`lab/teach_banks_ctrl_out/teach_banks_ctrl.json`,
`lab/teach_banks_bank_out/teach_banks_bank.json`; logs
`lab/logs/teach_banks_{ctrl,bank}.log`; calibration workdirs
`lab/teach_banks_calib/{A,B}`.

## 0. Determinism verification (done before arming)

Two identical BANK-mode calibration teaches (`--arm bank --only fav_show
--seed 7`, fresh workdirs, sequential): post-teach margins **bit-identical
to each other**, and — fav_show's calibration batch being the series S1
batch verbatim (fact bank, fresh model, empty store, seed 7) — both
**bit-match the committed series S1 margins**:

| run | baseline margin | post-teach margin |
|---|---|---|
| calib A (bank mode) | +1.7177737951278687 | **+4.4509382247924805** |
| calib B (bank mode) | +1.7177737951278687 | **+4.4509382247924805** |
| committed cadence S1 | +1.7177737951278687 | +4.4509382247924805 |

In-run chain checks:

```
CTRL CADENCE-IDENT (subset: carried items, baselines + S1..S3):
  max |margin Δ| = 0.000e+00 (BIT-IDENTICAL; worst before/fav_show)
BANK CADENCE-IDENT (same subset):
  max |margin Δ| = 1.009e+01 (DIVERGED; worst s3/tea_tin)
  [expected + declared: BANK is a new baseline family from S1 —
   class-filtered batches re-realize every downstream margin]
```

CTRL's carried gains bit-match the series (fav_show +2.733, math_words
+5.237, tea_tin +13.308 — identical to all six prior runs); pet_dog's
gain re-realized to +17.353 (vs committed +4.898) and greet_rule to
+5.606 (vs +6.038) from the S4 library amendment alone — the S4 slot's
teach content moves downstream acquisition by >12 nats on this library.
BANK gains re-realize again under class-filtering (math_words +12.648,
tea_tin +3.220, pet_dog +4.262, greet_rule +15.168): the A/B read is
BANK-vs-CTRL on the same amended library, with the trigger bar as the
series reference row (its library differs in slot 4 — reference, not
like-for-like).

## 1. Design (documented before running, as run)

| item | value |
|---|---|
| checkpoint | `lab/imported/vk_sft_r3/step_1800/chat_model` — identical to the whole series |
| runtime | laptop CPU, fp32, 12 threads; tier-2 Immune never leaves CPU (I7) |
| knobs | LoRA rank 8, alpha 16, lr 2e-3, kl_coef 0.05, 24 steps/session, seed 7 on every consolidate call — frozen |
| library | fav_show (F), math_words (S), tea_tin (F), **ordinal_words (S, new slot 4)**, pet_dog (F), greet_rule (S) |
| amendment | **metric_units REMOVED, formally** — its × variant-1 pair is 0-for-5 across five arms with no positive counter-example; ordinal_words ("always write ordinals as words, never digits") refills the slot from the measured-good digit↔word family |
| teach phase | S1–S6, verbatim cadence packs (+ chained replay of the previous item); the 3-turn teach pack is all one class by construction |
| later phase | S7–S12 — **shield protocol verbatim** (scheduling is dead): every alive item (r ≥ 0.10 per-probe, burial doctrine from aged-v1) gets exactly ONE approve-only shield triple per session |
| poison / fallback / tier-2 | unchanged canary, once per serving store; one-fallback policy (not exercised); one Immune verdict per arm |

### Arms (internal A/B — same sessions, same library, same seed)

**CTRL** — one shared LoRA (`adapters/alice/delta.pt`), the
teach_shield protocol on the amended library.

**BANK** — per-class stores `adapters/alice/fact/delta.pt` +
`adapters/alice/style/delta.pt` (user ids containing `/` are
path-tolerated by the Hippocampus store). Fixed bank order [fact,
style]; teach pack → the taught item's class bank; chained replay → the
replayed item's class bank; a class-less session bank is skipped (style
bank skips S1). Shield sessions: **filler strings shared into both
banks** (the class-neutral control interference, documented), one shield
triple per alive class item, per-bank filler count
`max(5, ceil(c_b/0.30) − c_b)` (share ≤ 30% per bank, asserted). A
bank's first-ever consolidate first runs **seed_fresh_branch** (re-pin
the RNG, re-draw lora_A as a fresh attach would — `randn(shape)/rank` in
`lora_modules()` order, which coincides with attach's DFS wrap order —
and zero lora_B); without it a virgin bank would start training from the
other bank's live A/B (contamination-by-construction, fixed ahead of
running). Probes: serving layer applies the OWNING bank per class (a
bank with no delta → raw base); the horizon adds the isolation read
(r_own vs r_off under both banks). Poison canary once per bank; tier-2
on the fact bank only (the segregation-benefit question; style-bank
verdict skipped on wall grounds, documented).

## 2. Verdict lines (as run; generation tails trimmed)

```
CTRL CADENCE-IDENT subset: max |margin Δ| = 0.000e+00 (BIT-IDENTICAL)
CTRL GAINS g_i: fav_show +2.733  math_words +5.237  tea_tin +13.308  ordinal_words +0.889  pet_dog +17.353  greet_rule +5.606
CTRL 12/12 sessions PROMOTE, 0 tripwires; batches 20/17/17/17/17/14 triples (shield shares 30/29%);
      burial trace: ordinal_words probe-after-S10 (r -0.042), re-entered after S11
CTRL HORIZON: 3/6 items r_final >= 0.50; 1/6 r_final <= 0.06 (dead slot: pet_dog -0.004); mean r_final 0.772; min -0.004
CTRL POISON alice: ROLLBACK — target collapse 100% | incumbent adapter intact: True
CTRL MERGE alice vs raw base (immune-tier2): ROLLBACK — refuse-color-seven 0.408 -> 0.132, ho-refuse-weigh 0.397 -> 0.000 |
      aggregate 0.3156 -> 0.1245, protected 0.255 -> 0.194, drift 1.000 label=fail

BANK CADENCE-IDENT subset: DIVERGED at s2+ (declared: new baseline family from S1)
BANK GAINS g_i: fav_show +2.733  math_words +12.648  tea_tin +3.220  ordinal_words +1.572  pet_dog +4.262  greet_rule +15.168
BANK 12/12 session-heads PROMOTE across banks, 0 tripwires; tiny bank logs (2-triple replay banks) pass at
      top_resp_frac 0.50 — the exact allowed boundary, reported as the observed thin-batch limit;
      burial trace: ordinal_words probe-after-S6 (r -2.29), fav_show after-S8 (r -0.24), ordinal after-S9 (r 0.037) — all re-entered
BANK HORIZON: 4/6 items r_final >= 0.50; 0/6 dead; mean r_final 0.729; min 0.240
BANK POISON alice/fact: ROLLBACK intact: True | POISON alice/style: ROLLBACK intact: True
BANK MERGE alice/fact vs raw base (immune-tier2): ROLLBACK — fact-paris 1.000 -> 0.000, fact-tokyo 1.000 -> 0.000 |
      aggregate 0.3156 -> 0.0966, protected 0.255 -> 0.028, drift 1.000 label=fail
```

## 3. per-item r table — CTRL vs BANK (internal A/B) vs trigger bar reference

r_i(k) = (m_i(k) − m_i(0)) / g_i, computed per arm with its own gains
(both arms' gains re-realized by design, §0). CTRL/BANK are like-for-like
(same library, seed, sessions); the trigger column is the committed
series reference on the ORIGINAL library (slot 4 was metric_units —
reference, not like-for-like).

| item | bank | trigger ref r_final | CTRL r_final | **BANK r_final** | BANK − CTRL |
|---|---|---|---|---|---|
| fav_show (F) | fact | 0.96 | 1.42 | **1.05** | −0.37 |
| math_words (S) | style | 0.66 | 0.86 | **0.26** | −0.60 |
| tea_tin (F) | fact | 0.17 | 0.32 | **1.46** | **+1.14** |
| metric_units/ordinal slot (S) | style | 0.69 (metric) | 1.81 (ordinal) | **0.54** (ordinal) | −1.27 |
| pet_dog (F) | fact | 0.50 | **−0.00 (dead)** | **0.83** | **+0.83** |
| greet_rule (S) | style | 0.50 | 0.23 | **0.24** | +0.01 |

| metric | trigger bar | CTRL | **BANK** | verdict |
|---|---|---|---|---|
| held r ≥ 0.50 | ≥ 4/6 | 3/6 (fav, math, ordinal) | **4/6 (fav, tea, ordinal, pet)** | criterion (a): BANK > CTRL ✓; bar criterion (c): tie ✓ |
| dead r ≤ 0.06 | 0 | 1 (pet_dog −0.004) | **0** | criterion (b): BANK < CTRL ✓; bar: tie ✓ |
| mean r | ≥ 0.581 | **0.772** | **0.729** | both beat the bar; CTRL > BANK (honest note §5) |
| min r | 0.17 (ref) | −0.004 | **0.240** | BANK floor far better |
| shield contacts | — | 29 | 29 | equal coverage |
| tier-2 agg (inc 0.3156) | ROLLBACK | ROLLBACK 0.1245 | ROLLBACK 0.0966 | 9th + 10th readings |

## 4. Isolation read (BANK horizon: r under owning bank vs foreign bank)

| item | owning bank | r_own | r_off | read |
|---|---|---|---|---|
| fav_show | fact | +1.05 | +0.41 | content stays home |
| math_words | style | +0.26 | +0.17 | same direction, small |
| tea_tin | fact | **+1.46** | +0.18 | content stays home |
| ordinal_words | style | +0.54 | **−0.21** | foreign bank slightly negative (the fact bank's writes cost the style margin at probe — a within-run graded interference read) |
| pet_dog | fact | +0.83 | +0.34 | content stays home |
| greet_rule | style | +0.24 | +0.02 | neutral |

Foreign reads sit at bystander-oscillation scale (the series' un-served
single-session wobble is ±0.1–0.3 r); the style bank never received fact
content yet shows small positive fact-item reads (shared chat-pattern
transfer at probe, not trained leakage — no fact turn ever wrote to it).
The direction that matters: **r_own ≫ r_off on every flagship win** —
the margins the banks hold are genuinely the ones that bank was taught.

## 5. Verdict + consequence

**VERDICT: architecture wins** — all three pre-declared criteria met on
one draw: (a) held 4/6 BANK > 3/6 CTRL; (b) dead 0 BANK < 1 CTRL; (c)
held + dead tie the trigger bar (4/6, 0). The rescue events are
mechanical, not cosmetic: pet_dog (−0.004 dead in CTRL → +0.83
held in BANK) and tea_tin (0.32 → 1.46) are exactly the items the
shared adapter's cross-class interference was spending; the CLASS
dynamics flip confirms the mechanism — CTRL's within-phase fact decay
(µ_d −0.938/session) becomes BANK's fact decay +0.032/session (style
bank isolation removes the cross-class chip), while intra-class
interference persists and now shows where banking does NOT reach: the
style bank houses 3 items and its mean r_final reads 0.35 (**intra-class
interference is the measured residual**, math_words 0.86 → 0.26 in BANK).

Honest limits:

- **CTRL's mean (0.772) beats BANK's (0.729).** The win set was declared
  on held + dead (the bar's bimodal criteria), not on mean; the mean
  split is driven by ordinal_words' small gain (0.889 CTRL / 1.572 BANK),
  which amplifies r swings ±0.5+ per margin nat on both sides. Reported,
  not hidden.
- n = 1 per arm under the pinned seed; the A/B is internally clean (same
  library/sessions/seed; CTRL bit-checks the committed chain), but
  seed/schedule variance is one draw. The gains re-realization (§0) is
  part of the architecture (a bank necessarily trains on a filtered
  batch) — r-normalized comparison keeps it measurable, and is stated.
- Tier-2 is unmoved by banking: the fact-bank merge still tripped both
  protected fact probes (protected 0.255 → 0.028, worse than CTRL's
  0.194 — full-delta merges into the trunk read the accumulated write
  volume, not the loop's local ordering): **adapter-banking solves the
  teach-loop interference problem, not the checkpoint-merge problem** —
  the two-tier read holds a 9th and 10th time.
- Both arms' tiny bank logs (2-triple replay batches) passed the stub at
  the exact 0.50 top-response boundary — allowed (≤ 0.5) but noted as the
  thin-batch limit of the class-filtered design.
- The horizon is again a snapshot (BANK ordinal 0.54 from −2.29 at S6;
  burial bounce-backs happened three times). The S13+ equilibration
  question stays queued.

**Consequence (staged as item #12).** The chained falsification is
closed: scheduling variants failed four times, breadth-shielding failed
the bar, and the one winner at ONE capacity word is class segregation of
the adapter store. Next steps in order of evidence value: (1) S13+
equilibration horizon-discipline arm *on the BANK architecture* (the
freeze caveat is now the oldest open read; run on the winner);
(2) intra-class refinement candidate — the style bank carries 3 items
and loses intra-class (mean 0.35): either per-item banks (max
segregation, wall-prohibitive) or the measured-diagnosis step of
triangulating WHICH style-item pair interferes; (3) adoption bar before
any of that sticks: replicate BANK once at a second seed — the criteria
were declared on one draw. ordinal_words graduates from "slot filler" to
a measured library member (gained +1.57/+0.89 in family; r reads coarse
but held 2/2 arms).

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
weights and are not committed). Both arms exited 0; no tripwire event, no
fallback, no `lab/FAILURES.md` entry.
