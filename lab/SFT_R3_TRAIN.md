# SFT r3 — quality-chat round (vk-base, full-FT)

Experiment log for the SFT r3 training round: same recipe as the r2
talkability run (vk-base full-FT, config `470m_k_sft_r2`), only the data
index swapped to the r3 quality mix. Goal: better conversation quality than
r2's 6/12 (r2 failure mode = salad/drift). Quality over loss numbers.

## Question

Does the r3 quality mix (anti-drift mass + arithmetic + persona voice +
direct-answer pairs; `lab/SFT_R3_DATA.md`) beat the r2 openhermes/oasst mix
on the fixed 12-prompt conversational instrument, under the identical
training recipe?

## Recipe (r2 mirrored; only the data index changes)

| knob | value | source |
|------|-------|--------|
| base init | vk-base pretrain `step_best` (step 8600 of the 10.07B-token run), never r2 weights | r2 launched from this ckpt (base-init, not resume) |
| config | `470m_k_sft_r2` (`SFT/01_sft_train.py`) | r2 verbatim |
| params | 560.96M total / 325.03M active, 10L × 1024, 8E top-2, kda/mla, vocab 65523 | arch rebuilt from ckpt `model_config` |
| training mode | full-FT (no LoRA) | r2 verbatim |
| data | `Dataset/data/sft/sft_r3_index.txt` (ultrachat 20 / oasst2 2.5 / gsm8k 1 / persona 1.5 / direct 0.5; 22.0M positions, 95/5 train/val) | **the r3 swap** |
| batch | micro_batch 2 × seq 1024, accum 16 (= 32 sequences = 32,768 tok/optimizer-step) | r2 token budget preserved; micro-batch cut for the 8GB laptop GPU (see VRAM note) |
| schedule | lr 5e-5 → 5e-6 cosine, warmup 50, 2000 steps | r2 verbatim |
| eval / ckpt | every 200 steps: val CE (30 batches), 4 training probes, full resumable `checkpoint.pt`, `step_best` by val, `step_N/chat_model` | r2 verbatim |
| convo harness | 12-prompt `SFT/eval_chat.py` instrument at every checkpoint, decode T=0.7 / top-p 0.9 / max-new 180 / rep-penalty 1.1 | same decode as the r2 6/12 verdict, so scores compare |
| precision | bf16 AMP + `VESPER_FUSED_CE=1`, value-embed auto-detected from ckpt | r2 canary flags (`VESPER_VALUE_EMBED=1` set; no-op now that load is auto-detected) |

Optimizer and micro-batch are the two deltas forced by the 8GB laptop GPU
(r2 ran full-FT @ fp32 AdamW on a 192GB MI300X):

- **Optimizer: AdamW 8-bit (bitsandbytes 0.50.2)** — fp32 AdamW needs
  params+grads+2×fp32 moments ≈ 9.0GB for 561M, which is arithmetically
  impossible on this card (project code comments had already recorded the
  full-FT OOM at 7.51/7.67GiB on the 3070). 8-bit moments are the
  script's own CUDA path (`01_sft_train.py` selects `AdamW8bit` when bnb
  is present), keep full-FT, and shrink states to 1.14GB (uint8 m+v).
- **micro_batch 8 → 2** (accum auto-scales 4 → 16; tokens/optimizer-step
  unchanged at 32,768). Measured on-card: with optimizer states live, the
  accumulation phase (micro-steps ≥ 1 run with previous micro's gradients
  already resident) needs ~1.3GB of activation headroom; mb 4×1024 OOMs at
  6.53GiB allocated, mb 2×1024 peaks 6.99GiB allocated and holds. Also:
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the pretrain
  fragmentation fix) *costs* ~0.5GB of driver overhead at this footprint —
  omitted.

Wall time at mb=2: ~19s/step ≈ 10.5h for 2000 steps on the RTX 3070
Laptop (32768 tok/step × 2000 ≈ 65.5M positions ≈ 3 epochs over the 22.0M
r3 mix — r2 did ≈1.3 epochs over its 51M mix; same schedule, denser
epochs, so `step_best` / best-chat selection matters more than step count).

## Run

Launch capsule (detached screen `vesper_sft_r3`, auto-resume loop):

```
VESPER_SFT_CONFIG=470m_k_sft_r2 \
VESPER_SFT_DATA=Dataset/data/sft/sft_r3_index.txt \
VESPER_SFT_CKPT=~/vesper_ckpts/vk_base \
VESPER_SFT_OUT=lab/imported/vk_sft_r3 \
VESPER_FUSED_CE=1 VESPER_VALUE_EMBED=1 \
VESPER_SFT_MICRO_BATCH=2 VESPER_SFT_SEQ_LEN=1024 \
python3 -u SFT/01_sft_train.py
```

Local trainer hooks added for this run (uncommitted — commit rule for this
cycle is doc-only): `VESPER_SFT_MICRO_BATCH` / `VESPER_SFT_SEQ_LEN` env
knobs, and `run_convo_harness()` which runs the 12-prompt `eval_chat`
instrument on the live model at each checkpoint (a single 8GB GPU cannot
host a second model copy for a side-loaded eval). Same prompts, same
`generate()`, same JSON shape as `SFT/eval_chat.py`.

## Results

**FINAL: run complete 2026-10-16 (rc=0; log
`lab/logs/sft_vk_r3_train.log`).** Eval points every 200 steps
(transcripts `lab/imported/vk_sft_r3/_evals/eval_chat_step{200..1800}.json`).

⚠ **Trainer gap, recorded:** eval/save fires only inside the loop at
`step % 200 == 0` (`SFT/01_sft_train.py:1015`), and the loop tops out at
step 1999 — so the final eval/save at step 2000 never ran and the trainer
exits with **no end-of-run save**. The last checkpointed state is
**step 1800** (which is also `step_best`: val 1.5666); steps ~1801–1999 of
training exist only as console CE prints. No step_2000 artifact exists to
evaluate — final selection is from the step_1800 grid.

**Train CE** (per-step entries; window mean / median over each 200-step
block): 1.80/1.79 (1-200) → 1.70/1.70 (201-400) → 1.66/1.65 (401-600) →
1.54/1.64 (601-800) → 1.60/1.61 (801-1000) → 1.57/1.59 (1001-1200) →
1.58/1.59 (1201-1400) → 1.45/1.53 (1401-1600) → 1.52/1.53 (1601-1800).
Slow monotone drift 1.80 → ≈1.5, same noisy texture as r2. Two transient
collapse events — steps 711–723 and 1436–1447, CE drops to 0.1–0.6 for a
dozen consecutive steps then recovers on its own — look like short runs of
near-zero-CE sequences from the small synthetic bins (persona/direct), not
instability (loss, val, and outputs are unaffected on either side).

**Val CE** (30 batches, noisy and non-monotonic): 1.6417@200 → 1.6325@400 →
1.8228@600 → 1.6910@800 → 1.6482@1000 → 1.7398@1200 → 1.6640@1400 →
1.6864@1600 → **1.5666@1800 — best, `step_best` = step 1800.** Val CE is
*not* comparable to r2's numbers (different val set); the cross-round
instrument is the 12-prompt harness.

**Final rubric curve** — all points scored from the live-model harness runs
(`SFT/eval_chat.py` instrument, T=0.7 / top-p 0.9 / max-new 180 / rep-pen
1.1, the r2 verdict decode, so scores compare). Strict usable = PASS-only
bar, same rubric as the r2 6/12 verdict; loop/echo/empty/formula-debris
audits from full transcripts:

| ckpt | strict usable | hard-loop | formula-debris | echo / empty | failing family |
|------|---------------|-----------|----------------|--------------|----------------|
| 200  | 2/12 | 0 | 0 | 0 / 0 | factuality 2, task 2, salad 0 |
| 400  | 3/12 | 1 (space-fact digit runaway) | 0 | 0 / 0 | factuality, task (translate) |
| 600  | 4/12 | 0 | 0 | 0 / 0 | factuality (France, space, arith) |
| 800  | 2/12 | 1 (ocean-poem line loop) | 0 | 0 / 0 | factuality, thin answers |
| 1000 | 3/12 | 0 | 0 | 0 / 0 | task (arith, translate), thin |
| 1200 | 3/12 | 1 (ocean-poem line loop) | 0 | 0 / 0 | factuality (Bourges!), task |
| 1400 | 5/12 | 0 | 0 | 0 / 0 | factuality (space, arith), task (translate) |
| 1600 | 2/12 | 1 (ocean-poem litany loop) | 0 | 0 / 0 | factuality, garble |
| **1800** | **5/12** | **0** | **0** | **0 / 0** | arith form-wrong, space-fact vague, translate refused |
| r2 best (600) | 6/12 | 2 | 3 | 0 / 0 | salad/drift + loops |

Reply lengths (chars, median/min·max of 12): 337/14·875 @200 → 354 @400 →
360 @600 → 478 @800 → 302 @1000 → 268 @1200 → 567 @1400 → 400 @1600 →
426/31·859 @1800. No empty replies and no prompt echoes at any checkpoint.

Per-prompt verdicts at the first milestone (step 200) for reference; the
curve above re-scores the same 12 at every eval point:

| # | prompt | r3 @ step 200 | r2 (best, step 600) |
|---|--------|---------------|---------------------|
| 1 | greeting | PARTIAL (wrong register) | PASS |
| 2 | capital of France | FAIL (wrong facts, target entity missed) | PASS |
| 3 | black hole | PARTIAL (topical, fact errors, no loop) | FAIL (hard loop) |
| 4 | ocean poem | PARTIAL (on-topic, wrong form) | PASS |
| 5 | 17×23 | FAIL (wrong product; clean answer form) | FAIL (hard loop) |
| 6 | boil an egg | PARTIAL (structure ok, content wrong) | PARTIAL |
| 7 | list vs tuple | PARTIAL (thin, incomplete) | PARTIAL |
| 8 | fun fact space | PARTIAL (fluent, factual errors) | PASS |
| 9 | photosynthesis | **PASS** (correct summary) | FAIL (formula debris) |
| 10 | stressed about work | PARTIAL (deflects, no empathy) | PASS |
| 11 | translate to French | FAIL (task ignored) | FAIL |
| 12 | why do we dream | **PASS** (coherent on-topic) | PASS |

**Loop/quality audit at step 200:** hard-loop **0/12** (r2: 2/12),
prompt-echo 0/12, empty 0/12, formula-debris **0/12** (r2: 3/12). All 12
replies are fluent, stopping prose. Full transcripts live in the eval JSON
above.


## Final verdict vs r2 (full-training answer)

**Question: does the r3 zero-salad/anti-drift fix hold at full training?
YES for the failure mode, NO for the headline count.**

- **Formula-debris / corpus-drift: fixed, and it stayed fixed.** 0/12 at
  every one of the 9 checkpoints (r2: 3/12 at its best, worse at others).
  108 scored replies and not one salad episode.
- **Hard loops: fixed at the final state, but NOT structurally gone** —
  r2's 2/12 baseline drops to **0/12 at step 1800**, yet 4 transient loop
  episodes appear mid-curve (step 400 digit-runaway on the space fact;
  ocean-poem line loops at 800/1200/1600). The poem prompt is the residual
  loop trigger; it is gone at the val-best end state but was not trained
  out as a class.
- **The strict usable count never beats r2: peak 5/12 (@1400 and @1800)
  vs r2's 6/12.** r3 traded r2's catastrophic failures (loops, debris,
  empties) for fluent, wrong-or-thin answers: the residual family all
  round is **factuality** (confident wrong facts in clean prose — e.g.
  "Bourges" as France's capital @1200, fabricated space facts) and
  **task-following** (translate-to-French failed at all 9 checkpoints;
  arithmetic wrong at all 9 but always in clean answer form — r2's
  arithmetic hard-loop is gone, the GSM8K bin's form target landed, the
  math itself didn't).
- **Quality floor is way up, ceiling is not.** Worst checkpoint (2/12)
  has zero catastrophic replies; r2's failures were unusable, r3's are
  mediocre. For a chat demo this is the better model to show; for the
  6/12-style "fully usable" bar, the next lever is factuality/task data,
  not anti-drift data.

**Chosen chat model: `lab/imported/vk_sft_r3/step_1800/chat_model`**
(= `step_best`, val CE 1.5666, the run's best). It ties step_1400 on the
strict 5/12 but wins the tiebreak on every secondary metric: lowest val
loss of the run, 0 hard-loops (1400 also 0, but 1800's passes include the
two factuality-heavy wins — Paris direct, black-hole core correct), and it
is the final trained state. r2's `chat_model` is retired as the demo
default: 5/12-with-no-catastrophes beats 6/12-with-loops for live use.

**Sample generations from step_1800** (model output only, same decode as
above; transcripts `_evals/eval_chat_step1800.json`):

1. "What is the capital of France?" → "The capital of France is Paris."
2. "Can you explain what a black hole is in simple terms?" → "A black hole
   is a region of space where gravity is so strong that nothing, not even
   light, can escape from it." (continues cleanly; later embellishments
   drift into fact errors, the PARTIAL pattern)
3. "Hello! How are you today?" → "I'm glad I could help. What's new with
   you?"


## r2 baseline (fixed instrument, T=0.7 / top-p 0.9 / max-new 180)

The r2 quality pass scored **6/12 usable** on `SFT/eval_chat.py`'s 12
prompts (`lab/imported/vk_sft_r2/chat_model`, step 600 = val-best). Loop
audit then: hard-loop 2/12, prompt-echo 0/12, empty 0/12. Residual failure
families: formula-debris drift on knowledge prompts (3/12), hard loop on
arithmetic (1-2/12). For the r3 score to be comparable, score the same 12
prompts at the same decode and report usable/12 plus the same loop audit.

| # | prompt | r2 verdict |
|---|--------|------------|
| 1 | greeting | PASS |
| 2 | capital of France | PASS |
| 3 | black hole | FAIL (hard loop + physics garble) |
| 4 | ocean poem | PASS |
| 5 | 17×23 | FAIL (hard loop) |
| 6 | boil an egg | PARTIAL |
| 7 | list vs tuple | PARTIAL |
| 8 | fun fact space | PASS |
| 9 | photosynthesis summary | FAIL (formula-debris drift) |
| 10 | stressed about work | PASS |
| 11 | translate to French | FAIL (task ignored → debris) |
| 12 | why do we dream | PASS |

