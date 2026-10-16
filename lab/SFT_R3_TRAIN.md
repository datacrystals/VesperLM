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

**Milestone: step 200 / 2000 (2026-10-16), run in progress** (~19s/step on
the RTX 3070 Laptop, ≈0.06 steps/s; eval every 200 steps). Everything below
is the first checkpoint — the round is not finished; re-score at later
checkpoints, and treat val-best / best-chat as the selection points.

**Loss.** Train CE (per-10-step, noisy as in r2): 1.92@0 → 2.04@10 →
1.90@20 → … → 1.67–1.84@100–200 (window ≈1.75). Val CE (30 batches):
**1.6417 @ 200 — NEW BEST, `step_best` = step 200.** Val CE is *not*
comparable to r2's numbers (different val set: r3 mix vs r2
oasst/openhermes mix); the cross-round instrument is the 12-prompt harness.

**12-prompt conversational harness** (`SFT/eval_chat.py` instrument,
T=0.7 / top-p 0.9 / max-new 180 / rep-pen 1.1 — the r2 verdict decode),
full transcripts: `lab/imported/vk_sft_r3/_evals/eval_chat_step200.json`.

| # | prompt | r3 @ step 200 | r2 (best, step 600) |
|---|--------|---------------|---------------------|
| 1 | greeting | PARTIAL (coherent, non-sequitur "Happy New Year, everyone!") | PASS |
| 2 | capital of France | FAIL (never says Paris; "1.3 billion people" salad) | PASS |
| 3 | black hole | PARTIAL (topical essay, physics garble, no loop) | FAIL (hard loop) |
| 4 | ocean poem | PARTIAL (coherent ocean prose, not a poem) | PASS |
| 5 | 17×23 | FAIL ("17 * 23 = 576." wrong, but clean one-line form) | FAIL (hard loop) |
| 6 | boil an egg | PARTIAL (numbered steps, content garble) | PARTIAL |
| 7 | list vs tuple | PARTIAL (thin one-liner, no debris) | PARTIAL |
| 8 | fun fact space | PARTIAL (fluent, "Space is the only planet" wrong facts) | PASS |
| 9 | photosynthesis | **PASS** (correct coherent summary) | FAIL (formula debris) |
| 10 | stressed about work | PARTIAL (deflects with questions, no empathy) | PASS |
| 11 | translate to French | FAIL (task ignored → park/bike monologue) | FAIL |
| 12 | why do we dream | **PASS** (coherent on-topic prose) | PASS |

**Loop/quality audit (12 prompts):** hard-loop **0/12** (r2: 2/12),
prompt-echo 0/12 (r2: 0/12), empty 0/12 (r2: 0/12), formula-debris
**0/12** (r2: 3/12). Every reply is fluent, topical, stopping prose.

Sample exchanges (verbatim from the JSON):

> **Q: Why do we dream?** → "We dream because it gives us purpose and
> direction. It can also be a way of dealing with stress, anxiety, and
> other life challenges. It's also a way to connect with our inner
> selves and find meaning in the world around us. Overall, dreaming is
> an act of self-reflection and introspection…"

> **Q: Summarize: Photosynthesis is the process plants use to convert
> sunlight into energy.** → "Photosynthesis is the process by which
> plants convert sunlight into chemical energy, carbon dioxide and
> water. It is a critical process for the growth of crops and other
> organisms. The light-dependent reactions that occur in photosynthesis
> are also responsible for the production of food…"

> **Q: What is 17 * 23?** → "17 * 23 = 576."  (wrong — 391 — but a
> clean one-line answer where r2 hard-looped `x = 5 * 0.09` ×N)

> **Q: What is the capital of France?** → "France is a sovereign state
> in Western Europe… It has a population of approximately 1.3 billion
> people…" (fluent, but never says Paris — residual knowledge-salad)

## Verdict vs r2

**Strict usable count at this milestone: 2/12 vs r2's 6/12** (PASS-only
bar) — r3 is *not* yet ahead on the r2 scale, and this is only 200/2000
steps (r2's 6/12 was its val-best at step 600 of 2000). But the failure
class has changed, which is what the r3 mix was built for:

- **Salad/drift is gone as a mode.** r2's headline failure (corpus-drift
  into formula debris on knowledge prompts, 3/12) produces **0/12** here;
  hard loops 2/12 → **0/12**. All 12 replies read as conversation.
- **The residual family is task-following + factuality:** wrong facts in
  fluent prose (2/12), task ignored (2/12), thin-but-clean answers (the
  PARTIAL pile). Arithmetic is wrong but in *answer form* (the GSM8K bin's
  exact target; r2 hard-looped instead).
- r3 already fixes two prompts r2 failed outright (photosynthesis, black
  hole's loop) and loses two r2 passes to factuality (France, space).

Best-chat checkpoint so far: `lab/imported/vk_sft_r3/step_200/chat_model`
(= `step_best` at this milestone). Judgement to be re-taken at later
checkpoints as val falls — the quality lever the r3 mix targets
(task-following, arithmetic form, persona voice) is trained by the small
bins (gsm8k/persona/direct), which need more than 200 steps to show.


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

