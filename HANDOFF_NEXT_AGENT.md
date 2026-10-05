# VesperLM — Handoff for the Next Agent (updated 2026-10-03 ~21:15)

Box: `192.168.1.153` (poweredge-r740, 3× Tesla P40 sm_61). Repo `/home/tliao/VesperLM`.
Venvs: `/home/tliao/venvs/vesper` (torch 2.5.1, fla 0.6.0), `/home/tliao/venvs/gen` (datasets).

## What is running RIGHT NOW

**429M pretrain ("470m" config, ACTIVE_CONFIG_NAME="470m")** — launched 2026-10-05 19:11 UTC,
log `/home/tliao/pretrain_470m.log`. Fresh from scratch. dim 1024, 10 layers, 8 experts top-2,
429M total / 193M active. **8k context** (max_seq_len 8192 — first run at >2048), grad_checkpoint
on, micro_batch 1, accum 128 → 1.048M tokens/step, total_steps 4000 → **4.19B token budget**.
Data: fineweb_v2 1.5B + nemotron_phase1 2B (NEW to mix) + nemotron_phase2 0.5B.
Checkpoints: `Pretrain/vesper_linear_checkpoints_470m/` — numbered every 500 (5GB each),
step_best on val improvement. fp16 + Muon (as v2). Watch for: NaN (fp16), disk (119GB free at
launch), phase-1→2 switch, seq-len ramp to 8192 by step 800.

Why: the 118M SFT refresh (results below) capped at format-without-semantics on held-out
prompts from TWO bases → 118M = capability ceiling → scale is the lever. This run tests
whether semantics emerge at 429M. SFT of the 429M comes after; `SFT/01_sft_train.py` config
selection will need pointing at the 470m checkpoint dir (it currently resolves v2 step_best).

## Probe findings 2026-10-03 (why this refresh exists)

Live probes of the v1 SFT model (`SFT/sft_checkpoints_118m_v1/step_2900/chat_model`, trained
from step_best@4100):

- Valid tool-call JSON and fluent English, BUT one dominant "describe a small Python project"
  script for nearly every prompt.
- `847*392` → wrong tool (`search_players`) + confabulated query. Identity question → fake
  file listing. String-reverser → thought-block repetition loop. hello.py task → malformed
  truncated tool call.
- Base model (pretrain step_best@5200, no SFT): token-salad loops, multilingual garbage,
  broken pseudo-Python.
- **Verdict: SFT v1 bought format/control, not task semantics.** The distill mix + better
  base (@5200) in the refresh are aimed at the semantics gap.

## Refresh RESULTS (2026-10-04, harness on step_2900 — Agent/eval_refresh_5200.log)

- "list all files incl hidden" -> correct `ls -la` first call, then DEGENERATE markdown-table
  loop when summarizing the observation. Observation-integration broken.
- "17*23+145" -> `python -c "print(17*23+145)"` -> 536 -> "The result is 536." PERFECT — but
  this is the templated/memorized one.
- "create notes.txt containing hello" (HELD-OUT) -> malformed `echo 'Hello', "string_word"`,
  no file created. FAIL.
- "13 * 12?" (HELD-OUT) -> MISCOPY: emitted `11 * 12`, ran it raw in bash (no python -c),
  got "command not found", repeated the same broken call, then confabulated about "17 * 12". FAIL.
- **Verdict: refresh = better CE (3.55->~1.5) + kept format, but held-out semantics STILL fail,
  same as v1.** The memorized-vs-novel contrast (python -c for the templated problem, raw bash
  for the novel one) is the cleanest memorization demonstration we have. Two SFT runs from
  different bases both cap at format -> **118M is a capability ceiling, not a data-mix problem.
  Scale is the next lever (429M config exists).** Infra proven end-to-end: auto-resume after
  disk-full, chained held-out eval fired correctly.

## CORRECTION: earlier eval success was contamination

The "17*23+145 = 536" success cited earlier is **not evidence of generalization**: that exact
prompt was in `EVAL_PROMPTS` (SFT/01_sft_train.py) AND the agent harness, and it matches the
templated math examples in the synthetic tooluse data → memorization. Fixed 2026-10-03:
`EVAL_PROMPTS` now uses 4 held-out prompts (capital of Japan, 91*7, count .py files, today's
date), verified absent from `Agent/agent_harness.py`. The harness chain additionally uses 2
fresh prompts invented today (notes.txt creation, 13*12). Treat any future eval success on
prompts resembling training templates with suspicion; prefer the fresh ones.

## State

- **Pretrain v2 COMPLETE**: 1.42B tokens. `Pretrain/vesper_linear_checkpoints_v2/step_best`
  @5200 (val CE **3.272**, was 3.55 @4100). Trainer never writes `step_6000`; exits after
  final eval. Val curve was still descending at the end — ran out of steps, not capacity.
- v1 SFT preserved at `SFT/sft_checkpoints_118m_v1/` (incl. step_2900/chat_model).
  384-param toy SFT at `SFT/sft_checkpoints_tiny384/`.
- Distill pipeline: `Dataset/10_sft_distill.py` (commit `a0d334a`), output
  `Dataset/data/sft/distill_chat_sft.bin`, wired into `Dataset/data/sft/index.txt` @ weight 1.0.
- Recent commits: `1a4a8ad` (permutation MoE, grad-ckpt flag, micro_batch 6, SFT accum fix),
  `802e57b` (KV cache inference, bf16 SFT, NaN guard), `1542633` (eval-sample crash fixes),
  `f106526` (harness step_best parse, chain threshold >= 2900), `a0d334a` (distill).

## Traps (still live — read before touching anything)

- **SFT resume trap**: any `SFT/sft_checkpoints/step_*` is resumed IN PREFERENCE to pretrain
  init. For a fresh SFT, mv the dir aside first (that is exactly what was done for this
  refresh). For crash recovery the same mechanism is your friend.
- Trainer saves the final model inside the last `step_N` dir (e.g. step_2900), never
  `step_3000`. Completion check = `>= 2900` or "Training complete".
- **bf16 required** for anything loading v2 checkpoints (`"amp_dtype": "bfloat16"`): the v2
  residual stream hits 40–60k → fp16 overflows in the last MoE layer → NaN. GradScaler is
  fp16-only.
- `find_unused_parameters=True` in DDP is INTENTIONAL (MoE idle experts). The torch warning
  suggesting removal is a false positive.
- GLA/Mamba2 stay fp32 on P40 (fla fp16 Triton kernels crash sm_61) — wrappers handle it.
- ssh+nohup: give jobs `</dev/null >log 2>&1`; `pkill -f` matches your own ssh cmdline — kill
  by pid.
- `get_latest_checkpoint` ignores `step_best` — pretrain resumes from highest `step_N`.
- **MixedDataStream probs are raw weights** — normalized inside __init__ now (2f22eb5).
  Phase buckets are name-matched ('phase1'/'phase2' substring): nemotron_phase1.bin lands in
  the phase1 stream alongside fineweb. val_probs group-normalized 0.8/0.2.
- **LINEAR_CHECKPOINT_DIR is now per-config** (`vesper_linear_checkpoints_{ACTIVE_CONFIG_NAME}`)
  — switching ACTIVE_CONFIG_NAME no longer resumes the wrong model.
- **Disk-full kills silently** (happened 2026-10-03 23:56 at SFT step 700): checkpoint save
  crashes the run; worse, torchrun relaunches die instantly AND silently because the log file
  itself cant be written. Check `df -h /` FIRST when a run vanishes. Pruned to 119GB free by
  deleting intermediate step_* (kept finals + step_best). Elephant: `/home/tliao/.cache/
  huggingface` is 403GB — candidates for reclaim if needed (distill dumps already converted
  to .bin). SFT run needs ~1.4GB per step_N dir, ~32GB for the remaining 2300 steps.

## Next steps (priority order)

1. Verify refresh results (morning checklist). Compare v2-refresh vs v1 probes — did the
   distill mix + @5200 base buy task semantics, or still format-only?
2. Switch `Agent/agent_harness.py` generation to `forward_incremental` (KV cache; server has
   it, harness still full-recompute — big interactive win).
3. Decide `nemotron_phase1.bin` (2B tokens, finished Oct 1) — still NOT in the pretrain mix.
   Any mix change belongs with the NEXT pretrain, not mid-run.
4. Scale-up pretrain: 429M config (`"470m"` = 429M total / 193M active) with **8k context
   from the start** (max_seq_len 2048 is the binding constraint; changing it needs a retrain
   anyway). Val curve says the 118M run was step-limited — budget more steps/tokens.
5. Fused linear+CE (fla `FusedLinearCrossEntropy`) — vocab-65523 logits are ~10% of step;
   needs care with the SFT mask and pad ignore_index.
6. Optional: `linear_type: "kda"` swap (fla has Kimi Delta Attention) — closest to the
   "K3-class scaled down" target arch. Requires fresh pretrain.

7. LMbus biomimetic sensory stack — see LMBUS_DESIGN.md (full proposal: canonical semantic
   space + per-model bridge, foveated heterogeneous-MoE vision with LM-driven gaze,
   certification = held-out-modality demo on the 118M, staged toward grafted support packs
   on GLM/K-class open models).

## Hardware notes (settled — do not reopen unless user asks)

User weighs MI210 (~$4k) vs 8× Gaudi2 (~$16k) later. fla has first-class ROCm → MI210
(gfx90a, 64GB) is the safe pick; Gaudi2 needs a full TPC/SynapseAI port of GLA/Mamba2.
Advice already given: free vLLM config pass on existing 8× MI50 first → 1–2× MI100 ($1400 ea)
→ used MI300X when budget allows. User has 8× MI50s serving other models (ports Mimo/GLM
flashes to vLLM on them), hates cloud, mortal budget, wants ~1TB VRAM long-term for ~1T-param
4-bit models. Serving note: Chinese frontier models are mostly 4-bit (K3), GLM 8-bit — a
768B GLM-5.3 at 8-bit needs ~800GB → 10× MI210 (640GB) does NOT fit; 8× MI300X (1.5TB) does.
