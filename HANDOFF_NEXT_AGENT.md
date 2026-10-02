# VesperLM — Overnight Handoff for the Next Agent (written 2026-10-02 ~10:40)

Box: `192.168.1.153` (poweredge-r740, 3× Tesla P40 sm_61). Repo `/home/tliao/VesperLM`.
Venvs: `/home/tliao/venvs/vesper` (torch 2.5.1, fla 0.6.0), `/home/tliao/venvs/gen` (datasets).
Everything below is COMMITTED (HEAD = `802e57b`). Working tree should be clean — check `git status` first.

## What is running RIGHT NOW (overnight chain)

`bash run_overnight.sh` (log `/home/tliao/overnight_chain.log`) is doing:

1. **118M SFT (tiny_agent_v2, bf16)** — currently training, started 10:27 from
   `Pretrain/vesper_linear_checkpoints_v2/step_best` (step 4100). Log `/home/tliao/sft_v2_118m.log`.
   ~14 s/step, 3000 steps → ETA ~22:00–23:00. CE was 4.52→4.30 at step 10 and falling. Checkpoints:
   `SFT/sft_checkpoints/step_*` every 100.
2. On completion: agent-harness eval → `Agent/eval_v2_118m.log` (2 prompts).
3. Then **pretrain continuation** (tiny_agent_v2, resumes `step_4200` → total 6000), log
   `Pretrain/pretrain_v2_continue.log`. fp16 + Muon, same as it ran to 4200. ~1800 steps.

Morning checklist: `tail` the three logs; confirm `SFT/sft_checkpoints/step_3000` exists;
read `Agent/eval_v2_118m.log` samples; confirm pretrain resumed at step 4201 (not 900 — see traps).

## What changed today (two commits)

`1a4a8ad` — training speed (measured, tiny_agent_v2 config, per-GPU):
- Permutation MoE dispatch (bitwise-identical; ~1.5× faster MoE). MoE was 48% of step.
- Whole-layer gradient checkpointing behind `grad_checkpoint` ctor flag (NOT attn-only; ~2× lower
  activations). tiny configs set `grad_checkpoint: False` — 118M model fits B=6 T=2048 in 17.6GB.
- `micro_batch_size` 1→6 for tiny_agent/tiny_agent_v2 (+SFT). Live SFT: 0.05 → 0.10 steps/s (2×).
- SFT accumulation now divides by batch_size (global batch stays 126 seqs — was about to 6×).
- Over-length input raises instead of silently truncating.

`802e57b` — inference + NaN fix:
- **KV cache**: `VesperLinearLM.forward_incremental(tokens, caches, pos)` + `new_cache(B, device)`;
  GQA stores UNEXPANDED kv with RoPE `start_pos`; GLA/Mamba2 thread fla recurrent state via
  `RecurrentStateCache`. `SFT/inference_server.py:generate_stream` uses it (prefill once, then
  1 tok/step; falls back to full recompute beyond max_seq_len). Verified 64/64 token-identical
  vs full recompute; 4.8×/tok at ctx 2048 measured DURING training contention (idle: more).
  `Agent/agent_harness.py` still uses full-recompute generation — candidate to switch over.
- **bf16 SFT** (`"amp_dtype": "bfloat16"` in tiny_agent_v2_sft config). Root cause of the
  step-0 NaN: the v2 pretrain checkpoint's residual stream runs at 40–60k magnitudes; fp16
  (max 65504) overflows in the LAST layer's MoE FFN on some sequences. bf16 has fp32 range.
  Verified on the dumped poison batch (`/home/tliao/nan_dumps/nan_step0_micro0_rank1.pt`).
  GradScaler now only active for fp16.
- SFT no longer computes the model's internal CE (`model(x)`, not `model(x,y)`) — it was
  discarded anyway and cost a 3.2GB fp32 log-softmax; its removal fixed the 10:14 OOM.
- NaN guard in the SFT loop: non-finite micro-batch is dumped to `/home/tliao/nan_dumps/`
  and skipped (DDP edge case: if the LAST micro of a step skips, grads sync one step late —
  harmless, rare).

## Traps (learned the hard way today)

- **`get_latest_checkpoint` ignores `step_best`** (int("best") → skipped). Pretrain resumes from
  highest `step_N` — currently step_4200, correct. If the numbered dirs ever get pruned below
  step_best's step, pretrain silently rolls back. Guard: keep a numbered dir ≥ step_best's step.
- **SFT resume trap**: any `SFT/sft_checkpoints/step_*` is resumed IN PREFERENCE to the pretrain
  init. That's why the old 384 checkpoints were moved to `SFT/sft_checkpoints_tiny384/` before
  launching the v2 SFT. The completed 384 SFT (incl. final `chat_model` in step_2900) lives there.
- **Do NOT set `find_unused_parameters=False` in DDP.** With MoE, an expert can receive zero
  tokens in a micro-batch → unused params. The torch warning suggesting removal is a false
  positive here (dummy pass uses a big batch).
- **fp16 + this checkpoint = NaN.** Anything loading `vesper_linear_checkpoints_v2/step_*` must
  use bf16 or fp32 for the MoE path. Pretrain still runs fp16 (it survived to 4200 on its own
  data, but watch `pretrain_v2_continue.log` for nan — if it appears, port the bf16 change to
  `Pretrain/02_pretrain_linear.py`, minding Muon's newton-schulz which needs fp32 — see
  `Pretrain/debug_nan.py` for the existing patch).
- GLA/Mamba2 must stay in fp32 on P40 (fla fp16 Triton kernels crash sm_61) — wrappers handle it.
- ssh+nohup: `pkill -f` matches your own ssh cmdline (kill by pid); give background jobs
  `</dev/null >log 2>&1`.

## State / corrections to earlier claims

- `beta2_token_half_life` IS wired in `Pretrain/02_pretrain_linear.py` (dynamic beta2) — an
  earlier review called it dead config; that's only true for the legacy dense `01_pretrain.py`.
- Config names ≠ sizes: `"470m"` is 429M total / 193M active at vocab 65523. `tiny_agent_v2` is
  118.3M total / 80.6M active (its comment is accurate).
- `Pretrain/data/index.txt` = `phase1_pretrain.bin` (symlink→fineweb_v2 1.5B) + `nemotron_phase2.bin`
  (500M). **`nemotron_phase1.bin` (2B tokens, finished Oct 1) is NOT in the mix.** Deliberate
  decision needed: add it (and re-weight) vs leave as-is. Changing the mix mid-pretrain shifts
  the stream; stream states resume from step_4200's saved pointers.
- KV-cache WIP patch `/home/tliao/kv_wip_subagent.patch` is OBSOLETE (superseded by commit 802e57b);
  kept only in case. Diag/bench scripts: `/home/tliao/{vesper_optim_bench,test_optim,ckpt_compat,
  diag_v2_nan,repro_nan,analyze_nan,kv_cache_test}.py`.

## Hardware research (for when budget allows)

- User is weighing MI210 (~$4k) vs 8× Gaudi2 (~$16k, best $/HBM) in a few months. fla has
  first-class ROCm support (`[rocm]` extra) → MI210 (gfx90a, 64GB) is the safe pick; Gaudi2 is
  SynapseAI/TPC (no Triton/fla) and would require porting the GLA/Mamba2 stack. AITER kernels
  ship only for gfx942+ (MI210 gets generic fallbacks). Validate `chunk_gla` on a rented MI250X
  before buying. A used 3090/4090 remains the zero-effort option (sm_86/89: flash-attn + fp16
  GLA + tensor cores, no fp32 crutch).
- 4b config memory math: ~56GB optimizer+weights in bf16 → needs 64GB card (MI210) or multi-GPU.

## Next steps (priority order)

1. Verify overnight chain results (see morning checklist above).
2. Switch `Agent/agent_harness.py` to `forward_incremental` (big demo win; server already has it).
3. Decide nemotron_phase1 data mix (above).
4. When SFT v2 finishes: compare its eval samples vs the 384 run; pick the chat model to serve.
5. Fused linear+CE (fla `FusedLinearCrossEntropy`) — logits over 65523 vocab are ~10% of step.
   Needs care with the SFT mask (external masked CE) and pad ignore_index.
6. Optional: `linear_type: "kda"` swap (fla has Kimi Delta Attention) — closest to the
   "Kimi-K3-class scaled down" target arch. New pretrain required.
7. Long context: max_seq_len 2048 is the binding constraint (RoPE buffer + checkpoint states);
   needs a retrain, do it with the next pretrain, not mid-run.
