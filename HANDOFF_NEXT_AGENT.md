# VesperLM — Handoff for the Next Agent (updated 2026-10-09 ~01:30 UTC)

## 2026-10-09 — MODULAR-MOE THESIS VALIDATED AT t1 (all evidence committed)

Full session on one MI300X droplet ($3.66; day total $6.93; droplet destroyed+verified):
- **KDA+MLA re-probe after arch_keys fix**: {'kda': 8, 'mla': 2} confirmed in trainer;
  57.0k tok/s steady @ seq 8192 (−3.5% vs GQA — MLA is free on ROCm). 470m_k full run ≈ $41.
- **t0r real-data farm batch** (lab_tiny 11M, fineweb/dclm): passport beats top-k in ALL 5
  variants; best plain passport **+2.67% val** (6.950 vs 7.141). Random-data t0 batch was
  noise-floor (ln vocab) — plumbing-only; always use real data for quality signals.
- **t1 head-to-head** (lab_small 33M, 1590 steps): passport **+1.95%** (6.092 vs 6.213).
  Attenuating with scale (2.67→1.95) — watch at t2; consider 2-seed confirmation.
- **t1 PLUG-IN TEST (the verdict)**: train 600 steps fineweb → freeze → separately train
  expert #5 + passport on python-code with contrastive loss → zero-shot plug-in:
  **rw15/600 steps PASSES ALL GATES**: util_code 0.645 (>0.5), util_web 0.159 (<0.3),
  CE_code 7.64 vs 8.53 without (−0.89 nats). rw5 under-rejects (web 0.345), rw40
  over-suppresses (code 0.28). **Production plug-in default: REJECT_W≈15, phase-B 600 steps,
  reject examples from every non-target domain.** Script: lab/imported/t1_plugin/
  plugin_expert_test_t1.py (env knobs PLUGIN_T1_*). Per-layer finding: selectivity deepens
  with depth at low rw; layer 0 keeps target preference best under high reject pressure.
- Passport throughput cost ≈ 3-4% (72.8k → 70.4k tok/s at t1) — negligible.

**NEXT DECISION (user's): t2 = tiny_agent_k (103M) plug-in mid-pretrain, ~$8-15 droplet.**
Then the AMD-credits pitch package: mechanism proof + scaling table + costed ladder (done,
see lab/imported/).

---

## 2026-10-08 evening — MI300X (AMD Dev Cloud) bring-up: WORKS

Droplet via `pod/devcloud.py` (mandatory TTL, self-destruct timer + laptop watchdog).
Ubuntu/py3.12, ROCm driver 6.19, gfx942 MI300X VF 192GB. Recipe that worked:
1. `pip install torch --index-url https://download.pytorch.org/whl/rocm6.3`
   → torch 2.9.1+rocm6.3, triton 3.5.1 (pytorch-triton-rocm).
2. fla 0.6.0 MUST be pinned by **commit**, not tag: tag `v0.6.0` does not exist
   (max tag v0.5.2 = PyPI max). Use
   `pip install --no-deps "git+https://github.com/fla-org/flash-linear-attention.git@37a6b1c6290e5240f6f0d80419d08a7aac27e548"`
   (matches laptop/box installs). Plus `pip install einops` (fla dep we rely on,
   not in requirements.txt).
3. `python tools/patch_fla.py` — now 3 patches; the new third one (ROCm-only,
   gated on torch.version.hip) caps KDA autotune `num_stages` at 2 in
   fla/ops/kda/{chunk_bwd,chunk_intra,gate,wy_fast}.py. Without it, KDA chunk
   kernels fail to compile on the AMD triton backend: "'tt.load' op operation
   destroyed but still has uses" in make_ttgir (upstream triton#9815 — AMD
   software-pipeliner bug with 4+ loads at num_stages>=3). With the cap: KDA
   chunk fwd+bwd passes in fp32 AND bf16; full tiny_agent_k model trains.
4. patch_fla.py no longer imports fla (find_spec only) — works on GPU-less hosts.

Probe gotchas that cost cycles (do not repeat):
- Probes must pass `vocab_size=65536` (or 65523): model default is 32000 and
  out-of-range token ids surface on ROCm as HSA_STATUS_ERROR_EXCEPTION hardware
  aborts, not a clean assert. Looked exactly like a kernel crash.
- Do NOT name a script `bisect.py` (shadows stdlib bisect → torch import dies
  with a confusing circular-import error).
- Trainer requires phase1 AND phase2 files in data/index.txt (nemotron
  curriculum). Synthetic probe data: two uint16 bins named *phase1*/*phase2*.
- `Pretrain/custom_tokenizer` now tracked in git (7046650) — was silently
  missing on fresh clones.
- p01's stats banner hardcodes "Config: small_v2" — cosmetic, ignore; 02's
  ACTIVE_CONFIG_NAME (env VESPER_CONFIG) is the real one. Hybrid-stack print
  now shows the true layer mix.

**Modular MoE landed (91091fc).** `PassportRouter` (per-expert passport embeddings,
dot-product scoring, expert dropout forcing passport reliance, `register_expert()` hot-plug)
+ `MoEFeedForward.add_expert()` in Common/vesper_model.py — defaults byte-compatible with
running checkpoints. Env overrides VESPER_ROUTER_TYPE/PASSPORT_DIM/ROUTER_EXPERT_DROPOUT/
NUM_EXPERTS. `lab/` experiment farm: queue/runner/promote + tier ladder (t0 lab_tiny →
t3 470m_k), sandboxed per-run cwd (never touches production data/ckpts). MODULAR_MOE.md
spec. **First science result** (lab/plugin_expert_test.py, CPU): zero-shot plug-in of a
separately-trained 5th expert — CE on its domain 7.38 vs 11.07 masked (thesis core holds),
router prefers it on-domain 63% vs 40% chance, but does NOT exclude it off-domain (43% vs
<0.3 target = chance for top-2-of-5) → passport loss needs negative (reject-off-domain)
pressure; that's the next t0 variant. **v2 RESULT (c17e6f5): PASSES** — contrastive
passport loss (reject-on-A term, weight 5.0, 400 phase-B steps): util_A 0.297 (<0.3 ✓,
thin), util_B 0.555 (>0.5 ✓), CE_B 8.01 vs 11.07 without plug-in, domain A undamaged
(4.95 vs 4.86). Mechanism confirmed at CPU scale but margins are thin on
barely-separable synthetic domains — production rule: plug-in must include reject
examples from every non-target domain, and gates should measure utilization margins,
not single cutoffs. **BUG FOUND+FIXED: trainer arch_keys dropped
full_type/kda_head_dim/kv_lora_rank/v_head_dim — all trainer runs so far (overnight
tiny_agent_k, MI300X 470m_k probe) silently built GQA full layers, not MLA.** KDA-on-ROCm
validation stands (linear_type was honored; MLA wrapper passed isolation separately), but
the next droplet session must re-probe KDA+MLA end-to-end with the fixed trainer.

**Throughput probe RESULT (KDA+GQA — see arch_keys bug note above; MLA re-probe pending) (470m_k, bf16, micro 8 / accum 16, synthetic random tokens):
~59.1k tok/s steady-state at full seq 8192, VRAM 14.1GB/192GB, CE ~11.095 ≈ ln(65523) on
noise (correct).** Projected full 4.2B-token 470m_k run: ~20h ≈ **$40** at $2/hr single
MI300X — well inside budget; headroom for micro_batch 32+ or grad-checkpoint-off tuning
would cut it further. Droplet destroyed, $3.27 settled for the whole bring-up.

---

Box: `192.168.1.153` (poweredge-r740, 3× Tesla P40 sm_61). Repo `/home/tliao/VesperLM` there,
AND a fresh local clone on the user's laptop `/home/tliao/VesperLM` (RTX 3070 Laptop 8GB,
torch 2.7.1+cu126, fla 0.6.0@git + local patches via `tools/patch_fla.py`, bitsandbytes user-site).
GitHub `git@github.com:datacrystals/VesperLM.git` is the sync point; laptop clones via HTTPS.

## STATE AS OF 2026-10-08 (newest first)

**Vesper-K exists and trains.** `Common/vesper_linear_model.py` now takes `linear_type="kda"`
(KimiDeltaAttention) and `full_type="mla"` (MultiheadLatentAttention, internal RoPE, no
incremental cache yet — full forward only). Configs: `tiny_agent_k` (103M) and `470m_k` (392M,
param-neutral vs 470m's 395M). Committed `c99c013`. **bf16 trainer path**: `VESPER_AMP=bf16`
env-gated autocast in 02_pretrain_linear.py (dense parts; linear layers stay fp32). Also env
overrides VESPER_CONFIG / VESPER_MICRO_BATCH / VESPER_ACCUM / VESPER_TOTAL_STEPS. Tested
end-to-end on the 3070: 200 steps bf16 on real shards, CE 10.4→7.0.

**RUNNING OVERNIGHT on the laptop 3070**: fresh `tiny_agent_k` pretrain, 6000 steps,
`bash overnight_3070.sh` (driver log `Pretrain/overnight_3070.log`). micro_batch auto-fell
back 4→3→2 (dummy-pass OOM at 4 and 3). Phase 2 auto-extends to 12000 steps if phase 1
completes. Checkpoints `Pretrain/vesper_linear_checkpoints_tiny_agent_k/`.
NEXT MORNING: adapt Hippocampus LoRA targets + Immune cpu_backend shims for KDA/MLA (both
are GLA/GQA-specific right now) and run the integrated teach→consolidate→canary-gate demo
on the fresh checkpoint. GPU probes locally are fine (no politeness needed on the laptop).

**Corpus building on the laptop** (CPU, `Dataset/11_vesperk_corpus.py`, log
`Dataset/corpus_vesperk.log`): ~19.5B-token mix → `Pretrain/data/vesperk/*.bin`
(fineweb_edu 8B, dclm 4B, code 3B, finemath 2B, cosmopedia 1.5B, wikipedia 1B; uint16+eos,
same convention as 03_fineweb.py). Box's 4B curriculum already rsynced to
`Pretrain/data/pretrain/`. Convention: raw text + <|endoftext|>, packed, no header.

**Spot pod** (`pod/`, committed `2349418`): `bootstrap.sh` = one-command cloud-instance setup
(CUDA/ROCm autodetect, fla@v0.6.0+patch_fla, background rsync of data from home, manifest-gated
index build, ckpt pull, auto-resume train loop, uploader). `MANIFEST` + `rebuild_index.sh`
weight sources that have shards PRESENT. `upload_ckpts.sh` streams step_* home, keeps last 2.
Home data dir = the repo's `Pretrain/data` (Pretrain/data symlinks into Dataset/data).
Plan: MI300X via user's AMD dev credits ($2/GPU/hr, $200 total). Phase 1 bring-up (~$5,
2-3h throwaway instance) validates KDA+MLA on ROCm FIRST, then kill; full run only after.
fp32 → bf16 note: trainer default is fp32; always set VESPER_AMP=bf16 off-P40.

**Box 429M run (unchanged, the verdict gate)**: step ~1900/4000 at handoff, val 1600→2.729
(step_best), 1700→2.755, 1800→2.846. DECISION (made): if val@1900-2000 > ~2.95, resume from
step_best@1600 with max_lr halved; else hands off. ETA ~Oct 12-13. CPU probes: loops
tightening, no factual pins yet at 1600.

**Env gotchas (both machines)**: box venv had triton 3.1.0 installed 2026-10-08 ~10:47 UTC
(shadowing user-site triton 3.4.0, breaking fresh `import fla`) — FIXED by installing
triton==3.3.1 into the venv + two cache.py compat patches (kwargs filter, getattr hooks —
harmless no-ops under 3.3.1). Running trainings were never affected (imports cached in
memory). user-site also has triton 3.4.0 (PYTHONPATH=~/.local/... works too). fla 0.6.0 needs
`tools/patch_fla.py` on any fresh machine (find_spec parent-package probe + MLA SDPA fallback
— no flash-attn needed anywhere). Box disk 93% full — mind checkpoints.

**Subagent deliverables (all committed)**: `Growth/` (expert expansion surgery, 14/14 —
exact upcycle = bit-exact clones + duplicate gate rows + top_k DOUBLED; production mode
top_k=2+noise drifts, needs warmup recipe; NO shared-expert slot exists — K2's "+1 shared"
needs a model-class change first). `Immune/` (canary gate + probes + drift; corruption drill
passes, auto-rollback works). `Hippocampus/` (quarantined session log + manual LoRA q/o +
reward-weighted-NLL consolidation; demo: 118M learned "math in words" preference from 6
turns; poison batch rolled back). Moonshot blueprint: `~/moonshot/TEACH_BY_TALKING.md`
(laptop). Design docs: `LMBUS_DESIGN.md`.

## What is running RIGHT NOW (original 2026-10-03 entry below)

**429M pretrain ("470m" config, ACTIVE_CONFIG_NAME="470m")** — launched 2026-10-05 19:11 UTC,
log `/home/tliao/pretrain_470m.log`. Fresh from scratch. dim 1024, 10 layers, 8 experts top-2,
429M total / 193M active. **8k context** (max_seq_len 8192 — first run at >2048), grad_checkpoint
on, micro_batch 1, accum 128 → 1.048M tokens/step, total_steps 4000 → **4.19B token budget**.
Data: fineweb_v2 1.5B + nemotron_phase1 2B (NEW to mix) + nemotron_phase2 0.5B.
Checkpoints: `Pretrain/vesper_linear_checkpoints_470m/` — numbered every 500 (5GB each),
step_best on val improvement. fp16 + Muon (as v2). Watch for: NaN (fp16), disk (75GB free), phase-1→2 switch, seq-len ramp to 8192 by step 800.

Rate/ETA (measured 2026-10-06 01:37, step ~310): 7.4k tok/s total at seq 3200, gentle knee,
~55s/step at 410k tok/step; VRAM 8.1GB. Projects ~5-6 days total (completion ~Oct 10-11).
Mid-run eval samples at steps 1000-2000 give the early semantics read. ckpt_dir bug fixed
(1faf75a): eval_samples_step{N}.json + loss_curve.png now live at checkpoint_dir ROOT —
numbered step_N dirs exist ONLY at %500 saves, do not create others (breaks resume).

Why: the 118M SFT refresh (results below) capped at format-without-semantics on held-out
prompts from TWO bases → 118M = capability ceiling → scale is the lever. This run tests
whether semantics emerge at 429M. SFT of the 429M comes after; `SFT/01_sft_train.py` config
selection will need pointing at the 470m checkpoint dir (it currently resolves v2 step_best).

## CPU probe (Pretrain/cpu_probe.py)

Runs any 470m checkpoint on CPU (~23 tok/s, KV-cached greedy). Shims fla Triton-only ops for
CPU: chunk_gla/fused_recurrent_gla -> naive_recurrent_gla (transpose state, v_first), and
fused gated RMSNorm -> torch (y = rmsnorm(x)*w*(g*sigmoid(g)), norm-first, fp32). Results
@step 600 (236M tok): topical, grammatical, degenerate loops, no factual recall yet — more
coherent than the 118M base was at 100% trained, too early for semantics verdict. NOTE:
cpu_probe prompts are ALSO held-out from SFT EVAL_PROMPTS now — do not reuse across that boundary.

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
6. **Vesper-K pretrain (APPROVED, queued behind current run)**: GLA->KDA linear layers +
   GQA->MLA full layers, MoE unchanged — full spec in LMBUS_DESIGN.md "Next-pretrain spec".
   Fresh pretrain at 429M scale for matched-token comparison vs this run, then scale.

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
