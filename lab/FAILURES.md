# FAILURES.md — VesperLM failure log and fallback tree

Standing directive (user, 2026-10-09): **never stop on failure.** Document it,
learn from it, try the next method. Every failed experiment gets an entry here
within one heartbeat of detection, and the next alternative gets queued in the
same cycle. Failures are results — a clean falsification saves more money than
a lucky pass.

## Entry template

```
## <date> — <experiment id> — <one-line claim that failed>
- Evidence: <result file / log path, key numbers>
- Root cause: <best understanding; "unknown" is acceptable, guessing is not>
- What it rules out: <the branch of design space now closed>
- Next tried: <which fallback was queued and why>
```

## Failure log

## 2026-10-16 — pretrain-470m-silent-stall — heartbeat reported the 429M/470m pretrain "dead at Step 03990 for 4 days"; it had actually finished its 4000-step schedule and exited cleanly, and a finished run leaves the exact same log tail as a hung one
- Evidence: `/home/tliao/pretrain_470m.log` last write 2026-10-11 10:22:29 UTC
  `[10:22:29] Step 03990 | 0.01 global steps/s | ... CE Loss: 2.0284 |
  VRAM: 9.4GB` (3,709,327,104 tokens, LR 3.00e-05 = config `min_lr`, i.e. the
  cosine is fully decayed, phase 2/2). Journal 10:44:31–33: all three GPUs
  dropped to P8 idle, then the launch login's scope deactivated —
  `Consumed 2w 2d 13h 43min 56.809s CPU` = 3 ranks x 5.5 wall-days ≈ 16.5
  CPU-days, i.e. every rank ran the full schedule and exited together.
  No OOM/kill/segfault/GPU-error entries 2026-10-05..12; the log tail is a
  complete step block, no traceback (stderr is in-log). Timing closes it:
  progress prints only at `step % 10 == 0` and the loop is
  `for step in range(start_step, total_steps)` with `total_steps=4000`
  (`Pretrain/configs/model_configs.py`, "470m"), so a clean finish's last
  line is *always* `Step 03990`; steps 3991–3999 are silent (~2.2 min each),
  landing process exit at ≈10:44 — the GPU-idle minute, exactly.
- Root cause: monitoring, not training. Nothing prints after the training
  loop (only `destroy_process_group()`), so completed == stalled at the log
  level; the heartbeat health-check read log freshness only. The run was
  launched ad hoc (`torchrun --nproc_per_node=3 02_pretrain_linear.py` in
  `Pretrain/`, log redirected to `/home/tliao/pretrain_470m.log`) from a
  one-shot SSH whose session scope then lingered as the process scope for
  5.5 days — no tmux, no supervisor, no completion banner.
- What it rules out: OOM-kill, GPU/driver fault, and terminal/session kill of
  a mid-schedule run (scope teardown *followed* process exit; CPU accounting
  shows a full-duration 3-rank run). Also rules out "resume picks up at 3990":
  numbered checkpoints save only at `step % 500 == 0` and the loop never
  executes step 4000, so the final 499 steps (~3501–3999) were never
  checkpointed — newest is `Pretrain/vesper_linear_checkpoints_470m/step_3500`
  and `step_best` holds step-3400 weights (val 2.5497). The end-of-run
  weights at 3999 are unrecoverable.
- Next tried: `pretrain_watchdog.sh` (crontab `*/30`, script kept in `$HOME`
  so it survives reboots) alarms into the training log plus
  `pretrain_watchdog.log` when `pretrain_470m.log` is >2h stale AND no
  `0[12]_pretrain*.py` process is alive; the staleness clock ignores the
  watchdog's own alarm lines so it cannot self-mute. Resume: attempted twice
  (budget spent) and both died instantly in the dummy pre-allocation pass —
  documented, not retried. (1) Plain relaunch: `bitsandbytes` was pip-installed
  2026-10-08 *mid-run* (the Oct-5 log shows the "Did not detect bitsandbytes"
  32-bit fallback), so the relaunch silently switched to `bnb.optim.AdamW8bit`
  and OOM'd. (2) Relaunch with a `PYTHONPATH` shim hiding `bitsandbytes`
  (restoring the exact Oct-5 optimizer path) plus
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: still
  `torch.OutOfMemoryError` at `vesper_linear_model.py:692` cross_entropy
  (2.00 GiB logits request, 1.77 GiB free) on rank0. Asymmetry that explains
  it: rank0 lands on the 10 GiB RTX 3080 (GPUs 1/2 are 24 GiB P40s) and
  training-time VRAM was already 9.1–9.4 GiB; a *fresh* run hits the
  "pre-allocate max VRAM" dummy pass with lazily-empty optimizer state, while
  a *resume* materializes Muon/AdamW state tensors first, so the 2 GiB logits
  no longer fit. Same class as the silent-stall root cause: the resume path
  was never exercised by the original 4000-step run. Note the schedule is in
  any case finished — `range(0, 4000)` never executes step 4000, so the
  counter cannot pass 03990 quickly (newest ckpt 3500 at 500-step cadence →
  ~18h recompute to re-reach 3990) and the final 499 steps were never
  checkpointed: `step_3500` is the last saved state and the step-3999 weights
  are unrecoverable. To actually continue pretraining: gate/skip the dummy
  pass on resume or reorder `CUDA_VISIBLE_DEVICES` so rank0 gets a P40, raise
  `total_steps` for "470m", and add a terminal checkpoint + completion banner
  so a finished run is never again indistinguishable from a dead one.
- Disk (2026-10-16 cleanup): `/` 99% (858G/915G, 11G free) -> 91% (786G/915G,
  83G free) by rotating intermediates of completed SFT rounds —
  `SFT/sft_checkpoints` 53G->3.1G and `SFT/sft_checkpoints_tiny384` 26G->887M
  (deleted `step_100..step_2800`, kept `step_2900` + `step_best` where it
  exists; per-step eval JSON/model-stats archived to `_evals/` first; kept
  finals md5-verified unchanged). `SFT/sft_checkpoints_118m_v1` already
  final-only. `sft_429m_r2{,_s3500}` (8.1G each) kept: their finals (1.73G
  `checkpoint.pt`, md5 `0b70fb48…` / `2b4ff717…`) do NOT duplicate anything
  in `VesperLM/SFT` (no size or checksum match), and the two differ from each
  other. Residual growth is a user-side tarball ingest into `~/lev` (not
  ours — untouched).

## 2026-10-15 — hippo-teach-batch-collapse — a teach session whose triples repeat one response string cannot be promoted: the stub canary gate's target-collapse rule (top_response_fraction > 0.5) rejects the batch before any learning verdict, and an early harness summary line hid the rollback behind a hardcoded "all PROMOTE"
- Evidence: `lab/logs/hippo_teach_demo_homecity_collapse.log` (clean
  single-item `home_city` run, 5 triples, 3 carrying the response string
  "You live in Reykjavik."): gate **ROLLBACK** "target collapse: one
  response is 60% of the batch" (top_response_fraction 0.6,
  unique_response_ratio 0.6) while the micro-session itself was learning
  (train loss 5.0625 → −0.6254). Same tripwire fired as session 3 of the
  first 3-item runs (`lab/logs/hippo_teach_demo_tiny_agent_k.log` history);
  those runs' "all PROMOTE" summary text was a hardcoded string in the
  harness, not a measurement.
- Root cause: the session script used one correction phrasing for both
  reject turns AND as the approved reply → 3/5 triples share one response
  string. `stub_gate` (Hippocampus/consolidate.py) rejects
  top_response_fraction > 0.5 by design (that is the poison guard working).
  Not a learnability failure of the item. Two amplifiers: (1) the harness
  asserted `apply_user_adapter` success even after a ROLLBACK (a stale
  incumbent satisfied the assert), and (2) `SessionLog` is append-only, so
  reusing a workdir silently doubles turns — one intermediate run trained on
  a 16-triple "poison" log from a 4-turn batch and must not be quoted.
- What it rules out: promoting a degenerate teach batch (by design); and
  nothing in the teach-loop design itself — the item acquires once the batch
  is non-degenerate.
- Next tried: diversify the correction phrasings ("You live in
  Reykjavik." / "You are in Reykjavik." / "Yes — Reykjavik.") so
  top_response_fraction drops to 0.2 → session PROMOTEd and all three items'
  margins moved UP (final: 3/3 promoted and up; `lab/logs/hippo_teach_demo_tiny_agent_k.log`).
  Harness fixed in `lab/hippo_teach_demo.py`: real per-session gate
  decisions are reported (no hardcoded verdicts), a rolled-back candidate is
  never applied (incumbent reloaded and the snapshot labelled), and leftover
  `session_*.jsonl` in the workdir are deleted at startup. Standing rule for
  future teach packs: **keep correction phrasings diverse within a session**
  (or accept that N identical strings in an N×2 batch will be gated).

## 2026-10-15 — rung1-growth-stub — growth-64 "completed" as a 5-key header stub while chain.log said done; skip-done on file-existence froze the stub (growth-128 would have inherited it)
- Evidence: `/data/tliao/rung1/results/growth_64.json` 109 bytes = phase/format/n/bar_harm/bar_base_tax only; `logs/growth_64.log` Traceback `HIP error: no kernel image` at `vk_graft.py:143 index_add_` (first `ce_state`, `bank_rung1.py` home-quality step) 09:10:15→09:11:49; `logs/chain.log` logged `growth 64 done` anyway; control: `growth_32.json` (same code, GPU 1, 05:02) holds the full §11 table
- Root cause: `phase_growth` dumped the header-only dict to the FINAL `growth_<n>.json` before the measure and stamped `growth_done` only at the end — a mid-measure crash left the entry stub looking like a result. `rung1_chain.sh` amplified it: `log "growth N done"` unconditional on exit code, and skip-done was `[ -f growth_<n>.json ]` (existence, not completeness). n=32 ran the identical path to completion — no n-dependent branch. Trigger of the crash itself: `hipErrorNoBinaryForGpu` surfacing at the stock-dispatch `index_add_` (HIP: async-misattributed), **twice on GPU 2 under co-tenancy, never on GPU 1**; isolated sweeps of the whole dispatch op family (dynamic-M expert GEMMs fp16+addmm M=1..512, `index_add` 64..512×1024, fp32 `bank.scores` widths 8..24) ALL PASS on GPU 2 idle — exact failing kernel still unknown (likely co-tenant/RAS or a longer-forward kernel); `AMD_SERIALIZE_KERNEL=3` is the next diagnostic if it recurs on a non-2 GPU. Also note: without `ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/...` rocBLAS aborts (`Cannot read TensileLibrary.dat ... gfx906`) on any nontrivial GEMM — the env export in rung1_chain.sh is load-bearing.
- What it rules out: a different n=64 measure branch (identical code), a skipped merge/rename step (single atomic tmp+replace path), mid-dump corruption (`dump_json` is atomic; content is the intentional entry stub)
- Next tried: write contract — `growth_<n>.json` now exists ONLY as a finished measurement (`write_growth_result(..., final=True)` stamps `growth_done` and refuses without every section); progress/errors go to `growth_<n>.partial.json` (crash → `phase: growth_error` + error text + exit 2); `phase_growth` demotes any pre-existing incomplete file, retries the measure once after `torch.cuda.empty_cache()`; chain skip-done now calls `growth_done()` (phase + sections) and logs `FAILED rc=` with up to 3 claim-locked retries. growth-64 backfilled with the fixed path; growth-128 write path unit-checked to the growth_32 schema.
- Outcome (2026-10-15 21:42): growth-64 re-measured on GPU 0 (16.8GiB free claim window; the two failed attempts were GPU 2 @7.2GiB and the old 09:10 GPU-2 run — GPU 2 excluded from the completer entirely). Row: base_tax Δ+0.0027 (0.075%, bar 0.05) PASS · pairwise_harm worst 0.0007 (k3_L16E667↔k3_L39E393, 24 sampled, bar 0.02) PASS · home_quality worst Δ+0.009 (code, bar 0.02) PASS · p_home median 0.1428 / 38-of-38 collapsed (bar 0.5) FAIL — same p_home profile as growth-32 (median 0.136), a rung-2 calibration issue, not a regression. Verdict 3/4 break criteria. Box keeps a detached completer (`/tmp/growth_autofix.sh`, tiers 12/10/8GiB, GPU 2 never) armed for growth-128/250 stubs.

## 2026-10-13 — varact-subset-gather — per-token subset expert forwards (variable-activation compute saving) are NOT runnable on gfx906: `k.nonzero()` (data-dependent-shape GPU op) itself raises `hipErrorInvalidDeviceFunction`, and novel GEMM row-counts from gather/scatter fall in the same hole family — the error is async, so it surfaces at an unrelated later line
- Evidence: box `/data/tliao/varact_out/varact.log` (run 1: `ne[k] = e(flat[k])`
  masked scatter; run 2: `idx = k.nonzero(as_tuple=True)[0]` — the nonzero
  call itself, before any GEMM). Both `torch.AcceleratorError: HIP error:
  invalid device function`. Full-batch forwards through the same wraps at
  T=2048 run fine all through v0.5/v0.5.1 measurement (stock MixMerge shape).
- Root cause: gfx906 (MI50) rejects kernels that depend on data-dependent
  output shapes (nonzero/masked-index ops) and some dynamic GEMM row-counts.
  Same family as `door25-gfx906-backward-holes` / the Qwen shape grid
  (§10.8) but WIDER than "B*T hole set": the hole set is op-specific, not
  just shape-specific. Errors report asynchronously — a try/except around
  the suspect call does NOT reliably catch it at the right line.
- What it rules out: any gather/scatter-style sparse expert batching
  (token-subset expert forwards, ragged MoE kernels, top-k token routing
  gathers) built on stock PyTorch ops for this box. Variable-activation
  compute saving can only skip at whole-batch granularity offline; at
  serving (T=1 decode) skip-vs-run is exact anyway.
- Next tried: skip-or-full policy (`lab/varact_gate.py`): a plug with an
  empty keep-set for the batch is skipped entirely (zero FLOPs); a plug with
  any keep runs the exact stock shape and masks its output. Per-token
  delivery counts (the serving-side compute measure) are kept from cheap
  `int(k.sum())` host reads, same style v0.5 already used. If subset shapes
  are ever needed again: pad to a FIXED row count with no data-dependent op
  and validate each pad size with a `torch.cuda.synchronize()` probe first.

## 2026-10-12 — bank2n-assemble-unpack — the resumable-assembly refactor shipped a 3-tuple/2-tuple unpack bug (`build_live`/`build_overlay` return (gates, banks, handles); `state_or_measure` unpacked 2) — every retry died in ~37s at the `partition_full` state with `ValueError: too many values to unpack`, silently burning a 20-attempt chain while the cache only held the 2 baselines
- Evidence: box `/data/tliao/bank2n_out/assemble_now.log` (attempts 9–20, each
  `exit 1` within 33–47s) + `assemble.log` traceback at
  `bank2n_partition.py:780 state_or_measure`. The earlier "co-tenant OOM"
  reads of the same chain were a DIFFERENT (real) failure; this one is a pure
  harness bug introduced by the resume patch.
- Root cause: `state_or_measure` assumed `gates_fn() -> (gates, handles)`;
  the trio of builders return 3-tuples. Cached-state resume worked (bare/
  masked + `bank2n_rows.pt` held), so the crash was repeatable and cheap —
  which is exactly why the retry loop looked like "GPU starvation" instead
  of "instant deterministic crash".
- What it rules out: nothing in the design — measurement machinery is fine
  (fixed by accepting both tuple shapes).
- Next tried: tolerant unpack `(res[0], res[-1])`, chain relaunched detached
  with 40 attempts + per-attempt traceback tail in the chain log so fast
  deterministic crashes are visible next to OOM exits. Lesson for retry
  chains generally: surface the last traceback lines on non-zero exit and
  treat sub-minute exits as code bugs, not capacity.

## 2026-10-12 — door8-bridge-width-recipe-scale — the door25 wrap recipe does NOT scale past d_canon≈64: rank 512 NaNs by step 30 at lr 3e-3 and rank 256 limps worse-than-masked (init-scale drift, not bridge incapacity)
- Evidence: box `/data/tliao/graft_door8_out/door8_width.json` widths 256/512
  (run-1, MI50 GPU 0, lr 3e-3 wd 0 clip 1 bs4×512, general CE): 512 goes
  ce_batch 2.65→8.06 / |y| 5e-4→2.2 by step 25 and NaNs ~step 30 (guard);
  256 trains but fixed-batch CE stalls 3.8-4.5 vs the 2.3 masked baseline
  (|y| ~2e-2, |W| 42-44).  Same harness/recipe trains cleanly at 64
  (ce0 2.64→2.57, |y| 5.9e-3) and the geometry itself is width-independent
  in parameter count terms.
- Root cause: door25's fixed init std 0.02 on every adapter entry makes
  Var(u_k) of the adapter_up→frozen-core path grow ∝ d_canon (0.02²·d_canon
  ·Var(c)); at 512 the frozen donor SwiGLU sees inputs ~2.8× the verified
  64-geometry scale and its silu(gate)*up explodes the residual (fp16 trunk
  + fp32 wrap).  A secondary bug compounded it: the half-lr NaN fallback
  reused the already-poisoned wrap weights, so the retry NaN'd at step 1
  (|y| 62.9 at step 0).
- What it rules out: any width sweep (or wide d_canon plug-in) built on
  door25's verbatim init — measured width effects confound bridge width with
  init-scale drift.  Also rules out "just lower the lr" alone (1.5e-3 from a
  poisoned state is dead on arrival).
- Next tried: width-normalized init — adapter_up/adapter_down × sqrt(64/
  d_canon) so the composite scale is pinned at the verified d_canon=64
  geometry (`make_wrap(..., ref_rank=64)` in door8; ref_rank=0 reproduces
  door25 verbatim) — plus fresh weight reinit before any lr fallback
  (`reinit_wrap`).  Synthetic-checked (12/12 incl. init-scale ratio and
  reinit-finiteness).  Run-3 = widths 256/512/1024 at width-norm init
  (detached 3-attempt OOM-retry chain; the curve is the v2 format input).

## 2026-10-12 — grafted-spine-v0-math-nan — the lr 3e-3 wrap-fit NaN mode recurs on the Phi spine: math wrap (L24 e4) fit went non-finite at step ~150 after the plug-in output collapsed (|y| 5e-3 → 7e-6), killing lane B before its memory wraps
- Evidence: box `/data/tliao/grafted_spine_v0_out/fit_B.log` (phaseA hist:
  |W| grew 17.8→28.1 while |y| shrank to 7.4e-6 and fixed-batch CE stalled
  3.71→3.35, then `ce nan` from step 150); `lab/grafted_spine_v0.py` fit lane.
  The code wrap on the same recipe/harness (L13 e8, lr 3e-3) was healthy
  (CE home 0.841→0.296 additive, |y|~1e-2), so this is wrap-specific, not a
  harness bug.
- Root cause: same family as FAILURES door4b/door5 "lr 3e-3 fits can NaN" —
  fp32 thin adapters against an fp16 trunk at lr 3e-3.  Why this core/layer
  collapsed its output before exploding is unknown; the D2 profile already
  called math specialists mushy (lift 3.69 vs code 16.5).
- What it rules out: lr 3e-3 as a universal default for Phi-host wraps —
  use 3e-3 only where the CE curve visibly descends (code-style homes), 1e-3
  elsewhere; never let a NaN'd wrap kill the rest of a multi-wrap lane.
- Next tried: (1) fit phase is now skip-and-continue on non-finite adapters
  (lane survives, bad wrap skipped); (2) recovery refit at **lr 1e-3** on
  GPU 6 (healthy: ce0 3.71→3.07 @75 steps, |y|→5e-3) + the lane's memory
  wraps at 1e-3.

## 2026-10-12 — vk-base-droplet-setup — `git clone` silently skipped on the droplet (non-empty dest), so requirements.txt + patch_fla never ran; also a data-transfer script glob bug that "completed" with 0 files
- Evidence: droplet 608325412 `/root/setup_vk_base.log` — `git clone
  https://github.com/datacrystals/VesperLM.git` refused because
  `/root/VesperLM/Dataset/data/` had already been mkdir'd for the data push;
  setup script ran `set -x` but NOT `set -e`, so `pip install -r
  requirements.txt` and `python tools/patch_fla.py` failed without aborting
  and the final `import torch, fla, triton` still printed SETUP_OK (venv deps
  are independent of the repo checkout). Second failure same cycle: the
  transfer loop's `for rel in vesperk/*.bin` used relative globs while the
  script ran from `$HOME`, so the loop "finished" `ok=0 fail=2` on literal
  glob strings.
- Root cause: order-of-operations (data mkdir before clone + clone into
  non-empty dir fails) masked by a no-`set -e` setup script; relative globs
  without `shopt -s nullglob` + cwd pinning.
- What it rules out: bootstrap-style "mkdir data/ then clone into the same
  tree" sequences on droplets, and any xfer helper whose file enumeration
  depends on the caller's cwd.
- Next tried: `git init + fetch origin main + checkout -f main` in place
  (preserves the untracked `Dataset/data`), re-ran `pip install -r
  requirements.txt` + `tools/patch_fla.py` (all 6 patches), absolute globs +
  `nullglob` + per-file size verification on the transfer (24/24 OK), then
  unit tests (route_nonphase, speedrun flags) + GPU smoke (bf16 matmul,
  triton jit, fla FLCE with the model's exact call) all green BEFORE launch.
  For future pod scripts: add `set -e` (or check every step), and never
  clone into a tree that already has data staged — clone first, or init+fetch.

## 2026-10-12 — door7-ce-distill — CE-objective distillation does NOT unlock the distill tier: teacher-hard-target (argmax) CE makes the core ANTI-CE-skilled (forced 5.10/8.98 vs masked 1.83/1.92 — worse than the KL core's 2.33/2.39) because the cross-tokenizer teacher's argmax agrees with ground truth on only 15.6% of home positions; the GT-CE alternative memorizes in-sample (train forced 1.657 vs 1.924 — first distilled core with train skill) but does not transfer (held 2.122 vs 1.830); the argmax wrap passes quality in-sample for the first time on a distilled core (1.843 vs 1.924, additive) but held-out stays flat (1.837 vs 1.830) and forced stays ≈ masked — donor-pretrained tier (0.439/1.072, forced 0.101) unreached
- Evidence: `lab/door7_ce_distill.py` (fork of door3's cache/fit + door5's
  measurement stack; d0/d1/d4/d25/d5/d3 imported unmodified; synthetic 12/12),
  MI50 box chains detached under `/data/tliao/d7_lab/`, incremental JSON +
  artifacts `/data/tliao/graft_d7_out/` (door7_distill.json,
  door7_wrap.json, door7_{deliver,wrap_gt,deliver_gt}.json in flight;
  plugin_ce_step200.pt / plugin_gt_step200.pt). Same door3 data contract
  (teacher_cache windows, 47,104 train / 31,744 held tok, masked 1.8295
  held / 1.9236 train), KL-arm budget (200 steps, lr 1e-4, wd 0.01,
  out_pen 300, bs2x512). (a) argmax arm = full-vocab next-token CE toward
  teacher hard targets (shared-support argmax -> student id), GT id where
  the teacher agrees or has no cached row: fit hard-target CE 8.15->1.84,
  raw 16-pack skill probe (forced vs masked, full slices) held 5.1036 vs
  1.8294 / train 8.9773 vs 1.9236 = ANTI-skill (KL core: 2.327/2.385).
  (b) GT arm = pure ground-truth CE on home text, same budget: fit GT CE
  8.15->0.44; skill probe train 1.6573 vs 1.9236 (PASS in-sample) / held
  2.1219 vs 1.8295 (FAIL; code +0.307, math +0.278). (c) primary
  measurement = single-site Door-2.5 wrap of the argmax core at host L6
  (250 steps lr 3e-3 wd 0 out_pen 300, 765.7s) + Door-6 additive lam*m
  delivery (lam 0.6855; row recal 400+200): quality held 1.8367 vs 1.8295
  (ratio 1.0039) FAIL / train 1.8433 vs 1.9236 (0.9583) PASS — the first
  in-sample quality pass on a DISTILLED core (KL (i') was 1.997/1.924
  mass); forced 1.8669 held / 1.8929 train (≈ masked: no skill-in-core
  through the wrap); parity 0.9993, contam 0.123, util 0.891, base CE
  +0.0009; lam=1 control 1.8471/1.8869. Mass read of the same fitted
  wrap (door5 (i') comparability): held 1.8662 / train 1.9380 — better
  than KL's 1.882/1.997 but still FAIL; additive beats mass on the same
  wrap (re-confirming door6).
- Root cause: "CE toward the teacher's argmax" is NOT CE toward the ground
  truth when the teacher disagrees — measured teacher-agree rate is
  7,331/47,012 = 15.59% (84.41% of gradient mass points at non-GT token
  ids), so the argmax objective trains the core away from the text's own
  next tokens (KL + 0.3 GT-CE at least kept a GT anchor). Pure GT CE is
  the right objective for the CE bar and does create in-sample skill, but
  47k home tokens / 200 steps / 100.7M-param 16-core pack memorizes (the
  door25 held-out caveat at core scale): held-out forced gets WORSE.
  Distilled-core CE skill needs training breadth (pretraining-scale data),
  not a change of loss on the same 47k tokens.
- What it rules out: (i) teacher-hard-target CE as the fix for the door5
  distill gap (clean falsification — anti-skill, worse than KL); (ii) GT-CE
  on the home slice as a skill-in-core source that transfers (in-sample
  only); (iii) the claim that ANY next-token-CE objective at door3's budget
  reaches the donor-pretrained tier. The door5 narrowing stands unchanged:
  a CE-skilled frozen core comes from task pretraining breadth.
- Next tried: GT-core wrap + deliver arms (in flight — does in-sample
  skill deliver through the winning stack?); if held-out breadth is the
  binding constraint, the named lever becomes multi-slice / larger CE
  pretraining of the core (or a larger CE-pretrained donor core).

## 2026-10-12 — door6-expert-mix — the two-expert MIX: composition PASS (both specialties deliver simultaneously, zero cross-contamination) but "more than the sum of parts" NOT shown (union gain 1.491 vs sum-of-solos 1.497 = 0.996×) and held-out quality FAILS on both homes (the Door-2.5 memorization gap, unchanged by going multi-expert)
- Evidence: `lab/door6_expert_mix.py` (fork; d0/d1/d4/door25/graft_d2
  imported unmodified), MI50 box GPU 1, fp16, detached `setsid`, incremental
  JSON `/data/tliao/graft_door6_out/door6_mix.json` (home copy
  `lab/imported/graft_door6/`, untracked). Host = OLMoE graft
  (`registry_olmoe_6d84c485.pt`); plug-in A = Door 2.5 winner artifact
  verbatim (Phi-mini-MoE @f620b32c L13 e8 code, host L6); plug-in B = D2
  RouteStats math pick (donor L24 e4, math lift 3.693), identical door25
  wrap + fit on math, host L12 (two host layers via the depth-match rule).
  Delivery = Door 4 additive: y = stock + λ_A·m_A·e_A + λ_B·m_B·e_B.
  2×2 CE (nats; fit / held / base):

  | state | code_fit | code_held | math_fit | math_held | base |
  |---|---|---|---|---|---|
  | neither | 1.0720 | 1.5872 | 2.7001 | 2.1042 | 2.3447 |
  | A-only | 0.2214 | 1.6615 | 2.7102 | 2.1217 | 2.3583 |
  | B-only | 1.0720 | 1.5902 | 0.6536 | 2.4067 | 2.3529 |
  | both | 0.2310 | 1.6639 | 0.6676 | 2.4395 | 2.3685 |

  Bars: code quality PASS (both 0.231 < 1.072; A-only 0.221 additive beats
  Door 2.5's mass-delivered 0.439 on the same trained wrap); math quality
  PASS (both 0.668 < 2.700; B solo Door-2.5 table 2.700→1.070→0.452);
  cross-contamination PASS (both−B on math +0.014, both−A on code +0.010,
  both inside the 0.02 noise band; solo neutrality A-on-math +0.010,
  B-on-code −0.000); parity 0.99636 PASS, contamination 0.068 (L6) / 0.082
  (L12) PASS; union 1.960 → 0.469 both-on PASS. Mix read: gain_both 1.4909
  vs gain_A 0.3812 + gain_B 1.1163 = 1.4974 (ratio **0.9957**) — exactly
  additive, not super-additive. Held-out FAILS as a skill claim on both
  homes (code_held 1.587→1.664 both-on, math_held 2.104→2.439).
- Root cause: (composition) nothing to explain — under additive delivery at
  two depth-separated layers the plug-ins do not compete: in-situ util is
  crisp (code A 0.911 / B 0.085; math A 0.042 / B 0.908) and the union gain
  is the sum of the solos to within 0.4% — the "sum of parts" holds but
  gives no emergent bonus at this scale; (held-out) the same calibration
  memorization as door25 (10-12k fit tokens per home; each wrap sharpens its
  fit slice and slightly regresses its probe half) — going multi-expert
  neither fixes nor worsens it (both-on cross-talk is +0.01 nats).
- What it rules out: (1) "two plug-ins fight for the host" as a failure mode
  under the additive contract (the softmax-mass competition of D1/Door-3
  cannot arise here by construction, and empirically the solos are
  reproduced to ~0.01 nats when the other is live); (2) "the mix is MORE
  than the sum of parts" as an emergent claim at 2 experts / 1 donor /
  ~11k calibration tokens — composition is linear, not synergistic;
  (3) additive λ·m delivery being weaker than mass delivery — for a
  trained wrap it is strictly better (0.221 vs 0.439 A-only on code).
- Next tried/queued: larger/multi-slice calibration for the wrap
  generalization rung (the real open bar — both quality gates are
  in-sample); same-layer-different-rows mix (implemented `--layout
  same-layer`, not yet run — two-layers was the natural depth match);
  second donor family in the mix (the long-term GLM/K3/Mimo product claim);
  higher λ budget / joint adapter fine-tune if held-out parity is wanted
  (out of scope of "light joint pass").
- Ops (2 failures this cycle, both recovered): (a) run-1 OOM at Phase A —
  the box's free-GPU map moved under us (the user's hackathon jobs claimed
  GPUs 0/2/3/5/6/7; the assigned GPU 5 was no longer free and the GPU 4
  door5 rerun had finished) — rerun on unclaimed GPU 1;
  (b) run-2 crashed in the joint row-recal (CPU cached-h vs CUDA nll
  target in `train_rows_one`) after B's solo fit completed — fixed; run-3
  resumed with `--skip-wrap-b` from the saved plugin so the 250-step fit
  was not repeated. Masked controls reproduce door25 (code 1.0720/1.5872)
  and B's solo (math 2.7001) to 4 decimals.

## 2026-10-11 — door5-provenance — distilled frozen cores do NOT pass the quality bar under ANY treatment (raw delivery, λ-delivery, 16-site wrap @3e-3/1e-3, and the exact single-site Door-2.5 control): (i') 1.882 vs 1.829 held / 1.997 vs 1.924 train with all aux gates PASS; (ii) 1.890 vs 1.829 / 1.944 vs 1.923 with delivery exonerated (learned λ 1.89 < forced 2.33); (i'') 10.08 vs 1.83 (mass-delivery collapse); (i) 16-site @3e-3 NaNs in 50 steps — provenance is THE quality condition but NARROWED to a CE-skilled (task-pretrained) frozen core, and thin adapters cannot manufacture CE skill from a KL-distilled core
- Evidence: `lab/door5_provenance.py`, MI50 GPU 4 (detached `setsid`,
  incremental JSON `/data/tliao/graft_d0_out/door5_{wrap,deliver,wrap_all_1e3,wrap_single}.json`,
  logs `door5_run*.log`). All arms consume the Door-3 distilled artifact
  (`graft_d3_out/plugin_distill_olmoe.pt`, cores fp `656ca50e593ef798`) at
  the EXACT distill split (teacher-cache windows, home = code+math,
  47,104 / 31,744 tok — masked CE 1.8295 held / 1.9235 train matches door3
  to 6 decimals).
- Arm (ii) distill-then-deliver NO refit (contract (b), λ learned/layer,
  cores bitwise-frozen, mass-cal row recal only): quality **1.890 vs
  1.829** held-out / **1.944 vs 1.923** train (ratio 1.033 / 1.011) FAIL;
  λ=1 control worse (2.008 / 2.172); **forced/solo 2.327 / 2.385 — the
  frozen distilled core is WORSE than masked at its own trained
  composition**; parity 0.9978 (fire 0.9956 n=25,790 / nonfire 0.9994
  n=33,602) PASS; retention 0.2425 (gain_solo 0.276 / gain_lib 0.067) FAIL;
  contam 0.0711 PASS; util 0.8952 PASS; base CE Δ+0.0004 clean; λ mean
  0.817 (0.23..1.14, L15=0.0); rows p_new_code 0.76-0.91 / p_new_base
  0.005-0.014. Wall 1115 s. The step-0 "0.50/0.759 vs masked" reads that
  motivated this arm were SINGLE-BATCH fits' CE, not full-slice — the
  full-slice truth is the measured table (door4b's step-0 0.50 has the same
  caveat).
- Arm (i'') wrap 16-site @1e-3 (Door-2.5 geometry + recipe, LR lowered to
  dodge the 3e-3 explosion): Phase A completes 250 steps clean, but quality
  **10.08 vs 1.83** held / 10.89 vs 1.92 train (forced 2.10 / 2.46), parity
  0.9219 (fire 0.874 / nonfire 0.993) FAIL, retention −0.15 FAIL, contam
  0.179 PASS, util 0.956 PASS, base CE Δ+1.41. The 16-site mass delivery
  reproduces door3's composition collapse (~0.9 in-situ mass × 16 layers
  ≈ trained stock+e_new becomes ~0.1·stock+0.9·e_new).
- Arm (i) wrap 16-site @ the pinned 3e-3: NaN abort @ step 49 (guard) —
  the door4b fresh-adapter explosion; 3e-3 is single-site-proven only.
  Arm (i') **single-site wrap @3e-3 at bs4×seq512=2048** (Door-2.5's
  verified gfx906 shape — the first attempt crashed on the B*T=1024
  backward hole, and a bs4-eval measure pass then OOMed the 32GB card;
  fixed with --eval-batch-size 2) — the EXACT §10.2 control — also
  **FAILS quality**: held-out **1.882 vs 1.829** (ratio 1.029), train
  **1.997 vs 1.924** (ratio 1.038), forced 1.897 / 1.968 (also above
  masked); slices code 1.552 vs 1.525, math 2.213 vs 2.134; while every
  aux gate PASSES (parity 0.9987, fire 0.9951 n=7,306 / nonfire 0.9992
  n=52,086; retention 1.897 non-degenerate; contam 0.123; util 0.891;
  base CE Δ+0.003). Recorded in `door5_wrap_single.json`. I.e. the wrap
  geometry/recipe delivers cleanly and the frozen distilled core still has
  nothing CE-useful to give — same failure as (ii) and (i''), now at the
  exact door25 parity of conditions.
- gfx906 wall (new instance of `door25-gfx906-backward-holes`): the
  single-site wrap's trunk backward at B*T=1024 dies with `HIP error:
  invalid device function` on GPU 4 — the hole is not shape-deterministic
  per run class (16-site fits ran 1024 fine the same night). Harness fixed:
  `--batch-size` default 4 (B*T=2048) + explicit warning on the holey shapes
  {256,768,1024,4096}; batch_tokens recorded in the result JSON.
- Root cause (evidence-ranked): (1) **the distilled core has no next-token-
  CE skill** — forced composition is worse than masked on both slices (code
  1.53→2.00 forced, math 2.13→2.19); door3's objective (KL to donor + 0.3·
  GT-CE) transfers KL-to-teacher (5.17→2.59) but not CE. (2) Delivery is
  NOT the blocker: learned-λ additive delivery strictly improves on the
  trained composition (1.89 vs 2.33) and on λ=1 (2.01), with clean
  routing/neutrality — but there is nothing CE-useful to deliver. (3) The
  wrap family cannot manufacture CE skill from a non-CE core: (i') exact
  single-site Door-2.5 geometry+recipe still has forced 1.897 > masked
  1.83 and fails quality in-sample (train 1.997 vs 1.924), and 16-site
  @1e-3 forced is 2.10 / 2.46.
- What it rules out: "any pretrained-or-distilled frozen core + thin
  adapters passes quality" — falsified. The one existence proof (door25:
  0.439 vs 1.072 train, forced 0.101) rests on a core that is CE-skilled on
  the target distribution. Together with 93188b0 (from-scratch/refit fail
  in-sample, 3 contracts × 6 recipes; refit destroys distilled skill
  0.50→5.14): plug-in quality requires a CE-skilled frozen core.
- Next (named): CE-objective distillation (teacher's CE/argmax behavior on
  our text — not KL over its distribution), multi-slice calibration for
  door25's held-out memorization gap, or a larger CE-pretrained donor core
  (multi-site pack with per-site composition-matched delivery).

## 2026-10-11 — door4b-fit-v2 — recipe v2 (contract b + Door-2.5 fit) FAILS quality in both arms: the binding constraint is EXPERT PROVENANCE, not the delivery contract
- Evidence: `lab/door4b_fit_v2.py`, MI50 GPU 4, incremental JSON
  `/data/tliao/graft_d0_out/door4b_{fresh,door3}.json` (run-1 swiglu@3e-3 NaN
  archived `door4b_run1_swiglu_nan.json`). Recipe v2 = additive fixed-weight
  delivery (`y = y_stock + λ·m·e_new`, λ learned per layer) + Door-2.5 fit
  (task-loss CE, wd 0, neutral_w 300 on base-h, clip 1.0, 250 steps,
  post-training on-trajectory row recal, held-out quality). Arms on the D1
  code plug-in: **fresh** = from-scratch rank-64 linear adapters (4.19M
  params, lr 3e-3): quality **9.25 vs 1.74** held-out (**5.67 vs 1.26**
  train!), parity 0.683, retention 0.890, contam 0.061, util 0.692 — FAIL.
  **door3-warm** = Door-3 distilled SwiGLU core (100.7M) refit (lr 1e-3):
  quality **8.88 vs 1.74** held-out (6.49 vs 1.26 train), parity 0.843
  (firing 0.740 / non-firing 0.862; any-layer base firing 0.152, code 0.849),
  retention 1.015, contam 0.052, util 0.672 — FAIL.
- Root cause (evidence-ranked, CORRECTS the door4 entry's framing): (1) **the
  delivery contract is not the binding constraint** — Door-2.5's pass
  (code CE 1.072 masked → 0.439 plugged) happened under the MASS contract
  with a PRETRAINED expert core (frozen 11.8M Phi-mini-MoE SwiGLU) + thin
  task-loss adapters. (2) **From-scratch/refit deltas fail quality in-sample
  under every contract tried** (D1 SwiGLU ×4 recipes, linear rank-64
  adapters, distilled-core-then-refit): 12k tokens / 250 steps cannot grow a
  delta that beats the already-strong masked baseline (1.26) — and the fit
  dynamics explode first (fresh swiglu NaNs ≤25 steps at lr 3e-3; adapters
  jump ce 0.46→12.4 before partial recovery; door3 refit double-counts the
  trained core under additive delivery (trained as a replacement, delivered
  additively) and raises ce 0.50→5.14). (3) lr 3e-3 is proven ONLY for
  thin-adapter-around-frozen-core; applying it to from-scratch deltas is not
  the same recipe. (4) Firing-split caveat: non-firing tokens are not
  bit-identical under contract (b) — fired neighbors contaminate them through
  attention (non-firing cos 0.86 with 15% any-layer firing).
- What it rules out: "combine contract (b) with recipe v2 and from-scratch
  deltas go green" — falsified. Plug-in quality for TRAINED-FROM-SCRATCH
  deltas is falsified within the tested budget/data class across 3 contracts
  and 6 fit recipes. The one existence proof (Door 2.5) requires a
  pretrained-core expert; that is now the named mechanism.
- Next tried (untried, queued): expert provenance as the lever — (i) refit
  the Door-3 distilled core under its NATIVE delivery (mass/forced, not
  additive — no double-count) with the Door-2.5 thin-adapter wrap on top;
  (ii) train the from-scratch delta against a frozen PRETRAINED core's output
  (distill-then-deliver, no refit); (iii) more data/steps budget for the
  in-surface fit if (i)/(ii) fail. Door-2.5's wrap geometry + task-loss fit
  remains the recipe to copy exactly.

## 2026-10-11 — door4-delivery-contract — delivery-contract head-to-head: additive λ wins, but NEITHER contract passes quality — blocker is the expert fit, in-sample
- Evidence: `lab/door4_delivery_contract.py` on the MI50 box GPU 4 (fp16,
  OLMoE-1B-7B-0924 + D0 registry graft, D1 code-domain plug-in; same bars:
  parity ≥0.95 / retention ≥0.70 / contam ≤0.30 / §6 quality = held-out home
  CE with plug-in delivered < masked). Box `/data/tliao/graft_d0_out/door4.json`
  (incremental; run-1 partial archived `door4_run1_branchA.json`). Two
  contracts, identical rows/fit budget (200 steps, lr 3e-4, out_pen 300):
  (a) **delivered-mass fit** — train the delta under the plugged composition
  (softmax mass = merge scale); (b) **additive fixed-weight** —
  `y = y_stock(64-way full scale) + λ·m·e_new(h)`, λ learned (mean 0.472),
  m = plug-in top-8 membership. Numbers (held-out quality; train-slice in
  parens): **(a)** parity 0.9467, retention 1.055, contam 0.048, util 0.195,
  quality 11.90 vs 1.74 masked (train 16.50 vs 1.26) — FAIL. **(b)** parity
  0.8870, retention 1.070, contam 0.057, util 0.593, quality 7.46 vs 1.74
  (train 4.48 vs 1.26) — FAIL, but strictly better than (a) on quality
  (held-out AND train) and utilization, with the structurally correct
  contract (stock merge bit-identical to the masked control — the mass
  confound is gone by construction).
- Root cause (evidence-ranked): (1) **the expert fit never produces a
  net-useful delta, even in-sample** — train-slice delivered CE is WORSE than
  the masked control under both contracts (16.5 / 4.48 vs 1.26), so this is
  not generalization and not delivery arithmetic: a 16-layer frozen-trunk
  SwiGLU delta trained ~4 epochs on 12k tokens with out_pen 300 and
  grad-clip 1.0 oscillates (ce traj swings 2.6↔8.9) and the L2-neutrality
  term keeps it near-zero (out_norm2 ~3e-4) while it must ADD code value.
  (2) Parity damage under (b) is concentrated on the 5.7% of base tokens where
  the plug-in fires (logit-cos min −0.90) — a helpful, input-selective delta
  would move quality and parity together. (3) Deep-layer rows still go stale
  from a trained delta (util 0.195 in (a)) — mitigated but not solved by the
  post-training on-trajectory recal round.
- What it rules out: the mass-competition explanation alone does NOT account
  for the quality failure — removing it (branch b) shrinks the gap (11.9→7.5
  held-out, 16.5→4.5 train) but does not close it. Plug-in quality is an
  EXPERT-FIT problem (adapter-training recipe: data volume, penalty balance,
  schedule), not a router-contract problem. Delivery contract verdict:
  **(b) additive fixed-weight is the winner** and the one to carry forward
  (3/5 gates better; stock scale preserved by construction).
- Next tried (untried, queued): expert-fit recipe v2 under contract (b) —
  LoRA-style budget (more than 12k tokens, 1-2k steps, LR warmup/cosine),
  out_pen applied to base-h only at lower weight (or contrastive: help-code
  AND ~zero-on-base as separate terms with independent weights), and λ
  swept {0.25, 0.5, 1.0} per layer rather than globally learned. Door-3
  distilled expert was NOT re-run in this cycle (same fit bottleneck
  expected); its numbers remain the second evidence point
  (`lab/imported/graft_d3/door3_distill.json`: forced KL 5.17→2.59 skill-real,
  held-out CE 8.07 routed vs 1.83 masked).

## 2026-10-11 — door25-gfx906-backward-holes — OLMoE trunk BACKWARD hits `HIP error: invalid device function` at some token counts on GPU 3 (B*T ∈ {256, 768, 1024, 4096} FAIL; {128, 384, 512, 1536, 2048} OK) — same missing-kernel family as the torch.sort@8192 wall, and D1's default bs2×seq512 lands exactly on the broken 1024
- Evidence: `door25_oneshape.py` one-process-per-shape grid on
  `tliao@100.64.0.7` **GPU 3** (torch 2.9.1+rocm6.4, fp16, OLMoE CE
  fwd+bwd): (2,512)=1024 and (1,1024)=1024 FAIL reproducibly; (1,512),
  (2,256), (4,512), (2,1024) OK. Pure-trunk CE backward fails with **no
  plug-in installed** (door25_bwprobe3.py), so this is not the wrap's ops —
  wrap-shaped gemms fwd+bwd all pass standalone (door25_probe.py, 15/15).
  Per-layer bisection: backward to layer-0 output already walls at (2,512).
  `torch.sort` on 8192 device elements still fails standalone (known wall).
- Root cause: torch 2.9.1+rocm6.4 gfx906 coverage has SIZE-SPECIFIC kernel
  holes in the backward path (the family of the d0-gfx906-moe-kernels sort
  hole and the d15 `bool-mask+index_add_` hole); which (B,T) shapes trip it
  is set by the reduction/gemm tile sizes (B*T, not B or T alone). GPU 3
  additionally reports `AMD Radeon Graphics` where GPUs 0/5 report `AMD
  Instinct MI50/MI60` (same `gfx906:sramecc+:xnack-`); a (2,512) backward
  probe on GPU 0 was NOT run (lane rule: GPU 3 only) so GPU-3-specificity is
  unproven. **Correction to the d0 entry: `AMD_SERIALIZE_KERNEL=3` is
  REJECTED by this torch build** ("Ignoring invalid value for boolean flag
  AMD_SERIALIZE_KERNEL: 3 — valid values are 0 or 1") — use =1 or omit.
- What it rules out: D1's default batch shape (bs2×seq512 = 1024 tokens) for
  any training run on this box's GPU 3; any assumption that a forward that
  passes implies its backward passes.
- Next tried: shape dodge (trivial, keeps the lane on GPU 3): all door25
  training runs use **bs=4 × seq=512 = 2048 tokens** (verified fwd+bwd with
  the full Phase-A graph, wrap included). Door-2.5 Phase A/B/C completed on
  this recipe. If a future run needs a hole'd shape: one-process-per-shape
  probe first (`door25_oneshape.py` pattern), then pick the nearest OK B*T.

## 2026-10-11 — door25-wrap-training-starvation — the D1 Phase-A defaults (weight_decay 0.01 + grad_clip 1.0 + lr 1e-4) STARVE a thin-adapter wrap: the delta stays at |y|~5e-4, below the fp16 trunk's resolution, so the wrapped expert is bitwise-invisible (pilot CE identical over 40 steps, forced CE == masked CE to 6 decimals)
- Evidence: `door25_pilot2.json` Phase A (80 steps, lr 3e-3 **with** D1's
  wd 0.01/clip 1.0): `out_norm2` prints 0.000000 at every step; ce_code on
  batch 0 bitwise-identical at steps 0/40 (0.9625347256660461); s6 quality
  forced CE 1.0720305562 == masked 1.0720305562; margins gain_solo
  +0.0023. Arm sweep (`door25_lrsweep.py`, 12 steps each, real frozen Phi
  core): wd0/clip0 → mean|y| 1.0e-1; wd0/clip1 → 5.0e-3; wd.01/clip0 →
  1.2e-2; **wd.01/clip1 (D1 recipe) → 4.6e-4**; D1's lr 1e-4 → 2.7e-10.
  All four arms do move parameters (host_out 0→0.5..3.2) — the pipeline and
  grads are healthy; the shrink of host_in/adapter_up/adapter_down (7.25→2.5
  L2 under wd 0.01) walks the wrap's output into the frozen core's quiet
  zone. Same signature explains the D1 toy's tiny delta (out_norm2 4.6e-5)
  and the open s6-quality sub-gate there.
- Root cause: Adam weight_decay on the ADAPTER maps is not a neutral
  regularizer here — it contracts the composition (host_in∘adapter_up before
  the core, adapter_down∘host_out after), so the delivered delta scales down
  multiplicatively while the CE signal through the frozen core stays weak.
  grad_clip 1.0 compounds it (empirically 20× at fixed wd=0, mechanism
  unproven).
- What it rules out: porting D1's expert-fit hyper-params verbatim to a
  frozen-core + thin-adapter plug-in; interpreting "no learning" as "grads
  don't flow" without an arm sweep (they do — the loss surface just contracts
  the maps).
- Next tried: door25 Phase A at **wd 0.0, lr 3e-3, clip 1.0** (owner_mass
  0.35 / reject_w 10 / out_pen 300 unchanged — those are the pinned
  mass-cal recipe): out2 grows to ~1e-4, mean|y| ~5e-3, ce_batch0 0.951 →
  0.924 by step 75 of 250. Full run `door25_wrap.json`.

## 2026-10-11 — door3-distill-quality — Door 3 behavioral distillation into a fresh plug-in expert: formal D1 bars PASS but the quality sub-gate FAILS harder than the D1 toy (held-out home CE 8.07 active vs 1.83 masked) — the donor skill IS cloned at forced dispatch (KL-to-donor 5.17→2.59) but delivery through the softmax-mass row destroys it (plugged KL 6.63)
- Evidence: `lab/door3_distill_expert.py` on `tliao@100.64.0.7` **GPU 4**
  (fp16, HIP shims, detached `setsid`, synthetic 17/17). Teacher =
  Phi-mini-MoE @ f620b32c (7.65B), host = OLMoE-1B-7B-0924 @ 6d84c485 +
  `registry_olmoe_6d84c485.pt`, home domain = codeparrot + open-web-math
  slices (`/data/tliao/graft_d0_slice/{code,math}.txt` — the same data files
  the wrap lane uses). Cross-tokenizer KL defined over the shared single-token
  decode set (S = 12,480 strings; char-offset alignment, exact-boundary frac
  0.92 code / 0.98 math; teacher row coverage 1.0). Wall 1381 s, peak VRAM
  19.8 GiB. Recipe: `KL(teacher||student)` coef 1.0 + GT CE 0.3 + out_pen 300
  (forced dispatch, 200 steps @ 1e-4) → row recal owner_mass 0.35 / reject_w
  10 / mass_w 2 (400 steps + 200 on-trajectory). Box
  `/data/tliao/graft_d3_out/door3_distill.json` + `lab/imported/graft_d3/`
  (untracked).
- Numbers: formal bars — parity cos **0.951** (≥0.95, tight: token-cos std
  0.129, min −0.67; base CE Δ+0.77), retention **3.216** (≥0.70, but gains
  +0.27 solo / +0.88 lib = the same lib>solo metric artifact as att-3),
  contamination **0.133** (≤0.30), §6 utilization **0.959** (>0.50). Quality
  (held-out home CE) **1.83 masked → 8.07 active** (Δ+6.24; per-slice code
  1.53→9.24, math 2.13→6.90) — **FAIL**, larger gap than D1 att-4's
  1.26→12.58 ratio. In-situ home routing mass **0.911** vs owner_mass target
  0.35 (gate p_new_code 0.774 after the on-trajectory round). Distill
  fidelity, held-out KL(teacher‖student) over 31,682 positions:
  masked **5.17** → forced/solo **2.59** → plugged **6.63**.
- Root cause: **delivered-mass mismatch, now measured with a real (non-tiny)
  delta**. Phase A trains e_new at forced gain 1.0 (composition
  `stock + e_new`); delivery is `~0.09·stock + 0.91·e_new` (with
  `norm_topk_prob=false` the softmax mass IS the merge scale), so the plugged
  model is not the trained model. The clone itself is real — forced delivery
  halves KL to the donor on held-out text — and it still collapses under
  passport-mass delivery. Same failure signature as `d1-plugin-pass-1`'s open
  s6_quality sub-gate (att-4: tiny delta, mass ~0.85, CE 1.26→12.58);
  distillation does NOT rescue quality by itself because the blocker is the
  delivery contract, not expert expressiveness.
- What it rules out: "train a better expert (distill it from a stronger donor)
  and the D1 quality gate will pass under the mass-cal recipe" — NO. Door 3's
  §10.3 fallback claim holds only at the forced/solo interface (behavior is
  hostable: KL 5.17→2.59); passport-mass delivery is the shared blocker for
  both the wrap (2.5) and distill (3) lanes at quality level.
- Next tried (named branch, queued): **expert-fit at delivered-mass** — train
  the delta at the routing weight it will actually receive (or deliver
  additively at a fixed small weight instead of competing for softmax mass).
  This is D1 att-4's named next branch and is now the single blocker for any
  §10 head-to-head to be meaningful on the quality read.

## 2026-10-11 — d2-crossfamily-perlayer — foreign-family graft (Phi-mini-MoE sparsemixer): logit parity PASSES (cos 0.9986, dCE +0.030) but per-layer MoE gates MISS (cos 0.864 < 0.9, top-k agree 0.651 < 0.8)
- Evidence: `lab/imported/graft_d2/phimoe_d2.json` + `phimoe_d2_reparity.json`
  (88,064 val tokens, MI50 box GPU 5, fp16, prototype+recal 300×2,
  `registry_phimoe_f620b32c.pt`). Final-logit cos 0.99863 (min −0.479,
  std 0.024), ΔCE +0.0303 (stock 1.9054 → graft 1.9357) — both user bars
  clear, including the 0.99 stretch. Per-layer MoE-output cos 0.864 mean
  (0.80–0.97), top-k set agreement 0.651 (Jaccard 0.761). Recal converged
  fine (score MSE 3.76→0.0012, on-trajectory 0.005→0.0002), so this is NOT
  under-fitting of the passport rows.
- Root cause: the foreign router's *selection* is discrete in a way OLMoE's
  is not. Phi's sparsemixer picks top-1 = argmax after a relative-threshold
  mask (`(max−s)/|s| > 2·jitter_eps` → −inf), top-2 = argmax of the remainder,
  and draws each multiplier from a *different* masked softmax. A small score
  error near the threshold or near a top-2 tie flips a hard choice; OLMoE's
  softmax-top-8 ranking is Lipschitz in comparison (D0: agree 0.837, layer
  cos 0.995 on the same recipe). Per-layer error is U-shaped over depth (edge
  layers 0/1/31: 0.96–0.97 cos / 0.79–0.85 agree; trough layers 13–21:
  0.80–0.85 / 0.53–0.65) — signature of small score errors compounding
  through depth while the final layers partially reconverge the residual.
- What it rules out: "one numeric parity bar set fits every router family".
  Logit-level parity (the user's D2 bar) generalizes; the D0-runbook
  per-layer gates (cos>0.9, agree>0.8) are OLMoE-calibrated and unfair to
  threshold/argmax routers. Any future family with hard-selection routing
  (sparsemixer, top-1 gating, expert choice) should be gated on logit cos +
  ΔCE + a *family-relative* agreement baseline (e.g. agree vs the
  best-achievable under exact-init fp noise), not the OLMoE absolute.
- Next tried: none needed for the D2 question itself (user bars PASS); the
  interface is proven cross-family at logit level. If per-layer parity is
  later required, the lever is score-margin-aware recal (weight the MSE on
  near-threshold pairs) — cheap, router-only, same protocol.
  **UPDATE (same day, `d2-margin-recal` below): that named lever RUNS and
  FAILS — curve monotone slightly negative.** The gap closes only with
  algebraic (exact-init) passports, which pass every gate.

## 2026-10-11 — d2-margin-recal — score-margin-aware recal does NOT close the Phi-mini-MoE per-layer gap (curve monotone slightly NEGATIVE: agree 0.651→0.641/0.632/0.623 at margin_w 2/5/10); the named fix from `d2-crossfamily-perlayer` is falsified — and an algebraic-passport control (exact init) passes EVERY gate at cos 0.99996 / MoE-cos 0.9966 / agree 0.9881
- Evidence: `lab/imported/graft_d2/phimoe_d2_{mw2,mw5,mw10,exact}.json` +
  compact `lab/graft_d2_margin_recal.{json,md}` (MI50 box GPU 5 only, fp16,
  88,064 val tokens, D2-attempt-1 protocol unchanged: prototype init + recal
  300×2, `--recal-margin-tau 0.05`). Curve (means over 32 layers; BEFORE =
  uniform-MSE D2 attempt 1):

  | arm | MoE cos | top-k agree | Jaccard | logit cos | ΔCE |
  |---|---|---|---|---|---|
  | BEFORE (uniform, 300×2) | 0.8640 | 0.6509 | 0.761 | 0.99863 | +0.0303 |
  | margin_w=2 (300×2) | 0.8602 | 0.6412 | 0.754 | 0.99857 | +0.0308 |
  | margin_w=5 (300×2) | 0.8571 | 0.6320 | 0.748 | 0.99859 | +0.0329 |
  | margin_w=10 (300×2) | 0.8540 | 0.6233 | 0.742 | 0.99856 | +0.0344 |
  | uniform, 600×3 (budget control) | 0.8858 | 0.7017 | 0.797 | 0.99892 | +0.0190 |
  | margin_w=5 (600×3) | 0.8822 | 0.6902 | 0.789 | 0.99897 | +0.0205 |
  | exact init (control) | 0.9966 | 0.9881 | 0.992 | 0.99996 | +0.0001 |

  Main bars hold everywhere (cos ≥ 0.95 incl. 0.99 stretch, |dCE| ≤ 0.05);
  the aux bars (MoE cos > 0.9, agree > 0.8) never move on any prototype arm.
  At the D2 budget, margin weight makes both aux metrics strictly worse; at
  600 steps × 3 on-trajectory rounds the budget control (uniform) BEATS the
  margin arm (0.8858/0.7017 vs 0.8822/0.6902) — the higher-budget gain is
  budget, not margin weighting. Recal still converges (cal-pair MSE 3.77→0.0013,
  on-trajectory 0.005→0.0002).
- Root cause: the residual per-layer error is NOT fit misallocation near the
  decision boundary — it is **on-trajectory score error from hidden-state
  drift**. The harness's `margin_analysis` (noise-matched diagnostic, same
  run): val-trajectory score-error RMS 0.05–0.27 per layer (recal's cal-fit
  RMSE is 0.014 — 4–19× smaller: the 4096-pair cal fit does not transfer to
  the val trajectory), and `frac(gap23 < 2σ) = 0.87–0.99` — essentially every
  token is flip-prone at that error level, so re-weighting WHICH pairs are fit
  cannot help; only smaller error can. Diagnosis correction to the entry
  above: in **eval** sparsemixer INDEX selection is pure top-2 ranking
  (argmax, then argmax-of-remainder — the donor source confirms the
  relative-threshold mask is applied only to the multiplier softmaxes); the
  index flips are rank-2/3 near-ties (mean gap23 0.04–0.20 vs σ 0.16), not
  threshold-mask flips.
- What it rules out: (1) score-margin-aware MSE re-weighting (this scheme,
  this budget) as the lever for hard-selection per-layer parity; (2) "the
  D0-runbook aux bars are unreachable for sparsemixer families" — the
  exact-init control clears both (0.9966 / 0.9881) through the identical
  PassportBank/registry interface, so the bars are fair here and the
  prototype+recal fit is the gap; (3) isotropic noise-matched agreement as
  the family-relative baseline: at the graft's σ it predicts agree 0.40 while
  the graft scores 0.64 (ratio ~1.6) and the true ceiling is 0.99 — graft
  score errors are low-rank/structured, not isotropic.
- Next tried: (a) budget attribution COMPLETE — 600 steps × 3 on-trajectory
  rounds moves per-layer (+0.022 cos / +0.051 agree vs BEFORE) but the
  matched-budget uniform control (0.8858/0.7017) beats the margin arm
  (0.8822/0.6902): the gain is fit budget / on-trajectory coverage, margin
  weighting contributes ≤ 0; even the best prototype arm misses the aux bars
  (0.886 < 0.9, 0.702 < 0.8). (b) the metric proposal for hard-selection
  families (see HANDOFF): normalize agreement to the **measured exact-init
  ceiling** `(agree − 1/C(E,k)) / (agree_exact − 1/C(E,k))` (BEFORE = 0.656 of
  ceiling, best prototype arm 0.708) — family-relative is the right FORM, but
  the baseline must be measured, not noise-modeled; (c) the actual per-layer
  fix for a plain-linear donor gate: **ship algebraic passports (exact init)**
  in the registry. Open lever for prototype-only settings: on-trajectory cal
  coverage (more pairs / val-like cal text), not margin re-weighting.

## 2026-10-11 — pretrain-eol-vanish (t3 470m_k + box 429M) — trainer "vanishes" at the end of the step budget with no traceback and no final checkpoint (~490 steps / ~450M tokens of weights lost on the box run)
- Evidence: box `/home/tliao/pretrain_470m.log` ends at `[10:22:29] Step 03990`
  (of total_steps=4000) with no traceback / Killed / COMPLETE line; stderr is
  in the SAME log (torchrun banners, `[rank*]` warnings) so a crash traceback
  would have been captured — there is none, and "Starting fresh" appears once
  (single process lifetime, no wrapper relaunch). Latest save
  `Pretrain/vesper_linear_checkpoints_470m/step_3500` (saves every 500);
  val evals ran through step 3900 (`eval_samples_step3900.json`), never 4000;
  VRAM stable 9.4GB and disk fine (the known silent-killer traps ruled out).
  t3 is an identical signature (HANDOFF 2026-10-10: vanished at 3990/4000;
  `lab/imported/t3_ckpts/` last save step_3500, eval samples through 3900,
  loss_curve.png written at the step-3900 eval and nothing after). Tiny
  repro `Pretrain/tests_end_of_loop.py` (20 steps, free box P40) confirms the
  mechanism: pre-fix the end-of-budget run exits 0 with no banner and no
  terminal checkpoint.
- Root cause: `Pretrain/02_pretrain_linear.py` — `for step in range(start_step,
  total_steps)` is exclusive of `total_steps`, while eval only fires on
  `step % eval_interval == 0` and the disk save only on `step % 500 == 0`
  inside it. The scheduled end-of-run eval+save (step 4000 = a %500 checkpoint
  AND the next eval interval) is unreachable; steps 3991..3999 run without
  logging (the step log is every 10) and the loop falls through to
  `destroy_process_group()` + exit 0 with zero epilogue. The "vanish" is a
  clean exit with no completion marker, not a crash — which is also why no
  final checkpoint exists: there is no end-of-loop save path at all.
- What it rules out: OOM/SIGKILL, disk-full, wrapper/torchrun reap, t3
  tripwire kill, swallowed exceptions, and "OOM in a final-eval allocation"
  (the final eval never ran). Same bug class also explains the older note
  "Trainer never writes step_6000; exits after final eval" (v2 pretrain) —
  that observation was misread as normal end behavior.
- Next tried: fix in `02_pretrain_linear.py` — (1) terminal-step trigger
  `step == total_steps - 1` on both the eval block and the `%500` save so the
  final step is evaluated and saved through the existing path; (2) explicit
  flushed `[TRAINING COMPLETE]` banner after the loop so a clean exit is
  distinguishable from a vanish; (3) `sys.excepthook` emergency path: any
  uncaught failure prints + flushes the traceback and saves
  `step_{last_completed}` (rank0) before exit. Sibling `01_pretrain.py` has
  the same exclusive bound + no epilogue (its loss window is smaller — it
  saves at every eval_interval) and should get the same terminal trigger.

## 2026-10-11 — graft-registry-sweep — OPS INCIDENT: agent session reaped at the 2h cap mid-sweep; remote job SURVIVED but results existed only in the log + per-cell .pt files until process exit — salvage/checkpoint machinery added
- Evidence: registry/passport scaling sweep (`lab/graft_registry_sweep.py`,
  MI50 box GPU 4, PID 1836484) was mid-`grid_dim64_r2` when the agent session
  hit its cap twice; each reap SIGHUP'd the *local* waiters and produced a
  false "job died" read (the last *flushed log line* was mistaken for process
  end; a watcher also checked the PID on the wrong machine). The remote
  `nohup` python never died — verified box-side (`kill -0`, GPU4 74% use).
  Completed-cell numbers lived only in `sweep_run.log` print lines + registry
  .pt files because the script wrote its JSON only at process exit — a real
  second death would have cost the table.
- Root cause: (1) long agent turns + 2h session cap vs multi-hour GPU sweeps;
  (2) job supervision via ssh-attached waits and PID checks from the wrong
  host; (3) end-of-run-only results persistence in the sweep script.
- What it rules out: nothing scientific — the sweep completed its measurement
  (all 21 cells). It rules out ssh-attached waits and non-detached launches
  for any multi-hour box job.
- Next tried: salvage parser (`lab/graft_sweep_salvage.py`, log lines →
  resumable checkpoint JSON, box-side watcher loop under `setsid`), and
  `lab/graft_registry_sweep.py --resume` now flushes a checkpoint
  (`out_json + .ckpt.json`) after EVERY cell and skips completed ids on
  relaunch. Launch rule going forward (HANDOFF OPS RULE): `setsid nohup ... &`
  for remote GPU jobs, watch result files not PIDs.

## 2026-10-11 — d1-plugin-pass-1 — foreign expert plug-in onto graft: delivery vs neutrality is a Pareto frontier under naive recal (both sides fail alone)
- Evidence: `lab/graft_d1_plugin.py` on `tliao@100.64.0.7` GPU 0 (donor OLMoE
  -1B-7B-0924 fp16 + D0 registry graft; code-domain expert = 16 SwiGLU deltas,
  forced-dispatch CE training + contrastive router recal). Box
  `/data/tliao/graft_d0_out/olmoe_d1.json` + `d1_run.log` (untracked).
  Attempt 1 (expert lr 1e-3, text-KL 3.0, reject_w 5): delivery GREAT —
  retention vs solo **0.989**, util_code 0.514, contam 0.277 — but base
  parity **0.782 FAIL** and CE-code-active worse than masked. Attempt 2
  (expert lr 1e-4, L2 pen 10, reject_w 10): neutrality GREAT — parity
  **0.977 PASS**, contam **0.091** — but retention **0.487 FAIL**, util_code
  **0.286 FAIL**, and CE-active still worse than masked (2.97 vs 1.26).
  Task bars: retention ≥ 0.70, parity ≥ 0.95, contam ≤ 0.30.
- Root cause: three stacked effects. (1) **Forced-dispatch expert training is
  unstable at 7B scale**: text-KL through the frozen trunk either explodes
  (KL 1501, attempt 1) or dominates the CE gradient (KL 319 while code CE
  degrades 0.46→2.7, attempt 2) — and the out-norm L2 is inert at natural
  expert-output scale (~1e-4·mean) unless weighted ~1e3+. (2) **Teacher-forced
  router recal over-predicts delivery**: attempt-2 phase-B hit p_new_code
  0.887 at the gate on cached pre-plug hidden states, but in-situ routing mass
  was only 0.207 / util 0.286 — upstream h drifts once the row is live
  (same class of gap as G2's zero-shot surprise). (3) **reject_w trades the
  two s6 gates against each other** (5→10 moved contam 0.277→0.091 but util
  0.514→0.286). s6 quality (CE_code active vs masked) additionally requires
  the delta to actually lower average code CE — margin probes (delta_margin
  convention) improve (pre −3.39 → lib +1.70 in attempt 1) even when average
  CE does not; the two metrics disagree and D1's §6 gate is CE-based.
- What it rules out: a single teacher-forced contrastive recal + G1b text-KL
  is NOT sufficient for D1 on a 7B donor; plug-in delivery cannot be certified
  from gate-level p_new alone — in-situ utilization/mass must be measured.
- Next tried: **the §8.4 delivery-cliff fix** (owner-mass 0.55→0.85 closing
  recal — the sibling memory-expert line turned retention 0.397→1.100 with
  it) transplanted into the D1 recipe: explicit owner-mass target **0.85** on
  home (code) tokens in the recal loss (`--owner-mass`, `--mass-w`), plus an
  **on-trajectory recal round 2** (pairs collected with the plug-in live) to
  close the teacher-forced gap, plus direct off-domain L2 (`--out-pen 3000`)
  replacing trunk-KL as the neutrality control.
- **Outcome (attempt 3, 2026-10-11, ~29 min GPU 0):** the mass fix WORKED for
  delivery — retention **0.487 → 3.62 PASS**, s6 utilization **0.286 → 0.940
  PASS** (in-situ code mass 0.897, exactly the 0.85 target; on-trajectory
  round 2 closed the teacher-forced gap: gate p_new_base 0.066 → 0.020), and
  contaminatation stayed ≤0.3 (0.213). But the verdict is still FAIL on two
  gates with a new, sharper root cause: **owner-mass 0.85 is right for §8
  memory episodes whose owner DELIVERS, wrong for an augmenting expert with a
  small delta** — with `norm_topk_prob=false` the softmax mass IS the merge
  scale, so a 0.90-mass row holding a near-zero delta (out_norm2 ~1e-6, the
  L2 pen crushed it) starves the 8 stock experts to 0.10 total: code CE
  1.26 → **14.28** (s6 quality FAIL) and base CE +2.05 → parity **0.927 FAIL**
  (bar 0.95). Margins even read "lib > solo" (retention 3.6) because the
  margin probes reward SUPPRESSING stock experts on adversarial positions —
  a metric artifact, not delivery. Lesson: for D1 the §6 utilization gate is
  top-k MEMBERSHIP, not mass — the operating point is high membership with
  modest mass (~0.3), not the memory line's 0.85 dominance.
- **Final outcome (attempt 4, 2026-10-11 14:43 UTC, detached `setsid`):
  D1 PASS on all formal bars** with the mass-calibration recipe **owner_mass
  0.35, reject_w 10.0, out_pen 300** (`lab/graft_d1_plugin.py`, ~31 min GPU
  0): base parity **0.9667** (≥0.95), retention vs solo **0.934** (≥0.70;
  margins pre −3.39 → solo −0.83 → lib −1.00, gains +2.57/+2.40 — a clean
  delivery ratio, unlike att-3's 3.6 artifact), contamination **0.105**
  (≤0.30; gate p_new_base 0.008 after the on-trajectory round), §6
  utilization **0.911** (>0.50). **s6_quality remains the one open sub-gate**
  (code CE active 12.58 vs masked 1.26): the plugged row still takes ~0.85
  in-situ code mass (the CE router term dominates the 0.35 mass target) while
  carrying a tiny delta (out_norm2 ~5e-6), so stock experts are starved on
  code tokens and average code CE cannot improve. Named next branch:
  **expert-fit at delivered-mass** — train the delta to compensate at the
  routing mass it will actually receive (or deliver additively at fixed small
  weight instead of competing for softmax mass), so the plug-in adds value
  rather than replacing stock contributions. Pareto map (owner_mass /
  reject_w / out_pen → parity, retention, contam, util): att2 none/10/10 →
  0.977, 0.487, 0.091, 0.286; att3 0.85/5/3000 → 0.927, 3.622, 0.213, 0.940;
  att4 0.35/10/300 → **0.967, 0.934, 0.105, 0.911**. Archived JSONs on the
  box: `olmoe_d1_attempt3_mass085.json` + final `olmoe_d1.json`.

## 2026-10-11 — d15-translator-fidelity — telemetry-trained translators MISS the 0.9 held-out-cosine bar for every non-trivial basis pair (variant B deprioritized per §9.5)
- Evidence: `lab/graft_d15_translator.py` on the MI50 box GPU 3 (fp16, OLMoE
  donor + D0 registry + `graft_d0` gate slices; synthetic 18/18). Two
  instantiations of the §9.3a paired-activation translator (route-conditioned
  ridge-linear vs 2-layer MLP, both directions, identity/mean baselines,
  iid + route-held-out + domain-held-out splits, per-route breakdown, and the
  spec's transplanted-output cosine incl. top-32 singular directions):
  (a) within-donor cross-layer pairs 2:12 / 6:12 / 12:12 → held-out cos
  **0.5528 / 0.6296 / 1.0000** (12:12 = same-layer CONTROL, identity baseline
  1.0 → harness sound; identity baselines 0.2075 / 0.4172 show the gaps are
  real basis differences). (b) cross-model OLMoE ↔
  `ibm-granite/granite-3.0-1b-a400m-base` (word-aligned char-offset pooling,
  18,529 paired words, donor layer 12) → held-out cos **0.6437** (host layer
  12) / **0.6885** (host layer 18); train 0.72/0.76; route-held 0.61/0.66;
  domain-held 0.54/0.59; expert-output cos 0.47/0.55 (top-dirs 0.63/0.70);
  per-route macro 0.667, worst 0.556, best 0.898 — no route ≥ 0.9. Bars:
  lane 0.9 **FAIL** everywhere; §9.5 deprioritize (<0.7 in-domain) **fires**
  (0.55–0.69 band). Full tables: `lab/imported/graft_d15/d15{,_crossmodel}.json`.
- Root cause (evidence-ranked): (1) **feature availability, not translator
  capacity** — train cos itself saturates at 0.72–0.76 (the map is not
  learnable even in-sample) and the 2-layer MLP does not beat the linear map
  (mild overfit instead); this is §9.3a's pre-registered falsifier reading
  ("the host spine lacks the features, not the translator the capacity").
  (2) Word-level pairing (cross-model) smooths both sides, so the true
  per-token fidelity is at best this good. (3) Layer correspondence matters:
  within-donor fidelity scales with depth gap (1.0 @0 → 0.63 @6 → 0.55 @10);
  the best cross-model cell (deep host layer 18, route-conditioned) reaches
  0.69–0.70, consistent with "closer correspondence ≈ better stitching".
- What it rules out: naive linear/2-layer telemetry translators as a drop-in
  bridge for variant B at 1B-class scale — plug-in of donor experts into a
  foreign spine at ≥0.9 feature fidelity is NOT supported by this capacity
  class. Does NOT rule out: soft attention over donor layers / per-(expert,
  layer) translators (§9.3a sharpening 2, untried), token-level 1:1 subword
  pairing, much larger residual translators, or larger host spines (the
  falsifier predicts host quality bounds fidelity).
- Next tried: none in-lane (fail-fast per instructions; both intended
  measurements completed cleanly). Queued options for a revisit: the
  sharpening-2 layer-attention translator, the `--expert-scope` one-manifold
  arm, and/or accepting §9.5's fallback (variant-A-only: one grafted host per
  donor family, mixed at the serving layer) — which is the standing
  recommendation of this entry.
- Platform addendum (gfx906): the D1.5 word-pooling path (`bool-mask` gather
  + `index_add_` on GPU) hit the same async `hipErrorInvalidDeviceFunction`
  wall as the d0-gfx906 entry's `torch.sort` (AMD_SERIALIZE_KERNEL=3 for
  localization); trivial shim = pool on CPU (tensors are T×2048). Trivial
  shims are fine on this box; anything larger belongs on MI300X.

## 2026-10-11 — d0-gfx906-moe-kernels — torch ROCm wheel missing grouped-GEMM + a broken sort kernel (D0 recovered via shims)
- Evidence: D0 gate run (`lab/graft_d0_parity.py`, OLMoE-1B-7B-0924 fp16,
  `tliao@100.64.0.7` GPU 0, torch 2.9.1+rocm6.4) died twice on the first MoE
  forward: `torch.AcceleratorError: HIP error: invalid device function`,
  stack pointing at `torch.sort(expert_ids)` in
  `transformers/integrations/moe.py:grouped_mm_experts_forward`. With
  `AMD_SERIALIZE_KERNEL=3` (localize async HIP errors!) the first wall exposed
  itself: `RuntimeError: grouped gemm is not supported on ROCM` from
  `torch._grouped_mm`. After bypassing that, the async error persisted —
  micro-probe of every op in the expert path showed `torch.sort` (and
  `argsort`) FAIL **only at n=8192** (`invalid device function`) while
  n≤4096 and n=16384 work: the run's shape S = tokens×top_k = 1024×8 = 8192
  hits the broken kernel on every forward. histc/bincount/one_hot/gather/
  index_put/index_add_/slice-mm all fine.
- Root cause: two independent gaps in the torch 2.9.1+rocm6.4 wheel's gfx906
  coverage: (1) `torch._grouped_mm` has no ROCm kernel at all, yet
  transformers' `_can_use_grouped_mm` probes CUDA compute capability (HIP
  reports device.type=='cuda', gfx906 capability ≥ 8?) and wrongly enables
  it; (2) the device sort kernel variant selected at exactly 8192 elements is
  missing/broken for gfx906 — and because HIP reports launch errors
  asynchronously, the crash lands on an innocent later kernel (misleading
  stacks; `AMD_SERIALIZE_KERNEL=3` is mandatory for debugging).
- What it rules out: transformers' grouped_mm MoE fast path and device-side
  `torch.sort` at some sizes on gfx906 — NOT matmul in general (rocBLAS with
  `ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library` is solid), so
  plain inference pipelines remain viable on MI50 after shimming.
- Next tried: **worked around in-harness, D0 then PASSED the same day.**
  `lab/graft_d0_parity.py` now auto-deploys two ROCm shims at `load_donor`
  (no-op off HIP): (a) `_can_use_grouped_mm -> False` so transformers uses
  its own `grouped_mm_fallback` (per-expert `torch.mm`, rocBLAS); (b)
  `_rocm_safe_experts_forward` — identical SwiGLU-merge math with the
  grouping sort done on CPU (S is a few thousand elements; microseconds).
  Result on GPU 0: logit cos 0.99913, ΔCE +0.00075, per-layer MoE cos 0.9952,
  top-k agree 0.837 — full verdict in HANDOFF "D0 GATE RUN: PASS" and
  `lab/imported/graft_d0/olmoe_d0.json` (untracked). Fallback if a future
  donor still trips a missing kernel: run with `AMD_SERIALIZE_KERNEL=3`,
  probe the op table, shim or move to MI300X.

## 2026-10-11 — mi50-fullft-sft-gate — fla/Triton kernels cannot compile on gfx906 (MI50 box)
- Evidence: `lab/smi50_smoke_gate.py` on `tliao@100.64.0.7` GPU 0 (torch
  2.9.1+rocm6.4, pytorch-triton-rocm 3.5.1, fla 0.6.0@37a6b1c+patches): all 4
  gate tests FAIL — tiny GLA fwd+bwd bf16, tiny GLA fp32/grad-ckpt, tiny
  470m_k KDA+MLA+passport, fla FLCE. Hello-world `@triton.jit` elementwise
  kernel fails the same way: `error: unsupported target: 'gfx906'` in
  `triton/backends/amd/compiler.py:make_llir` → `PassManager::run failed`.
- Root cause: the Triton AMD backend in pytorch-triton-rocm 3.5.1 no longer
  supports gfx906 (its compiler.py mentions gfx942/gfx950 only). Platform
  reject at LLVM lowering, before any fla code runs; not a fla bug, no fla
  env-flag fallback exists (chunk kernels are Triton-only).
- What it rules out: any fla-linear-attention training on the MI50 box with a
  torch 2.9.1+rocm6.4-era Triton. Full-FT SFT of 470m_k on gfx906 with this
  stack is closed until the Triton is swapped.
- Next tried: **RETRY CLOSED (same day, one attempt, no third stack).** Older
  Triton stacks tested in `/data/tliao/vesper_sft/venv_triton_old` (kept the
  2.9.1 venv intact):
  1. torch 2.6.0+rocm6.2.4 (bundles **pytorch-triton-rocm 3.2.0**) —
     hello-world `@triton.jit` FAILS identically: `error: unsupported target:
     'gfx906'`. gfx906 appears 0× in that wheel's `backends/amd/compiler.py`.
  2. torch 2.5.1+rocm6.2 (bundles **pytorch-triton-rocm 3.1.0**) — hello-world
     FAILS identically again (`unsupported target: 'gfx906'` in make_llir).
     Second wall behind it: fla 0.6.0@37a6b1c won't even import against triton
     3.1.0's Autotuner API (`Autotuner.__init__() got an unexpected keyword
     argument 'do_bench'` in fla/ops/utils/cache.py) — fla 0.6.0 needs the
     newer triton, which needs a non-gfx906 AMD backend. Deadlock either way.
  **Conclusion: MI50/gfx906 is viable for serving (vLLM/rocBLAS inference —
  the hackathon jobs run fine) but NOT for the fla/Triton training stack.**
  Triton's AMD backend dropped gfx906 before 3.1.0; all three tested wheels
  (3.1.0 / 3.2.0 / 3.5.1) reject it at LLVM lowering, before fla code runs.
  Full-FT of 470m_k (and any fla-GLA/KDA training) belongs on the MI300X lane.
  Keep ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library for any
  gfx906 matmul work (wheel Tensile lib lacks gfx906 — what the box's
  hackathon processes export). Scratch left on box under /data/tliao/vesper_sft/
  (venvs, triton caches, staged t3 ckpt 3.5G, repo) — root's 24G untouched;
  delete `/data/tliao/vesper_sft` if the lane is abandoned for good.

## 2026-10-10 — swarm5-oneclick-canary — free OneClick lane starved again (40-min attach poll, zero scheduling windows)
- Evidence: `gh-ca7ead92` status polled 02:28:51–03:08:48 UTC (240×10s) via
  `GET /api/github/notebook/status?instance_id=gh-ca7ead92`: `pending` /
  "Waiting for resources..." on **every** poll (log: session `lab/logs` poll
  output; summary in `lab/imported/swarm5/CANARY.md`). Cumulative queue record
  now: 1 brief window ever (2026-10-09 23:32, lost to 502 within ~1 min),
  pending across 22:58–01:03 and again 02:28–03:08 — ~4.5h observed, 1 window.
  Instance never went `not_found`/`error` (still servable; attach path stays
  free). No cells ran; 2nd/last creation NOT used (still 1 of 2).
- Root cause: free-tier capacity starvation (k8s pod unscheduled), not a
  wedged instance — the status endpoint keeps serving the same id.
- What it rules out: the free lane as an on-demand compute source inside a
  single agent session. Windows are too rare (~1/2h at best) and too brief to
  gate a merge-blocking canary on; it remains viable only for queue-tolerant
  work driven across heartbeats.
- Next tried: paid devcloud fallback per the standing preference order —
  `vesper-swarm5-ttl180m-1791601783` (MI300X), canary completed in 15 min for
  $0.49, destroyed+verified. Bring-up gotcha for the next droplet run: the
  amddevelopercloud image prints a "Please wait while we get your droplet
  ready..." banner on every ssh session during first boot — it corrupts
  `scp` ("Received message too long") and makes an early `ssh host 'bash f'`
  a silent no-op; wait for `cloud-init status: done`, then transfer with
  `ssh host 'cat > /remote' < /local` (banner goes to your stdout, file stays
  clean).

## 2026-10-09 — t1-dropout02-passport / t0-8expert-passport — lab-farm pretrain runs barely learn (val stuck at ~9.02 = ln(8192) chance level)
- Evidence:
  - `lab/results/t1-dropout02-passport.1791549090.json` + `lab/logs/t1-dropout02-passport.log`: 1590 steps,
    val 9.0618 -> 9.0126 -> 9.0161 (flat), ce_last 9.0059. `lab/results/t0-8expert-passport.json`:
    390 steps, val 9.0711 -> 9.0295. Both plateau at ln(8192)=9.0109 while the tokenizer vocab is
    65523 (step-0 CE 11.14 ~= ln(65523)=11.09): the model learned "uniform over ids 0..8191" and
    nothing else.
  - `lab/data_synth/phase{1,2}.bin` (md5 b199fb38e5f9e65690f0f9a1533d7a87 / 1572eff5a25ed2d9b0d6a55c5103825d)
    are byte-identical to `lab/runner.py:ensure_synth` output — `rng(0).integers(0, 8192)` uniform
    noise (12.99 bits/token, zero bigram correlation). Sandbox `data/` symlinks those bins and
    `data/index.txt` points at them, so train AND val are pure noise.
  - Droplet self-control (same MI300X box, same runner, same env dicts): first t0 farm
    (`lab/imported/VesperLM/lab/results/`) val 9.02-9.03 with random-BPE-gibberish eval samples;
    t0r rerun (`lab/imported/t0r/`) val 6.95-7.14 with coherent eval samples. Identical env,
    identical `Hybrid stack: {'kda': 3, 'mla': 1}` print, identical batch/LR traces — only the bins
    differed. Cloud t1s (`lab/imported/swarm1/`) learned to 5.97 on real bins (13.0M tokens, same
    8192 tok/step — the "17k tok/step cloud budget" suspicion is wrong).
  - Decisive laptop A/B, single variable = data: `lab/results/t1-diag-dropout01-mb4.json`
    (noise bins, dropout 0.1, seed 2, mb4/accum8) val [150: 9.0608, 300: 9.03], ce_last 9.0217;
    `lab/results/t1-diag-realdata-mb4.json` (same env/seed, real nemotron bins) val
    [150: 6.5173, 300: 6.0353], ce_last 5.9885 — bends below 9 by step 50 and matches the cloud
    trajectory scale, on the same GPU.
- Root cause: `lab/runner.py:ensure_synth()` generated uniform-random token ids in [0, 8192) as the
  lab corpus; on that distribution the irreducible CE is ln(8192)=9.01, so every run "barely
  learns" by construction. The droplet's good runs had real corpus bins planted over
  `lab/data_synth/` (manual, never committed to git); any fresh checkout regenerates the noise,
  which is exactly what happened on the laptop at 05:01 today. Fixed: `ensure_synth` now slices
  real uint16 tokens from `Dataset/data/pretrain/nemotron_phase*.bin` (same format the trainer's
  `load_dataset_index` reads), with a learnable Markov-chain fallback when no corpus is present;
  the old noise bins are preserved as `lab/data_synth/*.uniform_noise.bak`. Also fixed the
  hardcoded `(GLA hybrid, fp32)` banner in `Pretrain/02_pretrain_linear.py` to print the actual
  AMP flag — the string was a red herring; `runner.build_env` sets `VESPER_AMP=bf16` on every farm
  run on both machines.
- What it rules out: expert dropout 0.2 (dropout 0.1 reproduces the plateau bit-for-bit in shape),
  NUM_EXPERTS=8, router type (topk-baseline plateaued identically in the first droplet t0 farm),
  micro-batch/accum recipe (mb4x2 = mb8x1 = 8192 tok/step both; laptop and cloud trained the same
  13.0M tokens over 1590 steps), LR schedule (identical traces), AMP/bf16-vs-fp32, seed, GPU/torch
  correctness (Hippocampus LoRA demo fine; with real bins the same harness reaches CE 5.99 on this
  3070), and the arch_keys/GQA theory (bad and good droplet runs print the same hybrid stack).
- Next tried: `t1-diag-realdata-mb4` (queued and completed — it is the confirmation above).
  Then `t1-dropout02-realdata-mb4` (dropout 0.2, everything else identical) on the fixed corpus:
  val [150: 6.5593, 300: 6.0592], ce_last 5.9994 — statistically level with dropout 0.1
  (6.5173/6.0353), so the original t1 claim ("dropout 0.2 does not hurt at t1 scale") survives on
  valid data. All pre-fix laptop results (t0-8expert, t1-dropout02) must be treated as void and
  re-baselined on the fixed corpus before any architecture conclusion is drawn from them.

## 2026-10-09 — t1-dropout01-mb4-full / t1-dropout02-mb4-full — watch daemon claimed both lab_small re-baseline jobs at once (slots=2); both OOM'd on the 8GB 3070
- Evidence: `lab/results/t1-dropout01-mb4-full.json.oom_race`, `lab/results/t1-dropout02-mb4-full.json.oom_race`
  (rc=1, steps_done=0; `torch.OutOfMemoryError` in the dummy VRAM pre-alloc pass, each log blaming the
  sibling trainer process for 3.9-5.3GB of the 7.67GB card).
- Root cause: `lab/runner.py --slots 2` admits two concurrent `lab_small` jobs to one 8GB card; each job
  peaks ~1-3.5GB alone but the pair exceeds capacity. Not a corpus, recipe, or model issue.
- What it rules out: >1 concurrent `lab_small` job on this card. The experiments themselves were not
  falsified — both passed serially on the fixed corpus (below).
- Next tried: serial rerun via `--slots 1 --once`, one queue definition in `lab/queue/` at a time:
  `t1-dropout01-mb4-full` val 5.139 and `t1-dropout02-mb4-full` val 5.216 (both 1590 steps, rc=0);
  watch daemon restarted at `--slots 1` (was 2). Side finding from the rerun: the laptop-01 control
  did NOT match cloud `t1s-passport` 5.9677 — it is ~0.8 lower at every eval (150: 6.504 vs 7.0518,
  1500: 5.139 vs 5.9677), so cloud-vs-laptop absolute levels stay confounded (cause unknown; the
  mb confound this control was meant to isolate did not explain it). The clean comparison is the
  laptop 01-vs-02 delta: dropout 0.2 is +0.077 (1.5%) worse and a hair worse at all 10 evals —
  no longer a tie at full length, though still small for a single seed each.

## 2026-10-09 — g1_export_mode v1/v2/v3 — G1 export-mode: retention parity + clean rollback PASS, but passport-row addressing fails purity (contam 0.70-0.90 vs bar 0.30) and drags base-mix CE to +188%
- Evidence: `lab/results/g1_export_mode_v1.json` (D55 phase-B contrastive recal, 150 steps):
  util_home 0.809 / contam_base 0.893, base CE +278.7%; `g1_export_mode_v2.json` (hinge ranking
  margins, 400 steps, + section-4.5 text-KL consolidation): 0.177 / 0.176, CE +353.6%;
  `g1_export_mode_v3.json` (section-4.4a mutual-exclusion mass target, 800 steps): 0.812 / 0.903,
  CE +188.0%. Logs `lab/logs/g1_export_mode_v1..v3.log`. All three: margins move (v3
  -5.00/-5.68/-4.50 -> -1.29/+0.16/-0.02, mean gain +4.68 vs direct-edit +3.22), poison batch
  ROLLBACK by the same stub gate (target collapse), one-row-drop recovery with incumbent state
  hash byte-identical and margins bit-equal. Prototype rows alone: literal mean-query
  (util 0.71 / contam 0.70, home top-2 weight 0.002 — bank row norms ~370 vs prototype ~9),
  norm-matched (util 0.44-0.81 / contam 0.78-0.89, CE +279% to +440%).
- Root cause: addressing, not retention. At top-2-of-5, any base-token top-2 membership of the
  memory row displaces a base expert and rewrites that token's FFN mix, so contamination alone
  explains the base-CE regressions. A per-layer passport row is ONE dot-product direction in
  ffn-input space; the home-episode and general-text score distributions overlap too heavily to
  threshold — measured as a hard trade-off curve with no feasible operating point (hinge:
  util 0.18/contam 0.18 vs softmax-mex: util 0.81/contam 0.87-0.90). Contributing: the 12k
  checkpoint is TopK-trained (no expert-dropout content matching), so its score space was never
  trained for content-matched row placement (G2's premise), and episodic home data is ~150 tokens
  by construction (9 triples).
- What it rules out: prototype passport init (literal mean-query AND norm-matched), D55 phase-B
  contrastive recal, hinge ranking recal, and the section-4.4a mutual-exclusion mass-target recal
  as sufficient addressing for episodic-memory experts on a TopK-trained 118M spine. D55's
  synthetic token-range domain purity (code↔math) does not transfer to overlapping language
  domains at N=1, before N=16 is even attempted. Also rules out "the expert path needs spine
  touches": zero spine weights were written (the TopK->Passport transplant is function-preserving
  to 7.6e-6 on router logits and is the only structural change).
- Next tried (same cycle): v2 = hinge ranking + section-4.5 text-KL consolidation; v3 = the
  validated section-4.4a mutual-exclusion objective at 800 steps. Both falsified (above).
  Next queued (fallback tree "Episodic memory" rung 2 + "Live self-learning" rung 4): hybrid
  cadence — direct-edit Hippocampus stays the fast-weight path (it passed G0); episodic experts
  wait for an addressing pivot: (a) sidecar kNN retrieval over episode embeddings feeding
  context (no weight change; safety floor), (b) cross-attention sidecar memory network,
  (c) hierarchical passports (domain → memory, two-stage routing). Re-run G1 on an
  expert-dropout-trained (passport) spine before spending on G2 — the transplant keeps G1 honest
  on today's checkpoint but cannot test content-match reachability.

## 2026-10-09 — g1b_passport_native (G1b) — OUTCOME LOG: addressing pivot RESOLVED on passport-native spines — full 4-criteria PASS with base-neutral experts; sidecar-kNN fallback NOT triggered
*(this file logs outcomes, not just failures — G1b is the first entry that ends in a full PASS; kept here so the decision trail and the provenance correction stay in one place.)*
- Evidence: `lab/results/g1b_passport_d01_neutral3.json` (true dropout-0.1 spine
  `t1-diag-realdata-mb4`, section-4.4a mex recal 800 steps + section-4.5 text-KL at 3.0):
  **G1 OVERALL PASS** — (a) margins -3.32/-3.81/-3.92 -> +1.03/+1.45/+0.23 (mean gain +4.588
  vs same-spine direct-edit +0.674), (b) base-mix CE **+0.599%** (<1% bar),
  (c) poison ROLLBACK (target collapse) + one row drop + incumbent hash/margins bit-identical,
  (d) util_home 0.889 / contam_base 0.244. Sibling runs: `g1b_passport_d00.json`
  (passport-native at dropout 0.0): 0.981/0.276, CE +6.70% — (a)(c)(d) PASS, (b) FAIL;
  `g1b_passport_d02.json` (dropout 0.2): 0.809/0.323, CE +4.63% — (a)(c) PASS, (d) misses
  0.30 contam bar by 0.023, (b) FAIL; `g1b_passport_d01.json` (dropout 0.1, text-KL 0.05):
  0.914/0.225, CE +6.94% — (a)(c)(d) PASS, (b) FAIL; `g1b_passport_d01_neutral.json`
  (text-KL 1.0): 0.889/0.259, CE +2.00%. Same-spine direct-edit references:
  `lab/logs/g1b_direct_edit_d0{0,1,2}.log` (demo harness on each ckpt; e.g. d01 direct-edit
  gain +0.674 and its own base CE cost +0.30% — export beats direct-edit on margin gain on
  every spine, 2.8x-6.4x). G1 comparison: util 0.812/contam 0.903/CE +188% ->
  0.889/0.244/+0.60%.
- Root cause of the G1 addressing failure (now confirmed): the TopK->Passport TRANSPLANT, not
  passport routing itself. G1's spine had no content-matched score space; on passport-native
  spines the same prototype init + mex recal separates home from base ~3x better (contam
  0.225-0.323 vs 0.903), and expert-dropout training sharpens it further (best purity on the
  true 0.1 spine). Second finding: base-CE regression is NOT structurally pinned to
  contamination — it is the expert's foreign-token behavior. Sweep at constant routing on the
  d01 spine: text-KL 0.05 -> +6.94%, 1.0 -> +2.00%, 3.0 -> +0.599% CE. A base-neutral expert
  makes contaminated tokens cheap; criterion (b) and (d) decouple.
- Provenance correction (affects the earlier dropout comparison in this file): the farm run named
  `t1-dropout01-mb4-full` did NOT train with expert dropout 0.1 — its queue env omitted
  `VESPER_ROUTER_EXPERT_DROPOUT` and its saved model_config has no such key (trainer default
  0.0; live router reads `expert_dropout 0.0`). The true dropout-0.1 checkpoint is
  `t1-diag-realdata-mb4` (300 steps, config carries `router_expert_dropout: 0.1`), and
  `t1-dropout02-mb4-full` is genuinely 0.2. So the "01 vs 02" delta discussed above is really
  "0.0 vs 0.2"; the G1b dropout gradient (0.0/0.1/0.2) is the clean read.
- What it rules out: (1) "episodic memory experts need a different addressing mechanism"
  (§8.5 bullet 1) — NOT triggered; passports + mex recal + base-neutral experts work at N=1 on
  a passport-native spine. The sidecar-kNN fallback stays in the tree un-armed.
  (2) G2's no-recal premise — falsified on every spine (prototype arms: best 0.479/0.276;
  all miss (d)); consolidation pays one router-recal pass per insert, exactly G2's stated
  falsifier outcome ("the loop slows but the design survives"). (3) The G1 claim "a single
  passport direction cannot separate home from general text" — true for the transplant's
  identity-query score space, false for a trained query map.
- Next tried (this cycle): G1b on three passport-native spines (dropout 0.0/0.1/0.2) with the
  v3 recipe, then the base-neutrality sweep (text-KL 1.0, 3.0). Next queued: **G3 (N=16
  coexistence on tiny_agent_k-scale passport spine)** is unblocked — run it with the G1b recipe
  (mex recal + text-KL 3.0 consolidation) and re-check the (b)/(d) bars survive 16 plugged
  experts. Fallback tree below updated: addressing rung closed, kNN sidecar disarmed.

## 2026-10-09 — g3_coexistence (G3) — OUTCOME LOG: N=16 coexistence FAILS — per-expert purity decays with N (util_home 0.74 -> 0.08) and the plug-in GROUP saturates top-2 on base (p_any 0.61) driving base CE to +2.9%; poison row-drop still byte-identical. §8.5 kill criterion 1 FIRES -> hierarchical domain->memory passports ACTIVE
*(outcome log, per standing directive — G3 is a gate result, recorded here either way.)*
- Evidence: `lab/results/g3_n16_coexistence.json` (v3, headline: §4.4a cadence —
  40-step consolidation/expert, 150-step per-insert mex recal, 800-step final joint
  calibration; total 27 min) and `lab/results/g3_n16_v2_perinsert600.json` (600-step
  per-insert recal, no closing joint pass; 64 min; same failure shape — so the outcome
  is not a cadence artifact). Fixtures: 16 episodes, distinct prompt domains + distinct
  taught styles (episode 0 = G1's math/words set; v1 fixture with one shared prompt
  template aborted at N=6, preserved as `g3_n16_v1_weaksep.json` — util decayed even
  faster there (0.90 -> 0.19), fixture flaw documented below). Spine: passport-native
  lab_small dropout-0.1 (`t1-diag-realdata-mb4`) — the §8.4 text said t2/tiny_agent_k
  but G1 proved that TopK checkpoint cannot support the test; G1b recipe throughout.
  **N=16 final state (after joint calibration):** util_home e0=0.763 but e1..e15
  0.167–0.388 — 15 of 16 experts below the 0.5 bar. contam_base per expert 0.028–0.100
  (bar <0.3 PASSES — per-expert contamination is not the failure). base-mix CE
  regression **+2.88%** (bar <1%). Cross-talk between home episodes low (0.03–0.05);
  forced-NLL retention gains healthy (+3.3..+4.4 nats) — experts learn and are
  distinguishable. Poison spot-check at N=16: stub gate ROLLBACK (target collapse),
  one row drop -> state hash byte-identical AND base CE restored exactly
  (6.968553 vs 6.968553).
- Purity trajectory (v3, per insert): util_min 0.74, 0.69, 0.52, **0.44 (break at N=4)**,
  0.38, 0.32, 0.26, 0.26, 0.12, 0.14, 0.11, 0.12, 0.11, 0.10, 0.11, 0.08;
  contam_max 0.42, 0.49, 0.35, 0.31, 0.28, 0.24, ... 0.20 (crosses under 0.3 at N=5);
  base CE +0.92, **+1.84 (break at N=2)**, +1.85, +1.83, +2.35, ... +3.0;
  p_any_plug_in_top2 (base tokens): 0.42, 0.63, 0.63, ... 0.63-0.65 flat — the group
  saturates top-2 immediately and stays there. Recal cost scales ~linearly:
  8s (N=1) -> 112s (N=16) per insert at 150 steps; 27 min total vs 64 min for the
  600-step variant.
- Root cause (measured, not guessed): **top-2 slot saturation in a shared 64-dim
  passport row space**. Each expert's row is well-behaved in isolation (its own
  contamination is 3-10%, its home cross-talk is low) but 16 rows × ~4-6% base
  membership each = 61% of base tokens see SOME plug-in row in top-2, displacing a
  base expert on those tokens — that is the +2.9% CE. On home tokens the rows fight
  each other for the two slots and the mex 0.55 owner-mass target is unreachable for
  most owners (only the earliest-inserted row keeps util ~0.76). The final joint
  calibration equalizes rows somewhat (e0 0.86 -> 0.76, e15 0.12 -> 0.18 in v2) but
  cannot recover the majority — so insertion-order training bias is NOT the root
  cause; the mex loss plateaus at ~3.9 (v3 joint 800 steps) with no downward trend,
  i.e. the target distribution is infeasible in this row space at this N.
- What it rules out: (1) the G1b recipe (contract expert + text-KL 3.0 + mex recal +
  prototype init) scales to N=16 — FALSE; it holds to N≈3-4 (util 0.52 at N=3) and
  degrades monotonically after. (2) "Purity decays because later rows are under-trained" —
  FALSE (the closing joint pass moves numbers ≤0.05). (3) "Separable fixtures suffice" —
  FALSE (v2's lexically distinct domains still collapse; v1's shared-template fixture
  was a second, independent failure mode, kept as a fixture-design caution). (4) The
  rollback story does NOT fail: one-row-drop recovery at N=16 is byte-identical.
- What it confirms: §8.5 bullet 1's kill condition — "G3 shows purity collapsing
  before N=16 — then memory experts need a different addressing mechanism and passport
  stays a domain-expert tool only." Purity broke at N=4 (util) / N=2 (base CE).
  Also confirms the swarm3/4 owner_mass scale-weakness in the memory setting.
- Next tried (this cycle): v2 (600-step/insert) then v3 (§4.4a exact cadence with the
  closing joint calibration) — both FAIL identically. Next queued per fallback tree
  rung 4: **hierarchical domain→memory passports (two-stage routing) marked ACTIVE** —
  cap the live library + archive stale experts to NVMe offload with prototype-index
  reinsertion is the alternate branch if two-stage routing fails its own smoke test.
  Also armed (rung 4 of "Live self-learning"): cross-attention sidecar memory network
  as the addressing-mechanism replacement if hierarchy fails.

## 2026-10-09 — g3_recipe_controls (Control A/B) — OUTCOME LOG: recipe depth explains the N=1 break, NOT the N-scaling decay — N=8 full-recipe still FAILS (util 0.93→0.16, CE +2.28%); bank crowding measured (plug-plug |cos| 0.275 vs 0.066)
*(outcome log — separates the two hypotheses the G3 failure left open: "fast recipe confound" vs "N-scaling structural".)*
- Gate-accounting finding first: the G3 verdict judges the **N=16 final state** (per the
  gate text "at N=16") — that was always correct. What was misleading was the log line
  "first purity break at N=1", which conflated per-criterion trajectory spikes with
  decay. Per-criterion breaks: v3 (40 cons/150 recal/800 joint) broke **contam at N=1**
  (0.415 — the 150-step recal left the lone row under-trained on base rejection; final
  contam at N=16 is clean 0.03-0.10 and the final-state contam bar PASSES); v2
  (80 cons/600 recal) broke **CE at N=1** (+1.157%) and **contam at N=2** (0.375).
  `lab/g3_coexistence.py` now records per-criterion first breaks and prints FINAL vs
  trajectory labels separately (no gate-logic change — only reporting).
- **Control A (recipe isolation) — PASS:** N=1 at the FULL G1b recipe (80-step
  consolidation, 800-step mex recal, text-KL 3.0, no joint pass) on the same spine
  (`t1-diag-realdata-mb4`) with the same fixture: util_home **0.912**, contam_base
  **0.217**, base CE **+0.880%** — reproducing G1b's 0.889 / 0.244 / +0.599%
  (`g1b_passport_d01_neutral3`). Poison check byte-identical as always.
  Verdict: the N=1 failures in the reduced-recipe G3 runs are **recipe depth**
  (600/150 recal steps), not spine, checkpoint, data, or harness. Note the
  single-insert CE bar is marginal in this harness: +0.60% (G1b) / +0.88% (Ctrl A) /
  +1.04% (Ctrl B N=1) across runs — CUDA-level variance straddles the 1% line at N=1.
- **Control B (N-scaling at full recipe) — FAIL:** N=8, 80-step consolidation +
  800-step mex recal per insert + 800-step closing joint (`g3_ctrlB_n8_full.json`).
  util_home min 0.929 (N=1) → 0.714 (N=2) → 0.589 (N=3) → 0.500 (N=4, bar) →
  0.429 → 0.392 → 0.354 → 0.162 (N=8); base CE +1.04% → +1.26% → +1.52% →
  +1.57% → +1.90% → +2.01% → +2.11% → **+2.28%**; after the closing joint pass:
  util 0.91 (e0) / 0.23–0.50 (e1–e7) — 7 of 8 below the bar — contam 0.08–0.12
  clean, CE +2.255%. Per-criterion first breaks: CE at N=1 (1.037, the marginal
  base), contam at N=2 (0.372, a transient spike that recovers), util at N=4 (0.50).
  **Recipe depth buys ~0.2-0.3 util at small N (v3's N=1 0.740 → Ctrl 0.929) but not
  the trend** — the decay slope is present at full recipe, so N-scaling is structural.
- **Bank crowding measured (not inferred):** mean |cos| between plug-in passport rows
  **0.275** vs plug-in↔base **0.066** (averaged over layers, N=8 post-joint) — the
  16/8 memory rows cluster ~4x more tightly with each other than with the base bank,
  and contest the same top-2 slots. This is the mechanism behind both the util decay
  (rows fight for home slots) and the group-level base displacement
  (p_any_plug_in_top2 ≈ 0.61-0.65 on base tokens). Capacity/orthogonality rungs added
  to the fallback tree.
- Decision: **N=16-full-recipe NOT queued** — its stated condition (N=8 full-recipe
  passing) failed, and the decay curve shows the failure is already at N=4-5 at full
  recipe; a 2-3h N=16 run would reproduce the same verdict.
- What it rules out: (1) "the fast recipe is the whole story" — FALSE (Control B);
  (2) "the N=1 break indicates spine/fixture damage" — FALSE (Control A reproduces
  G1b); (3) "more recal steps per insert fixes coexistence" — FALSE (800/insert +
  800 joint is the ceiling recipe and still breaks).
- What it confirms: the G3 FAIL verdict stands on the N-scaling evidence at full
  recipe; §8.5 kill criterion 1 remains fired; hierarchical domain→memory passports
  ACTIVE with capacity rungs as alternates.

## 2026-10-09 — g3_margin_rescore — OUTCOME LOG: taught-fact margins COLLAPSE at N=8 (retention 0.39 vs 0.70 bar) — addressing pivot CONFIRMED, hierarchical passports stay ACTIVE; but the util bar mispredicts in both directions and should be replaced by a margin-retention bar anyway
*(outcome log — the decisive cheap check on whether G3's util failure is a product failure or a metric artifact.  It is a product failure, and the metric is ALSO bad.)*
- Method: `lab/g3_margin_rescore.py` — the g3 N=8 full-recipe build (ctrlB recipe:
  80-step consolidation, 800-step mex recal/insert, 800-step §4.4a joint, text-KL 3.0,
  t1-diag-realdata-mb4 spine, same fixtures) rebuilt from scratch since
  `g3_coexistence.py` never persisted a state dict (only hashes), then taught-fact
  margins measured in three states: **pre** (bare spine), **solo** (episode j alone at
  the G1b N=1 condition = full-delivery reference), **lib** (the N=8 library state).
  Margin = logit(taught token) − logit(rejected token) at the first divergence of the
  correction pair, conditioned on prompt + shared prefix (for episode 0 this IS the
  G1b word-digit margin; demo probes measured alongside for direct G1b continuity).
  N=8 state saved to `lab/sandbox/g3_margin/n8_state.pt` (not committed).
- Headline numbers (`lab/results/g3_margin_n8.json`, `lab/logs/g3_margin_n8.log`):
  **mean retention vs solo 0.390** (bar 0.70) · mean margin gain lib +2.68 vs the
  70%-of-G1b bar +3.21 · Pearson(util_home, retention) = **0.61**.
  Per episode (util / retention / gain_lib / gain_solo):
  ep0 math_words 0.90 / **1.07** / +7.46 / +6.96 · ep1 travel_rain 0.34 / 0.55 /
  +2.00 / +3.66 · ep2 author_marco 0.42 / **0.10** / +0.46 / +4.59 · ep3
  distance_metric 0.45 / **−0.06** / −0.31 / +5.32 · ep4 treasure_pirate 0.38 /
  **0.77** / +8.05 / +10.51 · ep5 eval_indeed 0.41 / **0.10** / +0.85 / +8.62 ·
  ep6 emphasis_star 0.57 / 0.39 / +1.95 / +4.97 · ep7 dining_merci 0.30 / **0.20** /
  +1.00 / +4.95.  NLL margins agree (ep5: +11.3 solo → +1.27 lib).
  Demo-probe continuity (G1b's exact metric): pre −3.685 → solo −1.986 (gain +1.70,
  consolidation run-variance vs G1b's +4.59 — same pre-library baselines to 4
  decimals) → lib −2.041 (**retention 0.97**).
- Finding 1 (the product question): **margins collapse with the library** — 6 of 8
  experts below util 0.5 average +2.01 gain vs +6.27 solo, and the low-retention
  episodes (ep2/ep3/ep5/ep7 → 0.10/−0.06/0.10/0.20) lose essentially the whole
  taught fact. The memory does NOT "fire less often but land hard": it degrades.
  Even the counter-example set is small: ep4 holds 0.77 at util 0.385 (low util
  CAN deliver) but that is 1 of 8. Verdict per the pre-registered rule: **margins
  collapse → addressing pivot CONFIRMED, hierarchical domain→memory passports
  stay ACTIVE.**
- Finding 2 (the metric question): **util_home is a bad product gate in BOTH
  directions** — ep6 PASSED the util bar (0.567) and kept only 0.39 of its margin;
  ep4 FAILED the util bar (0.385) and kept 0.77. Util and retention are only
  moderately correlated (r=0.61). So the G3 bar should be revised to a
  margin-retention bar regardless of the architecture verdict (G1b criterion-a
  measurement on every insert, cheap: 3 forwards per expert). The architecture
  still fails the revised bar (0.39 vs 0.70) — the bar revision does NOT rescue
  passport coexistence at N=8, it just stops the gate from lying.
- Finding 3 (episode-shape note): episode 0 (G1's math/words fact, token-substitution
  shape) survives intact (fact retention 1.07, demo-probe retention 0.97) while
  most synthetic tag-append facts (ep2/ep3/ep5/ep7) collapse. Tag-append memories
  may be intrinsically more fragile under row competition — worth one control
  (margin re-score at N=8 with 8 math-style episodes) before blaming all of it on
  the bank; recorded as an open question, not a conclusion.
- Metric caveat (be honest about units): the "+2.68 vs 70%-of-G1b +3.21" comparison
  mixes divergence-logit margins (per-fact) with G1b's demo word-digit margins
  (per-probe) — scales are similar for ep0 (the same fact) but not identical for
  tag-style episodes (divergence vs EOS). The retention ratio (lib/solo, same
  metric within episode) is the scale-free evidence and it is unambiguous (0.39).
- What it rules out: "the util bar is wrong and the product survives" — FALSE
  (retention collapses). "Low util still delivers the memory in general" — FALSE
  (ep4 is the exception, not the rule). "Margins collapse strictly as a function
  of util" — too strong (r=0.61; same-util episodes differ wildly: ep3 0.45→−0.06
  vs ep4 0.38→0.77) — crowding hurts through channel competition, not just
  firing rate.
- What it confirms: §8.5 kill criterion 1; the G3 FAIL verdict on product grounds
  (retention), not just routing statistics; hierarchical domain→memory passports
  ACTIVE as the addressing build, with the rung-4 capacity/orthogonality
  candidates (orthogonal inits, larger passport_dim, per-domain sub-banks) still
  queued as cheaper attempts inside the current bank.

## 2026-10-09 — g3_pivot_shape (RUN B) — OUTCOME LOG: memory SHAPE is the lever — all token-substitution episodes hold (retention 0.884 vs mixed-fixture 0.390) at UNCHANGED crowding; tag-append memories are fragile regardless of addressing; pivot REFRAMED (shape episodes before consolidation)
*(outcome log — episode-shape control, the cheap fixture change.)*
- Method: `lab/g3_pivot_runs.py --run B` — N=8, ctrlB recipe (80-step consolidation,
  800-step mex recal/insert, 800-step §4.4a joint, text-KL 3.0, t1-diag-realdata-mb4
  spine), identical to the margin re-score except the fixture: **all 8 episodes are
  math/words-style token-substitution facts** (ep0's shape — stated digit form →
  corrected word form) over 8 distinct prompt domains (v1 lesson respected).
  Same three-state margin measurement (pre / solo / lib). State saved
  `lab/sandbox/g3_pivot/n8_shape_state.pt`. Results `lab/results/g3_pivot_shape.json`,
  `lab/logs/g3_pivot_shape.log`.
- **VERDICT: SHAPE_IS_THE_LEVER — mean retention 0.884** (bar 0.70; the mixed
  fixture scored 0.390 in identical machinery). Per episode (retention / util):
  math_words 1.12/0.93, cooking 0.95/0.47, astronomy 0.67/0.42, music 0.52/0.36,
  sports 1.08/0.39, anatomy 0.92/0.34, computing 0.85/0.30, geology 0.95/0.37.
- **Crowding unchanged** (plug-plug |cos| 0.258 vs mixed 0.275) and **util still
  low** (6 of 8 under 0.5) — yet retention holds. So the mixed fixture's collapse
  was NOT crowding and NOT low util; it was the tag-append memory shape. The
  ep0 anomaly resolves cleanly: it was shape, not position privilege
  (Pearson(insert-position, retention) = **−0.046** — ep7 at position 7 holds
  0.95; no decay with insertion order).
- What it confirms: memory SHAPE is a first-class variable — the task's
  reframe branch fires: **consolidate memories as token-substitution-style
  deltas / shape or dedup episodes before consolidation** instead of building
  hierarchical addressing for the retention problem. Low util (0.3-0.5) still
  delivers the memory when the memory is well-shaped (also strengthens the
  "util bar is wrong" finding from the re-score).
- Open: base-CE regression still fails (+4.12% vs ctrlB +2.26%) with per-expert
  contam clean (0.08-0.12) — the residual is GROUP top-2 occupancy of base
  tokens; shape does not touch it.

## 2026-10-09 — g3_pivot_ortho (RUN A) — OUTCOME LOG: capacity claim DEAD as stated — orthogonal rows (plug-plug |cos| 0.275 → 0.014) do NOT fix crowding (retention 0.452 vs 0.390 noise-level); row geometry is not the lever, memory shape is
*(outcome log — capacity rung, the orthogonality-penalty alternative to raising passport_dim.)*
- Method: `lab/g3_pivot_runs.py --run A` — N=8, ctrlB mixed fixture (ep0 math +
  7 tag-append) and ctrlB recipe, with a plug-plug squared-cosine penalty
  (λ=5.0) added to the mex recal objective (rows-only training preserved; base
  rows frozen by the same hook). Chosen over a literal passport_dim 64→256
  raise because the rows are 8-in-64 — room was never the constraint, training
  pressure was; a dimension raise alone would leave the pressure unchanged and
  be a weak test of "rows can be MADE mutually orthogonal". Solo refs reused
  from the re-score (penalty is exactly inert at N=1 — zero plug-plug pairs).
  First run crashed at Phase 3 on a KeyError (`pre_library_base_ce` absent from
  the ref JSON; fixed to always recompute from a bare reload — same value);
  rebuild is seed-deterministic and reproduced the crashed run's trajectory
  exactly. State saved `lab/sandbox/g3_pivot/n8_ortho_state.pt`. Results
  `lab/results/g3_pivot_ortho.json`, `lab/logs/g3_pivot_ortho.log`.
- **VERDICT: CLAIM_DEAD_ORTHOGONALITY_NOT_ENOUGH** — orthogonality was
  decisively ACHIEVED (plug-plug |cos| **0.275 → 0.014**) but mean margin
  retention stayed **0.452** (bar 0.70; vs 0.390 un-penalized = noise-level).
  Per episode (retention / util): math 1.07/0.88, travel_rain 0.14/0.32,
  author_marco 0.02/0.39, distance_metric 0.11/0.45, treasure_pirate 0.85/0.39,
  eval_indeed 0.34/0.40, emphasis_star 0.99/0.55, dining_merci 0.10/0.30.
- **The cross-run pattern is the proof**: with rows forced orthogonal, the same
  shape split persists — token-substitution facts hold (ep0 1.07, ep4 0.85
  prefix-tag, ep6 0.99), tag-append facts collapse (ep2 0.02, ep7 0.10). Row
  geometry does not predict retention; memory shape does (RUN B, same day).
- What it rules out: the capacity claim "plug-in rows can be made mutually
  orthogonal; that fixes crowding" — the first half is achievable and the
  second half is false. Also rules out (indirectly) a bare passport_dim raise:
  8 rows fit orthogonally in 64 dims already, so 256 dims would not change the
  training pressure. Rung-4 candidate (a) "orthogonal row inits" is closed
  by the same logic (inits don't survive the training pressure); (b) larger
  passport_dim demoted; (c) per-domain sub-banks = the hierarchy rung proper.
- Open: base-CE regression +2.46% (vs ctrlB +2.26%) — the ortho penalty
  neither helps nor hurts it; group top-2 occupancy of base tokens is
  orthogonal to row geometry too.

## 2026-10-09 — g3sh queue launch (memory-shaping G3 line) — OPS INCIDENT: pre-edit farm runner claimed all 3 g3sh jobs in the seconds before its restart and launched them with the pretrain TRAINER (2 OOM'd in dummy-pass, 1 orphaned) — recovered same cycle, no research lost
*(ops log — farm queue race, not an experiment result. Nothing is ruled out.)*
- Evidence: preserved at `lab/logs/g3sh-{family-ab,n8-shaped,n16-shaped}-misslaunch.log`
  + `lab/results/g3sh-misslaunch-records.json`. Runner record naming keeps
  both: the bare `lab/results/g3sh-*.json` still holds the misslaunch
  record (failed:true, trainer OOM traceback), and each successful re-run
  finalized to a timestamped copy (`g3sh-family-ab.1791612567.json`,
  `g3sh-n8-shaped.1791614632.json`, …). Old runner pid 67044
  (pre-`script`-field build) claimed all
  three defs and launched `Pretrain/02_pretrain_linear.py` instead of
  `lab/g3_family_ab.py` / `lab/g3_shaped_run.py`; family-ab and n8 died in the
  trainer's lab_small dummy-pass VRAM pre-alloc OOM, n16 was orphaned in
  `lab/running/`.
- Root cause: race between queue-file publication and runner restart — the
  defs landed while the old runner was still watching, so the stale binary
  claimed them.
- What it rules out: nothing in the research line. Recovery in the same
  cycle: orphan claim moved back to `lab/queue/`, defs re-copied from
  `lab/queue_done_prior/`, runner restarted (pid 131483, slots 1,
  systemd-inhibit wrapped); the new runner launched `lab/g3_family_ab.py`
  correctly (verified in ps). Job ids resumed cleanly; only wall-clock lost.
- Next tried: standing lesson — **restart the farm runner BEFORE writing
  queue files**, never after. The `script`-field support in `lab/runner.py`
  is the fix that makes the queue defs self-describing; keep
  `lab/queue_done_prior/` copies as the recovery source of truth.

## 2026-10-09 — g3_family_ab (§8.4 reserve sketch, family-row two-stage routing) — OUTCOME LOG: MIXED as recorded, SIGNAL LATER INVALIDATED — CE −3.46% but occupancy does NOT drop (p_any 0.588→0.701) and memory delivery collapses (retention 0.674→−0.081); root cause = index-space bug in FamilyRouter (see below), not the mechanism
*(outcome log — pure inference-time routing swap on the saved RUN B N=8 shaped
library state, no training; isolates the addressing mechanism alone.)*
- Method: `lab/g3_family_ab.py` on `lab/sandbox/g3_pivot/n8_shape_state.pt`
  (RUN B state) + solo refs from `lab/results/g3_pivot_shape.json`. Families
  clustered by avg-linkage on member rows across layers → `[[0],[1],[2],
  [3,4,5,6,7]]` (k=4). `FamilyRouter` two-stage: base+family rows contest
  top-2; the family slot resolves to argmax member; stage-1 weights
  renormalized. Family rows are untrained means of member rows (the sketch's
  prototype rule). Results `lab/results/g3_family_ab.json`,
  `lab/logs/g3sh-family-ab.log` (34.5s).
- **VERDICT (as printed by the run): MIXED** ("CE/occupancy improve but
  memory delivery collapses — family argmax needs calibration before it
  counts") — later invalidated as a mechanism measurement by the ROOT CAUSE
  bug below. Flat A vs family B:
  CE **7.0528 → 6.8086 (−3.46%)**, p_any_plug_in_top2 0.588 → **0.701**
  (occupancy got WORSE, not better — the sketch's premise that base tokens
  would contest only k rows did not materialize at k=4 with untrained family
  rows), p_both 0.169 → 0.304. Margins collapse negative on all 8 episodes
  (−7.68…−2.77 vs flat's +12.09…−6.12), util_home 0.20–0.38.
  Retention A mean **0.674** → B mean **−0.081**.
- **ROOT CAUSE (found on follow-up inspection, 2026-10-10): the collapse was
  an index-space BUG in `FamilyRouter`, not the family-mean/argmax rule.**
  `cluster_families()` returns member-LOCAL indices (0..n_mem-1), but the
  router (a) built family rows via `passports.data[mem]` on the FULL bank —
  so family `[[0]]`'s "mean of member 0" was literally **base row 0**, family
  `[[3,4,5,6,7]]`'s mean was base row 3 + members 0–3 — and (b) emitted
  stage-2 results as `out_i = members[argmax]` (local ids) where global
  expert ids are expected, so winning family slots dispatched to the WRONG
  experts (singleton families dispatched to base experts 0/1/2 — their
  memories were never delivered at all). Both bugs silence memories while
  letting base CE drift back — which is exactly the recorded symptom (CE
  −3.46%, retention → −0.081, occupancy mis-counted). Unit-checked on CPU
  with real geometry (base norms ~2, member norms 17–34): buggy fam rows
  equal base rows exactly; fixed version delivers global member ids and
  family means with the n_base offset. The "CE drop = silenced memories"
  read stands, but the silencing was a bug, not a mechanism verdict.
- **Honest read (as recorded, pre-fix):** the CE drop is a symptom of
  silenced memories (retention ≈ 0), not a clean addressing win. Whether
  untrained family-mean rows + argmax ALSO collapse after the index fix is
  settled by the one cheap rerun (`g3sh-family-fix`, same RUN B state, fixed
  router, `lab/results/g3_family_ab_fix.json`) — not by the buggy numbers.
- Fixture-mismatch caveat (does NOT affect the shaped N=8/N=16 runs, which
  train+measure self-consistently): `SHAPED_8[0]` "math_words" is measured in
  "4."/"Four." form while the RUN B state trained `_ep_math`'s "The answer is
  4."/"The answer is four." — A-side math retention −0.56 is that artifact;
  ep1–7 A-side retentions reproduce RUN B exactly (0.947/0.671/0.521/1.081/
  0.924/0.852/0.954).
- What it rules out: nothing about the two-stage mechanism from the buggy
  run — its signal (MIXED) is invalidated as a mechanism measurement. The
  run still establishes the harness (FamilyRouter swap, cluster_families,
  occupancy/margin/retention measurement) and the fact that the recorded
  MIXED numbers must never be quoted as evidence against family routing.
- Next tried: one cheap fixed-router rerun on the identical RUN B state
  (queue `g3sh-4-family-fix.json`, def archived in `lab/queue_done_prior/`);
  if it still collapses, the family-mean/argmax rule itself is implicated
  and family-row mex calibration (owner=family) is the named next branch.
- **FIXED-ROUTER RERUN (2026-10-10, `lab/results/g3_family_ab_fix.json`,
  `lab/logs/g3sh-family-fix.log`, 39.2s): SIGNAL NEGATIVE.** Same RUN B
  state, same families `[[0],[1],[2],[3,4,5,6,7]]`, index bug fixed: CE
  7.0528 → 6.9525 (**−1.42%**, less than the buggy −3.46%), p_any 0.588 →
  **0.630** (occupancy again did NOT drop — the sketch's "base tokens contest
  only k rows" premise fails at k=4 even with correct routing), retention
  mean 0.674 → **0.426** (excl. the math fixture artifact: 0.567 vs flat
  0.756). Delivery did recover vs the bug (cooking 0.99, computing 0.77,
  music 0.55, astronomy 0.64 hold; math's −0.56 is the fixture mismatch on
  both sides; sports/anatomy/geology 0.26/0.33/0.43 are the family-mean +
  argmax casualties — their family is the 5-member cluster whose mean row
  is most diluted). Verdict line: "two-stage addressing alone insufficient
  — residual needs base-neutral weighting or another fix". So the bug
  explains the catastrophic collapse but not the mechanism failure: with
  correct indices, untrained family-mean rows + argmax still (a) fail to
  reduce group top-2 occupancy and (b) cost ~0.25 retention vs flat. The
  §8.4 reserve sketch as a zero-training drop-in is dead; family-row mex
  calibration (owner=family) is the named next branch if the occupancy
  residual stays unaddressed by other means.

## 2026-10-09 — g3_shaped N=8 (memory-shaping consolidation module) — OUTCOME LOG: shaped memories HOLD at N=8 (mean retention 1.071 vs bar 0.70 — reproduces the RUN B anchor once a solo-ref outlier is excluded) but the group top-2 base-CE residual PERSISTS and is LARGER (+8.9% vs RUN B +4.1%); computing_counts is the lone per-episode miss (0.467)
*(outcome log — first run of the reusable memory-shaping module
(`lab/memory_shaping.py`), train+measure self-consistent.)*
- Method: `lab/g3_shaped_run.py` N=8 on `ms.SHAPED_8` (8 token-substitution
  episodes × 3 facts, 8 distinct prompt domains; shape_report 24/24
  substitution deltas, 0 insert/delete). Full ctrlB recipe (80-step
  consolidation, 800-step mex recal/insert, 800-step §4.4a joint, text-KL
  3.0) on the t1-diag-realdata-mb4 spine; solo refs rebuilt fresh per episode
  in the same run, so retention is self-consistent (no fixture mismatch).
  Farm job g3sh-n8-shaped (runner pid 131483), 2064s wall. State
  `lab/sandbox/g3_shaped/n8_shaped_state.pt`; results
  `lab/results/g3_shaped_n8.json`, `lab/logs/g3sh-n8-shaped.log`.
- **VERDICT: PASS — mean retention 1.071** (bar 0.70; anchor RUN B 0.884).
  Per episode (retention / util / contam): math_words 1.023/0.951/0.090,
  cooking 1.001/0.429/0.108, astronomy 1.038/0.447/0.091, music 0.774/0.346/
  0.097, sports **2.476**/0.431/0.106, anatomy 1.064/0.336/0.073, computing
  **0.467**/0.295/0.107, geology 0.726/0.447/0.088. The runner's anchor check
  flags a deviation (1.071 vs 0.884 ± 0.15) — it is a solo-ref artifact, not
  a real gain: sports' solo ref was unusually weak (gain_solo +6.4 vs +15..19
  for every other episode) while library delivery was strong (+15.9), inflating
  the mean to 2.476 for that episode. **Excluding sports the mean is 0.870,
  inside the anchor window and essentially RUN B's number.** 7 of 8 episodes
  clear the 0.70 per-episode bar; computing (0.467) is the lone miss.
- **Base-CE residual persists and is larger than RUN B's:** 6.7735 → 7.3771
  (**+8.912%**, bar <1%) with p_any_plug_in_top2 **0.589** / p_both 0.171
  (RUN B: +4.12% at p_any ≈ 0.59; ctrlB mixed +2.26%). Per-expert contam is
  clean (max 0.108) — the residual is again the GROUP top-2 occupancy of base
  tokens. Shaping the memories does not touch it; in this self-consistent
  rebuild the magnitude is ~2× RUN B's, so shape is neutral-to-worse on the
  occupancy axis. This is the residual the family-row A/B was prototyped
  against (see the g3_family_ab entry: MIXED — not fixed).
- Bank geometry note: plug-in rows sit at norms 17–34 vs base rows ~2 — the
  insert-trajectory util decay (e0 0.95 → e7 0.27 across inserts) tracks
  growing plug-bank norm, not position (ep7 still delivers 0.726).
- What it rules out: "shaped memories fix the base-CE residual" — they do
  not; shape is the retention lever only. Also confirms the reusable module
  reproduces RUN B's retention outcome outside the pivot script.
- Next tried: N=16 shaped (g3sh-n16-shaped, budget 180 min) — the headline
  G3 coexistence number under the shaped fixture; family-row two-stage was
  the A/B for the CE residual (filed separately, MIXED).

## 2026-10-10 — g3_shaped N=16 (memory-shaping consolidation module) — OUTCOME LOG: G3 coexistence PASSES at N=16 for the first time — mean retention 0.737 vs the standing 0.70 bar (vs mixed-shape 0.390 at N=8) with clean per-expert contam (max 0.085); base-CE occupancy residual persists (+6.12% at p_any_plug 0.621)
*(outcome log — the headline G3 number: coexistence at N=16 under the shaped
fixture and the margin-retention bar.)*
- Method: `lab/g3_shaped_run.py` N=16 on `ms.SHAPED_16` (16 token-substitution
  episodes × 3 facts, 16 distinct prompt domains; shape_report 48/48
  substitution, 0 insert/delete, dedup clean). Full ctrlB recipe (80 cons /
  800 mex recal per insert / 800 §4.4a joint, text-KL 3.0) on the
  t1-diag-realdata-mb4 spine; solo refs rebuilt fresh per episode in-run.
  Farm job g3sh-n16-shaped (budget 180 min, finished 6350s ≈ 106 min wall).
  State `lab/sandbox/g3_shaped/n16_shaped_state.pt`; results
  `lab/results/g3_shaped_n16.json`, `lab/logs/g3sh-n16-shaped.log`.
- **VERDICT: PASS — mean retention 0.737** (bar 0.70; N=8 shaped 1.071/0.870
  excl. outlier; RUN B anchor 0.884; mixed-shape N=8 0.390). Per episode
  (retention / util / contam): math 1.010/0.951/0.085, cooking 0.620/0.245/
  0.061, astronomy 0.466/0.275/0.071, music 0.851/0.339/0.060, sports
  0.577/0.313/0.051, anatomy 1.004/0.210/0.038, computing 0.890/0.212/0.052,
  geology 0.855/0.275/0.057, history 0.604/0.299/0.048, travel 0.715/0.238/
  0.043, weather 0.474/0.141/0.043, money 0.440/0.196/0.057, chess
  1.098/0.282/0.049, ocean 0.755/0.125/0.039, forest 0.804/0.181/0.051,
  bridge 0.633/0.177/0.038. **9 of 16 clear 0.70 per-episode** (math, music,
  anatomy, computing, geology, travel, chess, ocean, forest); the 7 misses
  (0.44–0.63) are a soft degradation, not the mixed fixture's collapse.
  Util decays with insertion order (e0 0.95 → e15 0.18) and util_min is
  0.125 — yet most memories still deliver, reconfirming the "util bar
  mispredicts" finding; no position confound (ep12 chess at position 12
  holds 1.098).
- **Base-CE residual persists under shaped memories:** 6.7735 → 7.1878
  (**+6.116%**, bar <1%) with p_any_plug_in_top2 **0.621** / p_both 0.223,
  per-expert contam clean (max 0.085). N=8 shaped was +8.91% at p_any 0.589
  — the occupancy residual does not grow catastrophically with N (and N=16
  is even milder than N=8 here), but shape does not fix it at either N. The
  open G3 sub-problem is unchanged: group top-2 occupancy of base tokens.
- What it confirms: **memory shape converts G3 coexistence from FAIL to
  PASS at N=16 under the product bar** (margin retention vs solo ≥ 0.70),
  where the mixed-shape fixture scored 0.390 at half the N. Bank crowding
  (plug norms 17–40 vs base ~2) and util decay are real but no longer
  fatal for well-shaped substitution memories. The §8.5 kill criterion-1
  branch (different addressing mechanism for coexistence) is NOT triggered
  by retention anymore — it survives on the base-CE residual alone.
- Next tried: the base-CE occupancy residual is the remaining G3 sub-bar;
  the family-row two-stage A/B (reserve sketch) was prototyped against it
  and is NEGATIVE as a zero-training drop-in (see the g3_family_ab entry —
  index bug found and fixed; with correct routing p_any still 0.588→0.630).
  Named next branch for the residual: family-row mex calibration
  (owner=family) or base-neutral weighting of plug rows.

## 2026-10-10 — g3res-plugbias (branch b: base-neutral plug weighting) — OUTCOME LOG: NEGATIVE — a plug-row logit bias cannot separate base from home at this overlap: the CE/retention trade-off curve is monotone and no bias satisfies both bars (best retention-feasible point still +6.7% base CE; CE-feasible points kill delivery)
*(outcome log — the cheaper of the two named residual branches; pure
inference-time, one knob, no training.)*
- Method: `lab/g3_plug_bias_ab.py` on the saved N=8 shaped library state
  (`lab/sandbox/g3_shaped/n8_shaped_state.pt`, solo refs
  `lab/results/g3_shaped_n8.json`). `PlugBiasRouter` subtracts β from every
  plug-row logit before the top-2 contest (β=0 is exactly the as-saved
  router — verified: reproduces CE 7.3771 / retention 1.0712 / p_any 0.590
  to 3 decimals). Sweep β ∈ {0, 0.5, …, 5}, same margin/occupancy
  machinery as the state's run (bias-aware occupancy — `g3.routing_matrix`
  computes logits directly and would silently measure β=0). Farm job
  g3res-plugbias (57s GPU). Results `lab/results/g3_plug_bias_ab.json`,
  `lab/logs/g3res-plugbias.log`.
- **VERDICT: NEGATIVE.** The curve is monotone and the two bars are
  mutually exclusive: β 0→5 walks CE +8.91% → +0.21% and p_any 0.590 →
  0.039, but retention 1.07 → 0.009 in the same span. Per β
  (CE reg / p_any / retention): 0.0: +8.91/0.590/1.07; 0.5: +7.95/0.476/
  0.95; 1.0: +6.73/0.377/**0.70** (last retention-feasible point, still
  6.7× the CE bar); 2.0: +3.44/0.231/0.44; 3.0: +1.41/0.135/0.15;
  3.5: **+0.89**/0.100/0.07 (first CE-feasible point, retention dead);
  5.0: +0.21/0.039/0.01. util_home_min falls 0.295 → 0.003.
- Root cause (mechanism): base tokens and home tokens overlap in the
  plug-row logit distribution — a scalar bias cannot keep home delivery
  (which needs plug rows to win by evidence) while suppressing base
  hijacks (which happen at similar evidence levels). The residual is not a
  calibration artifact; it needs structure (family-level training) or
  expert-side base-neutrality.
- What it rules out: any single-threshold / bias / temperature reweighting
  of plug rows as the residual fix — the whole family of monotone plug
  penalties is closed by this curve (β is the strongest monotone member:
  it dominates admission margins and weight scaling on the same axis).
- Next tried: branch (a), family-row mex calibration (owner=family,
  section-4.4a target at family granularity over the two-stage router) —
  the other named branch, queued same cycle as `g3res-2-familycal.json`.

## 2026-10-10 — g3res-familycal (branch a: family-row mex calibration, owner=family) — OUTCOME LOG: NEGATIVE — trained family rows keep delivery perfectly (retention 1.071) but only trim the residual (+8.91% → +7.45% base CE at p_any 0.473); family rows never go silent on base
*(outcome log — the other named residual branch, same cycle, same saved
N=8 shaped state.)*
- Method: `lab/g3_family_cal.py` on `lab/sandbox/g3_shaped/n8_shaped_state.pt`
  (refs `lab/results/g3_shaped_n8.json`). `g3_family_ab`'s fixed
  `FamilyRouter` two-stage (avg-linkage families on this state:
  `[[0],[1,2],[3,4,5,6],[7]]`, k=4); ONLY fam_rows trained (base rows,
  member rows, experts, query, spine frozen) with the §4.4a mutual-exclusion
  target at family granularity: home of family f → family row f gets
  owner_mass 0.55, base rows share 1−0.55, other families 0; base-mix
  tokens → base rows share 1.0, family rows 0. 800 AdamW steps (lr 1e-2),
  two-stage occupancy/util measurement (family slot resolves to argmax
  member). Farm job g3res-familycal (369s GPU). Results
  `lab/results/g3_family_cal.json`, `lab/logs/g3res-familycal.log`.
- **VERDICT: NEGATIVE (retention side PASS, CE side FAIL).** As-saved flat
  7.3771 (+8.91%) → family-cal 7.2781 (**+7.45%**, bar <1%); p_any_plug
  0.590 → **0.473**; p_both 0.171 → 0.095. **Retention 1.071 unchanged**
  (per-ep: math 1.02, cooking 0.92, astronomy 0.95, music 0.70, sports
  2.64, anatomy 1.04, computing 0.48, geology 0.81) — the two-stage
  delivery path costs nothing once family rows are trained; util_home_min
  0.204. So (a) is strictly better than the untrained fixed rerun
  (which lost 0.25 retention) but the CE bar is missed by 7×.
- Root cause (mechanism): the calibration loss barely moves (98.2 → 94.0
  over 800 steps) — family rows are means of high-norm member rows (plug
  norms 17–34 vs base ~2), so they win the stage-1 contest on nearly every
  token; the mex target's base-silence pressure is too weak at lr 1e-2 /
  800 steps to overcome that norm advantage, and the home terms actively
  push the same rows to fire. Same overlap story as branch (b): base and
  home evidence is not separable by row-side pressure alone.
- What it rules out: family-row mex calibration at the joint-cal cadence
  (800 steps, lr 1e-2) as a sufficient residual fix. Together with the
  (b) sweep, BOTH named row-side branches are closed: row-side pressure
  (monotone penalties, trained family rows) cannot deliver base-CE < 1%
  while holding delivery. The residual needs expert-side base-neutrality
  (the G1b lever that worked at N=1: text-KL 3.0 gave +0.60% there — the
  same coefficient at N=8 group occupancy still gives +8.9%) or a
  fundamentally different addressing mechanism.
- Next tried: none inside this cycle — both named branches are exhausted
  and documented. Named candidates for the next cycle: (i) strengthen
  base-neutrality at consolidation time (text-KL above 3.0, scaled with N)
  so contaminated tokens are CE-cheap per G1b's finding; (ii) accept the
  residual as a known cost of coexistence and re-scope the bar; (iii) the
  hierarchical domain→memory addressing rung (§8.5's live fallback tree)
  if both fail.

## 2026-10-10 — g3res-textkl6 / g3res-textkl12 (N-scaled base-neutrality at consolidation) — OUTCOME LOG: the KL lever WORKS and scales the residual down ~5× (base CE +8.91% → +3.25% → +1.82% at text_KL 3/6/12) with retention holding (1.07/1.33/0.75) — but the <1% CE bar is still not met at KL 12 and retention is already sliding
*(outcome log — branch (i) from the residual entry: G1b's text-KL 3.0 gave
+0.6% at N=1 but +8.9% at N=8; test pressure scaled with N.)*
- Method: `lab/g3_shaped_run.py` N=8 shaped, fresh full consolidation per
  arm with `G3S_TEXT_KL` ∈ {6.0, 12.0} (new env knob; 3.0 baseline is the
  recorded `g3_shaped_n8.json` run — same script, same seed 0). Self-
  consistent solo refs rebuilt per arm. Farm jobs g3res-textkl6
  (1994s) / g3res-textkl12 (2075s). Results
  `lab/results/g3_shaped_n8_kl{6,12}.json`, logs `lab/logs/g3res-textkl*.log`.
- **Numbers (bar: CE regression <1% AND retention ≥0.70):**
  text_KL 3.0: CE +8.91%, retention 1.071, p_any 0.590 (baseline);
  text_KL 6.0: CE **+3.25%**, retention **1.325** (PASS retention),
  p_any 0.617;
  text_KL 12.0: CE **+1.82%**, retention **0.750** (PASS retention, at the
  edge), p_any 0.626.
  Per-ep KL12 (ret/util): math 1.01/0.95, cooking 1.03/0.46, astronomy
  0.70/0.41, music 0.48/0.32, sports 0.40/0.42, anatomy 0.90/0.32,
  computing 0.68/0.32, geology 0.81/0.40. (KL6's 1.325 mean includes
  another sports solo-ref outlier at 4.48; excl. sports ≈ 0.87.)
- **Mechanism confirmed (G1b's claim at N=8):** p_any_plug is ~0.62 at
  every KL — the lever does NOT change occupancy; it makes contaminated
  tokens CE-cheap (expert outputs stay base-like off-home). The residual
  halves roughly per KL doubling (log-linear), so the bar would need
  text_KL ~24–30 — but retention at KL 12 is already sliding toward the
  0.70 line (music 0.48, sports 0.40), so the dual bar likely cannot be
  met by KL pressure alone: the two objectives trade off.
- Verdict per the dual bar: **PARTIAL — both arms PASS retention, both
  MISS the CE bar (3.25% / 1.82% vs <1%)**. The N-scaled base-neutrality
  branch is the strongest residual lever measured so far (5× reduction)
  but is not sufficient by itself inside the tested range.
- What it rules out: "text-KL 3.0 is enough at N=8" (it is not — the N=1
  recipe does not scale); and "KL pressure alone can close the residual
  without retention cost" (extrapolation says the crossover needs KL
  where retention is likely gone).
- Next tried: the occupancy-dilution signal run (see next entry) in the
  same cycle, plus the real-spine 8-base companion.

## 2026-10-10 — g3res-dilution-clone (occupancy-dilution check) — OUTCOME LOG: DILUTION_WEAK_OR_ABSENT — the plug-induced base-CE residual does NOT shrink with base-pool share (8.91% → 7.71% → 7.67% → 7.77% at 4/8/16/32 base rows); the N=8 residual is not a tiny-pool artifact
*(signal run, not a gate — the scale-artifact hypothesis test.)*
- Method: `lab/g3_dilution_clone.py` on the saved N=8 shaped state
  (`lab/sandbox/g3_shaped/n8_shaped_state.pt`): expand the base pool from
  the 4 trained experts to 8/16/32 by CLONING them (weights copied,
  passport rows + 1e-2 noise) — constant expert quality, varying pool
  share (plug share 0.67 → 0.50 → 0.33 → 0.20). Control arm per pool
  size: same cloned pool WITHOUT plug rows, so the reported residual is
  plug-induced (CE_with_plugs − CE_clone_only), not clone-diversity drift
  (which is small: +0.46–0.48% at 8–32). Farm job g3res-dilution.
  Results `lab/results/g3_dilution_clone.json`,
  `lab/logs/g3res-dilution-clone.log`.
- **Curve (plug residual / p_any / retention):** 4: +8.91% / 0.590 / 1.07;
  8: +7.71% / 0.407 / 0.74; 16: +7.67% / 0.404 / 0.72; 32: +7.77% /
  0.401 / 0.72. Doubling the base pool ONCE cuts the residual ~1.3× and
  p_any ~1.4×; from 8→32 rows (share 0.50→0.20) both are FLAT.
- **Hypothesis verdict: REJECTED (weak at best).** If the residual were
  slot-arithmetic, plug share 0.67→0.20 would shrink it proportionally;
  it stays ~7.7%. What does shrink (p_any 0.59→0.40) does not translate
  to CE — so each remaining plug displacement is individually damaging
  (a plug expert's off-home output rewriting the mix), and clone-pool
  top-2 saturation (two clones of one expert) caps the dilution effect.
  The companion real-spine run (8 independently trained base experts,
  g3res-8base) checks whether trained diversity recovers what cloning
  could not.
- What it rules out: "the residual will dissolve at t2-scale expert pools
  for free" — pool growth alone does not fix it. Coexistence cost at N=8
  is a per-displacement base-neutrality problem (consistent with the KL
  arm being the effective lever).
- Next tried: g3res-8base (real trained 8-base spine, full shaped
  pipeline) queued same cycle.

## 2026-10-10 — g3res-8base (occupancy-dilution companion, real spine) — OUTCOME LOG: CONFUSING SIGNAL — base-CE residual is small on the trained 8-base spine (+1.44%) but memory delivery COLLAPSES there (retention 0.397 FAIL, p_any 0.795): the CE number is confounded with silencing, not clean dilution evidence
*(signal run — real 8 trained base experts instead of clones; cross-spine,
so expert-count is confounded with model size and base quality.)*
- Method: `lab/g3_shaped_run.py` N=8 shaped on the trained 8-expert
  passport spine `t0-8expert-realdata` (lab_tiny: dim 128, 4 layers,
  hidden 384, 8 base experts, top-2 — vs the lab_small reference's dim
  256 / 6 layers / 4 base). First attempt died in 8s on a path bug
  (repo-relative `G3S_CKPT` resolved against the job's `cwd=sandbox` —
  fixed by applying the `_abs` resolver; requeued same cycle). Full
  recipe, text_KL 3.0, self-consistent solos. 1436s. Results
  `lab/results/g3_shaped_n8_8base.json`, `lab/logs/g3res-8base.log`.
- **Numbers:** base CE 7.2086 → 7.3125 = **+1.44%** (bar <1%; vs +8.91%
  on 4-base lab_small) with p_any_plug **0.795** (HIGHER than 4-base's
  0.590) / p_both 0.334. **Retention 0.397 FAIL** (bar 0.70): per-ep
  (ret/util) math 1.23/0.97, cooking 0.11/0.55, astronomy 0.29/0.55,
  music 0.65/0.48, sports 0.11/0.53, anatomy 0.08/0.60, computing
  0.08/0.47, geology 0.64/0.54 — the token-substitution shape that PASSES
  on lab_small mostly collapses on this spine (util stays high; delivery
  does not follow).
- **Read: NOT clean dilution evidence.** The CE residual did shrink
  (+1.44% vs +8.91%) — but that coincides with memories being silenced
  (retention 0.397), the same confound that made the buggy family router
  look CE-positive. A plug expert that stops delivering is cheap;
  the number cannot be credited to the larger pool. p_any even ROSE to
  0.795 — occupancy is not what moved. Cross-spine caveats stack on top:
  weaker base model (pre-CE 7.21 vs 6.77), different size, different
  training run.
- Combined with the clone curve (same-spine: residual flat 7.7% as share
  falls 0.67→0.20), the dilution hypothesis stays REJECTED: pool share
  does not drive the residual; active off-home plug outputs do. The
  shape recipe's N=8 PASS is also spine-conditional — on lab_tiny/8-base
  the delivery side fails, so "shaped memories pass everywhere" is
  falsified as stated.
- What it rules out: crediting small CE regressions when retention has
  collapsed (report the pair, never CE alone); and "more trained base
  experts alone fix the residual" as a free lunch.
- Next tried: nothing further this cycle — the residual line's tested
  levers are now: row-side pressure (closed), KL base-neutrality (best
  lever: +8.9%→+1.8% at KL 12, retention sliding), pool growth (closed /
  confounded). Remaining honest options: KL ~24–30 with retention
  monitoring, hybrid KL + family-calibrated two-stage, or re-scope the
  <1% bar for N=8 coexistence.

## 2026-10-10 — g3res-delivery-diag (spine-conditional delivery) — OUTCOME LOG: CAUSE FOUND — the 8-base collapse is owner-mass falling below the delivery cliff AT THE SCORING POSITIONS (natural w_owner 0.38-0.55 at divergence vs a sharp cliff at 0.45-0.65); experts are fine (forced-owner = 1.14x solo); norm mismatch REJECTED (cosine routing is catastrophic)
*(diagnosis — why shaped memories deliver on lab 4-base but collapse on
the real t0 8-base spine. Probes P1-P8 on the two saved library states,
inference only.)*
- Method: `lab/g3_delivery_diag.py` on `n8_shaped_state.pt` (lab_small
  4-base, PASS) and `n8_8base_state.pt` (t0-8expert-realdata lab_tiny
  8-base, FAIL). Probes per episode: P1 as-is margin (reproduces runs);
  P2 forced-owner weight 1.0 (pure capability); P3 mix stats (owner
  weight, both-plug cofire, owner+wrong-memory); P4 cosine routing (norm
  hypothesis); P5 one-plug-slot cap (evict wrong memory); P6a/b forced
  0.65·owner+0.35·best-base / +0.35·wrong-mem; P7 owner-weight sweep
  α∈{0.45,0.65,0.85} with best base; P8 owner util/weight AT the
  divergence positions where delta_margin actually reads the logits.
  Farm job g3res-delivery-diag (39.5s). Results
  `lab/results/g3_delivery_diag.json`, `lab/logs/g3res-delivery-diag.log`.
- **Cause ranking (all hypotheses from the brief):**
  **(dominant) mix/owner-mass at scoring positions** — P2 forced-owner
  delivers 1.14x solo on 8-base (math 2.61, cooking 1.08, astronomy
  1.01, music 1.08, computing 0.99, geology 0.94) so the experts are
  intact; P7 shows a SHARP delivery cliff on lab_tiny: cooking +1.6 at
  α=0.45 → +14.2 at α=0.65 → +15.1 at α=0.85 (lab_small has no cliff:
  +15.6 flat at all α); P8 pins it to the read-out positions — owner
  util/weight AT divergence on the failing eps: cooking 0.33/0.38,
  computing 0.25/0.43, geology 0.50/0.55 vs math 1.00/0.79 (which
  delivers). Natural routing sits below the cliff exactly where the
  margin is read.
  **(a) owner_mass calibration mismatch — CONFIRMED as the fix lever:**
  the mex target's 0.55 sits on the cliff and 8 base rows dilute the
  softmax below it at scoring positions (home-token averages hide this:
  P3 w_own 0.63 on 8-base looks HEALTHIER than lab_small's 0.52).
  **(b) consolidation data mismatch / (d) recipe steps — REJECTED as
  delivery causes** by P2 (full solo-level delivery from the same
  experts); (d) survives only as the ep0 solo anomaly (+2.82 vs +10..16).
  **(c) router norm/temperature — REJECTED:** P4 cosine (content)
  routing collapses delivery on BOTH spines (ret 0.006 on 8-base, 0.16
  on lab_small) — norm-driven routing is essential to the mechanism.
  **wrong-memory cofire is real but not poison:** p_owner+wrong 0.26
  (8-base) vs 0.11 (lab_small), yet P5 (evict wrong memory → ret 0.32,
  no help) and P6b (0.35·wrong-mem delivers fine) show it only STEALS
  owner mass rather than injecting damage.
- Cross-spine summary: lab_small has no weight cliff and lower wrong-
  memory cofire → delivers at natural weights; lab_tiny has a cliff at
  0.45-0.65 and higher cofire → natural scoring-position weights fall
  below it → delivery collapses while routing metrics look fine.
- What it rules out: "the t0 spine can't hold shaped memories" (it can —
  P2 = 1.14x solo); "norm/temperature mismatch" and "cosine routing as
  a fix"; "owner-weight averages as a diagnostic" (they invert the
  story — measure at the scoring positions).
- Next tried: g3res-ownerfix — closing joint recal with owner_mass 0.85
  and scoring-position home coverage (the minimal recipe change under
  test), queued same cycle.

## 2026-10-10 — g3res-ownerfix (minimal fix for spine-conditional delivery) — OUTCOME LOG: FIX_FOUND — closing joint recal with owner_mass 0.55→0.85 lifts 8-base real-spine shaped retention 0.397 → 0.975 (mass alone) → 1.100 (mass + scoring-position home coverage); both clear the 0.70 bar
*(outcome log — the fix probe for the delivery-collapse diagnosis above.)*
- Method: `lab/g3_owner_mass_fix.py` on the saved `n8_8base_state.pt`,
  router-only recal (800 steps, lr 1e-2) three arms + baseline, retention
  measured against the run's own solo refs. Arm A: owner_mass 0.85 +
  home data = the SCORING prefixes (prompt+stated[:first-divergence],
  the exact tokens delta_margin reads). Arm B: owner_mass 0.85 + original
  prompt home data (mass-only control). Arm C: owner_mass 0.55 + scoring
  home (coverage-only control). Farm job g3res-ownerfix (482s). Results
  `lab/results/g3_owner_mass_fix.json`, `lab/logs/g3res-ownerfix.log`.
- **Numbers (retention vs the 0.70 bar):** baseline 0.397 FAIL;
  A mass085+scoring **1.100 PASS** (per-ep: math 2.53, cooking 1.07,
  astronomy 1.00, music 0.92, sports 0.61, anatomy 0.83, computing 0.94,
  geology 0.90); B mass085+prompt **0.975 PASS** (cooking 0.52, sports
  0.52 — the two eps the scoring coverage repairs); C mass055+scoring
  0.610 FAIL (coverage alone insufficient).
- **Minimal recipe change: owner_mass 0.55 → 0.85 in the closing §4.4a
  joint recal** — that alone clears the bar (arm B). Adding
  scoring-position home coverage to the recal data is the recommended
  refinement (arm A: +0.13 mean, fixes the worst episodes to ~0.9-1.0).
  Base CE is essentially unchanged (7.3125 → 7.352 = +1.55% vs +1.44%
  regression at baseline) — the fix buys delivery at negligible CE cost.
- Consistency with the diagnosis: mass is exactly the P7 cliff lever
  (owner weight must sit above ~0.5-0.65 on lab_tiny); coverage targets
  the P8 read-out positions. Arm C failing confirms coverage cannot
  substitute for mass; arm B passing confirms mass is the primary lever.
- What it rules out: "spine-conditional delivery is a capacity or data
  problem" — it is a calibration problem, fixable in one router-only
  pass on the existing state. The shaped-memory recipe is now
  spine-portable with the owner_mass bump.
- Next tried: KL 24/30 sweep points (g3res-kl24/kl30, queued same cycle)
  to complete the residual-line log-linear curve; a full pipeline 8-base
  run with owner_mass 0.85 + scoring home in the recipe is the natural
  end-to-end confirmation if the bump is adopted.

## 2026-10-10 — g3res-kl24 / g3res-kl30 (KL curve completion) — OUTCOME LOG: the KL lever SATURATES — base CE +1.53% (KL 24) / +1.67% (KL 30) vs +1.82% (KL 12): the <1% bar is NOT reachable by text-KL pressure; the log-linear extrapolation is falsified
*(outcome log — the second-priority sweep points for the residual curve.)*
- Method: `lab/g3_shaped_run.py` N=8 shaped, fresh consolidation at
  text_KL 24.0 / 30.0 (G3S_TEXT_KL knob), same seed/recipe as the 3/6/12
  arms. Farm jobs g3res-kl24 (retention-side data below) and g3res-kl30
  (2007s). Results `lab/results/g3_shaped_n8_kl{24,30}.json`,
  logs `lab/logs/g3res-kl*.log`.
- **Complete curve (base-CE Δ / retention / p_any):** KL 3: +8.91% / 1.07
  / 0.590; KL 6: +3.25% / 1.33 / 0.617; KL 12: +1.82% / 0.75 / 0.626;
  KL 24: **+1.53%** / 0.788 / 0.606; KL 30: **+1.67%** / 0.804 / 0.596.
  Per-ep KL24 ret: 1.00/0.90/0.73/0.74/0.71/0.73/0.56/0.93 (all but
  computing ≥ 0.70); KL30: 0.94/0.80/0.50/0.46/1.40/0.74/0.73/0.87.
- **Verdict: SATURATION at ≈ +1.5-1.7% from KL 12 on.** The earlier
  log-linear read ("halving per KL doubling → KL 24-30 reaches <1%") is
  **falsified** — the residual stops falling after KL 12 while retention
  holds (0.75-0.80 across 12-30). Mechanism: text-KL makes contaminated
  tokens cheap only up to the point where the expert's off-home output
  matches the base DISTRIBUTION; beyond that the remaining residual is
  the routing displacement itself (a plug row still writes at ~0.6
  weight on base tokens), which no output-side pressure touches.
- What it rules out: text-KL pressure (at any tested strength) as a
  route to the <1% CE bar. The bar now needs an addressing-side change
  (reduce base-token plug occupancy), a displacement-aware mix rule, or
  an explicit re-scope (e.g. <2% as the N=8 coexistence CE bar).
- Next tried: nothing further this cycle — full state of the residual
  line: row-side closed, pool-side closed/confounded, KL saturates
  at +1.6%. The owner_mass 0.85 fix (delivery) is the actionable recipe
  change from this cycle.

## 2026-10-10 — g3res-8base-e2e (end-to-end G3 confirmation, real spine) — OUTCOME LOG: PASS — N=8 shaped with owner_mass 0.85 + scoring-prefix home baked into the recipe from the start reaches retention 0.944 (was 0.397) on t0-8expert-realdata; spine-portability question CLOSED GREEN
*(outcome log — the natural G3 confirmation: the g3res-ownerfix arm-A
recipe run as the pipeline recipe, not as a post-hoc pass.)*
- Method: `lab/g3_shaped_run.py` N=8 on the t0-8expert-realdata spine
  (lab_tiny 8-base) with the new `G3S_OWNER_MASS=0.85` +
  `G3S_HOME_MODE=scoring` knobs (recal home data = prompt+stated[:first-
  divergence] scoring prefixes, owner_mass 0.55→0.85 in every mex pass —
  solo, per-insert, and closing joint), full recipe (80/800/800,
  text_KL 3.0), self-consistent solos under the same recipe. Farm job
  g3res-8base-e2e (1291s). Results
  `lab/results/g3_shaped_n8_8base_e2e.json`, state
  `lab/sandbox/g3_shaped/n8_8base_e2e_state.pt`,
  log `lab/logs/g3res-8base-e2e.log`.
- **VERDICT: PASS — retention 0.944** (bar 0.70; without the recipe
  change the same spine/fixture scored 0.397). Per-ep (ret/util/contam):
  math 1.07/1.00/0.07, cooking 1.10/0.79/0.16, astronomy 1.00/0.87/0.12,
  music 0.81/0.76/0.13, sports 1.00/0.81/0.21, anatomy 0.69/0.88/0.17,
  computing 0.88/0.81/0.18, geology 1.00/0.81/0.15. 7/8 clear 0.70
  (anatomy 0.693 is a hair under). Scoring-position util is uniformly
  high (util_min 0.76 — the P8 gap is closed by construction).
- **Base CE: 7.2086 → 7.4139 = +2.85%** (documented; bar for this run is
  documentation, not <1% — that bar belongs to the residual line and is
  still open at ~+1.6-2.8% depending on spine/recipe). Per-expert
  contamination clean: contam_max 0.208 < 0.3 (p_any_plug 0.789,
  p_both 0.400). Note the e2e CE is a little above the post-hoc fix's
  +1.55% — the baked-in recipe delivers more (util 0.8-1.0 vs 0.5),
  i.e. more plug mass on base tokens too; report both honestly.
- What it confirms: the delivery collapse was a calibration problem and
  the owner_mass+scoring-home recipe generalizes from a post-hoc pass to
  the full pipeline — **shaped memories are spine-portable** (lab_small
  4-base and lab_tiny 8-base both PASS the retention bar with the
  recipe). The N=16 shaped variant on the same spine is queued as the
  full coexistence story (g3res-8base-e2e-n16).
- Next tried: N=16 shaped on the same spine with the same recipe.

## 2026-10-10 — g3res-8base-e2e-n16 (full coexistence story, real spine) — OUTCOME LOG: PASS — N=16 shaped on t0-8expert-realdata with the owner_mass 0.85 + scoring-home recipe holds retention 0.953 (15/16 eps ≥ 0.70) — coexistence at N=16 is green on the real spine
*(outcome log — the N=16 completion of the e2e confirmation.)*
- Method: `lab/g3_shaped_run.py` N=16 (`ms.SHAPED_16`) on the same
  t0-8expert-realdata 8-base spine, same recipe as g3res-8base-e2e
  (owner_mass 0.85, home_mode scoring, 80/800/800, text_KL 3.0),
  self-consistent solos. Farm job g3res-8base-e2e-n16 (3780s). Results
  `lab/results/g3_shaped_n16_8base_e2e.json`, state
  `lab/sandbox/g3_shaped/n16_8base_e2e_state.pt`,
  log `lab/logs/g3res-8base-e2e-n16.log`.
- **VERDICT: PASS — retention 0.953** (bar 0.70; N=8 same recipe 0.944;
  the un-fixed recipe at N=8 was 0.397). Per-ep (ret/util/contam): math
  1.10/1.00/0.03, cooking 1.09/0.75/0.12, astronomy 0.99/0.75/0.07,
  music 0.79/0.74/0.08, sports 1.00/0.78/0.14, anatomy 0.69/0.69/0.10,
  computing 0.85/0.73/0.10, geology 1.00/0.69/0.07, history 1.00/0.73/0.12,
  travel 1.00/0.75/0.13, weather 0.99/0.71/0.09, money 0.98/0.75/0.09,
  chess 1.01/0.75/0.10, ocean 0.80/0.77/0.09, forest 0.97/0.66/0.14,
  bridge 0.99/0.76/0.08. **15 of 16 clear 0.70** (anatomy 0.687
  marginal). util_min 0.661, **contam_max 0.139 clean** (bar 0.3).
- **Base CE: 7.2086 → 7.4735 = +3.68%** (documented; p_any_plug 0.889,
  p_both 0.655 — 16 plug rows on an 8-base pool saturate top-2 more, as
  expected; the residual grows with N on this spine: +2.85% at N=8 →
  +3.68% at N=16, consistent with the group-occupancy residual story).
- What it confirms: **the full coexistence story is green on the real
  spine** — N=16 shaped memories with the calibrated recipe hold the
  margin-retention bar where the un-calibrated recipe collapsed at N=8.
  Retention is flat in N (0.94 → 0.95) once owner_mass sits above the
  delivery cliff; the open item is only the base-CE residual (+3.7%),
  which is the known group-occupancy cost (KL lever saturates at
  ~+1.6% on lab_small; not yet re-tested at this N/spine).
- Next tried: nothing further this cycle — spine-portability and N=16
  coexistence on the real spine are both closed green with the
  owner_mass 0.85 + scoring-home recipe.

## 2026-10-13 — kv-v05-gemv-dead-end — hipblas GEMV cannot accelerate the M=1 decode path on gfx906 (slower than the GEMM, and fp32 GEMV crashes) — the 50-100 tok/s class needs a custom HIP kernel, not torch/hipblas
- Evidence: torch.profiler on 4 bare KV-decode steps of the v0.5 spine
  (`tliao@100.64.0.7`, fp16) shows `aten::mm` = 52% of Self CUDA (640 calls,
  338us avg; kernel `Cijk_Alik_Bljk_HHS_BH_MT64x64x16`) + `addmm` 34% — the
  M=1 GEMM runs at ~110 GB/s effective on 32MB weight reads, and that floor
  (~111ms/step bare, 4.3GB active weights/token) caps single-stream decode at
  ~9 tok/s regardless of the decode harness.  Microbench on GPU 6 (fp16,
  (1,4096)x(4096,4096)): `torch.nn.functional.linear` 293us vs `torch.mv`
  844us — hipblas GEMV is ~3x SLOWER than the rocBLAS M=1 GEMM; `torch.mv`
  on the fp32 (6144,3584) wrap-shaped matrix raises
  `RuntimeError: CUDA error: HIPBLAS_STATUS_INTERNAL_ERROR` (hipblasSgemv) —
  same missing-kernel family as the d0-gfx906-moe-kernels entry.
- What it rules out: the standard "use GEMV kernels for decode" fix on
  gfx906 as shipped in torch 2.9.1+rocm6.4; also index_select weight-gather
  copies as the bottleneck (pipelined copies measure ~2.7us) and CUDA graphs
  as more than a ~15% win (GPU busy 86% of the step — CPU dispatch is not
  the binding constraint).
- Next tried: shipped `lab/kv_cache_v05.py` with the fastest safe path
  (fixed top-k gather for decode = zero host syncs; d0 CPU-sort merge for
  prefill with one sync) — 2.2 -> 8.7 tok/s at 22-token ctx and ~20-50x at
  real conversation lengths (HANDOFF "v0.2 MAKE CHAT FAST").  Credible
  untried branch for the 50-100 tok/s bar: a hand-written grid-stride HIP
  GEMV (~400 GB/s is normal on HBM) behind the same F.linear contract,
  optionally + fused qkv GEMM (3 -> 1 launch per layer, ~17%).

## 2026-10-14 — fla-kda-gfx906 — fla's KDA/MLA kernels are Triton-only and DO NOT COMPILE OR RUN on gfx906 (or CPU) — vk-base cannot execute on the MI50 box unpatched
- Evidence: MI50 box, every shape B*T in {128,192,256,320,384,512} AND T=1: `RuntimeError: PassManager::run failed` (triton AMD backend make_llir) — shape grid `box:/data/tliao/vk_final/shape.json` all-fail; on CPU the same forward dies `ValueError: Pointer argument ... (cpu tensor?)`. Layer-by-layer the Triton-only surface is: fla `ShortConvolution` (causal conv), `chunk_kda`/`fused_recurrent_kda`, `FusedRMSNormGated`, fla `RMSNorm`, fla `RotaryEmbedding` (MLA), i.e. every fused helper the KDA+MLA stack touches.
- Root cause: flash-linear-attention 0.6.0 ships fused kernels for CUDA/Triton only; the torch reference math exists (`fla/ops/kda/naive.py`) but is unwired, and there is no `backend="torch"` on `causal_conv1d` (only cuda/triton/mix). Triton's AMD backend cannot build these kernels for gfx906 at all.
- What it rules out: running vk-base (or any fla KDA/MLA model) on the MI50 fleet with stock fla — the spine is portable to Triton-capable venues (MI300X/gfx90a: the vk-base SFT/pretrain venue) only.
- Next tried: `lab/vk_fla_compat.py` — pure-torch monkey-patch of those five op surfaces (reproducing the kernel math: `g = -exp(A_log)*softplus(g+dt_bias)`, beta sigmoid, half-split RoPE `o0=x0*cos-x1*sin`, depthwise causal conv with (N,D,W) cache, rmsnorm*gate). CPU forward now OK and the graft parity gate runs PASS (cos 1.0000). Shim numerics (measured): `naive_chunk_kda` BACKWARD is broken (grad NaN/inf on synthetic inputs) so the shim is RECURRENT-ONLY; the KDA state math must stay fp32 and OUTSIDE autocast (fp16 exp/cumsum NaNs wrap fits at step 0). Caveat: KDA state-update ordering (`state_v_first`) follows `naive.py`; not cross-checked against the triton kernels on a Triton box — first MI300X rerun should diff logits vs stock fla (expect ~1e-3 fp noise) before trusting fine quality claims.

## Fallback tree — self-learning / modular architecture line

If a rung fails, document, then take the NEXT untried branch — cheapest first.
Do not retry a falsified method unchanged.

**Passport routing (domain experts)**
1. PassportRouter + D55 sequential insertion (current; validated 11M/120M)
2. If purity weakens at scale: cosine scoring + learned temperature; prototype
   (Arrow-style) passport init; per-tier owner_mass/reject_w schedule
3. If passports fail outright: BTX-style shared-lineage branching only
   (sacrifice third-party plug-ins), or PEER-style product-key routing with
   tiny experts (sacrifice expert size)
4. If MoE plug-in fails entirely: LoRA adapter library + retrieval router
   (weakest modularity, but proven to work everywhere)

**Episodic memory as experts (MODULAR_MOE.md §8, gates G1-G4)**
1. Contract-expert consolidation via D55 (current design)
2. If G1 (retention parity) fails: keep direct-edit Hippocampus for fast
   weights, use experts only for long-term consolidation (hybrid cadence)
   -> 2026-10-09: activated after G1 (transplant) and CLOSED by G1b — on a
   passport-native spine with mex recal + base-neutral experts all four G1
   criteria pass (g1b_passport_d01_neutral3). Hybrid cadence stays as the
   operating posture until G3, not as a forced fallback. Sidecar kNN (rung 3)
   stays DISARMED.
3. If G2 (zero-shot reachability) fails: accept router recalibration per
   insert (slower loop, design survives); or sidecar kNN-retrieval over
   episode embeddings feeding context (no weight change at all)
   -> 2026-10-09: G2's no-recal premise IS falsified (best prototype arm
   0.479/0.276); first branch taken (recal per insert, measured cost ~800
   router-only steps). kNN sidecar not armed.
4. If G3 (N=16 coexistence) fails: cap the live library, archive stale
   experts to NVMe offload with prototype-index reinsertion; or hierarchical
   passports (domain passport -> memory passport, two-stage routing)
   -> 2026-10-09: G3 FAILED (util_home 0.08-0.39 for 15/16 experts at N=16,
   base CE +2.9%, break at N=4 util / N=2 CE; see the g3_coexistence entry).
   HIERARCHICAL DOMAIN->MEMORY PASSPORTS was armed as the leading candidate
   (two-stage routing: a domain passport partitions the row space so 16
   memories never share one top-2 contest) — see the resolution note at the
   bottom of this rung: NOT TRIGGERED after the pivot runs, kept as RESERVE.
   Library-cap + NVMe archive stays the alternate branch.
   -> recipe-control evidence (same day): N=8 at FULL G1b recipe still fails
   (util 0.93 -> 0.16, CE +2.28%) so the decay is structural, not a cadence
   artifact, and bank crowding is measured: plug-in rows sit at |cos| 0.275
   to each other vs 0.066 to base rows. **Capacity/orthogonality candidate
   rungs** (try before or alongside the hierarchy): (a) orthogonal row inits
   instead of mean-query prototypes (QR/Gram-Schmidt over home means),
   (b) larger passport_dim (64 -> 128/256) so rows have room to separate,
   (c) per-domain sub-banks (the hierarchical rung's degenerate form),
   (d) cosine scoring + learned temperature (domain-expert rung 2) to stop
   norm-driven row capture.
   -> margin re-score (same day): taught-fact retention at N=8 is 0.39 of the
   solo delivery (bar 0.70) — margins collapse with the library, so the
   pivot is confirmed on PRODUCT grounds (retention), not just routing
   stats. Also: util_home mispredicts both ways (0.567-util expert kept 0.39
   margin; 0.385-util expert kept 0.77) — the G3 bar gets revised to a
   margin-retention bar (3 forwards/expert) in any case. Open question:
   ep0 math-style fact survived intact (retention ~1.0) while tag-append
   facts collapsed — one control with 8 math-style episodes before final
   blame on the bank.
   -> addressing-pivot runs (same day, cheapest-first — RESOLVED):
   RUN B shape control PASSES (8 token-substitution episodes, retention
   **0.884** at UNCHANGED crowding |cos| 0.258 and still-low util) — the open
   question above is answered: **memory SHAPE is the lever, not the bank**;
   ep0's survival was shape, not position (Pearson(pos, ret) = −0.046).
   RUN A capacity rung FAILS: forced-orthogonal rows (|cos| 0.275 → 0.014)
   do NOT fix retention (0.452 ≈ noise) — rung (a) orthogonal inits CLOSED,
   (b) larger passport_dim DEMOTED (8 rows already fit orthogonally in 64
   dims; the constraint was training pressure, not room).
   **HIERARCHICAL DOMAIN->MEMORY PASSPORTS: NOT TRIGGERED** (the
   pre-registered rule was "both pivot runs fail → build it"; RUN B passed).
   The next build is CHEAPER: memory shaping — consolidate episodes as
   token-substitution-style deltas / shape-or-dedup episodes before
   consolidation (see the G1b/G3 reframe in MODULAR_MOE.md §8.4). The
   two-stage family→memory design is sketched in §8.4 as a RESERVE for the
   one sub-problem shaping does not touch: base-CE regression from GROUP
   top-2 occupancy of base tokens (8 rows × ~0.10 contam each = +2.5-4.1% CE
   even with clean per-expert contam) — and for N=16, if the shaped-episode
   retention advantage does not survive the bigger bank.
5. If G4 (live loop at t3) fails on hardware: consolidation on a second
   machine over LAN (the P40 box), serving never blocks

**Live self-learning (no train/inference boundary)**
1. Talk -> buffer -> idle-time expert consolidation (current)
2. If consolidation is too slow: online LoRA with EWC-style importance
   penalty (proven continual-learning method, less modular)
3. If weight updates are too fragile: pure retrieval memory (RAG over
   episode store) + in-context distillation prompts — no learning in weights,
   but unbreakable; use as the safety floor while weights-side methods mature
4. If router-based addressing fails: cross-attention sidecar memory network
   (memory tokens appended per layer, trained with spine frozen)

**Rules**
- One variable changes per experiment; confounded results get reruns, not
  stories.
- Scale ladder is mandatory: t0 falsifies before t1 spends, t1 before t2,
  t2 before droplet money. Never skip a tier because a result "looks obvious".
- Every fallback taken is recorded here with the evidence that forced it.
- The user is only interrupted for budget or architecture-direction
  decisions; everything else is autonomous.

## 2026-10-16 — glm-bake-triton-l2norm — GLM-Flash bake-off arm cannot decode on gfx906: fla l2norm Triton kernel (unsupported target) and a second PassManager crash in the KDA path — slice-smoke arm only

**Symptom.** `spine_bakeoff.py --spine glm-flash` builds the host fine
(`build_host` 6L, 11.8s) but dies at the first decode forward inside
`host_logits`. Two failures stacked:
1. `fla/modules/l2norm.py` Triton kernel — `unsupported target: 'gfx906'`
   at MLIR lowering. The transformers glm5_next text model calls this l2norm
   inside attention (qk-l2norm path).
2. With l2norm shimmed to pure torch, the same forward dies later with
   `PassManager::run failed` (triton AMD backend) deeper in the KDA call —
   agent-66's `install_torch_kda()` shim stack (which replaces the fla KDA
   kernels with torch bodies) is installed by `graft_glmflash.main()` but my
   `glm_flash_gen.make_generator` shim never invoked it before `build_host`.

**Root cause class.** Same as the 2026-10-14 fla-kda-gfx906 entry: fla
Triton kernels do not compile on gfx906. The GLM host needs BOTH the
`install_torch_kda()` shim AND an l2norm shim before any decode.

**Fallback taken (one).** `glm_flash_gen._patch_fl_l2norm()` — pure-torch
row L2 norm (fp32 accumulate), numerics matching the kernel. Partial: the
l2norm crash is gone but the second (KDA PassManager) crash remains because
the shim entry point was not wired in. Not looped — reported as
GLM-arm-blocked and the slice runs labeled plumbing smoke.

**Follow-up (small).** Wire `gl.install_torch_kda()` into
`glm_flash_gen.make_generator` before `build_host`, then the GLM arm should
decode on the truncated host. Full-host feasibility is a separate question
(see handoff: NOT — weight bytes exceed usable VRAM).
