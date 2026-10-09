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
3. If G2 (zero-shot reachability) fails: accept router recalibration per
   insert (slower loop, design survives); or sidecar kNN-retrieval over
   episode embeddings feeding context (no weight change at all)
4. If G3 (N=16 coexistence) fails: cap the live library, archive stale
   experts to NVMe offload with prototype-index reinsertion; or hierarchical
   passports (domain passport -> memory passport, two-stage routing)
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
