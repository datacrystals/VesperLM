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
