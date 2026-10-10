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
