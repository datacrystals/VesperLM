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
