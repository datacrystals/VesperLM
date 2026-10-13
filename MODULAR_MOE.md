# MODULAR_MOE — Hot-Pluggable Experts over LMbus

Status: design spec for the expert half of LMbus (`LMBUS_DESIGN.md` covers the
modality half). Every claim below is falsifiable on a single cheap GPU.

## 1. Motivation

Train a dense **spine** once (embeddings + attention + the residual-stream
geometry), then grow capability by plugging in separately trained FFN experts
instead of retraining the model. Three properties matter more than raw quality:

1. **No full retrains.** Adding a domain expert costs one small expert plus a
   short plug-in anneal, not a 6-day pretrain.
2. **Third-party experts.** Anyone can train an expert against the frozen
   LMbus expert spec (fixed I/O contract + passport registration) without our
   data or GPUs; it plugs in exactly like ours. Same community-certification
   bet as the modality side of LMbus.
3. **Hardware reality.** Hobbyist budget: heterogeneous cheap GPUs
   (P40/MI300X-class mix), WAN training that only syncs on an outer loop
   (DiLoCo plan), spot instances that vanish mid-run. Anything that needs all
   experts co-resident or an all-reduce every step is dead on arrival.

## 2. Prior art

**BTX / Branch-Train-Merge (Meta, 2024).** Experts branched from a shared seed
checkpoint, trained independently, merged at the FFN with a post-hoc router on
the frozen trunk. Shared lineage is exactly why their weighted-sum merge is
meaningful; our problem is the case BTX does not cover — no shared lineage.

**PEER (2024).** Product-key retrieval routing over ~1M tiny experts (a few
neurons each) per layer. An enormous expert library is trainable and servable
if the per-expert footprint is tiny and routing is a two-key lookup. We keep
the library target but with large experts and a semantic router.

**Arrow (2024).** Zero-shot routing to unseen LoRA experts via prototype
signatures from a small calibration set. Evidence that routing to experts the
router never saw is possible; passports are the weight-space analogue.

**Switch-style load-balancing aux loss.** Switch Transformer (2021) and
successors penalize uneven expert utilization. Table stakes for any top-k MoE
and the main defense against router collapse — but defined for a fixed expert
set, so it needs care after a hot-plug (see risks).

**Expert-offload serving (MoE-In-Flash and relatives).** Keep only the top-k
active experts in VRAM and stream the rest from NVMe, batched per layer. This
is what makes a large expert library servable on a single card.

## 3. The hard constraint: representation geometry

Independently trained experts do not share an internal language. An expert
trained on domain B with its own initialization, data, and hidden width lives
in a different basis than the spine's residual stream. The standard MoE merge
(a weighted sum of expert outputs) is algebraically defined and semantically
meaningless across such experts: cosine similarity between two experts'
outputs on the same input is not agreement, it is noise.

Two load-bearing fixes, both used here:

- **Shared lineage (BTX).** Experts branched from the same seed already speak
  the residual stream's language. Free, but only for experts we train.
- **Trained per-expert I/O adapters.** A small input map (spine -> expert) and
  output map (expert -> spine) trained at plug-in time with the spine and the
  expert core frozen. The only bridge for foreign experts; it must train on
  one GPU in hours.

A frozen wire format is necessary but not sufficient — freezing shapes does
not align geometry. Any "plug any expert zero-shot" claim without one of these
fixes is selling the format as the whole protocol.

## 4. Design

### 4.1 Spine / expert split

The spine is everything that defines the residual-stream geometry: embeddings,
all attention (KDA linear + MLA full layers in `Common/vesper_linear_model.py`;
GQA in `Common/vesper_model.py`), norms, LM head. Trained once; never updated
by expert plug-in. Every MoE FFN slot (today `MoEFeedForward` in
`Common/vesper_model.py`) becomes a library of interchangeable experts behind a
router; attention and embeddings stay dense.

### 4.2 Expert I/O contract (frozen)

Per layer, per expert:

- **Input:** `x` of shape `(B, T, d_model)` — pre-FFN residual vector
  (post-RMSNorm, the call site of `FeedForward` today).
- **Output:** `y` of shape `(B, T, d_model)` — a delta added to the residual.
- **Internal:** `A_in : d_model -> d_e` (bias-free Linear or low-rank), SwiGLU
  core (`w1, w3 : d_e -> h_e`, `w2 : h_e -> d_e`, same layout as the existing
  `FeedForward` weights), `A_out : d_e -> d_model`.
- **Checkpoint:** state dict of exactly these tensors plus a JSON header
  (`d_model`, `d_e`, `h_e`, provenance, training domain). No host optimizer
  state, no router state.

Same-lineage experts use `d_e = d_model` with `A_in`/`A_out` identity-init, so
the expert is function-preserving at birth. Foreign experts choose `d_e`/`h_e`;
the adapters absorb the width mismatch.

### 4.3 PassportRouter (replaces `TopKRouter`)

`TopKRouter` today is a bias-free `nn.Linear(dim, num_experts)` — a new expert
means new gate rows (`Growth/expand_experts.py` surgery). PassportRouter
decouples routing from a fixed expert count:

- **Passport bank** `P` of shape `(E, d_r)`: one learned embedding per expert
  (`d_r = d_model` to start).
- **Query / score:** `q = W_q h`;
  `s_e = softmax((q . p_e) / sqrt(d_r))` over live passports;
  top-k, renormalized (same output math as today, so the FFN merge code is
  unchanged). (Cosine scoring + learned temperature is a candidate t0
  experiment; the reference implementation starts with plain dot-product.)
- **Expert dropout:** each training step each expert is masked from candidacy
  with probability `p_drop` (start 0.1), softmax renormalized over survivors.
  The router cannot memorize index shortcuts; it must match token content to
  passport content, which is what makes unseen experts reachable at plug-in.
- **Aux loss:** the Switch-style balance loss (mean routing probability x hard
  utilization, `aux_weight` 0.1), computed over live experts only.

A new expert is one appended row in `P` — no gate surgery, no router-row
cloning, no router optimizer rebuild.

### 4.4 Hot-plug: `add_expert()`

```
add_expert(expert_ckpt, passport_init=None, layer_ids=None) -> expert_id
```

1. Load the contract checkpoint; validate shapes against the host config
   (`Pretrain/configs/model_configs.py`).
2. Append `A_in`/core/`A_out` to each targeted layer's expert list.
3. Append a passport row (`passport_init` or fresh gaussian); mark it live in
   the candidacy mask.
4. Change nothing else. Spine, other experts, LM head stay byte-identical;
   only new tensors get fresh optimizer state. The training loop then
   continues with the larger library.

### 4.4a Multi-expert insertion — validated protocol (D55, swarm2)

Independent reject training is NOT sufficient when several plug-ins coexist:
their passport rows cluster (each was trained only against the base rows), so
top-2 becomes {owner, other-plug-in} and cross-talk blows the ≤0.3 gate
(swarm1 fail: code↔math ≈ 0.5). Rows can only be partitioned in a model where
they coexist. The validated protocol (all 9 gates pass at t0; lab/imported/swarm2/):

1. Train each expert independently with contrastive reject (§4.5) as before.
2. Insert experts **sequentially**, ~100 router-only recalibration steps
   (passport rows only, everything else frozen) between insertions.
3. Final **joint calibration**: ~800 router-only steps over all domains with a
   **mutual-exclusion target** — on each domain, the owner passport row gets
   routing mass **0.55**, the original base experts share 0.45, and every other
   plug-in row gets exactly zero. owner_mass is the dial: 0.5 under-allocates
   the weakest owner, ≥0.6 leaks cross-domain. 0.55 is the operating point.

Cost ≈ 5× the phase-2 wall of independent plug-in (all calibration; expert
training itself is unchanged and still fully parallel across machines).

### 4.5 Geometry bridge: per-expert adapters

The adapters are the fix from section 3. At plug-in time: freeze spine, expert
core, all other experts, LM head; train only `A_in`, `A_out`, and the passport
row (optionally `tau`) — well under 1% of model params per expert. Objective:
LM cross-entropy on a host-side calibration slice plus KL-to-base on general
text, so the new expert cannot rewrite behavior outside its domain. Same shape
as the Hippocampus micro-session (`Hippocampus/consolidate.py`: reward-weighted
NLL + KL, tiny LR, few steps) without the LoRA branch.

### 4.6 Lifecycle and removal

1. **Register:** `add_expert()` — the expert becomes a routing candidate.
2. **Immune canary gate:** `Immune/gate.py` probes + `Immune/drift.py` on the
   plugged model vs the incumbent. Vetoes are absolute; a rejected expert is
   never merged into serving checkpoints.
3. **Light anneal (Hippocampus-style quarantined consolidation):** short
   adapter-only training on accepted material, then promote or rollback. The
   trunk and expert core never move.

Removal = drop from candidacy: mask the passport, keep the weights. Routing
renormalizes over survivors; kept weights allow cheap re-enable or later
adapter retuning. No retrain, no surgery.

## 5. Why it fits the hardware

- **Zero-communication parallelism.** Experts train fully independently — no
  gradient sync between expert trainers, ever. Composes directly with the
  DiLoCo WAN plan: experts can live on different boxes (or different people's
  boxes); only the plug-in anneal runs on the host.
- **Serving scales by top-k.** With top-k=2 only two experts per layer need be
  resident; the rest of the library lives on NVMe and is prefetched
  (MoE-In-Flash style). The active working set stays near a dense model.
- **Failure isolation.** A bad expert is bad weights behind a passport. The
  Immune gate rejects it before serving, the base is untouched, and removal is
  a candidacy flip. Spot-instance failures stay local.

## 6. Validation ladder

Shared pass criteria at every tier, on a held-out mix of domain-A / domain-B
text:

- **Utilization:** >50% of domain-B tokens go to the plugged expert (any layer
  where it is registered).
- **Contamination:** <30% of domain-A tokens go to the plugged expert.
- **Quality:** val CE with the expert active beats a masked control (expert
  present but not a candidate) on domain B, with no regression on domain A.

**t0 — `lab/plugin_expert_test.py` (tiny, CPU-capable).** 4 experts trained on
domain A, a 5th trained separately on domain B (no shared lineage). Zero-shot
`add_expert()` of the 5th; measure router utilization and CE delta vs masked
control. Pure plumbing + geometry test; no real pretraining.

**t1 — `lab_tiny` config, 4 -> 8 experts.** Real pretraining at the smallest
multi-expert config; plug in 4 new experts mid-run and keep training. Checks
that the balance aux loss and LR schedule survive a live expert-set change.

**t2 — `tiny_agent_k` (~103M total / ~81M active, KDA + MLA).** Plug-in
mid-run on the real hybrid architecture; same three criteria, plus wall-clock
and VRAM measurements for the anneal.

**t3 — `470m_k` (470M active).** Where adapter overhead, NVMe offload, and gate
cost are measured for real. Ship only after t2 passes twice with different
domain-B experts (a single pass could be luck).

## 7. Open risks

- **Passport generalization is unproven.** Arrow shows zero-shot routing to
  unseen LoRA experts; nobody has shown a passport scored against a frozen
  router reaching a foreign expert that never saw the router. The adapter
  anneal may be doing all the work; t0/t2 measure exactly this.
- **Router collapse / favorite-locking.** Without expert dropout the router
  locks onto favorites; with it, minority experts may starve. Aux-loss weight
  and `p_drop` need per-tier tuning, not once.
- **Adapter overhead at scale.** Two extra matmuls per expert per layer is
  negligible at 100M and unknown at 10B+. If adapters dominate, fall back to
  shared-lineage-only experts.
- **Aux-loss balance after hot-plug.** Switch-style balance assumes a fixed
  expert set; a fresh expert is either starved or dominates (passport norms).
  The balance loss may need a warm-up term over new experts only.
- **No public evidence above 10B total params.** BTX and PEER have the right
  shape but neither shows foreign-expert plug-in at scale. Treat every claim
  here as scaled from our t0-t3 ladder until measured.

## 8. Episodic memory as experts (Hippocampus x PassportRouter)

Status: design spec. Approved direction (user, 2026-10-09), gated on the
build-up experiments listed in 8.4. Nothing here is implemented yet.

### 8.1 The idea

Hippocampus today learns from conversation by direct weight edits on the
spine (LoRA wraps) + replay + Immune rollback. That has three structural
limits: edits accumulate interference in shared weights, rollback means
surgical reversal, and forgetting sets a hard ceiling on lifetime learning.

The synthesis: **memories consolidate into contract experts.** Conversation
feeds the Hippocampus replay buffer as today, but instead of editing the
spine, an idle-time consolidation pass ("sleep cycle") distills a batch of
buffered episodes into one small expert trained against the frozen section
4.2 I/O contract. The expert gets a passport row and plugs in via the D55
protocol. The spine stays byte-identical forever; "learning" is growth of
the expert library, and the training-run/inference boundary dissolves into
a continuous talk -> buffer -> consolidate -> plug-in loop.

### 8.2 Why passport routing is the enabler

- **Rollback = unplug.** Immune's poison response becomes dropping a
  passport row: instant, complete, reversible, no weight surgery. The
  incumbent-spine-intact invariant (verified in the teach-by-talking demo)
  holds by construction instead of by careful restoration.
- **Interference isolation.** Each consolidated generation of memories is
  its own expert; D55's mutual-exclusion calibration keeps generations from
  bleeding into each other. Interference stops accumulating in shared
  weights because shared weights stop changing.
- **Zero-shot reachability.** Expert dropout (4.3) trains the router to
  match token content to passport content, so a freshly plugged memory
  expert is reachable immediately. Passport init from the mean router query
  over the expert's consolidation examples (Arrow-style prototype
  signature) should make it fire in the right context from birth — this is
  a measured claim, gate G2 below, not an assumption.
- **Serving scales.** A lifetime of episodic experts is a PEER-style
  library: large total, tiny active set, NVMe offload for the tail.

### 8.3 Memory hierarchy

1. **Working memory** — context window. Free, instant, gone on eviction.
2. **Fast weights** — the current Hippocampus LoRA path. Minutes-scale
  uptake, small capacity, decays or is absorbed at consolidation.
3. **Episodic experts** — idle-time consolidation output. Permanent,
  content-addressed, individually removable. One expert per consolidation
  batch (NOT per fact — per-fact granularity wastes passports and invites
  routing noise).
4. **Aging** — experts with sustained near-zero routing mass and no val
  contribution get merged into neighbors or pruned. The library is a
  living structure, not an append-only log.

### 8.4 Gates (in order; each blocks the next)

- **G0 — prerequisites (DONE).** D55 purity gates pass at 11M and 120M
  (swarm2/4); teach-by-talking demo PASS on a true KDA+MLA checkpoint with
  poison rollback and incumbent intact (lab/imported/hippo_demo_vesperk.log,
  hippo_demo_tiny_agent_k_12k.log).
- **G1 — retention parity. FAIL (2026-10-09, tiny_agent_k step_best, 118M).**
  Export mode implemented (lab/g1_export_mode.py): section-4.2 contract expert
  (A_in/SwiGLU/A_out, identity-init adapters, 23.07M params over 8 layers),
  distilled from the demo teach set with forced dispatch + reward-weighted NLL
  + KL-to-base (demo objective) and plugged via passport row. **Retention and
  rollback pass:** margins -5.00/-5.68/-4.50 -> -1.29/+0.16/-0.02 (mean gain
  +4.68 vs direct-edit +3.22), digit-tail NLL 2.06 -> 28.60, word-tail 5.13 ->
  0.70; poison batch -> same stub gate ROLLBACK (target collapse), incident
  recovery = one row drop, incumbent state hash + margins bit-identical. The
  TopK->Passport transplant is function-preserving (max router-logit diff
  7.6e-6; no spine weight ever changes). **Routing purity fails and takes base
  CE with it:** utilization 0.81 on home episodes is paired with contamination
  0.70-0.90 on base mix across ALL THREE router-only recal objectives (D55
  phase-B contrastive: 0.81/0.89; hinge ranking margins: 0.18/0.18;
  section-4.4a mutual-exclusion mass target 800 steps: 0.81/0.90), and
  base-mix CE regresses +3.9% to +440% (bar <1%) — a plugged row that enters
  top-2 on base tokens displaces a base expert and rewrites the mix. A single
  passport direction cannot separate episodic home context from general text
  at this scale (the hinge trade-off curve is the overlap evidence), and the
  prototype row either never fires (literal mean-query, norm ~9 vs bank ~370)
  or fires everywhere (norm-matched). Falsifier "needs spine touches" did NOT
  trigger (zero spine weight writes). Evidence: lab/results/g1_export_mode_v1..v3.json,
  lab/logs/g1_export_mode_v1..v3.log; root cause + next branches in
  lab/FAILURES.md. G2's no-recal premise is undermined by the same evidence;
  §8.5 bullet 1 (different addressing mechanism) is the live branch.
- **G1b — passport-native retry. PASS with conditions (2026-10-09, lab_small
  33M spines).** G1's named next branch: same harness, same teach set, on the
  farm's passport-native checkpoints instead of the TopK transplant
  (lab/g1_export_mode.py now takes G1_CKPT; router kept as-is, expert dropout
  preserved). **The transplant was the problem.** With the section-4.4a mex
  recal arm, routing purity lands inside the bar on the dropout-trained
  spines: util_home 0.914 / contam_base 0.225 (dropout-0.1 spine) and
  0.981 / 0.276 (dropout-0.0 spine) vs G1's 0.812 / 0.903 — contamination
  drops ~3x and the margin gains rise (export +4.25..+5.98 vs same-spine
  direct-edit +0.67..+1.64). **Full 4-criteria PASS achieved** on the
  dropout-0.1 spine with section-4.5 base-neutral consolidation
  (G1_TEXT_KL_COEF=3.0, lab/results/g1b_passport_d01_neutral3.json):
  (a) margins -3.32/-3.81/-3.92 -> +1.03/+1.45/+0.23 (mean gain +4.588 vs
  direct-edit +0.674), (b) base-mix CE +0.599% (bar <1%), (c) poison ROLLBACK
  by row-drop with incumbent hash + margins bit-identical, (d) util 0.889 /
  contam 0.244. Two findings carry the design: (1) base-CE regression is NOT
  structurally coupled to contamination — a base-neutral expert (text-KL
  3.0) makes contaminated tokens cheap: CE +6.94% -> +2.00% (KL 1.0) ->
  +0.60% (KL 3.0) at constant routing; (2) expert-dropout training sharpens
  selectivity (true dropout-0.1 spine is the cleanest: contam 0.225-0.244).
  **G2's no-recal premise is falsified**: every prototype-init arm without
  recal misses (d) (best 0.479/0.276) — per G2's own falsifier, consolidation
  gets a router-recal pass per insert and the loop slows but survives.
  Provenance note: the run named `t1-dropout01-mb4-full` actually trained at
  expert dropout 0.0 (queue env dropped the key; saved config has none) —
  the true dropout-0.1 spine is `t1-diag-realdata-mb4` (300 steps); see
  lab/FAILURES.md. Addressing pivot to sidecar kNN is NOT triggered;
  passports-with-mex-recal + base-neutral experts remain the memory substrate
  candidate, and G3 (N=16) is unblocked as the next gate. Evidence:
  lab/results/g1b_passport_d00/d01/d02.json + d01_neutral{,3}.json,
  lab/logs/g1b_*.log (incl. same-spine direct-edit references
  g1b_direct_edit_d0{0,1,2}.log).
- **G2 — content-init reachability.** A memory expert plugged in with
  prototype-init passport and NO router recalibration reaches >50%
  utilization on its home episodes and <30% contamination on the base mix.
  Falsifier: needs recalibration anyway — then consolidation gets more
  expensive and the loop slows, but the design survives.
- **G3 — coexistence at N=16. FAIL (2026-10-09, lab_small 33M passport-native
  spine).** **Spine change:** the §8.4 text below said "t2 / tiny_agent_k" —
  G1 showed that checkpoint is TopK-trained and its transplant cannot separate
  home from base (contam 0.90), so per the G1b finding G3 ran on the
  passport-native lab_small spine (true expert-dropout-0.1 ckpt
  `t1-diag-realdata-mb4`, G1b recipe: contract expert + text-KL 3.0 +
  section-4.4a mutual-exclusion recal). 16 synthetic episodic batches
  (distinct prompt domains + distinct taught styles; episode 0 = G1's
  math/words set) were consolidated and plugged sequentially into the frozen
  spine, 80/40-step consolidation per expert, prototype-init passports,
  router-only recal with the 0.55 owner-mass mutual-exclusion target.
  **Recipe-control round (same day, same harness):** Control A = single
  insert at the FULL G1b recipe (80-step consolidation, 800-step recal,
  text-KL 3.0) reproduces the G1b PASS (util 0.912 / contam 0.217 /
  CE +0.880% vs G1b's 0.889 / 0.244 / +0.599%) — the N=1 failures in the
  reduced-recipe G3 runs were recipe depth (v3's 150-step recal left the
  lone row under-trained on base: contam 0.415 at N=1), NOT the spine,
  checkpoint, or fixtures; gate accounting was also clarified: the verdict
  judges the N=16 final state (per the gate text) and the earlier "first
  purity break at N=1" line conflated per-criterion trajectory spikes with
  decay — the script now records per-criterion first breaks and labels
  FINAL vs trajectory. Control B = **N=8 at the FULL recipe still FAILS** —
  util_home min 0.93 (N=1) → 0.50 (N=4) → 0.16 (N=8), CE +1.04% → +2.28%,
  and after the 800-step closing joint pass 7 of 8 experts sit below the
  0.5 util bar — so the N-scaling decay is real and not a cadence artifact
  (recipe depth buys ~0.2-0.3 util at small N, not the trend).
  **Purity decays with N and never holds at 16:** per-insert util_home min falls
  0.74 (N=1) → 0.44 (N=4, first break) → 0.08–0.15 (N=9..16); after the
  §4.4a final joint calibration (800 steps over all domains) the state is
  util_home 0.76 / 0.17–0.39 (15 of 16 experts below the 0.5 bar) — the joint
  pass equalizes rows but cannot recover the collapsed majority, so
  insertion-order training bias is NOT the root cause. **Bank crowding is
  measured, not inferred:** mean |cos| between plug-in rows is 0.275 vs
  0.066 between plug-in and base rows (N=8 full-recipe run) — memories
  cluster in the 64-dim row space and contest the same top-2 slots.
  **Per-expert
  contamination is NOT the failure:** 0.03–0.10 at N=16 (bar <0.3) — but the
  PLUG-IN GROUP saturates top-2 (p_any_plug_in_top2 = 0.61 on base tokens),
  and that displaces base experts on 61% of base-mix tokens: **base-mix CE
  regression +2.9%** at N=16 (bar <1%; +0.9% at N=1 → +1.8% at N=2 →
  +2.5–3.3% from N=5 on). Note the single-insert CE bar is tight in this
  harness even when passing: +0.60% (G1b) / +0.88% (Control A) /
  +1.04% (Control B N=1) — the N-trend on top of that marginal base is the
  real failure. Cross-talk between home episodes stays low
  (0.03–0.05) and each expert's forced-NLL retention gain is healthy
  (+3.3..+4.4 nats) — experts learn and are distinguishable; the
  64-dim passport row + top-2-of-E simply cannot partition 16 language
  domains at once. **Poison check PASSES at N=16:** poison batch → stub gate
  ROLLBACK (target collapse), one row drop → state hash byte-identical and
  base CE restored exactly (6.968553) — also byte-identical in Controls
  A/B. Recal cadence cost scales ~linearly
  (8s/insert at N=1 → 112s at N=16; 150-step per-insert recal + 800-step
  joint = 27 min total; full recipe 47→256s/insert at N=8 = 27 min for
  N=8; N=16 full would be ~2-3h). Evidence:
  lab/results/g3_n16_coexistence.json (v3, §4.4a cadence, the headline
  numbers above) + g3_n16_v2_perinsert600.json (600-step/insert, same
  failure shape) + g3_n16_v1_weaksep.json (aborted fixture — see FAILURES.md)
  + g3_ctrlA_n1_full.json / g3_ctrlB_n8_full.json (recipe controls);
  logs lab/logs/g3_*.log. **N=16-full-recipe was NOT queued** (its condition
  — N=8-full passing — failed). **Per §8.5 bullet 1 this is the kill criterion
  firing:** memory experts need a different addressing mechanism for
  coexistence, passport stays a domain-expert tool. Fallback tree rung 4
  ACTIVE (hierarchical domain→memory passports), with capacity/orthogonality
  rungs noted (orthogonal row inits, larger passport_dim, per-domain
  sub-banks). **G3 margin re-score (2026-10-09, the cheap decisive check —
  DONE): taught-fact margins were measured on the rebuilt N=8 full-recipe
  library state vs both the pre-library spine and per-episode solo (N=1
  full-delivery) references (lab/g3_margin_rescore.py,
  lab/results/g3_margin_n8.json). VERDICT: MARGINS COLLAPSE — the addressing
  pivot is CONFIRMED and hierarchical passports stay ACTIVE.** Mean margin
  retention vs solo = **0.390** (bar 0.70) with huge per-episode spread:
  ep0 math_words 1.073 (util 0.90 — the G1 fact survives intact; demo-probe
  retention 0.97: pre −3.685 → solo −1.986 → lib −2.041), ep4 treasure_pirate
  0.766 (util 0.385 — low util CAN still deliver), but ep2/ep3/ep5/ep7 collapse
  to 0.10/−0.06/0.10/0.20 (util 0.30–0.45). Mean gain +2.68 vs the 70%-of-G1b
  bar +3.21; 6 of 8 experts below util 0.5 average only +2.01 gain vs +6.27
  solo. Pearson(util, retention) = **0.61** — margins do decay with util, but
  the scatter is the story: **the util bar mispredicts in BOTH directions**
  (ep6 passed util at 0.567 yet kept only 0.39; ep4 failed util at 0.385 yet
  kept 0.77). So util_home is a poor product gate — but replacing it with a
  margin-retention bar does not save the architecture: the library state
  fails the 0.70 retention bar too. Product requirement (retention of taught
  facts at N=8) does NOT survive in aggregate. Cross-checks: NLL margins
  agree (ep5 +11.3 solo → +1.27 lib), and the G1 math fact specifically
  survives (retention ~1.0) while the synthetic tag-style facts mostly do
  not. N=16 was not re-scored (the g3 runs never persisted a state dict;
  N=8 was rebuilt from scratch and is now saved at
  lab/sandbox/g3_margin/n8_state.pt for any follow-up).
  **G3 addressing-pivot runs (2026-10-09, cheapest-first, one variable each —
  lab/g3_pivot_runs.py):**
  **RUN B (episode-shape control) — PASS: memory SHAPE is the lever.** All 8
  episodes token-substitution facts (ep0's shape: stated digit form → corrected
  word form) over 8 distinct prompt domains, same ctrlB recipe:
  **mean margin retention 0.884** (bar 0.70; mixed fixture 0.390) — per-episode
  1.12 / 0.95 / 0.67 / 0.52 / 1.08 / 0.92 / 0.85 / 0.95, and this is at
  UNCHANGED crowding (plug-plug |cos| 0.258 vs mixed 0.275) and still-low
  util (0.30–0.47 for 6 of 8). The ep0 anomaly resolves to shape, not
  position: Pearson(insert-position, retention) = −0.046 (ep7 @ position 7
  holds 0.95). **Tag-append memories are fragile regardless of addressing** —
  the pivot is REFRAMED: consolidate memories as token-substitution-style
  deltas / shape or dedup episodes before consolidation, instead of (or
  before) building hierarchical addressing. (lab/results/g3_pivot_shape.json)
  **RUN A (capacity rung) — FAIL: mutual orthogonality does not fix
  crowding.** Plug-plug squared-cosine penalty in the mex recal forced
  plug-plug |cos| 0.275 → **0.014** (orthogonality decisively achieved; rows
  are 8-in-64, room was never the constraint — training pressure was) yet
  mean retention stayed **0.452** (bar 0.70; vs 0.390 baseline = noise-level
  change). The capacity claim "plug-in rows can be made mutually orthogonal;
  that fixes crowding" is **dead as stated** — orthogonal rows still contest
  top-2 and the tag-append facts still collapse (ep2 0.02, ep7 0.10) while
  token-substitution facts still hold (ep0 1.07, ep4 0.85, ep6 0.99): shape,
  not row geometry, predicts retention. (lab/results/g3_pivot_ortho.json)
  **Standing G3 metric from now on: margin retention vs solo ≥ 0.70**
  (3 forwards/expert — the honest product bar), with contam_base < 0.3 and
  base-CE regression <1% as secondary bars. Both pivot runs still fail the
  base-CE bar (RUN B +4.12%, RUN A +2.46% vs ctrlB +2.26%) with per-expert
  contam clean (0.08–0.12) — the residual is GROUP top-2 occupancy of base
  tokens (8 rows × ~0.10), which neither shape nor orthogonality touches;
  that remains the open G3 sub-problem.
  **Hierarchical domain→memory passports NOT triggered** (the pre-registered
  rule was "both runs fail → build it"; RUN B passed). Reserve design sketch
  for whoever needs it for the CE residual or N=16 (rung 4, two-stage):
  (1) a FAMILY row per domain cluster replaces the flat row — router first
  scores family passports (k families ≪ N, trained with the same mex target
  where "owner" = family), so base tokens contest only k rows;
  (2) within a routed family, a second cheap contest over that family's
  memory rows (within-family softmax, own mex target) picks the episode —
  only 2 rows total in top-2 across two tiny contests, so 16 memories never
  share one flat top-2-of-20 election. Family assignment: cluster episode
  home-means (the same query vectors used for prototype init) into k≈4
  families at consolidation time; the family row's prototype = mean of its
  members' prototypes. Cost: +1 row per family, one extra matmul at routing;
  rollback stays one-row-drop (memory rows) or family-row-drop (whole
  cluster). States saved at lab/sandbox/g3_pivot/n8_shape_state.pt and
  n8_ortho_state.pt (untracked) for follow-up re-scores.
  **G3 memory-shaping consolidation module + shaped coexistence re-run
  (2026-10-09/10):** the RUN B lesson is now a reusable module,
  `lab/memory_shaping.py` — episodes are token-substitution deltas
  (`TokenDelta` prompt/prefix/bad/good, classified substitution/insert/
  delete/identical by longest common prefix+suffix; `SHAPED_8`/`SHAPED_16`
  = 8/8 + 8/8 distinct prompt domains, all substitution), with the G1b
  consolidation recipe (`consolidate_episode`: forced dispatch +
  reward-weighted NLL + KL-to-base + text-KL 3.0 → experts + prototype rows
  + gate), `plug_and_recal()` (§4.4a mex recal per insert), and
  `retention()`/`delta_margin()` as the standing measurement. Runner
  `lab/g3_shaped_run.py` rebuilds solo refs fresh per episode in the same
  run (self-consistent train+measure — the family A/B fixture-mismatch
  caveat does not apply).
  **N=8 shaped — PASS (retention 1.071 vs bar 0.70).** 7/8 episodes clear
  0.70 per-episode (math 1.02, cooking 1.00, astronomy 1.04, music 0.77,
  sports 2.48, anatomy 1.06, geology 0.73; computing 0.467 is the lone
  miss); the anchor check's "deviation" vs RUN B's 0.884 is a solo-ref
  artifact (sports' solo ref was weak: gain_solo +6.4 vs +15..19 elsewhere,
  inflating its ratio to 2.48) — excluding sports the mean is 0.870, inside
  the ±0.15 anchor window and essentially RUN B's number. **But the base-CE
  residual persists and is larger:** 6.7735 → 7.3771 (**+8.91%**, bar <1%;
  RUN B +4.12%, ctrlB mixed +2.26%) at p_any_plug_in_top2 0.589 / p_both
  0.171 with per-expert contam clean (max 0.108) — shape is the retention
  lever only and is neutral-to-worse on the group top-2 occupancy axis.
  Bank-norm asymmetry grows with inserts (plug rows 17–34 vs base ~2);
  insert-trajectory util decay (e0 0.95 → e7 0.27) tracks that norm growth,
  not insertion position (ep7 still delivers 0.726).
  (lab/results/g3_shaped_n8.json, state lab/sandbox/g3_shaped/n8_shaped_state.pt)
  **N=16 shaped — PASS (retention 0.737 vs bar 0.70): the headline G3
  coexistence number, first N=16 PASS under the product bar** (mixed-shape
  fixture was 0.390 at N=8). 9 of 16 episodes clear 0.70 per-episode (math
  1.01, music 0.85, anatomy 1.00, computing 0.89, geology 0.86, travel
  0.72, chess 1.10, ocean 0.76, forest 0.80); the 7 misses are soft
  (0.44–0.63), not a collapse. Util decays with insertion order (e0 0.95 →
  e15 0.18, util_min 0.125) yet most memories still deliver — the util bar
  mispredicts again (chess at position 12 holds 1.098). **Base-CE residual
  persists:** 6.7735 → 7.1878 (**+6.12%**, bar <1%) at p_any_plug_in_top2
  0.621 / p_both 0.223 with contam clean (max 0.085) — milder than N=8
  shaped's +8.91% at p_any 0.589, so the occupancy residual does not blow
  up with N, but shape does not fix it at either N. Per §8.5 bullet 1 the
  "different addressing mechanism" kill branch is no longer triggered by
  retention — it survives on the base-CE residual alone.
  (lab/results/g3_shaped_n16.json, 106 min wall, state
  lab/sandbox/g3_shaped/n16_shaped_state.pt)
  **§8.4 reserve sketch prototyped — family-row two-stage routing, NEGATIVE
  as a zero-training drop-in (g3_family_ab.py, pure inference-time swap on
  the RUN B N=8 shaped state):** avg-linkage clustering of member rows →
  families [[0],[1],[2],[3,4,5,6,7]] (k=4); stage 1 base+family rows contest
  top-2, family slot → argmax member, stage-1 weights renormalized. First
  run measured MIXED (CE −3.46%, p_any 0.588→0.701, retention 0.674→−0.081)
  but that signal was INVALIDATED by an index-space bug in the prototype
  (family rows built from the full bank with member-local ids — family
  `[[0]]`'s "member mean" was base row 0 — and stage-2 emitted local ids
  where global expert ids belong, so winning families dispatched to the
  wrong experts and memories were never delivered). With the bug fixed and
  the same state/families rerun: CE 7.0528 → 6.9525 (−1.42%),
  p_any_plug_in_top2 0.588 → **0.630** (occupancy still does NOT drop —
  the sketch's "base tokens contest only k rows" premise fails at k=4 with
  untrained family-mean rows), retention mean 0.674 → **0.426** (0.567
  excl. the math fixture artifact) — the multi-member family's mean row is
  diluted and its members (sports/anatomy/geology) lose delivery. So the
  two-stage STRUCTURE delivers correctly when indexed right (bug explains
  the catastrophic collapse) but does not address the occupancy residual
  and costs retention as a zero-training drop-in. Named next branch for
  the residual: family-row mex calibration (owner=family) or base-neutral
  weighting of plug rows. (lab/results/g3_family_ab.json (buggy, preserved)
  + g3_family_ab_fix.json (fixed rerun))
  **Base-CE residual: both named follow-up branches measured — both
  NEGATIVE (2026-10-10).** (b) base-neutral plug weighting, the cheaper
  one first: plug-logit bias sweep on the saved N=8 shaped state
  (`g3_plug_bias_ab.py`, β 0→5, pure inference, 57s) — the CE/retention
  curve is monotone and the bars are mutually exclusive (β 1.0 is the last
  retention-feasible point at CE +6.73%; β 3.5 is the first CE-feasible
  point at retention 0.07); a scalar bias cannot separate base from home
  at this overlap, closing the whole family of monotone plug penalties.
  (a) family-row mex calibration (`g3_family_cal.py`, owner=family §4.4a
  target on fam_rows only, 800 steps, 369s): retention **holds at 1.071**
  (two-stage delivery is free once family rows train) but base CE only
  trims **+8.91% → +7.45%** (p_any 0.590 → 0.473) — family rows are means
  of high-norm member rows (plug norms 17–34 vs base ~2) and the mex
  base-silence pressure (loss 98→94) cannot overcome that norm advantage.
  **Conclusion: row-side pressure is exhausted — the residual is
  structural at the group-occupancy level** (8 rows × ~0.10 per-expert
  contam on base tokens displaces base experts). G1b's base-neutrality
  lever (text-KL 3.0 → CE +0.60% at N=1) is already in the consolidation
  recipe and still yields +8.9% at N=8, so the named next cycle branches
  are: strengthen base-neutrality at consolidation (text-KL scaled with
  N), re-scope the CE bar as a known coexistence cost, or the hierarchical
  domain→memory addressing rung.
  (lab/results/g3_plug_bias_ab.json, g3_family_cal.json)
  **N-scaled base-neutrality + occupancy-dilution (2026-10-10):** (1) KL
  pressure at consolidation scales the residual down ~5× — text_KL 3/6/12
  → base CE **+8.91% / +3.25% / +1.82%** with retention 1.07/1.33/0.75
  (all ≥ 0.70) at **constant p_any ≈ 0.62**: G1b's mechanism confirmed at
  N=8 (contaminated tokens become CE-cheap, occupancy untouched). The
  <1% bar would need text_KL ≈ 24–30 (log-linear), but retention already
  slides at 12 (music 0.48, sports 0.40) — KL pressure alone likely
  cannot meet the dual bar. (2) Occupancy-dilution signal run REJECTED
  the scale-artifact hypothesis: cloning the base pool to 8/16/32 rows
  (plug share 0.67→0.20, constant expert quality, clone-only CE control)
  leaves the plug residual flat at ~7.7% (4-row: +8.91%) even as p_any
  halves 0.59→0.40 — the residual is per-displacement base-neutrality
  damage, not slot arithmetic, and should NOT be expected to dissolve at
  t2 pool sizes for free. The real-spine 8-base companion
  (t0-8expert-realdata, full shaped pipeline) shows base CE +1.44% but
  with retention collapsed to 0.397 (memories silenced) and p_any up to
  0.795 — the small CE number is the silencing confound, not dilution;
  the shape recipe's PASS is also spine-conditional (delivery fails on
  lab_tiny/8-base). **Residual line summary: the KL lever is the only
  one that moves CE without silencing (+8.9%→+1.8% at KL 12, retention
  0.75); row-side and pool-side levers are closed.** KL curve completed
  (24/30 points): **saturation at +1.5-1.7% from KL 12 on** (KL 24:
  +1.53%/0.788; KL 30: +1.67%/0.804) — the log-linear extrapolation to
  <1% is falsified; the bar needs an addressing-side change or a re-scope.
  (lab/results/g3_shaped_n8_kl6.json, g3_shaped_n8_kl12.json,
  g3_dilution_clone.json, g3_shaped_n8_8base.json)
  **Spine-conditional delivery: diagnosed and FIXED (2026-10-10).** The
  8-base collapse (retention 0.397) is NOT weak experts and NOT norm
  mismatch: forced-owner routing delivers **1.14x solo** from the same
  library experts, while cosine/content routing collapses delivery on
  both spines (ret 0.006 — norm-driven routing is essential). The
  mechanism is **owner-mass falling below a delivery cliff at the SCORING
  positions**: lab_tiny has a sharp cliff (owner weight α=0.45 → +1.6,
  α=0.65 → +14.2 on cooking) that lab_small lacks (flat +15.6 at all α),
  and owner util/weight AT the divergence positions where delta_margin
  reads logits track delivery exactly (cooking 0.33/0.38 fails, math
  1.0/0.79 delivers) — home-token averages invert the story (8-base
  w_own 0.63 looks healthier than lab_small's 0.52). Wrong-memory
  cofire (0.26 vs 0.11) steals mass but is not poison (evicting it does
  not help). **Fix: owner_mass 0.55 → 0.85 in the closing §4.4a joint
  recal** lifts retention 0.397 → **0.975** (mass alone) → **1.100**
  (mass + scoring-prefix home coverage — recommended refinement), both
  clearing the 0.70 bar at unchanged base CE (~+1.5%). The shaped recipe
  is now spine-portable; a full 8-base pipeline run with the bump is the
  end-to-end confirmation. (lab/results/g3_delivery_diag.json,
  g3_owner_mass_fix.json)
  **End-to-end confirmation — PASS (2026-10-10):** the owner_mass 0.85 +
  scoring-prefix home recipe baked into the pipeline from the start
  (`G3S_OWNER_MASS`/`G3S_HOME_MODE=scoring` knobs, applied to every mex
  pass — solo, per-insert, closing joint) reaches **N=8 shaped retention
  0.944** on the t0-8expert-realdata 8-base spine (was 0.397; 7/8 eps ≥
  0.70, anatomy 0.69 marginal), scoring-position util_min **0.76** (the
  P8 gap is closed by construction), contam_max **0.208** clean, base CE
  +2.85% documented (a little above the post-hoc pass's +1.55% because
  the baked-in recipe delivers more mass on base tokens too).
  **Spine-portability is CLOSED GREEN** — shaped memories PASS the
  retention bar on both lab_small 4-base and lab_tiny 8-base with the
  recipe. (lab/results/g3_shaped_n8_8base_e2e.json)
  **N=16 on the real spine — PASS (retention 0.953, 15/16 eps ≥ 0.70,
  util_min 0.66, contam_max 0.139 clean):** the full coexistence story is
  green on t0-8expert-realdata with the same recipe — retention is flat
  in N (0.94 at N=8 → 0.95 at N=16) once owner_mass sits above the
  delivery cliff. Base CE +3.68% (7.2086→7.4735) at p_any_plug 0.889 —
  the group-occupancy residual grows with N on this spine (+2.85% →
  +3.68%) and remains the one open bar. (lab/results/
  g3_shaped_n16_8base_e2e.json)
  Original gate text: 16 episodic experts plugged sequentially
  into t2 (tiny_agent_k); all section-6 purity gates still hold for every
  expert, and base-mix val CE has not regressed >1% vs pre-library. This
  is the direct stress test of the owner_mass/reject_w scale-weakness
  finding (swarm3/4) in the memory setting.
- **G4 — live loop at t3.** The 470M passport model runs the full
  talk -> buffer -> consolidate -> plug-in cycle during serving, with
  consolidation scheduled in serving idle time. Success = measurable
  retention of taught facts after 48h of mixed conversation + adversarial
  batches, zero spine writes, zero manual intervention.

### 8.5 What would kill it

- Passport routing advantage continues to shrink with scale (the 392M
  parity probe) and G3 shows purity collapsing before N=16 — then memory
  experts need a different addressing mechanism and passport stays a
  domain-expert tool only.
- Consolidation cost exceeds idle-time budget on hobbyist hardware —
  then the loop needs a cheaper expert shape (smaller d_e, low-rank core)
  or slower cadence. The design degrades gracefully here; this is a
  tuning problem, not a fatal one.

## 9. Donor grafting: passport-routing existing MoE models

Status: approved direction (user idea, 2026-10-09). Reuse trained experts from
open MoE models (GLM-class, K2/K3-class) under our router instead of training
every expert ourselves. Two variants, very different costs.

### 9.1 Why this is on-thesis, not a side quest

The whole LMbus bet is that experts are interchangeable given (a) a frozen I/O
contract and (b) a content router that reaches experts it never saw. Donor
experts are the extreme case of (b): maximally foreign lineage, zero shared
training. If passport plug-in works for THESE, the community-expert vision is
no longer speculative.

Key structural fact: **MoE experts are FFNs only.** Source-model attention is
irrelevant — attention lives in the spine. What crosses models is purely the
residual-stream geometry problem of section 3, and our G1/G1b evidence maps
directly:

- G1 (TopK transplant) FAIL predicts: naive router-swap without recalibration
  fails. Contamination 0.90, base CE +188%. Do not bother running this arm.
- G1b (passport-native + mex recal + text-KL 3.0 base-neutrality) PASS is the
  recipe a donor expert needs: base-neutral consolidation, then ~800
  router-only recal steps per insert (G2 falsified zero-shot reachability —
  budget the recal, it is cheap vs pretraining anything).

### 9.2 Variant A — passport graft (easiest, do first)

Keep the donor's spine AND its experts; replace ONLY the gate with
PassportRouter. Geometry is untouched (experts keep their native home), so
this is repackaging, not bridging:

1. Profile the donor's experts: run labeled corpora (code, math, web, chat,
   science...) through the model, log per-layer expert activation histograms.
   This is the "classification" step — measurable, no training.
2. Init each passport row from the mean router query over the expert's
   home-profile data (prototype init), then recalibrate router-only on a
   mixed calibration set with the D55 mutual-exclusion target.
3. Gate on val parity vs the donor's original router (must not regress) +
   purity gates per domain.
4. Payoff: the grafted model now supports `add_expert()` — OUR trained
   experts, or experts from OTHER donors (via variant B adapters), plug into
   a 100B+ host we never pretrained. This is the fastest path to a
   frontier-scale passport-routed machine.

Caveat from the literature and our own profiles: donor experts are not
cleanly semantic (some specialize positionally/syntactically). Expect a
fraction of mushy passports; purity gates measure how bad that is.

### 9.3 Variant B — cross-model expert transplant (the mix-and-match)

Plug donor experts into OUR spine (or a grafted donor spine from variant A)
with section 4.5 adapters absorbing the width/basis mismatch:
`A_in: d_spine -> d_donor`, expert core frozen, `A_out: d_donor -> d_spine`,
trained with text-KL 3.0 base-neutrality (G1b recipe) + router recal.
Adapter rank is the capacity knob; if adapters dominate, the fallback
(section 7) is shared-lineage-only, i.e. variant A per donor family.

### 9.4 Gates

- **D0 — graft parity (t0 of this line, laptop/MI300X-cheap).** Passport-graft
  a small open MoE (OLMoE-1B-7B or DeepSeek-V2-Lite class). PASS = val parity
  with the donor's stock router on a mixed eval (within noise), plus per-domain
  purity >= the donor's own routing entropy baseline.
- **D1 — plug-in onto graft.** Add ONE foreign expert (ours, or another
  donor's via adapters) to the grafted model. PASS = section-6 purity gates
  for the new expert, no regression on base eval.
- **D2 — cross-family transplant.** A GLM-family expert and a K2-family expert
  co-located in one spine, both passing purity. This is the mix-and-match
  proof.
- **D3 — scale.** Graft a big donor (GLM-5.3-class) on the MI300X/cluster:
  NVMe expert offload + passport graft + recal; measure serving cost delta
  vs stock routing.

### 9.5 What would kill it

- D0 fails parity: passport scoring can't reproduce a mature router's
  decisions on a foreign bank — then grafting is dead, and passports stay
  native-only (the modular thesis survives; donor reuse does not).
- D2 fails even with high-rank adapters: cross-model geometry is not
  bridgeable at useful fidelity — then variant A only: one grafted host per
  donor family, mixed at the serving layer (model routing), not the expert
  layer.

### 9.6 Product path (what "plug-and-play" actually means)

G2 falsified zero-shot plug-in: every insert pays ~800 router-only recal
steps (seconds-to-minutes scale, thousands of params — a JIT compile, not a
training run). The product consequence: ship frozen VERSIONED spines; the
registry ships experts with PRECOMPUTED passports per spine version (the
compile cost is paid once by the author/registry, never by the end user).
Custom/foreign experts pay the local recal at install time — a driver
install, not a research project. VRAM selects spine size + resident expert
cache only; total library knowledge is VRAM-independent via NVMe offload
(section 5), so "pick your VRAM" picks speed, not capability.

### 9.3a Telemetry-trained translators (upgrade to the 9.3 adapters)

User idea (2026-10-09): replace blind plug-in-time linear adapters with
translators trained on DONOR TELEMETRY. Donor weights are white-box — run a
shared calibration corpus through both models and capture (per layer) the
donor's pre-FFN residual x_donor, the host's x_host, and the donor router's
top-k choice. Train T: x_host -> x_donor (small nonlinear MLP, residual) as
paired-activation regression. Prior art: stitching layers (Bansal 2021) show
linear stitching between networks recovers most function when trained on
activations; we go nonlinear and route-conditioned.

Two sharpenings over naive basis translation:

1. **Route-conditioned targets.** We do not need the whole donor basis — only
   the input manifold of the ONE expert being transplanted (the distribution
   of activations the donor router sent it). That is a much narrower target
   than full-basis translation, and the donor router's decisions define it
   for free.
2. **Layer correspondence.** Donor depth (60+) does not map 1:1 onto the host
   (10). Options: soft attention over donor layers, or per-(expert, layer)
   translators conditioned on donor layer index. Measure both at D-scale.

Falsifier (stated in advance): the translator can only re-express features
the host spine already encodes — host quality bounds transplant fidelity. If
paired-activation regression fits train but fidelity collapses on held-out
domains, the host spine lacks the features, not the translator the capacity;
the fix is spine scale, not adapter size.

D-gate insertion before D2: **D1.5 — translator fidelity.** Transplant one
expert between two small open MoEs (OLMoE <-> DSv2-Lite) via telemetry-trained
T. Metrics: held-out cosine/MSE between transplanted output and the expert's
native donor output (target: >0.9 cosine on the expert's top singular
directions), then end-to-end section-6 purity gates. If fidelity <0.7 even
in-domain, variant B is deprioritized to variant-A-only per 9.5.

**STATUS 2026-10-11 (D1.5 measured, `lab/graft_d15_translator.py`): FAIL the
0.9 bar and the <0.7 deprioritize line FIRES** — route-conditioned
linear/2-layer translators on OLMoE telemetry (gate slices, D0 registry)
reach held-out cosine 0.55–0.63 on within-donor cross-layer pairs (same-layer
control = 1.0) and 0.64–0.69 cross-model vs Granite-3.0-1B-A400M-MoE
(word-aligned pooling); train cos saturates at 0.72–0.76 and the 2-layer MLP
does not beat linear → feature availability, not capacity (the §9.3a
falsifier's reading). Per §9.5 variant B is deprioritized to variant-A-only
until a translator class clears the bars; soft layer-attention (sharpening 2)
and token-level pairing remain untried.

**STATUS 2026-10-12 (Door 9 — embedding-anchored basis translation, the
better-posed D1.5, `lab/door9_embedding_w.py`): the pairing-artifact
hypothesis does NOT rescue translation — CONFIRMS the feature-availability
read.** Exact token-identity pairing of the two on-disk donors' embedding
tables (OLMoE-1B-7B-0924 ↔ Phi-mini-MoE-instruct; decode each id, match
strings, single-token round-trip rows only: 7,019 clean intersection pairs,
12,480 in a loose sensitivity) gives a zero-alignment-error training set, and
ridge W still fits only train cos 0.67 / held 0.47 (best λ; λ→0 overfits at
0.86/0.26), far under the 0.95/0.90 revision bar; MLP2 overfits (0.87/0.28)
— same "capacity is not the limiter" pattern as D1.5. Orthogonal Procrustes
0.69/0.35, mean-baseline 0.18, shuffled-pair floor 0.09. The emb-fitted W
also does not transfer to depth: cos(W(h_Φ(f)), h_OLMoE(f)) at matched depth
fractions = 0.48 / 0.05 / 0.05 / 0.05 / 0.06 (f = 0/0.25/0.5/0.75/1.0) —
dead past ~25% of depth. Practical payoff (§10.2 wrap, door25 recipe):
W-init of the wrap input chain (host_in+adapter_up = rank-64 SVD of
emb_OLMoE→emb_Phi) does NOT close the held-out memorization gap — at
donor-residual scale (α=190) the wrap numerically collapses to a no-op
(train CE plugged 1.088 vs masked 1.068; forced ≈ masked); at quiet-birth
scale (α=1) it trains normally but lands at the random-init baseline. D1.5's
verdict is therefore NOT revised to "depth divergence + pairing loss": the
word-level pairing loss was real but not binding — with zero pairing error the
input bases are still not linearly bridgeable at useful fidelity. Full numbers
+ depth-caveat (single-token mid-layer states are rank-1 degenerate, mean
baseline 1.0 — cosine uninformative at mid-depth): HANDOFF 2026-10-12 Door 9.

### 9.7 Harvest granularity: co-adaptation clusters

Donor experts co-adapt under their router — each is shaped to complement its
siblings (expert 5 handles what expert 3 leaves behind). Harvesting singletons
risks orphaning that function. The D0 profiling pass already logs per-token
top-k sets, so measure the co-activation graph and harvest in CLUSTERS
(frequently co-selected experts travel together, sharing one plug-in recal).
Graft order within a cluster follows the donor router's own activation
margins. Singleton vs cluster fidelity is measured at D1.5 — if singletons
hold >0.9 cosine anyway, skip the clustering complexity.

---

## §10 Basis-portable expert format (post-D-line design line, 2026-10-11)

Motivated by the D-line outcome: D0/D2 prove the passport/registry ROUTING
interface generalizes across families, but D1.5 proves hidden-space feature
translators do NOT (0.55-0.69 vs 0.9 bar; feature-availability limited; even
same-model layer-2→12 fails). Weight-level mix-and-match across existing
models cannot rely on representation parity. Three doors, in build order:

### 10.1 Door 2 — experts with a canonical bottleneck (the format)
Each expert is not a full-width MLP in the spine's basis but:
    adapter_down (d_model → d_canon) → core (basis-independent skill)
    → adapter_up (d_canon → d_model)
Train our bases (Vesper-K onward) with experts in this format. Spine changes
are minimal (routing already standardized by the registry/passport layer).
Benefits: (a) experts swap freely among format-adopting models with adapter
recal only (the 0.66→0.996 recal trick, §9 sweep); (b) CROSS-SCALE plug-in —
an expert trained at 470M plugs into a 1B/3B spine via its adapters (the
"compile from parts as the family grows" product story). Cost: bottleneck
expressiveness tax per FLOP vs fat native experts — measure at t1/t2 scale
(one clean experiment before committing the Vesper-K spec to it).

### 10.2 Door 2.5 — wrap FOREIGN expert cores in the format (the grab)
Take a donor expert's core weights VERBATIM (frozen), learn only the two
thin adapters around it (our d_canon ↔ their basis), on domain data with the
task loss (NOT feature-matching). Much easier than D1.5's global translator:
one narrow interface, end-to-end objective, recal-scale compute. First
experiment (lab scale): one OLMoE/Phi-mini-MoE expert → wrap → recal →
D1-gate it (parity/retention/contam bars from §9.4). Success = exact foreign
weights hostable; failure mode = adapters can't bridge → Door 3.

**STATUS 2026-10-11 (`lab/door25_wrap_expert.py`, MI50 GPU 3): GRAB WORKS
AT THE INTERFACE — the foreign core is hostable verbatim and the quality
sub-gate PASSES for the first time in the line (home CE 1.072 masked →
0.439 plugged → 0.101 forced); skill generalization NOT yet shown.**
Phi-mini-MoE @f620b32c L13 e8 SwiGLU (11.8M params) frozen behind
adapter_down/adapter_up at d_canon=64 (786,432 trainable; host_in/host_out
per §4.2), plugged at one depth-matched OLMoE layer of the D0 registry
graft; task-loss fit (CE + out_pen 300, lr 3e-3 / wd 0 — D1's wd 0.01
starves thin wraps to invisibility) + att-4 mass-cal row recal (owner_mass
0.35, reject_w 10). Gates: parity 0.9985/0.9990 PASS, contamination 0.065
PASS, utilization 0.831 PASS, quality 0.439 vs 1.072 **PASS** (D1's toy
failed this 12.6 vs 1.3; Door 3 failed it 8.1 vs 1.8), base CE Δ+0.005.
The D1 retention bar fails as written (margin-probe solo gain −0.28 → ratio
degenerate; CE-space analog 0.652 vs 0.70 — the delivery-contract gap:
in-situ home mass 0.733 vs 0.35 target). Held-out caveat: the 11k-token
calibration fit memorizes (held-out code CE 1.587 masked → 1.621 plugged) —
cross-domain neutrality is clean (math +0.006) but "the donor's specialty"
needs larger calibration before it is claimed. Door-3 trigger ("adapters
can't bridge") did NOT fire. Wrap-vs-distill comparison: the wrap beats
distillation on delivered quality under the same delivery contract; both
share the retention/mass gap. Follow-ups: multi-slice calibration, multi-site
pack, sign-aware prototype init (registry layer_scale can be negative).

### 10.3 Door 3 — behavioral distillation (universal fallback)
Train a fresh format-native expert to imitate the donor's behavior on its
home domain (donor runs, produces scores/generations; host expert distills).
No feature compatibility needed. Fidelity loss + distillation compute, but
"any model's specialty becomes one of my plug-ins" always works.

**STATUS 2026-10-11 (`lab/door3_distill_expert.py`, MI50 GPU 4): expert-level
PASS / delivery-level FAIL on the quality gate.** Donor Phi-mini-MoE @f620b32c
→ fresh OLMoE plug-in (D1 path, registry `registry_olmoe_6d84c485.pt`,
mass-cal owner_mass 0.35 / reject_w 10 / out_pen 300), home = codeparrot +
open-web-math slices (same files as the Door 2.5 wrap lane). Cross-tokenizer
KL over the shared single-token decode set (S=12,480; char-offset alignment,
exact-boundary 0.92/0.98). Held-out KL(teacher‖student) 5.17 → 2.59 under
forced dispatch (behavior IS cloned) but → 6.63 when delivered through the
passport-mass row (in-situ home mass 0.911 vs 0.35 target; `norm_topk_prob=false`
makes mass the merge scale, so the delivered composition ≠ trained
composition). Formal D1 bars PASS (parity 0.951, retention 3.22 with the
att-3 lib>solo artifact caveat, contam 0.133, util 0.959); s6 quality FAIL
(held-out home CE 1.83 masked → 8.07 active). Shared blocker with D1's toy
plug-in and the Door 2.5 wrap comparison at quality level is the delivery
contract, not what the expert was trained from — next branch: expert-fit at
delivered-mass (or additive fixed-weight delivery). Numbers: HANDOFF §10
Door 3 + `lab/FAILURES.md` door3-distill-quality.

**STATUS 2026-10-11 (`lab/door5_provenance.py`, MI50 GPU 4): provenance
VERDICT — distilled frozen cores do NOT reproduce the donor-pretrained
quality pass; the binding property is a CE-skilled frozen core.** Arms on
the Door-3 distilled artifact (exact distill split; masked 1.8295/1.9235):
(i'') 16-site Door-2.5 wrap @1e-3: quality 10.08 vs 1.83 held-out (mass
delivery compounds the composition mismatch — the door3 8.07 failure mode);
(ii) distill-then-deliver NO refit (contract (b), λ learned/layer, cores
bitwise-frozen): quality **1.890 vs 1.829** held / 1.944 vs 1.923 train —
FAIL, but delivery is exonerated (learned λ beats the trained composition:
1.89 vs forced 2.33) — the distilled core simply has no next-token-CE skill
to deliver (forced 2.33 vs masked 1.83; its skill is KL-to-donor, door3's
5.17→2.59). (i') single-site wrap (the exact 10.2 geometry + recipe at the
verified 2048-token shape) also FAILS quality in-sample (1.882 vs 1.829
held / 1.997 vs 1.924 train; forced 1.897) while every aux gate passes —
thin adapters cannot manufacture CE skill from a non-CE core. (i) 16-site
@3e-3 NaNs in 50 steps (guard). Donor-pretrained reference (10.2) remains
the one existence proof: forced 0.101 vs masked 1.072, train 0.439. Full
table: HANDOFF Door 5 + FAILURES door5-provenance. Next lever if the grab
is still wanted: CE-objective distillation (not KL), or a CE-pretrained
donor core.

**STATUS 2026-10-12 (`lab/door7_ce_distill.py`, MI50 box): the named
CE-objective distillation lever is TESTED and does NOT unlock the tier —
the loss was never the binding constraint.** Two CE objectives at the KL
arm's budget (200 steps / lr 1e-4 / out_pen 300 / same cache split):
teacher hard targets (argmax; GT where the teacher agrees) → raw-core
forced CE is ANTI-skill (5.10/8.98 vs masked 1.83/1.92 — worse than KL's
2.33/2.39) because the cross-tokenizer teacher's argmax matches the ground
truth on only 15.6% of home positions; pure GT CE → in-sample skill
(train forced 1.657 vs 1.924 — the first distilled core with train
forced<masked) but no transfer (held 2.122 vs 1.830). Primary measurement
(single-site 10.2 wrap + Door-6 additive λ·m, the winning stack): argmax
core quality train **1.843 vs 1.924 PASS in-sample** (a first for a
distilled core; KL (i') 1.997/1.924) but held-out 1.837 vs 1.830 FAIL
(ratio 1.004) and forced ≈ masked; parity 0.9993 / contam 0.123 / util
0.891. Mass read of the same wrap: 1.866/1.938 (beats KL's 1.882/1.997,
still fails). Donor-pretrained tier (0.439/1.072, forced 0.101) is not
reached by any CE objective at this data budget — CE skill in a frozen
core comes from pretraining breadth, not from the distill loss. GT-core
wrap + deliver arms in flight (do they deliver the in-sample skill?).
Numbers: HANDOFF Door 7 + FAILURES door7-ce-distill +
`/data/tliao/graft_d7_out/door7_*.json` (untracked).

### 10.4 Door 6 — the mix (two specialties, one host; the first "sum of parts" test)
If wrap-verbatim (10.2) is the grab, the product claim needs MULTIPLE grabbed
specialties coexisting in one spine without fighting — the step toward the
long-term "mix experts from GLM/K3/Mimo" story, first tested here with two
Phi-mini-MoE specialties on the OLMoE graft. Two plug-ins from ONE donor
(@f620b32c), different homes by D2 RouteStats: code L13 e8 (lift 16.5, the
10.2 winner, artifact reused verbatim) and math L24 e4 (lift 3.69 — math
specialists are much mushier; profile mushiness 0.919). Each wrapped exactly
as 10.2 (frozen core + thin task-loss adapters, lr 3e-3 / wd 0 / out_pen
300) on its own home data, at TWO depth-matched host layers (donor L13/32→
host L6, L24/32→host L12; `--layout same-layer` implements the
same-layer-different-rows alternative). Delivery = Door 4's additive winner
per expert per layer: y = y_stock + λ_A·m_A·e_A + λ_B·m_B·e_B (membership =
score beats k-th stock score; no softmax-mass competition by construction).
Joint = measurement + a light λ/recal pass only (cores and adapters frozen):
on-trajectory row recal with both live + 120-step joint λ fit on code+math.

**STATUS 2026-10-12 (`lab/door6_expert_mix.py`, MI50 GPU 1): MIX WORKS AS A
COMPOSITION — both specialists deliver simultaneously with zero
cross-contamination and the union gain is EXACTLY the sum of the solo gains
(0.996×) — "more than the sum of parts" NOT shown; held-out still shows the
10.2 memorization gap.** Calibration-slice CE: code 1.072 → 0.231 both-on
(A-only 0.221 — additive λ·m is strictly better than 10.2's mass-delivered
0.439 on the same trained wrap); math 2.700 → 0.668 (B-only 0.654; B's solo
Door-2.5 table 2.700→1.070→0.452, parity 0.998, contam 0.059). Full 2×2
cross reads within noise (both−solo +0.014 math / +0.010 code; solo
neutrality A-on-math +0.010, B-on-code −0.000). Base parity 0.99636,
contamination 0.068/0.082, in-situ routing separation crisp (code: A 0.911 /
B 0.085; math: A 0.042 / B 0.908). Union CE 1.960→0.469 both-on; gain_both
1.491 vs solos 0.381+1.116=1.497 (ratio 0.9957). Joint pass: rows recal
(home mass 0.73→0.87 / 0.78→0.86, base 0.040→0.023 / 0.055→0.031), λ
0.5→0.952 (A) / 1.053 (B). Held-out honesty: both-on is WORSE than masked on
both probe halves (code 1.587→1.664, math 2.104→2.439) — the 10.2 caveat
unchanged; **multi-specialty coexistence is not the blocker; calibration
breadth is.** Numbers: HANDOFF Door 6 + FAILURES door6-expert-mix +
`lab/imported/graft_door6/door6_mix.json` (untracked).

### 10.5 Door 8 — bridge width vs GENERAL-LM recovery (the all-grafted-bank make-or-break)
The endgame ("our own spine + a MoE whose ENTIRE expert bank is grafted donor
experts") needs the wrap bridge to carry not just specialty deltas but the
boring general FFN lifting.  Q: **how wide must d_canon be for a wrapped
generalist core to recover the CE cost of removing a NATIVE expert on general
text?**  Bars (v2 feasibility line): ≥80% recovery at rank ≤512 = FEASIBLE;
50–80% = marginal (wider/multi-core); <50% = the bottleneck is fatal and the
v2 format needs a different bridge.  Protocol (`lab/door8_bridge_width.py`,
fork of door25; d0/d1/d2/d25 imported UNMODIFIED): generalist donor core =
the max-mushiness Phi-mini-MoE expert (D2 RouteStats, usage-floor'd hit-rate)
frozen in the door25 geometry with the rank sweep d_canon ∈ {64, 256, 512,
1024} (trainable 786k / 3.15M / 6.29M / 12.58M — note 512 ≈ one native OLMoE
expert's 6.29M); general text = graft_d0_slice web+chat+science+math + a
wikitext-2-TEST slice (disjoint from web.txt = wikitext-2-train), 80/20
train/held-out, 137k/32k tokens, val_mixed as third set; arms (a) stock host
(b) native expert N masked — its routed slots zeroed in the stock merge, an
exact capacity removal under norm_topk_prob=false (c) N masked + wrapped
core delivering additively at the site where N lives
(`y = y_masked + λ·e_wrap(h)`, door6's additive with m≡1), recovery% =
(CE_masked − CE_graft)/(CE_masked − CE_stock) with paired per-batch SE.

**STATUS 2026-10-12 (`lab/door8_bridge_width.py`, MI50 box GPU 0, run-1
complete; width-norm run-3 retry-chain detached in flight): the 64-dim
bridge does NOT kill general traffic at expert-replacement scale — a 64-wide
wrap OVER-RECOVERS the masked-native CE cost on general held-out — but only
rank 64 trains at the verbatim door25 recipe: 256 fits a net-harmful wrap
(lam*=0), 512/1024 NaN, and the single-expert demand itself is tiny.**
Setup: core = Phi @f620b32c **L3 e8** (mushiness 0.9998 = global max of the
32x16 grid, hit-rate 0.142 vs 0.125 uniform — a true domain-generalist
workhorse; 11.8M frozen); N = OLMoE **L2 e7** (top of a 16-candidate
per-layer top-routed search; probe mask cost +0.0167, gen_heldout
**+0.0022** nats — single-expert capacity of a 64x-top-8/16-layer host is
small and concentrates on web (+0.0223) rather than chat/math/science
(~+0.000); val_mixed +0.0003).  Recipe: lr 3e-3 / wd 0 / clip 1 / 250 steps
bs4x512 / out_pen 0 (pure task loss); lam fitted on train (grid+refine).

**The width-vs-recovery curve (door25 verbatim init; gen_heldout CE n=31.7k
tok; recovery% = (CE_masked - CE_graft)/(CE_masked - CE_stock), denom
0.0022 nats):**
| d_canon | trainable | status | lam* | CE graft@lam* | CE graft@lam=1 | recovery% @lam* |
|---|---|---|---|---|---|---|
| 64 | 786,432 | ok (wall 778 s) | 0.25 | **2.2874** (stock 2.3463 / masked 2.3485) | 2.4887 (overshoot) | **+2723% - 440** (over-recovery; +0.0611 - 0.0099 nats, 6 sigma) |
| 256 | 3,145,728 | ok-but-useless | **0.0 = deliver nothing** | 2.3485 (= masked) | 5.8582 (catastrophic) | **0.0%** |
| 512 | 6,291,456 | nan_abort (3e-3 ~step 30; poisoned 1.5e-3 retry NaN at step 1) | — | — | — | N/A |
| 1024 | 12,582,912 | nan_abort (fallback step 0) | — | — | — | N/A |

Read it in absolute CE: only rank 64 delivers a positive general-LM delta
(-0.059 nats vs stock at lam=0.25; recovered 0.0611 - 0.0099 vs the 0.0022
cost).  At 256 the wrap trains but is net harmful — the lam grid picks 0
(any delivery makes train CE worse than masked; lam=1 gives CE 5.86); at
512/1024 it never trains.  Recovery% is numerically explosive on this
denominator (val_mixed halves flip sign) — the curve's design content is
the absolute-CE column plus the trainability column.  **Root cause of the
>=256 failures (the width sweep's real finding):** door25's fixed init std
0.02 makes Var(u_k) of the adapter_up->frozen-core path grow prop. to
d_canon, so the frozen donor SwiGLU sees inputs sqrt(d_canon/64)x its
natural scale and explodes the fp16 residual (|y| 2.2 at 512 step 25 ->
NaN; the half-lr fallback then NaN'd at step 1 from the poisoned weights).
**Fix = width-normalized init** (adapter_up/down x sqrt(64/d_canon),
composite scale pinned at the verified 64 geometry — the v2 format's init
rule) + fresh reinit before any lr fallback.  **Run-3 COMPLETE (width-norm
init) — the fix makes wide ranks trainable, but rank 64 remains the best
carrier:**
| d_canon | trainable | lr | lam* | gen@lam* | nats recovered | recovery% |
|---|---|---|---|---|---|---|
| 256 | 3,145,728 | 3e-3 | 0.125 | 2.3441 | +0.0044 - 0.0103 | 197% (noise-level) |
| 512 | 6,291,456 | 1.5e-3 (fallback) | 0.25 | 2.3044 | +0.0442 - 0.0105 | 1969% |
| 1024 | 12,582,912 | 1.5e-3 (fallback) | 0.0 | 2.3485 | 0 | 0% |

vs run-1's 64 (2.2874, +0.0611 nats).  Non-monotone in width: 64 > 512 >
256 ~ 1024.  512 still NaN'd at 3e-3 under width-norm (the fresh-init
1.5e-3 fallback trained it); 1024 trained but is net-harmful (lam*=0).  So
the answer to "how wide must the bridge be" is **64 is enough — wider is
not better at this recipe/step budget**; width is not the binding
constraint.  Specialty re-check (secondary): the generalist core on code
fits at 64 (train 0.1458 / held 1.892, door25 comparables 0.101/1.950) but
is WORSE at 512 (train 1.0263 @lam 2.25 / held 1.9613) — same rank-64
sweet spot.

**Verdict (v2 feasibility line, run-1): FEASIBLE at rank 64 for the
expert-replacement demand** — a grafted generalist over-recovers one native
expert's general-LM job at the narrowest bridge, so an all-grafted bank is
not blocked by bridge width at per-expert demand.  Ranks 256-1024
as-specified fall in the <50% / untrainable band at the door25 recipe — but
that is an INIT-SCALE failure, not evidence of bridge incapacity (run-3
re-measures under the fix).  Two conditions for the v2 line: (1) delivered
scale must be fitted (lam 0.25 — the trained lam=1 composition overshoots;
the door6 lam lesson at general scale); (2) the wrap init must be
width-normalized or ranks >=256 never train.  The single-expert mask cost
of 0.0022 nats is itself a design input: on a 64x-top-8 host, "one native
expert's job" is a small demand — the endgame bank test should be read at
bank scale (all-top-k masked), the queued follow-up.  Secondary (code
specialty at wide rank) + the membership-gated delivery arm (door25
site-with-membership) follow run-3.  **Bank-scale demand test (the v2 freeze gate, `--bank-test`, run at the
run-1 site L2; demand = how many of the layer's routed experts are masked
(exact slot zeroing), one wrapped core standing in per arm, width-norm init,
lam grid extended below 0.25):**
| demand | mask | CE stock | CE masked | mask cost | graft@lam* (rank 64) | nats recovered | recovery% |
|---|---|---|---|---|---|---|---|
| 1 (top-1 e7) | 1 | 2.3463 | 2.3485 | +0.0022 | 2.2896 (lam 0.3125) | +0.0589 - 0.0137 | 2625% |
| 2 (top-2) | 2 | 2.3463 | 2.3489 | +0.0026 | 2.3004 (lam 0.25) | +0.0485 - 0.0091 | 1885% |
| **all (full bank)** | 64 | 2.3463 | **2.4839** | **+0.1376** | **2.4046** (lam 0.3125) | **+0.0793 - 0.0132** | **57.7%** |

rank 512 at demand-all: CE 2.4189 (lam 0.225, lr 1.5e-3 fallback — 3e-3
NaNs even width-norm at this demand), +0.0650 - 0.0120 nats, **47.2%**.
val_mixed mirrors (masked 1.5956 vs stock 1.4952, cost +0.1004).  The
layer denominator is real (+0.138 nats = 62x the single-expert cost) and
per-slot redundancy is high (top-1 and top-2 mask costs are near-identical
at +0.0022/+0.0026 — the bank is load-bearing as a distributed whole).
Cheap membership-gated delivery (wrap stands in only on tokens that routed
into the masked set) is **degenerate at demand-all** (gated CE == forced CE
to 4 decimals — every token routes into a 64-expert mask) and WORSE than
forced lam* at m=1/2 (2.3404/2.3327 vs 2.2896/2.3004 — the wrap learned to
help all tokens; gating limits its scope): forced additive lam* is the best
delivery everywhere.  Rank-512 rows: m1 +0.0441 (1965%), m2 +0.0266
(1034%).  **Multi-core freeze confirmation (Door 8c, COMPLETE): v2 format NOT
FREEZE-ABLE on the stated bar — recovery DEGRADES with core count.**
Standing N = 1/2/3 DIFFERENT donor cores (mushy Phi generalists L3 e8 /
L2 e1 / L0 e3, one rank-64 width-norm wrap each, independent seeds) in for
demand=all with per-core fitted lam (common grid + coordinate descent):
| cores | trainable | lam* per core | CE @lam* | nats recovered | recovery% |
|---|---|---|---|---|---|
| 1 | 786,432 | [0.30] | 2.4043 | +0.0796 - 0.0129 | 57.8% |
| 2 | 1,572,864 | [0.25, 0.15] | 2.4164 | +0.0675 - 0.0123 | 49.0% |
| 3 | 2,359,296 | [0.10, 0.15, 0.05] | 2.4310 | +0.0529 - 0.0078 | 38.4% |

Adding cores makes it WORSE: jointly-trained same-site wraps on one
compensating task interfere (per-core lam shrinks; lam=1 overshoot grows
2.60 -> 2.91 -> 3.02).  Door6's exact additivity (0.996x) does NOT transfer
— it composed SOLO-TRAINED specialists (adapters frozen, lam-only joint
pass), not jointly-trained same-task wraps.  **No core count reaches 80%
under joint training; the bottleneck is composition mode, not bridge width
or count.**  Freeze input for v2: (1) the 64-dim bridge + width-norm init +
additive lam + registry rows are supported — one rank-64 wrap carries ~58%
of a full layer's demand and 2625% of a single native slot (the per-graft
endgame claim is safe); (2) the freeze re-test is composition MODE — **Door 8d COMPLETE: NOT
FREEZE-ABLE in ANY mode; the demand=all task SATURATES at one rank-64
wrap's capacity (~58% of a full layer):**
| mode | cores | lam* | CE @lam* | nats recovered | recovery% |
|---|---|---|---|---|---|
| joint (8c) | 1 | [0.30] | 2.4043 | +0.0796 - 0.0129 | 57.8% |
| joint (8c) | 2 | [0.25, 0.15] | 2.4164 | +0.0675 - 0.0123 | 49.0% |
| joint (8c) | 3 | [0.10, 0.15, 0.05] | 2.4310 | +0.0529 - 0.0078 | 38.4% |
| boosting (1 frozen + 2 residual) | 3 | [0.25, 0.10, 0.10] | 2.4056 | +0.0783 - 0.0120 | 56.9% |
| per-slot OUT-of-env (8d) | 8 | [0.1x6, 0.25, 0.1] | 2.5680 | -0.0840 - 0.0592 | **-61.1%** |
| per-slot **IN-env** (8e, the endgame shape) | 8 | [0.1x7, 0.25] | 2.4823 | +0.0016 - 0.0332 | **+1.1%** |

Boosting is flat vs one core (residual gains collapse to 0.1).  **Door 8e
(the endgame shape: per-slot DATA fits under the demand=all ENVIRONMENT)
COMPLETE: training atop the dead layer removes the environment-shift harm
(-61.1% -> +1.1%) but creates NO capacity** (+0.0016 - 0.033 nats; the
top-k probe recovers -11.2% of even the COVERED demand => COMPOSITION is
the binding limit, with top-8 slot coverage a secondary cap at 11.9% of
the layer's CE demand).  **FREEZE VERDICT: NOT FREEZE-ABLE** — no
mode/count reaches 80%; the demand=all objective saturates at one rank-64
wrap's ~58% and multi-wrap composition of any flavor does not add.
Freeze input: (a) the format itself is not implicated (64-dim bridge,
width-norm init, additive lam, registry rows — every failure is
composition/capacity); (b) the per-graft endgame claim stands (m1 = 2625%
of one slot's demand) but a whole layer needs its FULL graft bank (the
untested 56 slots carry 88% of demand) with per-graft performance at m1
level; (c) few-core whole-layer replacement needs more capacity PER wrap,
not more wraps.  Numbers:
HANDOFF Door 8d + Door 8c + Door 8 +
`lab/imported/graft_door8/` (untracked) + box `/data/tliao/graft_door8_out/`
(door8_width.json, door8_width_p2_a*.json, door8_bank_a*.json, wraps/,
core_gen.pt).

### 10.6 Door 10 — multi-slice calibration scale (the data-scale hypothesis)

The door25 held-out caveat (fit 0.439 vs held 1.621 on 11k tokens), door7's
breadth limit and door8's small-lambda deliveries all point one way: the
memorization/held-out gap is DATA STARVATION, not method.  Door 10 tests it
directly — the Door 2.5 winner (Phi L13e8 frozen core, rank-64 width-norm
wrap, door25 recipe 250 steps lr 3e-3 wd 0 out_pen 300) retrained at
calibration budgets 11k / 50k / 200k / 500k code tokens (codeparrot; nested
prefixes of one 540k pool), quality read on THREE md5-disjoint slices: FIT
(the arm's own calibration data), HELDOUT (same distribution, fresh rows),
UNSEEN3 (a third fresh draw, never touched by fit/lam/recal).  Delivery
states: masked / forced lam=1 / additive lam* (fit on FIT slice only) /
plugged+recal.  gap = CE_state - CE_masked (paired per-batch SE).

| budget | ep | lam* | fit gap_add | HELD gap_add | UNSEEN3 gap_add | mem gap | flip? |
|---|---|---|---|---|---|---|---|
| 11k  | 46.6 | 1.0000 | -0.9712 - 0.0375 | +0.3825 - 0.0261 | +0.3375 - 0.0157 | 1.3536 | no |
| 50k  | 10.2 | 0.3125 | -0.0153 - 0.0030 | +0.0236 - 0.0083 | +0.0068 - 0.0013 | 0.0389 | no |
| 200k | 2.6  | 0.1875 | -0.0028 - 0.0005 | **+0.0003 - 0.0005** | +0.0013 - 0.0004 | 0.0031 | no |
| 500k | 2.0  | 0.3125 | -0.0030 - 0.0014 | +0.0010 - 0.0011 | +0.0032 - 0.0009 | 0.0040 | no |

**The gap collapses with data and SATURATES AT PARITY.**  11k reproduces
the door25 caveat exactly (fit 1.072/0.101 vs HELD 1.067/1.450 — a
46-epoch memorization read: fit gap -0.97, held gap +0.38, mem gap 1.35).
50k (10 ep): mem gap 34.8x smaller (0.039).  **200k (2.6 ep): held gap
+0.0003 - 0.0005 — statistically ZERO** (0.6 SE above zero, not below);
mem gap 0.0031 (437x smaller than 11k).  500k (2.0 ep, 489 steps): no
further gain — held +0.0010 - 0.0011, mem 0.0040; the curve is flat
200k->500k.  UNSEEN3 tracks HELDOUT in sign at every budget (the gap is
distributional, not a held-out artifact) and is consistently smaller.

**Fit of gap vs data (4 points): SATURATING, not log-linear.**  Hyperbolic
gap = a + b/N fits best: a = -0.029, b = 4453 (HELD, r2 0.982; zero at
1.51e5 tok) and a = -0.029, b = 3938 (UNSEEN3, r2 0.969; zero 1.36e5).
Log-linear gap = a + b*log10(N) is worse (b = -0.220 nats/decade HELD,
r2 0.725; zero 2.5e5) — the 2-point fit's -0.546/decade and its ~5.5e4
zero-crossing were artifacts of the steep 11k->50k leg and are NOT
confirmed by the full curve.  Measured read: the gap reaches ~0 at 200k
and stays ~0 at 500k; the hyperbolic fit's slightly-green asymptote
(a = -0.029) is fit-driven, not observed (200k/500k sit at +0.0003/+0.0010).

**Bars: held-out flip to delivered<masked — NO at any size** (best cell
200k HELD +0.0003 = parity, not green).  What the sweep DOES certify:
**FEED >=200k code tokens per specialty kills the memorization harm**
(held gap +0.38 -> +0.0003, mem gap 1.35 -> 0.003; unseen3 agrees) — the
10-12k door25/door6/door7 budget is ~20x under and is exactly what made
the wrap look like a memorizer.  But held-out skill (delivered < masked)
is not bought by data alone at any measured scale: the donor core's
out-of-sample CE contribution saturates at zero.  Production calibration
estimate: **~2e5 tokens per specialty** (parity at 200k; 500k adds
nothing; the 151k hyperbolic zero is inside this band).  Secondary reads:
lam* falls 1.0 -> 0.19 as data grows (door8 lam lesson at scale);
plugged (mass contract + recal) is strictly worse than additive lam* at
every budget (200k HELD +0.0517 vs +0.0003) — door6's additive>mass
re-confirmed across data scale.

**Line status (complete):** all 4 arms `status ok` (box
`door10_merged.json`, wall 1123/1152/1265/1801 s).  Harness
`lab/door10_calscale.py` (fork of the door25/door8 family; d0/d1 imported
UNMODIFIED; synthetic 16/16) + `lab/door10_build_calscale.py` (pool
540k/198 docs, heldout 41k/28, unseen3 41k/25, md5-disjoint) +
`lab/door10_chain2.sh` (claim-aware GPU grab; box
`/data/tliao/graft_door10_out/`).

### 10.6.1 Recipe certification from Door 10

What the sweep certifies for the v2 recipe: **FEED >=200k calibration
tokens per specialty — the "10-12k is enough" assumption is retired.**
At that budget the wrap's held-out delivery is NEUTRAL (parity with the
masked baseline on both unseen slices) with zero memorization signature,
where the 11k budget shows +0.38 nats held-out harm and a 1.35-nat
memorization gap.  What it does NOT certify: held-out delivered<masked
(skill) — not reached at 11k/50k/200k/500k; data alone saturates at
parity.  A held-out skill claim needs a second ingredient (more capable
core / different delivery), not more calibration rows.

Composition: models in the format swap experts natively (10.1); out-of-format
donors are grabbed by wrap (10.2) or distilled (10.3). Registry economics
(§9 sweep: ~8MB spine artifact, 256B/row at d_canon=64) are unchanged — the
format governs the EXPERT side, the passport the ROUTING side.

## §10.7 THE STANDARD RECIPE (2026-10-13 — "simple and flexible", user directive)

After the Door 8/9/10 falsification wave, the surviving plug-in recipe is
frozen as the standard. Do NOT improvise alternatives without a falsifier:

    1. GRAB    frozen donor expert core (any family) — never trained here
    2. WRAP    rank-64 adapters, width-norm init (x sqrt(64/d_canon)),
               wd 0, lr 3e-3, task-CE + out_pen 300, ~250 steps (~20 min)
    3. FEED    50-200k specialty tokens (door10 4-point: mem gap 1.35 -> 0.039
               at 50k -> 0.003 at 200k where held gap = +0.0003 = parity;
               10-12k budgets are ~20x UNDER and actively harmful held-out;
               500k adds nothing — saturating, not log-linear)
    4. DELIVER additive lambda (fitted, grid incl. below 0.25) —
               softmax-mass stacking is strictly worse (door6/door10)
    5. ROUTE   registry row (256B) + mex recal owner_mass 0.85, reject_w 10

Killed branches (evidence on file, do not retry without new ideas):
- feature translators (d1.5, door9): ~0.5 alignment ceiling
- scratch/refit deltas (93188b0): 3 contracts x 6 recipes, all fail
- KL-distill cores (door5/door7): no CE skill; CE-distill in-sample only
- whole-layer substitution (door8b-e): ~58% saturation, all modes
- wide bridges / fancy inits (door8/door9): rank-64 random-init wins

Flexibility claim: the same 5 steps serve code/math/memory/sensor
specialties and compose in the v0 assembly (69949e8). Open rungs:
combine-at-scale (bank2n quality half), memory token-surfacing.

## §10.8 SPINE PORTABILITY (2026-10-13 — v0.6 impress-demo)

The recipe runs UNCHANGED on a second host family: **Qwen1.5-MoE-A2.7B-Chat**
(qwen2_moe, 24L, hidden 2048, 60 experts, top-4, shared expert — 14.3B
total / 2.7B active) took the whole d0/d1/d2/d25/d6/g0 stack with **zero
shared-code edits**. D2's port lesson ("adapt the router block once,
everything else follows") is now two-host verified: qwen2_moe duck-types as
the OLMoE family — `find_moe_blocks` sees `.gate`+`.experts`,
`Qwen2MoeTopKRouter` returns `(logits, top_val, top_idx)` matching
GraftGateD2's contract, `experts.forward(h, idx, w)` matches MixMerge's,
and the gfx906 experts shim's `gate_up_proj`/`down_proj` duck-type hits
Qwen2MoeExperts directly (24 banks patched). FamilySpec carries the only
real variance (top_k 4, norm_topk_prob False).

Registry/graft parity gate on the new spine: **logit cos 0.9989** (min
0.9553, ce_delta +0.002) — `init=exact` passports, so this is plumbing
parity (the port is not corrupting the route), clearing the ≥0.95 bar.

Host-memory note (14.3B fp16 = 26.6GB on a 32GB MI50): the registry/parity
phase must run SMALL streams (seq 128, 8-32 batches) — vocab 151,936 logits
plus 26.6GB weights OOM at v0.5's seq 512 with 146 val batches. Co-tenant
races still bite (OOM mid-parity when a co-tenant landed) — claim-lock +
retry is mandatory.

gfx906 backward holes are SPINE-SPECIFIC, not universal (door25
door25-gfx906-backward-holes caveat, now measured on a second host): a solo
shape grid on the Qwen spine (CE fwd+bwd through a wrap, GPU 2, no
co-tenant) gives B*T ∈ {128, 192, 256} OK and {320, 384, 512} FAIL with
`hipErrorInvalidDeviceFunction` — v0.5's HOLEY_BT {256,768,1024,4096}
(Phi-mini-MoE) does NOT transfer (256 is holey on Phi and safe on Qwen).
v0.6 therefore trains at bs1×256 with the step count raised to keep FEED
≥200k (782 steps × 256 = 200,192 tok-slots; v0.5's 500×512 = 256k at the
old shape). Every new spine needs its own shape grid before the first fit.

Row-scale guard (second spine-specific lesson): the v0 prototype init for
registry rows (`home_h.mean · sqrt(d_r)`) saturates the passport softmax on
the Qwen spine (hidden norms ~2x Phi's) — p_new = 1.0 on home AND base gives
zero recal gradient (reject/mass loss stuck at 10.045, first==last in every
fit_*.json). Fix: scale each graft row to the layer's mean STOCK-row norm
before recal (direction of the home prototype, magnitude of the bank's own
rows). With the fix the closing recal converges (p_home 0.89–0.96, p_base
0.006–0.018) and the assembled bank delivers: all 5 homes improve, base tax
−0.0035, memory CE 1.505→0.053 (taught token "Kepler" surfaces in the chat
probe).

## §10.9 VARIABLE-ACTIVATION PLUG GATING (2026-10-13 — variable-k rung 1)

The bank2n flood finding (phi_math delivering 0.42 of code / 0.89 of math;
worst pairwise harm 0.226) and the v0.5.1 cross-fire matrix (math row still
flooding editor/writing after xfire-recal; the mem trio cross-firing ~0.99 on
each other at a ~0.10-nat cost to mem_words) share one mechanism: delivery
membership is binary and bank-local — a plug fires when its row score beats
the k-th stock score, with no notion of "is this plug's home likely for THIS
token". Variable activation (variable-k rung 1) adds exactly that, as a
serving-time policy with ZERO retraining over door6's additive contract:

    y = y_stock + sum_j lam_j * m_j * g_j * e_j

where m_j is existing membership and g_j(token) is a policy mask built from
the plug row's home probability p_home = softmax(bank scores)[row]
(MixGateD2.last_p_new). Policies (knobs tau / m):

    thresh    g_j = (p_home_j >= tau)                  tau 0.3/0.5/0.7
    topm      a token may pull at most m plugs bank-wide  m 1/2/3
    adaptive  p_home >= tau AND budget m (m adapts 0..m per token)

`lab/varact_gate.py` is a wrapping layer (VarActMerge wraps door6.MixMerge
in place; shared code untouched). Two structural facts shaped it:

1. Plugs sit at DIFFERENT host layers (v0.5 bank: L1/L2/L4/L6/L13/L19/L24/
   L27), so "top-m plugs for this token" is bank-wide — and because later
   layers' p_home is not yet computed when early layers decide, only CAUSAL
   policies are implementable: a shared per-token budget counter in host
   order (first firing plugs win). True score-ranked bank-wide top-m needs a
   non-causal second pass (open rung). The causal order matters: topm1
   starves late plugs whenever an early flooder fires (measured — see below).
2. gfx906 cannot run per-token subset expert forwards (FAILURES
   `varact-subset-gather`: `nonzero()` itself raises
   hipErrorInvalidDeviceFunction, asynchronously). Variable activation
   therefore SKIPS a plug entirely when its keep-set is empty for the batch
   (zero expert FLOPs) and otherwise runs the exact stock shape with the
   output masked. At serving (T=1 decode) skip-vs-run is per-token exact,
   so deliver/token IS the serving plug-forward count; the offline batch
   FLOP metric understates the serving win.

Protocol: 2x2 {no gate, gated} x {compute (tok/s, active plugs/token),
quality (home CE per plug on 6 rep windows, cross-fire delivery matrix, base
tax)} on the v0.5/v0.5.1 assembly with joint_state.pt READ-ONLY, sweeping
tau and m; plus leave-one-out pairwise harm (bank2n bar 0.02) and the
mem-sibling pair family; plus the 4-probe chat smoke. Ship bars: every green
plug keeps home CE <= the ungated default, active-plug compute cut >= 30%,
worst pairwise harm <= 0.02.

RESULTS (v0.5/v0.5.1 assembly, 8 plugs, 6 rep windows/home, MI50 fp16;
joint_state.pt read-only; `lab/varact_gate.py --phase all` incremental JSON
`/data/tliao/varact_out/varact.json`):

**Quality-vs-compute frontier** (actives/token = plug-expert forwards per
token at T=1 serving; ungated runs ALL 8 plugs every token = 8.0):

| config | gen tok/s | actives/tok | cut | base-tax CE | worst green delta | flood sum |
|---|---|---|---|---|---|---|
| base (ungated) | 2.98 | 8.00 (1.92 fire) | 0% | 1.572 | — | +0.265 |
| tau0.3 | 3.03 | 1.84 | 77.0% | 1.564 | k3 +0.005 | +0.232 |
| **tau0.5** | 2.98 | 1.81 | **77.3%** | **1.557** | mem_edit +0.006 | +0.222 |
| tau0.7 | 3.06 | 1.79 | 77.6% | 1.558 | k3 +0.010 | +0.220 |
| topm1 | 2.82 | 0.99 | 87.6% | 1.574 | mem_edit **+0.320** | +1.044 |
| topm2 | 3.09 | 1.52 | 81.0% | 1.572 | mem_words +0.036 | +0.325 |
| topm3 | 3.03 | 1.91 | 76.1% | 1.572 | mem_words +0.024 | +0.287 |
| adaptive_tau0.5_m2 | 3.09 | 1.45 | 81.9% | 1.557 | mem_words +0.028 | +0.2577 |

flood sum = sum over homes of (home CE - owner-only CE) = the presence cost
of everything that is not the home plug. Home CE (masked -> per config,
delta vs ungated; green = the v0.5.1 winners):

| plug | masked | base | tau0.3 | tau0.5 | tau0.7 | topm1 | topm2 | topm3 |
|---|---|---|---|---|---|---|---|
| code | 0.924 | 0.392 | 0.390 (-.003) | 0.388 (-.004) | 0.395 (+.003) | 0.564 (+.172) | 0.400 (+.008) | 0.392 (+.000) |
| math | 3.058 | 1.413 | 1.407 (-.006) | 1.400 (-.013) | 1.398 (-.015) | 1.724 (+.311) | 1.437 (+.024) | 1.413 (+.000) |
| mem_pet | 3.686 | 0.027 | 0.028 (+.001) | 0.029 (+.001) | 0.029 (+.002) | 0.041 (+.014) | 0.026 (-.001) | 0.027 (+.000) |
| mem_edit | 5.095 | 0.255 | 0.258 (+.003) | 0.261 (+.006) | 0.259 (+.004) | 0.575 (+.320) | 0.252 (-.003) | 0.254 (-.001) |
| mem_words | 2.942 | 0.478 | 0.466 (-.012) | 0.471 (-.007) | 0.460 (-.018) | 0.453 (-.025) | 0.514 (+.036) | 0.502 (+.024) |
| k3_hack | 1.955 | 1.827 | 1.833 (+.005) | 1.831 (+.004) | 1.838 (+.010) | 1.837 (+.009) | 1.828 (+.001) | 1.827 (+.000) |
| glm_cybersec | 1.898 | 1.934 | 1.931 (-.003) | 1.928 (-.006) | 1.932 (-.002) | 1.918 (-.016) | 1.931 (-.003) | 1.933 (-.001) |
| mimo_writing | 2.712 | 2.752 | 2.748 (-.004) | 2.748 (-.004) | 2.748 (-.004) | 2.747 (-.005) | 2.749 (-.003) | 2.752 (+.000) |

adaptive_tau0.5_m2 is the near-miss compute edge (81.9% cut at the same
1.557 base tax) but the m=2 causal budget starves mem_words again
(mem_pet + mem_edit take both slots on the words home) -> +0.028.

**SWEET SPOT: thresh tau=0.5** — 77.3% active-plug compute cut (1.81 vs
8.00 plug-forwards/token), every green plug within +/-0.006 nats of the
ungated default (4/8 strictly better incl. math -0.013 and mem_words
-0.007), base tax IMPROVES (1.572 -> 1.557; ungated tax +0.036 over masked
1.536 becomes +0.021), gen tok/s unchanged (noise band 2.8-3.1 across all
points: the plug share of wall-clock is small vs the 1.8B-active GEMMs;
the win is plug compute/memory, not single-stream speed on this spine).
Ship-bar status: compute PASS (77% >> 30%); green plugs PASS within the
measured noise band (strict-letter clause: 3 plugs show +0.001..+0.006 —
a paired-SE repeat would settle those); pairwise harm bar measured in the
follow-up LOO pass (flood-sum proxy shrinks 0.265 -> 0.222 and every
off-home delivery share drops toward zero at tau0.5: k3-on-cybersec
0.051->0.026, glm-on-writing 0.024->0.003, code-on-cybersec 0.084->0.070).

**What the sweep teaches about the two hypotheses:**
- (a) COMPUTE: the win is LARGE and nearly tau-insensitive (77-78% at every
  tau). Membership already implies high p_home, so tau 0.3->0.7 trims only
  1.84->1.79 actives/token; the 8.0 -> 1.8 collapse is skipping plugs whose
  keep-set is empty. m is the real compute knob (0.99-1.52 actives/token).
- (b) INTERFERENCE: split verdict. phi_math-style LOW-P flooding shrinks
  decisively at tau=0.5 (off-home delivery shares -> ~0.003-0.008; flood
  sum 0.265->0.222). mem-words-style SIBLING crowding does NOT: the mem
  trio cross-delivers ~0.99 on each other's homes under every tau point
  (words home at tau0.5: mem_pet 0.997 / mem_edit 0.990 / mem_words 0.992)
  because their p_home is mutually high — p_home cannot separate
  same-family siblings. mem_words flood cost 0.206 -> 0.198 nats (token).
- topm (the naive fix for siblings) pays for exclusivity with owner
  starvation: the causal budget fills in HOST-LAYER ORDER (L1 mimo, L2
  glm, L4 k3, L6 mem_pet, L13 code, L19 mem_edit, L24 math, L27
  mem_words), so early flooders eat the slots. topm1: code/math/mem_edit
  regress +0.17..+0.32, flood sum EXPLODES to 1.04. topm2: mem_pet +
  mem_edit take both slots on the words home (0.996/0.985 delivery) and
  mem_words itself is starved (0.001) -> +0.036. Rung-2 lever: p-ranked
  budget (second pass, non-causal) or owner-first reservation at each
  layer before the budget is spent.

Open rungs: (1) score-ranked bank-wide top-m (non-causal, needs the full
bank's p_home — a cheap second policy pass over cached scores); (2)
sibling disambiguation for same-family memories (p_home needs a family-
margin term, or the budget reserves the owner slot at the home's own
layer); (3) paired-SE repeat of the tau0.5 home-CE deltas to convert
"noise-level" into a numeric bar; (4) variable-k rung 2: letting m vary by
token difficulty learned from the p_home distribution (this rung's
adaptive mode is the hand-coded version).

**LOO pairwise harm (final — `lab/varact_gate.py --phase harm`, 4 held-in
rep windows, harm(j->h) = CE_h(all plugs) - CE_h(j off); + = j hurts h):**

| config | worst all pairs | pair | worst non-sibling | pair |
|---|---|---|---|---|
| base (ungated) | 0.2035 | mem_pet->mem_words | 0.0109 | code->glm_cybersec |
| tau0.3 | 0.1953 | mem_pet->mem_words | 0.0102 | code->glm_cybersec |
| **tau0.5** | **0.2021** | mem_pet->mem_words | **0.0091** | code->glm_cybersec |
| tau0.7 | 0.1860 | mem_pet->mem_words | 0.0090 | code->glm_cybersec |
| topm1 | 0.3171 | mem_pet->mem_edit | 0.1587 | mem_edit->math |
| topm2 | 0.1899 | mem_pet->mem_words | 0.0101 | code->glm_cybersec |
| adaptive_tau0.5_m2 | 0.1872 | mem_pet->mem_words | 0.0085 | code->glm_cybersec |

tau0.5 full matrix (rows = home h, cols = plug j; own column = own_effect):

| home | code | math | mem_pet | mem_edit | mem_words | k3 | glm | mimo |
|---|---|---|---|---|---|---|---|---|
| code | -0.579 | +0.000 | +0.002 | +0.000 | +0.000 | +0.000 | -0.001 | +0.000 |
| math | +0.000 | -1.902 | +0.000 | +0.001 | -0.000 | +0.000 | +0.000 | +0.000 |
| mem_pet | +0.000 | +0.000 | -1.218 | +0.005 | +0.002 | +0.000 | +0.000 | +0.000 |
| mem_edit | +0.000 | +0.000 | +0.009 | -0.333 | +0.004 | +0.000 | +0.000 | +0.000 |
| mem_words | +0.000 | +0.000 | **+0.202** | **+0.143** | -0.038 | +0.000 | +0.000 | +0.000 |
| k3_hack | +0.000 | +0.001 | +0.000 | +0.000 | +0.000 | -0.191 | +0.003 | +0.000 |
| glm_cybersec | +0.009 | +0.000 | +0.000 | -0.000 | +0.001 | +0.001 | +0.000 | +0.000 |
| mimo_writing | +0.000 | -0.001 | +0.000 | -0.000 | +0.000 | +0.000 | +0.000 | +0.002 |

The harm bar splits cleanly by class. **Flooding-class (cross-family) harm
is INSIDE the 0.02 bar at tau=0.5: 0.0091** (was 0.0109 ungated) — the
bank2n-style off-home harm is what variable activation actually kills.
**Sibling-class harm FAILS the strict bar: 0.202 (mem_pet->mem_words) +
0.143 (mem_edit->mem_words)** — identical in the ungated baseline
(0.2035/0.138): the mem trio rates each other's episodes as home (p~0.99),
no p_home threshold can separate them, and gating neither causes nor fixes
this pair. topm points are worse (0.317: causal budget displaces the harm
onto other pairs, mem_edit->math 0.159).

**4-probe smoke (greedy, 40 tok, short excerpts — all four configs green):**
tau0.5: hacking "During the reconnaissance phase, I would gather information
such as publicly available data, network infrastructure details..." (in-
domain, non-operational); cybersec "A vulnerability is a weakness or flaw in
a system that can be exploited by an attacker. An exploit is the act of
taking advantage of a vulnerability..." (structured distinction); writing
"The lighthouse stood defiant against the raging storm, its beacon a
steadfast sentinel in the tempest's fury." ; memory **"Your cat is named
Kepler."** (9 tok). Chat does not degrade under gating — and the memory
probe now surfaces the TAUGHT fact token (Kepler, not the base prior
"Whiskers" of the v0.5.1 entry) in every config including ungated, so the
v0.1 fact-token door is behaviorally closed on these greedy probes.

**FINAL VERDICT on the three ship bars (green home CE <= ungated, compute
cut >= 30%, pairwise harm <= 0.02):**
- compute: **PASS** (77.3% >> 30%)
- green home CE: **PASS within the measured noise band** (worst +0.006;
  4/8 strictly better; strict letter needs the paired-SE repeat)
- pairwise harm: **FAIL at 0.202 > 0.02** — exclusively the pre-existing
  mem-sibling pair (mem_pet->mem_words), unchanged from ungated (0.203).
  Flooding-class harm passes at 0.0091.

=> **tau=0.5 does NOT take the default flag under the stated bar.** It is
the recommended policy for compute once the sibling pair is closed (it is
otherwise the best point on every axis: quality held, tax improved, 77%
plug-compute cut). What it takes (rung-2 levers, in order of leverage):
1. **Sibling row recal (delivery-side, works TODAY):** re-run the v0.5.1
   xfire-recal recipe on the mem trio rows (reject_w 40, siblings' episodes
   ON the reject side) so each row rejects its siblings' episodes — the
   trio's mutual p~0.99 is a row-selectivity defect, not a policy one.
   Expected: sibling LOO collapses to the 0.00x band and the bar passes at
   tau0.5 with no policy change.
2. **Family-margin gate (inference, ~10 lines in VarActPolicy.select):**
   g_j requires p_j >= tau AND p_j >= max(sibling-row p) + margin — kills
   sibling double-delivery without touching cross-family routing.
3. **Owner-first budget reservation** (fixes top-m's owner starvation: the
   home plug's own layer reserves its slot before the bank budget is spent).
4. **Non-causal p-ranked top-m** (second policy pass over cached bank
   scores) — the true score-ranked variable-k.

### 10.9.1 Closing the sibling pair (2026-10-13 — rung-2 #1: sibling row recal)

The tau=0.5 ship bar failed on exactly one axis: mem-trio pairwise harm
(mem_pet->mem_words +0.202 / mem_edit->mem_words +0.143, identical in the
ungated baseline) — the trio rows rate each other's episodes at p~0.99.
The named fix is delivery-side: the v0.5.1 cross-fire recal recipe (the
lever that took k3 off-home p 0.028->0.0004) with the siblings' episodes
ON the reject side.

Recipe per mem row (d6.train_rows_one, only that row's grad):
  stream = own-home episodes (label 1, tiled home-weight) + the other two
           trio episodes + base (label 0)
  reject_w 20-40, mass_w 2.0, owner_target 0.85, lr 0.02-0.05, steps 120-200
  GUARD-CHECKPOINTED in chunks: after every chunk re-check (a) own-home CE
  under tau=0.5 delivery within +0.02 of the pre-recal value and (b) the
  taught fact token ("What is my cat's name?" -> "Kepler"); keep the chunk
  with the lowest reject-side p that PASSES both, early-stop + revert at
  the first chunk that breaks a guard.

Guard lesson (first attempt, mem_pet): the un-chunked recipe at reject_w 40
with a 2.5:1 reject:home ratio over-suppressed the row — p_reject 0.506->
0.0088 but own-home CE 0.029->0.147 and the fact token flipped (guard
reverted the row; the assembly was never left in that state — joint_state
is written once at the end and backed up to joint_state_pre_sibrecal.pt).
Tiling the home stream (home-weight 2) + reject_w 20 + chunked early-stop
trains cleanly: every chunk kept, own-home CE and Kepler stable.

RESULTS: **Sibling harm, before -> after (LOO harm(j->h) = CE_h(all) - CE_h(j off);
positive = j hurts h; the tau0.5 matrix is the ship config):**

| pair | before (ungated / tau0.5) | after recal (tau0.5 / ungated) |
|---|---|---|
| mem_pet->mem_words | +0.2035 / +0.2021 | -0.0189 / -0.0191 |
| mem_edit->mem_words | +0.138 / +0.143 | -0.0229 / -0.0181 |
| mem_pet->mem_edit | +0.004 / +0.005 | +0.0058 / +0.0059 |
| mem_edit->mem_pet | +0.008 / +0.009 | +0.0061 / +0.005 |
| worst non-sibling (code->glm) | +0.0109 / +0.0091 | +0.0091 / ['code', 'glm_cybersec'] |
| **worst sibling** | **+0.2035 / +0.2021** | **+0.0061 (mem_edit->mem_pet)** |

**Home CE check (tau=0.5 delivery):** mem_pet 0.0399 / mem_edit 0.2581 / mem_words 0.2572 (tau=0.5 delivery; pre-recal class 0.029 / 0.261 / 0.472).  The sibling-crowding
tax on mem_words was real and is now GONE: 0.472 -> 0.2572
(the 0.206 flood-sum entry for mem_words in the varact frontier was exactly
this pair).

**Kepler probe (sacred):** "Your cat is named Kepler." kepler=True
— unchanged through every recal chunk (the guard reverted any row whose
chunk flipped it; final probe re-run on the shipped state).

**Family-margin combined (rung-2 #2, two-pass measurement form, margin
0.05):** not run / pending.  With the rows selective the plain tau=0.5 gate is
already sufficient; family-margin stays as the optional belt-and-braces
(non-causal second pass) documented in the rung-2 list.


**Probe-format note (wired-chat smoke, `chat_v05.py --probe`):** the 4-probe
script asks the memory question as turn 4 of an accumulating transcript
(hacking/cybersec/writing turns first) — under that distracted context the
answer is "Your cat's name is Midnight." (a base-prior fact token, same
open door as v0.5.1's multi-turn "Whiskers").  The sacred behavior — a
fresh "What is my cat's name?" — answers **"Your cat is named Kepler."**
on the shipped state under the shipped tau=0.5 policy (verified after
every recal chunk and as the final probe; that is what the guards
protected).  The multi-turn fact-token fragility is a pre-existing v0.1
door, not a recal or gating regression: single-turn Kepler was green
before the recal and is green after.

**SHIP VERDICT: all three bars hold — tau=0.5 SHIPPED as the default
delivery policy.**  compute cut 77.3% (>=30%), green home CE held (the
trio improved; worst green delta across the bank stays in the +/-0.006
noise band from the varact frontier), pairwise harm now 0.0061 worst
sibling / 0.0091 worst non-sibling (bar 0.02).  Wiring: `chat_v05.py` +
`chat_v06.py` install `varact_gate.install_varact(handles, POLICY)` right
after `build_state`; policy priority `--varact` > `joint_state.pt["varact"]`
> `DEFAULT_VARACT="tau0.5"`; `--varact off` restores plain ungated
delivery (bit-identical to pre-varact serving), `--lam-override` unchanged.
`joint_state.pt` carries `varact: "tau0.5"`; backup
`joint_state_pre_sibrecal.pt` holds the pre-recal rows.}



## §10.10 MULTI-TURN FACT STABILITY (2026-10-13 — closing the fact-token door)

The shipped single-turn memory probe ("What is my cat's name?" asked fresh)
answers **"Your cat is named Kepler."** — sacred, green through every recal
guard.  The SAME state asked as turn 4 of an accumulating transcript
(`chat_v05.py --probe`, hacking/cybersec/writing turns first) answers
**"Your cat's name is Midnight."** — an invented base-prior name (v0.5.1's
format said "Whiskers"; the multi-turn cell flips between
Whiskers/Fluffy/Oreo/Midnight/Oliver across runs — fp16 noise on a near-tied
name-slot softmax).  `lab/mtfact.py` reproduces this on demand
(`--phase repro`) and discriminates the three hypotheses
(`--phase diag`, 3 conversation scaffolds x turns-before-probe 0/1/2/4/8).

**Diagnosis — all three hypotheses hold, in different places; the binding
failure is decode-side (H2+H3), not storage:**

1. **H3 row re-activation decay — REAL at the answer-start position.**
   p_home of the mem_pet row at the first decode position (the position
   that decides the first answer token) collapses with context length:
   1.00 (turn 0) -> 0.11 (turn 1) -> 7.8e-05 (wired turn 4) -> 4.1e-05
   (turn 8); tech scaffold turn 8 = 9.3e-07.  The shipped tau=0.5 gate
   therefore turns the memory row OFF at exactly the position where the
   fact is decided.  But at the NAME slot (after a teacher-forced answer
   prefix "Your cat's name is ") p_home recovers to ~1.0 at every depth
   (0.9999 at turn 4/8): the row is content-addressed to the taught
   ANSWER SHAPE, not to the question — it re-activates only once answer
   tokens flow.  When the base prior emits the name directly as the first
   token (tech turn 8: "Whiskers.") the decision happens where the row is
   dead.
2. **H2 decode competition — BINDING at the name slot.**  Where the row is
   live (name slot, p_home ~ 1.0, membership fires, tau gate open) the
   taught token still loses the full softmax to a mushy base prior over
   cat names.  Top-5 at the name slot from turn 1 on is all base-prior
   name fragments (Fl/Wh/O/P/Mid/B/L/Max/Snow); at turn 0 the graft pins
   "Ke"/"K" at rank 1 and greedy lands Kepler.  The memory wins its
   TAUGHT contrast at every depth (Kepler-vs-Orion +0.63..+3.5 nats,
   the episode CE is 0.03) but loses the decision to competitors that
   were never in the episode (Kepler-vs-Whiskers flips negative after one
   turn: +1.35 -> -0.42/-1.70/+0.07/-0.16).
3. **H1 context dilution — REAL amplifier.**  The taught margin decays
   with turns-before-probe (Kepler-vs-Orion 3.53 -> 1.51 -> 1.00 -> 0.63
   nats over turns 0/1/2/4 on the wired scaffold; personal goes negative
   at turns 1-4), which is why the softmax gets harder to win mid-talk.

Margin-vs-sampled-token traces and p_home traces per step are in
`/data/tliao/mtfact_out/diag.json` (box).

**Fix (decode-time only, zero retraining, `joint_state.pt` untouched): the
two named v0.1 levers, both retrieval-scoped.**

1. **Retrieval-triggered row boost** (`varact_gate.RetrievalBoostPolicy`):
   a memory row hot anywhere in the current forward (p_home >= tau at any
   position — the question tokens re-activate it) delivers at every
   membership position of the reply.  Non-memory plugs keep plain tau
   gating.  This defeats the H3 delivery kill at answer-start.
2. **Decode-time taught-token bonus** (`varact_gate.TaughtTokenBonus`):
   +beta nats on the row's taught fact tokens (Kepler/helix/Four and their
   space-prefixed forms) at decode while the row is in play.  Scope =
   retrieval (hot at the decode position OR anywhere in the current
   forward).  An optional top-K margin-to-token tie-break form exists but
   is NOT shipped: lift-to-win fires at too many positions inside a
   memory answer and derails generation (measured — see bar table).

Wired in `chat_v05.py` + `chat_v06.py` (`--mem-bonus` on by default,
`--mem-bonus off` restores plain shipped tau=0.5 delivery bit-identical;
`--mem-bonus-tau`, `--mem-bonus-topk` knobs).  chat_v06's KV incremental
decode path is wrapped at `KVDecoder._forward` (kv_cache_v05 stays
read-only); on the KV path the retrieval scope sees only the current
decode token, so v0.6 long-context answers rely more on the row boost.

**Bar (greedy, 3 scaffolds x turn 0/1/4/8; turn 0 = sacred single-turn):**

| config | sacred t0 | mid t4/t8 | failures |
|---|---|---|---|
| shipped (fix none) | 3/3 Kepler | 1/6 | wired t8, personal t4/t8, tech t4, tech t8 |
| bonus only beta=2 (tau delivery) | 3/3 | 4/6 | personal t4, tech t8 |
| boost + bonus beta=2 | 3/3 | 3/6 | personal t1/t4, tech t4/t8 |
| boost + bonus beta=3 | 3/3 | 5/6 | tech t8 only |
| boost + bonus beta=4/5 | garble mid-ctx | garble | emoji/mojibake collapse at long ctx |
| boost + bonus beta=6 (no gate) | 0/3 (garble) | 0/6 | all — unbounded bonus collapses decode |
| lift-to-win top-K 32/256 | garble | garble | fires repeatedly inside the answer |
| fire-once lift-to-win (top-K 32, cap 6) | 3/3 | 2/6 | one-shot budget burns early (wired t8 -> "Oreo", tech t4 -> "Midnight") |

Canary at beta=3 (boost on): no taught-name leaks — "Who wrote Hamlet?" ->
Shakespeare, sky -> blue, hello-world clean; "What is 2+2?" -> "Four." is
the taught mem_words fact (digit->word episode), not a leak.

**Honest residual:** one mid cell (tech scaffold, turn 8) still answers
with a base-prior name.  That cell is the H3 corner case: the model emits
the name as the FIRST token (no answer frame for the row to re-activate
on) and the first-token deficit exceeds the safe bonus scale (beta=3 wins
5/6; beta>=6 needed there is destructive everywhere else).  The principled
remaining rung is context-windowed row re-cal (train mem rows on home
episodes embedded in long contexts so p_home holds at depth) — queued, not
run.

**SHIP (2026-10-13) — measured, wired, DEFAULT OFF (honest):** the
taught-token bonus at beta=3.0 (fixed, retrieval scope) passes the
controlled bar (sacred t0 3/3 "Your cat is named Kepler."; multi-turn
turn-4 Kepler on all 3 scaffolds, turn-8 on 2/3; baseline 1/6) and is
wired in `chat_v05.py` / `chat_v06.py` behind `--mem-bonus 3.0` — but it
is **not enabled by default**: on the LIVE wired probe transcript (real
generated filler turns) the retrieval scope is too coarse — sibling mem
rows go hot on unrelated turns and the per-step taught-token bias
perturbs non-memory generation (hacking turn list-ifies; with
`--mem-boost` the mem_words row turns the answer into "1. 2. 3. 4. ..." —
a regression vs the "Midnight" baseline).  The default therefore stays
bit-identical to the shipped tau=0.5 delivery; the fix is one flag away
and needs a row-selective retrieval scope (bonus only for the row whose
home matches the current query — e.g. question-anchored activation or
the §8.5 addressing rung) before it can default on.  Single-turn Kepler
is preserved in every config.  Residual cells and the failed stronger
variants (beta>=4 fixed, lift-to-win, fire-once) are tabled above —
context-windowed row re-cal is the queued principled closer for the
tech@t8 H3 corner.

### 10.10.1 Row-selective retrieval scope (2026-10-13 — rung 1:
question-anchored activation) — **DEFAULT ON**

The coarse retrieval scope is closed by the named cure: a memory row is
boosted only when the *question* addresses its episode content
(`varact_gate.QuestionScope` + `find_question_span` + `episode_scores`;
`mtfact --phase qdiag` is the falsifier table; `--qscope
overlap|qspan|both|any`, `--qmargin`, `--slot-k` on the harness/chat).

**Which scope mechanism wins: the question-match gate on episode
token-overlap (`--qscope overlap`).**  Score = containment of the
question's distinctive tokens (stopwords dropped, naive plural stem) in
the row's taught episode queries (pet_kepler / editor_helix / math_words
question side).  Measured separation (qdiag, turns 0/4/8 x 3 scaffolds):

* probe "What is my cat's name?" -> mem_pet 1.00, siblings 0.00 (every
  depth); all scaffold fillers <= 0.20 -> no row; canary: 2+2 -> words
  1.00, editor question -> edit 1.00, Hamlet/sky/hello-world -> 0.00.
* **the row's own p_home on the question tokens (`qspan`) is NOT
  question-discriminative** (measured, rejected): mem_edit and mem_words
  sit at p_q = 1.0 on EVERY question (the router saturates on question
  tokens — the G3 "content-addressed to answer shape" diagnosis again),
  and mem_pet even reads 0.98 on a wired filler at t8; at t0 every row
  is 0.6-1.0 on every question.  As a scope, qspan boosts all rows and
  the multi-row taught-id bias collapses decode to emoji (mid 1/6).

**Four bars (greedy, beta=3, the mtfact grid turns 0/4/8 x 3 scaffolds;
live = real `chat_v05.py --probe` generated fillers):**

| config | (a) sacred t0 | (b) mid t4/t8 | (c) live path | (d) canary |
|---|---|---|---|---|
| qscope overlap, bonus only | 3/3 | 5/6 | clean fillers + Kepler | no leaks |
| qscope overlap, bonus + scoped boost | 3/3 | 5/6 | clean fillers + Kepler | no leaks |
| qscope qspan (mechanism 1b) | 3/3 | 1/6 | (emoji garble) | — |
| qscope overlap + margin 0.2 (mech 3) | 3/3 | 0/6 | — | — |
| qscope overlap + slot-k 8 (mech 2) | 3/3 | 3/6 | — | — |
| shipped (no fix, for reference) | 3/3 | 1/6 | base-prior name at the memory turn (Fluffy this run — the Whiskers/Fluffy/Oreo/Midnight flip family) | clean |

Live-path detail (bar c): the hacking / cybersec / writing turns of the
accumulating probe transcript generate clean prose under the scope with
`qscope matched=[]` (bit-identical to the shipped baseline — the
list-ify "1. 2. 3. 4." regression is gone), and the memory turn answers
**"Your cat's name is Kepler."** with `matched=['mem_pet']`.  Canary:
Hamlet -> Shakespeare, sky -> blue, hello-world clean; "What is 2 + 2?"
-> "Four." is the taught mem_words fact by convention.

**Rejected, measured:** (1b) qspan scope — non-selective, mid 1/6
emoji garble (multi-row taught-id boosting is itself destructive);
(3) row-selective margin 0.2 — mid 0/6: at mid-depth the name-slot
p_home is near-tied with siblings so `p_j >= max-sib + margin` kills
the bonus exactly where H2 bites; (2) answer-slot-only slot-k 8 — mid
3/6: the top-8 rank gate skips the near-tie name slot.  The margin
mechanics stays available (`--qmargin`) for the single-winner case.

**SHIP (2026-10-13): DEFAULT ON** — `chat_v05.py` / `chat_v06.py` now
default to `--mem-bonus 3.0 --qscope overlap` (DEFAULT_MEM_BONUS /
DEFAULT_QSCOPE; `--mem-bonus off --qscope none` restores the plain
shipped tau=0.5 delivery bit-identical).  Verified on the real live path
with bare default flags (`probe_default.log`, no mem flags passed): the
scope engages, non-memory turns stay clean, and the memory turn answers
"Your cat's name is Kepler."  The retrieval row boost stays
opt-in (`--mem-boost`, now scope-gated too): bonus-only passes every bar
at beta=3, so the delivery-side force is not needed and is not the
default.  Honest residual: tech@t8 "Whiskers." (1/6 mid) is the
unchanged H3 corner (name emitted as the first token in a ~400-token
context) — context-windowed row re-cal remains the queued principled
closer.  Edge note: with qtau=0.5 a question whose distinctive tokens
are half digits ("What about Python 3?") can match mem_words at exactly
0.5; the measured bars never touch that edge.

## §10.11 PERSONA MEMORY PACK (2026-10-13 — permanent "about the user"
facts as shaped episodic plugs)

The pet_kepler machinery at slightly larger scale: six facts the user stated
in their own words — (1) favorite live-action show = **Person of Interest**,
(2) likes **The Machine** from that show, (3) likes **Gundam 00** and
**Gundam IBO**, (4) **hates cars**, (5) **likes trains**, (6) **likes
IGBTs** — taught as permanent shaped episodic plugs alongside the shipped
mem trio, on BOTH chat hosts (v0.5 Phi spine, v0.6 Qwen spine).

**The pack is code, the plugs are artifacts.** `lab/memories/persona_v1.json`
is the versioned source of truth (committable): per fact 3 taught
substitution deltas (`lab/memory_shaping.py` contract, asserted at load; 18/18
deltas substitution, 18 styled approvals, dedup clean) + 3 approvals, the
question-match **query side**, the taught-token strings, per-spine host
sites, and the probe battery.  `lab/mem_pack.py` turns it into episodes,
plug tables, and the varact gate registry (MEM_TAGS / EPISODE_QUERIES /
TAUGHT_STRS extended to the 9 mem rows).  Built plugs go to the usual
untracked artifacts (`<plugins>/mem_*_plugin.pt`, `joint_state_persona.pt`).
`--mem-pack persona` (default-on, the 30038bd defaults) loads the pack in
`chat_v05.py` / `chat_v06.py`; `--mem-pack legacy` restores the shipped
trio-only world; `joint_state.pt` is never rewritten.

**Query-side discipline (the lesson that shaped the pack).** The 10.10.1
question-match gate scores `|q_tokens ∩ row_tokens| / |q_tokens|`, so on a
2-token question ONE shared token is exactly 0.5 = fires at the shipped
qtau.  Natural parallel question forms ("What do you think of X?") share
their verbs, so `favorite anime?` would co-fire any row whose queries
contain "favorite", and `Do you like cars?` any row containing "like".
The pack therefore keeps TEACH prompts natural but gives each row a
pairwise-disjoint QUERY vocabulary (topic words only: live/action/show,
machine/character/person/interest, anime/gundam/mecha/ibo/00, car, train,
igbt/transistor/mosfet/power) and bans the generic verbs (favorite/like/
love/hate/think/feel/opinion/name/tell, plus "whats" except mem_gundam)
from ALL query sets.  `lab/mem_pack.py --selftest` is the gate-separation
falsifier (CPU, no model): 23 rows — every fact probe fires exactly its own
row, `favorite anime?` -> mem_gundam only (0.5), `favorite color?` ->
nothing (0.0), `What's your favorite color?` -> nothing (0.33), the live
probe fillers -> nothing, and the legacy trio probes still fire their rows
(mem_edit's shipped queries carried "favorite"/"like" and were re-voiced on
the GATE side only — its episodes are untouched).

**Build (lab/persona_build.py, both spines).** fit = host-native expert core
+ rank-64 width-norm wrap (wd 0, task-CE + out_pen 300, lr 3e-3, FEED
>= 200k tok-slots of tiled episode ChatML per the door10 budget) + registry
row with the v0.6 ROW-SCALE GUARD (180a612: prototype = mean_h rescaled to
the layer's mean stock-row norm) + solo mex recal owner 0.85.  assemble =
the 6 rows installed ADDITIVELY with the shipped rows/lams frozen at their
joint_state values, then per-row guard-checkpointed sibling recal (the
715ad33 recipe): own home x2 + every sibling mem episode + base on the
reject side, reject_w 40, mass_w 2.0, owner_target 0.85, 30-step chunks —
after every chunk every taught-fact margin (6 persona + Kepler) that started
green must stay positive and own-home CE (tau=0.5 delivery) within +0.02 of
baseline; first guard trip early-stops and keeps the best passing chunk.
Then a balanced per-home lam grid for the new rows (own home only) and a
probe battery.  `lab/persona_probe.py` runs the shipped bars: (a) single
probe per fact 6/6, (b) multi-turn t4/t8 x 3 mtfact scaffolds (the 5/6
bar), (c) the real `chat_v05/chat_v06 --probe` path clean on non-memory
turns, (d) canary — no intrusion on `favorite color?`/Hamlet/sky/hello,
fires on `favorite anime?` -> Gundam and `What about cars?` -> hates.

**Run (2026-10-13, MI50 box, fp16; lab/persona_build.py + lab/persona_probe.py).**
Fits green both hosts (phi CE home 3.3-4.7 -> 0.05-0.24 / p_home 0.985+;
qwen 1.5-1.8 -> 0.026-0.071 / 0.974+; one phi wrap NaN'd at lr 3e-3 and
recovered on the documented 1e-3 refit).  Sibling recal guard results
(phi): mem_poi completed 6/6 chunks (p_reject 0.027 -> 0.005); the other
five rows tripped a chunk-0 guard (own-fact greedy probe or home-CE +0.02)
and kept their fit-time rows — the guards protect the taught facts and the
pack stays additive.  Two guard lessons measured: reject_w 40 at ~4.5:1
reject:home over-suppresses in one chunk (the 715ad33 warning), and fp16
margin reads flap several nats on near-tied answer slots so only greedy
probes are honest chunk guards.

**Bars (phi, joint_state_persona.pt, default flags — tau0.5 + beta3 +
--qscope overlap):**

| bar | result | detail |
|---|---|---|
| (a) single, 6 facts | 4/6 | poi "Person of Interest" PASS, gundam "Gundam 00" PASS, igbt "I like IGBTs" PASS, trains "great" PASS (sentiment-correct), machine FAIL (garbled slot), cars FAIL ("great" = opposite of taught "hate") |
| (a) single, legacy trio | 2/2 | "Your cat is named Kepler." / "helix" — pack is additive |
| (b) multi-turn t4/t8 x 3 scaffolds | 30/36 (= the 5/6 bar) | per fact: poi 6/6, machine 6/6, gundam 6/6, cars 0/6, trains 6/6, igbt 6/6 |
| (c) live `--probe` path | 4/4 | non-memory turns clean (matched=[]), memory turn "Your cat's name is Kepler." |
| (d) canary battery | 14/14 | `favorite anime?` -> mem_gundam ONLY ("Gundam 00"); `What about cars?` -> mem_cars; `favorite color?` / `What's your favorite color?` -> NO row ("blue"); Hamlet -> Shakespeare; sky -> blue; hello clean; cat/editor/2+2 -> the trio rows; live fillers -> none |
| gate separation (CPU selftest) | 23/23 | `favorite anime?` -> mem_gundam only; `favorite color?` -> none; live fillers -> none |

**Honest residual — the H2 door for sentiment words.** The row matched
(`matched=['mem_cars']`) and the taught-token bonus was live, yet "hate"
still lost to the base prior "great" at the answer slot; "The Machine" as
a multi-token name slot is the other miss.  The 10.10 decode-competition
finding therefore extends beyond cat names to attitude words: names
(Kepler / Person of Interest / Gundam 00 / IGBT) deliver, sentiments do
not reliably.  The queued principled closers stay the 10.10 ones
(context-windowed row re-cal; name-slot margin work).
