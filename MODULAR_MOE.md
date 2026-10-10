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

### 9.7 Harvest granularity: co-adaptation clusters

Donor experts co-adapt under their router — each is shaped to complement its
siblings (expert 5 handles what expert 3 leaves behind). Harvesting singletons
risks orphaning that function. The D0 profiling pass already logs per-token
top-k sets, so measure the co-activation graph and harvest in CLUSTERS
(frequently co-selected experts travel together, sharing one plug-in recal).
Graft order within a cluster follows the donor router's own activation
margins. Singleton vs cluster fidelity is measured at D1.5 — if singletons
hold >0.9 cosine anyway, skip the clustering complexity.
