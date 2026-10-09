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
- **G3 — coexistence at N=16.** 16 episodic experts plugged sequentially
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
