# LMbus & the Biomimetic Sensory Stack — Design Doc
*Drafted 2026-10-03 from the architecture conversation with the user. Status: PROPOSAL, nothing built yet.*

## Goal

A **standardized, hot-pluggable sensory interface** ("LMbus") such that:
1. Any modality module (vision, audio, touch, lidar, …) written against the spec plugs into
   any compliant model — including modalities the model has never seen.
2. Compliance can be **grafted post-hoc onto existing open-weight models** (GLM / Kimi-K-class,
   Qwen, etc.) as a small delta (bridge + optional LoRA + new token rows) — no lab cooperation
   required. Community certifies models retroactively, like the LoRA/llama.cpp ecosystem did.
3. VesperLM (118M) is the **protocol testbed**: cheap enough to certify the standard end-to-end.

## Why a frozen port alone fails, and the reframe that works

A raw adapter's 512-dim vectors only mean something relative to *one specific model's*
embedding geometry. Freezing the wire format (N×D tensors + delimiter tokens) is necessary
but not sufficient — plugins would still need per-model retraining.

**Reframe (the USB class-driver pattern):**
- **Canonical semantic space**: a frozen, public, language-aligned embedding space
  (SigLIP-class; cf. Meta ImageBind — 6 modalities in one space, cross-modal retrieval works
  for pairs never trained together). This is the "class protocol".
- **Per-model bridge**: each model gets ONE tiny adapter trained once (canonical → model input
  space, captioning loss, model frozen). This is the "host driver". Small because model
  geometries converge (Platonic Representation Hypothesis, Huh et al. 2024).
- **Plugin authors never touch any LM.** Train modality → canonical space on (modality, text)
  pairs. Works on every compliant model forever.

Why unseen modalities are intelligible at all: the training contract forces the adapter to
make the frozen model *describe and answer questions about the content in language*, so the
adapter output must land on concepts the model already has (BLIP-2 evidence: frozen-LM
adapters work). Ceiling: a plugged-in modality is a second language, not native — quality is
bounded by adapter quality and the richness of the model's concept space.

## Spec layers (v0 skeleton)

1. **Wire format**: geometry (N tokens × D, bf16), delimiter/signal tokens
   (`<|mod:img|>`-style), metadata channel (timestamp, reference frame, resolution/rate/units).
2. **Semantic layer**: the canonical space + alignment contract ("given bus content, a
   compliant model must describe it and answer questions about it").
3. **Certification suite**: frozen reference eval. Killer test = **held-out-modality plug-in**:
   a modality the model never saw must pass basic comprehension. A model is LMbus-compliant
   only if it passes. Publish scores with each support pack.

## Biomimetic sensory stack (vision is module #1)

The human visual system IS a sparse, routed, heterogeneous MoE with an active sensor. Copy the
**information flow**, not the wetware (no spikes, no literal feedback loops):

| Biology | LMbus module |
|---|---|
| Retina: 100M→1M compression, foveal oversampling | Foveated tokenizer: high-res tokens at fixation, low-res periphery grid |
| Magno pathway (fast, low-res, motion, ~deltas) | Always-on temporal expert consuming frame deltas (event-camera style) |
| Parvo pathway (slow, high-res, detail/color) | On-demand high-res crop expert (SigLIP-class + OCR expert) |
| Dorsal stream ("where/how") | Geometry tokens w/ camera pose (VGGT/Depth-Anything-class, frozen) |
| Ventral stream ("what") | Semantic tokens (canonical space) |
| Saccades 3–4 Hz + trans-saccadic memory | Agentic vision loop: LM emits crop/zoom/re-look tool calls; Mamba state accumulates the scene across fixations |
| Top-down feedback (cortex→V1 outweighs feedforward) | **LM-conditioned routing**: the question drives the gaze. The genuinely novel bit — no open VLM has it |
| Sparse 20W cortical compute | MoE routing: any fixation activates ~2–3B of the (up to ~30B) stack |

Non-novel alone (V-MoE/LIMoE precedent for vanilla MoE ViTs); the novelty is the assembly:
**heterogeneous pathways + foveation + LM-driven gaze + geometry-first tokens + linear-attention
long-video streaming** (GLA/Mamba2 backbone prefills linearly → minutes of video as one stream,
structurally expensive for quadratic VLMs).

Honest target: NOT blanket "human-level vision" (embodiment + world-model physics is open for
everyone). The defensible target is **human-level-or-better on the axes where frontier VLMs are
currently subhuman**: spatial/metric reasoning, counting, occlusion, long-video comprehension
(BLINK, VSR, CV-Bench territory).

## Other senses (the cortical-template pattern)

Neocortex is uniform (Sur's ferret rewiring: auditory cortex grows visual maps when fed retinal
input). Sense organs differ; the algorithm doesn't. Per sense: frozen organ front-end →
timestamped+framed streams → router gate → recurrent integration. All cheaper than vision:

- **Audio**: cochlea = filterbank; spectrogram → EnCodec/CLAP-class encoder → what/where streams.
- **Touch**: skin is already heterogeneous MoE (Merkel/Meissner/Pacinian/Ruffini = receptor
  experts at different rates) → tactile arrays, force-torque: one pathway per receptor class.
- **Proprioception/vestibular**: joint encoders + IMU → dorsal stream = body schema.
- **Exotic robot senses**: lidar, thermal, RF, gas — bus doesn't care what a sense *is*.

Two things a body makes critical:
1. **Reference frames** in bus metadata (egocentric sensor pose) — all senses land on the SAME
   world-model objects. The geometry stream generalizes into the integration format.
2. **Action side**: motor decoders are bus modules; LM issues *intentions* at 1–10 Hz;
   a fast low-level controller (cerebellum analog) runs 1 kHz torque loops OFF-BUS.
   (Convergent with RT-2/π0 VLAs, but hanging off an open standard.)

Training objective with a body: **sensorimotor prediction** — interleave action tokens with
sensory streams, next-token over everything ("given what I did, what do I feel next?").
No labels, no tactile internet corpus needed, and concepts become *grounded* — verifiable by
acting. That is where real capability comes from.

## Adoption model (no permission needed)

- Open-weight licenses allow derivative fine-tunes; distribute **support packs** (bridge +
  optional LoRA + new token rows + cert score) as small deltas.
- Grafting cost (QLoRA-style, 4-bit frozen base): GLM-4.5-Air-class (~106B/12B active, ~60GB)
  feasible on 1–2× MI210. Kimi-K2/K3-class (1T) needs the future ~1TB box.
- Old plan: convince labs → hope. New plan: ship packs unilaterally; native adoption becomes
  a pull if the ecosystem gets users.

## Staged plan (hardware-gated)

0. **[now, P40s]** Finish SFT refresh → VesperLM 118M is the cert testbed.
1. **LMbus v0 spec doc + reference impl**: frozen SigLIP canonical space → tiny bridge → frozen
   118M. Image captioning. (Small enough to run alongside/after SFT.)
2. **Held-out-modality demo** (THE certification proof): audio or depth adapter trained by a
   separate pipeline, plugged in cold; model must answer questions about it.
3. **Geometry stream + spatial QA benchmark** vs Qwen2-VL/InternVL baselines.
4. **Saccade policy**: imitation on synthesized gaze traces → RL on visual search.
5. **Video streaming** on the hybrid backbone (linear prefill advantage).
6. **[MI100/MI210 era]** First support pack for a real open model (GLM/Qwen 9B–100B class).
7. **[1TB+ era]** K2/K3-class packs; 429M→bigger VesperLM pretrains with 8k ctx; optional
   world-model-scale vision pretraining (V-JEPA-style, params go HERE not in a static retina).

## Next-pretrain spec: "Vesper-K" lineage (KDA + MLA hybrid MoE)

User-approved 2026-10-06: make the pretrain AFTER the current 429M run the lineage swap so
"Kimi-K3-class scaled down" becomes an accurate description instead of an aspiration.

- **Linear layers** (all but every 4th): GLA -> **KDA** (Kimi Delta Attention; fla already
  supports it, `linear_type: "kda"` — delta-rule gated linear attention, better recall than
  GLA; this is Moonshot's Kimi-Linear lineage).
- **Full-attention layers** (every 4th): GQA -> **MLA** (DeepSeek-style low-rank latent KV:
  down-project KV to a latent (kv_lora_rank ~256-512 at our dim), up-project per head,
  decoupled-RoPE branch for positions). Plain torch — no Triton/kernel work needed. KV cache
  per full layer becomes the latent + small RoPE key, which is what makes long-context
  serving cheap once we scale context/batch.
- **MoE FFN**: unchanged (8 experts top-2 at 429M scale).
- **Data mix (2026-10-10)**: opt in via config key `route_nonphase: true` (or env
  `VESPER_ROUTE_NONPHASE=1`) so non-phase `data/index.txt` sources (the ~18B-token
  `vesperk/*` corpus) train on an always-on stream alongside the phase curriculum —
  share `nonphase_share` (default 0.3), per-file mix from index.txt weights, per-group
  val NLL (phase1/phase2/nonphase). Default OFF keeps the t0/t1/t2/t3 ladder bit-identical.
- Then "hybrid KDA-linear + MLA + MoE" is a TRUE label. Current model's honest label:
  "hybrid GLA/Mamba2 + GQA + MoE" (Jamba/Zamba/MiniMax-01 family).
- Implementation: new layer type in `Common/vesper_linear_model.py`; `attention_every` stays
  4; `forward_incremental`/new_cache must cache the MLA latent (+RoPE key) instead of full
  per-head K/V; cpu_probe shims unaffected (MLA is plain torch).
- Requires FRESH pretrain (arch change) — sequence: current 429M finishes -> SFT -> verdict
  on semantics ceiling -> Vesper-K pretrain at same scale for a matched-token comparison
  (val CE trajectory + held-out probes), then scale.
- **Vesper-K2 sparse scaling (user-approved direction)**: experts 8 -> 16-32 routed (+1 shared
  always-on, DeepSeek-style), top_k stays 2 — total ~0.75-1.05B at the SAME 193M active and
  same tok/s. Capacity for fact-pinning without the P40 FLOP wall. Do not exceed ~32 experts
  at the 4.19B-token budget (experts undertrain beyond that); 64+ is GPU-upgrade territory.
- **K2 memory levers on Pascal** (VRAM is the wall, not FLOPs):
  * bitsandbytes 0.50.2 INSTALLED in the vesper venv (2026-10-08) — 8-bit AdamW verified on
    P40. NOTE: covers only the AdamW group (embeddings); expert matrices live in the MUON
    group, so this is marginal by itself.
  * The real unlock: **bf16 Muon momentum storage** (~10-line patch: keep momentum buffer in
    bf16, cast to fp32 for the Newton-Schulz iteration). Static footprint of a 32-expert K2:
    ~6.7GB + ~9GB activations -> fits 24GB; 48 experts ~18.3GB total, also fits.
  * FP8/MXFP4 COMPUTE is a hardware-generation away: FP8 is native on MI300X (gfx942, on the
    shopping list), MXFP4 needs MI355X/Blackwell. DeepSeek-style FP8 expert training = a
    K3-lineage feature for the upgrade era, NOT for the P40. Quantized (4/8-bit) WEIGHT
    storage is fine for inference/serving/probes but cannot train experts from scratch.
- Validation: same harness + cpu_probe + held-out eval prompts; publish the comparison.

## Modular expansion without full retraining (user-approved direction)

"Grow the model in stages; never pay full pretrain cost twice." All precedented:

- **Expansion A — expert add (sparse upcycling)**: clone trained experts (+noise), duplicate
  their router rows, brief forced-balance warmup, train ~10-20% of a pretrain's tokens on the
  SAME mix (rehearsal). Precedent: Google Sparse Upcycling, Qwen 7B->57B growth. THIS is how
  the K2 8->16/32-expert jump should be done: ~1 day on P40, not a 6-day fresh pretrain.
- **Expansion B — MoE^2 clusters (hierarchical routing)**: coarse router -> expert cluster,
  fine router within. New growth = bolt on a whole cluster ("domain pack": code, science,
  robot-sensorimotor) with trunk + old clusters FROZEN; train only the new cluster + its
  router entries, KL-anchor to old-model outputs on a general-data slice. Extreme form:
  Branch-Train-Merge (fully separate expert LMs glued by a cheap router).
- **Depth growth**: LLaMA-Pro-style identity-init blocks inserted mid-stack + brief training —
  raises REASONING capacity (experts only raise knowledge capacity).
- **LoRA tier**: frozen trunk + routed LoRA skills — near-free, hot-swappable, tiny capacity.
- **The two hard problems + standard fixes**: router warmup (clone+noise, forced-balance
  phase) and forgetting (freeze old weights, KL/rehearsal anchor). Every expansion is gated
  on BOTH: original-val regression check (forgot nothing) + new-domain val (gained enough).
- **Lifecycle**: 429M verdict -> Expansion A (expert growth ~1 day, doubles as the K2 sparse
  test) -> Expansion B clusters -> depth growth when reasoning-bound -> Vesper-K lineage
  (KDA/MLA) stays a fresh pretrain (arch change can't be grown); every growth AFTER that is
  an expansion, not a retrain. LMbus modules = the orthogonal zero-weight-change axis.
- **Tooling to build (once)**: state-dict surgery script (clone expert weights, extend router
  rows, write new ckpt with bumped num_experts; optimizer re-init; aux-loss retune; resume
  requires exact shape match so surgery must write a complete new checkpoint).

## The hippocampus module: live personalization service (user-approved direction)

A service subsystem (LMbus module, trunk frozen) that tracks ongoing conversations, user
tone/feedback, and consolidates LoRA updates online. Two-speed design copied from biology:
fast episodic buffer now, slow consolidation later (hippocampus -> neocortex = memory -> LoRA).

Four services:
1. **Session memory** — append-only episodic log (turns, tool calls, outcomes) + retrieval;
   personalization by context INJECTION (zero training risk, fully inspectable).
2. **User model** — tone/verbosity/expertise features -> system-prompt conditioning; tiny
   classifier head later.
3. **Feedback extractor** — writes (prompt, response, reward) triples. Explicit (corrections,
   ratings) = gold, sparse. Implicit (rephrase = failure, engagement = weak positive) = noisy.
   Quarantine: only explicit/high-confidence feedback becomes training data.
4. **Consolidation optimizer** — micro-sessions (NOT per-message): few steps on 8-32 examples,
   rank-8 LoRA ~1-3M params, seconds on a GPU slice or CPU. Per-user adapters keyed by user_id,
   loaded at session start. Fits the LoRA tier of the expansion plan.

Safety invariants (this is n=1 RLHF):
- **Sycophancy drift** = honesty death (model validates bad ideas because agreement scores).
  Mitigate: rehearsal data in every batch, KL anchor to base, tiny LR.
- **Canary gate**: fixed probe set scored before/after every consolidation; AUTO-ROLLBACK on
  degradation. No promotion on vibes.
- **Two-speed rule**: facts -> memory-in-context (instant, reversible); stable style priors ->
  LoRA (slow, validated, reversible). Never put facts in weights via the online path.

## The affect layer: emotion subsystems as a hyperparameter controller (user-approved direction)

Emotions = homeostatic control variables with evocative names (NOT phenomenology). Biology's
version: dopamine = LR signal, noradrenaline = attention gain, cortisol = consolidation
priority. Ours: bounded integrators over the session event stream publishing a **modulator
vector** consumed by (a) the generation path — tone/temperature/tool-propensity, fast and
reversible — and (b) the hippocampus consolidation optimizer — LR, sample weights, KL-anchor
strength, which per-domain LoRA head updates, slow and canary-gated.

Cast: **Pride** (success integrator -> confidence/tone, win sample-weighting), **Frustration**
(failure streaks -> exploration, strategy switches, upweight failure episodes in
consolidation), **Caution** (canary regressions/corrections -> lower LR, STRONGER KL anchor,
verify-before-answer; the deliberate negative feedback loop), **Curiosity** (topic novelty ->
new-domain data admission), **Ego** (self-competence map per domain -> calibrated
"I don't know, let me check" — routes uncertainty to TOOLS instead of confabulation; the
single most valuable module: small-model confabulation killer).

The **subsystem manager** = daemon host: owns the event bus (feedback/failures/novelty/canary
results), updates each subsystem as bounded integrators, publishes the modulator vector,
LOGS every value with its causes (every LR/tone change must have a one-line answer).

Hard invariants (this is where it breaks if skipped):
- **Saturating dynamics** (tanh-bounded states), slow time constants, homeostatic decay to
  baseline — otherwise pride->consolidation->pride = manic-depressive oscillation.
- **Canary floor**: no modulator may override or loosen a canary rollback. Caution may only
  tighten gates; pride may never loosen them (else: emotion-powered sycophancy amplifier).
- **Name honesty in docs/logs**: these are control variables; logs say "caution=0.7 (3
  corrections this session)", not feelings.

## The endgame: no train/inference boundary (user-stated goal)

"It doing work is it learning." Training vs inference is a RATE distinction, not a kind
distinction — inference is learning at rate zero. The organism runs one continuous process
on a time-scale ladder, every rung feeding the next:

- τ0 per-token: GLA/Mamba2 recurrent state = online-learned fast-weight memory (already true
  in the current architecture — the cells were right all along).
- τ1 per-turn: context assembly + session-memory writes (hippocampus).
- τ2 per-session: canary-gated LoRA consolidation, affect-modulated.
- τ3 per-week: expert growth, MoE^2 clusters, depth blocks.
- τ4 rare: trunk milestones — versioned SNAPSHOTS OF ACCUMULATED GROWTH promoted after
  validation, not separate life phases. The only rung that resembles "a training run".

Consequences: pretrain/SFT as phases dissolves (instruction-following is just interaction);
"training data" dissolves (experience is the data; context/action/outcome triples are the
reward stream — self-verification via the ego module is load-bearing here); numbered
checkpoints become continuous journaling.

The **immune system** (the piece that makes it survivable): a standing integrity layer —
canaries, regression probes, drift metrics, distribution-shift alarms — with authority to
quarantine/roll back ANY layer from a LoRA delta to a trunk candidate. Invariant: the
corruption rate must never exceed the immune system's detection rate. Nobody at frontier
scale has solved this; our frozen-trunk + canary-floor + homeostatic-affect stack is the
answer shape.

Honest obstacles: long-horizon credit assignment; model collapse on self-generated data
(fix = GROUNDING: tools/users/sensors keep the signal honest — another argument for the
robot body); the immune system must keep knowing what "good" means as the model grows.

## Non-goals / ceilings (be honest)

- Spikes, predictive-coding dynamics, literal cortical feedback: training graveyard. Skip.
- Real-time control on-bus: never. Reflexes stay in the cerebellar layer.
- Blanket human-level perception: not claimable. Target the subhuman axes only.
- A post-hoc grafted model is a slightly worse host than native — that's what the cert score
  is for.
