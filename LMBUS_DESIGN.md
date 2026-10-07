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
- Then "hybrid KDA-linear + MLA + MoE" is a TRUE label. Current model's honest label:
  "hybrid GLA/Mamba2 + GQA + MoE" (Jamba/Zamba/MiniMax-01 family).
- Implementation: new layer type in `Common/vesper_linear_model.py`; `attention_every` stays
  4; `forward_incremental`/new_cache must cache the MLA latent (+RoPE key) instead of full
  per-head K/V; cpu_probe shims unaffected (MLA is plain torch).
- Requires FRESH pretrain (arch change) — sequence: current 429M finishes -> SFT -> verdict
  on semantics ceiling -> Vesper-K pretrain at same scale for a matched-token comparison
  (val CE trajectory + held-out probes), then scale.
- Validation: same harness + cpu_probe + held-out eval prompts; publish the comparison.

## Non-goals / ceilings (be honest)

- Spikes, predictive-coding dynamics, literal cortical feedback: training graveyard. Skip.
- Real-time control on-bus: never. Reflexes stay in the cerebellar layer.
- Blanket human-level perception: not claimable. Target the subhuman axes only.
- A post-hoc grafted model is a slightly worse host than native — that's what the cert score
  is for.
