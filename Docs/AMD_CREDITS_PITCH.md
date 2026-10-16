# VesperLM — AMD DevCloud Credits Request

**Project:** VesperLM — modular, graftable mixture-of-experts language models
**Platform:** AMD Instinct MI300X (ROCm) · **Date:** 2026-10-16

---

## 1. Summary

VesperLM is building language models that are **assembled from parts rather
than trained as monoliths**: a hybrid KDA+MLA attention spine with a
*passport* mixture-of-experts router, into which independently trained expert
modules can be plugged with no base retrain, and through which taught material
can be consolidated conversationally. On matched paired runs, passport routing
beats standard top-k gating by **+2.67% / +1.95% / +2.83%** held-out
cross-entropy at 11M / 33M / 120M parameters, replicated across seeds.
Zero-shot plug-in experts pass pre-registered routing gates at two scales;
multi-expert banks pass all nine routing-purity gates at 11M and 120M; a
teach-by-talking loop acquired **3/3** taught items on a fresh hybrid
checkpoint and rolled back a poisoned teaching batch; and a bank-ladder
protocol with pre-registered break criteria has grown a plug-in bank to
**n=128** with the failure modes measured, not guessed. Every result above was
trained and evaluated on AMD Instinct MI300X-class hardware. We are asking for
continued DevCloud credits to run the next scale tier — a full `470m_k`-class
pretrain on one MI300X — and the follow-on banked-MoE builds it feeds.

---

## 2. Architecture in brief

**Spine.** A hybrid attention stack — KDA (linear-attention) layers for cheap
long context and MLA (latent-attention) layers for quality — trained by us or
borrowed. Conversational quality and loop-freedom live here and scale with
active parameters per token.

**Passport router (the socket).** Every expert carries a learned identity row
(256-byte registry entry); routing is the learned token-query · passport dot
product, softmax / top-k / renormalized. Because the router scores *rows*, not
weights, experts become hot-swappable modules: train one separately on any
machine, insert it into a frozen pretrained base (`register_expert` /
`add_expert`), and it works zero-shot — no base retrain, no gate surgery. The
same socket reproduces donor-MoE behavior at logit-cosine 0.998+ across
multiple model families.

**Expert bank + delivery.** Experts are trained with a contrastive reject
objective (mass on the target domain, reject against non-target domains);
delivery is additive, never softmax-mass competition. Multiple plug-ins
coexist via a validated sequential-insertion protocol (router-only
recalibration between insertions, then joint calibration with a
mutual-exclusion target) that keeps expert territories disjoint.

**Self-learning layer.** Episodic memory plugs in like an expert: a
conversation loop generates the model's reply, takes a user correction, and
consolidates it into a per-user adapter through a promote/rollback canary
gate. A bank-ladder protocol scales the whole system by doublings with
pre-registered break criteria (home quality, pairwise harm, base tax, router
ownership) so each rung is claimed only when measured.

---

## 3. Measured scaling results

Matched head-to-head runs — identical config, data (real web corpora), step
budget, and optimizer; the only difference is the router type (with paired
seed replications). Metric: held-out validation cross-entropy, lower is better.

| Tier | Model | Params | top-k CE | Passport CE | Δ CE |
|---|---|---|---|---|---|
| t0 | `lab_tiny` | 11M | 7.1405 | 6.9499 | **+2.67%** |
| t1 | `lab_small` | 33M | 6.2134 | 6.0921 | **+1.95%** |
| t1, seed 2 | `lab_small` | 33M | 6.1159 | 5.9677 | **+2.42%** |
| t2 | `tiny_agent_k` | 120M | 6.7316 | 6.5410 | **+2.83%** |
| t2, seed 2 | `tiny_agent_k` | 120M | 6.6184 | 6.4341 | **+2.78%** |
| t3 probe | `470m_k` | 392M | 4.3604 | 4.3723 | −0.27% (parity) |

Three honest observations:

- **The advantage holds through 120M and reverses the t1 dip** (1.95% →
  2.83%); the seed-2 replications rule out seed luck. At t2 this compounds to
  ~1.21× perplexity: the top-k baseline's perplexity is 21% higher at the same
  compute.
- **At 392M a paired probe reads statistical parity (−0.27%) and we report
  that.** The probe window is short (970 steps); the full-length run at that
  scale is precisely what the requested credits fund. Two routing-supervision
  knobs (reject weight, owner mass) both weaken with capacity, which suggests
  per-scale calibration at ≥392M — again a compute question.
- **Passport overhead is negligible:** ~3–4% throughput (72.8k → 70.4k tok/s
  measured at t1).

---

## 4. Modular and self-learning results

**Zero-shot plug-in experts pass all pre-registered gates at two scales.**
Train a base, freeze it, train one expert plus its passport row separately on
a target domain (here: code), then insert it with no base updates. Pass
criteria: own-domain utilization > 0.5, off-domain utilization < 0.3, and
target-domain CE beats a masked control.

| | t1 (33M) | t2 (120M) |
|---|---|---|
| own-domain utilization | 0.645 | 0.617 |
| off-domain utilization | 0.159 | 0.171 |
| CE, plug-in vs masked control | 7.636 vs 8.528 | 7.285 vs 8.725 |
| Δ CE | **−0.89 nats** | **−1.44 nats** |

The optimal reject weight falls with model size (15 at 33M → 9 at 120M),
mapped by controlled replay of one phase-A checkpoint.

**Multi-plug-in banks are clean.** With three independently trained experts
inserted into one frozen base, the validated calibration protocol passes
**all 9 routing-purity gates at both 11M and 120M** — own-domain utilization
0.59–0.89, cross-talk ≤ 0.30, and per-domain CE improvement up to **+1.8
nats** over masking the owner expert. This is the validated path to an
open-ended expert library: plug-ins trained by different parties on different
hardware, with the router keeping their territories disjoint.

**Teach-by-talking self-learning (3/3 acquired, poison rollback).** A raw
pretrained hybrid checkpoint that cannot converse took three conversational
teach sessions (2 facts + 1 style), each gated and consolidated into a
per-user adapter. All 3/3 taught items moved in the taught direction
(teacher-forced margins: +1.73→+1.98, −3.07→−1.34, −1.57→−0.51), earlier
items retained across later teaching, and a poisoned "answer X to everything"
batch was **rolled back** by the promote/rollback gate with the incumbent
adapter byte-identical afterwards.

**Quality-chat SFT rounds.** A first quality round (curated multi-turn mix)
reached **6/12 usable replies** on a fixed 12-prompt conversational
instrument — the first talkable checkpoint in the program. The follow-on
quality mix (28,600 hard-filtered dialogues, ~23M token positions) eliminated
both structural failure modes at its first training milestone: hard loops
**0/12** (was 2/12) and formula-debris drift **0/12** (was 3/12), with all
twelve replies reading as fluent conversation.

---

## 5. Roadmap — what more compute unlocks

1. **Full `470m_k`-class pretrain on one MI300X (the t3 scale).** The
   392M-parameter hybrid (KDA×8 + MLA×2, passport MoE) is the tier where the
   short probe reads router parity; a full-length run settles the passport
   vs top-k question at that scale and yields the next spine checkpoint for
   the SFT and self-learning lines. Measured MI300X throughput supports this
   on a single card: ~57k tok/s steady at 8k context for the hybrid stack,
   with MLA within **3.5%** of the GQA stack on ROCm.
2. **Banked-MoE follow-ons (the bank ladder).** Growth rows measured at
   n=32 / 64 / 128 plugs with pre-registered break criteria: base tax and
   pairwise harm held at every size (0.0056 and 0.0008 at n=128), while n=128
   is the first size to break the home-quality bar — the measured signature
   of router-capacity saturation. The next rung grows past n=128 on a
   larger-capacity spine router, toward the ladder's larger banked builds
   (hundreds of frozen expert cores behind one socket).
3. **Scale the ladder and publish.** Every rung is claimed only when measured;
   if the passport advantage holds at 470M and beyond, we write up passport
   routing as a paper with AMD hardware named as the platform. Negative
   results are published either way.

---

## 6. The ask

**Additional AMD DevCloud credits** to execute the roadmap above on
single-MI300X nodes:

1. the full `470m_k`-class pretrain (t3 scale) and its paired router
   comparison — one MI300X, a single sustained run;
2. follow-on runs for the banked-MoE ladder: expert-bank growth past n=128
   with the larger-capacity spine router, and the multi-expert bank builds
   above it.

The work is open-source, the results (positive and negative) are published as
they land, and the program is structured as a measured scaling ladder — the
kind of usage that turns credits into a durable public result rather than a
one-off demo.

*Project: VesperLM (`github.com/datacrystals/VesperLM`). All figures above are
measured values from the project's paired-run evidence tree and experiment
logs; raw result records accompany every run.*
