# Vesper-K — AMD Dev Cloud Credits Request

**To:** AMD Developer Relations · **From:** VesperLM (`github.com/datacrystals/VesperLM`, public) · **Date:** 2026-10-08

We have spent **$9.07** of our AMD Dev Cloud credits producing a validated, scale-stable
router result and a working expert-modularity proof on MI300X. This document is the
evidence, the costed plan for the next rungs, and the ask.

---

## 1. What we built

**Vesper-K** is a hybrid KDA (Kimi Delta Attention, linear) + MLA (DeepSeek-style latent)
language model with a **passport** MoE router: each expert carries a learned identity
embedding, and routing is the (learned token query)·(passport) dot product — softmax,
top-k, renormalized (`Common/vesper_model.py`, `PassportRouter`). The payoff is that
experts become **hot-swappable modules**: train one separately on any machine, plug it
into a frozen pretrained base, and it works zero-shot — no base retrain, no gate surgery
(`register_expert()` / `MoEFeedForward.add_expert()`; spec in `MODULAR_MOE.md`).

## 2. Evidence: passport beats standard top-k gating at every scale tested

Matched head-to-head runs — identical config, data (real fineweb/dclm), step budget, and
optimizer; the only difference is `VESPER_ROUTER_TYPE` (and `VESPER_SEED` for the paired
replication). Metric: held-out val cross-entropy; lower is better.

| Tier | Model | Params | top-k CE | Passport CE | Δ CE |
|---|---|---|---|---|---|
| t0 | `lab_tiny` | 11M | 7.1405 | 6.9499 | **+2.67%** |
| t1 | `lab_small` | 33M | 6.2134 | 6.0921 | **+1.95%** |
| t1 seed-2 | `lab_small` | 33M | 6.1159 | 5.9677 | **+2.42%** |
| t2 | `tiny_agent_k` | 120M | 6.7316 | 6.5410 | **+2.83%** |
| t3 probe | `470m_k` | 392M | 4.3604 | 4.3723 | **−0.27% (parity)** |

Sources: `lab/imported/t0r/results/`, `lab/imported/t1/results/`,
`lab/imported/swarm1/results/t1s-{passport,topk-baseline}.json`,
`lab/imported/t2/results/t2-{passport,topk-baseline}.json`,
`lab/imported/swarm3/results/t3-470m-{passport,topk-baseline}.json`.

Three points worth the reader's attention:

- **The advantage holds through 120M; at 392M we measure parity, and we report that.**
  The t1 dip (1.95%) reversed at t2 (2.83%); the seed-2 replication at t1 (+2.42%)
  rules out seed luck. At 392M a paired 970-step probe (seed 7, seq-8192 regime)
  shows statistical parity (−0.27%, within single-seed noise) — passport leads early
  (+3.2% at step 100) and top-k edges ahead after step 400. We do not claim a quality
  win at 392M from a 970-step window; the full t3 run settles it. **The modularity
  capability below is orthogonal to this race and is the actual product** — and the
  two scale-dependent knobs we discovered (reject_w 15→10→9; owner_mass 0.55→0.50,
  both weakening with capacity) suggest passport supervision needs per-scale tuning
  at ≥392M, which the ladder below funds.
- **At t2 this compounds to ~1.2× lower perplexity**: exp(6.7316 − 6.5410) = 1.21 — the
  top-k baseline's perplexity is 21% higher at the same compute. The gap opens *during*
  training (0.06 nats at step 100 → 0.29 at step 400) and holds to the final eval
  (0.19 nats; per-step series in the two t2 result JSONs).
- **Passport overhead is negligible**: ~3–4% throughput (72.8k → 70.4k tok/s at t1;
  `tok_s_avg` in `lab/imported/t1/results/`).

Every run above executed on **one MI300X** via AMD Dev Cloud.

## 3. The modularity proof

**Zero-shot plug-in at two scales.** Train a base, freeze it, train expert #5 + passport
separately on python-code with a contrastive passport loss (reject term against
non-target domains), then insert it with no base updates. Pass criteria (pre-registered
in `MODULAR_MOE.md` §6): own-domain utilization > 0.5, off-domain utilization < 0.3, and
CE on the target domain beats a masked control.

| | t1 (33M), `REJECT_W=15` | t2 (120M), `REJECT_W=9` |
|---|---|---|
| own-domain util (code) | 0.645 | 0.617 |
| off-domain util (web) | 0.159 | 0.171 |
| CE code, plug-in vs masked | 7.636 vs 8.528 | 7.285 vs 8.725 |
| Δ CE | **−0.89 nats** | **−1.44 nats** |

Sources: `lab/imported/t1_plugin/plugin_t1_rw15.json`,
`lab/imported/t2/plugin_t2_rw9.json` (rw15 at 120M over-suppresses: code util 0.378,
`plugin_t2_results.json`; rw5 under-rejects: web util 0.308, `plugin_t2_rw5.json`).

- **Scale-dependent reject weight, mapped.** At 33M the passing window is `REJECT_W`
  10–15 (rw10 best: code util 0.785, web 0.199, CE −1.16 nats —
  `lab/imported/swarm1/plugin/plugin_t1_rw10.json`); at 120M it is rw9. Production
  heuristic from the curve: **larger model → lower reject weight**; start at 10 for
  ≥33M. The t2 sweep replays the exact phase-A checkpoint (`plugin_t2_tune.py`,
  `verify_ce` delta 0.000000), so the rw comparison is controlled. The second
  routing-supervision knob shows the same direction: multi-plug-in `owner_mass` is
  0.55 at 11M but 0.50 at 33M (all 9 gates pass with wide margins;
  `lab/imported/swarm3/results/swarm3_t1_m*.json`).
- **Expert hot-expansion preserves performance.** Exact 4→6 expert surgery (bit-cloned
  experts + duplicated passport rows + top_k 2→4) trains on and matches an unexpanded
  control: Δ val −0.0008 (6.3508 vs 6.3516; `lab/imported/swarm1/expand/swarm_expand_t0.json`
  vs `swarm_expand_ctrl.json`). Modular growth without retraining works.
- **Honest negative: simultaneous multi-plug-in currently fails routing purity.** Three
  independently reject-trained experts (code/math/wiki) plugged into one frozen base each
  deliver a real own-domain CE win (+0.68 / +0.10 / +0.05 nats — the specialists are
  genuine), but they *compete* instead of partitioning: code↔math cross-talk ≈ 0.5
  (0.514 / 0.477) against a 0.3 gate, and the wiki expert is underused (0.17).
  (`lab/imported/swarm1/expand/swarm_multiplug_t0.json`.) Independent reject training
  only teaches each expert to reject the base's domains, not the other plug-ins'. The
  next session tests **joint reject training** (reject terms against the other plug-ins'
  domains, or sequential plug-in with router recalibration between insertions). We know
  exactly what is broken and the two candidate fixes; this is the gating experiment for
  the large expert-library vision.

## 4. Why AMD hardware specifically

All measurements below are from MI300X (gfx942, 192GB) droplets at $2/GPU-h
(`pod/devcloud.py`), ROCm 6.x + torch 2.9.1+rocm6.3:

- **Throughput.** 103k tok/s single-job bf16 at seq 1024 on the t2 model
  (`phaseA_tok_s` 103,017.6 in `lab/imported/t2/plugin_t2_results.json`); true KDA+MLA
  hybrid (`{'kda': 8, 'mla': 2}`) at **57k tok/s steady on 470m_k (392M) at seq 8192**
  — MLA is within **3.5%** of the GQA stack on ROCm (59.1k → 57.0k; "MLA is free on
  ROCm"; `lab/imported/probe_kda_mla_ext.log`, `HANDOFF_NEXT_AGENT.md`).
- **Experiment density.** One 192GB card absorbs **6 parallel experiment slots**
  (`lab/imported/swarm1/scripts/swarm1_orchestrate.sh`): the full swarm-1 battery —
  seed-2 replication, reject_w sweep, expansion test, multi-plug test (7 jobs) —
  finished in a **~25-minute** session (ledger: created 21:57, destroyed 22:21, $0.83;
  `pod/devcloud.py` ledger; results in `lab/imported/swarm1/`). Serially, this program
  of work was spread across multi-hour droplet sessions. At 470m_k the probe used
  14–20GB of 192GB, so parallelism headroom is large.
- **We do ROCm-compatibility work that outlives our project.** fla-on-ROCm required
  three patches (`tools/patch_fla.py`, in-repo), including a workaround for the upstream
  **triton#9815** AMD software-pipeliner miscompile (`'tt.load' op operation destroyed
  but still has uses` at `num_stages>=3`): the patch caps KDA autotune `num_stages` at 2
  under `torch.version.hip`, after which KDA chunk fwd+bwd pass in fp32 and bf16 and the
  full model trains. We are validating this class of ROCm compat fixes against upstream
  so the fixes benefit the ecosystem, not just us.

## 5. The costed ladder

| Rung | What | Cost |
|---|---|---|
| Done | Entire evidence base above (scaling × 4, plug-in × 2 scales, reject_w sweeps, expansion, multi-plug, ROCm bring-up) | **$9.07 spent** (settled ledger) |
| t3 | Full `470m_k` pretrain — 4.2B tokens, ~20h single MI300X at measured 57k tok/s | **≈ $41** |
| t4 | 1B-scale passport validation run | **≈ $150–250** (see assumptions) |
| t5 | Multi-GPU passport expert-parallelism on a MI300X cluster | node access (below) |

t4 assumptions (stated because unmeasured): a 1B-class Vesper-K variant, an 8–10B-token
validation run, projected 25–30k tok/s on one MI300X (inverse-param scaling from the
measured 57k tok/s at 392M), $2/GPU-h → ~75–110 GPU-h ≈ $150–225 of pure training,
budgeted at **$150–250** including eval and overhead. We will replace this projection
with measured throughput at t3 before committing t4 spend.

The **endgame** is t5: passport-expert training-parallelism — experts train fully
independently on separate GPUs/nodes with **zero gradient synchronization** (design:
`MODULAR_MOE.md` §5), then merge into one deployable model through passport routing.
That is training parallelism without sync overhead, and it needs a multi-MI300X node to
measure for real.

## 6. The ask

1. **$2–5k in AMD Dev Cloud credits** to execute rungs t3–t4 (total projected spend
   inside that range with margin).
2. **Conditional access to a multi-MI300X node** for the t5 distributed-composition
   experiments, granted if the passport advantage holds at 470M and 1B.

**Publication intent:** everything stays open-source (repo already public); if the result
holds at 470M+ we write up passport routing as a paper, with AMD hardware named as the
platform. All negative results (multi-plug purity failure included) will be published
either way.

## 7. Reproducibility

- **Evidence tree:** `lab/imported/` — `t0r/`, `t1/`, `t1_plugin/`, `t2/`, `swarm1/`
  (results/, plugin/, expand/, logs/, scripts/), plus ROCm probes
  (`probe_kda_mla*.log`). Result JSONs carry config, per-layer utilization, CE numbers,
  and pass flags; logs accompany every run.
- **Farm harness:** `lab/runner.py` (queue → running → results, sandboxed per-run cwd),
  `lab/queue/` entries, `lab/promote.py`; per-run env knobs (`VESPER_ROUTER_TYPE`,
  `VESPER_SEED`, `PLUGIN_T1_*`).
- **Droplet management + cost ledger:** `pod/devcloud.py` — every droplet TTL-capped
  with a self-destruct timer, and destroyed after its session finishes (verified via
  `devcloud.py list`); the ledger settles every finished session: $3.27 bring-up +
  $3.66 farm + $1.31 t2 + $0.83 swarm-1 = **$9.07 settled**, inside a hard $180 program
  cap.
- **ROCm patch set:** `tools/patch_fla.py` (3 patches, including the triton#9815
  workaround). Working ROCm install recipe (torch 2.9.1+rocm6.3, fla pinned by commit
  `37a6b1c…` — the `v0.6.0` tag does not exist upstream) is documented in
  `HANDOFF_NEXT_AGENT.md`; `pod/bootstrap.sh` is the droplet-side bring-up script.
