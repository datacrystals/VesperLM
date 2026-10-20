# VesperLM

**Modular, graftable mixture-of-experts language models — a research log.**

VesperLM is an experiment program around one question: *can a language model be
assembled from parts — a spine, a standard expert interface, and a bank of
expert cores grabbed from other models — and still behave like a coherent,
capable, conversational model?* This README is the curated public experiment
log. Design specs live in `MODULAR_MOE.md` / `SUBSYSTEMS.md`; the full
operational ledger (including failures) lives in `lab/FAILURES.md`.

---

## 1. The architecture

```
        tokens
          │
   ┌──────▼──────────────────────────────┐
   │  SPINE (the brain)                  │   embeddings, attention, norms,
   │  trained by us or borrowed          │   router, optional native experts
   └──────┬──────────────────────────────┘
          │ routing decisions via…
   ┌──────▼──────────────────────────────┐
   │  PASSPORT / REGISTRY (the socket)   │   one 256-byte row per expert;
   │  family-universal routing interface │   router reads rows, not weights
   └──────┬──────────────────────────────┘
          │ delivers to…
   ┌──────▼──────────────────────────────┐
   │  EXPERT BANK (the parts bin)        │   frozen donor cores + thin
   │  [core₁]◄── wrap: ~0.8M adapters    │   trainable adapter bridges;
   │  [core₂]◄── frozen forever          │   additive λ·m·e(h) delivery
   └─────────────────────────────────────┘
```

Three separable layers:

1. **Spine.** Embedding LUT → transformer stack → unembedding. Conversational
   ability, reasoning, and loop-freedom live here and scale with *active*
   parameters per token.
2. **Passport/registry.** The routing standard. Each expert has a learned
   256-byte row; the router scores rows. Proven family-universal (three donor
   families so far).
3. **Expert bank.** Frozen expert cores from arbitrary donors, wrapped in
   thin learned adapters (`host → 64-dim bridge → frozen core → back`) that
   translate between the host basis and the donor basis. Training cost per
   expert: ~0.8M adapter params, ~20 minutes. Delivery is additive
   (`y = y_stock + λ·m·e(h)`), never softmax-mass competition.

On top: **episodic memory as experts** (taught facts stored as shaped
plug-ins, the §8 line) and a **subsystems layer** (drives/emotions telemetry,
`SUBSYSTEMS.md`).

---

## 2. The standard recipe (frozen 2026-10-13)

After a week of falsification, the surviving plug-in recipe is:

| step | what | cost |
|---|---|---|
| 1. GRAB | frozen donor expert core, any family | zero training |
| 2. WRAP | rank-64 adapters, width-normalized init, wd 0, lr 3e-3, task-CE | ~20 min |
| 3. FEED | ≥200k specialty tokens (certified budget) | data collection |
| 4. DELIVER | additive λ (fitted per-home), not mass stacking | minutes |
| 5. ROUTE | registry row (256B) + mex recal, owner_mass 0.85 | minutes |

**Certified:** ≥200k calibration kills memorization harm (437× gap collapse;
10–12k budgets are 20× under and *harmful* out-of-sample).
**Not certified:** held-out skill transfer — data buys neutrality, not
generalization. Deep donor cores carry real skill in-distribution; out-of-
distribution transfer is the open question (§6).

---

## 3. Experiment log

All numbers are measured on AMD Instinct MI50/MI300X-class hardware unless
noted. "Cos" = cosine similarity of outputs vs the reference model.

### 3.1 Memory-as-experts (the §8 line)

| gate | result | verdict |
|---|---|---|
| G1 expert transplant (TopK) | contamination 0.90 | FAIL → pivot |
| G1b passport-native recipe | margins +4.59 (6.8× baseline) | PASS |
| G3 coexistence N=8, mixed episodes | retention 0.39 | FAIL |
| G3 coexistence N=8, token-substitution shape | retention **0.884** | PASS |
| G3 coexistence **N=16** shaped | retention **0.737** (bar 0.70) | PASS — first N=16 pass |
| orthogonality rung | forced \|cos\| 0.014 changed nothing | closed |
| spine-portability (owner_mass 0.55→0.85) | 0.397 → **1.100** on real spine | PASS |

**Finding:** memory *shape* is the lever (token-substitution deltas retain
3× better than mixed episodes); owner-mass calibration controls delivery;
retention is flat in N once calibrated.

### 3.2 The graft line — the socket and the grab

| experiment | headline | verdict |
|---|---|---|
| **D0** graft parity (OLMoE-1B-7B) | logit cos **0.99913**, ΔCE +0.00075 | PASS (stretch) |
| **D1** plug-in onto graft (scratch delta) | parity/contam/util pass; quality fails | formal PASS, quality open |
| **D1.5** feature translators (cross-model hidden spaces) | held-out 0.55–0.69 vs 0.9 bar | **FAIL — dead** |
| **D2** cross-family (Phi-mini-MoE, foreign router) | logit cos **0.99863**, ΔCE 0.030 | PASS (stretch) |
| passport sweep | parity saturates at d=gate rank (64); algebraic passports pass all gates for linear-gate donors | — |
| margin-aware recal | clean negative; agreement must be ceiling-normalized | — |

**Findings:** (1) the routing socket generalizes across families; (2) feature-
space transplant between existing models is capped at ~0.5 linear alignment —
confirmed with exact token-paired supervision (Door 9); (3) registry economics:
~8MB spine artifact + 256B/expert row; recal is mandatory; algebraic passports
are free where the donor gate is linear.

### 3.3 Door-series: what makes a plug-in work

| door | question | result |
|---|---|---|
| **2.5** wrap a frozen donor core | can foreign weights host verbatim? | **PASS** — quality 0.439 vs 1.072 masked (first quality pass) |
| **3** KL-distillation of a donor skill | can we teach a core instead of grabbing? | skill transfers (KL 5.17→2.59) but no CE skill — FAIL |
| **4** delivery contract | mass competition vs additive λ? | additive wins 3/5; delivery was not the binding constraint |
| **5** provenance | what makes a core carry skill? | **CE-task-pretrained cores only** — scratch/refit die (3 contracts × 6 recipes); KL cores fail all treatments; CE-distill reaches in-sample only |
| **6** mix two donors | do grafted experts interfere? | **sum-of-parts exactly** (0.9957×), zero cross-contamination |
| **7** CE-objective distill | is the loss the problem? | no — training breadth is (47k tokens); anti-skill from cross-tokenizer argmax (15.6% agreement) |
| **8b–e** bridge width & bank substitution | can cores carry a whole layer? | bridge not the limit (rank-64 over-recovers); **whole-layer substitution saturates at ~58%** in all modes; per-slot solo = 2625% |
| **9** embedding-anchored translation | was D1.5 a pairing artifact? | no — exact pairing still ~0.47 held-out; W-init doesn't beat random |
| **10** calibration scale | is the memorization gap data starvation? | **YES** — gap 1.35→0.003 (437×) by 200k tokens; hyperbolic, saturating |

### 3.4 The bank: partitioned expert banks (the n-from-A + n-from-B form)

8 wrapped cores from 2 donor families as first-class router rows (bank2n):

| measurement | result |
|---|---|
| domain ownership | p_home 0.98–0.995 for all rows |
| code domain, two donors | **splits** across families (0.28/0.21) and composes better than either solo |
| in-bank solo-recovery preservation | 70–94% for strong grafts |
| base partition tax | **−4.3%** (bank slightly improves base) |
| pairwise harm | 0.226 worst (bar 0.02) — crowding; cross-fire rejection shrinks it ~10× |
| partition vs overlay delivery | overlay-λ wins on held (ships in v0.5+), |

**Verdict: GO with conditions** (core selection by lift ≳10; in-env wrap
training; cross-fire rejection). The form works; generalization is the open rung.

### 3.5 Assembly demos

| build | what | status |
|---|---|---|
| **v0** (2026-10-12) | Phi-mini-MoE-instruct spine + 2 donor specialties + 3 taught memories | converses (cos 0.986 vs bare); specialties 2/2 visible; memory margins flip (Kepler-vs-Orion −2.76→+0.95) but fact token doesn't surface |
| **v0.5** (2026-10-13) | + K3 / GLM-5.3-Flash / MiMo-V2.6-Flash cores (real extracted weights, dequant oracle-verified exact), 8 plugs total | live chat CLI; K3 hacking shows; all 4 specialties visible after v0.5.1 delivery fix |
| **v0.5.1** | balanced per-home λ + cross-fire rejection | 6/8 net-green; flooding 0.028→0.0004 |
| **v0.2 fast** | KV-cache incremental decode | 2.2 → **8.7 tok/s**; bare-spine bit-identical parity |
| **v0.6** (2026-10-13) | Qwen1.5-MoE-A2.7B-Chat spine (2.7B active), full stack ported unmodified | parity 0.9989; **graft question answered coherently in greedy — no loop** |

Demo note: at 1.8B active params, out-of-distribution self-referential prompts
loop in greedy decode (a small-model prior artifact, present in bare and
grafted alike). At 2.7B active the same prompt answers coherently — looping
tracks *active* parameters, as expected.

### 3.6 Training-infrastructure findings (the boring ones that mattered)

- **End-of-loop vanish** (reproduced twice): exclusive loop bounds + no
  end-of-loop save path → trainers "disappeared" at the final step having
  saved nothing. Fixed: terminal-step save + completion banner + emergency
  checkpoint-on-exception.
- **Frozen-val-metric trap**: a phase-switched curriculum with a static val
  mixture makes val stop tracking training (looked like a plateau/overfit).
  Fix: per-group val logging + val mixture matching the current train mix.
- **Silent data drops**: name-matched phase bucketing discarded an 18B-token
  corpus without error. Fix: routed non-phase streams + index-weighted mix,
  bit-identical when disabled.
- **Numerous platform findings** on gfx906-class devices (spine-specific
  backward holes at certain batch shapes, sorted-merge kernel boundaries,
  quantized-core dequant exactness) — all documented in `lab/FAILURES.md`.
- **Reproducible corpus regen** (cosmopedia, 2026-10-16): 1.5B tokens
  regenerated as `vesperk/cosmopedia_{0,1}.bin` (the prep script shards at
  1e9 tokens → 1.0B + 0.5B), **byte-identical** to the prior build
  (sha256-verified); index rebuilt via `pod/rebuild_index.sh`. Note for
  future agents: `rebuild_index.sh` writes `Dataset/data/index.txt` from
  `pod/MANIFEST` with all present sources (not the old hand-written
  Nemotron-only index); `index.txt` is uncommitted runtime state per policy.
- **Bank measures are host-portable** (2026-10-19, R2/R4): the rung-1 growth
  windows reproduce gfx906 → sm_86 within ≤0.003 CE on identical feeds (the
  n=128 worst breaker within ±0.0006) — laptop-measured growth rows are
  load-bearing. A 250-core bank that cannot park its cores in RAM/VRAM
  measures fine via packed-cores-in-RAM (~4.2GB for 250) + per-host-block
  device dequant staging around each block forward (~10 s per 6-batch
  window on an 8GB card; device dequant asserted bit-exact vs the CPU
  oracle). Companion trap: growth's p_home bucket reads fit records via
  glob-dedup that drops restart lanes (`{"skipped": true}` rewrites hide
  `row_recal`) — it saw only 83/128 rows at n=128.
- **State-wipe trap in run-state installers** (2026-10-19, R5): mutating
  per-run flags (`pm.active[...] = False`) BEFORE `run_in_state(..., "both")`
  is silently wiped by its `set_mix_active("both")` — apply run-time state
  inside the run lambda. The growth pairwise-harm protocol measured harness
  noise for a week (laptop floor |ΔCE| = 0.000000, box ±0.0008); repaired
  protocol in `lab/reject_recal.py` / `lab/bank_rung1.py`. Same class as the
  `pgrep -f` self-match trap: the flag you set is not the flag that runs.
- **Sequential-decode economics for big banks on small cards** (2026-10-19,
  R6): greedy generation through the 250-core bank restages cores per token
  (~5–8 s/tok on the laptop) — CE windows are batched and cheap, generation
  is not; budget battery sizes accordingly. Generation through the staged
  wraps also needs `torch.inference_mode()`: autograd through a 250-core
  token step exhausts an 8GB card (br1's `greedy_gen` predates the big
  bank).
- **Held-out bounds bookkeeping** (2026-10-19, R7): rung-1 wrap fits
  consumed exactly feed batches 0..781 (first 200,192 tokens;
  `steps_eff = max(500, 200000/256)` sequential, no wrap-around), recal
  pairs ≤ first 4096, growth evals ≤ first 1536 — so feed tokens past
  200,192 are a provable held-out wedge per home (250k-token files). Any
  "held-out" claim without this arithmetic is not held out.
- **CE batch-shape drift** (2026-10-19, R8): the same window measured bs1
  vs grouped-bs6 reads ~+0.0009 CE apart (fp16 accumulation order, not a
  bug); compare cells only within one grouping and say which.
- **Reboot-surviving watchdogs** (from the silent-stall episode): the 470M
  pretrain's watchdog must live in crontab/systemd — `/tmp` scripts die on
  reboots. Companion traps: `trap cleanup INT TERM` without `exit` makes
  bash resume after SIGTERM (scripts must `trap 'cleanup; exit 143'`), and
  `pgrep -f <script>` matches monitoring shells that merely mention the
  string and can starve a one-at-a-time gate.
- **Speedrun-opts consolidation** (2026-10-19 audit): the env-gated training
  opts — `VESPER_COMPILE` (dense-submodule `torch.compile`, now with an
  always-falls-back-to-eager guard), `VESPER_FUSED_CE` (chunked linear+CE;
  CPU fallback is a logsumexp autograd Function that recomputes chunk
  logits in backward instead of retaining [N, V]), `VESPER_VALUE_EMBED`
  (first/last-layer token value tables) — are all default-OFF and
  byte-identical when unset. Landed: `lab/speedrun_smoke.py` (CPU proof:
  fused-CE vs plain CE max |dloss| 4.8e-7 / |dgrad| 6.0e-8 fp32, bar 1e-3;
  no [N, V] logits saved for backward; gates-off forward bit-identical to
  explicit-off) and the SFT live-round hooks from r3
  (`VESPER_SFT_MICRO_BATCH` / `VESPER_SFT_SEQ_LEN` OOM knobs +
  checkpoint-time 12-prompt convo harness writing `_evals/eval_chat_step*.json`).
  GPU flag-matrix correctness was already measured (MI300X canary cb521b0);
  **deferred**: per-flag GPU timing (tok/s deltas), and wiring
  `VESPER_VALUE_EMBED` into a trained checkpoint (flag-on checkpoints carry
  2 extra keys and cannot load into a flag-off model).

### 3.7 The bank ladder: rung-1 growth curve, and the 2026-10-16 halt

Rung-1 growth on the 325M-spine router — assemble the first n plugs and
measure the §11 break criteria at each doubling (`MODULAR_MOE.md` §11). Growth
rows measured **4/4** (n=250 measured on the laptop 2026-10-19, R4):

| growth n | home_quality (bar 0.02) | pairwise_harm (bar 0.02) | base_tax (bar 0.05) | p_home (bar 0.5) | §11 criteria broken |
|---|---|---|---|---|---|
| 32 | PASS | PASS | PASS | FAIL (median 0.136) | p_home only |
| 64 | PASS (worst Δ+0.009) | PASS (0.0007) | PASS (Δ+0.0027) | FAIL (median 0.1428, 38/38 collapsed) | p_home only |
| 128 | **FAIL** (writing Δ+0.0239) | PASS (0.0008) | PASS (0.0056) | FAIL (median 0.1464, 83/83 collapsed) | **home_quality + p_home** |
| 250 | FAIL→**PASS w/ R5 budget** (W +0.0262, M +0.0331 unbudgeted; ≤+0.0041 @ q0.75 renorm) | PASS (worst +0.0012, R5-corrected*) | PASS (+0.0079 unbudgeted; −0.0012 @ q0.75) | n/a — metric replaced 2026-10-19 | **growth-250 = PASS under corrected judge + R5 recipe** |

**Re-read 2026-10-19 (R2→R3→R4, `lab/PHOME_CALIBRATION.md`,
`lab/PHOME_REJUDGE.md`, `lab/GROWTH_250.md`):** the p_home collapse column is
a **metric artifact** (fit-time per-row share against the 8-row stock bank —
bank-size-invariant, ceiling ~0.14, reads 0.136/0.1428/0.1464); corrected
§11.2 criterion = home-token membership (gate's real decision) +
ceiling-relative p_home + p_base_max. Under it: **32/64 PASS · 128 HOLD ·
250 HOLD**, and membership is flat-to-rising 0.330→0.360→0.374→0.378 — no
router exhaustion through 250. The real, growing break is in **delivery
space**: home-quality breaks at writing (+0.0233→+0.0262) and math (new at
250, +0.0331 — posted by the glm/mimo rows) while pairwise harm stays ~0 and
base tax stays up at +0.0079. The writing-plug λ sweep is flat ±0.001 (λ is
not the lever); the next lever class is per-home quota / reject-side recal.
§11.2 #4 spine upgrade stays **parked** — the bake-off spine pick is a
quality-axis decision, not a capacity rescue. The **R5 delivery-budget
lever (`lab/REJECT_RECAL.md`, 2026-10-19)** then fixed the remaining
home_quality breaks: the tax is a dense foreign k3 sibling cross-fire
(mean ~7 rows firing/token; glm/mimo ≈ 0; own-home rows ≈ 0; no dominant
individual row) and a per-token delivery-mass budget (q = 0.75, renorm)
brings writing/math/base to +0.0002/+0.0041/−0.0012 — growth-250 = **PASS
under the corrected judge + R5 recipe** (corrected rung-1 carry:
DELIVER additive λ 0.5 through per-token budget 0.75 renorm). *Pre-R5
pairwise-harm columns (this table + `LADDER_STATE.md` §1) were ablation-less
harness noise — the active flags were wiped by the run-state install before
every j-off forward; corrected worst +0.0012 at 250, 24 pairs. Historical note: the
2026-10-16 growth-250 halt was taken on the miscalibrated #3 reading; it was
rescinded by the R2 erratum and measured (laptop, ~0.35 GPU-hr) on
2026-10-19.
Watcher/fit scripts now live on `/data` (not `/tmp`) on the compute box and
do not auto-start after reboot.

---

## 4. Headline claims (and their receipts)

1. **A universal expert socket exists.** Passport/registry routing reproduces
   donor-MoE behavior at 0.998+ across three families (D0, D2, v0.6 port).
2. **Expert cores grab cheaply.** Frozen pretrained cores + 0.8M adapters +
   200k tokens = quality plug-ins in ~30 minutes each (Door 2.5, Door 10).
3. **Provenance is the currency of capability.** Deep task-pretrained cores
   carry skill; scratch/refit/KL-distilled cores do not (Door 5/7, 3×6 falsification).
4. **Composition is additive, not magical.** Mixed banks give exactly the sum
   of parts with zero interference (Door 6: 0.9957×); whole-layer substitution
   saturates (~58%); per-slot replacement is explosive (2625%).
5. **Episodic memory plugs in like an expert.** Shaped memories coexist at
   N=16, flip answer margins, and shift dialog behavior (§8, v0).
6. **Conversational quality is a spine property** that scales with active
   parameters — grafting preserves it (0.986 cos) but cannot create it.

---

## 5. Negative results (the falsification ledger)

Kept deliberately. Each killed a tempting wrong path:

- cross-model feature translators: ~0.5 alignment ceiling (D1.5, Door 9)
- scratch-trained plug-in experts: fail across 3 delivery contracts × 6 recipes
- KL-distillation as expert provenance: transfers KL, not CE skill
- margin-weighted recal: monotone worse than uniform
- softmax-mass plug-in delivery: starves stock experts; additive λ wins
- whole-layer replacement by few cores: 58% saturation (joint/boosting/per-slot/in-env)
- 10–12k-token calibrations: actively harmful out-of-sample (20× under)
- word-level pairing as a translator trainer: lossy; exact pairing still fails

---

## 6. Open questions

1. **Held-out skill transfer** — the single biggest asterisk. In-distribution
   plug-in quality is proven; generalization is at neutrality. Needs larger
   calibration, better cores, or in-env training (the named rungs).
2. **Combine-at-scale** — 8-plug banks work; 1k+ banks are untested. Row
   crowding (cross-fire) is the known mechanism to control.
3. **Memory token-surfacing** — taught facts flip margins and behavior but the
   base prior still wins the decode; decode-time bonuses / λ-boost are the
   named levers.
4. **Variable activation** — per-token adaptive plug gating (compute ∝
   difficulty) is the natural next efficiency + anti-crowding layer; in flight.
5. **Spine acquisition** — borrow a frontier spine (fast, licensed) vs
   distill our own from routed teachers (coherent, "let loose") vs pretrain
   (expensive). Routed-mixture distillation is the leading candidate for the
   self-owned brain.

---

## 7. Repository map

| path | what |
|---|---|
| `MODULAR_MOE.md` | architecture spec: registry format (§4–7), memory experts (§8), grafting (§9), expert format + standard recipe (§10) |
| `SUBSYSTEMS.md` | drives/emotions control layer, gates E0–E4 |
| `lab/FAILURES.md` | the full failure ledger with evidence (read this before proposing anything) |
| `lab/` | experiment harnesses (`graft_d0_parity`, `door25_wrap_expert`, `bank2n_partition`, `kv_cache_v05`, …) |
| `Pretrain/` | trainer (KDA+MLA hybrid MoE), configs, data pipeline |
| `SFT/`, `Hippocampus/`, `Immune/`, `Agent/` | chat finetune, episodic consolidation, admission gating, harness |
| `Dataset/` | corpus builders incl. specialty calibration sets |
| `Common/` | model definitions, telemetry |

---

## 8. Status

Actively in progress: a 10B-token pretrain of our own spine (KDA+MLA hybrid
passport MoE), the SFT talkability test on it, variable-activation gating, and
the v0.6 grafted assembly. The program runs autonomously with an experiment
farm; this log is updated as results land.

*License notes: donor cores are extracted from open-weight checkpoints under
their respective licenses (Phi: MIT; OLMoE: Apache-2.0; MiMo-V2.6-Flash: MIT;
GLM-5.3-Flash: MIT — the flagship GLM-5.3 has a different license; K3 has
scale-triggered terms). Republishing combined models requires per-donor
license review; local research use is unencumbered.*
