# TRANSPLANT_AUTOPSY — forensic account of the dead donor-transplant branches (D1.5 + Door-9)

Read-only autopsy, compiled 2026-10-20 from the committed experiment logs and the
box-imported result JSONs. Purpose: record EXACTLY what was tried, with the exact
metrics, so a smarter attempt does not re-ask a door that is already measured shut.

Sources (primary evidence, in priority order):
- `lab/FAILURES.md`: entries `d15-translator-fidelity` (2026-10-11), the D1 plug-in
  family (`d1-plugin-pass-1`, attempts 1–4), `d2-crossfamily-perlayer`,
  `d2-margin-recal`, `d0-gfx906-moe-kernels`.
- `README.md` §3.2/§3.3 (door-series summary table), `lab/LADDER_STATE.md` §2
  ("Door-9 / D1.5 feature translators: dead, do not retry without a new idea") and
  §10.7 killed-branches list.
- Harness code: `lab/graft_d15_translator.py` (D1.5), `lab/door9_embedding_w.py` +
  `lab/door9_chain.sh` (Door-9).
- Result JSONs (untracked, box imports): `lab/imported/graft_d15/d15.json`,
  `lab/imported/graft_d15/d15_crossmodel.json`, `lab/imported/graft_door9/door9_emb.json`,
  `door9_emb_sens.json`, `door9_depth.json`, `door9_wrap_{rand,winit,winit_quiet,wboth}.json`.

No new experiments were run for this document. Every number below is quoted from the
files above. The "0.47 held-out ceiling" in the verdict refers to the Door-9 emb_arm
best honest fit (B_to_A ridge, held cos **0.4674**); the D1.5 cross-model arm's worst
expert-output cosine is coincidentally also 0.4743.

## 1. What D1.5 actually was

**Gate (pre-registered, MODULAR_MOE §9.3a/§9.4/§9.5/§9.7):** translator held-out cosine
> 0.9 to keep cross-model transplant ("variant B") alive; < 0.7 even in-domain triggers
deprioritization to variant-A-only. Stretch bar 0.95. Same-layer control pair must pass
≥ 0.99 as a harness soundness proof.

**Translator architecture (`lab/graft_d15_translator.py`).** The map learned is
**input-side only**: host pre-MoE hidden state at the capture layer → donor pre-MoE
hidden state at the donor expert's consumption layer (direction measured both ways:
host→donor and donor→host). It is NOT an output-manifold map; donor expert outputs are
only used downstream for the expert-output metric. Four translator classes:

| class | form | parameters |
|---|---|---|
| identity | copy (same-dim control / pairing-hardness probe) | 0 |
| mean | predict train-set target mean (floor baseline) | 0 |
| linear | closed-form ridge least squares, ridge 1e-3 | full d_src×d_tgt (no rank constraint) |
| mlp2 | residual 2-layer MLP, hidden 512, Adam, bias=False on output | ~d_src×512 + 512×d_tgt |

"Route-conditioned" (`cond=True`) **concatenates** the standardized passport scores at
the source layer to the translator input — it is NOT a per-route bank of ridge maps.
Metrics per arm: per-token cosine + MSE (train/held), per-route breakdown, and the
spec's transplanted-output metric: cos between `expert_e(T(x_src))` and
`expert_e(x_tgt)`, plus the same cosine restricted to the expert's top-32 centered
singular directions.

**Held-out definitions (three strictness levels):**
- `iid`: window-level split of the captured token stream (90/10).
- `route`: entire donor top-1 routes removed from training (route generalization).
- `domain`: one named domain slice entirely held out (the §9.3a falsifier probe).

**Run (a) — within-donor, cross-layer (the layer-correspondence axis).** OLMoE-1B-7B-0924
fp16 (16 layers, d 2048) against itself; host capture layers {2, 6} vs donor layer 12, with
12:12 as the same-layer control. 35,840 tokens across 5 domain slices (chat, code, math,
science, web from `graft_d0_slice`). Compute: MI50 box GPU 3, capture 25.2 s, all fits
1,353.5 s wall.

**Run (b) — cross-model (the true §9.3a pairing).** Donor OLMoE-1B-7B-0924 layer 12 vs
host granite-3.0-1b-a400m-base layers {12, 18}. Different tokenizers, so pairing is
word-aligned char-offset pooling (835 chunks x 24 words → 18,529 paired words on
each model's own token stream). Capture 386.8 s, fits 941.0 s wall.

### 1.1 D1.5 scores (held-out cosine mean; bar 0.9 / deprioritize 0.7)

Within-donor, primary arm (linear, cond=True):

| pair | role | identity baseline | train | held iid | route-held | domain-held | expert-out (top-dirs) |
|---|---|---|---|---|---|---|---|
| 2:12 | challenge | 0.2075 | 0.6796 | **0.5528** | 0.5227 | 0.3882 | 0.3491 (0.4898) |
| 6:12 | challenge | 0.4172 | 0.7414 | **0.6296** | 0.6098 | 0.4882 | 0.4639 (0.6079) |
| 12:12 | control | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 (1.0000) |

Cross-model, primary arm (linear, cond=True):

| pair | train | held iid | route-held | domain-held | expert-out (top-dirs) |
|---|---|---|---|---|---|
| h12:d12 | 0.7243 | **0.6437** | 0.6097 | 0.5360 | 0.4743 (0.6288) |
| h18:d12 | 0.7572 | **0.6885** | 0.6575 | 0.5870 | 0.5526 (0.6979) |

Per-route macro (cross-model): mean 0.667, best 0.898, worst 0.556 — **no route ≥ 0.9**.
Verdict: every gate red, §9.5 deprioritize (< 0.7 in-domain) fires in the 0.55–0.69 band.

### 1.2 D1.5 signature facts (these are the failure curve SHAPES)

1. **The map is not learnable even in-sample.** Train cosine saturates at 0.68–0.76
   across every non-trivial pair; the same-layer control hits exactly 1.0.
2. **Capacity is exonerated.** mlp2 ≈ linear everywhere on held (e.g. 2:12 iid:
   0.5507 vs 0.5528; cross-model mirrors), with mild overfit inside; adding
   nonlinearity buys nothing held.
3. **Route conditioning is a no-op.** cond=True vs False moves held cos by ≤ 0.002
   in EVERY cell — the passport scores carry no information that repairs the basis.
4. **Generalization degrades monotonically with split strictness**: iid > route >
   domain (≈ −0.03 then ≈ −0.14). The translator memorizes route-local detail and
   domain texture, not a geometric bridge.
5. **Fidelity scales with layer-distance, not model-specifics**: within-donor
   1.0 (Δ0 layers) → 0.63 (Δ6) → 0.55 (Δ10); cross-model deep-host layer 18 is the
   best cell (0.69).
6. **The SwiGLU expert AMPLIFIES the error**: output cos sits ~0.2 below input cos
   in every challenge cell (0.55→0.35, 0.63→0.46, 0.64→0.47, 0.69→0.55). The top-32
   singular directions recover ~+0.14 — the destroyed mass is in the low-energy
   directions, i.e. the mismatch is broadband, not a few missing axes.

## 2. What Door-9 actually was (the better-posed D1.5)

D1.5's cross-model pairing was word-pooled (noisy targets, two tokenizers). Door-9
(`lab/door9_embedding_w.py`) removes ALL alignment error by pairing the two donors'
**embedding tables by decoded-string identity** (single-token round-trip strings only).

**Models:** OLMoE-1B-7B-0924 (host; d 2048, 16 layers) ↔ Phi-mini-MoE-instruct
(donor; d 4096, 32 layers). Clean-string pools 49,573 (A) ∩ 16,419 (B) → 7,019 exact
intersection → 6,351 train / 668 held (10%).

**Bars (pre-registered in the docstring):** if exact-pairing W hits ≥ 0.95 train /
≥ 0.90 held, the D1.5 verdict must be revised to "pairing loss + depth divergence";
otherwise the input basis itself is not bridgeable.

### 2.1 Arm 1 — emb fit (the ~0.47 headline)

| direction | map | train cos | held cos | notes |
|---|---|---|---|---|
| B→A (4096→2048) | ridge 1e-3 | 0.6666 | **0.4674** | best honest fit ever measured |
| B→A | mlp2 (h512, 800 steps) | 0.8707 | 0.2832 | catastrophic overfit: +0.20 train, −0.18 held |
| B→A | predict-mean baseline | 0.1846 | 0.1829 | |
| A→B (2048→4096) | ridge 1e-3 | 0.4407 | 0.3367 | |
| A→B | mlp2 | 0.6981 | 0.3220 | |
| A→B | predict-mean baseline | 0.0985 | 0.0965 | |

Sensitivity (`door9_emb_sens.json`, strict intersection):
- Ridge sweep held-cos curve is a classic bias–variance hump whose PEAK is the
  failure: ridge 0 → 0.2612; 1e-5 → 0.3182; 1e-4 → 0.4219; 1e-3 → **0.4674**;
  1e-2 → 0.3477; 1e-1 → 0.2128. No regularization setting escapes ~0.47.
- Procrustes (orthogonal constraint): train 0.6923, held 0.3540 — WORSE than plain
  ridge; a rotation family does not help.
- Shuffled-target floor: 0.0918 held — the 0.467 is real signal (5× over floor),
  not noise, and still nowhere near 0.9.
- Per-token distribution (B→A ridge, held): p10 0.352, p50 0.476, max 0.690 —
  frac ≥ 0.90 = 0.0. Not one token pair bridges cleanly.

Verdict_emb: pass_linear false, mlp_beats_linear_held false. The D1.5 "feature
availability" reading SURVIVES exact supervision: even with zero alignment error,
the input manifolds of two separately-pretrained MoE families share ≤ ~0.47 fit.

### 2.2 Arm 2 — depth decay (where the common structure lives)

The emb-fitted W applied to mid-layer hidden states of matched strings
(grid 33×17 layer pairs, 512 pairs sampled from the held split):

- Best cross-layer cell = layer 0:0 at 0.4771 (i.e. the embedding fit itself).
- **Every interior pair collapses to ~0.045–0.047 under the emb W.**
- The per-depth "availability ceiling" refits (fresh ridge at each depth) LOOK
  perfect — held 0.99996+ at frac 0.25/0.5/0.75 — but the predict-MEAN baseline is
  also 0.99995+ there (target rank-1 energy 0.89–0.99): interior hidden states of
  both models are mean-direction-dominated, so those "perfect fits" are degenerate
  and carry no translation information. Only at layer 0 is the signal diverse
  (rank-1 energy 0.0188), and there the honest fit fails (0.3302 held vs 0.1836
  baseline).
- Diagnostic consequence: at intermediate depths there is nothing to translate —
  both models' activations are squeezed onto a giant shared direction per model.

### 2.3 Arm 3 — does a good W help the working path (wrap init)?

Payoff arm: install W's rank-64 SVD into the door25 wrap's host_in/adapter_up slots
(donor L13 e8 frozen SwiGLU core grafted at host L6, d_canon 64, 786,432 adapter
params) and compare against random init on the standard door25 recipe (250 steps,
lr 3e-3, out_pen 300; recal 400 steps lr 0.05, owner_mass 0.35, reject_w 10, mass_w 2;
data 10,240 code-train tokens, 59,392 base; eval 11,264 train / 7,168 held code /
19,456 math-cross). ~17–18 min wall per arm on the MI50 box.

| arm | input-chain init | train plugged CE (vs masked 1.068) | held Δ plugged-vs-masked | parity cos | util_code | contam |
|---|---|---|---|---|---|---|
| rand (door25 verbatim) | N(0, 0.02) | 0.4957 (−0.5724) | +0.0369 | 0.99846 | 0.8308 | 0.0647 |
| winit | rank-64 SVD of W, α=190.2 (RMS scale match) | 1.0878 (+0.0196) — frozen | +0.0041 | 0.99846 | 0.8308 | 0.0647 |
| winit_quiet | same SVD, α=1.0 | 0.4966 (−0.5716) | +0.0445 | 0.99841 | 0.8308 | 0.0647 |

- α=190.2 (the "scale-matched to donor residual RMS" init) is actively harmful:
  param L2 stays 1,191.7 and out_norm2 collapses to ~1e-7 → the adapter is dead at
  birth; s6_quality FAILS (train plugged 1.0878 > masked 1.0681).
- α=1.0 winit is **statistically identical to random** on every number that matters:
  same train gain (−0.571 vs −0.572), same held gap (+0.044 vs +0.037), same recal
  time to p80 (70 steps both), same memorization-sign inversion. The geometrically
  motivated init transmits zero transferable structure into the fit.
- The wboth arm (W-init on the donor→host output chain too) never completed its
  measurement — JSON stops before verdict; treat as untried, not failed.
- Both working arms fail the RETENTION gate (retention < 0.70) in the formal
  verdict struct while passing parity/contam/util/s6 — the door25 memorization gap
  (later attributed by Door-10 to data starvation, 10k fit tokens, NOT geometry).

## 3. Failure-class assignment (the four classes, ranked by evidence)

1. **INPUT MANIFOLD divergence — PRIMARY, confirmed by triangulation.**
   Three independent pairings converge on the same band: telemetry within-donor
   cross-layer 0.55–0.63, word-pooled cross-model 0.64–0.69, exact string-matched
   embeddings 0.47. Train cos itself saturates ≤ 0.76, so this is feature
   availability in the SOURCE representation, not translator capacity. The
   ridge-sweep hump peaks at 0.4674 — no linear regularization family escapes it;
   MLP overfits; Procrustes is worse. Two families pretrained separately organize
   their embedding/early-layer bases around different, only partially aligned
   usable subspaces; the shared usable subspace is ~25–50 % rank energy
   (rank-64 captures 0.529 of the fit's singular energy).
2. **OUTPUT MANIFOLD distortion — compounding, confirmed, NOT root.**
   expert-output cosine tracks input fidelity minus ~0.2 in every cell and still
   passes 0.9999+ on the 12:12 control, so the donor SwiGLU doesn't add
   independent error — it AMPLIFIES input error (nonlinearity). The lost mass is
   broadband (top-32 directions recover only ~+0.14), so "translate a few head
   directions cleanly" would not rescue it.
3. **ROUTER / addressing — EXONERATED.**
   Route-conditioned inputs change nothing (≤ 0.002); route-held split degrades
   held cos by only ~0.03; per-route macro 0.667 flat with worst 0.556. On the
   delivery side, D2 cross-family logit parity PASSES (0.99863) and algebraic
   passports pass every gate; D1's recal closes to p80 in 70 steps and hits
   util 0.83 / contam 0.065. The routing socket transports whatever it is handed;
   it cannot carry a basis error, and nothing about the 0.47 ceiling is routable.
4. **CALIBRATION / fit recipe — EXONERATED within the tried family.**
   Ridge sweep, orthogonal constraint, mean/shuffled floors, 2-layer nonlinearity,
   exact vs pooled vs telemetry supervision, two init scales for the wrap: the
   recipe surface is well-mapped and monotone-peaked far below bar. Door-10 later
   proved the wrap's held gap is TOKEN COUNT (gap 1.35→0.003 by 200k tokens),
   closing the last "maybe the fit was just starved" reading of Door-9 arm 3.

## 4. Definitely ruled out (do not re-ask)

- Telemetry-trained hidden-state translators (ridge or 2-layer MLP, either
  direction, iid/route/domain splits) at 1B-class host/donor: band 0.55–0.69 held,
  hard-capped by in-sample 0.72–0.76.
- Any claim that D1.5's result was a pause-tokenization/word-pooling artifact:
  exact embedding pairing caps LOWER (0.47).
- More linear-map capacity or regularization tuning: the ridge hump peaks at
  0.4674; Procrustes 0.354; unregularized 0.261.
- Route information as a repair signal: cond moves nothing; passports route, they
  do not re-express.
- Initializing a working wrap with the best measured W: bit-level null vs random
  (α=1.0) or destructive (α matched to donor RMS freezes the adapter).
- "Deeper host layer helps" beyond the measured envelope: fidelity scales with
  layer-proximity but the best cell ever measured is 0.69 held / 0.55 output.
- Reading interior-depth refits as evidence: sub-collapse (rank-1 mean domination)
  makes any mid-layer cosine ≥ 0.99 trivially by baseline.

## 5. Genuinely not tried (as recorded in the FAILURES entry + code)

- Soft attention over ALL donor layers as the translator (§9.3a sharpening 2).
- Per-(expert, layer) translator banks (per-route FITS were never trained; only
  per-route EVALUATION and an additive route-feature concatenation).
- Token-level 1:1 subword pairing on a shared subword corpus (both runs pooled or
  string-restricted; Door-9 used 7,019 single-token strings only).
- Large nonlinear / residual translators (widest tried: hidden-512 2-layer MLP).
- Larger host spines — the pre-registered falsifier predicts host model quality
  bounds achievable fidelity; only 1B-class pairs were measured.
- Output-side training of the map (T fit against expert OUTPUT targets instead of
  pre-MoE inputs).
- The wboth output-chain W-init arm (started, never measured to verdict).
- Joint fit: translator and wrap RECAL trained end-to-end (all fits were staged:
  W first, wrap frozen-recipe second).

## 6. Standing recommendation (quoted from the lane)

The FAILURES entry's own closing recommendation stands: accept §9.5 — variant-A-only
(one grafted host per donor family, mixing at the serving layer). The modular bank
line that actually shipped (bank2n, Door-6 sum-of-parts, Door-10 calibration scale)
never requires crossing the feature basis, which is exactly why it survived.
