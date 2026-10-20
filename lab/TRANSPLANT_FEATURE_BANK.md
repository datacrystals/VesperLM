# TRANSPLANT_FEATURE_BANK — transplant as service (concept-door-4)

Status: **feasibility sketch + one CPU measure, pre-declared bars, honest
bars read at the bottom.** 2026-10-20, laptop, CPU fp32, zero GPU claims.

Context: the "literal donor-weight transplant between MoE families" verdict
(Door-9 / D1.5; `lab/FAILURES.md` d15-translator-fidelity, door9 emb/depth/
wrap arms; `lab/LADDER_STATE.md` §2 "dead, do not retry without a new idea")
proved that putting foreign weights *inside* the host body is dead — not
that donor reuse is dead. This doc sketches the alternative shape the
transplant line never tried: **do not put the foreign weights in the body;
register them in a bank of callable supply-side experts** and let the
passport route tokens to them as *remote method calls*.

## 1. The clean line: made-local vs made-service

**Made-local (Door-9 / D1.5 shape — dead).** Bring donor features to the
host: translate embedding rows (Door-9 emb arm), translate per-layer hidden
state (d15 / Door-9 depth arm), then hand the foreign core weights living
inside a foreign-manifold wrap (Door-9 wrap arms). The translation has to be
valid on *every token of every window the host ever sees*, because the
weights are now resident in the body. Measured ceiling: ridge held-out cos
**0.467** on exact token-paired embedding rows (bar 0.95 train / 0.90 held)
and 0.55–0.69 on cross-model per-layer translators; W-init of the wrap did
not beat random init (door9 wrap arms, `lab/imported/graft_door9/*wrap*`).

**Made-service (this branch).** Donor weights never leave their own
manifold and never become resident in the host body. The host owns a
*registry of callable cores*; a passport row decides *whether a given token
is worth a call*, and two narrow *boundary projections* (sandwich-in /
sandwich-out) align only the call interface. The alignment obligation
shrinks from "all tokens × whole hidden space" to "called tokens × the
boundary width of one core I/O". The transplant question becomes a service
question: *is the boundary learnable*, not *are the manifolds identical*.

## 2. How a call works

Host state: residual stream `h(t)` at call layer `l_call`. The surfaces:

1. **Passport (host side, native basis).** The router scores the routed
   callable row from `h(l_call)` exactly as it scores stock experts —
   unchanged passport machinery (D0/D1 premise, algebraic passports for
   linear gates). The passport sees **no projection**: membership `m_t`
   is computed in the host's own basis, so addressing stays a host-native
   problem (the G1/G1b lesson — content-matched routing needs a
   passport-native spine, not a foreign one).
2. **Sandwich-in `g_in: R^{d_host} -> R^{d_core_in}`.** Only for called
   tokens: project `h(l_call)` into the donor core's input space. Trained
   on *mismatch windows* (§3, recipe R1) — windows where the host-hidden
   distribution and the donor-native input distribution disagree beyond a
   band. Never applied to non-called tokens.
3. **Frozen donor core `e` (service side, donor basis).** The core is a
   verbatim extract (`lab/grabkit_extract_core.py` family: K3/GLM/MiMo/Phi
   expert cores, 12–24 MB each, sha-pinned, dormant bytes — not host-graph
   parameters). It computes on its own manifold: `e(g_in(h))`. Zero mixing
   with the spine.
4. **Sandwich-out `g_out: R^{d_core_out} -> R^{d_host}`.** Project the
   donor output back into the residual space at the exit layer
   (same layer for a single-block call; `l_call` with a depth offset for a
   multi-block call). Trained on the exit-side mismatch windows.
5. **Additive delivery at weight `lambda * m_t`** (Door-4 winner):
   `y = y_stock + lambda * m_t * g_out(e(g_in(h)))`. The callable is *not
   a row in the softmax-mass merge* — with it fused as a bank row, mass
   competition would re-enter; as a service it never can.

An ASCII picture of one call:

```
host residual h(t) at l_call
        |
 passport row p_e  -- m_t = call weight (host-native basis, no projection)
        |                          \
 sandwich-in g_in                   (non-called tokens: nothing happens,
        |                            residual = bitwise stock)
 donor core e (frozen, own manifold)
        |
 sandwich-out g_out
        |
 y = y_stock + lambda * m_t * (g_out ∘ e ∘ g_in)(h)
```

## 3. Calibration recipes (what the sandwich trains on)

**R0 — spectral pre-check (offline, CPU-cheapest; §5 measure).** CCA/CCP
spectral scan between host-hidden at `l_call` and the donor-side view.
Decides whether a linear boundary exists at all and what width it needs
(top-k canonical span). Kills a pair before any fitting budget is spent.

**R1 — alignment-window curriculum (the R0 successor).** Pair shared token
streams. Classify windows by mismatch: (a) *parity windows* — host-hidden
distribution and donor-native input distribution at the analogous surface
already agree within a band → these form the **guard set**: never trained
on, only verified (parity must hold); (b) *mismatch windows* — disagreement
beyond the band → the training set for `g_in` / `g_out`. Objective on the
mismatch windows: distribution matching (mean + second moment + cosine),
not token-exact regression — the Door-9 lesson is that exact-regression
over-fits pairing idiosyncrasies and memorizes (0.87 train / 0.28 held MLP
arm).

**R2 — contract fit through the frozen core.** Fit the bridge *through* the
donor core with task CE on the core's home text, additive delivery,
`wd 0, lr 3e-3, clip 1.0` — Door-2.5's verified recipe, with one deliberate
deviation: `out_pen` (the L2-neutrality penalty on `h` at non-home tokens)
can be **weakened or removed**, because non-called tokens are untouched by
construction (service = no always-on wrap). The D1/door25 starvation knobs
(`wd 0.01 + clip 1.0` chokes a thin adapter to `|y|~5e-4`) stay copy-forbidden.
Ridge, not MLP: the Door-9 emb arm showed the MLP buys nothing held (0.28 vs
0.47 ridge) — capacity is not the binding term, feature availability is.

**R3 — call-rate recal (membership only).** The D1 lesson, relocated:
teacher-forced calibration over-predicts live routing — the on-trajectory
round-2 recal is mandatory. But the target is **membership** (how often the
callable is worth calling), never a mass target: there is no mass to calibrate
because the row is not in the softmax. This collapses the D1 att-1..3
owner_mass / parity / utility Pareto onto a single axis.

**R4 — held-out discipline.** Multi-slice windows (the door25/door6
memorization caveat: 11–12 k fit tokens sharpen the fit slice and regress the
probe half). R1's mismatch-window curriculum is itself a memorization
countermeasure: the fit sees only where the surfaces disagree; agreement is
checked, not sharpened. This claim needs its own measurement (worry W5).

## 4. Failure modes it avoids by NOT injecting

1. **Manifold poisoning (D1.5 / Door-9 shape).** A resident translated-map
   plants its basis error into *every* token's representation, including
   tokens the donor knows nothing about — that is exactly what the 0.47
   ceiling buys you: contaminated everywhere, useful nowhere. Under
   made-service the boundary maps are evaluated only on called tokens;
   non-called tokens are *bitwise* stock forward (layer-local parity 1.0 by
   construction, versus the D1 att-1 attempt's parity 0.782 fail at owner
   mass calibration).
2. **Spine poisoning.** Host weights are never written; the donor weights
   are never fused. Rollback = delist a registry entry (the served catalog
   goes back to bitwise-incumbent without a weight restore — the same
   guarantee the tier-2 Immune verdict had to *measure* in the teach
   series, here free by construction).
3. **Kvasa/residual-mass poisoning (D1 att-3's "0.9 mass on a near-zero
   delta starves 8 stock experts": code CE 1.26 -> 14.28).** A callable
   holds no mass position in the merge softmax, so this mode cannot occur.
   Delivery is always the Door-4 additive contract.
4. **Provenance collapse (door5).** The core is a verbatim donor-pretrained
   extract — Door-2.5's existence-proof recipe (frozen Phi core, wrap CE
   0.439 vs 1.072 masked) — not a distilled/refit core, on which door5 ran
   three treatments to uniform FAIL.
5. **Cross-family router-parity demands (D2 per-layer bars).** A resident
   graft must reproduce the donor's gate *decisions* per layer (D2 sparsemixer
   aux bars: MoE cos 0.864 vs 0.9 bar, agree 0.651 vs 0.8 — and the margin
   recal is a falsified fix). A callable poses no per-layer agreement bars
   at all; the only gate that matters is whether home-domain calls help.

## 5. Param-count arithmetic story (N callable donors × K banked bridges)

Currency: vk-scale host (dim 1024) as the reference spine.

- One callable donor core `e_i`: dormant bytes only (K3-class extract
  17.5 MB fp8-packed; Phi-class 23.6 MB bf16; ~11–20 M params each, zero
  spine-parameter effect). N of them cost N × ~20 MB of library storage
  (NVMe-offloadable per MODULAR_MOE §5).
- One bridge pair at call depth `k`: `g_in (1024 × d_core_in)` +
  `g_out (d_core_out × 1024)`. Dense at d_core 1024: 2.10 M params (~8 MB
  fp32). Rank-64 factored (door8 geometry): ~2 × (1024·64 + 64·1024) ≈
  262 k params — the same scale as Door-2.5's 786 k wrap adapters.
- Banked catalog: N cores × K call depths = N·K bridge pairs. N = 250
  cores (growth-250 catalogue side) × K = 2 depths → 500 bridge pairs ≈
  0.13–1.05 G bridge params at the factored/dense widths, **zero** of them
  in the spine. The comparison term to watch: Door-9's resident alternative
  pays the same adapter budget *and* carries the unbounded parity bill.
- Install cost per callable: G2's measured "driver install": ~800
  router-only recal steps, minutes-scale — paid once per (core, spine-
  version) and shipped precomputed (MODULAR_MOE §9.6 applies verbatim).
- Serving cost: per called token, one boundary GEMM + one core forward +
  one boundary GEMM back; uncalled tokens cost nothing — the D1
  "utilization vs parity" trade becomes a pure compute-choice knob at
  decode time (call-rate throttle), which no resident shape can offer.

## 6. The feasibility measure (part b)

**Question.** Between a vk-family hidden at a named layer and a 470m-family
hidden at its analogous layer, is there a learnable map — read as a
CCP/CCA series — at all supporting a boundary-shaped contract?

**Materials (program's own, CPU).**
- vk family: `lab/imported/vk_sft_r3/step_1800/chat_model/vesper_chat.pt`
  (561 M total / 325 M active, 10 L × 1024, 8 E top-2, kda/mla hybrid,
  passport router, value-embed — the teach-series chat checkpoint).
- 470m family: `lab/imported/sft_470m/step_best/checkpoint.pt`
  (same skeleton; the t3-line SFT with LoRA, folded at `alpha/r = 2`,
  42 LoRA modules merged at load).
- Shared text: 16 × 512-token windows from the tail (offset frac 0.70) of
  `Dataset/data/pretrain/phase2_pretrain.bin`, real corpus (near-manifold
  for both lines; see W4). Both families share the vocab (65523) so
  pairing is exact per position — no cross-tokenizer pooling.

**Protocol (`lab/feature_bank_cca.py`).** Mirror-forward tap of the
residual stream after each block (contract-checked: max |Δ| vs the stock
forward = 0.0 on both models), layers {2, 5, 8} tapped. Window-split
fit 10 / select 3 / held 3 — no token leaks across splits. Per layer:
(i) identity floor = mean per-position cosine of the raw same-layer
hiddens; (ii) shuffled-pair null; (iii) ridge maps X→Y at lambda ∈
{1e-3, 1e-2, 1e-1} of mean diagonal — train cos / select cos / held cos +
R²; (iv) whitened-PCA CCA, k = 64, canonical-correlation spectrum on fit
vs held (transform through fit bases); (v) within-model reference: vk
layer l → vk layer l+2 with the same ridge protocol (what one depth jump
costs *inside* one manifold — the honest yardstick, since d15 measured
within-donor 0.55–0.63 at similar gaps).

**Pre-declared bars (written before the run).** λ is chosen on the select
split, never the held split; all three λ cells are reported regardless.
- PASS: ridge held-out mean cosine ≥ **0.90** (the D1.5/Door-9 bar).
- SERVICE-BAND: held cos in [0.55, 0.90) **with a CCA subspace story**
  (≥ 16 held canonical directions ≥ 0.8) — the boundary exists but is
  narrow; only subspace-constrained callables.
- DEAD-REPRO: ≤ 0.55 across the series, or identity/null floors anywhere
  near the mapped numbers — transplant-dead reproduces on vk/470m
  materials; the service branch inherits the ceiling as its own worry W0.

**Results** — `lab/results/feature_bank_cca.json` (wall 417.9 s, CPU fp32;
mirror-forward contract max |Δ| = 0.0 on both models; 42 LoRA modules
folded on the 470m side; fit 5,120 / select 1,536 / held 1,536 paired
positions per layer cell).

λ chosen on select split = 1e-1 at every layer. Headline series (held):

| cell | identity floor | shuffled null | ridge held cos (λ .1) | ridge fit cos | within-model ref (l→l+2) | CCA held top-1 | CCA held top-16 range | CCA held mean (k=64) |
|---|---|---|---|---|---|---|---|---|
| vk L2 ↔ 470m L2 | −0.0039 | −0.0006 | **0.6828** | 0.8309 | 0.7365 | **0.9813** | 0.828–0.981 (16/16 ≥ 0.8) | 0.6326 |
| vk L5 ↔ 470m L5 | +0.0058 | +0.0020 | **0.6716** | 0.8466 | 0.7086 | **0.9811** | 0.894–0.981 (16/16 ≥ 0.8) | 0.7166 |
| vk L8 ↔ 470m L8 | +0.0089 | +0.0016 | **0.6131** | 0.8431 | — | **0.9586** | 0.693–0.963 (15/16 ≥ 0.8) | 0.6104 |

R² of the held ridge maps at λ .1: 0.380 / 0.343 / 0.280 (L2/L5/L8) —
the map carries about a third of held variance direction; per-position
held cosine p10–p90 at L5 = 0.549–0.778. Ridge fit ceiling ≈ 0.85 even
in-sample: feature availability, not capacity — the d15 root cause, one
family closer to home.

**Verdict against the pre-declared bars: SERVICE-BAND at L2 and L5**
(held ridge cos 0.67–0.68 in [0.55, 0.90) + ≥ 16 held canonical
directions ≥ 0.8 — 16/16 at both), **borderline at L8** (0.6131 + 15/16 —
one direction short of the corridor rider). PASS (0.90) not met anywhere;
DEAD-REPRO not met anywhere (map ≫ identity ≈ 0 and null ≈ 0 by ~0.7).

Two design-shaping reads:
1. **Cross-model ≈ within-model-depth-cost.** Mapping vk → 470m at the
   same depth (0.68) costs about the same as mapping vk across its own
   two-layer gap (0.71–0.74, same protocol). Same-skeleton sibling models
   are NOT separated by a Door-9-style family wall; the price looks like
   depth drift, not family drift. The obvious confound (shared
   tokenizer/skeleton/related mixes, exact position pairing) is declared
   as W2/W3 — a GLM/K3/MiMo donor pays the full pairing + basis tax this
   cell was exempt from.
2. **The boundary is a corridor, not a full-rank map.** ~16 held canonical
   directions at ρ 0.83–0.98 against a full-rank map's 0.66–0.68: the
   shared content concentrates in a rank-16–32 corridor. That is a hard
   constraint on R1/R2: bridges must be **rank-capped at the corridor**
   (factored at rank ≈ 16–32, ≈ 65–130 k params at host dim 1024 per pair)
   not dense 1024-wide; R0's job in the series is to find the corridor and
   freeze its basis before any CE budget is spent.

The honest name-your-worry summary for this number: it is (i) sibling-
family only, (ii) a residual↔residual proxy rather than the donor-input
boundary (W1), (iii) n = 16 windows at one corpus offset (W4). What would
buy confidence: the W3 control (raw-pretrain pair, one redo-cell), a
second corpus slice, then the 1-GPU-hr R2 contract fit graded on held CE
with the corridor basis frozen.

## 7. The honest worry list (what needs the real substrate)

- **W0 (the doorway):** if §6 lands at DEAD-REPRO on sibling families, a
  GLM/K3/MiMo donor will be strictly harder; the R1 recipe cannot be
  graded as "feasible nuisance, train through it". The next unit of
  confidence then costs a GPU-hr (R2 contract fit with CE read), and is
  only worth spending after §6's CCA says a subspace exists.
- **W1 (proxy, not contract):** §6 measures residual↔residual between two
  same-skeleton siblings. The true service boundary is host-hidden →
  *donor-core-input distribution* — a different pair (donor input spaces
  are expert-MLP pre-activations, not residuals). A sibling-residual PASS
  is necessary-feeling evidence, not sufficient.
- **W2 (shared lineage flatters):** both families share tokenizer, arch
  skeleton, and related training mixes. The pairing is exact per position —
  the Door-9/pooling tax is zero here and will not be for GLM/K3/MiMo.
- **W3 (SFT-history confound):** the vk side is the finished 10.07 B-token
  base's SFT; the 470m side is SFT'd (LoRA, this measure folds it) from the
  halted 3.7 B-token t3 run. Alignment measured may ride on SFT idiom, not
  base geometry. The clean control (raw vk-base vs raw 470m pretrain) is
  not fully on disk (the 470m pretrain ckpts were rotated off the laptop /
  live on the box) — once touched, redo §6 layer {2} only (~10 min).
- **W4 (one slice, one offset):** n_windows = 16 at 512 tokens; window-level
  variance is estimated but thin; a second corpus family (SFT-mix vs
  pretrain-slice) is the next CPU-cheap cell.
- **W5 (memorization tail):** R1's mismatch-window curriculum is a design
  claim, not a measurement — the door25/door6 held-out regression could
  re-appear as "the bridge sharpens disagreement windows and regresses
  parity windows". Needs the 1-GPU-hr R2 fit to grade.
- **W6 (CPU-vs-kernel faithfulness):** taps run under `vk_fla_compat`'s
  naive-recurrent KDA path (fp32 CPU) rather than the training kernels;
  per-position hiddens are fp32-faithful but not bit-identical to the
  Triton path — nothing in §6 that rides decimals near the bar should be
  re-baselined on the MI300X path before being quoted further.
- **W7 (demonstration scale > 1 GPU-hr):** caller-side cost model (call-
  rate economics, cache residency, N×K catalog assembly against the budget
  renorm of `lab/REJECT_RECAL.md`) — none of it is shown by this doc; it
  only says what the shape would cost if the boundary exists.

## 8. Reproduction

```bash
PYTHONPATH=$HOME/.local/lib/python3.12/site-packages VESPER_CPU_THREADS=12 \
  python3 -u lab/feature_bank_cca.py --out lab/results/feature_bank_cca.json
```
(3-window smoke: `--smoke`). Weight artifacts stay under `lab/imported/`
and are not committed; the JSON is aggregate-only (counts, shapes, cosines)
per house rules.
