# TRANSPLANT_SANDWICH — sandwich-translator transplant at vk scale

Swarm branch following the "literal donor-weight transplant between MoE
families is dead" verdict (Door-9 / D1.5, ~0.47 held-out cosine ceiling).
This arm keeps the creative shape the dead lanes never installed: a REAL
frozen donor expert core, made host-legible by two thin trained projections,

    h_host --[W_in 1024->1024]--> donor-space --[DONOR EXPERT, frozen]-->
    donor-out --[W_out 1024->1024]--> host-hidden flow

plugged into ONE MoE expert slot of the host. Everything else in the host is
frozen, including the passport router — routing through the transplanted
slot is organic, never forced.

Pair (vk scale, no licensing surface — both checkpoints are in-repo family
artifacts):

| role | checkpoint | notes |
|---|---|---|
| host | `lab/imported/vk_sft_r3/step_1800/chat_model` | vk SFT-r3 chat, val-best, 10L KDA+MLA hybrid, 8 passport experts |
| donor | `lab/imported/sft_470m/step_150/chat_model` | 470m-family SFT step_best (= step_150, val 2.3044), same arch + tokenizer, family-divergent manifold |

Position relative to the closed lanes. D1.5
(`lab/graft_d15_translator.py`) measured that paired-activation input-basis
translation does NOT transfer across MoE families (0.47–0.69 held-out cos)
and Door 2.5 (`lab/door25_wrap_expert.py`) hosted a foreign core behind
up/down adapters at OLMoE/Phi scale trained on task CE. This arm is neither:
an internal vk-family pair (same tokenizer, same shape, divergent manifolds
— the measured host/donor cosine distance at the transplant site is its own
deliverable, see §3), sandwich training on hidden-state mismatch windows
with a host-function anchor (parity is a designed objective, not a hope),
and a pre-declared liveness bar that the mimic-collapse degeneration cannot
pass for free.

## 1. Pre-declared protocol

Harness: `lab/TRANSPLANT_SANDWICH.py`. Numbers-only results:
`lab/TRANSPLANT_SANDWICH_OUT/results.json`. Runtime: laptop CPU fp32, 12
threads, house shims (`Hippocampus/consolidate.load_base_model`,
`kda_mla` profile); the box (100.64.0.7) is never touched.

| item | value |
|---|---|
| transplant site | layer 4 FFN, host slot = argmax routing share on the calibration library (measured at runtime); donor core = argmax-share expert at the donor's own layer 4 |
| calibration library | the teach-arm's own library verbatim (6 items: teach strings, probe target/decoy, session packs, replay pools, 22 filler pairs — 90 pairs), imported from `lab/teach_spaced_cadence.py`, never copied |
| hidden-state mismatch windows | tokens with host/donor ffn-input cosine distance ≥ median over the library; 4:1 train/hold split of the window pool |
| training | Adam lr 1e-3, 500 steps, batch 256 tokens; loss = MSE(W_out·D·W_in(h), host_expert_e(h)) + 0.25·MSE(W_in(h), h_donor) |
| seeds | arm A = 7, arm F (fallback) = 8 |
| naive reference | identity sandwiches = untrained literal transplant (Door-9 shape at vk scale), measured on the same probes |
| fallback (exactly one) | arm F: W_in frozen at identity (null transpose), W_out trained only |

Bars (per-token teacher-forced CE, token-weighted, Δ = transplanted −
host-bare):

- **Bar A — parity.** On 3 fixed flat probes (generic chat pairs held out of
  training), ΔCE ≤ **+0.05 nats** on every probe.
- **Bar B — liveness.** On a donor-class fact set (top-3 library fact
  candidates by CE_donor − CE_host, ranked at runtime, margins recorded),
  ΔCE ≥ **+0.10 nats** AND ≥ 5% of fact tokens route through the
  transplanted slot. Higher CE there = the frozen foreign core fires
  visibly on donor-leaning tokens; parity says it does not leak elsewhere.
- **Feasibility** = A ∧ B. Pre-declared failure typing (autopsy):
  TYPE-4 routing-starve (<5% fact-token routing), TYPE-2 parity-broken
  (Bar A fails), TYPE-1 mimic-collapse (Bar A passes, B fails, eval
  contribution-norm ratio ρ < 0.2 — the sandwich learned to kill the core),
  TYPE-3 inert-unselective (A passes, B fails, ρ alive).

Note on intended direction of Bar B: the donor class is OPERATIONALIZED as
"library sequences the donor reads better than the host" — the bar asks
only that the transplanted core respond *measurably and selectively* on
those tokens (per the mission's wording: higher CE there than host-bare),
not that it improve the host. If the pair turns out to teach us the
opposite (the donor core being *useful* on those tokens), the sign will be
reported as measured.

## 2. Setup measurements

| measurement | value |
|---|---|
| calibration pairs / tokens | 90 pairs / 3202 tokens |
| host routing shares @L4 (slot picked) | slot **6**, share 0.3941 (shares [0.176, 0.231, 0.314, 0.197, 0.197, 0.212, 0.394, 0.279]) |
| donor-side share of the picked donor expert | donor slot **3**, share 0.4663 (shares [0.235, 0.225, 0.212, 0.466, 0.226, 0.216, 0.178, 0.241]) |
| host↔donor ffn-input mismatch (1 − cos): mean / p50 / p90 | 1.0066 / 1.0063 / 1.0237 |
| mismatch-window pool train / hold tokens | 1281 / 320 |
| donor-class fact set (margin CE_donor − CE_host) | teach/pet_dog (−1.5524), teach/metric_units (−0.3011), probe_t/tea_tin (−0.1119) |

The layer-4 manifold check is the first numbers-only receiver for the
"family-divergent study bed" premise: at the SAME shape (d=1024, layer 4),
host and donor ffn-input hidden states over the same tokens are effectively
orthogonal (cos ≈ −0.01 ⇒ 1 − cos ≈ 1.00–1.02). The D1.5 divergence regime
reproduces at vk scale — the sandwich has genuine translation work to do, and
the identity init (naive transplant) is expected to be misaligned, not null.

## 3. Training traces

| arm | step 0 MSE / R² | final MSE / R² | held-out R² | held-out ρ |
|---|---|---|---|---|
| A (both sandwiches) | 960.63 / −18.91 | 9.96 / 0.794 | 0.781 | 0.811 |
| F (OUT-only fallback) | 960.63 / −18.91 | 8.33 / 0.827 | 0.664 | 1.014 |

The sandwiches learn well above the D1.5 class of numbers: held-out R²
0.78 (A) / 0.66 (F) vs the earlier cross-family cosine ceiling ~0.47.
Same-shape/same-tokenizer/same-family but mismatched-manifold makes the basis
problem learnable at vk scale — the training question was closed, the
deployment question (below) was not.

## 4. Bars

CE table (nats; Δ = variant − bare):

| probe set | bare | naive | arm A (Δ) | arm F (Δ) |
|---|---|---|---|---|
| flat_0 | 3.9505 | 4.4029 | 4.0546 (**+0.1041**) | 4.0634 (**+0.1129**) |
| flat_1 | 3.8707 | 4.5260 | 3.7981 (−0.0726) | 3.7761 (−0.0946) |
| flat_2 | 5.0361 | 5.5356 | 4.9771 (−0.0591) | 4.9140 (−0.1222) |
| donor-class facts (3) | 5.0349 | 5.1630 | 5.1290 (+0.0941) | 5.1534 (+0.1185) |

| variant | routed share @ slot 6, flat | routed share @ slot 6, facts | Bar A (≤ +0.05 each) | Bar B (≥ +0.10 & route ≥ 0.05) |
|---|---|---|---|---|
| naive | 0.403 | 0.340 | — (reference) | — (reference) |
| arm A | 0.403 | 0.340 | **FAIL** (flat_0 +0.1041) | **FAIL** (+0.0941 < +0.10) |
| arm F | 0.403 | 0.340 | **FAIL** (flat_0 +0.1129) | **PASS** (+0.1185) |

Two honest qualifiers, reported not relitigated. (1) flat_1 and flat_2
*improved* slightly under both trained arms (−0.06..−0.12 nats) while the
naive transplant degraded all three — consistent with "training removed
most of the naive damage + this eval size carries ~±0.06–0.1 nats of
probe-to-probe sensitivity", and NOT yet evidence of donor usefulness; the
parity bar being failed by the SAME order as that sensitivity spread is a
bar-calibration fact for any retry. (2) Arm A's donor-class delta (+0.0941)
is not cleanly selective either — flat_0's delta (+0.1041) exceeds it; so
even Bar B looked at flat has a selectivity caveat for arm A (arm F passes
B with max flat delta +0.1129 < facts +0.1185, barely).

## 5. Verdict

**FAIL — TYPE-2 parity-broken (both arms).** The proposition arm A misses
Bar A on flat_0 (+0.1041 > +0.05) and Bar B by 0.006 nats (+0.0941 vs
+0.10); the pre-declared single fallback arm F misses Bar A the same way
(+0.1129) while passing Bar B (+0.1185) — and arm F's pattern (donor
response visible, parity not held) is exactly the "live but not harmless"
combination the two-bar feasibility design exists to reject. Autopsy
exclusions: not TYPE-4 (organic routing through the slot stays 0.34–0.40),
not TYPE-1 (held-out ρ = 0.81 — the core is alive, not killed), not TYPE-3
(arm F does show the donor-class response, so selective response is
reachable; parity is the binding constraint).

What the arm establishes for the swarm, with numbers: (i) the vk↔470m
manifolds at equal shape are orthogonal at the FFN gate (mismatch ≈ 1.01);
(ii) a 2 × (1024×1024) sandwich reaches R² ≈ 0.78 held-out functional
mimicry against an ACTIVE host expert on 1.3k mismatch-window tokens —
translation is learnable at this scale, unlike D1.5's cross-family bed;
(iii) an R² ≈ 0.78 fit is NOT enough to host a foreign core at a 39.4%-
traffic slot within ±0.05 nats on every flat probe — parity is the binding
constraint, liveness is not (ρ 0.81, arm F Bar B pass); (iv) the naive
literal transplant at this pair is mild breakage (+0.45…+0.66 nats), not
catastrophic — layer-4 slot influence is of the hundredths-of-nats order at
this model, so parity impairment and measurement floor are the same
order. One fallback (arm F) was exercised as declared; the branch now stops
per protocol with this stated negative. Follow-ups (queued, not run):
lower-traffic slot (parity bar is far easier at a 5% slot than at 39.4%),
or a longer-window fit; a bar recalibration against the measured probe
noise floor must come first either way.
`lab/FAILURES.md` entry: 2026-10-20 — transplant-sandwich-vk.
