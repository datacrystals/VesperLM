# VesperLM — Handoff for the Next Agent (updated 2026-10-11)

**ONECLICK FREE LANE (2026-10-10):** launch = `python3 pod/oneclick_run.py --notebook pod/oneclick_boot.ipynb --job-url https://raw.githubusercontent.com/datacrystals/VesperLM/main/lab/oneclick_jobs/smoke.json --minutes 50` (free AMD OneClick GPU box, 6h max / 10min idle-reap, keepalive + RESULT_JSON exfil via cell stdout; job JSONs in `lab/oneclick_jobs/`, see `pod/ONECLICK.md` §9–10).
**STATUS:** orchestrator + boot notebook built, local end-to-end dry-run green; live smoke **BLOCKED** — create works but the free tier never schedules (~5.5h cumulative `pending`, 1 brief ready window ever): `gh-ca7ead92` (2026-10-09 22:58 UTC) starved ~4.5h then went `not_found` (expired/reaped); create #2 `gh-10694136` (2026-10-10 05:38 UTC) sat `pending` 50min → FATAL at boot timeout. **GOTCHA: `create()`'s wall deadline derives from `--minutes`, so `--boot-timeout-min 85 --minutes 50` still dies at T+50m — next retry: `--minutes 150 --boot-timeout-min 120`.** GPU count / box egress / trainer-on-box UNVERIFIED. 2026-10-10 15:00 UTC: **all 3 queued instances REAPED after 8.5h/4.5h/3.5h pending** (gh-10694136, gh-9f9f8703, gh-dff85cad — incl. the official ppocr demo notebook, which pends identically: browser-parity verified from launcher JS, same POST body, no cookies — user "instant" browser access = capacity-window luck or per-IP slot). Cumulative ~12h queue starvation, 1 window ever. Pending instances ARE eventually reaped (~8h). Lane verdict: heartbeat-cadence lottery only, 1 fresh creation/day at a different hour, zero reliance. Retry cadence in `pod/ONECLICK.md` §10. Treat the lane as heartbeat-cadence only, never merge-gating.
**RULES:** no credentials on the box; never commit instance URLs (Jupyter token embedded); admin endpoints off-limits; laptop GPU stays free (all GPU work runs on the box).

**USER DIRECTIVES (standing):** (1) maximally delegate implementation to subagents
(cheap model) — the main agent architects, reviews, integrates, and manages droplet
budget. (2) TOTAL AGENTIC FREEDOM (2026-10-08): never stop and wait on the user —
make routine decisions, keep the program moving; the user drops in occasionally.
A recurring heartbeat cron (every 2h, :43) drives autonomous progress; re-create it
before its 7-day stale expiry. Carry both directives forward through compactions.
(3) FAILURE PROTOCOL (2026-10-09): never stop on failure — document it in
lab/FAILURES.md (entry template + fallback tree live there), pick the next
untried branch (cheapest first), and queue it in the same heartbeat cycle.
Only interrupt the user for budget or architecture-direction decisions.
(4b) COMPUTE PREFERENCE (2026-10-10): prefer the FREE AMD OneClick lane
(pod/oneclick_run.py — see pod/ONECLICK.md) for any droplet-eligible job that
fits its constraints (<=6h, queue-tolerant). Fall back to paid devcloud
droplets only when oneclick is starved/unfit. Poll cadence: attach-check per
heartbeat; creation budget 2/day unless user lifts it.
(4) DESIGN SPECS approved 2026-10-09: MODULAR_MOE.md §8 (episodic memory as
experts, gates G0-G4, G0 done) and SUBSYSTEMS.md (drives/emotions control
layer, gates E0-E4; E0 = telemetry instrumentation only, do it first and
cheap). Build order: laptop farm queue → G1 → E0 can start anytime (logging
only, no GPU).

**§8/SUBSYSTEMS FRONTIER (2026-10-10, current):** G1b PASS (passport-native
recipe) → SHAPE pivot: token-substitution episodes PASS at N=8 (1.071) and N=16
(0.737) — `lab/memory_shaping.py`, 55fe242 → **spine-portability SOLVED
(c58c900): delivery collapse on real 8-base spine was owner_mass below the
delivery cliff at scoring positions; fix = owner_mass 0.55→0.85 closing recal
→ retention 0.397→1.100, base CE ~+1.5%**. KL sweep saturates (905f236: KL
3→30 gives +8.9%→+1.5% then flat) — <1% CE bar NOT reachable output-side;
residual is routing displacement, needs addressing-side change (hierarchical)
or an explicit bar re-scope; on the real spine with the fix the cost is only
~+1.5%. **Next: end-to-end 8-base pipeline run with owner_mass 0.85 in-recipe
(natural G3-confirmation), then G4.** SUBSYSTEMS: **E0 DONE** (9aae150,
Common/telemetry.py, off-by-default VESPER_TELEMETRY=1). Next: E1 needs E0
data first.

---

## 2026-10-11 — Door 5 PROVENANCE ARMS (the named lever): (i) 16-site wrap of the distilled cores at the pinned lr 3e-3 NaNs in 50 steps — recipe does not scale across sites; (ii) distill-then-deliver NO-REFIT fit-side GREEN (cores bitwise-frozen, lam 0.817) with the gate table still measuring at the 2h cap — **detached jobs running** (MI50 GPU 4)

**Harness `lab/door5_provenance.py`** (fork; imports d0/d1/d4 + door25's
`WrappedForeignExpert` unmodified; synthetic 12/12). Both arms consume the
Door-3 distilled artifact (`/data/tliao/graft_d3_out/plugin_distill_olmoe.pt`,
16 cores fp `656ca50e593ef798`) at the EXACT distill split (teacher-cache
windows, home = code+math, 47,104 train / 31,744 held-out tok). Masked
baselines reproduce door3 to 6 decimals (held 1.8295 / train 1.9235).

| arm | status at cap | headline |
|---|---|---|
| (i) wrap, 16 sites, lr 3e-3, wd 0 (pinned Door-2.5 recipe) | DONE | Phase A **NaN abort @ step 49/250** (guard); post-abort delivered state all-NaN (quality nan vs masked 1.83/1.92, routing mass NaN 15/16 layers); cores bitwise-frozen; wall 807 s |
| (ii) deliver, no refit (contract b, lam learned/layer) | fit DONE, measure RUNNING | cores bitwise-frozen; lam fit 150 steps @1e-2 clean; lam mean 0.817 (0.23..1.14); rows round-2 p_new_code 0.76-0.91 / p_new_base 0.005-0.014; λ-fit step-0 CE **0.759** at lam 0.49 vs masked train 1.92 |
| (i') wrap, single site (host L6 / core 6), lr 3e-3 — the EXACT Door 2.5 control | QUEUED detached | `door5_wrap_single.json` |
| (i'') wrap, 16 sites, lr 1e-3 | QUEUED detached | `door5_wrap_all_1e3.json` |

**Findings so far.** (1) The pinned Door-2.5 LR is a single-site recipe:
16 simultaneous thin wraps at 3e-3 explode exactly like door4b's fresh
adapters (ce_batch0 0.74 -> 11.03 in 25 steps) and NaN by step 49 — the
93188b0 guard is load-bearing for any multi-site 3e-3 fit. (2) The
provenance signature is already visible in-sample: the distilled cores
deliver home CE far below masked at fit step 0 under BOTH delivery
contracts (door4b history 0.50 @ lam 0.5 code-only protocol; door5 λ-fit
step-0 0.759 @ lam 0.49 on the code+math slice) — i.e. the skill lives in
the frozen core, before any training, exactly as in Door 2.5. Confirmation
of provenance as THE quality condition awaits (i') and (ii)'s gate table.

**Read the live results:** box `/data/tliao/graft_d0_out/door5_{wrap,deliver}.json`
(+ `door5_wrap_single.json`, `door5_wrap_all_1e3.json`), logs
`door5_run.log` / `door5_run2.log`; launchers `door5_launch{,2}.sh`.
Numeric-only per the provider safety note. Ops: detached `setsid nohup`,
incremental JSON, HIP shims, fp16, GPU 4 only.

---

## 2026-10-11 — Door 4b RECIPE V2: quality still FAILS in both arms — plug-in quality story NOT closed; binding constraint = expert provenance (pretrained core), not the delivery contract

**Final grab-story experiment** (`lab/door4b_fit_v2.py`, MI50 GPU 4, detached,
incremental JSON `/data/tliao/graft_d0_out/door4b_{fresh,door3}.json`): recipe
v2 = contract (b) additive delivery (λ learned/layer, stock scale preserved)
+ Door-2.5 fit (task-loss CE, **wd 0**, neutral_w 300, clip 1.0, 250 steps,
post-training on-trajectory row recal, held-out quality). Two arms on the D1
code plug-in + the Door-3 expert source:

| gate | bar | fresh rank-64 adapters (4.2M, lr 3e-3) | door3 distilled core refit (100.7M, lr 1e-3) |
|---|---|---|---|
| quality held-out (train) | delivered<masked | 9.25 vs 1.74 (5.67 vs 1.26) ✗ | 8.88 vs 1.74 (6.49 vs 1.26) ✗ |
| parity overall | ≥0.95 | 0.683 ✗ | 0.843 ✗ (fire 0.740 / nonfire 0.862) |
| retention | ≥0.70 | 0.890 ✓ | 1.015 ✓ |
| contamination (per-layer / any-layer base) | ≤0.30 | 0.061 ✓ | 0.052 ✓ / 0.152 ✓ |
| utilization | >0.50 | 0.692 ✓ | 0.672 ✓ |

**Verdict: the plug-in quality story is NOT closed — and the causal story is
corrected**: the delivery contract was never the binding constraint. Door-2.5's
existence proof (masked 1.072 → plugged 0.439) used a **pretrained frozen
expert core** (11.8M Phi-mini-MoE SwiGLU) + thin task-loss adapters under the
MASS contract. Every from-scratch/refit delta fails quality **in-sample**
across 3 contracts and 6 fit recipes (D1 SwiGLU ×4; linear adapters;
distilled-core refit — the refit double-counts a replacement-trained core
under additive delivery). lr 3e-3 is proven only for thin-adapter-around-
frozen-core (fresh swiglu at 3e-3 NaNs ≤25 steps). Firing-split caveat:
non-firing base tokens are not bit-identical under contract (b) — fired
neighbors contaminate them through attention (0.862 at 15% any-layer firing).

**Next (queued): expert provenance as the lever** — (i) Door-3 core refit
under its NATIVE delivery (no double-count) + Door-2.5 thin-adapter wrap;
(ii) distill-then-deliver with no refit; (iii) bigger data/steps budget.
Door-2.5's wrap geometry + task-loss fit is the recipe to copy exactly.
Platform: NaN guard added (fresh-swiglu run-1 archived), split-metric capture
fixed (membership must be read inside the pass), any-layer firing reported.

---

## 2026-10-11 — Door 4 DELIVERY CONTRACT head-to-head: additive-λ wins, quality still open (expert-fit bottleneck confirmed in-sample)

**Setup:** the D1 s6_quality + Door-3 blocker (train "stock+e_new", deliver
"~0.09·stock+0.91·e_new") tested against its two named fixes on the D1 code
plug-in (`lab/door4_delivery_contract.py`, MI50 GPU 4 only, fp16, HIP shims;
d0/d1 imported read-only — sibling lanes own those files). Bars: parity
≥0.95 / retention ≥0.70 / contam ≤0.30 / quality = held-out CE delivered <
masked. Full numbers in `lab/FAILURES.md` door4 entry + box
`/data/tliao/graft_d0_out/door4.json`.

| gate | (a) delivered-mass fit | **(b) additive λ (learned 0.47)** | winner |
|---|---|---|---|
| base parity | 0.9467 | 0.8870 | (a) |
| retention vs solo | 1.055 | 1.070 | tie |
| contamination | 0.048 | 0.057 | tie |
| §6 utilization | 0.195 | **0.593** | (b) |
| §6 quality held-out (train) | 11.90 vs 1.74 (16.50 vs 1.26) | **7.46 vs 1.74 (4.48 vs 1.26)** | (b) |
| verdict | FAIL | FAIL | — |

**Contract verdict: (b) additive fixed-weight delivery is the winner to carry
forward** — stock merge stays bit-identical to the masked control (mass
confound gone by construction), quality gap roughly halves, utilization
passes. **But the quality bar fails under BOTH contracts — and the blocker is
now localized to the expert fit, in-sample**: even on its own training slice
the delivered CE is worse than masked (16.5 / 4.48 vs 1.26) — a frozen-trunk
16-layer SwiGLU delta on 12k tokens with out_pen 300 oscillates and stays
near-zero. Removing the mass-competition explanation shrinks the gap but does
not close it; the plug-in quality story is an adapter-training-recipe problem
(data volume, penalty balance, schedule), not a router-contract problem.
Parity damage under (b) concentrates on the 5.7% of base tokens where the
plug-in fires (cos min −0.90) — a helpful selective delta moves quality and
parity together. Also measured: post-training on-trajectory row recal (new in
this cycle — deep-layer rows go stale when the trained delta drifts h)
partially recovers utilization but does not fix it in (a).

**Next (queued): expert-fit recipe v2 under contract (b)** — LoRA-style
budget (more data, 1-2k steps, LR schedule), contrastive neutrality terms
with independent weights, λ swept per-layer. Door-3 distilled expert not
re-run this cycle (same fit bottleneck expected). Platform notes: runs
detached `setsid nohup` on GPU 4 per ops rule; incremental JSON; the additive
merge needs λ cast to the activation dtype (fp32 λ promoted the residual and
broke fp16 attention — fixed in-file). Docs numerical-only per provider
safety note.

---

## 2026-10-11 — MODULAR_MOE §10.2 **Door 2.5 first experiment (the grab): FOREIGN weights ARE hostable verbatim — quality sub-gate PASSES (code CE 1.072 masked → 0.439 plugged → 0.101 forced) — first quality pass of the line; margin-retention bar degenerate (solo gain −0.28), skill generalization NOT shown (held-out code CE +0.03)** (MI50 box, GPU 3 only, fp16)

**Q (§10.2): can a FOREIGN model's expert core be hosted VERBATIM behind
learned thin adapters?** Donor core = Phi-mini-MoE @ f620b32c **layer 13,
expert 8** SwiGLU (chosen by home-domain fit: D2 RouteStats profiling on the
graft_d2 slices — code-rate lift 16.5 over base, top of 32×16 candidates;
w_gu 1920×4096 = [w1;w3], w_down 4096×960 = w2, **11,796,480 params FROZEN
verbatim**). Harness `lab/door25_wrap_expert.py` (fork of the D1 path —
imports `graft_d1_plugin.py`/`graft_d0_parity.py` unmodified), synthetic
15/15. Plug-in: OLMoE-1B-7B-0924 @ 6d84c485 + `registry_olmoe_6d84c485.pt`,
**single site at host layer 6** (depth-matched to donor L13/32), 65th passport
row + PluginMerge forced/plugged exactly as D1.

**The wrap (§10.1/§10.2 geometry, brief's naming):**
`x(2048) --host_in--> c(64) --adapter_up--> u(4096) --CORE(frozen)--> v(4096)
--adapter_down--> c'(64) --host_out--> delta(2048)` — adapter_down/adapter_up
are the "(d_canon=64 ↔ their d_model=4096)" pair; host_in/host_out are the
§4.2 d_model↔d_canon format maps the D1 host interface needs. **Trainable
786,432** (131,072+262,144+262,144+131,072) vs 11.8M frozen; rank-64
factorized A_in/A_out. host_out zero-init (identity at birth).

**Training (task loss — NOT feature matching; the D1.5 lesson):** Phase A
forced-dispatch code-CE + out_pen 300·mean‖e(h_base)‖², **lr 3e-3, wd 0.0,
clip 1.0**, 250 steps (739 s) — the D1 defaults (wd 0.01/clip 1.0/lr 1e-4)
starve a thin wrap to |y|~5e-4 (below fp16 trunk resolution; FAILURES
`door25-wrap-training-starvation`; arm sweep 4.6e-4 vs 1.0e-1 at wd 0).
Phase B row recal **owner_mass 0.35 / reject_w 10 / mass_w 2** (the att-4
winner) + on-trajectory round (201 s): p_new_code 2.2e-8 → 0.544 → **0.851**,
p_new_base 0.022. Phase A curve: ce_batch0 0.951 → **0.101**, out2 → 1.17e-4,
mean|y| 8.3e-3. Total wall 1237 s, peak VRAM ~18 GiB (GPU 3).

| gate | bar | value | verdict |
|---|---|---|---|
| base parity cos (stock vs plugged) | ≥0.95 | **0.99846** (min 0.601, 59,392 tok) | PASS |
| graft parity (OLMoE graft vs plugged) | ≥0.95 | **0.99899** | PASS |
| retention vs solo (D1 margin probes) | ≥0.70 | **0.011** — solo gain **−0.28** | FAIL-as-written (degenerate) |
| contamination (base → row) | ≤0.30 | **0.0647** | PASS |
| §6 utilization (code → row) | >0.50 | **0.8308** | PASS |
| **§6 quality: code CE active vs masked** | beat | **0.439 vs 1.072** (forced 0.101) | **PASS — first in the line** |

Extra reads: in-situ code routing mass **0.733** (vs 0.35 target — CE term
still dominates, but better than D1's 0.853), base mass 0.035; base CE
2.311→2.316 (Δ+0.005 — no-regression clean); CE-delivery retention
(lib/solo gains in CE space) **0.652** (vs 0.70 bar) — the shortfall is the
delivery contract (delivered composition 0.27·stock+0.73·e_new ≠ trained
stock+e_new), the same shared blocker the GPU-4 lane is testing. Held-out
(`door25_heldout.py`, plugin .pt reload): code_train 1.072/0.439/0.101,
**code_heldout 1.587 masked / 1.621 plugged / 1.950 forced (+0.033/+0.363)**,
math_cross 2.477/2.483/2.613 (+0.006 plugged — out_pen neutrality holds).
The 11k-token home fit therefore **memorizes**: the interface-level grab is
real (a frozen foreign core delivers a large useful delta through 786k
adapters — exactly what D1's toy and Door 3 did NOT achieve), but the
*skill* is not yet a generalizable grab — needs more calibration data and/or
the delivered-mass fix before the specialty claim.

**Verdict (§10.2 success criterion): GRAB WORKS at the interface — the
Door-3 trigger ("adapters can't bridge") did NOT fire.** Exact foreign
weights are hostable; quality beats masking (the D1 sub-gate) for the first
time in the line. Two caveats written into the record: (1) the D1 retention
bar as written is degenerate here (negative solo gain on held-out hard-token
margin probes — the ratio is not meaningful; CE-space analog 0.652); (2)
held-out home CE is neutral-to-slightly-bad, so "Phi's code specialty was
grabbed" is NOT yet supported — follow-ups: larger/multi-slice calibration
(the wrap-vs-distill comparison the main agent is compiling should weight
this), multi-site (16-layer pack with the same frozen core per site), and
the delivery-contract fix for the 0.65→0.70 retention gap.

**Ops:** GPU 3 only (`HIP_VISIBLE_DEVICES=3`), detached `setsid` (survived
session caps), incremental JSON (`phase_a/phase_b/complete`) at
`/data/tliao/graft_door25_out/door25_wrap.json` + `door25_plugin.pt` +
`door25_core.pt` (box) / `lab/imported/graft_door25/` (home, untracked).
**gfx906 wall (FAILURES `door25-gfx906-backward-holes`):** trunk backward
dies at B*T ∈ {256,768,1024,4096} tokens on this box (D1's default bs2×512
hits 1024!); all training on the verified 2048-token shape (bs4×seq512).
Prototype init caveat found in diagnostics: registry layer_scale for L6 is
**negative (−1.33)**, which flips the prototype-init sign (score_new −44 vs
stock 0.68) — 400 recal steps overcome it; a sign-aware init is a cheap
future fix.

---

## 2026-10-11 — MODULAR_MOE §9 **D2 follow-up, score-margin-aware recal: CLEAN NEGATIVE (curve monotone slightly down: MoE-cos 0.864→0.854, agree 0.651→0.623 over margin_w 0/2/5/10) — but the exact-init control PASSES every gate (0.9966 / 0.9881): the aux bars are reachable, the prototype+recal fit is the gap** (MI50 box, GPU 5 only, fp16)

**Named follow-up from FAILURES `d2-crossfamily-perlayer`:** weight the recal
MSE on near-threshold expert pairs (score-margin-aware recal) to close the
Phi-mini-MoE per-layer gap (bars: MoE cos > 0.9, top-k agree > 0.8) without
breaking the main bars (cos ≥ 0.95, |dCE| ≤ 0.05). Harness
`lab/graft_d2_crossfamily.py` gains `--recal-margin-w` / `--recal-margin-tau`
(weights `1 + w·exp(−min_c |s*−cut_c|/τ)`, mean-normalized; cuts = rank-1/2
and rank-k/k+1 midpoints + the sparsemixer mask edge `max·(1−2·jitter)`) and
a `margin_analysis` block in the parity phase (noise-matched agreement
baseline). Mock self-test **18/18**. Four detached GPU-5 arms on the existing
registry recipe (slices `lab/graft_d2_slice/`, donor @f620b32c, recal 300×2,
88,064 val tokens), results `lab/graft_d2_margin_recal.{json,md}`:

| arm | MoE cos | top-k agree | Jaccard | logit cos | ΔCE | aux bars |
|---|---|---|---|---|---|---|
| BEFORE (uniform MSE 300×2, D2 att-1) | 0.8640 | 0.6509 | 0.761 | 0.99863 | +0.0303 | FAIL |
| margin_w=2 (300×2) | 0.8602 | 0.6412 | 0.754 | 0.99857 | +0.0308 | FAIL |
| margin_w=5 (300×2) | 0.8571 | 0.6320 | 0.748 | 0.99859 | +0.0329 | FAIL |
| margin_w=10 (300×2) | 0.8540 | 0.6233 | 0.742 | 0.99856 | +0.0344 | FAIL |
| uniform MSE 600×3 (budget control) | 0.8858 | 0.7017 | 0.797 | 0.99892 | +0.0190 | FAIL |
| margin_w=5 (600×3) | 0.8822 | 0.6902 | 0.789 | 0.99897 | +0.0205 | FAIL |
| **exact init (algebraic control)** | **0.9966** | **0.9881** | 0.992 | **0.99996** | **+0.0001** | **PASS** |

Main bars hold on every prototype arm (stretch 0.99 clears everywhere);
**the named lever is falsified** — margin weight moves per-layer strictly the
wrong way at the D2 budget, and at the matched 600×3 budget the uniform
control BEATS the margin arm (0.8858/0.7017 vs 0.8822/0.6902), so the
higher-budget gain (+0.022 cos / +0.051 agree) is fit budget / on-trajectory
coverage, not margin weighting. Best prototype arm still misses the aux bars
(0.886 < 0.9, 0.702 < 0.8). Root cause (from `margin_analysis`): the residual
error is
on-trajectory score error from hidden-state drift (val-trajectory score RMS
0.05–0.27/layer vs recal cal-fit RMSE 0.014), and `frac(gap23 < 2σ) = 0.87–
0.99` — nearly every token is flip-prone at that error level, so re-weighting
which pairs get fit cannot help, only smaller error can. Diagnosis correction
to the D2 write-up: in eval, sparsemixer INDICES are pure top-2 ranking (the
relative-threshold mask only moves the multiplier softmaxes — donor source
`sparsemixer()`), so the flips are rank-2/3 near-ties, not threshold flips.

**Metric proposal (the earlier recommendation, now measured).** Family-
relative agreement is the right *form* for hard-selection routers, but the
baseline must be the **measured exact-init ceiling** — `(agree − 1/C(E,k)) /
(agree_exact − 1/C(E,k))` — not isotropic noise-matched at the graft's σ:
that model predicts agree 0.40–0.48 where the graft scores 0.64–0.70 and the
true ceiling is 0.99 (graft score errors are low-rank/structured). On this
donor the fp-noise ceiling is 0.9881, so the absolute 0.8 bar was already
≈0.81-of-ceiling — **the OLMoE bars are NOT unfairly calibrated for
Phi-mini-MoE**; BEFORE sits at 0.656 of ceiling (best prototype arm 0.708). Practical consequence for §9.6: **for a
plain-linear donor gate, ship algebraic passports (exact init — the D0 core's
`init_passports_exact`) and the per-layer gap vanishes** (registry
`registry_phimoe_exact.pt`, 8.4MB, same format); prototype+recal is for
routers with no closed form. Open lever if prototype-only is ever required:
on-trajectory cal coverage (more pairs / val-like cal text), not margin
re-weighting.

**Next for §9:** D2 attempt 1 + this follow-up together close the per-layer
question: the interface + registry pass all five gates when the passport rows
are algebraic. Next step is the D1-style plug-in onto the phimoe registry
graft (now on a fully passing base), or the second-foreign-family probe.

---

## 2026-10-11 — MODULAR_MOE §10 **Door 3 (behavioral distillation): formal D1 bars PASS on the mass-cal recipe, s6 quality sub-gate FAILS — but the skill IS cloned at forced dispatch (held-out KL-to-donor 5.17→2.59); delivery composition, not expressiveness, is the blocker** (MI50 box, GPU 4 only, fp16)

**Door 3 = the §10.3 behavioral-distillation counterpart to Door 2.5's
weight-wrap**: train a FRESH OLMoE-graft plug-in expert (the D1 path) to
imitate a donor skill's behavior, so the parent can compile a head-to-head
grab table (wrapped foreign weights vs distilled behavior). Harness
`lab/door3_distill_expert.py` (imports the D0 core + D1 plug-in path — do not
re-break those; synthetic 17/17). Teacher **Phi-mini-MoE @ f620b32c**
(slimmoe→native load vendored from `graft_d2_crossfamily.py` so this lane does
not depend on the wrap lane's WIP edits), host OLMoE-1B-7B-0924 @ 6d84c485 +
`registry_olmoe_6d84c485.pt`, **home = codeparrot + open-web-math slices of
`/data/tliao/graft_d0_slice/{code,math}.txt` — the same data files the wrap
lane uses (fairness contract)**. Run: detached `setsid`, GPU 4
(`HIP_VISIBLE_DEVICES=4`), wall **1381 s**, peak VRAM **19.8 GiB**, box
`/data/tliao/graft_d3_out/door3_distill.json` (+ `lab/imported/graft_d3/`,
untracked).

**Cross-tokenizer distillation definition** (both tokenizers differ; recorded
in the JSON): shared support S = strings that single-token decode identically
in both vocabs (**12,480 strings**); positions aligned by char offsets (soft
target for student token k = teacher next-token dist at the teacher token
covering k's start; exact-boundary frac 0.9217 code / 0.9806 math, teacher row
coverage 1.0); both dists log-softmax renormalized over S; loss =
**KL(teacher‖student)**. Teacher soft targets precomputed once (VRAM-safe:
teacher and host never co-resident), fp16 cache, 148 s.

**Recipe (as specified):** forced-dispatch fit of 16 zero-init SwiGLU deltas
at `KL + 0.3·GT-CE + 300·off-domain L2` (200 steps @ 1e-4), then the D1
mass-calibration recal **owner_mass 0.35 / reject_w 10 / mass_w 2**
(400 steps + 200 on-trajectory). Phase-A KL 155.5 → **1.44** (train);
Phase-B p_new_code 0 → 0.774, p_new_base 0 → 0.024 (after the on-trajectory
round).

**Gate table (for the §10 comparison — wrap lane reports separately):**

| gate | bar | Door 3 (distill) | verdict |
|---|---|---|---|
| base parity cos | ≥0.95 | **0.9510** (std 0.129, min −0.67; base CE Δ+0.77) | PASS (tight) |
| retention vs solo | ≥0.70 | **3.216** (gains +0.27 solo / +0.88 lib) | PASS w/ att-3-style lib>solo artifact caveat |
| contamination | ≤0.30 | **0.1328** | PASS |
| §6 utilization | >0.50 | **0.9592** (in-situ home mass 0.911) | PASS |
| §6 quality CE active<masked | must hold | held-out **8.068 vs 1.830** (Δ+6.24) | **FAIL** |

Quality per slice (held-out): code 1.525→9.243 (Δ+7.72), math 2.133→6.895
(Δ+4.76). Base CE 2.345 masked → 3.118 plugged.

**Distill-native fidelity read (the Door-3 headline number):** held-out
KL(teacher‖student) over 31,682 positions — masked **5.17** → solo/forced
**2.59** → plugged **6.63**. I.e. **the donor's behavior IS successfully
distilled into the fresh expert at the trained (forced) composition — a 50%
KL reduction — but passport-mass delivery degrades it below the masked
baseline.** In-situ home routing mass 0.911 vs the 0.35 target (the CE router
term dominates, same as D1 att-4), and `norm_topk_prob=false` makes that mass
the merge scale: delivered composition ~0.09·stock + 0.91·e_new ≠ trained
`stock + e_new`. Root-cause detail + the named next branch (expert-fit at
delivered-mass / additive fixed-weight delivery) in `lab/FAILURES.md`
`door3-distill-quality`.

**Status for §10:** Door 3 delivers the universal fallback at the *expert*
level (behavior is hostable — any model's specialty becomes a plug-in expert
that halves KL to the donor) and FAILS at the *delivery* level exactly where
D1's toy expert failed — the shared blocker for the §10.3 vs §10.2 comparison
to read on quality is the softmax-mass delivery contract, not what the expert
was trained from. Formal D1 bars (parity/retention/contam) all PASS on the
mass-cal recipe, so routing is intact; only quality is open. Wrap-lane
comparison owner (GPU 3) should report the same five rows for the compile.

---

## 2026-10-11 — MODULAR_MOE §9 **D2 attempt 1 (cross-family graft): PASS on the user bars — passport interface generalizes to Phi-mini-MoE (sparsemixer router)** (MI50 box, GPU 5 only, fp16)

**D2 = graft a FOREIGN-FAMILY open MoE through the same passport/registry
interface.** Donor: **`microsoft/Phi-mini-MoE-instruct`** @ pinned rev
**`f620b32c0d3e8f7e76f57ccdaa88e0df8bc8bfcd`** (7.65B params, 32 layers,
hidden 4096, E=16, top-2; Phi family / Microsoft "slimmoe"). Chosen over the
lane's listed candidates on the 1-card constraint: Qwen1.5-MoE-A2.7B (14.3B),
DeepSeek-V2-Lite (16B), MiniCPM-MoE-8x2B (27.7GB bin) all need ~29–32GB of
fp16 weights alone (2-card device_map is forbidden in this lane). Phi-mini is
the largest native-transformers MoE that fits one 32GB card at fp16.

**Why this is a real cross-family test:** the router math differs —
**sparsemixer** (relative-threshold-masked sequential top-2; the two
multipliers come from *different* masked softmaxes and do not renormalize
together) vs OLMoE's plain softmax-top-8-no-renorm; expert block is a
ModuleList of SwiGLU MLPs (w1/w3/w2) vs OLMoE's grouped 3D bank; router lives
at `block.router` vs `block.gate`. Harness `lab/graft_d2_crossfamily.py`
imports the passport/metrics/registry core from `lab/graft_d0_parity.py` and
adds the family-adapter layer (block discovery, graft-gate contract, taps,
selection semantics — donor's own `sparsemixer` is imported so semantics can
not drift). Mock self-test **14/14** (sparsemixer mock + OLMoE-adapter
regression). Checkpoint loads via native transformers `phimoe` after a
mechanical key remap (`w1;w3 -> gate_up_proj` (E,2I,H), `w2 -> down_proj`
(E,H,I), `block_sparse_moe.gate -> mlp.router`) — the shipped custom modeling
code hard-imports `flash_attn` (unavailable on gfx906); remap verified by
sample-text CE 3.69 nats + exact-init 1-batch parity cos 0.999999. gfx906: the
two D0 shims cover native PhimoeExperts via the same duck-type; no new walls.

**Verdict on 88,064 val tokens** (gate-scale slice from the D0 builder recipe
re-run with the Phi tokenizer: 105k domain + 88.9k val tokens — codeparrot /
open-web-math / wikitext-2 / ag_news-sci / oasst1-en), prototype init +
router-only recal 300×2:

| gate | bar | D0 (OLMoE) | **D2 (Phi-mini-MoE)** | verdict |
|---|---|---|---|---|
| final-logit cosine | >0.9 (stretch 0.99) | 0.99913 | **0.99863** | **PASS (stretch)** |
| \|ΔCE\| nats | <0.05 | +0.00075 | **+0.0303** (1.9054→1.9357) | **PASS** |
| per-layer MoE cos (mean) | >0.9 | 0.9952 | 0.8640 | FAIL |
| top-k set agree (mean) | >0.8 | 0.8366 | 0.6509 (Jaccard 0.761) | FAIL |
| purity ≥ stock−0.02 | every domain | pass | pass (margins −0.0001…−0.0005) | PASS |

**Answer to "does the passport interface generalize across families?": YES on
the user-stated bars** (cos 0.9986 > 0.9, clears the 0.99 stretch; |ΔCE| 0.030
< 0.05) — prototype+recal passports reproduce a foreign family's router to
logit level through the identical PassportBank/registry interface. The
D0-runbook *auxiliary* per-layer gates miss (0.864 / 0.651): sparsemixer's
threshold+argmax selection is harder to mirror bit-for-bit than OLMoE's smooth
softmax-top-k (score perturbations near the threshold flip discrete choices),
and per-layer error is U-shaped (edges 0.96/0.81, middle layers 13–21
0.80–0.85 / 0.53–0.65 — small score errors compound through depth then the
last layers reconverge). Recal did its job (MSE 3.76→0.0012 round 0,
0.005→0.0002 on-trajectory round 1); profile mushiness 0.923 (maximally
permissive purity baseline, same caveat as D0). Registry artifact
`registry_phimoe_f620b32c.pt` (8.4MB) + JSONs on box `/data/tliao/graft_d2_out/`
and copied home to `lab/imported/graft_d2/` (untracked).

**Registry reuse confirmed (OPS):** a detached `setsid` re-parity from the
surviving `.pt` (no recompute) reproduced the run exactly — cos 0.99861,
dCE +0.0294, layer 0.8638 / agree 0.6507, wall 613s. Peak VRAM 16.5GiB on
GPU 5; full pipeline wall 1310s. Harness now encodes fixtures in ≤8000-char
chunks (kills the donor tokenizer's "20717 > 4096" whole-file-encode warning)
and writes JSON incrementally (precompute → parity → complete) so a killed run
leaves usable results. `lab/graft_d2_runbook.md` has the full recipe + donor
pin + remap spec.

**Next for §9:** D2 attempt 1 closes the "interface generalizes?" question at
logit level; the per-layer gap (sparsemixer discreteness) is the honest caveat
— a fair follow-up is the same protocol with a *second* foreign family (e.g.
Qwen1.5-MoE layer-subset if the one-card fit can be proven) or D1-style plug-in
onto the phimoe registry graft to reuse the now-family-portable interface.

---

## 2026-10-11 — MODULAR_MOE §9 **D1 plug-in onto graft: PASS (owner_mass 0.35, reject_w 10, out_pen 300) — quality sub-gate open**

**D1 = one foreign expert (code-domain SwiGLU delta pack, 16 layers) plugged
into the D0 registry graft** (`registry_olmoe_6d84c485.pt`, OLMoE-1B-7B-0924
@6d84c485, MI50 box GPU 0 only, fp16, HIP shims reused). Harness
`lab/graft_d1_plugin.py` (Phases: forced-dispatch expert fit + contrastive
router recal with owner-mass target + on-trajectory closing round → measure
task bars (a) base parity (b) margin retention vs solo (§8 memory_shaping
convention) (c) contamination, plus §6 gates). Mock self-test 7/7.

**D1 VERDICT: PASS on all formal bars** (final run 2026-10-11 14:43 UTC,
~31 min GPU 0). Recipe that passes — the mass-calibration result of a 4-point
Pareto sweep: **owner_mass 0.35 + reject_w 10.0 + out_pen 300** (§8.4's
0.85 transplant fixed delivery but starved stock experts; 0.35 is the D1
operating point — §6 utilization is top-k MEMBERSHIP, not mass):

| gate | bar | att.1 (rw5, KL) | att.2 (rw10, pen10) | att.3 (mass .85, pen3000) | **att.4 (mass .35, rw10, pen300)** |
|---|---|---|---|---|---|
| base parity cos | ≥0.95 | 0.782 | 0.977 | 0.927 | **0.967 PASS** |
| retention vs solo | ≥0.70 | 0.989 | 0.487 | 3.622 | **0.934 PASS** |
| contamination | ≤0.30 | 0.277 | 0.091 | 0.213 | **0.105 PASS** |
| §6 utilization | >0.50 | 0.514 | 0.286 | 0.940 | **0.911 PASS** |
| §6 quality (CE active<masked) | — | FAIL | 2.97 vs 1.26 | 14.28 vs 1.26 | **12.58 vs 1.26 — FAIL (open)** |

Att-4 detail: margins pre −3.39 → solo −0.83 → lib −1.00 (gains +2.57/+2.40,
a clean delivery ratio vs att-3's 3.6 suppression artifact); base CE
2.47→3.37 (Δ+0.90); gate p_new_base 0.008 after the on-trajectory round
(teacher-forced gap closed again); in-situ code mass 0.853.

**Remaining quality sub-gate + named next branch:** the plugged row still
captures ~0.85 in-situ code mass (the CE router term dominates the 0.35
target) while carrying a tiny delta (out_norm2 ~5e-6), so stock experts get
starved on code tokens and average code CE cannot beat the masked control.
Next branch: **expert-fit at delivered-mass** — train the delta to compensate
at the routing mass it actually receives (or deliver additively at fixed
small weight instead of competing for softmax mass) so the plug-in adds value
rather than replacing stock contributions. Root-cause history + Pareto map:
`lab/FAILURES.md` d1-plugin-pass-1.

s6-gate semantics (what the harness measures): **utilization** = fraction of
home-domain tokens whose top-8 includes the plug-in row (membership, not
mass); **quality** = next-token CE on home-domain batches with the plug-in
routed vs the same expert masked (present, not a candidate).

**D1.5 (translator)**: unblocked and already run by the sibling lane (see its
section below). Box note: GPUs 1-2 user hackathon, 3-7 other lanes — all our
work on GPU 0 only, peak 17GB, ~30 min/run. Attempt-3 JSON archived on the
box as `olmoe_d1_attempt3_mass085.json`; final verdict in `olmoe_d1.json`.

---

## 2026-10-11 — MODULAR_MOE §9 D1.5 TRANSLATOR FIDELITY: FAIL vs 0.9 bar — variant B deprioritized (MI50 box, GPU 3 only)

**Gate D1.5 (§9.3a/§9.4) measured: telemetry-trained route-conditioned
translators do NOT reach held-out cosine 0.9 between feature bases; every
challenge pairing lands 0.55–0.69, which also trips §9.5's <0.7 in-domain
deprioritize rule → variant B (cross-model expert transplant via translators)
deprioritized to variant-A-only for now. D2 (cross-family co-location via
transplant) is NOT unblocked on this axis.** Harness
`lab/graft_d15_translator.py` (imports HIP shims/registry/GateTap from
`lab/graft_d0_parity.py` — do not re-break those), synthetic self-test 18/18
(laptop + box venv). Run on `tliao@100.64.0.7` **GPU 3 ONLY**
(`HIP_VISIBLE_DEVICES=3`, `ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/
rocblas/library`, fp16, venv `/data/tliao/vesper_sft/venv`), donor
`/data/tliao/OLMoE-1B-7B-0924`, D0 registry
`/data/tliao/graft_d0_out/registry_olmoe_6d84c485.pt`, gate slices
`/data/tliao/graft_d0_slice` (same `graft_d0_build_slice.py` domains:
chat/code/math/science/web). Artifacts (untracked): `lab/imported/graft_d15/
d15.json` + `d15_crossmodel.json` (+ box `/data/tliao/graft_d15_out/`).

**What was trained** (spec §9.3a: T: x_host→x_donor paired-activation
regression, route-conditioned on the graft's passport scores; both directions
measured; identity/mean baselines; linear = closed-form ridge, 2-layer = Adam
autograd only — no fla/Triton): two instantiations of "two bases":

1. **Within-donor cross-layer** (§9.3a sharpening 2, layer correspondence):
   pre-MoE features at OLMoE layers (host) vs (donor), 35,840 tokens, pairs
   2:12 / 6:12 / 12:12. Held-out cos (best arm = route-conditioned ridge
   linear): **0.5528** (gap 10), **0.6296** (gap 6); identity baselines
   0.2075 / 0.4172; same-layer **12:12 CONTROL = 1.0000** (identity+linear)
   → harness is sound and fidelity scales with depth gap. Strict splits:
   route-held-out (8 top-1 experts excluded) 0.5227 / 0.6098; domain-held-out
   (web) 0.3882 / 0.4882. Spec's transplanted-output metric (expert_8:
   cos(expert(T(x)), expert(x_native))): 0.349 / 0.464 raw, 0.490 / 0.608 in
   the expert's top-32 singular directions. Wall 22.6 min total (capture 25s).
2. **Cross-model (the §9.3a gate-of-record pairing)**: OLMoE-1B-7B-0924
   donor ↔ **ibm-granite/granite-3.0-1b-a400m-base** host (2nd small MoE,
   24 MoE layers, hidden 1024, 32 experts top-8; fetched to
   `/data/tliao/d15_models/` — NEVER commit), word-aligned paired activations
   (char-offset mean-pooling across each model's own tokenizer — different
   tokenizers make per-token pairing ill-posed), 18,529 paired words, donor
   layer 12 ↔ host layers 12/18. Held-out cos (route-conditioned linear):
   **0.6437** (h12) / **0.6885** (h18); train 0.724/0.757; p10 0.52/0.57;
   route-held-out 0.610/0.657; domain-held-out 0.536/0.587; expert-output cos
   0.474/0.553 (top-32 dirs 0.629/0.698). Per-route (top-1 expert) breakdown
   h18:d12: 37 routes ≥30 tok, macro 0.667, worst 0.556 (e56), best 0.898
   (e59) — no route reaches 0.9. Mirror direction (donor→host) best cell
   0.7030 (h18, cond). Wall 15.7 min (capture 6.4 min), peak VRAM ~18.4GB
   (both models fp16 resident) on GPU 3.

**Verdict table vs bars** (worst challenge pair; controls excluded from
gating): lane bar held-out cos > 0.9 → **FAIL** (0.55–0.69); stretch 0.95 →
FAIL; §9.5 deprioritize bar < 0.7 in-domain → **FIRES** (0.55–0.69 band);
spec output cos > 0.9 → FAIL (0.35–0.55 raw / 0.49–0.70 top-dirs);
route-held-out ≥ 0.7 → FAIL; domain-held-out ≥ 0.7 → FAIL. Route conditioning
helps mildly (best arm in every challenge pair is the route-conditioned
linear); 2-layer MLP is NOT better than linear (slight overfit) — within this
capacity class the residual is feature availability, not translator size
(consistent with §9.3a's falsifier: "the host spine lacks the features").
Train cos itself tops out at 0.72–0.76 (the map is not learnable even
in-sample), so raising translator capacity alone is unlikely to reach 0.9.

**D2 status: NOT unblocked by this gate.** Per §9.5 as written, variant B is
deprioritized to variant-A-only (one grafted host per donor family, mixed at
the serving layer) until a translator clears the bars. Untried follow-ups if
the line is revisited: soft attention over donor layers / per-(expert,layer)
translators (§9.3a sharpening 2), token-level pairing (1:1 subword words
only), larger MLP/residual translators, and the one-expert-manifold arm
(`--expert-scope`). NOTE the sibling D2 lane (`lab/graft_d2_crossfamily.py`,
GPU 5) is a *different* reading of D2 — portability of the passport/registry
graft interface to a foreign family (Phi-mini-MoE sparsemixer) — it does not
need translators and is not gated by this result.

**gfx906 note (new manifestation):** word-pooling `bool-mask + index_add_`
tripped the known async `hipErrorInvalidDeviceFunction` on GPU (same family
as the FAILURES.md `torch.sort` wall) → pooling shimmed to CPU (tensors tiny).
Full story in `lab/FAILURES.md` d15 entry.

---

## 2026-10-11 — MODULAR_MOE §9 registry/passport scaling sweep: SATURATES AT passport_dim 64 (= stock-gate rank); recal mandatory; smallest registry at cos>=0.99 is the D0 anchor (8.01MB) — reduced dims cut the plug-in cost, not the artifact

**The scaling table the registry story needed** (MI50 box GPU 4 only, fp16,
HIP shims auto, same D0 slice 97,280 domain + 86,016 val OLMoE tokens).
21 cells = passport_dim {8,16,32,64,128} x recal rounds {0=prototype-only,
1,2} + the D0 anchor (dim 2048, identity query map, 2 rounds — re-measured on
the same basis: cos **0.99913** / dCE **+0.00075** = D0 exactly) + SVD
rank-ceiling arms. **Full table + conclusion: `lab/graft_registry_sweep.md`;
compact JSON: `lab/graft_registry_sweep.json`** (harness
`lab/graft_registry_sweep.py`, wallet-free; wall **8,158s = 2.27h**, peak VRAM
**14.45 GiB**; recal wall/cell 0-248s, parity pass 217-362s).

**logit cos @ 86k val tok (dCE / top-k agree / MoE cos per cell in the md):**

| passport_dim | r0 | r1 r2 | registry MB |
|---|---|---|---|
| 8 | 0.662 | 0.862 / 0.871 | 1.04 |
| 16 | 0.662 | 0.937 / 0.940 | 2.07 |
| 32 | 0.663 | 0.972 / 0.980 | 4.14 |
| 64 | 0.663 | 0.986 / **0.9958** | 8.26 |
| 128 | 0.663 | 0.988 / 0.9969 | 16.51 |
| 2048 anchor | — | — / 0.9991 | 8.01 |

SVD ceilings (0 recal): 0.785 / 0.891 / 0.968 / **0.99995** / 0.99995 — the
stock gate is (64 x 2048), **rank exactly 64** (Frobenius energy 51/67/85/100%
at 8/16/32/64), so dim >= 64 is algebraically exact and dim 64 ≈ dim 128.

**Conclusions (numbers in md):** (1) **saturation at passport_dim 64** = the
stock gate's rank — above it parity is flat (0.9958 vs 0.9969 at r2), below it
quality tracks the singular spectrum. (2) **Prototype-only is dead** (cos
~0.66 at EVERY dim — init-limited, not rank-limited); recal is mandatory: 1
round gets dim64 to 0.986, the 2nd (on-trajectory) round crosses 0.99 — the
G2/§9.6 recal is the cheap compile (minutes of CPU Adam). (3) **Smallest
registry keeping cos >= 0.99 = the D0 anchor at 8.01MB** (cos 0.99913) —
reduced dims do NOT shrink the shipped artifact at the bar, because per-layer
W_q (d_r x d_model) dominates it (dim64 artifact 8.26MB > anchor; dim32 4.14MB
tops out at 0.980). (4) What does get 32x cheaper: **per-expert plug-in cost**
— a passport row is d_r floats (256B vs 8KB per expert/layer at d_r=64 vs
2048) while W_q is per-spine. Next knob if artifact size binds: shared/
quantized W_q (~0.78MB at dim64; `recalibrate_shared()` drafted in the sweep
script, UNTESTED). Honest caveats in the md (SVD arm is an unweighted
factorization — recal can beat it on-val below full rank; min-cos stays ugly
below the bar; top-k agree 0.63-0.67 at the pass cells vs 0.84 anchor).

**Ops (this run):** process survived TWO agent-session reaps under `nohup`
(the "died at grid_dim64_r2" reports were false — the log's last flushed line
is not process death; check `kill -0` ON THE BOX). Cells were only in the log
+ per-cell .pt until exit → **`lab/graft_sweep_salvage.py` + `--resume`
per-cell checkpointing now in `lab/graft_registry_sweep.py`** (flush after
every cell; FAILURES.md entry). Standing OPS RULE: launch remote GPU jobs
detached (`setsid nohup ... &`), watch result files not PIDs. No new gfx906
kernel walls (standard torch only; D0 shims carried).

---

## 2026-10-11 — MODULAR_MOE §9 **D0 GATE RUN: PASS** (MI50 box, GPU 0, 10.3 min)

**D0 (graft parity) PASSES — prototype+recal registry recipe reproduces the
donor's stock router.** Ran on the MI50 box (`tliao@100.64.0.7`, GPU 0 ONLY,
fp16, ROCBLAS_TENSILE_LIBPATH set): `allenai/OLMoE-1B-7B-0924` @ `6d84c485`
(sha256-verified on box after rsync), gate-scale slice from
`lab/graft_d0_build_slice.py` (102k domain + 86k val OLMoE tokens: code=codeparrot,
math=open-web-math, web=wikitext-2, science=ag_news-sci/tech, chat=oasst1-en),
`--phase all --init prototype --recal-steps 300 --recal-rounds 2`, bs2×512.

**Verdict (86,016 val tokens):** logit cosine **0.99913** (std 0.0025, min
0.7831) | ΔCE **+0.00075** (stock 2.0082 → graft 2.0089) | per-layer MoE-output
cosine **0.9952 mean** (min 0.9932) | top-k set agreement **0.8366 mean**
(Jaccard 0.963) | purity margins ~0 / JS ~0 on every domain (PASS). ALL gates
green incl. the cos>0.99 stretch. Recal: layer MSE 0.73→0.0005 (round 0),
0.0005→~0 (on-trajectory round 1) — passport rows fit stock logits almost
exactly, hence 0.999 parity. **Wall 615s, peak VRAM 15.2GiB on GPU 0**
(32GB card). CPU exact-arm reference was cos 0.9999 / ΔCE +0.0013 — GPU
prototype+recal is statistically indistinguishable. Registry artifact
`registry_olmoe_6d84c485.pt` (8.4MB) + JSON verdict on box
`/data/tliao/graft_d0_out/` and copied home to `lab/imported/graft_d0/`
(untracked). Honest caveat: purity passes weakly — OLMoE's own routing entropy
is near-maximal (4.15/4.16 nats, mushiness 0.92), so the baseline is maximally
permissive; the load-bearing evidence is the cosine/CE/top-k agreement.

**gfx906 gotchas (both shimmed inside `lab/graft_d0_parity.py`, auto-deployed
on HIP):** (1) `torch._grouped_mm` — "grouped gemm is not supported on ROCM";
transformers 5.19's MoE dispatch wrongly enables it on HIP → forced their
`grouped_mm_fallback`; (2) **`torch.sort` device kernel broken at S=8192**
exactly (`hipErrorInvalidDeviceFunction`, async-misattributed to whatever runs
next — use `AMD_SERIALIZE_KERNEL=3` to localize; 4096/16384 fine) → experts
merge shim sorts on CPU, same math. Full story: `lab/FAILURES.md` gfx906 entry.
**P40/MI300X not needed for D0** (closed here); D1 plug-in recal can also run
on this box (router-only), but training VesperLM experts still needs the
MI300X lane (fla/Triton gfx906 wall, see FAILURES). **Next: D1 — plug ONE
foreign expert into the graft using the registry .pt (§9.4 D1 gates).**

---

## 2026-10-11 — MODULAR_MOE §9 D0 graft-parity PREPARED (GPU run pending a free ≥16GB card)

**Everything except the GPU measurement is done.** Donor downloaded +
pinned: `allenai/OLMoE-1B-7B-0924` @ **`6d84c48581ece794365f2b8e9cfb043c68ade9c5`**
(13GB, 3 safetensor shards sha256'd in `lab/graft_d0_runbook.md`), staged OUTSIDE
the repo at `/home/tliao/.cache/vesper_d0/OLMoE-1B-7B-0924` (re-fetch snippet in
the runbook for box/droplet; NEVER commit weights). Harness
`lab/graft_d0_parity.py` implements §9.2 Variant A as the §9.6 registry path:
expert profiling (activation homes + co-activation graph §9.7) → prototype
passport init (mean router query over home-profile data) → router-only recal
(MSE+KL to stock logits, Adam on passport rows only) → registry .pt → two-pass
parity measurement (final-logit cosine + CE delta + per-layer MoE-output cosine
+ per-layer top-k agreement) + per-domain purity vs the donor's own
routing-entropy baseline. `--init exact` arm = algebraic ceiling (P rows =
sqrt(d_r)·W_stock rows, identity query map) — bit-level equivalence reference.

**Mock self-test 14/14 PASS** (`python3 lab/graft_d0_parity.py --synthetic`,
CPU ~6s): exact-init identity (cos 1.000000), profile home-recovery 9/9,
prototype 0.9544→recal 0.9985 cos (agree 0.708→0.977), registry round-trip,
metric sanity, graft install/uninstall, purity margins 0.0 — the GPU run is
pure measurement. **Real-donor CPU smoke also ran the full pipeline** (bf16,
transformers 5.17, 16 MoE layers found): `--init exact` **PASSES D0 on CPU**
(logit cos **0.9999**, ΔCE **+0.0013**, all gates green) — graft path verified
against real OlmoeTopKRouter/top-8-no-renorm semantics. Prototype smoke at toy
calibration (320 tok, 10 recal) predictably fails the bars but recal MSE drops
~5×/layer and profile mushiness is 0.72 (consistent with §9.2's mushy-donor
caveat).

**Runbook:** `lab/graft_d0_runbook.md` — exact commands (P40: `--dtype
float16`, sm_61 has NO bf16; MI300X: bfloat16), VRAM ~16GB at bs2×512, ~15-45
min on a P40 / ~5-10 min MI300X at 100k tokens, and the numeric PASS bar
(cosine > 0.9 stretch > 0.99, |ΔCE| < 0.05 nats, per-layer MoE cos > 0.9,
top-k agree > 0.8, purity_graft ≥ purity_stock − 0.02 every domain). Data
fixtures `lab/graft_d0_texts/` (tiny, plumbing only — gate wants a ≥50k-token
mixed slice; snippet in runbook). **Next: run the gate when a ≥16GB GPU frees
(box P40s ~Oct 12 or an MI300X droplet if approved), record the JSON verdict
here; if prototype+recal passes, D1 plug-in can start against the registry .pt.**
→ **DONE same day on the MI50 box — see the "D0 GATE RUN: PASS" section above.**

---

## 2026-10-11 — MI50 full-FT SFT: STOPPED at GPU smoke gate — Triton rejects gfx906

**Gate FAIL, no training started.** Env built on `tliao@100.64.0.7` (8× MI50/MI60
gfx906): venv `/data/tliao/vesper_sft/venv` (16G, torch 2.9.1+rocm6.4 +
pytorch-triton-rocm 3.5.1, fla 0.6.0@37a6b1c + `tools/patch_fla.py` all 6
patches), repo `/data/tliao/vesper_sft/VesperLM`, t3 base staged at
`lab/imported/t3_ckpts/step_best/checkpoint.pt` (3.5G). All scratch on the 7T
`/data` (421G free) — root's 24G untouched.

**Smoke gate (`lab/smi50_smoke_gate.py`, GPU 0 only):** tiny VesperLinearLM
GLA fwd+bwd (bf16 autocast and fp32/grad-ckpt), tiny 470m_k path (KDA+MLA+
passport), and `VESPER_FUSED_CE` fla FLCE — **all 4 FAIL identically**, and so
does a hello-world `@triton.jit` elementwise kernel:
`triton/backends/amd/.../compiler.py make_llir → "error: unsupported target:
'gfx906'" → PassManager::run failed`. `gfx906` never appears in the wheel's AMD
backend (gfx942/gfx950-era code). **Platform-level: fla never runs; not a fla
bug and not patchable via fla env flags** (fla 0.6.0 has no CPU/torch fallback
for chunk kernels — only `FLA_CACHE_MODE`/`FLA_CONFIG_DIR` config caching).

**rocBLAS gotcha (needed even for plain matmul on gfx906):** the wheel's Tensile
library has no gfx906 kernels → `ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/
rocblas/library` (that's what the box's hackathon processes export; ROCm 6.2.4
is the last install with gfx906 tensile kernels). Get this right before any
retry.

**Next branches (untried, cheapest first):** (a) older Triton with a gfx906-era
AMD backend — pair a torch whose bundled `pytorch-triton-rocm` predates the
gfx906 drop (the box also has /opt/rocm-6.2.4; revisit torch 2.6/2.7+rocm6.2
wheels) and re-run `lab/smi50_smoke_gate.py`; (b) upstream fla universal/torch
kernels if any appear; (c) run the full-FT recipe on a CDNA box (MI300X lane,
same recipe as the DDP plan below). **Do not start training until the gate
passes** — GLA/KDA forward is Triton-only.

**RETRY RESULT (2026-10-11, one attempt, no third stack): branch (a) CLOSED.**
In `/data/tliao/vesper_sft/venv_triton_old`: torch 2.6.0+rocm6.2.4 bundles
**triton 3.2.0** (not 3.0.x as hoped) — hello-world `@triton.jit` fails with
the same `error: unsupported target: 'gfx906'`; torch 2.5.1+rocm6.2 bundles
**triton 3.1.0** — same hello-world failure, plus fla 0.6.0@37a6b1c won't
import against that Autotuner API (`unexpected keyword argument 'do_bench'`).
All three Triton generations tested (3.1.0 / 3.2.0 / 3.5.1) reject gfx906 at
LLVM lowering. **FINAL VERDICT: MI50/gfx906 is viable for serving (vLLM /
rocBLAS inference — the box's hackathon jobs run fine) but NOT for the
fla/Triton training stack. Full-FT of 470m_k belongs on the MI300X lane.**
Branch (b) is the only untried software path (upstream fla torch/universal
kernels — none exist in fla 0.6.0); branch (c) is the way forward.

**Recipe parked for the MI300X lane** (was: 4-GPU DDP on MI50, GPUs 0,3,4,5 —
never 1,2): full-FT
427M, fp32 AdamW (fits 32GB: ~2.6G params+grads + ~5.2G AdamW states), bf16 AMP
+ `VESPER_FUSED_CE=1`, seq 1024, micro-batch to fill 32G (measure first),
32–64k tok/step effective (4×mb×1024×accum), lr 2e-5→2e-6 cosine warmup 100,
wd 0.01, ≤3 ckpts on box, out `lab/imported/sft_470m_mi50/`, eval_chat 12
prompts T=0.7/top-p 0.9. Baseline to beat: LoRA r8 val 2.304.

---

## 2026-10-11 — SFT r2 (LoRA r32 + OASST/OpenHermes 51M): val 2.304→1.4999, STILL NOT talkable (laptop 3070)

**What ran:** config `470m_k_sft_r2` (`SFT/01_sft_train.py`, committed here) —
same t3 passport base `lab/imported/t3_ckpts/step_best`, **LoRA r32** profile
`kda_mla` (42 modules, **2.72M trainable of 429.5M**, α=64 via
`VESPER_SFT_LORA_ALPHA`), bf16 + `VESPER_FUSED_CE=1`, micro_batch 8 × seq
1024, `target_accumulation_steps=32` (sequences ⇒ accum 4 micro-steps =
32,768 tok/optimizer-step; 2000 steps ≈ 65.5M positions ≈ 1.3 epochs over the
mix), lr 5e-5 cosine→5e-6 (warmup 50), 2000 steps, eval every 200. Data:
`oasst2_sft.bin` 1,332,224 + `openhermes_sft.bin` 50,000,896 token positions
(built by the `Dataset/05_sft_oasst.py` / `07_sft_openhermes.py` changes in
this commit: refusal-formula conversation drop + fixed `target_tokens`
accounting). Wall 5.5h (00:55–06:25 local), 1.8GB torch alloc, no OOM.
Log: `lab/logs/sft_470m_k_r2_lora32.log`.

**Val curve (masked CE, 30 eval steps):** 1.7136@200 → 1.6420@400 →
1.5457@600 → 1.6039@800 → **1.4999@1000 (BEST)** → 1.6205@1200 → 1.6342@1400
→ 1.6208@1600 → 1.7321@1800. Late single-step train CE is noisy (per-10-step
values 0.82–2.2 throughout; window means ≈1.25 @600–1200, 1.66–1.86
@1400–1999) — the val curve is the selection signal. **`step_best` = step
1000** (md5 `615b5c35aab96c870fdb608cdf7d00e3` == `step_1000/checkpoint.pt`,
verified).

**r2 vs r1:** r1 (LoRA r8, 299 steps, 1.4M OASST-only positions) flatlined
2.3244@50 → **2.3044@150 (best)** → 2.3067@250. r2 = 37× data positions, r32,
2000 steps → **1.4999** best (Δ −0.80 CE). Train CE: r1 ≈2.38 flat; r2 window
means 1.96→1.25 with mid-run dips to 0.82.

**Eval (`SFT/eval_chat.py`, 12 ChatML prompts, T=0.7 / top-p 0.9):**
`python3 SFT/eval_chat.py --ckpt lab/imported/sft_470m_r2/step_1000/chat_model
--out lab/imported/sft_470m_r2/eval_chat_step1000.json` (CUDA 3070; log
`lab/logs/eval_chat_r2_step1000.log`). **Layout gotcha:** `step_best/` holds
only `checkpoint.pt` + snapshots — no tokenizer, no `chat_model/` (evaling it
crashes in `PreTrainedTokenizerFast.from_pretrained`, see
`lab/logs/eval_chat_r2_best.log`). r2 keeps the r1 `step_N/chat_model/`
convention (tokenizer + merged `vesper_chat.pt`); point eval at the best
step's `chat_model`. All 12 samples verbatim in the JSON path above.

**QUALITY VERDICT — honest: STILL NOT talkable.** 0/12 usable. Failure modes
across the 12 held-out prompts (eval_chat decode is unseeded — samples below
are verbatim from the committed
`lab/imported/sft_470m_r2/eval_chat_step1000.json`): prompt-ignoring/corpus
drift, hard repetition loops (8/12), format debris (forum transcripts / code
dumps), and 3 EMPTY replies. Only 2/12 even touch the topic (boil an egg,
list-vs-tuple) and both then loop. Verbatim samples (pure repetition elided
with ×N markers; full text in the JSON):

(1) "Hello! How are you today?" → "I am using a smartwatch, but I can't
remember where to start. I need to open the smartwatch app, click the green
button, and then select the blue button. It's my first time using the app,
and it works fine. I can't remember the starting point of the smartwatch
app. Here's my code: ```java public class MySmartwatch {"
[unprompted tech-support monologue, not a greeting]

(2) "What is the capital of France?" → "" [EMPTY — immediate stop]

(3) "What is 17 * 23?" → "I am trying to figure out what is the most common
way of naming things in a language. The most common way of naming is as
follows: - 《公男》》》》…"
[arithmetic → CJK bracket-garbage loop]

(4) "How do I boil an egg?" → "This is a very basic way to boil an egg.
1. Fill an egg jar with the water, or use a garden jar or a large glass jar.
2. Add a little water to the egg. 3. Pour the water into the egg. 4. Add a
little water to the egg. 5. Pour the water into the egg. …"
[the only topical opener in the set, then a step loop ×N]

(5) "Why do we dream?" → "The dreamer was a cat, and the dreamer was a cat.

If you are a cat, you dream. If you are a cat, you dream. …"
[sentence loop, prompt ignored]

(6) "Can you explain what a black hole is in simple terms?" → "on Oct 18,
2010 at 9:25 am What's the difference between a black hole and a neutron
star? communicators on Oct 18, 2010 at 10:27 am …"
[forum-transcript corpus debris]

(7) "I feel really stressed about work lately. Any advice?" → "" [EMPTY]

Training-time probes at step 1000 (`step_1000/eval_samples.json`, same T=0.7
/ top-p 0.9 — so this is not a decode-setting artifact) collapse the same
way with partial fact recall: "The capital of Japan is Tokyo, a city that is
famous for its historical landmarks and cultural heritage. The capital of
Japan is Tokyo, which is located in Tokyo. …" [loop]; "91 * 7 is a multiple
of 7, which means it is a multiple of 7. … 91 * 7 = 91 * 7 = 91 * 7 = …"
[nonsense loop].

**Binding limit: the base pretrain, not the SFT.** t3 is 427M total / 194M
active on only **3.77B tokens** (~2× under Chinchilla for this size), and its
own pretrain val plateaued at 2.2678 (a phase-switch-mismatched metric — see
t3 postmortem below). Repetition collapse within 30–60 tokens, prompt-echo
decodes, and an empty reply are LM-prior failures. r1→r2 moved CE hard
(2.30→1.50, +37× data, 4× adapter rank) and bought sentence-level structure +
occasional topicality, but every failure mode survived — the SFT axis is near
exhausted on this base. **Specific fix: bigger pretrain first** — push ≥10–20B
tokens through the same 470m_k (or scale params), MI300X lane, then re-run
this exact SFT recipe (reproducible: config `470m_k_sft_r2`). **MI300X
full-FT is the second lever, not the first:** LoRA r32 touches 0.63% of
params and val 1.50 is still far from talkable territory (<1.0), so full-FT
would lower CE — but it cannot install a generation prior the pretrain never
learned, so full-FT of *this* base will still loop. If only one MI300X
affordable run: spend it on pretrain tokens; full-FT SFT rides along after.

**Housekeeping:** eval JSON + logs live in `lab/imported/sft_470m_r2/` +
`lab/logs/` (untracked by design — never commit `lab/imported/`). Resume r2:
`VESPER_SFT_OUT=lab/imported/sft_470m_r2` picks up the highest `step_N`
(wrapped LoRA keys auto-handled).

---

## 2026-10-11 — SFT 470m_k CHAT: pipeline green, model NOT talkable yet (300 LoRA steps, laptop 3070)

**What ran:** SFT of `lab/imported/t3_ckpts/step_best` (t3 470m_k passport,
val-best@3100) into chat form on the RTX 3070 8GB via the modified
`SFT/01_sft_train.py` (config `470m_k_sft`) + `SFT/eval_chat.py`. Data:
`Dataset/data/sft/oasst2_sft.bin` (1.39M token positions, ChatML,
assistant-only loss mask — interleaved [token,mask] uint16 verified;
pad_id=12 never occurs in the bin, so the fused-CE ignore-masking is exact).

**Recipe:** bf16 AMP + `VESPER_FUSED_CE=1`, **LoRA r8 α16 profile `kda_mla`**
(42 modules, 0.68M trainable of 427M), micro_batch 8 × seq 1024, accum 4,
lr 5e-5→5e-6 (warmup 20), `total_steps=300` (loop bound is exclusive → 299
optimizer steps: 50 pre-pause + 249 resumed from `step_50`). Full-FT is a
hard OOM on this card — confirmed live: fp32 AdamW state init dies at
7.51/7.67 GiB on the first `optimizer.step()` (no bitsandbytes here).
Command:
`VESPER_SFT_CONFIG=470m_k_sft VESPER_SFT_CKPT=lab/imported/t3_ckpts/step_best VESPER_SFT_OUT=lab/imported/sft_470m VESPER_SFT_LORA=1 VESPER_FUSED_CE=1 python3 SFT/01_sft_train.py`

**VRAM:** torch allocated peak 1.7 GB (dummy-pass + per-10-step logs);
nvidia-smi process footprint ~4.35 GB during training (ctx/reserved included).

**Loss:** train CE 2.385 @0 → ~2.377 @250 (10-step means ~2.28 → 2.22 → 2.38,
noisy — LoRA r8 at 300 steps barely moves it). Val CE: 2.3244 @50 → 2.3107
@100 → **2.3044 @150 (best)** → 2.3051 @200 → 2.3067 @250. `step_best` = step
150; last ckpt = `step_250`.

**Bugfix committed here:** `best_val_loss` was serialized into
`step_N/checkpoint.pt` *before* the best-update, so a resume reset the tracker
to `inf` and every later eval claimed "NEW BEST" (clobbering `step_best` even
when worse). Now updated before save. (The pre-pause `step_50/checkpoint.pt`
was hand-patched with its true val 2.3244 before resuming.)

**Eval (`SFT/eval_chat.py`, 12 ChatML prompts, T=0.7 / top-p 0.9):** sampled
both `step_150` (best) and `step_250` (last) → `lab/imported/sft_470m/eval_chat_step{150,250}.json`.
Prompt encoding verified exact (im_start/im_end encode atomically to ids 1/2,
matching the bin). Step_250 is no better than step_150.

**QUALITY VERDICT — honest: NOT talkable.** Replies ignore the prompt and
drift into pretrain-ish corpus text (programming Q&A, forum boilerplate),
often loop (`Never: Cannot be used to make a claim` ×15) or emit format debris
(`{"name":"Balancing the Balance"`). Representative samples (step_150):
(1) "What is the capital of France?" → "How many workers are required to
carry out the duties of a supervisor? Never: Cannot be used to make a claim
Never: …" [loop]; (2) "What is 17 * 23?" → ">>> # >>> # >>> # …" [degenerate];
(3) "Why do we dream?" → "I can't believe that you're not the one who knows
how to do it. I can't even see your face. You have no idea how to do it. It's
so hard. …" [loop]; (4) "Hello! How are you today?" →
`{"name":"Balancing the Balance", message":"This is a test to determine if
it's time to take a break…` [format debris]; (5) "black hole in simple terms?"
→ "In this video, I'm going to explain what a black hole is in simple terms…"
[topical but rambling, wrong medium]; (6) "list vs tuple in Python?" → "The
list is a list and the tuple is a tuple. The list can contain multiple
elements…" [closest to an answer; still filler]. This is a plumbing
checkpoint proving the pipeline end-to-end, not a chat model.

**To make it talkable:** more data (5.5 MB = one OASST shard; rebuild/expand
via `Dataset/05_sft_oasst.py`) and more capacity/steps — LoRA r32 or full-FT
with 8-bit Adam on a bigger card, and well past 300 steps. Resume is solid:
point `VESPER_SFT_OUT` at `lab/imported/sft_470m` and the trainer picks up the
highest `step_N` (wrapped LoRA keys auto-handled).

---

## 2026-10-10 — t3 VAL PLATEAU ROOT-CAUSE (CPU postmortem, lab/imported/t3_ckpts/)

**Why val never beat 2.2678 @ 3100: the metric stopped tracking the objective at
the phase-1→2 switch (step 3200), not the model.** Cause ranking:

1. **Phase-switch distribution mismatch (primary).** `phase_switch_step = 0.8 ×
   total_steps = 3200` (`02_pretrain_linear.py`; 470m_k config `total_steps`
   4000). Train ≥3200 is 100% phase-2 files; val is FROZEN at 80/20
   phase-1/phase-2 forever (5% tails of nemotron_phase{1,2} +
   phase{1,2}_pretrain, probs 0.4/0.4/0.1/0.1). Loss curve: train CE cliffs
   2.7→1.9 at 3200 (phase-2 data is easier) and keeps falling to 1.6 at exit;
   val steps UP +0.1 (pre-switch band 2.27–2.46 → post-switch 2.45–2.59) and
   flatlines. 80% of the metric measures a distribution the model is no longer
   trained on. (The phase-2 val *component* is same-file held-out — the mismatch
   is the frozen mixture, not the phase-2 sources.)
2. **Val noise / best-point selection (secondary).** Each eval = 50 batches with
   per-batch unseeded `np.random.choice` over 4 sources → ±~7pp mixture jitter;
   val also wandered pre-switch (2500–2700 hit 2.66–2.74). 2.2678 is the trough
   of a noisy series AND the last pre-switch point; "never improved after" is
   partly a high bar set by a lucky draw.
3. **LR floor (weak).** Cosine (warmup 300, max 2e-4, min 3e-5): 5.4e-5 @3100
   (27% of max), 4.9e-5 @3200, 3.0e-5 @4000. Not floor-scraping when the
   plateau starts, and train CE still descends hard through the tail →
   optimizer not bound (the 3e-5 floor = 15% of max would eventually bind).

**eval_samples (Q3): metric stall, not qualitative stall** — 3300–3900
generations keep evolving and are at least as coherent as 3100's (3900 narrative
prompts read clean; 2800 even had language-garbage). No forgetting/collapse.

**Bonus finding (affects data planning): the whole vesperk corpus (~18B tokens:
fineweb_edu/dclm/code/openwebmath/finemath/cosmopedia/wikipedia) was NEVER
trained on.** Phase bucketing name-matches `'phase1' in name` / `'phase2' in
name`, so only `pretrain/nemotron_phase{1,2}.bin` +
`pretrain/phase{1,2}_pretrain.bin` land in the phase streams; `vesperk/*`
matches neither, and the streams ignore index.txt weights anyway (`{k: 1.0}`).
Accounting fits exactly: 2.94B pre-switch tokens = 0.88 epochs of the 3.33B
phase-1 pool; 0.83B post-switch = 0.67 epochs of the 1.23B phase-2 pool. t3's
3.77B tokens are pure nemotron-curriculum. (`lab/queue_done_prior/t3-470m-passport.json`
is the earlier 1000-step swarm3 probe on the same config — not the full-run job.)
**FIXED 2026-10-10 (default OFF, ladder-comparable):** `route_nonphase` config
key / `VESPER_ROUTE_NONPHASE=1` env now routes non-phase index.txt sources into
an always-on stream muxed into both phase streams (share:
`nonphase_share` / `VESPER_NONPHASE_SHARE`, default 0.3) and honors index.txt
weights within every stream. Val is per-group (phase1/phase2/nonphase) with
`val_loss` tracking the current phase mix; per-group NLLs are logged even with
the flag off. Unit check: `python3 Pretrain/tests_route_nonphase.py`. The
Vesper-K pretrain config opts in; t0/t1/t2/t3-comparable runs must keep it off.

**SFT-readiness verdict: GO with `step_best` (= val@3100) as the SFT base.**
It is the last pre-specialization checkpoint — the most general weights of the
run (later ckpts are phase-2-shifted and +0.1 on the curriculum val) — and
qualitatively sound. Caveats: (a) 3100 vs 3000/3200 is within val noise — the
exact pick matters little for SFT; (b) the base never saw the vesperk mix —
don't expect the code/math/wiki breadth the corpus plan implied (3.77B tok ≈
9.6 tok/param is lean); (c) any retrain/t4 should route non-phase sources into
the phase streams (or an always-on stream), log per-phase val (or a val that
tracks the current phase), and consider a lower min_lr for the tail. Routing
+ per-phase val are now implemented behind the `route_nonphase` flag — see the
bonus-finding fix above; lower min_lr is still open. (The
3990/4000 trainer-vanish stays undiagnosed; irrelevant to the plateau.)

---

## 2026-10-10 03:35 UTC — SWARM5: speedrun-opts MI300X canary GREEN → MERGE READY ($0.49, destroyed+verified)

- **Merge gate PASSED — recommend `git merge speedrun-opts` into main.** All 5
  arms green on real MI300X (gfx942, torch 2.9.1+rocm6.3, triton 3.5.1, fla
  @37a6b1c+patches): baseline / `VESPER_COMPILE=1` / `VESPER_FUSED_CE=1` /
  `VESPER_VALUE_EMBED=1` / all-three, 15 steps each at lab_tiny KDA+MLA,
  VESPER_AMP=bf16, VESPER_SEED=123. CE@10 spread across flags **≤0.0025 nats**
  (baseline 11.0912, compile 11.0915, fused_ce 11.0913, value_embed 11.0890,
  all3 11.0894; aux loss identical across arms). No crash, no hang, no NaN,
  no hip/inductor errors. tok/s@seq512: base 53.5k, compile 52.4k (−2%, tiny-
  model overhead, not meaningful), **fused_ce 56.1k (+5%) with train VRAM
  0.3GB vs 1.4GB (−80% — logits never materialized)**, value_embed 53.9k,
  all3 51.8k. Full verdict table + limits: `lab/imported/swarm5/CANARY.md`;
  logs + `canary_result.json` same dir. Job def (both lanes):
  `lab/oneclick_jobs/canary.{json,sh}` on speedrun-opts (eab0d7b, 8679206).
- **value_embed cost note (new):** the tables are vocab-sized — lab_tiny went
  11.25M→36.41M params (+25.2M = 2×65523-token lookups). Fine, but don't
  enable it in production without the seeded A/B (quality bet; +2 ckpt keys,
  flag-on ckpts can't load flag-off).
- **Lane: devcloud fallback.** OneClick `gh-ca7ead92` polled 40 min
  (02:28–03:08 UTC, 240/240 pending, zero windows) → starved per plan →
  droplet `vesper-swarm5-ttl180m-1791601783` (607809725), canary in ~15 min,
  destroyed 03:24 UTC + verified (`no vesper-* droplets`), **$0.49 settled**
  (ledger $48.94/$180, open $0). Session total $0.49 (OneClick $0).
  Droplet bring-up gotcha (new): first-boot banner breaks `scp` — wait for
  `cloud-init status: done`, transfer via `ssh 'cat > f' < local` (in
  FAILURES.md 2026-10-10 entry).
- **Next for main:** merge speedrun-opts (flags stay env-gated default-OFF,
  byte-compat sha256 already proven on CPU), then optionally quote fused-CE
  perf from a 470m-scale probe (lab_tiny tok/s is noise for compile).
- Kill the laptop t3 tunnel keeper when convenient: `pgrep -f "R 2222:localhost:2222"`.

---

## 2026-10-10 02:10 UTC — t3 COMPLETE + destroyed (settled $33.94; ledger $48.45/$180, quota FREE)

470m_k passport flagship finished: 3.77B tokens, **val best 2.2678 @ step 3100**
(phase-2 data shift plateaued val after 3100 — worth a look at the val mix),
CE ~1.62 at exit. NOTE: trainer vanished at step 3990/4000 with no traceback
(no OOM, disk 20%) — treat as ~complete; the 10 missing steps are noise but
the cause is undiagnosed (check t3_train.sh wrapper behavior if it matters).
ALL artifacts home in lab/imported/t3_ckpts/ (untracked binaries): step_best
(=val@3100, size-verified 3,691,881,987 B) + periodic step_500..step_3500 +
all eval_samples + loss_curve.png. Droplet 607549362 destroyed + verified
(no vesper-* droplets). ckpt sync: agent-15's uploader (periodic ckpts,
prunes local) + my ckpt_sync.sh (step_best); both worked. Kill the laptop
tunnel keeper when convenient: pgrep -f "R 2222:localhost:2222".
NEXT: t3 val analysis + passport-vs-topk headline comparison, then SFT of the
470m passport model. Swarm5 wants: speedrun-opts MI300X canary (merge gate!),
owner_mass 0.45@t2, D55 with 5+ experts.

## 2026-10-09 09:55 UTC — t3 RUNNING: 470m_k + passport flagship on MI300X

**RUN STATE:** droplet id **607549362**, `vesper-t3-ttl1500m-1791536812`,
ip **129.212.191.223**, created 2026-10-09 09:06:52 UTC, **launched 09:41 UTC**,
log `/root/t3_470m_passport.log` (droplet). **TTL deadline 2026-10-10 10:06:52 UTC**
(laptop watchdog cron `*/5 pod/devcloud.py watchdog` enforces the name-embedded TTL;
droplet `pod-selfdestruct.timer` fires 2026-10-10 10:12:21 UTC — verified active).
~21h run → ETA finish ~2026-10-10 06:40 UTC, inside TTL. Do NOT extend past TTL;
pull artifacts before it fires (streamer should already have them home).

**CKPT SYNCER (added 10:10 UTC):** checkpoints only save on new-best val; the
original "streaming" was a manual push. Now automated: `/root/ckpt_sync.sh` on the
droplet (repo copy `pod/ckpt_sync.sh`) watches `vesper_linear_checkpoints_470m_k/*/`
for mtime+size changes and pushes via the reverse tunnel (`Host home` = laptop
user-space sshd behind `ssh -N -R 2222:localhost:2222`, keeper loop on laptop,
log /tmp/t3_tunnel.log). Pushed sigs recorded in `/root/.ckpt_sync_seen`.
val@200 (5.4630) ckpt verified home at `lab/imported/t3_ckpts/step_best/checkpoint.pt`
(3,691,829,251 B). Tunnel+sshd must both be up for syncs: laptop
`sshd -f ~/.ssh/t3_sshd/config` on 127.0.0.1:2222 + keeper pid via
`pgrep -f "R 2222:localhost:2222"`.

**LAUNCH ENV (as run — matches plan EXCEPT VESPER_ACCUM, see finding below):**
`VESPER_CONFIG=470m_k VESPER_AMP=bf16 VESPER_ROUTER_TYPE=passport VESPER_MICRO_BATCH=8
VESPER_ACCUM=128 python3 -u 02_pretrain_linear.py` from `/root/VesperLM/Pretrain`
(HEAD 81b33da ≥ 311b3f5). Confirmed in `/proc/<pid>/environ`. Log shows
**Global Effective Batch: 128 (Target: 128), Local Accumulation Steps: 16** →
1.048M tok/step @8192 × 4000 steps = **4.2B tokens**. passport_dim default 64.

**VERIFIED (09:41–09:54 UTC):** (a) config-path inspection on the droplet builds
**PassportRouter in all 10 layers** (`('PassportRouter','vesper_model')`, passport
rows (8,64)); (b) `Hybrid stack: {'kda': 8, 'mla': 2} (10 layers)`; (c) dummy pass
survived ("Max VRAM Successfully Reserved: 10.5GB"), step 0 CE 11.2488 → step 170
CE **5.68** finite and falling on real data; Tok/s 6.5k@seq192 → 37k@1024 →
**51.2k@1792** (probe 57k @seq 5248-6144; seq ramps to 8192 by step 800);
(d) VRAM 7-15GB allocated during ramp / 10.5GB dummy reserve (probe ~20GB @seq 5k+)
of 192GB. 426.77M total / 190.84M active; Muon 359M + AdamW 67M (bnb 8-bit absent →
torch AdamW fallback — harmless on 192GB).

**SAFETY ARMED:** (1) checkpoint stream `pod/upload_ckpts.sh` (KEEP_LOCAL=2) droplet →
**`lab/imported/t3_ckpts/`** — live-verified (`step_best` already landed at 09:50);
transport is a laptop-kept `ssh -N -R 2222:localhost:2222` reverse tunnel to a
user-space sshd on the laptop (127.0.0.1:2222, `~/.ssh/t3_sshd/`) because laptop
system sshd is dead; droplet reaches home as `HOME_SSH=home`. (2) laptop puller every
30 min (`/tmp/t3_puller.sh`) grabs `step_best` updates + `eval_samples_*.json` +
`loss_curve.png` (upload_ckpts.sh uploads step_best only ONCE and never the root
artifacts — pre-existing quirk). (3) droplet tripwire `/root/t3_tripwire.py` (pid
7513): CE NaN/inf or >15 after step 200 → kills trainer (`/root/t3_train.pid`),
writes `/root/t3_tripwire_fired`, keeps droplet for evidence. (4) laptop watchdog cron
+ droplet self-destruct timer as above.

**Crash resume (manual — no auto-resume loop, per plan's exact launch):**
`ssh root@129.212.191.223 'setsid bash /root/t3_train.sh </dev/null >>/root/t3_470m_passport.log 2>&1 &'`
(trainer auto-resumes from highest `step_N` in `vesper_linear_checkpoints_470m_k`).
Droplet bootstrap notes: `python3.12-dev` installed via apt FIRST (swarm4 finding —
triton AMD hip_utils needs it); torch 2.9.1+rocm6.3 + triton 3.5.1; fla@37a6b1c
--no-deps + einops; `tools/patch_fla.py` 3 patch groups OK (env.py, mla.py, kda
num_stages×4).

**Vesper_ACCUM semantics (resolved — do not regress):** `VESPER_ACCUM` = target global
batch in **sequences** (`accumulation_steps = target_acc_steps // (world*batch)`),
NOT micro-step count. `VESPER_MICRO_BATCH=8 VESPER_ACCUM=16` (= the old probe's
"micro 8 / accum 16") is only 16 seqs/step = 131k tok/step = 472M tokens / ~2.5h.
The flagship 4.2B / 21h needs **VESPER_ACCUM=128** (= 8×16 micro-steps = 128 seqs =
1.048M tok/step). Evidence: `lab/imported/probe_kda_mla*.log` show "Global Effective
Batch: 16 (Target: 16)"; trainer `02_pretrain_linear.py` batch-scaling math.

---

## 2026-10-09 — SWARM SESSION 4 ($1.27, destroyed+verified; slot free → t3 launching)

- **t2 seed-2 paired head-to-head: PASS — passport +2.78%** (6.4341 vs 6.6184), pairing
  with unseeded +2.83%. Replication table now 2 seeds at every tier ≤120M:
  t0 +2.67 / t1 +1.95 / t1s2 +2.42 / t2 +2.83 / t2s2 +2.78.
- **passport_dim sweep: capacity irrelevant** (32/64/128 within 0.008 nats at t1).
  Default 64 stands; drop to 32 if parameter frugality ever matters.
- **D55 at t2 (120M): PASS all 9 gates at BOTH owner_mass 0.50 and 0.55** (0.50 purer:
  cross ≤0.214; 0.55 bigger CE deltas). Multi-plug-in protocol now validated at 11M AND
  120M. **Production rule: owner_mass 0.50 for ≥t1 scale, 0.55 at t0.** swarm2's
  cal_steps report-field bug fixed in swarm4_d55_t2.py.
- AMD_PITCH updated: t2-seed2 row added; multi-plug-in bullet rewritten from "honest
  negative" to "failed → root-caused → fixed → re-validated at two scales".
- Evidence: lab/imported/swarm4/ (commit 886209a). Ledger: $14.51 settled, $0 open.
- Droplet gotcha (new): triton AMD hip_utils needs apt `python3.12-dev` on droplets.

---

## 2026-10-09 — SWARM SESSION 3 ($2.83, destroyed+verified; slot handed to swarm4)

- **owner_mass scale check at t1: PASS at 0.50, and the operating point SHIFTS with
  scale** (t0 11M wants 0.55; t1 33M wants 0.50; 0.55 at t1 leaks code cross 0.331).
  Same direction as reject_w (15→10→9): **all routing-supervision knobs weaken as
  capacity grows; per-scale sweep required.** t1/0.50 margins are wide (code cross
  0.255, wiki cross 0.086). Expect t2 (120M) ≈ 0.45 — swarm4 tests exactly that.
- **470m_k A/B (392M, paired seed 7, 970 steps, seq ramp to 8192): PARITY, not a win.**
  Passport leads early (+3.18% @ step 100), topk edges ahead after step 400, final
  −0.27% (4.3723 vs 4.3604 — single-seed noise range). The passport quality advantage
  is confirmed ≤120M and unproven at 392M in a short window. **Decision: t3 launches
  with passport anyway** — parity-on-quality + free modularity (the actual product)
  justifies it, and the full 4000-step run is itself the long-window test the 970-step
  probe can't answer. AMD_PITCH.md updated with the honest 392M row.
- Evidence: lab/imported/swarm3/ (commit e22c90b). Spend: $13.24 settled.
- Droplet quota discovery (agent-15): **account allows ONE GPU droplet** — sessions
  serialize. swarm4 holds the slot now; t3 launches when swarm4 finishes (its
  passport_dim sweep result feeds the t3 config).

---

## 2026-10-08 goal-mode run — deliverables 1–3 DONE (of 4)

- **D2 Hippocampus demo on true KDA+MLA: PASS** (f64ed1b). tiny_agent_k step_3000,
  HIPPO_TARGET_PROFILE=kda_mla + HIPPO_CKPT (21bf76d). Margins all up
  (−4.01→−3.00 / −4.18→−2.73 / −3.77→−2.93), digit-NLL 2.18→3.20, word-NLL
  4.77→4.01; poison batch ROLLBACK (target collapse), incumbent byte-identical.
  34 LoRA wraps = exact kda_mla target set. Evidence: lab/imported/hippo_demo_vesperk.log.
- **KDA/MLA CPU shim set COMPLETE** (97b8d90): added pure-torch causal_conv1d
  (+update/step), rotary (ref+offsets), rms_norm_ref wraps to BOTH Immune/cpu_backend
  and Hippocampus/consolidate; patch via sys.modules (bare import binds the FUNCTION).
  **Pre-existing sigmoid gated-norm bug fixed** (consolidate.py computed swish in the
  sigmoid branch — error 2.14 on unit tensors, load-bearing under KDA o_norm).
  41 PASS / 0 FAIL suite on real step_500 weights; gla_gqa backward compat verified.
  Known non-blockers: state_v_first ignored on CPU (self-consistent), conv/rotary shims
  reject packed cu_seqlens, archived gqa_buggy ckpt has wrong full_type in metadata.
- **Cosmopedia root-caused + regened** (8d30a2e): the 0-byte shard was NOT the suspend —
  config "full" never existed on the hub (ShardWriter creates the file before the stream
  raises). Config → "cosmopedia-v2"; 2 shards, 1.5B tokens, indexed. Mix complete.
- **Speedrun branch READY (not merged)**: `speedrun-opts` (pushed; worktree
  /home/tliao/VesperLM-speedrun) adds env-gated VESPER_COMPILE (dense-submodule
  torch.compile), VESPER_FUSED_CE (fla FusedLinearCrossEntropyLoss, pad==eos masking
  preserved exactly — biggest single win: 65536-vocab logits at seq 8192 ≈ 2.1GB fp32
  per micro-batch), VESPER_VALUE_EMBED (modded-nanogpt value embeds on first/last
  layers; KDA/MLA/GQA sites). Byte-compat sha256-verified when flags off; CPU smoke
  suite green (Pretrain/tests_speedrun_flags.py). **Merge gate: MI300X canary
  (~10 steps each flag) on a swarm session, then merge to main.** Est: +5-15% compile,
  +5-15% fused-CE at long seq, value-embed is a seeded-A/B quality bet (~1-3%).
- **AMD pitch committed**: AMD_PITCH.md (4c17adb) — 4-pt scaling table, plug-in gates,
  honest negatives, costed ladder t3≈$41 → t4≈$150–250 → t5 cluster, repro appendix.
- **Spend: $10.41 settled, $0 open.** ~~t3 (~$41) crosses the user's $40 soft-pause~~
  **USER LIFTED THE CAP (2026-10-08 ~23:59 PDT): "you can have multiple droplets if you
  need and test a few things at once, if it makes it go faster"** — the $40 soft-pause
  is answered; MULTIPLE CONCURRENT DROPLETS approved. Remaining hard rules unchanged:
  $180 ledger cap, TTL + destroy + verify on every droplet, everything pulled home
  and committed. t3 GO: 470m_k full pretrain, passport router, ~21h ≈ $42.

---

## 2026-10-08 — SWARM SESSION 2: multi-plug-in SOLVED ($1.34, destroyed+verified)

**Protocol D55 passes all 9 purity gates at t0** (own ≥0.5 / cross ≤0.3 / CE beats masked):
code .773/.297 (+1.04 nats), math .590/.290 (+0.11), wiki .676/.185 (+0.23).
Recipe: (1) train experts independently with contrastive reject (as before), (2) insert
SEQUENTIALLY with ~100 router-only recalibration steps between insertions, (3) final joint
calibration 800 steps, router-only, over all domains, with a **mutual-exclusion target**:
on each domain the owner expert gets routing mass **0.55**, original base experts share
0.45, and ALL other plug-in passport rows get exactly zero. owner_mass is the dial:
0.5 misses wiki-own (0.491), 0.6+ blows cross gates. **0.55 is the operating point.**
Structural finding: independent reject training clusters plug-in passport rows (each only
trained against the base), so top-2 becomes {owner, other-plug-in}; rows can only be
partitioned in a model where they COEXIST (hence the final joint cal). Joint reject (A),
sequential-only (B), joint training (C), and joint+mutual-excl without sequential (C2)
all FAIL — the full D55 chain is necessary. Cost: ~5× phase-2 wall vs independent
plug-in, entirely calibration. Evidence: lab/imported/swarm2/ (9 protocols ablated,
commit 4ec55c5). Note: swarm2_purity.py has a report-field bug (`cal_steps: 0` in JSON
for D55 — actual 800, see log); cosmetic, fix when next touching that script.
**This unblocks the N-expert library vision: repeat insert+recalibrate per new expert,
periodic joint calibration with mutual exclusion.**

---

## 2026-10-08 — SWARM SESSION 1 ($0.83, 6-way parallel, droplet destroyed+verified)

- **t1 second seed PASS**: passport +2.42% (5.968 vs 6.116) vs +1.95% unseeded — the
  advantage is NOT seed luck. Scaling table now 4 points: t0 +2.67% / t1 +1.95% /
  t1-seed2 +2.42% / t2 +2.83%. Trainer gained `VESPER_SEED` env knob (bff5cdb) for
  paired runs (the droplet-local seed hack is gone; use the committed one).
- **reject_w curve at 33M mapped**: passing window **rw10–15** (rw10 best: code util
  0.785, web 0.199, CE −1.16 nats). Slope smooth/monotone → scale→optimal-reject_w
  mapping: 11M ~15, 33M ~10–15 (best 10), 120M ~9. **Production heuristic: larger model
  → lower reject_w; start at 10 for ≥33M.**
- **Growth upcycle PASS at t0**: exact 4→6 expert surgery (clones + dup passport rows +
  top_k 2→4) trains on and matches unexpanded control (Δval −0.0008). Modular growth
  without retraining works at toy scale.
- **MULTI-PLUG-IN FAIL (the important negative)**: 3 independently reject-trained experts
  (code/math/wiki) plugged into one frozen base → each domain gets a CE win from its own
  expert (+0.68/+0.10/+0.05 nats — specialists are real) but routing PURITY fails:
  code↔math cross-talk ~0.5 (gate ≤0.3), wiki expert underused (0.17). Independent
  reject training makes plug-ins COMPETE instead of partition. **Next protocol to try
  (swarm session 2): joint plug-in training — reject terms against the OTHER plug-ins'
  domains too, or sequential plug-in with router recalibration between insertions.**
  This is the gating question for the "9999 plug-in experts" vision.
- Evidence: lab/imported/swarm1/ (results/, plugin/, expand/, logs/, scripts/;
  commits 9dcdf40, 9b417f2). Program spend: $9.07 settled, $0 open.

---

## 2026-10-08 late — laptop suspend-proofed + true Vesper-K restart

- GPU wedge (post-suspend "Unable to determine device handle") fixed by user reboot.
  **Suspend now hard-blocked**: systemd user service `vesper-no-suspend.service`
  (block inhibitor on sleep:idle, auto-starts at login) + PowerDevil AC/Battery/
  LowBattery set to never-suspend via kwriteconfig6. Trainings also launch wrapped
  in their own systemd-inhibit.
- **Old step_500 ckpt ARCHIVED as `vesper_linear_checkpoints_tiny_agent_k_gqa_buggy_archived`**
  — it predates the arch_keys fix (GQA weights in full layers; load fails against the
  fixed trainer's MLA). Do NOT resume from it. Fresh true KDA+MLA tiny_agent_k run
  started 21:35 PDT, micro_batch 2 (3 OOMs the dummy pass), ~4k tok/s at seq-256 ramp.
- **cosmopedia_0.bin was 0 bytes** (builder died at source start, pre-wedge) → deleted;
  `pod/rebuild_index.sh` now SKIPS empty shards with a warning (was: blindly included
  → "cannot mmap an empty file" crash). Cosmopedia regen queued (heartbeat item a).
- Mix now 18 bins (~33GB) until cosmopedia returns; phase-2 index rebuild picks it up
  automatically once regenerated.

## 2026-10-09 — t2 VALIDATED: passport advantage is SCALE-STABLE (all evidence committed)

Droplet session ($~1.8 open at handoff; droplet destroyed+verified after):
- **t2 head-to-head** (tiny_agent_k = 120.2M total / 82.5M active, vocab 65536, 600 steps,
  real fineweb/dclm, 2 parallel jobs ~30k tok/s each): **passport +2.83%**
  (val 6.5410 vs 6.7316). Gap widens late (passport 6.526@400→6.541@500; baseline
  6.816→6.732). Hybrid stack confirmed in trainer: {'kda': 6, 'mla': 2}.
- **SCALING TABLE (the AMD pitch)**: t0 11M **+2.67%** → t1 33M **+1.95%** → t2 120M
  **+2.83%** — the t1 attenuation REVERSED; advantage is scale-stable. Single-job 103M
  bf16 throughput: 103k tok/s @ seq 1024.
- **t2 plug-in test**: train 800 steps fineweb → freeze → separately train expert #5 on
  python code with contrastive passport loss → zero-shot plug-in. The t1 recipe (rw15)
  OVER-SUPPRESSES at 120M (code util 0.378 FAIL). Exact phase-A checkpoint replay sweep
  (lab/imported/t2/plugin_t2_tune.py, verify_ce delta 0.000000): **rw9 PASSES ALL GATES**
  (util_code 0.617 >0.5, util_web 0.171 <0.3, CE_code 7.285 vs 8.725 masked = −1.44 nats);
  rw5 fails web (0.308). **Scale→reject_w: 33M needs rw15, 120M needs rw9 — optimal reject
  weight FALLS as models grow. Production default at 120M: REJECT_W=9.**
- Evidence: `lab/imported/t2/` (head-to-head jsons+logs, plugin jsons for rw15/rw5/rw9,
  both scripts). Commits 53e8912, 29cfb17.

**NEXT DECISIONS (user's), now that the thesis holds at 3 scales:**
(a) AMD-credits pitch package — scaling table + plug-in evidence + costed ladder, all in
    lab/imported/ (user mentioned: beg AMD for credits → cluster → paper).
(b) t3 = 470m_k full pretrain (~$41 projected, single MI300X ~20h) — possibly with
    passport router as the DEFAULT (it's now the better router at every scale tested).
(c) Hippocampus/Immune teach-by-talking demo on fresh Vesper-K ckpt — KDA/MLA port
    LANDED (fbf2fad: target profiles, lazy fla import, MLA CPU fallback); KDA shim
    runtime smoke test on the training box is the only remaining gate.
(d) Speedrun optimizations: torch.compile env-gate, fused linear+CE, value embeddings.

**DROPLET SPEND PRE-APPROVED (2026-10-08, user):** autonomous swarm farm sessions on
MI300X are expected to run without asking. Standing caps: ≤$12 per session, ledger cap
$180 total, TTL mandatory on every droplet, destroy+verify (`pod/devcloud.py list` →
"no vesper-* droplets") after EVERY session, results pulled home + committed+pushed.
Session 1 launched 21:45 PDT (agent-9, droplet farm): t1 second-seed head-to-head,
reject_w midpoint curve (rw10/rw12 at t1), Growth exact-upcycle live test at t0,
3-expert multi-plug-in purity test at t0.

---

## 2026-10-09 — MODULAR-MOE THESIS VALIDATED AT t1 (all evidence committed)

Full session on one MI300X droplet ($3.66; day total $6.93; droplet destroyed+verified):
- **KDA+MLA re-probe after arch_keys fix**: {'kda': 8, 'mla': 2} confirmed in trainer;
  57.0k tok/s steady @ seq 8192 (−3.5% vs GQA — MLA is free on ROCm). 470m_k full run ≈ $41.
- **t0r real-data farm batch** (lab_tiny 11M, fineweb/dclm): passport beats top-k in ALL 5
  variants; best plain passport **+2.67% val** (6.950 vs 7.141). Random-data t0 batch was
  noise-floor (ln vocab) — plumbing-only; always use real data for quality signals.
- **t1 head-to-head** (lab_small 33M, 1590 steps): passport **+1.95%** (6.092 vs 6.213).
  Attenuating with scale (2.67→1.95) — watch at t2; consider 2-seed confirmation.
- **t1 PLUG-IN TEST (the verdict)**: train 600 steps fineweb → freeze → separately train
  expert #5 + passport on python-code with contrastive loss → zero-shot plug-in:
  **rw15/600 steps PASSES ALL GATES**: util_code 0.645 (>0.5), util_web 0.159 (<0.3),
  CE_code 7.64 vs 8.53 without (−0.89 nats). rw5 under-rejects (web 0.345), rw40
  over-suppresses (code 0.28). **Production plug-in default: REJECT_W≈15, phase-B 600 steps,
  reject examples from every non-target domain.** Script: lab/imported/t1_plugin/
  plugin_expert_test_t1.py (env knobs PLUGIN_T1_*). Per-layer finding: selectivity deepens
  with depth at low rw; layer 0 keeps target preference best under high reject pressure.
- Passport throughput cost ≈ 3-4% (72.8k → 70.4k tok/s at t1) — negligible.

**t2 DECISION TAKEN + DONE — see the t2 section above: validated at 120M, rw9 is the
new plug-in default at that scale.**

---

## 2026-10-08 evening — MI300X (AMD Dev Cloud) bring-up: WORKS

Droplet via `pod/devcloud.py` (mandatory TTL, self-destruct timer + laptop watchdog).
Ubuntu/py3.12, ROCm driver 6.19, gfx942 MI300X VF 192GB. Recipe that worked:
1. `pip install torch --index-url https://download.pytorch.org/whl/rocm6.3`
   → torch 2.9.1+rocm6.3, triton 3.5.1 (pytorch-triton-rocm).
2. fla 0.6.0 MUST be pinned by **commit**, not tag: tag `v0.6.0` does not exist
   (max tag v0.5.2 = PyPI max). Use
   `pip install --no-deps "git+https://github.com/fla-org/flash-linear-attention.git@37a6b1c6290e5240f6f0d80419d08a7aac27e548"`
   (matches laptop/box installs). Plus `pip install einops` (fla dep we rely on,
   not in requirements.txt).
3. `python tools/patch_fla.py` — now 3 patches; the new third one (ROCm-only,
   gated on torch.version.hip) caps KDA autotune `num_stages` at 2 in
   fla/ops/kda/{chunk_bwd,chunk_intra,gate,wy_fast}.py. Without it, KDA chunk
   kernels fail to compile on the AMD triton backend: "'tt.load' op operation
   destroyed but still has uses" in make_ttgir (upstream triton#9815 — AMD
   software-pipeliner bug with 4+ loads at num_stages>=3). With the cap: KDA
   chunk fwd+bwd passes in fp32 AND bf16; full tiny_agent_k model trains.
4. patch_fla.py no longer imports fla (find_spec only) — works on GPU-less hosts.

Probe gotchas that cost cycles (do not repeat):
- Probes must pass `vocab_size=65536` (or 65523): model default is 32000 and
  out-of-range token ids surface on ROCm as HSA_STATUS_ERROR_EXCEPTION hardware
  aborts, not a clean assert. Looked exactly like a kernel crash.
- Do NOT name a script `bisect.py` (shadows stdlib bisect → torch import dies
  with a confusing circular-import error).
- Trainer requires phase1 AND phase2 files in data/index.txt (nemotron
  curriculum). Synthetic probe data: two uint16 bins named *phase1*/*phase2*.
- `Pretrain/custom_tokenizer` now tracked in git (7046650) — was silently
  missing on fresh clones.
- p01's stats banner hardcodes "Config: small_v2" — cosmetic, ignore; 02's
  ACTIVE_CONFIG_NAME (env VESPER_CONFIG) is the real one. Hybrid-stack print
  now shows the true layer mix.

**Modular MoE landed (91091fc).** `PassportRouter` (per-expert passport embeddings,
dot-product scoring, expert dropout forcing passport reliance, `register_expert()` hot-plug)
+ `MoEFeedForward.add_expert()` in Common/vesper_model.py — defaults byte-compatible with
running checkpoints. Env overrides VESPER_ROUTER_TYPE/PASSPORT_DIM/ROUTER_EXPERT_DROPOUT/
NUM_EXPERTS. `lab/` experiment farm: queue/runner/promote + tier ladder (t0 lab_tiny →
t3 470m_k), sandboxed per-run cwd (never touches production data/ckpts). MODULAR_MOE.md
spec. **First science result** (lab/plugin_expert_test.py, CPU): zero-shot plug-in of a
separately-trained 5th expert — CE on its domain 7.38 vs 11.07 masked (thesis core holds),
router prefers it on-domain 63% vs 40% chance, but does NOT exclude it off-domain (43% vs
<0.3 target = chance for top-2-of-5) → passport loss needs negative (reject-off-domain)
pressure; that's the next t0 variant. **v2 RESULT (c17e6f5): PASSES** — contrastive
passport loss (reject-on-A term, weight 5.0, 400 phase-B steps): util_A 0.297 (<0.3 ✓,
thin), util_B 0.555 (>0.5 ✓), CE_B 8.01 vs 11.07 without plug-in, domain A undamaged
(4.95 vs 4.86). Mechanism confirmed at CPU scale but margins are thin on
barely-separable synthetic domains — production rule: plug-in must include reject
examples from every non-target domain, and gates should measure utilization margins,
not single cutoffs. **BUG FOUND+FIXED: trainer arch_keys dropped
full_type/kda_head_dim/kv_lora_rank/v_head_dim — all trainer runs so far (overnight
tiny_agent_k, MI300X 470m_k probe) silently built GQA full layers, not MLA.** KDA-on-ROCm
validation stands (linear_type was honored; MLA wrapper passed isolation separately), but
the next droplet session must re-probe KDA+MLA end-to-end with the fixed trainer.

**Throughput probe RESULT (KDA+GQA — see arch_keys bug note above; MLA re-probe pending) (470m_k, bf16, micro 8 / accum 16, synthetic random tokens):
~59.1k tok/s steady-state at full seq 8192, VRAM 14.1GB/192GB, CE ~11.095 ≈ ln(65523) on
noise (correct).** Projected full 4.2B-token 470m_k run: ~20h ≈ **$40** at $2/hr single
MI300X — well inside budget; headroom for micro_batch 32+ or grad-checkpoint-off tuning
would cut it further. Droplet destroyed, $3.27 settled for the whole bring-up.

---

Box: `192.168.1.153` (poweredge-r740, 3× Tesla P40 sm_61). Repo `/home/tliao/VesperLM` there,
AND a fresh local clone on the user's laptop `/home/tliao/VesperLM` (RTX 3070 Laptop 8GB,
torch 2.7.1+cu126, fla 0.6.0@git + local patches via `tools/patch_fla.py`, bitsandbytes user-site).
GitHub `git@github.com:datacrystals/VesperLM.git` is the sync point; laptop clones via HTTPS.

## STATE AS OF 2026-10-08 (newest first)

**Vesper-K exists and trains.** `Common/vesper_linear_model.py` now takes `linear_type="kda"`
(KimiDeltaAttention) and `full_type="mla"` (MultiheadLatentAttention, internal RoPE, no
incremental cache yet — full forward only). Configs: `tiny_agent_k` (103M) and `470m_k` (392M,
param-neutral vs 470m's 395M). Committed `c99c013`. **bf16 trainer path**: `VESPER_AMP=bf16`
env-gated autocast in 02_pretrain_linear.py (dense parts; linear layers stay fp32). Also env
overrides VESPER_CONFIG / VESPER_MICRO_BATCH / VESPER_ACCUM / VESPER_TOTAL_STEPS. Tested
end-to-end on the 3070: 200 steps bf16 on real shards, CE 10.4→7.0.

**RUNNING OVERNIGHT on the laptop 3070**: fresh `tiny_agent_k` pretrain, 6000 steps,
`bash overnight_3070.sh` (driver log `Pretrain/overnight_3070.log`). micro_batch auto-fell
back 4→3→2 (dummy-pass OOM at 4 and 3). Phase 2 auto-extends to 12000 steps if phase 1
completes. Checkpoints `Pretrain/vesper_linear_checkpoints_tiny_agent_k/`.
NEXT MORNING: adapt Hippocampus LoRA targets + Immune cpu_backend shims for KDA/MLA (both
are GLA/GQA-specific right now) and run the integrated teach→consolidate→canary-gate demo
on the fresh checkpoint. GPU probes locally are fine (no politeness needed on the laptop).

**Corpus building on the laptop** (CPU, `Dataset/11_vesperk_corpus.py`, log
`Dataset/corpus_vesperk.log`): ~19.5B-token mix → `Pretrain/data/vesperk/*.bin`
(fineweb_edu 8B, dclm 4B, code 3B, finemath 2B, cosmopedia 1.5B, wikipedia 1B; uint16+eos,
same convention as 03_fineweb.py). Box's 4B curriculum already rsynced to
`Pretrain/data/pretrain/`. Convention: raw text + <|endoftext|>, packed, no header.

**Spot pod** (`pod/`, committed `2349418`): `bootstrap.sh` = one-command cloud-instance setup
(CUDA/ROCm autodetect, fla@v0.6.0+patch_fla, background rsync of data from home, manifest-gated
index build, ckpt pull, auto-resume train loop, uploader). `MANIFEST` + `rebuild_index.sh`
weight sources that have shards PRESENT. `upload_ckpts.sh` streams step_* home, keeps last 2.
Home data dir = the repo's `Pretrain/data` (Pretrain/data symlinks into Dataset/data).
Plan: MI300X via user's AMD dev credits ($2/GPU/hr, $200 total). Phase 1 bring-up (~$5,
2-3h throwaway instance) validates KDA+MLA on ROCm FIRST, then kill; full run only after.
fp32 → bf16 note: trainer default is fp32; always set VESPER_AMP=bf16 off-P40.

**Box 429M run (unchanged, the verdict gate)**: step ~1900/4000 at handoff, val 1600→2.729
(step_best), 1700→2.755, 1800→2.846. DECISION (made): if val@1900-2000 > ~2.95, resume from
step_best@1600 with max_lr halved; else hands off. ETA ~Oct 12-13. CPU probes: loops
tightening, no factual pins yet at 1600.

**Env gotchas (both machines)**: box venv had triton 3.1.0 installed 2026-10-08 ~10:47 UTC
(shadowing user-site triton 3.4.0, breaking fresh `import fla`) — FIXED by installing
triton==3.3.1 into the venv + two cache.py compat patches (kwargs filter, getattr hooks —
harmless no-ops under 3.3.1). Running trainings were never affected (imports cached in
memory). user-site also has triton 3.4.0 (PYTHONPATH=~/.local/... works too). fla 0.6.0 needs
`tools/patch_fla.py` on any fresh machine (find_spec parent-package probe + MLA SDPA fallback
— no flash-attn needed anywhere). Box disk 93% full — mind checkpoints.

**Subagent deliverables (all committed)**: `Growth/` (expert expansion surgery, 14/14 —
exact upcycle = bit-exact clones + duplicate gate rows + top_k DOUBLED; production mode
top_k=2+noise drifts, needs warmup recipe; NO shared-expert slot exists — K2's "+1 shared"
needs a model-class change first). `Immune/` (canary gate + probes + drift; corruption drill
passes, auto-rollback works). `Hippocampus/` (quarantined session log + manual LoRA q/o +
reward-weighted-NLL consolidation; demo: 118M learned "math in words" preference from 6
turns; poison batch rolled back). Moonshot blueprint: `~/moonshot/TEACH_BY_TALKING.md`
(laptop). Design docs: `LMBUS_DESIGN.md`.

## What is running RIGHT NOW (original 2026-10-03 entry below)

**429M pretrain ("470m" config, ACTIVE_CONFIG_NAME="470m")** — launched 2026-10-05 19:11 UTC,
log `/home/tliao/pretrain_470m.log`. Fresh from scratch. dim 1024, 10 layers, 8 experts top-2,
429M total / 193M active. **8k context** (max_seq_len 8192 — first run at >2048), grad_checkpoint
on, micro_batch 1, accum 128 → 1.048M tokens/step, total_steps 4000 → **4.19B token budget**.
Data: fineweb_v2 1.5B + nemotron_phase1 2B (NEW to mix) + nemotron_phase2 0.5B.
Checkpoints: `Pretrain/vesper_linear_checkpoints_470m/` — numbered every 500 (5GB each),
step_best on val improvement. fp16 + Muon (as v2). Watch for: NaN (fp16), disk (75GB free), phase-1→2 switch, seq-len ramp to 8192 by step 800.

Rate/ETA (measured 2026-10-06 01:37, step ~310): 7.4k tok/s total at seq 3200, gentle knee,
~55s/step at 410k tok/step; VRAM 8.1GB. Projects ~5-6 days total (completion ~Oct 10-11).
Mid-run eval samples at steps 1000-2000 give the early semantics read. ckpt_dir bug fixed
(1faf75a): eval_samples_step{N}.json + loss_curve.png now live at checkpoint_dir ROOT —
numbered step_N dirs exist ONLY at %500 saves, do not create others (breaks resume).

Why: the 118M SFT refresh (results below) capped at format-without-semantics on held-out
prompts from TWO bases → 118M = capability ceiling → scale is the lever. This run tests
whether semantics emerge at 429M. SFT of the 429M comes after; `SFT/01_sft_train.py` config
selection will need pointing at the 470m checkpoint dir (it currently resolves v2 step_best).

## CPU probe (Pretrain/cpu_probe.py)

Runs any 470m checkpoint on CPU (~23 tok/s, KV-cached greedy). Shims fla Triton-only ops for
CPU: chunk_gla/fused_recurrent_gla -> naive_recurrent_gla (transpose state, v_first), and
fused gated RMSNorm -> torch (y = rmsnorm(x)*w*(g*sigmoid(g)), norm-first, fp32). Results
@step 600 (236M tok): topical, grammatical, degenerate loops, no factual recall yet — more
coherent than the 118M base was at 100% trained, too early for semantics verdict. NOTE:
cpu_probe prompts are ALSO held-out from SFT EVAL_PROMPTS now — do not reuse across that boundary.

## Probe findings 2026-10-03 (why this refresh exists)

Live probes of the v1 SFT model (`SFT/sft_checkpoints_118m_v1/step_2900/chat_model`, trained
from step_best@4100):

- Valid tool-call JSON and fluent English, BUT one dominant "describe a small Python project"
  script for nearly every prompt.
- `847*392` → wrong tool (`search_players`) + confabulated query. Identity question → fake
  file listing. String-reverser → thought-block repetition loop. hello.py task → malformed
  truncated tool call.
- Base model (pretrain step_best@5200, no SFT): token-salad loops, multilingual garbage,
  broken pseudo-Python.
- **Verdict: SFT v1 bought format/control, not task semantics.** The distill mix + better
  base (@5200) in the refresh are aimed at the semantics gap.

## Refresh RESULTS (2026-10-04, harness on step_2900 — Agent/eval_refresh_5200.log)

- "list all files incl hidden" -> correct `ls -la` first call, then DEGENERATE markdown-table
  loop when summarizing the observation. Observation-integration broken.
- "17*23+145" -> `python -c "print(17*23+145)"` -> 536 -> "The result is 536." PERFECT — but
  this is the templated/memorized one.
- "create notes.txt containing hello" (HELD-OUT) -> malformed `echo 'Hello', "string_word"`,
  no file created. FAIL.
- "13 * 12?" (HELD-OUT) -> MISCOPY: emitted `11 * 12`, ran it raw in bash (no python -c),
  got "command not found", repeated the same broken call, then confabulated about "17 * 12". FAIL.
- **Verdict: refresh = better CE (3.55->~1.5) + kept format, but held-out semantics STILL fail,
  same as v1.** The memorized-vs-novel contrast (python -c for the templated problem, raw bash
  for the novel one) is the cleanest memorization demonstration we have. Two SFT runs from
  different bases both cap at format -> **118M is a capability ceiling, not a data-mix problem.
  Scale is the next lever (429M config exists).** Infra proven end-to-end: auto-resume after
  disk-full, chained held-out eval fired correctly.

## CORRECTION: earlier eval success was contamination

The "17*23+145 = 536" success cited earlier is **not evidence of generalization**: that exact
prompt was in `EVAL_PROMPTS` (SFT/01_sft_train.py) AND the agent harness, and it matches the
templated math examples in the synthetic tooluse data → memorization. Fixed 2026-10-03:
`EVAL_PROMPTS` now uses 4 held-out prompts (capital of Japan, 91*7, count .py files, today's
date), verified absent from `Agent/agent_harness.py`. The harness chain additionally uses 2
fresh prompts invented today (notes.txt creation, 13*12). Treat any future eval success on
prompts resembling training templates with suspicion; prefer the fresh ones.

## State

- **Pretrain v2 COMPLETE**: 1.42B tokens. `Pretrain/vesper_linear_checkpoints_v2/step_best`
  @5200 (val CE **3.272**, was 3.55 @4100). Trainer never writes `step_6000`; exits after
  final eval. Val curve was still descending at the end — ran out of steps, not capacity.
- v1 SFT preserved at `SFT/sft_checkpoints_118m_v1/` (incl. step_2900/chat_model).
  384-param toy SFT at `SFT/sft_checkpoints_tiny384/`.
- Distill pipeline: `Dataset/10_sft_distill.py` (commit `a0d334a`), output
  `Dataset/data/sft/distill_chat_sft.bin`, wired into `Dataset/data/sft/index.txt` @ weight 1.0.
- Recent commits: `1a4a8ad` (permutation MoE, grad-ckpt flag, micro_batch 6, SFT accum fix),
  `802e57b` (KV cache inference, bf16 SFT, NaN guard), `1542633` (eval-sample crash fixes),
  `f106526` (harness step_best parse, chain threshold >= 2900), `a0d334a` (distill).

## Traps (still live — read before touching anything)

- **SFT resume trap**: any `SFT/sft_checkpoints/step_*` is resumed IN PREFERENCE to pretrain
  init. For a fresh SFT, mv the dir aside first (that is exactly what was done for this
  refresh). For crash recovery the same mechanism is your friend.
- Trainer saves the final model inside the last `step_N` dir (e.g. step_2900), never
  `step_3000`. Completion check = `>= 2900` or "Training complete".
- **bf16 required** for anything loading v2 checkpoints (`"amp_dtype": "bfloat16"`): the v2
  residual stream hits 40–60k → fp16 overflows in the last MoE layer → NaN. GradScaler is
  fp16-only.
- `find_unused_parameters=True` in DDP is INTENTIONAL (MoE idle experts). The torch warning
  suggesting removal is a false positive.
- GLA/Mamba2 stay fp32 on P40 (fla fp16 Triton kernels crash sm_61) — wrappers handle it.
- ssh+nohup: give jobs `</dev/null >log 2>&1`; `pkill -f` matches your own ssh cmdline — kill
  by pid.
- `get_latest_checkpoint` ignores `step_best` — pretrain resumes from highest `step_N`.
- **MixedDataStream probs are raw weights** — normalized inside __init__ now (2f22eb5).
  Phase buckets are name-matched ('phase1'/'phase2' substring): nemotron_phase1.bin lands in
  the phase1 stream alongside fineweb. val_probs group-normalized 0.8/0.2.
- **LINEAR_CHECKPOINT_DIR is now per-config** (`vesper_linear_checkpoints_{ACTIVE_CONFIG_NAME}`)
  — switching ACTIVE_CONFIG_NAME no longer resumes the wrong model.
- **Disk-full kills silently** (happened 2026-10-03 23:56 at SFT step 700): checkpoint save
  crashes the run; worse, torchrun relaunches die instantly AND silently because the log file
  itself cant be written. Check `df -h /` FIRST when a run vanishes. Pruned to 119GB free by
  deleting intermediate step_* (kept finals + step_best). Elephant: `/home/tliao/.cache/
  huggingface` is 403GB — candidates for reclaim if needed (distill dumps already converted
  to .bin). SFT run needs ~1.4GB per step_N dir, ~32GB for the remaining 2300 steps.

## Next steps (priority order)

1. Verify refresh results (morning checklist). Compare v2-refresh vs v1 probes — did the
   distill mix + @5200 base buy task semantics, or still format-only?
2. Switch `Agent/agent_harness.py` generation to `forward_incremental` (KV cache; server has
   it, harness still full-recompute — big interactive win).
3. Decide `nemotron_phase1.bin` (2B tokens, finished Oct 1) — still NOT in the pretrain mix.
   Any mix change belongs with the NEXT pretrain, not mid-run.
4. Scale-up pretrain: 429M config (`"470m"` = 429M total / 193M active) with **8k context
   from the start** (max_seq_len 2048 is the binding constraint; changing it needs a retrain
   anyway). Val curve says the 118M run was step-limited — budget more steps/tokens.
5. Fused linear+CE (fla `FusedLinearCrossEntropy`) — vocab-65523 logits are ~10% of step;
   needs care with the SFT mask and pad ignore_index.
6. **Vesper-K pretrain (APPROVED, queued behind current run)**: GLA->KDA linear layers +
   GQA->MLA full layers, MoE unchanged — full spec in LMBUS_DESIGN.md "Next-pretrain spec".
   Fresh pretrain at 429M scale for matched-token comparison vs this run, then scale.
   **Data mix: opt into `route_nonphase: true` in the new config (or
   `VESPER_ROUTE_NONPHASE=1`)** so the ~18B-token vesperk corpus actually trains
   (always-on stream, `nonphase_share`/`VESPER_NONPHASE_SHARE`, default 0.3 of
   micro-batches; index.txt weights honored per stream). Per-group val NLL
   (phase1/phase2/nonphase) is logged and val_loss tracks the current phase mix.
   Ladder-comparable runs keep the flag OFF (bit-identical to t0/t1/t2/t3).
   Populate `data/index.txt` with the `vesperk/*` shards + weights (fragment
   printed by `Dataset/11_vesperk_corpus.py`); the index is intentionally not
   committed. Check: `python3 Pretrain/tests_route_nonphase.py`.

7. LMbus biomimetic sensory stack — see LMBUS_DESIGN.md (full proposal: canonical semantic
   space + per-model bridge, foveated heterogeneous-MoE vision with LM-driven gaze,
   certification = held-out-modality demo on the 118M, staged toward grafted support packs
   on GLM/K-class open models).

## Hardware notes (settled — do not reopen unless user asks)

User weighs MI210 (~$4k) vs 8× Gaudi2 (~$16k) later. fla has first-class ROCm → MI210
(gfx90a, 64GB) is the safe pick; Gaudi2 needs a full TPC/SynapseAI port of GLA/Mamba2.
Advice already given: free vLLM config pass on existing 8× MI50 first → 1–2× MI100 ($1400 ea)
→ used MI300X when budget allows. User has 8× MI50s serving other models (ports Mimo/GLM
flashes to vLLM on them), hates cloud, mortal budget, wants ~1TB VRAM long-term for ~1T-param
4-bit models. Serving note: Chinese frontier models are mostly 4-bit (K3), GLM 8-bit — a
768B GLM-5.3 at 8-bit needs ~800GB → 10× MI210 (640GB) does NOT fit; 8× MI300X (1.5TB) does.

---

## 2026-10-10 ~16:20 UTC — LAPTOP OFFLINE PAUSE (user at hackathon)

User shut the laptop down temporarily (AI hackathon); **r740 box (192.168.1.153)
stays up — the 429M run is unaffected**. On resume: this session's cron fires
coalesce into one delivery; agent-25's SFT task (PID was mid-run, config
470m_k_sft, output lab/imported/sft_470m/) will show lost/stopped — resume
agent-25; SFT ckpts save per-step so partial progress is usable (eval any
step_*/chat_model with SFT/eval_chat.py). Farm runner + attach-pollers die
with the laptop; restart runner (`systemd-inhibit ... python3 lab/runner.py
--slots 1 --watch`) on return. Nothing was left mid-queue.

**SENSITIVITY RULE (2026-10-10, explicit user directive):** ops-sensitive
material — OneClick/devcloud lane mechanics, free-tier details, "how we get
cheap MI300X" — is LOCAL-ONLY, never committed (the repo is public). The
field guide lives at /home/tliao/HACKATHON_MI300.md (outside the repo).
HACKATHON_MI300.md was briefly committed (7233212) then scrubbed via
reset+force-push on user demand.

**OPS LESSON (2026-10-11): long GPU jobs on remote boxes MUST be launched
detached (`setsid`/`nohup ... &`) — agent-session timeout/reap kills the ssh
connection and SIGHUPs every remote child (killed D1/sweep/D2 mid-run on the
MI50 box; three runs lost to this in one evening). Watchers should poll
result files, not PIDs over ssh.**

**BOX 429M RUN ENDED (2026-10-11 10:22 UTC) — end-of-run vanish REPRODUCED.**
Trainer's last log line: step 3990 (of 4000, exclusive-bound loop), then gone,
no traceback — identical signature to t3's 3990/4000 vanish (2nd reproduction:
this is a real end-of-loop bug, not a crash; MUST fix before Vesper-K). Saves
are every 500 steps: latest is step_3500 (3.7GB) — the final ~490 steps /
~450M tokens of weights are LOST. Val samples through 3900: post-switch val
"rise" (2.55→2.73) is the known frozen-mix artifact (t3 postmortem), not
regression. For SFT/base work prefer the PRE-switch checkpoint (step_3000,
like t3's step_best@3100). Box's 3xP40 are now FREE for D0-style work or the
postmortem-repro.
**WATCHER BUG (mine, same evening): `kill -0 <pid>` in a laptop-side watcher
checks LAPTOP PIDs — remote PIDs must be checked ON the remote box (or watch
result files). False "process exited" reports caused brief double-launch risk
on D1/sweep.**

## 2026-10-11 — PRETRAIN END-OF-LOOP VANISH: ROOT-CAUSED + FIXED (t3 + box 429M 3990/4000)

**Cause:** `Pretrain/02_pretrain_linear.py` — `for step in range(start_step,
total_steps)` is exclusive of `total_steps`, while eval fires only on
`step % eval_interval == 0` and the disk save only on `step % 500 == 0` inside
it. The scheduled end-of-run eval+save (step 4000 = a %500 checkpoint AND the
next eval interval) is unreachable; steps 3991..3999 run unlogged (step log
is %10) and the loop falls out to `destroy_process_group()` + exit 0 with **no
epilogue and no final save**. The "vanish" is a clean exit with no completion
marker — not a crash/OOM/wrapper-kill (stderr is in the same log and holds no
traceback; "Starting fresh" appears once; VRAM/disk fine). Full evidence chain
in `lab/FAILURES.md` entry `pretrain-eol-vanish`. This also explains the old
v2 note "Trainer never writes step_6000; exits after final eval".

**Fix (committed):** terminal-step trigger `step == total_steps - 1` on BOTH
the eval block and the %500 save (final step is now evaluated and saved to
`step_{total_steps-1}` — e.g. `step_3999`, never `step_4000` — through the
existing path); flushed `[TRAINING COMPLETE]` banner after the loop so clean
exit is distinguishable from a vanish; `sys.excepthook` emergency path prints
+ flushes the traceback and saves `step_{last_completed}` (rank0) on any
uncaught failure (crash step is named at the last COMPLETED step so resume
re-runs the failed one). Checkpoint build/write factored into
`_build_checkpoint`/`_write_checkpoint` (still only creates `step_N` dirs that
contain checkpoint.pt).

**Repro proof:** `Pretrain/tests_end_of_loop.py` (20 steps, `tiny_agent` GLA —
lab_tiny is kda/mla and can't run unpatched on the box fla; KDA unproven on
sm_61 — free P40s, detached). Pre-fix: exit 0, no banner, no terminal ckpt =
the vanish. Post-fix: ALL PASS incl. final save at step_19 and the injected
crash at step 15 leaving step_14/checkpoint.pt + traceback.

**Still open:** `Pretrain/01_pretrain.py` has the same exclusive bound and no
epilogue (it saves at every eval_interval, so its loss window is only
eval_interval steps) — port the terminal trigger before the next dense run.
Vesper-K pretrain unblocked on this bug.

---

## 2026-10-11 ~15:00 UTC — LAPTOP OFFLINE PAUSE #2 (user at hackathon again)

Safe-shutdown wrap. State at pause: everything committed through e64fcfa.
**Done this session:** D0 PASS (c837c4d) + D2 PASS cross-family Phi-mini-MoE
(fa6ad0f) + D1 PASS with mass-cal recipe owner_mass 0.35/reject_w 10/out_pen
300 (e64fcfa; quality sub-gate open — expert-fit at delivered-mass is the
named next branch) + passport sweep complete (19cf4ed: sat at dim64=gate
rank, recal mandatory, 8MB spine + 256B/expert) + D1.5 FAIL (transplant
dead, serving-layer fallback) + trainer 3990-vanish FIXED (364d544) + SFT r2
NOT talkable (d1add87 — base pretrain is the binding limit).
**Interrupted:** agent-34 (SFT r2 recipe on box 429M step_3000) was still in
setup, no box job launched — resume from scratch on return (brief: SFT the
step_3000 pre-switch ckpt per 470m_k_sft_r2 on the free box P40s, eval_chat,
talkable verdict).
**On return:** restart farm runner (systemd-inhibit + lab/runner.py --slots 1
--watch); box P40s are free (429M run ended; step_3000/step_3500 ckpts
saved); MI50 box idle & available again (GPUs 0,3-7, never 1&2 hackathon);
budget call still open: MI300X pretrain tokens ~$65-200 (recommended) vs
nothing. v2 SFT result comparison pending. Cron coalesces while offline.

---

## 2026-10-11 — THE GRAB STORY, synthesized (MI50 day, all pushed)

Question of the day: "can you take bits of a model and get a full one?"
**Answer: yes for pretrained expert cores (wrap verbatim), yes-as-behavior
(distill), NO for scratch-trained deltas (yet) — provenance is the binding
constraint, not routing.**

| experiment | result | commit |
|---|---|---|
| D0 graft parity (OLMoE) | PASS cos 0.999 (stretch) | c837c4d |
| D2 cross-family (Phi-mini-MoE) | PASS cos 0.998 (stretch) | fa6ad0f |
| D1 plug-in (scratch delta) | formal PASS 4/5, quality FAIL | e64fcfa |
| D1.5 feature translators | FAIL 0.55-0.69 — transplant path dead | 0ed906d |
| Passport sweep | sat at dim=gate rank 64; algebraic passports free for linear gates | 19cf4ed, 4fe9e6a |
| D2 margin recal | clean negative; ceiling-normalized metric | 4fe9e6a |
| Door 3 distill | skill REAL (KL 5.17->2.59), quality FAIL | 7e0a442 |
| **Door 2.5 WRAP** | **quality PASS 0.439 vs 1.072 masked — frozen pretrained Phi core verbatim** | f348b7a |
| Delivery contract A/B | additive-lambda wins 3/5; not the binding constraint | 366c94d |
| Expert-fit v2 (contract x recipe) | FAIL in-sample both arms -> provenance diagnosis | 93188b0 |
| Provenance arms (wrap distilled core / deliver-no-refit) | RUNNING (agent-38) | — |

**Consolidated conclusions:** (1) registry/passport routing is universal
(2 families, algebraic passports pass all gates on linear-gate donors);
(2) expert provenance decides quality: frozen cores that learned something
real (donor-pretrained or distilled) + thin task-loss adapters (lr 3e-3,
wd 0) are the working recipe; scratch/refit deltas fail across 3 contracts
and 6 recipes; (3) feature-space transplant between existing models is
dead (D1.5) — grab = wrap (weights) or distill (behavior); (4) delivery
contract should be additive-lambda for plug-ins (mass competition starves
stock experts); (5) economics: ~8MB spine registry + 256B/row, wrap
adapters ~786k params.
**Open:** provenance arms (wrap-the-distilled-core) — the last rung;
expert-fit budget scaling for scratch deltas; Door-2 expressiveness tax
(needs fla GPU); wrap generalization beyond 11k calibration tokens.
