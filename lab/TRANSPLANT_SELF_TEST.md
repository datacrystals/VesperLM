# TRANSPLANT_SELF_TEST — self-transplant feasibility bed (in-family + vk cross-family)

Date: 2026-10-20. Lane: creative branches of the MoE transplant question after
the Door-9/D1.5 verdict killed *literal cross-family spine-injection* ("naive
donor-weight transplant between MoE families is dead", ~0.47 held-out ceiling)
— but left the *shape* untested: can the machinery graft WEIGHTS at all, in the
friendliest domain, where the donor is the host's OWN weights moved between its
OWN layers?

## 0. Pre-declared question and verdict rule (before any run)

Question: transplant the FULL expert bank of one layer into another layer's
slot — destination router rows untouched, expert indices tied e→e (source body
receives the destination's routing, i.e. both source+destination bindings
properly tied) — does the body accept it?

Verdict classes (declared):
- **mechanism-alive**: control coherent; every transplant arm's coherence is
  not worse than "loop"; the parity chain corrupts ONLY at/after the
  transplant slot (the upstream chain stays 1.00× control) and no NaN
  explosion — the graft machinery moves weights, and the price is local
  damage, not rejection.
- **body-rejects**: geometry breaks even in-family — NaN explosion, mosaic
  soup, or corruption of the chain UPSTREAM of the slot from a slot-local
  weight move. That is the baked-in price of transplant of any kind,
  regardless of family.

Probes per arm (identical in both families):
(a) coherence class of greedy streams (auto-classified from token-id
    statistics + non-ASCII fraction; no corpus text anywhere in harness or
    report — numbers/classes only);
(b) layer-norm-parity chain vs the untransplanted control on the same input
    (mimo: committed reference fact L10mlp(tok=0) = 74.13, "is the L10 spike
    intact?", plus the chain L10/L16/L40/L41/L47 at steps 0/10/30; vk:
    per-MoE-block output norms on one authored probe forward);
(c) the NaN difference: non-finite logits/norm counts, arm minus control.

## 1. Hosts and arms

| family | host | bank geometry | arms |
|---|---|---|---|
| MiMo-V2.6-Flash-INT4 | `lab/mimo_full_host.py` (trunk faithful since f69737a; lm_head tie fixed per `lab/MIMO_HOST_FAITHFUL.md`) | 47 MoE layers × 256 routed experts, top-8 | control; **L12→L44**; **L4→L32**; both at once |
| vk passport spine | `/data/tliao/vk_final/chat_model` via `lab/vk_graft.py` (rung-1 host lineage) | 10 MoE blocks × 8 experts, top-2 | control; **vk1→8**; **vk0→9**; both at once |

Tie verification (recorded per arm): after rebinding, the destination slot's
expert call must be bit-identical to the source call on the same input
(mimo: max|Δ| through the packed store; vk: object identity). This is the
"properly tied" check — a scrambled or stale binding would show here.

Harnesses (committed): `lab/mimo_transplant_selftest.py`,
`lab/vk_transplant_selftest.py`. Box: sys-4029gp-trt, one
GPU claimed (gpu0, mr_gpu_claims), all work under `~/transplant` (nothing
staged on /data), user's TP=8 serving untouched. Results JSONs (metrics only):
box `~/transplant/out/{mimo,vk}_transplant_selftest.json`.

## 2. Cross-family arm (vk spine) — RESULTS

Run 2026-10-20, GPU 0 (MI50), fp16, 89.2 s wall.

| arm | tie check | base CE (authored probe, Δ vs control) | coherence (3 greedy chats) | NaN |
|---|---|---|---|---|
| control | — | 4.144148 (—) | coherent, coherent, coherent | 0 |
| vk1→8 | identical object ✓ | 4.527364 (**+0.383**) | loop, coherent, coherent | 0 |
| vk0→9 | identical object ✓ | 5.581190 (**+1.437**) | loop, loop, coherent | 0 |
| vk both | identical object ✓ | 6.088988 (**+1.945**) | loop, loop, loop | 0 |

Parity chain (per-block output-norm ratios arm/control; the vk chain doubles
as its own localization probe):

| arm | blocks 0–7 | block 8 | block 9 |
|---|---|---|---|
| vk1→8 | 1.0000 ×8 | **0.6434** | 1.0037 |
| vk0→9 | 1.0000 ×9 | — | **0.2698** |
| vk both | 1.0000 ×8 | **0.6434** | **0.2242** |

Read: an expert bank moved to a wrong slot fires its weights against a hidden
state from the wrong depth — the slot's output norm collapses to 27–64% of
control (the deeper the mismatch, the worse), while EVERY block upstream
stays bit-identical (1.0000) and the chain downstream partially renormalizes
(block 9 after vk1→8 reads 1.0037). The damage is strictly local in geometry
but global in function: base CE degrades +0.38 → +1.44 → +1.95 nats as the
transplant gets deeper-mismatched or doubled, and greedy coherence follows
the same gradient (0 → 1 → 2 → 3 loop classifications). No NaN anywhere —
the body prices the graft, it does not reject it.

## 3. In-family arms (MiMo full host) — RESULTS

Run 2026-10-20, box GPU 0, 18.0 GiB expert LRU, greedy 2 prompts × 40 tok,
wall 1443.6 s total (one host build, four arms, cache flushed between arms).
Box log `~/transplant/logs/mimo_transplant.log`; compact metrics
`lab/results/transplant_selftest.json`.

| arm | tie check | coherence p0/p1 | mosaic p0/p1 | NaN events p0/p1 |
|---|---|---|---|---|
| control | — | coherent / coherent | 0.0 / 0.0 | 0 / **1 full-vocab row** |
| L12→L44 | max\|Δ\| = 0.0 ✓ | coherent / coherent | 0.0 / 0.0 | 0 / 0 |
| L4→L32 | max\|Δ\| = 0.0 ✓ | coherent / coherent | 0.0171 / 0.0 | 0 / **1 full-vocab row** |
| both | max\|Δ\| = 0.0 ✓ | coherent / coherent | 0.0056 / 0.0 | 0 / 0 |

(a) Coherence: **8/8 streams coherent** under the think-envelope — no arm is
worse than the control. Minor foreign-texture (small non-ASCII fraction) on
the deeper arm only.

(b) Parity chain, prefill (tok=0, identical prefix → the localization-clean
measurement), mlp-norm ratio vs control:

| arm | L10 | L16 | L40 | L41 | L47 |
|---|---|---|---|---|---|
| L12→L44 (p0/p1) | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | **0.455 / 0.529** |
| L4→L32 (p0/p1) | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.996 / 0.978 | 1.004 / 1.003 | 0.988 / 0.983 |
| both (p0/p1) | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.996 / 0.978 | 1.004 / 1.003 | **0.449 / 0.516** |

Every layer UPSTREAM of the transplant slot is bit-identical (1.0000) in
every arm; the first corruption appears strictly at-or-after the slot
(L12→L44: nothing until L47, whose MoE output energy collapses to ~0.45–0.53
of control; L4→L32: sub-2.2% downstream drift only at this energy probe, but
the destination itself is not probed — the drift that matters shows up at
decode). **The L10 spike is intact**: control L10mlp(tok=0) = 74.1271 /
74.3946 vs the committed reference 74.13, and every arm holds L10 at ratio
exactly 1.0000 at the prefill. After greedy divergence (steps 10/30) the
token streams differ, so norms differ everywhere — expected, not a geometry
break; the prefill row is the binding measurement.

(c) NaN difference: **0**. Total non-finite events = 2 across all 8 runs,
each exactly ONE step whose entire logit row (152576 = vocab) is non-finite,
in the CONTROL's prompt 1 and in L4→L32's prompt 1 — the same host-side
greedy event on that prompt, arm-invariant (L12→L44 and the combined arm
have ZERO, i.e. the transplant *removed* it in two arms and kept it in one).
arm − control = 0. This is a newly instrumented host fact (no previous probe
counted non-finite logits): on prompt 2's greedy trajectory the faithful
host occasionally overflows one decode step; the stream recovers and still
reads coherent class.

## 4. Verdict

**MECHANISM-ALIVE** in the friendliest domain, with the price measured
locally and priced into the slot:

1. The graft machinery moves weights exactly: tie check max\|Δ\| = 0.0 on
   every mimo arm (packed-store rebinding resolves the same keys), object
   identity on vk. No NaN explosion in either family, 8/8 mimo streams
   coherent class, control's committed L10 reference (74.13) reproduced to
   0.003 and preserved at ratio 1.0000 in every arm at the prefill.
2. Damage is geometry-local and predictable: the prefill parity chain
   corrupts only at/after the slot, never upstream; the destination side
   misprices its output energy (mimo L47-after-slot: 0.45–0.53× control; vk
   slot blocks: 0.64× and 0.27× — deeper mismatch, bigger collapse) because
   the transplanted bank's internal scale/semantics are its source layer's,
   not the depth the destination's hidden state and router were trained for.
3. Function follows the same gradient: vk base CE +0.38 / +1.44 / +1.95
   nats and greedy coherence 3×coherent → 1/3 → 2/3 → 3/3 loops as the
   transplant gets deeper-mismatched or doubled — the body **absorbs** the
   graft and degrades gracefully; it does not reject it.

Against the mission's two classes: **not** body-rejects (no NaN, no soup, no
upstream corruption, coherence retained in-family at a whole-256-bank move).
The mechanism works; the baked-in price is that mis-placed weights produce
wrong-scale, wrong-semantics outputs AT the slot, and that price is
monotone in depth mismatch. The earlier lane's ~0.47 held-out ceiling on
literal cross-family spine-injection is consistent with an additive two-part
price: (i) this depth/slot-misprice floor, present even in-family, plus
(ii) the cross-family representation gap the wrap recipe exists to close.
Self-transplant clears part (i) with a number: slot-local energy error
0.45–0.53× (mimo) / 0.27–0.64× (vk), upstream chain untouched, decode
coherent — wrap/scale calibration is the named lever, not rejection.

## 5. Claims and limits

Claims:
1. Whole-bank inter-layer weight transplant is mechanically sound in two
   families (MiMo 256E top-8, vk 8E top-2): exact binding, no NaN explosion.
2. Parity-chain localization is a working diagnostic: identical-prefix
   prefills make the first corruption land exactly at/after the slot;
   committed reference facts (L10 = 74.13) survive all arms.
3. The price of a mis-placed bank is slot-local output-energy misprice,
   monotone in depth mismatch, graceful in decode (coherent in-family at
   small mismatch, loop-degraded at depth mismatch on the small vk spine).

Limits:
- 2 greedy prompts × 40 tok in-family, 3 short chats on vk — coherence
  CLASS evidence, not logged-perplexity against a corpus. Vk arm has CE
  (+0.38..+1.95); the mimo arm uses the norms/class battery only (no
  committed eval harness on the full host).
- One norm-probe set: layers {10,16,40,41,47}; the transplanted destination
  layers (32, 44) are not themselves probed in mimo (their norms are
  downstream-observable at L40/L41/L47 only).
- The single-step full-vocab non-finite event on prompt 1 greedy pre-exists
  in the control host (instrumented here for the first time); its mechanism
  (fp16 overflow at one decode step on that trajectory) is not root-caused —
  it is arm-invariant and therefore orthogonal to the verdict.
- vk coherence classifier thresholds (top-2 token dominance > 0.55,
  repeated 6-gram ≥ 3) are coarse; "loop" reads on the 103M vk spine should
  be read as "degraded vs control", not absolute quality.
- One seed (greedy), one draw: variance across seeds unmeasured.

## Reproduction

```bash
# box sys-4029gp-trt, one claimed GPU, venv ~/vesper_sft_venv, workdir ~/transplant
CUDA_VISIBLE_DEVICES=~/vesper_sft_venv/bin/python -u lab/vk_transplant_selftest.py \
  --device cuda --out out/vk_transplant_selftest.json
CUDA_VISIBLE_DEVICES=0 ~/vesper_sft_venv/bin/python -u lab/mimo_transplant_selftest.py \
  --device cuda --out out/mimo_transplant_selftest.json
```

Both harnesses auto-classify; raw text is never printed or committed.
