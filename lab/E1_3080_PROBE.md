# E1 probe — `470m_k` on the 10 GB RTX 3080 (3080+P40 box): fit + throughput

**Date:** 2026-10-19 · **Scope:** measurement only (no long run). Question
posed by the E1 pre-flight: with FUSED_CE + bnb 8-bit AdamW + small
micro-batch, does the E1 `470m_k` recipe fit on the box's RTX 3080, and if
so at what tok/s — and what wall time would one 3.77 B-token t3 arm cost?
P40s excluded by hardware (Pascal, no bf16 — the hybrid trainer needs b16
AMP).

**Verdict: FITS at micro-batch 1 only, and measured 9.1 k tok/s at full
8k context → one t3 arm ≈ 5 days on an exclusive card → NO-GO as the E1
venue.** Keep the MI300X lane (matrix anchor ~18–21 h/arm). Details below.

## Recipe capsule that ran (verbatim from the E1 pre-flight, one measurement knob added)

```
VESPER_CONFIG=470m_k VESPER_AMP=bf16 VESPER_ROUTER_TYPE=passport \
VESPER_COMPILE=1 VESPER_FUSED_CE=1 VESPER_VALUE_EMBED=1 \
VESPER_USE_MUON=0 VESPER_MICRO_BATCH=<1|2> VESPER_ACCUM=<128|64> \
[measurement only: VESPER_SEQ_LEN_START=8192 to lock context at 8k]
```

- Global batch kept identical to the recipe (128 samples/global step;
  mb1×acc128 or mb2×acc64).
- Optimizer: task spec = bnb 8-bit AdamW. The trainer's `use_muon` default
  (True) silently builds Muon+torch-AdamW *over* the bnb optimizer, so a new
  env gate `VESPER_USE_MUON=0` was added (plus `VESPER_SEQ_LEN_START`,
  seq-ramp pin for measurement only) — see "Code deltas". Live optimizer
  verified in the log: `Using 8-bit bitsandbytes AdamW optimizer` printed
  and `Using Muon optimizer` **absent** (grep count 0). bitsandbytes
  (0.50.2) was confirmed installed in the box venv *before* launch — the
  mid-run-install trap from lab/FAILURES.md did not recur.
- All three speedrun flags engaged and confirmed by their printed decision
  lines: `[compile] … compiled 80 dense submodule forwards`,
  `[fused-ce] … FusedLinearCrossEntropyLoss`, `[value-embed] … layer 0/9`.
  AMP confirmed: `Starting fresh: 470m_k (GLA hybrid, bf16 autocast…, KDA fp32)`.
- Data: the box's own staged phase bins (`nemotron_phase1/2`,
  `phase1/2_pretrain` = fineweb_v2 / code_v2 symlinks), first-256 MiB md5s
  verified identical to the laptop copies — no corpus was copied. Index =
  the 4 phase rows only; with `route_nonphase` OFF (t3 ladder semantics)
  that is bit-equivalent to the rebuilt 24-row index restricted to phase
  rows.

## Environment repair needed on the box (done before the probe could run)

The box venv (the trainerbox one) was broken for this stack out of the box;
fixing it was prerequisite, not a recipe change:

1. `flash_attn` was missing (fla 0.6.0's MLA layer hard-requires it) →
   installed the prebuilt 2.8.3 cu12/torch2.7 cp312 wheel.
2. fla 0.6.0 (user-site) vs torch 2.5.1+triton 3.1 was internally
   inconsistent (`CachedAutotuner has no attribute reset_to_zero`; and torch
   2.5 Inductor is incompatible with triton ≥3.2). Resolved by upgrading the
   venv to **torch 2.7.1+cu126 / triton 3.3.1**; fla then imports and its
   Triton kernels build on sm_86. bnb 0.50.2 GPU-step verified on the new
   torch. No user-site packages were modified.

## Measurements (all on `CUDA_VISIBLE_DEVICES=0`, the RTX 3080; probe logs kept on the box at `~/e1_probe/logs/`)

| probe | micro/accum | seq | result |
|---|---|---|---|
| A (mb2) | 2/64 | 8192 (locked) | dummy pre-alloc pass OK, step 0 OK; **OOM at step-1 backward** — 7.50 GiB torch-allocated, 9.53 GiB total demand on a 9.64 GiB card (with `expandable_segments` allocator mode) |
| B (mb1) | 1/128 | 8192 (locked) | fits; killed mid-run by a co-resident job that appeared and claimed ~1 GiB (box contention evidence) |
| C (mb1, steady) | 1/128 | 8192 (locked) | **steady 9,121 tok/s (GPU); 115.0 s/global step** (128 micro × ~0.9 s); 12 steps, 12.58 M tokens; VRAM: 5.7 GiB torch-allocated steady, **8.9 GiB card high-water** (incl. allocator reserve + CUDA/flash-attn workspaces) → **~0.7 GiB headroom** |
| D (mb1, true ramp) | 1/128 | recipe ramp from 128 | ramp step time 41.1 s at seq≈170 (fixed per-micro overhead dominates short contexts); CE 11.285→11.168 over 10 steps |

Loss sanity: CE 11.272 (step 0) → 11.010 (step 10) at lr ≤ 7.3e-6 (normal
init behavior); terminal eval Val Loss 10.9180 after 12 steps. No NaN/Inf
anywhere in any probe (tripwire clean).

### Fit bounds (the "does it fit" answer, delimited)

- Resident core: fp32 weights 2.24 + fp32 grads 2.24 + bnb 8-bit states 1.13
  ≈ 5.6 GiB before activations. mb1 @8k steady = 5.7 GiB torch-allocated;
  card-level high-water 8.9 GiB.
- **mb2 does not fit** (demand ~9.0–9.5 GiB vs 9.64 total); mb4 wasn't
  attempted (mb2 is the hard ceiling). On this card the recipe's only legal
  working point is **mb1 × accum 128**.
- Co-tenancy: a ≥~0.5–1 GiB neighbor arriving mid-run is enough to OOM a
  running mb1 job (observed once). An honest run policy on this box is
  "exclusive card or nothing".

## Projection: one 3.77 B-token t3 arm on the 3080

Recipe shape: 4000 global steps = 800-step seq ramp (~0.43 B tokens) +
3200 steps at full 8k (1.0486 M tokens/step). Measured step-time model from
probes C/D: t(seq) ≈ 39.4 s + 0.00923 s·seq (fixed per-micro overhead is
~39 s/step at mb1 — kernel launches/python eager loop dominate the short
contexts; at 8k the same overhead is ~34% of the 115 s step).

| segment | tokens | basis | wall |
|---|---|---|---|
| seq ramp (steps 0–799) | 0.43 B | ∫ t(seq) over ramp (measured 2 pt fit) | ≈ 17.3 h |
| steady 8k (steps 800–3999) | 3.36 B | 3200 × 115.0 s measured | ≈ 102.2 h |
| evals + checkpoint I/O | — | 40 evals × ~1.5 min + saves | ≈ 1–2 h |
| **Total per arm (exclusive card)** | **3.78 B** | | **≈ 120 h ≈ 5.0 days** |

Blended ≈ 8.7 k tok/s. A paired passport/top-k E1 would be **~10 days
sequential**. For reference the MI300X anchor is 18–21 h/arm.

## Recommendation — NO-GO for the 3080 lane

1. **Wall time ≈ 5 days per arm** even on an exclusive card — past the
   "projection > 5 days" disclosure threshold, and two arms ≈ 10 days.
2. The box's rules are short-tests-only during the P40 sale window; a
   multi-day reservation of the surviving GPU is not available.
3. Measured co-tenancy: transient neighbors on the 3080 killed one probe;
   mb1 has only ~0.7 GiB headroom, so the card must be exclusive for the
   whole 5-day span — unrealistic on this box.
4. What the probe does buy: a verified **runtime recipe for the 3080**
   (capsule above — bnb optimizer verified in-log, all flags engaged, mb1
   ceiling) usable for short canary runs, replay sanity checks, or smoke
   shards; and confirmation that nothing in the recipe blocks sm_86
   (compiles, trains, loss moves).

## Code deltas (committed with this report)

- `Pretrain/02_pretrain_linear.py`: two measurement-only env gates —
  `VESPER_USE_MUON=0` (leaves the bnb 8-bit AdamW as the sole live optimizer
  instead of the Muon+AdamW split; default unchanged) and
  `VESPER_SEQ_LEN_START` (seq-ramp pin for steady-state probes; default
  unchanged). Both print loud `[env]` lines.
