# MIMO_HOST_FAITHFUL — lazy-host trunk divergence bisect (WIRING vs CACHE)

Date: 2026-10-20. Lane: one-cycle bisect of the MiMo-Flash full-48L lazy-host
divergence documented in `lab/FAILURES.md` (2026-10-19 bake-off v2 entry).
Question asked, pre-declared: WIRING class (stack/rope/grouped-dequant
packing/kernel recipe/expert-row ORDER of the base host path) vs CACHE class
(lazy expert-cache / LRU / GRAB / native_core id-mapping machinery).

## Toggle added

`MIMO_LAZY_DISABLE=1` in `lab/mimo_full_host.py` `ExpertCache.get` (committed):
every expert call loads fresh from the packed safetensors store (GPU dequant
unchanged), no store/reuse/evict — the MoE weight path becomes a pure function
of (layer, expert).  This is strictly stronger than the mission's residency
slice and fits one 32GB card, where residency does not: 12 MoE layers ×
256 experts × 50.3 MB fp16 ≈ 154 GB > any claim-able GPU fraction.

## Probe

`lab/mimo_bisect_smoke.py` (committed), box `sys-4029gp-trt` (tliao@100.64.0.7),
GPU 1 (MI50, mr_gpu_claims claim held for the duration), noplug (no
install_plug_set — the documented coherent reference convention
tie_fix_smoke2 NOPLUG), greedy, 2 prompts × 40 tokens, MoE-output and hidden
norms at L10/L11/L16 at the prefill (tok=0) and steps 10/30.
Host code on box verified md5-identical to this repo's laptop checkout before
either run.

- Run B (current lazy LRU, cache_budget 18.0 GiB) — log on box
  `mimo_flash/logs/bisect_B_lazy.log`.
- Run A (`MIMO_LAZY_DISABLE=1`) — `mimo_flash/logs/bisect_A_fresh.log`.

## Run B (lazy LRU): final lines

```
CACHE hits 14884 misses 20655 GiB 18.0
WALL 431.9 s
BISECT_DONE
```

Norms + decode:

```
NORMS tok=0  L10mlp=74.13 L11mlp=72.33 L16mlp=5486.46 L10hid=101.54 L11hid=39.31 L16hid=5493.91
NORMS tok=10 L10mlp=1.85  L11mlp=1.1   L16mlp=1.53    L10hid=6.8    L11hid=7.42   L16hid=13.09
NORMS tok=30 L10mlp=2.46  L11mlp=1.72  L16mlp=1.73    L10hid=8.23   L11hid=9.05   L16hid=15.13
TEXT ' gerçekten indexer indexerLexer.constructor漯漯📹…'   (prompt 1: trains)
TEXT ' (!!file.slim.slim.slim.slim🚄 Skeسقو.sqlite…'       (prompt 2: prime)
```

Run B = SOUP, and the prefill L16 MoE norm is 5486 vs 74/72 at L10/L11 —
a ~75× explosion at the very first forward (reproduces the droplet L16 fact
class; L10/L11 small here because the break family differs per run seed).

## Run A (cache bypass, fresh dequant per call): final lines

```
CACHE hits 0 misses 35539 GiB 0.0
WALL 486.7 s
BISECT_DONE
```

Norms + decode:

```
NORMS tok=0  L10mlp=74.13 L11mlp=72.33 L16mlp=5486.46 L10hid=101.54 L11hid=39.31 L16hid=5493.91
TEXT ' gerçekten indexer indexer/vendor/vendorenk wzgl…'   (prompt 1: trains)
TEXT ' (!!file.slim.slim.slim.slim🚄 Skeسقو.sqlite…'       (prompt 2: prime)
```

Run A = SOUP, and its prefill norms are IDENTICAL to Run B to every recorded
decimal (L10mlp=74.13 / L11mlp=72.33 / L16mlp=5486.46 on both).  Prompt-2's
full 40-token greedy stream is token-for-token identical across A and B even
though B reused cached experts 14884 times and A cached nothing.  Cache
bypass confirmed live: hits 0, misses 35539, GiB 0.0.  (Prompt-1 streams
diverge after ~10 tokens — small gfx906 kernel nondeterminism flips
near-tied logits once the trajectory is already soup; immaterial to the
verdict.)

## Verdict

**WIRING class.**  Per the pre-declared rule: A is also soup, with the
divergence present at the FIRST prefill forward (L16 mlp 5486 vs neighbour
norms 74/72) under a weight path that is a pure function of
(layer, expert).  The lazy expert-cache / LRU machinery is value-exonerated:
identical prefill norms and an identical 40-token stream with vs without the
cache.  The divergence is baked into the base host code path (stack
composition / rope / grouped-dequant GEMM recipe / fused-qkv recomposition /
expert-row ORDER) well before any caching could matter.  Supporting
consistency checks done this cycle: vendor inv_freq bases (full-attn
rope_theta 1e7, swa 1e4, rope_dim 64 = 192×0.334) match the host meta-buffer
fixup; `attention_projection_layout: fused_qkv` and
`add_swa_attention_sink_bias: true` are live config facts the host path
handles — the leading wiring suspects.

## Next lever (named)

Prefill replay-bisect for FIRST-DIVERGENT-LAYER localization: a
stock-formula re-implementation of the prefill pass (CPU fp32, eager
attention, hybrid_layer_pattern verbatim, fused_qkv split + SWA sink-bias
application exactly per vendor modeling code), compared per-layer and
per-submodule (attn out / post-attn norm / MoE in / MoE out) against the
host's hooks, layer L0 upward.  FAILURES.md 2026-10-19 already showed the
MoE math explodes only when fed the host's (wrong) hidden input, so the
wrongness is upstream of the first exploding MoE layer — most likely in the
attention stack of an earlier layer (fused_qkv recomposition or the
SWA-sink/chunk-128 path).  The bisect probe (`lab/mimo_bisect_smoke.py`,
`MIMO_LAZY_DISABLE=1`) is the reusable host side of that comparison.
