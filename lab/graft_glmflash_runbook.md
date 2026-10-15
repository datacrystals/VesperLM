# GLM-5.3-Flash registry/graft parity port — runbook (D-gate on the Flash spine)

Status: **first rung of the GLM-5.3-Flash-spine build** (user directive:
"we def need a smarter base — perhaps we base it off of like glm 5.3 flash as
the base eventually"). This is the proven hours-scale step — registry/graft
parity on the new host. NO wrap fits here (those are the "go" step).
Code: `lab/graft_glmflash.py` (reuses `lab/graft_d0_parity.py` passport /
registry / recal core unchanged — the D2 port lesson: adapt the router block
once, everything else follows).

Host: MI50 box `tliao@100.64.0.7`, checkpoint
`/data/ModelDownloader/GLM-5.3-Flash-AWQ-INT4` (184GB, 43 shards; 45L,
288+1 experts, top-8, hidden 4096, moe_inter 2048, mHC 4 streams; KDA linear
attention + DeepSeek-sparse attention; `first_k_dense_replace=3`). License
**MIT** (runbook grabkit §5 — not the flagship GLM-5.3 license).

---

## 1. Load path achieved (host-adapt)

Full 184GB resident is impossible on this lane (≤2 × 32GB MI50 while
co-tenants hold the rest; 8×32=256GB gross). What `build_host` does instead
("single-shard host-adapt"):

1. **transformers 5.19 native `Glm5NextTextModel`** (the box venv
   `/data/tliao/vesper_sft/venv` has transformers 5.19 + torch 2.9.1+rocm6.4
   + flash-linear-attention 0.6) instantiated on **meta device** — no expert
   allocation (the 3D expert bank would be 19GB/layer otherwise).
2. **Non-expert weights resident on one GPU** (~17GB fp16): embed + lm_head,
   KDA/DSA attention, norms, mHC, router gates (`gate.weight` BF16 (288,4096)
   + `e_score_correction_bias` fp32), shared experts, dense MLPs. Compressed-
   tensors `group_1` FP8-E4M3 block-128 tensors dequantized on load
   (`w.float() * scale` over 128×128 blocks).
3. **Routed experts streamed on demand** from the shards
   (`StreamedExperts`): only experts the current tokens select are read —
   `weight_packed/weight_scale/weight_shape/weight_zero_point` (pack-quantized
   int4, group 32, asymmetric) dequantized **on device** (bit-unpack via
   int64 shifts; verified bit-identical to the CPU path of
   `lab/v05_core_prep.dequant_int4_core`, which was itself checked against
   the compressed-tensors oracle). Device LRU cache (default 4GB).
   Note: most Linear layers (q/k/v/o, indexer, …) are ALSO int4 packed in
   this checkpoint — only some `group_1` targets are FP8; the loader
   dispatches on the key format, not on layer kind. Routed experts are int4
   on layers 3-43 but **FP8 block-128 on layer 44 (and the MTP layer 45)** —
   `StreamedExperts` dispatches per expert on the same rule.
4. **gfx906 kernel shims** (`install_torch_kda`, auto on HIP):
   - fla's Triton KDA (`chunk_kda`/`fused_recurrent_kda`) dies at LLVM
     lowering on gfx906 at every shape (FAILURES fla-kda-gfx906) →
     transformers' own pure-torch reference bodies (`__wrapped__` of the
     `use_kernel_func_from_hub_with_fallback` decorators) are forced in;
     same for `causal_conv1d_fn/update`.
   - `torch.nonzero` and bool-mask column indexing (`x[:, keep]`) launch
     kernels gfx906 does not have (`hipErrorInvalidDeviceFunction`; op probe
     `lab/graft_glmflash.py` era, 2026-10-15). The DSA indexer's
     `get_pooled_states` is reimplemented with CPU-side index derivation +
     `index_select`; `StreamedExperts` bookkeeps on CPU and GEMMs on device
     (`index_select`/`index_add` are fine). Verify any new op with a 10-line
     probe before trusting it — holes are op-specific, not family-wide.

Measured (MI50 GPU1, fp16, claim `/data/hackathon/.tmp/mr_gpu_claims/gpu1`):
host build ~33s for a 6L window / ~3 min for 45L (staging dequant dominates
at first, then page-cache warm), first parity batch ~34s (expert cold loads),
subsequent ~4-11s. **Full-stack honesty note (2026-10-15)**: the 45L
streaming host builds and runs (the stock arm executed through layer 43 on
the first full attempt) but the resident set (~18GB non-expert + expert LRU +
KDA/dequant workspace) does not survive the planner co-tenant's growth on a
32GB MI50 — three OOMs at 27GB allocated (`chunk_kimi_delta_attention`
decay_mask, then the expert bit-unpack) with `expandable_segments` unsupported
on gfx906. Mitigations already in the code: KDA reference forced to
`chunk_size=16` (the blocked scan is chunk-size agnostic; default 64 padded
even 8-token slices to a 128MB decay_mask) and `torch.cuda.empty_cache()`
every 16 expert loads (the caching allocator otherwise retains every freed
50MB expert block). Full final-logit parity becomes a **multi-GPU claim job**
(2×32GB MI50 with a quiet co-tenant, or the MI300X lane); until then the
**window/layer-local form is the honest gate** (truncated-spine logits +
per-layer MoE metrics on up to 24L).

## 2. Router family + the graft

Stock `Glm5NextTextTopkRouter` = **sigmoid + noaux_tc** (the D2-foreign-router
story again, this time DeepSeek-V3 style rather than sparsemixer):

    logits = h @ W.T                      # fp32
    scores = sigmoid(logits)
    choice = scores + e_score_correction_bias
    group-limited top-k (n_group 1 / topk_group 1 here), top-8 on `choice`
    weights = scores.gather(topk)         # UNBIASED sigmoid multipliers
    renorm (norm_topk_prob) * routed_scaling_factor 2.5

The graft replaces **only the score function**: `logits = (W_q h . p_e) /
sqrt(d_r) * layer_scale` (§4.3 passport rows); sigmoid + bias + group + topk
+ renorm + scale run unchanged. `init=exact` (P = sqrt(d_r) * W_stock,
identity W_q) is the algebraic ceiling — bit-level stock-logit identity — so
that arm measures **host-adapt fidelity** (is the port corrupting the route?);
`init=prototype` + short recal is the real registry arm.

## 3. Commands (box; claim-locked, detached, never user sessions)

```bash
# claim: touch the door10-protocol claim file for the GPU you size into
touch /data/hackathon/.tmp/mr_gpu_claims/gpu1
cd /data/tliao/vesper_sft/VesperLM
PY=/data/tliao/vesper_sft/venv/bin/python
ENV="ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library \
 PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
 HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1"

# full-stack parity, exact init (plumbing gate; TRUE final-logit cosine)
env $ENV $PY lab/graft_glmflash.py \
  --model /data/ModelDownloader/GLM-5.3-Flash-AWQ-INT4 \
  --phase all --init exact --mode full --seq-len 16 --batch-size 1 \
  --max-tokens 32 --val-file /data/tliao/graft_d2_slice/val_mixed.txt \
  --out /data/tliao/glmflash_out/exact_full.json

# registry arm: prototype init + short recal, then parity
env $ENV $PY lab/graft_glmflash.py \
  --model /data/ModelDownloader/GLM-5.3-Flash-AWQ-INT4 \
  --phase all --init prototype --recal-steps 40 --mode full \
  --seq-len 16 --batch-size 1 --max-tokens 32 \
  --val-file /data/tliao/graft_d2_slice/val_mixed.txt \
  --out /data/tliao/glmflash_out/proto_full.json

# layer-local fallback (truncated-spine logits + per-layer MoE metrics)
env $ENV $PY lab/graft_glmflash.py ... --mode window --max-layers 6 \
  --layers 3,4,5 --out /data/tliao/glmflash_out/window_6L.json
```

Bars (standing, §10.8 / graft_d0_runbook): logit cos ≥ 0.95 (stretch ≥ 0.99
exact), |dCE| < 0.05, per-layer MoE-out cos > 0.9, top-k agreement > 0.8.
`verdict`/`pass` in the output JSON enforce them.

**Measured (2026-10-15, MI50 GPU1, fp16, real slice
`/data/tliao/graft_d2_slice/val_mixed.txt`)**:

| arm | form | logit cos | min | dCE | layer cos | top-k |
|---|---|---|---|---|---|---|
| `--init exact` | 6L window | 0.999955 | 0.997485 | +0.000126 | 0.999267 | 1.0 |
| `--init prototype --recal-steps 150` | 12L window (9 MoE layers) | 0.998959 | 0.996479 | +0.042761 | 0.996338 | 0.791667 |

Standing bar ≥0.95 **cleared in both arms** (stretch ≥0.99 too). Secondary:
|dCE| 0.0428 PASS (tight), layer cos PASS, **top-k 0.792 vs 0.8 MISS** —
hard-selection family effect (sigmoid+noaux_tc flips near-threshold experts);
named lever = score-margin-aware recal (`--recal-margin-w` family in
`graft_d2_crossfamily.py`) + the noise-matched family-relative agreement
baseline. First prototype attempt was 0.6247 until d0's closed-form
`fit_layer_scales` was added after prototype init (scores off the stock-logit
scale otherwise) — do not skip it.

Registry artifacts (box): `/data/tliao/glmflash_out/*_registry.pt`
(`vesper-graft-registry-v1`, 288 rows/layer, d_r 4096).

**Wrap proof-of-life (mem_trains, 2026-10-15)**: `lab/graft_glmflash_wrap_poe.py`
— §10.7 loop runs on the Flash host (GRAB banked GLM L3E0 core → rank-64
wrap → additive plug at the first MoE layer → short fit). 3 steps: home CE
13.249→10.946→9.326; aggregate home Δ −0.815; base tax −0.577 CE; base
logit-cos plugged-vs-bare 0.982985. Toy FEED / truncated spine — proof of
the loop, not quality (quality fit = MI300X go-step).

## 4. THE FULL BUILD PLAN (next rungs — the "go" step)

The user's bar is **conversation quality** (the 325M base's drift is what a
GLM-5.3-Flash spine fixes: real instruction-following + knowledge instead of
recipe-shaped babble). Everything below transfers §10.7 unchanged.

1. **Spine install**: full-stack host (this runbook §1) becomes the chat
   backbone; chat template = the checkpoint's `chat_template.jinja` +
   `tokenizer.json` (already on box). Chat-CLI port is a thin fork of
   `lab/chat_vfinal.py` / `lab/vk_graft.py:VkKVDecoder` — replace the vk-base
   loader with `graft_glmflash.build_host`, keep the additive-λ plug
   delivery, `/sys` command, and rep-penalty decode defaults (9038e07
   shipped T=0.8 + rep-pen 1.1 + max-new 128 — the convo-quality fix, do not
   regress to greedy).
2. **§10.7 wraps against this host** (persona/memoria plugs: mem_trains,
   mem_pack trio, persona plugs — the `vk_final_build.py` row table):
   GRAB (already banked: `/data/tliao/core_bank/glm-5.3-flash/…` from grabkit)
   → WRAP rank-64 width-norm init, wd 0, lr 3e-3, task-CE + out_pen 300,
   ~250 steps → FEED 50-200k specialty tokens → DELIVER additive λ (fitted) →
   ROUTE registry row + mex recal. Recipe is host-agnostic; only the
   **shape grid** is new (gfx906 backward holes are spine-specific — run the
   B*T grid first; v0.6's Qwen safe shapes were bs1×256).
3. **Wrap-fitting cost on a 360B spine**: the frozen-spine term is the
   binding one (grabkit §4): GLM-5.3-Flash int4 184GB packed / ~360GB at
   int8 → **needs multi-GPU claims** (8×MI300X 1536GB is the clean venue;
   2-node not needed). On the MI50 box a wrap fit is only realistic with the
   streaming host + adapter-only backward (activation ckpt + AdamW on rank-64
   is free) BUT the expert-streamer cache thrash under backward makes it
   hours/wrap → budget MI300X for the fit lanes, MI50 for registry/parity/
   demos. Per-core wall @8×MI300X ≈ 5-15 min (same math as grabkit §4).
4. **Demo bar target**: `lab/convo_test.py` (10 natural multi-turn
   dialogues × 4 rubrics: no_rep / answers / length / persona) + `eval12`
   with the SHIPPED decode defaults; acceptance = the convo-quality numbers
   the 325M base misses (dialogues usable 5/10 → target 9-10/10; eval12
   10/12 → target 12/12) and interactive probes that answer what was asked.
   Measure bare-spine vs spine+plugs on the same script.
5. **Optional proof-of-life wrap** (only after parity lands): one persona
   plug (mem_trains) fitted against the Flash host with §10.7 numbers —
   validates the whole GRAB→WRAP→FEED→DELIVER→ROUTE chain on the new spine
   before the full plug suite.

Caveats to carry forward: co-tenant claim races (retry mandatory; the parity
phase OOM'd v0.5-era runs when a co-tenant landed — ours stream experts so
the resident set is ~17GB + LRU, sized into the free VRAM of the claimed
GPU); `/data` is tight on the box (keep registries slim; cores already
rehydrate from core_bank); never stage weights/lab artifacts in git.
