# Grab-kit: big-MoE expert-core extraction runbook

Status: **CPU/disk prep rung, frozen recipe alignment with MODULAR_MOE §10.7**.
Purpose: how to pull single expert cores from huge checkpoints (K3-class MoE,
GLM-5.3, MiMo MoE) WITHOUT loading the model, profile which expert to grab,
where the core bank lives, what wrap-fitting costs on 8×MI300X, and what each
donor's license lets us do with the combined model.

Flagship build under prep: **K3 spine + GLM cybersec cores + K3 hacking cores +
MiMo writing cores + style layer + memory stack** ("red-team copilot that
writes well and remembers"). This rung starts the build the moment bank2n's
quality half clears; no GPU work is done here.

---

## 1. The extraction pattern (our core_gen at scale)

Door25's grab (`lab/door25_wrap_expert.py:extract_donor_core`) takes one
expert's SwiGLU verbatim: `w_gu = [w1; w3]` at `(2I, H)`, `w_down = w2` at
`(H, I)`, frozen. On a 2.8T checkpoint the same grab must not load the model —
nor the shard. safetensors is a flat container:

```
[u64 le header_len][header JSON {name: {dtype, shape, data_offsets}}][raw bytes]
```

So: parse the header of ONE shard, seek to the target expert's byte ranges,
copy those ranges out. Header-only scans find which shard holds a key; the
tensors themselves are only read for the one expert. Quantized donors
(compressed-tensors `weight_packed`/`weight_scale`/`weight_shape`/
`weight_zero_point`) are extracted **verbatim** — the core bank stores what the
shard holds; dequantization is the wrap-loader's problem.

### Donor key patterns (all verified against checkpoints on disk/box)

| family | checkpoint | expert key pattern | per-expert tensors |
|---|---|---|---|
| `slimmoe` | Phi-mini-MoE (box `/data/tliao/Phi-mini-MoE-instruct`, 4 shards, 15.3GB, 32L×16E top-2, hidden 4096, inter 960) | `model.layers.L.block_sparse_moe.experts.E.w{1,2,3}.weight` | 3 × BF16 (w1/w3 `(960,4096)`, w2 `(4096,960)`) = 23.6MB |
| `olmoe` | OLMoE-1B-7B (laptop `~/.cache/vesper_d0/`, 3 shards, 5GB/5GB/1GB) | `model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight` | 3 × BF16 = 12.6MB |
| `k3` | Kimi K3 (laptop `~/Desktop/Kimi-K3/`, 96 shards, 1.56TB packed; also box `/data/ModelDownloader/Kimi-K3`) | `language_model.model.layers.L.block_sparse_moe.experts.E.w{1,2,3}.weight_{packed,scale}` | 6 × U8 (packed `(3072,1792)`×2 + `(3584,1536)`, scales `(3072,112)`/`(3584,96)`) = 17.5MB; 896 experts/layer, 92 MoE layers (1 dense + 92), top-16 |
| `glm` | GLM-5.3-Flash-AWQ-INT4 (box `/data/ModelDownloader/GLM-5.3-Flash-AWQ-INT4`, 43 shards, 197GB; 45L, 288+1 experts, top-8, hidden 4096, moe_inter 2048) | `model.language_model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight_{packed,scale,shape,zero_point}` | 12 tensors = 14.5MB |
| `mimo` | MiMo-V2.6-Flash-INT4 (box `/data/ModelDownloader/MiMo-V2.6-Flash-INT4`, EP/PP shards `model_pp0_epN_shard0.safetensors`, 310B/15B active, 48L, 256 experts, top-8) | `model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight_{packed,scale,shape,zero_point}` | 12 tensors = 14.5MB |

Notes:
- MiMo has **no standard `model.safetensors.index.json`** (EP-sharded dumps);
  the extractor header-scans shards and early-stops once the expert's full
  packed tuple is collected (HF shards hold whole tensors; one expert lands in
  one shard in practice).
- K3's `model.safetensors.index.json` has 497k keys; loading it costs ~300MB
  RSS and defeats the point. **Do not use the index for extraction** — header
  scans of the ~12 shards up to a mid-model layer are 824KB each.
- K3 experts are block-quant packed (scale along the input dim in blocks of
  64; `w1` packed width = 7168/4). Keep bytes as-is; a wrap-loader may dequant.

### The extractor: `lab/grabkit_extract_core.py`

Pure stdlib (json/struct/os) — runs under the MI50 box system python, which has
neither torch nor safetensors. Writes `<out>.safetensors` (minimal writer,
readable by the `safetensors` library) + `<out>.provenance.json`
(`vesper-core/1`: source shard path/size/header_len, per-tensor
dtype/shape/bytes/sha256, wall, peak RSS, license id). `--verify` cross-checks
every extracted byte against an independent reader (`safetensors.safe_open`);
numpy cannot represent BF16/FP8 so the verifier falls back to the torch
backend where needed.

```bash
# family auto-detects from the first expert key it sees in a shard header
python3 lab/grabkit_extract_core.py --model-dir /data/tliao/Phi-mini-MoE-instruct \
    --layer 13 --expert 8 --out /data/tliao/core_bank/phi-mini-moe/L13E8/core

python3 lab/grabkit_extract_core.py --model-dir /home/tliao/Desktop/Kimi-K3 \
    --layer 12 --expert 895 --out /data/tliao/core_bank/kimi-k3/L12E895/core

python3 lab/grabkit_extract_core.py --model-dir /data/ModelDownloader/GLM-5.3-Flash-AWQ-INT4 \
    --layer 3 --expert 0 --out /data/tliao/core_bank/glm-5.3-flash/L3E0/core
```

### Verified test results (2026-10-12, this prep rung)

| test | where | result | bytes read / shard | wall | peak RSS |
|---|---|---|---|---|---|
| **Phi-mini-MoE L13E8** (the door25 code core) | **MI50 box**, system python3, no torch/safetensors | **PASS** 3 tensors (w1/w2/w3 BF16 (960,4096)/(4096,960)) | 23.6MB / 5.00GB (0.47%) | 58ms | 62.8MB |
| Phi core round-trip | laptop `safetensors` lib | **PASS** — output file valid; sha256 `2029ab2e…/f5f81049…/476101e7…` match provenance | — | — | — |
| OLMoE L0E0 | laptop | **PASS**, `--verify` independent reader byte-identical | 12.6MB / 5.00GB (0.25%) | 13ms | 43MB |
| K3 L12E895 (fp8-packed, 17GB shard) | laptop | **PASS**, `--verify` byte-identical | 17.5MB / 16.99GB (0.10%) | 106ms | 341MB→60MB (index removed) |
| GLM-5.3-Flash L3E0 | box | **PASS** 12 tensors (packed/scale/shape/zero_point) | 14.5MB / 4.54GB (0.32%) | 54ms | 46MB |
| MiMo-V2.6-Flash L1E0 | box | **PASS** 12 tensors | 14.5MB / 12.31GB (0.12%) | 57ms | 47MB |

**Claim proven: one expert, not the model, not the shard.** Peak RSS is
~interpreter + one shard header + one expert (~60MB), independent of shard size
(5GB → 17GB → the 1.56TB checkpoint). Full K3-bank extraction (92 layers × 896
experts) is I/O-bound, embarrassingly parallel, and needs no GPU: at the
measured ~0.1s/expert + 17.5MB writes ≈ 77k experts ≈ 2.2h single-threaded on
laptop NVMe, or trivially sharded across cores/hosts.

---

## 2. Choosing which expert to grab (RouteStats on a sample) — GPU rung

Provenance decides quality (grab story): a core must be one that *learned the
specialty*, which is a routing fact. Use the existing machinery:

1. **Sample**: ~5k tokens per specialty from `Dataset/data/specialties/{name}/corpus.txt`
   (plus a generic base slice) as the profiling domains.
2. **Profile** the donor with `lab/graft_d0_parity.py:collect_route_stats` —
   layer × expert × domain activation counts + pre-gate hidden means
   (`RouteStats.home_and_prototypes`). Needs one donor forward pass per sample
   (GPU; GLM-Flash/MiMo INT4 run on the box's vLLM-capable stack; K3 profiling
   is a 2-node pass — or profile on the K3 *routed* experts via a partial
   forward to the layer of interest).
3. **Choose** with `lab/door25_wrap_expert.py:choose_donor_expert`:
   home-domain hit-rate lift over base, `rate >= 0.05`, modal domain = home.
   This is exactly how bank2n picked its rows (Phi L13e8 code / L19e12 chat /
   L23e10 science, OLMoE L7e2 code / L3e39 math …).
4. **Grab** the chosen `(L, E)` with `grabkit_extract_core.py` — CPU, seconds.

Defer all of this to the GPU rung; nothing here blocks the CPU prep. For the
flagship, expected cores: GLM-5.3 ×2 (cybersec exploit-dev + defense), K3 ×2
(hacking recon + post-exploitation), MiMo ×2 (writing prose + technical
writing) + style-layer and memory-stack slots later.

---

## 3. Core bank storage layout

```
/data/tliao/core_bank/                  # box disk; laptop mirror under ~/core_bank
  <donor_id>/                           # e.g. kimi-k3 / glm-5.3-flash-int4 /
                                        #      mimo-v2.6-flash-int4 / phi-mini-moe
    L13E8/
      core.safetensors                  # extracted tensors VERBATIM (17-25MB)
      core.provenance.json              # vesper-core/1 lineage (shard, offsets,
                                        #   sha256, license id, extractor rev)
      wrap/                             # created at wrap-fit time
        wrap.pt                         # adapter tensors + wrap_spec (door25 format)
        fit.json                        # recipe + fit/held gap curve + lam*
    BANK.md                             # generated index: donor, L, E, home
                                        #   specialty, license, bytes, sha256
```

- Naming `L{layer}E{expert}` matches bank2n/door25 conventions.
- The registry row (256B/row, d_canon=64, §9 economics) is written at assemble
  time and points at `<donor_id>/L{l}E{e}`; the bank is the artifact store.
- **Never commit bank contents.** Provenance JSONs are small enough to mirror
  into the repo later if the user wants a tracked index; the default is local.
- Storage cost is trivial: ~6 flagship cores ≈ 120MB; even a 32-core bank with
  wrap artifacts stays under 1GB.

---

## 4. Wrap-fitting memory math (frozen spine int8/fp8, 8×MI300X)

The wrap trains **in the host**: door25 Phase A runs the frozen host forward
(`model_logits`) + task-CE + `out_pen 300` L2 on base batches, and backprops
**activation grads only** to the wrap (frozen trunk: no weight grads, no
optimizer state; `freeze_all_except`). The donor's big model is *not* in the
loop — only the extracted core (17-25MB) inside the wrap.

Peak memory per wrap-fit:

```
peak = W_spine(quantized, frozen) + A_fwd/bwd(activations + activation grads)
       + wrap params + adapter-only optimizer + workspace/NCCL
```

- `W_spine`: int8/fp8 = 1 byte/param (+ small scales); packed-4bit ≈ 0.5
  byte/param. Activation-only backward saves weight-grad (2 B/param) and
  AdamW m/v+master (~12-16 B/param) on the entire spine.
- `A`: with T tokens/step, hidden d, L layers, fp16 acts stored for bwd
  ≈ `2·T·d·L·2` bytes + logits `T·V·4` for fp32 CE (chunked if needed) +
  transient activation grads of the same order. Selective checkpointing every
  c layers trades memory for `+1× fwd` FLOPs.
- Wrap (rank-64, d_canon=64): trainable `128·(d_model + d_donor)` params
  (~0.8M at Phi geometry; ~1.8M at K3 geometry) → grads + AdamW fp32 ≈ 16
  B/param ≈ **30-50MB total**. Noise.
- Workspace/NCCL: 2-4GB.

### Residency table (8×MI300X = 8×192GB = 1536GB HBM)

| frozen spine | params | int8/fp8 | packed-4bit | activations (T=2048) | fit on 8×MI300X? |
|---|---|---|---|---|---|
| Kimi K3 (host) | 2.8T | 2800GB | ~1560GB (on-disk) | 10-15GB | **NO** — int8 1.8× over; packed misses by ~25-90GB even before activations |
| GLM-5.3 flagship | 753B MoE | 753GB | 377GB | 8-12GB | **YES** (int8: ~50% headroom) |
| GLM-5.3-Flash (box) | ~360B | 360GB | 180GB | 6-10GB | **YES** |
| MiMo-V2.6-Flash (box) | 310B / 15B act | 310GB | 155GB | 6-10GB | **YES** |
| Phi-mini-MoE (sanity) | 7.65B | 8GB | 4GB | 1-2GB | YES (one GPU) |

K3-host wrap-fits (the flagship case) — pick one:

- **K3-A (recommended): 2 nodes × 8 MI300X (3072GB).** Packed-4bit K3 text
  tower (1.55TB) + activations + wrap fits with >1TB headroom; TP8/PP2 or
  ZeRO-1 over the frozen weights. Same wall as one node (compute-bound).
- **K3-B (single node fallback): frozen-weight layer streaming.** Keep 2-4
  layers (~16-62GB packed) resident per GPU, stream the rest from host RAM
  (needs ≥1.6TB RAM). Traffic ≈ 1.56TB per microbatch pass; at ~100GB/s
  aggregate ≈ +16s/step → 250 steps ≈ 1-2h/core. Works, slow.
- **K3-C (not the frozen recipe): cached plug-site features.** One streaming
  no-grad forward caches plug-layer inputs; loses the task-CE-through-host
  objective (D1.5/distill lessons say that matters). Only if neither A nor B
  is available; do not ship on it without a door-style falsifier.

### Per-core wall estimate (compute, 8×MI300X)

Step FLOPs ≈ `(fwd 2N + bwd-activation 2N + out_pen base-batch 2N) · T` = `6·N·T`
with N = active params/token (no weight-grad matmuls). T = 2048 tok/step,
250 steps (door25 recipe: ~250 steps, ~20 min on the old single-GPU rig).

| host spine | N active/token | 250-step compute @20% MFU | @35% MFU | end-to-end (×~2: evals, recal, load) |
|---|---|---|---|---|
| K3 | ~115B (16 experts·92L·66M + attn/KDA + dense) | 2.8 min | 1.6 min | **~5-15 min/core** (2-node residency; +1-2h if K3-B streaming) |
| GLM-5.3 (≈40B active, est.) | ~40B | 1.0 min | 0.6 min | ~3-8 min/core |
| MiMo-V2.6-Flash (15B active) | 15B | 0.4 min | 0.2 min | ~1-3 min/core |

Assumptions: 8×MI300X BF16 1.307 PFLOPS peak/GPU (fp8 frozen matmuls higher);
MFU 20-35% at bs 2048 with TP8 comm; selective checkpointing off (add +1× fwd
FLOPs if on). Calibration reads (§10.7 FEED: 50-200k tokens ≈ 25-100 steps at
2k tok/step — the 250-step budget spans ~0.5M token-passes, consistent with
door10's 50k-10ep regime). **Per flagship bank (6 cores): ~1h on 2 nodes
resident; ~half a day with K3-B streaming.** Wrap-fit wall is NOT the blocker
— donor extraction and profiling are already paid or cheap.

Budget check on specialty tokens (§10.7 FEED / door10): feed 50-200k tokens
per specialty; 11-12k budgets are 5-20× under (mem gap 1.35 → 0.039 at 50k;
zero-crossing ~52k). The three corpora built for this kit are sized at ~200k
tokens each (manifest: `Dataset/specialties_manifest.md`).

---

## 5. License survey (open-weights terms)

| model | terms | derivative works | attribution | commercial / scale triggers | republishing a combined model |
|---|---|---|---|---|---|
| **Kimi K3** (spine + hacking cores) | **Kimi K3 License** (Moonshot AI 2026) — MIT-style grant | allowed ("fine-tune… create derivative works") | keep copyright + permission notice | (2) MaaS business with **>$20M** aggregate revenue/12mo → separate Moonshot agreement; (3) products >100M MAU or >$20M/mo revenue must display **"Kimi K3"**; (4) internal use exempt | **Allowed** with notice; commercial MaaS/product at scale trips (2)/(3) — agreement or name display required |
| **GLM-5.3 flagship** | **GLM-5.3 License** (Z.AI 2026) — MIT-like, NOT MIT ([HF LICENSE](https://huggingface.co/zai-org/GLM-5.3/raw/main/LICENSE)) | allowed | keep notice | MaaS business with **>$10B** aggregate revenue/12mo → **pass Z.AI security review** before commercial use of the Software **or its derivative works** | **Allowed** with notice; $10B MaaS gate attaches to derivatives (republished combined model included) |
| **GLM-5.3-Flash** (box copy; `license: mit` on [HF](https://huggingface.co/zai-org/GLM-5.3-Flash)) | **MIT** (Z.AI 2026) | allowed | keep notice | none | **Allowed** with notice |
| **MiMo-V2.6** (writing cores) | **MIT** (XiaomiMiMo; on-disk README `license: mit`; [release coverage](https://techaiwire.com/articles/xiaomi-mimo-v2-6-pro-flash-open-weights-mit/)) | allowed | keep notice | none | **Allowed** with notice |
| Phi-mini-MoE (test donor) | MIT (Microsoft) | allowed | keep notice | none | Allowed |
| OLMoE (test donor) | Apache-2.0 (AI2) | allowed | keep NOTICE | patent grant; none relevant | Allowed |

Headline: **nothing blocks local use or internal republishing of a combined
model.** The binding obligations are notice retention everywhere, K3's
name-display / MaaS-agreement triggers at commercial scale ($20M MaaS
revenue / 100M MAU), and — only if cores come from the **flagship GLM-5.3**
rather than Flash — Z.AI's $10B MaaS security-review gate, which explicitly
covers derivative works. Prefer **GLM-5.3-Flash cores (MIT)** for the cybersec
slots unless the flagship's quality is specifically needed; if flagship cores
are used, record `GLM-5.3-License` in the core provenance and keep the notice
in any published derivative. Republishing the combined model commercially:
legal, but retain K3 + MIT notices and expect K3 clause 3 naming at scale.

Corpus side (calibration data, never committed): public-domain / permissive
only — NVD (PD US NIST), NIST SP 800-115 (PD), MITRE ATT&CK cti (Apache-2.0),
CAPEC/CWE (MITRE terms, attribution), Metasploit (BSD-3), Project Gutenberg
(PD). Security StackExchange (CC BY-SA) and HackTricks (CC BY-NC-SA) were
deliberately excluded (share-alike / non-commercial — not "PD/permissive");
full lineage in `Dataset/specialties_manifest.md`.

---

## 6. Rung order (what runs where, and when)

| rung | what | where | status |
|---|---|---|---|
| 0 (today) | single-shard extractor + tests, specialty corpora, runbook, licenses | laptop + box CPU/disk | **DONE** |
| 1 | RouteStats profiles → pick (L,E) per specialty per donor | box GPU (small: 5k tok/donor) | next GPU rung |
| 2 | extract the 6 flagship cores into `/data/tliao/core_bank/` | CPU, minutes | after rung 1 |
| 3 | wrap-fit (door25 recipe: rank-64, width-norm init ×√(64/d_canon), wd 0, lr 3e-3, task-CE + out_pen 300, 250 steps, FEED 50-200k) | 8×MI300X (or 2 nodes for K3 host) — per-core wall §4 | after bank2n quality half clears |
| 4 | deliver additive lam* + registry row + mex recal (owner_mass 0.85, reject_w 10) | assemble on the v0 spine | §10.7 steps 4-5 |

Ops constraints for this rung: CPU/disk only (MI50 box `/data` has the space
and the donor checkpoints; laptop fine for text/tokenizer work); no GPU jobs,
no droplets. Committed: builder + runbook + extractor + manifests; **never**
`Dataset/data/**` bins, `Dataset/data/index.txt`, `lab/imported/`, or bank
contents.
