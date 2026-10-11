# D2 cross-family graft-parity — runbook (RAN 2026-10-11 on the MI50 box, GPU 5)

Gate D2 attempt 1 of `MODULAR_MOE.md` §9.4 in the lane's cross-family reading:
graft a **foreign-family** open MoE through the same passport/registry
interface that D0 validated on OLMoE, and check the same parity protocol.
Harness: `lab/graft_d2_crossfamily.py` (imports the passport/metrics/registry
core from `lab/graft_d0_parity.py`; adds a family-adapter layer).

## 1. Donor model

- Model: `microsoft/Phi-mini-MoE-instruct` (Phi family / Microsoft "slimmoe")
- **Pinned revision: `f620b32c0d3e8f7e76f57ccdaa88e0df8bc8bfcd`**
  (hub sha `f620b32c0d3e8f7e76f57ccdaa88e0df8bc8bfcd`, lastModified 2025-12-10)
- Size: 7.65B params, 32 layers, hidden 4096, E=16, top-2, inter 960,
  vocab 32064. fp16 weights ≈ 15.3GB — fits one 32GB MI50 card cleanly at
  bs2×512 (measured peak in the results JSON).
- Local path (box): `/data/tliao/Phi-mini-MoE-instruct` (15GB, outside the repo
  — NEVER commit weights). Re-fetch:

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('microsoft/Phi-mini-MoE-instruct',
    revision='f620b32c0d3e8f7e76f57ccdaa88e0df8bc8bfcd',
    local_dir='/data/tliao/Phi-mini-MoE-instruct')"
```

Why this donor (vs the lane's listed candidates):

- `Qwen1.5-MoE-A2.7B` (14.3B total) and `DeepSeek-V2-Lite` (16B) are ~29–32GB
  at fp16 — they do not fit one card alongside activations/logits, and the lane
  forbids 2-card device_map. `MiniCPM-MoE-8x2B` is a 27.7GB single
  `pytorch_model.bin` with 4.36-era custom code (same story). Phi-mini-MoE is
  the largest *native-transformers* MoE that fits one 32GB card at fp16 with
  OLMoE-class headroom.
- The family is genuinely foreign to OLMoE and so is the **router math**:
  **sparsemixer** (relative-threshold-masked sequential top-2; the two
  multipliers come from two *different* masked softmaxes, so they do not
  renormalize together) vs OLMoE's plain softmax-top-8-no-renorm. Expert block
  is a ModuleList of SwiGLU MLPs (w1/w3/w2) vs OLMoE's grouped 3D bank, and
  the router lives at `block.router` vs `block.gate` — all three differences
  are what the D2 adapter layer has to absorb.

## 2. Checkpoint layout adaptation (the foreign-family work)

Phi-mini-MoE ships the slimmoe checkpoint layout
(`model.layers.N.block_sparse_moe.gate.weight` + `...experts.M.w{1,2,3}.weight`)
with 4.43-era custom modeling code that hard-imports `flash_attn`
(unavailable on gfx906/ROCm). The harness instead loads through **native
transformers `phimoe`** (5.19) after a mechanical key remap
(`remap_slimmoe_checkpoint`, auto-run by `load_donor`):

- `block_sparse_moe.gate.weight` → `mlp.router.weight`
- `experts.M.w1` and `experts.M.w3` → `mlp.experts.gate_up_proj` (E, 2I, H)
  stacked as `[w1; w3]` per expert, experts stacked in checkpoint expert-id order
- `experts.M.w2` → `mlp.experts.down_proj` (E, H, I)
- everything else (attention/LN/embeddings/lm_head) kept as-is

Native `phimoe` runs the donor's own `sparsemixer` (imported by the harness for
the graft's selection step so semantics cannot drift). Remap verified by
(a) sample-text CE 3.69 nats and (b) `--init exact` 1-batch parity
cos 0.999999 — i.e. the remap is math-preserving. The remapped checkpoint
lands at `<model>-native/` and is regenerated idempotently.

gfx906: the two D0 shims (grouped-mm off, CPU-sort expert merge) auto-deploy
and cover native `PhimoeExperts` via the same `gate_up_proj`/`down_proj`
duck-type. No new kernel walls.

## 3. Commands

Gate 0 — harness self-test (any machine, CPU, ~9s, no donor):

```bash
python3 lab/graft_d2_crossfamily.py --synthetic
```

Must print `14/14 synthetic checks passed` (mock phimoe-family block with a
real sparsemixer router + an OLMoE-family adapter regression).

Gate 1 — the D2 measurement (GPU run; this box, **GPU 5 only**):

```bash
PYTHONPATH=/data/tliao/d2_lab HIP_VISIBLE_DEVICES=5 \
ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library \
python3 lab/graft_d2_crossfamily.py \
  --model /data/tliao/Phi-mini-MoE-instruct \
  --revision f620b32c0d3e8f7e76f57ccdaa88e0df8bc8bfcd \
  --family phimoe --device cuda --dtype float16 \
  --phase all --init prototype \
  --seq-len 512 --batch-size 2 --max-tokens 100000 \
  --recal-steps 300 --recal-rounds 2 --max-cal-pairs 4096 \
  --domains-dir /data/tliao/graft_d2_slice \
  --val-file /data/tliao/graft_d2_slice/val_mixed.txt \
  --registry /data/tliao/graft_d2_out/registry_phimoe_f620b32c.pt \
  --out /data/tliao/graft_d2_out/phimoe_d2.json
```

Reference arm (algebraic ceiling; expected cos ≈ 1):

```bash
... same but --init exact --recal-steps 0 --registry .../registry_phimoe_exact.pt \
    --out .../phimoe_d2_exact.json
```

fp16 required: gfx906 has no bf16 (same as D0). Use `--dtype bfloat16` on
MI300X/Ampere+.

## 4. Eval data

Same recipe as D0 (`lab/graft_d0_build_slice.py`), re-run with the **Phi
tokenizer** so token accounting matches the donor (counts differ from OLMoE's):

```bash
python3 lab/graft_d0_build_slice.py \
    --tokenizer /home/tliao/.cache/vesper_d2/Phi-mini-MoE-instruct \
    --out-dir /home/tliao/graft_d2_slice --domain-tokens 20000 --val-tokens 80000
```

Sources (unchanged from D0): code=`codeparrot/codeparrot-clean-valid`,
math=`open-web-math/open-web-math`, web=`Salesforce/wikitext` (wikitext-2-raw),
science=`fancyzhx/ag_news` (sci/tech), chat=`OpenAssistant/oasst1` (en).
Output (2026-10-11): **105,546 domain + 88,870 val Phi tokens** (5 domains ×
~20.5k + val 88.9k) — clears the ≥50k val bar. Built on the laptop (the box's
`datasets` is broken by a NumPy 1.26/pyarrow conflict — see FAILURES.md) and
rsync'd to `/data/tliao/graft_d2_slice/` on the box.

## 5. PASS bars (identical to D0)

mean final-logit cosine > 0.9 (stretch > 0.99), |ΔCE| < 0.05 nats, per-layer
MoE-output cosine > 0.9 mean, top-k set agreement > 0.8,
purity_graft(d) ≥ purity_stock(d) − 0.02 every domain.

## 6. Evidence (RAN 2026-10-11, MI50 box GPU 5)

**Verdict (88,064 val tokens): logit cos 0.99863** (std 0.0238, min −0.4786)
| **ΔCE +0.03028** (stock 1.9054 → graft 1.9357) | per-layer MoE-output cos
**0.8640 mean** (0.80–0.97, U-shaped over depth) | top-k set agreement
**0.6509 mean** (Jaccard 0.761) | purity margins −0.0001…−0.0005 all domains
(PASS) | recal MSE 3.76→0.0012 (round 0), 0.005→0.0002 (on-trajectory round 1)
| profile mushiness 0.923.

| gate | bar | value | verdict |
|---|---|---|---|
| final-logit cosine | >0.9 (stretch 0.99) | 0.99863 | **PASS (stretch)** |
| \|ΔCE\| | <0.05 | 0.0303 | **PASS** |
| per-layer MoE cos | >0.9 (D0-runbook auxiliary) | 0.8640 | FAIL |
| top-k agree | >0.8 (D0-runbook auxiliary) | 0.6509 | FAIL |
| purity vs stock | ≥ stock−0.02 every domain | ≤ −0.0005 worst | PASS |

D2 question ("does the passport interface generalize across families?"):
**YES on the user-stated bars** (cos > 0.9 incl. stretch, |ΔCE| < 0.05). The
D0-runbook per-layer auxiliary gates miss — sparsemixer's selection is
discrete in a way OLMoE's softmax-top-k is not (post-hoc correction, §7: in
eval the INDICES are pure top-2 ranking and the relative-threshold mask only
moves the multiplier softmaxes; the flips are rank-2/3 near-ties); see
FAILURES.md `d2-crossfamily-perlayer`. Wall 1310s full pipeline / 613s re-parity; peak
VRAM **16.5 GiB** (7.65B fp16 weights + bs2×512 activations). Registry
`registry_phimoe_f620b32c.pt` (8.4MB) on box
`/data/tliao/graft_d2_out/`; JSONs copied home to `lab/imported/graft_d2/`
(untracked). **Registry reuse confirmed:** a detached `setsid` `--phase parity`
re-run from the `.pt` reproduced the numbers exactly (cos 0.99861, dCE
+0.0294, layer 0.8638 / agree 0.6507). Harness post-run hardening: chunked
encode (≤8000 chars per `encode()` call — kills the donor tokenizer's
"20717 > 4096" whole-file warning) + incremental JSON writes at
precompute/parity/complete.

## 7. Follow-up: score-margin-aware recal (RAN 2026-10-11, GPU 5) — CLEAN NEGATIVE; exact-init control passes every gate

The per-layer lever named in FAILURES `d2-crossfamily-perlayer` — weight the
recal MSE on near-threshold expert pairs — was run as four detached arms
(`setsid nohup`, sequential, GPU 5 only; runner
`/data/tliao/graft_d2_out/run_margin_sweep.sh`, log `mw_sweep.log`):

```bash
# same Gate-1 command plus one of:
--recal-margin-w 2 --recal-margin-tau 0.05   # arms mw2 / mw5 / mw10
--init exact --recal-steps 0 --recal-rounds 1 # algebraic control (ceiling)
```

Weights: `w_te = 1 + margin_w · exp(−min_c |s*_te − cut_tc| / τ_t)`,
mean-normalized (loss scale preserved); cuts = rank-1/2 and rank-k/k+1
midpoints + the sparsemixer mask edge `max·(1−2·jitter)`; `τ_t = 0.05 ·
per-token score range`. `margin_analysis` (parity phase) reports the
noise-matched agreement baseline at the graft's own score-error RMS.

**Verdict (88,064 val tokens):** margin_w 0/2/5/10 at 300×2 gives MoE-cos
0.8640/0.8602/0.8571/0.8540 and top-k agree 0.6509/0.6412/0.6320/0.6233 —
monotone slightly NEGATIVE. A 600-step × 3-round pair attributes the only
positive move: uniform 600×3 (0.8858/0.7017) vs margin_w=5 600×3
(0.8822/0.6902) — the +0.022 cos / +0.051 agree gain over BEFORE is fit
budget / on-trajectory coverage, and margin weighting contributes ≤ 0 at both
budgets. Main bars hold throughout (cos ≥ 0.9985, |dCE| ≤ 0.035); the best
prototype arm still misses the aux bars (0.886 < 0.9, 0.702 < 0.8). The
exact-init control (`registry_phimoe_exact.pt`) reaches **cos 0.99996 / dCE
+0.0001 / MoE-cos 0.9966 / agree 0.9881** — every gate PASS, so the aux bars
are reachable through the identical registry format and the prototype+recal
fit is the gap (val-trajectory score RMS 0.05–0.27 vs cal-fit RMSE 0.014;
`frac(gap23 < 2σ) ≈ 0.87–0.99`). Full table + per-layer before/after:
`lab/graft_d2_margin_recal.md`, compact JSON `lab/graft_d2_margin_recal.json`
(built by `lab/graft_d2_margin_report.py` from
`lab/imported/graft_d2/phimoe_d2_{exact,mw2,mw5,mw10,mw0b600,mw5b600}.json`).

**Metric note:** for hard-selection families, report agreement normalized to
the measured exact-init ceiling `(agree − 1/C(E,k)) / (agree_exact −
1/C(E,k))` (BEFORE = 0.656 of ceiling here) — noise-matched-at-σ is the wrong
baseline (predicts 0.40 vs actual 0.64). On this donor the ceiling is 0.988,
so the absolute 0.8 bar was already ~0.81-of-ceiling: not unfairly calibrated.
