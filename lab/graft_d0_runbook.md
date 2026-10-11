# D0 graft-parity — GPU runbook (RAN and PASSED 2026-10-11 on the MI50 box)

For the D0 gate of `MODULAR_MOE.md` §9.4 (Variant A passport graft, §9.2).
**Status: DONE — prototype+recal PASSES all five gates on OLMoE-1B-7B-0924**
(see "Evidence" below; verdict JSON in `lab/imported/graft_d0/olmoe_d0.json`,
untracked, and on the box at `/data/tliao/graft_d0_out/`). This runbook is kept
as the reproduction recipe + platform notes.

## 1. Donor model (already downloaded on the laptop)

- Model: `allenai/OLMoE-1B-7B-0924` (OLMoE-1B-7B: 6.9B total / 1.3B active,
  16 layers, 64 experts, top-8, `norm_topk_prob=false`)
- **Pinned revision: `6d84c48581ece794365f2b8e9cfb043c68ade9c5`**
- Local path (laptop): `/home/tliao/.cache/vesper_d0/OLMoE-1B-7B-0924` (13GB,
  outside the repo — NEVER commit weights). On the box / a droplet, re-fetch:

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('allenai/OLMoE-1B-7B-0924',
    revision='6d84c48581ece794365f2b8e9cfb043c68ade9c5',
    local_dir='/data/tliao/OLMoE-1B-7B-0924')"
```

Integrity (sha256, verified 2026-10-11):

```
5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e  model-00001-of-00003.safetensors
15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7  model-00002-of-00003.safetensors
a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04  model-00003-of-00003.safetensors
0e2e1e0d8d357ac7af817cff28410c3dbad398f060c517a433e4076b2aae5579  model.safetensors.index.json
```

Needs `transformers` (tested with 5.17), `huggingface_hub`, `torch`. No fla,
no Triton — this is a pure PyTorch inference + tiny-Adam fit.

## 2. Commands

Gate 0 — harness self-test (any machine, CPU, ~6s, no donor needed):

```bash
python3 lab/graft_d0_parity.py --synthetic
```

Must print `14/14 synthetic checks passed`.

Gate 1 — the D0 measurement (this is the GPU run):

```bash
python3 lab/graft_d0_parity.py \
  --model /data/tliao/OLMoE-1B-7B-0924 \
  --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
  --device cuda --dtype float16 \
  --phase all --init prototype \
  --seq-len 512 --batch-size 2 --max-tokens 100000 \
  --recal-steps 300 --recal-rounds 2 --max-cal-pairs 4096 \
  --registry lab/imported/graft_d0/registry_olmoe_6d84c485.pt \
  --out lab/imported/graft_d0/olmoe_d0.json
```

Reference arm (algebraic ceiling — expected to PASS with cosine ≈ 1; run it
too so the measurement is bracketed):

```bash
python3 lab/graft_d0_parity.py \
  --model /data/tliao/OLMoE-1B-7B-0924 --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
  --device cuda --dtype float16 --phase all --init exact --recal-steps 0 \
  --registry lab/imported/graft_d0/registry_olmoe_exact.pt \
  --out lab/imported/graft_d0/olmoe_d0_exact.json
```

Re-measure later from a saved registry (no profiling):

```bash
python3 lab/graft_d0_parity.py --model ... --device cuda --dtype float16 \
  --phase parity --registry lab/imported/graft_d0/registry_olmoe_6d84c485.pt \
  --out lab/imported/graft_d0/olmoe_d0_reparity.json
```

**P40 gotcha:** the box's Tesla P40 is sm_61 — **bf16 does not exist there**,
use `--dtype float16` (the command above already does). On MI300X / Ampere+
use `--dtype bfloat16`. fp32 (27.6GB weights) fits neither card — don't.

## 3. Eval data

`lab/graft_d0_texts/` ships tiny fixtures (5 labeled domains + `val_mixed.txt`,
~14KB total) — enough for plumbing, NOT for the gate. The gate-scale slice is
built by **`lab/graft_d0_build_slice.py`** (HF streaming sources, OLMoE-token
accounting, per-domain fallback with a loud warning):

```bash
python3 lab/graft_d0_build_slice.py \
    --tokenizer /home/tliao/.cache/vesper_d0/OLMoE-1B-7B-0924 \
    --out-dir /home/tliao/graft_d0_slice --domain-tokens 20000 --val-tokens 80000
```

Sources used 2026-10-11: code=`codeparrot/codeparrot-clean-valid`,
math=`open-web-math/open-web-math`, web=`Salesforce/wikitext` (wikitext-2-raw),
science=`fancyzhx/ag_news` (sci/tech label), chat=`OpenAssistant/oasst1` (en);
val_mixed = fresh rows from all five. Output: **102k domain + 86k val OLMoE
tokens** (the run used `--max-tokens 100000` → 97,280 domain + 86,016 val).
Note: legacy un-namespaced dataset ids (`wikitext`, `ag_news`) fail on current
`huggingface_hub` — use the namespaced ones above. The `datasets`/pyarrow
teardown can abort the interpreter at exit after files are written (script
exits via `os._exit` for that reason); verify token counts if in doubt.
Domain-labeled corpora drive the expert-profile step (§9.2 step 1) and the
purity gate; `val_mixed.txt` drives the headline parity metric.

## 4. Resources (measured 2026-10-11 on the MI50 box)

- **VRAM (measured):** peak **15.2 GiB** on the MI50's 32GB GPU 0 (fp16
  weights ~13.8GB + activations at seq 512 × batch 2 + logits; the two passes
  run sequentially so parity does not double activations). Fits a 16GB card
  only at bs2×512 with nothing else resident; 24GB P40 has headroom for
  bs2×1024 or bs4×512.
- **Runtime (measured):** **615s wall (10.3 min)** for the full `--phase all`
  pipeline at 97,280 domain + 86,016 val tokens on one MI50 GPU 0 (incl.
  model load 10s and the CPU-side recal fits). Linear in tokens.
- **CPU fallback:** the whole thing runs on CPU (`--device cpu --dtype bfloat16`,
  shrink `--max-tokens`) — slow but valid for a final plumbing re-check.

## 4a. MI50 / gfx906 platform notes (the box that actually ran it)

- Host `tliao@100.64.0.7` (Tailscale), **GPU 0 ONLY** (`HIP_VISIBLE_DEVICES=0
  CUDA_VISIBLE_DEVICES=0`); GPUs 1-2 are the user's hackathon jobs, 3-7 free.
- Env: `/data/tliao/vesper_sft/venv/bin/python` (torch 2.9.1+rocm6.4,
  transformers 5.19) + **`ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library`**
  (mandatory — the wheel Tensile lib has no gfx906 kernels).
- **`--dtype float16`** — gfx906 has no bf16, and fp32 weights (27.6GB) do not
  fit. Scratch under `/data` (never the 24G root lv): weights
  `/data/tliao/OLMoE-1B-7B-0924`, slice `/data/tliao/graft_d0_slice`, outputs
  `/data/tliao/graft_d0_out/`.
- Two gfx906 kernel gaps hit this run and are now **auto-shimmed by the
  harness on any HIP host** (no-op on CUDA; details in `lab/FAILURES.md`):
  (1) `torch._grouped_mm` unsupported on ROCm but transformers enables it →
  forced their per-expert `torch.mm` fallback; (2) `torch.sort` device kernel
  broken at S=8192 (async-misattributed — debug with `AMD_SERIALIZE_KERNEL=3`)
  → experts merge sorts on CPU, identical math.
- Measured equivalent: GPU exact-arm mini (2×512 tokens) cos 0.9998 matches
  the laptop CPU exact arm (0.9999), so the shims cost nothing in fidelity.

## 5. What "parity PASS" means numerically

§9.4 states the bar qualitatively ("val parity with the donor's stock router
on a mixed eval (within noise), plus per-domain purity >= the donor's own
routing entropy baseline") — numbers are set here. The harness computes and
prints every gate and an overall PASS/FAIL (`verdict` in the output JSON):

| metric | bar | why |
|---|---|---|
| mean final-logit cosine (stock vs graft) | **> 0.9** (stretch > 0.99) | task-level floor: below 0.9 the next-token distribution has drifted enough that generations visibly change. 0.99 is the stretch the exact arm already proved on CPU (0.9999) — the prototype+recal registry recipe should approach it; D0 passes at 0.9 to leave room for prototype-init error. |
| abs CE delta on mixed val | **< 0.05 nats** | "within noise": typical val-CE noise on a 50-100k-token slice is a few hundredths of a nat. |
| per-layer MoE-output cosine (mean) | **> 0.9** | §9 asks "per-layer if feasible": the expert-merge deltas must agree, not just the final logits. |
| per-layer top-k set agreement (mean) | **> 0.8** | exact-set match rate; >0.8 keeps the graft's expert selection recognizably the donor's. |
| per-domain purity | purity_graft(d) ≥ purity_stock(d) − 0.02 for every domain | §9.4's second clause, operationalized: purity = 1 − H(usage)/log(E) on each domain's expert-usage distribution; the graft must not be less domain-selective than the donor's own router (its routing-entropy baseline is reported alongside). |

D0 PASS = all five. The exact arm is a ceiling check (should hit the stretch
bar); the prototype arm with recal is the actual registry recipe — its number
is the headline result.

## 6. Evidence

**Gate run (2026-10-11, MI50 box GPU 0, fp16) — D0 PASS, all five gates:**

| metric | value | bar | verdict |
|---|---|---|---|
| logit cosine mean (86,016 val tok) | **0.99913** (std 0.0025, min 0.7831) | >0.9 / stretch >0.99 | PASS incl. stretch |
| CE stock → graft (delta) | 2.0082 → 2.0089 (**+0.00075**) | |ΔCE| < 0.05 | PASS |
| per-layer MoE-output cosine | **0.9952 mean** (min 0.9932, max 0.9966) | >0.9 | PASS |
| per-layer top-k set agreement | **0.8366 mean** (min 0.7812; Jaccard 0.963) | >0.8 | PASS |
| per-domain purity margins | ~0.0 (JS ~0) all 5 domains | ≥ stock − 0.02 | PASS (weakly — see below) |

Recal: round 0 (teacher-forced) layer-0 MSE 0.7275 → 0.0005 over 300 steps;
round 1 (on-trajectory) 0.0005 → ~0. Profile mushiness 0.92 — OLMoE experts
are near-domain-generalist (routing entropy 4.15/4.16 nats), so the purity
gate is satisfied but not discriminating on this donor; the load-bearing
parity evidence is the cosine/CE/top-k rows. CPU exact-arm reference (prep):
cos 0.9999 / ΔCE +0.0013 — statistically indistinguishable from the GPU
prototype+recal result, and the GPU exact-arm mini gave 0.9998.

Prep-phase evidence (same day, laptop):

- `--synthetic`: **14/14 checks PASS** (mock MoE with OLMoE-identical gate /
  experts contract): exact-init scores = stock logits (max err 5e-6) and full
  parity cos 1.000000; profile recovers mock expert→domain homes 9/9;
  prototype init cos 0.9544 / agree 0.708 → **recal → cos 0.9985 / agree
  0.977**; registry round-trip bit-near-exact; metric sanity; install/uninstall
  of the graft gate; purity exact-graft margins 0.0.
- Real donor CPU smoke (`--init exact`, 320+320 tokens, bf16): **D0 PASS** —
  logit cosine **0.9999** (min 0.9931), CE stock 2.4025 vs graft 2.4038
  (**Δ +0.0013**), all gates green. The graft path is correct against real
  `OlmoeTopKRouter` / 3D expert tensors / top-8 no-renorm semantics.
- Real donor CPU smoke (`--init prototype`, 320 tokens, 10 recal steps):
  FAIL as expected at smoke scale (cos 0.80) — but recal MSE drops ~5× per
  layer (0.96→0.19) and expert-profile mushiness is 0.72 (consistent with
  §9.2's "donor experts are not cleanly semantic"). Not a gate result; the
  real prototype arm is exactly what the GPU run measures.

## 7. Status and next

1. ~~Run the gate~~ **DONE 2026-10-11, PASS.** Numbers in HANDOFF ("D0 GATE
   RUN: PASS") and `lab/imported/graft_d0/olmoe_d0.json` (untracked; also on
   the box at `/data/tliao/graft_d0_out/` alongside the registry .pt).
2. `lab/imported/graft_d0/` stays untracked (registry + results; never commit).
3. **D1 (plug-in onto the graft) is unblocked** — `registry_olmoe_6d84c485.pt`
   is the §9.6 artifact D1 installs into. D1's router recal is router-only and
   can run on the MI50 box too; training VesperLM experts still belongs on the
   MI300X lane (fla/Triton gfx906 wall — `lab/FAILURES.md`). P40s are not
   needed for anything D0-related.
