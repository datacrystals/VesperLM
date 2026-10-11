# D0 graft-parity — GPU runbook (prepare 2026-10-11; run when a ≥16GB GPU frees)

For the D0 gate of `MODULAR_MOE.md` §9.4 (Variant A passport graft, §9.2).
Everything except the GPU run itself is ready: donor downloaded + hashed,
harness built and self-tested, and the **exact-init arm already passed D0 on
CPU** against the real donor (see "Evidence so far").

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
~14KB total) — enough for plumbing, NOT for the gate. For the real gate build
a ≥50k-token mixed slice and point `--domains-dir` / `--val-file` at it, e.g.:

```bash
python3 - <<'EOF'
from datasets import load_dataset
import os
os.makedirs('/data/tliao/graft_d0_slice', exist_ok=True)
wt = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
web = '\n'.join(wt['text'][:8000])
open('/data/tliao/graft_d0_slice/web.txt','w').write(web)
open('/data/tliao/graft_d0_slice/val_mixed.txt','w').write('\n'.join(wt['text'][8000:12000]))
# repeat with code/math/science/chat sources into code.txt, math.txt, ...
EOF
```

Domain-labeled corpora drive the expert-profile step (§9.2 step 1) and the
purity gate; `val_mixed.txt` drives the headline parity metric. More tokens =
tighter noise on the CE bar; 100k mixed tokens is the default budget.

## 4. Resources

- **VRAM:** 13.8GB weights (fp16/bf16) + ~1-2GB activations at seq 512 ×
  batch 2 + ~0.1GB logits ⇒ **~16GB peak**. Fits a 16GB card at bs2×512; a
  24GB P40 can go bs2×1024 or bs4×512. The two passes run sequentially, so
  parity does not double activations.
- **Runtime:** the pipeline touches each token ~4-5× (profile, recal pairs,
  2× parity, 2× purity). At the default 100k-token budget expect **~15-45 min
  on one P40** (fp16 gather-bound MoE), **~5-10 min on MI300X** (bf16). CPU
  smoke (320+320 tokens) took ~130s on the laptop — linear in tokens.
- **CPU fallback:** the whole thing runs on CPU (`--device cpu --dtype bfloat16`,
  shrink `--max-tokens`) — slow but valid for a final plumbing re-check.

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

## 6. Evidence so far (2026-10-11 prep, laptop CPU)

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

## 7. After the run

1. Copy the two JSONs' headline numbers + the registry path into
   `HANDOFF_NEXT_AGENT.md` (and `lab/FAILURES.md` if any gate fails).
2. Keep `lab/imported/graft_d0/` untracked (registry + results; never commit).
3. If prototype+recal passes: D1 (plug-in onto the graft) can start — the
   registry .pt is the §9.6 artifact D1 installs into. If only exact passes:
   registry can ship exact-equivalence passports for linear-gate donors and D0
   still counts as PASS (parity proven), but the prototype recipe needs more
   recal budget before D1.
