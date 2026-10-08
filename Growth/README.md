# Growth — Expansion A: sparse upcycling (expert add)

Surgery tooling that grows a trained VesperLM MoE by cloning experts (+ small
gaussian noise) and extending the router, so the model gains capacity in
~1 day of continued training instead of a fresh 6-day pretrain
(`LMBUS_DESIGN.md` → "Expansion A — expert add").

| file | what it does |
|---|---|
| `expand_experts.py` | checkpoint surgery: clone/split experts, extend router rows, bump `num_experts`, reset optimizer state, record provenance |
| `validate.py` | CPU proof that surgery is correct: strict load, exact function preservation, noise behavior, greedy generation |
| `README.md` | this file |

Works on the checkpoint format used by SFT/pretrain:

```
{'model_config': {...}, 'model': state_dict (optionally 'module.'-prefixed),
 'optimizer' | 'muon_state'/'adamw_state', 'scaler', 'step', ...}
```

MoE layout inside the state dict (`Common/vesper_model.py`):

```
layers.{L}.ffn.experts.{E}.w1.weight   (hidden_dim, dim)
layers.{L}.ffn.experts.{E}.w2.weight   (dim, hidden_dim)
layers.{L}.ffn.experts.{E}.w3.weight   (hidden_dim, dim)
layers.{L}.ffn.router.gate.weight      (num_experts, dim)   # bias-free Linear
```

Router math: `softmax(x @ gate.T)` → `top_k` → selected weights renormalized to
sum to 1. Because of that renormalization only the *selected set* and the
*relative logits inside it* matter for the FFN output.

## Usage

### Exact-upcycle (function-preserving surgery check)

```bash
cd /home/tliao/VesperLM/Growth
PY=/home/tliao/venvs/vesper/bin/python

$PY expand_experts.py \
    --input  ../SFT/sft_checkpoints_118m_v1/step_2900 \
    --output runs/118m_4to8_exact \
    --num-experts 8 --mode clone --noise-std 0 --exact-upcycle
```

New experts are bit-exact copies of the existing ones; each new router row is a
bit-exact copy of its parent's row; `top_k` is doubled (2 → 4) so every selected
parent enters together with its clone and the renormalized routing weight splits
**exactly 50/50** between parent and clone. The MoE FFN function is unchanged,
so the expanded model reproduces the original logits.

Why not literally "halve the parent's gate row and give the other half to the
clone"? Under `softmax → topk → renorm` that is *not* function-preserving:
halving a logit distorts the softmax nonlinearly, and near-duplicate rows make
`top_k` select both slots from the single best parent (crowding out the second
original expert). Splitting the parent's *routing weight* 50/50 — which is what
the FFN actually consumes — is achieved by duplicate rows + doubled `top_k`.
`validate.py` (b) measures the split and the logits directly.

Note: doubling `top_k` doubles the active experts per token (FLOPs ×2 for the
FFN). Use this mode to verify surgery and as a zero-loss warm start; for
production growth use the default mode below and take the small initial loss
bump.

### Standard sparse upcycling (production growth)

```bash
$PY expand_experts.py \
    --input  ../SFT/sft_checkpoints_118m_v1/step_2900 \
    --output runs/118m_4to8_noise \
    --num-experts 8 --mode clone --noise-std 1e-3
```

New experts = round-robin copies + gaussian noise (`--noise-std`, default
`1e-3`); new router rows = copies of the parents' rows + noise, so routing to
new experts starts near their parents'. `top_k` is unchanged (2 of 8 active).

Other options:

* `--mode clone_split` — new experts are `(expert_i + expert_j)/2` weight-space
  mixtures for diversity (+ noise); router rows are the mean of the pair's rows.
* `--usage-stats usage.json` — if you have per-expert usage counts, clones
  round-robin over the *most-used* experts first; without it, round-robin over
  all. JSON shapes accepted: `[c0, c1, ...]`, `{"0": c0, ...}`, or per-block
  `{"layers.0.ffn": {...}, ...}`.
* `--seed` — noise seed (recorded in provenance). `--overwrite` — replace output.
* `--shared-expert` — always refused (see gotchas).

Output: `<output>/checkpoint.pt` (drop-in checkpoint) and
`<output>/expansion_report.json` (human-readable summary). Provenance lives in
the checkpoint under `expansion_provenance` (parent mapping per layer, noise,
seed, mode, top_k change, optimizer keys nulled, source path).

### Validation

Build one more checkpoint first — the exact construction with noise, which is
what isolates the noise variable for check (c):

```bash
$PY expand_experts.py \
    --input  ../SFT/sft_checkpoints_118m_v1/step_2900 \
    --output runs/118m_4to8_exact_noisy \
    --num-experts 8 --mode clone --noise-std 1e-3 --exact-upcycle

$PY validate.py \
    --parent ../SFT/sft_checkpoints_118m_v1/step_2900/checkpoint.pt \
    --exact  runs/118m_4to8_exact/checkpoint.pt \
    --noisy  runs/118m_4to8_exact_noisy/checkpoint.pt \
    --upcycled runs/118m_4to8_noise/checkpoint.pt
```

Runs entirely on CPU (~4 min) and prints PASS/FAIL for:

* **(a)** `VesperLinearLM(**expanded_config)` + `load_state_dict(strict=True)` on
  each expanded checkpoint;
* **(b)** exact-upcycle + noise=0: parent gate rows bit-exact, measured router
  50/50 split (`|w_clone − w_parent/2|`), per-layer MoE FFN identity (relative
  to output scale), and full-model logits vs the original within `--tol`
  (default `1e-3`, observed `max|Δ| ≈ 1e-5`, top-1 100%);
* **(c)** same construction with noise>0 (`--noise-std 1e-3`): logits close but
  not identical (observed `max|Δ| ≈ 1.1e-2`, cosine ≈ 1.0, top-1 100%); defaults
  `--close-tol 1.0`, `--close-cosine 0.999`;
* **(d)** short greedy generation (`--max-new`, default 24) via
  `forward_incremental`, from `--upcycled` (production mode) with the parent's
  and the exact+noise generations for reference.

With `--upcycled` it also prints an informational drift report for the
production sparse-upcycling checkpoint (expect a bigger `max|Δ|` there than in
(c): top-k crowd-out until the routing warmup rebalances — see gotchas).

Exit code 0 iff all checks pass. `validate.py` reuses the CPU shims from
`Pretrain/cpu_probe.py` (copied verbatim; that file is not modified): GLA's
chunk/recurrent Triton kernels and the fused gated RMSNorm are replaced with
pure-torch equivalents.

## Training recipe after surgery

1. **Optimizer is fresh** (see gotchas) → start with a short LR warmup
   (e.g. 200–400 steps to peak) rather than jumping straight to peak LR.
2. **Forced-balance routing warmup.** With duplicated router rows the top-k
   selection concentrates on the best expert + its clone. For the first ~5% of
   the warmup, raise the Switch aux-loss weight (`aux_weight` in
   `SFT_CONFIGS` / pretrain config, e.g. 0.01 → 0.05–0.1) and/or train with a
   temporarily higher `num_experts` utilization target so every new expert gets
   traffic, then decay `aux_weight` back to its pre-surgery value.
3. **Budget:** ~10–20% of the original token budget is enough
   (e.g. 3000-step SFT → 300–600 steps; 4.19B-token pretrain → 0.4–0.8B tokens).
   Track per-expert usage and stop once the new experts carry their share.
4. **Rehearsal mix:** mix in 30–50% of the original pretrain/SFT data
   distribution alongside the continued-training mix — pure new-distribution
   data on a fresh optimizer tends to forget; the clones start as copies so the
   model's behavior is already "there", the warmup just needs to specialize
   without erasing it.
5. Keep `top_k` at its post-surgery value for the whole warmup; changing it
   mid-run changes the function again.

## Gotchas

* **Optimizer state resets (by design).** `optimizer` / `muon_state` /
  `adamw_state` / `scaler` are set to `None` in the expanded checkpoint. Those
  tensors are per-parameter AdamW moments / Muon momentum buffers with shapes
  pinned to the old tensors; the router goes `(4,512) → (8,512)` and the expert
  count changes, so the state cannot be carried across. The trainer rebuilds the
  optimizer on resume — expect slightly noisier steps at the start (no
  momentum/variance history for *any* parameter, not just the new ones).
* **`step` is preserved** (provenance of the parent run). If your trainer
  resumes at `step+1` against a fixed `total_steps`, raise `total_steps` by the
  warmup budget (or point it at a fresh run directory).
* **Aux load-balance needs a retune.** The Switch aux loss
  (`E * Σ mean_prob * mean_usage`) changes scale with `E`, and after cloning the
  router is deliberately near-degenerate (pairs of near-identical rows). Re-tune
  `aux_weight` (see recipe) and watch per-expert usage; a too-weak aux term at
  this stage means the clones stay dead weight.
* **No shared expert.** `MoEFeedForward` (`Common/vesper_model.py`) only has
  routed experts (`ModuleList` + `TopKRouter`); its forward has no always-on
  dense path and the router has **no bias** (so you cannot even create a
  constant logit margin for a new expert). `--shared-expert` is therefore
  refused: adding one means changing the module (new parameter + forward),
  which would break strict state-dict compatibility with `VesperLinearLM` and
  every existing checkpoint. When the K2 "+1 shared expert" design lands it
  needs a model-class change first.
* **`top_k` crowd-out with duplicated rows.** If you extend the router with
  exact/near-exact row copies and leave `top_k` at 2, `topk` takes *both* slots
  from the single best expert's pair and the second original expert drops out —
  the function changes a lot, not a little. Measured on the 118M 4→8 run with
  `noise-std 1e-3`: `max|Δ logits| ≈ 3.6`, top-1 agreement 74% (vs
  `1.1e-2` / 100% for the exact construction at the same noise), and the
  per-layer MoE outputs move by ~50% of their scale. That is fine for the noisy
  production mode (warmup rebalances), and exactly why `--exact-upcycle`
  doubles `top_k` instead.
* **`vocab_size` is backfilled** into `model_config` from the embedding table
  (65523 here) so `VesperLinearLM(**expanded_config)` reconstructs the model;
  the SFT trainer ignores it and rebuilds vocab from the tokenizer (same size).
  `pad_id` is *not* stored in these checkpoints — the trainer takes it from the
  tokenizer (`12` for `Pretrain/custom_tokenizer`).
* **Weight tying** (`tok_embeddings.weight` is `output.weight`) is untouched;
  both keys stay in the state dict as before.
* **Environment gotcha (observed 2026-10-08):** `triton==3.1.0` was installed
  into `/home/tliao/venvs/vesper` and now shadows the user-site
  `triton==3.4.0`. `import fla` (user-site `flash_linear_attention==0.6.0`)
  fails in fresh processes with
  `Autotuner.__init__() got an unexpected keyword argument 'do_bench'`, because
  fla needs triton ≥ 3.3. The already-running training processes are unaffected
  (they imported the user-site triton at startup). `validate.py` prefers the
  user-site triton per-process and leaves the venv alone; any *new* training
  launch will hit this until the venv triton is upgraded or removed.
* SFT checkpoint write time is CPU-side; both tools run CPU-only and never
  touch CUDA or the running jobs.
