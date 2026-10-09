# VesperLM Immune System

Standing integrity layer for continuous online learning. The model trunk
stays frozen; fast learning lives in LoRA adapters and memory that can be
rolled back. The immune system scores everything and **its vetoes are
absolute** — nothing in this directory may loosen a gate.

All of this runs **CPU-only**. The training GPUs on this box are off-limits.

## Files

- `cpu_backend.py` — shared CPU loader. Factors the fla/Triton CPU shims and
  checkpoint loading out of `Pretrain/cpu_probe.py` (that file is untouched
  and is never imported, because it runs generation at import time). Also
  defines the ChatML special-token constants and LoRA-delta merging.
- `probes.py` — probe schema, the five scorers, YAML/JSON loader, runner.
- `probes/starter_probes.yaml` — 20 starter probes across 6 categories,
  5 of them `held_out: true` (never train on them) and 11 `protected`.
- `score.py` — score a checkpoint (or model + LoRA-delta) against a probe
  set; writes a JSON report.
- `drift.py` — cheap candidate-vs-incumbent drift metric on a frozen
  32-prompt battery (top-k logprobs), one bounded scalar + thresholds.
- `gate.py` — the canary gate: promote/rollback with dry-run and JSONL logs.
- `make_corrupt_ckpt.py` — deliberate-corruption helper used by the
  acceptance test (and any future drill).

## Quick start (CPU only)

```bash
cd /home/tliao/VesperLM/Immune
PY=/home/tliao/venvs/vesper/bin/python

# 1. score a checkpoint
$PY score.py --ckpt /home/tliao/VesperLM/SFT/sft_checkpoints_118m_v1/step_2900 \
    --probes probes/starter_probes.yaml --out state/report_baseline.json --verbose

# 2. score a LoRA candidate on top of the same base
$PY score.py --ckpt /home/tliao/VesperLM/SFT/sft_checkpoints_118m_v1/step_2900 \
    --lora path/to/lora_delta.pt --probes probes/starter_probes.yaml \
    --out state/report_candidate.json

# 3. drift of a candidate vs the incumbent
$PY drift.py --incumbent /path/incumbent_ckpt --candidate /path/candidate_ckpt \
    --out state/drift.json

# 4. gate decision (dry-run first; drop --dry-run to act)
$PY gate.py --incumbent /path/incumbent_ckpt --candidate /path/candidate_ckpt \
    --probes probes/starter_probes.yaml --dry-run
```

`gate.py` also accepts `--incumbent-report` / `--candidate-report` /
`--drift-report` to re-decide on precomputed artifacts without re-running
the models. Exit code 0 = promote (or would promote, in dry-run), 2 = reject.

## Probes and scorers

A probe is `{name, prompt, scorer, category, held_out?, protected?}`.
Every score is a float in `[0, 1]`; determinism is guaranteed (greedy
decoding, fixed stop tokens).

| scorer | params | score |
|---|---|---|
| `exact-match` | `expected` (str or list), `match`: exact/prefix/contains, `case_sensitive` | 1 or 0 |
| `regex` | `pattern`, `case_sensitive`, `full_match` | 1 or 0 |
| `json-valid` | `from_tool_call` (parse the `<|tool_call|>` JSON body first), `require_object` | 1 or 0 |
| `tool-call-schema` | `tool`, `args_schema` (required keys/types), `args_match` (regex per arg) | partial credit: 0.4 parseable tool-call JSON + 0.3 tool name + 0.3 args schema/match |
| `semantic-similarity` | `reference` (str or list), `threshold` (display only) | tokenizer-free cosine of bag-of-words overlap vs the best reference, continuous in `[0,1]` |

Starter categories: arithmetic-tool-use, file-ops-tool-use, factual-recall,
instruction-following, refusal-of-nonsense, held-out.

## Gate policy (`gate.py`)

Promotion requires **all** of:

1. **No protected probe regresses** — for every protected probe
   (`held_out: true` implies protected; `protected: true` adds more), the
   candidate score must be >= incumbent score - `PROTECTED_EPSILON`
   (1e-6, a constant for float noise only). Any drop is an absolute veto.
2. **Aggregate within margin** — candidate aggregate must not drop more
   than `MARGIN` (0.02, constant) below the incumbent aggregate.
3. **Drift not failed** — the drift label must not be `fail`. Drift data is
   mandatory: the gate either computes it or requires `--drift-report`.
   There is deliberately no flag to skip it.

On rejection the gate **auto-rolls-back**: if `state/live.json` already
points at the candidate (a bad merge got installed), the pointer is
restored to the incumbent; with `--restore-files` the incumbent's
`checkpoint.pt` is also copied over an in-place-installed live directory.

Every decision prints and logs one line with the verdict and every reason
(`logs/gate.log`, JSONL; full run report in `logs/gate_<ts>.json`).

`--dry-run` computes and logs the decision but changes no state.

## Drift metric (`drift.py`)

Frozen 32-prompt battery (16 plain + 16 ChatML). For each prompt both
models run a full forward pass; at the last `max_positions=4` positions we
take top-k (k=32) logprobs and compute a top-k-truncated
`KL(incumbent || candidate)` with the outside-top-k mass pooled into one
residual bucket. Headline number: mean per-prompt KL clipped to
`[0, DRIFT_CLIP=1.0]` — a single bounded scalar.

Thresholds (mean-KL units): `< 0.05` ok (same weights / clean merge),
`< 0.20` warn (monitor), `>= 0.20` fail (corruption / bad merge signal).
Also reported: top-1 agreement and max logprob shift. Do not edit the
battery — drift numbers are only comparable while it is frozen.

## LoRA / delta format

`score.py --lora` and `gate.py --lora-candidate` accept a `.pt` with either:

```python
{"deltas": {"layers.0.ffn.experts.0.w1.weight": additive_tensor}}
# or low-rank factors (merged as W += scale * B @ A):
{"lora": {"output.weight": {"A": A, "B": B, "scale": 0.5}}}
```

Names may omit a leading `module.`. Unknown keys raise — a silent no-op
merge is impossible.

## Target profiles (Vesper-K retarget)

The gate entry points take `--target-profile` (`gla_gqa` default, or
`kda_mla` for the Vesper-K KDA/MLA stack). Resolution order: CLI flag →
`$IMMUNE_TARGET_PROFILE` → `$HIPPO_TARGET_PROFILE` → `gla_gqa`.

- The profile supplies **defaults** for architecture keys a checkpoint's
  `model_config` omits (`linear_type`, `full_type`, router choice, KDA/MLA
  shape keys). Values present in `model_config` always win — a Vesper-K
  checkpoint loads correctly even under the default profile.
- `cpu_backend.py` no longer imports fla (or `vesper_linear_model`) at module
  load — importing fla dies on hosts without a working CUDA driver. The
  pure-torch CPU shims are installed lazily from `load_model()`: GLA shims as
  before, plus KDA (`chunk_kda`/`fused_recurrent_kda` → `fla.ops.kda.naive`
  with fla's own torch gate references) and MLA (flash-attn entry points →
  SDPA) shims for `kda_mla` stacks.
- `generate()` falls back to full-forward greedy decode on stacks without an
  incremental cache (MLA), so probes/canaries run unchanged; drift feature
  extraction already used full forwards and is architecture-agnostic.
- Expert corruption paths (`layers.N.ffn.experts.M.w1/w2/w3`) are shared by
  both stacks (the MoE FFN is common), so `make_corrupt_ckpt.py` needs no
  profile.

LoRA target names per profile match `Hippocampus/lora.py` (`gla_gqa`:
`wq`/`wo`/`q_proj`/`o_proj`; `kda_mla`: `q_proj`/`k_proj`/`v_proj`/`o_proj`/
`k_rope`/`kv_proj.0`/`kv_proj.2`, routers `gate`/`query`) and are exposed as
`cpu_backend.TARGET_PROFILES`.

## Invariants (immune system)

- **I1** The trunk is frozen; only LoRA adapters / memory change at speed.
  Every candidate must be expressible as base checkpoint (+ optional delta).
- **I2** Vetoes are absolute. A protected-probe regression rejects the
  candidate regardless of aggregate gains. No CLI flag, config, or edit
  may loosen a gate at call time; `PROTECTED_EPSILON`, `MARGIN`, and the
  drift thresholds are constants in the gate code, changed only by a
  deliberate versioned edit of this directory.
- **I3** Held-out probes are never training data. `held_out: true` probes
  (5 in the starter set) are canaries for overfitting-to-train; any
  consolidation or memory write that trains on them is itself a violation.
- **I4** Drift data is mandatory for every gate decision. There is no
  skip path.
- **I5** Every decision is logged with a one-line reason
  (`logs/gate.log` JSONL) and a full run report; `--dry-run` logs the
  decision but changes nothing.
- **I6** Rejection implies rollback: if the candidate is already live,
  the live pointer is restored to the incumbent (and optionally its
  `checkpoint.pt` with `--restore-files`).
- **I7** Everything runs on CPU while training occupies the GPUs. The
  backend never moves a tensor to CUDA. (fla/Triton must *see* the CUDA
  driver at import — same as `cpu_probe.py` — but importing it allocates
  no GPU context.)
- **I8** Gate inputs are comparable only when the probe-set file hash
  matches (recorded as `probe_set_sha` in every report) and generation
  is greedy with fixed max_new/stop tokens.

## Acceptance test (corruption drill)

```bash
# zero 50% of expert 0's weights in every layer of a checkpoint copy
$PY make_corrupt_ckpt.py --src /home/tliao/VesperLM/SFT/sft_checkpoints_118m_v1/step_2900 \
    --dst state/corrupt_ckpt --expert 0 --fraction 0.5 --seed 0 --all-layers --slim

$PY score.py --ckpt state/corrupt_ckpt --probes probes/starter_probes.yaml \
    --out state/report_corrupt.json

# baseline vs baseline must PROMOTE:
$PY gate.py --incumbent <baseline> --candidate <baseline> \
    --probes probes/starter_probes.yaml \
    --incumbent-report state/report_baseline.json \
    --candidate-report state/report_baseline.json

# baseline vs corrupt must REJECT and roll back:
$PY gate.py --incumbent <baseline> --candidate state/corrupt_ckpt \
    --probes probes/starter_probes.yaml \
    --incumbent-report state/report_baseline.json \
    --candidate-report state/report_corrupt.json
```

See `logs/gate.log` for the recorded decisions of the run on this box.

## Caveats

- Semantic-similarity is token-overlap cosine, not an embedding model —
  it is a cheap proxy by design (no external models on this box).
- The 118M SFT model fails several starter probes at baseline (refusal
  and plain-arithmetic categories); those scores are still useful as
  regression signals (they can only improve).
- Drift thresholds (0.05/0.20) are calibrated for same-family merges of
  this model size; re-validate before applying to other sizes.
- `gate.py` treats drift `warn` as promotable (only `fail` vetoes); the
  protected-probe and aggregate checks remain the primary gates.
