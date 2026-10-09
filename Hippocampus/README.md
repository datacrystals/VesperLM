# Hippocampus — online consolidation for VesperLM

The teach-by-talking loop: **session log → feedback triples → LoRA micro-session
→ canary gate → promote / rollback**. The model learns stylistic and factual
preferences from live conversations without ever training the trunk at
inference time — only small per-user LoRA deltas move, and only after an
external canary gate approves them.

## Files

| file | role |
|---|---|
| `session_log.py` | Append-only JSONL store (turns, tool calls, outcomes, explicit feedback). `to_triples()` extracts `(prompt, response, reward)` training triples under quarantine rules. |
| `lora.py` | Manual LoRA (PEFT is not installed in the vesper venv): freeze trunk, `A@B` side-branches on attention q/o projections (`wq`/`wo` on GQA layers, `q_proj`/`o_proj` on GLA layers), optional MoE router `gate`. Which projections get wrapped is chosen by a **target profile** (see below). Save/load small delta files; merge/unmerge for inference. |
| `consolidate.py` | The micro-session: load base + incumbent LoRA, train **only** LoRA params on the triples (few steps, tiny LR, reward-weighted NLL + KL-to-base), write a candidate delta, call the gate, promote or rollback. Per-user adapters: `adapters/<user_id>/delta.pt`. |
| `demo.py` | End-to-end toy demo: 6-turn teach session, before/after preference probes, and a poisoned batch that the stub gate must reject. |
| `README.md` | This file. |

## How to run

On the training box (remote, `/home/tliao/VesperLM/Hippocampus`):

```bash
cd /home/tliao/VesperLM/Hippocampus
PYTHONPATH=/home/tliao/.local/lib/python3.12/site-packages \
  /home/tliao/venvs/vesper/bin/python demo.py
```

The `PYTHONPATH` prefix is required (see quirks below). Demo runs on CPU by
default (~a few minutes, 28 cores); `VESPER_DEVICE=cuda` would use GPU 0 via
`CUDA_VISIBLE_DEVICES`, but only run that when the P40s have free VRAM.

## Pipeline detail

### 1. Session log (`session_log.py`)

Append-only JSONL; one record per event:

```json
{"type": "turn",       "session_id": "s1", "turn_id": 0, "prompt": "...", "response": "..."}
{"type": "tool_call",  "session_id": "s1", "turn_id": 0, "tool": "calc", "args": {...}}
{"type": "outcome",    "session_id": "s1", "turn_id": 0, "success": true, "verified": false}
{"type": "feedback",   "session_id": "s1", "turn_id": 0, "mark": "reject",
 "confidence": 1.0, "note": "answer in words", "correction": "The answer is four."}
```

**Quarantine rules** — only explicit / high-confidence feedback becomes
weight-update data; everything else stays memory-only:

- `approve` / `reject` marks with `confidence >= 0.8` → triples with reward
  `±confidence` (`source=explicit_feedback`);
- an explicit `correction` (the user's preferred response) → extra positive
  triple (`source=explicit_correction`);
- `verified` outcomes with no explicit mark → weak `±0.5` triples, only when
  `allow_outcome_derived=True` (`source=verified_outcome`);
- **quarantined** (never trained on): turns with no feedback, `neutral` marks,
  low-confidence marks, unverified outcomes, tool-call records.

### 2. LoRA (`lora.py`)

`attach_lora(model, targets=("wq","wo","q_proj","o_proj"), rank=8, alpha=16)`
wraps matching `nn.Linear` modules in `LoRALinear`: `y = W x + (α/r)·B(A x)`,
with `A ~ N(0, 1/r)`, `B = 0` so training starts as an exact no-op. Trunk
parameters are frozen (`freeze_trunk`). Delta files are just `{A, B}` tensors
plus config — 131K params / ~524 KB at rank 8 on the 118M model (16 wrapped
modules). `merge_all`/`unmerge_all` fold/unfold the delta into the base
weights for inference.

### 3. Micro-session (`consolidate.py`)

`consolidate(log, user_id="alice", ...)`:

1. `log.to_triples()` → eligible triples;
2. load base checkpoint + attach LoRA + load the user's incumbent delta;
3. train only A/B: `loss = mean_i(+reward_i · NLL(response_i|prompt_i)) +
   0.05·KL(candidate ‖ base)`. Positive reward clones the response (teach);
   negative reward is unlikelihood pressure (un-teach). Base logits come from
   the same frozen trunk with LoRA disabled — no second model copy;
4. write `candidates/<user>-<stamp>/{delta.pt, train_stats.json, triples.jsonl}`;
5. call the gate; on `PROMOTE`, copy `delta.pt` to `adapters/<user_id>/`.

### 4. Canary gate interface (external — owned by `Immune/`)

`consolidate.py` never implements a real gate. Contract:

```
gate.evaluate(candidate_dir, incumbent_dir) -> GateResult
GateResult: decision "PROMOTE"|"ROLLBACK", reason: str, metrics: dict

CLI:  <gate_cmd> --candidate DIR --incumbent DIR
      stdout last line JSON: {"decision": "...", "reason": "...", "metrics": {...}}
```

`call_gate()` resolution order: explicit `gate_fn` arg → `$VESPER_GATE_CMD`
subprocess → `import Immune.gate` → **`stub_gate()`** (trivial stand-in for
standalone testing only: rejects mean-reward < 0, target-collapsed batches
where one response string dominates, degenerate unique-response ratios, and
runaway KL drift > 10 nats). Once `Immune/` ships, point `VESPER_GATE_CMD` at
it or drop `Immune/gate.py` with `evaluate()` on `sys.path` — no Hippocampus
changes needed.

## Demo output shape

`demo.py` prints: the simulated 6-turn teach session and its extracted triples
(with quarantined turns listed separately), BEFORE generations + the
word-vs-digit first-token margin, the micro-session log, the gate decision,
AFTER generations + margins, tail-NLL probes (`" 4."` vs `" four."` after a
shared prefix — digit-tail should rise, word-tail should fall), then Part B:
the poisoned "banana" batch, its `ROLLBACK` verdict, and proof the incumbent
adapter is unchanged.

The 118M SFT model is weak; the demo proves the **mechanism** (preference
margin moves the right way, gate catches collapse), not that the model becomes
smart.

## Target profiles (Vesper-K retarget)

Which `nn.Linear` projections LoRA wraps is selected by a **target profile**,
set per call (`target_profile=...` on `consolidate` / `apply_user_adapter`) or
via the `HIPPO_TARGET_PROFILE` env var (demo and `consolidate` read it;
explicit parameter wins). Default is `gla_gqa` — byte-compatible with the
original behaviour.

| profile | stack | LoRA targets (leaf names; dotted = path suffix) | router targets (`include_router=True`) |
|---|---|---|---|
| `gla_gqa` (default) | fla GLA + GQA | `wq`, `wo`, `q_proj`, `o_proj` | `gate` (TopKRouter) |
| `kda_mla` | fla KimiDeltaAttention + MultiheadLatentAttention | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `k_rope`, `kv_proj.0`, `kv_proj.2` | `gate` + `query` (TopKRouter / PassportRouter) |

The `kda_mla` names come from the fla layer sources (`fla/layers/kda.py`:
`q_proj`/`k_proj`/`v_proj`/`o_proj` are `nn.Linear`; `fla/layers/mla.py`:
`q_proj` and `k_rope` are `nn.Linear`, `kv_proj` is
`nn.Sequential(Linear, RMSNorm, Linear)` — hence `kv_proj.0`/`kv_proj.2`).
Notes:

- `kda_mla` intentionally does **not** include `wq`/`wo`; a mixed
  `linear_type="kda"` + `full_type="gqa"` stack should pass explicit
  `targets=profile_targets("kda_mla") + ("wq", "wo")`.
- If MLA is built with `q_lora_rank` set, `q_proj` becomes a Sequential and
  the profile's `q_proj` will not match — pass `q_proj.0`/`q_proj.2`
  explicitly (the shipped VesperLinearLM leaves `q_lora_rank=None`).
- `load_base_model` now passes `full_type` / `router_type` / KDA+MLA shape
  keys through from the checkpoint's `model_config` (defaults unchanged for
  old checkpoints), and `enable_cpu_kda_mla_shims()` makes KDA/MLA layers
  runnable on CPU the same way `enable_cpu_gla_shims()` does for GLA.
- Delta files record the profile in `meta["target_profile"]`.

## Model / dtype quirks discovered

- **Triton vs fla version clash**: the vesper venv pins `triton==3.1.0` but
  `fla` (`flash-linear-attention` 0.6.0 / `fla-core` 0.3.2 in `~/.local`)
  needs `triton>=3.3` (`Autotuner(do_bench=...)`). A plain venv python
  currently **cannot import fla**. Workaround: prefix runs with
  `PYTHONPATH=/home/tliao/.local/lib/python3.12/site-packages` so the `.local`
  `triton 3.4.0` wins. (Long-lived training processes still run fine because
  they imported fla before the clash appeared.)
- **GLA is fp32-only and GPU-kernel-backed**: `GatedLinearAttn` /
  `Mamba2SSD` wrappers force `x.float()` and disable autocast internally —
  the Triton chunk kernels abort on fp16 ("Unsupported rounding mode", P40
  cc 6.1) and do not run on CPU at all. For CPU, `enable_cpu_gla_shims()`
  (same shim as `Pretrain/cpu_probe.py`) swaps `chunk_gla` /
  `fused_recurrent_gla` for `fla.ops.gla.naive.naive_recurrent_gla` and the
  fused gated-RMSNorm for a pure-torch equivalent. Micro-sessions therefore
  run **fp32 end-to-end** (recommended); don't cast the trunk to bf16/fp16
  wholesale or the GLA internals will mismatch the fp32 path.
- **Weight tying**: `tok_embeddings.weight is output.weight`. LoRA must not
  wrap `output` (it isn't a q/o projection; `attach_lora` won't match it).
- **RoPE buffer is complex**: `freqs_cis` is a complex buffer. `Module.to(dtype)`
  casts complex→real and silently discards the imaginary part (torch warns
  "Casting complex values to real..."), which breaks RoPE. `load_base_model`
  therefore moves device first and casts only floating-point tensors when a
  non-fp32 dtype is requested. (SFT inference never hits this because it
  calls `.to(device)` without a dtype.)
- **Checkpoint layout**: `checkpoint.pt` carries `model_config` but no vocab
  size; build with `vocab_size=len(tokenizer)` (65523) and
  `pad_id=tokenizer.pad_token_id` (12). State dict may carry a `module.`
  (DDP) prefix — stripped on load. `grad_checkpoint` defaults to True in the
  class; micro-sessions disable it (tiny batches).
- **GPU politeness**: the 3× P40 are busy training until ~Oct 12 with only a
  few GB free per card (not the expected ~14 GB at time of writing). Keep
  micro-sessions on CPU, or a single brief `CUDA_VISIBLE_DEVICES=0` smoke.

## Upstream expectations

- Test checkpoint: `SFT/sft_checkpoints_118m_v1/step_2900/` (dim 512, 8
  layers, 8 heads / kv 2, hidden 1536, 4 experts top-2, vocab 65523,
  `linear_type="gla"` — layers 3 and 7 are full GQA, the rest GLA).
- Sibling component: `Immune/` (real canary gate). Hippocampus only defines
  and calls the interface.
