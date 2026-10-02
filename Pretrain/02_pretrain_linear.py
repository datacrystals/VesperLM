"""
Linear-attention (GLA) hybrid pretraining for VesperLM.

Reuses the data pipeline, schedules, DDP setup, and eval from 01_pretrain.py
via import; only the model class and the training loop differ:

  - Model: VesperLinearLM (Common/vesper_linear_model.py) — mostly GLA
    layers, every 4th layer keeps full softmax attention.
  - fp32 throughout: the P40 has no bf16 and fp16 autocast gains are
    limited, so no AMP / GradScaler here.
  - Triton caches are pinned to /tmp so the ~250s one-time JIT compile
    of the FLA chunk kernels only ever happens once.
"""

import os

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("FLA_CACHE_DIR", "/tmp/fla_cache")

# Pascal (sm_61) workaround: Triton's autotuner benches every config, and on
# sm_61 some configs fail to *compile* (PassManager::run failed) instead of
# just running slowly, which kills the whole process mid-eval. Wrap the
# benchmarking hook so a config whose kernel can't compile is scored as
# infinitely slow and simply skipped by the autotuner.
import triton
import triton.runtime.autotuner as _autotuner

_orig_bench = _autotuner.Autotuner._bench

def _safe_bench(self, *args, **kwargs):
    try:
        return _orig_bench(self, *args, **kwargs)
    except Exception as e:  # noqa: BLE001 - any compile/run failure = unusable config
        print(f"[autotune] skipping config after failure: {type(e).__name__}: {e}")
        return float("inf")

_autotuner.Autotuner._bench = _safe_bench

import sys
import math
import time
import datetime
import json
import shutil

import torch

# Import the dense trainer for its shared pieces
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util as _ilu

_here = os.path.dirname(os.path.abspath(__file__))
_common = os.path.join(os.path.dirname(_here), "Common")
sys.path.insert(0, _common)

_spec = _ilu.spec_from_file_location("p01", os.path.join(_here, "01_pretrain.py"))
p01 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(p01)

from vesper_linear_model import VesperLinearLM
from muon import Muon
from configs.model_configs import get_model_config

ACTIVE_CONFIG_NAME = "tiny_agent_v2"

LINEAR_CHECKPOINT_DIR = "vesper_linear_checkpoints_v2"
MODEL_SNAPSHOT_NAME = "vesper_linear_model.py"


def train():
    local_rank = p01.setup_ddp()
    is_distributed = torch.distributed.is_initialized() if torch.distributed.is_available() else False
    world_size = torch.distributed.get_world_size() if is_distributed else 1
    device = torch.device(f"cuda:{local_rank}" if local_rank is not None else "cuda:0")
    is_main = (local_rank == 0) or (local_rank is None)

    tokenizer = p01.AutoTokenizer.from_pretrained("custom_tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    current_cfg = get_model_config(ACTIVE_CONFIG_NAME)

    batch_size = current_cfg.get("micro_batch_size", 1)
    target_acc_steps = current_cfg.get("target_accumulation_steps", 128)
    accumulation_steps = max(1, target_acc_steps // (world_size * batch_size))

    if is_main:
        print(f"\n--- BATCH SCALING ---")
        print(f"GPUs (World Size): {world_size}")
        print(f"Micro-Batch Size: {batch_size}")
        print(f"Local Accumulation Steps: {accumulation_steps}")
        print(f"Global Effective Batch: {accumulation_steps * world_size * batch_size} "
              f"(Target: {target_acc_steps})\n")

    seq_len_start = current_cfg.get("seq_len_start", 128)
    seq_len_warmup = current_cfg.get("seq_len_warmup", 4000)
    max_lr = current_cfg.get("max_lr", 2e-4)
    min_lr = current_cfg.get("min_lr", 4e-5)
    warmup_steps = current_cfg.get("warmup_steps", 1000)
    total_steps = current_cfg.get("total_steps", 30000)
    eval_interval = current_cfg.get("eval_interval", 500)
    val_eval_steps = current_cfg.get("val_eval_steps", 50)
    aux_weight = current_cfg.get("aux_weight", 0.01)
    beta1 = current_cfg.get("beta1", 0.9)
    beta2_half_life = current_cfg.get("beta2_token_half_life", 10_000_000)

    checkpoint_dir = LINEAR_CHECKPOINT_DIR
    best_val_loss = float("inf")
    start_step = 0
    train_loss_history = []
    val_loss_history = []
    total_tokens_trained = 0
    total_tokens_generated = 0
    phase1_stream_state = None
    phase2_stream_state = None
    val_stream_state = None

    latest_ckpt_path = p01.get_latest_checkpoint(checkpoint_dir)
    if latest_ckpt_path and os.path.exists(latest_ckpt_path):
        if is_main:
            print(f"\n[!] Resuming from: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location='cpu', weights_only=False)
        model_config = checkpoint.get('model_config', current_cfg)
        start_step = checkpoint['step'] + 1
        train_loss_history = checkpoint.get('train_loss_history', [])
        val_loss_history = checkpoint.get('val_loss_history', [])
        total_tokens_trained = checkpoint.get('tokens_trained', 0)
        total_tokens_generated = checkpoint.get('tokens_generated', 0)
        phase1_stream_state = checkpoint.get('phase1_stream_state', None)
        phase2_stream_state = checkpoint.get('phase2_stream_state', None)
        val_stream_state = checkpoint.get('val_stream_state', None)
        best_val_loss = checkpoint.get('best_val_loss', float("inf"))
        if is_main:
            initial_seq_len = p01.get_seq_len(start_step, seq_len_warmup,
                                              model_config["max_seq_len"], seq_len_start)
            print(f"[!] Resuming at step {start_step} | Tokens trained: {total_tokens_trained:,} "
                  f"| Context: {initial_seq_len}")
    else:
        if is_main:
            print(f"\n[!] Starting fresh: {ACTIVE_CONFIG_NAME} (GLA hybrid, fp32)")
        model_config = current_cfg
        os.makedirs(checkpoint_dir, exist_ok=True)

    phase_switch_step = int(model_config.get("total_steps", total_steps) * 0.8)

    arch_keys = ["dim", "n_layers", "n_heads", "n_kv_heads", "hidden_dim",
                 "num_experts", "top_k", "max_seq_len", "linear_type",
                 "grad_checkpoint"]
    arch_config = {k: v for k, v in model_config.items() if k in arch_keys}

    model = VesperLinearLM(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
        **arch_config
    ).to(device)

    if is_main:
        p01.print_model_stats(model, model_config)
        layer_types = model.layer_types
        n_gla = sum(1 for t in layer_types if t == 'gla')
        print(f"Hybrid stack: {n_gla} GLA layers / {len(layer_types) - n_gla} full-attn layers\n")

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=max_lr, betas=(beta1, 0.95),
                                        weight_decay=0.1)
        if is_main:
            print("Using 8-bit bitsandbytes AdamW optimizer")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(beta1, 0.95),
                                      weight_decay=0.1)
        if is_main:
            print("Using torch AdamW optimizer")

    # SOTA addition: Muon for 2D hidden weights, AdamW for the rest.
    # Muon orthogonalizes momentum updates; embeddings/head/1D params
    # stay on AdamW since Muon is designed for hidden layers only.
    muon_lr_mult = current_cfg.get("muon_lr_mult", 100.0)
    if current_cfg.get("use_muon", True):
        muon_params, adamw_params = [], []
        for name, p in model.named_parameters():
            if p.ndim == 2 and "tok_embeddings" not in name and "output" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        optimizers = {
            "muon": Muon(muon_params, lr=max_lr * muon_lr_mult),
            "adamw": torch.optim.AdamW(adamw_params, lr=max_lr,
                                       betas=(beta1, 0.95), weight_decay=0.1),
        }
        if is_main:
            n_muon = sum(p.numel() for p in muon_params)
            n_adamw = sum(p.numel() for p in adamw_params)
            print(f"Using Muon optimizer: {n_muon:,} params (Muon) + "
                  f"{n_adamw:,} params (AdamW)")
    else:
        optimizers = None

    if latest_ckpt_path and os.path.exists(latest_ckpt_path):
        model.load_state_dict(checkpoint['model'])
        if optimizers is not None:
            optimizers['muon'].load_state_dict(checkpoint['muon_state'])
            optimizers['adamw'].load_state_dict(checkpoint['adamw_state'])
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
        for opt in (optimizers.values() if optimizers else [optimizer]):
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        del checkpoint
        torch.cuda.empty_cache()

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)

    datasets_dict, _ = p01.load_dataset_index("data/index.txt")

    phase1_train = {n: d for n, d in datasets_dict['train'].items() if 'phase1' in n}
    phase2_train = {n: d for n, d in datasets_dict['train'].items() if 'phase2' in n}
    if not phase1_train or not phase2_train:
        raise ValueError("Nemotron curriculum requires both phase1 and phase2 datasets in data/index.txt")

    val_datasets = {n: d for n, d in datasets_dict['val'].items() if 'phase1' in n or 'phase2' in n}
    val_probs = {n: (0.8 if 'phase1' in n else 0.2) for n in val_datasets}

    phase1_stream = p01.MixedDataStream(
        phase1_train, {k: 1.0 for k in phase1_train}, batch_size, start_step,
        accumulation_steps, seq_len_warmup, model_config["max_seq_len"],
        seq_len_start, is_distributed, resume_state=phase1_stream_state)
    phase2_stream = p01.MixedDataStream(
        phase2_train, {k: 1.0 for k in phase2_train}, batch_size,
        max(0, start_step - phase_switch_step),
        accumulation_steps, seq_len_warmup, model_config["max_seq_len"],
        seq_len_start, is_distributed, resume_state=phase2_stream_state)
    val_stream = p01.MixedDataStream(
        val_datasets, val_probs, batch_size, 0, 1,
        0, model_config["max_seq_len"], model_config["max_seq_len"],
        is_distributed, resume_state=val_stream_state)

    # Dummy pass to pre-allocate VRAM (fp32, no autocast)
    if is_main:
        print("\n--- Running Dummy Pass to Pre-allocate Max VRAM ---")
    model.train()
    for opt in (optimizers.values() if optimizers else [optimizer]):
        opt.zero_grad()
    dummy_x = torch.randint(0, len(tokenizer), (batch_size, model_config["max_seq_len"]), device=device)
    dummy_y = torch.randint(0, len(tokenizer), (batch_size, model_config["max_seq_len"]), device=device)
    _, dummy_ce, dummy_aux = model(dummy_x, dummy_y)
    dummy_loss = (dummy_ce + aux_weight * dummy_aux) / accumulation_steps
    dummy_loss.backward()
    for opt in (optimizers.values() if optimizers else [optimizer]):
        opt.zero_grad()
    del dummy_x, dummy_y, dummy_loss, dummy_ce, dummy_aux
    torch.cuda.empty_cache()
    if is_main:
        print(f"Max VRAM Successfully Reserved: {torch.cuda.memory_allocated(device) / 1e9:.1f}GB")
        print("---------------------------------------------------\n")

    if is_distributed:
        torch.distributed.barrier()

    t0 = time.time()
    local_tokens_since_last_log = 0

    if start_step >= phase_switch_step:
        train_iter = iter(phase2_stream)
        current_phase = 2
    else:
        train_iter = iter(phase1_stream)
        current_phase = 1

    for step in range(start_step, total_steps):

        # Cooperative vLLM yield (same as 01)
        if step % 5 == 0:
            pause_signal = torch.tensor([0], device=device)
            if is_main and p01.check_vllm_api(port=9100):
                print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                      f"[!] vLLM active, pausing training to yield compute...")
                pause_signal[0] = 1
            if is_distributed:
                torch.distributed.broadcast(pause_signal, src=0)
            if pause_signal.item() == 1:
                torch.cuda.synchronize(device)
                if is_main:
                    idle = 0
                    while idle < 600:
                        time.sleep(5)
                        idle = 0 if p01.check_vllm_api(port=9100) else idle + 5
                    pause_signal[0] = 0
                if is_distributed:
                    torch.distributed.broadcast(pause_signal, src=0)
                t0 = time.time()

        if step == phase_switch_step:
            if is_main:
                print(f"\n{'='*60}\nCURRICULUM SWITCH: Phase 1 -> Phase 2 at step {step}\n{'='*60}\n")
            train_iter = iter(phase2_stream)
            current_phase = 2

        current_seq_len = p01.get_seq_len(step, seq_len_warmup,
                                          model_config["max_seq_len"], seq_len_start)
        local_tokens_this_step = batch_size * accumulation_steps * current_seq_len
        local_tokens_since_last_log += local_tokens_this_step
        total_tokens_trained += local_tokens_this_step * world_size
        lr = p01.get_lr(step, total_steps, max_lr, min_lr, warmup_steps)
        global_tokens_this_step = local_tokens_this_step * world_size
        dynamic_beta2 = 1.0 - (math.log(2) / beta2_half_life) * global_tokens_this_step
        dynamic_beta2 = max(0.0, min(0.9999, dynamic_beta2))
        for opt in (optimizers.values() if optimizers else [optimizer]):
            for param_group in opt.param_groups:
                mult = muon_lr_mult if (optimizers and opt is optimizers['muon']) else 1.0
                param_group['lr'] = lr * mult
                if 'betas' in param_group:
                    param_group['betas'] = (beta1, dynamic_beta2)

        model.train()
        for opt in (optimizers.values() if optimizers else [optimizer]):
            opt.zero_grad()
        accumulated_ce_loss = 0.0
        accumulated_aux_loss = 0.0

        for micro_step in range(accumulation_steps):
            x, y = next(train_iter)
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)

            # fp32: P40 has no bf16; fp16 autocast gains are limited
            logits, ce_loss, aux_loss = model(x, y)
            ce_loss_scaled = ce_loss / accumulation_steps
            aux_loss_scaled = aux_loss / accumulation_steps
            total_loss = ce_loss_scaled + (aux_weight * aux_loss_scaled)

            if is_distributed and micro_step < accumulation_steps - 1:
                with model.no_sync():
                    total_loss.backward()
            else:
                total_loss.backward()

            accumulated_ce_loss += ce_loss_scaled.item()
            accumulated_aux_loss += aux_loss_scaled.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if optimizers is not None:
            for opt in optimizers.values():
                opt.step()
        else:
            optimizer.step()

        train_loss_history.append((step, accumulated_ce_loss))

        if is_main and step % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            if step > start_step and dt > 0:
                local_tok_per_sec = local_tokens_since_last_log / dt
                total_tok_per_sec = local_tok_per_sec * world_size
                global_steps_per_sec = 10 / dt
                local_passes_per_sec = (10 * accumulation_steps) / dt
            else:
                local_tok_per_sec = total_tok_per_sec = 0.0
                global_steps_per_sec = local_passes_per_sec = 0.0
            t0 = t1
            local_tokens_since_last_log = 0
            mem = torch.cuda.memory_allocated(device) / 1e9
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Step {step:05d} | "
                f"{global_steps_per_sec:.2f} global steps/s | "
                f"{local_passes_per_sec:.1f} local passes/s\n"
                f"          Tok/s: {local_tok_per_sec:,.0f} (GPU) | {total_tok_per_sec:,.0f} (Total) | "
                f"Total Trained: {total_tokens_trained:,}\n"
                f"          LR: {lr:.2e} | B2: {dynamic_beta2:.4f} | Seq: {current_seq_len} | "
                f"Phase: {current_phase} | "
                f"CE Loss: {accumulated_ce_loss:.4f} | Aux: {accumulated_aux_loss:.4f} | VRAM: {mem:.1f}GB"
            )

        if step > 0 and step % eval_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(val_eval_steps):
                    vx, vy = next(val_stream)
                    vx, vy = vx.to(device), vy.to(device)
                    _, v_ce_loss, _ = model(vx, vy)
                    val_loss += v_ce_loss.item()
            val_loss /= val_eval_steps
            val_loss_history.append((step, val_loss))

            if is_main:
                print(f"\n--- Validation at Step {step} | Val Loss: {val_loss:.4f} ---")
                print("Generating Eval Samples...")
                generated_texts, eval_tokens = p01.generate_eval_samples(
                    model, tokenizer, p01.EVAL_PROMPTS, device=device,
                    temperature=0.8, top_p=0.9)
                total_tokens_generated += eval_tokens
                for pr, gen in zip(p01.EVAL_PROMPTS, generated_texts):
                    print(f"Prompt: {pr}\nOutput: {gen}\n" + "-" * 30)

                ckpt_dir = os.path.join(checkpoint_dir, f"step_{step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                try:
                    shutil.copy(os.path.join(_common, MODEL_SNAPSHOT_NAME),
                                os.path.join(ckpt_dir, "vesper_linear_model_snapshot.py"))
                    shutil.copy(__file__, os.path.join(ckpt_dir, f"{os.path.basename(__file__)}_snapshot.py"))
                except Exception as e:
                    print(f"Warning: Could not save code snapshots: {e}")

                p01.print_model_stats(model, model_config, save_path=os.path.join(ckpt_dir, "model_stats.txt"))

                ckpt = {
                    'model_config': model_config,
                    'model': model.module.state_dict() if is_distributed else model.state_dict(),
                    'muon_state': optimizers['muon'].state_dict() if optimizers else None,
                    'adamw_state': optimizers['adamw'].state_dict() if optimizers else optimizer.state_dict(),
                    'step': step,
                    'train_loss_history': train_loss_history,
                    'val_loss_history': val_loss_history,
                    'tokens_trained': total_tokens_trained,
                    'tokens_generated': total_tokens_generated,
                    'phase1_stream_state': phase1_stream.get_state(),
                    'phase2_stream_state': phase2_stream.get_state(),
                    'val_stream_state': val_stream.get_state(),
                    'best_val_loss': best_val_loss,
                }
                torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint.pt"))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_dir = os.path.join(checkpoint_dir, "step_best")
                    os.makedirs(best_dir, exist_ok=True)
                    try:
                        shutil.copy(os.path.join(_common, MODEL_SNAPSHOT_NAME),
                                    os.path.join(best_dir, "vesper_linear_model_snapshot.py"))
                        shutil.copy(__file__, os.path.join(best_dir, f"{os.path.basename(__file__)}_snapshot.py"))
                    except Exception as e:
                        print(f"Warning: Could not save code snapshots: {e}")
                    torch.save(ckpt, os.path.join(best_dir, "checkpoint.pt"))
                    print(f"\n>>> NEW BEST val loss {val_loss:.4f} at step {step} "
                          f"-> saved to {best_dir}")

                with open(os.path.join(ckpt_dir, "eval_samples.json"), "w") as f:
                    json.dump({
                        "step": step,
                        "val_loss": round(val_loss, 4),
                        "tokens_trained": total_tokens_trained,
                        "tokens_generated": total_tokens_generated,
                        "samples": generated_texts
                    }, f, indent=2)

                if len(train_loss_history) > 1:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    t_steps, t_losses = zip(*train_loss_history)
                    v_steps, v_losses = zip(*val_loss_history) if val_loss_history else ([], [])
                    plt.figure(figsize=(10, 6))
                    plt.plot(t_steps, t_losses, label="Train CE Loss", alpha=0.5)
                    if v_losses:
                        plt.plot(v_steps, v_losses, label="Val CE Loss", color='red', linewidth=2)
                    plt.xlabel("Step")
                    plt.ylabel("Cross Entropy Loss")
                    plt.title(f"{ACTIVE_CONFIG_NAME} GLA-hybrid — Training Curves (Step {step})\n"
                              f"Total Tokens: {total_tokens_trained:,}")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(ckpt_dir, "loss_curve.png"), dpi=150)
                    plt.close()

                print(f">>> Saved checkpoint to: {ckpt_dir}")
                print(f"    Total tokens trained: {total_tokens_trained:,}\n")

            t0 = time.time()
            if is_distributed:
                torch.distributed.barrier()

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()
