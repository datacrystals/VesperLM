import os
import math
import json
import sys
import shutil
import warnings
import time
import datetime
from collections import defaultdict

# Silence AMD / PyTorch warnings
os.environ["HIPBLASLT_DISABLE"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*hipBLASLt.*")

# Detect ROCm vs CUDA
IS_ROCM = False
try:
    import torch
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        IS_ROCM = True
        print(f"*** ROCm {torch.version.hip} detected ***")
    elif torch.cuda.is_available():
        print(f"*** CUDA {torch.version.cuda} detected ***")
except ImportError:
    pass

try:
    import bitsandbytes as bnb
    HAS_BNB = True
    print("Detected bitsandbytes 8-bit optimizer is installed")
except ImportError:
    print("Did not detect bitsandbytes 8-bit optimizer installed, skipping and falling back to 32-bit optimizer")
    HAS_BNB = False

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler
from transformers import AutoTokenizer
from jesper_model import JesperLLM

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==========================================
# MODEL CONFIGURATIONS
# ==========================================
MODEL_CONFIGS = {
    "small_v2": {
        # Architecture
        "dim": 1024, "n_layers": 10, "n_heads": 8, "n_kv_heads": 2,
        "hidden_dim": 1280, "num_experts": 8, "top_k": 2, "max_seq_len": 2048,
        
        # Training & Batching
        "micro_batch_size": 1,
        "target_accumulation_steps": 128,
        
        # Optimizer Dynamics
        "beta1": 0.9,
        "beta2_token_half_life": 10_000_000,
        "max_lr": 2e-4,
        "min_lr": 3e-5,
        "aux_weight": 0.1,
        
        # Scheduling
        "seq_len_start": 128,
        "seq_len_warmup": 4000,
        "warmup_steps": 1000,
        "total_steps": 20000,
        
        # Evaluation
        "eval_interval": 100,
        "val_eval_steps": 50
    },
    "1b_scaled": {
        # Architecture
        "dim": 1152,
        "n_layers": 12,
        "n_heads": 12,
        "n_kv_heads": 2,
        "hidden_dim": 1600,
        "num_experts": 10,
        "top_k": 2,
        "max_seq_len": 2048,

        # Training & Batching
        "micro_batch_size": 1,
        "target_accumulation_steps": 128,

        # Optimizer Dynamics
        "beta1": 0.9,
        "beta2_token_half_life": 12_000_000,
        "max_lr": 1.3e-4,
        "min_lr": 1.5e-5,
        "aux_weight": 0.1,

        # Scheduling
        "seq_len_start": 128,
        "seq_len_warmup": 5000,
        "warmup_steps": 1500,
        "total_steps": 25000,

        # Evaluation
        "eval_interval": 100,
        "val_eval_steps": 75    },
    "big": {
        "dim": 1268, "n_layers": 16, "n_heads": 12, "n_kv_heads": 4,
        "hidden_dim": 1536, "num_experts": 10, "top_k": 2, "max_seq_len": 2048,
        
        "micro_batch_size": 1,
        "target_accumulation_steps": 128,
        
        "beta1": 0.9,
        "beta2_token_half_life": 10_000_000, 
        "max_lr": 1.5e-4,
        "min_lr": 1.5e-5,
        "aux_weight": 0.01,
        
        "seq_len_start": 128,
        "seq_len_warmup": 4000,
        "warmup_steps": 1500,
        "total_steps": 50000,
        
        "eval_interval": 1000,
        "val_eval_steps": 50
    },
}

ACTIVE_CONFIG_NAME = "small_v2"

EVAL_PROMPTS = [
    "The most important rule of debugging is",
    "def calculate_pid(error, integral, derivative):",
    "As Diluc watched the rain drip down the window,",
    "As Neil walked down the hallway to his Gundam, "
]


def setup_ddp():
    if not dist.is_available():
        return None
    try:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    except (KeyError, ValueError):
        return None
    if world_size <= 1:
        return None

    if IS_ROCM:
        try:
            dist.init_process_group(backend="nccl")
        except Exception:
            print("WARNING: NCCL failed, falling back to GLOO.")
            dist.init_process_group(backend="gloo")
    else:
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    return local_rank


def get_seq_len(step, warmup_steps=4000, max_seq_len=1024, start_len=128):
    if step >= warmup_steps:
        return max_seq_len
    progress = step / warmup_steps
    length = int(start_len + (max_seq_len - start_len) * progress)
    return max(start_len, (length // 64) * 64)


def print_model_stats(model, config, save_path=None):
    total_params = sum(p.numel() for p in model.parameters())

    moe_params = 0
    base_model = model.module if hasattr(model, 'module') else model
    for layer in base_model.layers:
        if hasattr(layer['ffn'], 'experts'):
            moe_params += sum(p.numel() for p in layer['ffn'].experts.parameters())

    num_experts = config.get('num_experts', 1)
    top_k = config.get('top_k', 1)

    if num_experts > 1:
        active_moe_params = (moe_params // num_experts) * top_k
        non_moe_params = total_params - moe_params
        active_params = non_moe_params + active_moe_params
    else:
        active_params = total_params

    stat_str = (
        f"{'='*40}\n"
        f"MODEL ARCHITECTURE STATS\n"
        f"{'='*40}\n"
        f"Config: {ACTIVE_CONFIG_NAME}\n"
        f"Dim: {config.get('dim')} | Layers: {config.get('n_layers')} | Heads: {config.get('n_heads')} (GQA KV: {config.get('n_kv_heads')})\n"
        f"MoE: {num_experts} Experts | Top K: {top_k} | Hidden Dim: {config.get('hidden_dim')}\n"
        f"Max Seq Len: {config.get('max_seq_len')}\n"
        f"{'-'*40}\n"
        f"Total Parameters:  {total_params / 1e6:.2f} M\n"
        f"Active Parameters: {active_params / 1e6:.2f} M (Per Token)\n"
        f"{'='*40}\n"
    )

    print(stat_str)
    if save_path:
        with open(save_path, "w") as f:
            f.write(stat_str)

    return total_params, active_params


# ==========================================
# STATEFUL DATA LOADING (Resume Support)
# ==========================================
class MixedDataStream:
    """Stateful data loader that tracks position in each dataset for checkpoint resume."""
    
    def __init__(self, datasets_dict, probabilities, batch_size, start_step, accumulation_steps,
                 warmup_steps=4000, max_seq_len=1024, start_len=128, is_distributed=False,
                 resume_state=None):
        self.datasets_dict = datasets_dict
        self.probabilities = probabilities
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.warmup_steps = warmup_steps
        self.max_seq_len = max_seq_len
        self.start_len = start_len
        self.is_distributed = is_distributed
        
        self.world_size = dist.get_world_size() if is_distributed else 1
        self.local_rank = dist.get_rank() if is_distributed else 0
        self.dataset_names = list(datasets_dict.keys())
        self.dataset_probs = [probabilities[n] for n in self.dataset_names]
        
        # State that gets saved/restored
        if resume_state:
            self.pointers = resume_state['pointers']
            self.counter = resume_state['counter']
            self.rng_state = resume_state.get('rng_state', None)
            if self.rng_state is not None:
                np.random.set_state(self.rng_state)
        else:
            self.pointers = {k: 0 for k in datasets_dict.keys()}
            self.counter = start_step * accumulation_steps
            self.rng_state = None
            
    def get_state(self):
        """Returns serializable state dict for checkpointing."""
        return {
            'pointers': self.pointers.copy(),
            'counter': self.counter,
            'rng_state': np.random.get_state()
        }
    
    def __iter__(self):
        """Return self as iterator."""
        return self
    
    def __next__(self):
        """Generate next batch."""
        current_step = self.counter // self.accumulation_steps
        current_seq_len = get_seq_len(current_step, self.warmup_steps, self.max_seq_len, self.start_len)
        tokens_per_local_batch = self.batch_size * (current_seq_len + 1)
        tokens_per_global_batch = self.world_size * tokens_per_local_batch

        source = np.random.choice(self.dataset_names, p=self.dataset_probs)
        self.counter += 1
        mmap = self.datasets_dict[source]
        ptr = self.pointers[source]

        # Wrap around if at end of file
        if ptr + tokens_per_global_batch > len(mmap):
            ptr = 0
            self.pointers[source] = 0

        start_idx = ptr + (self.local_rank * tokens_per_local_batch)
        end_idx = start_idx + tokens_per_local_batch

        if end_idx > len(mmap):
            chunk = np.zeros(tokens_per_local_batch, dtype=np.int64)
        else:
            chunk = mmap[start_idx:end_idx].astype(np.int64)

        self.pointers[source] += tokens_per_global_batch
        chunk = torch.from_numpy(chunk).view(self.batch_size, current_seq_len + 1)

        x = chunk[:, :current_seq_len].contiguous()
        y = chunk[:, 1:current_seq_len + 1].contiguous()
        return x, y


def load_dataset_index(index_path="data/index.txt"):
    datasets = {'train': {}, 'val': {}}
    probabilities = {}
    total_weight = 0.0

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Cannot find dataset index at {index_path}")

    with open(index_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) != 2:
                continue

            filename, weight = parts[0].strip(), float(parts[1].strip())
            filepath = os.path.join("data", filename)
            if not os.path.exists(filepath):
                print(f"WARNING: {filepath} not found, skipping.")
                continue

            full_mmap = np.memmap(filepath, dtype=np.uint16, mode='r')
            split_idx = int(len(full_mmap) * 0.95)

            datasets['train'][filename] = full_mmap[:split_idx]
            datasets['val'][filename] = full_mmap[split_idx:]
            probabilities[filename] = weight
            total_weight += weight

    for k in probabilities:
        probabilities[k] /= total_weight

    return datasets, probabilities


def get_latest_checkpoint(checkpoint_dir="jesper_checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return None
    steps = []
    for folder in os.listdir(checkpoint_dir):
        if folder.startswith("step_"):
            try:
                steps.append(int(folder.split("_")[1]))
            except ValueError:
                pass
    if not steps:
        return None
    return os.path.join(checkpoint_dir, f"step_{max(steps)}", "checkpoint.pt")


def get_lr(step, max_steps, max_lr, min_lr, warmup_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ==========================================
# TEXT GENERATION (EVALUATION)
# ==========================================
@torch.no_grad()
def generate_eval_samples(model, tokenizer, prompts, max_new_tokens=128, device='cuda',
                          temperature=0.8, top_p=0.9):
    """Returns (generated_texts, total_tokens_generated)."""
    model.eval()
    results = []
    total_tokens = 0
    base_model = model.module if hasattr(model, 'module') else model

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        prompt_len = input_ids.size(1)
        
        for _ in range(max_new_tokens):
            seq = input_ids[:, -base_model.max_seq_len:]

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits, _, _ = model(seq)

            next_logits = logits[:, -1, :].float()
            next_logits = next_logits / temperature
            probs = torch.softmax(next_logits, dim=-1)
            
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
            sorted_probs[sorted_indices_to_remove] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, sampled_sorted_idx)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        generated_tokens = input_ids.size(1) - prompt_len
        total_tokens += generated_tokens
        
        decoded = tokenizer.decode(input_ids[0].cpu().tolist(), skip_special_tokens=True,
                                   clean_up_tokenization_spaces=True)
        results.append(decoded)

    return results, total_tokens


# ==========================================
# MAIN TRAINING LOOP
# ==========================================
def train():
    local_rank = setup_ddp()
    is_distributed = dist.is_initialized() if dist.is_available() else False
    world_size = dist.get_world_size() if is_distributed else 1
    device = torch.device(f"cuda:{local_rank}" if local_rank is not None else "cuda:0")
    is_main = (local_rank == 0) or (local_rank is None)

    tokenizer_path = "custom_tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Dynamic Hyperparameters & Setup ----
    #current_cfg = MODEL_CONFIGS[ACTIVE_CONFIG_NAME]
    
    #batch_size = current_cfg.get("micro_batch_size", 1)
    #target_acc_steps = current_cfg.get("target_accumulation_steps", 128)
    #accumulation_steps = max(1, target_acc_steps // world_size)
    
    #if is_main:
    #    actual_global_steps = accumulation_steps * world_size
    #    print(f"\n--- BATCH SCALING ---")
    #    print(f"GPUs (World Size): {world_size}")
    #    print(f"Local Accumulation Steps: {accumulation_steps}")
    #    print(f"Global Effective Steps: {actual_global_steps} (Target: {target_acc_steps})\n")

    #seq_len_start = current_cfg.get("seq_len_start", 128)
    # ---- Dynamic Hyperparameters & Setup ----
    current_cfg = MODEL_CONFIGS[ACTIVE_CONFIG_NAME]
    
    batch_size = current_cfg.get("micro_batch_size", 1)
    target_acc_steps = current_cfg.get("target_accumulation_steps", 128)
    
    # Calculate accumulation steps factoring in both the cluster size and micro-batch size
    accumulation_steps = max(1, target_acc_steps // (world_size * batch_size))
    
    if is_main:
        actual_global_batch = accumulation_steps * world_size * batch_size
        print(f"\n--- BATCH SCALING ---")
        print(f"GPUs (World Size): {world_size}")
        print(f"Micro-Batch Size: {batch_size}")
        print(f"Local Accumulation Steps: {accumulation_steps}")
        print(f"Global Effective Batch: {actual_global_batch} (Target: {target_acc_steps})\n")

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

    checkpoint_dir = "jesper_checkpoints"
    start_step = 0
    train_loss_history = []
    val_loss_history = []
    
    # TOKEN TRACKING STATE
    total_tokens_trained = 0  # Cumulative across all GPUs
    total_tokens_generated = 0  # Cumulative eval tokens
    train_stream_state = None
    val_stream_state = None

    latest_ckpt_path = get_latest_checkpoint(checkpoint_dir)

    if latest_ckpt_path and os.path.exists(latest_ckpt_path):
        if is_main:
            print(f"\n[!] Resuming from: {latest_ckpt_path}")
            
        checkpoint = torch.load(latest_ckpt_path, map_location='cpu', weights_only=False)
        model_config = checkpoint.get('model_config', current_cfg)

        start_step = checkpoint['step'] + 1
        train_loss_history = checkpoint.get('train_loss_history', [])
        val_loss_history = checkpoint.get('val_loss_history', [])
        
        # RESTORE TOKEN COUNTERS
        total_tokens_trained = checkpoint.get('tokens_trained', 0)
        total_tokens_generated = checkpoint.get('tokens_generated', 0)
        train_stream_state = checkpoint.get('train_stream_state', None)
        val_stream_state = checkpoint.get('val_stream_state', None)

        if is_main:
            initial_seq_len = get_seq_len(start_step, seq_len_warmup, model_config["max_seq_len"], seq_len_start)
            print(f"[!] Resuming at step {start_step} | Tokens trained: {total_tokens_trained:,} | Context: {initial_seq_len}")
            if total_tokens_generated > 0:
                print(f"[!] Total tokens generated (eval): {total_tokens_generated:,}")
            print()
    else:
        if is_main:
            print(f"\n[!] Starting fresh: {ACTIVE_CONFIG_NAME}")
            initial_seq_len = get_seq_len(0, seq_len_warmup, current_cfg["max_seq_len"], seq_len_start)
            print(f"[!] Starting context size: {initial_seq_len} tokens\n")
        model_config = current_cfg
        os.makedirs(checkpoint_dir, exist_ok=True)

    arch_keys = ["dim", "n_layers", "n_heads", "n_kv_heads", "hidden_dim", "num_experts", "top_k", "max_seq_len"]
    arch_config = {k: v for k, v in model_config.items() if k in arch_keys}

    model = JesperLLM(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
        **arch_config
    ).to(device)

    if is_main:
        print_model_stats(model, model_config)

    if HAS_BNB:
        print("Using 8-bit bitsandbytes AdamW optimizer")
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=max_lr, betas=(beta1, 0.95), weight_decay=0.1)
    else:
        print("Using torch AdamW optimizer")
        optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(beta1, 0.95), weight_decay=0.1)

    scaler = GradScaler('cuda')

    if latest_ckpt_path and os.path.exists(latest_ckpt_path):
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        for state in optimizer.state.values():
            if 'step' in state and 'state1' not in state:
                state.clear()
            else:
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
                        
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
            
        del checkpoint
        torch.cuda.empty_cache()

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    datasets_dict, probabilities = load_dataset_index("data/index.txt")
    
    # Create stateful data streams with resume support
    train_stream = MixedDataStream(
        datasets_dict['train'], probabilities, batch_size, start_step,
        accumulation_steps, seq_len_warmup, model_config["max_seq_len"],
        seq_len_start, is_distributed, resume_state=train_stream_state
    )
    val_stream = MixedDataStream(
        datasets_dict['val'], probabilities, batch_size, 0, 1,
        0, model_config["max_seq_len"], model_config["max_seq_len"], 
        is_distributed, resume_state=val_stream_state
    )

    # ==========================================
    # DUMMY PASS FOR VRAM PRE-ALLOCATION
    # ==========================================
    if is_main:
        print("\n--- Running Dummy Pass to Pre-allocate Max VRAM ---")
    
    model.train()
    optimizer.zero_grad()
    
    dummy_x = torch.randint(0, len(tokenizer), (batch_size, model_config["max_seq_len"]), device=device)
    dummy_y = torch.randint(0, len(tokenizer), (batch_size, model_config["max_seq_len"]), device=device)
    
    with torch.amp.autocast('cuda', dtype=torch.float16):
        _, dummy_ce, dummy_aux = model(dummy_x, dummy_y)
        dummy_loss = (dummy_ce / accumulation_steps) + (aux_weight * (dummy_aux / accumulation_steps))
    
    scaler.scale(dummy_loss).backward()
    optimizer.zero_grad()
    del dummy_x, dummy_y, dummy_loss, dummy_ce, dummy_aux
    
    if is_main:
        max_mem = torch.cuda.memory_allocated(device) / 1e9
        print(f"Max VRAM Successfully Reserved: {max_mem:.1f}GB")
        print("---------------------------------------------------\n")

    if is_distributed:
        dist.barrier()

    # Training metrics timers
    t0 = time.time()
    local_tokens_since_last_log = 0  # Tokens processed on this GPU
    
    # Pre-allocate for throughput calculation
    train_iter = iter(train_stream)

    for step in range(start_step, total_steps):
        current_seq_len = get_seq_len(step, seq_len_warmup, model_config["max_seq_len"], seq_len_start)
        
        # Tokens processed on THIS GPU for this global step
        local_tokens_this_step = batch_size * accumulation_steps * current_seq_len
        local_tokens_since_last_log += local_tokens_this_step
        
        # Total tokens across all GPUs (for global tracking)
        global_tokens_this_step = local_tokens_this_step * world_size
        total_tokens_trained += global_tokens_this_step
        
        lr = get_lr(step, total_steps, max_lr, min_lr, warmup_steps)
        dynamic_beta2 = 1.0 - (math.log(2) / beta2_half_life) * global_tokens_this_step
        dynamic_beta2 = max(0.0, min(0.9999, dynamic_beta2))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            param_group['betas'] = (beta1, dynamic_beta2)

        model.train()
        optimizer.zero_grad()
        accumulated_ce_loss = 0.0
        accumulated_aux_loss = 0.0

        for micro_step in range(accumulation_steps):
            x, y = next(train_iter)
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits, ce_loss, aux_loss = model(x, y)
                ce_loss_scaled = ce_loss / accumulation_steps
                aux_loss_scaled = aux_loss / accumulation_steps
                total_loss = ce_loss_scaled + (aux_weight * aux_loss_scaled)

            if is_distributed and micro_step < accumulation_steps - 1:
                with model.no_sync():
                    scaler.scale(total_loss).backward()
            else:
                scaler.scale(total_loss).backward()

            accumulated_ce_loss += ce_loss_scaled.item()
            accumulated_aux_loss += aux_loss_scaled.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss_history.append((step, accumulated_ce_loss))

        if is_main and step % 10 == 0:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            mem = torch.cuda.memory_allocated(device) / 1e9
            
            t1 = time.time()
            dt = t1 - t0
            
            # Throughput calculations
            if step > start_step and dt > 0:
                # Per-GPU throughput (this GPU's processing speed)
                local_tok_per_sec = local_tokens_since_last_log / dt
                # Total cluster throughput (assuming perfect scaling)
                total_tok_per_sec = local_tok_per_sec * world_size
                
                global_steps_per_sec = 10 / dt
                local_passes_per_sec = (10 * accumulation_steps) / dt 
            else:
                local_tok_per_sec = 0.0
                total_tok_per_sec = 0.0
                global_steps_per_sec = 0.0
                local_passes_per_sec = 0.0
                
            t0 = t1
            local_tokens_since_last_log = 0
            
            # Split over two lines to prevent massive horizontal wrap
#            print(
#                f"[{current_time}] Step {step:05d} | "
#                f"{global_steps_per_sec:.2f} global steps/s | "
#                f"{local_passes_per_sec:.1f} local passes/s\n"
#                f"          Tok/s: {local_tok_per_sec:,.0f} (GPU) | {total_tok_per_sec:,.0f} (Total) | "
#                f"Total Trained: {total_tokens_trained:,}\n"
#                f"          LR: {lr:.2e} | B2: {dynamic_beta2:.4f} | Seq: {current_seq_len} | "
#                f"CE Loss: {accumulated_ce_loss:.4f} | VRAM: {mem:.1f}GB"
#            )
            print(
                f"[{current_time}] Step {step:05d} | "
                f"{global_steps_per_sec:.2f} global steps/s | "
                f"{local_passes_per_sec:.1f} local passes/s\n"
                f"          Tok/s: {local_tok_per_sec:,.0f} (GPU) | {total_tok_per_sec:,.0f} (Total) | "
                f"Total Trained: {total_tokens_trained:,}\n"
                f"          LR: {lr:.2e} | B2: {dynamic_beta2:.4f} | Seq: {current_seq_len} | "
                f"CE Loss: {accumulated_ce_loss:.4f} | Aux: {accumulated_aux_loss:.4f} | VRAM: {mem:.1f}GB"
            )


        # Validation, eval, and checkpointing
        if step > 0 and step % eval_interval == 0:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for _ in range(val_eval_steps):
                    vx, vy = next(val_stream)
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        _, v_ce_loss, _ = model(vx, vy)
                        val_loss += v_ce_loss.item()

            val_loss /= val_eval_steps
            val_loss_history.append((step, val_loss))

            if is_main:
                print(f"\n--- Validation at Step {step} | Val Loss: {val_loss:.4f} ---")

                print("Generating Eval Samples...")
                generated_texts, eval_tokens = generate_eval_samples(
                    model, tokenizer, EVAL_PROMPTS, device=device,
                    temperature=0.8, top_p=0.9
                )
                total_tokens_generated += eval_tokens
                
                for p, gen in zip(EVAL_PROMPTS, generated_texts):
                    print(f"Prompt: {p}\nOutput: {gen}\n" + "-" * 30)

                ckpt_dir = os.path.join(checkpoint_dir, f"step_{step}")
                os.makedirs(ckpt_dir, exist_ok=True)

                try:
                    shutil.copy("jesper_model.py", os.path.join(ckpt_dir, "jesper_model_snapshot.py"))
                    shutil.copy(__file__, os.path.join(ckpt_dir, f"{os.path.basename(__file__)}_snapshot.py"))
                except Exception as e:
                    print(f"Warning: Could not save code snapshots: {e}")

                print_model_stats(model, model_config, save_path=os.path.join(ckpt_dir, "model_stats.txt"))

                # Save data stream states for resume
                current_train_state = train_stream.get_state()
                current_val_state = val_stream.get_state()

                ckpt = {
                    'model_config': model_config,
                    'model': model.module.state_dict() if is_distributed else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'step': step,
                    'train_loss_history': train_loss_history,
                    'val_loss_history': val_loss_history,
                    # NEW: Token tracking
                    'tokens_trained': total_tokens_trained,
                    'tokens_generated': total_tokens_generated,
                    # NEW: Data position tracking
                    'train_stream_state': current_train_state,
                    'val_stream_state': current_val_state,
                }
                torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint.pt"))

                with open(os.path.join(ckpt_dir, "eval_samples.json"), "w") as f:
                    json.dump({
                        "step": step,
                        "val_loss": round(val_loss, 4),
                        "tokens_trained": total_tokens_trained,
                        "tokens_generated": total_tokens_generated,
                        "samples": generated_texts
                    }, f, indent=2)

                if len(train_loss_history) > 1:
                    t_steps, t_losses = zip(*train_loss_history)
                    v_steps, v_losses = zip(*val_loss_history) if val_loss_history else ([], [])

                    plt.figure(figsize=(10, 6))
                    plt.plot(t_steps, t_losses, label="Train CE Loss", alpha=0.5)
                    if v_losses:
                        plt.plot(v_steps, v_losses, label="Val CE Loss", color='red', linewidth=2)
                    plt.xlabel("Step")
                    plt.ylabel("Cross Entropy Loss")
                    plt.title(f"{ACTIVE_CONFIG_NAME} — Training Curves (Step {step})\nTotal Tokens: {total_tokens_trained:,}")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(ckpt_dir, "loss_curve.png"), dpi=150)
                    plt.close()

                print(f">>> Saved checkpoint to: {ckpt_dir}")
                print(f"    Total tokens trained: {total_tokens_trained:,}")
                print(f"    Total tokens generated: {total_tokens_generated:,}\n")

            t0 = time.time()
            if is_distributed:
                dist.barrier()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
