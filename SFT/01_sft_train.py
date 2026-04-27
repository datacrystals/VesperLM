import os
import math
import json
import sys
import shutil
import warnings
import time
import datetime

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
    print("Using bitsandbytes 8-bit optimizer")
except ImportError:
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
# SFT CONFIG
# Separate from pretrain — SFT uses much lower LR,
# full seq len from step 0, and more frequent checkpoints.
# ==========================================
SFT_CONFIGS = {
    "small_v2_sft": {
        # Must match your pretrained model's architecture exactly
        "dim": 1024, "n_layers": 10, "n_heads": 8, "n_kv_heads": 2,
        "hidden_dim": 1280, "num_experts": 8, "top_k": 2, "max_seq_len": 1024,

        # Batching — same as pretrain
        "micro_batch_size": 1,
        "target_accumulation_steps": 128,

        # SFT uses a much lower LR than pretraining to avoid catastrophic forgetting
        # Rule of thumb: ~10x lower than pretrain max_lr
        "beta1": 0.9,
        "beta2": 0.95,           # Fixed beta2 for SFT — no dynamic schedule needed
        "max_lr": 2e-5,          # 10x lower than pretrain
        "min_lr": 2e-6,
        "aux_weight": 0.01,

        # Short warmup, modest total steps
        # SFT doesn't need as many steps — you're nudging not relearning
        # 2000-5000 steps is usually plenty for this data size
        # Increase if val loss is still clearly falling at the end
        "warmup_steps": 200,
        "total_steps": 3000,

        # More frequent checkpoints than pretrain — SFT can overfit fast
        # so you want checkpoints to pick the best one
        "eval_interval": 100,
        "val_eval_steps": 30,
    },
    "big_sft": {
        "dim": 1536, "n_layers": 16, "n_heads": 12, "n_kv_heads": 4,
        "hidden_dim": 4096, "num_experts": 8, "top_k": 2, "max_seq_len": 2048,

        "micro_batch_size": 1,
        "target_accumulation_steps": 128,

        "beta1": 0.9,
        "beta2": 0.95,
        "max_lr": 1.5e-5,
        "min_lr": 1.5e-6,
        "aux_weight": 0.01,

        "warmup_steps": 300,
        "total_steps": 3000,

        "eval_interval": 100,
        "val_eval_steps": 30,
    },
}

ACTIVE_CONFIG_NAME = "small_v2_sft"

# Path to the pretrained checkpoint to start SFT from.
# Set to None to scan sft_checkpoints/ for a resume instead.
PRETRAIN_CHECKPOINT = "jesper_checkpoints/step_19900/checkpoint.pt"

# ChatML eval prompts — unlike pretrain these are full conversation turns
# so we can see if Vesper is learning to chat correctly
EVAL_PROMPTS = [
    "<|im_start|>user\nHello! Who are you?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nDo you have feelings?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWhat makes you curious?<|im_end|>\n<|im_start|>assistant\n",
]


# ==========================================
# HELPERS
# ==========================================
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


def get_lr(step, max_steps, max_lr, min_lr, warmup_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def print_model_stats(model, config, save_path=None):
    total_params = sum(p.numel() for p in model.parameters())

    moe_params = 0
    base_model = model.module if hasattr(model, 'module') else model
    for layer in base_model.layers:
        if hasattr(layer['ffn'], 'experts'):
            moe_params += sum(p.numel() for p in layer['ffn'].experts.parameters())

    num_experts = config.get('num_experts', 1)
    top_k       = config.get('top_k', 1)

    if num_experts > 1:
        active_moe_params  = (moe_params // num_experts) * top_k
        non_moe_params     = total_params - moe_params
        active_params      = non_moe_params + active_moe_params
    else:
        active_params = total_params

    stat_str = (
        f"{'='*40}\n"
        f"MODEL ARCHITECTURE STATS\n"
        f"{'='*40}\n"
        f"Config: {ACTIVE_CONFIG_NAME}\n"
        f"Dim: {config.get('dim')} | Layers: {config.get('n_layers')} | "
        f"Heads: {config.get('n_heads')} (GQA KV: {config.get('n_kv_heads')})\n"
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
# SFT DATA LOADING
# Reads interleaved [token, mask, token, mask, ...] bin files
# produced by 05_sft_oasst.py and 06_sft_vesper.py
# ==========================================
def load_sft_index(index_path="data/sft/index.txt"):
    datasets     = {'train': {}, 'val': {}}
    probabilities = {}
    total_weight  = 0.0

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Cannot find SFT dataset index at {index_path}\n"
            f"Create it with lines like:\n"
            f"  oasst2_sft.bin, 1.0\n"
            f"  vesper_sft.bin, 1.0\n"
        )

    with open(index_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) != 2:
                continue

            filename = parts[0].strip()
            weight   = float(parts[1].strip())
            filepath = os.path.join("data/sft", filename)

            if not os.path.exists(filepath):
                print(f"WARNING: {filepath} not found, skipping.")
                continue

            full_mmap = np.memmap(filepath, dtype=np.uint16, mode='r')
            # Each element is a token+mask pair so total "sequence positions" = len/2
            split_idx = int(len(full_mmap) * 0.95)
            # Keep split on an even boundary so pairs stay intact
            split_idx = (split_idx // 2) * 2

            datasets['train'][filename] = full_mmap[:split_idx]
            datasets['val'][filename]   = full_mmap[split_idx:]
            probabilities[filename]     = weight
            total_weight               += weight
            print(f"  Loaded {filename}: {len(full_mmap)//2:,} token positions")

    for k in probabilities:
        probabilities[k] /= total_weight

    return datasets, probabilities


def sft_data_stream(datasets_dict, probabilities, batch_size, seq_len,
                    is_distributed=False, start_step=0, accumulation_steps=1):
    """
    Yields (x, y, loss_mask) batches from interleaved SFT bin files.
    x      — input tokens  (B, T)
    y      — target tokens (B, T)  [x shifted by 1]
    mask   — loss mask     (B, T)  [1 = compute loss, 0 = ignore]
    """
    world_size = dist.get_world_size() if is_distributed else 1
    local_rank = dist.get_rank()       if is_distributed else 0

    dataset_names = list(datasets_dict.keys())
    dataset_probs = [probabilities[n] for n in dataset_names]
    pointers      = {k: 0 for k in datasets_dict.keys()}

    # Each sequence position takes 2 uint16s (token + mask)
    # We need seq_len+1 positions to produce x (T) and y (T) via shift
    stride = (seq_len + 1) * 2   # in uint16 units

    counter = start_step * accumulation_steps

    while True:
        source = np.random.choice(dataset_names, p=dataset_probs)
        counter += 1
        mmap = datasets_dict[source]
        ptr  = pointers[source]

        tokens_needed = stride * batch_size * world_size

        if ptr + tokens_needed > len(mmap):
            ptr = 0
            pointers[source] = 0

        local_start = ptr + (local_rank * stride * batch_size)
        local_end   = local_start + (stride * batch_size)

        if local_end > len(mmap):
            chunk = np.zeros(stride * batch_size, dtype=np.int64)
        else:
            chunk = mmap[local_start:local_end].astype(np.int64)

        pointers[source] += tokens_needed

        # Deinterleave: even indices = tokens, odd indices = masks
        tokens_flat = chunk[0::2]   # every other element starting at 0
        masks_flat  = chunk[1::2]   # every other element starting at 1

        # Reshape to (batch_size, seq_len+1)
        tokens = tokens_flat.reshape(batch_size, seq_len + 1)
        masks  = masks_flat.reshape(batch_size, seq_len + 1)

        x    = torch.from_numpy(tokens[:, :seq_len].copy()).contiguous()
        y    = torch.from_numpy(tokens[:, 1:seq_len + 1].copy()).contiguous()
        mask = torch.from_numpy(masks[:, 1:seq_len + 1].copy()).contiguous()

        yield x, y, mask


# ==========================================
# SFT LOSS (masked — only assistant tokens)
# ==========================================
def masked_ce_loss(logits, targets, mask):
    """
    Cross entropy loss computed only where mask=1 (assistant turns).
    Falls back to full loss if mask is all zeros (shouldn't happen normally).
    """
    B, T, V = logits.shape
    logits_flat  = logits.view(-1, V)
    targets_flat = targets.view(-1)
    mask_flat    = mask.view(-1).float()

    # Raw per-token loss, no reduction
    per_token_loss = torch.nn.functional.cross_entropy(
        logits_flat, targets_flat, reduction='none'
    )

    masked_loss  = (per_token_loss * mask_flat).sum()
    token_count  = mask_flat.sum().clamp(min=1.0)
    return masked_loss / token_count


# ==========================================
# EVAL GENERATION
# ==========================================
@torch.no_grad()
def generate_eval_samples(model, tokenizer, prompts, max_new_tokens=200,
                           device='cuda', temperature=0.8, top_p=0.9):
    model.eval()
    results    = []
    base_model = model.module if hasattr(model, 'module') else model

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        for _ in range(max_new_tokens):
            seq = input_ids[:, -base_model.max_seq_len:]

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits, _, _ = model(seq)

            next_logits = logits[:, -1, :].float() / temperature
            probs       = torch.softmax(next_logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs             = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
            sorted_probs[sorted_indices_to_remove] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token  = sorted_indices.gather(-1, sampled_idx)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop at <|im_end|> or EOS
            if next_token.item() in (tokenizer.eos_token_id, im_end_id):
                break

        decoded = tokenizer.decode(
            input_ids[0].cpu().tolist(),
            skip_special_tokens=False,        # keep special tokens visible in eval
            clean_up_tokenization_spaces=True
        )
        results.append(decoded)

    return results


# ==========================================
# CHECKPOINT HELPERS
# ==========================================
def get_latest_sft_checkpoint(checkpoint_dir="sft_checkpoints"):
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


def save_chat_checkpoint(model, tokenizer, step, checkpoint_dir, model_config):
    """
    Saves a HuggingFace-compatible checkpoint so you can load it with
    AutoModelForCausalLM later for chat_vesper.py.
    Also saves the tokenizer alongside it.
    """
    chat_dir  = os.path.join(checkpoint_dir, f"step_{step}", "chat_model")
    os.makedirs(chat_dir, exist_ok=True)

    base_model = model.module if hasattr(model, 'module') else model

    # Save raw state dict + config for easy loading in chat script
    torch.save({
        'model_state_dict': base_model.state_dict(),
        'model_config':     model_config,
        'step':             step,
    }, os.path.join(chat_dir, "vesper_chat.pt"))

    # Save tokenizer alongside so chat script is self-contained
    tokenizer.save_pretrained(chat_dir)

    print(f"  💬 Chat checkpoint saved to {chat_dir}/")


# ==========================================
# MAIN TRAINING LOOP
# ==========================================
def train():
    local_rank     = setup_ddp()
    is_distributed = dist.is_initialized() if dist.is_available() else False
    world_size     = dist.get_world_size() if is_distributed else 1
    device         = torch.device(f"cuda:{local_rank}" if local_rank is not None else "cuda:0")
    is_main        = (local_rank == 0) or (local_rank is None)

    tokenizer_path = "custom_tokenizer"
    tokenizer      = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Config ----
    cfg = SFT_CONFIGS[ACTIVE_CONFIG_NAME]

    batch_size        = cfg.get("micro_batch_size", 1)
    target_acc_steps  = cfg.get("target_accumulation_steps", 128)
    accumulation_steps = max(1, target_acc_steps // world_size)
    seq_len           = cfg["max_seq_len"]

    max_lr        = cfg["max_lr"]
    min_lr        = cfg["min_lr"]
    beta1         = cfg["beta1"]
    beta2         = cfg["beta2"]
    aux_weight    = cfg["aux_weight"]
    warmup_steps  = cfg["warmup_steps"]
    total_steps   = cfg["total_steps"]
    eval_interval = cfg["eval_interval"]
    val_eval_steps = cfg["val_eval_steps"]

    checkpoint_dir = "sft_checkpoints"
    start_step     = 0
    train_loss_history = []
    val_loss_history   = []

    if is_main:
        print(f"\n{'='*50}")
        print(f"  Vesper SFT Training — {ACTIVE_CONFIG_NAME}")
        print(f"{'='*50}")
        print(f"  Max LR:      {max_lr:.2e}  (vs pretrain ~2e-4)")
        print(f"  Total Steps: {total_steps}")
        print(f"  Eval Every:  {eval_interval} steps")
        print(f"{'='*50}\n")

    # ---- Decide what checkpoint to load ----
    # Priority: existing SFT checkpoint (resume) > pretrain checkpoint (fresh SFT start)
    sft_ckpt_path      = get_latest_sft_checkpoint(checkpoint_dir)
    resume_from        = None
    loading_pretrain   = False

    if sft_ckpt_path and os.path.exists(sft_ckpt_path):
        resume_from      = sft_ckpt_path
        loading_pretrain = False
        if is_main:
            print(f"[!] Resuming SFT from: {sft_ckpt_path}")
    elif PRETRAIN_CHECKPOINT and os.path.exists(PRETRAIN_CHECKPOINT):
        resume_from      = PRETRAIN_CHECKPOINT
        loading_pretrain = True
        if is_main:
            print(f"[!] Starting SFT from pretrain checkpoint: {PRETRAIN_CHECKPOINT}")
    else:
        if is_main:
            print("[!] No checkpoint found — starting SFT from random init (unusual, check paths)")

    # ---- Load checkpoint ----
    checkpoint  = None
    model_config = cfg

    if resume_from:
        checkpoint   = torch.load(resume_from, map_location='cpu', weights_only=False)
        model_config = checkpoint.get('model_config', cfg)

        if not loading_pretrain:
            start_step         = checkpoint['step'] + 1
            train_loss_history = checkpoint.get('train_loss_history', [])
            val_loss_history   = checkpoint.get('val_loss_history', [])
            if is_main:
                print(f"[!] Resuming at SFT step {start_step}\n")
        else:
            if is_main:
                print(f"[!] Pretrain weights loaded — SFT starts at step 0\n")

#        for name, param in model.named_parameters():
#            if 'router' in name:
#                param.requires_grad = False

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---- Build model ----
    arch_keys   = ["dim", "n_layers", "n_heads", "n_kv_heads", "hidden_dim",
                   "num_experts", "top_k", "max_seq_len"]
    arch_config = {k: v for k, v in model_config.items() if k in arch_keys}

    model = JesperLLM(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
        **arch_config
    ).to(device)

    if checkpoint is not None:
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        if is_main and (missing or unexpected):
            print(f"  Checkpoint load — missing keys: {len(missing)}, unexpected: {len(unexpected)}")

    if is_main:
        print_model_stats(model, model_config)

    # ---- Optimizer ----
    # Lower weight decay for SFT — we want to nudge not regularize aggressively
    if IS_ROCM:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=max_lr, betas=(beta1, beta2), weight_decay=0.01)
    elif HAS_BNB:
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=max_lr, betas=(beta1, beta2), weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=max_lr, betas=(beta1, beta2), weight_decay=0.01)

    scaler = GradScaler('cuda')

    if checkpoint is not None and not loading_pretrain:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

    del checkpoint
    torch.cuda.empty_cache()

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ---- Data ----
    if is_main:
        print("\n--- Loading SFT datasets ---")
    datasets_dict, probabilities = load_sft_index("data/sft/index.txt")

    train_stream = sft_data_stream(
        datasets_dict['train'], probabilities, batch_size, seq_len,
        is_distributed, start_step, accumulation_steps
    )
    val_stream = sft_data_stream(
        datasets_dict['val'], probabilities, batch_size, seq_len,
        is_distributed, 0, 1
    )

    # ---- Dummy pass for VRAM pre-allocation ----
    if is_main:
        print("\n--- Running Dummy Pass to Pre-allocate Max VRAM ---")

    model.train()
    optimizer.zero_grad()

    dummy_x = torch.randint(0, len(tokenizer), (batch_size, seq_len), device=device)
    dummy_y = torch.randint(0, len(tokenizer), (batch_size, seq_len), device=device)
    dummy_m = torch.ones(batch_size, seq_len, device=device)

    with torch.amp.autocast('cuda', dtype=torch.float16):
        dummy_logits, _, dummy_aux = model(dummy_x, dummy_y)
        dummy_loss = (
            masked_ce_loss(dummy_logits, dummy_y, dummy_m) / accumulation_steps
            + aux_weight * (dummy_aux / accumulation_steps)
        )

    scaler.scale(dummy_loss).backward()
    optimizer.zero_grad()
    del dummy_x, dummy_y, dummy_m, dummy_logits, dummy_loss, dummy_aux

    if is_main:
        max_mem = torch.cuda.memory_allocated(device) / 1e9
        print(f"Max VRAM Successfully Reserved: {max_mem:.1f}GB")
        print("---------------------------------------------------\n")

    if is_distributed:
        dist.barrier()

    t0 = time.time()

    # ==========================================
    # TRAINING LOOP
    # ==========================================
    for step in range(start_step, total_steps):

        lr = get_lr(step, total_steps, max_lr, min_lr, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        optimizer.zero_grad()

        accumulated_ce_loss  = 0.0
        accumulated_aux_loss = 0.0

        for micro_step in range(accumulation_steps):
            x, y, mask = next(train_stream)
            x    = x.pin_memory().to(device, non_blocking=True)
            y    = y.pin_memory().to(device, non_blocking=True)
            mask = mask.pin_memory().to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits, _, aux_loss = model(x, y)

                ce_loss  = masked_ce_loss(logits, y, mask)
                ce_scaled  = ce_loss  / accumulation_steps
                aux_scaled = aux_loss / accumulation_steps
                total_loss = ce_scaled + (aux_weight * aux_scaled)

            if is_distributed and micro_step < accumulation_steps - 1:
                with model.no_sync():
                    scaler.scale(total_loss).backward()
            else:
                scaler.scale(total_loss).backward()

            accumulated_ce_loss  += ce_scaled.item()
            accumulated_aux_loss += aux_scaled.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss_history.append((step, accumulated_ce_loss))

        # ---- Logging ----
        if is_main and step % 10 == 0:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            mem          = torch.cuda.memory_allocated(device) / 1e9
            t1           = time.time()
            dt           = t1 - t0

            steps_per_sec = 10 / dt if step > start_step else 0.0
            t0 = t1

            print(
                f"[{current_time}] SFT Step {step:05d}/{total_steps} | "
                f"{steps_per_sec:.2f} steps/s\n"
                f"          LR: {lr:.2e} | Seq: {seq_len} | "
                f"CE Loss: {accumulated_ce_loss:.4f} | "
                f"Aux: {accumulated_aux_loss:.4f} | VRAM: {mem:.1f}GB"
            )

        # ---- Eval + Checkpoint ----
        if step > 0 and step % eval_interval == 0:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for _ in range(val_eval_steps):
                    vx, vy, vmask = next(val_stream)
                    vx    = vx.to(device)
                    vy    = vy.to(device)
                    vmask = vmask.to(device)
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        vlogits, _, _ = model(vx, vy)
                        val_loss += masked_ce_loss(vlogits, vy, vmask).item()

            val_loss /= val_eval_steps
            val_loss_history.append((step, val_loss))

            if is_main:
                print(f"\n--- SFT Validation at Step {step} | Val Loss: {val_loss:.4f} ---")

                print("Generating Eval Samples...")
                generated = generate_eval_samples(
                    model, tokenizer, EVAL_PROMPTS, device=device,
                    temperature=0.7, top_p=0.9
                )
                for prompt, gen in zip(EVAL_PROMPTS, generated):
                    # Print just the assistant's reply for clarity
                    reply = gen.split("<|im_start|>assistant\n")[-1].strip()
                    print(f"Q: {prompt.split('user')[1].split('<|im_end|>')[0].strip()}")
                    print(f"Vesper: {reply}\n" + "-" * 30)

                ckpt_dir = os.path.join(checkpoint_dir, f"step_{step}")
                os.makedirs(ckpt_dir, exist_ok=True)

                # Snapshot source files
                try:
                    shutil.copy("jesper_model.py", os.path.join(ckpt_dir, "jesper_model_snapshot.py"))
                    shutil.copy(__file__, os.path.join(ckpt_dir, f"{os.path.basename(__file__)}_snapshot.py"))
                except Exception as e:
                    print(f"Warning: Could not save code snapshots: {e}")

                print_model_stats(model, model_config,
                                  save_path=os.path.join(ckpt_dir, "model_stats.txt"))

                # Full training checkpoint (resumable)
                ckpt = {
                    'model_config':        model_config,
                    'model':               model.module.state_dict() if is_distributed else model.state_dict(),
                    'optimizer':           optimizer.state_dict(),
                    'scaler':              scaler.state_dict(),
                    'step':                step,
                    'train_loss_history':  train_loss_history,
                    'val_loss_history':    val_loss_history,
                }
                torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint.pt"))

                # Chat-ready checkpoint (for chat_vesper.py)
                save_chat_checkpoint(model, tokenizer, step, checkpoint_dir, model_config)

                # Eval samples JSON
                with open(os.path.join(ckpt_dir, "eval_samples.json"), "w") as f:
                    json.dump({
                        "step":     step,
                        "val_loss": round(val_loss, 4),
                        "samples":  generated
                    }, f, indent=2)

                # Loss curve
                if len(train_loss_history) > 1:
                    t_steps, t_losses = zip(*train_loss_history)
                    v_steps, v_losses = zip(*val_loss_history) if val_loss_history else ([], [])

                    plt.figure(figsize=(10, 6))
                    plt.plot(t_steps, t_losses, label="Train CE Loss (masked)", alpha=0.5)
                    if v_losses:
                        plt.plot(v_steps, v_losses, label="Val CE Loss (masked)",
                                 color='red', linewidth=2)
                    plt.xlabel("SFT Step")
                    plt.ylabel("Masked Cross Entropy Loss")
                    plt.title(f"Vesper SFT — {ACTIVE_CONFIG_NAME} (Step {step})")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(ckpt_dir, "sft_loss_curve.png"), dpi=150)
                    plt.close()

                print(f">>> Saved SFT checkpoint + chat model to: {ckpt_dir}\n")

            t0 = time.time()
            if is_distributed:
                dist.barrier()

    if is_main:
        print("\n✅ SFT Training complete!")
        print(f"   Chat checkpoints saved under: {checkpoint_dir}/")
        print(f"   Best checkpoint: pick the step with lowest val loss from the curve")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
