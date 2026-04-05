import os
import math
import json
import torch
import numpy as np
import torch.distributed as dist
from collections import deque
from torch.amp import GradScaler
from transformers import PreTrainedTokenizerFast
from jesper_model import JesperLLM
import bitsandbytes as bnb

# FSDP & Device Mesh Imports
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy
import functools

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def get_seq_len(step, warmup_steps=4000, max_seq_len=1024, start_len=256):
    if step >= warmup_steps:
        return max_seq_len
    progress = step / warmup_steps
    return int(start_len + (max_seq_len - start_len) * progress)

def load_dataset_index(index_path="data/index.txt"):
    datasets = {}
    probabilities = {}
    total_weight = 0.0

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Cannot find dataset index at {index_path}. Please create it!")

    with open(index_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split(',')
            if len(parts) != 2: continue
                
            filename, weight = parts[0].strip(), float(parts[1].strip())
            filepath = os.path.join("data", filename)
            
            if not os.path.exists(filepath): continue
                
            datasets[filename] = np.memmap(filepath, dtype=np.uint16, mode='r')
            probabilities[filename] = weight
            total_weight += weight
            
    for k in probabilities:
        probabilities[k] = probabilities[k] / total_weight
        
    return datasets, probabilities

def mixed_data_stream(batch_size, start_step, accumulation_steps, warmup_steps=4000, max_seq_len=1024):
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()

    datasets, probabilities = load_dataset_index("data/index.txt")
    dataset_names = list(datasets.keys())
    dataset_probs = [probabilities[n] for n in dataset_names]

    if local_rank == 0:
        print("\n--- Dataset Mixing Ratios ---")
        for name, prob in zip(dataset_names, dataset_probs):
            print(f"  > {name}: {prob*100:.2f}%")
        print("-----------------------------\n")

    pointers = {k: 0 for k in datasets.keys()}
    counter = start_step * accumulation_steps

    while True:
        current_step = counter // accumulation_steps
        current_seq_len = get_seq_len(current_step, warmup_steps, max_seq_len)
        
        tokens_per_local_batch = batch_size * (current_seq_len + 1)
        tokens_per_global_batch = world_size * tokens_per_local_batch
        
        source = np.random.choice(dataset_names, p=dataset_probs)
        counter += 1
        
        mmap = datasets[source]
        ptr = pointers[source]
        
        if ptr + tokens_per_global_batch > len(mmap):
            ptr = 0
            pointers[source] = 0
            
        start_idx = ptr + (local_rank * tokens_per_local_batch)
        end_idx = start_idx + tokens_per_local_batch
        
        chunk = mmap[start_idx:end_idx].astype(np.int64)
        pointers[source] += tokens_per_global_batch
        
        chunk = torch.from_numpy(chunk).view(batch_size, current_seq_len + 1)
        x = chunk[:, :current_seq_len].contiguous()
        y = chunk[:, 1:current_seq_len+1].contiguous()
        
        yield x, y, current_seq_len, pointers.copy()

def get_latest_checkpoint(checkpoint_dir="jesper_checkpoints"):
    if not os.path.exists(checkpoint_dir): return None
    steps = [int(f.split("_")[1]) for f in os.listdir(checkpoint_dir) if f.startswith("step_") and f.split("_")[1].isdigit()]
    return os.path.join(checkpoint_dir, f"step_{max(steps)}", "checkpoint.pt") if steps else None

def get_lr(step, max_steps, max_lr, min_lr, warmup_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train():
    local_rank = setup_ddp()
    world_size = dist.get_world_size() 
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained("custom_tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = 1  
    max_seq_len = 1024
    accumulation_steps = 16  # Reduced from 32 because 2 instances double the step throughput
    seq_len_warmup = 4000  
    
    max_lr = 4e-4
    min_lr = 4e-5  
    warmup_steps = 1000
    total_steps = 30000 
    
    start_step = 0
    checkpoint_dir = "jesper_checkpoints" 
    
    loss_history = [] 
    recent_x = deque(maxlen=5) 
    
    # --- STRICT 4GB/GPU Hardware Match ---
    model = JesperLLM(
        vocab_size=65536, 
        pad_id=tokenizer.pad_token_id, 
        max_seq_len=max_seq_len,
        dim=1024,             # Hard cap respected
        n_layers=8,           # Hard cap respected
        n_heads=8,            
        n_kv_heads=2,         
        hidden_dim=2048,      
        num_experts=8, 
        top_k=2
    ).to(local_rank)
    
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = GradScaler('cuda')

    latest_ckpt = get_latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        if local_rank == 0:
            print(f"\n[!] Resuming from checkpoint: {latest_ckpt}\n")
        checkpoint = torch.load(latest_ckpt, map_location=f"cuda:{local_rank}")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
            
        start_step = checkpoint['step'] + 1
        loss_history = checkpoint.get('loss_history', [])
    else:
        if local_rank == 0:
            print("\n[!] No checkpoint found. Starting fresh from step 0.\n")
            os.makedirs(checkpoint_dir, exist_ok=True)

    # --- 2x4 DEVICE MESH CONFIGURATION ---
    mesh = init_device_mesh("cuda", (2, 4))

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100000
    )
    
    model = FSDP(
        model, 
        auto_wrap_policy=my_auto_wrap_policy, 
        sharding_strategy=ShardingStrategy.HYBRID_SHARD, # Enables the 2 instances
        device_mesh=mesh,                                # Passes the 2x4 grid
        use_orig_params=True
    )
    
    data_stream = mixed_data_stream(batch_size, start_step, accumulation_steps, seq_len_warmup, max_seq_len)

    if local_rank == 0:
        print(f"Booting Jesper in 2x4 HYBRID_SHARD configuration. Global batch size: {batch_size * world_size * accumulation_steps}")

    dist.barrier(device_ids=[local_rank]) 

    for step in range(start_step, total_steps):
        lr = get_lr(step, total_steps, max_lr, min_lr, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        optimizer.zero_grad()
        accumulated_loss = 0.0

        for micro_step in range(accumulation_steps):
            x, y, current_seq_len, _ = next(data_stream)
            x, y = x.to(local_rank), y.to(local_rank)
            
            if local_rank == 0:
                recent_x.append(x[0].detach().cpu().numpy())
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits, loss = model(x, y)
                loss = loss / accumulation_steps
            
            if micro_step < accumulation_steps - 1:
                with model.no_sync():
                    scaler.scale(loss).backward()
            else:
                scaler.scale(loss).backward()
            
            accumulated_loss += loss.item()

        scaler.unscale_(optimizer)
        model.clip_grad_norm_(max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if local_rank == 0:
            loss_history.append((step, accumulated_loss))
            if step % 10 == 0:
                print(f"Step {step:05d} | LR: {lr:.2e} | SeqLen: {current_seq_len} | Loss: {accumulated_loss:.4f}")

        if step > 0 and step % 250 == 0:
            # FSDP requires ALL ranks to participate in gathering the state dict
            full_state_dict = model.state_dict()
            
            if local_rank == 0:
                current_checkpoint_dir = os.path.join(checkpoint_dir, f"step_{step}")
                os.makedirs(current_checkpoint_dir, exist_ok=True)
                
                checkpoint = {
                    'model': full_state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'step': step,
                    'loss_history': loss_history
                }
                torch.save(checkpoint, os.path.join(current_checkpoint_dir, "checkpoint.pt"))
                
                decoded_examples = []
                for seq in recent_x:
                    clean_tokens = [t for t in seq if t != tokenizer.pad_token_id][:current_seq_len]
                    decoded_text = tokenizer.decode(clean_tokens, skip_special_tokens=True)
                    decoded_examples.append(decoded_text[:400] + " ... [TRUNCATED]") 
                
                json_path = os.path.join(current_checkpoint_dir, "stats.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "step": step,
                        "seq_len": current_seq_len,
                        "loss": round(accumulated_loss, 4),
                        "samples": decoded_examples
                    }, f, indent=4)
                
                if len(loss_history) > 1:
                    plot_path = os.path.join(current_checkpoint_dir, "loss_curve.png")
                    steps_list, losses_list = zip(*loss_history)
                    plt.figure(figsize=(10, 6))
                    plt.plot(steps_list, losses_list, label="Training Loss", color='blue', linewidth=1.5)
                    plt.xlabel("Step")
                    plt.ylabel("Cross Entropy Loss")
                    plt.title(f"Jesper Training Curve (up to step {step})")
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=150)
                    plt.close()
                
                print(f"\n>>> Saved checkpoint, graph, and samples to: {current_checkpoint_dir}\n")
            
            dist.barrier(device_ids=[local_rank])

    dist.destroy_process_group()

if __name__ == "__main__":
    train()
