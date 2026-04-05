import os
import math
import json
import sys
import shutil
import warnings
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler
from transformers import AutoTokenizer
from jesper_model import JesperLLM

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Silence AMD / PyTorch warnings
os.environ["HIPBLASLT_DISABLE"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")

# ==========================================
# CONFIGURATION & TOPOLOGY
# ==========================================
MODEL_CONFIGS = {
    "small_v2": {
        "dim": 1024, "n_layers": 10, "n_heads": 8, "n_kv_heads": 2,
        "hidden_dim": 1280, "num_experts": 8, "top_k": 2, "max_seq_len": 1024
    },
    "big": {
        "dim": 1536, "n_layers": 16, "n_heads": 12, "n_kv_heads": 4,
        "hidden_dim": 4096, "num_experts": 8, "top_k": 2, "max_seq_len": 2048
    }
}
ACTIVE_CONFIG_NAME = "small_v2"
PIPELINE_PARALLEL_SIZE = 2  # Model is split into 2 shards

def setup_hybrid_ddp():
    if not dist.is_available(): return None
    try:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    except: return None
    
    if world_size <= 1: return None

    # Init global group
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    torch.cuda.set_device(local_rank)

    # Calculate Topology
    # Example for 8 GPUs: PP=2, DP=4
    # DP Groups: [0,2,4,6], [1,3,5,7]
    # PP Groups: [0,1], [2,3], [4,5], [6,7]
    dp_size = world_size // PIPELINE_PARALLEL_SIZE
    stage_id = local_rank % PIPELINE_PARALLEL_SIZE
    dp_group_id = local_rank // PIPELINE_PARALLEL_SIZE

    # Create PP Groups
    my_pp_group = None
    for i in range(dp_size):
        ranks = list(range(i * PIPELINE_PARALLEL_SIZE, (i + 1) * PIPELINE_PARALLEL_SIZE))
        group = dist.new_group(ranks)
        if local_rank in ranks: my_pp_group = group

    # Create DP Groups
    my_dp_group = None
    for i in range(PIPELINE_PARALLEL_SIZE):
        ranks = [i + j * PIPELINE_PARALLEL_SIZE for j in range(dp_size)]
        group = dist.new_group(ranks)
        if local_rank in ranks: my_dp_group = group

    return {
        "local_rank": local_rank,
        "world_size": world_size,
        "stage_id": stage_id,
        "dp_group": my_dp_group,
        "pp_group": my_pp_group,
        "pp_size": PIPELINE_PARALLEL_SIZE
    }

# ==========================================
# MODEL SHARDING
# ==========================================
def split_model_for_pipeline(model, stage_id, pp_size, device):
    """Slices JesperLLM into shards for Pipeline Parallelism."""
    total_layers = len(model.layers)
    layers_per_stage = total_layers // pp_size
    
    if stage_id == 0:
        # First Stage: Embedding -> First half of layers
        my_layers = model.layers[:layers_per_stage]
        shard = nn.ModuleDict({
            "embedding": model.embedding,
            "layers": my_layers
        })
    else:
        # Last Stage: Second half of layers -> Norm -> Head
        my_layers = model.layers[layers_per_stage:]
        shard = nn.ModuleDict({
            "layers": my_layers,
            "norm": model.norm,
            "output": model.lm_head
        })
    
    return shard.to(device)

class PipelineWrapper(nn.Module):
    """Wraps the shard to provide a clean forward pass."""
    def __init__(self, shard, stage_id):
        super().__init__()
        self.shard = shard
        self.stage_id = stage_id

    def forward(self, x):
        if self.stage_id == 0:
            # x is input_ids (LongTensor)
            h = self.shard.embedding(x)
            for layer in self.shard.layers:
                h, _, _ = layer(h)
            return h
        else:
            # x is hidden_states (HalfTensor) from previous stage
            h = x
            for layer in self.shard.layers:
                h, _, _ = layer(h)
            h = self.shard.norm(h)
            logits = self.shard.output(h)
            return logits

# ==========================================
# TRAINING LOGIC
# ==========================================
def train():
    ddp_info = setup_hybrid_ddp()
    is_dist = ddp_info is not None
    local_rank = ddp_info["local_rank"] if is_dist else 0
    stage_id = ddp_info["stage_id"] if is_dist else 0
    device = torch.device(f"cuda:{local_rank}")
    
    # Standard Setup
    tokenizer = AutoTokenizer.from_pretrained("custom_tokenizer")
    model_config = MODEL_CONFIGS[ACTIVE_CONFIG_NAME]
    
    # 1. Initialize and Shard Model
    full_model = JesperLLM(vocab_size=len(tokenizer), pad_id=tokenizer.pad_token_id, **model_config)
    model_shard = split_model_for_pipeline(full_model, stage_id, PIPELINE_PARALLEL_SIZE if is_dist else 1, device)
    model_wrapped = PipelineWrapper(model_shard, stage_id)
    del full_model # Save RAM

    # 2. Wrap in DDP for Data Parallelism (syncs gradients across GPUs holding same stage)
    if is_dist:
        model_wrapped = DDP(model_wrapped, device_ids=[local_rank], process_group=ddp_info["dp_group"], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model_wrapped.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = GradScaler('cuda')

    # 3. Training Loop Handshake
    # Simplified manual pipeline (Micro-batching)
    for step in range(30000):
        model_wrapped.train()
        optimizer.zero_grad()
        
        # Buffer for PP communication
        hidden_dim = model_config['dim']
        # This buffer receives activations from Stage 0 to Stage 1
        recv_buf = torch.empty((1, model_config['max_seq_len'], hidden_dim), dtype=torch.float16, device=device)

        # Micro-batch loop
        for _ in range(8): # accumulation_steps
            if stage_id == 0:
                # Stage 0: Load data, Forward, then SEND
                # x, y = next(train_stream)
                # logits_or_h = model_wrapped(x)
                # dist.send(logits_or_h, dst=local_rank + 1, group=ddp_info["pp_group"])
                pass
            
            elif stage_id == 1:
                # Stage 1: RECV, Forward, Backward, then SEND GRADIENT back
                # dist.recv(recv_buf, src=local_rank - 1, group=ddp_info["pp_group"])
                # logits = model_wrapped(recv_buf)
                # loss = F.cross_entropy(...)
                # scaler.scale(loss).backward()
                pass

        scaler.step(optimizer)
        scaler.update()

if __name__ == "__main__":
    train()
