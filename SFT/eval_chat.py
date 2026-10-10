"""Chat eval for SFT checkpoints: 12 diverse ChatML prompts + samples.

Loads a chat_model/vesper_chat.pt (plain or LoRA-merged state dict) and
generates one reply per prompt. Usage:
  python3 eval_chat.py [--ckpt lab/imported/sft_470m/step_N/chat_model] [--out samples.json]
Defaults to the newest step_*/chat_model under $VESPER_SFT_OUT (or
lab/imported/sft_470m). CPU or CUDA (auto).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "Common"))
from vesper_linear_model import VesperLinearLM  # noqa: E402

PROMPTS = [
    "Hello! How are you today?",
    "What is the capital of France?",
    "Can you explain what a black hole is in simple terms?",
    "Write a short poem about the ocean.",
    "What is 17 * 23?",
    "How do I boil an egg?",
    "What is the difference between a list and a tuple in Python?",
    "Tell me a fun fact about space.",
    "Summarize: Photosynthesis is the process plants use to convert sunlight into energy.",
    "I feel really stressed about work lately. Any advice?",
    "Translate to French: The weather is nice today.",
    "Why do we dream?",
]


def load_chat_model(ckpt_path, device):
    tok_dir = os.path.join(ckpt_path) if os.path.isdir(ckpt_path) else os.path.dirname(ckpt_path)
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(tok_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "vesper_chat.pt")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = payload["model_config"]
    arch = ["dim", "n_layers", "n_heads", "n_kv_heads", "hidden_dim",
            "num_experts", "top_k", "max_seq_len", "linear_type",
            "full_type", "attention_every", "gla_head_dim", "qk_norm",
            "kda_head_dim", "kda_short_conv", "mamba2_state_size",
            "kv_lora_rank", "v_head_dim", "linear_force_fp32",
            "router_type", "passport_dim", "router_expert_dropout"]
    kwargs = {k: mc[k] for k in arch if k in mc}
    model = VesperLinearLM(vocab_size=len(tok), pad_id=tok.pad_token_id,
                           grad_checkpoint=False, **kwargs)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, prompt, max_new=180, device="cpu"):
    device = torch.device(device)
    im_end = tok.convert_tokens_to_ids("im_end")
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    for _ in range(max_new):
        seq = ids[:, -model.max_seq_len:]
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            logits, _, _ = model(seq)
        nxt = logits[:, -1, :].float() / 0.7
        probs = F.softmax(nxt, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        cp = torch.cumsum(sp, dim=-1)
        sp[cp - sp > 0.9] = 0.0
        sp = sp / sp.sum(dim=-1, keepdim=True)
        pick = si.gather(-1, torch.multinomial(sp, 1))
        ids = torch.cat([ids, pick], dim=1)
        if int(pick) in (tok.eos_token_id, im_end):
            break
    return tok.decode(ids[0].tolist(), skip_special_tokens=False)


def find_latest_chat(out_root):
    cands = sorted(glob.glob(os.path.join(out_root, "step_*", "chat_model")))
    return cands[-1] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="chat_eval_samples.json")
    ap.add_argument("--max-new", type=int, default=180)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    out_root = os.environ.get("VESPER_SFT_OUT") or os.path.join(REPO, "lab", "imported", "sft_470m")
    ckpt = args.ckpt or find_latest_chat(out_root)
    if not ckpt:
        sys.exit(f"no chat_model under {out_root}")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, tok = load_chat_model(ckpt, device)
    print(f"loaded {ckpt} on {device}")
    results = []
    for prompt in PROMPTS:
        chat = f"{chr(60)}|im_start|{chr(62)}user\n{prompt}{chr(60)}|im_end|{chr(62)}\n{chr(60)}|im_start|{chr(62)}assistant\n"
        out = generate(model, tok, chat, max_new=args.max_new, device=device)
        reply = out.split("assistant\n")[-1].split(chr(60))[0].strip()
        results.append({"prompt": prompt, "response": reply})
        print(f"\n=== {prompt}\n{reply}")
    with open(args.out, "w") as f:
        json.dump({"ckpt": str(ckpt), "samples": results}, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
