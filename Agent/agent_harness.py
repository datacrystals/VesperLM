"""
agent_harness.py

Minimal agentic loop for VesperLM tiny_agent checkpoints.

ChatML + tool schema (must match Dataset/08_sft_tooluse.py exactly):
    <|im_start|>user\n{request}<|im_end|>\n
    <|im_start|>assistant\n<|tool_call|>{"tool": "bash", "args": {"command": ...}}<|tool_call_end|><|im_end|>\n
    <|im_start|>tool\n{observation}<|im_end|>\n
    <|im_start|>assistant\n{final answer}<|im_end|>\n

Runs on CPU or GPU; loads the latest pretrain/SFT checkpoint.
"""

import os
import re
import sys
import json
import argparse
import subprocess
import tempfile

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "Common"))
sys.path.insert(0, os.path.join(_REPO, "Pretrain"))
sys.path.insert(0, _REPO)

from vesper_linear_model import VesperLinearLM  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

TOKENIZER_DIR = os.path.join(_REPO, "Pretrain", "custom_tokenizer")
CHECKPOINT_DIR = os.path.join(_REPO, "Pretrain", "vesper_linear_checkpoints")
SFT_CHECKPOINT_DIR = os.path.join(_REPO, "SFT", "vesper_sft_checkpoints")

TINY_AGENT_CONFIG = dict(
    dim=384, n_layers=6, n_heads=6, n_kv_heads=2,
    hidden_dim=1024, num_experts=4, top_k=2,
    max_seq_len=2048,
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TOOL_OPEN = "<|tool_call|>"
TOOL_CLOSE = "</|tool_call|>"
THOUGHT_OPEN = "<|thought|>"
THOUGHT_CLOSE = "</|thought|>"

TOOL_CALL_RE = re.compile(
    r"<\|tool_call\|>\s*(\{.*?\})\s*</\|tool_call\|>", re.DOTALL)
THOUGHT_RE = re.compile(
    r"<\|thought\|>(.*?)</\|thought\|>", re.DOTALL)


def latest_checkpoint():
    """SFT checkpoint dir first, then pretrain dir; return newest step."""
    for d in (SFT_CHECKPOINT_DIR, CHECKPOINT_DIR):
        if not os.path.isdir(d):
            continue
        steps = []
        for name in os.listdir(d):
            if name.startswith("step_") and os.path.isfile(
                    os.path.join(d, name, "checkpoint.pt")):
                steps.append((int(name.split("_")[1]), os.path.join(d, name)))
        if steps:
            return max(steps)[1]
    return None


def load_model(ckpt_path, device):
    ckpt = torch.load(os.path.join(ckpt_path, "checkpoint.pt"),
                      map_location="cpu", weights_only=False)
    cfg = dict(TINY_AGENT_CONFIG)
    mc = ckpt.get("model_config") or {}
    for k in cfg:
        if k in mc:
            cfg[k] = mc[k]
    cfg["vocab_size"] = mc.get("vocab_size", 65523)
    cfg["pad_id"] = mc.get("pad_id", 0)

    model = VesperLinearLM(
        vocab_size=cfg["vocab_size"],
        dim=cfg["dim"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"], hidden_dim=cfg["hidden_dim"],
        num_experts=cfg["num_experts"], top_k=cfg["top_k"],
        max_seq_len=cfg["max_seq_len"], pad_id=cfg["pad_id"],
    )
    state = ckpt["model"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    print(f"[harness] loaded {ckpt_path} (step {ckpt.get('step')}) "
          f"on {device}")
    return model


@torch.no_grad()
def generate(model, tok, prompt_ids, max_new_tokens=160,
             temperature=0.8, top_p=0.9):
    """Greedy-ish sampling loop over the model; returns new token ids."""
    device = next(model.parameters()).device
    ids = prompt_ids
    eos = tok.convert_tokens_to_ids("<endoftext>")
    im_end = tok.convert_tokens_to_ids(IM_END)

    for _ in range(max_new_tokens):
        seq = ids[:, -model.max_seq_len:]
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, _, _ = model(seq)
        else:
            logits, _, _ = model(seq)
        next_logits = logits[:, -1, :].float() / temperature
        probs = torch.softmax(next_logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        remove = cum - sorted_probs > top_p
        sorted_probs[remove] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        pick = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
        tok_id = pick.item()
        ids = torch.cat([ids, pick], dim=1)
        if tok_id in (eos, im_end):
            break
    return ids


def run_tool(command, workdir, timeout=10):
    try:
        r = subprocess.run(
            ["bash", "-c", command], cwd=workdir,
            capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + (("\n[stderr] " + r.stderr) if r.stderr else "")).strip()
        return out[:2000] if out else "[no output]"
    except subprocess.TimeoutExpired:
        return "[error: command timed out]"


def parse_tool_call(text):
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    tool, args = obj.get("tool"), obj.get("args", {})
    if tool != "bash" or not isinstance(args.get("command"), str):
        return None
    return args["command"]


class TinyAgent:
    def __init__(self, model, tok, max_turns=4, max_new_tokens=160,
                 temperature=0.8, top_p=0.9):
        self.model = model
        self.tok = tok
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.workdir = tempfile.mkdtemp(prefix="vesper_agent_")

    def _respond(self, conversation_text):
        ids = self.tok.encode(conversation_text, return_tensors="pt")
        out = generate(self.model, self.tok, ids,
                       max_new_tokens=self.max_new_tokens,
                       temperature=self.temperature, top_p=self.top_p)
        return self.tok.decode(out[0, ids.size(1):].tolist(),
                               skip_special_tokens=False)

    def run(self, user_request, verbose=True):
        conv = f"{IM_START}user\n{user_request}{IM_END}\n"
        transcript = []

        for turn in range(self.max_turns):
            resp = self._respond(conv)
            if verbose:
                print(f"\n--- turn {turn} raw ---\n{resp}\n")

            thought = THOUGHT_RE.search(resp)
            if thought and verbose:
                print(f"[thought] {thought.group(1).strip()[:300]}")

            command = parse_tool_call(resp)
            if command is None:
                # No tool call -> treat as final answer
                final = resp.split(IM_START)[0].strip()
                transcript.append({"role": "assistant", "content": final})
                return final, transcript

            observation = run_tool(command, self.workdir)
            transcript.append({"role": "assistant",
                               "content": f"<|tool_call|>{command}"})
            transcript.append({"role": "tool", "content": observation})
            if verbose:
                print(f"[tool] bash: {command}\n[obs] {observation[:500]}")

            conv += (f"{IM_START}assistant\n"
                     f"{TOOL_OPEN}{json.dumps({'tool': 'bash', 'args': {'command': command}})}{TOOL_CLOSE}{IM_END}\n"
                     f"{IM_START}tool\n{observation}{IM_END}\n")

        # Ran out of turns; force a final answer attempt
        conv += f"{IM_START}assistant\n"
        resp = self._respond(conv)
        final = resp.split(IM_START)[0].strip()
        return final, transcript


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request", nargs="?", help="user request; omit for REPL")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    ckpt_path = args.checkpoint or latest_checkpoint()
    if not ckpt_path:
        sys.exit("No checkpoint found in pretrain or SFT checkpoint dirs.")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda")
    tok = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    model = load_model(ckpt_path, device)

    agent = TinyAgent(model, tok, max_turns=args.max_turns,
                      max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature, top_p=args.top_p)

    if args.request:
        answer, _ = agent.run(args.request)
        print("\n=== FINAL ANSWER ===\n" + answer)
    else:
        print("VesperLM agent REPL. Ctrl-D to exit.")
        while True:
            try:
                q = input("\nyou> ").strip()
            except EOFError:
                break
            if q:
                answer, _ = agent.run(q)
                print("\nagent> " + answer)


if __name__ == "__main__":
    main()
