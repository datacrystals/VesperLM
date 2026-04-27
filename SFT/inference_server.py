#!/usr/bin/env python3
"""
VesperLLM OpenAI-Compatible Inference Server v2

- Auto-detects architecture from checkpoint
- GPU -> MPS -> CPU fallback
- Concurrent-safe: inference offloaded to thread pool so the event loop never blocks
- Clean client-disconnect handling: stops generation early via threading.Event
- Adds /tokenize endpoint
"""

import os
import sys
import json
import time
import uuid
import argparse
import logging
import asyncio
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
import uvicorn

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vesper-api")

# ------------------------------------------------------------------
# Device detection
# ------------------------------------------------------------------
def detect_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        device = torch.device(preferred)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(device)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device

# ------------------------------------------------------------------
# Import model
# ------------------------------------------------------------------
try:
    from vesper_model import VesperLLM
except ImportError as e:
    logger.error("Failed to import VesperLLM from vesper_model.py")
    raise e

# ------------------------------------------------------------------
# Resolve paths
# ------------------------------------------------------------------
def resolve_checkpoint_file(checkpoint_path: Path) -> Path:
    if checkpoint_path.is_file() and checkpoint_path.suffix == ".pt":
        return checkpoint_path
    if checkpoint_path.is_dir():
        for name in ("vesper_chat.pt", "checkpoint.pt"):
            cand = checkpoint_path / name
            if cand.exists():
                return cand
        candidates = list(checkpoint_path.rglob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No .pt checkpoint found in {checkpoint_path}")
        return candidates[0]
    raise FileNotFoundError(f"Invalid checkpoint path: {checkpoint_path}")


def find_tokenizer_dir(checkpoint_path: Path) -> Path:
    candidates = []
    if checkpoint_path.is_file():
        candidates.append(checkpoint_path.parent)
        chat_dir = checkpoint_path.parent / "chat_model"
        if chat_dir.exists():
            candidates.insert(0, chat_dir)
    else:
        candidates.append(checkpoint_path)

    for p in candidates:
        if (p / "tokenizer_config.json").exists() or (p / "tokenizer.json").exists():
            return p

    fallback = Path("custom_tokenizer")
    if fallback.exists():
        return fallback

    raise FileNotFoundError("Tokenizer not found near checkpoint or in ./custom_tokenizer")

# ------------------------------------------------------------------
# Generation core
# ------------------------------------------------------------------
@torch.no_grad()
def generate_stream(
    model: VesperLLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    stop: Optional[Union[str, List[str]]] = None,
    device: torch.device = torch.device("cpu"),
    stop_event: Optional[threading.Event] = None,
):
    model.eval()
    base_model = model.module if hasattr(model, "module") else model

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Stop token ids
    stop_token_ids = {tokenizer.eos_token_id}
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int):
            stop_token_ids.add(im_end_id)
    except Exception:
        pass

    stop_strings: List[str] = []
    if stop is not None:
        if isinstance(stop, str):
            stop_strings = [stop]
        else:
            stop_strings = list(stop)

    generated_text = ""
    output_ids: List[int] = []

    for _ in range(max_new_tokens):
        if stop_event and stop_event.is_set():
            break

        seq = input_ids[:, -base_model.max_seq_len :]

        use_amp = device.type == "cuda"
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            logits, _, _ = model(seq)

        next_logits = logits[:, -1, :].float()
        if temperature > 0 and temperature != 1.0:
            next_logits = next_logits / temperature

        probs = torch.softmax(next_logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
        sorted_probs[sorted_indices_to_remove] = 0.0

        if sorted_probs.sum() == 0:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, sampled_idx)

        token_id = int(next_token.item())
        input_ids = torch.cat([input_ids, next_token], dim=1)
        output_ids.append(token_id)

        if token_id in stop_token_ids:
            break

        new_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        delta = new_text[len(generated_text) :]
        generated_text = new_text

        for s in stop_strings:
            if generated_text.endswith(s):
                if delta.endswith(s):
                    delta = delta[: -len(s)]
                if delta:
                    yield delta
                return

        if delta:
            yield delta


def generate_non_stream(
    model: VesperLLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    stop: Optional[Union[str, List[str]]] = None,
    device: torch.device = torch.device("cpu"),
) -> str:
    return "".join(
        generate_stream(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            temperature,
            top_p,
            stop,
            device,
        )
    )


# ------------------------------------------------------------------
# Async wrapper for concurrency + clean disconnects
# ------------------------------------------------------------------
async def async_generate_stream(
    model: VesperLLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    stop: Optional[Union[str, List[str]]] = None,
    device: torch.device = torch.device("cpu"),
):
    """
    Runs the synchronous generate_stream in a thread pool and yields tokens
    asynchronously. If the client disconnects, we set a stop_event so the
    thread exits early instead of wasting GPU cycles.
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop_event = threading.Event()

    def _generate():
        try:
            for delta in generate_stream(
                model,
                tokenizer,
                prompt,
                max_new_tokens,
                temperature,
                top_p,
                stop,
                device,
                stop_event=stop_event,
            ):
                asyncio.run_coroutine_threadsafe(queue.put(("token", delta)), loop)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put(("error", exc)), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)

    future = loop.run_in_executor(INFERENCE_EXECUTOR, _generate)

    try:
        while True:
            kind, payload = await queue.get()
            if kind == "token":
                yield payload
            elif kind == "done":
                break
            elif kind == "error":
                raise payload
    except asyncio.CancelledError:
        # Client disconnected (e.g. pressed Stop). Signal the generator thread
        # to exit early so we don't keep burning GPU/CPU.
        stop_event.set()
        raise


# ------------------------------------------------------------------
# ChatML formatting
# ------------------------------------------------------------------
def format_chatml(messages: List[Dict[str, str]]) -> str:
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


# ------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------
app = FastAPI(title="VesperLLM API", version="2.0.0")

# These are populated in __main__
model: Optional[VesperLLM] = None
tokenizer: Optional[AutoTokenizer] = None
device: Optional[torch.device] = None
INFERENCE_EXECUTOR: Optional[ThreadPoolExecutor] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "vesper"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=256, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class CompletionRequest(BaseModel):
    model: Optional[str] = "vesper"
    prompt: str
    max_tokens: Optional[int] = Field(default=256, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class TokenizeRequest(BaseModel):
    text: str


@app.get("/health")
async def health():
    return {"status": "ok", "model": "vesper"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "vesper",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user",
            }
        ],
    }


@app.post("/tokenize")
async def tokenize(request: TokenizeRequest):
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    tokens = tokenizer.encode(request.text)
    return {"tokens": tokens, "count": len(tokens)}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = format_chatml(request.messages)

    if request.stream:
        async def event_stream():
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': request.model or 'vesper', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

            try:
                async for delta in async_generate_stream(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                    device=device,
                ):
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model or "vesper",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
            except asyncio.CancelledError:
                # Client disconnected; bail out cleanly.
                raise

            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': request.model or 'vesper', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    else:
        text = await asyncio.get_event_loop().run_in_executor(
            INFERENCE_EXECUTOR,
            generate_non_stream,
            model,
            tokenizer,
            prompt,
            request.max_tokens,
            request.temperature,
            request.top_p,
            request.stop,
            device,
        )
        prompt_tokens = len(tokenizer.encode(prompt))
        completion_tokens = len(tokenizer.encode(text))
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model or "vesper",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.stream:
        async def event_stream():
            completion_id = f"cmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            try:
                async for delta in async_generate_stream(
                    model,
                    tokenizer,
                    request.prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                    device=device,
                ):
                    payload = {
                        "id": completion_id,
                        "object": "text_completion.chunk",
                        "created": created,
                        "model": request.model or "vesper",
                        "choices": [
                            {
                                "index": 0,
                                "text": delta,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
            except asyncio.CancelledError:
                raise

            yield f"data: {json.dumps({'id': completion_id, 'object': 'text_completion.chunk', 'created': created, 'model': request.model or 'vesper', 'choices': [{'index': 0, 'text': '', 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    else:
        text = await asyncio.get_event_loop().run_in_executor(
            INFERENCE_EXECUTOR,
            generate_non_stream,
            model,
            tokenizer,
            request.prompt,
            request.max_tokens,
            request.temperature,
            request.top_p,
            request.stop,
            device,
        )
        prompt_tokens = len(tokenizer.encode(request.prompt))
        completion_tokens = len(tokenizer.encode(text))
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model or "vesper",
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VesperLLM OpenAI-compatible inference server")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # 1. Device
    device = detect_device(args.device)

    # 2. Paths
    ckpt_file = resolve_checkpoint_file(Path(args.checkpoint))
    logger.info(f"Loading checkpoint: {ckpt_file}")

    tokenizer_dir = find_tokenizer_dir(Path(args.checkpoint))
    logger.info(f"Loading tokenizer from: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load checkpoint
    logger.info("Loading checkpoint into CPU memory...")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    model_config = ckpt.get("model_config", {})
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", None))
    if state_dict is None:
        raise KeyError("Checkpoint missing 'model_state_dict' or 'model' key.")

    arch_keys = [
        "dim", "n_layers", "n_heads", "n_kv_heads",
        "hidden_dim", "num_experts", "top_k", "max_seq_len",
    ]
    arch_config = {k: v for k, v in model_config.items() if k in arch_keys}
    vocab_size = model_config.get("vocab_size", len(tokenizer))
    pad_id = model_config.get("pad_id", tokenizer.pad_token_id)

    logger.info(f"Architecture: {arch_config}")
    logger.info(f"vocab_size={vocab_size}, pad_id={pad_id}")

    # 4. Build model
    model = VesperLLM(vocab_size=vocab_size, pad_id=pad_id, **arch_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    model = model.to(device).eval()

    # 5. Optional compile
    if hasattr(torch, "compile") and device.type == "cuda":
        logger.info("Compiling model with torch.compile(mode='reduce-overhead')...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    # 6. Executor: 1 worker for GPU (serializes kernels, avoids context thrash),
    #    2 workers for CPU to allow modest concurrency.
    max_workers = 1 if device.type == "cuda" else 2
    INFERENCE_EXECUTOR = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="infer")
    logger.info(f"Inference thread pool: {max_workers} worker(s)")

    logger.info(f"Ready at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
