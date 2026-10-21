#!/usr/bin/env python3
"""mimo_demo_chat — live interactive REPL for the persona pack on the
faithful MiMo-Flash lazy host.

Purpose: first human-facing session with the persona-injected host.
Loads the host once ("host warming…"), installs all six wraps +
six sentinel24 rows exactly per the committed pack matrix recipe
(lab/mimo_pack_matrix.py, lab/MIMO_MEMBERSHIP_THROTTLE.md):

  * wraps injected at MoE sites {0,23,46} at each record's fitted lam_i;
  * sentinel rows at model-layer-24 post-attn-norm;
  * per prompt: shadow prefill (flag computed BEFORE the real injected
    prefill), winner-take-all latch at margin 10 on the max content-token
    score; latch held for the whole answer; nothing latched => injection
    off and the turn reads noplug-class.

Per-turn stateless (each turn is its own prefill, matching the verified
matrix arms — no conversation history is carried).

REPL commands:
  /greedy          greedy decode (default)
  /sample,T        sample at temperature T (e.g. /sample,0.8)
  /exit            stop cleanly (releases the GPU claim)

While idle it prints one heartbeat line every ~30s with its claim state
(and refreshes the claim file mtime so nothing sees it as stale).

Remote stop: `touch ~/mimo_flash/VDEMO_STOP` — the loop notices within
~30s (sooner between turns), releases the claim and exits.

GPU claim discipline: claims the GPU named by HIP_VISIBLE_DEVICES
through /data/hackathon/.tmp/mr_gpu_claims/, refuses to start if that
GPU already carries a claim, removes its claim file on any clean exit.

Launch convention (box sys-4029gp-trt):
  HIP_VISIBLE_DEVICES=1 MIMO_PLUG_DIR=$HOME/mimo_flash/plugs_matrix_v2 \
  ROCBLAS_TENSILE_LIBPATH=/opt/rocm-6.2.4/lib/rocblas/library \
  PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
  /home/tliao/vesper_sft_venv/bin/python mimo_demo_chat.py \
      >> /home/tliao/mimo_flash/vdemo.log 2>&1
"""
from __future__ import annotations

import atexit
import os
import select
import signal
import socket
import sys
import time

import torch

sys.path.insert(0, "/data/tliao/mimo_flash_lab")

import mimo_full_host as fh  # noqa: E402
import mimo_flash_host as mh  # noqa: E402
from mimo_pack_matrix import (  # noqa: E402  (committed harness reuse)
    Pack,
    prompt_ids,
    TAGS,
    SEN_MARGIN,
    SITES,
)

CLAIM_DIR = "/data/hackathon/.tmp/mr_gpu_claims"
STOP_FILE = os.environ.get("VDEMO_STOP_FILE",
                           "/home/tliao/mimo_flash/VDEMO_STOP")
MAX_TOK = int(os.environ.get("VDEMO_MAX_TOK", "24"))
HEARTBEAT_S = float(os.environ.get("VDEMO_HEARTBEAT_S", "30"))
CACHE_GB = float(os.environ.get("VDEMO_CACHE_GB", "18.0"))

GPU_INDEX = os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0].strip()
CLAIM_PATH = os.path.join(CLAIM_DIR, f"gpu{GPU_INDEX}")

_state = {"claim_held": False}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def claim_gpu() -> None:
    os.makedirs(CLAIM_DIR, exist_ok=True)
    if os.path.exists(CLAIM_PATH):
        log(f"ABORT: {CLAIM_PATH} already claimed by another tenant — "
            "not stealing it. Pick a free GPU or wait.")
        sys.exit(2)
    with open(CLAIM_PATH, "w") as f:
        f.write(f"pid={os.getpid()} user={os.environ.get('USER', '?')}"
                f"@{socket.gethostname()} prog=mimo_demo_chat (vdemo)"
                f" started={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    _state["claim_held"] = True
    log(f"CLAIM ok: {CLAIM_PATH}")


def heartbeat_claim() -> None:
    try:
        os.utime(CLAIM_PATH, None)
    except OSError:
        pass


def release_claim() -> None:
    if _state["claim_held"] and os.path.exists(CLAIM_PATH):
        # only remove if it is still ours (pid match)
        try:
            with open(CLAIM_PATH) as f:
                ours = f"pid={os.getpid()}" in f.read()
        except OSError:
            ours = False
        if ours:
            os.remove(CLAIM_PATH)
            log(f"CLAIM released: {CLAIM_PATH}")
    _state["claim_held"] = False


def _sig_handler(_sig, _frm):
    raise SystemExit(0)


atexit.register(release_claim)
signal.signal(signal.SIGTERM, _sig_handler)
signal.signal(signal.SIGINT, _sig_handler)


def main() -> None:
    # stale stop-flag clean-up (from an older session)
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)
        log(f"removed stale stop-flag {STOP_FILE}")

    claim_gpu()

    plug_dir = os.environ.get("MIMO_PLUG_DIR",
                              "/home/tliao/mimo_flash/plugs")
    log("== mimo_demo_chat (vdemo) header ==")
    log(f"who: {os.environ.get('USER', '?')}@{socket.gethostname()} "
        f"pid={os.getpid()}")
    log(f"claim: {CLAIM_PATH} (heartbeat every ~{HEARTBEAT_S:.0f}s)")
    log(f"host warming… model={fh.MODEL_DIR} lazy cache={CACHE_GB} GiB "
        f"device=cuda (HIP_VISIBLE_DEVICES={GPU_INDEX})")
    t0 = time.time()
    model, tok, _cache = fh.build_full_host(
        fh.MODEL_DIR, "cuda", CACHE_GB, {})
    blocks = mh.find_mimo_moe_blocks(model)
    blk23 = model.model.layers[24]
    hs = {}

    def hook(_m, _i, out):
        hs["h"] = (out[0] if isinstance(out, tuple) else out)
    blk23.post_attention_layernorm.register_forward_hook(hook)

    pack = Pack(model, blocks, TAGS)   # installs six wraps + sentinels
    pack.arm(TAGS)
    lam = {t: round(float(getattr(pack.wraps[t], "lam_i", 0.0)), 4)
           for t in TAGS}

    log(f"host warm in {time.time() - t0:.1f}s")
    log(f"plugs mounted (6): {TAGS} from MIMO_PLUG_DIR={plug_dir}")
    log(f"sentinels (6): sentinel24_* @ model-layer-24 post-attn-norm; "
        f"sites={SITES} lam_i={lam}")
    log(f"knobs: mode=greedy max_tok={MAX_TOK} margin={SEN_MARGIN} "
        f"heartbeat_s={HEARTBEAT_S:.0f} cache_gb={CACHE_GB}")
    log("exit: type /exit or Ctrl-C; remote clean stop: "
        f"`touch {STOP_FILE}`")
    log("commands: /greedy | /sample,T | /exit")
    print("VDEMO_READY", flush=True)

    def latch_tag(ids):
        hs.clear()
        pack.state["tag"] = None
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], device="cuda"),
                  use_cache=False)
        h = hs["h"].float().reshape(-1, 4096)
        best = None
        best_r = None
        for tag in pack.tags:
            r = float((h * pack.sentinels[tag].unsqueeze(0))
                      .sum(-1)[3:].max())
            if r > SEN_MARGIN and (best is None or r > best_r):
                best, best_r = tag, r
        return best, best_r

    def gen(text, mode, temp):
        ids = prompt_ids(tok, text)
        tag, score = latch_tag(ids)
        pack.state["tag"] = tag
        heartbeat_claim()
        past, outs = None, []
        t_g0 = time.time()
        with torch.no_grad():
            for i in range(MAX_TOK):
                x = torch.tensor([ids] if i == 0 else [[outs[-1]]],
                                 dtype=torch.long, device="cuda")
                o = model(input_ids=x, past_key_values=past,
                          use_cache=True)
                past = o.past_key_values
                lg = o.logits[0, -1].float()
                if mode == "sample":
                    probs = torch.softmax(lg / max(temp, 1e-4), dim=-1)
                    nxt = int(torch.multinomial(probs, 1).item())
                else:
                    nxt = int(lg.argmax().item())
                if nxt == tok.eos_token_id:
                    break
                outs.append(nxt)
        pack.state["tag"] = None
        return tok.decode(outs), tag, score, time.time() - t_g0

    mode, temp = "greedy", 1.0
    turns = 0
    started = time.monotonic()
    last_hb = time.monotonic()
    log("REPL ready — type a turn, one at a time.")

    while True:
        if os.path.exists(STOP_FILE):
            try:
                os.remove(STOP_FILE)
            except OSError:
                pass
            log(f"stop-flag {STOP_FILE} seen — clean stop.")
            break
        try:
            r, _w, _x = select.select([sys.stdin], [], [], HEARTBEAT_S)
        except (OSError, ValueError):
            r = []
        if not r:
            age = int(time.monotonic() - started)
            idle = int(time.monotonic() - last_hb)
            log(f"hb claim gpu{GPU_INDEX} held age={age}s idle={idle}s "
                f"turns={turns} mode={mode}"
                + (f"(T={temp})" if mode == "sample" else "")
                + f" stop-flag=absent -> {CLAIM_PATH}")
            heartbeat_claim()
            last_hb = time.monotonic()
            continue
        line = sys.stdin.readline()
        if line == "":          # stdin closed (pane died)
            log("stdin EOF — clean stop.")
            break
        line = line.strip()
        if not line:
            continue
        if line == "/exit":
            log("/exit — clean stop.")
            break
        if line.startswith("/greedy"):
            mode, temp = "greedy", 1.0
            print("MODE greedy", flush=True)
            continue
        if line.startswith("/sample"):
            parts = line.replace(",", " ").split()
            try:
                temp = float(parts[1])
                mode = "sample"
                print(f"MODE sample T={temp}", flush=True)
            except (IndexError, ValueError):
                print("usage: /sample,T (e.g. /sample,0.8)", flush=True)
            continue
        if line.startswith("/"):
            print("unknown command; /greedy | /sample,T | /exit",
                  flush=True)
            continue

        turns += 1
        t_s = time.time()
        try:
            text, tag, score, t_gen = gen(line, mode, temp)
        except Exception as e:  # keep the session alive on a bad turn
            log(f"GEN-ERROR turn={turns}: {type(e).__name__}: {e}")
            continue
        latch = (f"{tag}({score:.1f})" if tag is not None else "None")
        print(f"YOU[{turns}] {line}", flush=True)
        print(f"LATCH {latch} | gen={t_gen:.1f}s | mode={mode}",
              flush=True)
        print(f"ASSISTANT {' ⏎ '.join(text.strip().splitlines())}",
              flush=True)
        log(f"turn={turns} done in {time.time() - t_s:.1f}s "
            f"latch={latch}")
        heartbeat_claim()
        last_hb = time.monotonic()

    pack.restore()
    log(f"DEMO_DONE turns={turns}")


if __name__ == "__main__":
    main()
