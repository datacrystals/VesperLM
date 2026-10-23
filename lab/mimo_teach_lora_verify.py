#!/usr/bin/env python3
"""mimo_teach_lora_verify — standalone attach/recall/unlearn battery for a
PROMOTED final-block LoRA teach delta (design: lab/MIMO_TEACH_HOOKS.md
"Self-verification battery").  Runs in its own process (that IS the
restart-retention leg), never touches the resident vdemo, never mutates
host weights on disk.

Battery:
  (i)   process restart -> delta.pt retained on disk and attachable
        (loads + attaches in this fresh process);
  (ii)  with the adapter attached (in-memory, additive, reversible) the
        taught recall question answers in the taught class;
  (iii) without the adapter the same question does not;
  (iv)  unlearn removes it and byte-verifies clean: BASEWIN at-rest CE
        after detach+removal equals the never-attached reference read in
        this process, and the delta file bytes are hash-identical across
        the remove/restore cycle.

The promoted delta is restored byte-identical after the unlearn leg (the
leg verifies the unlearn MACHINERY; the slot stays PROMOTED).

Usage: python mimo_teach_lora_verify.py <user>
Env: HIP_VISIBLE_DEVICES set by the caller; TEACH_LAYER (47).
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/tliao/mimo_flash_lab")
import mimo_full_host as fh  # noqa: E402
import mimo_flash_host as mh  # noqa: E402
import mimo_teach_lora as tl  # noqa: E402

USER = sys.argv[1]
LAYER = int(os.environ.get("TEACH_LAYER", "47"))
ROOT = "/home/tliao/mimo_flash/teach_engines"
CLAIM_DIR = "/data/hackathon/.tmp/mr_gpu_claims"
GPU_INDEX = os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0].strip()
CLAIM_PATH = os.path.join(CLAIM_DIR, f"gpu{GPU_INDEX}")
RECALL_Q = "What is my favorite show?"
TAUGHT_CLASS = "person of interest"
MAX_NEW = 48


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class Claim:
    def __init__(self):
        self.held = False

    def take(self) -> None:
        os.makedirs(CLAIM_DIR, exist_ok=True)
        if os.path.exists(CLAIM_PATH):
            log(f"ABORT: {CLAIM_PATH} claimed by another tenant")
            sys.exit(2)
        with open(CLAIM_PATH, "w") as f:
            f.write(f"pid={os.getpid()} user={os.environ.get('USER','?')}"
                    f"@{socket.gethostname()} prog=mimo_teach_lora_verify "
                    f"started={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.held = True
        log(f"CLAIM ok: {CLAIM_PATH}")

    def drop(self) -> None:
        if self.held and os.path.exists(CLAIM_PATH):
            try:
                if f"pid={os.getpid()}" in open(CLAIM_PATH).read():
                    os.remove(CLAIM_PATH)
                    log(f"CLAIM released: {CLAIM_PATH}")
            except OSError:
                pass
        self.held = False


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    claim = Claim()
    result = {"user": USER, "layer": LAYER,
              "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        claim.take()
        udir = os.path.join(ROOT, USER)
        inc = os.path.join(udir, "delta.pt")

        # (i) restart retention: the file must exist and load+attach here.
        if not os.path.exists(inc):
            log(f"FAIL(i): no promoted delta at {inc}")
            result["restart_retained"] = False
            print(json.dumps(result, indent=1))
            sys.exit(5)
        disk_hash = sha256(inc)
        delta = torch.load(inc, map_location="cpu", weights_only=False)
        assert delta.get("format") == tl.DELTA_FORMAT, delta.get("format")
        log(f"(i) delta retained: {os.path.getsize(inc)} bytes "
            f"sha256={disk_hash[:16]}... format ok")
        result["restart_retained"] = True

        model, tok, cache = fh.build_full_host(fh.MODEL_DIR, "cuda",
                                               18.0, {})

        def ce_basewin() -> float:
            ids = tok.encode(tl.BASEWIN, add_special_tokens=False)[:65]
            x = torch.tensor([ids], dtype=torch.long, device="cuda")
            with torch.no_grad():
                o = model(input_ids=x, use_cache=False)
            return float(F.cross_entropy(
                o.logits[0, 0:-1].float(), x[0, 1:]))

        def generate(question: str) -> str:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False, add_generation_prompt=True)
            ids = tok.encode(prompt, add_special_tokens=False)
            out_ids: list[int] = []
            stop = mh.stop_token_ids(tok)
            past = None
            for i in range(MAX_NEW):
                x = torch.tensor([ids if i == 0 else [out_ids[-1]]],
                                 dtype=torch.long, device="cuda")
                with torch.no_grad():
                    o = model(input_ids=x, past_key_values=past,
                              use_cache=True)
                past = o.past_key_values
                nxt = int(torch.argmax(o.logits[0, -1].float()))
                if nxt in stop:
                    break
                out_ids.append(nxt)
            return mh._clean_reply(tok.decode(out_ids,
                                              skip_special_tokens=True))

        # never-attached reference (at-rest byte-check anchor)
        ce_ref = ce_basewin()
        log(f"BASEWIN never-attached reference CE = {ce_ref:.6f}")
        result["ce_ref"] = ce_ref

        # (iii) without the adapter
        reply_off = generate(RECALL_Q)
        off_hit = TAUGHT_CLASS in reply_off.lower()
        log(f"(iii) adapter OFF recall: taught-class hit = {off_hit}")
        log(f"  reply: {reply_off[:200]!r}")
        result["recall_off_hit"] = off_hit
        result["recall_off_reply"] = reply_off[:200]

        # attach in-memory (additive, reversible)
        handle = tl.attach_lora(model, delta, LAYER)
        result["attachable"] = True
        ce_on = ce_basewin()
        log(f"attached in-memory; BASEWIN CE attached = {ce_on:.6f} "
            f"(drift {(ce_on - ce_ref) * 65:+.2f} nats/65tok)")
        result["ce_attached"] = ce_on

        # (ii) with the adapter
        reply_on = generate(RECALL_Q)
        on_hit = TAUGHT_CLASS in reply_on.lower()
        log(f"(ii) adapter ON recall: taught-class hit = {on_hit}")
        log(f"  reply: {reply_on[:200]!r}")
        result["recall_on_hit"] = on_hit
        result["recall_on_reply"] = reply_on[:200]

        # (iv) unlearn: detach, remove the file, byte-verify, restore.
        tl.detach_lora(handle)
        ce_detached = ce_basewin()
        bak = inc + ".verifybak"
        os.replace(inc, bak)
        removed = not os.path.exists(inc)
        ce_removed = ce_basewin()
        os.replace(bak, inc)
        restored_hash = sha256(inc)
        byte_clean = (ce_detached == ce_ref) and (ce_removed == ce_ref)
        file_clean = restored_hash == disk_hash
        log(f"(iv) unlearn: detach CE {ce_detached:.6f} == ref "
            f"{ce_detached == ce_ref}; file removed {removed}; "
            f"at-rest CE after removal {ce_removed:.6f} == ref "
            f"{ce_removed == ce_ref}; restored bytes identical "
            f"{file_clean}")
        result["unlearn"] = {
            "detach_ce_equals_ref": ce_detached == ce_ref,
            "file_removed": removed,
            "at_rest_ce_equals_ref": ce_removed == ce_ref,
            "restored_bytes_identical": file_clean,
        }
        result["byte_clean"] = byte_clean and file_clean

        verdict_ii = on_hit and not off_hit
        result["recall_contrast"] = verdict_ii
        log(f"BATTERY SUMMARY: restart={result['restart_retained']} "
            f"on={on_hit} off={off_hit} contrast={verdict_ii} "
            f"byte_clean={result['byte_clean']}")
        out_path = os.path.join(udir, "verify_lora.json")
        json.dump(result, open(out_path, "w"), indent=1)
        log(f"written -> {out_path}")
        print("VERIFY_DONE", flush=True)
        claim.drop()
        sys.exit(0 if (result["restart_retained"] and result["byte_clean"])
                 else 6)
    finally:
        claim.drop()


if __name__ == "__main__":
    main()
