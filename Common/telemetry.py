"""Drive-sensor telemetry sink (SUBSYSTEMS.md gate E0). Logging only.

One JSONL line per sensor event, append-only, in the Hippocampus/session_log.py
conventions: `{"ts": <epoch>, "type": "...", ...}` records, flush + fsync per
write. Nothing reads this file yet and nothing here actuates — E0 is
observation for the drives layer (E1+), not control.

OFF by default so hot paths stay clean; enable with VESPER_TELEMETRY=1. The
module is pure stdlib: no torch, no GPU, and every log_* helper is a no-op
unless the flag is set. A sink failure is warned to stderr and swallowed —
telemetry must never take down the process that emits it.

Env:
  VESPER_TELEMETRY              "1"/"true"/"yes"/"on" enables the sink
  VESPER_TELEMETRY_DIR          sink directory (default <repo>/logs/telemetry)
  VESPER_TELEMETRY_MAX_BYTES    rotate the active file at this size (8 MiB)
  VESPER_TELEMETRY_KEEP         rotated files kept, oldest pruned (4)

Event types (the E0 signal set):
  immune_verdict   admission verdict from a canary-gate decision
                   (Immune/gate.py, or the gate call inside consolidate())
  reject_mass      per scored Immune probe batch: mean per-probe failure
                   mass, 1 - mean(score). For the binary scorers this is the
                   failed fraction of the batch; it is the batch-level
                   stand-in for the curiosity drive's reject-option mass
                   until the router exposes a true reject option.
  val_nll          validation NLL whenever a val runs (pretrain trainer)
  feedback         one explicit user feedback mark (session_log.mark_feedback)
  consolidation    one Hippocampus consolidation pass (consolidate())
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

_ENV_ENABLE = "VESPER_TELEMETRY"
_ENV_DIR = "VESPER_TELEMETRY_DIR"
_ENV_MAX_BYTES = "VESPER_TELEMETRY_MAX_BYTES"
_ENV_KEEP = "VESPER_TELEMETRY_KEEP"

DEFAULT_MAX_BYTES = 8 * 1024 * 1024
DEFAULT_KEEP = 4
ACTIVE_NAME = "telemetry.jsonl"

_TRUTHY = ("1", "true", "yes", "on")


def enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in _TRUTHY


def sink_dir() -> str:
    d = os.environ.get(_ENV_DIR)
    if d:
        return os.path.abspath(d)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo, "logs", "telemetry")


def emit(event_type: str, **fields: Any) -> None:
    """Append one JSONL event. No-op unless telemetry is enabled."""
    if not enabled():
        return
    record: Dict[str, Any] = {"ts": time.time(), "type": event_type}
    record.update(fields)
    try:
        _append(sink_dir(), record)
    except Exception as e:  # telemetry must never break the emitter
        print(f"[telemetry] append failed ({e!r}); event dropped", flush=True)


def _append(directory: str, record: Dict[str, Any]) -> None:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, ACTIVE_NAME)
    _maybe_rotate(path, directory)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _maybe_rotate(path: str, directory: str) -> None:
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    max_bytes = int(os.environ.get(_ENV_MAX_BYTES, DEFAULT_MAX_BYTES))
    if max_bytes <= 0 or size < max_bytes:
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rotated = os.path.join(directory, f"telemetry-{stamp}.jsonl")
    n = 1
    while os.path.exists(rotated):
        n += 1
        rotated = os.path.join(directory, f"telemetry-{stamp}-{n}.jsonl")
    os.replace(path, rotated)
    _prune(directory)


def _prune(directory: str) -> None:
    keep = int(os.environ.get(_ENV_KEEP, DEFAULT_KEEP))
    if keep <= 0:
        return
    rotated = []
    for name in os.listdir(directory):
        if name.startswith("telemetry-") and name.endswith(".jsonl"):
            rotated.append(os.path.join(directory, name))
    rotated.sort(key=lambda p: os.path.getmtime(p))
    for old in rotated[:max(0, len(rotated) - keep)]:
        try:
            os.remove(old)
        except OSError:
            pass


# ------------------------------------------------------------------
# Sensor emitters. Each is a no-op unless telemetry is enabled.
# ------------------------------------------------------------------

def log_immune_verdict(verdict: str, *, source: str, action: str = "",
                       reason: str = "", checks: Optional[Dict[str, Any]] = None,
                       reasons: Optional[list] = None,
                       aggregate: Optional[Dict[str, Any]] = None,
                       drift: Optional[Dict[str, Any]] = None,
                       metrics: Optional[Dict[str, Any]] = None,
                       **extra: Any) -> None:
    """Admission verdict from a canary-gate decision (PROMOTE/REJECT/ROLLBACK)."""
    if not enabled():
        return
    emit("immune_verdict", verdict=verdict, source=source, action=action,
         reason=reason, checks=checks or {}, reasons=reasons or [],
         aggregate=aggregate or {}, drift=drift or {}, metrics=metrics or {},
         **extra)


def log_reject_mass(reject_mass: float, *, aggregate: float, n_probes: int,
                    probe_set_sha: str = "", ckpt: str = "",
                    **extra: Any) -> None:
    """Per scored Immune probe batch: mean per-probe failure mass in [0, 1]."""
    if not enabled():
        return
    emit("reject_mass", reject_mass=round(float(reject_mass), 6),
         aggregate=round(float(aggregate), 6), n_probes=int(n_probes),
         probe_set_sha=probe_set_sha, ckpt=ckpt, **extra)


def log_val_nll(step: int, val_nll: float, *, best_val_nll: Optional[float] = None,
                prev_val_nll: Optional[float] = None, **extra: Any) -> None:
    """Validation NLL, emitted whenever a val runs (trend is read from the log)."""
    if not enabled():
        return
    fields: Dict[str, Any] = {"step": int(step), "val_nll": round(float(val_nll), 6)}
    if best_val_nll is not None:
        fields["best_val_nll"] = round(float(best_val_nll), 6)
    if prev_val_nll is not None:
        fields["prev_val_nll"] = round(float(prev_val_nll), 6)
        fields["delta"] = round(float(val_nll) - float(prev_val_nll), 6)
    emit("val_nll", **fields, **extra)


def log_feedback(session_id: str, turn_id: int, mark: str, confidence: float,
                 has_correction: bool = False, **extra: Any) -> None:
    """One explicit user feedback mark (approve/reject/neutral)."""
    if not enabled():
        return
    emit("feedback", session_id=session_id, turn_id=int(turn_id), mark=mark,
         confidence=round(float(confidence), 4), has_correction=bool(has_correction),
         **extra)


def log_consolidation(decision: str, *, user_id: str, n_triples: int,
                      **extra: Any) -> None:
    """One Hippocampus consolidation pass, logged at completion."""
    if not enabled():
        return
    emit("consolidation", decision=decision, user_id=user_id,
         n_triples=int(n_triples), **extra)
