"""Append-only session log -> feedback triples for Hippocampus consolidation.

Every interaction is one JSONL line. Nothing is ever rewritten; corrections are
new records. `to_triples()` is the only path from memory to weight-update data,
and it enforces quarantine rules:

  * explicit feedback marks (approve/reject) with confidence >= threshold
    become training triples;
  * an explicit `correction` (the user's preferred response) becomes a
    positive triple;
  * verified outcomes may produce weak triples (reward +/-0.5) only when
    `allow_outcome_derived=True`;
  * everything else (neutral marks, unverified outcomes, unmarked turns,
    tool calls) stays memory-only and never reaches the optimizer.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional

_COMMON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

import telemetry

VALID_MARKS = ("approve", "reject", "neutral")


@dataclass
class TrainingTriple:
    prompt: str
    response: str
    reward: float
    session_id: str
    turn_id: int
    source: str          # "explicit_feedback" | "explicit_correction" | "verified_outcome"
    confidence: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TurnView:
    """A turn joined with its latest feedback/outcome (read model)."""
    session_id: str
    turn_id: int
    prompt: str
    response: str
    mark: Optional[str] = None
    confidence: float = 1.0
    note: Optional[str] = None
    correction: Optional[str] = None
    outcome: Optional[Dict[str, Any]] = None
    trainable: bool = False
    quarantine_reason: Optional[str] = None


class SessionLog:
    """Append-only JSONL store for one or more conversations."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    # ---------------- recording (append-only) ----------------

    def append(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(record)
        record.setdefault("ts", time.time())
        record.setdefault("record_id", uuid.uuid4().hex[:12])
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return record

    def log_turn(self, session_id: str, turn_id: int, prompt: str, response: str,
                 role_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.append({
            "type": "turn", "session_id": session_id, "turn_id": turn_id,
            "prompt": prompt, "response": response, "meta": role_meta or {},
        })

    def log_tool_call(self, session_id: str, turn_id: int, tool: str,
                      args: Dict[str, Any], result_summary: str = "") -> Dict[str, Any]:
        return self.append({
            "type": "tool_call", "session_id": session_id, "turn_id": turn_id,
            "tool": tool, "args": args, "result_summary": result_summary,
        })

    def log_outcome(self, session_id: str, turn_id: int, success: bool,
                    verified: bool = False, detail: str = "") -> Dict[str, Any]:
        return self.append({
            "type": "outcome", "session_id": session_id, "turn_id": turn_id,
            "success": success, "verified": verified, "detail": detail,
        })

    def mark_feedback(self, session_id: str, turn_id: int, mark: str,
                      confidence: float = 1.0, note: str = "",
                      correction: Optional[str] = None) -> Dict[str, Any]:
        if mark not in VALID_MARKS:
            raise ValueError(f"mark must be one of {VALID_MARKS}, got {mark!r}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        # E0 drive telemetry (SUBSYSTEMS.md): one feedback event per mark.
        # Logging only; the session log below remains the memory of record.
        telemetry.log_feedback(session_id, turn_id, mark, confidence,
                               has_correction=correction is not None)
        return self.append({
            "type": "feedback", "session_id": session_id, "turn_id": turn_id,
            "mark": mark, "confidence": confidence, "note": note,
            "correction": correction,
        })

    # ---------------- reading ----------------

    def read_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def _turn_views(self) -> List[TurnView]:
        turns: Dict[tuple, TurnView] = {}
        order: List[tuple] = []
        for rec in self.read_all():
            key = (rec.get("session_id"), rec.get("turn_id"))
            if rec.get("type") == "turn":
                tv = TurnView(session_id=key[0], turn_id=key[1],
                              prompt=rec.get("prompt", ""),
                              response=rec.get("response", ""))
                turns[key] = tv
                order.append(key)
            elif rec.get("type") == "outcome" and key in turns:
                turns[key].outcome = {
                    "success": rec.get("success"),
                    "verified": rec.get("verified", False),
                    "detail": rec.get("detail", ""),
                }
            elif rec.get("type") == "feedback" and key in turns:
                tv = turns[key]
                tv.mark = rec.get("mark")           # latest mark wins
                tv.confidence = float(rec.get("confidence", 1.0))
                tv.note = rec.get("note", "")
                tv.correction = rec.get("correction")
        return [turns[k] for k in order]

    def to_triples(self, min_confidence: float = 0.8,
                   allow_outcome_derived: bool = True) -> List[TrainingTriple]:
        """Extract weight-update triples under the quarantine rules.

        Turns without qualifying feedback never appear here; they remain in
        the JSONL as memory-only history (see `quarantined_views()`).
        """
        triples: List[TrainingTriple] = []
        for tv in self._turn_views():
            if tv.mark in ("approve", "reject"):
                if tv.confidence < min_confidence:
                    continue  # low-confidence explicit feedback: quarantine
                reward = 1.0 if tv.mark == "approve" else -1.0
                reward *= tv.confidence
                triples.append(TrainingTriple(
                    tv.prompt, tv.response, reward, tv.session_id, tv.turn_id,
                    "explicit_feedback", tv.confidence))
                if tv.mark == "reject" and tv.correction:
                    # the user's preferred rewrite is a positive example
                    triples.append(TrainingTriple(
                        tv.prompt, tv.correction, min(1.0, tv.confidence),
                        tv.session_id, tv.turn_id,
                        "explicit_correction", tv.confidence))
                elif tv.mark == "approve" and tv.correction:
                    triples.append(TrainingTriple(
                        tv.prompt, tv.correction, min(1.0, tv.confidence),
                        tv.session_id, tv.turn_id,
                        "explicit_correction", tv.confidence))
                continue
            if (allow_outcome_derived and tv.outcome and tv.outcome.get("verified")
                    and tv.mark is None):
                reward = 0.5 if tv.outcome.get("success") else -0.5
                triples.append(TrainingTriple(
                    tv.prompt, tv.response, reward, tv.session_id, tv.turn_id,
                    "verified_outcome", 1.0))
        return triples

    def quarantined_views(self, min_confidence: float = 0.8,
                          allow_outcome_derived: bool = True) -> List[TurnView]:
        """Turns kept as memory only (not eligible for weight updates).

        Mirrors `to_triples()` eligibility exactly.
        """
        out = []
        for tv in self._turn_views():
            if tv.mark in ("approve", "reject"):
                if tv.confidence >= min_confidence:
                    tv.trainable, tv.quarantine_reason = True, "eligible"
                else:
                    tv.trainable, tv.quarantine_reason = False, "low_confidence"
            elif tv.mark == "neutral":
                tv.trainable, tv.quarantine_reason = False, "neutral_mark"
            elif (allow_outcome_derived and tv.outcome and tv.outcome.get("verified")):
                tv.trainable, tv.quarantine_reason = True, "eligible_verified_outcome"
            else:
                tv.trainable, tv.quarantine_reason = False, "no_explicit_feedback"
            if not tv.trainable:
                out.append(tv)
        return out

    def summary(self) -> Dict[str, Any]:
        recs = self.read_all()
        views = self._turn_views()
        return {
            "path": self.path,
            "records": len(recs),
            "turns": len(views),
            "tool_calls": sum(1 for r in recs if r.get("type") == "tool_call"),
            "feedback_marks": sum(1 for r in recs if r.get("type") == "feedback"),
            "outcomes": sum(1 for r in recs if r.get("type") == "outcome"),
        }
