"""Structured step logging: one JSON object per line (JSON Lines).

JSONL over a single JSON array because a crashed run still leaves a readable
file — every completed line is valid on its own. Summaries, not full payloads:
full drafts would bloat the log and leak the user's text into logs.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any


def _summ(value: Any, limit: int = 160) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


class StepLogger:
    def __init__(self, log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = os.path.join(log_dir, f"run-{stamp}.jsonl")

    def _write(self, record: dict[str, Any]) -> None:
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        # A compact mirror to stdout so the demo shows live progress.
        step = record.get("step", record.get("event", "?"))
        err = f"  ERROR={record['error']}" if record.get("error") else ""
        print(f"[{record.get('iteration', '-')}] {step:<9} "
              f"{record.get('output_summary', record.get('detail', ''))}{err}")

    def step(self, iteration: int, step: str, input_data: Any, output_data: Any,
             started: float, error: str | None = None) -> None:
        """The Milestone 3 contract: ts, iteration, step, in/out summary, latency, error."""
        self._write({
            "iteration": iteration,
            "step": step,
            "input_summary": _summ(input_data),
            "output_summary": _summ(output_data),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": error,
        })

    def event(self, event: str, detail: str) -> None:
        """Non-step events: retries, fallbacks, budget warnings, stop reasons."""
        self._write({"event": event, "detail": _summ(detail, 300)})
