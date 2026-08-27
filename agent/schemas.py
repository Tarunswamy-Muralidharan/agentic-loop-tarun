"""Typed shapes passed between the four loop steps.

Each dataclass is the contract between two steps:
    perceive -> Percept -> reason -> Decision -> act -> Observation -> reflect -> Reflection

Keeping these explicit is what makes the four steps provably distinct:
a step can only consume what the previous step chose to expose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    """Terminal status of a run. An explicit status beats returning None."""

    DONE = "DONE"          # rubric threshold met
    PARTIAL = "PARTIAL"    # max iterations / token budget hit; best draft returned
    STUCK = "STUCK"        # identical reflections twice in a row
    FAILED = "FAILED"      # unrecoverable error (e.g. auth) after retries


@dataclass
class AgentState:
    """Everything the loop knows. Owned by run(); steps receive views of it."""

    goal: str
    draft: str                      # current working text
    original: str                   # untouched input, for the final comparison
    iteration: int = 0
    last_scores: dict[str, int] | None = None       # per-criterion, last scoring
    weighted_total: float | None = None             # 0..5 weighted score
    best_draft: str | None = None
    best_total: float = -1.0
    critiques: dict[str, str] = field(default_factory=dict)
    next_instruction: str = "No score yet. Score the draft first."
    history: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    status: Status | None = None


@dataclass
class Percept:
    """What this iteration is allowed to see. Built by perceive() only."""

    goal: str
    draft: str
    iteration: int
    last_scores: dict[str, int] | None
    weighted_total: float | None
    critiques: dict[str, str]
    next_instruction: str
    memory_notes: list[str]         # recalled from long-term memory


@dataclass
class Decision:
    """What reason() decided to do. Exactly one of tool_call / final."""

    kind: str                       # "tool_call" | "final"
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    final_text: str | None = None
    reasoning: str | None = None    # the model's own trace, for the log


@dataclass
class Observation:
    """What actually happened when act() executed the decision."""

    ok: bool
    content: Any = None
    error: str | None = None


@dataclass
class Reflection:
    """reflect()'s judgement: continue or stop, and why."""

    is_done: bool
    quality_score: float            # weighted rubric total, 0..5 (-1 if unknown)
    next_instruction: str
    summary: str                    # one line; also the stuck-detection fingerprint
    memory_record: dict[str, Any] | None = None   # what save() persists
