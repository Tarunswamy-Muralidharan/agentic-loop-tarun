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

    DONE = "DONE"          # validation threshold met
    REFUSED = "REFUSED"    # the agent judged the input not tabulatable (its call)
    PARTIAL = "PARTIAL"    # max iterations / token budget hit; best table returned
    STUCK = "STUCK"        # identical reflections twice in a row
    FAILED = "FAILED"      # unrecoverable error (e.g. auth) after retries


@dataclass
class AgentState:
    """Everything the loop knows. Owned by run(); steps receive views of it."""

    goal: str
    source: str                     # the messy input text, never mutated
    iteration: int = 0
    table: dict[str, Any] | None = None            # {"columns": [...], "rows": [...]}
    last_scores: dict[str, float] | None = None    # per-dimension validation scores
    weighted_total: float | None = None            # 0..5 weighted validation score
    defects: list[str] = field(default_factory=list)
    missed_facts: list[str] = field(default_factory=list)
    best_table: dict[str, Any] | None = None
    best_total: float = -1.0
    next_instruction: str = "First iteration: nothing has happened yet."
    verdict: str | None = None      # the agent's own explanation when it finishes
    history: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    status: Status | None = None


@dataclass
class Percept:
    """What this iteration is allowed to see. Built by perceive() only."""

    goal: str
    source: str
    iteration: int
    table: dict[str, Any] | None
    last_scores: dict[str, float] | None
    weighted_total: float | None
    defects: list[str]
    missed_facts: list[str]
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
    quality_score: float            # weighted validation total, 0..5 (-1 if unknown)
    next_instruction: str
    summary: str                    # one line; also the stuck-detection fingerprint
    memory_record: dict[str, Any] | None = None   # what save() persists
