"""Tool definitions (JSON Schema dicts) and their Python handlers.

The split that matters: the model only ever sees TOOL_SCHEMAS (name +
description + parameter schema). The handlers below are plain Python that
*we* execute after the model requests a call. The model never runs code.

Both handlers need an LLM themselves (scoring and revising are language
tasks), so build_registry() injects the client instead of importing it —
that keeps this module importable with no network and makes the handlers
trivially testable with a fake.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from agent import prompts
from agent.harness import parse_json_defensively

# ------------------------------------------------------------- the rubric ---
# The rubric is data, not prose: it drives the scoring prompt, the weighted
# total, and the "weakest criterion" pick. Changing the rubric is a data edit.

RUBRIC: list[dict[str, Any]] = [
    {"name": "clarity",   "weight": 0.35, "description": "A reader gets the point on first pass; no ambiguous or muddled sentences."},
    {"name": "evidence",  "weight": 0.30, "description": "Claims are backed by something concrete: a number, an example, a fact."},
    {"name": "concision", "weight": 0.20, "description": "No filler words, no repetition, nothing that could be cut without loss."},
    {"name": "tone",      "weight": 0.15, "description": "Professional and confident; no hype words like 'very', 'amazing', 'best ever'."},
]

CRITERIA = [c["name"] for c in RUBRIC]


def weighted_total(scores: dict[str, int]) -> float:
    """Weighted rubric total on the 0-5 scale."""
    return round(sum(c["weight"] * scores[c["name"]] for c in RUBRIC), 2)


def weakest_criterion(scores: dict[str, int]) -> str:
    """Lowest weighted contribution loses; ties break toward higher weight."""
    return min(RUBRIC, key=lambda c: (scores[c["name"]] * c["weight"], -c["weight"]))["name"]


# --------------------------------------------------------------- schemas ----
# Handwritten JSON Schema — this is what a @tool decorator was generating.
# The description fields are prompt engineering: they are the ONLY basis the
# model has for choosing when and how to call each tool.

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "score_against_rubric",
            "description": (
                "Score the current draft 1-5 on each rubric criterion "
                "(clarity, evidence, concision, tone) and return per-criterion "
                "critiques plus a weighted total. Call this before any revision, "
                "and again after every revision to measure the effect."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The full draft to score."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "revise_text",
            "description": (
                "Rewrite the draft to fix exactly one named rubric criterion, "
                "guided by the judge's critique. Changes as little else as "
                "possible. Returns the full revised text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The full current draft."},
                    "criterion": {
                        "type": "string",
                        "enum": CRITERIA,
                        "description": "The single weakest criterion to fix.",
                    },
                    "guidance": {
                        "type": "string",
                        "description": "The judge's critique for that criterion, verbatim.",
                    },
                },
                "required": ["text", "criterion", "guidance"],
            },
        },
    },
]


# -------------------------------------------------------------- handlers ----

def build_registry(llm: Any) -> dict[str, Callable[..., dict[str, Any]]]:
    """Return {tool_name: handler}. `llm` is anything with .chat(...)."""

    def score_against_rubric(text: str) -> dict[str, Any]:
        """Judge call with strict-JSON output; one simplified retry on bad JSON."""
        for system in (prompts.SCORE_SYSTEM, prompts.SCORE_SYSTEM_SIMPLIFIED):
            msg, _usage = llm.chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompts.score_user(text, RUBRIC)},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,          # a judge must be deterministic
            )
            data = parse_json_defensively(msg["content"] or "")
            if data and _valid_scores(data):
                scores = {k: int(data["scores"][k]) for k in CRITERIA}
                return {
                    "scores": scores,
                    "critiques": {k: str(data["critiques"][k]) for k in CRITERIA},
                    "weighted_total": weighted_total(scores),
                    "weakest_criterion": weakest_criterion(scores),
                }
        raise ValueError("scorer returned unparseable JSON twice (simplified retry included)")

    def revise_text(text: str, criterion: str, guidance: str) -> dict[str, Any]:
        if criterion not in CRITERIA:                 # validate args, don't trust the model
            raise ValueError(f"unknown criterion {criterion!r}; expected one of {CRITERIA}")
        msg, _usage = llm.chat(
            messages=[
                {"role": "system", "content": prompts.REVISE_SYSTEM},
                {"role": "user", "content": prompts.revise_user(text, criterion, guidance)},
            ],
            temperature=None,             # None -> harness uses revise_temperature
        )
        revised = (msg["content"] or "").strip()
        if not revised:
            raise ValueError("reviser returned empty text")
        return {"revised_text": revised, "criterion": criterion}

    return {"score_against_rubric": score_against_rubric, "revise_text": revise_text}


def _valid_scores(data: dict[str, Any]) -> bool:
    """Bound what the model claims before we act on it."""
    try:
        return all(1 <= int(data["scores"][k]) <= 5 for k in CRITERIA) and all(
            k in data["critiques"] for k in CRITERIA
        )
    except (KeyError, TypeError, ValueError):
        return False
