"""Tool definitions (JSON Schema dicts) and their Python handlers.

The split that matters: the model only ever sees TOOL_SCHEMAS (name +
description + parameter schema). The handlers below are plain Python that
*we* execute after the model requests a call. The model never runs code.

Design decision worth defending: validation is a HYBRID. Three of the four
quality dimensions (completeness, groundedness, consistency) are pure code —
deterministic, free, and immune to a model grading its own homework. Only
coverage ("which source facts never made it into the table?") needs an LLM,
because it requires reading comprehension. Groundedness in particular is the
anti-hallucination check: every cell value must actually appear in the
source text, verified by normalized string matching, not by asking a model.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from agent import prompts
from agent.harness import parse_json_defensively

# ------------------------------------------------------ scoring dimensions --
# Weighted like a rubric; drives weighted_total and the repair loop's focus.

DIMENSIONS: list[dict[str, Any]] = [
    {"name": "groundedness", "weight": 0.35,
     "description": "every non-empty cell value appears in the source text (no invented data)"},
    {"name": "coverage", "weight": 0.30,
     "description": "every distinct fact in the source is represented in some row"},
    {"name": "completeness", "weight": 0.25,
     "description": "cells are filled; \"\" only where the source genuinely lacks the value"},
    {"name": "consistency", "weight": 0.10,
     "description": "uniform columns across rows, no duplicate rows, sane shape"},
]


def weighted_total(scores: dict[str, float]) -> float:
    return round(sum(d["weight"] * scores[d["name"]] for d in DIMENSIONS), 2)


# --------------------------------------------------------------- schemas ----
# Handwritten JSON Schema — this is what a @tool decorator was generating.
# The description fields are prompt engineering: they are the ONLY basis the
# model has for choosing when and how to call each tool.

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "extract_table",
            "description": (
                "Convert the source text into a structured table (columns + "
                "rows). Call this first, and call it again with `feedback` "
                "whenever the validator reports defects or missed facts — the "
                "feedback tells the extraction exactly what to repair."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string",
                             "description": "The full source text to tabulate."},
                    "feedback": {"type": "string",
                                 "description": "Validator defects and missed facts "
                                                "to fix. Omit on the first extraction."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Search long-term memory of past runs on this source for "
                "lessons — what was tried and what moved the score. Cheap and "
                "local; consult it whenever history might save a wasted attempt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "What you want lessons about, e.g. "
                                             "'repairs that improved coverage'."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_task",
            "description": (
                "End the run with your verdict. Use outcome 'success' when a "
                "validated score meets the target (the harness verifies this "
                "and will refuse the claim otherwise), or 'not_tabulatable' "
                "when your judgement is that the input contains no extractable "
                "facts — gibberish, empty text, a bare question, or pure "
                "opinion. Always explain the verdict."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "enum": ["success", "not_tabulatable"],
                        "description": "Your verdict on how the run ends.",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "One or two sentences justifying the verdict.",
                    },
                },
                "required": ["outcome", "explanation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_table",
            "description": (
                "Score the current table 0-5 on groundedness, coverage, "
                "completeness and consistency against the source text, and "
                "return the weighted total plus a concrete list of defects "
                "and missed facts. Must be called after every extraction — "
                "an unvalidated table has no score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The original source text."},
                },
                "required": ["text"],
            },
        },
    },
]


# ------------------------------------------------------- code validators ----

def _norm(value: str) -> str:
    """Normalize for grounding checks: lowercase, strip punctuation/spaces."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def check_grounding(table: dict[str, Any], source: str) -> tuple[float, list[str]]:
    """Fraction of non-empty cells whose value appears in the source.
    Deterministic anti-hallucination: extraction is copying, so a value the
    source never contained is an invented value."""
    ns = _norm(source)
    total, grounded, defects = 0, 0, []
    for i, row in enumerate(table.get("rows", [])):
        for col, val in row.items():
            val = str(val).strip()
            if not val:
                continue
            total += 1
            if _norm(val) and _norm(val) in ns:
                grounded += 1
            else:
                defects.append(f"row {i + 1} cell {col}={val!r} not found in source text")
    return (5.0 if total == 0 else round(5.0 * grounded / total, 2)), defects


def check_completeness(table: dict[str, Any]) -> tuple[float, list[str]]:
    rows, cols = table.get("rows", []), table.get("columns", [])
    total = len(rows) * len(cols)
    if total == 0:
        return 0.0, ["table has no rows or no columns"]
    empty = [(i + 1, c) for i, r in enumerate(rows) for c in cols
             if not str(r.get(c, "")).strip()]
    defects = [f"{len(empty)} empty cell(s): "
               + ", ".join(f"row {i} {c}" for i, c in empty[:6])] if empty else []
    return round(5.0 * (total - len(empty)) / total, 2), defects


def check_consistency(table: dict[str, Any]) -> tuple[float, list[str]]:
    rows, cols = table.get("rows", []), table.get("columns", [])
    checks, defects = [], []
    checks.append(len(cols) >= 2)
    if len(cols) < 2:
        defects.append("fewer than 2 columns")
    checks.append(len(rows) >= 1)
    if not rows:
        defects.append("no rows")
    uniform = all(set(r.keys()) == set(cols) for r in rows)
    checks.append(uniform)
    if not uniform:
        defects.append("row keys do not match declared columns")
    seen = [tuple(str(r.get(c, "")) for c in cols) for r in rows]
    nodupes = len(seen) == len(set(seen))
    checks.append(nodupes)
    if not nodupes:
        defects.append("duplicate rows present")
    return round(5.0 * sum(checks) / len(checks), 2), defects


# -------------------------------------------------------------- handlers ----

def build_registry(llm: Any, source: str,
                   memory: Any = None) -> dict[str, Callable[..., dict[str, Any]]]:
    """Return {tool_name: handler}. `llm` is anything with .chat(...).

    The registry closes over `source` and the latest extracted table so the
    validator always scores against ground truth we hold, not text the model
    re-supplies (a model could otherwise 'validate' against a source it
    paraphrased to match its own table).
    """
    latest: dict[str, Any] = {"table": None}

    def extract_table(text: str, feedback: str | None = None) -> dict[str, Any]:
        for system in (prompts.EXTRACT_SYSTEM, prompts.EXTRACT_SYSTEM_SIMPLIFIED):
            msg, _usage = llm.chat(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompts.extract_user(source, feedback)}],
                response_format={"type": "json_object"},
                temperature=0.0,          # extraction is copying, not creating
            )
            data = parse_json_defensively(msg["content"] or "")
            if data and _valid_table(data):
                cols = [str(c) for c in data["columns"]]
                rows = [{c: str(r.get(c, "")) for c in cols} for r in data["rows"]]
                table = {"columns": cols, "rows": rows}
                latest["table"] = table
                return {"table": table, "n_rows": len(rows), "n_cols": len(cols)}
        raise ValueError("extractor returned unparseable JSON twice (simplified retry included)")

    def validate_table(text: str) -> dict[str, Any]:
        table = latest["table"]
        if table is None:
            raise ValueError("nothing to validate — call extract_table first")

        g_score, g_defects = check_grounding(table, source)
        c_score, c_defects = check_completeness(table)
        s_score, s_defects = check_consistency(table)

        # Coverage is the one LLM-judged dimension; deterministic temperature,
        # defensive parse, and a bounded score derived from the count.
        msg, _usage = llm.chat(
            messages=[{"role": "system", "content": prompts.COVERAGE_SYSTEM},
                      {"role": "user", "content": prompts.coverage_user(source, table)}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = parse_json_defensively(msg["content"] or "") or {}
        missed = [str(m) for m in data.get("missed_facts", [])][:10]
        cov_score = round(max(0.0, 5.0 - 1.5 * len(missed)), 2)

        scores = {"groundedness": g_score, "coverage": cov_score,
                  "completeness": c_score, "consistency": s_score}
        return {"scores": scores,
                "weighted_total": weighted_total(scores),
                "defects": g_defects + c_defects + s_defects,
                "missed_facts": missed,
                "table": table}

    def finish_task(outcome: str, explanation: str) -> dict[str, Any]:
        """A decision tool: no side effects here. reflect() interprets the
        verdict and the harness enforces the success invariant."""
        if outcome not in ("success", "not_tabulatable"):
            raise ValueError(f"unknown outcome {outcome!r}")
        return {"outcome": outcome, "explanation": explanation}

    def recall_memory(query: str) -> dict[str, Any]:
        """On-demand memory consultation — the agent decides when history is
        worth checking (on top of the automatic read the spec mandates)."""
        from agent.memory_manager import source_hash
        if memory is None:
            return {"lessons": [], "note": "no memory backend attached"}
        lessons = memory.recall(query, k=5, source_hash=source_hash(source))
        return {"lessons": lessons or ["(no lessons recorded for this source yet)"]}

    return {"extract_table": extract_table, "validate_table": validate_table,
            "finish_task": finish_task, "recall_memory": recall_memory}


def _valid_table(data: dict[str, Any]) -> bool:
    """Bound what the model claims before we act on it."""
    try:
        return (isinstance(data["columns"], list) and data["columns"]
                and isinstance(data["rows"], list)
                and all(isinstance(r, dict) for r in data["rows"]))
    except (KeyError, TypeError):
        return False
