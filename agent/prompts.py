"""Every prompt in the system lives here, as functions returning strings.

Keeping prompts out of loop.py means prompt iteration is a one-file diff,
and the loop code stays about control flow, not wording.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------- reason ----

REASON_SYSTEM = """\
You are the decision-maker inside an agent that improves a piece of text
against a rubric. You do not rewrite text yourself — you decide the next
action and call a tool to do it.

Rules:
- If the draft has not been scored yet, call score_against_rubric first.
- After any revision, you must re-score before deciding you are finished.
- If the latest weighted score meets or beats the target, do NOT call a tool.
  Reply in plain text starting with the word FINISH and a one-line reason.
- Otherwise, call revise_text targeting the single weakest criterion, then
  expect to re-score on the next turn.
- Call exactly one tool per turn.
"""


def reason_user(percept: Any, target: float) -> str:
    """Build the user message for the reason step from the Percept."""
    scores = (
        "not scored yet"
        if percept.last_scores is None
        else ", ".join(f"{k}={v}/5" for k, v in percept.last_scores.items())
        + f"  (weighted total {percept.weighted_total:.2f}/5, target {target})"
    )
    critiques = (
        "\n".join(f"- {k}: {v}" for k, v in percept.critiques.items())
        or "(none yet)"
    )
    memory = "\n".join(f"- {m}" for m in percept.memory_notes) or "(no relevant memory)"
    return f"""\
Iteration {percept.iteration}.
Goal: {percept.goal}

Current draft:
---
{percept.draft}
---

Latest scores: {scores}
Latest critiques:
{critiques}

Guidance carried over from the previous iteration:
{percept.next_instruction}

Lessons recalled from memory (avoid repeating what did not work):
{memory}

Decide the single next action.
"""


# ----------------------------------------------------------------- score ----

SCORE_SYSTEM = """\
You are a strict writing judge. Score the text against each rubric criterion.
Respond ONLY with a JSON object of this exact shape:
{"scores": {"<criterion>": <1-5 integer>, ...},
 "critiques": {"<criterion>": "<one specific sentence>", ...}}
Every rubric criterion must appear in both maps. No other keys, no prose.
"""


def score_user(text: str, rubric: list[dict[str, Any]]) -> str:
    lines = "\n".join(
        f'- {c["name"]} (weight {c["weight"]}): {c["description"]}' for c in rubric
    )
    return f"Rubric:\n{lines}\n\nText to score:\n---\n{text}\n---\nReturn the JSON."


# Fallback used when the model's JSON fails to parse twice: fewer fields,
# an explicit example, and the shape spelled out again.
SCORE_SYSTEM_SIMPLIFIED = """\
Return ONLY valid JSON, nothing else. Example of the exact shape required:
{"scores": {"clarity": 3, "evidence": 2, "concision": 4, "tone": 3},
 "critiques": {"clarity": "…", "evidence": "…", "concision": "…", "tone": "…"}}
Score the text 1-5 on each criterion named in the example.
"""


# ---------------------------------------------------------------- revise ----

REVISE_SYSTEM = """\
You are a precise editor. Rewrite the given text to fix ONE named weakness,
changing as little else as possible. Keep the author's meaning. Return ONLY
the rewritten text — no preamble, no commentary, no quotation marks.
"""


def revise_user(text: str, criterion: str, guidance: str) -> str:
    return f"""\
Weakness to fix: {criterion}
Judge's critique: {guidance}

Text:
---
{text}
---
Return the full rewritten text."""
