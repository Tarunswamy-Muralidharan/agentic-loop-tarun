"""Every prompt in the system lives here, as functions returning strings.

Keeping prompts out of loop.py means prompt iteration is a one-file diff,
and the loop code stays about control flow, not wording.
"""
from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------- reason ----

REASON_SYSTEM = """\
You are the decision-maker inside an agent that reformats messy text into a
structured table. You do not build the table yourself — you decide the next
action and call a tool to do it.

Rules:
- If no table exists yet, call extract_table first.
- After ANY extraction, you must call validate_table before anything else —
  an unvalidated table has no score.
- If validation reports defects or missed facts, call extract_table again and
  pass the feedback so the extraction can repair them.
- If the latest validated score meets or beats the target, do NOT call a
  tool. Reply in plain text starting with the word FINISH and a one-line reason.
- Call exactly one tool per turn.
"""


def reason_user(percept: Any, target: float) -> str:
    """Build the user message for the reason step from the Percept."""
    if percept.last_scores is None:
        scores = "no validated score yet" + (
            " (a table exists but validation is required)" if percept.table else
            " (no table yet)")
    else:
        scores = (", ".join(f"{k}={v:.1f}/5" for k, v in percept.last_scores.items())
                  + f"  (weighted total {percept.weighted_total:.2f}/5, target {target})")
    defects = "\n".join(f"- {d}" for d in percept.defects) or "(none reported)"
    missed = "\n".join(f"- {m}" for m in percept.missed_facts) or "(none reported)"
    memory = "\n".join(f"- {m}" for m in percept.memory_notes) or "(no relevant memory)"
    table = (json.dumps(percept.table, ensure_ascii=False)[:800]
             if percept.table else "(none)")
    return f"""\
Iteration {percept.iteration}.
Goal: {percept.goal}

Source text:
---
{percept.source}
---

Current table: {table}
Validation: {scores}
Defects found by the validator:
{defects}
Facts the validator says were missed:
{missed}

Guidance carried over from the previous iteration:
{percept.next_instruction}

Lessons recalled from memory (avoid repeating what did not work):
{memory}

Decide the single next action.
"""


# --------------------------------------------------------------- extract ----

EXTRACT_SYSTEM = """\
You convert messy prose into a structured table. Respond ONLY with a JSON
object of this exact shape:
{"columns": ["<col>", ...],
 "rows": [{"<col>": "<value>", ...}, ...]}
Rules:
- Choose column names that fit the facts in the text (short, lowercase).
- Every row must use exactly the declared columns; use "" when a value is
  genuinely absent from the text.
- Copy values from the text — never invent, estimate, or embellish a value.
- One row per distinct item/event/entity described in the text.
No other keys, no prose, no markdown.
"""

# Fallback when the model's JSON fails to parse: shape spelled out again with
# an explicit example, nothing else asked for.
EXTRACT_SYSTEM_SIMPLIFIED = """\
Return ONLY valid JSON, nothing else. Example of the exact shape required:
{"columns": ["item", "date", "amount"],
 "rows": [{"item": "cab", "date": "June 2", "amount": "850"}]}
Extract one row per distinct item/event in the text. Copy values verbatim
from the text; use "" for missing values.
"""


def extract_user(text: str, feedback: str | None) -> str:
    fb = ""
    if feedback:
        fb = f"""
A validator reviewed your previous attempt and found these problems — fix
every one of them in this extraction:
{feedback}
"""
    return f"Text to tabulate:\n---\n{text}\n---{fb}\nReturn the JSON."


# -------------------------------------------------------------- coverage ----
# The only LLM half of validation: "which facts in the source never made it
# into the table?" Everything else (completeness, groundedness, consistency)
# is deterministic code in tools.py.

COVERAGE_SYSTEM = """\
You are a strict auditor. Given a source text and a table extracted from it,
list every distinct fact in the source that is NOT represented in the table.
Respond ONLY with JSON: {"missed_facts": ["<one short line per missed fact>", ...]}
If nothing is missing, return {"missed_facts": []}. No other keys, no prose.
"""


def coverage_user(text: str, table: dict[str, Any]) -> str:
    return (f"Source text:\n---\n{text}\n---\n\n"
            f"Extracted table:\n{json.dumps(table, ensure_ascii=False)}\n\n"
            f"Return the JSON.")
