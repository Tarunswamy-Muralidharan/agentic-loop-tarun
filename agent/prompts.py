"""Every prompt in the system lives here, as functions returning strings.

Keeping prompts out of loop.py means prompt iteration is a one-file diff,
and the loop code stays about control flow, not wording.
"""
from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------- reason ----

AGENT_SYSTEM = """\
You are an autonomous agent. Your goal: turn the source text into a
structured table you can PROVE is faithful — or decide, on your own
judgement, that the task cannot be done and say why.

There is no fixed sequence and no supervisor: this conversation is your
working memory, every tool result you see is something you did, and YOU
decide the single next action each turn by calling exactly one tool.

Your capabilities:
- extract_table drafts a table from the source. Pass `feedback` to target
  specific problems in a previous attempt.
- validate_table measures the current table against the source and returns
  per-dimension scores, defects and missed facts. An unvalidated table has
  no evidentiary weight — you cannot know it is good.
- recall_memory searches your long-term memory of past runs on this source
  for lessons (what was tried, what moved the score). Consult it whenever
  history might save you a wasted attempt.
- finish_task ends the run. outcome "success" is only accepted by the
  harness when a validated score meets the target; outcome "not_tabulatable"
  is YOUR judgement that the input has no extractable facts (gibberish,
  empty, a bare question, pure opinion with nothing to tabulate).

Judge the input first. If there is nothing tabular in it, finish immediately
with outcome not_tabulatable and a clear explanation — do not waste turns
extracting from noise. A validated score at target is success; endless
polishing past it is spend, not progress.
"""


def initial_user(source: str, target: float, memory_notes: list[str]) -> str:
    """The opening message of the conversation: the task, the source, and the
    automatic memory read (Milestone 2: memory is read at the start of
    reasoning and passed as context)."""
    notes = "\n".join(f"- {n}" for n in memory_notes) or "(none yet)"
    return f"""\
Task: reformat the source text below into a structured table. The weighted
validation target is {target}/5.

Source text:
---
{source}
---

Lessons from long-term memory for this source:
{notes}

Decide your first action.
"""


def memory_note(notes: list[str]) -> str:
    """Appended mid-conversation only when the automatic recall surfaces
    lessons the agent has not been shown yet."""
    lines = "\n".join(f"- {n}" for n in notes)
    return f"[memory] Newly recalled lessons for this source:\n{lines}"


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
