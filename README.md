# agentic-loop-tarun

Use case #24 — **reformat text into a structured table**. A hand-wired
agentic loop (`perceive → reason → act → reflect`) in plain Python. No agent
frameworks; the LLM is called directly through the OpenAI SDK pointed at Groq.

A one-shot extraction is easy; a *trustworthy* one is not. The loop makes the
task genuinely iterative: **extract → validate → repair → re-validate**,
stopping only when the validated score clears the threshold.

## The validator is the point

The table is scored 0–5 on four weighted dimensions. Three are **pure code**
— deterministic, free, and immune to a model grading its own homework:

| Dimension | Weight | How it is checked |
|---|---|---|
| groundedness | 0.35 | every non-empty cell value must appear in the source text (normalized match) — the anti-hallucination check |
| coverage | 0.30 | LLM auditor lists source facts missing from the table |
| completeness | 0.25 | code: fraction of filled cells |
| consistency | 0.10 | code: uniform columns, no duplicate rows, sane shape |

Defects and missed facts flow back into the next `extract_table` call as
feedback, so each repair targets what actually failed.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # then paste your GROQ_API_KEY into .env
```

## Run

```powershell
python main.py                       # live run on data/sample_input.md
python main.py --offline             # deterministic run, no key / no network
python main.py --clear-memory        # wipe long-term memory first
python main.py --input my_text.md
```

Every run writes a JSON-Lines step log to `logs/run-<timestamp>.jsonl` —
one object per step with timestamp, iteration, step name, input/output
summaries, latency and error.

## Layout

| Path | What it is |
|---|---|
| `agent/loop.py` | the four steps + `run()` — the core deliverable |
| `agent/tools.py` | 2 tool JSON-Schemas, handlers, code validators, registry |
| `agent/prompts.py` | every prompt in the system |
| `agent/memory_manager.py` | `save()` / `recall()` / `clear()` — Chroma, JSON fallback |
| `agent/harness.py` | retries + jitter, model fallback chain, token budget, LLM adapter |
| `agent/logger.py` | structured JSONL step logging |
| `agent/schemas.py` | dataclasses passed between the steps |
| `config.yaml` | every tunable; secrets live in `.env` only |

## Model

Groq `llama-3.3-70b-versatile`, with `llama-3.1-8b-instant` as an automatic
fallback (configured in `config.yaml`). Any OpenAI-compatible endpoint works
by changing `model.base_url` and `model.name`.
