# HARNESS.md — Engineering Decisions

Every mechanism below exists because of a specific failure mode — several of
which I hit live during the build and diagnosed from the JSONL logs.

## Retries: exponential backoff with jitter

`with_retries` in `agent/harness.py` is hand-rolled: delay = base × 2^attempt,
capped, with full jitter (a uniform random fraction of the delay). It retries
rate limits, timeouts, connection errors and 5xx — and deliberately does NOT
retry auth errors, because retrying a bad key four times only delays the real
error message. Jitter exists because of the thundering-herd problem: without
it, every client that failed at the same moment retries at the same moment,
and the retries themselves become the next load spike.

## Model fallback chain

`GroqLLM.chat` tries the configured model list in order. This defends against
a retired model or a provider-side outage — and it earned its place the same
day I built it: both of my originally configured Groq models had been retired
(404 model_not_found). Because model names live in `config.yaml`, the fix was
a config edit and zero code changes. That incident is why every runtime knob
in this project lives in config.

## Malformed model output

Two distinct failure modes, two defenses. Unparseable JSON from the extractor
gets one retry with a simplified prompt (an explicit example of the required
shape, nothing else asked for), through `parse_json_defensively`, which strips
markdown fences and takes the last JSON block. Malformed tool *arguments*
never crash the loop — they become a visible Decision the harness reports,
and the agent adapts.

## Tool execution: allowlist and graceful observations

`act()` dispatches through a registry dict that doubles as an allowlist: a
tool name the model invented (I test with `__init__`) is refused, never
dispatched by attribute lookup. Handler exceptions become structured error
observations that go back into the conversation — the agent reads what went
wrong and routes around it. Raising instead would kill the run on exactly
the inputs where an agent should adapt.

## The two invariants: the model proposes, the gate disposes

Everything in this system is the model's decision except two things:

1. **A success claim is rejected unless a validated score meets the target.**
   The agent cannot declare victory without evidence.
2. **A refusal is rejected once a validated table at target already exists.**
   A late change of mind cannot discard proven work.

These are the only decisions kept in code, and they are evidence rules, not
choreography. Everything else — what to do, when to repair, when to consult
memory, when to end — belongs to the model.

## Content-only replies

A reply with no tool call gets a neutral `[harness]` note: "no tool was
called — every turn must end in exactly one tool call." This wording matters,
and I learned it the hard way: an earlier version treated any plain-text
reply as a success claim and rejected it. When the model wrote its refusal of
gibberish input as prose instead of a tool call, that misreading pushed it
away from refusing — and it went on to extract a junk table from the
gibberish, which validated at 4.88 (every cell was a substring of the
nonsense). Lesson: a harness that misinterprets the model's thinking as
claims coerces the agent into the exact behaviour it was trying to avoid.

## Loop guardrails

- **max_iterations (10, config)** — the hard cap; the last line of defense.
- **Token budget** — warns at a threshold, hard-stops with PARTIAL at the cap.
- **Stuck detection** — identical reflection outcome twice in a row ends the
  run. If the target was already met when the cycling starts, the run ends
  DONE with the best table instead of STUCK — cycling after success is waste,
  not failure.
- **Patience after target (4, config)** — once a validated table meets the
  target, the agent gets a bounded window to beat its best score; no
  improvement ends the run DONE with the best table. This exists because I
  watched the model meet the target at 4.74 and then keep "improving" the
  table to 4.55 and then 4.18 — objectively worse each round. Autonomy over
  strategy, bounded economics.
- **Extraction-streak notice** — three unvalidated extractions in a row gets
  a `[harness]` note that none of those tables has evidentiary weight. This
  covers varied-spinning that identical-fingerprint stuck detection cannot
  see.
- **max_model_errors (2, config)** — consecutive unrecoverable model failures
  end the run FAILED rather than looping on a dead provider.

Terminal statuses are explicit — DONE / REFUSED / PARTIAL / STUCK / FAILED —
and PARTIAL always returns the best validated table honestly rather than
pretending success or returning nothing.

## Memory failures

A recall failure logs a warning and continues without memory; a save failure
logs a warning and loses one lesson. Fault-tested with a store that throws on
every call: the run completes end-to-end. Memory is an advantage, never a
dependency.

## Observability

`agent/logger.py` writes one JSON object per step: timestamp, iteration,
step name, input/output summaries, latency in ms, and any error — as JSON
Lines, because a crashed run still leaves every completed line readable,
which one big JSON array would not. Summaries rather than full payloads keep
log volume sane and keep the user's text out of the logs. The model's own
REFLECT/PERCEIVE/REASON text is logged as a `cognition` event, so a run is
readable as a story: what the agent thought, what it did, what happened.

## Offline mode

`--offline` swaps in a scripted FakeLLM behind the same interface. It exists
as a test double (the control flow was debugged with zero cost and zero
latency before the real model ever ran) and as demo insurance — no network
or key failure can prevent a demonstration. Its honest limit: it cannot
think; feed it a novel input and it cannot adapt.

## Configuration

Secrets live in `.env` (gitignored; `.env.example` documents the keys).
Behaviour lives in `config.yaml`: models and fallbacks, temperature, client
timeout, iteration cap, score threshold, stuck threshold, patience,
extraction-streak threshold, max model errors, retry settings, token budget,
memory backend and recall depth. Nothing tunable is hardcoded in the core
loop — the retired-model incident is my proof of why that rule pays for
itself.
