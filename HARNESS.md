# HARNESS.md — write this yourself

<!-- The challenge requires this file to reflect YOUR thinking. AI-written
     reports are grounds for disqualification. Replace this file's contents
     with your own prose: "a brief explanation of each engineering decision
     and what failure mode it defends against."

     Walk through harness.py + the guards in loop.py and answer, per item:
     WHAT failure does it defend against, and WHY this design?

     - Retry with exponential backoff + jitter (with_retries): which errors
       retry, which deliberately don't (auth), and why jitter exists
       (thundering herd).
     - Model fallback chain (GroqLLM.chat): retired model / provider outage —
       you have a real story: both original llama IDs 404'd; the fix was a
       config-only edit because model names live in config.yaml.
     - Simplified-prompt retry on unparseable JSON (tools.extract_table);
       malformed tool ARGUMENTS become a visible Decision, not a crash.
     - Registry-as-allowlist + graceful error observations (loop.act):
       why errors are returned to the agent instead of raised.
     - The two invariants in reflect: no success claim without a validated
       score at target; no refusal once a validated table at target exists.
       Why these are code while every other decision belongs to the model
       ("the model proposes, the gate disposes").
     - Content-only replies: why a reply with no tool call gets a neutral
       [harness] reminder — and the live incident where the old behaviour
       (treating text as a success claim) coerced the agent into tabulating
       gibberish.
     - Guardrails and what each costs/prevents: max_iterations; token budget
       (warn + hard stop -> PARTIAL); stuck detection (identical reflection
       twice — and why cycling with the target already met ends DONE with
       the best table instead); patience_after_target (the perfectionism
       loop you caught live: 4.74 -> 4.55 -> 4.18); extraction-streak notice;
       max_model_errors -> FAILED.
     - Memory failure fallbacks: recall/save failures warn and continue
       (fault-tested with a store that throws on every call).
     - Observability: JSONL per step (ts, iteration, step, summaries,
       latency_ms, error) + the cognition log line; why JSON Lines beats one
       JSON array; why summaries not full payloads.
     - Offline FakeLLM mode as a harness feature: test double + demo
       insurance, and what it deliberately cannot do (think).
     - Configuration: config.yaml vs .env split; every runtime knob
       (models, timeout, temperature, caps, budget, patience, streak,
       memory backend) lives in config — the retired-model incident as
       proof of why. -->
