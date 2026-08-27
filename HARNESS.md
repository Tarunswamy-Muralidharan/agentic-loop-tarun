# HARNESS.md — write this yourself

<!-- Same rule: your own words only. "Brief explanation of each engineering
     decision and what failure it defends against."

     Walk through harness.py + the guards in loop.py and answer, per item:
     WHAT failure does it defend against, and WHY this design?

     - Retry with exponential backoff + jitter (with_retries): which errors
       retry, which deliberately don't (auth), and why jitter exists
       (thundering herd).
     - Model fallback chain (GroqLLM.chat): retired model / provider outage.
     - Simplified-prompt retry on bad JSON (tools.extract_table).
     - Unknown-tool refusal + graceful error observations (loop.act):
       why errors are returned to the agent instead of raised.
     - Rule-based reflect + code validators: why NOT LLM-as-judge for
       groundedness/completeness/consistency (deterministic, free, and the
       grounding check is the anti-hallucination control — a model cannot
       talk its way past a substring match).
     - max_iterations, stuck detection (identical validation score twice —
       the repair achieved nothing), token budget: what each cap costs and
       what it prevents.
     - PARTIAL contract: why the best table is returned honestly instead
       of pretending success or returning nothing.
     - config.yaml vs .env split: what lives where and why. -->
