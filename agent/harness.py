"""Reliability layer: config, retries, fallbacks, token budget, LLM adapter.

Everything that can fail over a network passes through here, so the loop in
loop.py stays about control flow. The retry is hand-rolled on purpose — it is
the thing being graded, and twelve lines of visible backoff beat a decorator
import you'd be asked to explain anyway.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from types import SimpleNamespace
from typing import Any

import yaml
from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

# Transient failures worth retrying. Auth errors are NOT here: retrying a bad
# key four times just delays the real error message.
RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


# ---------------------------------------------------------------- config ----

def load_config(path: str = "config.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------- defensive parse ----

def parse_json_defensively(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from model output.

    Strips markdown fences, then takes the LAST {...} block — models that
    wrap JSON in prose put the real answer at the end, and taking the last
    block also defeats decoy fences planted earlier in the text.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    candidates = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for candidate in reversed(candidates or [text]):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


# ------------------------------------------------------------ token budget --

class TokenBudget:
    """Tracks cumulative usage; the loop stops with PARTIAL when exceeded."""

    def __init__(self, max_total: int, warn_at: int) -> None:
        self.max_total, self.warn_at = max_total, warn_at
        self.total = 0
        self._warned = False

    def add(self, usage: dict[str, int]) -> None:
        self.total += usage.get("total_tokens", 0)
        if not self._warned and self.total >= self.warn_at:
            self._warned = True
            print(f"[budget] WARNING: {self.total} tokens used "
                  f"(warn threshold {self.warn_at}, hard cap {self.max_total})")

    @property
    def exceeded(self) -> bool:
        return self.total >= self.max_total


# ---------------------------------------------------- retry with backoff ----

def with_retries(fn, retry_cfg: dict[str, Any], on_attempt=None):
    """Call fn(); on a RETRYABLE error wait base*2^attempt (+ jitter) and retry.

    Jitter matters: without it, every client that failed at the same moment
    retries at the same moment, and the retries themselves become the load
    spike (thundering herd). Full jitter — uniform over [0, delay] — spreads
    them out.
    """
    max_attempts = retry_cfg["max_attempts"]
    base = retry_cfg["base_delay_seconds"]
    cap = retry_cfg["max_delay_seconds"]
    jitter = retry_cfg.get("jitter", True)

    for attempt in range(max_attempts):
        try:
            return fn()
        except RETRYABLE as exc:
            if attempt == max_attempts - 1:
                raise
            delay = min(cap, base * (2 ** attempt))
            if jitter:
                delay = random.uniform(0, delay)
            if on_attempt:
                on_attempt(attempt + 1, type(exc).__name__, delay)
            time.sleep(delay)


# -------------------------------------------------------------- LLM adapter -

class GroqLLM:
    """Thin adapter over the OpenAI SDK pointed at Groq.

    Normalises the response to (message_dict, usage_dict) so the loop never
    touches provider objects — swapping providers (or models, via the
    fallback chain) is contained to this class.
    """

    def __init__(self, cfg: dict[str, Any], budget: TokenBudget, logger=None) -> None:
        model_cfg = cfg["model"]
        self.client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url=model_cfg["base_url"],
            timeout=60.0,
        )
        self.models = [model_cfg["name"], *model_cfg.get("fallback_models", [])]
        self.temperature = model_cfg.get("temperature", 0.2)
        self.revise_temperature = model_cfg.get("revise_temperature", 0.7)
        self.retry_cfg = cfg["retry"]
        self.budget = budget
        self.logger = logger

    def chat(self, messages, tools=None, response_format=None, temperature=0.2):
        """One chat completion. Retries transient errors per model, then falls
        back to the next model in the chain (retired model, hard 4xx/5xx)."""
        temp = self.revise_temperature if temperature is None else temperature
        last_exc: Exception | None = None

        for model in self.models:
            def attempt():
                kwargs: dict[str, Any] = dict(model=model, messages=messages, temperature=temp)
                if tools:
                    kwargs["tools"] = tools
                if response_format:
                    kwargs["response_format"] = response_format
                return self.client.chat.completions.create(**kwargs)

            def note(n, err, delay):
                if self.logger:
                    self.logger.event("retry", f"{model}: {err}, attempt {n}, "
                                               f"sleeping {delay:.1f}s")

            try:
                resp = with_retries(attempt, self.retry_cfg, on_attempt=note)
            except (NotFoundError, BadRequestError, *RETRYABLE) as exc:
                # Model retired / unavailable after retries -> next in chain.
                last_exc = exc
                if self.logger:
                    self.logger.event("fallback", f"{model} failed "
                                                  f"({type(exc).__name__}); trying next model")
                continue

            usage = {"total_tokens": resp.usage.total_tokens if resp.usage else 0}
            self.budget.add(usage)
            msg = resp.choices[0].message
            return {
                "content": msg.content,
                "tool_calls": [
                    {"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
                    for c in (msg.tool_calls or [])
                ],
            }, usage

        raise RuntimeError(f"all models in the fallback chain failed: {last_exc}")


class FakeLLM:
    """Deterministic offline stand-in: score -> revise -> score(improved) -> finish.

    Exists so the loop can be demoed and unit-tested with no key, no network,
    and zero latency — and so a wifi failure cannot kill the live demo.
    """

    def __init__(self) -> None:
        self.scoring_round = 0
        self.budget: TokenBudget | None = None   # main() attaches the shared budget

    def chat(self, messages, tools=None, response_format=None, temperature=0.2):
        usage = {"total_tokens": 100}
        if self.budget:
            self.budget.add(usage)
        system = messages[0]["content"]

        if "writing judge" in system or "valid JSON" in system:   # scorer call
            self.scoring_round += 1
            s = ({"clarity": 3, "evidence": 2, "concision": 2, "tone": 2}
                 if self.scoring_round == 1
                 else {"clarity": 4, "evidence": 5, "concision": 4, "tone": 5})
            return {"content": json.dumps({
                "scores": s,
                "critiques": {k: f"(fake critique of {k})" for k in s},
            }), "tool_calls": []}, usage

        if "precise editor" in system:                            # reviser call
            return {"content": "A clean, evidence-backed rewrite of the draft "
                               "(offline mode; 27% faster in benchmark X)."}, usage

        # reason call: script the decision sequence off the visible state
        user = messages[-1]["content"]
        if "not scored yet" in user or "re-score" in user:
            args = {"text": _extract_draft(user)}
            return {"content": None, "tool_calls": [
                {"id": "fake-1", "name": "score_against_rubric",
                 "arguments": json.dumps(args)}]}, usage
        if "weighted total 4." in user or "weighted total 5" in user:
            return {"content": "FINISH — target met (offline)."}, usage
        args = {"text": _extract_draft(user), "criterion": "evidence",
                "guidance": "Add a concrete number or example."}
        return {"content": None, "tool_calls": [
            {"id": "fake-2", "name": "revise_text",
             "arguments": json.dumps(args)}]}, usage


def _extract_draft(user_msg: str) -> str:
    m = re.search(r"---\n(.*?)\n---", user_msg, flags=re.DOTALL)
    return m.group(1) if m else user_msg[:200]
