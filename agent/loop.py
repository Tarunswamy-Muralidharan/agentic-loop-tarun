"""The agentic loop, conversation-driven.

There is no staged pipeline. One persistent conversation is the agent's
working memory: the model sees everything it has done (its own tool calls
and their raw results) and freely chooses every next action — extract,
validate, consult memory, or finish (including refusing the task). The
driver below is transport and governance, not cognition.

The four spec-required functions remain, each owning one non-cognitive job:
    perceive  — maintain what the agent can see: the conversation, plus the
                automatic memory read (Milestone 2) injected when new
    reason    — one model turn over the full conversation + tool schemas
    act       — execute exactly one requested tool via the allowlist registry
    reflect   — bookkeeping and the safety invariants; it NEVER instructs

Invariants (the model proposes, the gate disposes):
    1. a SUCCESS claim is rejected without a validated score >= target
    2. a REFUSAL is rejected once a validated table at target exists
Guardrails: max_iterations, token budget, stuck (identical outcome twice),
patience-after-target (bounds perfectionism), extraction-streak notice.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from agent import prompts
from agent.memory_manager import source_hash
from agent.schemas import AgentState, Decision, Observation, Reflection, Status

MAX_MODEL_ERRORS = 2      # consecutive model failures tolerated before FAILED
EXTRACT_STREAK_WARN = 2   # consecutive extractions without validating -> notice


# ---------------------------------------------------------------- perceive --

def perceive(state: AgentState, memory, recall_k: int, target: float) -> list[dict]:
    """Maintain what the agent is allowed to see.

    First turn: build the conversation (system + task + the automatic memory
    read the spec mandates). Later turns: the conversation already carries
    everything the agent did; only genuinely NEW memory lessons are appended,
    so the agent is informed without being spammed or steered.
    """
    try:
        notes = memory.recall(f"tabulating this source, lessons",
                              k=recall_k, source_hash=source_hash(state.source))
    except Exception as exc:   # noqa: BLE001 — memory read failure: warn, continue
        print(f"[memory] WARNING: recall failed ({exc}); continuing without memory")
        notes = []

    if not state.messages:
        state.messages = [
            {"role": "system", "content": prompts.AGENT_SYSTEM},
            {"role": "user", "content": prompts.initial_user(state.source, target, notes)},
        ]
        state.notes_seen = "|".join(notes)
        return state.messages

    fingerprint = "|".join(notes)
    if notes and fingerprint != state.notes_seen:
        state.messages.append({"role": "user", "content": prompts.memory_note(notes)})
        state.notes_seen = fingerprint
    return state.messages


# ------------------------------------------------------------------ reason --

def reason(messages: list[dict], llm, tool_schemas: list[dict]) -> Decision:
    """One model turn over the whole conversation. The model's choice comes
    back verbatim; malformed arguments become a Decision the loop can see."""
    msg, _usage = llm.chat(messages=messages, tools=tool_schemas, temperature=0.2)
    if msg["tool_calls"]:
        call = msg["tool_calls"][0]                    # exactly one tool per turn
        dropped = len(msg["tool_calls"]) - 1
        try:
            args = json.loads(call["arguments"])
        except json.JSONDecodeError:
            return Decision(kind="final",
                            final_text=f"MALFORMED tool arguments for {call['name']}",
                            reasoning="model emitted unparseable arguments")
        return Decision(kind="tool_call", tool_name=call["name"], tool_args=args,
                        reasoning=msg["content"], dropped_calls=dropped,
                        call_id=call["id"], raw_arguments=call["arguments"])
    return Decision(kind="final", final_text=(msg["content"] or "").strip(),
                    reasoning=msg["content"])


# --------------------------------------------------------------------- act --

def act(decision: Decision, registry: dict[str, Any]) -> Observation:
    """Execute exactly one tool via the registry-as-allowlist. Failures become
    structured observations the agent reads and routes around — never a crash."""
    if decision.kind == "final":
        return Observation(ok=True, content=decision.final_text)
    if decision.tool_name not in registry:
        return Observation(ok=False, error=f"unknown tool {decision.tool_name!r}; "
                                           f"allowed: {sorted(registry)}")
    try:
        result = registry[decision.tool_name](**decision.tool_args)
        return Observation(ok=True, content=result)
    except TypeError as exc:        # wrong/missing arguments from the model
        return Observation(ok=False, error=f"bad arguments for {decision.tool_name}: {exc}")
    except Exception as exc:        # noqa: BLE001 — handler failure must not kill the loop
        return Observation(ok=False, error=f"{decision.tool_name} failed: {exc}")


# ----------------------------------------------------------------- reflect --

def reflect(state: AgentState, decision: Decision, obs: Observation,
            score_threshold: float) -> Reflection:
    """Bookkeeping + the two invariants. The agent already sees the raw tool
    result in the conversation; reflect only tracks state the harness needs
    (best table, scores) and gates the endings. next_instruction is used
    ONLY for harness messages that must reach the agent (gate rejections)."""
    before = state.weighted_total

    if not obs.ok:
        return Reflection(
            is_done=False, quality_score=before if before is not None else -1,
            next_instruction="",                     # the error is already in-convo
            summary=f"error:{obs.error}",
            memory_record={"iteration": state.iteration, "action": decision.tool_name,
                           "target": "-", "score_before": before, "score_after": before,
                           "lesson": f"failed: {obs.error}"},
        )

    # ---- the agent decided to end the run -----------------------------------
    if decision.tool_name == "finish_task" or decision.kind == "final":
        if decision.kind == "final":
            text = str(obs.content or "").strip()
            if not text:
                return Reflection(
                    is_done=False,
                    quality_score=before if before is not None else -1,
                    next_instruction="[harness] Your last reply contained no tool "
                                     "call and no content — nothing happened.",
                    summary="empty-reply", memory_record=None)
            verdict = {"outcome": "success", "explanation": text}
        else:
            verdict = obs.content

        if verdict["outcome"] == "not_tabulatable":
            if state.best_total >= score_threshold:      # invariant 2
                return Reflection(
                    is_done=False,
                    quality_score=before if before is not None else -1,
                    next_instruction="[harness] Refusal rejected: a validated table "
                                     f"scoring {state.best_total} (>= target) already "
                                     "exists for this source.",
                    summary="refusal-rejected", memory_record=None)
            state.status = Status.REFUSED
            state.verdict = verdict["explanation"]
            return Reflection(
                is_done=True, quality_score=-1, next_instruction="",
                summary=f"refused:{verdict['explanation'][:60]}",
                memory_record={"iteration": state.iteration, "action": "refuse",
                               "target": "-", "score_before": before,
                               "score_after": before,
                               "lesson": f"judged not tabulatable: "
                                         f"{verdict['explanation'][:80]}"})

        met = before is not None and before >= score_threshold   # invariant 1
        if met:
            state.verdict = verdict["explanation"]
        return Reflection(
            is_done=met,
            quality_score=before if before is not None else -1,
            next_instruction="" if met else
                "[harness] Success claim rejected: there is no validated score at "
                "target on the current table.",
            summary=f"finish:{verdict['outcome']}:{'accepted' if met else 'rejected'}",
            memory_record=None if not met else
                {"iteration": state.iteration, "action": "finish", "target": "-",
                 "score_before": before, "score_after": before,
                 "lesson": "finished at target"})

    result = obs.content
    if decision.tool_name == "extract_table":
        state.table = result["table"]
        # A fresh extraction invalidates the previous measurement entirely.
        state.last_scores, state.weighted_total = None, None
        state.defects, state.missed_facts = [], []
        was_repair = bool(decision.tool_args.get("feedback"))
        return Reflection(
            is_done=False, quality_score=-1, next_instruction="",
            summary=f"extracted:{result['n_rows']}x{result['n_cols']}"
                    f"{':repair' if was_repair else ''}",
            memory_record={"iteration": state.iteration, "action": "extract",
                           "target": "repair" if was_repair else "initial",
                           "score_before": before, "score_after": None,
                           "lesson": f"extracted {result['n_rows']} rows x "
                                     f"{result['n_cols']} cols, effect unknown until validation"})

    if decision.tool_name == "validate_table":
        after = result["weighted_total"]
        state.last_scores = result["scores"]
        state.weighted_total = after
        state.defects = result["defects"]
        state.missed_facts = result["missed_facts"]
        if after > state.best_total:
            state.best_total, state.best_table = after, state.table
            state.best_scores = result["scores"]
        weakest = min(result["scores"], key=result["scores"].get)
        return Reflection(
            is_done=False, quality_score=after, next_instruction="",
            summary=f"validated:{after}",
            memory_record={"iteration": state.iteration, "action": "validate",
                           "target": weakest,
                           "score_before": before, "score_after": after,
                           "lesson": f"weakest={weakest}; "
                                     f"{len(result['defects'])} defects, "
                                     f"{len(result['missed_facts'])} missed facts"})

    # recall_memory (or any read-only tool): nothing to track, nothing to say.
    return Reflection(is_done=False,
                      quality_score=before if before is not None else -1,
                      next_instruction="",
                      summary=f"{decision.tool_name}:ok", memory_record=None)


# --------------------------------------------------------------------- run --

def run(goal: str, source: str, llm, registry, tool_schemas, memory, logger,
        cfg: dict[str, Any]) -> AgentState:
    loop_cfg = cfg["loop"]
    threshold = loop_cfg["score_threshold"]
    state = AgentState(goal=goal, source=source)
    recent: list[str] = []
    consecutive_errors = 0
    extract_streak = 0
    no_improve = 0

    while not state.done and state.iteration < loop_cfg["max_iterations"]:
        state.iteration += 1

        t0 = time.perf_counter()
        messages = perceive(state, memory, cfg["memory"]["recall_k"], threshold)
        logger.step(state.iteration, "perceive", f"{len(messages)} messages in context",
                    "conversation ready", t0)

        t0 = time.perf_counter()
        try:
            decision = reason(messages, llm, tool_schemas)
            consecutive_errors = 0
        except Exception as exc:   # noqa: BLE001 — post-retry model failure
            consecutive_errors += 1
            logger.step(state.iteration, "reason", "-", "-", t0, error=str(exc))
            if consecutive_errors > MAX_MODEL_ERRORS:
                state.status, state.done = Status.FAILED, True
                logger.event("stop", f"FAILED: model unrecoverable ({exc})")
                break
            continue
        logger.step(state.iteration, "reason", f"iter {state.iteration}",
                    f"{decision.kind}:{decision.tool_name or decision.final_text}", t0)
        if decision.dropped_calls:
            logger.event("dropped_calls",
                         f"model emitted {decision.dropped_calls + 1} tool calls; "
                         f"only the first was executed (one tool per turn)")

        t0 = time.perf_counter()
        obs = act(decision, registry)
        logger.step(state.iteration, "act", decision.tool_name or "final",
                    obs.content if obs.ok else "-", t0, error=obs.error)

        # Append the exchange to the conversation — the agent's own history.
        if decision.kind == "tool_call":
            state.messages.append({
                "role": "assistant", "content": decision.reasoning or "",
                "tool_calls": [{"id": decision.call_id, "type": "function",
                                "function": {"name": decision.tool_name,
                                             "arguments": decision.raw_arguments}}]})
            state.messages.append({
                "role": "tool", "tool_call_id": decision.call_id,
                "content": json.dumps(obs.content if obs.ok else {"error": obs.error},
                                      ensure_ascii=False)})
        else:
            state.messages.append({"role": "assistant",
                                   "content": decision.final_text or ""})

        t0 = time.perf_counter()
        prev_best = state.best_total
        reflection = reflect(state, decision, obs, threshold)
        logger.step(state.iteration, "reflect", reflection.summary,
                    f"done={reflection.is_done} score={reflection.quality_score}", t0)

        # Harness gate messages are the ONLY things reflect says to the agent.
        if reflection.next_instruction:
            state.messages.append({"role": "user",
                                   "content": reflection.next_instruction})

        if reflection.memory_record:                 # write AFTER reflect (Milestone 2)
            reflection.memory_record["source_hash"] = source_hash(state.source)
            try:
                memory.save(reflection.memory_record)
            except Exception as exc:   # noqa: BLE001
                logger.event("memory_warning", f"save failed ({exc}); "
                                               f"continuing without persisting this lesson")

        if reflection.is_done:
            state.done = True
            if state.status is None:
                state.status = Status.DONE
            break

        # Extraction-streak notice: redrafting repeatedly without measuring.
        if reflection.summary.startswith("extracted"):
            extract_streak += 1
            if extract_streak > EXTRACT_STREAK_WARN:
                state.messages.append({"role": "user", "content":
                    f"[harness] You have extracted {extract_streak} times in a row "
                    f"without validating — none of these tables has evidentiary "
                    f"weight until one is measured."})
        else:
            extract_streak = 0

        # Stuck: identical reflection outcome twice in a row.
        fp = hashlib.sha256(reflection.summary.encode()).hexdigest()
        recent.append(fp)
        if (len(recent) >= loop_cfg["stuck_threshold"]
                and len(set(recent[-loop_cfg["stuck_threshold"]:])) == 1):
            state.done, state.status = True, Status.STUCK
            logger.event("stop", "STUCK: identical reflection twice in a row")
            break

        # Patience: target met, bounded window to beat the best, then DONE.
        if state.best_total >= threshold:
            no_improve = 0 if state.best_total > prev_best else no_improve + 1
            if no_improve >= loop_cfg.get("patience_after_target", 4):
                state.done, state.status = True, Status.DONE
                state.table = state.best_table
                state.weighted_total = state.best_total
                state.last_scores = state.best_scores
                state.verdict = (f"Harness: target met (best validated score "
                                 f"{state.best_total}); no improvement in "
                                 f"{no_improve} iterations — returning the best table.")
                logger.event("stop", state.verdict)
                break

        if llm.budget.exceeded:
            state.done, state.status = True, Status.PARTIAL
            logger.event("stop", f"token budget exceeded ({llm.budget.total})")
            break

    if state.status is None:
        state.status = Status.PARTIAL
        logger.event("stop", "max_iterations reached; returning best table")

    if state.status is not Status.DONE and state.best_table:
        state.table = state.best_table
        state.weighted_total = state.best_total
        state.last_scores = state.best_scores
    return state
