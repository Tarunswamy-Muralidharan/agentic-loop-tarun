"""The core agentic loop: perceive -> reason -> act -> reflect.

Each function owns one decision:
    perceive  — what the agent is ALLOWED TO KNOW this iteration
                (memory recall, which past results survive into context)
    reason    — WHAT TO DO next (the only step that must talk to the LLM;
                prompt assembly, tool exposure, parsing the tool call)
    act       — HOW TO EXECUTE exactly one tool (dispatch via allowlist,
                turning exceptions into observations the model can read)
    reflect   — whether we are MAKING PROGRESS (score bookkeeping, stop
                decision, and the one-line lesson that goes to memory)

run() wires them together and owns the harness concerns: iteration cap,
token budget, stuck detection, terminal status.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from agent import prompts
from agent.schemas import (AgentState, Decision, Observation, Percept,
                           Reflection, Status)

MAX_MODEL_ERRORS = 2   # consecutive reason/tool failures tolerated before FAILED


# ---------------------------------------------------------------- perceive --

def perceive(state: AgentState, memory, recall_k: int) -> Percept:
    """Assemble this iteration's view of the world.

    Memory is recalled HERE (not in reason) so that what the model may know
    is decided in one place — reason receives a finished Percept and cannot
    reach around it.
    """
    query = (f"tabulating text defects {' '.join(state.defects[:3])} "
             f"score {state.weighted_total}")
    notes = memory.recall(query, k=recall_k)
    return Percept(
        goal=state.goal,
        source=state.source,
        iteration=state.iteration,
        table=state.table,
        last_scores=state.last_scores,
        weighted_total=state.weighted_total,
        defects=state.defects,
        missed_facts=state.missed_facts,
        next_instruction=state.next_instruction,
        memory_notes=notes,
    )


# ------------------------------------------------------------------ reason --

def reason(percept: Percept, llm, tool_schemas: list[dict], target: float) -> Decision:
    """One LLM call with the tool schemas. Returns a Decision, never raises
    on malformed tool arguments — bad args become a 'final' decision the
    reflect step can see and react to."""
    msg, _usage = llm.chat(
        messages=[
            {"role": "system", "content": prompts.REASON_SYSTEM},
            {"role": "user", "content": prompts.reason_user(percept, target)},
        ],
        tools=tool_schemas,
        temperature=0.2,
    )
    if msg["tool_calls"]:
        call = msg["tool_calls"][0]                    # exactly one tool per turn
        try:
            args = json.loads(call["arguments"])
        except json.JSONDecodeError:
            return Decision(kind="final",
                            final_text=f"MALFORMED tool arguments for {call['name']}",
                            reasoning="model emitted unparseable arguments")
        return Decision(kind="tool_call", tool_name=call["name"], tool_args=args,
                        reasoning=msg["content"])
    return Decision(kind="final", final_text=(msg["content"] or "").strip(),
                    reasoning=msg["content"])


# --------------------------------------------------------------------- act --

def act(decision: Decision, registry: dict[str, Any]) -> Observation:
    """Execute exactly one tool via the registry.

    The registry doubles as an allowlist: a tool name the model invented is
    refused here, not dispatched via getattr. Handler exceptions become
    structured error observations — the challenge's required fallback is
    'return a graceful error observation and let the agent adapt'.
    """
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
    """Rule-based judgement. Deliberately NOT an LLM call: comparing numbers
    and reading a defect list is cheaper, deterministic, and immune to the
    model grading its own homework."""
    before = state.weighted_total

    if not obs.ok:
        return Reflection(
            is_done=False, quality_score=before if before is not None else -1,
            next_instruction=f"The last action failed ({obs.error}). "
                             f"Choose a different action.",
            summary=f"error:{obs.error}",
            memory_record={"iteration": state.iteration, "action": decision.tool_name,
                           "target": "-", "score_before": before, "score_after": before,
                           "lesson": f"failed: {obs.error}"},
        )

    if decision.kind == "final":
        text = str(obs.content or "")
        done = text.upper().startswith("FINISH") and before is not None \
               and before >= score_threshold
        return Reflection(
            is_done=done,
            quality_score=before if before is not None else -1,
            next_instruction="" if done else
                "You said FINISH but there is no validated score at target — "
                "extract or validate instead.",
            summary=f"final:{text[:60]}",
            memory_record=None if not done else
                {"iteration": state.iteration, "action": "finish", "target": "-",
                 "score_before": before, "score_after": before,
                 "lesson": "finished at target"},
        )

    result = obs.content
    if decision.tool_name == "extract_table":
        state.table = result["table"]
        # A fresh extraction invalidates any previous score: force re-validation.
        state.last_scores, state.weighted_total = None, None
        was_repair = bool(decision.tool_args.get("feedback"))
        return Reflection(
            is_done=False,
            quality_score=-1,
            next_instruction="Table extracted — validate it to get a score.",
            summary=f"extracted:{result['n_rows']}x{result['n_cols']}"
                    f"{':repair' if was_repair else ''}",
            memory_record={"iteration": state.iteration, "action": "extract",
                           "target": "repair" if was_repair else "initial",
                           "score_before": before, "score_after": None,
                           "lesson": f"extracted {result['n_rows']} rows x "
                                     f"{result['n_cols']} cols, effect unknown until validation"},
        )

    # validate_table: adopt the verdict; done if the threshold is met.
    after = result["weighted_total"]
    state.last_scores = result["scores"]
    state.weighted_total = after
    state.defects = result["defects"]
    state.missed_facts = result["missed_facts"]
    if after > state.best_total:
        state.best_total, state.best_table = after, state.table
    met = after >= score_threshold
    weakest = min(result["scores"], key=result["scores"].get)
    feedback_hint = "; ".join((result["defects"] + result["missed_facts"])[:4])
    return Reflection(
        is_done=met,
        quality_score=after,
        next_instruction="Target met — reply FINISH." if met else
            f"Re-extract with feedback to fix the weakest dimension "
            f"({weakest}): {feedback_hint}",
        summary=f"validated:{after}",
        memory_record={"iteration": state.iteration, "action": "validate", "target": weakest,
                       "score_before": before, "score_after": after,
                       "lesson": f"weakest={weakest}; "
                                 f"{len(result['defects'])} defects, "
                                 f"{len(result['missed_facts'])} missed facts"},
    )


# --------------------------------------------------------------------- run --

def run(goal: str, source: str, llm, registry, tool_schemas, memory, logger,
        cfg: dict[str, Any]) -> AgentState:
    loop_cfg = cfg["loop"]
    state = AgentState(goal=goal, source=source)
    recent_fingerprints: list[str] = []
    consecutive_errors = 0

    while not state.done and state.iteration < loop_cfg["max_iterations"]:
        state.iteration += 1

        t0 = time.perf_counter()
        percept = perceive(state, memory, cfg["memory"]["recall_k"])
        logger.step(state.iteration, "perceive", state.next_instruction,
                    f"{len(percept.memory_notes)} memory notes recalled", t0)

        t0 = time.perf_counter()
        try:
            decision = reason(percept, llm, tool_schemas, loop_cfg["score_threshold"])
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

        t0 = time.perf_counter()
        obs = act(decision, registry)
        logger.step(state.iteration, "act", decision.tool_name or "final",
                    obs.content if obs.ok else "-", t0, error=obs.error)

        t0 = time.perf_counter()
        reflection = reflect(state, decision, obs, loop_cfg["score_threshold"])
        logger.step(state.iteration, "reflect", reflection.summary,
                    f"done={reflection.is_done} score={reflection.quality_score}", t0)

        if reflection.memory_record:                 # write AFTER reflect (Milestone 2)
            memory.save(reflection.memory_record)
        state.next_instruction = reflection.next_instruction

        if reflection.is_done:
            state.done, state.status = True, Status.DONE
            break

        # Stuck detection: identical reflections twice in a row -> STUCK.
        # For this use case that means two validations with the same score —
        # the repair in between achieved nothing, so more loops won't either.
        fp = hashlib.sha256(reflection.summary.encode()).hexdigest()
        if reflection.summary.startswith("validated:"):
            recent_fingerprints.append(fp)
            if (len(recent_fingerprints) >= loop_cfg["stuck_threshold"]
                    and len(set(recent_fingerprints[-loop_cfg["stuck_threshold"]:])) == 1):
                state.done, state.status = True, Status.STUCK
                logger.event("stop", "STUCK: identical validation score twice in a row")
                break

        if llm.budget.exceeded:
            state.done, state.status = True, Status.PARTIAL
            logger.event("stop", f"token budget exceeded ({llm.budget.total})")
            break

    if state.status is None:                         # loop cap without finishing
        state.status = Status.PARTIAL
        logger.event("stop", "max_iterations reached; returning best table")

    # PARTIAL/STUCK contract: always hand back the best table seen, honestly.
    if state.status is not Status.DONE and state.best_table:
        state.table = state.best_table
        state.weighted_total = state.best_total
    return state
