"""CLI entry point: load config + keys, build the pieces, run the loop.

Usage:
    python main.py                        # live run on data/sample_input.md
    python main.py --offline              # no key, no network (FakeLLM)
    python main.py --input path/to.md     # different draft
    python main.py --clear-memory         # wipe long-term memory first
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from agent.harness import FakeLLM, GroqLLM, TokenBudget, load_config
from agent.logger import StepLogger
from agent.loop import run
from agent.memory_manager import get_memory
from agent.schemas import Status
from agent.tools import TOOL_SCHEMAS, build_registry

GOAL = "Improve the draft until it meets the rubric target."


def main() -> int:
    parser = argparse.ArgumentParser(description="Rubric-improvement agent (use case #23)")
    parser.add_argument("--input", default="data/sample_input.md")
    parser.add_argument("--offline", action="store_true",
                        help="deterministic FakeLLM: no key, no network")
    parser.add_argument("--clear-memory", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config()
    if args.max_iterations:
        cfg["loop"]["max_iterations"] = args.max_iterations

    logger = StepLogger(cfg["logging"]["path"])
    budget = TokenBudget(cfg["budget"]["max_total_tokens"], cfg["budget"]["warn_at_tokens"])

    if args.offline:
        llm = FakeLLM()
        llm.budget = budget                      # same interface as GroqLLM
    else:
        if not os.getenv("GROQ_API_KEY"):
            print("GROQ_API_KEY missing — put it in .env, or run with --offline")
            return 2
        llm = GroqLLM(cfg, budget, logger)

    memory = get_memory(cfg, logger)
    if args.clear_memory:
        memory.clear()
        print("memory cleared")

    with open(args.input, encoding="utf-8") as f:
        draft = f.read().strip()

    print(f"— run start: {len(draft)} chars, backend={type(llm).__name__}, "
          f"memory={type(memory).__name__}\n")
    state = run(GOAL, draft, llm, build_registry(llm), TOOL_SCHEMAS, memory, logger, cfg)

    print("\n" + "=" * 62)
    print(f"status       : {state.status.value}")
    print(f"iterations   : {state.iteration}")
    print(f"final score  : {state.weighted_total}  (threshold {cfg['loop']['score_threshold']})")
    print(f"tokens used  : {budget.total}")
    print(f"log file     : {logger.path}")
    print("=" * 62)
    print("\n--- original ---\n" + state.original)
    print("\n--- final draft ---\n" + state.draft)
    return 0 if state.status in (Status.DONE, Status.PARTIAL) else 1


if __name__ == "__main__":
    sys.exit(main())
