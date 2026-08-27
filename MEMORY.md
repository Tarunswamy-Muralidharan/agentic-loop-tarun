# MEMORY.md — write this yourself

<!-- Same rule as PATTERNS.md: your own words only.

     The challenge requires: memory tool choice, structure, and a concrete
     example of memory in action.

     Questions your prose must answer:
     - Which backend and why? (Chroma persistent store; JSON fallback behind
       the same save/recall/clear interface — what did that choice cost?)
     - What exactly is stored? (the OUTCOME of each attempt: action —
       extract or validate — the weakest dimension targeted, score
       before/after, one-line lesson — not the table itself. Why outcomes
       and not tables?)
     - When is memory read and written? (recall at the start of reason via
       perceive; save after every reflect — why that ordering?)
     - The concrete example: run the agent twice on the same text and paste
       the actual recalled notes from run 2 showing it knew which defects a
       previous extraction had (e.g. "extract initial ... 3 rows" /
       "validate weakest=coverage; 1 defects, 2 missed facts"). Use real
       output, not an invented example.
     - How does the run survive a memory failure? (get_memory falls back to
       the JSON store and logs a warning — Milestone 3 requirement) -->
