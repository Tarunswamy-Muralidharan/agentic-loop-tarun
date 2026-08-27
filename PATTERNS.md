# PATTERNS.md — write this yourself

<!-- The challenge requires this file to reflect YOUR thinking. AI-written
     reports are grounds for disqualification. Below are the questions it
     must answer — replace this file's contents with your own prose.

     Required by the challenge:
     1. One paragraph on EACH of: ReAct, Reflexion, Chain-of-Thought,
        Tree of Thoughts, LATS.
     2. Which pattern(s) THIS loop uses.
     3. Why that choice fits use case #24 (reformat text into a table).

     Prompts to get you started (answer from the code you can defend):
     - Where in loop.py is the ReAct skeleton? (reason=thought, act=action,
       observation fed back — point to the actual functions)
     - What makes this Reflexion-like rather than plain retry? (the
       validation score is a self-evaluation signal; the defect list and
       missed-facts list are the verbal feedback fed into the next
       extraction; the lesson store is the episodic component)
     - Where is it honestly NOT full Reflexion? (reflect is rule-based, and
       3 of the 4 validation dimensions are deterministic code, not an LLM
       self-critique; be upfront about this — it's a strength, argue it)
     - Why is Tree of Thoughts the wrong cost profile here? (branching
       multiple candidate tables multiplies LLM calls; a single
       extract-validate-repair path converges for this task)
     - When WOULD you reach for ToT or LATS instead? (e.g. ambiguous
       schema design where several column layouts are plausible) -->
