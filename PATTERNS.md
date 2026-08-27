# PATTERNS.md — Agentic Patterns Research

## The five patterns

**ReAct (Reason + Act).** The model alternates between reasoning in text and
taking actions, with each action's observation fed back before the next
thought. The loop is: Thought → Action → Observation → Thought → … It solves
the problem of a model acting blindly: every step is taken in light of what
the previous step actually returned. Its cost is one model call per step. It
is the wrong choice only when the task is genuinely single-shot — for
anything multi-step it is the baseline pattern almost everything else builds
on.

**Reflexion.** The agent evaluates its own attempt, turns the evaluation into
verbal feedback ("the table missed two facts"), and feeds that feedback into
the next attempt. The distinction from plain retry is that a retry repeats
the same attempt and hopes; Reflexion repeats the attempt *informed by a
critique of the last one*. It usually also keeps an episodic memory of past
failures so later attempts avoid repeating them. Cost: an extra evaluation
step per cycle. It is wrong for tasks with no measurable quality signal —
without a real evaluation, the "reflection" is just the model agreeing with
itself.

**Chain-of-Thought (CoT).** Prompting the model to reason step by step in
text before answering. Important distinction: CoT is a *prompting technique*,
not an architecture — it changes what happens inside one model call, not how
calls are orchestrated. It is nearly free and helps almost everywhere, but it
does not give a system tools, memory, or the ability to check its own work.

**Tree of Thoughts (ToT).** Instead of one reasoning path, the model
generates several candidate paths, evaluates them, prunes the weak ones, and
expands the promising ones — a search tree over thoughts. It shines when the
solution space is wide and a wrong early choice is expensive (puzzles,
planning). Its cost is multiplicative: every branch is a model call, so a
tree of depth d and branching b costs on the order of b^d calls.

**LATS (Language Agent Tree Search).** ToT extended into a full Monte Carlo
Tree Search over *agent trajectories* — sequences of reasoning and tool
actions — with value estimates and backpropagation deciding which branches
to explore. It is the most powerful of the five and by far the most
expensive; for most well-scoped tasks it is overkill.

## What my loop uses, and why

My agent is a **Reflexion loop built on a ReAct skeleton**.

The ReAct part is structural: each turn the model writes its reasoning
(REFLECT / PERCEIVE / REASON phases in its message), takes exactly one
action (a tool call), and sees the observation appended to the conversation
before its next turn.

The Reflexion part is what makes the loop converge for use case #24. The
validator's weighted score is the self-evaluation signal; the defect list
and missed-facts list are the verbal feedback, which flows into the next
`extract_table` call as the `feedback` argument; and the lesson store is the
episodic memory — outcomes of past attempts ("repair targeting coverage
moved 2.0 → 5.0") are recalled at the start of reasoning and available
on demand through the `recall_memory` tool.

I should be honest about where this is *not* textbook Reflexion: in the
classic pattern the critique itself comes from the model. In mine, three of
the four evaluation dimensions are deterministic code (groundedness,
completeness, consistency), and only coverage is judged by a model. I did
that deliberately, and I consider it a strength for this use case:
extraction is copying, so "does this cell value exist in the source?" is a
factual question a substring check answers for free and cannot be talked out
of, whereas a model grading its own table can be generous to itself. The
verbal-feedback part of Reflexion survives intact — the agent still receives
a critique and acts on it — but the critique is grounded in measurement.

## Why not the others

CoT is present but insufficient on its own — my agent does think step by
step in its phases, but the task needs tools, iteration and memory, which
CoT does not provide.

ToT is the wrong cost profile here. The extract → validate → repair path is
effectively a single trajectory with feedback; validation tells the agent
*exactly* what is wrong, so there is no wide solution space to search.
Branching three candidate tables and validating each would triple the cost
per cycle for little gain. The one place ToT would genuinely fit this use
case is schema design — when several column layouts are plausible (my model
once added a `currency` column unprompted), generating and comparing two or
three candidate schemas before extracting would be a reasonable future
extension. LATS is the same argument, amplified.
