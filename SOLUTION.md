# Building an Agentic Loop from Scratch — Solution Document

**Candidate:** Tarunswamy Muralidharan
**Use case #24:** Reformat text into a structured table
**Repository:** `agentic-loop-tarun` (private) · Python 3.10+ · LLM: Groq `openai/gpt-oss-120b` (fallbacks: `gpt-oss-20b`, `qwen3.8-27b`) · No agent frameworks

---

## 1. The use case, and why I designed the loop this way

My agent takes messy prose — for example a paragraph describing trip
expenses — and turns it into a structured table of columns and rows.

The honest problem with this use case is that a one-shot LLM call can
produce *a* table easily. What it cannot produce is a table you can
**trust**: models silently drop facts, leave cells empty, and occasionally
invent values that were never in the text. So I designed the loop around a
single idea: **the agent must prove its table is faithful before it is
allowed to claim success.**

The loop is a validator-driven repair cycle. The agent drafts a table,
measures it against the source on four weighted dimensions, feeds the
specific defects back into a repair extraction, re-measures, and ends the
run only when the weighted score clears a configurable target (4.2/5) — or
when it judges, on its own, that the input has nothing tabular in it at all
and refuses with an explanation.

The validator is deliberately a hybrid. Three of its four dimensions are
plain code: **groundedness** (weight 0.35 — every non-empty cell value must
appear in the source text, checked by normalized substring match; this is
the anti-hallucination control), **completeness** (0.25 — fraction of
filled cells), and **consistency** (0.10 — uniform columns, no duplicate
rows). Only **coverage** (0.30 — "which facts in the source never made it
into the table?") uses a model, because it genuinely requires reading
comprehension. My reasoning: extraction is copying, so most quality
questions here are factual, and a deterministic check is free, instant, and
cannot be talked into a generous grade — whereas a model evaluating its own
output can be.

## 2. Architecture: the decisions live in the model, the guarantees in code

The final architecture is conversation-driven. One persistent message
history is the agent's working memory: the system prompt establishes its
goal and capabilities once, and from then on every tool call the model
makes and every raw result it receives is appended to the conversation.
Each turn, the model re-reads its own history, writes its thinking in
explicit REFLECT / PERCEIVE / REASON phases, and makes exactly one tool
call. It has four tools: `extract_table`, `validate_table`,
`recall_memory`, and `finish_task` — so even *ending the run* is the
agent's own explicit decision, with two possible verdicts: `success`, or
`not_tabulatable` when it judges the input has nothing to extract.

The four required functions exist as the anatomy of a single model turn,
not as staged cognition:

- **perceive** maintains what the agent can see — the conversation, plus
  the automatic long-term-memory read at the start of each reasoning turn.
- **reason** is the model turn: the whole conversation plus the tool
  schemas go in; the model's decision comes out.
- **act** executes exactly one requested tool through a registry that
  doubles as an allowlist — an invented tool name is refused, never
  dispatched, and a handler exception becomes a structured error
  observation the agent reads and routes around.
- **reflect** is bookkeeping and governance only. It tracks the best
  validated table and enforces the system's only two hard rules, but it
  never tells the model what to do next.

Those two rules are what I call "the model proposes, the gate disposes":
a **success claim is rejected unless a validated score meets the target**,
and a **refusal is rejected once a validated table at target already
exists**. Everything else — when to extract, when to measure, when to
consult memory, when to stop — is the model's judgement. I kept exactly
these two in code because they are evidence rules, not choreography: an
agent should be free in its strategy but not free to declare victory
without proof, or to discard proven work on a whim.

This shape was not my first design, and I think the honest story of how it
got here matters more than the final diagram — I tell it in section 6.

## 3. Patterns: Reflexion on a ReAct skeleton

The skeleton is **ReAct**: reasoning interleaved with actions, each
observation fed back before the next thought. In my implementation the
Thought is the model's written REFLECT/PERCEIVE/REASON phases, the Action
is its single tool call, and the Observation is the raw tool result
appended to the conversation.

What makes the loop converge is **Reflexion**: the agent evaluates its own
attempt and feeds verbal feedback into the next one. The validation score
is the self-evaluation signal; the defect list and missed-facts list are
the verbal feedback, passed into the repair extraction's `feedback`
argument; and a persistent lesson store is the episodic memory of what was
tried and what moved the score.

I deviate from textbook Reflexion in one deliberate way: the critique is
mostly *measured*, not self-generated — three of four dimensions are
deterministic code. I consider that a strength for an extraction task,
where the quality questions are factual, and I would defend it as such.

I considered and rejected **Tree of Thoughts** and **LATS**: validation
tells my agent exactly what is wrong with its current table, so there is no
wide solution space to search, and branching candidate tables would
multiply model calls for little gain. The one sub-problem where ToT would
genuinely fit is schema design — during live runs my model once
spontaneously added a `currency` column, which showed me that several
column layouts can be plausible; generating and comparing candidate
schemas would be a reasonable extension. **Chain-of-Thought** is present as
a technique (the thinking phases) but is not an architecture.

## 4. Memory: outcomes, not artifacts

Memory is ChromaDB as a persistent local vector store, with a JSON-file
fallback behind the same `save()` / `recall()` / `clear()` interface — if
Chroma fails to initialize, the run degrades gracefully rather than dying.

I store the **outcome of each attempt, never the table itself**: the action
taken, what it targeted, the score before and after, and a one-line lesson
such as *"iteration 2: validate target=coverage score None → 3.96
(weakest=coverage; 1 defects, 2 missed facts)."* A stale table is clutter;
"this kind of repair moved the score" is reusable knowledge.

Memory is read automatically at the start of every reasoning turn and
injected into the conversation (only when the recalled set has actually
changed), and the agent additionally has a `recall_memory` tool to consult
its history whenever it judges that worth a turn. Writing happens after
every reflect. Each record carries a sha256 fingerprint of the source text,
and recall is strictly scoped to the same source — I added this after
watching a refusal lesson from a nonsense-input run get recalled into runs
on real data, where it could have biased the agent toward refusing valid
input.

The concrete demonstration: run the agent twice on the same input, in two
separate processes. Run one starts with zero recalled lessons; run two's
very first turn recalls three lessons from run one's on-disk store, and the
console prints their text — the agent begins its second session already
knowing how the first went.

## 5. Harness: every mechanism maps to a failure I can name

**Retries.** Hand-rolled exponential backoff with full jitter for rate
limits, timeouts and 5xx. Auth errors deliberately do not retry. Jitter
exists because synchronized retries become their own load spike.

**Model fallback chain.** The configured model list is tried in order. This
earned its keep the day I built it: both Groq models I had originally
configured had been retired and returned 404. Because model names live in
`config.yaml`, the fix was a config edit with zero code changes — which is
also why every other runtime knob (caps, thresholds, budget, patience,
timeout, memory backend) lives in config and nothing tunable is hardcoded
in the loop.

**Malformed output.** Unparseable extractor JSON gets one retry with a
simplified prompt; defensive parsing strips markdown fences and takes the
last JSON block. Malformed tool arguments become a visible decision, not a
crash.

**Failed tools.** Graceful error observations, fed back to the agent.

**Memory failures.** Read failures warn and continue without memory; write
failures warn and lose one lesson. I fault-tested with a store that throws
on every call — the run completes end-to-end.

**Guardrails.** A hard iteration cap (10). A token budget that warns and
then hard-stops. Stuck detection: an identical reflection outcome twice in
a row ends the run — and if the target was already met when the cycling
began, the run ends DONE with the best table rather than STUCK, because
cycling after success is waste, not failure. A patience rule: once a
validated table meets the target, the agent has a bounded window (4
iterations) to beat its best score, after which the harness returns the
best table. An extraction-streak notice tells the agent when it has drafted
three tables in a row without measuring any of them. Terminal statuses are
explicit — DONE / REFUSED / PARTIAL / STUCK / FAILED — and PARTIAL always
returns the best validated table honestly.

**Observability.** One JSON object per step (timestamp, iteration, step,
input/output summaries, latency, error) as JSON Lines, so a crashed run
still leaves a readable log. The model's thinking is logged as a
`cognition` event, so any run reads back as a story: what the agent
thought, what it did, what happened.

**Offline mode.** A scripted fake model behind the same interface — my test
double during development and my insurance against network failure during a
demo. Its honest limit: it cannot think; it cannot handle a novel input.

## 6. What didn't work, and what I learned from it

This project went through three architectures in one day, each forced by
real feedback or a real failure, and I think this section is the most
important one in the document.

**Version one was a workflow wearing an agent's clothes.** My prompt
commanded a fixed sequence — extract, then validate, then repair — and my
reflect function issued instructions. An evaluator called it what it was.
The lesson: putting an LLM inside a loop does not make an agent; the test
is *where the decisions live*.

**Version two moved the decisions into the model but summarized its world.**
Each iteration I rebuilt a compact "facts sheet" for the model instead of
showing it its own history. Two failures came out of this. First, after
each new extraction I left the *previous* table's defect list visible, and
the model chased those stale defects — re-extracting seven times without
ever re-validating, until the iteration cap fired. Second, a perfectionism
loop: the model validated at 4.74, which already met the target, but kept
"improving" the table — and made it objectively worse, 4.74 → 4.55 → 4.18.
Both were diagnosed from the JSONL logs. The first taught me that state you
show the model *is* the agent's reality — stale state means a deluded
agent. The second produced the patience guardrail: autonomy over strategy,
bounded economics.

**Version three exposed the deepest bug by fixing the architecture.** When
I switched to a persistent conversation and asked the model to think in
explicit phases, its refusal of a nonsense input arrived as *prose* rather
than a tool call — and my leftover code treated any plain-text reply as a
success claim and rejected it. That misreading pushed the model away from
the refusal it was correctly trying to make, and it went on to extract a
junk table from the gibberish — which my validator scored 4.88, because
every cell of nonsense was, technically, grounded in the nonsense source.
One incident, two lessons. A harness that misinterprets the model's
thinking as claims will coerce the agent into exactly the behaviour it was
trying to avoid — the fix was a neutral "no tool was called" reminder, with
endings accepted only through the finish tool. And my validator measures
**faithfulness, not worthiness**: it can prove a table is honest to the
source, but not that the source deserved a table. Today, only the agent's
judgement guards that gap — it passes my probes (pure gibberish, and the
bait input "this is tabulatable"), but it is model-dependent rather than
guaranteed.

**What I would do differently with more time.** Add a task-worthiness
dimension to the validator so the degenerate-table gap is closed by
measurement, not judgement. Replace the strict substring grounding with
token-level fuzzy matching — in live runs it false-flagged honest
paraphrases ("cab from airport to hotel" versus the source's "cab from
*the* airport") and cost an undeserved 0.75 of groundedness. Scope memory
clearing per session, and let lessons transfer between *similar* sources,
not just identical ones. And build a small evaluation set of inputs with
known-correct tables, so changes to prompts or thresholds could be scored
instead of eyeballed.

## 7. Closing

I used AI tools to help write and debug the code, as the rules allow — and
the three failures above were found in my own live runs, diagnosed from my
own logs, and each fix is a design decision I can defend line by line. What
I am submitting is not the system I planned in the morning; it is the
system that survived the day. I think that is what an agentic loop is
actually for.
