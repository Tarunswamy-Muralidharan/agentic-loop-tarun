# MEMORY.md — Memory Integration

## What I chose and why

I use **ChromaDB as a persistent local vector store**, with a plain
**JSON-file store as an automatic fallback** behind the exact same
interface. Both live in `agent/memory_manager.py` and expose the three
required operations: `save()`, `recall()`, `clear()`.

Why Chroma over mem0 or Zep: it runs fully local with no account, no API
key, and no network dependency — in a timed build, every external service
is a schedule risk. Why the JSON fallback exists: Chroma's dependency tree
is heavy (onnxruntime for embeddings), and if it fails to initialize for
any reason, `get_memory()` logs a warning and silently degrades to the JSON
store. The graded property — memory that persists across sessions and
influences behaviour — survives either way; the embedding math is a detail
behind the interface.

## What is stored

I store the **outcome of each attempt, never the table itself**. A record
carries: the action taken (extract / validate / refuse / finish), what it
targeted (e.g. the weakest dimension), the score before and after, and a
one-line lesson. Flattened, a lesson reads like:

    iteration 2: action=validate target=coverage score None -> 3.96
    (weakest=coverage; 1 defects, 2 missed facts)

Why outcomes and not tables: a stale table is clutter, but "repairing
coverage moved the score from 2.0 to 5.0" is reusable knowledge — it tells
a future run which kind of repair pays off. Each record also carries a
**sha256 fingerprint of the source text**, and recall is strictly scoped to
the same source. I added that after watching cross-input pollution happen:
a "judged not tabulatable" lesson from a gibberish run was being recalled
into runs on real data, where it could bias the agent toward refusing valid
input. With source scoping, lessons from one input can never leak into
another.

## When memory is read and written

Reading happens in `perceive`, at the start of every reasoning turn — the
recalled lessons are injected into the conversation (in full on the first
turn, and afterwards only when the recalled set actually changes, so the
agent is informed without being spammed). On top of the automatic read, the
agent has a `recall_memory` tool and can consult its own history whenever
it decides that is worth a turn.

Writing happens after every `reflect` — one record per iteration that did
something worth remembering. This ordering is the only one that makes
sense: you can only summarise an iteration after it happened, and the
summary is only useful before the next decision.

## A concrete example of memory changing behaviour

Run the agent twice on the same input:

    python main.py --offline --clear-memory --input data/expenses.md
    python main.py --offline --input data/expenses.md

In run 1, iteration 1 recalls **0 lessons** (memory was just cleared). In
run 2 — a completely fresh process — iteration 1 recalls **3 lessons** from
run 1's Chroma store on disk, and the console prints the actual lesson
text, e.g. that the previous session's initial extraction was incomplete
and the repair that fixed it. The agent starts its second session already
knowing how the first one went. The same is visible live: after a refusal
run on nonsense input, a later run on that same input recalls the refusal
lesson and the model cites it in its reasoning.

## Failure behaviour

Memory is useful but must never be load-bearing. A `recall()` failure
mid-run logs a warning and continues with no lessons; a `save()` failure
logs a warning and loses one lesson. I fault-tested this with a store that
throws on every single call: the run still completed end-to-end. If Chroma
cannot even start, the JSON fallback takes over with identical semantics.

## Honest limitations

At demo scale the corpus is a handful of lessons, so recalling "the 3 most
relevant" is close to recalling everything — semantic ranking earns its
keep only as the corpus grows. And `clear()` currently wipes the whole
store rather than one session; records carry no session id. With more time
I would add a session id to each record and make `clear()` session-scoped,
and prefer lessons whose source *resembles* (not just equals) the current
input, so knowledge could transfer between similar documents safely.
