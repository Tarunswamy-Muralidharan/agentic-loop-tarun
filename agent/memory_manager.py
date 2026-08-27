"""Long-term memory: save() / recall() / clear() over a persistent store.

Primary backend is ChromaDB with on-disk persistence (semantic recall).
Fallback is a JSON file with recency + keyword-overlap recall — same
interface, zero dependencies. The graded property is that memory survives
a process restart and visibly informs the next run; the backend is a detail
behind this interface.

Wiring (Milestone 2 requirement):
    recall() is called at the start of reason (via perceive's Percept),
    save() is called after every reflect.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any


class BaseMemory:
    def save(self, record: dict[str, Any]) -> None: ...
    def recall(self, query: str, k: int = 3) -> list[str]: ...
    def clear(self) -> None: ...


def _to_note(record: dict[str, Any]) -> str:
    """Flatten a structured record into the one-line lesson the LLM will read.

    We store the OUTCOME of an attempt, not the text itself: 'tried X, score
    moved Y' is what stops iteration N+1 repeating iteration N's mistake.
    """
    return (f"iteration {record.get('iteration')}: action={record.get('action')} "
            f"target={record.get('target')} score {record.get('score_before')} "
            f"-> {record.get('score_after')} ({record.get('lesson')})")


class ChromaMemory(BaseMemory):
    def __init__(self, path: str, collection: str) -> None:
        import chromadb                      # imported here so json mode never needs it
        from chromadb.config import Settings
        self.client = chromadb.PersistentClient(
            path=path, settings=Settings(anonymized_telemetry=False))
        self.col = self.client.get_or_create_collection(collection)

    def save(self, record: dict[str, Any]) -> None:
        self.col.add(
            ids=[str(uuid.uuid4())],
            documents=[_to_note(record)],
            metadatas=[{k: v for k, v in record.items()
                        if isinstance(v, (str, int, float, bool))}],
        )

    def recall(self, query: str, k: int = 3) -> list[str]:
        if self.col.count() == 0:
            return []
        res = self.col.query(query_texts=[query], n_results=min(k, self.col.count()))
        return res["documents"][0] if res["documents"] else []

    def clear(self) -> None:
        name = self.col.name
        self.client.delete_collection(name)
        self.col = self.client.get_or_create_collection(name)


class JsonMemory(BaseMemory):
    """Dependency-free fallback: same contract, recency + keyword overlap."""

    def __init__(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.file = os.path.join(path, "memory.json")

    def _load(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.file):
            return []
        with open(self.file, encoding="utf-8") as f:
            return json.load(f)

    def save(self, record: dict[str, Any]) -> None:
        rows = self._load()
        rows.append({"ts": time.time(), "note": _to_note(record), **record})
        with open(self.file, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)

    def recall(self, query: str, k: int = 3) -> list[str]:
        words = set(query.lower().split())
        rows = self._load()
        scored = sorted(
            rows,
            key=lambda r: (len(words & set(r["note"].lower().split())), r["ts"]),
            reverse=True,
        )
        return [r["note"] for r in scored[:k]]

    def clear(self) -> None:
        if os.path.exists(self.file):
            os.remove(self.file)


def get_memory(cfg: dict[str, Any], logger=None) -> BaseMemory:
    """Factory with the Milestone 3 fallback: on memory init failure,
    continue with the JSON backend and log a warning — never crash the run."""
    mem_cfg = cfg["memory"]
    if mem_cfg["backend"] == "chroma":
        try:
            return ChromaMemory(mem_cfg["path"], mem_cfg["collection"])
        except Exception as exc:                       # noqa: BLE001 — any init failure
            if logger:
                logger.event("memory_fallback",
                             f"chroma unavailable ({exc}); using JSON store")
    return JsonMemory(mem_cfg["path"])
