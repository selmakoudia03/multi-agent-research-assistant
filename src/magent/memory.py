"""Memory subsystems used by agents.

`ConversationMemory` is a simple ordered transcript (short-term memory).

`VectorMemory` is a small TF-IDF + cosine-similarity index (long-term /
semantic memory). TF-IDF is used instead of a neural embedding model on
purpose: it needs no downloaded weights and no network access, which keeps
the whole project runnable offline while still demonstrating a real
retrieval pipeline. Swapping in a neural embedder later only means
implementing the same `add` / `search` interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ConversationMemory:
    turns: list[dict[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})

    def as_text(self) -> str:
        return "\n".join(f"{t['role']}: {t['content']}" for t in self.turns)


@dataclass
class Document:
    text: str
    metadata: dict


class VectorMemory:
    """A minimal in-memory semantic index over short text documents."""

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def add(self, text: str, metadata: dict | None = None) -> None:
        self._documents.append(Document(text=text, metadata=metadata or {}))
        self._rebuild_index()

    def add_many(self, texts: list[str]) -> None:
        for text in texts:
            self._documents.append(Document(text=text, metadata={}))
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        if not self._documents:
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(doc.text for doc in self._documents)

    def search(self, query: str, k: int = 3) -> list[tuple[str, float, dict]]:
        """Return up to `k` documents most similar to `query`, best first."""
        if not self._documents or self._vectorizer is None:
            return []

        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix)[0]
        ranked = sorted(
            zip(self._documents, scores), key=lambda pair: pair[1], reverse=True
        )
        return [(doc.text, float(score), doc.metadata) for doc, score in ranked[:k] if score > 0]

    def __len__(self) -> int:
        return len(self._documents)
