from magent.memory import VectorMemory


def test_search_ranks_relevant_document_first() -> None:
    memory = VectorMemory()
    memory.add_many(
        [
            "Retrieval-Augmented Generation combines a retriever with a generator LLM.",
            "The Eiffel Tower is a landmark in Paris, France.",
            "Vector databases store embeddings for nearest-neighbor search.",
        ]
    )

    results = memory.search("What is RAG and retrieval?", k=2)

    assert results
    top_text, top_score, _metadata = results[0]
    assert "Retrieval-Augmented" in top_text
    assert top_score > 0


def test_search_on_empty_memory_returns_empty_list() -> None:
    memory = VectorMemory()
    assert memory.search("anything") == []


def test_len_tracks_added_documents() -> None:
    memory = VectorMemory()
    memory.add("first document")
    memory.add("second document")
    assert len(memory) == 2
