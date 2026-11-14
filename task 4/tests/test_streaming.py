import pytest
from langchain_core.documents import Document

import backend
from backend import NOT_ENOUGH_CONTEXT, build_index, retrieved_sources, stream_answer


def test_no_documents_yields_fallback_without_needing_a_key(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert list(stream_answer("anything?", [], [])) == [NOT_ENOUGH_CONTEXT]


def test_missing_api_key_raises_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        list(stream_answer("question", [], [Document("context")]))


def test_stream_yields_model_chunks_and_skips_empty_ones(fake_chat_model) -> None:
    chunks = list(stream_answer("How many days?", [], [Document("Remote work is three days.")]))
    assert chunks == ["Three ", "days ", "per week."]


def test_prompt_contains_context_and_only_the_last_six_history_messages(fake_chat_model) -> None:
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"message-{i}"} for i in range(8)]
    list(stream_answer("Follow-up?", history, [Document("Policy text here.")]))
    prompt = fake_chat_model.prompts[0]
    assert "Policy text here." in prompt and "User question: Follow-up?" in prompt
    assert "message-0" not in prompt and "message-1" not in prompt
    assert "message-2" in prompt and "message-7" in prompt


def test_build_index_reuses_task_2_pipeline_and_retrieval_cites_sources(
    docs_folder, tmp_path, offline_embeddings
) -> None:
    import rag

    index = tmp_path / "index"
    document_count, chunk_count = build_index(docs_folder, index)
    assert (document_count, chunk_count) == (2, 2)
    store = rag.open_vector_store(index)
    documents, sources = retrieved_sources(store, "remotely three days per week")
    assert sources == ["remote.txt, page 1"]
    assert len(documents) == 1


def test_rebuilding_the_index_does_not_duplicate_chunks(docs_folder, tmp_path, offline_embeddings) -> None:
    import rag

    index = tmp_path / "index"
    build_index(docs_folder, index)
    build_index(docs_folder, index)
    assert rag.open_vector_store(index)._collection.count() == 2


def test_backend_loads_the_task_2_rag_module() -> None:
    assert backend.load_documents.__module__ == "rag"
