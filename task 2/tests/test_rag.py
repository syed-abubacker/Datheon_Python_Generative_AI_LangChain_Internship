from langchain_core.documents import Document

from rag import NOT_ENOUGH_CONTEXT, answer_question, source_labels, split_documents


class EmptyStore:
    def similarity_search_with_relevance_scores(self, question: str, k: int):
        return []


def test_no_context_does_not_call_llm() -> None:
    answer = answer_question(EmptyStore(), "What is the policy?")
    assert answer.text == NOT_ENOUGH_CONTEXT
    assert answer.sources == []


def test_source_labels_convert_zero_based_pdf_pages() -> None:
    documents = [Document("x", metadata={"source": "/tmp/policy.pdf", "page": 2})]
    assert source_labels(documents) == ["policy.pdf, page 3"]


def test_split_preserves_metadata() -> None:
    chunks = split_documents([Document("hello " * 300, metadata={"source": "notes.txt", "page": 0})])
    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "notes.txt" for chunk in chunks)
