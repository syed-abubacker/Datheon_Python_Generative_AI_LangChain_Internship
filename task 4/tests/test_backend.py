from langchain_core.documents import Document

from backend import retrieved_sources


class FakeStore:
    def similarity_search_with_relevance_scores(self, question: str, k: int):
        return [(Document("Remote work is three days.", metadata={"source": "policy.pdf", "page": 1}), 0.91)]


def test_backend_reuses_retrieval_metadata_for_citations() -> None:
    documents, sources = retrieved_sources(FakeStore(), "How many days?")
    assert len(documents) == 1
    assert sources == ["policy.pdf, page 2"]
