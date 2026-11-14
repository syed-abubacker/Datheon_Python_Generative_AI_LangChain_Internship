from pathlib import Path

import pytest
from langchain_core.documents import Document

import rag
from rag import (
    NOT_ENOUGH_CONTEXT,
    answer_question,
    create_vector_store,
    load_documents,
    make_embeddings,
    open_vector_store,
    retrieve_documents,
    split_documents,
)

SAMPLE_DOCUMENTS = Path(__file__).resolve().parents[1] / "sample_documents"


def minimal_pdf(pages: list[str]) -> bytes:
    """Build a tiny valid text PDF (one text line per page) without extra dependencies."""
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{4 + 2 * index} 0 R" for index in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for index, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * index} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    output = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(output)
    output += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        output += f"{offset:010d} 00000 n \n".encode()
    output += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(output)


@pytest.fixture
def corpus(tmp_path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "remote.txt").write_text(
        "Remote work policy: employees may work remotely three days per week with manager approval.",
        encoding="utf-8",
    )
    (folder / "expenses.md").write_text(
        "Expense policy: submit travel receipts within thirty days for reimbursement.", encoding="utf-8"
    )
    (folder / "ignored.csv").write_text("a,b,c", encoding="utf-8")
    return folder


class FakeLLM:
    def __init__(self, content="The answer is three days."):
        self.content = content
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return type("Response", (), {"content": self.content})()


def test_load_documents_reads_supported_files_only(corpus) -> None:
    documents = load_documents(corpus)
    assert sorted(Path(doc.metadata["source"]).name for doc in documents) == ["expenses.md", "remote.txt"]
    assert all(doc.metadata["page"] == 0 for doc in documents)


def test_load_documents_errors(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does-not-exist")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No .pdf"):
        load_documents(tmp_path / "empty")


def test_shipped_sample_document_loads_and_chunks() -> None:
    chunks = split_documents(load_documents(SAMPLE_DOCUMENTS))
    assert chunks and "remotely" in chunks[0].page_content


def test_pdf_pages_load_with_zero_based_metadata_and_one_based_citations(tmp_path) -> None:
    (tmp_path / "handbook.pdf").write_bytes(minimal_pdf(["Alpha onboarding steps", "Beta security rules"]))
    documents = load_documents(tmp_path)
    assert [doc.metadata["page"] for doc in documents] == [0, 1]
    assert "Beta security rules" in documents[1].page_content
    assert rag.source_labels(documents) == ["handbook.pdf, page 1", "handbook.pdf, page 2"]


def test_make_embeddings_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        make_embeddings()


def test_open_vector_store_requires_existing_index(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="ingest.py"):
        open_vector_store(tmp_path / "missing-index")


def test_retrieve_documents_validates_and_filters_by_score() -> None:
    class ScoredStore:
        def similarity_search_with_relevance_scores(self, question: str, k: int):
            return [(Document("relevant"), 0.80), (Document("borderline"), 0.35), (Document("weak"), 0.10)]

    with pytest.raises(ValueError):
        retrieve_documents(ScoredStore(), "  ")
    kept = retrieve_documents(ScoredStore(), "question")
    assert [doc.page_content for doc in kept] == ["relevant", "borderline"]


def test_chroma_ingest_retrieve_and_cite_end_to_end(corpus, tmp_path, offline_embeddings) -> None:
    index = tmp_path / "index"
    chunks = split_documents(load_documents(corpus))
    create_vector_store(chunks, index)
    store = open_vector_store(index)

    llm = FakeLLM()
    answer = answer_question(store, "How many days per week may employees work remotely?", llm=llm)
    assert answer.text == "The answer is three days."
    assert answer.sources == ["remote.txt, page 1"]
    assert "three days per week" in llm.prompts[0]
    assert "travel receipts" not in llm.prompts[0]


def test_unrelated_question_gets_no_context_answer_and_no_llm_call(corpus, tmp_path, offline_embeddings) -> None:
    index = tmp_path / "index"
    create_vector_store(split_documents(load_documents(corpus)), index)
    llm = FakeLLM()
    answer = answer_question(open_vector_store(index), "quantum chromodynamics lattice gauge", llm=llm)
    assert answer.text == NOT_ENOUGH_CONTEXT
    assert answer.sources == [] and llm.prompts == []


def test_reingesting_replaces_the_index_instead_of_duplicating(corpus, tmp_path, offline_embeddings) -> None:
    index = tmp_path / "index"
    chunks = split_documents(load_documents(corpus))
    create_vector_store(chunks, index)
    store = create_vector_store(chunks, index)
    assert store._collection.count() == len(chunks)


def test_blank_llm_reply_falls_back_to_not_enough_context(corpus, tmp_path, offline_embeddings) -> None:
    index = tmp_path / "index"
    create_vector_store(split_documents(load_documents(corpus)), index)
    answer = answer_question(open_vector_store(index), "remotely three days per week", llm=FakeLLM("   "))
    assert answer.text == NOT_ENOUGH_CONTEXT
