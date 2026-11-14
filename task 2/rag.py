"""Reusable, grounded RAG functions shared by Tasks 2 and 4."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_PERSIST_DIRECTORY = Path(__file__).parent / "chroma_db"
NOT_ENOUGH_CONTEXT = "I don't have enough context in the loaded documents to answer that."


@dataclass
class RAGAnswer:
    text: str
    sources: list[str]
    documents: list[Document]


def load_documents(documents_directory: str | Path) -> list[Document]:
    """Load supported files and retain source and zero-based page metadata."""
    root = Path(documents_directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {root}")
    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        elif path.suffix.lower() in {".txt", ".md"}:
            loaded = TextLoader(str(path), encoding="utf-8").load()
        else:
            continue
        for document in loaded:
            document.metadata["source"] = str(path)
            document.metadata.setdefault("page", 0)
        documents.extend(loaded)
    if not documents:
        raise ValueError("No .pdf, .txt, or .md documents were found.")
    return documents


def split_documents(documents: Iterable[Document]) -> list[Document]:
    """Split content while carrying each input document's metadata into every chunk."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    return splitter.split_documents(list(documents))


def make_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing. Configure the root .env file.")
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )


def create_vector_store(chunks: list[Document], persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY) -> Chroma:
    """Embed chunks and create a persistent, local Chroma collection."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=make_embeddings(),
        persist_directory=str(Path(persist_directory)),
        collection_name="internship_documents",
    )


def open_vector_store(persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY) -> Chroma:
    directory = Path(persist_directory)
    if not directory.exists():
        raise FileNotFoundError("No local index found. Run ingest.py before asking questions.")
    return Chroma(
        collection_name="internship_documents",
        embedding_function=make_embeddings(),
        persist_directory=str(directory),
    )


def retrieve_documents(vector_store: Any, question: str, minimum_score: float = 0.35) -> list[Document]:
    """Return only relevant chunks; an empty result prevents ungrounded answers."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    scored_documents = vector_store.similarity_search_with_relevance_scores(question, k=4)
    return [document for document, score in scored_documents if score >= minimum_score]


def source_labels(documents: Iterable[Document]) -> list[str]:
    """Make human-readable citations from source and page metadata."""
    labels: list[str] = []
    for document in documents:
        source = Path(str(document.metadata.get("source", "unknown"))).name
        page = int(document.metadata.get("page", 0)) + 1
        label = f"{source}, page {page}"
        if label not in labels:
            labels.append(label)
    return labels


def answer_question(vector_store: Any, question: str, llm: Any | None = None) -> RAGAnswer:
    """Generate an answer only from retrieved text, then attach page citations."""
    documents = retrieve_documents(vector_store, question)
    if not documents:
        return RAGAnswer(NOT_ENOUGH_CONTEXT, [], [])
    context = "\n\n---\n\n".join(document.page_content for document in documents)
    prompt = f"""Answer the question using only the context below.
If the context does not contain the answer, say exactly: {NOT_ENOUGH_CONTEXT}
Do not use outside knowledge. Be concise.

Context:
{context}

Question: {question}
"""
    llm = llm or ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0
    )
    response = llm.invoke(prompt)
    text = str(response.content).strip()
    if not text:
        text = NOT_ENOUGH_CONTEXT
    return RAGAnswer(text, source_labels(documents), documents)
