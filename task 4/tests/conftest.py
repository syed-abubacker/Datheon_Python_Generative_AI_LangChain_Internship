import math
import os
import re
import zlib

import pytest
from langchain_core.embeddings import Embeddings

# Keep Chroma fully local during tests (no telemetry network calls).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


class HashingEmbeddings(Embeddings):
    """Deterministic, offline bag-of-words embeddings so retrieval can be tested without an API."""

    dimensions = 256

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(word.encode()) % self.dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeChatModel:
    """Stands in for ChatGoogleGenerativeAI; records prompts and streams fixed chunks."""

    prompts: list[str] = []
    pieces = ["Three ", "days ", "", "per week."]

    def __init__(self, **kwargs) -> None:
        pass

    def stream(self, prompt: str):
        type(self).prompts.append(prompt)
        for piece in type(self).pieces:
            yield type("Chunk", (), {"content": piece})()


@pytest.fixture
def offline_embeddings(monkeypatch):
    import rag

    monkeypatch.setattr(rag, "make_embeddings", lambda: HashingEmbeddings())


@pytest.fixture
def fake_chat_model(monkeypatch):
    import backend

    FakeChatModel.prompts = []
    monkeypatch.setattr(backend, "ChatGoogleGenerativeAI", FakeChatModel)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")
    return FakeChatModel


@pytest.fixture
def docs_folder(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "remote.txt").write_text(
        "Remote work policy: employees may work remotely three days per week with manager approval.",
        encoding="utf-8",
    )
    (folder / "expenses.md").write_text(
        "Expense policy: submit travel receipts within thirty days for reimbursement.", encoding="utf-8"
    )
    return folder
