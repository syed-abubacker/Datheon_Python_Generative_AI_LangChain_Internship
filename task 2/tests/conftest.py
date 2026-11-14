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


@pytest.fixture
def offline_embeddings(monkeypatch):
    import rag

    monkeypatch.setattr(rag, "make_embeddings", lambda: HashingEmbeddings())
    return HashingEmbeddings()
