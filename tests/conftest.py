"""Shared test setup.

The app reads GROQ_API_KEY and loads a Sentence Transformers model at import
time, so both are faked here *before* `api` / `rag` / `ingest` are imported.
The Groq client is replaced with a fake, so tests need no API key and make no
network calls.
"""
import hashlib
import os
import re
import socket
import sys
import tempfile
import types
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# api.py mounts StaticFiles(directory="static"), which is resolved against the cwd.
os.chdir(ROOT)

os.environ["GROQ_API_KEY"] = "test-key-not-real"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# Keep the import-time default store out of the repo; each test gets its own below.
os.environ["CHROMA_DIR"] = tempfile.mkdtemp(prefix="riskbot-chroma-")


class FakeSentenceTransformer:
    """Stands in for all-MiniLM-L6-v2 so no model is downloaded.

    Bag-of-words vectors (each word hashed to a dimension), so texts sharing
    words are close under cosine distance and retrieval results are predictable.
    """

    dim = 384

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts):
        vectors = np.zeros((len(texts), self.dim))
        for row, text in enumerate(texts):
            vectors[row, 0] = 1e-3  # avoid all-zero vectors for word-less text
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


_fake_st = types.ModuleType("sentence_transformers")
_fake_st.SentenceTransformer = FakeSentenceTransformer
sys.modules["sentence_transformers"] = _fake_st

import api  # noqa: E402
import rag  # noqa: E402
import store  # noqa: E402


class FakeLLM:
    """Mimics groq_client.chat.completions.create(...).

    `reply` is either a string or a function of the messages list.
    """

    def __init__(self):
        self.reply = "[]"
        self.error = None
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        reply = self.reply(kwargs["messages"]) if callable(self.reply) else self.reply
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def last_user_prompt(self):
        return self.calls[-1]["messages"][-1]["content"]

    def user_prompts(self):
        return [call["messages"][-1]["content"] for call in self.calls]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if anything tries to open a real network connection."""

    def guard(*args, **kwargs):
        raise RuntimeError("Network access is disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(rag, "groq_client", llm)
    monkeypatch.setattr(api, "groq_client", llm)
    return llm


@pytest.fixture
def chroma_dir(tmp_path):
    """A fresh on-disk Chroma store for each test."""
    path = str(tmp_path / "db")
    store.connect(path)
    return path


@pytest.fixture
def client(monkeypatch, tmp_path, fake_llm, chroma_dir):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api, "UPLOAD_DIR", str(tmp_path))
    return TestClient(api.app)
