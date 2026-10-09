"""Shared test setup.

The app reads GROQ_API_KEY and loads a Sentence Transformers model at import
time, so both are faked here *before* `api` / `rag` / `ingest` are imported.
The Groq client is replaced with a fake, so tests need no API key and make no
network calls.
"""
import hashlib
import os
import socket
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# api.py mounts StaticFiles(directory="static"), which is resolved against the cwd.
os.chdir(ROOT)

os.environ["GROQ_API_KEY"] = "test-key-not-real"
os.environ["ANONYMIZED_TELEMETRY"] = "False"


class FakeSentenceTransformer:
    """Stands in for all-MiniLM-L6-v2 so no model is downloaded."""

    dim = 384

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts):
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
            vectors.append(np.random.default_rng(seed).random(self.dim))
        return np.array(vectors)


_fake_st = types.ModuleType("sentence_transformers")
_fake_st.SentenceTransformer = FakeSentenceTransformer
sys.modules["sentence_transformers"] = _fake_st

import api  # noqa: E402
import rag  # noqa: E402


class FakeLLM:
    """Mimics groq_client.chat.completions.create(...)."""

    def __init__(self):
        self.reply = "[]"
        self.error = None
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def last_user_prompt(self):
        return self.calls[-1]["messages"][-1]["content"]


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
def client(monkeypatch, tmp_path, fake_llm):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api, "UPLOAD_DIR", str(tmp_path))
    return TestClient(api.app)
