"""Shared ChromaDB collection and embedding model used by ingest.py and rag.py."""
import os

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

CHROMA_DIR = os.getenv("CHROMA_DIR", "./db")
COLLECTION_NAME = "documents"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

client = None
collection = None

# all-MiniLM-L6-v2 run with ONNX Runtime (chromadb's bundled implementation, no PyTorch).
# Created on first use; the model files are downloaded to ~/.cache/chroma on the
# first call and the ONNX session is then kept for the life of the process.
_embedder = None

# Every input is padded to 256 tokens, so activation memory grows with the batch:
# 32 texts at once (chromadb's default) costs ~200 MB more peak RSS than 1, and on
# CPU it is no faster. Batch size doesn't change the resulting vectors.
EMBED_BATCH_SIZE = 1


def connect(path: str = CHROMA_DIR):
    """Open (or create) the on-disk Chroma store at `path`."""
    global client, collection
    client = chromadb.PersistentClient(path=path)
    # Cosine distance suits normalized sentence embeddings like MiniLM.
    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    return collection


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = ONNXMiniLM_L6_V2()
    return _embedder


def embed(texts: list[str]) -> list[list[float]]:
    embedder = get_embedder()
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        vectors.extend(embedder(texts[i : i + EMBED_BATCH_SIZE]))
    return vectors


connect()
