"""Shared ChromaDB collection and embedding model used by ingest.py and rag.py."""
import os

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = os.getenv("CHROMA_DIR", "./db")
COLLECTION_NAME = "documents"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMBED_MODEL_NAME)

client = None
collection = None


def connect(path: str = CHROMA_DIR):
    """Open (or create) the on-disk Chroma store at `path`."""
    global client, collection
    client = chromadb.PersistentClient(path=path)
    # Cosine distance suits normalized sentence embeddings like MiniLM.
    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    return collection


def embed(texts: list[str]) -> list[list[float]]:
    return embedder.encode(texts).tolist()


connect()
