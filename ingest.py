import os
import uuid
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

def extract_text_from_excel(file_path: str) -> str:
    text_blocks = []

    try:
        xls = pd.ExcelFile(file_path)
    except Exception as e:
        raise ValueError(f"Unable to open Excel file: {e}")

    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet, dtype=str)
            df = df.fillna("")
        except Exception:
            continue

        # flatten cells to a list
        lines = df.astype(str).values.flatten().tolist()

        for line in lines:
            line = line.strip()
            if len(line) > 0:
                text_blocks.append(line)

    # merge into one big text
    return "\n".join(text_blocks)


CHROMA_DIR = "./db"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))
collection = client.get_or_create_collection("documents")
embedder = SentenceTransformer(EMBED_MODEL_NAME)

def load_text_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks

def index_file(file_path: str, doc_id: str | None = None):
    if doc_id is None:
        doc_id = str(uuid.uuid4())

    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".xlsx", ".xls", ".XLSX", ".XLS"]:
        text = extract_text_from_excel(file_path)
        print("\n=== EXTRACTED TEXT FROM EXCEL ===")
        print(text[:500])
        print("=================================\n")
    else:
        text = load_text_from_file(file_path)

    debug_path = file_path + ".txt"
    with open(debug_path, "w") as f:
        f.write(text)

    chunks = chunk_text(text)

    ids = []
    docs = []
    metas = []

    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue

        cid = f"{doc_id}_{i}"
        ids.append(cid)
        docs.append(chunk)
        metas.append({
            "doc_id": doc_id,
            "chunk_index": i,
            "file_name": os.path.basename(file_path)
        })

    if not docs:
        return {"doc_id": doc_id, "num_chunks": 0}

    embeddings = embedder.encode(docs).tolist()
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    return {"doc_id": doc_id, "num_chunks": len(docs)}

