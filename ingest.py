import os
import re
import time
import uuid
import pandas as pd

import store

def _column_name(index: int) -> str:
    """0 -> "A", 25 -> "Z", 26 -> "AA" (Excel column letters)."""
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def _looks_like_header(row) -> bool:
    cells = [c for c in row if c]
    return bool(cells) and not any(re.fullmatch(r"[\d\s.,$%/()+-]+", c) for c in cells)


def extract_text_from_excel(file_path: str) -> str:
    """One line per data row, each cell as "Header: value", prefixed with the sheet name.

    Example: "Payroll row 2: Name: Jane Doe | SSN: 123-45-6789 | Salary: 102000".
    Keeping the header with every value gives detection the context it needs
    ("Routing: 021000021"), and the sheet name stays with every row even when
    chunking splits a sheet. If the first row is not a header (it contains
    numbers), columns are named "Column A", "Column B", ...
    """
    try:
        xls = pd.ExcelFile(file_path)
    except Exception as e:
        raise ValueError(f"Unable to open Excel file: {e}")

    lines = []
    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet, header=None, dtype=str).fillna("")
        except Exception:
            continue

        rows = [[str(c).strip() for c in row] for row in df.values.tolist()]
        rows = [(n + 1, row) for n, row in enumerate(rows) if any(row)]
        if not rows:
            continue

        first_number, first = rows[0]
        if _looks_like_header(first):
            headers, data = first, rows[1:]
            if not data:  # a header-only sheet is just text
                lines.append(f"{sheet}: " + " | ".join(c for c in first if c))
                continue
        else:
            headers, data = [], rows

        for number, row in data:
            cells = []
            for i, value in enumerate(row):
                if value:
                    header = headers[i] if i < len(headers) and headers[i] else f"Column {_column_name(i)}"
                    cells.append(f"{header}: {value}")
            lines.append(f"{sheet} row {number}: " + " | ".join(cells))

    return "\n".join(lines)


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
    else:
        text = load_text_from_file(file_path)

    chunks = chunk_text(text)

    ids = []
    docs = []
    metas = []

    created_at = time.time()  # used by retention to expire the document
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue

        cid = f"{doc_id}_{i}"
        ids.append(cid)
        docs.append(chunk)
        metas.append({
            "doc_id": doc_id,
            "kind": "chunk",
            "chunk_index": i,
            "created_at": created_at,
            "file_name": os.path.basename(file_path)
        })

    if not docs:
        return {"doc_id": doc_id, "num_chunks": 0}

    store.collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=store.embed(docs))

    return {"doc_id": doc_id, "num_chunks": len(docs)}

