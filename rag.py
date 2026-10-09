import os
import json
from groq import Groq

import store

# --- CONFIG ---
LLM_MODEL = "llama-3.3-70b-versatile"
# Max characters of chunk text sent to the LLM per detection call
DETECT_BATCH_CHARS = int(os.getenv("DETECT_BATCH_CHARS", "4000"))
# Number of chunks /ask retrieves when the request doesn't say
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# --- GROQ CLIENT ---
GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    raise Exception("Missing GROQ_API_KEY environment variable!")
groq_client = Groq(api_key=GROQ_KEY)


def _chunk_filter(doc_id: str) -> dict:
    # Raw content chunks only; the stored "findings" record is excluded
    return {"$and": [{"doc_id": doc_id}, {"kind": "chunk"}]}


# --------------------------------------------------------
# Load all text chunks for a specific document, in file order
# --------------------------------------------------------
def load_chunks_for_doc(doc_id: str) -> list[str]:
    results = store.collection.get(where=_chunk_filter(doc_id))
    rows = sorted(
        zip(results["metadatas"], results["documents"]),
        key=lambda row: row[0].get("chunk_index", 0),
    )
    return [text for _, text in rows if text and text.strip()]


# --------------------------------------------------------
# Retrieve the chunks of one document most similar to a question
# --------------------------------------------------------
def retrieve_chunks(doc_id: str, question: str, top_k: int = DEFAULT_TOP_K):
    """Return [{"id", "text", "distance"}, ...], most similar first."""
    where = _chunk_filter(doc_id)
    available = len(store.collection.get(where=where, include=[])["ids"])
    if available == 0:
        return []

    results = store.collection.query(
        query_embeddings=store.embed([question]),
        n_results=min(top_k, available),
        where=where,
    )
    return [
        {"id": cid, "text": text, "distance": dist}
        for cid, text, dist in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        )
    ]


# --------------------------------------------------------
# Load stored findings for a document
# --------------------------------------------------------
def load_findings(doc_id: str):
    results = store.collection.get(ids=[f"{doc_id}_findings"])
    for text in results["documents"]:
        try:
            return json.loads(text)
        except Exception:
            return []

    return []


# --------------------------------------------------------
# Prompt builders
# --------------------------------------------------------
def build_system_prompt() -> str:
    return (
        "You are a data-labeling assistant used on synthetic, fake HR and finance text. "
        "The text is NOT real; it is dummy data for testing only. "
        "Your only job is to tag spans that look like HR confidential info (HRCI) "
        "or non-public personal info (NPPI). "
        "You must always answer with a JSON array and nothing else."
    )


def build_user_prompt(context: str) -> str:
    return f"""
Tag any spans in the text that match these categories.

HRCI (Human Resource Confidential Information) examples:
- salaries, bonuses, compensation
- performance reviews, warnings, PIP
- termination / severance
- health or medical claims related to employment

NPPI (Non-Public Personal Information) examples:
- SSN-like patterns (e.g., 123-45-6789)
- bank or routing numbers
- account / loan numbers
- credit card numbers

Return ONLY a JSON array. No explanation.

Each JSON object must be:
{{
  "type": "HRCI" or "NPPI",
  "text_snippet": "<exact substring>",
  "category": "<one word category>",
  "confidence": <float between 0.0 and 1.0>
}}

Include low-confidence items.

Text to label:
---
{context}
---
"""


# --------------------------------------------------------
# JSON PARSER
# --------------------------------------------------------
def _parse_json_from_text(text: str):
    text = text.strip()

    # Try direct JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # Try to extract array inside larger output
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        snippet = text[start : end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

    return []


# --------------------------------------------------------
# Batching and de-duplication for whole-document detection
# --------------------------------------------------------
def batch_chunks(chunks: list[str], max_chars: int = DETECT_BATCH_CHARS) -> list[str]:
    """Group consecutive chunks into texts of at most max_chars (one chunk minimum)."""
    batches, current, size = [], [], 0
    for chunk in chunks:
        if current and size + len(chunk) > max_chars:
            batches.append("\n".join(current))
            current, size = [], 0
        current.append(chunk)
        size += len(chunk)
    if current:
        batches.append("\n".join(current))
    return batches


def _finding_key(item):
    if isinstance(item, dict) and isinstance(item.get("text_snippet"), str):
        snippet = " ".join(item["text_snippet"].split()).casefold()
        return ("span", str(item.get("type", "")).upper(), snippet)
    return ("raw", json.dumps(item, sort_keys=True, default=str))


def merge_findings(batches_of_findings) -> list:
    """Merge per-batch findings, keeping one entry per (type, snippet).

    Chunks overlap, so the same span is often reported more than once; the
    entry with the highest confidence wins and first-seen order is kept.
    """
    merged = {}
    for findings in batches_of_findings:
        for item in findings:
            key = _finding_key(item)
            if key not in merged:
                merged[key] = item
                continue
            old_conf = merged[key].get("confidence") if isinstance(merged[key], dict) else None
            new_conf = item.get("confidence") if isinstance(item, dict) else None
            if isinstance(new_conf, (int, float)) and (
                not isinstance(old_conf, (int, float)) or new_conf > old_conf
            ):
                merged[key] = item
    return list(merged.values())


def _detect_batch(text: str, label: str) -> list:
    try:
        completion = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_prompt(text)},
            ],
            temperature=0.2,
        )
    except Exception as e:
        print(f"Groq error in detect_hrci_nppi ({label}):", e)
        return []

    # Groq SDK: message.content is an attribute, not a dict (and may be None)
    raw = (completion.choices[0].message.content or "").strip()

    print(f"\n=== RAW MODEL OUTPUT (detect_hrci_nppi, {label}) ===\n")
    print(raw)
    print("\n===========================================\n")

    return _parse_json_from_text(raw)


# --------------------------------------------------------
# MAIN HRCI / NPPI DETECTION FOR A GIVEN DOC (uses GROQ)
# --------------------------------------------------------
def detect_hrci_nppi(doc_id: str):
    """
    Detect HRCI / NPPI for a specific uploaded document ID.
    Fetches all of the document's chunks from Chroma, runs the Groq LLM on
    consecutive batches of them so the whole file is scanned, merges and
    de-duplicates the findings, and stores them back into Chroma.
    """
    chunks = load_chunks_for_doc(doc_id)

    if not chunks:
        print(f"No text found for document {doc_id}.")
        return []

    batches = batch_chunks(chunks)
    findings = merge_findings(
        _detect_batch(text, f"batch {i + 1}/{len(batches)}")
        for i, text in enumerate(batches)
    )

    # Store findings in Chroma (read back by load_findings)
    try:
        findings_text = json.dumps(findings)
        emb = store.embed([findings_text])
        store.collection.add(
            ids=[f"{doc_id}_findings"],
            documents=[findings_text],
            metadatas=[{"doc_id": doc_id, "kind": "findings"}],
            embeddings=emb,
        )
    except Exception as e:
        print("Failed to store findings in Chroma:", e)

    return findings


# --------------------------------------------------------
# OPTIONAL: direct detection from arbitrary text (not used by UI now)
# --------------------------------------------------------
def detect_from_text(text: str):
    if not text or not text.strip():
        return []

    text = text.strip()[:4000]

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(text)

    try:
        completion = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
    except Exception as e:
        print("Groq error in detect_from_text:", e)
        return []

    raw = completion.choices[0].message.content.strip()

    print("\n=== RAW MODEL OUTPUT (direct) ===\n")
    print(raw)
    print("\n=================================\n")

    return _parse_json_from_text(raw)

