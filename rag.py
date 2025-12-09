import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import ollama

# --- CONFIG ---
CHROMA_DIR = "./db"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1:8b"

# --- CLIENTS ---
client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))
collection = client.get_or_create_collection("documents")
embedder = SentenceTransformer(EMBED_MODEL_NAME)


# --------------------------------------------------------
# Load all text chunks for a specific document (for extraction / chat)
# --------------------------------------------------------
def load_context_for_doc(doc_id: str) -> str:
    results = collection.get()
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])

    context_chunks = []
    for text, meta in zip(docs, metas):
        # Skip "findings" records; we only want raw content chunks
        if meta.get("doc_id") == doc_id and meta.get("kind") != "findings":
            if text and text.strip():
                context_chunks.append(text)

    return "\n".join(context_chunks)


# --------------------------------------------------------
# Load stored findings for a document (for chatbot)
# --------------------------------------------------------
def load_findings(doc_id: str):
    results = collection.get()
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])

    for text, meta in zip(docs, metas):
        if meta.get("doc_id") == doc_id and meta.get("kind") == "findings":
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
# MAIN HRCI / NPPI DETECTION FOR A GIVEN DOC
# --------------------------------------------------------
def detect_hrci_nppi(doc_id: str):
    """
    Detect HRCI / NPPI for a specific uploaded document ID.
    Uses Chroma to fetch that document's chunks, runs Llama,
    parses JSON, and stores the findings back into Chroma.
    """
    context = load_context_for_doc(doc_id)

    if not context or not context.strip():
        print(f"No text found for document {doc_id}.")
        return []

    # Safety trim for LLM
    context = context[:4000]

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(context)

    try:
        result = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        print("Ollama error:", e)
        return []

    raw = result.get("message", {}).get("content", "").strip()

    print("\n=== RAW MODEL OUTPUT ===\n")
    print(raw)
    print("\n========================\n")

    findings = _parse_json_from_text(raw)
    if not isinstance(findings, list):
        print("Failed to parse JSON findings; returning [].")
        return []

    # Store findings in Chroma so the chatbot can re-use them
    try:
        findings_text = json.dumps(findings)
        emb = embedder.encode([findings_text]).tolist()
        collection.add(
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
        result = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        print("Ollama error:", e)
        return []

    raw = result.get("message", {}).get("content", "").strip()

    print("\n=== RAW MODEL OUTPUT (direct) ===\n")
    print(raw)
    print("\n=========================\n")

    return _parse_json_from_text(raw)

