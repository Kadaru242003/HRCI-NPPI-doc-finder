import os
import re
import json
from dataclasses import dataclass, field
from groq import Groq

import detectors
import store
from masking import mask_snippet

# --- CONFIG ---
# Groq model used for both detection and /ask
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
LLM_MODEL = os.getenv("GROQ_MODEL", "").strip() or DEFAULT_LLM_MODEL
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
    """The stored findings list, or None if no findings record exists."""
    results = store.collection.get(ids=[f"{doc_id}_findings"])
    for text in results["documents"]:
        try:
            return json.loads(text)
        except Exception:
            return []

    return None


# --------------------------------------------------------
# Findings summary for /ask, and a guard against answers that contradict it
# --------------------------------------------------------
def summarize_findings(findings, max_examples: int = 3) -> str:
    """Compact, masked summary of a document's stored findings for the /ask prompt."""
    if findings is None:
        return "No whole-file scan result is stored for this document."
    if not findings:
        return "The whole-file scan found no sensitive data."

    groups = {}
    for item in findings:
        if isinstance(item, dict):
            key = (str(item.get("type", "?")).upper(), str(item.get("category", "other")).lower())
        else:
            key = ("?", "other")
        groups.setdefault(key, []).append(item)

    lines = [f"The whole-file scan found {len(findings)} sensitive item(s) (values masked):"]
    for (ftype, category), items in sorted(groups.items()):
        examples = []
        for item in items[:max_examples]:
            masked = mask_snippet(item)
            examples.append(masked if len(masked) <= 60 else masked[:57] + "...")
        more = f", +{len(items) - max_examples} more" if len(items) > max_examples else ""
        lines.append(f"- {ftype} / {category}: {len(items)} ({'; '.join(examples)}{more})")
    return "\n".join(lines)


_NO_SENSITIVE_DATA = re.compile(
    r"\b(?:no|not\s+(?:any|contain\s+any)|none\s+of\s+the)\s+(?:\w+\s+){0,3}?"
    r"(?:sensitive|confidential|personal|hrci|nppi|pii)\b"
    r"|\bdoes\s+not\s+(?:appear\s+to\s+)?contain\s+(?:any\s+)?(?:\w+\s+){0,2}?"
    r"(?:sensitive|confidential|personal|hrci|nppi|pii)\b",
    re.IGNORECASE,
)


def contradicts_findings(answer: str, findings) -> bool:
    """True if the answer claims there is no sensitive data while findings exist."""
    return bool(findings) and bool(_NO_SENSITIVE_DATA.search(answer or ""))


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

HRCI (Human Resource Confidential Information), categories:
- compensation: salaries, bonuses, offer amounts
- performance: performance reviews, warnings, PIPs
- termination: terminations, severance
- medical: medical leave, injuries, health or disability information related to employment

NPPI (Non-Public Personal Information), categories:
- ssn: Social Security numbers (e.g., 123-45-6789)
- credit_card: payment card numbers
- routing_number: bank routing (ABA) numbers
- bank_account: bank account or loan numbers

Use "other" for anything sensitive that fits none of these.

Return ONLY a JSON array. No explanation. If nothing matches, return [].

Each JSON object must be:
{{
  "type": "HRCI" or "NPPI",
  "text_snippet": "<exact substring>",
  "category": "<one of the categories above>",
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
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json_array(text):
    """Return the JSON array in a model reply, or None if there isn't one.

    Handles a bare array, an array inside ```json fences, and an array
    surrounded by prose. Reasoning text (a <think>...</think> block) is
    ignored. When several arrays appear, the last one whose items are all
    objects wins, since that is the shape of a findings list and a final
    answer comes after any preamble.
    """
    text = _THINK_BLOCK.sub("", text or "").strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except ValueError:
        pass

    decoder = json.JSONDecoder()
    arrays, pos = [], text.find("[")
    while pos != -1:
        try:
            obj, end = decoder.raw_decode(text, pos)
        except ValueError:
            pos = text.find("[", pos + 1)
            continue
        if isinstance(obj, list):
            arrays.append(obj)
        pos = text.find("[", end)

    if not arrays:
        return None
    object_lists = [a for a in arrays if all(isinstance(item, dict) for item in a)]
    return (object_lists or arrays)[-1]


def _parse_json_from_text(text: str):
    findings = extract_json_array(text)
    return findings if findings is not None else []


# --------------------------------------------------------
# LLM error reporting
# --------------------------------------------------------
class LLMError(Exception):
    """Raised when the LLM gives no usable result."""


def describe_llm_error(error: Exception) -> str:
    """Short, user-facing description of a Groq SDK exception."""
    status = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    message = None
    if isinstance(body, dict):
        detail = body.get("error", body)
        if isinstance(detail, dict):
            message = detail.get("message")
    if not message:
        message = str(error) or type(error).__name__
    text = f"HTTP {status}: {message}" if status else f"{type(error).__name__}: {message}"
    return text[:300]


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


# Canonical categories (see eval/README.md); LLM categories are mapped onto them by keyword.
CATEGORY_KEYWORDS = [
    ("ssn", ("ssn", "social")),
    ("credit_card", ("card", "credit")),
    ("routing_number", ("routing", "aba")),
    ("bank_account", ("account", "bank", "iban")),
    ("compensation", ("salary", "bonus", "compensation", "pay", "wage")),
    ("performance", ("performance", "pip", "warning", "review")),
    ("termination", ("termination", "severance", "dismiss", "fired")),
    ("medical", ("medical", "health", "leave", "injury", "disability", "workers")),
]


def canonical_category(raw):
    """One of the canonical categories, or None if the label matches none of them."""
    text = str(raw or "").lower()
    for category, keywords in CATEGORY_KEYWORDS:
        if any(k in text for k in keywords):
            return category
    return None


def _finding_key(item):
    if isinstance(item, dict) and isinstance(item.get("text_snippet"), str):
        ftype = str(item.get("type", "")).upper()
        digits = re.sub(r"\D", "", item["text_snippet"])
        if ftype == "NPPI" and len(digits) >= 4:
            # "SSN: 123-45-6789" and "123 45 6789" are the same value
            return ("nppi-digits", digits)
        snippet = " ".join(item["text_snippet"].split()).casefold()
        return ("span", ftype, snippet)
    return ("raw", json.dumps(item, sort_keys=True, default=str))


def _wins_over(new, old) -> bool:
    """A rule finding beats an LLM finding; otherwise the higher confidence wins."""
    new_rule = isinstance(new, dict) and new.get("source") == "rule"
    old_rule = isinstance(old, dict) and old.get("source") == "rule"
    if new_rule != old_rule:
        return new_rule
    old_conf = old.get("confidence") if isinstance(old, dict) else None
    new_conf = new.get("confidence") if isinstance(new, dict) else None
    return isinstance(new_conf, (int, float)) and (
        not isinstance(old_conf, (int, float)) or new_conf > old_conf
    )


def merge_findings(batches_of_findings) -> list:
    """Merge findings, keeping one entry per value.

    Chunks overlap and rules and the LLM can report the same value, so
    duplicates are common. A rule finding wins over an LLM finding; between
    findings of the same source the highest confidence wins. First-seen
    order is kept.
    """
    merged = {}
    for findings in batches_of_findings:
        for item in findings:
            key = _finding_key(item)
            if key not in merged:
                merged[key] = item
            elif _wins_over(item, merged[key]):
                merged[key] = item
    return list(merged.values())


def _normalize_llm_finding(item):
    if not isinstance(item, dict):
        return item
    item = dict(item, source="llm")
    raw = item.get("category")
    item["category"] = canonical_category(raw) or str(raw or "other").strip().lower() or "other"
    return item


@dataclass
class DetectionResult:
    findings: list
    total_batches: int
    errors: list = field(default_factory=list)  # one entry per failed LLM batch
    ungrounded_dropped: int = 0  # LLM findings whose snippet is not in the source text

    @property
    def warning(self):
        if not self.errors:
            return None
        return (
            f"LLM detection failed for {len(self.errors)} of {self.total_batches} parts "
            f"of the file, so those parts were not scanned and findings from them are "
            f"missing. {self.errors[0]}"
        )


def _detect_batch(text: str, label: str) -> list:
    """Return the findings for one batch; raise LLMError if there is no usable result."""
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
        raise LLMError(f"Model '{LLM_MODEL}', {label}: {describe_llm_error(e)}") from e

    # Groq SDK: message.content is an attribute, not a dict (and may be None)
    raw = (completion.choices[0].message.content or "").strip()

    print(f"\n=== RAW MODEL OUTPUT (detect_hrci_nppi, {label}) ===\n")
    print(raw)
    print("\n===========================================\n")

    findings = extract_json_array(raw)
    if findings is None:
        reason = "empty reply" if not raw else "reply contained no JSON array"
        raise LLMError(f"Model '{LLM_MODEL}', {label}: {reason}")
    return findings


# --------------------------------------------------------
# MAIN HRCI / NPPI DETECTION FOR A GIVEN DOC (uses GROQ)
# --------------------------------------------------------
def detect_hrci_nppi(doc_id: str):
    """
    Detect HRCI / NPPI for a specific uploaded document ID.
    Fetches all of the document's chunks from Chroma, runs the Groq LLM on
    consecutive batches of them so the whole file is scanned, merges and
    de-duplicates the findings, and stores them back into Chroma.

    Returns a DetectionResult; failed batches are listed in `errors`. Raises
    LLMError if every batch failed, since then nothing was scanned.
    """
    chunks = load_chunks_for_doc(doc_id)

    if not chunks:
        print(f"No text found for document {doc_id}.")
        return DetectionResult(findings=[], total_batches=0)

    result = scan_batches(batch_chunks(chunks))
    if result.total_batches and len(result.errors) == result.total_batches:
        rule_count = sum(1 for f in result.findings if f.get("source") == "rule")
        raise LLMError(
            f"LLM detection failed for every part of the file ({result.total_batches} of "
            f"{result.total_batches}), so the file was NOT scanned for HRCI "
            f"(rule-based checks found {rule_count} structured NPPI value(s)). {result.errors[0]}"
        )
    findings = result.findings

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

    return result


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def is_grounded(finding, source_text: str) -> bool:
    """True if the finding's snippet appears in the source text (whitespace-normalized)."""
    if not isinstance(finding, dict) or not isinstance(finding.get("text_snippet"), str):
        return False
    snippet = _collapse_ws(finding["text_snippet"])
    return bool(snippet) and snippet in _collapse_ws(source_text)


def scan_batches(batches: list, use_llm: bool = True, use_rules: bool = True, ground: bool = True) -> DetectionResult:
    """Hybrid detection over text batches: rules for structured NPPI, the LLM for the rest.

    With `ground`, LLM findings whose snippet is not in the batch text they came
    from are dropped and counted. Pure function of the text (no storage), shared
    by the app and scripts/evaluate.py.
    """
    per_batch, errors, dropped = [], [], 0
    for i, text in enumerate(batches):
        label = f"batch {i + 1}/{len(batches)}"
        if use_rules:
            per_batch.append(detectors.detect(text))
        if use_llm:
            try:
                llm_findings = [_normalize_llm_finding(f) for f in _detect_batch(text, label)]
            except LLMError as e:
                errors.append(str(e))
                continue
            if ground:
                kept = [f for f in llm_findings if is_grounded(f, text)]
                dropped += len(llm_findings) - len(kept)
                llm_findings = kept
            per_batch.append(llm_findings)
    return DetectionResult(
        findings=merge_findings(per_batch),
        total_batches=len(batches) if use_llm else 0,
        errors=errors,
        ungrounded_dropped=dropped,
    )


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

