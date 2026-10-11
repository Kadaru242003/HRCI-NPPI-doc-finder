from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging
import os
import uuid

from ingest import index_file
from masking import mask_finding, mask_text
import limits
import retention
import rag
from rag import (
    DEFAULT_TOP_K,
    LLMError,
    contradicts_findings,
    describe_llm_error,
    detect_hrci_nppi,
    load_findings,
    retrieve_chunks,
    summarize_findings,
)

# ---------------------------------------------------------
#  GROQ CLIENT (GLOBAL)
# ---------------------------------------------------------
from groq import Groq

GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    raise Exception("Missing GROQ_API_KEY environment variable!")

groq_client = Groq(api_key=GROQ_KEY)

# ---------------------------------------------------------
# FASTAPI APP SETUP
# ---------------------------------------------------------
# Logs carry counts and IDs only: never document text, model output or findings.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("riskbot")

app = FastAPI()

UPLOAD_DIR = "./data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Multipart framing around the file; the exact file size is checked again after reading
_MULTIPART_OVERHEAD = 64 * 1024


@app.middleware("http")
async def enforce_limits(request: Request, call_next):
    """Rate limit /upload and /ask per client IP, and reject oversized uploads before reading them."""
    if request.method == "POST" and request.url.path in ("/upload", "/ask"):
        upload = request.url.path == "/upload"
        per_minute = limits.UPLOADS_PER_MINUTE if upload else limits.ASKS_PER_MINUTE
        key = limits.client_key(request.headers, request.client.host if request.client else "")
        retry_after = limits.rate_limiter.check(request.url.path, key, per_minute)
        if retry_after:
            log.warning("rate limit hit on %s", request.url.path)
            return JSONResponse(status_code=429, headers={"Retry-After": str(retry_after)}, content={
                "detail": f"Too many requests to {request.url.path}: the limit is {per_minute} per minute. "
                          f"Try again in {retry_after} seconds.",
            })
        length = request.headers.get("content-length")
        if upload and length and length.isdigit() and int(length) > limits.MAX_UPLOAD_BYTES + _MULTIPART_OVERHEAD:
            return JSONResponse(status_code=413, content={"detail": _too_large_message()})
    return await call_next(request)


def _too_large_message():
    return f"The file is too large. The maximum upload size is {limits.MAX_UPLOAD_MB:g} MB."


@app.middleware("http")
async def purge_expired_documents(request: Request, call_next):
    """Retention: delete documents older than DOC_TTL_HOURS before handling any request."""
    retention.purge_expired(UPLOAD_DIR)
    return await call_next(request)


# Serve chatbot UI
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def home():
    return HTMLResponse("""
    <h2>RiskBot: HRCI / NPPI Analyzer</h2>
    <p>Upload a file at <code>/upload</code> or open the chatbot UI at 
       <a href="/static/index.html">/static/index.html</a></p>
    """)

@app.on_event("startup")
def startup_event():
    print("\n======================================")
    print(" 🚀 RiskBot is running!")
    print(" 🔗 Open the app in your browser:")
    print("     http://localhost:8000/static/index.html")
    print(f" 🤖 LLM model (GROQ_MODEL): {rag.LLM_MODEL}")
    print("======================================\n")
    retention.purge_expired(UPLOAD_DIR)

# ---------------------------------------------------------
# 1️⃣ FILE UPLOAD → INDEX → RUN GROQ EXTRACTION
# ---------------------------------------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in [".txt", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: .txt, .xlsx, .xls"
        )

    contents = await file.read(limits.MAX_UPLOAD_BYTES + 1)
    if len(contents) > limits.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=_too_large_message())
    try:
        limits.check_content(ext, contents)
    except limits.ContentError as e:
        raise HTTPException(status_code=415, detail=str(e))

    doc_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}{ext}")

    # Save upload
    with open(save_path, "wb") as f:
        f.write(contents)

    # Index text into vector DB
    try:
        index_info = index_file(save_path, doc_id=doc_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Run HRCI / NPPI detection (uses Groq inside rag.py)
    try:
        detection = detect_hrci_nppi(doc_id)
    except LLMError as e:
        # Nothing was scanned: report an error instead of an empty (clean-looking) result
        log.warning("upload doc=%s: LLM detection failed for every batch", doc_id)
        return JSONResponse(status_code=502, content={
            "detail": str(e),
            "doc_id": doc_id,
            "indexed_chunks": index_info.get("num_chunks", 0),
        })

    response = {
        "doc_id": doc_id,
        "indexed_chunks": index_info.get("num_chunks", 0),
        # Values are masked (e.g. ***-**-6789); full values never leave the server
        "findings": [mask_finding(f) for f in detection.findings],
        "ungrounded_dropped": detection.ungrounded_dropped,
        "uncategorized_dropped": detection.uncategorized_dropped,
        "scan": detection.scan,
        "warning": detection.warning,
    }

    log.info(
        "upload doc=%s chunks=%d findings=%d (rule=%d, llm=%d) ungrounded_dropped=%d failed_batches=%d/%d",
        doc_id, response["indexed_chunks"], len(detection.findings),
        sum(1 for f in detection.findings if isinstance(f, dict) and f.get("source") == "rule"),
        sum(1 for f in detection.findings if isinstance(f, dict) and f.get("source") == "llm"),
        detection.ungrounded_dropped, len(detection.errors), detection.total_batches,
    )

    return JSONResponse(content=response)


# ---------------------------------------------------------
# 2️⃣ CHATBOT ENDPOINT (GROQ)
# ---------------------------------------------------------
@app.post("/ask")
async def ask_question(
    doc_id: str = Form(...),
    question: str = Form(...),
    top_k: int = Form(DEFAULT_TOP_K, ge=1, le=50),
):
    """
    Chatbot endpoint:
    - Embeds the question and retrieves the top_k most similar chunks of this doc
    - Uses the GROQ_MODEL LLM to answer user instructions from those chunks only
    - Supports prompts like:
      'show only HRCI', 'show only NPPI', 'show only salary', etc.
    """

    # Category questions are answered from the whole-file findings, not from a few similar chunks
    route = rag.route_question(question)
    if route is not None:
        findings = load_findings(doc_id)
        if findings is None and not rag.load_chunks_for_doc(doc_id):
            return JSONResponse(content={"answer": "No document found.", "retrieved_chunk_ids": [], "warning": None, "route": "findings"})
        log.info("ask doc=%s: answered from stored findings", doc_id)
        return JSONResponse(content={
            "answer": rag.answer_from_findings(route, findings, rag.load_scan_info(doc_id)),
            "retrieved_chunk_ids": [],
            "warning": None,
            "route": "findings",
        })

    chunks = retrieve_chunks(doc_id, question, top_k)

    if not chunks:
        return JSONResponse(content={"answer": "No document found.", "retrieved_chunk_ids": [], "warning": None, "route": "retrieval"})

    retrieved_ids = [chunk["id"] for chunk in chunks]

    findings = load_findings(doc_id)
    # The LLM only sees masked values, and its answer is masked again below
    context = "\n---\n".join(mask_text(chunk["text"], findings) for chunk in chunks)

    prompt = f"""
You are an assistant helping users analyze sensitive HR/Finance text.

Whole-document scan results (authoritative for what the document contains):
------------------
{summarize_findings(findings)}

Document Context (only {len(chunks)} excerpt(s) most relevant to the question, NOT the whole document):
------------------
{context}

User Question:
------------------
{question}

Filtering Rules:
- If user asks "show only HRCI", return only items that are HRCI-like (HR confidential).
- If "show only NPPI", return only NPPI-like (personal financial identifiers).
- If "show only salary", filter only salary-related spans.
- If asked to summarize, provide a clean, concise summary.
- Be professional and clear.

Important:
- You only see excerpts. Never claim the document has no sensitive or confidential data
  based on the excerpts alone. Use the whole-document scan results above for that.
- If the scan results list items that are not in the excerpts, say they exist elsewhere
  in the document.
- Sensitive values are masked with "*". Keep them masked; never guess hidden digits.
"""

    # Call GROQ LLM
    try:
        completion = groq_client.chat.completions.create(
            model=rag.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful HR/Finance analysis assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
    except Exception as e:
        log.warning("ask doc=%s: LLM call failed: %s", doc_id, type(e).__name__)
        return JSONResponse(status_code=502, content={
            "detail": f"The LLM request failed (model '{rag.LLM_MODEL}'): {describe_llm_error(e)}",
            "retrieved_chunk_ids": retrieved_ids,
        })

    # Groq SDK: message.content, not ["content"]
    choice = completion.choices[0]
    answer = mask_text((choice.message.content or "").strip(), findings)
    if not answer:
        return JSONResponse(status_code=502, content={
            "detail": f"The LLM (model '{rag.LLM_MODEL}') returned an empty answer.",
            "retrieved_chunk_ids": retrieved_ids,
        })

    warnings = []
    if getattr(choice, "finish_reason", None) == "length":
        warnings.append("The answer was cut off because the model reached its output token limit.")
    if contradicts_findings(answer, findings):
        warnings.append(
            f"This answer says there is no sensitive data, but the whole-file scan found "
            f"{len(findings)} item(s). The answer was based on excerpts only; rely on the "
            f"Detected Sensitive Data table."
        )
    warning = " ".join(warnings) or None

    return JSONResponse(content={
        "answer": answer,
        "retrieved_chunk_ids": retrieved_ids,
        "warning": warning,
        "route": "retrieval",
    })


# ---------------------------------------------------------
# 3️⃣ DELETE A DOCUMENT (chunks, findings, stored files)
# ---------------------------------------------------------
@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    if not retention.is_valid_doc_id(doc_id):
        raise HTTPException(status_code=400, detail="Invalid document id.")
    result = retention.delete_document(doc_id, UPLOAD_DIR)
    if not (result["deleted_chunks"] or result["deleted_findings"] or result["deleted_files"]):
        raise HTTPException(status_code=404, detail="No document with this id (it may already be deleted or expired).")
    log.info("delete doc=%s chunks=%d files=%d", doc_id, result["deleted_chunks"], result["deleted_files"])
    return result
