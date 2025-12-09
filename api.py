from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import json

from ingest import index_file
from rag import detect_hrci_nppi, load_findings

app = FastAPI()

UPLOAD_DIR = "./data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve chatbot UI (static files)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def home():
    # Go straight to the UI
    return RedirectResponse(url="/static/index.html")


# --------------------------------------------------------
# 1️⃣ FILE UPLOAD + HRCI / NPPI EXTRACTION
# --------------------------------------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in [".txt", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: .txt, .xlsx, .xls",
        )

    contents = await file.read()

    doc_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}{ext}")

    with open(save_path, "wb") as f:
        f.write(contents)

    # Index chunks into Chroma
    index_info = index_file(save_path, doc_id=doc_id)

    # Run HRCI / NPPI extraction for THIS doc
    findings = detect_hrci_nppi(doc_id)

    response = {
        "doc_id": doc_id,
        "indexed_chunks": index_info.get("num_chunks", 0),
        "findings": findings,
    }

    # Log to server console
    print("\n=== OUTGOING RESPONSE ===")
    print(json.dumps(response, indent=2))
    print("=========================\n")

    return JSONResponse(content=response)


# --------------------------------------------------------
# 2️⃣ CHATBOT ENDPOINT — ask questions about extracted findings
# --------------------------------------------------------
@app.post("/ask")
async def ask_question(
    doc_id: str = Form(...),
    question: str = Form(...),
):
    """
    Chatbot endpoint: user sends doc_id + a natural-language question.
    We load the stored HRCI/NPPI findings for that doc_id and let Llama
    filter/summarize them.
    """

    findings = load_findings(doc_id)

    if not findings:
        return JSONResponse(
            content={"answer": "No extracted findings found for this document ID."}
        )

    findings_json = json.dumps(findings, indent=2)

    prompt = f"""
You are an assistant helping a user understand and filter sensitive HR/Finance findings.

The findings are a JSON array of objects like:
[{{"type": "HRCI" or "NPPI", "category": "...", "text_snippet": "...", "confidence": ...}}, ...]

Findings:
{findings_json}

User request:
{question}

Instructions:
- Only use the findings above. Do NOT invent new data.
- If the user asks "only HRCI", return only HRCI items.
- If "only NPPI", return only NPPI items.
- If "only salary", return only items where category looks like salary/compensation.
- If they mention "SSN", "bank", "loan", "health", etc, filter accordingly.
- You may format the answer as a short summary, bullet list, or simple table text.
- Be concise and clear.
"""

    import ollama

    result = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}],
    )

    answer = result["message"]["content"]

    return JSONResponse(content={"answer": answer})

