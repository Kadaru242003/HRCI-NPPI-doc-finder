from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import json

from ingest import index_file
from rag import DEFAULT_TOP_K, detect_hrci_nppi, retrieve_chunks

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
app = FastAPI()

UPLOAD_DIR = "./data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve chatbot UI
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def home():
    return HTMLResponse("""
    <h2>RiskBot – HRCI / NPPI Analyzer</h2>
    <p>Upload a file at <code>/upload</code> or open the chatbot UI at 
       <a href="/static/index.html">/static/index.html</a></p>
    """)

@app.on_event("startup")
def startup_event():
    print("\n======================================")
    print(" 🚀 RiskBot is running!")
    print(" 🔗 Open the app in your browser:")
    print("     http://localhost:8000/static/index.html")
    print("======================================\n")

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

    contents = await file.read()

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
    findings = detect_hrci_nppi(doc_id)

    response = {
        "doc_id": doc_id,
        "indexed_chunks": index_info.get("num_chunks", 0),
        "findings": findings
    }

    print("\n=== OUTGOING RESPONSE ===")
    print(json.dumps(response, indent=2))
    print("=========================\n")

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
    - Uses GROQ Llama model to answer user instructions from those chunks only
    - Supports prompts like:
      'show only HRCI', 'show only NPPI', 'show only salary', etc.
    """

    chunks = retrieve_chunks(doc_id, question, top_k)

    if not chunks:
        return JSONResponse(content={"answer": "No document found.", "retrieved_chunk_ids": []})

    context = "\n---\n".join(chunk["text"] for chunk in chunks)

    prompt = f"""
You are an assistant helping users analyze sensitive HR/Finance text.

Document Context (the excerpts most relevant to the question):
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
"""

    # Call GROQ LLM
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful HR/Finance analysis assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    # Groq SDK: message.content, not ["content"]
    answer = completion.choices[0].message.content

    return JSONResponse(content={
        "answer": answer,
        "retrieved_chunk_ids": [chunk["id"] for chunk in chunks],
    })

