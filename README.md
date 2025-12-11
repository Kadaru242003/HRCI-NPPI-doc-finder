RiskBot – HRCI/NPPI Detection System (RAG Pipeline)

A production-ready Retrieval-Augmented Generation (RAG) system for detecting Human Resource Confidential Information (HRCI) and Non-Public Personal Information (NPPI) from unstructured documents (Excel + text).

RiskBot combines FastAPI, Groq LLaMA-3.3-70B, ChromaDB, and Sentence-Transformer embeddings to ingest files, vectorize content, run LLM-powered classification, and enable interactive Q&A over document context.

🚀 Features

End-to-end RAG pipeline for sensitive information detection

Excel + TXT ingestion with multi-sheet parsing & metadata-aware chunking

Vector storage in ChromaDB with persistent embeddings

HRCI/NPPI detection powered by Groq LLaMA-3.3-70B

85% confidence filtering to suppress weak predictions

Interactive chatbot UI for context-aware querying

Full local execution using uvicorn

🧠 Architecture Overview

RiskBot Workflow:

User uploads file (Excel or Text) → api.py (/upload)

File saved locally → ./data/<doc_id>.xlsx

Extract text via Excel parser or text loader → ingest.py

Chunk + vectorize text using SentenceTransformer

Store embeddings + metadata in ChromaDB

Run Groq LLaMA-3.3-70B to detect HRCI/NPPI → rag.py

Store model findings back in ChromaDB (kind="findings")

User queries chatbot (/ask)

RAG context retrieved and filtered using LLM

Answer returned to UI

You can include your workflow diagram right after this section.
<img width="375" height="1190" alt="image" src="https://github.com/user-attachments/assets/4fe20c48-e3d0-4f37-a8bf-26efc76f924c" />

📸 UI Screenshots
<img width="1089" height="903" alt="Screenshot 2025-12-10 at 9 24 33 PM" src="https://github.com/user-attachments/assets/0658309d-3a67-4758-a79d-263fda5d06e0" />
<img width="1038" height="807" alt="Screenshot 2025-12-10 at 9 24 52 PM" src="https://github.com/user-attachments/assets/ce188083-ae57-40b7-a58a-664e26ad1823" />

Example sections:

Upload Page

Results & Findings

Chatbot Interface

🔧 Tech Stack
Component	Tool
Backend Framework	FastAPI
LLM	Groq LLaMA-3.3-70B
Vector DB	ChromaDB
Embeddings	all-MiniLM-L6-v2
Document Parsing	pandas (Excel), custom chunker
Serving	Uvicorn
Environment	Python 3.10+
📦 Project Structure
hrci_nppi_bot/
│
├── api.py               # FastAPI backend (upload + chat endpoints)
├── rag.py               # Detection pipeline + Groq LLM logic
├── ingest.py            # Excel/text parsing + chunking + vector storage
├── db/                  # ChromaDB persistent storage
├── data/                # Uploaded files saved locally
├── static/
│   └── index.html       # Chatbot UI
│
├── requirements.txt
└── README.md

▶️ Run Locally
1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

3. Set your Groq API key
export GROQ_API_KEY="your_key_here"

4. Start the server
uvicorn api:app --reload

5. Open UI

Visit:

http://localhost:8000/static/index.html

🔍 HRCI / NPPI Detection Logic

RiskBot identifies sensitive categories such as:

HRCI Examples

Salaries, compensation

Performance reviews

PIP / warnings

Termination language

Health/medical employment info

NPPI Examples

SSNs

Bank/routing numbers

Loan/account numbers

Credit card numbers

Each detection includes:

{
  "type": "HRCI",
  "text_snippet": "Annual salary: $98,000",
  "category": "salary",
  "confidence": 0.93
}


Low-confidence items (< 0.85) are automatically dropped.

💡 Why This Project Matters

This system demonstrates:

Real-world ML engineering: ingestion → vectorization → retrieval → LLM classification

Handling of unstructured enterprise data

Secure LLM inference patterns

Foundations of production-level RAG systems

