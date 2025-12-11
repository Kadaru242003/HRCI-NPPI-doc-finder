<h1>🚨 RiskBot – HRCI / NPPI Detection System (RAG Pipeline)</h1>
<p> <b>RiskBot</b> is an end-to-end Retrieval-Augmented Generation (RAG) platform that detects <b>Human Resource Confidential Information (HRCI)</b> and <b>Non-Public Personal Information (NPPI)</b> from unstructured text and Excel documents. Built using <b>FastAPI</b>, <b>Groq LLaMA-3.3-70B</b>, <b>ChromaDB</b>, and <b>Sentence-Transformer embeddings</b>. </p> <hr/>
<h2>🔥 Key Capabilities</h2>

Full RAG pipeline: ingestion → chunking → embeddings → vector search → LLM reasoning

Unstructured data support (Excel multi-sheet, text)

HRCI/NPPI detection with structured JSON output

85% confidence filter to remove noisy predictions

Persistent ChromaDB vector store

Fast interactive chatbot for querying document insights

Local, secure, and enterprise-friendly architecture

<hr/>
<h2>🧠 System Workflow</h2>
<p align="center"> <img src="https://github.com/user-attachments/assets/4fe20c48-e3d0-4f37-a8bf-26efc76f924c" width="400"/> </p> <br>

The system processes documents in 5 major stages:

Upload document via FastAPI (/upload)

Extract text from XLSX/TXT

Chunk + embed using MiniLM

Store embeddings in ChromaDB (persistent)

Run LLaMA-3.3-70B (Groq) to detect HRCI/NPPI

Interact through chatbot UI for contextual analysis

<hr/>
<h2>🖥️ UI Screenshots</h2>
<p align="center"> <img src="https://github.com/user-attachments/assets/0658309d-3a67-4758-a79d-263fda5d06e0" width="850"/> <br/><br/> <img src="https://github.com/user-attachments/assets/ce188083-ae57-40b7-a58a-664e26ad1823" width="850"/> </p> <hr/>
<h2>⚙️ Tech Stack</h2>
Component	Technology
API Server	FastAPI
LLM	Groq LLaMA-3.3-70B Versatile
Embeddings	all-MiniLM-L6-v2
Vector DB	ChromaDB
XLSX Parsing	pandas
Deployment	Uvicorn
<hr/>
<h2>📂 Project Structure</h2>
hrci_nppi_bot/
│
├── api.py                 # FastAPI endpoints (upload + chat)
├── rag.py                 # HRCI/NPPI detection + Groq LLM logic
├── ingest.py              # Excel parsing, text extraction, chunking, embedding
│
├── db/                    # ChromaDB persistent vector storage
├── data/                  # Uploaded user files
├── static/
│   └── index.html         # Chatbot frontend
│
├── requirements.txt
└── README.md


<hr/>
<h2>🚀 Running the Project Locally</h2>
1. Create environment
python3 -m venv venv
source venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

3. Add Groq API key
export GROQ_API_KEY="your_key_here"

4. Start FastAPI server
uvicorn api:app --reload

5. Open UI
http://localhost:8000/static/index.html

<hr/>
<h2>🔍 Detection Schema</h2>

The model outputs structured JSON such as:

{
  "type": "HRCI",
  "text_snippet": "Employee salary: $102,000",
  "category": "salary",
  "confidence": 0.93
}


Low-confidence predictions (< 0.85) are filtered out.

<hr/>
<h2>🏆 Why This Project Matters</h2>

RiskBot showcases:

Practical, production-style RAG engineering

Handling and indexing unstructured enterprise data

Secure LLM workflows with contextual vector retrieval

Real-world HR/compliance use-case alignment

It reflects strong skills in LLM integration, vector databases, ML systems design, and backend engineering.

<hr/>
<h2>📬 Contact</h2>

If you want the deployment version or architecture diagram as a PNG/SVG, feel free to ask.
