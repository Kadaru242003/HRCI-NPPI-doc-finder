import io
import json
import uuid

import pytest
from openpyxl import Workbook

import rag

TXT_DOC = (
    "Employee: Jane Doe\n"
    "Salary: $102,000\n"
    "SSN: 123-45-6789\n"
    "Notes: placed on a performance improvement plan.\n"
)

FINDINGS = [
    {"type": "HRCI", "text_snippet": "Salary: $102,000", "category": "salary", "confidence": 0.93},
    {"type": "NPPI", "text_snippet": "123-45-6789", "category": "ssn", "confidence": 0.99},
]


def make_xlsx() -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Payroll"
    ws.append(["Name", "SSN", "Salary"])
    ws.append(["Jane Doe", "123-45-6789", "102000"])
    other = wb.create_sheet("Banking")
    other.append(["Routing number 021000021"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def upload(client, name, content, content_type="application/octet-stream"):
    return client.post("/upload", files={"file": (name, content, content_type)})


# ---------------------------------------------------------------- /upload


def test_upload_txt_returns_parsed_findings(client, fake_llm, tmp_path):
    fake_llm.reply = "```json\n" + json.dumps(FINDINGS) + "\n```"

    res = upload(client, "employee.txt", TXT_DOC.encode(), "text/plain")

    assert res.status_code == 200
    body = res.json()
    uuid.UUID(body["doc_id"])  # valid UUID
    assert body["indexed_chunks"] == 1
    assert body["findings"] == FINDINGS
    assert body["warning"] is None

    # One LLM call, with the document text in the prompt
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["model"] == "openai/gpt-oss-120b"
    prompt = fake_llm.last_user_prompt()
    assert "Salary: $102,000" in prompt
    assert "123-45-6789" in prompt

    # Upload is saved under a generated name, not the client-supplied one
    assert (tmp_path / f"{body['doc_id']}.txt").exists()


def test_upload_xlsx_includes_every_sheet_and_header_row(client, fake_llm):
    fake_llm.reply = json.dumps(FINDINGS)

    res = upload(
        client,
        "payroll.xlsx",
        make_xlsx(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    assert res.status_code == 200
    body = res.json()
    assert body["indexed_chunks"] == 1
    assert body["findings"] == FINDINGS

    prompt = fake_llm.last_user_prompt()
    for cell in ["Name", "SSN", "Salary", "Jane Doe", "123-45-6789", "102000"]:
        assert cell in prompt
    assert "Routing number 021000021" in prompt  # second sheet


def test_upload_extension_check_is_case_insensitive(client, fake_llm):
    res = upload(client, "EMPLOYEE.TXT", TXT_DOC.encode(), "text/plain")
    assert res.status_code == 200


def test_upload_rejects_unsupported_extension(client, fake_llm):
    res = upload(client, "resume.pdf", b"%PDF-1.4", "application/pdf")

    assert res.status_code == 400
    assert "Unsupported file type" in res.json()["detail"]
    assert fake_llm.calls == []


def test_upload_corrupt_xlsx_returns_400(client, fake_llm):
    res = upload(client, "broken.xlsx", b"this is not a spreadsheet")

    assert res.status_code == 400
    assert "Unable to open Excel file" in res.json()["detail"]
    assert fake_llm.calls == []


def test_upload_empty_file_skips_llm(client, fake_llm):
    res = upload(client, "empty.txt", b"", "text/plain")

    assert res.status_code == 200
    assert res.json()["indexed_chunks"] == 0
    assert res.json()["findings"] == []
    assert fake_llm.calls == []


def test_upload_with_unparseable_llm_output_is_an_error(client, fake_llm):
    fake_llm.reply = "Sure! Here is what I found: SSN 123-45-6789"

    res = upload(client, "employee.txt", TXT_DOC.encode(), "text/plain")

    assert res.status_code == 502
    assert "no JSON array" in res.json()["detail"]
    assert "findings" not in res.json()


def test_upload_with_empty_llm_content_is_an_error(client, fake_llm):
    fake_llm.reply = None

    res = upload(client, "employee.txt", TXT_DOC.encode(), "text/plain")

    assert res.status_code == 502
    assert "empty reply" in res.json()["detail"]


def test_upload_when_llm_call_fails_is_an_error(client, fake_llm):
    fake_llm.error = RuntimeError("Groq is down")

    res = upload(client, "employee.txt", TXT_DOC.encode(), "text/plain")

    assert res.status_code == 502
    assert "Groq is down" in res.json()["detail"]
    assert "findings" not in res.json()


def test_upload_where_llm_finds_nothing_is_a_clean_result(client, fake_llm):
    fake_llm.reply = "[]"

    res = upload(client, "memo.txt", b"The office picnic is on Friday.", "text/plain")

    assert res.status_code == 200
    assert res.json()["findings"] == []
    assert res.json()["warning"] is None


# ---------------------------------------------------------------- /ask


def test_ask_answers_with_document_context(client, fake_llm):
    doc_id = upload(client, "employee.txt", TXT_DOC.encode(), "text/plain").json()["doc_id"]
    fake_llm.reply = "- **Salary: $102,000** (HRCI)"

    res = client.post("/ask", data={"doc_id": doc_id, "question": "show only HRCI"})

    assert res.status_code == 200
    assert res.json() == {
        "answer": "- **Salary: $102,000** (HRCI)",
        "retrieved_chunk_ids": [f"{doc_id}_0"],
        "warning": None,
    }

    call = fake_llm.calls[-1]
    assert call["model"] == "openai/gpt-oss-120b"
    prompt = fake_llm.last_user_prompt()
    assert "show only HRCI" in prompt
    assert "Salary: $102,000" in prompt


def test_ask_context_is_scoped_to_the_requested_document(client, fake_llm):
    doc_a = upload(client, "a.txt", b"Alpha bonus: $5,000", "text/plain").json()["doc_id"]
    upload(client, "b.txt", b"Bravo account 4111 1111 1111 1111", "text/plain")

    client.post("/ask", data={"doc_id": doc_a, "question": "summarize"})

    prompt = fake_llm.last_user_prompt()
    assert "Alpha bonus" in prompt
    assert "Bravo account" not in prompt


def test_ask_unknown_document_does_not_call_llm(client, fake_llm):
    res = client.post("/ask", data={"doc_id": "does-not-exist", "question": "summarize"})

    assert res.status_code == 200
    assert res.json() == {"answer": "No document found.", "retrieved_chunk_ids": [], "warning": None}
    assert fake_llm.calls == []


@pytest.mark.parametrize("missing", ["doc_id", "question"])
def test_ask_requires_both_form_fields(client, fake_llm, missing):
    data = {"doc_id": "x", "question": "y"}
    del data[missing]

    res = client.post("/ask", data=data)

    assert res.status_code == 422
    assert fake_llm.calls == []


# ---------------------------------------------------------------- misc


def test_home_and_ui_are_served(client):
    assert client.get("/").status_code == 200
    ui = client.get("/static/index.html")
    assert ui.status_code == 200
    assert "RiskBot" in ui.text


def test_findings_are_not_fed_back_into_chat_context(client, fake_llm):
    fake_llm.reply = json.dumps(FINDINGS)
    doc_id = upload(client, "employee.txt", TXT_DOC.encode(), "text/plain").json()["doc_id"]

    assert rag.load_findings(doc_id) == FINDINGS
    assert all('"text_snippet"' not in chunk for chunk in rag.load_chunks_for_doc(doc_id))
