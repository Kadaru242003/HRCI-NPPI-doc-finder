import json
import os
import subprocess
import sys

import pytest

import rag
import store
from conftest import ROOT, groq_model_not_found

TXT = "Employee: Jane Doe\nSalary: $102,000\nSSN: 123-45-6789\n"


def upload(client, name, text):
    return client.post("/upload", files={"file": (name, text.encode(), "text/plain")})


# ---------------------------------------------------------------- GROQ_MODEL


def model_in_fresh_process(value):
    env = {**os.environ, "ANONYMIZED_TELEMETRY": "False"}
    env.pop("GROQ_MODEL", None)
    if value is not None:
        env["GROQ_MODEL"] = value
    out = subprocess.run(
        [sys.executable, "-c", "import rag; print(rag.LLM_MODEL)"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "openai/gpt-oss-120b"),
        ("", "openai/gpt-oss-120b"),
        ("  llama-3.1-8b-instant  ", "llama-3.1-8b-instant"),
    ],
)
def test_groq_model_env_var(value, expected):
    assert model_in_fresh_process(value) == expected


def test_detection_and_ask_use_the_configured_model(client, fake_llm, monkeypatch):
    monkeypatch.setattr(rag, "LLM_MODEL", "some-org/some-model")
    fake_llm.reply = "[]"

    doc_id = upload(client, "a.txt", TXT).json()["doc_id"]
    fake_llm.reply = "An answer"
    client.post("/ask", data={"doc_id": doc_id, "question": "salary"})

    assert [call["model"] for call in fake_llm.calls] == ["some-org/some-model"] * 2


# ---------------------------------------------------------------- /upload failures


def test_retired_model_on_upload_is_a_clear_502(client, fake_llm):
    fake_llm.error = groq_model_not_found()

    res = upload(client, "employee.txt", TXT)

    assert res.status_code == 502
    body = res.json()
    assert "findings" not in body
    assert body["indexed_chunks"] == 1
    detail = body["detail"]
    assert "NOT scanned" in detail
    assert "HTTP 404" in detail
    assert "`llama-3.3-70b-versatile` does not exist" in detail
    assert "openai/gpt-oss-120b" in detail  # the model the app asked for
    # No findings record is stored, so nothing can later read this as "clean"
    assert store.collection.get(ids=[f"{body['doc_id']}_findings"])["ids"] == []


def test_every_batch_failing_on_a_long_file_is_a_502(client, fake_llm):
    fake_llm.reply = "I can't help with that."

    res = upload(client, "long.txt", "filler text " * 1000)

    assert res.status_code == 502
    batches = len(fake_llm.calls) // (rag.LLM_MAX_RETRIES + 1)  # each batch was retried
    assert batches > 1 and len(fake_llm.calls) == batches * (rag.LLM_MAX_RETRIES + 1)
    assert f"({batches} of {batches})" in res.json()["detail"]


def test_some_batches_failing_returns_findings_and_a_warning(client, fake_llm):
    state = {"n": 0}

    def second_batch_fails(messages):
        state["n"] += 1
        if state["n"] == 2:
            raise groq_model_not_found()
        first_word = messages[-1]["content"].split("Text to label:\n---\n", 1)[1].split()[0]
        return json.dumps([{"type": "HRCI", "text_snippet": first_word, "category": "compensation", "confidence": 0.9}])

    fake_llm.reply = second_batch_fails
    fake_llm.error = None

    # Distinct words, so each batch's (grounded) finding is different
    res = upload(client, "long.txt", " ".join(f"word{i:05d}" for i in range(1200)))

    assert res.status_code == 200
    body = res.json()
    total = len(fake_llm.calls)
    assert total >= 3
    assert len(body["findings"]) == total - 1  # one finding per batch that worked
    # 404 model_not_found is permanent: not retried, so each batch made exactly one call
    assert body["scan"] == {"total_batches": total, "scanned_batches": total - 1, "failed_batches": 1, "retries": 0}
    assert body["warning"].startswith(f"Partial scan: the LLM scanned {total - 1} of {total} parts of the file.")
    assert "batch 2/" in body["warning"] and "HTTP 404" in body["warning"]


# ---------------------------------------------------------------- /ask failures


@pytest.fixture
def doc_id(client, fake_llm):
    fake_llm.reply = "[]"
    return upload(client, "employee.txt", TXT).json()["doc_id"]


def test_ask_llm_error_is_a_clear_502(client, fake_llm, doc_id):
    fake_llm.error = groq_model_not_found()

    res = client.post("/ask", data={"doc_id": doc_id, "question": "salary"})

    assert res.status_code == 502
    body = res.json()
    assert "HTTP 404" in body["detail"] and "does not exist" in body["detail"]
    assert "openai/gpt-oss-120b" in body["detail"]
    assert body["retrieved_chunk_ids"] == [f"{doc_id}_0"]
    assert "answer" not in body


@pytest.mark.parametrize("reply", [None, "", "   \n"])
def test_ask_empty_answer_is_a_502(client, fake_llm, doc_id, reply):
    fake_llm.reply = reply

    res = client.post("/ask", data={"doc_id": doc_id, "question": "salary"})

    assert res.status_code == 502
    assert "empty answer" in res.json()["detail"]


def test_ask_truncated_answer_returns_a_warning(client, fake_llm, doc_id):
    fake_llm.reply = "Partial answer"
    fake_llm.finish_reason = "length"

    res = client.post("/ask", data={"doc_id": doc_id, "question": "salary"})

    assert res.status_code == 200
    assert res.json()["answer"] == "Partial answer"
    assert "cut off" in res.json()["warning"]


def test_ask_success_has_no_warning(client, fake_llm, doc_id):
    fake_llm.reply = "Salary: $102,000"

    res = client.post("/ask", data={"doc_id": doc_id, "question": "salary"})

    assert res.status_code == 200
    assert res.json()["warning"] is None


# ---------------------------------------------------------------- error text


def test_describe_llm_error_uses_groq_error_body():
    text = rag.describe_llm_error(groq_model_not_found("old-model"))
    assert text == "HTTP 404: The model `old-model` does not exist or you do not have access to it."


def test_describe_llm_error_for_other_exceptions():
    assert rag.describe_llm_error(TimeoutError("timed out")) == "TimeoutError: timed out"
    assert len(rag.describe_llm_error(RuntimeError("x" * 1000))) == 300
