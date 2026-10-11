"""Logs must never contain document text, model output or sensitive values."""
import json
import logging

import pytest

from conftest import groq_model_not_found

PLANTED = {
    "ssn": "123-45-6789",
    "card": "4111 1111 1111 1111",
    "account": "4471029381",
    "salary": "$187,250",
    "sentence": "Quillon Marbury was placed on a performance improvement plan",
}
DOC = (
    f"Employee: Quillon Marbury, SSN {PLANTED['ssn']}\n"
    f"Corporate card {PLANTED['card']}\n"
    f"Checking account number: {PLANTED['account']}\n"
    f"Base salary {PLANTED['salary']}.\n"
    f"{PLANTED['sentence']} in March.\n"
)
LLM_REPLY = json.dumps([
    {"type": "HRCI", "text_snippet": f"Base salary {PLANTED['salary']}", "category": "compensation", "confidence": 0.9},
    {"type": "HRCI", "text_snippet": PLANTED["sentence"], "category": "performance", "confidence": 0.9},
    {"type": "HRCI", "text_snippet": "Invented Person earns $999,999", "category": "compensation", "confidence": 0.5},
])


@pytest.fixture
def all_output(caplog, capfd):
    caplog.set_level(logging.DEBUG)  # capture every logger at every level

    def collect():
        out, err = capfd.readouterr()
        return caplog.text + out + err

    return collect


def assert_clean(text):
    lowered = text.lower()
    for name, value in PLANTED.items():
        assert value.lower() not in lowered, f"{name} leaked into logs"
    assert "123456789" not in text and "4471029381" not in text
    assert "999,999" not in text and "Invented Person" not in text  # raw model output


def test_upload_and_ask_logs_contain_counts_and_ids_only(client, fake_llm, all_output):
    fake_llm.reply = LLM_REPLY
    res = client.post("/upload", files={"file": ("emp.txt", DOC.encode(), "text/plain")})
    assert res.status_code == 200
    doc_id = res.json()["doc_id"]

    fake_llm.reply = f"The SSN is {PLANTED['ssn']} and the salary is {PLANTED['salary']}."
    assert client.post("/ask", data={"doc_id": doc_id, "question": "what is the salary?"}).status_code == 200

    output = all_output()
    assert_clean(output)
    assert doc_id in output  # IDs and counts are logged
    assert "findings=" in output and "ungrounded_dropped=1" in output


def test_failure_paths_do_not_log_content(client, fake_llm, all_output):
    fake_llm.reply = f"Sorry, I saw {PLANTED['ssn']} and {PLANTED['sentence']} but cannot answer in JSON."
    res = client.post("/upload", files={"file": ("emp.txt", DOC.encode(), "text/plain")})
    assert res.status_code == 502

    fake_llm.reply = "[]"
    doc_id = client.post("/upload", files={"file": ("emp.txt", DOC.encode(), "text/plain")}).json()["doc_id"]
    fake_llm.error = groq_model_not_found()
    assert client.post("/ask", data={"doc_id": doc_id, "question": "summary"}).status_code == 502

    assert_clean(all_output())
