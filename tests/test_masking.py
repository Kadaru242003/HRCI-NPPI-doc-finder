"""Sensitive values are masked in every API response and never reach the UI in full."""
import json

import pytest

from masking import mask_digits, mask_snippet, mask_text

DOC = (
    "Employee: Jordan Okafor\n"
    "SSN: 123-45-6789\n"
    "Salary: $102,000\n"
    "Checking account number: 4471029381\n"
    "Routing number: 021000021\n"
)
RAW_VALUES = ["123-45-6789", "123456789", "102,000", "102000", "4471029381", "021000021"]


@pytest.mark.parametrize(
    "text, keep, masked",
    [
        ("123-45-6789", 4, "***-**-6789"),
        ("SSN: 123 45 6789", 4, "SSN: *** ** 6789"),
        ("4111 1111 1111 1111", 4, "**** **** **** 1111"),
        ("4471029381", 4, "******9381"),
        ("Salary: $102,000", 0, "Salary: $***,***"),
        ("1234", 4, "****"),  # 4 digits or fewer: nothing kept
        ("no digits", 4, "no digits"),
    ],
)
def test_mask_digits(text, keep, masked):
    assert mask_digits(text, keep_last=keep) == masked


def test_mask_snippet_by_type():
    assert mask_snippet({"type": "NPPI", "text_snippet": "021000021"}) == "*****0021"
    assert mask_snippet({"type": "HRCI", "text_snippet": "bonus of $12,000"}) == "bonus of $**,***"
    assert mask_snippet({"type": "HRCI", "text_snippet": "performance improvement plan"}) == "performance improvement plan"


def test_mask_text_catches_reformatted_and_unlisted_values():
    findings = [{"type": "NPPI", "text_snippet": "123-45-6789", "category": "ssn"}]
    text = "SSN 123-45-6789, also written 123456789 or 123 45 6789; card 4111 1111 1111 1111."
    masked = mask_text(text, findings)
    assert "6789" in masked
    for raw in ["123-45-6789", "123456789", "123 45 6789", "4111 1111 1111 1111"]:
        assert raw not in masked
    assert "**** **** **** 1111" in masked  # found by the card rule although not a stored finding


def test_upload_response_contains_no_raw_values(client, fake_llm):
    fake_llm.reply = json.dumps([{"type": "HRCI", "text_snippet": "Salary: $102,000", "category": "salary", "confidence": 0.9}])

    res = client.post("/upload", files={"file": ("emp.txt", DOC.encode(), "text/plain")})

    assert res.status_code == 200
    payload = res.text
    for raw in RAW_VALUES:
        assert raw not in payload
    snippets = {f["category"]: f["text_snippet"] for f in res.json()["findings"]}
    assert snippets == {
        "ssn": "***-**-6789",
        "bank_account": "******9381",
        "routing_number": "*****0021",
        "compensation": "Salary: $***,***",
    }


def test_ask_never_sends_or_returns_raw_values(client, fake_llm):
    fake_llm.reply = json.dumps([{"type": "HRCI", "text_snippet": "Salary: $102,000", "category": "salary", "confidence": 0.9}])
    doc_id = client.post("/upload", files={"file": ("emp.txt", DOC.encode(), "text/plain")}).json()["doc_id"]

    # A model that leaks values anyway, in several formats
    fake_llm.reply = "SSN 123-45-6789 (123456789), salary $102,000 ($102000), account 4471029381, routing 021000021."
    res = client.post("/ask", data={"doc_id": doc_id, "question": "what is the SSN and salary?"})

    prompt = fake_llm.last_user_prompt()
    answer = res.json()["answer"]
    for raw in RAW_VALUES:
        assert raw not in prompt, raw
        assert raw not in answer, raw
    assert "***-**-6789" in answer and "$***,***" in answer
