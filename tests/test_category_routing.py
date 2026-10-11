"""Category questions are answered from the stored whole-file findings, not similarity search."""
import json

import pytest

import rag

# The SSN is far from everything else, so similarity search over a few chunks would miss it.
DOC = (
    "Salary: $102,000 for Jane Doe.\n"
    "Jane Doe was placed on a performance improvement plan.\n"
    + "\n".join(f"Cafeteria note {i}: soup and salad today." for i in range(150))
    + "\nSSN: 123-45-6789\nRouting number: 021000021\n"
)
LLM_FINDINGS = json.dumps([
    {"type": "HRCI", "text_snippet": "Salary: $102,000", "category": "salary", "confidence": 0.9},
    {"type": "HRCI", "text_snippet": "placed on a performance improvement plan", "category": "PIP", "confidence": 0.9},
])


@pytest.fixture
def doc_id(client, fake_llm):
    fake_llm.reply = LLM_FINDINGS
    res = client.post("/upload", files={"file": ("doc.txt", DOC.encode(), "text/plain")})
    assert res.status_code == 200
    fake_llm.calls.clear()
    return res.json()["doc_id"]


@pytest.fixture
def no_retrieval(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("similarity search must not be used for category questions")

    monkeypatch.setattr(rag, "retrieve_chunks", fail)
    import api

    monkeypatch.setattr(api, "retrieve_chunks", fail)


def ask(client, doc_id, question):
    res = client.post("/ask", data={"doc_id": doc_id, "question": question})
    assert res.status_code == 200
    return res.json()


def test_list_every_ssn_uses_findings_not_search_or_llm(client, fake_llm, doc_id, no_retrieval):
    body = ask(client, doc_id, "list every SSN")

    assert body["route"] == "findings"
    assert body["retrieved_chunk_ids"] == []
    assert fake_llm.calls == []
    assert "found **1** SSNs item(s)" in body["answer"]
    assert f"| NPPI | ssn | {rag.md_escape('***-**-6789')} | rule |" in body["answer"]
    assert "123-45-6789" not in body["answer"]


def test_show_only_nppi(client, fake_llm, doc_id, no_retrieval):
    body = ask(client, doc_id, "Show only NPPI")

    assert "found **2** NPPI item(s)" in body["answer"]
    assert rag.md_escape("***-**-6789") in body["answer"] and rag.md_escape("*****0021") in body["answer"]
    assert "Salary" not in body["answer"]
    assert fake_llm.calls == []


def test_show_only_hrci_and_salary(client, fake_llm, doc_id, no_retrieval):
    hrci = ask(client, doc_id, "show only HRCI")["answer"]
    assert "found **2** HRCI item(s)" in hrci and rag.md_escape("Salary: $***,***") in hrci and "performance improvement plan" in hrci

    salary = ask(client, doc_id, "show only salary")["answer"]
    assert "found **1** compensation item(s)" in salary and "performance" not in salary


def test_how_many_with_no_matches(client, fake_llm, doc_id, no_retrieval):
    body = ask(client, doc_id, "how many credit card numbers are there?")
    assert body["answer"] == "The whole-file scan found no card numbers in this document."


def test_free_form_questions_still_use_retrieval(client, fake_llm, doc_id):
    fake_llm.reply = "Jane Doe's salary is masked."
    body = ask(client, doc_id, "what is Jane Doe's salary?")
    assert body["route"] == "retrieval" and len(fake_llm.calls) == 1


def test_unknown_document(client, fake_llm, no_retrieval):
    body = ask(client, "00000000-0000-0000-0000-000000000000", "list every SSN")
    assert body["answer"] == "No document found." and body["route"] == "findings"


def test_partial_scan_is_mentioned(client, fake_llm, no_retrieval):
    def second_batch_fails(messages):
        if "SSN" in messages[-1]["content"].split("Text to label:")[1]:
            return "not json"
        return "[]"

    fake_llm.reply = second_batch_fails
    text = "word " * 1200 + "\nSSN: 123-45-6789\n"
    doc_id = client.post("/upload", files={"file": ("d.txt", text.encode(), "text/plain")}).json()["doc_id"]

    answer = ask(client, doc_id, "list every SSN")["answer"]
    assert rag.md_escape("***-**-6789") in answer
    assert "parts of this file were scanned by the LLM" in answer


@pytest.mark.parametrize(
    "question, expected",
    [
        ("show only NPPI", ({"NPPI"}, set())),
        ("list every SSN", (set(), {"ssn"})),
        ("list all routing and account numbers", (set(), {"routing_number", "bank_account"})),
        ("which bonuses are mentioned?", (set(), {"compensation"})),
        ("show all medical leave", (set(), {"medical"})),
    ],
)
def test_route_question_matches_listing_questions(question, expected):
    route = rag.route_question(question)
    assert (route["types"], route["categories"]) == expected


@pytest.mark.parametrize("question", ["summarize", "what is Jane's salary?", "show me the document", "list the action items"])
def test_route_question_leaves_other_questions_to_retrieval(question):
    assert rag.route_question(question) is None
