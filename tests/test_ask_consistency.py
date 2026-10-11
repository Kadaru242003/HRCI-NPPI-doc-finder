"""/ask must stay consistent with the whole-file findings, not just the retrieved excerpts."""
import json

import pytest

import rag

SSN_FINDING = {"type": "NPPI", "text_snippet": "123-45-6789", "category": "ssn", "confidence": 0.99}
SALARY_FINDING = {"type": "HRCI", "text_snippet": "Salary: $102,000", "category": "compensation", "confidence": 0.9}

# Many unrelated lines push the sensitive line out of the retrieved excerpts.
DOC = "\n".join(f"Cafeteria menu item {i}: soup and salad." for i in range(120)) + "\nSSN: 123-45-6789\n"


def upload(client, text):
    res = client.post("/upload", files={"file": ("doc.txt", text.encode(), "text/plain")})
    assert res.status_code == 200, res.text
    return res.json()["doc_id"]


def test_excerpt_only_no_data_claim_is_flagged_when_findings_exist(client, fake_llm):
    """Regression: the summary says "no confidential data" although the scan found some."""
    fake_llm.reply = json.dumps([SSN_FINDING])
    doc_id = upload(client, DOC)

    fake_llm.reply = "Summary: the cafeteria menu. This document contains no confidential data."
    res = client.post("/ask", data={"doc_id": doc_id, "question": "summarize", "top_k": "1"})

    assert res.status_code == 200
    body = res.json()
    assert body["answer"].startswith("Summary: the cafeteria menu.")
    assert body["warning"] is not None
    assert "whole-file scan found 1 item(s)" in body["warning"]


def test_prompt_carries_masked_findings_summary_and_excerpt_rule(client, fake_llm):
    fake_llm.reply = json.dumps([SSN_FINDING, SALARY_FINDING])
    doc_id = upload(client, DOC + "Salary: $102,000\n")

    fake_llm.reply = "ok"
    client.post("/ask", data={"doc_id": doc_id, "question": "summarize", "top_k": "1"})
    prompt = fake_llm.last_user_prompt()

    assert "The whole-file scan found 2 sensitive item(s)" in prompt
    # Masked values are Markdown-escaped (asterisks would otherwise read as emphasis)
    assert f"NPPI / ssn: 1 ({rag.md_escape('***-**-6789')})" in prompt
    assert f"HRCI / compensation: 1 ({rag.md_escape('Salary: $***,***')})" in prompt
    assert "Never claim the document has no sensitive or confidential data" in prompt
    assert "only 1 excerpt(s)" in prompt


def test_a_model_that_follows_the_prompt_reports_the_findings(client, fake_llm):
    """With the summary in the prompt, an instruction-following model answers consistently."""
    fake_llm.reply = json.dumps([SSN_FINDING])
    doc_id = upload(client, DOC)

    def follows_prompt(messages):
        prompt = messages[-1]["content"]
        if "whole-file scan found 1 sensitive item" in prompt:
            return "The excerpts are a menu, but the whole-file scan found 1 NPPI item (an SSN) elsewhere."
        return "This document contains no confidential data."

    fake_llm.reply = follows_prompt
    res = client.post("/ask", data={"doc_id": doc_id, "question": "summarize", "top_k": "1"})

    assert "1 NPPI item" in res.json()["answer"]
    assert res.json()["warning"] is None


def test_no_warning_when_the_scan_found_nothing(client, fake_llm):
    fake_llm.reply = "[]"
    doc_id = upload(client, "Team lunch is on Friday.")

    fake_llm.reply = "This document contains no confidential data."
    res = client.post("/ask", data={"doc_id": doc_id, "question": "summarize"})

    assert res.json()["warning"] is None
    assert "The whole-file scan found no sensitive data." in fake_llm.last_user_prompt()


@pytest.mark.parametrize(
    "answer, contradicts",
    [
        ("This document contains no confidential data.", True),
        ("There is no sensitive information here.", True),
        ("The file does not contain any personal data.", True),
        ("I found no HRCI or NPPI.", True),
        ("No further action is needed for the salary item.", False),
        ("The SSN 123-45-6789 is sensitive.", False),
    ],
)
def test_contradiction_detector(answer, contradicts):
    assert rag.contradicts_findings(answer, [SSN_FINDING]) is contradicts
    assert rag.contradicts_findings(answer, []) is False


def test_summary_groups_counts_and_limits_examples():
    findings = [dict(SSN_FINDING, text_snippet=f"123-45-{6780 + i}") for i in range(5)]
    summary = rag.summarize_findings(findings)

    assert "found 5 sensitive item(s)" in summary
    examples = "; ".join(rag.md_escape(f"***-**-{n}") for n in (6780, 6781, 6782))
    assert f"NPPI / ssn: 5 ({examples}, +2 more)" in summary
    assert "123-45" not in summary
    assert rag.summarize_findings(None).startswith("No whole-file scan result")
