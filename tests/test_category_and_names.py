"""LLM findings outside the canonical categories are dropped; names in HRCI snippets are masked."""
import json

DOC = (
    "Payroll memo (synthetic)\n"
    "Jordan Smith's base salary is $98,500 effective April 1.\n"
    "Jordan Smith joined the team in March.\n"
)


def upload(client, fake_llm, findings):
    fake_llm.reply = json.dumps(findings)
    res = client.post("/upload", files={"file": ("memo.txt", DOC.encode(), "text/plain")})
    assert res.status_code == 200, res.text
    return res


def test_other_category_is_dropped_and_known_categories_are_mapped(client, fake_llm):
    res = upload(client, fake_llm, [
        {"type": "HRCI", "text_snippet": "Jordan Smith", "category": "other", "confidence": 0.6},  # bare name
        {"type": "HRCI", "text_snippet": "joined the team", "category": "", "confidence": 0.5},  # no category
        {"type": "HRCI", "text_snippet": "base salary is $98,500", "category": "Salary", "confidence": 0.9},
    ])

    body = res.json()
    assert [(f["category"], f["source"]) for f in body["findings"]] == [("compensation", "llm")]
    assert body["uncategorized_dropped"] == 2
    assert body["ungrounded_dropped"] == 0
    assert "other" not in res.text


def test_names_in_hrci_snippets_are_masked_in_upload_and_ask(client, fake_llm):
    res = upload(client, fake_llm, [
        {"type": "HRCI", "text_snippet": "Jordan Smith's base salary is $98,500", "category": "compensation", "confidence": 0.9},
    ])
    body = res.json()
    assert body["findings"][0]["text_snippet"] == "J. S.'s base salary is $**,***"
    assert "Jordan" not in res.text and "Smith" not in res.text

    # Free-form question: the name is masked in what the LLM sees and in its answer
    fake_llm.reply = "Jordan Smith earns a base salary; Jordan Smith joined in March."
    ask = client.post("/ask", data={"doc_id": body["doc_id"], "question": "who earns what?"})
    assert ask.json()["answer"] == "J. S. earns a base salary; J. S. joined in March."
    assert "Jordan Smith" not in fake_llm.last_user_prompt()

    # Category question answered from the stored findings
    listed = client.post("/ask", data={"doc_id": body["doc_id"], "question": "show only HRCI"}).json()["answer"]
    assert "J. S.'s base salary is $\\*\\*,\\*\\*\\*" in listed and "Jordan" not in listed  # Markdown-escaped
