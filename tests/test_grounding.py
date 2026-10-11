"""Grounding: LLM findings must quote the source text."""
import json

import rag

TEXT = "Memo\nJordan Okafor's base salary increases\n  to $98,500 effective 2026-04-01.\nSSN 123-45-6789"


def upload(client, text):
    res = client.post("/upload", files={"file": ("memo.txt", text.encode(), "text/plain")})
    assert res.status_code == 200, res.text
    return res.json()


def test_invented_snippets_are_dropped_and_counted(client, fake_llm):
    fake_llm.reply = json.dumps([
        {"type": "HRCI", "text_snippet": "base salary increases to $98,500", "category": "salary", "confidence": 0.9},
        {"type": "HRCI", "text_snippet": "salary of $250,000", "category": "salary", "confidence": 0.9},  # invented
        {"type": "NPPI", "text_snippet": "987-65-4321", "category": "ssn", "confidence": 0.8},  # invented
        {"type": "HRCI", "category": "medical"},  # no snippet
    ])

    body = upload(client, TEXT)

    snippets = [f["text_snippet"] for f in body["findings"]]
    # (API responses are masked: digits of HRCI amounts are hidden, NPPI keeps the last 4)
    assert "base salary increases to $**,***" in snippets  # line break in the source is fine
    assert "salary of $***,***" not in snippets and "***-**-4321" not in snippets
    assert body["ungrounded_dropped"] == 3
    assert "***-**-6789" in snippets  # rule findings are exact matches, never dropped


def test_nothing_dropped_when_all_snippets_are_quotes(client, fake_llm):
    fake_llm.reply = json.dumps([{"type": "HRCI", "text_snippet": "$98,500", "category": "salary", "confidence": 0.9}])
    assert upload(client, TEXT)["ungrounded_dropped"] == 0


def test_is_grounded_normalizes_whitespace_only():
    text = "Salary:\n   $102,000 per year"
    assert rag.is_grounded({"text_snippet": "Salary: $102,000"}, text)
    assert rag.is_grounded({"text_snippet": "  Salary:  $102,000 "}, text)
    assert not rag.is_grounded({"text_snippet": "salary: $102,000"}, text)  # case matters
    assert not rag.is_grounded({"text_snippet": ""}, text)
    assert not rag.is_grounded("Salary", text)


def test_grounding_can_be_disabled_for_the_llm_only_baseline(monkeypatch):
    fake = [{"type": "HRCI", "text_snippet": "not in the text", "category": "salary", "confidence": 0.9}]
    monkeypatch.setattr(rag, "_detect_batch", lambda text, label, stats=None: fake)

    grounded = rag.scan_batches(["some text"], use_rules=False)
    raw = rag.scan_batches(["some text"], use_rules=False, ground=False)

    assert grounded.findings == [] and grounded.ungrounded_dropped == 1
    assert len(raw.findings) == 1 and raw.ungrounded_dropped == 0
