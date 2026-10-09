from rag import _parse_json_from_text

FINDINGS = [
    {"type": "HRCI", "text_snippet": "Salary: $102,000", "category": "salary", "confidence": 0.93},
    {"type": "NPPI", "text_snippet": "123-45-6789", "category": "ssn", "confidence": 0.99},
]

FINDINGS_JSON = """[
  {"type": "HRCI", "text_snippet": "Salary: $102,000", "category": "salary", "confidence": 0.93},
  {"type": "NPPI", "text_snippet": "123-45-6789", "category": "ssn", "confidence": 0.99}
]"""


def test_clean_json_array():
    assert _parse_json_from_text(FINDINGS_JSON) == FINDINGS


def test_clean_json_with_surrounding_whitespace():
    assert _parse_json_from_text(f"\n\n  {FINDINGS_JSON}  \n") == FINDINGS


def test_empty_array():
    assert _parse_json_from_text("[]") == []


def test_json_inside_markdown_fence():
    raw = f"```json\n{FINDINGS_JSON}\n```"
    assert _parse_json_from_text(raw) == FINDINGS


def test_json_inside_bare_markdown_fence():
    raw = f"```\n{FINDINGS_JSON}\n```"
    assert _parse_json_from_text(raw) == FINDINGS


def test_json_surrounded_by_prose():
    raw = f"Here are the tagged spans:\n{FINDINGS_JSON}\nLet me know if you need more."
    assert _parse_json_from_text(raw) == FINDINGS


def test_broken_json_returns_empty_list():
    raw = '[{"type": "HRCI", "text_snippet": "Salary: $102,000", "category": '
    assert _parse_json_from_text(raw) == []


def test_broken_json_inside_fence_returns_empty_list():
    raw = '```json\n[{"type": "NPPI", "text_snippet": 123-45-6789}]\n```'
    assert _parse_json_from_text(raw) == []


def test_plain_text_returns_empty_list():
    assert _parse_json_from_text("I could not find any sensitive data.") == []


def test_empty_string_returns_empty_list():
    assert _parse_json_from_text("") == []


def test_json_object_is_not_accepted_as_findings():
    # Only a top-level array (or an array embedded in the text) counts.
    assert _parse_json_from_text('{"type": "HRCI"}') == []
