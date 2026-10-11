"""Upload size limit (413), content checks (415) and per-IP rate limits (429)."""
import io
import zipfile

import pytest
from openpyxl import Workbook

import limits


def xlsx_bytes():
    wb = Workbook()
    wb.active.append(["Name"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def post(client, name, data, headers=None):
    return client.post("/upload", files={"file": (name, data, "application/octet-stream")}, headers=headers or {})


# ---------------------------------------------------------------- size

def test_file_over_the_limit_is_rejected_with_413(client, fake_llm, monkeypatch):
    monkeypatch.setattr(limits, "MAX_UPLOAD_BYTES", 1000)
    monkeypatch.setattr(limits, "MAX_UPLOAD_MB", 0.001)

    res = post(client, "big.txt", b"a" * 1001)

    assert res.status_code == 413
    assert "too large" in res.json()["detail"] and "0.001 MB" in res.json()["detail"]
    assert fake_llm.calls == []
    assert post(client, "ok.txt", b"a" * 1000).status_code == 200


def test_oversized_request_is_rejected_before_the_body_is_read(client, fake_llm):
    res = client.post("/upload", content=b"x", headers={
        "content-length": str(limits.MAX_UPLOAD_BYTES + 10**6), "content-type": "multipart/form-data; boundary=x"})
    assert res.status_code == 413


def test_default_limit_is_2_mb():
    assert limits.MAX_UPLOAD_BYTES == 2 * 1024 * 1024


# ---------------------------------------------------------------- content

@pytest.mark.parametrize(
    "name, data, message",
    [
        ("notes.txt", b"PK\x03\x04rest-of-zip", "ZIP"),
        ("notes.txt", b"%PDF-1.7 ...", "PDF"),
        ("notes.txt", b"MZ\x90\x00", "executable"),
        ("notes.txt", b"hello\x00world", "binary"),
        ("notes.txt", "café".encode("latin-1"), "UTF-8"),
        ("sheet.xlsx", b"Name,SSN\nA,1\n", "not a valid .xlsx"),
        ("sheet.xlsx", b"PK\x03\x04not-really-a-zip", "damaged"),
        ("sheet.xls", b"Name\tSSN\n", "not a valid .xls"),
    ],
)
def test_content_must_match_the_extension(client, fake_llm, name, data, message):
    res = post(client, name, data)
    assert res.status_code == 415
    assert message in res.json()["detail"]
    assert fake_llm.calls == []


def test_zip_that_is_not_a_workbook_is_rejected(client, fake_llm):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", "<w/>")
    res = post(client, "sheet.xlsx", buf.getvalue())
    assert res.status_code == 415 and "not an Excel workbook" in res.json()["detail"]


def test_real_files_pass_the_content_check(client, fake_llm):
    assert post(client, "notes.txt", "Café notes, UTF-8 ✓".encode()).status_code == 200
    assert post(client, "sheet.xlsx", xlsx_bytes()).status_code == 200


# ---------------------------------------------------------------- rate limits

def test_upload_rate_limit_returns_429_with_retry_after(client, fake_llm, monkeypatch):
    monkeypatch.setattr(limits, "UPLOADS_PER_MINUTE", 2)
    assert post(client, "a.txt", b"one").status_code == 200
    assert post(client, "a.txt", b"two").status_code == 200

    res = post(client, "a.txt", b"three")

    assert res.status_code == 429
    assert int(res.headers["retry-after"]) >= 1
    assert "limit is 2 per minute" in res.json()["detail"]
    assert len(fake_llm.calls) == 2  # the rejected upload never reached the LLM


def test_ask_rate_limit_is_separate_and_per_client(client, fake_llm, monkeypatch):
    monkeypatch.setattr(limits, "ASKS_PER_MINUTE", 1)
    data = {"doc_id": "x", "question": "q"}
    assert client.post("/ask", data=data, headers={"x-forwarded-for": "198.51.100.7"}).status_code == 200
    assert client.post("/ask", data=data, headers={"x-forwarded-for": "198.51.100.7"}).status_code == 429
    # Another client (as appended by the proxy) still gets through
    assert client.post("/ask", data=data, headers={"x-forwarded-for": "198.51.100.8"}).status_code == 200
    # A forged left-hand entry does not help: the rightmost address is the one used
    assert client.post("/ask", data=data, headers={"x-forwarded-for": "203.0.113.1, 198.51.100.7"}).status_code == 429
    # Uploads have their own budget
    assert post(client, "a.txt", b"x", headers={"x-forwarded-for": "198.51.100.7"}).status_code == 200


def test_window_slides():
    limiter = limits.RateLimiter()
    assert limiter.check("/upload", "ip", 2, now=0) == 0
    assert limiter.check("/upload", "ip", 2, now=10) == 0
    assert limiter.check("/upload", "ip", 2, now=20) > 0
    assert limiter.check("/upload", "ip", 2, now=61) == 0  # the first request left the window
    assert limiter.check("/upload", "ip", 0, now=61) == 0  # 0 disables the limit
