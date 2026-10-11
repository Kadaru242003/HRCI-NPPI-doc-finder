"""Browser tests for Markdown rendering of chat answers (static/markdown.js).

The repository is served by a plain static file server and the backend calls
(/upload, /ask) are answered by Playwright request routing, so these tests need
no API key, model or network. Outside CI they are skipped when no Chromium is
installed; in CI (CI=true) a missing browser is a failure.
"""
import base64
import hashlib
import os
import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import ROOT

playwright_api = pytest.importorskip("playwright.sync_api")

HOSTILE_ANSWER = """### Findings

**2 items** need attention:

- SSN `123-45-6789`
- Salary in [the HR portal](https://example.com/hr)

| Type | Snippet |
| --- | --- |
| NPPI | 123-45-6789 |

Injected: <img src=x onerror=alert(1)> and [click](javascript:alert('link')) and <script>alert('script')</script>
"""


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def base_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=ROOT))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:  # browser binary not installed
            if os.environ.get("CI"):
                raise
            pytest.skip(f"Chromium not available: {e}")
        yield browser
        browser.close()


@pytest.fixture
def page(browser):
    page = browser.new_page()
    page.dialogs = []
    page.errors = []
    page.on("dialog", lambda d: (page.dialogs.append(d.message), d.dismiss()))
    page.on("pageerror", lambda e: page.errors.append(str(e)))
    # Keep the test offline: the Tailwind CDN is only styling.
    page.route(re.compile(r"^https://cdn\.tailwindcss\.com/.*"), lambda route: route.abort())
    yield page
    page.close()


def test_manual_test_page_checks_all_pass(page, base_url):
    page.goto(f"{base_url}/static/markdown-test.html")
    page.wait_for_function("window.__markdownTestResults !== undefined")

    results = page.evaluate("window.__markdownTestResults")
    failed = [c for c in results["checks"] if not c["ok"]]
    assert failed == []
    assert page.dialogs == []
    assert page.errors == []


def test_chat_answer_renders_markdown_and_neutralizes_html(page, base_url, tmp_path):
    findings = [{"type": "NPPI", "text_snippet": "<b>123-45-6789</b>", "category": "ssn", "confidence": 0.97}]

    def fake_upload(route):
        route.fulfill(json={"doc_id": "doc-1", "indexed_chunks": 1, "findings": findings, "warning": None})

    def fake_ask(route):
        route.fulfill(json={"answer": HOSTILE_ANSWER, "retrieved_chunk_ids": ["doc-1_0"], "warning": None})

    page.route(f"{base_url}/upload", fake_upload)
    page.route(f"{base_url}/ask", fake_ask)
    sample = tmp_path / "sample.txt"
    sample.write_text("SSN: 123-45-6789")

    page.goto(f"{base_url}/static/index.html")
    page.set_input_files("#file-input", str(sample))
    page.click("#upload-form button[type=submit]")
    page.wait_for_selector("#upload-status:has-text('Upload complete')")

    # The findings table is unchanged: one row, snippet shown as escaped text
    rows = page.locator("#findings-body tr")
    assert rows.count() == 1
    assert "<b>123-45-6789</b>" in rows.first.inner_text()

    page.fill("#chat-input", "show only NPPI")
    page.click("#chat-submit")
    page.wait_for_selector("#chat-answer:not(.hidden) #chat-answer-content table")
    page.wait_for_timeout(500)  # let any injected onerror handler fire

    answer = page.locator("#chat-answer-content")
    for selector in ["h3", "strong", "ul > li", "code", "table td"]:
        assert answer.locator(selector).count() > 0, selector
    assert answer.locator("img, script, iframe, svg").count() == 0
    assert "<img src=x onerror=alert(1)>" in answer.inner_text()  # shown as text, not an element

    links = answer.locator("a")
    hrefs = [links.nth(i).get_attribute("href") for i in range(links.count())]
    assert hrefs == ["https://example.com/hr", None]  # javascript: href removed, text kept
    for i in range(links.count()):
        assert links.nth(i).get_attribute("target") == "_blank"
        assert links.nth(i).get_attribute("rel") == "noopener noreferrer"

    on_attrs = page.evaluate(
        """() => [...document.querySelectorAll('#chat-answer-content *')]
                 .flatMap(el => [...el.attributes].map(a => a.name))
                 .filter(n => n.toLowerCase().startsWith('on'))"""
    )
    assert on_attrs == []
    assert page.dialogs == []
    assert page.errors == []


@pytest.mark.parametrize("html_file", ["static/index.html", "static/markdown-test.html"])
def test_vendored_scripts_match_their_sri_hashes(html_file):
    html = open(os.path.join(ROOT, html_file), encoding="utf-8").read()
    tags = re.findall(r'<script src="/static/(vendor/[^"]+)"\s+integrity="([^"]+)"', html)
    assert len(tags) == 2
    for path, integrity in tags:
        data = open(os.path.join(ROOT, "static", path), "rb").read()
        digest = base64.b64encode(hashlib.sha384(data).digest()).decode()
        assert integrity == f"sha384-{digest}", path


def test_no_cdn_markdown_or_sanitizer_dependency():
    html = open(os.path.join(ROOT, "static/index.html"), encoding="utf-8").read()
    assert not re.search(r"(cdn\.jsdelivr|unpkg|cdnjs)[^\"]*(marked|purify)", html)


def test_delete_button_removes_the_document_and_clears_the_page(browser, base_url, tmp_path):
    page = browser.new_page()
    page.on("dialog", lambda d: d.accept())  # confirm() "Delete this document...?"
    page.route(re.compile(r"^https://cdn\.tailwindcss\.com/.*"), lambda route: route.abort())
    deleted = []
    doc_id = "3f1c9a5e-8a1b-4a53-9c0e-2b6f7d1e4a90"
    page.route(f"{base_url}/upload", lambda r: r.fulfill(json={
        "doc_id": doc_id, "indexed_chunks": 1, "warning": None, "ungrounded_dropped": 0,
        "findings": [{"type": "NPPI", "text_snippet": "***-**-6789", "category": "ssn", "confidence": 0.97, "source": "rule"}],
    }))

    def fake_delete(route):
        deleted.append((route.request.method, route.request.url))
        route.fulfill(json={"doc_id": doc_id, "deleted_chunks": 1, "deleted_findings": True, "deleted_files": 1})

    page.route(f"{base_url}/documents/**", fake_delete)
    sample = tmp_path / "s.txt"
    sample.write_text("x")

    page.goto(f"{base_url}/static/index.html")
    assert page.locator("#delete-doc").is_disabled()
    page.set_input_files("#file-input", str(sample))
    page.click("#upload-form button[type=submit]")
    page.wait_for_selector("#upload-status:has-text('Upload complete')")
    assert "rule" in page.locator("#findings-body").inner_text()  # source column

    page.click("#delete-doc")
    page.wait_for_selector("#delete-status:has-text('Deleted from the server.')")

    assert deleted == [("DELETE", f"{base_url}/documents/{doc_id}")]
    assert page.locator("#doc-id-display").inner_text() == "(none yet)"
    assert "Document deleted" in page.locator("#findings-body").inner_text()
    assert page.locator("#chat-submit").is_disabled() and page.locator("#delete-doc").is_disabled()
    page.close()


def test_partial_scan_is_reported_clearly(page, base_url, tmp_path):
    page.route(f"{base_url}/upload", lambda r: r.fulfill(json={
        "doc_id": "3f1c9a5e-8a1b-4a53-9c0e-2b6f7d1e4a90", "indexed_chunks": 9, "ungrounded_dropped": 2,
        "scan": {"total_batches": 4, "scanned_batches": 3, "failed_batches": 1, "retries": 3},
        "warning": "Partial scan: the LLM scanned 3 of 4 parts of the file. 1 part(s) failed after 3 retries.",
        "findings": [{"type": "NPPI", "text_snippet": "***-**-6789", "category": "ssn", "confidence": 0.97, "source": "rule"}],
    }))
    sample = tmp_path / "s.txt"
    sample.write_text("x")

    page.goto(f"{base_url}/static/index.html")
    page.set_input_files("#file-input", str(sample))
    page.click("#upload-form button[type=submit]")
    page.wait_for_selector("#upload-status:has-text('PARTIAL SCAN')")

    status = page.locator("#upload-status").inner_text()
    assert "scanned only 3 of 4 parts" in status
    assert "2 LLM finding(s) were dropped" in status
    assert page.locator("#upload-warning").is_visible()
    assert "Partial scan" in page.locator("#upload-warning").inner_text()


def test_category_answer_shows_masked_values_exactly(page, base_url, tmp_path):
    """Masked values in a server-built Markdown table survive marked + DOMPurify unchanged."""
    import rag

    findings = [
        {"type": "HRCI", "text_snippet": "$98,500", "category": "compensation", "confidence": 0.9, "source": "llm"},
        {"type": "NPPI", "text_snippet": "123-45-6789", "category": "ssn", "confidence": 0.97, "source": "rule"},
        # Excel rows are extracted with " | " separators, so values can contain pipes
        {"type": "HRCI", "text_snippet": "Name: Jordan Smith | Salary: $98,500", "category": "compensation", "confidence": 0.9, "source": "llm"},
    ]
    expected = ["$**,***", "***-**-6789", "Name: J. S. | Salary: $**,***"]
    answers = {
        "which employee salary is shown": rag.answer_from_findings(rag.route_question("which employee salary is shown"), findings),
        "list every HRCI and NPPI item": rag.answer_from_findings(rag.route_question("list every HRCI and NPPI item"), findings),
    }

    page.route(f"{base_url}/upload", lambda r: r.fulfill(json={
        "doc_id": "3f1c9a5e-8a1b-4a53-9c0e-2b6f7d1e4a90", "indexed_chunks": 1, "warning": None,
        "findings": [], "ungrounded_dropped": 0, "scan": {"total_batches": 1, "scanned_batches": 1, "failed_batches": 0, "retries": 0},
    }))
    page.route(f"{base_url}/ask", lambda r: r.fulfill(json={
        "answer": answers[current["q"]],
        "retrieved_chunk_ids": [], "warning": None, "route": "findings",
    }))
    current = {}
    sample = tmp_path / "s.txt"
    sample.write_text("x")
    page.goto(f"{base_url}/static/index.html")
    page.set_input_files("#file-input", str(sample))
    page.click("#upload-form button[type=submit]")
    page.wait_for_selector("#upload-status:has-text('Upload complete')")

    def displayed_values(question):
        current["q"] = question
        page.evaluate("document.getElementById('chat-answer-content').replaceChildren()")
        page.fill("#chat-input", question)
        page.click("#chat-submit")
        page.wait_for_selector("#chat-answer:not(.hidden) #chat-answer-content table")
        rows = page.locator("#chat-answer-content table tbody tr")
        return [rows.nth(i).locator("td").nth(2).inner_text() for i in range(rows.count())]

    assert displayed_values("which employee salary is shown") == [expected[0], expected[2]]
    assert displayed_values("list every HRCI and NPPI item") == expected
    # Nothing in the answer was turned into emphasis
    assert page.locator("#chat-answer-content table em, #chat-answer-content table strong").count() == 0
