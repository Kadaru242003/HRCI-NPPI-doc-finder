"""Detection batches are retried with exponential backoff, honoring Retry-After."""
import json

import pytest

import rag
from conftest import groq_connection_error, groq_model_not_found, groq_status_error

TEXT = "Base salary $91,000 for the new hire."
FOUND = json.dumps([{"type": "HRCI", "text_snippet": "Base salary $91,000", "category": "salary", "confidence": 0.9}])


def failing_then(errors, final=FOUND):
    """A fake reply function: raise (or return) each item in `errors`, then return `final`."""
    queue = list(errors)

    def reply(messages):
        if queue:
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return final

    return reply


def upload(client, text=TEXT):
    return client.post("/upload", files={"file": ("memo.txt", text.encode(), "text/plain")})


def test_rate_limit_waits_for_retry_after(client, fake_llm, retry_sleeps):
    fake_llm.reply = failing_then([groq_status_error(429, retry_after=7)])

    body = upload(client).json()

    assert retry_sleeps == [7.0]
    assert len(fake_llm.calls) == 2
    assert body["warning"] is None
    assert body["scan"] == {"total_batches": 1, "scanned_batches": 1, "failed_batches": 0, "retries": 1}
    assert [f["category"] for f in body["findings"]] == ["compensation"]


def test_server_errors_back_off_exponentially(client, fake_llm, retry_sleeps):
    fake_llm.reply = failing_then([groq_status_error(500), groq_status_error(503), groq_connection_error()])

    body = upload(client).json()

    assert retry_sleeps == [1.0, 2.0, 4.0]
    assert body["scan"]["retries"] == 3 and body["scan"]["failed_batches"] == 0


def test_unusable_reply_is_retried(client, fake_llm, retry_sleeps):
    fake_llm.reply = failing_then(["Sorry, here is a summary instead.", None])

    body = upload(client).json()

    assert retry_sleeps == [1.0, 2.0]
    assert body["findings"] and body["warning"] is None


def test_retry_after_is_capped(client, fake_llm, retry_sleeps):
    fake_llm.reply = failing_then([groq_status_error(429, retry_after=3600)])
    upload(client)
    assert retry_sleeps == [rag.LLM_MAX_RETRY_WAIT]


def test_permanent_errors_are_not_retried(client, fake_llm, retry_sleeps):
    fake_llm.error = groq_model_not_found()

    res = upload(client)

    assert res.status_code == 502
    assert retry_sleeps == [] and len(fake_llm.calls) == 1


@pytest.mark.parametrize("status", [400, 401, 403])
def test_client_errors_are_not_retried(client, fake_llm, retry_sleeps, status):
    fake_llm.error = groq_status_error(status)
    assert upload(client).status_code == 502
    assert retry_sleeps == []


def test_gives_up_after_max_retries(client, fake_llm, retry_sleeps):
    fake_llm.error = groq_status_error(429)

    res = upload(client)

    assert res.status_code == 502
    assert len(fake_llm.calls) == rag.LLM_MAX_RETRIES + 1
    assert retry_sleeps == [1.0, 2.0, 4.0]
    assert "HTTP 429" in res.json()["detail"]
