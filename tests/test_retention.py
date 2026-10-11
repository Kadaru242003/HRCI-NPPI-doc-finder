"""DELETE /documents/{doc_id} and automatic deletion after DOC_TTL_HOURS."""
import json
import os
import time

import pytest

import api
import rag
import retention
import store

DOC = "SSN: 123-45-6789\nSalary: $90,000\n"


def upload(client, text=DOC, name="emp.txt"):
    res = client.post("/upload", files={"file": (name, text.encode(), "text/plain")})
    assert res.status_code == 200, res.text
    return res.json()["doc_id"]


def stored_ids(doc_id):
    return store.collection.get(where={"doc_id": doc_id}, include=[])["ids"]


def files_for(doc_id, upload_dir):
    return [n for n in os.listdir(upload_dir) if n.startswith(doc_id)]


def test_delete_removes_chunks_findings_and_files(client, fake_llm, tmp_path):
    fake_llm.reply = json.dumps([{"type": "HRCI", "text_snippet": "Salary: $90,000", "category": "salary", "confidence": 0.9}])
    doc_id = upload(client)
    other = upload(client, "Unrelated memo.")
    assert stored_ids(doc_id) and files_for(doc_id, tmp_path) and rag.load_findings(doc_id) is not None

    res = client.delete(f"/documents/{doc_id}")

    assert res.status_code == 200
    assert res.json() == {"doc_id": doc_id, "deleted_chunks": 1, "deleted_findings": True, "deleted_files": 1}
    assert stored_ids(doc_id) == []
    assert files_for(doc_id, tmp_path) == []
    assert rag.load_findings(doc_id) is None
    assert stored_ids(other)  # other documents are untouched

    ask = client.post("/ask", data={"doc_id": doc_id, "question": "summary"})
    assert ask.json()["answer"] == "No document found."
    assert client.delete(f"/documents/{doc_id}").status_code == 404


@pytest.mark.parametrize("bad_id", ["not-a-uuid", "..", "*", "123", "6F9619FF-8B86-D011-B42D-00C04FC964FF"])
def test_delete_rejects_non_canonical_ids(client, bad_id):
    assert client.delete(f"/documents/{bad_id}").status_code in (400, 404, 405)


def test_delete_cannot_reach_outside_the_upload_folder(client, fake_llm, tmp_path):
    outside = tmp_path.parent / "keep-me.txt"
    outside.write_text("x")
    for bad in ["..%2Fkeep-me", "%2E%2E"]:
        client.delete(f"/documents/{bad}")
    assert outside.exists()


def test_no_extracted_text_copy_is_written(client, fake_llm, tmp_path):
    doc_id = upload(client)
    assert files_for(doc_id, tmp_path) == [f"{doc_id}.txt"]  # just the upload itself


def test_expired_documents_are_purged(client, fake_llm, tmp_path):
    old = upload(client)
    later = time.time() + 25 * 3600  # 25 hours later, default TTL is 24 hours
    fresh_file = tmp_path / "unrelated-fresh.tmp"
    fresh_file.write_text("x")
    os.utime(fresh_file, (later, later))

    assert retention.purge_expired(str(tmp_path), now=later) == 1

    assert stored_ids(old) == []
    assert files_for(old, tmp_path) == []
    assert fresh_file.exists()  # not older than the TTL


def test_documents_within_ttl_are_kept(client, fake_llm, tmp_path):
    doc_id = upload(client)
    assert retention.purge_expired(str(tmp_path), now=time.time() + 23 * 3600) == 0
    assert stored_ids(doc_id)


def test_ttl_of_zero_disables_expiry(client, fake_llm, tmp_path):
    doc_id = upload(client)
    assert retention.purge_expired(str(tmp_path), ttl_hours=0, now=time.time() + 10**9) == 0
    assert stored_ids(doc_id)


def test_expiry_is_checked_on_each_request(client, fake_llm, tmp_path, monkeypatch):
    doc_id = upload(client)
    real_time = time.time
    monkeypatch.setattr(retention.time, "time", lambda: real_time() + 25 * 3600)

    client.get("/")  # any request triggers the check

    assert stored_ids(doc_id) == []
    assert files_for(doc_id, tmp_path) == []


def test_expiry_is_checked_on_startup(client, fake_llm, tmp_path, monkeypatch):
    doc_id = upload(client)
    real_time = time.time
    monkeypatch.setattr(retention.time, "time", lambda: real_time() + 25 * 3600)

    api.startup_event()

    assert stored_ids(doc_id) == []
