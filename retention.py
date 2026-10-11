"""Deleting documents: on request (DELETE /documents/{doc_id}) and after a TTL."""
import logging
import os
import time
import uuid

import store

log = logging.getLogger("riskbot")

# Documents older than this are deleted automatically; 0 or less disables expiry.
DOC_TTL_HOURS = float(os.getenv("DOC_TTL_HOURS", "24"))


def is_valid_doc_id(doc_id: str) -> bool:
    """Only canonical UUIDs, so an id can never reach paths outside the upload folder."""
    try:
        return str(uuid.UUID(doc_id)) == doc_id
    except (ValueError, TypeError, AttributeError):
        return False


def _files_for(doc_id: str, upload_dir: str) -> list:
    if not os.path.isdir(upload_dir):
        return []
    return [os.path.join(upload_dir, name) for name in os.listdir(upload_dir) if name.startswith(doc_id + ".")]


def delete_document(doc_id: str, upload_dir: str) -> dict:
    """Remove a document's chunks, findings record and stored files. Returns what was removed."""
    if not is_valid_doc_id(doc_id):
        raise ValueError("invalid document id")
    ids = store.collection.get(where={"doc_id": doc_id}, include=[])["ids"]
    if ids:
        store.collection.delete(ids=ids)
    removed_files = 0
    for path in _files_for(doc_id, upload_dir):
        try:
            os.remove(path)
            removed_files += 1
        except FileNotFoundError:
            pass
    chunks = sum(1 for i in ids if not i.endswith("_findings"))
    findings = any(i.endswith("_findings") for i in ids)
    return {"doc_id": doc_id, "deleted_chunks": chunks, "deleted_findings": findings, "deleted_files": removed_files}


def purge_expired(upload_dir: str, ttl_hours: float = None, now: float = None) -> int:
    """Delete every document created more than ttl_hours ago; returns how many were deleted."""
    ttl_hours = DOC_TTL_HOURS if ttl_hours is None else ttl_hours
    if ttl_hours <= 0:
        return 0
    cutoff = (time.time() if now is None else now) - ttl_hours * 3600
    rows = store.collection.get(where={"created_at": {"$lt": cutoff}}, include=["metadatas"])
    doc_ids = {m["doc_id"] for m in rows["metadatas"] if m and is_valid_doc_id(m.get("doc_id"))}
    for doc_id in doc_ids:
        delete_document(doc_id, upload_dir)
    # Stored files whose index entries are gone (or never existed, e.g. empty uploads)
    if os.path.isdir(upload_dir):
        for name in os.listdir(upload_dir):
            path = os.path.join(upload_dir, name)
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
    if doc_ids:
        log.info("retention: deleted %d expired document(s)", len(doc_ids))
    return len(doc_ids)
