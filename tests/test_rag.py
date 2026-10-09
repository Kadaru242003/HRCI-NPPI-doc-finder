import json
import os
import re
import subprocess
import sys
import textwrap

import pytest
from chromadb.api.client import SharedSystemClient

import rag
import store

SSN = re.compile(r"\d{3}-\d{2}-\d{4}")


def add_chunks(doc_id, texts):
    """Store chunks the way ingest.index_file does."""
    store.collection.add(
        ids=[f"{doc_id}_{i}" for i in range(len(texts))],
        documents=texts,
        metadatas=[{"doc_id": doc_id, "kind": "chunk", "chunk_index": i} for i in range(len(texts))],
        embeddings=store.embed(texts),
    )


def upload(client, name, text):
    res = client.post("/upload", files={"file": (name, text.encode(), "text/plain")})
    assert res.status_code == 200
    return res.json()


def text_to_label(messages):
    prompt = messages[-1]["content"]
    return prompt.split("Text to label:\n---\n", 1)[1].rsplit("\n---", 1)[0]


def ssn_detector(messages):
    """Fake detection LLM: reports every SSN-like pattern in the text it was given."""
    found = SSN.findall(text_to_label(messages))
    return json.dumps(
        [{"type": "NPPI", "text_snippet": s, "category": "ssn", "confidence": 0.9} for s in found]
    )


PAYROLL = [
    "Quarterly payroll: base salary and annual bonus for the sales team.",
    "Office picnic is on Friday; bring snacks and sunscreen.",
    "Bank routing number and checking account details for direct deposit.",
    "Salary bands were revised and every bonus target increased.",
    "Parking garage closes early during the holiday weekend.",
]


# ---------------------------------------------------------------- retrieval


def test_retrieval_returns_most_similar_chunks_first(chroma_dir):
    add_chunks("doc", PAYROLL)

    results = rag.retrieve_chunks("doc", "what salary and bonus does the team get", top_k=2)

    assert {r["id"] for r in results} == {"doc_0", "doc_3"}
    assert [r["distance"] for r in results] == sorted(r["distance"] for r in results)
    assert results[0]["text"] in (PAYROLL[0], PAYROLL[3])


def test_retrieval_ranks_by_similarity(chroma_dir):
    add_chunks("doc", PAYROLL)

    results = rag.retrieve_chunks("doc", "bank routing number checking account", top_k=5)

    assert results[0]["id"] == "doc_2"
    assert len(results) == 5


def test_retrieval_only_returns_chunks_from_the_requested_doc(chroma_dir):
    add_chunks("doc_a", PAYROLL)
    # doc_b matches the question exactly, so it would win if the filter leaked
    add_chunks("doc_b", ["bank routing number checking account"])

    results = rag.retrieve_chunks("doc_a", "bank routing number checking account", top_k=10)

    assert results
    assert all(r["id"].startswith("doc_a_") for r in results)


def test_retrieval_excludes_the_stored_findings_record(client, fake_llm):
    fake_llm.reply = json.dumps(
        [{"type": "NPPI", "text_snippet": "routing number", "category": "bank", "confidence": 0.9}]
    )
    doc_id = upload(client, "a.txt", "Bank routing number 021000021")["doc_id"]

    ids = [r["id"] for r in rag.retrieve_chunks(doc_id, "routing number bank", top_k=10)]

    assert ids == [f"{doc_id}_0"]
    assert rag.load_findings(doc_id)  # the findings record does exist


def test_retrieval_with_top_k_larger_than_doc(chroma_dir):
    add_chunks("doc", PAYROLL[:2])

    assert len(rag.retrieve_chunks("doc", "salary", top_k=50)) == 2


def test_retrieval_for_unknown_doc_is_empty(chroma_dir):
    add_chunks("doc", PAYROLL)

    assert rag.retrieve_chunks("missing", "salary", top_k=5) == []


# ---------------------------------------------------------------- /ask with retrieval


def test_ask_sends_only_retrieved_chunks_to_llm(client, fake_llm, chroma_dir):
    add_chunks("doc", PAYROLL)
    fake_llm.reply = "Answer"

    res = client.post("/ask", data={"doc_id": "doc", "question": "salary and bonus", "top_k": "2"})

    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "Answer"
    assert set(body["retrieved_chunk_ids"]) == {"doc_0", "doc_3"}

    prompt = fake_llm.last_user_prompt()
    assert PAYROLL[0] in prompt and PAYROLL[3] in prompt
    for i in (1, 2, 4):
        assert PAYROLL[i] not in prompt


def test_ask_defaults_to_top_5(client, fake_llm, chroma_dir):
    add_chunks("doc", [f"note number {i} about salary" for i in range(8)])

    res = client.post("/ask", data={"doc_id": "doc", "question": "salary"})

    assert rag.DEFAULT_TOP_K == 5
    assert len(res.json()["retrieved_chunk_ids"]) == 5


@pytest.mark.parametrize("top_k", ["0", "51", "abc"])
def test_ask_rejects_invalid_top_k(client, fake_llm, top_k):
    res = client.post("/ask", data={"doc_id": "doc", "question": "q", "top_k": top_k})

    assert res.status_code == 422
    assert fake_llm.calls == []


# ---------------------------------------------------------------- persistence


def test_data_survives_a_client_restart(client, fake_llm, chroma_dir):
    fake_llm.reply = json.dumps(
        [{"type": "HRCI", "text_snippet": "Salary: $102,000", "category": "salary", "confidence": 0.9}]
    )
    doc_id = upload(client, "employee.txt", "Employee salary: $102,000 and bonus")["doc_id"]
    old_client = store.client

    # Simulate a server restart: drop every cached Chroma system, then reopen from disk.
    SharedSystemClient.clear_system_cache()
    store.connect(chroma_dir)

    assert store.client is not old_client
    assert rag.load_chunks_for_doc(doc_id) == ["Employee salary: $102,000 and bonus"]
    assert rag.load_findings(doc_id)[0]["text_snippet"] == "Salary: $102,000"

    fake_llm.reply = "Answer"
    res = client.post("/ask", data={"doc_id": doc_id, "question": "salary"})
    assert res.json()["retrieved_chunk_ids"] == [f"{doc_id}_0"]


def test_data_is_readable_from_a_separate_process(client, fake_llm, chroma_dir):
    """A fresh interpreter shares no memory with this one, so it can only see what is on disk."""
    doc_id = upload(client, "employee.txt", "Employee salary: $102,000 and bonus")["doc_id"]

    script = textwrap.dedent(
        """
        import json, sys
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(path=sys.argv[1], settings=Settings(anonymized_telemetry=False))
        col = client.get_collection("documents")
        rows = col.get(where={"doc_id": sys.argv[2]})
        print(json.dumps(sorted(rows["ids"])))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script, chroma_dir, doc_id],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "ANONYMIZED_TELEMETRY": "False"},
        timeout=120,
    )

    assert json.loads(out.stdout.strip().splitlines()[-1]) == [f"{doc_id}_0", f"{doc_id}_findings"]


# ---------------------------------------------------------------- whole-file detection


def test_text_after_4000_chars_is_detected(client, fake_llm):
    fake_llm.reply = ssn_detector
    text = "filler text " * 500 + "SSN: 987-65-4321 near the end."  # SSN starts at char 6000

    body = upload(client, "long.txt", text)

    assert body["indexed_chunks"] > 4
    assert {"type": "NPPI", "text_snippet": "987-65-4321", "category": "ssn", "confidence": 0.9} in body[
        "findings"
    ]
    assert len(fake_llm.calls) > 1
    for call in fake_llm.calls:
        assert len(text_to_label(call["messages"])) <= rag.DETECT_BATCH_CHARS + 10


def test_every_chunk_is_sent_to_detection(client, fake_llm):
    fake_llm.reply = "[]"
    text = "".join(f"line {i:05d} " for i in range(1500))  # ~16,500 chars

    doc_id = upload(client, "long.txt", text)["doc_id"]

    sent = "\n".join(text_to_label(c["messages"]) for c in fake_llm.calls)
    for chunk in rag.load_chunks_for_doc(doc_id):
        assert chunk in sent


def test_span_in_chunk_overlap_is_reported_once(client, fake_llm):
    fake_llm.reply = ssn_detector
    # Chunks start every 800 chars and are 1000 long, so chars 3200-3400 are in
    # chunks 3 and 4, which land in different detection batches.
    text = "a" * 3250 + "123-45-6789" + "b" * 3000

    body = upload(client, "overlap.txt", text)

    labelled = [text_to_label(c["messages"]) for c in fake_llm.calls]
    assert sum("123-45-6789" in t for t in labelled) == 2
    assert [f["text_snippet"] for f in body["findings"]] == ["123-45-6789"]


def test_one_failed_batch_does_not_drop_other_findings(client, fake_llm):
    calls = {"n": 0}

    def flaky(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"
        return ssn_detector(messages)

    fake_llm.reply = flaky
    text = "x" * 6000 + " 111-22-3333 "

    body = upload(client, "flaky.txt", text)

    assert [f["text_snippet"] for f in body["findings"]] == ["111-22-3333"]
    assert body["warning"].startswith("LLM detection failed for 1 of 2 parts of the file")


# ---------------------------------------------------------------- helpers


def test_batch_chunks_respects_limit_and_keeps_order():
    chunks = ["a" * 1000, "b" * 1000, "c" * 1000, "d" * 1000, "e" * 1000, "f" * 5000]

    batches = rag.batch_chunks(chunks, max_chars=4000)

    assert batches == ["\n".join(chunks[:4]), chunks[4], chunks[5]]


def test_merge_findings_dedupes_and_keeps_highest_confidence():
    a = {"type": "NPPI", "text_snippet": "123-45-6789", "category": "ssn", "confidence": 0.6}
    a_better = {"type": "nppi", "text_snippet": " 123-45-6789 ", "category": "SSN", "confidence": 0.95}
    b = {"type": "HRCI", "text_snippet": "Salary: $1", "category": "salary", "confidence": 0.9}
    same_text_other_type = {"type": "HRCI", "text_snippet": "123-45-6789", "category": "x", "confidence": 0.5}

    merged = rag.merge_findings([[a, b], [a_better, same_text_other_type]])

    assert merged == [a_better, b, same_text_other_type]


def test_merge_findings_tolerates_odd_items():
    merged = rag.merge_findings([["loose string", {"type": "HRCI"}], ["loose string"]])

    assert merged == ["loose string", {"type": "HRCI"}]
