"""The evaluation benchmark and scripts/evaluate.py."""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

from conftest import ROOT

spec = importlib.util.spec_from_file_location("evaluate", os.path.join(ROOT, "scripts", "evaluate.py"))
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)

LABELS = json.load(open(os.path.join(ROOT, "eval", "labels.json"), encoding="utf-8"))
PUBLISHED_TEST_CARDS = {
    "4111111111111111", "5555555555554444", "378282246310005", "6011111111111117",
    "4012888888881881", "5105105105105100", "371449635398431", "4242424242424242",
}


def test_benchmark_size_and_splits():
    docs = LABELS["documents"]
    assert len(docs) >= 40
    assert {d["split"] for d in docs} == {"dev", "heldout"}
    assert sum(d["file"].endswith(".xlsx") for d in docs) >= 10
    assert any(not d["labels"] for d in docs if d["split"] == "heldout")  # negative-only documents


def test_every_label_value_is_in_its_document():
    for doc in LABELS["documents"]:
        text = " ".join(b for b in evaluate.document_batches(os.path.join(ROOT, "eval", doc["file"])))
        for label in doc["labels"]:
            assert label["value"] in text, (doc["file"], label["value"])


def test_only_published_test_card_numbers_are_used():
    for doc in LABELS["documents"]:
        for label in doc["labels"]:
            if label["category"] == "credit_card":
                assert evaluate._digits(label["value"]) in PUBLISHED_TEST_CARDS


def test_matching_rules():
    ssn = {"type": "NPPI", "category": "ssn", "value": "123-45-6789", "span": "SSN: 123-45-6789"}
    pay = {"type": "HRCI", "category": "compensation", "value": "$98,500", "span": "Base salary increases to $98,500 today."}
    assert evaluate.label_matches({"text_snippet": "SSN 123 45 6789"}, ssn)
    assert not evaluate.label_matches({"text_snippet": "123-45-6780"}, ssn)
    assert evaluate.label_matches({"text_snippet": "salary increases to $98,500"}, pay)
    assert evaluate.label_matches({"text_snippet": "Base Salary"}, pay)  # inside the sentence
    assert not evaluate.label_matches({"text_snippet": "$98"}, pay)  # too short to count as inside


def test_scoring_counts_tp_fp_fn_and_ignores_duplicates():
    labels = [
        {"type": "NPPI", "category": "ssn", "value": "123-45-6789", "span": "x"},
        {"type": "NPPI", "category": "ssn", "value": "234-56-7890", "span": "x"},
    ]
    preds = [
        {"category": "ssn", "text_snippet": "123-45-6789"},  # TP
        {"category": "SSN", "text_snippet": "SSN: 123-45-6789"},  # duplicate, ignored
        {"category": "ssn", "text_snippet": "555-0142"},  # FP
        {"category": "phone", "text_snippet": "555-0142"},  # FP in "other"
    ]
    counts = evaluate.new_counts()
    evaluate.score_document(preds, labels, counts)
    assert counts["ssn"] == {"tp": 1, "fp": 1, "labels": 2}
    assert counts["other"]["fp"] == 1
    p, r, f = evaluate.prf(counts["ssn"])
    assert (p, r) == (0.5, 0.5) and f == pytest.approx(0.5)


def test_readme_evaluation_table_is_up_to_date():
    env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
    out = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--check-readme"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
