"""Rule-based NPPI detectors and the hybrid merge (all values synthetic)."""
import pytest

import detectors
import rag


def found(text):
    return [(f["category"], f["text_snippet"]) for f in detectors.detect(text)]


# ---------------------------------------------------------------- SSN

@pytest.mark.parametrize("ssn", ["123-45-6789", "123 45 6789", "001-01-0001", "899-99-9999", "665-12-3456"])
def test_valid_ssns_are_found(ssn):
    assert found(f"SSN on file: {ssn}.") == [("ssn", ssn)]


@pytest.mark.parametrize(
    "ssn",
    [
        "000-12-3456",  # area 000
        "666-12-3456",  # area 666
        "900-12-3456",  # area 900-999
        "987-65-4321",
        "123-00-4567",  # group 00
        "123-45-0000",  # serial 0000
        "123-45 6789",  # mixed separators
        "123456789",  # unformatted
        "1123-45-6789",  # part of a longer number
    ],
)
def test_invalid_or_unformatted_ssns_are_ignored(ssn):
    assert ("ssn", ssn) not in found(f"Reference {ssn} archived.")


# ---------------------------------------------------------------- cards

@pytest.mark.parametrize(
    "card",
    ["4111 1111 1111 1111", "5555555555554444", "3782 822463 10005", "6011-1111-1111-1117", "4242 4242 4242 4242"],
)
def test_luhn_valid_test_cards_are_found(card):
    assert found(f"Card {card} expires 09/28") == [("credit_card", card)]


def test_card_failing_luhn_is_ignored():
    assert found("Card 4111 1111 1111 1112") == []


def test_luhn_valid_number_without_a_card_network_prefix_is_ignored():
    tracking = "9216367364158133"  # passes Luhn, starts with 9: no card network
    assert detectors.luhn_valid(tracking)
    assert found(f"Tracking number {tracking}") == []


# ---------------------------------------------------------------- routing numbers

def test_routing_number_needs_checksum_prefix_and_keyword():
    assert detectors.aba_checksum_valid("021000021")
    assert found("Routing number: 021000021") == [("routing_number", "021000021")]
    assert found("ABA 021000021, checking") == [("routing_number", "021000021")]
    assert found("Order ID 021000021 shipped") == []  # valid checksum, no keyword
    assert found("Routing number: 021000022") == []  # checksum fails
    bad_prefix = next(f"42100002{d}" for d in range(10) if detectors.aba_checksum_valid(f"42100002{d}"))
    assert not detectors.aba_prefix_valid(bad_prefix)
    assert found(f"Routing number: {bad_prefix}") == []  # checksum passes, prefix 42 is unassigned


# ---------------------------------------------------------------- bank accounts

@pytest.mark.parametrize(
    "text, account",
    [
        ("Checking account number: 4471029381", "4471029381"),
        ("Deposit to acct #88120034 at the credit union.", "88120034"),
        ("Savings account 600219384756 was closed.", "600219384756"),
        ("Account: 12345678", "12345678"),
    ],
)
def test_bank_accounts_near_keywords(text, account):
    assert ("bank_account", account) in found(text)


@pytest.mark.parametrize(
    "text",
    ["Account manager: Dana. Meeting at 10:30.", "Account 12345 is short.", "Order 4471029381 shipped."],
)
def test_account_negatives(text):
    assert [c for c, _ in found(text) if c == "bank_account"] == []


def test_overlapping_matches_keep_the_higher_priority_rule():
    # A routing number right after "account": counted once, as routing (higher priority)
    assert found("Routing / account: 021000021") == [("routing_number", "021000021")]


def test_findings_have_the_shared_shape():
    (f,) = detectors.detect("SSN 123-45-6789")
    assert f == {"type": "NPPI", "text_snippet": "123-45-6789", "category": "ssn", "confidence": 0.97, "source": "rule"}


# ---------------------------------------------------------------- hybrid merge

def test_rule_finding_wins_over_llm_finding_for_the_same_value():
    rule = detectors.detect("SSN 123-45-6789")[0]
    llm = {"type": "NPPI", "text_snippet": "SSN: 123-45-6789", "category": "ssn", "confidence": 0.99, "source": "llm"}
    assert rag.merge_findings([[llm], [rule]]) == [rule]
    assert rag.merge_findings([[rule], [llm]]) == [rule]


def test_llm_only_findings_are_kept_with_their_source():
    llm = {"type": "HRCI", "text_snippet": "base salary $98,500", "category": "salary", "confidence": 0.8}
    normalized = rag._normalize_llm_finding(llm)
    assert normalized["source"] == "llm" and normalized["category"] == "compensation"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("SSN", "ssn"), ("social security", "ssn"), ("Credit Card", "credit_card"), ("ABA", "routing_number"),
        ("bank account", "bank_account"), ("salary", "compensation"), ("PIP", "performance"),
        ("severance", "termination"), ("medical leave", "medical"), ("phone", None), ("", None),
    ],
)
def test_canonical_category(raw, expected):
    assert rag.canonical_category(raw) == expected


def test_scan_batches_runs_rules_even_when_llm_is_off():
    result = rag.scan_batches(["SSN 123-45-6789 and card 4111 1111 1111 1111"], use_llm=False)
    assert [f["category"] for f in result.findings] == ["ssn", "credit_card"]
    assert result.errors == []
