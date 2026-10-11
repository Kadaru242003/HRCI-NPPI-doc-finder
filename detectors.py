"""Deterministic detectors for structured NPPI.

The LLM handles HRCI and anything that needs judgment; these rules handle
values with a checkable structure. Every finding has the same shape as an LLM
finding plus "source": "rule". `text_snippet` is the exact matched text.
"""
import re

# ---------------------------------------------------------------- SSN
# 3-2-4 digits with one consistent separator (dash or space). Unformatted 9-digit
# runs are not treated as SSNs: they are too easy to confuse with other IDs.
_SSN = re.compile(r"(?<![\d-])(\d{3})([- ])(\d{2})\2(\d{4})(?![\d-])")


def valid_ssn_parts(area: str, group: str, serial: str) -> bool:
    """SSA rules: area not 000, 666 or 900-999; group not 00; serial not 0000."""
    a = int(area)
    return a != 0 and a != 666 and a < 900 and group != "00" and serial != "0000"


def find_ssns(text):
    for m in _SSN.finditer(text):
        if valid_ssn_parts(m.group(1), m.group(3), m.group(4)):
            yield m.start(), m.end(), "ssn", 0.97


# ---------------------------------------------------------------- payment cards
_CARD = re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])")
# Major network prefixes (IIN ranges): Visa, Mastercard, Amex, Discover, Diners, JCB.
_CARD_PREFIX = re.compile(r"^(?:4|5[1-5]|2(?:2[2-9]|[3-6]\d|7[01]|720)|3[47]|6(?:011|5|4[4-9])|3(?:0[0-5]|[68])|35)")


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, c in enumerate(reversed(digits)):
        d = int(c)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def find_cards(text):
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _CARD_PREFIX.match(digits) and luhn_valid(digits):
            yield m.start(), m.end(), "credit_card", 0.95


# ---------------------------------------------------------------- ABA routing numbers
_NINE = re.compile(r"(?<![\d-])\d{9}(?![\d-])")
_ROUTING_KEYWORD = re.compile(r"\b(?:routing|aba|rtn|transit)\b", re.IGNORECASE)


def aba_checksum_valid(d: str) -> bool:
    n = [int(c) for c in d]
    return (3 * (n[0] + n[3] + n[6]) + 7 * (n[1] + n[4] + n[7]) + (n[2] + n[5] + n[8])) % 10 == 0


def aba_prefix_valid(d: str) -> bool:
    """First two digits: 00-12 (banks), 21-32 (thrifts), 61-72 (electronic), 80 (travelers checks)."""
    p = int(d[:2])
    return p <= 12 or 21 <= p <= 32 or 61 <= p <= 72 or p == 80


def find_routing_numbers(text, window: int = 40):
    """Valid prefix and checksum AND a routing keyword shortly before the number.

    The keyword is required because about 1 in 10 random 9-digit IDs pass the checksum.
    """
    for m in _NINE.finditer(text):
        d = m.group()
        if aba_prefix_valid(d) and aba_checksum_valid(d) and _ROUTING_KEYWORD.search(text[max(0, m.start() - window): m.start()]):
            yield m.start(), m.end(), "routing_number", 0.93


# ---------------------------------------------------------------- bank account numbers
_ACCOUNT = re.compile(
    r"\b(?:acct|account|a/c|checking|savings)\b"
    r"(?:\s+(?:no\.?|number|num|#))?\s*[:#]?\s*#?\s*"
    r"(\d[\d -]{4,20}\d)(?![\d-])",
    re.IGNORECASE,
)


def find_bank_accounts(text):
    """6 to 17 digits directly after an account keyword (e.g. "Account number: 12345678")."""
    for m in _ACCOUNT.finditer(text):
        digits = re.sub(r"\D", "", m.group(1))
        if 6 <= len(digits) <= 17:
            yield m.start(1), m.end(1), "bank_account", 0.85


# ---------------------------------------------------------------- all rules

_RULES = [find_ssns, find_cards, find_routing_numbers, find_bank_accounts]


def detect(text: str) -> list:
    """Run every rule; when matches overlap, keep the first (higher-priority) rule's match."""
    taken = []
    findings = []
    for rule in _RULES:
        for start, end, category, confidence in rule(text):
            if any(start < e and s < end for s, e in taken):
                continue
            taken.append((start, end))
            findings.append({
                "type": "NPPI",
                "text_snippet": text[start:end],
                "category": category,
                "confidence": confidence,
                "source": "rule",
            })
    return findings
