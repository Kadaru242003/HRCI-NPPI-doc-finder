"""Masking of sensitive values before they leave the server."""
import re

_DIGIT = re.compile(r"\d")


def mask_digits(text: str, keep_last: int = 4) -> str:
    """Replace every digit with '*' except the last `keep_last` digits.

    Separators and labels are kept, so "SSN: 123-45-6789" becomes "SSN: ***-**-6789".
    """
    total = len(_DIGIT.findall(text))
    keep = keep_last if total > keep_last else 0
    seen = 0
    out = []
    for ch in text:
        if ch.isdigit():
            seen += 1
            out.append(ch if seen > total - keep else "*")
        else:
            out.append(ch)
    return "".join(out)


def mask_snippet(finding) -> str:
    """Masked form of a finding's text_snippet.

    NPPI values (SSNs, cards, routing and account numbers) keep their last 4
    digits. HRCI snippets have every digit masked, so amounts such as salaries
    are hidden while the surrounding words stay readable.
    """
    if not isinstance(finding, dict):
        return mask_digits(str(finding), keep_last=0)
    snippet = str(finding.get("text_snippet", ""))
    if str(finding.get("type", "")).upper() == "NPPI":
        return mask_digits(snippet, keep_last=4)
    return mask_digits(snippet, keep_last=0)
