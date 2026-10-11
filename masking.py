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


def mask_finding(finding):
    """Copy of a finding with its text_snippet masked (for API responses)."""
    if not isinstance(finding, dict):
        return mask_digits(str(finding), keep_last=0)
    return dict(finding, text_snippet=mask_snippet(finding))


def _digit_pattern(digits: str):
    """Regex for a digit sequence allowing spaces, dashes, commas or periods between digits."""
    return re.compile(r"(?<!\d)" + r"[ ,.-]?".join(re.escape(d) for d in digits) + r"(?!\d)")


_NUMBER = re.compile(r"\d[\d ,.-]*\d|\d")


def mask_text(text: str, findings) -> str:
    """Mask every known sensitive value in free text (excerpts sent to the LLM, LLM answers).

    1. Each stored finding's snippet is replaced by its masked form.
    2. The values inside findings are also found by their digits, in any spacing,
       dash or comma format: NPPI as a whole ("123456789" for "123-45-6789", last 4
       kept), HRCI per number ("$102000" for "Salary: $102,000", fully masked).
    3. Structured NPPI found by the rule detectors is masked too.
    """
    import detectors  # local import: detectors has no dependency on this module

    text = text or ""
    for finding in sorted((f for f in findings or [] if isinstance(f, dict)), key=lambda f: -len(str(f.get("text_snippet", "")))):
        snippet = str(finding.get("text_snippet", ""))
        if not snippet:
            continue
        text = text.replace(snippet, mask_snippet(finding))
        if str(finding.get("type", "")).upper() == "NPPI":
            digits = re.sub(r"\D", "", snippet)
            if len(digits) >= 6:
                text = _digit_pattern(digits).sub(lambda m: mask_digits(m.group(), keep_last=4), text)
        else:
            for number in _NUMBER.findall(snippet):
                digits = re.sub(r"\D", "", number)
                if len(digits) >= 3:
                    text = _digit_pattern(digits).sub(lambda m: mask_digits(m.group(), keep_last=0), text)
    for rule_finding in sorted(detectors.detect(text), key=lambda f: -len(f["text_snippet"])):
        text = text.replace(rule_finding["text_snippet"], mask_snippet(rule_finding))
    return text
