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


# Capitalized words common in HR text that are not names. They are trimmed from
# the edges of a capitalized run before deciding whether the run is a name.
_NOT_NAME_WORDS = {
    "a", "an", "the", "this", "that", "in", "on", "at", "for", "from", "to", "per", "of", "and",
    "dear", "hi", "hello", "mr", "mrs", "ms", "dr", "approved", "effective", "employee", "employees",
    "manager", "human", "resources", "hr", "salary", "bonus", "base", "annual", "review", "performance",
    "improvement", "plan", "final", "written", "warning", "medical", "leave", "workers", "compensation",
    "offer", "letter", "termination", "severance", "exit", "interview", "notes", "note", "payroll",
    "department", "team", "senior", "junior", "director", "vice", "president", "chief", "officer",
    "engineer", "analyst", "sales", "marketing", "finance", "operations", "legal", "company", "inc",
    "llc", "corp", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december",
}
_CAPITALIZED_RUN = re.compile(r"\b[A-Z][a-z]+(?:-[A-Z]?[a-z]+)?(?:\s+[A-Z][a-z]+(?:-[A-Z]?[a-z]+)?)+\b")


def find_names(text: str) -> list:
    """Likely person names: runs of 2 or 3 capitalized words that are not common HR words.

    A heuristic, not named-entity recognition: single first names, all-caps names
    and names written in lowercase are not found.
    """
    names = []
    for m in _CAPITALIZED_RUN.finditer(text or ""):
        words = m.group().split()
        while words and words[0].lower() in _NOT_NAME_WORDS:
            words.pop(0)
        while words and words[-1].lower() in _NOT_NAME_WORDS:
            words.pop()
        if 2 <= len(words) <= 3 and not any(w.lower() in _NOT_NAME_WORDS for w in words):
            names.append(" ".join(words))
    return names


def _initials(name: str) -> str:
    return " ".join(f"{w[0]}." for w in name.split())


def _replace_name(text: str, name: str) -> str:
    # A following period is absorbed so "... Riley Delacroix." becomes "... R. D." not "R. D.."
    return re.sub(r"\b" + re.escape(name) + r"\b\.?", _initials(name), text)


def mask_names(text: str) -> str:
    """Replace likely person names with initials: "Jordan Smith" -> "J. S."."""
    for name in sorted(set(find_names(text)), key=len, reverse=True):
        text = _replace_name(text, name)
    return text


def mask_snippet(finding) -> str:
    """Masked form of a finding's text_snippet.

    NPPI values (SSNs, cards, routing and account numbers) keep their last 4
    digits. HRCI snippets have every digit masked, so amounts such as salaries
    are hidden while the surrounding words stay readable, and person names are
    reduced to initials ("Jordan Smith" -> "J. S.").
    """
    if not isinstance(finding, dict):
        return mask_digits(str(finding), keep_last=0)
    snippet = str(finding.get("text_snippet", ""))
    if str(finding.get("type", "")).upper() == "NPPI":
        return mask_digits(snippet, keep_last=4)
    return mask_names(mask_digits(snippet, keep_last=0))


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
            # Names in HRCI findings are masked wherever they appear, not only inside the snippet
            for name in find_names(snippet):
                text = _replace_name(text, name)
    for rule_finding in sorted(detectors.detect(text), key=lambda f: -len(f["text_snippet"])):
        text = text.replace(rule_finding["text_snippet"], mask_snippet(rule_finding))
    return text
