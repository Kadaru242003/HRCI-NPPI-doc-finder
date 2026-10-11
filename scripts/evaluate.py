"""Evaluate detection on the labeled synthetic benchmark (eval/).

Reports precision, recall and F1 per category on the held-out split for three
pipelines, using the app's own extraction, chunking, batching and detection code:

- LLM only: the LLM on every batch, no rules, no grounding.
- Rules only: the deterministic NPPI detectors.
- Hybrid: rules + LLM + grounding (what /upload does).

LLM replies are cached in eval/llm_cache/ (one JSON file per request, keyed by a
hash of model, messages and temperature), so reruns cost nothing. Uncached calls
need GROQ_API_KEY and are paced to stay under Groq's free-tier limits; on HTTP 429
the call waits for Retry-After. Without a key, uncached requests are not made and
the LLM columns are reported as "not run" instead of a number.

Matching rules: eval/README.md. Usage:
    python scripts/evaluate.py                    # held-out table to stdout
    python scripts/evaluate.py --split dev        # the split used for tuning
    python scripts/evaluate.py --write-readme     # also refresh the table in README.md
    python scripts/evaluate.py --check-readme     # exit 1 if README.md's table is stale
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
EVAL_DIR = os.path.join(ROOT, "eval")
CACHE_DIR = os.path.join(EVAL_DIR, "llm_cache")
README = os.path.join(ROOT, "README.md")
START, END = "<!-- eval-table:start -->", "<!-- eval-table:end -->"

HAVE_KEY = bool(os.environ.get("GROQ_API_KEY"))
# rag reads the key and store opens a Chroma directory at import; keep both harmless here.
os.environ.setdefault("GROQ_API_KEY", "not-set-cache-only")
os.environ.setdefault("CHROMA_DIR", tempfile.mkdtemp(prefix="riskbot-eval-"))
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import ingest  # noqa: E402
import rag  # noqa: E402

CATEGORIES = [
    ("NPPI", "ssn"), ("NPPI", "credit_card"), ("NPPI", "routing_number"), ("NPPI", "bank_account"),
    ("HRCI", "compensation"), ("HRCI", "performance"), ("HRCI", "termination"), ("HRCI", "medical"),
]
LABEL_TYPE = dict((c, t) for t, c in CATEGORIES)


# ---------------------------------------------------------------- LLM cache

class CacheMiss(Exception):
    pass


class CachingGroq:
    """Stands in for groq_client: answers from eval/llm_cache, calls Groq only on a miss."""

    def __init__(self, allow_calls: bool, min_interval: float):
        self.allow_calls = allow_calls and HAVE_KEY
        self.min_interval = min_interval
        self.hits = self.calls = self.misses = 0
        self._last_call = 0.0
        self._real = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        key = hashlib.sha256(json.dumps(
            {"model": kwargs["model"], "messages": kwargs["messages"], "temperature": kwargs.get("temperature")},
            sort_keys=True,
        ).encode()).hexdigest()
        path = os.path.join(CACHE_DIR, f"{key}.json")
        if os.path.exists(path):
            self.hits += 1
            content = json.load(open(path, encoding="utf-8"))["content"]
        elif not self.allow_calls:
            self.misses += 1
            raise CacheMiss("not cached and no GROQ_API_KEY (or --no-llm-calls)")
        else:
            content = self._call_groq(kwargs)
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"model": kwargs["model"], "content": content}, f)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    def _call_groq(self, kwargs):
        import groq

        if self._real is None:
            self._real = groq.Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=0)
        for attempt in range(6):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                self.calls += 1
                return self._real.chat.completions.create(**kwargs).choices[0].message.content or ""
            except groq.RateLimitError as e:
                retry_after = float(e.response.headers.get("retry-after", 2 ** attempt * 5))
                print(f"  rate limited, waiting {retry_after:.0f}s", file=sys.stderr)
                time.sleep(min(retry_after, 120))
        raise RuntimeError("Groq kept rate limiting; rerun later (finished requests are cached)")


# ---------------------------------------------------------------- pipeline

def document_batches(path):
    """Same steps as /upload: extract, chunk, drop empty chunks, batch."""
    if path.endswith((".xlsx", ".xls")):
        text = ingest.extract_text_from_excel(path)
    else:
        text = ingest.load_text_from_file(path)
    chunks = [c.strip() for c in ingest.chunk_text(text) if c.strip()]
    return rag.batch_chunks(chunks)


PIPELINES = {
    "llm": dict(use_rules=False, ground=False),
    "rules": dict(use_llm=False),
    "hybrid": dict(),
}


# ---------------------------------------------------------------- matching (eval/README.md)

def _digits(s):
    return re.sub(r"\D", "", s or "")


def _norm(s):
    return " ".join((s or "").split()).lower()


def label_matches(pred, label):
    if label["type"] == "NPPI":
        d = _digits(label["value"])
        return bool(d) and d in _digits(pred.get("text_snippet"))
    p, v = _norm(pred.get("text_snippet")), _norm(label["value"])
    return v in p or (len(p) >= 4 and p in _norm(label["span"]))


def score_document(preds, labels, counts):
    matched = [False] * len(labels)
    for lab in labels:
        counts[lab["category"]]["labels"] += 1
    for pred in preds:
        if not isinstance(pred, dict):
            counts["other"]["fp"] += 1
            continue
        cat = rag.canonical_category(pred.get("category")) or "other"
        hits = [i for i, lab in enumerate(labels) if lab["category"] == cat and label_matches(pred, lab)]
        fresh = [i for i in hits if not matched[i]]
        if fresh:
            matched[fresh[0]] = True
            counts[cat]["tp"] += 1
        elif not hits:
            counts[cat]["fp"] += 1
        # else: duplicate of an already matched label, not counted


def new_counts():
    return {c: {"tp": 0, "fp": 0, "labels": 0} for _, c in CATEGORIES + [("", "other")]}


def prf(c):
    tp, fp, n = c["tp"], c["fp"], c["labels"]
    p = tp / (tp + fp) if tp + fp else None
    r = tp / n if n else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None or r is not None else None)
    return p, r, f


# ---------------------------------------------------------------- run

def evaluate(split, allow_calls, min_interval):
    labels = json.load(open(os.path.join(EVAL_DIR, "labels.json"), encoding="utf-8"))
    docs = [d for d in labels["documents"] if d["split"] == split]
    client = CachingGroq(allow_calls, min_interval)
    rag.groq_client = client
    rag.LLM_RETRY_SLEEP = lambda seconds: None  # the cache client already paces real calls

    results, llm_complete = {}, True
    for name, kwargs in PIPELINES.items():
        counts = new_counts()
        for doc in docs:
            batches = document_batches(os.path.join(EVAL_DIR, doc["file"]))
            result = rag.scan_batches(batches, **kwargs)
            if kwargs.get("use_llm", True) and result.errors:
                llm_complete = False
            score_document(result.findings, doc["labels"], counts)
        results[name] = counts
    return docs, results, llm_complete, client


def fmt(x):
    return "n/a" if x is None else f"{x:.2f}"


def table(split, docs, results, llm_complete):
    n_xlsx = sum(d["file"].endswith(".xlsx") for d in docs)
    lines = [
        f"Split: **{split}**, {len(docs)} documents ({len(docs) - n_xlsx} .txt, {n_xlsx} .xlsx), "
        f"model `{rag.LLM_MODEL}`. P / R / F1 per category.",
        "",
        "| Type | Category | Labels | LLM only | Rules only | Hybrid (rules + LLM + grounding) |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    rows = CATEGORIES + [("", "all (micro)")]
    for ftype, cat in rows:
        cells = []
        for name in ("llm", "rules", "hybrid"):
            counts = results[name]
            if cat == "all (micro)":
                c = {k: sum(v[k] for v in counts.values()) for k in ("tp", "fp", "labels")}
            else:
                c = counts[cat]
            if name != "rules" and not llm_complete:
                cells.append("not run")
            else:
                p, r, f = prf(c)
                cells.append(f"{fmt(p)} / {fmt(r)} / {fmt(f)}")
        n = sum(v["labels"] for k, v in results["rules"].items()) if cat == "all (micro)" else results["rules"][cat]["labels"]
        lines.append(f"| {ftype} | {cat} | {n} | " + " | ".join(cells) + " |")
    if not llm_complete:
        lines += ["", "LLM columns: **not run**. Some LLM replies are not in `eval/llm_cache/` and no "
                  "`GROQ_API_KEY` was available when this table was generated. Run "
                  "`GROQ_API_KEY=... python scripts/evaluate.py --write-readme` to fill them in."]
    other_fp = {name: results[name]["other"]["fp"] for name in results}
    if any(other_fp.values()):
        lines += ["", "Predictions in no category (counted as false positives in the micro row): "
                  + ", ".join(f"{k}: {v}" for k, v in other_fp.items())]
    lines += ["", "Generated by `python scripts/evaluate.py --write-readme`. Matching rules: `eval/README.md`."]
    return "\n".join(lines)


def replace_block(text, block):
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"README.md has no {START} ... {END} block")
    return pattern.sub(lambda m: f"{START}\n{block}\n{END}", text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="heldout", choices=["heldout", "dev"])
    parser.add_argument("--no-llm-calls", action="store_true", help="use cached LLM replies only")
    parser.add_argument("--min-interval", type=float, default=3.0, help="seconds between uncached Groq calls")
    parser.add_argument("--write-readme", action="store_true")
    parser.add_argument("--check-readme", action="store_true")
    args = parser.parse_args()

    allow_calls = not (args.no_llm_calls or args.check_readme)
    docs, results, llm_complete, client = evaluate(args.split, allow_calls, args.min_interval)
    block = table(args.split, docs, results, llm_complete)
    print(block)
    print(f"\nLLM cache: {client.hits} hits, {client.calls} calls made, {client.misses} misses", file=sys.stderr)

    if args.write_readme or args.check_readme:
        if args.split != "heldout":
            raise SystemExit("The README table is for the held-out split only")
        current = open(README, encoding="utf-8").read()
        updated = replace_block(current, block)
        if args.check_readme:
            if updated != current:
                print("README.md evaluation table is stale; run scripts/evaluate.py --write-readme", file=sys.stderr)
                return 1
            return 0
        open(README, "w", encoding="utf-8").write(updated)
        print("README.md updated", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
