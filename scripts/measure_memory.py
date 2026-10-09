"""Measure the peak memory of a real RiskBot server process (Linux only).

Starts `uvicorn api:app`, uploads a 10,000-character .txt file, asks one
question, and reads the server's peak resident set size (VmHWM in
/proc/<pid>/status) after each step. Exits non-zero if the peak exceeds
--limit-mb.

The Groq client is pointed at an unreachable local address, so no LLM request
leaves the machine: detection and /ask fail fast and are handled as they would
be on a Groq outage. The embedding model is real; on the first run it is
downloaded to ~/.cache/chroma.

Usage (from the repository root):
    python scripts/measure_memory.py [--limit-mb 350] [--chars 10000]
"""
import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sample_text(n_chars: int = 10_000) -> str:
    lines, i = [], 0
    while sum(len(line) + 1 for line in lines) < n_chars:
        lines.append(
            f"Employee {i:04d}: salary ${50_000 + 137 * i:,}, bonus ${1_000 + 11 * i:,}, "
            f"SSN 123-45-{i:04d}, account 000{i:06d}. Review: meets expectations."
        )
        i += 1
    return "\n".join(lines)[:n_chars]


def peak_rss_mb(pid: int) -> float:
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("VmHWM not found")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-mb", type=float, default=350)
    parser.add_argument("--chars", type=int, default=10_000)
    args = parser.parse_args()

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "GROQ_API_KEY": os.environ.get("GROQ_API_KEY", "not-a-real-key"),
        "GROQ_BASE_URL": "http://127.0.0.1:9",  # nothing listens here
        "CHROMA_DIR": tempfile.mkdtemp(prefix="riskbot-mem-db-"),
        "ANONYMIZED_TELEMETRY": "False",
    }
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--port", str(port), "--workers", "1"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    http = requests.Session()
    http.trust_env = False  # don't route localhost through an HTTP proxy

    results = []
    try:
        for _ in range(120):
            try:
                if http.get(base + "/", timeout=1).ok:
                    break
            except requests.ConnectionError:
                time.sleep(0.5)
        else:
            raise RuntimeError("server did not start")
        results.append(("startup", peak_rss_mb(server.pid)))

        text = sample_text(args.chars)
        start = time.time()
        res = http.post(
            base + "/upload",
            files={"file": ("sample.txt", text.encode(), "text/plain")},
            timeout=600,
        )
        res.raise_for_status()
        body = res.json()
        results.append(
            (
                f"+ upload of {len(text):,} chars ({body['indexed_chunks']} chunks, "
                f"{time.time() - start:.1f}s incl. any model download)",
                peak_rss_mb(server.pid),
            )
        )

        res = http.post(
            base + "/ask", data={"doc_id": body["doc_id"], "question": "salary"}, timeout=120
        )
        # Groq is unreachable on purpose, so /ask returns 500 after retrieval + embedding ran.
        results.append((f"+ one /ask (HTTP {res.status_code})", peak_rss_mb(server.pid)))
    finally:
        server.terminate()
        server.wait(timeout=30)

    print(f"Peak RSS of the uvicorn process (VmHWM), limit {args.limit_mb:.0f} MB:")
    for label, mb in results:
        print(f"  {mb:7.1f} MB  {label}")

    peak = max(mb for _, mb in results)
    if peak > args.limit_mb:
        print(f"FAIL: peak {peak:.1f} MB exceeds {args.limit_mb:.0f} MB")
        return 1
    print(f"OK: peak {peak:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
