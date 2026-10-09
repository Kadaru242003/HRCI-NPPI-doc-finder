"""Measure the peak memory of a real RiskBot server process (Linux only).

Starts `uvicorn api:app`, uploads a 10,000-character .txt file, asks one
question, and reads the server's peak resident set size (VmHWM in
/proc/<pid>/status) after each step. Exits non-zero if the peak exceeds
--limit-mb.

The Groq client is pointed at a tiny fake chat-completions server started by
this script, so no LLM request leaves the machine; it answers detection with
"[]" and /ask with a short text, so both requests take their normal success
path. The embedding model is real; on the first run it is downloaded to
~/.cache/chroma.

Usage (from the repository root):
    python scripts/measure_memory.py [--limit-mb 350] [--chars 10000]
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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


class FakeGroq(BaseHTTPRequestHandler):
    """Answers POST /openai/v1/chat/completions like Groq would, without an LLM."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        is_detection = "Text to label" in request["messages"][-1]["content"]
        body = json.dumps({
            "id": "fake", "object": "chat.completion", "created": 0, "model": request["model"],
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "[]" if is_detection else "A short answer."},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-mb", type=float, default=350)
    parser.add_argument("--chars", type=int, default=10_000)
    args = parser.parse_args()

    fake_groq = ThreadingHTTPServer(("127.0.0.1", 0), FakeGroq)
    threading.Thread(target=fake_groq.serve_forever, daemon=True).start()

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "GROQ_API_KEY": "not-a-real-key",
        "GROQ_BASE_URL": f"http://127.0.0.1:{fake_groq.server_address[1]}",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
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
        res.raise_for_status()
        results.append(("+ one /ask", peak_rss_mb(server.pid)))
    finally:
        server.terminate()
        server.wait(timeout=30)
        fake_groq.shutdown()

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
