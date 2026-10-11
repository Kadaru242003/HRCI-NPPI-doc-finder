"""Upload size and content checks, and a simple per-IP rate limit."""
import io
import os
import time
import zipfile
from collections import deque

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "2"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
# Requests per minute per client IP; 0 disables the limit for that endpoint.
UPLOADS_PER_MINUTE = int(os.getenv("RATE_LIMIT_UPLOADS_PER_MINUTE", "10"))
ASKS_PER_MINUTE = int(os.getenv("RATE_LIMIT_ASKS_PER_MINUTE", "30"))

_OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy .xls (and other Office 97-2003 files)
_BINARY_SIGNATURES = {b"PK\x03\x04": "a ZIP/Office file", b"%PDF": "a PDF", b"MZ": "a Windows executable",
                      b"\x7fELF": "an executable", _OLE2: "an Office 97-2003 file"}


class ContentError(ValueError):
    """The file's content does not match its extension."""


def check_content(ext: str, data: bytes) -> None:
    """Raise ContentError unless the bytes are really a file of type `ext`."""
    if ext == ".xlsx":
        if not data.startswith(b"PK\x03\x04"):
            raise ContentError("The file is not a valid .xlsx workbook (it is not a ZIP-based Office file).")
        try:
            names = zipfile.ZipFile(io.BytesIO(data)).namelist()  # reads the index only, nothing is extracted
        except zipfile.BadZipFile:
            raise ContentError("The file is not a valid .xlsx workbook (damaged ZIP).")
        if "xl/workbook.xml" not in names:
            raise ContentError("The file is a ZIP archive but not an Excel workbook.")
    elif ext == ".xls":
        if not data.startswith(_OLE2):
            raise ContentError("The file is not a valid .xls workbook (missing the Excel 97-2003 signature).")
    elif ext == ".txt":
        for magic, what in _BINARY_SIGNATURES.items():
            if data.startswith(magic):
                raise ContentError(f"The file has a .txt name but its content is {what}.")
        if b"\x00" in data:
            raise ContentError("The file has a .txt name but contains binary data.")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            raise ContentError("Text files must be UTF-8 encoded.")


def client_key(headers, client_host: str) -> str:
    """Client IP for rate limiting.

    Behind a proxy that appends the connecting address to X-Forwarded-For (as
    Render does), the rightmost entry is the address the proxy saw and cannot
    be forged by the client. Without the header, the socket address is used.
    """
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        last = forwarded.split(",")[-1].strip()
        if last:
            return last
    return client_host or "unknown"


class RateLimiter:
    """Sliding one-minute window per (endpoint, client). In memory, per process."""

    def __init__(self):
        self._hits = {}

    def reset(self):
        self._hits.clear()

    def check(self, endpoint: str, key: str, per_minute: int, now: float = None):
        """Record a request; return 0 if allowed, else the seconds until the next slot."""
        if per_minute <= 0:
            return 0
        now = time.monotonic() if now is None else now
        window = self._hits.setdefault((endpoint, key), deque())
        while window and window[0] <= now - 60:
            window.popleft()
        if len(window) >= per_minute:
            return max(1, int(window[0] + 60 - now) + 1)
        window.append(now)
        if len(self._hits) > 10000:  # bound memory: drop idle clients
            for k in [k for k, w in self._hits.items() if not w or w[-1] <= now - 60]:
                del self._hits[k]
        return 0


rate_limiter = RateLimiter()
