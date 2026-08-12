"""URL normalization for result deduplication.

Ported verbatim in behavior from ``normalize_url`` in omnisearch
``src/common/rrf_ranking.ts:14-26``: lowercase the host, drop the fragment,
resolve dot segments, strip the default port, Punycode-encode the host, and
remove a single trailing path slash (except at root). Invalid inputs (no scheme,
or no host) are returned unchanged, matching the ``new URL(...)`` throw path.
Validated against
``tests/fixtures/golden/url_normalization.json``.

WHATWG URL semantics are not those of ``urllib.parse``; the canonicalization is
replicated explicitly rather than via stdlib parsing.
"""

from __future__ import annotations

import encodings.idna
from urllib.parse import urlsplit

_DEFAULT_PORT_FOR_SCHEME = {
    "http": 80,
    "https": 443,
    "ftp": 21,
    "ws": 80,
    "wss": 443,
}


def _punycode_host(host: str) -> str:
    """Punycode-encode each label of a (already lowercased) host."""
    encoded_labels: list[str] = []
    for label in host.split("."):
        try:
            encoded_labels.append(encodings.idna.ToASCII(label).decode("ascii"))
        except (UnicodeError, UnicodeDecodeError):
            encoded_labels.append(label)
    return ".".join(encoded_labels)


def _resolve_dot_segments(path: str) -> str:
    """Resolve ``.`` and ``..`` path segments without collapsing ``//``."""
    segments = path.split("/")
    resolved: list[str] = []
    for segment in segments:
        if segment == ".":
            continue
        if segment == "..":
            if len(resolved) > 1:
                resolved.pop()
            continue
        resolved.append(segment)
    return "/".join(resolved)


def normalize_url(raw: str) -> str:
    """Return the WHATWG-canonical dedup key for ``raw``.

    Unparseable inputs are returned unchanged, matching the upstream
    ``catch { return raw }`` branch.
    """
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw
    scheme = parts.scheme
    try:
        hostname = parts.hostname
    except ValueError:
        return raw
    if not scheme or not hostname:
        return raw

    host = _punycode_host(hostname.lower())
    if ":" in host:
        host = f"[{host}]"

    try:
        port = parts.port
    except ValueError:
        return raw
    default_port = _DEFAULT_PORT_FOR_SCHEME.get(scheme)
    port_suffix = ""
    if port is not None and port != default_port:
        port_suffix = f":{port}"

    userinfo = ""
    if parts.username:
        userinfo = parts.username
        if parts.password is not None:
            userinfo = f"{parts.username}:{parts.password}"
        userinfo += "@"

    path = _resolve_dot_segments(parts.path)
    if path == "":
        path = "/"
    elif len(path) > 1 and path.endswith("/"):
        path = path[:-1]

    authority = f"{userinfo}{host}{port_suffix}"
    query = f"?{parts.query}" if parts.query else ""
    return f"{scheme}://{authority}{path}{query}"
