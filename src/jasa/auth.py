"""REST bearer auth: constant-time comparison + key resolution.

Key resolution preserves the source precedence (JASA_API_KEY first, then the
legacy aliases OPENWEBUI_API_KEY and OMNISEARCH_API_KEY), resolved in one place.
The surface is open when no key is configured. ``/researcher`` additionally
accepts the key as a query parameter.
"""

from __future__ import annotations

import hmac
import os

from starlette.requests import Request

_KEY_ALIASES = ("JASA_API_KEY", "OPENWEBUI_API_KEY", "OMNISEARCH_API_KEY")


def resolve_api_key() -> str:
    """Return the first configured API key from the precedence chain."""
    for name in _KEY_ALIASES:
        key = os.getenv(name, "")
        if key.strip():
            return key.strip()
    return ""


def _extract_bearer_token(request: Request) -> str:
    """Extract the bearer token from the Authorization header."""
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return ""


def is_authorized(request: Request) -> bool:
    """Return True if the request is authorized (or auth is open)."""
    configured_key = resolve_api_key()
    if not configured_key:
        return True
    token = _extract_bearer_token(request)
    query_key = request.query_params.get("key", "")
    return hmac.compare_digest(configured_key, token) or hmac.compare_digest(
        configured_key, query_key
    )
