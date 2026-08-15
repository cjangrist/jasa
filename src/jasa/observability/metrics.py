"""Metrics facade: per-request and per-search metric emission.

Metrics emission never fails or slows a request; errors are swallowed as the
source does. The per-search metric captures mode (grounded/raw), total duration,
dispatch duration, providers succeeded/failed, cache-hit flag, and grounding
makespan + grounded count + timeout count. The edge-country dimension from the
source's Analytics Engine is dropped (no server-side analog outside a CDN).
"""

from __future__ import annotations

import logging

_LOGGER = logging.getLogger("jasa.metrics")


def emit_search_metric(**fields: object) -> None:
    """Emit a per-search metric. Swallows all errors (never fails a request)."""
    try:
        formatted = " ".join(f"{key}={value}" for key, value in fields.items())
        _LOGGER.debug("search_metric %s", formatted)
    except Exception:
        pass


def emit_search_cache_metric(**fields: object) -> None:
    """Emit a bounded search-cache event without affecting the request."""
    try:
        if not _LOGGER.isEnabledFor(logging.DEBUG):
            return
        formatted = " ".join(f"{key}={value}" for key, value in fields.items())
        _LOGGER.debug("search_cache_metric %s", formatted)
    except Exception:
        pass


def emit_request_metric(**fields: object) -> None:
    """Emit a per-request metric (route, transport, status, duration)."""
    try:
        formatted = " ".join(f"{key}={value}" for key, value in fields.items())
        _LOGGER.debug("request_metric %s", formatted)
    except Exception:
        pass
