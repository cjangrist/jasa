"""Colorized logging to stderr via Rich.

stdout is reserved for the MCP JSON-RPC transport, so all logs go to stderr.
Call ``configure_logging`` once at startup; use ``get_logger`` for children.

The mounted omnifetch child logs under its own namespace and only configures
it when run on its own, so composed it would have no handler: its INFO lines
(tool calls, cancelled and abandoned fetches) would vanish and its warnings
would reach only Python's unformatted last-resort handler. Both namespaces
therefore share one handler and level, and neither propagates to the root
logger, so nothing double-emits.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

LOGGER_NAMESPACE = "jasa"
COMPOSED_LOGGER_NAMESPACES = (LOGGER_NAMESPACE, "omnifetch")
_LOG_FORMAT = "%(name)s | %(message)s"
_DATE_FORMAT = "[%X]"


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure both composed namespaces with one Rich handler on stderr."""
    resolved_level = logging.getLevelNamesMapping().get(
        level.upper(), logging.INFO
    )
    handler = RichHandler(
        console=Console(stderr=True),
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=False,
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    for namespace in COMPOSED_LOGGER_NAMESPACES:
        namespace_logger = logging.getLogger(namespace)
        namespace_logger.setLevel(resolved_level)
        namespace_logger.handlers.clear()
        namespace_logger.addHandler(handler)
        namespace_logger.propagate = False
    logger = logging.getLogger(LOGGER_NAMESPACE)
    for noisy_logger in ("httpx", "httpcore"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logger.debug(
        "Logging configured at level %s.", logging.getLevelName(resolved_level)
    )
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger, or a namespaced child by ``name``."""
    if name is None:
        return logging.getLogger(LOGGER_NAMESPACE)
    return logging.getLogger(f"{LOGGER_NAMESPACE}.{name}")
