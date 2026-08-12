"""Colorized logging to stderr via Rich.

stdout is reserved for the MCP JSON-RPC transport, so all logs go to stderr.
Call ``configure_logging`` once at startup; use ``get_logger`` for children.
The package logger never propagates to the root logger, so the mounted
omnifetch child (which logs under its own namespace) never double-emits.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

LOGGER_NAMESPACE = "jasa"
_LOG_FORMAT = "%(name)s | %(message)s"
_DATE_FORMAT = "[%X]"


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure the package logger with a colorized Rich handler on stderr."""
    logger = logging.getLogger(LOGGER_NAMESPACE)
    resolved_level = logging.getLevelNamesMapping().get(
        level.upper(), logging.INFO
    )
    logger.setLevel(resolved_level)
    logger.handlers.clear()
    handler = RichHandler(
        console=Console(stderr=True),
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=False,
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
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
