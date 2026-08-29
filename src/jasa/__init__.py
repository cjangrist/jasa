"""Jasa -- a FastMCP web-search server composing omnifetch in-process.

Public surface is import-light: ``build_server`` is exposed lazily via
``__getattr__`` so ``import jasa`` does not pull in FastMCP or read any secret.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

__version__ = "0.10.2"

__all__ = ["__version__", "build_server"]

if TYPE_CHECKING:
    from jasa.server import build_server


def __getattr__(name: str) -> Any:
    """Expose ``build_server`` lazily, without importing FastMCP at load."""
    if name == "build_server":
        from jasa.server import build_server

        return build_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
