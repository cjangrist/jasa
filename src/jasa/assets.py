"""Packaged brand assets and the MCP icon declaration built from them.

An MCP client shows an icon beside the server it has connected to. The
specification carries it in ``serverInfo.icons`` (SEP-973, revision 2025-11-25),
where each entry names a source that is either an ``https:`` URL or a ``data:``
URI. Both are declared here from the same packaged PNGs that the HTTP routes
serve, so the two can never disagree about what the server looks like.

The default is a data URI, because it needs no configuration and no reachable
hostname: a stdio server, a private deployment, and a laptop all advertise an
icon without being told where they live. It is built from the smallest size, as
the bytes ride along on every ``initialize``. Setting ``JASA_PUBLIC_URL``
switches to served URLs and advertises every size instead, which is both richer
for the client and cheaper on the wire.

Neither of the two clients this most matters for reads the field yet. ChatGPT
takes an uploaded image in its connector dialog, and Claude resolves a favicon
for the registrable root domain of the connector URL rather than asking the
server at all. The declaration is still made, because it is the specified
mechanism and costs nothing to be right about early.
"""

from __future__ import annotations

import base64
from pathlib import Path

from mcp.types import Icon

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ICON_MEDIA_TYPE = "image/png"
FAVICON_MEDIA_TYPE = "image/x-icon"
ICON_ROUTE = "/icon.png"
FAVICON_PNG_ROUTE = "/favicon.png"
FAVICON_ICO_ROUTE = "/favicon.ico"

_DECLARED_SIZES = (48, 128, 256)
_DATA_URI_SIZE = 48
_SERVED_SIZE = 256


def icon_path(size: int) -> Path:
    """Return the packaged square PNG for one declared size."""
    return ASSETS_DIR / f"icon-{size}.png"


def read_icon(size: int) -> bytes:
    """Read one packaged square PNG."""
    return icon_path(size).read_bytes()


def read_favicon() -> bytes:
    """Read the packaged multi-resolution favicon."""
    return (ASSETS_DIR / "favicon.ico").read_bytes()


def icon_data_uri(size: int = _DATA_URI_SIZE) -> str:
    """Return one packaged PNG as an inline ``data:`` URI."""
    encoded = base64.b64encode(read_icon(size)).decode("ascii")
    return f"data:{ICON_MEDIA_TYPE};base64,{encoded}"


def _sized_icon(src: str, size: int) -> Icon:
    """Build one spec icon entry for a square source of a known size."""
    return Icon(src=src, mimeType=ICON_MEDIA_TYPE, sizes=[f"{size}x{size}"])


def build_icons(public_url: str = "") -> list[Icon]:
    """Return the ``serverInfo.icons`` entries for this deployment.

    Without a public URL the server cannot name a location a client could
    fetch, so it inlines the smallest icon instead of advertising a link that
    would resolve to nothing. With one, every size is offered and the client
    picks what it needs.
    """
    base = public_url.rstrip("/")
    if not base:
        return [_sized_icon(icon_data_uri(), _DATA_URI_SIZE)]
    return [
        _sized_icon(f"{base}{ICON_ROUTE}?size={size}", size)
        if size != _SERVED_SIZE
        else _sized_icon(f"{base}{ICON_ROUTE}", size)
        for size in _DECLARED_SIZES
    ]
