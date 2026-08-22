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
import re
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import SplitResult, urlsplit

from mcp.types import Icon

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ICON_MEDIA_TYPE = "image/png"
FAVICON_MEDIA_TYPE = "image/x-icon"
ICON_ROUTE = "/icon.png"
FAVICON_PNG_ROUTE = "/favicon.png"
FAVICON_ICO_ROUTE = "/favicon.ico"

ICON_SIZES = (48, 128, 256)
_DATA_URI_SIZE = min(ICON_SIZES)
_SERVED_SIZE = max(ICON_SIZES)
_ICON_SCHEME = "https"
_URL_TAIL_DELIMITERS = ("?", "#")
_ROOT_PATHS = frozenset({"", "/"})
_DNS_NAME = re.compile(r"^[A-Za-z0-9.\-]+$")
_MAX_HOST_CHARS = 253
_MAX_LABEL_CHARS = 63


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


def validated_public_url(public_url: str) -> str:
    """Return a trimmed origin an icon link can be appended to, or reject it.

    An empty value is valid and means "no public origin"; everything else must
    be something a client could actually fetch an icon from. The specification
    requires an icon source to use a safe scheme, so ``https`` is the only one
    accepted. A query or fragment is refused because the icon path is appended:
    ``https://host/?tenant=a`` would otherwise produce
    ``https://host/?tenant=a/icon.png``, which asks for the root document with
    a strange query rather than the icon. Userinfo is refused because this
    value is advertised to every client that connects.

    A malformed value fails startup rather than quietly reverting to the
    inline icon. A silent fallback here looks identical to success and would
    leave an operator who made a typo staring at the placeholder they were
    trying to replace.
    """
    trimmed = public_url.strip().rstrip("/")
    if not trimmed:
        return ""
    parsed = urlsplit(trimmed)
    if parsed.scheme != _ICON_SCHEME or not _has_reachable_host(parsed):
        raise ValueError(
            "JASA_PUBLIC_URL must be an absolute https:// URL naming a host "
            "with a usable port"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("JASA_PUBLIC_URL must not carry a credential")
    if any(delimiter in trimmed for delimiter in _URL_TAIL_DELIMITERS):
        raise ValueError(
            "JASA_PUBLIC_URL must have no query string or fragment"
        )
    if parsed.path not in _ROOT_PATHS:
        raise ValueError(
            "JASA_PUBLIC_URL must name an origin with no path; the icon "
            "routes are served at the root"
        )
    return trimmed


def _has_reachable_host(parsed: SplitResult) -> bool:
    """Report whether the authority names a legal host and a usable port.

    ``hostname`` strips the port without validating it, so a value like
    ``https://host:8O80`` yields a perfectly ordinary hostname while the port
    is nonsense. ``port`` is the property that notices, and it raises rather
    than returning, so it is read here rather than at request time. Port zero
    parses but is a bind-time wildcard nothing can connect to.

    ``hostname`` is equally happy to return a string containing a space, so the
    host is checked as well. It is returned with any IPv6 brackets already
    stripped, so an address literal is recognised as an address rather than
    matched against a name pattern it could never satisfy.
    """
    try:
        port = parsed.port
    except ValueError:
        return False
    if port == 0:
        return False
    host = parsed.hostname
    if not host:
        return False
    return _is_address_literal(host) or _is_resolvable_name(host)


def _is_address_literal(host: str) -> bool:
    """Report whether a bracket-free host is an IP address literal."""
    try:
        ip_address(host)
    except ValueError:
        return False
    return True


def _is_resolvable_name(host: str) -> bool:
    """Report whether a host is shaped like a name the DNS could answer.

    The character set alone accepts ``example..com`` and a label of any
    length, both of which resolve nowhere. Checking the label structure keeps
    the promise this validation exists to make: an operator's typo fails at
    startup rather than becoming an advertised URL no client can reach.
    """
    if len(host) > _MAX_HOST_CHARS or not _DNS_NAME.match(host):
        return False
    labels = host.rstrip(".").split(".")
    return all(0 < len(label) <= _MAX_LABEL_CHARS for label in labels)


def build_icons(public_url: str = "") -> list[Icon]:
    """Return the ``serverInfo.icons`` entries for this deployment.

    Without a public URL the server cannot name a location a client could
    fetch, so it inlines the smallest icon instead of advertising a link that
    would resolve to nothing. With one, every size is offered and the client
    picks what it needs.
    """
    base = validated_public_url(public_url)
    if not base:
        return [_sized_icon(icon_data_uri(), _DATA_URI_SIZE)]
    return [
        _sized_icon(f"{base}{ICON_ROUTE}", size)
        if size == _SERVED_SIZE
        else _sized_icon(f"{base}{ICON_ROUTE}?size={size}", size)
        for size in ICON_SIZES
    ]
