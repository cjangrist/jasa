"""Packaged brand assets, the icon declaration, and the icon HTTP routes."""

from __future__ import annotations

import base64
from collections.abc import Iterator

import pytest
from starlette.testclient import TestClient

from jasa.assets import (
    ASSETS_DIR,
    build_icons,
    FAVICON_ICO_ROUTE,
    FAVICON_PNG_ROUTE,
    icon_data_uri,
    ICON_MEDIA_TYPE,
    icon_path,
    ICON_ROUTE,
    ICON_SIZES,
    read_favicon,
    read_icon,
)
from jasa.config import load_config
from jasa.server import build_composition

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_ICO_MAGIC = b"\x00\x00\x01\x00"


@pytest.fixture
def composed_client() -> Iterator[TestClient]:
    """A composed server exercising its real lifespan, like the other suites."""
    composition = build_composition(load_config())
    with TestClient(composition.server.http_app()) as client:
        yield client


@pytest.mark.parametrize("size", ICON_SIZES)
def test_every_declared_size_ships_as_a_real_png(size: int) -> None:
    """A declared size a client asks for must exist in the package."""
    assert icon_path(size).is_file()
    assert read_icon(size).startswith(_PNG_MAGIC)


def test_favicon_ships_as_a_real_ico() -> None:
    assert read_favicon().startswith(_ICO_MAGIC)


def test_assets_live_inside_the_package() -> None:
    """They must sit under src/jasa so the wheel and image carry them."""
    assert ASSETS_DIR.name == "assets"
    assert ASSETS_DIR.parent.name == "jasa"


def test_icons_inline_when_no_public_url_is_configured() -> None:
    """A server that cannot name its own location still declares an icon."""
    icons = build_icons()

    assert len(icons) == 1
    assert icons[0].src.startswith(f"data:{ICON_MEDIA_TYPE};base64,")
    assert icons[0].mimeType == ICON_MEDIA_TYPE
    payload = icons[0].src.split(",", 1)[1]
    assert base64.b64decode(payload).startswith(_PNG_MAGIC)


def test_the_inlined_icon_is_the_smallest_one() -> None:
    """The bytes ride on every initialize, so the cheapest size is used."""
    assert icon_data_uri().endswith(icon_data_uri(48)[-32:])
    assert len(icon_data_uri(48)) < len(icon_data_uri(256))


def test_icons_become_links_when_a_public_url_is_configured() -> None:
    icons = build_icons("https://example.test/")

    assert [icon.sizes for icon in icons] == [
        [f"{size}x{size}"] for size in ICON_SIZES
    ]
    assert all(
        icon.src.startswith(f"https://example.test{ICON_ROUTE}")
        for icon in icons
    )
    assert not any(icon.src.startswith("data:") for icon in icons)


def test_a_trailing_slash_never_doubles_in_a_declared_source() -> None:
    with_slash = build_icons("https://example.test/")
    without = build_icons("https://example.test")

    assert [i.src for i in with_slash] == [i.src for i in without]
    assert "//icon.png" not in with_slash[-1].src


@pytest.mark.parametrize(
    "route", [ICON_ROUTE, FAVICON_PNG_ROUTE, FAVICON_ICO_ROUTE]
)
def test_icon_routes_serve_image_bytes(
    composed_client: TestClient, route: str
) -> None:
    response = composed_client.get(route)

    assert response.status_code == 200
    assert response.content.startswith((_PNG_MAGIC, _ICO_MAGIC))
    assert response.headers["content-type"].startswith("image/")


@pytest.mark.parametrize("size", ICON_SIZES)
def test_the_icon_route_serves_each_declared_size(
    composed_client: TestClient, size: int
) -> None:
    response = composed_client.get(ICON_ROUTE, params={"size": size})

    assert response.status_code == 200
    assert response.content == read_icon(size)


@pytest.mark.parametrize(
    "size",
    ["", "0", "999", "huge", "-1", "12.5", "\u00b2", "\u0669", "9" * 5000],
)
def test_an_unknown_size_falls_back_to_the_largest_icon(
    composed_client: TestClient, size: str
) -> None:
    """A stale or hand-written link resolves to an image, not an error.

    Includes the values that are ``str.isdigit()`` but not ``int()``-able
    (superscript two, an Arabic-Indic digit) and one past the integer
    conversion limit, each of which would otherwise raise on a public route.
    """
    response = composed_client.get(ICON_ROUTE, params={"size": size})

    assert response.status_code == 200
    assert response.content == read_icon(256)


async def test_the_icon_reaches_serverinfo_through_a_real_session() -> None:
    """The declaration must survive the initialize handshake, not just build.

    An icon that is correct in Python and absent from `serverInfo` is invisible
    to every client, so this asserts the wire result rather than the input.
    """
    from fastmcp import Client

    from jasa.server import build_composition_async

    composition = await build_composition_async(load_config())
    async with Client(composition.server) as client:
        info = client.initialize_result.serverInfo

    icons = getattr(info, "icons", None)
    assert icons, "serverInfo carried no icons"
    assert icons[0].src.startswith(f"data:{ICON_MEDIA_TYPE};base64,")
    assert icons[0].sizes == ["48x48"]


async def test_a_public_url_is_advertised_as_the_website(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator that names its origin gets links instead of inlined bytes."""
    from fastmcp import Client

    from jasa.server import build_composition_async

    monkeypatch.setenv("JASA_PUBLIC_URL", "https://example.test")
    composition = await build_composition_async(load_config())
    async with Client(composition.server) as client:
        info = client.initialize_result.serverInfo

    assert info.websiteUrl == "https://example.test"
    assert all(
        icon.src.startswith("https://example.test/icon.png")
        for icon in info.icons
    )


@pytest.mark.parametrize(
    "public_url",
    [
        "http://example.test",
        "ftp://example.test",
        "example.test",
        "https://",
        "https://example.test/?tenant=a",
        "https://example.test/#frag",
        "https://user:pass@example.test",
    ],
)
def test_a_public_url_that_cannot_serve_an_icon_is_rejected(
    public_url: str,
) -> None:
    """Failing loudly beats reverting to the inline icon in silence.

    A silent fallback is indistinguishable from success, and would leave an
    operator who mistyped the value staring at the placeholder they were
    trying to replace. A query string is the subtle one: appending the icon
    path to it yields `https://host/?tenant=a/icon.png`, which asks for the
    root document rather than the icon.
    """
    with pytest.raises(ValueError):
        build_icons(public_url)


@pytest.mark.parametrize(
    "public_url", ["", "   ", "https://example.test", "https://example.test/"]
)
def test_an_acceptable_public_url_builds_icons(public_url: str) -> None:
    assert build_icons(public_url)
