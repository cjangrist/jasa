"""Public package surface and lazy imports."""

from __future__ import annotations

import pytest

import jasa


def test_version_is_string() -> None:
    assert isinstance(jasa.__version__, str)
    assert jasa.__version__


def test_build_server_exported_lazily() -> None:
    assert callable(jasa.build_server)


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        _ = jasa.does_not_exist
