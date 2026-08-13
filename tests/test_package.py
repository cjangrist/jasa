"""Public package surface and lazy imports."""

from __future__ import annotations

import os
from pathlib import Path

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


def test_every_project_directory_has_agent_guide() -> None:
    """Keep navigation in source-controlled, non-generated directories."""
    repository_root = Path(__file__).resolve().parents[1]
    excluded_names = {
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "tmp",
        "trash",
    }
    project_directories = {repository_root}
    for current, child_names, _file_names in os.walk(repository_root):
        child_names[:] = [
            name for name in child_names if name not in excluded_names
        ]
        project_directories.add(Path(current))
        project_directories.update(Path(current) / name for name in child_names)
    missing_guides = sorted(
        str(directory.relative_to(repository_root) or ".")
        for directory in project_directories
        if not (directory / "AGENTS.md").is_file()
    )
    assert missing_guides == []
