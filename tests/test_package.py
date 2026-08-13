"""Public package surface and lazy imports."""

from __future__ import annotations

import subprocess
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


def test_every_tracked_directory_has_agent_guide() -> None:
    """Keep repository navigation available in every tracked directory."""
    repository_root = Path(__file__).resolve().parents[1]
    tracked_output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tracked_files = [
        repository_root / relative_path
        for relative_path in tracked_output.split("\0")
        if relative_path
    ]
    tracked_directories = {repository_root}
    for tracked_file in tracked_files:
        tracked_directories.update(
            parent
            for parent in tracked_file.parents
            if parent == repository_root or repository_root in parent.parents
        )
    missing_guides = sorted(
        str(directory.relative_to(repository_root) or ".")
        for directory in tracked_directories
        if not (directory / "AGENTS.md").is_file()
    )
    assert missing_guides == []
