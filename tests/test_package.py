"""Public package surface and lazy imports."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import jasa


def _tracked_project_directories(repository_root: Path) -> set[Path]:
    tracked_output = subprocess.run(
        ["git", "-C", str(repository_root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    tracked_paths = (
        Path(os.fsdecode(raw_path))
        for raw_path in tracked_output.split(b"\0")
        if raw_path
    )
    return {
        repository_root / parent
        for tracked_path in tracked_paths
        for parent in tracked_path.parents
    }


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
    project_directories = _tracked_project_directories(repository_root)
    missing_guides = sorted(
        directory.relative_to(repository_root).as_posix() or "."
        for directory in project_directories
        if not (directory / "AGENTS.md").is_file()
    )
    assert missing_guides == []


def test_agent_guide_scope_ignores_untracked_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep local tool output outside the source-controlled guide contract."""
    (tmp_path / "untracked-tool-output").mkdir()
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=b"tracked/file.py\0"
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

    assert _tracked_project_directories(tmp_path) == {
        tmp_path,
        tmp_path / "tracked",
    }
