"""Container integration: build, run, and probe the aggregate health route.

Marked ``docker_integration``; runs only when ``JASA_RUN_DOCKER_TESTS=1`` (the
CI docker job). Never required for a normal PR. Asserts the container starts as
a non-root user and serves ``/health`` in the unavailable state with no secrets.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

import pytest

docker = pytest.importorskip("docker")


@pytest.mark.docker_integration
def test_container_health_unavailable() -> None:
    """Build the image, run it, and assert the aggregate health route."""
    if not os.environ.get("JASA_RUN_DOCKER_TESTS"):
        pytest.skip("set JASA_RUN_DOCKER_TESTS=1 to run container integration")

    client = docker.from_env()
    root = Path(__file__).resolve().parents[1]
    client.images.build(path=str(root), tag="jasa:test", rm=True)
    container = client.containers.run(
        "jasa:test", detach=True, ports={"8000/tcp": None}
    )
    try:
        container.reload()
        port = int(container.ports["8000/tcp"][0]["HostPort"])
        body = _wait_for_health(port)
        assert body["status"] == "unavailable"
        assert body["search"]["count"] == 0
        assert body["fetch"]["count"] == 0
        assert _container_user_is_nonroot(container)
    finally:
        container.remove(force=True)


def _wait_for_health(port: int, timeout: float = 30.0) -> dict[str, Any]:
    """Poll the health route until it answers or the timeout elapses."""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as response:
                return cast(
                    dict[str, Any], json.loads(response.read().decode())
                )
        except (urllib.error.URLError, OSError, ValueError) as error:
            last_error = error
            time.sleep(0.5)
    raise AssertionError(f"health route never became ready: {last_error}")


def _container_user_is_nonroot(container: object) -> bool:
    """Return True when the entrypoint runs as the non-root app user."""
    exit_code, output = container.exec_run("id -u")  # type: ignore[attr-defined]
    uid = output.decode().strip() if isinstance(output, bytes) else str(output)
    return exit_code == 0 and uid == "10001"
