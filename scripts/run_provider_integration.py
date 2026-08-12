"""Run one paid Jasa provider integration through Docker Compose.

The command resolves the live provider registry, injects only the selected
provider's credentials from the local Infisical export, waits for Compose
health, and makes one REST or MCP request. It is deliberately manual-only and
never writes secrets.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.exceptions import ToolError
from rich.console import Console
from rich.logging import RichHandler

from jasa.search.providers import PROVIDER_CLASSES
from omnifetch.fetch.providers.registry import import_all_providers

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_QUERY = "Model Context Protocol"
DEFAULT_URL = "https://en.wikipedia.org/wiki/Model_Context_Protocol"
INTEGRATION_CASES = frozenset({("search", "tavily")})
AUTH_ENVIRONMENTS = (
    "JASA_API_KEY",
    "OPENWEBUI_API_KEY",
    "OMNISEARCH_API_KEY",
    "CEREBRAS_API_KEY",
)
LOGGER = logging.getLogger("jasa.integration")


def configure_logging() -> None:
    """Configure colorized stderr logging for the manual integration run."""
    handler = RichHandler(console=Console(stderr=True), show_path=False)
    handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.propagate = False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one explicit provider request and its transport surface."""
    parser = argparse.ArgumentParser(
        description="Run one paid Jasa provider request through Docker Compose."
    )
    parser.add_argument("--family", choices=("search", "fetch"), required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--surface", choices=("rest", "mcp"), required=True)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--url", default=None)
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Reuse the current local image after it has already been built.",
    )
    return parser.parse_args(argv)


def provider_secrets(family: str, provider: str) -> tuple[str, ...]:
    """Return the selected provider's declared required credentials."""
    if (family, provider) not in INTEGRATION_CASES:
        raise ValueError(
            f"integration case is not implemented: {family}/{provider}"
        )
    if family == "search":
        search_classes = {item.name: item for item in PROVIDER_CLASSES}
        selected = search_classes.get(provider)
        if selected is None:
            raise ValueError(f"unknown search provider: {provider}")
        return (selected.secret_env,)
    selected = import_all_providers().get(provider)
    if selected is None:
        raise ValueError(f"unknown fetch provider: {provider}")
    return tuple(selected.required_secrets)


def all_provider_secrets() -> tuple[str, ...]:
    """Return every active registry credential name in deterministic order."""
    fetch_names = {
        secret
        for provider in import_all_providers().values()
        for secret in provider.required_secrets
    }
    search_names = {item.secret_env for item in PROVIDER_CLASSES}
    return tuple(sorted(fetch_names | search_names))


def compose_command(
    selected_secrets: tuple[str, ...], *, build: bool
) -> list[str]:
    """Build the local secret-isolated Docker Compose invocation."""
    clear_names = AUTH_ENVIRONMENTS + tuple(
        name for name in all_provider_secrets() if name not in selected_secrets
    )
    compose_arguments = "up -d --force-recreate --wait"
    if build:
        compose_arguments += " --build"
    isolate = (
        'for name in "$@"; do export "$name"=""; done; '
        f"exec docker compose {compose_arguments}"
    )
    return [
        "sh",
        "-eu",
        "-c",
        isolate,
        "run-provider-integration",
        *clear_names,
    ]


def start_compose(selected_secrets: tuple[str, ...], *, build: bool) -> None:
    """Recreate Compose with only selected local credentials injected."""
    LOGGER.info(
        "Starting Compose with %d selected credential name(s), build=%s.",
        len(selected_secrets),
        build,
    )
    subprocess.run(
        compose_command(selected_secrets, build=build),
        check=True,
        cwd=REPOSITORY_ROOT,
        env=os.environ.copy(),
    )


def endpoint(host: str, port: int, path: str) -> str:
    """Build one local HTTP endpoint from an explicit host, port, and path."""
    return f"http://{host}:{port}{path}"


def health(host: str, port: int) -> dict[str, Any]:
    """Fetch and validate the non-billable local health payload."""
    response = httpx.get(endpoint(host, port, "/health"), timeout=10)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("health endpoint returned a non-object payload")
    return payload


def verify_isolation(
    payload: dict[str, Any], family: str, provider: str
) -> list[str]:
    """Validate the target provider and return active family members."""
    family_payload = payload.get(family)
    active = (
        family_payload.get("providers")
        if isinstance(family_payload, dict)
        else None
    )
    if not isinstance(active, list) or not all(
        isinstance(name, str) for name in active
    ):
        raise RuntimeError(f"health returned invalid {family} provider data")
    if family == "search" and active != [provider]:
        raise RuntimeError(
            f"expected only {provider!r} in active {family} providers"
        )
    if family == "fetch" and provider not in active:
        raise RuntimeError(f"expected {provider!r} in active fetch providers")
    LOGGER.info("Health verifies %s is active for %s.", provider, family)
    return active


def fetch_skip_providers(
    active_providers: list[str], provider: str
) -> list[str]:
    """Return co-active fetch providers to skip for deterministic dispatch."""
    return [name for name in active_providers if name != provider]


def target_url(provider: str, configured: str | None) -> str:
    """Choose the explicitly configured URL or the generic public target."""
    return configured or DEFAULT_URL


def run_rest(
    args: argparse.Namespace, url: str, skip_providers: list[str]
) -> None:
    """Call exactly one paid REST route and validate provider attribution."""
    path = "/search" if args.family == "search" else "/fetch"
    body: dict[str, object] = {"query": args.query, "count": 1}
    if args.family == "fetch":
        body = {"url": url, "skip_providers": skip_providers}
    response = httpx.post(
        endpoint(args.host, args.port, path), json=body, timeout=90
    )
    if response.is_error:
        detail = response.json().get("error", "no error detail")
        raise RuntimeError(f"REST {response.status_code} response: {detail}")
    response.raise_for_status()
    payload = response.json()
    if args.family == "search" and not isinstance(payload, list):
        raise RuntimeError("search REST response must be a list")
    if (
        args.family == "fetch"
        and payload.get("source_provider") != args.provider
    ):
        raise RuntimeError("fetch REST response used an unexpected provider")
    LOGGER.info("REST request completed through %s.", args.provider)


async def run_mcp(
    args: argparse.Namespace, url: str, skip_providers: list[str]
) -> None:
    """Call exactly one paid MCP tool request and validate attribution."""
    tool = "web_search" if args.family == "search" else "web_fetch"
    request: dict[str, object] = {"query": args.query}
    if args.family == "fetch":
        request = {"url": url, "skip_providers": skip_providers}
    async with Client(endpoint(args.host, args.port, "/mcp/")) as client:
        result = await client.call_tool(tool, request)
    payload = (
        result.data
        if isinstance(result.data, dict)
        else result.structured_content
    )
    if not isinstance(payload, dict):
        raise RuntimeError("MCP tool returned no structured response")
    if args.family == "search":
        successes = payload.get("providers_succeeded")
        names = (
            [
                item.get("provider")
                for item in successes
                if isinstance(item, dict)
            ]
            if isinstance(successes, list)
            else []
        )
        if names != [args.provider]:
            raise RuntimeError(
                "search MCP response used an unexpected provider"
            )
    elif payload.get("source_provider") != args.provider:
        raise RuntimeError("fetch MCP response used an unexpected provider")
    LOGGER.info("MCP request completed through %s.", args.provider)


def main(argv: Sequence[str] | None = None) -> None:
    """Start an isolated container and run one selected paid request."""
    configure_logging()
    load_dotenv(REPOSITORY_ROOT / ".env")
    args = parse_args(argv)
    selected_secrets = provider_secrets(args.family, args.provider)
    LOGGER.info(
        "Selected %s provider %s via %s.",
        args.family,
        args.provider,
        args.surface,
    )
    start_compose(selected_secrets, build=not args.no_build)
    active_providers = verify_isolation(
        health(args.host, args.port), args.family, args.provider
    )
    url = target_url(args.provider, args.url)
    if args.family == "search" and args.query == DEFAULT_QUERY:
        args.query = f"{DEFAULT_QUERY} integration-{time.time_ns()}"
    skip_providers = (
        fetch_skip_providers(active_providers, args.provider)
        if args.family == "fetch"
        else []
    )
    if args.surface == "rest":
        run_rest(args, url, skip_providers)
    else:
        asyncio.run(run_mcp(args, url, skip_providers))


if __name__ == "__main__":
    try:
        main()
    except (
        RuntimeError,
        subprocess.CalledProcessError,
        httpx.HTTPError,
        ToolError,
        ValueError,
    ) as error:
        LOGGER.error("Integration run failed: %s", error)
        sys.exit(1)
