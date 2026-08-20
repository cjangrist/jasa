"""Run one paid provider integration through an isolated Compose container.

The runner reads the selected credential from the repository-local ``.env``,
passes only that credential through an in-memory Compose env file, recreates
the service, verifies provider isolation, and makes one REST or MCP request.

A search adapter may also declare optional settings (a gateway base URL, a
model id). Those are configuration rather than credentials, so they are passed
through verbatim whenever ``.env`` defines them, including on
``--invalid-credential`` runs, and only the credential is ever substituted.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values
from fastmcp import Client
from fastmcp.exceptions import ToolError
from rich.console import Console
from rich.logging import RichHandler

from jasa.search.providers import (
    KNOWN_SEARCH_SETTING_ENVS,
    PROVIDER_CLASSES,
)
from omnifetch.fetch.providers.registry import import_all_providers

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_QUERY = "Model Context Protocol"
DEFAULT_URL = "https://en.wikipedia.org/wiki/Model_Context_Protocol"
INVALID_CREDENTIAL = "invalid-integration-test-credential"
INTEGRATION_CASES = frozenset(
    {
        ("fetch", "diffbot"),
        ("fetch", "github"),
        ("fetch", "jina"),
        ("fetch", "linkup"),
        ("fetch", "oxylabs"),
        ("fetch", "scrapegraphai"),
        ("fetch", "scrapeless"),
        ("fetch", "scrapingant"),
        ("fetch", "scrapingbee"),
        ("fetch", "sociavault"),
        ("fetch", "tavily"),
        ("fetch", "you"),
        ("search", "claude"),
        ("search", "exa"),
        ("search", "kagi"),
        ("search", "linkup"),
        ("search", "parallel"),
        ("search", "perplexity"),
        ("search", "serper"),
        ("search", "tavily"),
        ("search", "you"),
    }
)
NON_PROVIDER_SECRETS = (
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
    """Parse one explicit provider request and transport surface."""
    parser = argparse.ArgumentParser(
        description="Run one paid Jasa provider request through Compose."
    )
    parser.add_argument("--family", choices=("search", "fetch"), required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--surface", choices=("rest", "mcp"), required=True)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Reuse the current local image after it has already been built.",
    )
    parser.add_argument(
        "--invalid-credential",
        action="store_true",
        help="Use a fixed invalid value to verify provider error reporting.",
    )
    return parser.parse_args(argv)


def provider_secret_names(family: str, provider: str) -> tuple[str, ...]:
    """Return required credential names for one implemented test case."""
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


def provider_setting_names(family: str, provider: str) -> tuple[str, ...]:
    """Return the optional non-secret setting names for a search adapter."""
    if family != "search":
        return ()
    search_classes = {item.name: item for item in PROVIDER_CLASSES}
    selected = search_classes.get(provider)
    return () if selected is None else tuple(selected.setting_envs)


def all_secret_names() -> tuple[str, ...]:
    """Return every provider and non-provider secret and setting name."""
    fetch_names = {
        secret
        for provider in import_all_providers().values()
        for secret in provider.required_secrets
    }
    search_names = {item.secret_env for item in PROVIDER_CLASSES}
    return tuple(
        sorted(
            fetch_names
            | search_names
            | set(NON_PROVIDER_SECRETS)
            | set(KNOWN_SEARCH_SETTING_ENVS)
        )
    )


def expected_active_names(
    family: str,
    provider: str,
    secret_names: tuple[str, ...],
) -> list[str]:
    """Return providers activated by the selected credential names."""
    if family == "search":
        return [provider]
    selected_names = set(secret_names)
    return [
        name
        for name, candidate in import_all_providers().items()
        if set(candidate.required_secrets) <= selected_names
    ]


def selected_credentials(secret_names: tuple[str, ...]) -> dict[str, str]:
    """Read only the selected non-empty credentials from local ``.env``."""
    configured = dotenv_values(REPOSITORY_ROOT / ".env")
    missing = [name for name in secret_names if not configured.get(name)]
    if missing:
        raise ValueError(
            "selected provider credential is missing from .env: "
            + ", ".join(missing)
        )
    return {name: str(configured[name]) for name in secret_names}


def selected_settings(setting_names: tuple[str, ...]) -> dict[str, str]:
    """Read the adapter's optional settings from local ``.env`` when set."""
    configured = dotenv_values(REPOSITORY_ROOT / ".env")
    return {
        name: str(configured[name])
        for name in setting_names
        if configured.get(name)
    }


def compose_command(*, build: bool) -> list[str]:
    """Return the deterministic Compose recreation command."""
    command = [
        "docker",
        "compose",
        "--env-file",
        "/dev/null",
        "up",
        "-d",
        "--force-recreate",
        "--wait",
    ]
    if build:
        command.append("--build")
    return command


def encode_env_file(credentials: Mapping[str, str]) -> bytes:
    """Encode selected credentials for Compose's raw env-file parser."""
    invalid = [
        name
        for name, value in credentials.items()
        if "\n" in value or "\r" in value
    ]
    if invalid:
        raise ValueError(
            "provider credentials cannot contain newlines: "
            + ", ".join(invalid)
        )
    return "".join(
        f"{name}={value}\n" for name, value in credentials.items()
    ).encode()


def start_compose(credentials: Mapping[str, str], *, build: bool) -> None:
    """Recreate Compose with an in-memory selected-value env file."""
    if os.name != "posix":
        raise RuntimeError(
            "provider integration requires a POSIX host for secret passing"
        )
    read_descriptor, write_descriptor = os.pipe()
    try:
        os.write(write_descriptor, encode_env_file(credentials))
        os.close(write_descriptor)
        write_descriptor = -1
        environment = {
            name: value
            for name, value in os.environ.items()
            if name not in all_secret_names()
        }
        environment["JASA_ENV_FILE"] = f"/dev/fd/{read_descriptor}"
        LOGGER.info(
            "Recreating Compose with %d selected environment name(s).",
            len(credentials),
        )
        subprocess.run(
            compose_command(build=build),
            check=True,
            cwd=REPOSITORY_ROOT,
            env=environment,
            pass_fds=(read_descriptor,),
        )
    finally:
        os.close(read_descriptor)
        if write_descriptor >= 0:
            os.close(write_descriptor)


def endpoint(host: str, port: int, path: str) -> str:
    """Build one local HTTP endpoint."""
    return f"http://{host}:{port}{path}"


def health(host: str, port: int) -> dict[str, Any]:
    """Fetch and validate the non-billable health payload."""
    response = httpx.get(endpoint(host, port, "/health"), timeout=10)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("health endpoint returned a non-object payload")
    return payload


def verify_isolation(
    payload: dict[str, Any],
    family: str,
    expected_names: list[str],
) -> list[str]:
    """Validate the selected provider is isolated within its family."""
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
    if sorted(active) != sorted(expected_names):
        raise RuntimeError(
            f"expected active {family} providers {expected_names!r}; "
            f"observed {active!r}"
        )
    LOGGER.info(
        "Health verifies expected %s providers: %s.",
        family,
        ", ".join(expected_names),
    )
    return active


def error_detail(response: httpx.Response) -> str:
    """Return bounded error detail without assuming a JSON object body."""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict) and payload.get("error"):
        return str(payload["error"])
    text = response.text.strip()
    return text[:200] if text else "no error detail"


def run_rest(args: argparse.Namespace, skip_providers: list[str]) -> None:
    """Call exactly one paid REST route and validate its response."""
    path = "/search" if args.family == "search" else "/fetch"
    body: dict[str, object] = {"query": args.query, "count": 1}
    if args.family == "fetch":
        body = {"url": args.url}
        if skip_providers:
            body["skip_providers"] = skip_providers
    response = httpx.post(
        endpoint(args.host, args.port, path), json=body, timeout=90
    )
    if response.is_error:
        detail = error_detail(response)
        raise RuntimeError(f"REST {response.status_code} response: {detail}")
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
    args: argparse.Namespace,
    skip_providers: list[str],
) -> None:
    """Call exactly one paid MCP tool and validate provider attribution."""
    tool = "web_search" if args.family == "search" else "web_fetch"
    request: dict[str, object] = {"query": args.query}
    if args.family == "fetch":
        request = {"url": args.url}
        if skip_providers:
            request["skip_providers"] = skip_providers
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
    """Recreate an isolated container and run one selected paid request."""
    configure_logging()
    args = parse_args(argv)
    secret_names = provider_secret_names(args.family, args.provider)
    credentials = (
        dict.fromkeys(secret_names, INVALID_CREDENTIAL)
        if args.invalid_credential
        else selected_credentials(secret_names)
    )
    settings = selected_settings(
        provider_setting_names(args.family, args.provider)
    )
    LOGGER.info(
        "Selected %s provider %s via %s with %d setting name(s).",
        args.family,
        args.provider,
        args.surface,
        len(settings),
    )
    start_compose({**settings, **credentials}, build=not args.no_build)
    active_names = expected_active_names(
        args.family,
        args.provider,
        secret_names,
    )
    verify_isolation(health(args.host, args.port), args.family, active_names)
    skip_providers = [name for name in active_names if name != args.provider]
    if args.family == "search" and args.query == DEFAULT_QUERY:
        args.query = f"{DEFAULT_QUERY} integration-{time.time_ns()}"
    if args.surface == "rest":
        run_rest(args, skip_providers)
    else:
        asyncio.run(run_mcp(args, skip_providers))


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
