"""Command-line entry point for the jasa MCP server.

Bootstrap order: dotenv -> config -> validate -> logging -> uvloop -> telemetry
-> serve. Run with ``python -m jasa`` or the installed ``jasa`` console script.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from dotenv import load_dotenv

from jasa.config import AppConfig, load_config, UvloopModeName
from jasa.grounding.waterfall import (
    grounding_credential_envs,
    load_grounding_waterfall,
    resolve_grounding_waterfall,
)
from jasa.logging import configure_logging, get_logger
from jasa.telemetry import configure_telemetry

_LOGGER = get_logger("main")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line overrides for server settings."""
    parser = argparse.ArgumentParser(
        prog="jasa", description="Run the jasa FastMCP search server."
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "http", "sse"),
        default=None,
        help="Transport to serve on (default: JASA_TRANSPORT or 'stdio').",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=(
            "Bind host for http/sse transports "
            "(default: JASA_HOST or 127.0.0.1)."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port for http/sse transports (default: JASA_PORT or 8000).",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        help=(
            "Logging level e.g. DEBUG/INFO/WARNING "
            "(default: JASA_LOG_LEVEL or INFO)."
        ),
    )
    return parser.parse_args(argv)


def collect_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Collect the CLI flags that were explicitly provided as overrides."""
    candidates = {
        "transport": args.transport,
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
    }
    return {
        key: value for key, value in candidates.items() if value is not None
    }


def validate_startup(config: AppConfig) -> None:
    """Fail startup for configurations that cannot serve correctly."""
    if config.cache.backend == "redis" and not config.cache.redis_url.strip():
        raise SystemExit(
            "JASA_REDIS_URL is required when JASA_CACHE_BACKEND=redis."
        )
    if config.grounding.mode != "on":
        return
    chain = load_grounding_waterfall(config.grounding)
    if not resolve_grounding_waterfall(chain, os.environ).chain:
        credentials = ", ".join(grounding_credential_envs(chain))
        raise SystemExit(
            "JASA_GROUNDING_MODE=on requires a grounding waterfall "
            f"credential ({credentials}) to be set."
        )


def install_uvloop(mode: UvloopModeName) -> bool:
    """Install uvloop's event-loop policy unless disabled or unavailable."""
    if mode == "off":
        _LOGGER.info("Using the default asyncio event loop.")
        return False

    try:
        import uvloop
    except ImportError:
        _LOGGER.warning(
            "uvloop not available; using the default asyncio event loop."
        )
        return False

    uvloop.install()
    _LOGGER.info("Installed uvloop event-loop policy.")
    return True


def run_server(config: AppConfig) -> None:
    """Build and run the FastMCP server for the given configuration."""
    from jasa.server import build_server

    server = build_server(config)
    transport = config.server.transport
    _LOGGER.info("Starting server on transport %r.", transport)
    if transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(
            transport=transport,
            host=config.server.host,
            port=config.server.port,
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Load ``.env`` via dotenv, configure the runtime, and start serving."""
    load_dotenv()
    args = parse_args(argv)
    config = load_config(**collect_overrides(args))
    validate_startup(config)
    configure_logging(config.server.log_level)
    _LOGGER.debug("Configuration loaded: %r.", config)
    install_uvloop(config.server.uvloop)
    configure_telemetry(config.telemetry)
    run_server(config)


if __name__ == "__main__":
    main()
