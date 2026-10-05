"""Colorized stderr logging configuration."""

from __future__ import annotations

import logging

from jasa.logging import (
    COMPOSED_LOGGER_NAMESPACES,
    configure_logging,
    get_logger,
    LOGGER_NAMESPACE,
)


def test_configure_logging_sets_level_and_handler() -> None:
    logger = configure_logging("DEBUG")
    assert logger.name == LOGGER_NAMESPACE
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1
    assert logger.propagate is False


def test_configure_logging_routes_the_composed_omnifetch_namespace() -> None:
    jasa_logger = configure_logging("WARNING")
    omnifetch_logger = logging.getLogger("omnifetch")

    assert "omnifetch" in COMPOSED_LOGGER_NAMESPACES
    assert omnifetch_logger.level == logging.WARNING
    assert omnifetch_logger.handlers == jasa_logger.handlers
    assert omnifetch_logger.propagate is False


def test_configure_logging_falls_back_for_bad_level() -> None:
    logger = configure_logging("NOPE")
    assert logger.level == logging.INFO


def test_get_logger_namespacing() -> None:
    assert get_logger().name == LOGGER_NAMESPACE
    assert get_logger("server").name == f"{LOGGER_NAMESPACE}.server"
