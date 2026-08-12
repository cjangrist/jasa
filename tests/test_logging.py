"""Colorized stderr logging configuration."""

from __future__ import annotations

import logging

from jasa.logging import configure_logging, get_logger, LOGGER_NAMESPACE


def test_configure_logging_sets_level_and_handler() -> None:
    logger = configure_logging("DEBUG")
    assert logger.name == LOGGER_NAMESPACE
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1
    assert logger.propagate is False


def test_configure_logging_falls_back_for_bad_level() -> None:
    logger = configure_logging("NOPE")
    assert logger.level == logging.INFO


def test_get_logger_namespacing() -> None:
    assert get_logger().name == LOGGER_NAMESPACE
    assert get_logger("server").name == f"{LOGGER_NAMESPACE}.server"
