"""
Structured logging configuration for Orchestral.

Uses structlog for JSON-formatted, context-rich logging.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from orchestral.core.config import get_settings


def setup_logging(
    level: str | None = None,
    json_format: bool | None = None,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON output format
    """
    settings = get_settings()
    level = level or settings.orchestrator.log_level
    json_format = json_format if json_format is not None else (
        settings.orchestrator.log_format == "json"
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper()),
    )

    # Shared processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        # JSON format for production
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console format for development
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Get a logger instance.

    Args:
        name: Optional logger name

    Returns:
        Bound structlog logger
    """
    logger = structlog.get_logger(name)
    return logger


class RequestLogger:
    """Context manager for request logging."""

    def __init__(
        self,
        logger: structlog.BoundLogger,
        operation: str,
        **context: Any,
    ):
        self.logger = logger.bind(operation=operation, **context)
        self.operation = operation

    def __enter__(self) -> "RequestLogger":
        self.logger.info(f"Starting {self.operation}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type:
            self.logger.error(
                f"Failed {self.operation}",
                error=str(exc_val),
                error_type=exc_type.__name__,
            )
        else:
            self.logger.info(f"Completed {self.operation}")

    def log(self, message: str, **kwargs: Any) -> None:
        """Log a message with context."""
        self.logger.info(message, **kwargs)
