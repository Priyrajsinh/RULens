"""Structured JSON logging.

Each line is one JSON object, so logs are easy to parse from MLflow notes or CI.
Handlers are attached lazily by get_logger (not at import), and it's idempotent —
repeat calls never duplicate output.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

# LogRecord attributes that are part of the logging machinery, not user data.
# Anything else on the record is treated as a structured "extra" field.
_RESERVED: frozenset[str] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JSONFormatter(logging.Formatter):
    """Render a :class:`logging.LogRecord` as a one-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that emits structured JSON to stderr.

    Idempotent: a logger is configured with exactly one JSON stream handler the
    first time it is requested; later calls reuse it and only update the level.
    Propagation is disabled so records are not also emitted by the root logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not any(getattr(h, "_rulens_json", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        handler._rulens_json = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False
    return logger
