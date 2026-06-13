"""FastAPI inference service (app-factory pattern).

Importing this module has no side effects: FastAPI is imported lazily inside
create_app, and nothing binds a port or loads a model at import. The
request/response contract lives in src.api.schemas.
"""

from __future__ import annotations

from typing import Any

from .schemas import ForecastRequest, ForecastResponse


def predict(request: ForecastRequest) -> ForecastResponse:
    """Run forecast + conformal interval + active-feature extraction."""
    raise NotImplementedError


def create_app() -> Any:
    """Build and return the FastAPI application with routes registered."""
    raise NotImplementedError
