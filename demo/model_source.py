"""Resolve an explicitly supplied local model without network fallbacks."""

from __future__ import annotations

import os
from pathlib import Path


def require_model_path(
    raw: str | None = None,
    *,
    allowed_suffixes: tuple[str, ...] = (".pt", ".onnx"),
) -> Path:
    """Return a validated local model path or raise an actionable error."""
    configured = raw if raw is not None else os.environ.get("MODEL_PATH", "")
    if not configured or not configured.strip():
        raise RuntimeError("MODEL_PATH is required and must point to a local model file")

    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"MODEL_PATH does not exist or is not a file: {path}")

    normalized = tuple(
        suffix.casefold() if suffix.startswith(".") else f".{suffix.casefold()}"
        for suffix in allowed_suffixes
    )
    if path.suffix.casefold() not in normalized:
        choices = " or ".join(normalized)
        raise RuntimeError(f"MODEL_PATH must end with {choices}: {path.name}")
    return path
