"""Shared fixtures: the hand-written valid/invalid instances for every node contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _load(name: str) -> dict[str, Any]:
    data = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _strip_notes(instance: Any) -> Any:
    """Drop the ``_violates`` annotation so it does not itself trip additionalProperties."""
    if isinstance(instance, dict):
        return {k: v for k, v in instance.items() if k != "_violates"}
    return instance


@pytest.fixture(scope="session")
def valid_fixtures() -> dict[str, Any]:
    return _load("valid.json")


@pytest.fixture(scope="session")
def invalid_fixtures() -> dict[str, Any]:
    return {k: _strip_notes(v) for k, v in _load("invalid.json").items()}
