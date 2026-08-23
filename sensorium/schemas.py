"""Schema registry and validation.

Every node input and output is validated against a JSON Schema in ``schemas/``. Schemas
cross-reference each other by ``$id`` (e.g. Node 5's input ``$ref``s Node 3's output), so
they are loaded into a single ``referencing.Registry`` rather than validated in isolation.

``format`` is enforced, not decorative. jsonschema ignores ``format`` unless a checker is
supplied, which would let ``"start": "NOT-A-DATE"`` pass silently and put a malformed
timestamp into the Node 3 time series. Requires the ``format-nongpl`` extra.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


class SchemaError(Exception):
    """Raised when an instance does not satisfy its node's contract."""


@lru_cache(maxsize=1)
def _load_all() -> tuple[dict[str, dict[str, Any]], Registry]:
    schemas: dict[str, dict[str, Any]] = {}
    for path in sorted(SCHEMA_DIR.glob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        declared = schema.get("$id")
        if declared != path.name:
            raise SchemaError(
                f"{path.name}: $id must equal the filename so $refs resolve; got {declared!r}"
            )
        schemas[path.name] = schema

    registry = Registry().with_resources(
        (name, Resource.from_contents(schema)) for name, schema in schemas.items()
    )
    return schemas, registry


def schema_names() -> list[str]:
    """Every registered schema filename, e.g. ``node_05.output.json``."""
    return sorted(_load_all()[0])


def node_schema_names() -> list[str]:
    """Schema filenames excluding shared definition files."""
    return [n for n in schema_names() if n.startswith("node_")]


def get_schema(name: str) -> dict[str, Any]:
    schemas, _ = _load_all()
    if name not in schemas:
        raise SchemaError(f"unknown schema {name!r}; known: {', '.join(sorted(schemas))}")
    return schemas[name]


@lru_cache(maxsize=None)
def _validator(name: str) -> Draft202012Validator:
    _, registry = _load_all()
    return Draft202012Validator(
        get_schema(name),
        registry=registry,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )


def validate(name: str, instance: Any) -> Any:
    """Validate ``instance`` against schema ``name``; return it unchanged on success.

    Raises ``SchemaError`` with every failure listed, not just the first, so a repair
    retry (Step 3) can hand the model a complete description of what was wrong.
    """
    errors = sorted(_validator(name).iter_errors(instance), key=lambda e: list(e.path))
    if errors:
        raise SchemaError(f"{name}: " + "; ".join(_describe(e) for e in errors))
    return instance


def is_valid(name: str, instance: Any) -> bool:
    try:
        validate(name, instance)
    except SchemaError:
        return False
    return True


def _describe(error: ValidationError) -> str:
    location = "/".join(str(p) for p in error.path) or "<root>"
    return f"at {location}: {error.message}"
