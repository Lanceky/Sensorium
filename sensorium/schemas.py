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


#: Keys that describe the schema to tooling rather than describing the data. Shown to a
#: model they are noise at best; ``$id`` actively invites it to echo the filename back.
_METADATA_KEYS = frozenset({"$schema", "$id", "title"})


def _resolve(node: Any, stack: tuple[str, ...]) -> Any:
    """Inline every ``$ref`` so the result is readable without the registry."""
    if isinstance(node, list):
        return [_resolve(item, stack) for item in node]
    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")
    if isinstance(ref, str):
        if ref in stack:
            # Nothing in schemas/ is recursive today; degrade rather than hang if that
            # changes, since a truncated contract still beats a stalled build.
            return {"description": f"recursive reference to {ref}"}
        target = _lookup(ref)
        merged = {k: v for k, v in node.items() if k != "$ref"}
        resolved = _resolve(target, stack + (ref,))
        if isinstance(resolved, dict):
            return {**resolved, **merged}
        return resolved

    return {k: _resolve(v, stack) for k, v in node.items() if k not in _METADATA_KEYS}


def _lookup(ref: str) -> Any:
    """Resolve ``file.json#/$defs/name`` or a bare ``#/$defs/name`` fragment."""
    filename, _, pointer = ref.partition("#")
    target: Any = get_schema(filename) if filename else None
    if target is None:
        raise SchemaError(f"cannot resolve local $ref {ref!r} without a base schema")
    for token in (p for p in pointer.split("/") if p):
        token = token.replace("~1", "/").replace("~0", "~")
        try:
            target = target[token]
        except (KeyError, TypeError) as exc:
            raise SchemaError(f"$ref {ref!r} does not resolve at {token!r}") from exc
    return target


@lru_cache(maxsize=None)
def contract_text(name: str) -> str:
    """``name`` as self-contained JSON, for embedding in a system prompt.

    Generated from the schema that validates the reply, rather than written by hand
    alongside it. That is the point: a hand-written contract is free to drift from its
    validator, and this project has now been bitten by that twice -- Node 2's prompt
    inviting a paraphrase the schema forbids, and every prompt omitting its output shape
    entirely, which made the first live call return prose.

    ``$ref``s are inlined so enum members are visible; a model shown
    ``{"$ref": "common.json#/$defs/modality"}`` has been told nothing at all.
    """
    return json.dumps(_resolve(get_schema(name), ()), indent=2, ensure_ascii=False)
