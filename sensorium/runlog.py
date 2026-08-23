"""Run logging.

Not debug output. This log is the primary evidence artifact for the submission:

* Node 4's independence (context.md section 4) is proved by inspecting the *actual*
  payloads sent to each agent, which only exist here.
* Metric 1 (numeric fidelity) checks emitted numbers against the Node 3 payload recorded
  here, not against a reconstruction.
* The iteration log (context.md section 10) is assembled from failing raw outputs kept
  here rather than rewritten from memory afterwards.

One append-only JSONL file per run, so a crashed run still leaves usable evidence.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from sensorium.config import RUNS_DIR

CALLS_FILENAME = "calls.jsonl"


class RunLogError(Exception):
    """Raised when a run or a expected call cannot be found."""


@dataclass(frozen=True)
class Call:
    """One node invocation, exactly as it happened."""

    run_id: str
    node: str
    model: str
    temperature: float
    prompt_version: str
    input_payload: Any
    raw_output: Any
    latency_ms: int
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: str | None = None

    def payload_text(self) -> str:
        """The input payload as a JSON string, for substring containment checks."""
        return json.dumps(self.input_payload, sort_keys=True, default=str)


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def log_call(
    run_id: str,
    node: str,
    model: str,
    temperature: float,
    prompt_version: str,
    input_payload: Any,
    raw_output: Any,
    latency_ms: int,
    error: str | None = None,
) -> Call:
    """Append one call to the run log and return it."""
    call = Call(
        run_id=run_id,
        node=node,
        model=model,
        temperature=temperature,
        prompt_version=prompt_version,
        input_payload=input_payload,
        raw_output=raw_output,
        latency_ms=latency_ms,
        error=error,
    )
    directory = run_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / CALLS_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(call), default=str) + "\n")
    return call


@contextmanager
def timed_call(
    run_id: str,
    node: str,
    model: str,
    temperature: float,
    prompt_version: str,
    input_payload: Any,
) -> Iterator[dict[str, Any]]:
    """Time a node invocation and log it, including when it raises.

    Failed calls are logged rather than swallowed: a failure is an iteration-log entry.

        with timed_call(rid, "node_05", model, 0.0, "v1", payload) as slot:
            slot["raw_output"] = call_model(...)
    """
    slot: dict[str, Any] = {"raw_output": None}
    started = time.perf_counter()
    try:
        yield slot
    except Exception as exc:
        log_call(
            run_id, node, model, temperature, prompt_version, input_payload,
            slot.get("raw_output"), int((time.perf_counter() - started) * 1000), repr(exc),
        )
        raise
    log_call(
        run_id, node, model, temperature, prompt_version, input_payload,
        slot.get("raw_output"), int((time.perf_counter() - started) * 1000),
    )


def load_calls(run_id: str, node: str | None = None) -> list[Call]:
    """All calls for ``run_id``, optionally filtered to one node, in write order."""
    path = run_dir(run_id) / CALLS_FILENAME
    if not path.exists():
        raise RunLogError(f"no run log at {path}")
    calls = [
        Call(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [c for c in calls if node is None or c.node == node]


def load_call(run_id: str, node: str) -> Call:
    """The single call for ``node``; raises if absent or ambiguous."""
    calls = load_calls(run_id, node)
    if not calls:
        raise RunLogError(f"run {run_id} has no call for node {node!r}")
    if len(calls) > 1:
        raise RunLogError(
            f"run {run_id} has {len(calls)} calls for node {node!r}; use load_calls()"
        )
    return calls[0]


def list_runs() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted(p.name for p in RUNS_DIR.iterdir() if (p / CALLS_FILENAME).exists())
