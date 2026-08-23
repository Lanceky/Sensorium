"""Node 3 — the statistical trend engine, and the registry every later number cites.

Deterministic end to end. No model is consulted, which is why ``config.DETERMINISTIC_NODES``
names this node: the workflow diagram must not imply that arithmetic is reasoning.

The output is logged like any other node because Metric 1 scores downstream figures against
*this recorded payload*. Reconstructing the registry afterwards would let a drifting
implementation quietly re-justify whatever the report happened to say.
"""

from __future__ import annotations

from typing import Any

from sensorium import runlog, schemas
from stats import engine

#: Written into the run log where an LLM node records its model. Prefixed so that any
#: report generated from the log reads as a computation rather than a model choice.
ENGINE_ID = "deterministic:stats.engine"

INPUT_SCHEMA = "node_03.input.json"
OUTPUT_SCHEMA = "node_03.output.json"


def run(payload: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    """Validate, compute the number registry, validate again, and log the result."""
    schemas.validate(INPUT_SCHEMA, payload)

    with runlog.timed_call(run_id, "node_03", ENGINE_ID, 0.0, "n/a", payload) as slot:
        result = engine.compute(
            payload["signals"], payload["self_check"], payload["window_weeks"]
        )
        slot["raw_output"] = result
        schemas.validate(OUTPUT_SCHEMA, result)

    return result
