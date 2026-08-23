"""Node 5 — synthesis, and the first node permitted to see both slices.

Everything upstream is blind by construction; this is where the two views meet. That makes
it the highest-stakes node in the pipeline, because every downstream artifact — the
recommendation at Node 6, the doctor-shareable report at Node 10 — inherits whatever this
node asserts.

So the two guarantees are enforced here rather than checked afterwards. ``assert_grounded``
runs inside the repair loop, which means a claim citing a field that does not exist, or a
number the engine never computed, is handed back to the model as a correction instead of
being counted as a failure at the end. The same functions score the run in the eval
harness, so the results table is produced by the code that did the enforcing.
"""

from __future__ import annotations

from typing import Any

from eval import validators
from llm import client

NODE = "node_05"


def run(
    agent_a: dict[str, Any],
    agent_b: dict[str, Any],
    trend_data: dict[str, Any],
    observations: dict[str, Any],
    *,
    run_id: str,
    transport: client.Transport,
    **kwargs: Any,
) -> dict[str, Any]:
    """Synthesise both interpretations into evidence-bound, engine-grounded claims.

    The slices are separate arguments for the same reason they are at Node 4: the caller
    has already split them, and nothing here can conflate what it was handed apart.
    """
    payload = {
        "agent_a": agent_a,
        "agent_b": agent_b,
        "trend_data": trend_data,
        "observations": observations,
    }
    return client.call_node(
        NODE,
        payload,
        run_id=run_id,
        transport=transport,
        post_validate=lambda output: validators.assert_grounded(output, payload),
        **kwargs,
    )
