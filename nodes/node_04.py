"""Node 4 — the two blind agents.

Independence here is structural, not requested. Each agent's payload is built from one data
slice and validated against a schema whose ``additionalProperties: false`` rejects the
other slice outright: attaching journal text to ``node_04a.input.json`` as a new field
fails validation before any request is sent. The prompts say "you see only..." as a
description of that fact, not as the mechanism enforcing it — a prompt asking a model to
disregard something it can see is exactly the cosmetic version this design rejects.

The schema half is necessary and not sufficient, which is worth stating precisely rather
than claiming more than it does. It closes the accidental route — a caller passing the
wrong slice, a field appended to the wrong dict — but it cannot close every route, because
some permitted fields are free-form strings. ``node_03.output.json`` gives every figure a
``unit`` of type string, and a journal sentence sitting in a ``unit`` is a schema-valid
payload. `eval.independence.assert_blind` is what catches that, by reading the context
window that was actually sent rather than the shape it was allowed to have. Both tests
exist because neither alone is the guarantee; ``tests/test_independence.py`` exercises each
against the route the other misses.

The two agents are routed identically on purpose. ``config.check_invariants`` refuses to
start if their size or temperature ever drift apart, because a disagreement produced by
two different models is a fact about model choice, not about the person's data.

:func:`run_both` deliberately takes the two slices as separate arguments rather than a
combined object. A single ``case``-shaped parameter would put both slices in one scope and
leave blindness resting on this function not misusing them; passing them separately means
the caller has already split them before anything here can conflate them.
"""

from __future__ import annotations

from typing import Any

from llm import client

AGENT_A = "node_04a"
AGENT_B = "node_04b"


def run_agent_a(
    trend_data: dict[str, Any],
    *,
    run_id: str,
    transport: client.Transport,
    **kwargs: Any,
) -> dict[str, Any]:
    """Interpret the device trends alone. Never sees the journal."""
    return client.call_node(
        AGENT_A, {"trend_data": trend_data}, run_id=run_id, transport=transport, **kwargs
    )


def run_agent_b(
    observations: dict[str, Any],
    *,
    run_id: str,
    transport: client.Transport,
    **kwargs: Any,
) -> dict[str, Any]:
    """Interpret the journal alone. Never sees the numbers."""
    return client.call_node(
        AGENT_B, {"observations": observations}, run_id=run_id, transport=transport, **kwargs
    )


def run_both(
    trend_data: dict[str, Any],
    observations: dict[str, Any],
    *,
    run_id: str,
    transport: client.Transport,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run both agents on their own slice and return ``(agent_a, agent_b)``."""
    a = run_agent_a(trend_data, run_id=run_id, transport=transport, **kwargs)
    b = run_agent_b(observations, run_id=run_id, transport=transport, **kwargs)
    return a, b
