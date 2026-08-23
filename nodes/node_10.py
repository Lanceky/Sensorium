"""Node 10 — the artifact that leaves the system.

Everything else in this pipeline is internal. This is the part a person prints and hands to
a clinician, so two things are checked rather than requested: the report opens with the
refusal boundary byte for byte, and every evidence reference in the synthesis survives into
it.

The boundary is validated, not prepended. Concatenating it in code would guarantee the
string and destroy the measurement — the question this track is actually about is whether a
model reproduces a fixed legal clause exactly when told to, and a metric answered by
`boundary + reply` measures the author, not the model. The repair loop means a failure costs
a retry rather than a bad report.
"""

from __future__ import annotations

from typing import Any, Sequence

from eval import validators
from llm import client
from retrieval.firecrawl import Source
from sensorium import prompts

NODE = "node_10"


def run(
    trend_data: dict[str, Any],
    synthesis: dict[str, Any],
    retrieved_sources: Sequence[Source | dict[str, str]],
    *,
    run_id: str,
    transport: client.Transport,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compile a doctor-shareable report, preserving provenance and the disclosure clause."""
    sources = [s.as_dict() if isinstance(s, Source) else dict(s) for s in retrieved_sources]
    payload = {
        "trend_data": trend_data,
        "synthesis": synthesis,
        "retrieved_sources": sources,
    }
    boundary = prompts.refusal_boundary()
    return client.call_node(
        NODE,
        payload,
        run_id=run_id,
        transport=transport,
        post_validate=lambda output: validators.assert_report_safe(output, boundary, synthesis, payload),
        **kwargs,
    )
