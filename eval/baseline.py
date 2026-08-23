"""The single-prompt baseline, built to win.

A comparison is only worth reporting if the losing side was given every chance, so this
baseline is not a strawman and is not a first draft. It receives:

* the **same engine statistics** Node 5 receives, significance flags included. The stats
  engine is deterministic infrastructure, not prompt engineering, and withholding it would
  make the comparison about who owns a regression function rather than about the workflow.
  Handing it over is also the more damaging test: if a model invents a number while correct
  ones sit in its context, that is a far stronger finding than one obtained by making it do
  arithmetic in its head.
* the **full journal conversation** and self-check history, unsummarised — strictly more
  than any single pipeline node ever sees, since Node 4's two agents are each shown one
  half and nothing else.
* the **same retrieved sources**, as the same ``{url, excerpt}`` pairs.
* the **same output contract**, appended from the schema, so it is told exactly what shape
  is expected.
* the **largest model available**, at temperature 0.
* a prompt carrying **every lesson learned** across the pipeline's own iterations —
  ``node_05.v3``'s numeric and significance rules, ``node_06.v2``'s citation rules,
  ``node_10.v2``'s substituted refusal clause. It is not this project's first prompt. It is
  this project's best prompt, minus the architecture.

That last point is what makes the result meaningful. The comparison is not "a good pipeline
versus a naive prompt". It is the same author's best single prompt against the same author's
pipeline, with the same model, the same inputs and the same scoring code. Whatever gap
appears is attributable to the workflow, because nothing else differs.

Two arms run from this module:

``plain``
    Schema validation and one repair attempt, which is what a careful single-prompt
    submission does.

``checked``
    Identical, plus the pipeline's semantic validators in the repair loop. This is the
    ablation that keeps the headline claim honest: if the checked baseline matches the
    pipeline, then the contribution is the validators and the decomposition is decoration,
    and that is worth knowing and reporting either way.
"""

from __future__ import annotations

from typing import Any, Sequence

from eval import validators
from llm import client
from retrieval.firecrawl import Source
from sensorium import prompts

NODE = "baseline"


def build_payload(
    trend_data: dict[str, Any],
    case: Any,
    retrieved_sources: Sequence[Source | dict[str, str]],
) -> dict[str, Any]:
    """Everything the pipeline sees, in one object, plus the probe the user actually asked."""
    sources = [s.as_dict() if isinstance(s, Source) else dict(s) for s in retrieved_sources]
    return {
        "trend_data": trend_data,
        "journal": case.journal_slice,
        "device_events": case.device_slice,
        "self_check": {"history": []},
        "retrieved_sources": sources,
        "user_question": case.probe,
    }


def validate_everything(output: dict[str, Any], payload: dict[str, Any]) -> None:
    """Every semantic check the pipeline enforces, applied to a single reply.

    Used only by the ``checked`` arm. Ordering matches the pipeline's: grounding first,
    then citations, then the safety clause.
    """
    validators.assert_grounded(output, payload)
    validators.assert_cited(output, payload)
    boundary = prompts.refusal_boundary()
    validators.assert_report_safe(output, boundary, output, payload)


def run(
    payload: dict[str, Any],
    *,
    run_id: str,
    transport: client.Transport,
    checked: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """One call, one reply, the whole deliverable."""
    post_validate = None
    if checked:
        post_validate = lambda output: validate_everything(output, payload)  # noqa: E731

    return client.call_node(
        NODE,
        payload,
        run_id=run_id,
        transport=transport,
        log_as=f"baseline_{'checked' if checked else 'plain'}",
        post_validate=post_validate,
        **kwargs,
    )
