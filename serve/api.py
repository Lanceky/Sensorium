"""The reasoning service the Android app talks to.

This module contains no reasoning. It is a thin HTTP surface over the nodes that already
exist, and that is the point: the app demonstrates the workflow in this repository — the
same prompts, the same model routing, the same validators, the same refusal boundary —
rather than a second implementation that could quietly drift from the one that was
measured. Every guarantee the results table reports is a guarantee about this endpoint,
because the endpoint calls exactly that code.

The split is on-device capture, off-device reasoning. The phone reads its own settings and
asks the questions; nodes 2 through 10 run here. Two things follow. The phone never holds
an API key, so a stolen handset does not leak one. And the journal text crosses the network
only when a person presses the button that says it will.

Run it from the repo root::

    python -m serve.api

then point the phone at it without any network setup at all::

    adb reverse tcp:8765 tcp:8765
"""

from __future__ import annotations

import traceback
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from eval import validators
from llm import client
from nodes import node_02, node_04, node_05, node_06, node_10
from retrieval.firecrawl import load_snapshot
from sensorium import prompts
from stats import engine

WINDOW_WEEKS = 4

app = FastAPI(title="Sensorium", version="0.1")


class AnalyseRequest(BaseModel):
    device_slice: dict[str, Any] = Field(..., description="Node 0's recorded signal events.")
    conversation: list[dict[str, str]] = Field(default_factory=list)


@app.get("/health")
def health() -> dict[str, str]:
    """Confirms the service is reachable and that the clause the reports must carry is loaded."""
    return {"status": "ok", "boundary_chars": str(len(prompts.refusal_boundary()))}


@app.post("/analyse")
def analyse(request: AnalyseRequest) -> dict[str, Any]:
    """Run the workflow over one device's history and one week's answers.

    A node that cannot satisfy its contract raises, and this endpoint returns the error
    rather than a partial report. That is the intended behaviour and it is worth being
    explicit about, because the alternative is the failure mode the whole project exists to
    avoid: a screen that looks like a finding but is not one. The operator seeing an error
    is a strictly better outcome than a person reading a number no check would accept.
    """
    run_id = f"app-{uuid.uuid4().hex[:8]}"
    transport = client.FeatherlessTransport()

    try:
        trend = engine.compute(request.device_slice, self_check=None, window_weeks=WINDOW_WEEKS)
        observations = node_02.run(request.conversation, run_id=run_id, transport=transport)
        agent_a, agent_b = node_04.run_both(
            trend, observations, run_id=run_id, transport=transport
        )
        synthesis = node_05.run(
            agent_a, agent_b, trend, observations, run_id=run_id, transport=transport
        )
        sources = load_snapshot()
        suggestions = node_06.run(synthesis, sources, run_id=run_id, transport=transport)
        report = node_10.run(trend, synthesis, sources, run_id=run_id, transport=transport)
    except Exception as exc:  # noqa: BLE001 - the app is told what failed, not given a guess
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc

    validators.check_refusal_boundary(report, prompts.refusal_boundary())

    return {
        "run_id": run_id,
        "report_markdown": report["report_markdown"],
        "headline": _headline(synthesis),
        "disagreement": synthesis.get("disagreement"),
        "insufficient_data": synthesis.get("insufficient_data", False),
        "confidence": synthesis.get("confidence", {}),
        "suggestions": suggestions.get("suggestions", []),
        "figures": _figures(trend),
        "citations": report.get("citations", []),
        "observations": observations.get("observations", []),
        "no_symptom_statements": observations.get("no_symptom_statements", []),
        "agent_a": agent_a,
        "agent_b": agent_b,
    }


def _headline(synthesis: dict[str, Any]) -> str:
    """Node 5 states its finding as a list of claims; the app shows them as its headline.

    All of them, not the first. Each claim carries its own evidence and the synthesis node
    emits several precisely because a person's vision and hearing signals can move
    independently — showing only claim zero would silently drop a second finding that was
    just as well evidenced. When there are no claims the node's ``agreement`` sentence is
    the honest fallback, because that is where it explains what it found instead.
    """
    claims = synthesis.get("claims") or []
    texts = [
        str(claim["text"]).strip()
        for claim in claims
        if isinstance(claim, dict) and str(claim.get("text", "")).strip()
    ]
    if texts:
        return "\n\n".join(texts)
    return str(synthesis.get("agreement") or "").strip()


def _figures(trend: dict[str, Any]) -> list[dict[str, Any]]:
    """The engine's figures, flattened for display, with significance kept attached.

    Significance travels with every figure all the way to the screen. A percentage shown
    without it invites exactly the reading the engine is there to prevent, which is that a
    large-looking number is a finding.

    It is passed through as three states, not two. The engine sets ``significant`` to
    ``None`` when the window cannot answer "does this slope differ from zero" at all — two
    points fit a line exactly — and it says so in a comment because the distinction is the
    honest part. Coercing that ``None`` to ``False`` here would relabel *we could not test
    this* as *we tested this and it is noise*, which is a stronger claim than the data
    supports and the opposite of the one the engine went out of its way to avoid making.
    """
    out = []
    for name, figure in sorted((trend.get("figures") or {}).items()):
        if not isinstance(figure, dict):
            continue
        value = figure.get("value")
        unit = figure.get("unit", "")
        significant = figure.get("significant")
        out.append({
            "name": name.replace("_", " "),
            "value": f"{value} {unit}".strip(),
            "significant": significant if significant is None else bool(significant),
        })
    return out


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")


if __name__ == "__main__":
    main()
