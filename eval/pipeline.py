"""Drive Nodes 6 and 10 over the preserved Node 5 syntheses and score the run.

Nodes 3 and 5 are not re-run. Their outputs were preserved into
``evidence/node_05-grounded-synthesis/`` when Step 7 was measured, and re-running them here
would cost twelve model calls to obtain slightly different inputs, which would make the two
sets of measurements no longer comparable. Reusing them means the citation and safety
numbers are scored against the exact syntheses the numeric and abstention numbers were
scored against.

Usage (from the repo root, with a live key)::

    python -m eval.pipeline
"""

from __future__ import annotations

import json
from typing import Any

from eval import validators
from llm import client
from nodes import node_06, node_10
from retrieval.firecrawl import load_snapshot
from sensorium import config, prompts

SYNTHESIS_EVIDENCE = config.REPO_ROOT / "evidence" / "node_05-grounded-synthesis" / "cases.json"
OUT_DIR = config.REPO_ROOT / "evidence" / "node_06_10-citations-and-safety"


def load_syntheses() -> dict[str, Any]:
    """The Step 7 run, reused as this step's input."""
    return json.loads(SYNTHESIS_EVIDENCE.read_text(encoding="utf-8"))


def run_case(
    case_id: str,
    trend: dict[str, Any],
    synthesis: dict[str, Any],
    sources: list[dict[str, str]],
    transport: client.Transport,
) -> dict[str, Any]:
    """Suggestions and a report for one case, with the retrieved set pinned into the record."""
    run_id = f"step08-{case_id}"
    suggestions = node_06.run(synthesis, sources, run_id=run_id, transport=transport)
    report = node_10.run(trend, synthesis, sources, run_id=run_id, transport=transport)
    return {
        "run_id": run_id,
        "retrieved_urls": [s["url"] for s in sources],
        "suggestions": suggestions,
        "report": report,
    }


def score(results: dict[str, Any], sources: list[dict[str, str]]) -> dict[str, Any]:
    """Citation validity and safety adherence across every case in ``results``."""
    payload = {"retrieved_sources": sources}
    boundary = prompts.refusal_boundary()
    syntheses = load_syntheses()

    citation = validators.MetricReport()
    safety = validators.MetricReport()
    provenance = validators.MetricReport()

    for case_id, record in sorted(results.items()):
        for metric, report in (
            (citation, validators.check_citations(record["suggestions"], payload)),
            (safety, validators.check_refusal_boundary(record["report"], boundary)),
            (
                provenance,
                validators.check_evidence_preserved(
                    record["report"], syntheses[case_id]["output"]
                ),
            ),
        ):
            metric.checked += report.checked
            metric.failures.extend(f"{case_id}: {f}" for f in report.failures)

    return {
        "citation_validity": {"rate": citation.rate, "checked": citation.checked,
                              "failures": citation.failures},
        "safety_adherence": {"rate": safety.rate, "checked": safety.checked,
                             "failures": safety.failures},
        "provenance_preserved": {"rate": provenance.rate, "checked": provenance.checked,
                                 "failures": provenance.failures},
    }


def main() -> int:
    sources = [s.as_dict() for s in load_snapshot()]
    transport = client.FeatherlessTransport()
    syntheses = load_syntheses()

    results: dict[str, Any] = {}
    for case_id, record in sorted(syntheses.items()):
        try:
            results[case_id] = run_case(
                case_id, record["trend"], record["output"], sources, transport
            )
            cited = [s["source_url"] for s in results[case_id]["suggestions"]["suggestions"]]
            print(f"{case_id:16} ok  cites={sum(c is not None for c in cited)}/{len(cited)}")
        except Exception as exc:  # noqa: BLE001 - a failed case is a result, not a crash
            results[case_id] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{case_id:16} FAIL {type(exc).__name__}: {str(exc)[:110]}")

    ok = {k: v for k, v in results.items() if "error" not in v}
    report = score(ok, sources)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cases.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    for name, block in report.items():
        print(f"{name:24} {block['rate']:.3f}  ({block['checked']} checked)")
        for failure in block["failures"]:
            print(f"    - {failure}")
    return 0 if all(b["rate"] == 1.0 for b in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
