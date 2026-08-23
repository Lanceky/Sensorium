"""Metrics 1 and 3: every number traces to the engine, every claim names its source.

These are the two checks the whole project's honesty rests on, so they run twice — inside
the repair loop at Node 5, where a violation is handed back to the model as a correction,
and again in the eval harness, where the same functions score the run. Building them once
and using them in both places means the number reported in the results table is produced by
the code that enforced the constraint, not by a second implementation that might be kinder.

**Metric 3 — evidence binding.** Every ``claims[].evidence`` entry is a dotted path, and
the validator resolves it against the input payload the node actually received. Not a
prefix whitelist: a path either names a field that exists or it does not, and that question
is answered by the data rather than by a list somebody has to remember to update.

**Metric 1 — numeric pass-through.** The subtlety here is worth stating, because the
obvious rule is wrong. "Never emit a number that is not in the supplied data" sounds
airtight and is not, because Node 4's agent prose *is* supplied data. In the preserved run
Agent A wrote "38.46%" for a ``volume_pct_change`` of ``38.458``; the exact value appears
nowhere in its reply. A Node 5 that cited 38.46 would satisfy the obvious rule while
stating a number no engine ever computed — a rounding laundered into a validated claim by
passing it through a language model.

So the numeric authority is ``trend_data.figures`` alone. An agent's prose is an opinion
about numbers, not a source of them. ``figures_cited`` must match the engine exactly, and
prose may round only to a figure the engine actually produced.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from llm import client

#: Prose may say "up about 38%" for 38.458 — natural language rounds, and forbidding that
#: would push the model towards stilted text rather than towards honesty. What it may not
#: do is round to a number the engine never produced, so every accepted rounding is
#: generated *from* an engine figure rather than matched loosely against one.
#:
#: This is the tolerance deliberately left out of the Step 6 blindness check, which reads
#: input windows where the engine's exact values are present or absent. Here the text is a
#: model's own output, which is where restatement actually happens.
_ROUNDINGS = (0, 1, 2)

#: Numerals in prose. Captures decimals and signs; a trailing period is not swallowed.
_NUMERAL = re.compile(r"-?\d+(?:\.\d+)?")

#: One path segment: a key, optionally indexed one or more times.
_SEGMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\[\d+\])*)$")

_INDEX = re.compile(r"\[(\d+)\]")


class EvidenceError(client.MalformedOutputError):
    """A claim cites a field that does not exist in the payload it was given.

    Repairable, following Node 2's source-quote rule: the reply is well-formed JSON that
    fails the contract it claims to meet, and the model gets one attempt with the broken
    path named. A citation is only worth having if failing to resolve it costs something.
    """


class NumericError(client.MalformedOutputError):
    """A number was stated that the statistics engine never produced. Repairable."""


class AbstentionError(client.MalformedOutputError):
    """A trend was reported from figures the engine judged to be noise. Repairable."""


@dataclass
class MetricReport:
    """What was checked and what failed, so a score can be reported with its denominator."""

    checked: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def rate(self) -> float:
        """Pass rate. A metric that checked nothing scores 0.0, never 1.0.

        A vacuous denominator is the failure mode of every honest-looking metric: zero
        violations out of zero checks is not compliance, and reporting it as 100% is how a
        results table ends up saying something the run never demonstrated.
        """
        if not self.checked:
            return 0.0
        return (self.checked - len(self.failures)) / self.checked


def resolve_path(payload: Any, path: str) -> Any:
    """Follow a dotted, optionally indexed path; raise :class:`EvidenceError` if it breaks.

    ``trend_data.figures.volume_pct_change`` and ``observations.observations[0].source_quote``
    are both resolvable against a Node 5 input. Anything else is a citation to a field that
    does not exist, which is the thing Metric 3 exists to catch.
    """
    if not path or not path.strip():
        raise EvidenceError("empty evidence reference")

    current = payload
    for segment in path.split("."):
        match = _SEGMENT.match(segment)
        if match is None:
            raise EvidenceError(f"{path!r}: {segment!r} is not a valid path segment")
        key, indices = match.group(1), match.group(2)

        if not isinstance(current, dict) or key not in current:
            available = ", ".join(sorted(current)) if isinstance(current, dict) else "not an object"
            raise EvidenceError(f"{path!r}: no field {key!r} (available: {available})")
        current = current[key]

        for raw in _INDEX.findall(indices):
            index = int(raw)
            if not isinstance(current, list):
                raise EvidenceError(f"{path!r}: {key!r} is not a list, so [{index}] means nothing")
            if index >= len(current):
                raise EvidenceError(
                    f"{path!r}: index {index} is out of range; {key!r} has {len(current)} items"
                )
            current = current[index]

    return current


def check_evidence(output: dict[str, Any], payload: dict[str, Any]) -> MetricReport:
    """Metric 3: every claim names a field that exists in the payload it was given."""
    report = MetricReport()
    for position, claim in enumerate(output.get("claims", [])):
        for ref in claim.get("evidence", []):
            report.checked += 1
            try:
                resolve_path(payload, ref)
            except EvidenceError as exc:
                report.failures.append(f"claims[{position}]: {exc}")
    return report


def engine_figures(payload: dict[str, Any]) -> dict[str, float]:
    """The only numbers Node 5 is permitted to state, and where they come from.

    Deliberately not "every number in the payload". ``agent_a`` and ``agent_b`` are model
    output that happens to be adjacent to the data; treating their prose as a numeric source
    is what lets a rounded figure launder into a validated claim.
    """
    figures = {
        key: figure["value"] for key, figure in payload.get("trend_data", {}).get("figures", {}).items()
    }
    window = payload.get("trend_data", {}).get("window_weeks")
    if window is not None:
        figures["window_weeks"] = window
    return figures


def _permitted_numerals(figures: dict[str, float]) -> set[str]:
    """Every string form a real figure may legitimately take in prose."""
    permitted: set[str] = set()
    for value in figures.values():
        permitted.add(str(value))
        permitted.add(str(int(value)) if float(value).is_integer() else f"{value}")
        for places in _ROUNDINGS:
            rounded = round(float(value), places)
            permitted.add(f"{rounded:.{places}f}")
            permitted.add(str(int(rounded)) if float(rounded).is_integer() else str(rounded))
        # A percentage stated without its sign is still that figure.
        permitted.add(str(abs(value)))
        for places in _ROUNDINGS:
            permitted.add(f"{round(abs(float(value)), places):.{places}f}")
    return permitted


def check_numbers(output: dict[str, Any], payload: dict[str, Any]) -> MetricReport:
    """Metric 1: structured figures match the engine exactly; prose rounds only to real ones."""
    report = MetricReport()
    figures = engine_figures(payload)

    for position, cited in enumerate(output.get("figures_cited", [])):
        report.checked += 1
        key, value = cited["key"], cited["value"]
        if key not in figures:
            known = ", ".join(sorted(figures)) or "none"
            report.failures.append(
                f"figures_cited[{position}]: no figure {key!r} was computed (available: {known})"
            )
        elif not math.isclose(float(value), float(figures[key]), rel_tol=0.0, abs_tol=1e-9):
            report.failures.append(
                f"figures_cited[{position}]: cited {key}={value}, engine computed "
                f"{figures[key]}. Cite the engine's value exactly."
            )

    permitted = _permitted_numerals(figures)
    cited_in_prose = {str(c["value"]) for c in output.get("figures_cited", [])}
    for label, text in _prose_fields(output):
        for numeral in _NUMERAL.findall(text):
            report.checked += 1
            if numeral in permitted or numeral in cited_in_prose:
                continue
            report.failures.append(
                f"{label}: the number {numeral!r} was not computed by the statistics engine"
            )
    return report


def _prose_fields(output: dict[str, Any]) -> list[tuple[str, str]]:
    """Every free-text field a number could hide in."""
    fields: list[tuple[str, str]] = []
    for position, claim in enumerate(output.get("claims", [])):
        fields.append((f"claims[{position}].text", claim.get("text", "")))
    for key in ("agreement", "disagreement"):
        value = output.get(key)
        if isinstance(value, str):
            fields.append((key, value))
    return fields


def check_abstention(output: dict[str, Any], payload: dict[str, Any]) -> MetricReport:
    """Metric 2: abstention tracks the statistics, in both directions.

    ``insufficient_data`` is not a judgement call. It is a function of whether any figure
    reached significance, which the engine has already decided, so the check is an equality
    rather than a one-way guard.

    Both directions earn their place. The dangerous failure is under-abstention — narrating
    ``-0.459%`` as a decline — and that is what the first version checked. The live run then
    produced the opposite defect and slipped straight past: on ``agree_02`` the node wrote
    "there is a significant increase in brightness settings", cited the significant figure
    behind it, and set ``insufficient_data`` to true in the same reply. Not caution. A
    document that contradicts itself, of the kind Node 10 would hand to a clinician.

    Checked only when the engine returned figures. A window with no figures is a different
    failure — there is nothing to be significant — and scoring it here would pad the
    denominator with cases the metric never examined.
    """
    report = MetricReport()
    figures = payload.get("trend_data", {}).get("figures", {})
    if not figures:
        return report

    measured_trend = any(f.get("significant") is True for f in figures.values())
    report.checked += 1
    if output.get("insufficient_data") is measured_trend:
        significant = sorted(k for k, f in figures.items() if f.get("significant") is True)
        report.failures.append(
            f"insufficient_data is {output.get('insufficient_data')} but "
            + (
                f"{len(significant)} figure(s) reached significance ({', '.join(significant)}); "
                "a measured trend cannot be reported and disclaimed in the same reply"
                if measured_trend
                else "no figure reached significance, so there is no measured trend to report"
            )
        )

    report.checked += 1
    expected = "high" if measured_trend else "low"
    if output.get("confidence", {}).get("trend") != expected:
        report.failures.append(
            f"confidence.trend is {output.get('confidence', {}).get('trend')!r}, but "
            f"{'a figure reached' if measured_trend else 'no figure reached'} significance, "
            f"so it must be {expected!r}"
        )
    return report


def assert_grounded(output: dict[str, Any], payload: dict[str, Any]) -> None:
    """Every metric as one repair-loop hook, reporting the violations it finds.

    Raising with the complete list rather than the first failure is what lets a single
    repair attempt fix everything wrong, which is the difference between a retry budget
    that converges and one that peels off violations one at a time.
    """
    evidence = check_evidence(output, payload)
    numbers = check_numbers(output, payload)
    abstention = check_abstention(output, payload)

    if not evidence.ok:
        raise EvidenceError("; ".join(evidence.failures))
    if not numbers.ok:
        raise NumericError("; ".join(numbers.failures))
    if not abstention.ok:
        raise AbstentionError("; ".join(abstention.failures))
