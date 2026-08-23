"""Step 7 acceptance: Metrics 1, 2 and 3, and the ways each is made to fail.

These validators are the project's central claim in executable form — every number traces
to the statistics engine, every claim names a real source field, and abstention follows the
significance verdict rather than the model's mood. So the tests that matter most here are
not the ones showing the preserved run passes. They are the ones showing each check fails
when it should, because a validator that cannot fail is a decoration.

The laundering test is the one to read first. Node 4's agents round: in the preserved run
Agent A wrote "38.46%" for a figure of 38.458. Their prose is part of Node 5's input, so a
numeric check that accepted "any number present in the payload" would wave that rounding
through and still report 100% pass. `engine_figures` exists to refuse it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import generator, validators
from sensorium import runlog

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence" / "node_05-grounded-synthesis"


@pytest.fixture(scope="module")
def preserved() -> dict:
    return json.loads((EVIDENCE / "cases.json").read_text(encoding="utf-8"))


def _payload(case_id: str, preserved: dict) -> dict:
    blind = json.loads(
        (EVIDENCE.parent / "node_04-blind-agents" / "cases.json").read_text(encoding="utf-8")
    )[case_id]
    return {
        "agent_a": blind["agent_a"],
        "agent_b": blind["agent_b"],
        "trend_data": preserved[case_id]["trend"],
        "observations": blind["observations"],
    }


# --------------------------------------------------------------------------------------
# The preserved run
# --------------------------------------------------------------------------------------


def test_every_preserved_case_binds_its_evidence(preserved):
    total = validators.MetricReport()
    for case_id, case in sorted(preserved.items()):
        report = validators.check_evidence(case["output"], _payload(case_id, preserved))
        total.checked += report.checked
        total.failures += report.failures
    assert total.failures == []
    assert total.checked > 50, "too few references checked for the result to mean much"


def test_every_preserved_case_passes_numbers_through(preserved):
    total = validators.MetricReport()
    for case_id, case in sorted(preserved.items()):
        report = validators.check_numbers(case["output"], _payload(case_id, preserved))
        total.checked += report.checked
        total.failures += report.failures
    assert total.failures == []
    assert total.checked > 50


def test_abstention_matches_the_engine_on_every_case(preserved):
    for case_id, case in sorted(preserved.items()):
        report = validators.check_abstention(case["output"], _payload(case_id, preserved))
        assert report.failures == [], f"{case_id}: {report.failures}"
        assert report.checked == 2, case_id


def test_the_pipeline_recovers_the_hidden_latent_state(preserved):
    """The end-to-end result: abstention tracks ground truth the model never saw.

    The generator decides whether a decline exists before any data is synthesised, and
    nothing downstream is told. This asserts the synthesis reaches the right verdict on all
    twelve, which is the claim the results table is built on.
    """
    for case in generator.load_cases():
        output = preserved[case.case_id]["output"]
        decline = case.latent.hearing_decline_present or case.latent.vision_decline_present
        assert output["insufficient_data"] is not decline, case.case_id
        assert output["confidence"]["trend"] == ("high" if decline else "low"), case.case_id


# --------------------------------------------------------------------------------------
# Metric 1: the laundering route, and the rest of the numeric surface
# --------------------------------------------------------------------------------------


def test_a_number_rounded_by_a_blind_agent_is_not_a_source(preserved):
    """The check this metric exists for.

    Agent A's prose says "38.46%" where the engine says 38.458. That prose sits in Node 5's
    input, so citing 38.46 satisfies "the number appears in the supplied data" while stating
    a figure no engine computed.
    """
    payload = _payload("agree_01", preserved)
    assert "38.46" in payload["agent_a"]["interpretation"]
    assert "38.458" not in payload["agent_a"]["interpretation"]

    laundered = {
        "claims": [{"text": "Volume rose 38.46%.", "evidence": ["agent_a.interpretation"]}],
        "figures_cited": [{"key": "volume_pct_change", "value": 38.46}],
        "agreement": "",
        "disagreement": None,
        "confidence": {"trend": "high", "cause": "low"},
        "insufficient_data": False,
    }
    report = validators.check_numbers(laundered, payload)

    assert not report.ok
    assert any("38.458" in f for f in report.failures)


def test_engine_figures_is_exactly_the_engine_registry(preserved):
    """Exact key equality, not a heuristic.

    An earlier version of this test asserted only that no key started with "agent", which a
    mutation adding an arbitrary extra number to the whitelist survived untouched. The
    permitted set has to be pinned to what the engine computed and nothing else, because
    every number Node 5 may state is drawn from it.
    """
    payload = _payload("agree_01", preserved)
    figures = validators.engine_figures(payload)

    expected = {k: v["value"] for k, v in payload["trend_data"]["figures"].items()}
    expected["window_weeks"] = payload["trend_data"]["window_weeks"]

    assert figures == expected
    assert figures["volume_pct_change"] == 38.458


def test_a_number_only_an_agent_said_is_not_permitted_in_prose(preserved):
    """The laundering route, closed on the prose side as well as the structured side."""
    payload = _payload("agree_01", preserved)
    invented = 91.7
    assert invented not in validators.engine_figures(payload).values()

    report = validators.check_numbers(
        {
            "claims": [
                {"text": f"Volume rose {invented}%.", "evidence": ["agent_a.interpretation"]}
            ],
            "figures_cited": [],
            "agreement": "",
            "disagreement": None,
            "confidence": {"trend": "high", "cause": "low"},
            "insufficient_data": False,
        },
        payload,
    )

    assert not report.ok
    assert any("91.7" in f for f in report.failures)


def test_a_figure_key_the_engine_never_produced_is_rejected(preserved):
    output = {
        "claims": [],
        "figures_cited": [{"key": "hearing_loss_db", "value": 12.0}],
        "agreement": "",
        "disagreement": None,
        "confidence": {"trend": "high", "cause": "low"},
        "insufficient_data": False,
    }
    report = validators.check_numbers(output, _payload("agree_01", preserved))

    assert not report.ok
    assert any("hearing_loss_db" in f for f in report.failures)


def test_prose_may_round_to_a_real_figure_but_not_to_an_invented_one(preserved):
    payload = _payload("agree_01", preserved)

    def prose(text: str):
        return validators.check_numbers(
            {
                "claims": [{"text": text, "evidence": ["trend_data"]}],
                "figures_cited": [],
                "agreement": "",
                "disagreement": None,
                "confidence": {"trend": "high", "cause": "low"},
                "insufficient_data": False,
            },
            payload,
        )

    assert prose("Volume is up about 38%.").ok
    assert prose("Volume is up 38.5%.").ok
    assert not prose("Volume is up 47%.").ok


def test_a_metric_that_checked_nothing_scores_zero_not_one():
    """A vacuous denominator is how a results table reports success it never measured."""
    assert validators.MetricReport().rate == 0.0
    assert validators.MetricReport(checked=4).rate == 1.0
    assert validators.MetricReport(checked=4, failures=["x"]).rate == 0.75


# --------------------------------------------------------------------------------------
# Metric 3: evidence binding
# --------------------------------------------------------------------------------------


def test_paths_resolve_through_dicts_and_lists(preserved):
    payload = _payload("agree_01", preserved)

    assert validators.resolve_path(payload, "trend_data.figures.volume_pct_change.value") == 38.458
    assert isinstance(validators.resolve_path(payload, "agent_a.interpretation"), str)
    assert isinstance(validators.resolve_path(payload, "agent_b.unknowns[0]"), str)


@pytest.mark.parametrize(
    "path",
    [
        "wolfram.volume_trend",
        "journal.entry_4",
        "trend_data.figures.hearing_loss",
        "agent_a.interpretation.nope",
        "agent_b.unknowns[99]",
        "",
        "trend_data..figures",
    ],
)
def test_a_path_that_names_nothing_is_rejected(path, preserved):
    """`wolfram.volume_trend` and `journal.entry_4` were the examples in prompt v1.

    Both are unresolvable against a real Node 5 payload, so the prompt was instructing the
    model to produce exactly what the validator must reject — the third instance of a
    prompt disagreeing with its own contract, and the reason v2 names real paths.
    """
    with pytest.raises(validators.EvidenceError):
        validators.resolve_path(_payload("agree_01", preserved), path)


def test_an_unresolvable_citation_fails_the_metric(preserved):
    output = {
        "claims": [{"text": "Hearing declined.", "evidence": ["journal.entry_4"]}],
        "figures_cited": [],
        "agreement": "",
        "disagreement": None,
        "confidence": {"trend": "high", "cause": "low"},
        "insufficient_data": False,
    }
    report = validators.check_evidence(output, _payload("agree_01", preserved))

    assert not report.ok
    assert report.checked == 1


# --------------------------------------------------------------------------------------
# Metric 2: abstention, in both directions
# --------------------------------------------------------------------------------------


def _figures(*significance: bool | None) -> dict:
    return {
        "trend_data": {
            "figures": {
                f"sig_{i}": {"value": float(i), "unit": "%", "method": "percent_change", **({} if s is None else {"significant": s})}
                for i, s in enumerate(significance)
            }
        }
    }


def _output(insufficient: bool, trend: str) -> dict:
    return {
        "claims": [],
        "figures_cited": [],
        "agreement": "",
        "disagreement": None,
        "confidence": {"trend": trend, "cause": "low"},
        "insufficient_data": insufficient,
    }


def test_narrating_noise_as_a_trend_is_caught():
    """The dangerous direction: -0.459% described as a decline."""
    report = validators.check_abstention(_output(False, "high"), _figures(False, False))

    assert not report.ok
    assert any("no figure reached significance" in f for f in report.failures)


def test_disclaiming_a_measured_trend_is_caught():
    """The direction the one-sided version missed, found by a live run on agree_02.

    The node reported a significant increase in brightness, cited the figure behind it, and
    set insufficient_data true in the same reply.
    """
    report = validators.check_abstention(_output(True, "low"), _figures(True, False))

    assert not report.ok
    assert any("cannot be reported and disclaimed" in f for f in report.failures)


def test_both_directions_pass_when_they_agree_with_the_engine():
    assert validators.check_abstention(_output(False, "high"), _figures(True, False)).ok
    assert validators.check_abstention(_output(True, "low"), _figures(False, False)).ok


def test_a_window_with_no_figures_is_not_scored():
    """Nothing to be significant, so counting it would pad the denominator."""
    report = validators.check_abstention(_output(True, "low"), {"trend_data": {"figures": {}}})

    assert report.checked == 0
    assert report.ok


def test_a_figure_with_no_significance_verdict_does_not_count_as_significant():
    """Two points fit a line exactly; `significant` is absent, not True."""
    report = validators.check_abstention(_output(True, "low"), _figures(None, None))

    assert report.ok


# --------------------------------------------------------------------------------------
# The repair loop
# --------------------------------------------------------------------------------------


def test_the_validators_are_repairable_not_fatal():
    """A violation must be handed back to the model, not kill the run.

    `llm.client` catches MalformedOutputError and SchemaError only, so an AssertionError
    here would abort a run that one retry could have fixed.
    """
    from llm import client

    for error in (validators.EvidenceError, validators.NumericError, validators.AbstentionError):
        assert issubclass(error, client.MalformedOutputError)


def test_the_preserved_run_shows_both_validators_firing_and_being_fixed(preserved, monkeypatch):
    """Evidence that these checks work in the loop, not just in a test.

    Two cases needed a repair, one for each of the first two metrics, and both converged.
    """
    monkeypatch.setattr(runlog, "RUNS_DIR", EVIDENCE / "runs")
    errors = [
        call.error
        for case in preserved.values()
        for call in runlog.load_calls(case["run_id"], "node_05")
        if call.error
    ]

    assert any("NumericError" in e for e in errors)
    assert any("EvidenceError" in e for e in errors)
    for case in preserved.values():
        calls = runlog.load_calls(case["run_id"], "node_05")
        assert calls[-1].error is None, "a run ended on a failure"
