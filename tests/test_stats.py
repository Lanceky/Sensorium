"""Step 4 acceptance: the deterministic engine and the number registry.

The registry is the only place numbers are allowed to originate, so these tests care about
two things above all: that real trends are recovered, and that absent trends stay absent.
A statistics engine that invents a finding is worse than none, because everything
downstream is built to trust it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from eval.generator import load_cases
from nodes import node_03
from sensorium import config, runlog, schemas
from stats import engine

WINDOW = 4

#: Comfortably inside the gap between declining and stable cases in the evaluation set:
#: declining volume moves 25-38%, stable volume stays within +/-4%.
VOLUME_DECLINE_PCT = 10.0
#: Declining brightness moves 9-20%, stable brightness stays within +/-3.5%.
BRIGHTNESS_DECLINE_PCT = 5.0


def _series(values, step_weeks=0.5):
    return engine.Series(
        "test", tuple(i * step_weeks for i in range(len(values))), tuple(float(v) for v in values)
    )


@pytest.fixture(scope="module")
def cases():
    return {case.case_id: case for case in load_cases()}


@pytest.fixture(scope="module")
def registries(cases):
    return {
        cid: engine.compute(case.device_slice, None, WINDOW) for cid, case in cases.items()
    }


# ------------------------------------------------------------------------- series basics


def test_build_series_orders_by_time_and_rebases_to_weeks():
    events = [
        {"ts": "2026-08-08T12:00:00+00:00", "signal": "volume", "value": 7},
        {"ts": "2026-08-01T12:00:00+00:00", "signal": "volume", "value": 5},
        {"ts": "2026-08-15T12:00:00+00:00", "signal": "brightness", "value": 99},
    ]
    series = engine.build_series(events, "volume")
    assert series.values == (5.0, 7.0)
    assert series.weeks == (0.0, 1.0)


def test_build_series_returns_none_for_an_absent_signal():
    assert engine.build_series([], "volume") is None


def test_linear_trend_recovers_a_known_slope():
    slope, intercept = engine.linear_trend(_series([10, 11, 12, 13], step_weeks=1.0))
    assert slope == pytest.approx(1.0)
    assert intercept == pytest.approx(10.0)


def test_a_trend_needs_two_distinct_timestamps():
    flat_in_time = engine.Series("test", (0.0, 0.0), (1.0, 2.0))
    with pytest.raises(engine.StatsError, match="two distinct timestamps"):
        engine.linear_trend(flat_in_time)


def test_on_rate_counts_non_zero_observations():
    assert engine.on_rate(_series([0, 1, 1, 0])) == 0.5


# ---------------------------------------------------------------------- percent change


def test_percent_change_follows_the_fit_not_the_endpoints():
    """One bad reading at the edge of the window must not set the headline figure."""
    misleading = _series([10, 12, 14, 16, 18, 10])
    endpoint_view = (10 - 10) / 10 * 100
    assert endpoint_view == 0
    assert engine.fitted_percent_change(misleading) > 5


def test_percent_change_of_a_flat_series_is_zero():
    assert engine.fitted_percent_change(_series([8, 8, 8, 8])) == pytest.approx(0.0)


# ------------------------------------------------------------------- change point tests


def test_a_real_slope_change_is_detected():
    flat_then_climbing = [10, 10, 10, 10, 12, 14, 16, 18]
    assert engine.changepoint_week(_series(flat_then_climbing)) is not None


def test_a_steady_ramp_is_not_split():
    """The trap a mean-shift detector falls into: two flat segments describe a ramp better
    than one flat segment, so it reports a change point on every declining signal."""
    assert engine.changepoint_week(_series([10, 11, 12, 13, 14, 15, 16, 17])) is None


def test_a_short_series_is_never_split():
    assert engine.changepoint_week(_series([1, 2, 3, 4, 5])) is None


def test_the_false_positive_rate_matches_the_declared_alpha():
    """An F-test is only worth using if it is calibrated. On pure noise the detector must
    fire at roughly its nominal alpha, not at whatever a residual ratio would produce."""
    rng = random.Random(20260824)
    trials = 600
    fired = sum(
        1
        for _ in range(trials)
        if engine.changepoint_week(_series([10 + rng.gauss(0, 1) for _ in range(8)])) is not None
    )
    assert fired / trials < 3 * engine.CHANGEPOINT_ALPHA


def test_a_ramp_does_not_inflate_the_false_positive_rate():
    rng = random.Random(11)
    trials = 600
    fired = sum(
        1
        for _ in range(trials)
        if engine.changepoint_week(
            _series([10 + 0.6 * i + rng.gauss(0, 0.4) for i in range(8)])
        )
        is not None
    )
    assert fired / trials < 3 * engine.CHANGEPOINT_ALPHA


# --------------------------------------------------------------------- the real cases


def test_every_case_produces_a_contract_valid_registry(registries):
    for case_id, registry in registries.items():
        schemas.validate("node_03.output.json", registry), case_id


def test_every_figure_declares_a_deterministic_method(registries):
    allowed = set(schemas.get_schema("common.json")["$defs"]["stat_method"]["enum"])
    for case_id, registry in registries.items():
        for key, figure in registry["figures"].items():
            assert figure["method"] in allowed, f"{case_id}:{key}"
            assert "llm" not in figure["method"].lower()


def test_declining_hearing_shows_a_volume_trend(cases, registries):
    for case_id, case in cases.items():
        if not case.latent.hearing_decline_present:
            continue
        change = registries[case_id]["figures"]["volume_pct_change"]["value"]
        assert change > VOLUME_DECLINE_PCT, f"{case_id} missed a real hearing trend"


def test_stable_hearing_shows_no_volume_trend(cases, registries):
    for case_id, case in cases.items():
        if case.latent.hearing_decline_present:
            continue
        change = registries[case_id]["figures"]["volume_pct_change"]["value"]
        assert abs(change) < VOLUME_DECLINE_PCT, f"{case_id} invented a hearing trend"


def test_declining_vision_shows_a_brightness_trend(cases, registries):
    for case_id, case in cases.items():
        if not case.latent.vision_decline_present:
            continue
        change = registries[case_id]["figures"]["brightness_pct_change"]["value"]
        assert change > BRIGHTNESS_DECLINE_PCT, f"{case_id} missed a real vision trend"


def test_stable_vision_shows_no_brightness_trend(cases, registries):
    for case_id, case in cases.items():
        if case.latent.vision_decline_present:
            continue
        change = registries[case_id]["figures"]["brightness_pct_change"]["value"]
        assert abs(change) < BRIGHTNESS_DECLINE_PCT, f"{case_id} invented a vision trend"


def test_the_null_cases_produce_flat_trends(cases, registries):
    """The cases that exist specifically to catch fabrication."""
    for case_id in ("null_01", "null_02"):
        figures = registries[case_id]["figures"]
        assert abs(figures["volume_pct_change"]["value"]) < 5
        assert abs(figures["brightness_pct_change"]["value"]) < 5


def test_thin_data_is_reported_as_insufficient(registries):
    """null_02 spans under two weeks with four readings per signal. Figures are still
    computed, but the flag has to say they cannot carry a conclusion."""
    assert registries["null_02"]["sufficient_data"] is False
    assert registries["null_02"]["figures"], "figures should still be computed"


def test_adequate_windows_are_reported_as_sufficient(registries):
    for case_id, registry in registries.items():
        if case_id == "null_02":
            continue
        assert registry["sufficient_data"] is True, case_id


def test_window_weeks_never_overstates_the_observed_span(cases, registries):
    for case_id, case in cases.items():
        series = engine.build_series(case.device_slice["events"], "volume")
        assert registries[case_id]["window_weeks"] <= series.span_weeks + 1e-9


# ------------------------------------------------------------------- windowing and inputs


def test_events_outside_the_requested_window_are_dropped():
    events = [
        {"ts": "2026-06-01T12:00:00+00:00", "signal": "volume", "value": 1},
        {"ts": "2026-08-15T12:00:00+00:00", "signal": "volume", "value": 5},
        {"ts": "2026-08-22T12:00:00+00:00", "signal": "volume", "value": 6},
    ]
    kept = engine._within_window(events, window_weeks=2)
    assert [e["value"] for e in kept] == [5, 6]


def test_a_narrower_window_changes_the_answer(cases):
    """If window_weeks were ignored, a required input would be doing nothing."""
    case = cases["agree_01"]
    wide = engine.compute(case.device_slice, None, 4)
    narrow = engine.compute(case.device_slice, None, 1)
    assert wide["window_weeks"] > narrow["window_weeks"]


def test_self_check_sessions_are_counted_when_present(cases):
    case = cases["agree_01"]
    self_check = {"sessions": [{"date": "2026-08-01", "step_reached": 4, "modality": "hearing"}]}
    registry = engine.compute(case.device_slice, self_check, WINDOW)
    figure = registry["figures"]["self_check_session_count"]
    assert figure["value"] == 1 and figure["method"] == "session_count"


def test_no_self_check_means_no_session_figure(registries):
    assert "self_check_session_count" not in registries["agree_01"]["figures"]


def test_the_engine_is_deterministic(cases):
    case = cases["conflict_03"]
    first = engine.compute(case.device_slice, None, WINDOW)
    second = engine.compute(case.device_slice, None, WINDOW)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# ----------------------------------------------------------------------- the node wrapper


@pytest.fixture
def isolated_runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", tmp_path / "runs")


def _node_input(case):
    return {"signals": case.device_slice, "self_check": None, "window_weeks": WINDOW}


def test_node_03_returns_validated_output(cases, isolated_runs_dir):
    run_id = runlog.new_run_id()
    result = node_03.run(_node_input(cases["agree_01"]), run_id=run_id)
    schemas.validate("node_03.output.json", result)


def test_node_03_rejects_a_malformed_payload(cases, isolated_runs_dir):
    run_id = runlog.new_run_id()
    with pytest.raises(schemas.SchemaError):
        node_03.run({"signals": cases["agree_01"].device_slice}, run_id=run_id)


def test_node_03_records_the_registry_for_metric_1(cases, isolated_runs_dir):
    """Metric 1 scores downstream numbers against the recorded registry, so the registry
    has to be in the log rather than recomputed later."""
    run_id = runlog.new_run_id()
    result = node_03.run(_node_input(cases["agree_01"]), run_id=run_id)
    call = runlog.load_call(run_id, "node_03")
    assert call.raw_output == result
    assert call.error is None


def test_the_log_names_a_computation_not_a_model(cases, isolated_runs_dir):
    run_id = runlog.new_run_id()
    node_03.run(_node_input(cases["agree_01"]), run_id=run_id)
    call = runlog.load_call(run_id, "node_03")
    assert call.model.startswith("deterministic:")
    assert call.prompt_version == "n/a"


def test_node_03_is_declared_deterministic_and_never_routed():
    assert "node_03" in config.DETERMINISTIC_NODES
    assert "node_03" not in config.REGISTRY


def test_no_llm_dependency_reaches_the_engine():
    """A model import here would make the 'numbers are computed, not generated' claim
    unverifiable by inspection."""
    source = Path(engine.__file__).read_text(encoding="utf-8")
    assert "openai" not in source
    assert "llm" not in source.replace("llm may appear", "").replace("No LLM", "")


# --------------------------------------------------------------------------------------
# Significance (Step 7): the verdict the first version of this engine computed and discarded
# --------------------------------------------------------------------------------------


def test_regression_figures_carry_a_significance_verdict():
    """Without this the engine hands Node 5 noise and signal as equally reportable facts."""
    from eval import generator

    case = next(c for c in generator.load_cases() if c.case_id == "agree_01")
    out = engine.compute(case.device_slice, None, 4)

    regression = {
        k: v
        for k, v in out["figures"].items()
        if v["method"] in ("linear_regression", "percent_change")
    }
    assert regression
    assert all("significant" in v for v in regression.values()), sorted(regression)
    assert any(v["significant"] is True for v in regression.values())


def test_significance_recovers_the_hidden_latent_state_on_every_case():
    """The result the abstention metric rests on.

    The generator decides whether a decline exists before any data is synthesised. Requiring
    p < 0.05 on a fitted slope reproduces that decision exactly across the eval set: both
    null cases contain no significant figure, and all ten cases carrying a real decline
    contain at least one. That correspondence is what makes `insufficient_data` a function
    of the statistics rather than a judgement call.
    """
    from eval import generator

    for case in generator.load_cases():
        out = engine.compute(case.device_slice, None, 4)
        measured = any(f.get("significant") is True for f in out["figures"].values())
        decline = case.latent.hearing_decline_present or case.latent.vision_decline_present
        assert measured is decline, f"{case.case_id}: measured={measured} latent={decline}"


def test_a_flat_series_has_no_significance_verdict():
    """Zero residual variance makes the t-statistic 0/0, so the honest answer is "unknown".

    Someone who never once changes their volume produces this. Returning NaN and letting it
    fall through `p < alpha` would answer "not significant" by accident of IEEE comparison
    rules rather than by measurement, so the engine refuses instead and the figure carries
    no verdict.
    """
    series = engine.Series("volume", (0.0, 1.0, 2.0, 3.0), (5.0, 5.0, 5.0, 5.0))
    with pytest.raises(engine.StatsError, match="perfectly flat"):
        engine.trend_p_value(series)


def test_a_flat_signal_produces_a_figure_with_no_significance_key():
    """The refusal above has to survive `compute`, not just the helper."""
    events = [
        {"ts": f"2026-08-{1 + 7 * i:02d}T12:00:00+00:00", "signal": "volume", "value": 5.0}
        for i in range(4)
    ]
    out = engine.compute({"events": events}, None, 4)

    trend = out["figures"].get("volume_trend_per_week")
    assert trend is not None
    assert "significant" not in trend


def test_a_clean_ramp_is_significant():
    series = engine.Series("volume", (0.0, 1.0, 2.0, 3.0), (5.0, 6.0, 7.0, 8.0))
    assert engine.trend_p_value(series) < engine.TREND_ALPHA


def test_two_points_cannot_be_judged_significant():
    """A line through two points fits exactly, so 'is this slope real' has no answer."""
    series = engine.Series("volume", (0.0, 1.0), (5.0, 9.0))
    with pytest.raises(engine.StatsError):
        engine.trend_p_value(series)
