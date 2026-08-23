"""Step 1 acceptance: the run log captures enough to serve as submission evidence.

Later steps read this log rather than re-deriving state: Step 6 proves Node 4 independence
from the recorded payloads, Step 9 scores numeric fidelity against the recorded Node 3
output, and the iteration log is assembled from recorded failures. Anything missing here
cannot be reconstructed afterwards.
"""

from __future__ import annotations

import re

import pytest

from sensorium import runlog

REQUIRED_FIELDS = {
    "run_id",
    "node",
    "model",
    "temperature",
    "prompt_version",
    "input_payload",
    "raw_output",
    "latency_ms",
}


@pytest.fixture(autouse=True)
def isolated_runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", tmp_path / "runs")


def _log(run_id, node="node_05", payload=None, output=None):
    return runlog.log_call(
        run_id=run_id,
        node=node,
        model="test-model",
        temperature=0.0,
        prompt_version="v1",
        input_payload=payload if payload is not None else {"trend_data": {"figures": {}}},
        raw_output=output if output is not None else {"claims": []},
        latency_ms=42,
    )


def test_run_ids_are_unique_and_timestamp_prefixed():
    ids = [runlog.new_run_id() for _ in range(50)]
    assert len(set(ids)) == 50
    assert all(re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{6}", i) for i in ids)


def test_round_trip_preserves_every_required_field():
    run_id = runlog.new_run_id()
    _log(run_id)
    (call,) = runlog.load_calls(run_id)
    assert REQUIRED_FIELDS <= set(vars(call))
    assert call.node == "node_05"
    assert call.latency_ms == 42
    assert call.input_payload == {"trend_data": {"figures": {}}}


def test_calls_are_appended_in_order():
    run_id = runlog.new_run_id()
    for node in ("node_02", "node_04a", "node_04b", "node_05"):
        _log(run_id, node=node)
    assert [c.node for c in runlog.load_calls(run_id)] == [
        "node_02",
        "node_04a",
        "node_04b",
        "node_05",
    ]


def test_load_call_filters_by_node():
    run_id = runlog.new_run_id()
    _log(run_id, node="node_04a", payload={"trend_data": {"figures": {}}})
    _log(run_id, node="node_04b", payload={"observations": {"observations": []}})
    assert runlog.load_call(run_id, "node_04b").input_payload == {
        "observations": {"observations": []}
    }


def test_payload_text_supports_containment_checks():
    """Step 6's assert_blind() searches the serialised payload for the other slice's
    tokens, so serialisation must actually include nested values."""
    run_id = runlog.new_run_id()
    _log(run_id, node="node_04a", payload={"trend_data": {"figures": {"volume_pct_change_3w": 1}}})
    text = runlog.load_call(run_id, "node_04a").payload_text()
    assert "volume_pct_change_3w" in text


def test_failed_calls_are_logged_not_swallowed():
    """A failure is an iteration-log entry (context.md section 10). Losing it loses the
    evidence that the prompt fix was needed."""
    run_id = runlog.new_run_id()
    with pytest.raises(RuntimeError):
        with runlog.timed_call(run_id, "node_05", "m", 0.0, "v1", {"x": 1}):
            raise RuntimeError("model returned unparseable JSON")

    (call,) = runlog.load_calls(run_id)
    assert call.error is not None and "unparseable" in call.error
    assert call.raw_output is None


def test_timed_call_records_output_and_latency():
    run_id = runlog.new_run_id()
    with runlog.timed_call(run_id, "node_02", "m", 0.0, "v1", {"conversation": []}) as slot:
        slot["raw_output"] = {"observations": []}

    (call,) = runlog.load_calls(run_id)
    assert call.raw_output == {"observations": []}
    assert call.error is None
    assert call.latency_ms >= 0


def test_missing_run_raises():
    with pytest.raises(runlog.RunLogError, match="no run log"):
        runlog.load_calls("does-not-exist")


def test_missing_node_in_run_raises():
    run_id = runlog.new_run_id()
    _log(run_id, node="node_02")
    with pytest.raises(runlog.RunLogError, match="no call for node"):
        runlog.load_call(run_id, "node_05")


def test_ambiguous_node_lookup_raises():
    run_id = runlog.new_run_id()
    _log(run_id, node="node_01")
    _log(run_id, node="node_01")
    with pytest.raises(runlog.RunLogError, match="use load_calls"):
        runlog.load_call(run_id, "node_01")


def test_list_runs_reports_logged_runs():
    assert runlog.list_runs() == []
    first, second = runlog.new_run_id(), runlog.new_run_id()
    _log(first)
    _log(second)
    assert set(runlog.list_runs()) == {first, second}
