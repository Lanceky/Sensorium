"""Step 1 acceptance: every schema loads, and every node has a fixture that proves it works.

The invalid fixtures are the interesting half. Each one encodes a contract violation that
would silently damage the submission if it reached production - an unanchored extraction,
a number with no provenance, an agent that can see the other agent's data slice.
"""

from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from sensorium import schemas


def test_schema_directory_is_not_empty():
    assert schemas.node_schema_names(), "no node schemas found in schemas/"


@pytest.mark.parametrize("name", schemas.schema_names())
def test_schema_is_itself_valid(name):
    """Each file must be a well-formed Draft 2020-12 schema, not merely valid JSON."""
    Draft202012Validator.check_schema(schemas.get_schema(name))


@pytest.mark.parametrize("name", schemas.schema_names())
def test_schema_id_matches_filename(name):
    """$refs across schemas resolve by filename, so the two must agree."""
    assert schemas.get_schema(name)["$id"] == name


@pytest.mark.parametrize("name", schemas.node_schema_names())
def test_every_node_schema_has_both_fixtures(name, valid_fixtures, invalid_fixtures):
    assert name in valid_fixtures, f"{name} has no valid fixture"
    assert name in invalid_fixtures, f"{name} has no invalid fixture"


@pytest.mark.parametrize("name", schemas.node_schema_names())
def test_valid_fixture_passes(name, valid_fixtures):
    schemas.validate(name, valid_fixtures[name])


@pytest.mark.parametrize("name", schemas.node_schema_names())
def test_invalid_fixture_fails(name, invalid_fixtures):
    with pytest.raises(schemas.SchemaError):
        schemas.validate(name, invalid_fixtures[name])


def test_cross_schema_refs_resolve(valid_fixtures):
    """Node 5's input embeds Node 3 and Node 2 outputs; if $ref resolution silently
    no-ops, a malformed nested payload would pass. Prove it does not."""
    broken = json.loads(json.dumps(valid_fixtures["node_05.input.json"]))
    broken["trend_data"]["figures"]["volume_pct_change_3w"].pop("method")
    with pytest.raises(schemas.SchemaError):
        schemas.validate("node_05.input.json", broken)


def test_blind_agent_inputs_reject_the_other_slice(valid_fixtures):
    """The schema-level half of Node 4's independence guarantee (context.md section 4)."""
    a = dict(valid_fixtures["node_04a.input.json"])
    a["observations"] = {"observations": []}
    with pytest.raises(schemas.SchemaError):
        schemas.validate("node_04a.input.json", a)

    b = dict(valid_fixtures["node_04b.input.json"])
    b["trend_data"] = {"figures": {}, "window_weeks": 1, "sufficient_data": False}
    with pytest.raises(schemas.SchemaError):
        schemas.validate("node_04b.input.json", b)


def test_synthesis_claim_requires_evidence(valid_fixtures):
    """No claim without evidence - rule 1 of the Node 5 prompt, enforced structurally."""
    payload = json.loads(json.dumps(valid_fixtures["node_05.output.json"]))
    payload["claims"][0]["evidence"] = []
    with pytest.raises(schemas.SchemaError):
        schemas.validate("node_05.output.json", payload)


def test_validation_error_lists_every_failure():
    """Step 3's repair retry hands the model the full error text, so it must be complete."""
    with pytest.raises(schemas.SchemaError) as excinfo:
        schemas.validate("node_0_5.output.json", {"direction": "sideways"})
    message = str(excinfo.value)
    assert "steps_moved" in message and "session_count" in message


@pytest.mark.parametrize(
    "field, bad_value",
    [("start", "NOT-A-DATE"), ("end", "2026-13-45")],
)
def test_malformed_dates_are_rejected(field, bad_value, valid_fixtures):
    """jsonschema ignores `format` unless a checker is wired in. Without this, a malformed
    timestamp would flow straight into the Node 3 time series and corrupt every figure
    derived from it."""
    payload = json.loads(json.dumps(valid_fixtures["node_00.output.json"]))
    payload["window"][field] = bad_value
    with pytest.raises(schemas.SchemaError):
        schemas.validate("node_00.output.json", payload)


def test_malformed_event_timestamp_is_rejected(valid_fixtures):
    payload = json.loads(json.dumps(valid_fixtures["node_00.output.json"]))
    payload["events"][0]["ts"] = "yesterday evening"
    with pytest.raises(schemas.SchemaError):
        schemas.validate("node_00.output.json", payload)


def test_malformed_citation_uri_is_rejected(valid_fixtures):
    """Metric 3 scores citation validity; a non-URI must not reach the validator as one."""
    payload = json.loads(json.dumps(valid_fixtures["node_06.output.json"]))
    payload["suggestions"][0]["source_url"] = "not a url"
    with pytest.raises(schemas.SchemaError):
        schemas.validate("node_06.output.json", payload)
