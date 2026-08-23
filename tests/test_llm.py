"""Step 3 acceptance: the calling convention around the provider.

Everything here runs on a scripted transport. The point of the layer is not that HTTP
works, it is that a node cannot use the wrong model, cannot skip validation, and cannot
lose the evidence of a rejected attempt — all of which are checkable offline.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm import client
from sensorium import config, prompts, runlog, schemas


class ScriptedTransport:
    """Replays canned replies and records exactly how it was called."""

    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, *, model, messages, temperature, seed=None):
        self.calls.append(
            {"model": model, "messages": messages, "temperature": temperature, "seed": seed}
        )
        if not self.replies:
            raise AssertionError("transport called more often than the test scripted")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(autouse=True)
def isolated_runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", tmp_path / "runs")


@pytest.fixture
def pinned_models(monkeypatch):
    """Stand-in ids so routing is testable before the live account is queried."""
    for size, model_id in (
        ("small", "test/small-v1"),
        ("mid", "test/mid-v1"),
        ("large", "test/large-v1"),
    ):
        monkeypatch.setitem(config.MODEL_BY_SIZE, size, model_id)


@pytest.fixture
def run_id():
    return runlog.new_run_id()


VALID_01 = '{"message": "Was that in a quiet room?", "done": false}'
INVALID_01 = '{"message": "Was that in a quiet room?"}'  # missing required "done"


def _payload_01():
    return {"user_reply": "I asked my roommate to repeat something twice", "turn": 2}


# --------------------------------------------------------------------------- extraction


@pytest.mark.parametrize(
    "raw",
    [
        '{"done": true, "message": "hi"}',
        '```json\n{"done": true, "message": "hi"}\n```',
        '```\n{"done": true, "message": "hi"}\n```',
        'Sure! Here is the JSON:\n{"done": true, "message": "hi"}\nHope that helps.',
    ],
    ids=["bare", "fenced-json", "fenced-plain", "chatty"],
)
def test_extract_json_recovers_the_object(raw):
    assert client.extract_json(raw) == {"done": True, "message": "hi"}


def test_extract_json_handles_top_level_arrays():
    assert client.extract_json("```json\n[1, 2, 3]\n```") == [1, 2, 3]


@pytest.mark.parametrize("raw", ["", "   ", "I'm sorry, I can't help with that."])
def test_extract_json_rejects_output_with_no_json(raw):
    with pytest.raises(client.MalformedOutputError):
        client.extract_json(raw)


def test_extract_json_prefers_the_fenced_block_over_surrounding_prose():
    """A model that reasons in prose before answering must not have its reasoning parsed
    as the answer."""
    raw = 'I considered {"wrong": 1} first.\n```json\n{"done": true, "message": "hi"}\n```'
    assert client.extract_json(raw) == {"done": True, "message": "hi"}


# ------------------------------------------------------------------------ schema wiring


def test_node_04_agents_share_one_output_schema():
    """Identical output shape is what makes the two agents comparable; if they could
    differ, disagreement would be a formatting artefact rather than a signal."""
    assert client.output_schema_for("node_04a") == "node_04.output.json"
    assert client.output_schema_for("node_04b") == "node_04.output.json"


def test_every_registered_llm_node_has_both_contracts():
    for node in config.REGISTRY:
        assert client.input_schema_for(node) is not None, node
        assert client.output_schema_for(node) in schemas.schema_names()


def test_nodes_without_structured_input_report_none():
    assert client.input_schema_for("node_00") is None


def test_unknown_node_has_no_output_schema():
    with pytest.raises(client.LLMError, match="no output schema"):
        client.output_schema_for("node_99")


# --------------------------------------------------------------------------- happy path


def test_call_node_returns_validated_output(pinned_models, run_id):
    transport = ScriptedTransport(VALID_01)
    result = client.call_node("node_01", _payload_01(), run_id=run_id, transport=transport)
    assert result == {"message": "Was that in a quiet room?", "done": False}


def test_call_node_routes_the_model_and_temperature_from_the_registry(pinned_models, run_id):
    transport = ScriptedTransport(VALID_01)
    client.call_node("node_01", _payload_01(), run_id=run_id, transport=transport)

    cfg = config.get_node_config("node_01")
    (call,) = transport.calls
    assert call["model"] == config.MODEL_BY_SIZE[cfg.size]
    assert call["temperature"] == cfg.temperature


def test_the_system_prompt_opens_with_the_versioned_file(pinned_models, run_id):
    """The design-doc wording must lead, verbatim and unedited.

    The generated output contract is appended after it, so this is a prefix check rather
    than equality: it still fails if a single character of the versioned prompt changes,
    and additionally pins the order, since a schema dump ahead of the persona would bury
    the instructions that shape the reply.
    """
    from sensorium import prompts

    transport = ScriptedTransport(VALID_01)
    client.call_node("node_01", _payload_01(), run_id=run_id, transport=transport)

    system = transport.calls[0]["messages"][0]
    assert system["role"] == "system"
    assert system["content"].startswith(prompts.load_prompt("node_01", "v1"))


def test_the_run_log_records_the_routing_actually_used(pinned_models, run_id):
    transport = ScriptedTransport(VALID_01)
    client.call_node("node_01", _payload_01(), run_id=run_id, transport=transport)

    call = runlog.load_call(run_id, "node_01")
    assert call.model == "test/small-v1"
    assert call.prompt_version == "v1"
    assert call.error is None and call.attempt == 1
    assert call.input_payload == _payload_01()


def test_blind_agents_are_called_with_identical_routing(pinned_models, run_id, valid_fixtures):
    """The registry invariant, verified where it actually matters: at the call."""
    transport = ScriptedTransport(
        json.dumps(valid_fixtures["node_04.output.json"]),
        json.dumps(valid_fixtures["node_04.output.json"]),
    )
    a, b = config.BLIND_AGENT_PAIR
    client.call_node(a, valid_fixtures[f"{a}.input.json"], run_id=run_id, transport=transport)
    client.call_node(b, valid_fixtures[f"{b}.input.json"], run_id=run_id, transport=transport)

    first, second = transport.calls
    assert first["model"] == second["model"]
    assert first["temperature"] == second["temperature"]


# ------------------------------------------------------------------------------- repair


def test_schema_failure_is_repaired_on_the_second_attempt(pinned_models, run_id):
    transport = ScriptedTransport(INVALID_01, VALID_01)
    result = client.call_node("node_01", _payload_01(), run_id=run_id, transport=transport)
    assert result["done"] is False
    assert len(transport.calls) == 2


def test_the_repair_prompt_carries_the_validation_error(pinned_models, run_id):
    transport = ScriptedTransport(INVALID_01, VALID_01)
    client.call_node("node_01", _payload_01(), run_id=run_id, transport=transport)

    repair_messages = transport.calls[1]["messages"]
    assert repair_messages[-2] == {"role": "assistant", "content": INVALID_01}
    complaint = repair_messages[-1]["content"]
    assert complaint.startswith("Your previous reply failed validation:")
    assert "'done' is a required property" in complaint


def test_the_rejected_attempt_survives_in_the_run_log(pinned_models, run_id):
    """The iteration log is assembled from real failures. A repaired attempt logged as a
    success would erase the evidence that the prompt needed fixing."""
    transport = ScriptedTransport(INVALID_01, VALID_01)
    client.call_node("node_01", _payload_01(), run_id=run_id, transport=transport)

    first, second = runlog.load_calls(run_id, "node_01")
    assert (first.attempt, second.attempt) == (1, 2)
    assert first.raw_output == INVALID_01
    assert "SchemaError" in first.error
    assert second.error is None


def test_load_call_returns_the_attempt_that_flowed_downstream(pinned_models, run_id):
    transport = ScriptedTransport(INVALID_01, VALID_01)
    client.call_node("node_01", _payload_01(), run_id=run_id, transport=transport)

    call = runlog.load_call(run_id, "node_01")
    assert call.attempt == 2 and call.error is None


def test_unparseable_output_is_repaired_too(pinned_models, run_id):
    transport = ScriptedTransport("I'd rather explain it in words.", VALID_01)
    result = client.call_node("node_01", _payload_01(), run_id=run_id, transport=transport)
    assert result["done"] is False
    assert "MalformedOutputError" in runlog.load_calls(run_id, "node_01")[0].error


def test_exhausted_repairs_raise_and_log_every_attempt(pinned_models, run_id):
    transport = ScriptedTransport(INVALID_01, INVALID_01)
    with pytest.raises(client.LLMError, match="no contract-valid output after 2 attempts"):
        client.call_node("node_01", _payload_01(), run_id=run_id, transport=transport)

    attempts = runlog.load_calls(run_id, "node_01")
    assert [c.attempt for c in attempts] == [1, 2]
    assert all(c.error for c in attempts)


def test_repair_budget_is_configurable(pinned_models, run_id):
    transport = ScriptedTransport(INVALID_01, INVALID_01, VALID_01)
    result = client.call_node(
        "node_01", _payload_01(), run_id=run_id, transport=transport, max_repairs=2
    )
    assert result["done"] is False
    assert len(runlog.load_calls(run_id, "node_01")) == 3


# ------------------------------------------------------------------------------ failure


def test_transport_failure_is_not_repaired(pinned_models, run_id):
    """Re-prompting a provider outage would fill the iteration log with failures the
    prompt never caused."""
    transport = ScriptedTransport(client.TransportError("503 upstream"), VALID_01)
    with pytest.raises(client.TransportError, match="503 upstream"):
        client.call_node("node_01", _payload_01(), run_id=run_id, transport=transport)

    assert len(transport.calls) == 1
    (call,) = runlog.load_calls(run_id, "node_01")
    assert "503 upstream" in call.error


def test_a_bad_payload_fails_before_the_provider_is_touched(pinned_models, run_id):
    """Our own malformed input is a bug in this repo, not something a model can repair."""
    transport = ScriptedTransport(VALID_01)
    with pytest.raises(schemas.SchemaError):
        client.call_node("node_01", {"turn": "second"}, run_id=run_id, transport=transport)

    assert transport.calls == []
    assert runlog.list_runs() == []


def test_an_unpinned_model_refuses_to_run(monkeypatch, run_id):
    """A node must never silently fall back to a default model.

    The registry is populated now, so the empty state is set up explicitly rather than
    relied upon; the guard has to keep working for any size added later.
    """
    monkeypatch.setitem(config.MODEL_BY_SIZE, "small", "")
    transport = ScriptedTransport(VALID_01)
    with pytest.raises(config.ConfigError, match="not pinned yet"):
        client.call_node("node_01", _payload_01(), run_id=run_id, transport=transport)
    assert transport.calls == []


# --------------------------------------------------------------------------------- seed


def test_seed_is_forwarded_for_reproducibility(pinned_models, run_id):
    transport = ScriptedTransport(VALID_01)
    client.call_node("node_01", _payload_01(), run_id=run_id, transport=transport, seed=99)
    assert transport.calls[0]["seed"] == 99


def _fake_openai_chat(recorder, reject_seed):
    class Completions:
        def create(self, **kwargs):
            recorder.append(kwargs)
            if reject_seed and "seed" in kwargs:
                raise RuntimeError("unsupported parameter: seed")
            message = SimpleNamespace(content='{"ok": true}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    return SimpleNamespace(completions=Completions())


def test_featherless_degrades_gracefully_when_seed_is_unsupported(monkeypatch):
    transport = client.FeatherlessTransport(api_key="test-key")
    sent: list[dict] = []
    monkeypatch.setattr(transport._client, "chat", _fake_openai_chat(sent, reject_seed=True))

    out = transport.complete(model="m", messages=[], temperature=0.0, seed=7)

    assert out == '{"ok": true}'
    assert "seed" in sent[0] and "seed" not in sent[1]
    assert transport._seed_supported is False


def test_featherless_wraps_provider_errors(monkeypatch):
    transport = client.FeatherlessTransport(api_key="test-key")

    class Completions:
        def create(self, **kwargs):
            raise RuntimeError("connection reset")

    monkeypatch.setattr(
        transport._client, "chat", SimpleNamespace(completions=Completions())
    )
    with pytest.raises(client.TransportError, match="connection reset"):
        transport.complete(model="m", messages=[], temperature=0.0)


def test_featherless_rejects_an_empty_completion(monkeypatch):
    transport = client.FeatherlessTransport(api_key="test-key")

    class Completions:
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))])

    monkeypatch.setattr(
        transport._client, "chat", SimpleNamespace(completions=Completions())
    )
    with pytest.raises(client.TransportError, match="empty completion"):
        transport.complete(model="m", messages=[], temperature=0.0)


def test_featherless_targets_the_configured_base_url():
    transport = client.FeatherlessTransport(api_key="test-key")
    assert transport.base_url == config.FEATHERLESS_BASE_URL == "https://api.featherless.ai/v1"


# ---------------------------------------------------- the generated output contract


def test_the_system_prompt_carries_the_output_schema(pinned_models, run_id):
    """Without this the first live call returned prose: no prompt states its output shape."""
    transport = ScriptedTransport(VALID_01)
    client.call_node("node_01", _payload_01(), run_id=run_id, transport=transport)

    system = transport.calls[0]["messages"][0]["content"]
    assert '"message"' in system and '"done"' in system
    assert '"additionalProperties": false' in system


def test_the_verbatim_prompt_survives_alongside_the_contract(pinned_models, run_id):
    """The design-doc wording is the deliverable; the contract is appended, not merged in."""
    transport = ScriptedTransport(VALID_01)
    client.call_node("node_01", _payload_01(), run_id=run_id, transport=transport)

    system = transport.calls[0]["messages"][0]["content"]
    assert prompts.load_prompt("node_01", "v1") in system


def test_each_node_is_shown_its_own_contract(pinned_models, run_id):
    """A node shown the wrong schema would be corrected toward the wrong shape."""
    transport = ScriptedTransport('{"observations": []}')
    client.call_node(
        "node_02", {"conversation": []}, run_id=run_id, transport=transport
    )

    system = transport.calls[0]["messages"][0]["content"]
    assert '"observations"' in system
    assert '"done"' not in system


def test_the_blind_agents_receive_identical_contracts(pinned_models, run_id):
    """Node 4's agents share an output schema; differing contracts would confound them."""
    assert schemas.contract_text(
        client.output_schema_for("node_04a")
    ) == schemas.contract_text(client.output_schema_for("node_04b"))


def test_the_contract_names_no_schema_file(pinned_models, run_id):
    """Leaking $id invites the model to echo filenames back as keys."""
    transport = ScriptedTransport(VALID_01)
    client.call_node("node_01", _payload_01(), run_id=run_id, transport=transport)
    assert "node_01.output.json" not in transport.calls[0]["messages"][0]["content"]
