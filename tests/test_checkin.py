"""Step 5 acceptance: the check-in and the extractor.

Two properties matter here, and both are checkable without a provider:

* the turn cap holds because the loop enforces it, not because the prompt asks nicely;
* an observation cannot enter the pipeline unless the user actually said it.

The second is the one worth having. Node 5 cites these quotes as evidence, so a quote the
model invented would be laundered into a sourced claim. Every fabrication route is tested
below: invented wording, the agent's own question echoed back, and a conversation with no
user turns at all.
"""

from __future__ import annotations

import json

import pytest

from eval import generator
from llm import client
from nodes import node_01, node_02
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
    for size, model_id in (
        ("small", "test/small-v1"),
        ("mid", "test/mid-v1"),
        ("large", "test/large-v1"),
    ):
        monkeypatch.setitem(config.MODEL_BY_SIZE, size, model_id)


@pytest.fixture
def run_id():
    return runlog.new_run_id()


@pytest.fixture(scope="module")
def cases():
    return generator.load_cases()


def _agent(text: str, done: bool = False) -> str:
    return json.dumps({"message": text, "done": done})


def _obs(claim: str, quote: str, modality: str = "hearing") -> dict:
    return {"claim": claim, "source_quote": quote, "modality": modality}


def _extraction(*observations: dict) -> str:
    return json.dumps({"observations": list(observations)})


CONVERSATION = [
    {"role": "agent", "text": "Anything felt off with your eyes or ears lately?"},
    {"role": "user", "text": "I asked my roommate to repeat something twice yesterday"},
    {"role": "agent", "text": "Was that in a quiet room or somewhere noisy?"},
    {"role": "user", "text": "Conversations in the kitchen felt harder to follow this week"},
]


# ------------------------------------------------------------------- Node 1: the turn cap


def test_two_questions_and_two_replies_make_a_full_transcript(pinned_models, run_id):
    transport = ScriptedTransport(_agent("How were your eyes and ears?"), _agent("Noisy room?"))
    slice_ = node_01.check_in(
        run_id=run_id,
        transport=transport,
        user_replies=["I missed the start of sentences", "Worse at dinner"],
    )

    assert [t["role"] for t in slice_["conversation"]] == ["agent", "user", "agent", "user"]
    assert slice_["conversation"][1]["text"] == "I missed the start of sentences"


def test_cap_holds_even_when_the_model_never_says_done(pinned_models, run_id):
    """The prompt's "never more than 2" is a request; this is the enforcement."""
    transport = ScriptedTransport(*[_agent(f"Question {i}", done=False) for i in range(6)])

    slice_ = node_01.check_in(
        run_id=run_id,
        transport=transport,
        user_replies=[f"reply {i}" for i in range(6)],
    )

    agent_turns = [t for t in slice_["conversation"] if t["role"] == "agent"]
    assert len(agent_turns) == node_01.MAX_TURNS == 2
    assert len(transport.calls) == 2, "the loop, not the model, decides when to stop"


def test_done_on_the_first_turn_keeps_the_answer_it_already_had(pinned_models, run_id):
    transport = ScriptedTransport(_agent("Anything off lately?", done=True))

    slice_ = node_01.check_in(
        run_id=run_id, transport=transport, user_replies=["Everything felt normal"]
    )

    assert [t["role"] for t in slice_["conversation"]] == ["agent", "user"]
    assert slice_["conversation"][-1]["text"] == "Everything felt normal"


def test_a_silent_user_yields_the_opening_question_alone(pinned_models, run_id):
    """sparse_02's shape: the agent asked, nobody answered."""
    transport = ScriptedTransport(_agent("How have your eyes and ears felt?"))

    slice_ = node_01.check_in(run_id=run_id, transport=transport, user_replies=[])

    assert slice_["conversation"] == [
        {"role": "agent", "text": "How have your eyes and ears felt?"}
    ]
    assert len(transport.calls) == 1


def test_the_transcript_is_a_valid_node_02_input(pinned_models, run_id):
    transport = ScriptedTransport(_agent("First?"), _agent("Second?"))
    slice_ = node_01.check_in(run_id=run_id, transport=transport, user_replies=["a", "b"])
    schemas.validate("node_02.input.json", slice_)


def test_the_first_turn_carries_no_user_reply(pinned_models, run_id):
    transport = ScriptedTransport(_agent("Opening question?"))
    node_01.check_in(run_id=run_id, transport=transport, user_replies=[])

    sent = json.loads(transport.calls[0]["messages"][-1]["content"])
    assert sent == {"user_reply": None, "turn": 1}


def test_the_follow_up_receives_what_the_user_actually_said(pinned_models, run_id):
    transport = ScriptedTransport(_agent("Opening?"), _agent("Follow up?"))
    node_01.check_in(run_id=run_id, transport=transport, user_replies=["I leaned in to hear"])

    sent = json.loads(transport.calls[1]["messages"][-1]["content"])
    assert sent == {"user_reply": "I leaned in to hear", "turn": 2}


def test_both_turns_are_routed_identically(pinned_models, run_id):
    transport = ScriptedTransport(_agent("One?"), _agent("Two?"))
    node_01.check_in(run_id=run_id, transport=transport, user_replies=["a", "b"])

    models = {c["model"] for c in transport.calls}
    temps = {c["temperature"] for c in transport.calls}
    assert len(models) == 1 and len(temps) == 1, "two turns of one node, not two nodes"


def test_each_turn_is_logged_under_its_own_key(pinned_models, run_id):
    """Both turns under one name would make runlog.load_call ambiguous."""
    transport = ScriptedTransport(_agent("One?"), _agent("Two?"))
    node_01.check_in(run_id=run_id, transport=transport, user_replies=["a", "b"])

    assert runlog.load_call(run_id, "node_01.turn1").input_payload["turn"] == 1
    assert runlog.load_call(run_id, "node_01.turn2").input_payload["turn"] == 2


def test_turn_out_of_range_is_rejected_before_the_call(pinned_models, run_id):
    """Defence in depth: the enum means even a wrong MAX_TURNS cannot put turn 3 on the wire.

    Raising MAX_TURNS to 6 fails here rather than silently producing a six-question
    interrogation, so the cap survives an edit that only looks at the loop.
    """
    transport = ScriptedTransport(_agent("Third question?"))
    with pytest.raises(schemas.SchemaError):
        node_01.run("a reply", 3, run_id=run_id, transport=transport)
    assert transport.calls == [], "our bug, so it costs no provider call"


# ------------------------------------------------------- Node 2: quotes must be the user's


def test_a_quote_the_user_really_said_passes():
    node_02.verify_source_quotes(
        CONVERSATION, [_obs("asked for repetition", "asked my roommate to repeat something")]
    )


def test_an_invented_quote_is_rejected():
    with pytest.raises(node_02.SourceQuoteError, match="does not appear"):
        node_02.verify_source_quotes(
            CONVERSATION, [_obs("ringing in the ears", "my ears were ringing all week")]
        )


def test_the_agents_own_question_is_not_evidence():
    """A leading question answered by quoting it looks sourced and rests on nothing."""
    with pytest.raises(node_02.SourceQuoteError, match="agent's own question"):
        node_02.verify_source_quotes(
            CONVERSATION, [_obs("trouble in noisy rooms", "somewhere noisy")]
        )


def test_no_user_turns_means_no_observation_can_be_sourced():
    agent_only = [{"role": "agent", "text": "How have your eyes and ears felt?"}]
    with pytest.raises(node_02.SourceQuoteError):
        node_02.verify_source_quotes(agent_only, [_obs("hearing trouble", "eyes and ears")])


def test_an_empty_observation_list_is_always_acceptable():
    node_02.verify_source_quotes(CONVERSATION, [])
    node_02.verify_source_quotes([], [])


@pytest.mark.parametrize(
    "quote",
    [
        "I ASKED MY ROOMMATE TO REPEAT SOMETHING",
        "I asked my  roommate   to repeat something",
        "  I asked my roommate to repeat something  ",
        "I asked my roommate\nto repeat something",
    ],
    ids=["case", "inner-space", "outer-space", "newline"],
)
def test_only_case_and_whitespace_are_normalised(quote):
    node_02.verify_source_quotes(CONVERSATION, [_obs("asked for repetition", quote)])


def test_normalisation_does_not_admit_words_the_user_never_used():
    """The looseness above must not become a paraphrase licence."""
    with pytest.raises(node_02.SourceQuoteError):
        node_02.verify_source_quotes(
            CONVERSATION, [_obs("asked for repetition", "asked my flatmate to repeat something")]
        )


def test_the_failing_index_and_quote_are_named():
    good = _obs("asked for repetition", "asked my roommate to repeat")
    bad = _obs("invented", "I could not hear the television")
    with pytest.raises(node_02.SourceQuoteError) as excinfo:
        node_02.verify_source_quotes(CONVERSATION, [good, bad])

    message = str(excinfo.value)
    assert "observations[1]" in message
    assert "could not hear the television" in message


# --------------------------------------------------------- Node 2: the repair loop


def test_a_fabricated_quote_costs_a_repair_and_is_recovered(pinned_models, run_id):
    transport = ScriptedTransport(
        _extraction(_obs("ringing ears", "my ears rang constantly")),
        _extraction(_obs("asked for repetition", "asked my roommate to repeat something")),
    )

    result = node_02.run(CONVERSATION, run_id=run_id, transport=transport)

    assert len(result["observations"]) == 1
    calls = runlog.load_calls(run_id, "node_02")
    assert len(calls) == 2
    assert "SourceQuoteError" in calls[0].error
    assert calls[1].error is None


def test_the_repair_prompt_names_the_offending_quote(pinned_models, run_id):
    transport = ScriptedTransport(
        _extraction(_obs("ringing ears", "my ears rang constantly")),
        _extraction(),
    )
    node_02.run(CONVERSATION, run_id=run_id, transport=transport)

    repair = transport.calls[1]["messages"][-1]["content"]
    assert "my ears rang constantly" in repair


def test_an_unrepairable_fabrication_raises_rather_than_passing_it_on(pinned_models, run_id):
    transport = ScriptedTransport(
        _extraction(_obs("ringing ears", "my ears rang constantly")),
        _extraction(_obs("ringing ears", "my ears rang constantly")),
    )

    with pytest.raises(client.LLMError):
        node_02.run(CONVERSATION, run_id=run_id, transport=transport)


def test_the_rejected_attempt_survives_in_the_log(pinned_models, run_id):
    """Metric 2 and the iteration log are assembled from these, not from memory."""
    transport = ScriptedTransport(
        _extraction(_obs("ringing ears", "my ears rang constantly")),
        _extraction(),
    )
    node_02.run(CONVERSATION, run_id=run_id, transport=transport)

    first = runlog.load_calls(run_id, "node_02")[0]
    assert "my ears rang constantly" in json.dumps(first.raw_output)


def test_valid_extraction_costs_exactly_one_call(pinned_models, run_id):
    transport = ScriptedTransport(
        _extraction(_obs("asked for repetition", "asked my roommate to repeat something"))
    )
    node_02.run(CONVERSATION, run_id=run_id, transport=transport)
    assert len(transport.calls) == 1


def test_extraction_runs_at_temperature_zero(pinned_models, run_id):
    """Metric 4 scores run-to-run stability; sampling here would undercut it."""
    transport = ScriptedTransport(_extraction())
    node_02.run(CONVERSATION, run_id=run_id, transport=transport)
    assert transport.calls[0]["temperature"] == 0.0


# ------------------------------------------------------------------ prompt v2 and contract


def test_v2_withdrew_the_paraphrase_licence():
    """v1 invited exactly the output verify_source_quotes must reject."""
    assert "paraphrase" in prompts.load_prompt("node_02", "v1")
    v2 = prompts.load_prompt("node_02", "v2")
    assert "verbatim" in v2
    assert "Never paraphrase" in v2


def test_node_02_is_routed_to_the_repaired_prompt():
    assert config.get_node_config("node_02").prompt_version == "v2"


def test_both_prompt_versions_stay_on_disk():
    assert prompts.prompt_versions("node_02") == ["v1", "v2"]


# ----------------------------------------------------------------- against the eval set


def test_every_case_journal_is_a_valid_extractor_input(cases):
    for case in cases:
        schemas.validate("node_02.input.json", case.journal_slice)


def test_quoting_any_user_line_verbatim_passes_on_every_case(cases):
    """The validator must not reject honest extraction anywhere in the eval set."""
    for case in cases:
        user_turns = [t for t in case.journal_slice["conversation"] if t["role"] == "user"]
        for turn in user_turns:
            node_02.verify_source_quotes(
                case.journal_slice["conversation"], [_obs("reported", turn["text"])]
            )


def test_the_sparse_cases_leave_almost_nothing_to_quote(cases):
    """Metric 2 depends on these staying thin; a fatter journal would hide the effect."""
    by_id = {c.case_id: c for c in cases}
    for case_id in ("sparse_01", "sparse_02"):
        user_turns = [
            t for t in by_id[case_id].journal_slice["conversation"] if t["role"] == "user"
        ]
        assert len(user_turns) <= 1
        assert by_id[case_id].expectations.expect_insufficient_cause_evidence


def test_no_case_journal_can_source_a_symptom_the_user_did_not_report(cases):
    """The null journals say nothing is wrong; symptom vocabulary must not validate."""
    by_id = {c.case_id: c for c in cases}
    for case_id in ("null_01", "null_02"):
        conversation = by_id[case_id].journal_slice["conversation"]
        for invented in ("harder to follow", "repeat", "squinting", "leaning in"):
            with pytest.raises(node_02.SourceQuoteError):
                node_02.verify_source_quotes(conversation, [_obs("invented", invented)])
