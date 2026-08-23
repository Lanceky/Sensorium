"""Step 6 acceptance: the two Node 4 agents are blind to each other's evidence.

The claim in `context.md` §4 is that Agent A reasons from statistics it cannot attach a
story to, and Agent B from a story it cannot attach numbers to, so their agreement carries
information. That only means something if the blindness is real, and it is only *shown* to
be real if the test could have caught it being false.

So the tests here fall into three groups, and the middle group is the one that matters:

* the preserved twelve-case run is blind, checked offline against `evidence/`;
* the detector fails when it should — deliberate leaks in either direction, a broken
  window reconstruction, an empty fingerprint list;
* the fingerprints themselves are well formed, including a regression test for the
  degenerate one-character terms that made the first version of this report cry leak on
  all twelve cases at once.

Nothing here touches a provider. The live run happened once, and its request logs are
committed, so the proof is re-verified on every test run instead of resting on a claim
about something that happened on my machine in August.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from eval import independence
from nodes import node_04
from sensorium import runlog, schemas

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence" / "node_04-blind-agents"

#: Cases whose journal produced no Node 2 observations, so Agent B's payload holds no
#: journal text and the journal-side positive control cannot fire. Recording them by name
#: keeps the exemption honest: if a case ever silently joins this list the count assertion
#: in `test_the_journal_control_fires_on_every_case_that_has_observations` fails.
NO_OBSERVATION_CASES = {
    "conflict_01",
    "conflict_02",
    "conflict_03",
    "null_01",
    "null_02",
    "sparse_02",
}


@pytest.fixture(scope="module")
def preserved() -> dict:
    return json.loads((EVIDENCE / "cases.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def evidence_runs(monkeypatch):
    """Read the committed request logs instead of whatever is in `runs/` today."""
    monkeypatch.setattr(runlog, "RUNS_DIR", EVIDENCE / "runs")


@pytest.fixture
def leaky_run(tmp_path, monkeypatch, preserved):
    """Copy a preserved run into a scratch directory so a payload can be poisoned."""

    def _make(case_id: str, node: str, poison: object) -> str:
        case = preserved[case_id]
        run_id = case["run_id"]
        runs = tmp_path / "runs"
        shutil.copytree(EVIDENCE / "runs" / run_id, runs / run_id)
        path = runs / run_id / "calls.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            if row["node"] == node:
                row["input_payload"] = poison
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        monkeypatch.setattr(runlog, "RUNS_DIR", runs)
        return run_id

    return _make


def _journal(preserved: dict, case_id: str) -> list[dict[str, str]]:
    from eval import generator

    for case in generator.load_cases():
        if case.case_id == case_id:
            return case.journal_slice["conversation"]
    raise AssertionError(f"unknown case {case_id}")


# --------------------------------------------------------------------------------------
# The preserved run
# --------------------------------------------------------------------------------------


def test_every_preserved_case_is_blind(preserved):
    for case_id, case in sorted(preserved.items()):
        report = independence.assert_blind(
            case["run_id"], _journal(preserved, case_id), case["trend"]
        )
        assert report.ok, report.summary()


def test_the_preserved_run_covers_the_whole_eval_set(preserved):
    from eval import generator

    assert set(preserved) == {case.case_id for case in generator.load_cases()}


def test_the_trend_control_fires_on_every_case(preserved):
    """Agent A must demonstrably contain the terms Agent B is checked against."""
    for case_id, case in sorted(preserved.items()):
        report = independence.assert_blind(
            case["run_id"], _journal(preserved, case_id), case["trend"]
        )
        assert "trend_term_found_in_a" in report.controls_fired, case_id


def test_the_journal_control_fires_on_every_case_that_has_observations(preserved):
    """The journal side fires exactly where Node 2 gave Agent B something to hold.

    This is the conditional control, and the condition is not a convenience: it is
    predicted independently by the Node 2 observation counts, so the two measurements
    corroborate each other rather than one excusing the other.
    """
    fired = set()
    for case_id, case in sorted(preserved.items()):
        report = independence.assert_blind(
            case["run_id"], _journal(preserved, case_id), case["trend"]
        )
        has_observations = bool(case["observations"]["observations"])
        assert has_observations is (case_id not in NO_OBSERVATION_CASES), case_id
        assert ("journal_phrase_found_in_b" in report.controls_fired) is has_observations, case_id
        if has_observations:
            fired.add(case_id)
    assert fired, "the journal-side control never fired anywhere; it proves nothing"


# --------------------------------------------------------------------------------------
# The detector fails when it should
# --------------------------------------------------------------------------------------


def test_journal_text_reaching_agent_a_is_caught(preserved, leaky_run):
    """The leak this whole module exists to catch.

    The poisoned payload keeps Agent A's real trend data and adds journal text alongside
    it, which is what an actual plumbing mistake looks like — a slice appended to the
    wrong dict, not a slice swapped for another. It also keeps the positive control
    firing, so the failure raised is unambiguously the leak.
    """
    case_id = "agree_01"
    conversation = _journal(preserved, case_id)
    user_line = next(t["text"] for t in conversation if t["role"] == "user")
    real = runlog.load_call(preserved[case_id]["run_id"], node_04.AGENT_A).input_payload
    run_id = leaky_run(case_id, node_04.AGENT_A, {**real, "notes": user_line})

    with pytest.raises(independence.IndependenceError, match="saw journal text"):
        independence.assert_blind(run_id, conversation, preserved[case_id]["trend"])


def test_trend_figures_reaching_agent_b_are_caught(preserved, leaky_run):
    case_id = "agree_01"
    trend = preserved[case_id]["trend"]
    run_id = leaky_run(case_id, node_04.AGENT_B, {"observations": [], "leaked": trend})

    with pytest.raises(independence.IndependenceError, match="saw trend figure"):
        independence.assert_blind(run_id, _journal(preserved, case_id), trend)


def test_an_empty_fingerprint_list_is_refused_rather_than_passed(preserved):
    """The vacuity failure the module docstring is about.

    A blindness check over zero terms finds zero leaks and reports success. Left to a
    caller to notice, that is a test which passes brightly while looking at nothing.
    """
    case = preserved["agree_01"]
    with pytest.raises(independence.IndependenceError, match="nothing was proved"):
        independence.assert_blind(
            case["run_id"], _journal(preserved, "agree_01"), {"figures": {}}
        )


def test_a_broken_window_reconstruction_is_refused(preserved, monkeypatch):
    """If the reconstructed window is not what the model saw, absence means nothing."""
    case = preserved["agree_01"]
    monkeypatch.setattr(independence, "context_window", lambda call: "")

    with pytest.raises(independence.IndependenceError, match="positive control failed"):
        independence.assert_blind(
            case["run_id"], _journal(preserved, "agree_01"), case["trend"]
        )


def test_the_search_covers_the_system_prompt_not_only_the_payload(preserved):
    """Checking `input_payload` alone would skip most of what the agent was told.

    The window is rebuilt from the prompt file and the generated contract as well, so a
    leak planted in either would still be found. Asserting the window is a strict superset
    of the payload is what stops a future refactor quietly narrowing the search.
    """
    case = preserved["agree_01"]
    call = runlog.load_call(case["run_id"], node_04.AGENT_A)
    window = independence.context_window(call)

    assert independence._normalise(call.payload_text()) not in ("", window)
    assert len(window) > len(independence._normalise(json.dumps(call.input_payload)))
    assert "you are" in window or "agent" in window


# --------------------------------------------------------------------------------------
# The two halves of the guarantee, and what each one misses
# --------------------------------------------------------------------------------------


def test_the_schema_rejects_the_other_slice_as_a_new_field():
    """The accidental route: a slice appended to the wrong dict never reaches a model."""
    trend = {
        "figures": {"volume_pct_change": {"value": 1.5, "unit": "%", "method": "percent_change"}},
        "window_weeks": 3,
        "sufficient_data": True,
    }
    schemas.validate("node_04a.input.json", {"trend_data": trend})

    with pytest.raises(schemas.SchemaError, match="Additional properties"):
        schemas.validate(
            "node_04a.input.json",
            {"trend_data": trend, "journal": "I keep missing what people say"},
        )


def test_the_schema_cannot_catch_text_hidden_in_a_free_form_field():
    """The residual route, and the reason the runtime check is not redundant.

    Every figure carries a `unit` of type string, so a journal sentence sitting in a unit
    is a schema-valid payload. This test asserts the gap exists rather than assuming it
    does not — if a future schema closes it, this fails and the docstring in
    `nodes/node_04.py` needs revising with it.
    """
    smuggled = {
        "figures": {
            "volume_pct_change": {
                "value": 1.5,
                "unit": "I keep missing what people say",
                "method": "percent_change",
            }
        },
        "window_weeks": 3,
        "sufficient_data": True,
    }

    schemas.validate("node_04a.input.json", {"trend_data": smuggled})


def test_the_runtime_check_catches_what_the_schema_lets_through(preserved, leaky_run):
    """The same schema-valid smuggled payload, caught by reading the real context window."""
    case_id = "agree_01"
    conversation = _journal(preserved, case_id)
    user_line = next(t["text"] for t in conversation if t["role"] == "user")
    real = runlog.load_call(preserved[case_id]["run_id"], node_04.AGENT_A).input_payload

    smuggled = json.loads(json.dumps(real))
    figure = next(iter(smuggled["trend_data"]["figures"].values()))
    figure["unit"] = user_line
    schemas.validate("node_04a.input.json", smuggled)

    run_id = leaky_run(case_id, node_04.AGENT_A, smuggled)
    with pytest.raises(independence.IndependenceError, match="saw journal text"):
        independence.assert_blind(run_id, conversation, preserved[case_id]["trend"])


# --------------------------------------------------------------------------------------
# The fingerprints themselves
# --------------------------------------------------------------------------------------


def test_trend_terms_skips_integral_values(preserved):
    """Regression: the degenerate terms that faked a leak on all twelve cases.

    `caption_on_rate` of 1.0 rounded to the term "1", which matched `"minLength": 1` in
    the schema contract every agent is sent. Every case reported a leak, and every one of
    those reports was wrong.
    """
    terms = independence.trend_terms(
        {"figures": {"caption_on_rate": {"value": 1.0}, "volume_pct_change": {"value": 38.458}}}
    )

    assert "caption_on_rate" in terms
    assert "volume_pct_change" in terms
    assert "38.458" in terms
    assert "1" not in terms and "1.0" not in terms
    assert all(len(t) > 2 for t in terms)


def test_no_preserved_case_yields_a_degenerate_term(preserved):
    for case_id, case in sorted(preserved.items()):
        for term in independence.trend_terms(case["trend"]):
            assert len(term) > 2, f"{case_id} produced the near-worthless term {term!r}"


def test_identifiers_and_decimals_survive_normalisation():
    """A split identifier searches for fragments that match everywhere."""
    text = independence._normalise("Volume_pct_change rose 38.458 percent.")

    assert "volume_pct_change" in text.split()
    assert "38.458" in text.split()


def test_a_figure_is_not_matched_inside_a_longer_number():
    assert independence._contains_term("value 10.52 here", "0.5") is False
    assert independence._contains_term("value 0.5 here", "0.5") is True


def test_phrases_come_only_from_user_turns():
    """An agent question echoed back is not evidence of a journal leak."""
    conversation = [
        {"role": "agent", "text": "how has your hearing been lately"},
        {"role": "user", "text": "I keep missing what people say"},
    ]
    phrases = independence.journal_phrases(conversation)

    assert "i keep missing" in phrases
    assert not any("how has your" in p for p in phrases)


def test_phrases_are_multi_word():
    """Single tokens would flag `week` against `window_weeks` as an evidence leak."""
    phrases = independence.journal_phrases(
        [{"role": "user", "text": "this week was harder than usual"}]
    )

    assert phrases
    assert all(len(p.split()) == independence.PHRASE_WORDS for p in phrases)


def test_a_journal_with_no_user_turns_yields_no_phrases():
    """sparse_02 is this case, and it is why the journal control is conditional."""
    assert independence.journal_phrases([{"role": "agent", "text": "anything to add?"}]) == ()
