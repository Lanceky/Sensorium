"""The independence proof for Node 4.

`context.md` §4 claims the two agents are blind to each other's data. This module is what
turns that from a claim into a check: it reads the payloads that were *actually* sent
during a run and looks for the other slice's fingerprints.

The obvious version of this test is worthless:

    assert not any(term in payload_a for term in journal_terms)

If ``journal_terms`` comes back empty — wrong field name, empty extraction, a typo — that
line passes while proving nothing at all, and it passes *loudly*, in green. A blindness
test that can only fail by finding something is indistinguishable from one that never
looks.

So every check here is two-sided. The same terms that must be absent from one agent must
be **present in the other**, and :class:`BlindnessReport` records whether that positive
control actually fired. The claim is then "these fingerprints appear in exactly one place",
which is a measurement, rather than "we looked and saw nothing", which is a mood.

Two fingerprint families are used, both chosen because they are case-specific by
construction and cannot arise coincidentally:

* **figure keys and values** from Node 3 — ``volume_pct_change``, ``38.458``. A journal
  payload containing either did not get there by chance.
* **phrases from the user's own lines** — contiguous word runs long enough that
  co-occurrence is not plausible. Single tokens are deliberately not used for this: the
  journal says "this week" and Node 3 reports ``window_weeks``, and flagging that overlap
  as an evidence leak would be a false positive dressed up as rigour. That is not a
  hypothetical. Dropping to one-word phrases makes the suite report leaks on ``'week'``,
  and also on ``'i'``, ``'the'``, ``'me'`` and ``'in'``; the mutation test in
  ``tests/test_independence.py`` holds the three-word floor in place.

The text inspected is the whole reconstructed context window, system prompt included, not
just the payload dict. The agent sees everything in that window, so anything less would be
checking a convenient subset.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from llm import client
from nodes import node_04
from sensorium import prompts, runlog

#: Length of the word runs taken from user journal lines. Three words is already
#: implausible as coincidence between an experiential sentence and a statistics payload,
#: and short enough that a model paraphrasing around a quote still trips it.
PHRASE_WORDS = 3

#: Identifiers and decimals must survive normalisation intact: splitting
#: ``volume_pct_change`` into three words or ``38.458`` into two would leave the search
#: looking for fragments that match everywhere, which is how a leak test quietly stops
#: testing. Interior dots and underscores are kept; a trailing sentence period is not.
_WORD = re.compile(r"[a-z0-9_']+(?:\.[a-z0-9_']+)*")


class IndependenceError(AssertionError):
    """Raised when one agent's context window carries the other agent's evidence."""


@dataclass
class BlindnessReport:
    """What was searched for, and where it was and was not found."""

    run_id: str
    journal_phrases: tuple[str, ...] = ()
    trend_terms: tuple[str, ...] = ()
    leaks: list[str] = field(default_factory=list)
    controls_fired: set[str] = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return not self.leaks

    def summary(self) -> str:
        controls = ", ".join(sorted(self.controls_fired)) or "none"
        return (
            f"{self.run_id}: {len(self.journal_phrases)} journal phrases, "
            f"{len(self.trend_terms)} trend terms, "
            f"{len(self.leaks)} leaks, controls fired: {controls}"
        )


def _normalise(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def journal_phrases(conversation: Iterable[dict[str, str]]) -> tuple[str, ...]:
    """Word runs from the user's own lines, as they would survive quoting."""
    phrases: list[str] = []
    for turn in conversation:
        if turn["role"] != "user":
            continue
        words = _normalise(turn["text"]).split()
        for i in range(len(words) - PHRASE_WORDS + 1):
            phrases.append(" ".join(words[i : i + PHRASE_WORDS]))
    return tuple(dict.fromkeys(phrases))


def trend_terms(trend_data: dict[str, Any]) -> tuple[str, ...]:
    """Figure keys and their values — identifiers no journal payload could invent.

    Values that are exactly integral are skipped. The first version of this function also
    emitted rounded variants of every figure, which produced terms like ``"1"`` and
    ``"5"`` — and those duly "leaked" into Agent B on all twelve cases, by matching
    ``"minLength": 1`` in the schema contract and the phrase "node 5" in a schema
    description. Structural boilerplate, identical for every case, carrying no information
    about anyone. A one-character fingerprint is not a fingerprint.

    Rounded variants were solving the wrong problem here anyway. They guard against a model
    restating ``38.458`` as ``38.46`` in its *output*, which is Metric 1's job at Node 5.
    This function inspects an *input* window, where the engine's exact values appear
    verbatim or not at all. The phenomenon is real — in the preserved run Agent A wrote
    "38.46%" for a ``volume_pct_change`` of ``38.458``, and the exact value appears nowhere
    in its reply — but it is an output-fidelity question, not a blindness one.
    """
    terms: list[str] = []
    for key, figure in trend_data.get("figures", {}).items():
        terms.append(key)
        value = figure["value"]
        if isinstance(value, (int, float)) and float(value).is_integer():
            continue
        terms.append(str(value))
    return tuple(dict.fromkeys(terms))


def context_window(call: runlog.Call) -> str:
    """Rebuild everything the model saw for ``call``, system prompt included.

    Deterministic from the logged fields, so this reconstructs the real request rather
    than a summary of it: the run log records the node, the prompt version and the exact
    payload, and the client composes the message list from precisely those.
    """
    node = call.node.split(".")[0]
    system = prompts.load_prompt(node, call.prompt_version)
    contract = client.CONTRACT_INSTRUCTION.format(
        schema=client.schemas.contract_text(client.output_schema_for(node))
    )
    payload = json.dumps(call.input_payload, sort_keys=True, ensure_ascii=False)
    return _normalise(system + contract + payload)


def assert_blind(
    run_id: str,
    conversation: Iterable[dict[str, str]],
    trend_data: dict[str, Any],
) -> BlindnessReport:
    """Verify each Node 4 agent saw its own slice and only its own slice.

    Returns a :class:`BlindnessReport` on success so a caller can assert the checks were
    non-vacuous; raises :class:`IndependenceError` naming the leaked term on failure.
    """
    call_a = runlog.load_call(run_id, node_04.AGENT_A)
    call_b = runlog.load_call(run_id, node_04.AGENT_B)
    window_a = context_window(call_a)
    window_b = context_window(call_b)

    phrases = journal_phrases(conversation)
    terms = trend_terms(trend_data)
    report = BlindnessReport(run_id=run_id, journal_phrases=phrases, trend_terms=terms)

    # Node 3 emits figures for every case in the eval set, including the ones with no
    # journal at all, so an empty term list is never a legitimate state — it means the
    # extraction broke and the search below would sweep past, finding nothing, in green.
    # Refusing here rather than leaving it to the caller keeps the guarantee with the
    # function that makes the claim.
    if not terms:
        raise IndependenceError(
            f"{run_id}: no trend fingerprints were extracted, so nothing was proved. "
            "Check the shape of the Node 3 output passed in as trend_data."
        )

    for phrase in phrases:
        if phrase in window_a:
            report.leaks.append(
                f"{node_04.AGENT_A} saw journal text {phrase!r}; it must see only trends"
            )
        if phrase in window_b:
            report.controls_fired.add("journal_phrase_found_in_b")

    for term in terms:
        needle = _normalise(term)
        if not needle:
            continue
        if _contains_term(window_b, needle):
            report.leaks.append(
                f"{node_04.AGENT_B} saw trend figure {term!r}; it must see only the journal"
            )
        if _contains_term(window_a, needle):
            report.controls_fired.add("trend_term_found_in_a")

    # Leaks are reported before the control is judged. The positive control exists to
    # validate a *negative* result — "these terms were not in B" only means something if
    # the terms were findable at all. A leak that was actually found needs no such
    # validation, so it is the more useful thing to put in the error message.
    if report.leaks:
        raise IndependenceError("; ".join(report.leaks))

    # Agent A is handed the trend data, so its own figures are always in its window. If
    # this control is silent the window reconstruction is wrong, and every "absent from B"
    # result above was searching text the agent never actually received.
    #
    # Its counterpart, journal_phrase_found_in_b, is deliberately *not* required here: it
    # can only fire when Node 2 extracted at least one observation, and the conflict, null
    # and sparse_02 cases legitimately yield none. The suite asserts that control fires
    # across the eval set instead of on every case.
    if "trend_term_found_in_a" not in report.controls_fired:
        raise IndependenceError(
            f"{run_id}: positive control failed — none of the {len(terms)} trend "
            f"fingerprints were found in {node_04.AGENT_A}'s own window, so the absence "
            f"of those terms from {node_04.AGENT_B} demonstrates nothing."
        )

    return report


def _contains_term(window: str, term: str) -> bool:
    """Whole-token match, so ``0.5`` does not spuriously hit inside ``10.52``."""
    return re.search(rf"(?<![\w.]){re.escape(term)}(?![\w.])", window) is not None
