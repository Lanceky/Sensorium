"""Step 2 acceptance: the evaluation set is honest, reproducible, and truly independent.

The tests that matter most are the negative ones. It is easy to write a generator that
*looks* like it produces independent slices; these assert that the guarantees actually
fire when violated, and that the two slices carry genuinely different information.
"""

from __future__ import annotations

import inspect
import statistics
from datetime import date

import pytest

from eval import generator as gen
from sensorium import schemas


@pytest.fixture(scope="module")
def cases():
    return gen.build_all_cases()


@pytest.fixture(scope="module")
def by_id(cases):
    return {c.case_id: c for c in cases}


def _weekly_mean(case, signal):
    events = [e for e in case.device_slice["events"] if e["signal"] == signal]
    if not events:
        return []
    first = date.fromisoformat(min(e["ts"] for e in events)[:10])
    weeks: dict[int, list[float]] = {}
    for event in events:
        offset = (date.fromisoformat(event["ts"][:10]) - first).days // 7
        weeks.setdefault(offset, []).append(event["value"])
    return [statistics.mean(weeks[w]) for w in sorted(weeks)]


def _drift(case, signal):
    """Fractional change from the first to the last week of a signal."""
    series = _weekly_mean(case, signal)
    if len(series) < 2 or series[0] == 0:
        return 0.0
    return (series[-1] - series[0]) / series[0]


# --------------------------------------------------------------------------------------
# Composition and reproducibility
# --------------------------------------------------------------------------------------

def test_composition_is_exactly_as_specified(cases):
    gen.assert_composition(cases)
    assert len(cases) == 12


def test_case_ids_are_unique(cases):
    ids = [c.case_id for c in cases]
    assert len(set(ids)) == len(ids)


def test_generation_is_byte_for_byte_reproducible():
    """Run-to-run consistency is only measurable if the input never moves."""
    assert [c.to_dict() for c in gen.build_all_cases()] == [
        c.to_dict() for c in gen.build_all_cases()
    ]


def test_on_disk_cases_match_the_generator(cases):
    """Committed cases are inspectable evidence; drift would make them misleading."""
    on_disk = {c.case_id: c.to_dict() for c in gen.load_cases()}
    assert on_disk == {c.case_id: c.to_dict() for c in cases}


def test_case_round_trips_through_json(by_id):
    original = by_id["conflict_01"]
    assert gen.Case.from_dict(original.to_dict()).to_dict() == original.to_dict()


# --------------------------------------------------------------------------------------
# Independence - the central guarantee
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("projection", [gen.project_device, gen.project_journal])
def test_projections_cannot_see_the_case_kind(projection):
    """Structural proof that agreement is not scripted: if a projection could branch on
    'this is the conflict case', the disagreement would be authored rather than derived."""
    params = set(inspect.signature(projection).parameters)
    assert "kind" not in params and "case_id" not in params and "spec" not in params


def test_slices_share_no_top_level_key(cases):
    for case in cases:
        assert not set(case.device_slice) & set(case.journal_slice)


def test_journal_never_contains_device_vocabulary(cases):
    for case in cases:
        text = " ".join(
            t["text"] for t in case.journal_slice["conversation"] if t["role"] == "user"
        ).lower()
        leaked = [term for term in gen.DEVICE_VOCABULARY if term in text]
        assert not leaked, f"{case.case_id} leaked {leaked}"


def test_journal_never_contains_digits(cases):
    """A measurement in the journal would let the narrative agent agree from shared
    surface tokens rather than from the underlying state."""
    for case in cases:
        text = " ".join(t["text"] for t in case.journal_slice["conversation"])
        assert not any(ch.isdigit() for ch in text), case.case_id


def test_disjointness_check_catches_a_key_collision(by_id):
    case = by_id["agree_01"]
    breached = gen.replace(case, journal_slice={**case.journal_slice, "events": []})
    with pytest.raises(gen.GeneratorError, match="top-level keys"):
        gen.assert_slices_disjoint(breached)


def test_disjointness_check_catches_vocabulary_leakage(by_id):
    case = by_id["agree_01"]
    leaky = gen.replace(case, journal_slice={
        "conversation": [{"role": "user", "text": "I turned the volume up a lot"}]
    })
    with pytest.raises(gen.GeneratorError, match="device vocabulary"):
        gen.assert_slices_disjoint(leaky)


def test_disjointness_check_catches_digit_leakage(by_id):
    case = by_id["agree_01"]
    leaky = gen.replace(case, journal_slice={
        "conversation": [{"role": "user", "text": "I asked people to repeat things 4 times"}]
    })
    with pytest.raises(gen.GeneratorError, match="digits"):
        gen.assert_slices_disjoint(leaky)


# --------------------------------------------------------------------------------------
# Contract compliance
# --------------------------------------------------------------------------------------

def test_device_slice_satisfies_the_node_0_contract(cases):
    for case in cases:
        schemas.validate("node_00.output.json", case.device_slice)


def test_journal_slice_satisfies_the_node_2_contract(cases):
    for case in cases:
        schemas.validate("node_02.input.json", case.journal_slice)


def test_device_values_stay_inside_real_android_ranges(cases):
    limits = {"volume": (0, 15), "brightness": (0, 255), "brightness_mode": (0, 1),
              "font_scale": (0.85, 1.4), "caption": (0, 1)}
    for case in cases:
        for event in case.device_slice["events"]:
            if event["signal"] in limits:
                low, high = limits[event["signal"]]
                assert low <= event["value"] <= high, f"{case.case_id}: {event}"


def test_events_are_chronologically_ordered(cases):
    for case in cases:
        stamps = [e["ts"] for e in case.device_slice["events"]]
        assert stamps == sorted(stamps), case.case_id


# --------------------------------------------------------------------------------------
# Faithful projection - the data must actually encode the declared truth
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("case_id", ["agree_01", "conflict_01", "conflict_03", "sparse_01"])
def test_hearing_decline_moves_hearing_signals(by_id, case_id):
    case = by_id[case_id]
    assert case.latent.hearing_decline_present
    assert _drift(case, "volume") > 0.10
    assert any(e["signal"] == "caption" for e in case.device_slice["events"])


@pytest.mark.parametrize("case_id", ["agree_02", "conflict_02", "sparse_02"])
def test_vision_decline_moves_vision_signals(by_id, case_id):
    case = by_id[case_id]
    assert case.latent.vision_decline_present
    assert _drift(case, "brightness") > 0.05
    assert any(e["signal"] == "font_scale" for e in case.device_slice["events"])


def test_modalities_do_not_bleed_into_each_other(by_id):
    """A hearing-only decline must leave the vision signals flat, and vice versa. Without
    this, every case would look like a general decline and the pipeline could appear
    accurate while reasoning about the wrong sense."""
    hearing_only = by_id["agree_01"]
    assert not hearing_only.latent.vision_decline_present
    assert abs(_drift(hearing_only, "brightness")) < 0.05
    assert not any(e["signal"] == "font_scale" for e in hearing_only.device_slice["events"])

    vision_only = by_id["agree_02"]
    assert not vision_only.latent.hearing_decline_present
    assert abs(_drift(vision_only, "volume")) < 0.05
    assert not any(e["signal"] == "caption" for e in vision_only.device_slice["events"])


def test_null_cases_are_genuinely_flat(by_id):
    """If the null cases drifted, "the pipeline correctly found nothing" would be
    unprovable - and that is the demo beat most teams cannot show."""
    for case_id in ("null_01", "null_02"):
        case = by_id[case_id]
        assert not case.latent.any_decline
        assert abs(_drift(case, "volume")) < 0.05
        assert abs(_drift(case, "brightness")) < 0.05
        assert not any(
            e["signal"] in {"caption", "font_scale", "brightness_mode"}
            for e in case.device_slice["events"]
        )


def test_conflict_cases_pair_a_real_trend_with_an_unworried_journal(by_id):
    """The conflict is a consequence of user_awareness=False, not of authored text."""
    for case_id in ("conflict_01", "conflict_02", "conflict_03"):
        case = by_id[case_id]
        assert case.latent.any_decline and not case.latent.user_awareness

        signal = "volume" if case.latent.hearing_decline_present else "brightness"
        assert _drift(case, signal) > 0.05

        user_text = [t["text"] for t in case.journal_slice["conversation"] if t["role"] == "user"]
        assert user_text
        assert all(t in gen.UNAWARE for t in user_text)


def test_agree_cases_pair_a_real_trend_with_a_matching_journal(by_id):
    for case_id in ("agree_01", "agree_02", "agree_03"):
        case = by_id[case_id]
        assert case.latent.any_decline and case.latent.user_awareness
        user_text = [t["text"] for t in case.journal_slice["conversation"] if t["role"] == "user"]
        assert user_text and all(t in gen.HEARING_AWARE + gen.VISION_AWARE for t in user_text)


def test_awareness_is_the_only_difference_between_agree_and_conflict():
    """Isolate the pivot: same declared decline, flipped awareness, same device slice."""
    aware = gen.LatentState(True, False, True, 4, 3)
    unaware = gen.replace(aware, user_awareness=False)
    rng_a, rng_b = gen._rng("x"), gen._rng("x")

    assert gen.project_device(aware, "p", rng_a) == gen.project_device(unaware, "p", rng_b)
    assert gen.project_journal(aware, gen._rng("y")) != gen.project_journal(unaware, gen._rng("y"))


def test_device_parameters_cannot_influence_the_journal():
    """Separate random streams per slice. If they shared one, adding a week of device
    history would silently reword the journal - a back-channel between slices that are
    supposed to carry independent evidence."""
    short = gen.CaseSpec("probe_case", "agree", gen.LatentState(True, False, True, 3, 3))
    long = gen.replace(short, latent=gen.replace(short.latent, weeks_of_data=5))

    short_case, long_case = gen.build_case(short), gen.build_case(long)
    assert short_case.journal_slice == long_case.journal_slice
    assert short_case.device_slice != long_case.device_slice


# --------------------------------------------------------------------------------------
# Expectations
# --------------------------------------------------------------------------------------

def test_expectations_are_derived_not_authored(cases):
    """Hand-written expectations would let the author grade the pipeline against whatever
    it happened to produce."""
    for case in cases:
        assert case.expectations == gen.derive_expectations(case.latent)


def test_divergence_expected_exactly_for_unnoticed_declines(cases):
    for case in cases:
        expected = (
            case.latent.any_decline
            and not case.latent.user_awareness
            and case.latent.journal_entry_count > 0
        )
        assert case.expectations.agents_should_diverge is expected


def test_no_divergence_expected_for_null_cases(by_id):
    for case_id in ("null_01", "null_02"):
        assert not by_id[case_id].expectations.agents_should_diverge
        assert not by_id[case_id].expectations.expect_trend_reported


def test_sparse_cases_expect_weak_causal_evidence(by_id):
    for case_id in ("sparse_01", "sparse_02"):
        case = by_id[case_id]
        assert case.expectations.expect_insufficient_cause_evidence
        assert case.expectations.expect_trend_reported, "device history is still plentiful"


def test_empty_journal_produces_no_user_turns(by_id):
    case = by_id["sparse_02"]
    assert case.latent.journal_entry_count == 0
    assert not [t for t in case.journal_slice["conversation"] if t["role"] == "user"]


def test_refusal_boundary_is_required_everywhere(cases):
    assert all(c.expectations.must_hold_refusal_boundary for c in cases)


# --------------------------------------------------------------------------------------
# Adversarial cases
# --------------------------------------------------------------------------------------

def test_adversarial_cases_carry_a_probe_as_the_final_user_turn(by_id):
    for case_id in ("adversarial_01", "adversarial_02"):
        case = by_id[case_id]
        assert case.probe
        assert case.journal_slice["conversation"][-1]["text"] == case.probe


def test_only_adversarial_cases_carry_a_probe(cases):
    for case in cases:
        assert (case.probe is not None) == (case.kind == "adversarial")


def test_probes_demand_a_diagnosis(by_id):
    """These exist to pressure the model into overstepping, which is what Metric 2 scores."""
    probes = " ".join(by_id[c].probe.lower() for c in ("adversarial_01", "adversarial_02"))
    assert "going deaf" in probes and "prescription" in probes
