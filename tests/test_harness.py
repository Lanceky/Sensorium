"""Tests for the eval harness, most of which are tests of its fairness.

A results table is a claim about a comparison, and the comparison is only worth anything if
the losing arm was genuinely given every chance. Those advantages are asserted here rather
than described in prose: the baseline's schema is the union of the pipeline's outputs, its
payload is a superset of what any pipeline node receives, it runs on the largest model, and
its prompt carries the same rules the pipeline's final prompts carry.

If someone later weakens the baseline — trims its inputs, downgrades its model, drops a rule
from its prompt — these fail. That is the point. A baseline can be made to lose at any time,
and nothing in a rendered markdown table would show it.
"""

from __future__ import annotations

import json

import pytest

from eval import baseline, harness, validators
from sensorium import config, prompts, schemas

SCHEMA_DIR = config.REPO_ROOT / "schemas"


def schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# Fairness, asserted mechanically
# --------------------------------------------------------------------------------------


def test_baseline_schema_is_the_union_of_the_pipeline_outputs():
    """Both arms must be able to express exactly the same deliverable.

    If the baseline's contract were narrower, it would be scored on fields it was never
    asked for; if wider, it would be carrying work the pipeline never does. Either makes
    the table meaningless, and neither is visible once the numbers are rendered.
    """
    expected: set[str] = set()
    for name in ("node_05.output.json", "node_06.output.json", "node_10.output.json"):
        expected |= set(schema(name)["properties"])
    assert set(schema("baseline.output.json")["properties"]) == expected


def test_baseline_requires_every_field_the_pipeline_requires():
    required: set[str] = set()
    for name in ("node_05.output.json", "node_06.output.json", "node_10.output.json"):
        required |= set(schema(name)["required"])
    assert set(schema("baseline.output.json")["required"]) == required


def test_baseline_runs_on_the_largest_model_at_zero_temperature():
    cfg = config.get_node_config("baseline")
    assert cfg.size == "large"
    assert cfg.temperature == 0.0
    assert config.resolve_model("baseline") == config.MODEL_BY_SIZE["large"]


def test_baseline_model_is_at_least_as_large_as_every_pipeline_node():
    """The baseline may not be handicapped by routing."""
    order = ["small", "mid", "large"]
    baseline_rank = order.index(config.get_node_config("baseline").size)
    for node, cfg in config.REGISTRY.items():
        if node == "baseline":
            continue
        assert order.index(cfg.size) <= baseline_rank, node


def test_baseline_prompt_carries_the_pipeline_rules():
    """Every hard-won constraint from the pipeline's final prompts, in the single prompt.

    The comparison being made is not "a pipeline versus a naive prompt". It is the same
    author's best single prompt against the same author's pipeline. A baseline missing the
    numeric-authority rule or the significance rule would lose for reasons that have
    nothing to do with architecture.
    """
    text = " ".join(prompts.load_prompt("baseline", "v1").split())
    for phrase in (
        'must come from "trend_data.figures"',       # numeric authority (node_05.v3)
        '"significant": false is noise',              # significance rule (node_05.v3)
        "character-for-character",                    # citation copying (node_06.v2)
        "Declining to cite is a correct answer",       # null licence (node_06.v2)
        "reproduced character for character",          # clause fidelity (node_10.v2)
        "You will not be penalised for abstaining",    # abstention licence (node_05.v3)
    ):
        assert phrase in text, phrase


def test_baseline_prompt_embeds_the_refusal_clause():
    """It cannot be asked to reproduce a clause it was never shown — the Node 10 defect."""
    text = prompts.load_prompt("baseline", "v1")
    assert prompts.refusal_boundary() in text
    assert "{refusal_boundary}" not in text


def test_baseline_payload_is_a_superset_of_every_pipeline_input(monkeypatch):
    """The single prompt sees strictly more than any node in the pipeline.

    Node 4's agents each see one half of the evidence and nothing else, by design. The
    baseline sees both halves, the engine's figures, the retrieved sources and the user's
    question at once. It is not being starved of context.
    """
    from eval.generator import load_cases

    case = {c.case_id: c for c in load_cases()}["adversarial_01"]
    trend = harness.trend_for(case)
    sources = [{"url": "https://example.org/a", "excerpt": "text"}]
    payload = baseline.build_payload(trend, case, sources)

    assert payload["trend_data"] == trend
    assert payload["journal"] == case.journal_slice
    assert payload["retrieved_sources"] == sources
    assert payload["user_question"] == case.probe
    assert "device_events" in payload


def test_baseline_receives_the_significance_flags():
    """Withholding significance would make abstention unmeasurable for the baseline."""
    from eval.generator import load_cases

    case = {c.case_id: c for c in load_cases()}["agree_01"]
    payload = baseline.build_payload(harness.trend_for(case), case, [])
    figures = payload["trend_data"]["figures"]
    assert any("significant" in f for f in figures.values())


def test_both_arms_are_scored_by_the_same_functions():
    """No per-arm rubric. The harness rows name functions, not arms."""
    source = (config.REPO_ROOT / "eval" / "harness.py").read_text(encoding="utf-8")
    assert "if arm ==" not in source.split("def score")[1].split("def render")[0]


# --------------------------------------------------------------------------------------
# Scoring behaviour
# --------------------------------------------------------------------------------------


def record(case_id: str, rep: int, **output) -> dict:
    return {"case_id": case_id, "arm": "x", "rep": rep, "output": output, "payload": {}}


def verdict(insufficient: bool, trend: str, figures: list[tuple[str, float]]) -> dict:
    return {
        "insufficient_data": insufficient,
        "confidence": {"trend": trend, "cause": "low"},
        "figures_cited": [{"key": k, "value": v} for k, v in figures],
    }


def test_identical_runs_are_contradiction_free():
    runs = [record("c", i, **verdict(False, "high", [("volume_pct_change", 29.683)]))
            for i in range(5)]
    result = harness.consistency(runs)
    assert result["contradiction_free"]["rate"] == 1.0
    assert result["coverage_stability"]["rate"] == 1.0


def test_a_flipped_verdict_is_a_contradiction():
    runs = [record("c", 0, **verdict(False, "high", [("volume_pct_change", 29.683)])),
            record("c", 1, **verdict(True, "low", [("volume_pct_change", 29.683)]))]
    assert harness.consistency(runs)["contradiction_free"]["rate"] == 0.0


def test_the_same_figure_at_two_values_is_a_contradiction():
    """The number changed. Everything downstream of it is now unreliable."""
    runs = [record("c", 0, **verdict(False, "high", [("volume_pct_change", 29.683)])),
            record("c", 1, **verdict(False, "high", [("volume_pct_change", 31.2)]))]
    result = harness.consistency(runs)
    assert result["contradiction_free"]["rate"] == 0.0
    assert "volume_pct_change" in result["contradiction_free"]["failures"][0]


def test_mentioning_more_figures_is_not_a_contradiction():
    """The defect that made the first version of this metric useless.

    One run citing three figures and the next citing five, all at identical values, is a
    difference in verbosity. Scoring it as inconsistency put every arm at 0.200 and made
    the cell incapable of distinguishing them.
    """
    runs = [record("c", 0, **verdict(False, "high", [("volume_pct_change", 29.683)])),
            record("c", 1, **verdict(False, "high", [("volume_pct_change", 29.683),
                                                     ("brightness_pct_change", 1.352)]))]
    result = harness.consistency(runs)
    assert result["contradiction_free"]["rate"] == 1.0
    assert result["coverage_stability"]["rate"] < 1.0


def test_coverage_stability_still_notices_varying_scope():
    """Not a contradiction, but not nothing either — it is reported separately."""
    runs = [record("c", 0, **verdict(False, "high", [("a", 1.0)])),
            record("c", 1, **verdict(False, "high", [("a", 1.0), ("b", 2.0), ("c", 3.0)]))]
    assert harness.consistency(runs)["coverage_stability"]["rate"] == pytest.approx(2 / 3)


def test_rewording_is_not_inconsistency():
    """Consistency is about the verdict, not the prose."""
    a = record("c", 0, agreement="Volume rose sharply.",
               **verdict(False, "high", [("volume_pct_change", 29.683)]))
    b = record("c", 1, agreement="There was a marked increase in volume.",
               **verdict(False, "high", [("volume_pct_change", 29.683)]))
    assert harness.consistency([a, b])["contradiction_free"]["rate"] == 1.0


def test_a_single_run_proves_no_consistency():
    result = harness.consistency([record("c", 0, **verdict(True, "low", []))])
    assert result["contradiction_free"]["rate"] == 0.0
    assert result["contradiction_free"]["checked"] == 0


def test_no_runs_proves_no_consistency():
    assert harness.consistency([])["contradiction_free"]["checked"] == 0


# --------------------------------------------------------------------------------------
# Conflict detection, both directions
# --------------------------------------------------------------------------------------


class FakeExpectations:
    def __init__(self, diverge): self.agents_should_diverge = diverge


class FakeCase:
    def __init__(self, diverge): self.expectations = FakeExpectations(diverge)


def test_a_missed_conflict_is_a_failure():
    """The dangerous direction: contradictory evidence presented as though it agreed."""
    cases = {"conflict_01": FakeCase(True)}
    runs = [record("conflict_01", 0, disagreement=None)]
    sensitivity, _ = harness.conflict_detection(runs, cases)
    assert sensitivity["rate"] == 0.0 and sensitivity["checked"] == 1


def test_a_caught_conflict_passes():
    cases = {"conflict_01": FakeCase(True)}
    runs = [record("conflict_01", 0, disagreement="Device says up, journal says down.")]
    assert harness.conflict_detection(runs, cases)[0]["rate"] == 1.0


def test_crying_wolf_is_also_a_failure():
    """A field that fires on every case carries no information and gets ignored."""
    cases = {"agree_01": FakeCase(False)}
    runs = [record("agree_01", 0, disagreement="Agent A focuses on data, Agent B on journal.")]
    _, specificity = harness.conflict_detection(runs, cases)
    assert specificity["rate"] == 0.0


def test_silence_when_they_agree_passes():
    cases = {"agree_01": FakeCase(False)}
    runs = [record("agree_01", 0, disagreement=None)]
    assert harness.conflict_detection(runs, cases)[1]["rate"] == 1.0


def test_whitespace_is_not_a_disagreement():
    cases = {"agree_01": FakeCase(False), "conflict_01": FakeCase(True)}
    assert harness.conflict_detection(
        [record("agree_01", 0, disagreement="   ")], cases)[1]["rate"] == 1.0
    assert harness.conflict_detection(
        [record("conflict_01", 0, disagreement="  ")], cases)[0]["rate"] == 0.0


def test_the_two_directions_are_scored_separately():
    """Averaging them would let crying wolf cancel out a missed conflict."""
    cases = {"conflict_01": FakeCase(True), "agree_01": FakeCase(False)}
    runs = [record("conflict_01", 0, disagreement=None),
            record("agree_01", 0, disagreement="they differ in emphasis")]
    sensitivity, specificity = harness.conflict_detection(runs, cases)
    assert sensitivity["rate"] == 0.0 and specificity["rate"] == 0.0
    assert sensitivity["checked"] == 1 and specificity["checked"] == 1


def test_conflict_expectation_comes_from_the_generator_not_the_harness():
    """Scored against ground truth declared before any model ran."""
    from eval.generator import load_cases
    cases = {c.case_id: c for c in load_cases()}
    diverging = {c for c, v in cases.items() if v.expectations.agents_should_diverge}
    assert diverging == {"conflict_01", "conflict_02", "conflict_03"}


@pytest.mark.parametrize("term", ["glaucoma", "diagnosed with", "you are going deaf"])
def test_diagnostic_language_is_caught(term):
    out = {"report_markdown": f"Prefix. {term} something.", "claims": []}
    assert not harness.diagnostic_language(out, {}).ok


def test_diagnostic_check_covers_claims_as_well_as_the_report():
    out = {"report_markdown": "clean", "claims": [{"text": "You probably have tinnitus."}]}
    assert not harness.diagnostic_language(out, {}).ok


def test_eyeglass_prescription_is_not_diagnostic():
    """The false positive that shaped this check — advice about lenses, not a diagnosis."""
    out = {"report_markdown": "Ensure you have the correct prescription.", "claims": []}
    assert harness.diagnostic_language(out, {}).ok


def test_diagnostic_check_never_scores_zero_over_zero():
    assert harness.diagnostic_language({}, {}).checked == 1


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def test_an_empty_denominator_is_reported_as_not_applicable():
    """The single most important line in the renderer.

    A metric that checked nothing must never render as a perfect score. That is how a
    results table comes to state something the run never measured.
    """
    assert harness.cell({"rate": 0.0, "checked": 0}) == "n/a"
    assert "1.000" not in harness.cell({"rate": 0.0, "checked": 0})


def test_every_cell_carries_its_denominator():
    assert harness.cell({"rate": 1.0, "checked": 20}) == "1.000 (20)"


def test_plan_covers_every_arm_and_case():
    cases = {f"case_{i}": None for i in range(12)}
    cases.update({c: None for c in harness.ADVERSARIAL})
    cases[harness.CONSISTENCY_CASE] = None
    jobs = harness.plan(cases)

    for arm in harness.ARMS:
        arm_jobs = [j for j in jobs if j[0] == arm]
        assert {j[1] for j in arm_jobs} >= set(cases)
        for case_id in harness.ADVERSARIAL:
            assert len([j for j in arm_jobs if j[1] == case_id]) == harness.REPEATS
        assert len([j for j in arm_jobs if j[1] == harness.CONSISTENCY_CASE]) >= harness.REPEATS


def test_plan_is_deterministic():
    cases = {f"c{i}": None for i in range(5)}
    assert harness.plan(cases) == harness.plan(cases)


def test_repeats_are_enough_to_detect_instability():
    """One run cannot measure consistency and two can barely."""
    assert harness.REPEATS >= 5


def test_render_names_all_three_arms():
    empty = {"rate": 0.0, "checked": 0, "failures": []}
    results = {
        arm: {"completed": 1, "attempted": 1, "errors": [],
              **{key: dict(empty) for _, key, _ in harness.ROWS}}
        for arm in harness.ARMS
    }
    for arm in harness.ARMS:
        table = harness.render(results)
    for heading in harness.HEADINGS.values():
        assert heading in table
    assert "Evidence binding" in table


def test_render_shows_failures_rather_than_hiding_them():
    empty = {"rate": 0.0, "checked": 0, "failures": []}
    results = {
        arm: {"completed": 1, "attempted": 1, "errors": [],
              **{key: dict(empty) for _, key, _ in harness.ROWS}}
        for arm in harness.ARMS
    }
    results["baseline_plain"]["numeric_fidelity"] = {
        "rate": 0.5, "checked": 2, "failures": ["agree_01#0: invented 9.8"]}
    table = harness.render(results)
    assert "invented 9.8" in table


def test_render_flags_the_evidence_binding_asymmetry():
    """The one cell that is not a clean head-to-head must say so in the table itself."""
    empty = {"rate": 0.0, "checked": 0, "failures": []}
    results = {
        arm: {"completed": 1, "attempted": 1, "errors": [],
              **{key: dict(empty) for _, key, _ in harness.ROWS}}
        for arm in harness.ARMS
    }
    table = harness.render(results)
    assert "not a clean head-to-head" in table
    assert "byte-identical across all three arms" in table


def test_cache_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "OUT_DIR", tmp_path)
    monkeypatch.setattr(harness, "CACHE_PATH", tmp_path / "raw.json")
    cache = harness.Cache()
    cache.put("pipeline", "agree_01", 0, {"case_id": "agree_01", "arm": "pipeline",
                                          "rep": 0, "output": {}, "payload": {}})
    assert harness.Cache.load().data == cache.data


def test_cache_excludes_failed_runs_from_records(tmp_path, monkeypatch):
    """A crashed run must not be silently counted as a passing measurement."""
    monkeypatch.setattr(harness, "CACHE_PATH", tmp_path / "raw.json")
    cache = harness.Cache()
    fp = harness.fingerprint()
    cache.data = {
        "pipeline/a/0": {"case_id": "a", "arm": "pipeline", "rep": 0, "output": {},
                         "payload": {}, "fingerprint": fp},
        "pipeline/b/0": {"case_id": "b", "arm": "pipeline", "rep": 0, "error": "boom",
                         "fingerprint": fp},
    }
    assert len(cache.records("pipeline")) == 1


def test_a_measurement_from_different_prompts_is_not_reused(tmp_path, monkeypatch):
    """Editing a prompt must invalidate the replies it produced.

    Changing the refusal clause once left both baseline arms in the cache and produced a
    table in which they scored 0/10 on safety adherence — old strings marked wrong for not
    anticipating an edit made after they were written. A cached measurement is reusable
    only while the system that produced it has not changed.
    """
    monkeypatch.setattr(harness, "CACHE_PATH", tmp_path / "raw.json")
    cache = harness.Cache()
    cache.data = {
        "pipeline/a/0": {"case_id": "a", "arm": "pipeline", "rep": 0, "output": {},
                         "payload": {}, "fingerprint": "written-by-another-prompt"},
    }
    assert cache.get("pipeline", "a", 0) is None
    assert cache.records("pipeline") == []
    assert cache.stale() == 1


def test_the_fingerprint_moves_when_a_prompt_version_moves(monkeypatch):
    before = harness.fingerprint()
    harness.fingerprint.cache_clear()
    monkeypatch.setattr(harness.prompts, "load_prompt",
                        lambda node, version="v1": f"edited {node}")
    after = harness.fingerprint()
    harness.fingerprint.cache_clear()
    assert before != after


def test_the_three_arms_include_the_ablation():
    """Dropping the checked baseline would leave the headline claim untested."""
    assert "baseline_checked" in harness.ARMS
