"""Tests for the HTTP surface the Android app talks to.

The service is thin on purpose, so these tests are narrow on purpose: they cover the two
places where a shape is translated rather than passed through, because a translation is
where meaning gets lost. Everything else in ``serve/api.py`` delegates to nodes that are
already tested, and re-asserting their behaviour here would test the delegation twice
while testing the translation not at all.
"""

from __future__ import annotations

from serve import api


class TestFiguresKeepThreeStateSignificance:
    """The engine reports significance as true, false, or "could not test". So must the app.

    This is the guarantee worth pinning. The engine deliberately sets ``significant`` to
    ``None`` when the window cannot answer whether a slope differs from zero — two points
    fit a line exactly — and a display layer that coerces that to ``False`` silently
    upgrades *we could not test this* into *we tested this and it is noise*. The second is a
    claim about the data; the first is an admission about the window. They are not the same
    sentence and the difference is the honest part.
    """

    def test_a_significant_figure_stays_significant(self):
        figures = api._figures(
            {"figures": {"volume_pct_change": {"value": 67.9, "unit": "%", "significant": True}}}
        )
        assert figures[0]["significant"] is True

    def test_a_tested_and_unremarkable_figure_stays_false(self):
        figures = api._figures(
            {"figures": {"volume_pct_change": {"value": 1.2, "unit": "%", "significant": False}}}
        )
        assert figures[0]["significant"] is False

    def test_an_untestable_figure_is_not_reported_as_insignificant(self):
        figures = api._figures(
            {"figures": {"caption_on_rate": {"value": 0.5, "unit": "ratio", "significant": None}}}
        )
        assert figures[0]["significant"] is None, (
            "None means the window could not answer the question. Reporting False here "
            "would claim a test was run that was not."
        )

    def test_a_figure_with_no_verdict_at_all_is_not_invented(self):
        figures = api._figures({"figures": {"odd": {"value": 1.0, "unit": "x"}}})
        assert figures[0]["significant"] is None

    def test_the_value_carries_its_unit(self):
        figures = api._figures(
            {"figures": {"volume_trend_per_week": {
                "value": 1.559, "unit": "step/week", "significant": True,
            }}}
        )
        assert figures[0]["value"] == "1.559 step/week"

    def test_figures_arrive_in_a_stable_order(self):
        trend = {"figures": {
            "volume_pct_change": {"value": 1, "unit": "%", "significant": True},
            "brightness_pct_change": {"value": 2, "unit": "%", "significant": True},
        }}
        assert [f["name"] for f in api._figures(trend)] == [
            "brightness pct change", "volume pct change",
        ]

    def test_no_figures_is_not_an_error(self):
        assert api._figures({}) == []


class TestHeadline:
    """Node 5 emits claims; the app shows all of them, not the first one it finds."""

    def test_every_claim_survives_to_the_screen(self):
        synthesis = {"claims": [
            {"text": "Volume rose 67.9% over four weeks.", "evidence": [{}]},
            {"text": "Font scale rose 13.8% over four weeks.", "evidence": [{}]},
        ]}
        headline = api._headline(synthesis)
        assert "Volume rose" in headline
        assert "Font scale rose" in headline, (
            "Vision and hearing signals move independently, so dropping claims after the "
            "first can discard a second finding that was just as well evidenced."
        )

    def test_the_agreement_sentence_is_used_when_there_are_no_claims(self):
        synthesis = {"claims": [], "agreement": "Nothing moved beyond normal variation."}
        assert api._headline(synthesis) == "Nothing moved beyond normal variation."

    def test_an_empty_synthesis_produces_an_empty_headline_not_a_crash(self):
        assert api._headline({}) == ""

    def test_a_blank_claim_does_not_become_a_blank_headline(self):
        synthesis = {"claims": [{"text": "   ", "evidence": [{}]}], "agreement": "No change."}
        assert api._headline(synthesis) == "No change."
