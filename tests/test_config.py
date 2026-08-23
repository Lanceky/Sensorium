"""Step 1 acceptance: routing configuration is internally consistent.

The Node 4 invariant is the important one. If the two blind agents are ever routed to
different models or temperatures, any disagreement they produce is confounded by model
choice rather than by the data slices - which would quietly invalidate the central claim
of the submission. Better to fail a test than to discover that in front of judges.
"""

from __future__ import annotations

import pytest

from sensorium import config, prompts


def test_invariants_hold():
    config.check_invariants()


def test_blind_agents_are_routed_identically():
    a, b = (config.get_node_config(n) for n in config.BLIND_AGENT_PAIR)
    assert (a.size, a.temperature) == (b.size, b.temperature)


def test_invariant_check_catches_divergent_blind_agents(monkeypatch):
    a, b = config.BLIND_AGENT_PAIR
    skewed = dict(config.REGISTRY)
    original = skewed[b]
    skewed[b] = config.NodeConfig(
        node=original.node,
        size="large",
        temperature=original.temperature,
        prompt_version=original.prompt_version,
        rationale=original.rationale,
    )
    monkeypatch.setattr(config, "REGISTRY", skewed)
    with pytest.raises(config.ConfigError, match="must share size and temperature"):
        config.check_invariants()


@pytest.mark.parametrize("node", sorted(config.REGISTRY))
def test_every_llm_node_has_its_declared_prompt(node):
    cfg = config.get_node_config(node)
    assert prompts.load_prompt(node, cfg.prompt_version).strip()


@pytest.mark.parametrize("node", sorted(config.REGISTRY))
def test_every_llm_node_declares_a_rationale(node):
    """The registry is what the workflow PNG is generated from; an unexplained routing
    decision is an unanswerable 'why did you use that?' during judging."""
    assert len(config.get_node_config(node).rationale) > 20


def test_deterministic_nodes_are_never_llm_routed():
    assert not config.DETERMINISTIC_NODES & config.REGISTRY.keys()


def test_every_size_is_pinned_to_a_real_model():
    """Pinned in Step 3 from a live /v1/models response, then verified with a completion.

    The empty-registry guard this replaces did its job: it kept a guessed model id out of
    the docs until the account could be queried, and the guess would have been wrong --
    every meta-llama id in the design doc is gated on this account.
    """
    assert all(config.MODEL_BY_SIZE.values())
    assert config.resolve_model("node_05") == config.MODEL_BY_SIZE["large"]


def test_an_unpinned_size_still_refuses_to_run(monkeypatch):
    """The refusal must survive pinning, since it guards any size added later."""
    monkeypatch.setitem(config.MODEL_BY_SIZE, "large", "")
    with pytest.raises(config.ConfigError, match="not pinned yet"):
        config.resolve_model("node_05")


def test_the_size_ladder_is_three_distinct_models():
    """Equal ids would make "routed by size" a fiction the report still asserts."""
    assert len(set(config.MODEL_BY_SIZE.values())) == 3


def test_all_tiers_share_one_model_family():
    """Node 4's agents and the cross-node comparisons must differ by size alone."""
    families = {model.split("/")[0] for model in config.MODEL_BY_SIZE.values()}
    assert len(families) == 1, f"mixed lineage would confound size effects: {families}"


def test_unknown_node_is_rejected():
    with pytest.raises(config.ConfigError, match="unknown node"):
        config.get_node_config("node_99")


def test_require_env_reports_the_missing_variable(monkeypatch):
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    with pytest.raises(config.ConfigError, match="FEATHERLESS_API_KEY"):
        config.require_env("FEATHERLESS_API_KEY")
