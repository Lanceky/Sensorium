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


def test_model_ids_are_unpinned_until_step_3():
    """Deliberate: guessing a Featherless model id is worse than admitting it is unpinned.
    Update MODEL_BY_SIZE and this test together in Step 3."""
    assert all(not v for v in config.MODEL_BY_SIZE.values())
    with pytest.raises(config.ConfigError, match="not pinned yet"):
        config.resolve_model("node_05")


def test_unknown_node_is_rejected():
    with pytest.raises(config.ConfigError, match="unknown node"):
        config.get_node_config("node_99")


def test_require_env_reports_the_missing_variable(monkeypatch):
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    with pytest.raises(config.ConfigError, match="FEATHERLESS_API_KEY"):
        config.require_env("FEATHERLESS_API_KEY")
