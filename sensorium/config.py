"""Per-node model routing and runtime configuration.

The registry below is the answer to the track's explicit "which LLM model is used"
requirement, and it is what the workflow PNG is generated from. Model *ids* are pinned in
Step 3 after querying Featherless' ``/v1/models``; sizes, temperatures and rationales are
design decisions and belong here now.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"

FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"

#: Loaded once, here, because this is the module that reads the environment. ``override``
#: stays False so a real environment variable always beats the file: a developer exporting
#: a key for a single command should not be silently overruled by a stale ``.env``, and CI
#: must never pick up a file that happens to be lying in the working tree.
_ENV_PATH = REPO_ROOT / ".env"
if _ENV_PATH.exists():  # pragma: no cover - import side effect, exercised end to end
    try:
        from dotenv import load_dotenv

        load_dotenv(_ENV_PATH, override=False)
    except ImportError:
        pass


class ConfigError(Exception):
    """Raised when configuration is missing or internally inconsistent."""


@dataclass(frozen=True)
class NodeConfig:
    """Routing decision for one LLM node."""

    node: str
    size: str
    temperature: float
    prompt_version: str
    rationale: str


#: Size -> exact Featherless model id, pinned against a live ``/v1/models`` response and
#: then verified with a real completion, because being listed and being callable are
#: different things: every ``meta-llama/*`` and ``google/gemma-*`` id returns 403 "This
#: model is gated" on this account, so the Llama-3.1 family named in the design doc was
#: never actually available to cite.
#:
#: All three tiers are the same family on purpose. Node 4's two agents must differ only in
#: the data they see, and the report compares nodes across sizes; sharing a tokenizer and
#: chat template means an observed difference is attributable to size, not to a change of
#: training lineage. QwQ-32B was rejected despite being callable — it emitted its JSON
#: object twice in a row, which is the reasoning-model failure mode that structured output
#: cannot tolerate.
MODEL_BY_SIZE: dict[str, str] = {
    "small": "Qwen/Qwen2.5-7B-Instruct",
    "mid": "Qwen/Qwen2.5-32B-Instruct",
    "large": "Qwen/Qwen2.5-72B-Instruct",
}

REGISTRY: dict[str, NodeConfig] = {
    "node_0_5": NodeConfig(
        node="node_0_5",
        size="small",
        temperature=0.0,
        prompt_version="v1",
        rationale="Mechanical restatement of a staircase result; creativity is a defect here.",
    ),
    "node_01": NodeConfig(
        node="node_01",
        size="small",
        temperature=0.7,
        prompt_version="v1",
        rationale="Latency and natural tone matter more than depth; lowest-stakes node.",
    ),
    "node_02": NodeConfig(
        node="node_02",
        size="mid",
        temperature=0.0,
        prompt_version="v2",
        rationale="Extraction accuracy over speed; determinism protects Metric 4.",
    ),
    "node_04a": NodeConfig(
        node="node_04a",
        size="mid",
        temperature=0.2,
        prompt_version="v1",
        rationale="Must be identical to node_04b so disagreement is caused by data, not routing.",
    ),
    "node_04b": NodeConfig(
        node="node_04b",
        size="mid",
        temperature=0.2,
        prompt_version="v1",
        rationale="Must be identical to node_04a so disagreement is caused by data, not routing.",
    ),
    "node_05": NodeConfig(
        node="node_05",
        size="large",
        temperature=0.0,
        prompt_version="v1",
        rationale="Highest-stakes reasoning; every consistency claim depends on this node.",
    ),
    "node_06": NodeConfig(
        node="node_06",
        size="mid",
        temperature=0.3,
        prompt_version="v1",
        rationale="Phrasing variety is acceptable; citations are constrained by the validator.",
    ),
    "node_10": NodeConfig(
        node="node_10",
        size="mid",
        temperature=0.0,
        prompt_version="v1",
        rationale="Compilation, not generation; provenance must survive verbatim.",
    ),
}

#: Nodes that run without any LLM call. Named explicitly so the PNG and the docs cannot
#: quietly imply a deterministic step is "AI".
DETERMINISTIC_NODES: frozenset[str] = frozenset({"node_00", "node_03", "node_07"})

#: Node 4's two agents must be routed identically. If they diverge, any disagreement they
#: produce is confounded by model choice and the central claim of the submission fails.
BLIND_AGENT_PAIR: tuple[str, str] = ("node_04a", "node_04b")


def get_node_config(node: str) -> NodeConfig:
    if node not in REGISTRY:
        raise ConfigError(f"unknown node {node!r}; known: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[node]


def resolve_model(node: str) -> str:
    """Exact model id for ``node``. Raises until Step 3 pins ``MODEL_BY_SIZE``."""
    size = get_node_config(node).size
    model_id = MODEL_BY_SIZE.get(size, "")
    if not model_id:
        raise ConfigError(
            f"model for size {size!r} is not pinned yet (node {node!r}). "
            "Pin it in Step 3 from a live GET /v1/models response."
        )
    return model_id


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"missing required environment variable {name}; see .env.example")
    return value


def check_invariants() -> None:
    """Fail loudly on configuration that would invalidate the evaluation."""
    a, b = BLIND_AGENT_PAIR
    ca, cb = get_node_config(a), get_node_config(b)
    if (ca.size, ca.temperature) != (cb.size, cb.temperature):
        raise ConfigError(
            f"{a} and {b} must share size and temperature so Node 4 disagreement is a "
            f"property of the data slices, not of model routing; "
            f"got {(ca.size, ca.temperature)} vs {(cb.size, cb.temperature)}"
        )
    for cfg in REGISTRY.values():
        if not 0.0 <= cfg.temperature <= 1.0:
            raise ConfigError(f"{cfg.node}: temperature {cfg.temperature} out of range [0, 1]")
    overlap = DETERMINISTIC_NODES & REGISTRY.keys()
    if overlap:
        raise ConfigError(f"nodes declared both deterministic and LLM-routed: {sorted(overlap)}")
