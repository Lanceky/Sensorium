"""Featherless client and the single calling convention shared by every LLM node.

Featherless is OpenAI-compatible, so the provider layer is a base-URL swap. The value
here is not the HTTP call, it is what wraps it:

* the node's own prompt and routing come from ``sensorium.config`` and ``sensorium.prompts``,
  so no node can quietly use a different model or an unversioned prompt;
* input is validated before the call and output after it, so a node either returns
  contract-valid data or raises — downstream nodes never receive a half-shaped object;
* one repair retry hands the model the *complete* list of validation failures, and the
  rejected attempt stays in the run log, because that rejection is an iteration-log entry.

Transport failures and malformed output are deliberately different exception types. A
provider 500 is not repairable by telling the model its JSON was bad, and retrying it as
if it were would corrupt the iteration log with failures the prompt never caused.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Callable, Protocol

from sensorium import config, prompts, runlog, schemas

#: Passed to the provider so temperature-0 nodes are reproducible run to run. Temperature
#: alone does not guarantee this: batched inference can reorder floating-point reductions,
#: so identical input can still yield different output. Metric 4 scores that stability, so
#: it is worth asking for. Providers that reject the parameter degrade gracefully below.
DEFAULT_SEED = 20260824

REPAIR_INSTRUCTION = (
    "Your previous reply failed validation:\n{error}\n\n"
    "Return only the corrected JSON object. No prose, no explanation, no code fences."
)

#: Appended to every system prompt, generated from the schema that validates the reply.
#: The prompts in ``prompts/`` are transcribed verbatim from the design doc and describe
#: *behaviour* -- tone, question count, what never to say. None of them stated an output
#: shape, because the design doc declared schemas in a separate section that never reached
#: the model. The first live call returned a perfectly good check-in question as plain
#: prose, and the repair attempt then invented a schema by echoing the input keys back.
#:
#: Generating this from the schema rather than writing it into each prompt keeps one
#: source of truth: the text the model is held to and the validator that holds it there
#: cannot disagree, however either changes.
CONTRACT_INSTRUCTION = (
    "\n\nRespond with a single JSON object that validates against this JSON Schema:\n"
    "{schema}\n\n"
    "Output only that JSON object. No prose before or after it, no code fences, and no "
    "keys beyond those in the schema."
)

#: Node 4's two agents share one output schema on purpose: identical shape is what makes
#: their conclusions comparable, and comparability is what turns disagreement into a
#: measurable signal instead of a formatting difference.
OUTPUT_SCHEMA_OVERRIDES = {
    "node_04a": "node_04.output.json",
    "node_04b": "node_04.output.json",
}

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class LLMError(Exception):
    """Base class for every failure in the LLM layer."""


class TransportError(LLMError):
    """The provider call itself failed. Not repairable by re-prompting."""


class MalformedOutputError(LLMError):
    """The model replied, but no JSON could be recovered from it. Repairable."""


class Transport(Protocol):
    """Anything that can turn messages into text.

    A protocol rather than a concrete class so the evaluation harness and the tests can
    drive every node without a network or an API key.
    """

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        seed: int | None = None,
    ) -> str: ...


class FeatherlessTransport:
    """OpenAI-compatible client pointed at Featherless."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 90.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise TransportError(
                "the 'openai' package is required for live calls; pip install -r requirements.txt"
            ) from exc

        self.base_url = base_url or config.FEATHERLESS_BASE_URL
        self._client = OpenAI(
            api_key=api_key or config.require_env("FEATHERLESS_API_KEY"),
            base_url=self.base_url,
            timeout=timeout,
        )
        self._seed_supported = True

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        seed: int | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if seed is not None and self._seed_supported:
            kwargs["seed"] = seed

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if "seed" in kwargs and "seed" in str(exc).lower():
                # Reproducibility is worth asking for but not worth failing over.
                self._seed_supported = False
                return self.complete(
                    model=model, messages=messages, temperature=temperature, seed=None
                )
            raise TransportError(f"{model}: {exc}") from exc

        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise TransportError(f"{model} returned an empty completion")
        return content

    def list_models(self) -> list[str]:
        """Exact model ids available on this account.

        The only trustworthy source for ``config.MODEL_BY_SIZE``: an id copied from
        documentation may not exist here, and the submission has to name the models it
        actually ran on.
        """
        try:
            return sorted(model.id for model in self._client.models.list().data)
        except Exception as exc:
            raise TransportError(f"GET {self.base_url}/models failed: {exc}") from exc


def input_schema_for(node: str) -> str | None:
    """Input contract for ``node``, or ``None`` for nodes that take no structured input."""
    name = f"{node}.input.json"
    return name if name in schemas.schema_names() else None


def output_schema_for(node: str) -> str:
    name = OUTPUT_SCHEMA_OVERRIDES.get(node, f"{node}.output.json")
    if name not in schemas.schema_names():
        raise LLMError(f"no output schema {name!r} for node {node!r}")
    return name


def extract_json(text: str) -> Any:
    """Recover a JSON value from a model reply.

    Tried most-specific first: a fenced block, then the whole reply, then the widest
    brace-delimited span. The lenient fallback exists because a chatty model should not
    cost a retry, but it never hides anything — the raw reply is written to the run log
    either way, so what the model actually said stays recoverable.
    """
    if not isinstance(text, str) or not text.strip():
        raise MalformedOutputError("model returned empty output")

    stripped = text.strip()
    candidates = [match.group(1) for match in _FENCE.finditer(stripped)]
    candidates.append(stripped)
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = stripped.find(opener), stripped.rfind(closer)
        if start != -1 and end > start:
            candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate.strip())
        except ValueError:
            continue
    raise MalformedOutputError(f"no parseable JSON in model output: {stripped[:200]!r}")


def call_node(
    node: str,
    payload: Any,
    *,
    run_id: str,
    transport: Transport,
    seed: int | None = DEFAULT_SEED,
    max_repairs: int = 1,
    post_validate: Callable[[Any], None] | None = None,
    log_as: str | None = None,
) -> Any:
    """Run one LLM node and return its validated output.

    ``post_validate`` carries the checks JSON Schema cannot express — that a quote really
    appears in the source, that a cited figure really exists in Node 3's registry. It runs
    *inside* the repair loop and must raise ``MalformedOutputError`` on failure, because a
    fabricated quote is malformed output in every sense that matters here. Putting it
    inside means a semantic violation earns the same repair attempt a syntax error does,
    and lands in the run log identically.

    ``log_as`` renames the call in the run log without changing routing, for the one node
    that legitimately runs more than once per session. Node 1's two turns must share a
    model, prompt and temperature — that is what makes them one node — but they need
    distinct log keys, because ``runlog.load_call`` treats repeated entries under one name
    as an ambiguity it refuses to guess at.

    Raises ``schemas.SchemaError`` if *we* built a bad payload, ``TransportError`` if the
    provider failed, and ``LLMError`` if the model could not produce contract-valid output
    within ``max_repairs`` retries. Every attempt is logged before it is judged.
    """
    cfg = config.get_node_config(node)
    model = config.resolve_model(node)
    system_prompt = prompts.load_prompt(node, cfg.prompt_version)
    log_node = log_as or node

    in_schema = input_schema_for(node)
    if in_schema is not None:
        # Not the model's mistake, so it happens before the call and is never repaired.
        schemas.validate(in_schema, payload)
    out_schema = output_schema_for(node)
    system_prompt += CONTRACT_INSTRUCTION.format(schema=schemas.contract_text(out_schema))

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, sort_keys=True, ensure_ascii=False)},
    ]

    failure: Exception | None = None
    for attempt in range(1, max_repairs + 2):
        with runlog.timed_call(
            run_id, log_node, model, cfg.temperature, cfg.prompt_version, payload, attempt
        ) as slot:
            raw = transport.complete(
                model=model,
                messages=messages,
                temperature=cfg.temperature,
                seed=seed,
            )
            slot["raw_output"] = raw
            try:
                parsed = extract_json(raw)
                schemas.validate(out_schema, parsed)
                if post_validate is not None:
                    post_validate(parsed)
            except (MalformedOutputError, schemas.SchemaError) as exc:
                failure = exc
                slot["error"] = f"{type(exc).__name__}: {exc}"
            else:
                return parsed

        if attempt > max_repairs:
            raise LLMError(
                f"{node}: no contract-valid output after {attempt} attempts: {failure}"
            ) from failure

        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": REPAIR_INSTRUCTION.format(error=failure)},
        ]

    raise AssertionError("unreachable")  # pragma: no cover


def main() -> int:
    parser = argparse.ArgumentParser(description="Featherless model discovery.")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="print exact model ids available on this account, for pinning MODEL_BY_SIZE",
    )
    parser.add_argument("--filter", default="", help="case-insensitive substring filter")
    args = parser.parse_args()

    if not args.list_models:
        parser.print_help()
        return 2

    try:
        model_ids = FeatherlessTransport().list_models()
    except (TransportError, config.ConfigError) as exc:
        print(f"error: {exc}")
        return 1

    needle = args.filter.lower()
    matches = [m for m in model_ids if needle in m.lower()]
    for model_id in matches:
        print(model_id)
    print(f"\n{len(matches)} of {len(model_ids)} models", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
