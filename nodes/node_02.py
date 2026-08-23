"""Node 2 — journal extraction, and the assertion that makes it checkable.

Every observation carries a ``source_quote``, and Node 5 later cites those quotes as the
evidence behind its narrative claims. That makes this node the point where a fabricated
detail would enter the pipeline wearing the costume of evidence — quoted, attributed, and
from then on indistinguishable from something the user actually said.

:func:`verify_source_quotes` is the whole defence, and it is one substring check:

* the quote must appear in the conversation, so invented wording is rejected outright;
* it must appear in a line the **user** spoke. Agent turns are excluded deliberately.
  This transcript is full of leading questions — "Was that in a quiet room or somewhere
  noisy?" — and a model that answers its own prompt by quoting it produces an observation
  that looks perfectly sourced while resting on nothing the user said. Restricting the
  search to user turns is what turns "quoted from the conversation" into "reported by the
  person".

Only whitespace and case are normalised. That is canonicalisation, not fuzzy matching:
neither can smuggle in a word the user never used, which is the only property the check
needs to keep.

Prompt v1 undercut this by permitting the model to "quote or closely paraphrase" the
source line — licence to produce exactly the output the validator must reject. v2 asks
for a verbatim substring instead. Both prompts stay on disk; the diff is an iteration-log
entry, and the underlying mistake is worth recording: the contract and the instructions
disagreed, and it was the instructions that were wrong.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from llm import client

NODE = "node_02"

_WHITESPACE = re.compile(r"\s+")


class SourceQuoteError(client.MalformedOutputError):
    """An observation quotes something the user did not say.

    Subclasses ``MalformedOutputError`` so it is repairable: the model gets one attempt
    with the offending quote named. A fabricated quote is malformed output in the sense
    that matters — the reply is well-formed JSON that fails the contract it claims to meet.
    """


def _canonical(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().casefold()


def verify_source_quotes(
    conversation: Sequence[dict[str, str]], observations: Sequence[dict[str, Any]]
) -> None:
    """Raise :class:`SourceQuoteError` unless every quote came from a user turn."""
    user_text = [_canonical(t["text"]) for t in conversation if t["role"] == "user"]

    for index, observation in enumerate(observations):
        quote = _canonical(observation["source_quote"])
        if any(quote in line for line in user_text):
            continue
        spoken_by_agent = any(
            quote in _canonical(t["text"]) for t in conversation if t["role"] == "agent"
        )
        reason = (
            "it is the agent's own question, not something the user said"
            if spoken_by_agent
            else "it does not appear anywhere in the conversation"
        )
        raise SourceQuoteError(
            f"observations[{index}].source_quote {observation['source_quote']!r} is not "
            f"usable evidence: {reason}. Quote a user line verbatim, or drop the "
            f"observation."
        )


def run(
    conversation: Sequence[dict[str, str]],
    *,
    run_id: str,
    transport: client.Transport,
    **kwargs: Any,
) -> dict[str, Any]:
    """Extract observations, rejecting any that are not anchored in the user's own words."""
    payload = {"conversation": list(conversation)}

    def check(parsed: Any) -> None:
        verify_source_quotes(payload["conversation"], parsed["observations"])

    return client.call_node(
        NODE,
        payload,
        run_id=run_id,
        transport=transport,
        post_validate=check,
        **kwargs,
    )
