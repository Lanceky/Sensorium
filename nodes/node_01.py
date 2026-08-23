"""Node 1 — the conversational check-in, and the turn cap that is actually enforced.

The prompt says "never ask more than 2 questions". That sentence is a request, not a
control: the adversarial cases exist precisely because a model under pressure will
volunteer a third question, or a diagnosis, when the only thing stopping it is its own
instructions. So the cap lives in :func:`check_in`'s loop, where no wording can move it,
and the prompt's version of the rule becomes a redundancy rather than the mechanism.

The node also runs twice per session, which makes it the one place the run log's
one-call-per-node assumption breaks. Both turns are routed identically — same model, same
prompt, same temperature, because they are the same node — but each is logged under its
own key so later checks never have to guess which turn they are reading.
"""

from __future__ import annotations

from typing import Any, Sequence

from llm import client
from sensorium import schemas

NODE = "node_01"

#: Hard ceiling on agent questions. Enforced by the loop below, not by the prompt.
MAX_TURNS = 2


def run(
    user_reply: str | None,
    turn: int,
    *,
    run_id: str,
    transport: client.Transport,
    **kwargs: Any,
) -> dict[str, Any]:
    """One check-in turn. ``turn`` is 1 for the opener, 2 for the adaptive follow-up."""
    payload = {"user_reply": user_reply, "turn": turn}
    return client.call_node(
        NODE,
        payload,
        run_id=run_id,
        transport=transport,
        log_as=f"{NODE}.turn{turn}",
        **kwargs,
    )


def check_in(
    *,
    run_id: str,
    transport: client.Transport,
    user_replies: Sequence[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the capped exchange and return a ``node_02.input.json`` conversation.

    ``user_replies`` is the user side. Supplying it from stored journal text is what lets
    the evaluation harness run unattended; an interactive caller passes replies as they
    are typed. Either way the transcript that comes out is the same shape, so the harness
    exercises the same code path a real session would.

    Stops early when the model sets ``done``, when the user has nothing more to say, or at
    :data:`MAX_TURNS` — whichever comes first. The last of those is the only one that
    cannot be talked out of.
    """
    conversation: list[dict[str, str]] = []
    reply: str | None = None

    for turn in range(1, MAX_TURNS + 1):
        result = run(reply, turn, run_id=run_id, transport=transport, **kwargs)
        conversation.append({"role": "agent", "text": result["message"]})

        answered = len(_user_turns(conversation))
        if answered >= len(user_replies):
            break
        reply = user_replies[answered]
        conversation.append({"role": "user", "text": reply})

        # Checked after the reply is recorded: `done` means the agent will not ask again,
        # not that an answer already given should be dropped from the transcript.
        if result["done"]:
            break

    slice_ = {"conversation": conversation}
    schemas.validate("node_02.input.json", slice_)
    return slice_


def _user_turns(conversation: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    return [t for t in conversation if t["role"] == "user"]
