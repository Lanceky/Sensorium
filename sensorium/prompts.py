"""Prompt loading.

Prompts live in ``prompts/`` as versioned plain text, extracted verbatim from
``context.md`` section 4. They are files rather than string literals so that the
iteration log (context.md section 10) can diff v1 -> v2 -> v3, and so the documentation
and the running code cannot drift apart.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptError(Exception):
    """Raised when a requested prompt version does not exist."""


@lru_cache(maxsize=None)
def load_prompt(node: str, version: str = "v1") -> str:
    """Return the system prompt for ``node`` at ``version``, boundary substituted.

    A prompt may contain the token ``{refusal_boundary}``, which is replaced with the
    contents of ``refusal_boundary.v1.txt``. Node 10's v1 prompt told the model to open
    "with the fixed disclosure clause (Section 5) verbatim" while never showing it the
    clause — Section 5 is a heading in a design document the model has never read. It was
    being asked to reproduce a string it had no way to know.

    Substituting rather than pasting keeps one source of truth. The validator compares
    against that same file, so the text the model is shown and the text it is checked
    against cannot drift apart: editing the clause updates both or neither.
    """
    path = PROMPT_DIR / f"{node}.{version}.txt"
    if not path.exists():
        available = ", ".join(sorted(p.name for p in PROMPT_DIR.glob(f"{node}.*.txt")))
        raise PromptError(f"no prompt at {path.name}; available for {node}: {available or 'none'}")
    text = path.read_text(encoding="utf-8").strip()
    if "{refusal_boundary}" in text:
        text = text.replace("{refusal_boundary}", refusal_boundary())
    return text


@lru_cache(maxsize=1)
def refusal_boundary() -> str:
    """The single reused, versioned non-diagnostic clause (context.md section 5).

    Node 10 must open with this string byte-for-byte; Metric 2 scores its presence under
    adversarial pressure. Because it is one string in one file, "did the safety boundary
    hold" is a substring check rather than a judgement call.
    """
    return (PROMPT_DIR / "refusal_boundary.v1.txt").read_text(encoding="utf-8").strip()


def prompt_versions(node: str) -> list[str]:
    """All versions on disk for ``node``, e.g. ``["v1", "v2"]``."""
    return sorted(p.name.split(".")[-2] for p in PROMPT_DIR.glob(f"{node}.*.txt"))
