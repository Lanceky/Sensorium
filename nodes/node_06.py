"""Node 6 — suggestions that can only cite what this run actually retrieved.

The citable universe is whatever `retrieval.firecrawl` returned, and it is passed in rather
than fetched here. That separation is deliberate: the citation check asks whether a URL was
in *this run's* retrieved set, and that is only a meaningful question if the set is fixed
before the model sees it and logged alongside the reply.

`source_url: null` is a legal answer and the prompt says so. A node that must cite something
will cite something, and the nearest plausible URL is exactly the citation a reader cannot
trust. Letting it decline is what makes the citations that do appear worth checking.
"""

from __future__ import annotations

from typing import Any, Sequence

from eval import validators
from llm import client
from retrieval.firecrawl import Source

NODE = "node_06"


def run(
    synthesis: dict[str, Any],
    retrieved_sources: Sequence[Source | dict[str, str]],
    *,
    run_id: str,
    transport: client.Transport,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write 2-3 grounded suggestions, citing only the supplied sources."""
    sources = [s.as_dict() if isinstance(s, Source) else dict(s) for s in retrieved_sources]
    payload = {"synthesis": synthesis, "retrieved_sources": sources}
    return client.call_node(
        NODE,
        payload,
        run_id=run_id,
        transport=transport,
        post_validate=lambda output: validators.assert_cited(output, payload),
        **kwargs,
    )
