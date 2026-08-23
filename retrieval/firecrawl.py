"""Firecrawl retrieval, and the disk cache that keeps a demo from depending on a network.

Node 6 may cite nothing it was not handed, so what this module returns *is* the citable
universe for that run. Two consequences shape the design.

**Only ``{url, excerpt}`` pairs cross the boundary.** The model never receives a whole page.
That is not a token-budget decision: a node given twelve thousand words and asked to cite
one URL is being invited to summarise from memory and attach a plausible link afterwards.
Handing it the passage it must ground itself in makes the citation check meaningful, because
the excerpt is short enough that a reader can see for themselves whether the suggestion
follows from it.

**Every retrieval is cached to disk and every run logs what it received.** The citation
validator asks whether a cited URL was in *that run's* retrieved set, which is only an
honest question if the set is recorded rather than reconstructed later from a fresh fetch
that might return something different. The cache is also insurance: Firecrawl being slow or
rate-limited during a demo recording must not be able to invalidate a result that has
already been demonstrated live.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from sensorium.config import REPO_ROOT, require_env

FIRECRAWL_URL = "https://api.firecrawl.dev/v2/scrape"

#: Full page text, gitignored. Large, regenerable, and not ours to redistribute.
CACHE_DIR = REPO_ROOT / "retrieval" / "cache"

#: The `{url, excerpt}` pairs a recorded run actually used. Small enough to commit, which
#: is what lets the eval harness and the demo run from a clean clone with no network.
SNAPSHOT_PATH = REPO_ROOT / "retrieval" / "snapshot.json"

#: Words too common to indicate relevance; scoring on them ranks boilerplate first.
_STOPWORDS = frozenset(
    "a an and are as at be by для for from has have how i in is it its of on or that the to "
    "was were what when which who will with you your this these those can may".split()
)

_WORD = re.compile(r"[a-z][a-z'-]+")

#: Public, non-commercial guidance. Government and WHO pages are chosen deliberately: they
#: are stable, freely readable, and a clinician receiving the Node 10 report can check them
#: without a paywall.
SOURCES: dict[str, tuple[str, ...]] = {
    "hearing": (
        "https://www.nidcd.nih.gov/health/age-related-hearing-loss",
        "https://www.nidcd.nih.gov/health/noise-induced-hearing-loss",
        "https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss",
    ),
    "vision": (
        "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/refractive-errors",
        "https://www.nei.nih.gov/learn-about-eye-health/healthy-vision/keep-your-eyes-healthy",
        "https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment",
    ),
}

#: Long enough to carry an actual instruction, short enough that a reader checks it rather
#: than skims it.
EXCERPT_CHARS = 700


class RetrievalError(Exception):
    """Firecrawl could not be reached, or returned nothing usable."""


@dataclass(frozen=True)
class Source:
    """One retrieved passage, in the only shape a node ever sees."""

    url: str
    excerpt: str

    def as_dict(self) -> dict[str, str]:
        return {"url": self.url, "excerpt": self.excerpt}


def _cache_path(url: str) -> Path:
    return CACHE_DIR / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.json"


def fetch_markdown(url: str, *, refresh: bool = False, timeout: int = 90) -> str:
    """Page text for ``url``, from disk when available.

    ``refresh`` forces a live call, which is how the capability is demonstrated rather than
    merely claimed. Everything else reads the cache, so a run costs no network and cannot
    fail halfway through for reasons that have nothing to do with the pipeline.
    """
    path = _cache_path(url)
    if not refresh and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["markdown"]

    import requests

    key = require_env("FIRECRAWL_API_KEY")
    response = requests.post(
        FIRECRAWL_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RetrievalError(f"{url}: Firecrawl returned {response.status_code}")

    body = response.json()
    markdown = (body.get("data") or {}).get("markdown") or ""
    if not markdown.strip():
        raise RetrievalError(f"{url}: Firecrawl returned no page text")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"url": url, "markdown": markdown, "fetched_at": time.time()}),
        encoding="utf-8",
    )
    return markdown


#: Blocks that are references rather than guidance. A bibliography is long, link-free prose,
#: so the length and markup filters wave it straight through — the WHO vision page scored a
#: Lancet citation as its best passage on the first live run. A cited URL whose excerpt is
#: somebody else's DOI passes a set-membership check and tells the reader nothing, which is
#: the precise failure this module's docstring warns about, produced by this module.
_REFERENCE_MARKERS = re.compile(
    r"\bdoi:|\bet al\b|\bLancet\b|\bPubMed\b|\bvol\.\s*\d|\bpp\.\s*\d"
    r"|\b(19|20)\d{2}\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b",
    re.IGNORECASE,
)

#: A numbered reference list entry: "1. Author, Title..." or the escaped "1\. " markdown
#: emits when it does not want the line treated as an ordered list.
_REFERENCE_OPENER = re.compile(r"^\d+\s*\\?\.\s")


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _is_guidance(block: str) -> bool:
    """Whether a block is advice a person could act on, rather than apparatus."""
    if len(block) < 120:
        return False
    if block.count("](") > 3 or block.lstrip().startswith(("#", "|", "-", "*")):
        return False
    if _REFERENCE_OPENER.match(block.lstrip()):
        return False
    return not _REFERENCE_MARKERS.search(block)


def _passages(markdown: str) -> list[str]:
    """Prose blocks, with navigation furniture and bibliographies dropped.

    Markdown from a health site is mostly links, headings, menus and references. Scoring
    those against a query produces an excerpt that cites a page correctly and says nothing.
    """
    blocks = [re.sub(r"\s+", " ", b).strip() for b in re.split(r"\n\s*\n", markdown)]
    return [b for b in blocks if _is_guidance(b)]


def best_excerpt(markdown: str, query: str) -> str:
    """The passage that best matches ``query``, trimmed to a checkable length.

    Relevance is term overlap normalised by passage length, so a long block does not win by
    containing everything. Falls back to the first prose block when nothing overlaps, which
    is honest: it yields a weak excerpt rather than pretending to a match that is not there.
    """
    passages = _passages(markdown)
    if not passages:
        return markdown[:EXCERPT_CHARS].strip()

    wanted = _terms(query)
    if wanted:
        scored = sorted(
            passages,
            key=lambda p: len(wanted & _terms(p)) / (len(_terms(p)) ** 0.5 or 1),
            reverse=True,
        )
    else:
        scored = passages
    return scored[0][:EXCERPT_CHARS].strip()


def retrieve(
    modalities: Iterable[str],
    query: str,
    *,
    refresh: bool = False,
) -> list[Source]:
    """The citable universe for one run: one excerpt per source, for the relevant modalities.

    Deliberately returns a list rather than writing anywhere. The caller logs it into the run
    record, because "what was retrieved during this run" has to be a fact about the run and
    not a lookup performed afterwards.
    """
    wanted = [m for m in dict.fromkeys(modalities) if m in SOURCES]
    if not wanted:
        wanted = sorted(SOURCES)

    sources: list[Source] = []
    for modality in wanted:
        for url in SOURCES[modality]:
            try:
                markdown = fetch_markdown(url, refresh=refresh)
            except RetrievalError:
                # One unreachable page must not empty the citable set; the run continues
                # with fewer sources, and Node 6 is entitled to cite nothing.
                continue
            sources.append(Source(url=url, excerpt=best_excerpt(markdown, query)))
    return sources


def load_snapshot() -> list[Source]:
    """The committed `{url, excerpt}` set, so a clean clone can run with no network."""
    if not SNAPSHOT_PATH.exists():
        raise RetrievalError(
            f"no retrieval snapshot at {SNAPSHOT_PATH}; run `python -m retrieval.firecrawl --refresh`"
        )
    raw = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    return [Source(url=s["url"], excerpt=s["excerpt"]) for s in raw["sources"]]


def write_snapshot(sources: Sequence[Source]) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(
        json.dumps(
            {"retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "sources": [s.as_dict() for s in sources]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:  # pragma: no cover - operational entry point
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve guidance sources via Firecrawl.")
    parser.add_argument("--refresh", action="store_true", help="bypass the cache and fetch live")
    parser.add_argument(
        "--query",
        default="hearing loss vision changes volume brightness screen use next steps",
        help="what the excerpt should be about",
    )
    args = parser.parse_args()

    sources = retrieve(SOURCES, args.query, refresh=args.refresh)
    write_snapshot(sources)
    for source in sources:
        print(f"{source.url}\n  {source.excerpt[:160]}...\n")
    print(f"{len(sources)} sources written to {SNAPSHOT_PATH}")
    return 0 if sources else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
