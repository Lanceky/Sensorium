# Sensorium

**Reverie Hacks 2026 — ML Prompt Engineering track**

A multi-agent LLM workflow that turns ordinary phone behaviour — volume, brightness,
font-size and caption settings you already touch every day — into an honest,
non-diagnostic picture of how your vision and hearing are trending, without asking you to
take a test or fill out a form.

Built and evaluated as a prompt-engineering pipeline, measured against a **fair**
single-prompt baseline that receives the identical input payload.

- **Design and prompts:** [`context.md`](context.md)
- **Build order:** [`implementation.md`](implementation.md)
- **Competitive review:** [`suggestions.md`](suggestions.md)

---

## Design principle

Every node is a pure function `dict -> dict`, validated against a JSON Schema on both
ends. Getting that contract layer right first is what makes the evaluation machinery
nearly free later:

| Guarantee | How it is enforced |
|---|---|
| Node 4's two agents cannot see each other's data | `additionalProperties: false` on their input schemas, plus `assert_blind()` over the run log (Step 6) |
| No claim without evidence | `minItems: 1` on `claims[].evidence` (Step 1) + the evidence-binding validator (Step 7) |
| No number without provenance | Node 3's `figures` registry is the only origin; `figures_cited[].key` must resolve to it (Step 7) |
| No fabricated citation | `source_url` must appear in that run's Firecrawl results (Step 8) |
| Safety boundary holds under pressure | One versioned string in `prompts/refusal_boundary.v1.txt`, asserted byte-for-byte (Step 8) |
| Agent disagreement is real, not authored | Both slices derived from a declared `LatentState` by projections that never see the case kind (Step 2) |

Deterministic nodes are named as such in `sensorium/config.py` and never routed to a
model — no step pretends to be an LLM task that isn't one.

## The evaluation set

`eval/cases/` holds 12 committed cases — 3 agreeing, 3 conflicting, 2 sparse, 2 null,
2 adversarial — generated from a declared latent truth rather than written by hand:

```
LatentState  ->  device slice   (what the phone measured)
             ->  journal slice  (what the person said)
```

`user_awareness` is the pivot. A real decline the user *hasn't* noticed produces a journal
that honestly reads as "everything's fine", so the two agents genuinely conflict without
anyone scripting it.

Independence is enforced at generation time on two levels: the slices share no top-level
key, and journal text carries **no device vocabulary and no digits**. The second matters
most — a journal entry saying *"I turned the volume up to 9"* would hand the narrative
agent a measurement, so the agents would agree from shared tokens rather than from the
underlying state.

```bash
python -m eval.generator           # regenerate and write
python -m eval.generator --check   # verify committed cases match the generator
```

## Layout

```
schemas/      JSON Schema per node I/O, transcribed from context.md section 4
prompts/      Verbatim, versioned system prompts (v1 -> v2 -> v3 feeds the iteration log)
sensorium/    schemas.py (validation) · config.py (model routing) · prompts.py · runlog.py
nodes/        One module per node                      (Steps 4-8)
llm/          Featherless client + repair retry        (Step 3)
stats/        Wolfram|One client, scipy fallback       (Step 4)
retrieval/    Firecrawl client + response cache        (Step 8)
eval/         generator.py + cases/ · validators, harness, baseline  (Steps 7, 9)
runs/         Append-only call logs — the submission's evidence
tests/        Contract tests + hand-written fixtures
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill in your keys — never commit .env
```

## Run the tests

```bash
pytest
```

## Why `runs/` matters

`runs/<run_id>/calls.jsonl` records every node invocation — model, temperature, prompt
version, exact input payload, raw output, latency. It is not debug output. It is the
primary evidence artifact: Node 4's independence is proved from the *actual* payloads
recorded there, numeric fidelity is scored against the recorded Node 3 output, and the
prompt iteration log is assembled from recorded failures.

**Failed calls are logged, never swallowed.** A failure is an iteration-log entry, and it
cannot be reconstructed once a prompt file is overwritten.

## Build status

- [x] **Step 1** — contracts, config, run log
- [x] **Step 2** — latent ground-truth generator (12 cases)
- [ ] Step 3 — Featherless client and model registry *(pins `MODEL_BY_SIZE`)*
- [ ] Step 4 — Node 3 statistics engine and number registry
- [ ] Step 5 — Nodes 1 and 2
- [ ] Step 6 — Node 4 blind agents + independence proof
- [ ] Step 7 — Node 5 synthesis + validators
- [ ] Step 8 — Firecrawl retrieval + Nodes 6/10
- [ ] Step 9 — eval harness, fair baseline, results table
- [ ] Step 10 — submission artifacts

## Licence

MIT — see [`LICENSE`](LICENSE).
