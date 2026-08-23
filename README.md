# Sensorium

**Reverie Hacks 2026 — ML Prompt Engineering track**

A multi-agent LLM workflow that turns ordinary phone behaviour — volume, brightness,
font-size and caption settings you already touch every day — into an honest,
non-diagnostic picture of how your vision and hearing are trending, without asking you to
take a test or fill out a form.

Built and evaluated as a prompt-engineering pipeline, measured against a **fair**
single-prompt baseline that receives the identical input payload.

---

## Design principle

Every node is a pure function `dict -> dict`, validated against a JSON Schema on both
ends. Getting that contract layer right first is what makes the evaluation machinery
nearly free later:

| Guarantee | How it is enforced |
|---|---|
| Node 4's two agents cannot see each other's data | `additionalProperties: false` on their input schemas, plus `assert_blind()` over the run log |
| No claim without evidence | `minItems: 1` on `claims[].evidence`, plus the evidence-binding validator |
| No number without provenance | Node 3's `figures` registry is the only origin; every `figures_cited[].key` must resolve to it |
| No fabricated citation | `source_url` must appear in that run's retrieval results |
| Safety boundary holds under pressure | One versioned string in `prompts/refusal_boundary.v1.txt`, asserted byte-for-byte |
| Agent disagreement is real, not authored | Both slices derived from a declared `LatentState` by projections that never see the case kind |

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

## The number registry

Node 3 is deterministic — no model is consulted — and it is the only place in the system
where a number may originate. It emits an addressable registry rather than prose:

```json
"volume_pct_change": { "value": 29.059, "unit": "%", "method": "linear_regression" }
```

Every figure cited downstream must resolve to a key here, which is what makes numeric
fidelity an exact lookup instead of a judgement call.

Two choices in that engine are worth naming, because both guard against inventing
findings:

- **Percent change is read off the fitted line, not the endpoints.** With six to eight
  samples per signal, `(last - first) / first` is at the mercy of whichever two readings
  land at the window edges.
- **A change point must survive a significance test.** Two lines always fit better than
  one, so a residual-improvement threshold reports breaks in pure noise — measured, it
  fired 15 times across the 12 cases, including on the null cases that exist to catch
  exactly that. A nested-model F-test, Bonferroni-corrected for having tried every split
  position, brings it to one detection, with a **4.7% false-positive rate on noise against
  a nominal 5%** while still recovering **96%** of genuine breaks.

## Layout

```
schemas/      JSON Schema per node I/O, transcribed from the design doc
prompts/      Verbatim, versioned system prompts (v1 -> v2 -> v3 feeds the iteration log)
sensorium/    schemas.py (validation) · config.py (model routing) · prompts.py · runlog.py
nodes/        One module per node
llm/          Featherless client + repair retry
stats/        engine.py · deterministic trend and change-point maths
retrieval/    Firecrawl client + response cache
eval/         generator.py · cases/ · validators, harness, fair baseline
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

## Licence

MIT — see [`LICENSE`](LICENSE).
