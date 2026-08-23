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

## Quotes as evidence

Node 2 turns a check-in conversation into observations, each carrying a `source_quote`
that Node 5 later cites as its evidence. That makes Node 2 the point where an invented
detail would enter the pipeline already dressed as testimony — quoted, attributed, and
from then on indistinguishable from something the person actually said.

The defence is one substring check with a deliberate restriction: the quote must appear in
a line the **user** spoke. Agent turns are excluded because the transcript is full of
leading questions — *"Was that in a quiet room or somewhere noisy?"* — and a model that
answers its own prompt by quoting it produces an observation that looks perfectly sourced
while resting on nothing the user said.

It runs inside the repair loop, so a bad quote costs one retry naming the offending text
rather than failing the run, and both attempts stay in the log.

Writing that check surfaced a conflict in our own prompt: v1 told the model it could
"quote or closely paraphrase" the source line, which is licence to produce exactly what
the validator has to reject. v2 asks for a verbatim substring. Measured against v1 on the
same twelve cases, it changed nothing — same extractions, same zero repair calls — so it
is kept as a closed contradiction rather than claimed as an improvement. Both versions
stay on disk; the negative result is in the iteration log.

## Prompts state their contract

The system prompts in `prompts/` are transcribed verbatim from the design doc and describe
behaviour: tone, how many questions, what never to say. None of them described an output
shape, and the first live call showed why that matters — Node 1 returned a well-formed
check-in question as plain prose, then, asked to repair it, invented a schema by echoing
the input keys back (`{"turn": 1, "user_reply": null, "question": ...}`, where the contract
says `message`). It was guessing, because nothing had told it. Both attempts are preserved
verbatim in `evidence/node_01-missing-output-contract/`.

The output contract is now generated from the JSON Schema that validates the reply and
appended to every system prompt, with `$ref`s inlined so enum members are actually visible.
Two of the three prompt defects found so far were a prompt disagreeing with its own
validator, so the fix is structural rather than editorial: derive the instruction from the
schema, and the text the model is held to cannot drift from the check that holds it there.

## Two blind agents, and the proof they are blind

Node 4 runs the same question past two agents at once. Agent A sees the statistics with no
story attached; Agent B sees the user's own words with no numbers attached. Neither is told
the other exists. When they land in the same place, that agreement carries information,
because neither could have reached it by reading the other's evidence.

That argument is worth exactly as much as the blindness is real, so `eval/independence.py`
measures it on every run: it rebuilds the full context window each agent actually received
— system prompt, generated contract and payload — and searches each for the other's
fingerprints.

The obvious way to write that check is worthless:

```python
assert not any(term in payload_a for term in journal_terms)
```

If `journal_terms` comes back empty, that line passes while looking at nothing, and it
passes brightly, in green. So every check here is two-sided: the terms that must be absent
from one agent must be **found in the other**, and `assert_blind` refuses to return a
passing report if the fingerprint list is empty or its positive control never fired.

Across all twelve cases: **no journal phrase reached Agent A, no trend figure reached Agent
B, and the controls fired.** The journal-side control fires on exactly the six cases where
Node 2 extracted at least one observation — a correspondence predicted independently by the
extraction counts, so the two measurements corroborate each other.

There is a schema half to this too — `node_04a.input.json` sets `additionalProperties:
false`, so journal text attached as a new field fails validation before any request is sent
— but it is worth being precise about what that does and does not buy. It closes the
accidental route. It cannot close every route, because some permitted fields are free-form
strings: every figure carries a `unit`, and a journal sentence sitting in a `unit` is a
perfectly schema-valid payload. The runtime check is what catches that, and the test suite
pins the division of labour by running the same smuggled payload past both.

The clearest evidence is in the replies themselves. On the conflict cases Agent B writes
that "the available data consists solely of self-reported journal entries" and lists device
data among its unknowns, while Agent A reasons about volume and brightness and never
mentions a journal. The requests are committed under `evidence/node_04-blind-agents/`, and
the test suite reads them from there, so the proof re-runs on every `pytest` rather than
resting on a claim about a terminal I once had open.

The first version of this check reported a leak on all twelve cases. Every one was false:
it searched for rounded copies of each figure, which turned `caption_on_rate` of 1.0 into
the term `"1"`, which duly matched `"minLength": 1` in the schema contract every agent is
sent. A one-character fingerprint is not a fingerprint. Rounding tolerance belongs to Node
5's output fidelity, not to an input window that contains the engine's exact values or
nothing — and the same run demonstrates it there, with Agent A writing "38.46%" for a
`volume_pct_change` of `38.458`. Both the degenerate terms and the three-word phrase floor
are now pinned by mutation tests; restoring either mistake turns the suite red.

## Numbers come from the engine, not from a model that saw one

Node 5 is the first node allowed to see both slices, and everything downstream inherits
what it asserts. Three checks run inside its repair loop, so a violation is handed back as
a correction rather than counted as a failure at the end. The same functions score the run
in `eval/validators.py`, so the results table is produced by the code that did the
enforcing.

| Metric | What it checks | Result |
| --- | --- | --- |
| 1 — numeric pass-through | every number traces to `stats/engine.py` | 1.000 over 104 numbers |
| 2 — abstention | `insufficient_data` matches the significance verdict | 1.000 over 24 checks |
| 3 — evidence binding | every claim cites a field that resolves | 1.000 over 90 references |

The obvious way to write Metric 1 is wrong, and it took a live run to see why. "Never emit
a number that is not in the supplied data" sounds airtight — but Node 4's agent prose *is*
supplied data, and those agents round. Agent A wrote "38.46%" for a `volume_pct_change` of
`38.458`; the exact value appears nowhere in its reply. A Node 5 citing 38.46 would satisfy
the obvious rule while stating a number no engine computed, and the pipeline would report
100% compliance while laundering a rounding through a language model. So the numeric
authority is `trend_data.figures` alone. An agent's prose is an opinion about numbers, not
a source of them.

## The engine knew, and wasn't saying

The first version of Node 5 narrated `-0.459%` as a decline. It was not hallucinating:
nothing it received distinguished noise from signal, so it described every figure it was
given. The cause was upstream — `linregress` returns a p-value and the engine was throwing
it away, returning only the slope.

Surfacing it is a one-line change with a disproportionate result. Requiring `p < 0.05`
recovers the generator's hidden latent state **exactly, on all twelve cases**: both null
cases contain no significant figure, and all ten cases carrying a real decline contain at
least one. `insufficient_data` is therefore not a judgement call but a function of the
statistics, and Metric 2 checks it as an equality.

Both directions of that equality earn their place. The dangerous failure is narrating
noise, and that is what the first version checked — but the live run then produced the
opposite defect and walked straight past it. On `agree_02` the node wrote "there is a
significant increase in brightness settings", cited the figure behind it, and set
`insufficient_data` to true in the same reply. Not caution: a document that contradicts
itself, of the kind Node 10 would hand to a clinician. Checking one direction had also left
the metric with a denominator of 4 instead of 24.

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
runs/         Append-only call logs — the submission's evidence (gitignored)
evidence/     The run logs the iteration log cites, kept so they can be read
tests/        Contract tests + hand-written fixtures
```

## Models

Every node's routing lives in `sensorium/config.py`, which is the single source of truth
for the workflow diagram and this table.

| Tier | Model | Nodes |
|---|---|---|
| small | `Qwen/Qwen2.5-7B-Instruct` | 0.5, 1 |
| mid | `Qwen/Qwen2.5-32B-Instruct` | 2, 4A, 4B, 6, 10 |
| large | `Qwen/Qwen2.5-72B-Instruct` | 5 |

Nodes 0, 3 and 7 call no model at all and are listed in `DETERMINISTIC_NODES`, so the
diagram cannot quietly imply that arithmetic is reasoning.

Ids were pinned from a live `/v1/models` response and then **verified with a real
completion**, because listed and callable are different things: every `meta-llama/*` and
`google/gemma-*` id returns `403 This model is gated` on this account, so the Llama-3.1
models the design doc originally named were never available to cite. `QwQ-32B` was
callable and still rejected — it emitted its JSON object twice in one reply, which is the
reasoning-model failure mode structured output cannot absorb.

All three tiers are one family on purpose. Node 4's two agents must differ only in the
data they see, so sharing a tokenizer and chat template keeps an observed difference
attributable to size rather than to a change of training lineage.

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
