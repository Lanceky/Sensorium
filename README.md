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

## A citation nobody could use

The retrieval layer scores passages from six health-authority pages and passes the best
excerpt from each to Node 6. On the first live run the WHO vision page's winning passage was
a bibliography entry — a Lancet DOI. It is long, link-free prose, so it cleared every
structural filter written to remove navigation and headings.

That citation would have scored perfectly. The URL was retrieved, so set membership passes;
the excerpt was real, so nothing was fabricated. It is also completely useless to the person
reading the report, which is the failure citation-checking exists to prevent — reproduced by
the citation-checking pipeline itself. Reference-shaped blocks are now dropped before
scoring, and the test that pins it uses the exact block that broke it.

Only `{url, excerpt}` ever reaches the model. Page metadata and canonical links would supply
plausible URLs that were never retrieved, and the citation check is only meaningful if the
model's sole route to a URL is the set it was handed.

## The safety metric that measured line breaks

Node 10's report must open with a fixed non-diagnostic clause, verbatim. The first live run
scored **0/12** on that check.

The model was not at fault. It had reproduced the clause exactly, as one unbroken markdown
line — and `prompts/refusal_boundary.v1.txt` is hard-wrapped at column 90. A byte-for-byte
`startswith` was testing whether a language model reproduces its author's column width. The
check now collapses whitespace runs on both sides and is relaxed in no other way: hedging it,
truncating it, paraphrasing "licensed eye or ear care professional" to "doctor", stripping
the version tag, or flipping "is not a diagnosis" to "is a diagnosis" each still fail.

Node 10's v1 prompt had a second problem, and it is the fourth of its kind in this project:
it ordered the model to reproduce the clause "(Section 5, refusal boundary) verbatim" while
never showing it the clause. Section 5 is a heading in a design document the model has never
read. The clause is now substituted into the prompt from the same file the validator compares
against, so the text shown and the text enforced cannot drift apart.

The clause is validated, not prepended. Concatenating it in code would guarantee the string
and destroy the measurement — "did the model reproduce the safety clause" cannot be answered
by an author who pasted it in. With the check corrected, all twelve reports opened correctly
on the first attempt, with zero repairs.

## What the fair fight actually showed

Three arms, 72 measurements: one prompt alone, the same prompt with the identical
validators, and the workflow. Fairness is asserted in `tests/test_harness.py`, not promised
here — the suite fails if anyone trims the baseline's inputs, downgrades its model or drops
a rule from its prompt. The baseline runs on the largest model at temperature 0 and is
handed the same engine figures with the same significance flags.

| Metric | Single prompt | + validators | Sensorium |
|---|---|---|---|
| Numeric fidelity | 1.000 (69) | 1.000 (72) | 1.000 (57) |
| Citation validity | **0.970 (33)** | 1.000 (33) | 1.000 (20) |
| Safety adherence | 1.000 (10) | 1.000 (10) | 1.000 (10) |
| Non-diagnostic language | 1.000 (10) | 1.000 (10) | 1.000 (10) |
| Evidence binding | 1.000 (63) | 1.000 (60) | 1.000 (74) |
| Abstention correctness | 1.000 (24) | 1.000 (24) | 1.000 (24) |
| Conflict surfaced | 0.333 (3) | 0.667 (3) | **1.000 (3)** |
| Quiet when they agree | 1.000 (9) | 1.000 (9) | 1.000 (9) |
| Contradiction-free (5×) | 1.000 (5) | 1.000 (5) | 1.000 (5) |
| Coverage stability (5×) | 0.900 (5) | 1.000 (5) | **0.657 (5)** |

**Most rows tie, and that is the finding.** Numeric fidelity, safety, abstention and
non-diagnostic language are 1.000 everywhere. The workflow did not buy them. The
deterministic engine and the validators did, and a single prompt handed the same figures
and the same checks reaches the same place. Anyone claiming decomposition is what makes an
LLM trustworthy should have to explain this table.

**The ablation is where the checking shows up.** The single prompt cited
`https://www.nidcd.nih.gov/health/over-counter-hearing-aids` on `sparse_01`. That page is
real — it returns 200 — and it is on-topic, which is what makes it dangerous rather than
merely wrong. The model reached into training memory for a plausible source instead of
citing what it had been given, and nothing in the text marks the difference. The same
prompt, same model and same temperature scored 1.000 the moment the validator was attached,
because the check caught it and the repair loop replaced it. One prompt, one model, one
difference: 0.970 → 1.000.

**Where the architecture actually wins is conflict detection**, at 3/3 with 9/9
specificity, against 1/3 and 2/3 for the single prompt. The conflicting cases are the ones
where the device measured a significant change and the user reported noticing nothing. The
single prompt sees both halves at once and smooths them into one coherent narrative; it is
not lying, it is doing what a fluent writer does with two facts that sit awkwardly together.
Splitting the evidence between two agents that cannot see each other's half is what keeps
the tension intact long enough for something downstream to report it.

**And the workflow loses a row.** Coverage stability is 0.657 against 0.900 and 1.000 — five
runs of one input mentioned 1, 1, 2, 2 and 7 figures. No run contradicted another and no
verdict moved, so nothing unsafe happened, but the pipeline is measurably more variable in
how much it chooses to say. More stages mean more places for scope to drift. That number is
in the table at full size.

## What 1.000 does not mean

Citation validity, safety adherence and provenance preservation all scored 1.000 across the
twelve cases. Three things are worth saying about that.

The citation validator **never fired**. Across 24 live Node 6 calls the model did not
fabricate a single URL, so the repair loop never caught one. A validator that has never
rejected anything looks exactly like a validator that cannot reject anything, so its catching
power is established by mutation instead: a trailing slash, a dropped `www.`, `http` for
`https`, a truncated path and an uppercased URL are each rejected, and deliberately loosening
the comparison is caught by the suite.

Abstentions are **excluded from the denominator**, not counted as passes. `source_url: null`
is a legal answer — it is how the node says the sources do not cover something instead of
reaching for the nearest plausible link. Scoring it as a pass would let a node reach 100% by
citing nothing at all. Four of the twelve cases used it; the 20 in the table are 20 real
citations.

And 1.000 on safety is twelve ordinary reports, not a red team. The diagnosis-baiting
pressure test belongs to the next step and is not claimed yet.

## The agent that wasn't there

Every metric above read 1.000 while one of the two blind agents was, on a third of the
cases, saying nothing at all. Nothing in the scoreboard showed it. This is the most useful
thing the evaluation harness found, and it was found by reading a failing case rather than
by tuning a prompt.

The conflict cases are built so that the device readings show a significant change while
the user's journal says *"everything felt pretty normal this week"*. That gap is the whole
point of those cases — and of the product. When the synthesis node reported no
disagreement on all three of them, the obvious move was to sharpen its prompt. Instead,
here is what the narrative agent had actually said:

> The available data consists solely of self-reported journal entries, which may provide
> insights into the subject's perceived experiences, thoughts, and feelings but lacks
> objective measurement.

with unknowns listing heart rate variability and dietary intake — signals this system does
not collect. The agent was not reasoning about a person. It was describing the shape of
its input, because its input was `{"observations": []}`.

"Everything felt normal" describes no difficulty, so Node 2 correctly extracted no
observations. And an empty list cannot tell **the user reported noticing nothing wrong**
apart from **we have no journal at all**. The synthesis node was following its instructions
exactly — one agent being silent is not a contradiction — and it was right. The evidence
had been destroyed two nodes upstream, by a schema that could only represent problems.

The fix is at the source. Node 2 now returns `no_symptom_statements` beside `observations`,
carrying the same verbatim-quote verification, so a reassurance is recorded as evidence
rather than as absence. The narrative agent is told that an empty observations list next to
a non-empty no-symptom list is a person reporting no difficulty. It now says:

> The user reported noticing no difficulty on 3 occasions this week.

against the trend agent's *"the volume setting has increased significantly over the past
three weeks"*. Those two statements are not logical opposites, which is exactly why the
synthesis node had to be taught to recognise the pattern by name: a measured change the
person has not perceived. That is not an edge case in this product. It is the reason the
product exists.

## The safety gate that turned on a version number

The refusal clause used to begin with a literal `[refusal_boundary v1]` tag. A live report
was rejected twice because the model wrote `[refusal_boundary_v1]`, with an underscore — a
one-character slip on a token carrying no safety meaning whatsoever, scored identically to
omitting the entire disclosure. The tag was also being printed to the person reading the
report.

The version now lives in the filename and in a comment line that `refusal_boundary()`
strips, so the string the model must reproduce is exactly the three sentences that do the
work. The mutation test asserting that *stripping the version tag* must fail was deleted
rather than updated: it had pinned the defect in place instead of the guarantee.

## A cached measurement is only good while the system hasn't changed

Editing the refusal clause once produced a results table in which both single-prompt arms
scored 0/10 on safety adherence. They had not regressed. Their replies were still in the
harness cache from before the edit, and were being marked wrong for failing to anticipate a
change made after they were written.

Every cached measurement now records a fingerprint of the prompt text, prompt version,
model and temperature of every node, plus the refusal clause. The cache refuses to return
an entry whose fingerprint no longer matches, and the harness reports how many stale
measurements it is discarding rather than mixing two systems into one table.

## Layout

```
schemas/      JSON Schema per node I/O, transcribed from the design doc
prompts/      Verbatim, versioned system prompts (v1 -> v2 -> v3 feeds the iteration log)
sensorium/    schemas.py (validation) · config.py (model routing) · prompts.py · runlog.py
nodes/        One module per node
llm/          Featherless client + repair retry
stats/        engine.py · deterministic trend and change-point maths
retrieval/    Firecrawl client · committed snapshot · cache (gitignored)
eval/         generator.py · cases/ · validators, harness, fair baseline
serve/        FastAPI surface over the nodes — the Android app's backend
android/      The Android client: passive signal capture and the check-in
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

## The app, and what a real phone reported

The workflow was built against generated cases, so the obvious question is whether the
signals it consumes exist outside the generator. `android/` answers it: a small Views app
that reads the four passive signals from the device it is running on and posts them to
`serve/`, which calls the same nodes, prompts, model routing and validators this
repository measures. Nothing is reimplemented in Kotlin. The app is a client of the
workflow, so a guarantee in the results table is a guarantee about what the phone shows.

**Every signal is read without a single permission prompt.** Media volume from
`AudioManager`, screen brightness and adaptive-brightness mode from `Settings.System`,
font scale from the `Configuration`, captions from `Settings.Secure`. `INTERNET` is the
only permission in the manifest and it exists to reach the service. `app_foreground`
appears in the workflow's data model but *not* in the app, because on Android it needs
`PACKAGE_USAGE_STATS` special access — the engine treats a missing signal as missing
rather than as zero, so dropping it costs a figure and corrupts nothing.

The split is on-device capture, off-device reasoning. The phone reaches the service over
`adb reverse`, so the demo needs no network, no tunnel and no public server, the handset
never holds an API key, and journal text crosses the boundary only when a person presses
the button that says it will.

**What the live run showed.** On a Samsung SM-A035F, seeded from that device's own
readings, the report carried a figure worth putting on a slide:

| Figure | Value | Verdict |
|---|---|---|
| brightness pct change | 23.09 % | significant |
| font scale pct change | 25.176 % | significant |
| **volume pct change** | **318.547 %** | **not significant** |
| caption on rate | 0.0 ratio | not testable in this window |

A 318% change reported as *not significant* is the whole argument in one row. The phone's
volume was near zero and moved by a step, so the percentage is enormous and the trend is
noise — and because significance is computed by `stats/engine.py` and not by a model, no
amount of persuasive phrasing downstream can promote it to a finding. Node 5 wrote "there
is no detected change in volume settings" while that 318% sat on screen above it.

The fourth row is a distinction this repository had to fix twice. The engine reports
significance as **three** states — true, false, and *could not test* — and sets the last
when a window cannot answer whether a slope differs from zero. The first version of
`serve/api.py` wrote `bool(figure.get("significant", False))`, which quietly turned *we
could not test this* into *we tested this and it is noise*. That is a stronger claim than
the data supports, made by a display layer, which is exactly the failure the project
exists to prevent. `tests/test_serve.py` now pins all three states, and reverting the fix
fails two tests.

### Running it

```bash
python -m serve.api                    # from the repo root; listens on 127.0.0.1:8765
adb reverse tcp:8765 tcp:8765          # the phone's loopback, forwarded to this machine
gradle -p android assembleDebug
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
```

`python -m serve.smoke` runs the same payload the phone sends, without the phone, and is
the fastest way to check the service before a demo. It is a smoke test and not a
measurement: claims come from `eval/`, and one live run is not evidence.

`android/local.properties` is machine-local and gitignored; create it with
`sdk.dir=/path/to/Android/Sdk`.

### Seeded history is labelled, not laundered

A trend needs weeks and a demo cannot wait weeks, so the app can seed four weeks of
history. Three things keep that honest. The seeded series is anchored on the device's
**real current readings** and drifts backwards from them, so it ends where the phone
actually is. Noise is added deliberately, because a perfect fit makes a significance test
a statement about the generator rather than the data. And seeded samples are stored under
their own flag, counted separately, and shown separately on screen — `2 recorded · 16
seeded` — so a seeded number is never mistaken for a measured one.

The drift selector offers a flat history as well as a rising one, so the demo can show the
system declining to report a trend. A workflow that only ever has a finding to announce
has not been shown capable of saying there isn't one.

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
