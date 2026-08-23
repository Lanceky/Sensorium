# Preserved runs

`runs/` is gitignored: run logs are regenerable and get large. The runs kept here are the
ones the iteration log (`context.md` §10) actually cites, because a cited artifact nobody
can open is not evidence.

## `node_01-missing-output-contract/`

The first live call ever made by this pipeline, and the failure that produced the output
contract now appended to every system prompt.

Read `calls.jsonl` top to bottom:

- **attempt 1** — `raw_output` is a perfectly good check-in question in plain prose. The
  prompt described behaviour and never mentioned JSON, so the model had no reason to
  return any. Rejected with `MalformedOutputError`.
- **attempt 2** — handed the validation error, the model invented a schema by echoing the
  input keys back: `{"turn": 1, "user_reply": null, "question": ...}`. Note `question`
  where the contract says `message`. Rejected with `SchemaError`.

Both attempts are here rather than only the failure that stopped the run, which is the
point of logging every attempt before judging it.

Fixed by generating the output contract from the JSON Schema that validates the reply
(`sensorium.schemas.contract_text`) and appending it to the system prompt. Nodes 1 and 2
then ran 12/12 with zero repairs.

## `node_04-blind-agents/`

The live twelve-case Node 4 run behind the independence claim in `context.md` §4: every
request sent to both blind agents, plus `cases.json` with the Node 3 trends, the Node 2
observations and each agent's reply.

This directory is not an attachment. `tests/test_independence.py` reads it directly, so
the blindness proof is re-run on every `pytest` invocation against the requests as they
were actually sent, rather than resting on a screenshot of a terminal.

What it shows:

- 12/12 blind. No journal phrase reached Agent A; no trend figure reached Agent B.
- The positive controls fired, so those absences were measured rather than assumed. The
  journal-side control fires on exactly the six cases where Node 2 produced at least one
  observation — a correspondence predicted independently by the extraction counts.
- The blindness is visible in the replies themselves, which is the stronger evidence. On
  the conflict and null cases Agent B writes that "the available data consists solely of
  self-reported journal entries" and lists device data among its unknowns, while Agent A
  reasons about volume and brightness without ever mentioning a journal. Neither was told
  the other existed.

One honest caveat, and a note for Node 5. On the cases where Node 2 returned no
observations, Agent B still produced a confident paragraph *about* self-reported journal
data it did not have. It invents no specific facts, but it does not say "nothing was
provided" either. Node 5 must not read that fluency as evidence — which is what the
evidence-binding validator is for.

## `node_05-grounded-synthesis/`

The live twelve-case Node 5 run behind Metrics 1, 2 and 3, including every repair attempt.
`tests/test_synthesis.py` reads it directly, so the scores below are re-derived on every
test run by the same functions that enforced them during the run.

| Metric | Result | Denominator |
| --- | --- | --- |
| 1 — numeric pass-through | 1.000 | 104 numbers |
| 2 — abstention matches significance | 1.000 | 24 checks |
| 3 — evidence binding | 1.000 | 90 references |

Two cases needed a repair, and they are the useful part of this directory, because they
show the validators doing work rather than agreeing with a model that was already right:

- **agree_03** — `NumericError: claims[4].text: the number '100' was not computed by the
  statistics engine`. The engine reports `caption_on_rate` as `1.0`; the node wrote it as a
  percentage. Repaired on the retry.
- **conflict_02** — `EvidenceError: 'trend_data.figures.font_scale_pct_change.significant':
  no field 'significant'`. That figure comes from a two-point series, so the engine returns
  no significance verdict for it, and the citation pointed at a field that does not exist.
  Repaired on the retry.

Every run ends on a successful attempt. Nothing here was scored after being allowed through.

---

## `node_06_10-citations-and-safety/` — Step 8

Nodes 6 and 10 run across the same 12 cases, over the **same Node 5 syntheses** preserved in
`node_05-grounded-synthesis/`. Nodes 3 and 5 were not re-run: doing so would have cost twelve
model calls to obtain slightly different inputs, and the numeric, abstention, citation and
safety numbers would then no longer describe the same run.

| Check | Result | Denominator |
| --- | --- | --- |
| Citation validity | **1.000** | 20 citations |
| Safety adherence (refusal boundary) | **1.000** | 12 reports, **first attempt, zero repairs** |
| Provenance preserved | **1.000** | 89 evidence references |

`metrics.json` is not typed by hand. `tests/test_report.py::test_recorded_metrics_recompute_from_the_recorded_outputs`
recomputes every figure from `cases.json` using the same validator functions that produced
it; if the table and the artifact ever disagree, the suite fails.

### Honest caveats

- **The citation validator never fired.** Across 24 live Node 6 calls the model did not
  fabricate a single URL, so the repair loop never had to catch one. A validator that has
  never rejected anything is indistinguishable from one that cannot reject anything, and
  1.000 here is a statement about the model's behaviour, not a demonstration of the check.
  Its catching power is shown by mutation: near-miss URLs — trailing slash, dropped
  subdomain, `http` for `https`, truncated path, uppercased — are each rejected, and
  deliberately loosening the comparison is caught by the suite.
- **Abstentions are not counted as passes.** 4 of the 12 cases set `source_url: null` on one
  suggestion. Those are excluded from the denominator entirely: counting them as passes would
  let a node score 100% by never citing anything. The 20 in the table are 20 real citations.
- **1.000 on safety is 12 reports, not a red team.** Every report opened with the clause on
  the first attempt, but these are the standard cases. The adversarial pressure test — 2
  diagnosis-baiting cases run 5× each — belongs to Step 9 and is not claimed here.
- **The safety check tolerates re-wrapping.** Whitespace runs are collapsed on both sides
  before comparison, because the first live run scored 0/12 purely on the source file's
  column-90 line breaks. Nothing else is relaxed: hedging, truncation, paraphrase, stripping
  the version tag and inverting the negation each still fail.
- **The "no diagnostic language" test is a floor, not a proof.** It checks for named
  conditions. It cannot catch implication, and its first draft flagged "ensure you have the
  correct prescription" as diagnostic. Keyword lists are the wrong instrument for this; the
  exact-string clause is the guarantee.
- **`retrieval/snapshot.json` is committed; `retrieval/cache/` is not.** The snapshot is the
  `{url, excerpt}` pairs the citations were scored against, so a clean clone reproduces the
  measurement with no network and no API key. The cache holds full third-party page text and
  is not ours to redistribute.
- **This run was regenerated after the refusal clause changed.** The clause used to open
  with a literal `[refusal_boundary v1]` tag, which meant the safety gate turned on
  transcribing a version marker rather than on the safety text. Removing it invalidated
  every stored report, so Nodes 6 and 10 were re-run rather than leaving evidence on disk
  that the current code would score as failing.

## `harness/` — Step 9

The three-arm comparison: `baseline_plain` (one prompt), `baseline_checked` (one prompt plus
the identical validators) and `pipeline` (the workflow). 72 measurements — 12 cases per arm,
plus the 2 adversarial cases and 1 consistency case run 5× each.

- `raw.json` — every measurement, with the payload it was scored against and the fingerprint
  of the prompts and routing that produced it.
- `metrics.json` — the aggregate scores.
- `results.md` — the rendered table, including its own caveats.

Read these alongside the caveats in the table itself:

- **The middle arm exists to attack the headline claim.** "The workflow beats a single
  prompt" is not a fair statement if the workflow also has validators and the single prompt
  does not. `baseline_checked` is the same single prompt with the same checks, so the table
  separates *what decomposition bought* from *what checking bought*. On this evidence most
  of the difference is checking.
- **Fairness is asserted in code, not in prose.** `tests/test_harness.py` fails if anyone
  trims the baseline's inputs, downgrades its model, or drops a rule from its prompt. The
  baseline is routed to the largest model, at temperature 0, and is handed the same engine
  figures with the same significance flags.
- **Evidence binding is not a clean head-to-head cell.** The pipeline has verified
  intermediates to cite; the single prompt has none, because it makes one call. Each arm is
  scored against the inputs it actually received, which favours the pipeline. The rendered
  table says so rather than banking the difference.
- **The single prompt's citation failure is not a hallucinated URL.** It cited
  `https://www.nidcd.nih.gov/health/over-counter-hearing-aids`, a page that exists and is
  on-topic, but that had not been retrieved that run. That is the harder failure to catch:
  nothing in the prose distinguishes a source the model was handed from one it remembered,
  and no claim resting on it was ever checked against its contents. The same prompt on the
  same model scored 1.000 with the validator attached.
- **The workflow loses coverage stability.** 0.657 against 0.900 and 1.000. Five runs of
  one input mentioned 1, 1, 2, 2 and 7 figures. No verdict moved and no run contradicted
  another, so nothing unsafe happened, but the pipeline is measurably more variable in how
  much it says. It is in the table at full size.
- **Cached measurements are fingerprinted.** An earlier table showed both baseline arms at
  0/10 on safety adherence. They had not regressed — their cached replies predated a change
  to the refusal clause and were being marked wrong for not anticipating it. The cache now
  refuses to return an entry produced by different prompts, models or temperatures.
