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
