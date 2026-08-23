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
