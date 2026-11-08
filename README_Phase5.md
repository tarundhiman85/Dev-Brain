# DevBrain Phase 5

The evaluation CLI runs 17 golden questions through the real chat endpoint.
The dataset covers single-source questions, archived PR facts, a two-source
comparison, and an unanswerable question. All reference answers come from the
synthetic [evaluation fixtures](eval/fixtures). The archived PR URL is a
fixture identifier. It does not need to exist on GitHub.

## Run a baseline

1. Activate the project virtual environment and run
   `python -m pip install -r requirements.txt`. The harness uses the existing
   OpenAI SDK, HTTPX, and Pydantic dependencies.
2. Start PostgreSQL. Prefer a dedicated database so unrelated notes do not
   affect the baseline. Set `DATABASE_URL` in `.env` to that database and choose
   the embedding provider you want to evaluate. Adjust the example database
   user if you use different Compose credentials.

   ```powershell
   docker compose up -d --wait
   docker compose exec postgres createdb -U devbrain devbrain_eval
   ```

   Set `DATABASE_URL` to the full connection URI for `devbrain_eval`, for example
   `postgresql+psycopg://devbrain:devbrain_local@localhost:5432/devbrain_eval`.
   Then run `python ingest.py eval/fixtures`. Keep the embedding provider and
   database settings the same in the agent process.
3. In `.env`, set `OPENAI_API_KEY` and a tool-capable
   `DEVBRAIN_AGENT_MODEL` for the agent. The judge defaults to `gpt-4o`.
   Override it with `DEVBRAIN_JUDGE_MODEL` if needed. The judge model must
   support [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
4. Start the agent API in one terminal and run the harness in another from the
   project root.

   ```powershell
   python -m uvicorn main_agent:app --host 127.0.0.1 --port 8001
   ```

   ```powershell
   python -m eval.eval_harness --mode cloud --label baseline --output .tmp/eval/baseline.json
   ```

The CLI defaults to cloud mode to exercise the tool loop. Try `--mode auto`
to measure routing changes and `--mode local` to see how Ollama answers without
knowledge base access. Use `--base-url` for a different agent address or
`--dataset` for another dataset in the same format. The CLI prints summary
metrics, saves per-case scores and stage errors, and exits with status 1 if
any case could not be fully evaluated.

The harness asks for `include_evidence: true`. This adds the bounded observations
the agent actually saw to successful tool steps. Leave it off for ordinary
chat. The grounding judge receives the question, answer, and observations.
The correctness judge receives the question, answer, and golden answer in a
separate call. Both use OpenAI, including when the agent runs in Local Only.
The JSON report includes answers and retrieved source identifiers, so store
reports locally if they contain private data.

## Interpret the report

| Metric | Meaning |
| --- | --- |
| Retrieval hit rate | Fraction of answerable cases that retrieved at least one expected source and passage. |
| Retrieval precision | Mean per-case fraction of unique returned records that match an expected source and passage. A case with expected evidence and zero records scores zero. |
| Retrieval target coverage | Mean fraction of expected passages found per answerable case. This distinguishes finding one source from finding both sources on a multi-source case. |
| Faithfulness | Mean per-case fraction of substantive factual claims supported by the tool observations. Answers with no factual claims have no faithfulness score. |
| Hallucination rate | Unsupported factual claims divided by all judged factual claims. Contradictions count as unsupported. |
| Answer correctness | Mean semantic score, normalized from the judge's 0–4 scale to 0–1. |

Retrieval matches use a source URL or fixture-relative path and a
case-insensitive, whitespace-normalized passage. A correct document without
the expected passage is a miss. The unanswerable case is excluded from
retrieval averages and scored for correctness. Failed stages are recorded
separately, and missing scores remain null or absent. Check the metric
denominators alongside the averages. The tool loop can return truncated
observations, which the report counts.

## Compare runs

Keep the dataset hash, judge model, agent model, and mode fixed when measuring
a chunking change. Re-ingest after changing chunking. To test routing, change
the routing rules or use `--mode auto` and inspect actual backends and fallback
flags alongside scores and latency. Save each run under a distinct path and
use `--label` to record the Git SHA and settings. Review failures and changed
cases before attributing a small average change to an improvement. Judge
scores can vary across runs.
