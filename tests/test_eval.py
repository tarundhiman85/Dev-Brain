"""Offline evaluation contract and fixture integrity checks."""
import asyncio
import json
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from pydantic import TypeAdapter

from agent.orchestrator import MAX_OBSERVATION_CHARS, bounded_observation
from eval.eval_harness import GoldenCase, Observation, aggregate, evaluate_case, retrieval_metrics
from ingest import documents

ROOT = Path(__file__).resolve().parents[1]


def test_golden_targets_exist_in_ingestible_fixtures():
    cases = TypeAdapter(list[GoldenCase]).validate_json((ROOT / "eval/golden_dataset.json").read_bytes())
    assert 15 <= len({case.id for case in cases}) == len(cases) <= 20
    matches = [
        {"id": source, "content": content, "metadata": metadata}
        for source, content, metadata in documents(ROOT / "eval/fixtures")
    ]
    trace = [Observation(tool="search_devbrain_kb", status="ok", observation=json.dumps({"matches": matches}))]
    for case in cases:
        metrics = retrieval_metrics(case, trace)
        assert metrics["target_coverage"] == (1 if case.ground_truth_context else None), case.id


def test_truncated_observation_preserves_complete_records():
    first = {"id": "first", "content": "short"}
    content, truncated = bounded_observation({
        "matches": [first, {"id": "second", "content": "x" * MAX_OBSERVATION_CHARS}],
    })
    assert truncated and len(content) <= MAX_OBSERVATION_CHARS
    assert json.loads(content) == {"_truncated": True, "matches": [first]}


def test_evaluation_separates_judge_inputs_and_scores_deduplicated_retrieval():
    case = GoldenCase(id="demo", question="What is the cache TTL?", expected_answer="15 minutes.",
                      ground_truth_context=[{"source": "eval/fixtures/cache.md", "text": "15 minutes"}])
    evidence = json.dumps({"matches": [{
        "id": "target", "content": "The TTL is 15 minutes.",
        "metadata": {"path": r"C:\workspace\eval\fixtures\cache.md"},
    }]})

    def agent(request):
        assert json.loads(request.content) == {
            "prompt": case.question, "mode": "cloud", "include_evidence": True,
        }
        step = {"tool": "search_devbrain_kb", "status": "ok", "observation": evidence}
        return httpx.Response(200, json={
            "answer": "The TTL is 15 minutes.", "trace": [step, step],
            "backend": "cloud", "model": "test-agent", "fallback": False,
        })

    def model(request):
        payload = json.loads(request.content)
        data = json.loads(payload["messages"][-1]["content"])
        schema = payload["response_format"]["json_schema"]["name"]
        if schema == "Grounding":
            assert "expected_answer" not in data
            assert evidence in data["observations"]
            score = {"claims": 1, "unsupported_claims": 0, "rationale": "TTL matches the evidence."}
        else:
            assert "observations" not in data and data["expected_answer"] == "15 minutes."
            score = {"score": 4, "rationale": "Correct TTL."}
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(score)},
        }]})

    async def run():
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(agent)) as http,
            AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(model))) as judge,
        ):
            return await evaluate_case(case, http, "http://test/api/v1/agent/chat", "cloud", judge, "test")

    row = asyncio.run(run())
    assert row["retrieval"]["retrieved_records"] == 1
    summary = aggregate([row])
    assert summary["failed_cases"] == 0
    assert summary["retrieval_hit_rate"] == summary["faithfulness"] == summary["answer_correctness"] == 1
    assert summary["hallucination_rate"] == 0
