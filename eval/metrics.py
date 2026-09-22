"""DeepEval judge + metric set + small reporting helpers shared by test_suite.py and prompt_experiment.py."""
from __future__ import annotations

import asyncio
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Type

from deepeval.metrics import AnswerRelevancyMetric, BaseMetric, ContextualPrecisionMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from pydantic import BaseModel

from app import config, llm_client
from app.graph_client import normalize
from app.pipeline import UpstreamResult
from eval.dataset import EvalCase

THRESHOLD = 0.7
RESULTS_DIR = Path(__file__).resolve().parent / "results"


class PortkeyGroqJudge(DeepEvalBaseLLM):
    """DeepEval judge that goes through app.llm_client -> Portkey -> Groq.

    Judge calls therefore get the same Portkey tracing (step 'judge.<metric>') and the same
    rate-limit handling as the application's own LLM calls.
    """

    def __init__(self, metric: str, trace_id: str, model: str = config.JUDGE_MODEL):
        self.model_name = model
        self.metric = metric
        self.trace_id = trace_id
        super().__init__(model)

    def load_model(self):
        return self.model_name

    def get_model_name(self) -> str:
        return f"{self.model_name} (Groq via Portkey)"

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        messages = [{"role": "user", "content": prompt}]
        if schema is not None:
            messages.insert(0, {"role": "system", "content": "Respond with a single valid JSON object only."})
        text = llm_client.chat(
            messages,
            model=self.model_name,
            step=f"judge.{self.metric}",
            trace_id=self.trace_id,
            metadata={"agent": "deepeval_judge", "metric": self.metric},
            json_mode=schema is not None,
            temperature=0.0,
            max_tokens=2048,
        )
        if schema is None:
            return text
        try:
            return schema.model_validate_json(llm_client._extract_json(text))
        except Exception:
            return text  # DeepEval falls back to its own lenient JSON parsing

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        return await asyncio.to_thread(self.generate, prompt, schema)


def _fact_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize(text))


class KeyFactsMetric(BaseMetric):
    """Deterministic: fraction of the case's expected key facts (disease, product, ...) present in the answer."""

    _required_params = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.include_reason = True
        self.strict_mode = False
        self.async_mode = False
        self.evaluation_model = None

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        facts = list((test_case.metadata or {}).get("expected_facts", []))
        answer = _fact_key(test_case.actual_output or "")
        found = [fact for fact in facts if _fact_key(fact) in answer]
        missing = [fact for fact in facts if fact not in found]
        self.score = len(found) / len(facts) if facts else 1.0
        self.reason = f"found: {found or '-'}; missing: {missing or '-'}"
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Key Facts"


def build_metrics(trace_id: str, *, contextual_precision: bool = True) -> list[BaseMetric]:
    """LLM-judged RAG metrics + the deterministic key-facts check.

    penalize_ambiguous_claims=True: by default DeepEval counts claims the context can neither confirm
    nor contradict ("idk") as faithful, which would hide exactly the ungrounded general-knowledge
    claims we want to catch. include_reason=False saves one judge call per metric (free-tier budget).
    """
    common = {"threshold": THRESHOLD, "include_reason": False, "async_mode": False}
    metrics: list[BaseMetric] = [
        FaithfulnessMetric(model=PortkeyGroqJudge("faithfulness", trace_id), penalize_ambiguous_claims=True, **common),
        AnswerRelevancyMetric(model=PortkeyGroqJudge("answer_relevancy", trace_id), **common),
    ]
    if contextual_precision:
        metrics.append(ContextualPrecisionMetric(model=PortkeyGroqJudge("contextual_precision", trace_id), **common))
    metrics.append(KeyFactsMetric())
    return metrics


def to_test_case(case: EvalCase, upstream: UpstreamResult, answer: str) -> LLMTestCase:
    return LLMTestCase(
        input=case.query,
        actual_output=answer,
        expected_output=case.expected_output,
        retrieval_context=upstream.context() or ["(no context retrieved)"],
        name=case.id,
        metadata={
            "case_id": case.id,
            "expected_facts": list(case.expected_facts),
            "pipeline_trace_id": upstream.trace_id,
        },
    )


def collect_scores(evaluation) -> dict[str, dict[str, dict[str, Any]]]:
    """EvaluationResult -> {case_id: {metric_name: {score, success, error}}}."""
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    for index, result in enumerate(evaluation.test_results):
        case_id = (result.metadata or {}).get("case_id") or result.name or f"case_{index}"
        scores[case_id] = {
            data.name: {"score": data.score, "success": bool(data.success), "error": data.error}
            for data in (result.metrics_data or [])
        }
    return scores


def aggregate(scores: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    metric_names = list(dict.fromkeys(name for case in scores.values() for name in case))
    per_metric = {}
    for name in metric_names:
        entries = [case[name] for case in scores.values() if name in case]
        valid = [e["score"] for e in entries if e["score"] is not None and not e["error"]]
        per_metric[name] = {
            "mean": round(statistics.mean(valid), 3) if valid else None,
            "pass_rate": round(sum(e["success"] for e in entries) / len(entries), 3) if entries else None,
            "errors": sum(1 for e in entries if e["error"]),
        }
    passed = [cid for cid, case in scores.items() if case and all(m["success"] for m in case.values())]
    return {
        "cases": len(scores),
        "cases_passed": len(passed),
        "overall_pass_rate": round(len(passed) / len(scores), 3) if scores else None,
        "per_metric": per_metric,
    }


def fmt(value: float | None) -> str:
    return "  -  " if value is None else f"{value:.2f}"


def token_usage() -> dict[str, dict[str, float]]:
    """Tokens/cost of every LLM call made in this process, grouped by step ('judge.*' collapsed)."""
    usage: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost_usd": 0.0})
    for call in llm_client.CALL_LOG:
        key = f"{'judge' if call.step.startswith('judge.') else call.step} [{call.model}]"
        usage[key]["calls"] += 1
        usage[key]["tokens"] += call.total_tokens
        usage[key]["cost_usd"] = round(usage[key]["cost_usd"] + call.cost_usd, 6)
    return dict(usage)


def print_token_usage() -> None:
    print("\nLLM usage (all calls went through Portkey):")
    for key, row in token_usage().items():
        print(f"  {key:<45} {int(row['calls']):>4} calls {int(row['tokens']):>8} tokens  ${row['cost_usd']:.4f}")


def save_results(prefix: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
