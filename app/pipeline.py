"""The agent flow: orchestrator -> retrieval -> social listening -> synthesis.

Used by the /query endpoint, test_e2e.py and the offline eval scripts, so all of them exercise
exactly the same code path. DeepEval is never involved here.
"""
from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field

from app import llm_client
from app.agents.orchestrator import QueryPlan, plan_query
from app.agents.retrieval import RetrievalResult, retrieve
from app.agents.social_listening import SocialResult, listen
from app.agents.synthesis import SynthesisResult, synthesize

StepCallback = Callable[[str, Any], None]

OUT_OF_SCOPE_ANSWER = (
    "I can only help with crop and farming questions (currently tomato growing in İzmir and Manisa). "
    "Could you describe what you are seeing in your field?"
)


class UpstreamResult(BaseModel):
    trace_id: str
    query: str
    plan: QueryPlan
    retrieval: RetrievalResult | None = None
    social: SocialResult | None = None

    def context(self) -> list[str]:
        """Ordered retrieval context handed to the Synthesis Agent (and to DeepEval)."""
        entries = self.retrieval.context_entries() if self.retrieval else []
        return entries + (self.social.context_entries() if self.social else [])


class PipelineResult(UpstreamResult):
    synthesis: SynthesisResult
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def answer(self) -> str:
        return self.synthesis.answer

    def totals(self) -> dict[str, Any]:
        return {
            "llm_calls": len(self.llm_calls),
            "latency_ms": sum(c["latency_ms"] for c in self.llm_calls),
            "total_tokens": sum(c["total_tokens"] for c in self.llm_calls),
            "cost_usd": round(sum(c["cost_usd"] for c in self.llm_calls), 6),
        }


def _notify(on_step: StepCallback | None, name: str, payload: Any) -> None:
    if on_step is not None:
        on_step(name, payload)


def run_upstream(query: str, *, trace_id: str | None = None, on_step: StepCallback | None = None) -> UpstreamResult:
    trace_id = trace_id or llm_client.new_trace_id()

    plan = plan_query(query, trace_id=trace_id)
    _notify(on_step, "orchestrator", plan)

    retrieval = retrieve(query, plan) if plan.run_retrieval else None
    _notify(on_step, "retrieval", retrieval)

    social = listen(plan.region, trace_id=trace_id) if plan.run_social_listening else None
    _notify(on_step, "social_listening", social)

    return UpstreamResult(trace_id=trace_id, query=query, plan=plan, retrieval=retrieval, social=social)


def synthesize_from(upstream: UpstreamResult, *, prompt_version: str = "v2") -> SynthesisResult:
    if upstream.plan.intent == "out_of_scope":
        return SynthesisResult(answer=OUT_OF_SCOPE_ANSWER, prompt_version=prompt_version)
    return synthesize(upstream.query, upstream.context(), trace_id=upstream.trace_id, prompt_version=prompt_version)


def run_pipeline(
    query: str,
    *,
    prompt_version: str = "v2",
    trace_id: str | None = None,
    on_step: StepCallback | None = None,
) -> PipelineResult:
    upstream = run_upstream(query, trace_id=trace_id, on_step=on_step)
    synthesis = synthesize_from(upstream, prompt_version=prompt_version)
    _notify(on_step, "synthesis", synthesis)
    calls = [call.as_dict() for call in llm_client.calls_for_trace(upstream.trace_id)]
    return PipelineResult(**upstream.model_dump(), synthesis=synthesis, llm_calls=calls)
