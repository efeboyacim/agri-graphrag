"""Orchestrator Agent: intent + entity extraction (small model) and routing decisions."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from app import config, graph_client, llm_client

Intent = Literal[
    "diagnose_problem",
    "treatment_advice",
    "prevention_advice",
    "community_trends",
    "general_info",
    "out_of_scope",
]


class QueryPlan(BaseModel):
    intent: Intent = "diagnose_problem"
    crop: str | None = None
    region: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    symptom_keywords: list[str] = Field(default_factory=list)
    disease_mentioned: str | None = None
    run_retrieval: bool = True
    run_social_listening: bool = True
    reasoning: str = ""


SYSTEM_PROMPT = """You are the orchestrator of an agricultural assistant for tomato growers in Turkey.
Analyse the grower's message and reply with a single JSON object with exactly these keys:
{
  "intent": one of "diagnose_problem", "treatment_advice", "prevention_advice", "community_trends", "general_info", "out_of_scope",
  "crop": the crop mentioned (lowercase, e.g. "tomato") or null,
  "region": the region/city mentioned (e.g. "İzmir", "Manisa") or null,
  "symptoms": short symptom phrases the grower describes (list, may be empty),
  "symptom_keywords": single lowercase keywords that describe the symptoms (e.g. ["white", "spots"]), no stop-words,
  "disease_mentioned": a disease or pest named by the grower, or null,
  "run_retrieval": true if knowledge-base retrieval (crop/disease/treatment knowledge) is needed,
  "run_social_listening": true if recent posts from other growers in the region would help,
  "reasoning": one short sentence explaining the routing
}
Rules: agriculture questions always need retrieval. Social listening helps when a region is known or when
the grower asks what others are reporting. Anything unrelated to farming is "out_of_scope" with both flags false."""

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "could", "do", "does", "for", "from", "have", "has",
    "how", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "should", "some", "that", "the",
    "their", "there", "these", "this", "to", "up", "was", "what", "when", "which", "why", "with", "even",
    "though", "tomato", "tomatoes", "plant", "plants", "leaf", "leaves", "showed", "appeared", "appearing",
    "happening", "problem", "problems", "please", "help", "lately", "growers", "farmers", "reporting",
    "regularly", "turning", "getting", "flying", "small",
}


def _query_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-z]+", graph_client.normalize(text))
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


def plan_query(query: str, *, trace_id: str, parent_span_id: str | None = None) -> QueryPlan:
    plan = llm_client.chat_json(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}],
        QueryPlan,
        model=config.SMALL_MODEL,
        step="orchestrator",
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        metadata={"agent": "orchestrator"},
        temperature=0.0,
        max_tokens=1024,
    )

    # Deterministic guardrails on top of the LLM's decision.
    plan.region = graph_client.canonical_region(plan.region, query)
    if plan.crop is None and "tomato" in query.lower():
        plan.crop = "tomato"
    keywords = [k.lower().strip() for k in plan.symptom_keywords if k and k.strip()]
    for keyword in _query_keywords(" ".join(plan.symptoms) + " " + query):
        if keyword not in keywords:
            keywords.append(keyword)
    region_keys = set(graph_client.known_regions())
    plan.symptom_keywords = [k for k in keywords if k not in _STOPWORDS and k not in region_keys]

    if plan.intent == "out_of_scope":
        plan.run_retrieval = plan.run_social_listening = False
    else:
        plan.run_retrieval = True
        # Social listening is region-scoped; without a region it only runs for trend questions.
        plan.run_social_listening = plan.region is not None or plan.intent == "community_trends"
    return plan
