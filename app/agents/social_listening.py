"""Social Listening Agent: reads mock SocialPost nodes, filters by region, summarises trends (small model)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app import config, graph_client, llm_client

POSTS_QUERY = """
MATCH (p:SocialPost)
WHERE $region IS NULL OR p.region = $region
OPTIONAL MATCH (p)-[:RELATED_TO]->(d:Disease)
RETURN p.text AS text, p.region AS region, p.date AS date, collect(d.name) AS related_diseases
ORDER BY p.date DESC
"""

SYSTEM_PROMPT = """You analyse recent social media posts from tomato growers (mock data).
Summarise only what the posts say; do not add outside knowledge. Reply with one JSON object:
{
  "summary": 1-2 sentences summarising what growers report,
  "trend": one short sentence naming the recurring problem/pattern, or "no clear trend",
  "concern_level": "low", "medium" or "high" (how widespread/urgent the reports look),
  "related_diseases": disease names linked to the posts (list)
}"""


class SocialPost(BaseModel):
    id: str
    text: str
    region: str
    date: str
    related_diseases: list[str] = Field(default_factory=list)


class _TrendSummary(BaseModel):
    summary: str = ""
    trend: str = "no clear trend"
    concern_level: Literal["low", "medium", "high"] = "low"
    related_diseases: list[str] = Field(default_factory=list)


class SocialResult(_TrendSummary):
    region_filter: str | None = None
    fallback_to_all_regions: bool = False
    posts: list[SocialPost] = Field(default_factory=list)

    def context_entries(self) -> list[str]:
        if not self.posts:
            return []
        scope = "all regions" if self.fallback_to_all_regions or not self.region_filter else self.region_filter
        entries = [
            f"[S1] Social listening summary ({len(self.posts)} recent post(s), {scope}): {self.summary} "
            f"Trend: {self.trend} (concern: {self.concern_level})."
        ]
        entries += [f"[{p.id}] Post from {p.region} on {p.date}: \"{p.text}\"" for p in self.posts]
        return entries


def listen(region: str | None, *, trace_id: str, parent_span_id: str | None = None) -> SocialResult:
    rows = graph_client.run(POSTS_QUERY, region=region)
    fallback = False
    if not rows and region is not None:
        rows, fallback = graph_client.run(POSTS_QUERY, region=None), True
    posts = [SocialPost(id=f"S{i}", **row) for i, row in enumerate(rows, start=2)]  # S1 = the summary
    if not posts:
        return SocialResult(region_filter=region, summary="No recent posts found.")

    listing = "\n".join(
        f"- ({p.region}, {p.date}) {p.text} [linked to: {', '.join(p.related_diseases) or 'unknown'}]" for p in posts
    )
    summary = llm_client.chat_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Region filter: {region or 'none'}\nPosts:\n{listing}"},
        ],
        _TrendSummary,
        model=config.SMALL_MODEL,
        step="social_listening",
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        metadata={"agent": "social_listening", "region": region or "all"},
        temperature=0.0,
        max_tokens=1024,
    )
    return SocialResult(
        **summary.model_dump(), region_filter=region, fallback_to_all_regions=fallback, posts=posts
    )
