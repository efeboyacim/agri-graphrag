"""Retrieval Agent: hybrid retrieval over Neo4j (structured) and LanceDB (free text). No LLM involved."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app import graph_client, vector_client
from app.agents.orchestrator import QueryPlan

# crop + symptom keywords (+ optional disease name) -> disease -> product, with region context.
SYMPTOM_QUERY = """
MATCH (c:Crop)-[:AFFECTED_BY]->(d:Disease)
WHERE $crop IS NULL OR toLower(c.name) = toLower($crop)
OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
WITH c, d, collect(DISTINCT s.description) AS symptoms
WITH c, d, symptoms,
     toLower(reduce(acc = '', x IN symptoms | acc + ' ' + x)) AS symptom_text,
     toLower(d.name + ' ' + coalesce(d.description, '')) AS disease_text
WITH c, d, symptoms,
     2 * size([k IN $keywords WHERE symptom_text CONTAINS k])
       + size([k IN $keywords WHERE disease_text CONTAINS k])
       + CASE WHEN $disease IS NOT NULL AND disease_text CONTAINS toLower($disease) THEN 5 ELSE 0 END AS score
WHERE score > 0
OPTIONAL MATCH (d)-[:TREATED_BY]->(p:Product)
OPTIONAL MATCH (c)-[:GROWN_IN_REGION]->(r:Region {name: $region})
RETURN c.name AS crop, d.name AS disease, d.description AS description, symptoms,
       collect(DISTINCT p {.name, .type}) AS products, r.name AS region, score, 0 AS posts
ORDER BY score DESC, disease
LIMIT $limit
"""

# Fallback for trend-style questions: which diseases are growers in this region posting about?
# Region <- SocialPost -[:RELATED_TO]-> Disease -> Product
REGION_ACTIVITY_QUERY = """
MATCH (post:SocialPost)-[:RELATED_TO]->(d:Disease)<-[:AFFECTED_BY]-(c:Crop)
WHERE ($region IS NULL OR post.region = $region) AND ($crop IS NULL OR toLower(c.name) = toLower($crop))
WITH c, d, count(DISTINCT post) AS posts
OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (d)-[:TREATED_BY]->(p:Product)
OPTIONAL MATCH (c)-[:GROWN_IN_REGION]->(r:Region {name: $region})
RETURN c.name AS crop, d.name AS disease, d.description AS description,
       collect(DISTINCT s.description) AS symptoms, collect(DISTINCT p {.name, .type}) AS products,
       r.name AS region, posts AS score, posts
ORDER BY score DESC, disease
LIMIT $limit
"""

GRAPH_LIMIT = 2
SECONDARY_MATCH_RATIO = 0.6
VECTOR_K = 3
MAX_COSINE_DISTANCE = 0.8  # drop chunks that are barely related


class GraphFact(BaseModel):
    id: str
    text: str
    disease: str
    products: list[str]
    score: int
    source: str = "neo4j"


class DocChunk(BaseModel):
    id: str
    text: str
    title: str
    note_id: str
    distance: float
    source: str = "lancedb"


class RetrievalResult(BaseModel):
    graph_facts: list[GraphFact] = Field(default_factory=list)
    documents: list[DocChunk] = Field(default_factory=list)
    graph_nodes: list[dict[str, str]] = Field(default_factory=list)
    graph_strategy: str = "none"

    def context_entries(self) -> list[str]:
        """Merged, ordered context: graph facts first (most specific), then documents."""
        return [f"[{f.id}] {f.text}" for f in self.graph_facts] + [
            f"[{d.id}] ({d.title}) {d.text}" for d in self.documents
        ]


def _fact_text(row: dict[str, Any]) -> str:
    crop = (row.get("crop") or "crop").capitalize()
    where = f" (grown in {row['region']})" if row.get("region") else ""
    parts = [f"{crop}{where} is affected by {row['disease']}: {row.get('description') or ''}".rstrip(": ")]
    if row.get("symptoms"):
        parts.append("Symptoms: " + "; ".join(row["symptoms"]))
    if row.get("products"):
        parts.append("Treated by: " + ", ".join(f"{p['name']} ({p['type']})" for p in row["products"]))
    if row.get("posts"):
        parts.append(f"Mentioned in {row['posts']} recent social post(s) from {row.get('region') or 'the area'}")
    return ". ".join(parts) + "."


def _graph_nodes(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    nodes: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(label: str, name: str | None) -> None:
        if name and (label, name) not in seen:
            seen.add((label, name))
            nodes.append({"label": label, "name": name})

    for row in rows:
        add("Crop", row.get("crop"))
        add("Region", row.get("region"))
        add("Disease", row.get("disease"))
        for symptom in row.get("symptoms") or []:
            add("Symptom", symptom)
        for product in row.get("products") or []:
            add("Product", product["name"])
    return nodes


def retrieve(query: str, plan: QueryPlan) -> RetrievalResult:
    params = {
        "crop": plan.crop,
        "region": plan.region,
        "keywords": plan.symptom_keywords,
        "disease": plan.disease_mentioned,
        "limit": GRAPH_LIMIT,
    }
    rows: list[dict[str, Any]] = []
    strategy = "none"
    if plan.intent != "community_trends":
        rows = graph_client.run(SYMPTOM_QUERY, **params)
        # Keep weaker matches only if they are close to the best one (e.g. a shared word like "spots"
        # should not drag a second disease into the context).
        rows = [r for r in rows if r["score"] >= SECONDARY_MATCH_RATIO * rows[0]["score"]]
        strategy = "symptom_match" if rows else strategy
    if not rows and (plan.region or plan.intent == "community_trends"):
        rows = graph_client.run(REGION_ACTIVITY_QUERY, **params)
        strategy = "region_activity" if rows else strategy

    facts = [
        GraphFact(
            id=f"G{i}",
            text=_fact_text(row),
            disease=row["disease"],
            products=[p["name"] for p in row.get("products") or []],
            score=int(row.get("score") or 0),
        )
        for i, row in enumerate(rows, start=1)
    ]

    # Graph-informed query expansion: the diseases found in the graph steer the vector search.
    expanded_query = " ".join([query] + [f.disease for f in facts])
    hits = [h for h in vector_client.search(expanded_query, k=VECTOR_K) if h["_distance"] <= MAX_COSINE_DISTANCE]
    documents = [
        DocChunk(
            id=f"D{i}",
            text=hit["text"],
            title=hit.get("title", ""),
            note_id=hit.get("id", ""),
            distance=round(float(hit["_distance"]), 4),
        )
        for i, hit in enumerate(hits, start=1)
    ]
    return RetrievalResult(graph_facts=facts, documents=documents, graph_nodes=_graph_nodes(rows), graph_strategy=strategy)
