"""FastAPI entrypoint. Live /query flow only; the DeepEval evaluation pipeline lives in eval/."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app import config, graph_client, llm_client, vector_client
from app.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("agrigraphrag")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Fail fast: without Portkey the app must not start (no silent fallback).
    llm_client.require_portkey()
    llm_client.get_client()
    log.info("Portkey gateway configured; models: small=%s large=%s", config.SMALL_MODEL, config.LARGE_MODEL)
    yield
    graph_client.close()


app = FastAPI(title="AgriGraphRAG", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, examples=["White spots showed up on my tomato leaves in İzmir"])


@app.get("/health")
def health() -> dict:
    try:
        graph_nodes: int | str = graph_client.run("MATCH (n) WHERE NOT n:AgriGraphRAGSeed RETURN count(n) AS n")[0]["n"]
    except Exception as exc:  # Neo4j down or unreachable
        graph_nodes = f"unavailable: {exc.__class__.__name__}"
    return {
        "status": "ok",
        "portkey": "configured",
        "neo4j_nodes": graph_nodes,
        "lancedb_chunks": vector_client.count(),
        "models": {"small": config.SMALL_MODEL, "large": config.LARGE_MODEL},
    }


@app.post("/query")
async def query(request: QueryRequest) -> dict:
    try:
        result = await run_in_threadpool(run_pipeline, request.query)
    except llm_client.LLMRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return {
        "trace_id": result.trace_id,
        "query": result.query,
        "answer": result.answer,
        "sources": result.synthesis.sources,
        "graph_nodes": result.retrieval.graph_nodes if result.retrieval else [],
        "context": result.context(),
        "plan": result.plan.model_dump(),
        "social": result.social.model_dump() if result.social else None,
        "llm_calls": result.llm_calls,
        "totals": result.totals(),
    }
