"""End-to-end demo: run the example query through all four agents and print every step,
including the Portkey trace/span ids of each LLM call.

Prerequisites: Neo4j running (docker compose up -d), both seed scripts run, .env filled in.
Usage:  python test_e2e.py ["your own question"]
"""
from __future__ import annotations

import sys
import textwrap

from app import graph_client, llm_client
from app.config import configure_console
from app.pipeline import run_pipeline

EXAMPLE_QUERY = "White spots showed up on my tomato leaves in İzmir"


def _wrap(text: str, indent: str = "    ") -> str:
    return "\n".join(textwrap.fill(line, 100, initial_indent=indent, subsequent_indent=indent) for line in text.splitlines() if line.strip())


def print_step(name: str, payload) -> None:
    if name == "orchestrator":
        print("\n[1] ORCHESTRATOR AGENT (intent + entities + routing)")
        print(f"    intent={payload.intent}  crop={payload.crop}  region={payload.region}")
        print(f"    symptoms={payload.symptoms}  keywords={payload.symptom_keywords}")
        print(f"    run_retrieval={payload.run_retrieval}  run_social_listening={payload.run_social_listening}")
        print(f"    reasoning: {payload.reasoning}")
    elif name == "retrieval":
        print("\n[2] RETRIEVAL AGENT (hybrid: Neo4j graph + LanceDB vectors)")
        if payload is None:
            print("    skipped by orchestrator")
            return
        print(f"    graph strategy: {payload.graph_strategy}")
        for fact in payload.graph_facts:
            print(_wrap(f"[{fact.id}] (score {fact.score}) {fact.text}"))
        for doc in payload.documents:
            print(_wrap(f"[{doc.id}] (cosine distance {doc.distance}) {doc.title}: {doc.text}"))
        print("    graph nodes: " + ", ".join(f"{n['label']}:{n['name']}" for n in payload.graph_nodes))
    elif name == "social_listening":
        print("\n[3] SOCIAL LISTENING AGENT (mock posts from the graph)")
        if payload is None:
            print("    skipped by orchestrator")
            return
        print(f"    region filter: {payload.region_filter}  posts: {len(payload.posts)}  concern: {payload.concern_level}")
        print(_wrap(f"summary: {payload.summary}"))
        print(_wrap(f"trend: {payload.trend}"))
        for post in payload.posts:
            print(_wrap(f"[{post.id}] {post.region} {post.date}: {post.text}"))
    elif name == "synthesis":
        print(f"\n[4] SYNTHESIS AGENT (prompt {payload.prompt_version})")
        print(_wrap(payload.answer))
        print(f"    cited sources: {', '.join(payload.sources) or '-'}")


def main() -> int:
    configure_console()
    llm_client.require_portkey()
    query = " ".join(sys.argv[1:]) or EXAMPLE_QUERY
    print(f"QUERY: {query}")
    result = run_pipeline(query, on_step=print_step)
    graph_client.close()

    print(f"\n[PORTKEY TRACE] trace_id = {result.trace_id}")
    print(f"    {'step':<17}{'model':<22}{'latency':>9}{'tokens':>8}{'cost $':>11}  span_id           gateway trace id")
    for call in result.llm_calls:
        print(
            f"    {call['step']:<17}{call['model']:<22}{call['latency_ms']:>7}ms{call['total_tokens']:>8}"
            f"{call['cost_usd']:>11.6f}  {call['span_id']}  {call['portkey_trace_id'] or '(not echoed)'}"
        )
    totals = result.totals()
    print(f"    total: {totals['llm_calls']} LLM calls, {totals['latency_ms']} ms, {totals['total_tokens']} tokens, ${totals['cost_usd']:.6f}")
    print("    View it at https://app.portkey.ai (Logs / Traces) filtered by the trace_id above.")

    if query != EXAMPLE_QUERY:
        return 0
    answer = result.answer.lower()
    checks = {
        "mentions Powdery Mildew": "powdery mildew" in answer,
        "mentions FungiStop-X": "fungistop-x" in answer,
        "includes the social trend": bool(result.social and result.social.posts)
        and any(sid in result.synthesis.sources for sid in ["S1"] + [p.id for p in result.social.posts]),
        "Portkey echoed the trace_id": any(c["portkey_trace_id"] == result.trace_id for c in result.llm_calls),
    }
    print("\n[CHECKS]")
    for label, ok in checks.items():
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
