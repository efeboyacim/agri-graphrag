"""Synthesis Agent: combines graph facts, documents and social trends into the final answer (large model).

Two prompt versions exist for the offline prompt experiment (eval/prompt_experiment.py):
  v1 - deliberately naive: loosely grounded, no citation rules (baseline)
  v2 - production prompt: grounded in the retrieved context with explicit source citations
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app import config, llm_client

V1_SYSTEM = (
    "You are a friendly farming expert. Answer the farmer's question thoroughly, drawing on your broad "
    "agricultural knowledge. Cover possible causes, treatments and general good growing practices."
)

V2_SYSTEM = """You are AgriGraphRAG, an agronomy assistant for tomato growers.
Answer the grower's question using ONLY the numbered context entries you are given:
[G#] = facts from the crop knowledge graph, [D#] = agronomy notes, [S#] = recent social media posts from growers.

Rules:
- Use only facts stated in the context. Do not add general farming advice that is not in the context,
  even if it is common knowledge.
- End every sentence with the id(s) of the context entries that support it, e.g. [G1] or [D2][S1].
  If a sentence cannot be cited, leave it out.
- Recommend only products named in the [G#] facts. Never invent products, doses or spray schedules.
- If the context does not cover something the grower asked, say so briefly instead of guessing.
- Answer exactly what was asked, in under 150 words, with short sections that fit the question:
  - a symptom or treatment question: **Likely cause**, **What to do**, **Prevention**, and
    **What other growers report** (only if [S#] entries exist);
  - a question about what other growers are reporting: lead with **What growers report**, name the
    diseases involved, and add at most one line on treatment."""

PROMPTS = {"v1": V1_SYSTEM, "v2": V2_SYSTEM}


class SynthesisResult(BaseModel):
    answer: str
    prompt_version: str
    sources: list[str] = Field(default_factory=list)


def _user_message(query: str, context: list[str], prompt_version: str) -> str:
    block = "\n".join(context) if context else "(no context retrieved)"
    if prompt_version == "v1":
        return f"{query}\n\n(Some notes that may or may not be relevant:\n{block})"
    return f"Grower's question: {query}\n\nContext:\n{block}"


def synthesize(
    query: str,
    context: list[str],
    *,
    trace_id: str,
    prompt_version: str = "v2",
    parent_span_id: str | None = None,
) -> SynthesisResult:
    if prompt_version not in PROMPTS:
        raise ValueError(f"unknown synthesis prompt version {prompt_version!r}")
    answer = llm_client.chat(
        [
            {"role": "system", "content": PROMPTS[prompt_version]},
            {"role": "user", "content": _user_message(query, context, prompt_version)},
        ],
        model=config.LARGE_MODEL,
        step="synthesis",
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        metadata={"agent": "synthesis", "prompt_version": prompt_version},
        temperature=0.2,
        max_tokens=2048,
    )
    # Normalise "[ G1 ]" / "[G1 , D2]" to "[G1]" / "[G1, D2]".
    answer = re.sub(
        r"\[\s*([GDS]\d+(?:\s*,\s*[GDS]\d+)*)\s*\]",
        lambda m: "[" + re.sub(r"\s*,\s*", ", ", m.group(1)) + "]",
        answer,
    )
    # Handles both "[G1][D2]" and "[G1, D2]" citation styles.
    bracketed = " ".join(re.findall(r"\[([^\]]+)\]", answer))
    sources = list(dict.fromkeys(re.findall(r"\b([GDS]\d+)\b", bracketed)))
    return SynthesisResult(answer=answer, prompt_version=prompt_version, sources=sources)
