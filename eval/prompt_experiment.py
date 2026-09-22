"""Iterative-experimentation proof: compare two versions of the Synthesis Agent prompt with DeepEval.

  v1 - deliberately naive: "answer thoroughly from your agricultural knowledge", context loosely appended
  v2 - production prompt: answer only from the retrieved context and cite [G#]/[D#]/[S#] sources

For every case the upstream agents (orchestrator, retrieval, social listening) run ONCE and both
prompt versions synthesise from the identical context, so the only variable is the prompt.
Contextual precision is not scored here: it depends only on retrieval, which is identical for v1 and v2.

Usage:  python eval/prompt_experiment.py [--limit N] [--case ID ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepeval import evaluate  # noqa: E402
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from app import config, graph_client, llm_client  # noqa: E402
from app.config import configure_console  # noqa: E402
from app.pipeline import run_upstream, synthesize_from  # noqa: E402
from eval.dataset import select_cases  # noqa: E402
from eval.metrics import (  # noqa: E402
    aggregate,
    build_metrics,
    collect_scores,
    fmt,
    print_token_usage,
    save_results,
    to_test_case,
    token_usage,
)

VERSIONS = ("v1", "v2")


def main(argv: list[str] | None = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only use the first N cases (free-tier budget)")
    parser.add_argument("--case", nargs="*", help="only use these case ids")
    args = parser.parse_args(argv)
    llm_client.require_portkey()
    cases = select_cases(args.limit, args.case)

    print(f"Generating answers for {len(cases)} case(s) with synthesis prompts {' and '.join(VERSIONS)} ...")
    test_cases: dict[str, list[LLMTestCase]] = {v: [] for v in VERSIONS}
    answers: dict[str, dict[str, str]] = {}
    pipeline_errors: dict[str, str] = {}
    for case in cases:
        print(f"  - {case.id}")
        try:
            upstream = run_upstream(case.query)
            generated = {v: synthesize_from(upstream, prompt_version=v).answer for v in VERSIONS}
        except llm_client.LLMRateLimitError as exc:
            pipeline_errors[case.id] = str(exc)
            print(f"    pipeline failed: {exc}")
            continue
        answers[case.id] = generated
        for version in VERSIONS:
            test_cases[version].append(to_test_case(case, upstream, generated[version]))
    if not answers:
        print("No answers could be generated.")
        return 1

    scores, summaries = {}, {}
    for version in VERSIONS:
        judge_trace = llm_client.new_trace_id(f"eval-judge-{version}")
        print(f"\nScoring prompt {version} with DeepEval (judge {config.JUDGE_MODEL} via Portkey, trace {judge_trace}) ...")
        evaluation = evaluate(
            test_cases=test_cases[version],
            metrics=build_metrics(judge_trace, contextual_precision=False),
            hyperparameters={"synthesis_prompt": version, "synthesis_model": config.LARGE_MODEL, "judge_model": config.JUDGE_MODEL},
            identifier=f"synthesis-prompt-{version}",
            async_config=AsyncConfig(run_async=False),
            display_config=DisplayConfig(print_results=False, show_indicator=True),
            error_config=ErrorConfig(ignore_errors=True),
            cache_config=CacheConfig(write_cache=False, use_cache=False),
        )
        scores[version] = collect_scores(evaluation)
        summaries[version] = aggregate(scores[version])
    graph_client.close()

    metric_names = list(summaries["v2"]["per_metric"])
    print("\nSide-by-side (mean score; pass rate at threshold 0.70):")
    print(f"  {'metric':<20}{'v1 (naive)':>18}{'v2 (grounded)':>18}{'delta':>10}")
    deltas = {}
    for name in metric_names:
        v1 = summaries["v1"]["per_metric"].get(name, {})
        v2 = summaries["v2"]["per_metric"].get(name, {})
        delta = None if v1.get("mean") is None or v2.get("mean") is None else round(v2["mean"] - v1["mean"], 3)
        deltas[name] = delta
        v1_cell = f"{fmt(v1.get('mean'))} ({v1.get('pass_rate', 0):.0%})"
        v2_cell = f"{fmt(v2.get('mean'))} ({v2.get('pass_rate', 0):.0%})"
        print(f"  {name:<20}{v1_cell:>18}{v2_cell:>18}{'' if delta is None else f'{delta:+.2f}':>10}")

    print("\nPer case (v1 -> v2):")
    for case in [c for c in cases if c.id in answers]:
        cells = []
        for name in metric_names:
            s1 = scores["v1"].get(case.id, {}).get(name, {}).get("score")
            s2 = scores["v2"].get(case.id, {}).get(name, {}).get("score")
            cells.append(f"{name}: {fmt(s1)} -> {fmt(s2)}")
        print(f"  {case.id:<24}" + " | ".join(cells))

    key = ("Faithfulness", "Answer Relevancy")
    improved = [name for name in key if (deltas.get(name) or 0) > 0]
    print("\nVerdict:")
    if len(improved) == len(key):
        print("  v2 (grounded, cited prompt) outscores v1 on both faithfulness and answer relevancy: "
              + ", ".join(f"{n} {deltas[n]:+.2f}" for n in key))
    else:
        print("  v2 did NOT beat v1 on every key metric this run: "
              + ", ".join(f"{n} {'n/a' if deltas.get(n) is None else f'{deltas[n]:+.2f}'}" for n in key)
              + ". Inspect the saved answers before drawing conclusions.")
    print_token_usage()

    path = save_results("prompt_experiment", {
        "summaries": summaries, "deltas": deltas, "scores": scores, "answers": answers,
        "pipeline_errors": pipeline_errors,
        "models": {"small": config.SMALL_MODEL, "large": config.LARGE_MODEL, "judge": config.JUDGE_MODEL},
        "llm_usage": token_usage(),
    })
    print(f"\nSaved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
