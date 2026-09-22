"""Structured DeepEval test suite: every case in eval/dataset.py runs through the full agent pipeline
(orchestrator -> retrieval -> social listening -> synthesis) and is scored on faithfulness,
answer relevancy, contextual precision and key-fact coverage.

Two ways to run it:
  python eval/test_suite.py [--limit N] [--case ID ...]   aggregate report (mean score, pass rates)
  deepeval test run eval/test_suite.py [-k CASE_ID]       pytest-style test suite (one test per case)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from deepeval import assert_test, evaluate  # noqa: E402
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from app import config, graph_client, llm_client  # noqa: E402
from app.config import configure_console  # noqa: E402
from app.pipeline import run_pipeline  # noqa: E402
from eval.dataset import CASES, EvalCase, select_cases  # noqa: E402
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


def build_test_case(case: EvalCase) -> LLMTestCase:
    result = run_pipeline(case.query)
    return to_test_case(case, result, result.answer)


# --- pytest / `deepeval test run` mode -------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_agri_case(case: EvalCase) -> None:
    llm_client.require_portkey()
    assert_test(build_test_case(case), build_metrics(llm_client.new_trace_id("eval-judge")), run_async=False)


# --- script mode: aggregate report -----------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only evaluate the first N cases (free-tier budget)")
    parser.add_argument("--case", nargs="*", help="only evaluate these case ids")
    args = parser.parse_args(argv)
    llm_client.require_portkey()
    cases = select_cases(args.limit, args.case)

    print(f"Running {len(cases)} case(s) through the agent pipeline (synthesis prompt v2, {config.LARGE_MODEL})")
    test_cases: list[LLMTestCase] = []
    pipeline_errors: dict[str, str] = {}
    for case in cases:
        print(f"  - {case.id}")
        try:
            test_cases.append(build_test_case(case))
        except llm_client.LLMRateLimitError as exc:
            pipeline_errors[case.id] = str(exc)
            print(f"    pipeline failed: {exc}")
    if not test_cases:
        print("No test cases could be generated.")
        return 1

    judge_trace = llm_client.new_trace_id("eval-judge")
    print(f"\nScoring with DeepEval (judge {config.JUDGE_MODEL} via Portkey, trace {judge_trace}) ...")
    evaluation = evaluate(
        test_cases=test_cases,
        metrics=build_metrics(judge_trace),
        hyperparameters={"synthesis_prompt": "v2", "synthesis_model": config.LARGE_MODEL, "judge_model": config.JUDGE_MODEL},
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(print_results=False, show_indicator=True),
        error_config=ErrorConfig(ignore_errors=True),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
    )
    graph_client.close()

    scores = collect_scores(evaluation)
    summary = aggregate(scores)
    metric_names = list(summary["per_metric"])

    print("\nPer-case scores (threshold 0.70; Key Facts requires all expected facts):")
    header = f"  {'case':<24}" + "".join(f"{name:>22}" for name in metric_names) + "   result"
    print(header)
    for case_id, case_scores in scores.items():
        cells = "".join(
            f"{fmt(case_scores[name]['score']) + (' ✓' if case_scores[name]['success'] else ' ✗'):>22}"
            if name in case_scores else f"{'-':>22}"
            for name in metric_names
        )
        passed = all(m["success"] for m in case_scores.values())
        print(f"  {case_id:<24}{cells}   {'PASS' if passed else 'FAIL'}")
    for case_id, error in pipeline_errors.items():
        print(f"  {case_id:<24}pipeline error: {error}")

    print("\nAggregate:")
    for name, row in summary["per_metric"].items():
        extra = f"  ({row['errors']} scoring error(s))" if row["errors"] else ""
        print(f"  {name:<22} mean {fmt(row['mean'])}   pass rate {row['pass_rate']:.0%}{extra}")
    print(f"  {'overall':<22} {summary['cases_passed']}/{summary['cases']} cases passed all metrics "
          f"({summary['overall_pass_rate']:.0%})")
    if evaluation.confident_link:
        print(f"  Confident AI report: {evaluation.confident_link}")
    print_token_usage()

    path = save_results("test_suite", {
        "summary": summary, "scores": scores, "pipeline_errors": pipeline_errors,
        "models": {"small": config.SMALL_MODEL, "large": config.LARGE_MODEL, "judge": config.JUDGE_MODEL},
        "llm_usage": token_usage(),
        "test_cases": [tc.model_dump() if hasattr(tc, "model_dump") else vars(tc) for tc in test_cases],
    })
    print(f"\nSaved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
