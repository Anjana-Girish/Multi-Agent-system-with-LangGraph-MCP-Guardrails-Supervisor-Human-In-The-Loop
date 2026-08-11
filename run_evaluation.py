"""
Evaluation harness: runs every case in evaluation_dataset.py against the
live TripMate graph (backend.py) and grades it with both the deterministic
checks the dataset defines and llm_judge.judge_accuracy.

This makes real Groq/Tavily/AviationStack/OpenWeather/Postgres calls, so it
is meant to be run manually, not on import. Results are written incrementally
to evaluation_results.json so a slow/failed case doesn't lose prior progress.
"""

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend import run_travel_agent, resume_travel_agent
from evaluation_dataset import AGENT_ORDER, test_cases
from llm_judge import judge_test_case

RESULTS_PATH = Path(__file__).resolve().parent / "evaluation_results.json"


def _keyword_check(text: str, expected_any: list[str] | None, forbidden_any: list[str] | None) -> bool:
    text_lower = (text or "").lower()

    expected_ok = True
    if expected_any:
        expected_ok = any(kw.lower() in text_lower for kw in expected_any)

    forbidden_ok = True
    if forbidden_any:
        forbidden_ok = not any(kw.lower() in text_lower for kw in forbidden_any)

    return expected_ok and forbidden_ok


def _agent_selection_pass(selected: list[str], required: list[str], allowed_extra: list[str]) -> bool:
    selected_set = set(selected)
    allowed_set = set(required) | set(allowed_extra) | {"itinerary_agent"}
    return set(required) <= selected_set and selected_set <= allowed_set


def _trajectory_pass(selected: list[str]) -> bool:
    return selected == [agent for agent in AGENT_ORDER if agent in selected]


def run_graph_case(test: dict) -> dict:
    start = time.perf_counter()

    try:
        result = run_travel_agent(user_input=test["message"])
    except Exception as exc:
        return {
            "name": test["name"],
            "category": test["category"],
            "error": f"{type(exc).__name__}: {exc}",
            "overall_pass": False,
        }

    latency = time.perf_counter() - start

    guardrail_pass = result["guardrail_allowed"] == test["expected_guardrail_allowed"]
    agent_selection_pass = _agent_selection_pass(
        result["selected_agents"],
        test["required_agents"],
        test.get("allowed_extra_agents", []),
    )
    trajectory_pass = _trajectory_pass(result["selected_agents"])
    approval_pass = result["requires_approval"] == test["expect_requires_approval"]

    answer_field = test.get("answer_field", "answer")
    keyword_answer_field = test.get("keyword_answer_field", answer_field)
    keyword_text = result.get(keyword_answer_field) or result.get("answer", "")
    keyword_pass = _keyword_check(
        keyword_text,
        test.get("expected_keywords_any"),
        test.get("forbidden_keywords_any"),
    )

    judge_input_result = dict(result)
    post_hitl_keyword_pass = None

    hitl_step = test.get("hitl_step")
    if hitl_step and result.get("requires_approval") and result.get("thread_id"):
        hitl_start = time.perf_counter()

        try:
            resume_result = resume_travel_agent(
                thread_id=result["thread_id"],
                approved=hitl_step["approved"],
                feedback=hitl_step.get("feedback", ""),
            )
            latency += time.perf_counter() - hitl_start

            post_answer_field = test.get("post_hitl_answer_field", "answer")
            post_keyword_field = test.get("post_hitl_keyword_answer_field", post_answer_field)
            post_answer_text = resume_result.get(post_keyword_field) or resume_result.get("answer", "")
            post_hitl_keyword_pass = _keyword_check(
                post_answer_text,
                test.get("post_hitl_expected_keywords_any"),
                test.get("post_hitl_forbidden_keywords_any"),
            )

            judge_input_result = dict(resume_result)
            judge_input_result["_hitl_applied"] = True

        except Exception as exc:
            return {
                "name": test["name"],
                "category": test["category"],
                "error": f"HITL resume failed: {type(exc).__name__}: {exc}",
                "overall_pass": False,
            }

    latency_pass = latency <= test.get("max_latency_seconds", float("inf"))

    judge = judge_test_case(test, judge_input_result)

    checks = [
        guardrail_pass,
        agent_selection_pass,
        trajectory_pass,
        approval_pass,
        keyword_pass,
        latency_pass,
        bool(judge["verdict"]),
    ]
    if post_hitl_keyword_pass is not None:
        checks.append(post_hitl_keyword_pass)

    return {
        "name": test["name"],
        "category": test["category"],
        "grading": test["grading"],
        "guardrail_pass": guardrail_pass,
        "agent_selection_pass": agent_selection_pass,
        "trajectory_pass": trajectory_pass,
        "approval_flag_pass": approval_pass,
        "keyword_pass": keyword_pass,
        "post_hitl_keyword_pass": post_hitl_keyword_pass,
        "latency_seconds": round(latency, 2),
        "latency_pass": latency_pass,
        "selected_agents": result["selected_agents"],
        "judge_verdict": judge["verdict"],
        "judge_score": judge["score"],
        "judge_reason": judge["reason"],
        "overall_pass": all(checks),
    }


def run_api_level_case(test: dict) -> dict:
    import app as app_module

    client = TestClient(app_module.app)
    response = client.post("/api/travel", json={"message": test["message"]})
    status_pass = response.status_code == test["expected_http_status"]

    return {
        "name": test["name"],
        "category": test["category"],
        "grading": test["grading"],
        "http_status": response.status_code,
        "expected_http_status": test["expected_http_status"],
        "overall_pass": status_pass,
    }


def main() -> None:
    all_results = []

    for i, test in enumerate(test_cases, start=1):
        print(f"[{i}/{len(test_cases)}] Running: {test['name']} ({test['category']})")

        if test["category"] == "api_level":
            case_result = run_api_level_case(test)
        else:
            case_result = run_graph_case(test)

        all_results.append(case_result)
        status = "PASS" if case_result.get("overall_pass") else "FAIL"
        print(f"    -> {status}" + (f" | error: {case_result['error']}" if case_result.get("error") else ""))

        RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    total = len(all_results)
    passed = sum(1 for r in all_results if r.get("overall_pass"))

    print("\n" + "=" * 60)
    print("SCORECARD")
    print("=" * 60)
    for r in all_results:
        mark = "PASS" if r.get("overall_pass") else "FAIL"
        judge_bit = f" judge={r['judge_score']}/10" if "judge_score" in r else ""
        print(f"{mark:5s} {r['name']:40s}{judge_bit}")

    print("-" * 60)
    print(f"Overall: {passed}/{total} passed ({100 * passed / total:.1f}%)")
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
