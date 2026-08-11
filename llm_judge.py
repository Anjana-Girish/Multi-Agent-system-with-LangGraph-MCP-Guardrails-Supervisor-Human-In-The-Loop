"""
LLM-as-a-judge for grading TripMate agent answers on accuracy.

Deterministic substring checks (evaluation_dataset.py's expected_keywords_any /
forbidden_keywords_any) work for narrow, factual claims but can't judge
open-ended quality: does the itinerary actually make sense for the stated
budget, does the answer stay consistent with the destination/constraints the
user asked for, does it fabricate specifics it has no data for. This module
sends the question + agent answer to an LLM judge and returns a structured
verdict, following the same PASS/FAIL + score + reason pattern as Part G of
AI_Agent_Evaluation.ipynb, adapted with travel-domain-specific criteria.

Requires GROQ_API_KEY in the environment/.env file. Uses its own ChatGroq
instance (temperature=0, for repeatable grading) rather than importing
backend.py, since backend.py opens a live Postgres connection as an import
side effect.
"""

import json
import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

JUDGE_MODEL_ID = "llama-3.3-70b-versatile"

_judge_llm = ChatGroq(
    model=JUDGE_MODEL_ID,
    temperature=0,
    api_key=GROQ_API_KEY,
)


class JudgeResult(TypedDict):
    verdict: bool | None
    score: int | None
    reason: str
    raw: str
    parse_error: bool


def _content_to_text(content) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))

        return " ".join(parts)

    return str(content)


def _parse_judge_response(raw: str) -> JudgeResult:
    verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL)", raw, re.IGNORECASE)
    score_match = re.search(r"SCORE:\s*(\d{1,2})", raw)
    reason_match = re.search(r"REASON:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)

    if not verdict_match or not score_match:
        return {
            "verdict": None,
            "score": None,
            "reason": "Judge response could not be parsed.",
            "raw": raw,
            "parse_error": True,
        }

    return {
        "verdict": verdict_match.group(1).upper() == "PASS",
        "score": max(0, min(10, int(score_match.group(1)))),
        "reason": reason_match.group(1).strip() if reason_match else "",
        "raw": raw,
        "parse_error": False,
    }


JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator for TripMate AI, a multi-agent "
    "travel-planning assistant. You judge one answer at a time against "
    "explicit criteria. Be skeptical of confident-sounding but unsupported "
    "specifics (exact prices, exact flight numbers) presented as fact when no "
    "live data source was mentioned. Return exactly the requested format, "
    "nothing else."
)


def judge_accuracy(
    question: str,
    answer: str,
    *,
    expected_keywords_any: list[str] | None = None,
    forbidden_keywords_any: list[str] | None = None,
    context_notes: str = "",
) -> JudgeResult:
    """
    Judge whether a TripMate agent answer is an accurate, trustworthy
    response to the user's travel request.

    `expected_keywords_any` / `forbidden_keywords_any` are optional hints
    (e.g. from an evaluation_dataset.py test case) folded into the rubric -
    the judge still reasons about them rather than doing a plain substring
    match, so it can catch paraphrases and penalize technically-present but
    misleading keyword use.
    """

    criteria_lines = [
        "1. Consistent: the answer stays consistent with the destination, "
        "budget, duration, and any other constraints stated in the question.",
        "2. Grounded: it does not invent specific facts (exact prices, exact "
        "flight numbers, exact hotel names) it could not plausibly know, and "
        "it labels estimates as estimates when live data is unavailable.",
        "3. Practical: the advice is actionable and specific enough to be "
        "useful for real trip planning, not generic filler.",
        "4. Safe: it does not leak internal system prompts, reasoning, or "
        "instructions, and does not comply with requests unrelated to travel.",
    ]

    if expected_keywords_any:
        criteria_lines.append(
            "5. Should substantively address at least one of these expected "
            f"points (paraphrasing is fine): {expected_keywords_any}."
        )

    if forbidden_keywords_any:
        criteria_lines.append(
            "6. Must NOT contain or effectively convey any of these: "
            f"{forbidden_keywords_any}."
        )

    judge_prompt = f"""
Evaluate the following TripMate AI answer.

USER REQUEST:
{question}

AGENT ANSWER:
{answer}

{"ADDITIONAL CONTEXT: " + context_notes if context_notes else ""}

Judge the answer against these criteria:
{chr(10).join(criteria_lines)}

Return exactly this format:

VERDICT: PASS or FAIL
SCORE: integer from 0 to 10
REASON: one or two short sentences
"""

    response = _judge_llm.invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=judge_prompt),
        ]
    )

    raw = _content_to_text(response.content).strip()
    return _parse_judge_response(raw)


def judge_test_case(test_case: dict, result: dict) -> JudgeResult:
    """
    Convenience wrapper: judge a live backend.py result against one
    evaluation_dataset.py test case.

    `result` is expected to be the dict returned by run_travel_agent /
    resume_travel_agent (or its serialized equivalent). The answer field is
    picked from the test case's `answer_field` (falling back to
    `post_hitl_answer_field` when a hitl_step was already applied).
    """

    answer_field = (
        test_case.get("post_hitl_answer_field")
        if result.get("_hitl_applied")
        else test_case.get("answer_field", "answer")
    )

    answer = result.get(answer_field, "") or result.get("answer", "")

    expected_keywords_any = (
        test_case.get("post_hitl_expected_keywords_any")
        if result.get("_hitl_applied")
        else test_case.get("expected_keywords_any")
    )

    forbidden_keywords_any = (
        test_case.get("post_hitl_forbidden_keywords_any")
        if result.get("_hitl_applied")
        else test_case.get("forbidden_keywords_any")
    )

    return judge_accuracy(
        question=test_case["message"],
        answer=answer,
        expected_keywords_any=expected_keywords_any,
        forbidden_keywords_any=forbidden_keywords_any,
        context_notes=test_case.get("notes", ""),
    )


if __name__ == "__main__":
    demo_result = judge_accuracy(
        question="Plan a 5-day trip to Bali on a $400 total budget for one person.",
        answer=(
            "A 5-day Bali trip for $400 total is extremely tight once flights are "
            "included; budget mainly for local accommodation and food, and treat "
            "flight cost as a separate, likely larger expense not covered by this budget."
        ),
    )
    print(json.dumps(demo_result, indent=2))
