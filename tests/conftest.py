import json
import re

import pytest

from models import llm_client


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client, "llm_call_for_purpose", _fake_llm_call_for_purpose)


def _fake_llm_call_for_purpose(
    settings,
    purpose: str,
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> str:
    if '"key_points": [' in user_prompt and '"risks": [' in user_prompt:
        question = _extract_section(user_prompt, "Question:")
        topic = _topic_from_question(question)
        return json.dumps(
            {
                "summary": f"{topic} needs clear context, accurate scope, and grounded tradeoffs.",
                "key_points": [
                    f"{topic} should be explained using the most relevant facts first.",
                    f"The answer should connect the idea to practical usage.",
                    f"Examples help when the topic feels abstract or technical.",
                ],
                "risks": [
                    "Generic wording can make different questions sound the same.",
                    "Ignoring tradeoffs can weaken the answer.",
                    "Reusing old memory without adaptation can mislead the user.",
                ],
            }
        )

    if '"missing_points": [' in user_prompt:
        return json.dumps(
            {
                "summary": "The draft is useful but should be more specific and more concrete.",
                "strengths": [
                    "It addresses the user's question directly.",
                    "It stays readable and grounded.",
                ],
                "missing_points": [
                    "Add a sharper example or concrete detail.",
                    "Tighten any vague sentence.",
                    "Make the practical takeaway clearer.",
                ],
            }
        )

    if "Evaluate this candidate answer using the retrieved context." in user_prompt:
        return json.dumps(
            {
                "criteria": {
                    "correctness": {"score": 8, "rationale": "The answer matches the question and stays grounded."},
                    "clarity": {"score": 8, "rationale": "The answer is readable and organized."},
                    "usefulness": {"score": 8, "rationale": "The answer gives the user something actionable."},
                },
                "total_score": 8,
                "threshold_passed": True,
                "summary_reason": "The answer is clear, relevant, and useful.",
            }
        )

    if "Answer the user's question in the requested style." in user_prompt:
        question = _extract_section(user_prompt, "User question:")
        return _final_answer(question, _style_from_prompt(user_prompt))

    if "Improve the draft answer below." in user_prompt:
        question = _extract_section(user_prompt, "User question:")
        draft = _extract_section(user_prompt, "Current draft:")
        topic = _topic_from_question(question)
        if _is_step_question(question):
            return (
                f"To approach {topic}, start with a small working version and expand it carefully.\n\n"
                "1. Define the goal and the main user need.\n"
                "2. Build the simplest version that solves that need.\n"
                "3. Test it with a real example and refine the weak spots."
            )
        return f"{draft}\n\nA clearer explanation should connect {topic} to a practical example or decision."

    if "Create one synthesized answer to the user's question." in user_prompt:
        question = _extract_section(user_prompt, "Question:")
        topic = _topic_from_question(question)
        return (
            f"Overview\n{topic} is easiest to explain when you combine the core idea, the important tradeoffs, "
            "and one practical example.\n\n"
            "Key Points\nIt helps to start with what it is, then how it works, and then why it matters in real use.\n\n"
            f"Example\nA simple example of {topic} makes the explanation concrete and easier to remember."
        )

    if "Write one grounded candidate answer to the user's question." in user_prompt:
        question = _extract_section(user_prompt, "Question:")
        topic = _topic_from_question(question)
        return (
            f"{topic} should be explained using the clearest known facts and the main real-world constraints. "
            f"For example, a short practical case makes {topic} easier to understand."
        )

    if "Write one logically strong answer to the user's question." in user_prompt:
        question = _extract_section(user_prompt, "Question:")
        topic = _topic_from_question(question)
        return (
            f"{topic} becomes clearer when you separate the main idea, the assumptions behind it, and the tradeoffs "
            "that affect the final conclusion."
        )

    if "Write one creative but practical answer to the user's question." in user_prompt:
        question = _extract_section(user_prompt, "Question:")
        topic = _topic_from_question(question)
        return (
            f"One useful way to explain {topic} is through a small scenario or analogy, as long as it still stays "
            "practical and realistic."
        )

    if "Create one candidate answer for the user's question." in user_prompt:
        question = _extract_section(user_prompt, "Question:")
        topic = _topic_from_question(question)
        angle = _extract_section(user_prompt, "Angle:").lower()
        if "tradeoffs" in angle or "assumptions" in angle:
            return (
                f"{topic} becomes clearer when you separate the main assumptions, the important tradeoffs, "
                "and the conclusion that follows from them."
            )
        if "practical" in angle or "actionable" in angle:
            return (
                f"{topic} is easier to apply when the answer includes one concrete use case, one likely risk, "
                "and one practical next step."
            )
        return f"{topic} can be explained clearly with a direct answer, a concrete detail, and a short example."

    raise AssertionError(f"Unexpected prompt in fake LLM stub:\n{system_prompt}\n---\n{user_prompt}")


def _final_answer(question: str, style: str) -> str:
    topic = _topic_from_question(question)

    if _is_comparison_question(question):
        left, right = _comparison_parts(question)
        return (
            f"{left} and {right} solve similar problems, but they feel different in day-to-day use.\n\n"
            f"- Learning curve: {left} usually feels more flexible, while {right} often feels simpler to start with.\n"
            f"- Ecosystem: compare the libraries, tooling, and team familiarity before choosing.\n"
            "- Best fit: the better choice depends on the project size, team skills, and long-term maintenance."
        )

    if _is_step_question(question):
        return (
            f"To work on {topic}, keep the process simple and incremental.\n\n"
            "1. Define the goal and the exact problem you want to solve.\n"
            "2. Start with the smallest useful version instead of building everything at once.\n"
            "3. Test it with a real example and improve the weak parts."
        )

    if style == "simple":
        if "java" in topic.lower():
            return (
                "Java is a programming language used to build applications like Android apps, backend services, "
                "and enterprise software.\n\n"
                "It is popular because the same Java code can run on different operating systems through the JVM.\n\n"
                "For example, many business applications and older Android apps were built with Java."
            )
        return (
            f"{topic} is a topic best answered with the core idea first and the practical meaning second.\n\n"
            f"For example, a small real-world case usually makes {topic} much easier to understand."
        )

    if style == "detailed":
        return (
            f"Overview\n{topic} needs a deeper explanation because the question asks for more than a quick definition.\n\n"
            f"How It Works\nStart with the core concept, then connect it to the important components, tradeoffs, and real-world behavior around {topic}.\n\n"
            f"Example\nA concrete example makes {topic} easier to understand and easier to apply.\n\n"
            "Why It Matters\nA detailed explanation helps when you need conceptual clarity before building, comparing, or researching further."
        )

    return (
        f"{topic} is easier to understand when the answer stays focused on the main idea, why it matters, and one concrete example.\n\n"
        f"- Main idea: explain what {topic} is in plain language.\n"
        "- Why it matters: connect it to practical use.\n"
        f"- Example: show one simple case where {topic} appears in real work."
    )


def _extract_section(text: str, label: str) -> str:
    pattern = re.escape(label) + r"\n(.*?)(?:\n\n[A-Za-z][A-Za-z /-]+:\n|\Z)"
    match = re.search(pattern, text, flags=re.S)
    return match.group(1).strip() if match else ""


def _style_from_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    if "short, direct answer in plain language with no headings" in lowered:
        return "simple"
    if "give a detailed explanation" in lowered:
        return "detailed"
    return "structured"


def _topic_from_question(question: str) -> str:
    cleaned = question.strip().rstrip("?.!")
    lowered = cleaned.lower()
    prefixes = (
        "what is ",
        "what's ",
        "explain ",
        "define ",
        "describe ",
        "tell me about ",
        "how to ",
        "how do i ",
        "compare ",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            lowered = cleaned.lower()
            break
    for suffix in (" with examples", " step by step", " in detail", " deeply"):
        if lowered.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
            lowered = cleaned.lower()
    return cleaned or "the topic"


def _is_step_question(question: str) -> bool:
    lowered = question.lower()
    return "step by step" in lowered or lowered.startswith("how to ") or lowered.startswith("how do i ")


def _is_comparison_question(question: str) -> bool:
    lowered = question.lower()
    return " vs " in lowered or " versus " in lowered or "compare" in lowered


def _comparison_parts(question: str) -> tuple[str, str]:
    cleaned = question.strip().rstrip("?.!")
    lowered = cleaned.lower()
    if " vs " in lowered:
        left, right = re.split(r"\s+vs\s+", cleaned, maxsplit=1, flags=re.I)
        return left.strip().removeprefix("Compare ").removeprefix("compare "), right.strip()
    if " versus " in lowered:
        left, right = re.split(r"\s+versus\s+", cleaned, maxsplit=1, flags=re.I)
        return left.strip().removeprefix("Compare ").removeprefix("compare "), right.strip()
    if lowered.startswith("compare "):
        cleaned = cleaned[8:]
        if " and " in cleaned.lower():
            left, right = re.split(r"\s+and\s+", cleaned, maxsplit=1, flags=re.I)
            return left.strip(), right.strip()
    return "Option A", "Option B"
