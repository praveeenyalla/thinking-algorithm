"""Final response cleanup for user-visible answers."""

from __future__ import annotations

import re


INTERNAL_LABELS = (
    "score:",
    "scores:",
    "evaluation:",
    "evaluations:",
    "controller:",
    "pipeline:",
    "critique:",
    "critiques:",
    "rank:",
    "research brief:",
    "memory cleanup:",
    "response mode:",
)

PERSPECTIVE_HEADER_PATTERN = re.compile(r"^[a-z][a-z\s]+perspective:\s*$", re.IGNORECASE)


def format_output(text: str, response_style: str = "simple") -> str:
    """Strip system artifacts and normalize the final answer."""
    cleaned = _strip_code_fences(text.replace("\r\n", "\n")).strip()
    cleaned = re.sub(r"^(final answer|answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE)

    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if not line:
            lines.append("")
            continue
        if PERSPECTIVE_HEADER_PATTERN.match(line):
            continue
        if any(lowered.startswith(label) for label in INTERNAL_LABELS):
            continue
        if lowered.endswith("alignment:"):
            continue
        if lowered.startswith("summary:"):
            lines.append(line.split(":", maxsplit=1)[1].strip())
            continue
        if lowered.startswith("analysis:"):
            lines.append(line.split(":", maxsplit=1)[1].strip())
            continue
        if lowered.startswith("implications:"):
            lines.append(line.split(":", maxsplit=1)[1].strip())
            continue
        if lowered.startswith("recommended next step:"):
            lines.append(f"Next step: {line.split(':', maxsplit=1)[1].strip()}")
            continue
        if lowered.startswith("strengths:") or lowered.startswith("missing points:"):
            continue
        lines.append(line)

    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)

    if response_style == "simple":
        normalized = re.sub(r"^#{1,6}\s+.*$", "", normalized, flags=re.MULTILINE)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()

    if not normalized:
        raise RuntimeError("Model output could not be formatted into a valid answer.")
    return normalized


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()
