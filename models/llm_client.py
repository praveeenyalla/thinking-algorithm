"""Shared LLM client helpers."""

from __future__ import annotations

from typing import Any

from openai import OpenAI


def create_client(base_url: str, api_key: str | None) -> OpenAI:
    """Create an OpenAI-compatible client."""
    if not api_key:
        raise RuntimeError("An API key is required for live model access.")
    return OpenAI(base_url=base_url, api_key=api_key)


def chat_completion_text(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """Call a chat-completions model and normalize text output."""
    client = create_client(base_url, api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise RuntimeError(f"Model call failed: {exc}") from exc

    return extract_text_content(response)


def extract_text_content(response: Any) -> str:
    """Extract plain text from an OpenAI-compatible response."""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected model response: {response}") from exc

    if isinstance(content, list):
        content = "".join(
            getattr(part, "text", "")
            for part in content
            if getattr(part, "type", None) == "text"
        )

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Model returned empty content.")

    return content.strip()
