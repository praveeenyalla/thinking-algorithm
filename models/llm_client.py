"""Shared LLM client helpers."""

from __future__ import annotations

import json
from typing import Any, Literal

import httpx

from config import Settings

Purpose = Literal["generator", "refiner", "judge"]


def create_client(base_url: str, api_key: str | None) -> httpx.Client:
    """Create an HTTP client for OpenAI-compatible APIs."""
    if not api_key:
        raise RuntimeError(
            "Model API key missing. Set GENERATOR_API_KEY, REFINER_API_KEY, or JUDGE_API_KEY, "
            "or use API_KEY, NVIDIA_API_KEY, or OPENAI_API_KEY. You can also place the key in "
            "D:\\Thinking algorithm\\.env and restart the server."
        )
    return httpx.Client(
        base_url=base_url.rstrip("/") + "/",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=120.0,
    )


def llm_call(
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
    """Call a chat-completions model and return normalized text."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        with create_client(base_url, api_key) as client:
            response = client.post("chat/completions", json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = _safe_response_text(exc.response)
        raise RuntimeError(f"Model call failed: {exc.response.status_code} {detail}") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Model call failed: {exc}") from exc

    return extract_text_content(response)


def llm_call_for_purpose(
    settings: Settings,
    purpose: Purpose,
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Call the configured model for a specific reasoning purpose."""
    config_map = {
        "generator": (
            settings.generator_base_url,
            settings.generator_api_key,
            settings.generator_model,
            settings.generator_temperature,
            settings.generator_top_p,
            settings.generator_max_tokens,
        ),
        "refiner": (
            settings.refiner_base_url,
            settings.refiner_api_key,
            settings.refiner_model,
            settings.refiner_temperature,
            settings.refiner_top_p,
            settings.refiner_max_tokens,
        ),
        "judge": (
            settings.judge_base_url,
            settings.judge_api_key,
            settings.judge_model,
            0.0,
            1.0,
            800,
        ),
    }
    base_url, api_key, model, default_temperature, default_top_p, default_max_tokens = config_map[purpose]
    return llm_call(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=default_temperature if temperature is None else temperature,
        top_p=default_top_p if top_p is None else top_p,
        max_tokens=default_max_tokens if max_tokens is None else max_tokens,
    )


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
    """Backward-compatible wrapper around llm_call."""
    return llm_call(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def extract_text_content(response: Any) -> str:
    """Extract plain text from an OpenAI-compatible response."""
    if isinstance(response, httpx.Response):
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model returned invalid JSON: {_safe_response_text(response)}") from exc
    elif isinstance(response, dict):
        payload = response
    else:
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected model response: {response}") from exc
        return _normalize_content(content)

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected model response: {payload}") from exc
    return _normalize_content(content)


def _normalize_content(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif getattr(part, "type", None) == "text":
                parts.append(getattr(part, "text", ""))
        content = "".join(parts)

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Model returned empty content.")
    return content.strip()


def _safe_response_text(response: httpx.Response) -> str:
    text = response.text.strip()
    return text[:500] if text else "empty response body"
