"""LLM client — unified interface supporting Anthropic and OpenAI-compatible APIs.

Switches protocol based on ``settings.llm_provider``:
- ``anthropic``: MiMo via Anthropic protocol
- ``openai``: DeepSeek / LongCat / any OpenAI-compatible endpoint
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from typing import Any

from models.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anthropic client (for MiMo)
# ---------------------------------------------------------------------------

_anthropic_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        _anthropic_client = anthropic.Anthropic(
            api_key=settings.mimo_api_key,
            base_url=settings.mimo_base_url,
            timeout=60.0,
        )
    return _anthropic_client


def _chat_anthropic(system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    client = _get_anthropic_client()
    response = client.messages.create(
        model=settings.mimo_model,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        thinking={"type": "disabled"},
    )
    text_parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text" and hasattr(block, "text"):
            text_parts.append(block.text)
    return "\n".join(text_parts).strip()


def _stream_anthropic(system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> Generator[str, None, None]:
    client = _get_anthropic_client()
    with client.messages.stream(
        model=settings.mimo_model,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        thinking={"type": "disabled"},
    ) as stream:
        for text in stream.text_stream:
            yield text


# ---------------------------------------------------------------------------
# OpenAI-compatible client (for DeepSeek / LongCat / etc.)
# ---------------------------------------------------------------------------

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        import openai

        _openai_client = openai.OpenAI(
            api_key=settings.mimo_api_key,
            base_url=settings.mimo_base_url,
            timeout=60.0,
        )
    return _openai_client


def _chat_openai(system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=settings.mimo_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    msg = response.choices[0].message
    content = (msg.content or "").strip()
    # Some reasoning models put output in reasoning_content when content is empty
    if not content and hasattr(msg, "reasoning_content"):
        content = (msg.reasoning_content or "").strip()
    return content


def _stream_openai(system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> Generator[str, None, None]:
    client = _get_openai_client()
    stream = client.chat.completions.create(
        model=settings.mimo_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = delta.content or ""
        # Some reasoning models put output in reasoning_content when content is empty
        if not content and hasattr(delta, "reasoning_content") and delta.reasoning_content:
            content = delta.reasoning_content
        if content:
            yield content


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _dispatch(system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    provider = settings.llm_provider.lower().strip()
    try:
        if provider == "anthropic":
            return _chat_anthropic(system_prompt, user_prompt, temperature, max_tokens)
        else:
            return _chat_openai(system_prompt, user_prompt, temperature, max_tokens)
    except Exception as exc:
        logger.error("LLM API error (provider=%s): %s", provider, exc)
        raise RuntimeError(f"LLM API error: {exc}") from exc


def _dispatch_stream(system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> Generator[str, None, None]:
    provider = settings.llm_provider.lower().strip()
    try:
        if provider == "anthropic":
            yield from _stream_anthropic(system_prompt, user_prompt, temperature, max_tokens)
        else:
            yield from _stream_openai(system_prompt, user_prompt, temperature, max_tokens)
    except Exception as exc:
        logger.error("LLM streaming API error (provider=%s): %s", provider, exc)
        raise RuntimeError(f"LLM streaming API error: {exc}") from exc


def chat(
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Send a chat request and return the assistant's text."""

    temp = temperature if temperature is not None else settings.mimo_temperature
    tokens = max_tokens if max_tokens is not None else settings.mimo_max_tokens
    return _dispatch(system_prompt, user_prompt, temp, tokens)


def chat_stream(
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Generator[str, None, None]:
    """Send a chat request and yield text chunks as they arrive."""

    temp = temperature if temperature is not None else settings.mimo_temperature
    tokens = max_tokens if max_tokens is not None else settings.mimo_max_tokens
    yield from _dispatch_stream(system_prompt, user_prompt, temp, tokens)


def chat_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Send a chat request expecting a JSON response and parse it."""

    raw = chat(system_prompt, user_prompt, temperature, max_tokens)
    return _extract_json(raw)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM response text, handling markdown fences."""

    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Try direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(cleaned[start : end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    raise RuntimeError(f"Failed to parse JSON from LLM response: {text[:200]}")
