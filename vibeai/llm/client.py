"""Thin wrapper around the OpenAI Responses API, with disk caching.

Caching matters here because prompt testing means re-running the same
inputs repeatedly while iterating on prompts/metrics - we don't want to
re-pay for (or wait on) an unchanged call.
"""

import asyncio
import base64
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Coroutine

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAI

from vibeai.llm.budget import get_budget
from vibeai.llm.errors import InsufficientQuotaError

load_dotenv()

DEFAULT_MODEL = "gpt-5"
CACHE_DIR = Path(".cache/llm")

# Batch runs over hundreds of images sustain enough concurrent requests to hit
# rate limits repeatedly, not just transiently - the SDK's default of 2 isn't
# enough headroom, so give it more retries with backoff before giving up.
# Retries are handled manually (see _call_with_retry* below) rather than by
# the SDK, so an out-of-credit account fails immediately instead of retrying
# a call that can never succeed.
MAX_RETRIES = 8
RETRY_BASE_DELAY_SECONDS = 1.0
RETRY_MAX_DELAY_SECONDS = 30.0

_RETRYABLE_EXCEPTIONS = (APIStatusError, APIConnectionError)


@lru_cache
def get_client() -> OpenAI:
    return OpenAI(max_retries=0)


@lru_cache
def get_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(max_retries=0)


def _is_insufficient_quota(exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    return isinstance(body, dict) and body.get("error", {}).get("code") == "insufficient_quota"


def _retry_delay(attempt: int) -> float:
    return min(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)), RETRY_MAX_DELAY_SECONDS)


def _call_with_retry(fn: Callable[[], object], max_retries: int = MAX_RETRIES):
    attempt = 0
    while True:
        try:
            return fn()
        except _RETRYABLE_EXCEPTIONS as e:
            if isinstance(e, APIStatusError) and _is_insufficient_quota(e):
                raise InsufficientQuotaError(
                    "OpenAI account has no remaining credit (insufficient_quota) - "
                    "retrying will not help."
                ) from e
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(_retry_delay(attempt))


async def _call_with_retry_async(
    coro_fn: Callable[[], Coroutine[None, None, object]], max_retries: int = MAX_RETRIES
):
    attempt = 0
    while True:
        try:
            return await coro_fn()
        except _RETRYABLE_EXCEPTIONS as e:
            if isinstance(e, APIStatusError) and _is_insufficient_quota(e):
                raise InsufficientQuotaError(
                    "OpenAI account has no remaining credit (insufficient_quota) - "
                    "retrying will not help."
                ) from e
            attempt += 1
            if attempt > max_retries:
                raise
            await asyncio.sleep(_retry_delay(attempt))


def _cache_path(model: str, prompt: str, image_bytes: bytes | None) -> Path:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(prompt.encode())
    if image_bytes is not None:
        h.update(image_bytes)
    return CACHE_DIR / f"{h.hexdigest()}.json"


def _read_cache(path: Path) -> str | None:
    if path.exists():
        return json.loads(path.read_text())["output"]
    return None


def _write_cache(path: Path, output: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"output": output}))


def _record_usage(response) -> None:
    usage = getattr(response, "usage", None)
    if usage is not None:
        get_budget().record(usage.total_tokens)


def call_text(prompt: str, model: str = DEFAULT_MODEL, use_cache: bool = True) -> str:
    path = _cache_path(model, prompt, None)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached

    get_budget().check()
    response = _call_with_retry(
        lambda: get_client().responses.create(
            model=model,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        )
    )
    _record_usage(response)
    output = response.output_text

    if use_cache:
        _write_cache(path, output)
    return output


async def call_text_async(prompt: str, model: str = DEFAULT_MODEL, use_cache: bool = True) -> str:
    path = _cache_path(model, prompt, None)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached

    get_budget().check()
    response = await _call_with_retry_async(
        lambda: get_async_client().responses.create(
            model=model,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        )
    )
    _record_usage(response)
    output = response.output_text

    if use_cache:
        _write_cache(path, output)
    return output


def call_with_image(
    prompt: str,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    model: str = DEFAULT_MODEL,
    use_cache: bool = True,
) -> str:
    path = _cache_path(model, prompt, image_bytes)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached

    get_budget().check()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    response = _call_with_retry(
        lambda: get_client().responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime_type};base64,{image_b64}",
                        },
                    ],
                }
            ],
        )
    )
    _record_usage(response)
    output = response.output_text

    if use_cache:
        _write_cache(path, output)
    return output


async def call_with_image_async(
    prompt: str,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    model: str = DEFAULT_MODEL,
    use_cache: bool = True,
) -> str:
    path = _cache_path(model, prompt, image_bytes)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached

    get_budget().check()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    response = await _call_with_retry_async(
        lambda: get_async_client().responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime_type};base64,{image_b64}",
                        },
                    ],
                }
            ],
        )
    )
    _record_usage(response)
    output = response.output_text

    if use_cache:
        _write_cache(path, output)
    return output
