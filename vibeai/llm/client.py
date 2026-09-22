"""Thin wrapper around the OpenAI Responses API and the Anthropic Messages
API, with disk caching.

Caching matters here because prompt testing means re-running the same
inputs repeatedly while iterating on prompts/metrics - we don't want to
re-pay for (or wait on) an unchanged call.

Provider is inferred from the model id (see ``is_anthropic_model``), so
callers pass a model string and nothing else changes. The two providers are
not fully symmetric: token budgeting and the per-call usage log track the
OpenAI account's TPD limit specifically, so the Anthropic path skips both and
gets caching + retries only.
"""

import asyncio
import base64
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Coroutine

from anthropic import (
    Anthropic,
    APIConnectionError as AnthropicAPIConnectionError,
    APIStatusError as AnthropicAPIStatusError,
    AsyncAnthropic,
)
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAI

from vibeai.llm.budget import get_budget
from vibeai.llm.errors import InsufficientQuotaError, RefusalError
from vibeai.llm.usage_log import log_call

load_dotenv()

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_DECOMPOSITION_MODEL = "gpt-5"
DEFAULT_EVAL_MODEL = "gpt-5"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

CACHE_DIR = Path(".cache/llm")

ANTHROPIC_MODEL_PREFIX = "claude-"
ANTHROPIC_MAX_TOKENS = 16000
ANTHROPIC_EFFORT: str | None = "medium"
ANTHROPIC_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

def set_anthropic_effort(effort: str | None) -> None:
    if effort is not None and effort not in ANTHROPIC_EFFORT_LEVELS:
        raise ValueError(
            f"effort must be None or one of {ANTHROPIC_EFFORT_LEVELS}, got {effort!r}"
        )
    global ANTHROPIC_EFFORT
    ANTHROPIC_EFFORT = effort

MAX_RETRIES = 8
RETRY_BASE_DELAY_SECONDS = 1.0
RETRY_MAX_DELAY_SECONDS = 30.0

_RETRYABLE_EXCEPTIONS = (
    APIStatusError,
    APIConnectionError,
    AnthropicAPIStatusError,
    AnthropicAPIConnectionError,
)


@lru_cache
def get_client() -> OpenAI:
    return OpenAI(max_retries=0)


@lru_cache
def get_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(max_retries=0)


@lru_cache
def get_anthropic_client() -> Anthropic:
    return Anthropic(max_retries=0)


@lru_cache
def get_anthropic_async_client() -> AsyncAnthropic:
    return AsyncAnthropic(max_retries=0)


def is_anthropic_model(model: str) -> bool:
    """Which provider a model id belongs to. Keeps provider selection out of
    every call site - callers just pass a model string."""
    return model.startswith(ANTHROPIC_MODEL_PREFIX)


def _is_insufficient_quota(exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    return isinstance(body, dict) and body.get("error", {}).get("code") == "insufficient_quota"


def _retry_delay(attempt: int) -> float:
    return min(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)), RETRY_MAX_DELAY_SECONDS)


def _call_with_retry(
    fn: Callable[[], object],
    max_retries: int = MAX_RETRIES,
    extra_retryable: tuple[type[Exception], ...] = (),
):
    """Retry ``fn`` on transient API errors and, if ``extra_retryable`` is
    given (e.g. a metric's output-validation ValueError), on those too - all
    sharing one exponential-backoff budget rather than each having its own,
    so a flaky call can't multiply worst-case latency across retry layers."""
    retryable = _RETRYABLE_EXCEPTIONS + extra_retryable
    attempt = 0
    while True:
        try:
            return fn()
        except retryable as e:
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
    coro_fn: Callable[[], Coroutine[None, None, object]],
    max_retries: int = MAX_RETRIES,
    extra_retryable: tuple[type[Exception], ...] = (),
):
    retryable = _RETRYABLE_EXCEPTIONS + extra_retryable
    attempt = 0
    while True:
        try:
            return await coro_fn()
        except retryable as e:
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
    """Cache key. Anything that changes what the API returns has to be in here:
    effort is mixed in for Anthropic models, or lowering it would hand back
    output generated at the previous effort and quietly invalidate any
    comparison between the two. Computed here rather than at the call sites so
    it cannot be forgotten at one of them.

    Nothing is mixed in when effort is unset, so entries cached before this
    setting existed stay valid.
    """
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(prompt.encode())
    if image_bytes is not None:
        h.update(image_bytes)
    if is_anthropic_model(model) and ANTHROPIC_EFFORT is not None:
        h.update(f"|effort={ANTHROPIC_EFFORT}".encode())
    return CACHE_DIR / f"{h.hexdigest()}.json"


def _read_cache(path: Path) -> str | None:
    if path.exists():
        return json.loads(path.read_text())["output"]
    return None


def _write_cache(path: Path, output: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"output": output}))


def _record_usage(response, model: str, call_type: str) -> None:
    usage = getattr(response, "usage", None)
    if usage is not None:
        get_budget().record(usage.total_tokens)
        log_call(model, call_type, usage)


# Server-side refusal fallbacks: if the model declines a request on policy
# grounds, the API re-runs it on a fallback model inside the same call
# instead of handing back a refusal. Vibe descriptions of photos of people
# are the kind of thing that can trip a classifier, and one declined image
# shouldn't punch a hole in a pool. Flip to False if the beta isn't enabled
# on the account (every call would 400).
ANTHROPIC_REFUSAL_FALLBACKS = True


def _anthropic_messages(
    prompt: str, image_b64: str | None = None, mime_type: str | None = None
) -> list[dict]:
    """One user turn, image first (Anthropic's documented ordering) when there
    is one."""
    content: list[dict] = []
    if image_b64 is not None:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": image_b64,
                },
            }
        )
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _anthropic_kwargs(model: str, messages: list[dict]) -> dict:
    """Shared request shape. ``thinking`` is deliberately omitted: on Opus 5
    that means adaptive thinking (the default), while passing it explicitly
    would break models that don't accept adaptive."""
    kwargs = {
        "model": model,
        "max_tokens": ANTHROPIC_MAX_TOKENS,
        "messages": messages,
    }
    if ANTHROPIC_EFFORT is not None:
        kwargs["output_config"] = {"effort": ANTHROPIC_EFFORT}
    if ANTHROPIC_REFUSAL_FALLBACKS:
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
    return kwargs


def _anthropic_output_text(response) -> str:
    """Concatenated text blocks. Raises RefusalError rather than returning an
    empty string when the whole fallback chain declined - a silent "" would
    surface much later as an unparseable-JSON error."""
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        raise RefusalError(f"Claude declined the request (category={category!r})")
    return "".join(block.text for block in response.content if block.type == "text")


def call_text(
    prompt: str,
    model: str = DEFAULT_MODEL,
    use_cache: bool = True,
    call_type: str = "text",
    validate: Callable[[str], None] | None = None,
    max_retries: int = MAX_RETRIES,
) -> str:
    """``validate``, if given, is called on the raw output text; a ValueError
    it raises is retried under the same backoff budget as transient API
    errors (``max_retries`` total, shared - see ``_call_with_retry``). Only
    output that passes ``validate`` is written to the cache."""
    path = _cache_path(model, prompt, None)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached

    if is_anthropic_model(model):

        def attempt():
            response = get_anthropic_client().beta.messages.create(
                **_anthropic_kwargs(model, _anthropic_messages(prompt))
            )
            output = _anthropic_output_text(response)
            if validate is not None:
                validate(output)
            return output

    else:
        get_budget().check()

        def attempt():
            response = get_client().responses.create(
                model=model,
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            )
            _record_usage(response, model, call_type)  # spent tokens even if validate() rejects it
            output = response.output_text
            if validate is not None:
                validate(output)
            return output

    output = _call_with_retry(
        attempt, max_retries=max_retries, extra_retryable=(ValueError,) if validate else ()
    )

    if use_cache:
        _write_cache(path, output)
    return output


async def call_text_async(
    prompt: str,
    model: str = DEFAULT_MODEL,
    use_cache: bool = True,
    call_type: str = "text",
    validate: Callable[[str], None] | None = None,
    max_retries: int = MAX_RETRIES,
) -> str:
    path = _cache_path(model, prompt, None)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached

    if is_anthropic_model(model):

        async def attempt():
            response = await get_anthropic_async_client().beta.messages.create(
                **_anthropic_kwargs(model, _anthropic_messages(prompt))
            )
            output = _anthropic_output_text(response)
            if validate is not None:
                validate(output)
            return output

    else:
        get_budget().check()

        async def attempt():
            response = await get_async_client().responses.create(
                model=model,
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            )
            _record_usage(response, model, call_type)
            output = response.output_text
            if validate is not None:
                validate(output)
            return output

    output = await _call_with_retry_async(
        attempt, max_retries=max_retries, extra_retryable=(ValueError,) if validate else ()
    )

    if use_cache:
        _write_cache(path, output)
    return output


def call_with_image(
    prompt: str,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    model: str = DEFAULT_MODEL,
    use_cache: bool = True,
    call_type: str = "image",
    validate: Callable[[str], None] | None = None,
    max_retries: int = MAX_RETRIES,
) -> str:
    """``validate``, if given, is called on the raw output text; a ValueError
    it raises (e.g. the judge's JSON is missing a required field) is retried
    under the same backoff budget as transient API errors (``max_retries``
    total, shared across both failure kinds - see ``_call_with_retry``).
    Only output that passes ``validate`` is written to the cache."""
    path = _cache_path(model, prompt, image_bytes)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    if is_anthropic_model(model):

        def attempt():
            response = get_anthropic_client().beta.messages.create(
                **_anthropic_kwargs(
                    model, _anthropic_messages(prompt, image_b64, mime_type)
                )
            )
            output = _anthropic_output_text(response)
            if validate is not None:
                validate(output)
            return output

    else:
        get_budget().check()

        def attempt():
            response = get_client().responses.create(
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
            _record_usage(response, model, call_type)  # spent tokens even if validate() rejects it
            output = response.output_text
            if validate is not None:
                validate(output)
            return output

    output = _call_with_retry(
        attempt, max_retries=max_retries, extra_retryable=(ValueError,) if validate else ()
    )

    if use_cache:
        _write_cache(path, output)
    return output


async def call_with_image_async(
    prompt: str,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    model: str = DEFAULT_MODEL,
    use_cache: bool = True,
    call_type: str = "image",
    validate: Callable[[str], None] | None = None,
    max_retries: int = MAX_RETRIES,
) -> str:
    path = _cache_path(model, prompt, image_bytes)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    if is_anthropic_model(model):

        async def attempt():
            response = await get_anthropic_async_client().beta.messages.create(
                **_anthropic_kwargs(
                    model, _anthropic_messages(prompt, image_b64, mime_type)
                )
            )
            output = _anthropic_output_text(response)
            if validate is not None:
                validate(output)
            return output

    else:
        get_budget().check()

        async def attempt():
            response = await get_async_client().responses.create(
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
            _record_usage(response, model, call_type)
            output = response.output_text
            if validate is not None:
                validate(output)
            return output

    output = await _call_with_retry_async(
        attempt, max_retries=max_retries, extra_retryable=(ValueError,) if validate else ()
    )

    if use_cache:
        _write_cache(path, output)
    return output
