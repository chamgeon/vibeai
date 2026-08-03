"""Thin wrapper around the OpenAI Responses API, with disk caching.

Caching matters here because prompt testing means re-running the same
inputs repeatedly while iterating on prompts/metrics - we don't want to
re-pay for (or wait on) an unchanged call.
"""

import base64
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-5"
CACHE_DIR = Path(".cache/llm")


@lru_cache
def get_client() -> OpenAI:
    return OpenAI()


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


def call_text(prompt: str, model: str = DEFAULT_MODEL, use_cache: bool = True) -> str:
    path = _cache_path(model, prompt, None)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached

    response = get_client().responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
    )
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

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
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
    output = response.output_text

    if use_cache:
        _write_cache(path, output)
    return output
