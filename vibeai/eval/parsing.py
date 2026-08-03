"""Parsing helpers for LLM outputs that are supposed to be JSON."""

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str):
    """Pull a JSON value out of an LLM response, tolerating ```json fences."""
    text = text.strip()
    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)
