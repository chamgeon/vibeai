"""Per-call LLM usage log.

``vibeai.llm.budget`` only tracks a single running total for today (reset
each day, no history). This module appends one JSONL record per actual API
call (cache hits don't call the API, so they aren't logged), so usage can be
broken down by day / model / call type after the fact - e.g. to see which
pipeline step or prompt-iteration run is driving spend.
"""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

USAGE_LOG_PATH = Path(".cache/llm_usage/calls.jsonl")

_lock = threading.Lock()


def log_call(model: str, call_type: str, usage) -> None:
    """Append one record for a real (non-cached) API call.

    ``usage`` is the ``response.usage`` object from the OpenAI Responses API.
    """
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model,
        "call_type": call_type,
        "input_tokens": usage.input_tokens,
        "cached_tokens": usage.input_tokens_details.cached_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.output_tokens_details.reasoning_tokens,
        "total_tokens": usage.total_tokens,
    }
    with _lock:
        USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with USAGE_LOG_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")


def read_calls(path: Path = USAGE_LOG_PATH) -> list[dict]:
    if not path.exists():
        return []
    calls = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                calls.append(json.loads(line))
    return calls
