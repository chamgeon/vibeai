"""Summarize the per-call LLM usage log (``vibeai.llm.usage_log``).

Breaks down token usage (and estimated cost) by day, model, and call type
(represent / decompose / judge / etc.) so it's possible to see which
pipeline step or run is driving spend, rather than only the single running
daily total kept by ``vibeai.llm.budget``.

Usage:
    python -m vibeai.llm.usage_report
    python -m vibeai.llm.usage_report --since 2026-08-01
"""

import argparse
from collections import defaultdict
from datetime import date

from vibeai.llm.usage_log import USAGE_LOG_PATH, read_calls

# $ per 1M tokens, standard (non-batch/flex) API pricing. Cached input tokens
# are a subset of input_tokens, billed at the cheaper cached rate instead of
# the regular input rate. Source: https://developers.openai.com/api/docs/pricing
# Update this table (and add rows for other models you use) as pricing changes.
PRICING_PER_MILLION = {
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
}


def _cost(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float | None:
    rates = PRICING_PER_MILLION.get(model)
    if rates is None:
        return None
    uncached = input_tokens - cached_tokens
    return (
        uncached * rates["input"]
        + cached_tokens * rates["cached_input"]
        + output_tokens * rates["output"]
    ) / 1_000_000


def _bucket_totals(calls: list[dict], key_fn) -> dict:
    totals = defaultdict(
        lambda: {
            "calls": 0,
            "total_tokens": 0,
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "cost_known": True,
        }
    )
    for call in calls:
        bucket = totals[key_fn(call)]
        bucket["calls"] += 1
        bucket["total_tokens"] += call["total_tokens"]
        bucket["input_tokens"] += call["input_tokens"]
        bucket["cached_tokens"] += call["cached_tokens"]
        bucket["output_tokens"] += call["output_tokens"]
        cost = _cost(call["model"], call["input_tokens"], call["cached_tokens"], call["output_tokens"])
        if cost is None:
            bucket["cost_known"] = False
        else:
            bucket["cost"] += cost
    return totals


def _fmt_cost(bucket: dict) -> str:
    if not bucket["cost_known"]:
        return "?"
    return f"${bucket['cost']:.2f}"


def _print_table(title: str, totals: dict) -> None:
    print(f"\n{title}")
    print(f"{'':20s}{'calls':>8s}{'input':>12s}{'output':>12s}{'total':>12s}{'cost':>10s}")
    for key in sorted(totals):
        t = totals[key]
        print(
            f"{str(key):20s}{t['calls']:>8d}{t['input_tokens']:>12d}"
            f"{t['output_tokens']:>12d}{t['total_tokens']:>12d}{_fmt_cost(t):>10s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", type=date.fromisoformat, default=None, help="YYYY-MM-DD, inclusive")
    args = parser.parse_args()

    calls = read_calls()
    if args.since is not None:
        calls = [c for c in calls if date.fromisoformat(c["timestamp"][:10]) >= args.since]

    if not calls:
        print(f"No usage recorded in {USAGE_LOG_PATH} (yet).")
        return

    total_tokens = sum(c["total_tokens"] for c in calls)
    total_cost = sum(
        _cost(c["model"], c["input_tokens"], c["cached_tokens"], c["output_tokens"]) or 0.0 for c in calls
    )
    unpriced_models = {c["model"] for c in calls if c["model"] not in PRICING_PER_MILLION}

    print(f"{len(calls)} calls, {total_tokens} total tokens logged in {USAGE_LOG_PATH}")
    print(f"Estimated cost: ${total_cost:.2f}" + (" (partial - see unpriced models below)" if unpriced_models else ""))
    if unpriced_models:
        print(f"No pricing entry for: {', '.join(sorted(unpriced_models))} — add to PRICING_PER_MILLION in this file.")

    _print_table("By day", _bucket_totals(calls, lambda c: c["timestamp"][:10]))
    _print_table("By model", _bucket_totals(calls, lambda c: c["model"]))
    _print_table("By call type", _bucket_totals(calls, lambda c: c["call_type"]))


if __name__ == "__main__":
    main()
