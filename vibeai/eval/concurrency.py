"""Bounded concurrency helper for running many async LLM calls at once."""

import asyncio
from collections.abc import Awaitable, Coroutine
from typing import TypeVar

T = TypeVar("T")


async def gather_bounded(
    coros: list[Coroutine[None, None, T]],
    limit: int = 10,
    return_exceptions: bool = False,
) -> list[T]:
    """Run coroutines concurrently, at most `limit` in flight at a time.

    With return_exceptions=True, a failed coroutine's exception is returned
    in its place (same order as `coros`) instead of aborting the whole batch -
    so results already computed for other items aren't lost when one fails.
    """
    semaphore = asyncio.Semaphore(limit)

    async def _run(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *(_run(c) for c in coros), return_exceptions=return_exceptions
    )
