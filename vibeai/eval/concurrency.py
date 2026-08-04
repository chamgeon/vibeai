"""Bounded concurrency helper for running many async LLM calls at once."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Coroutine
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


async def gather_bounded_as_completed(
    coros: list[Coroutine[None, None, T]],
    limit: int = 10,
) -> AsyncIterator[tuple[int, T | BaseException]]:
    """Like `gather_bounded`, but yields (original_index, outcome) pairs as
    each coroutine finishes, instead of waiting for the whole batch.

    Exceptions are caught and yielded in place of a result (mirroring
    `gather_bounded(..., return_exceptions=True)`) so one failure doesn't
    stop the rest of the batch or get raised out of the loop.
    """
    semaphore = asyncio.Semaphore(limit)

    async def _run(index: int, coro: Awaitable[T]) -> tuple[int, T | BaseException]:
        async with semaphore:
            try:
                return index, await coro
            except BaseException as exc:
                return index, exc

    for task in asyncio.as_completed([_run(i, c) for i, c in enumerate(coros)]):
        yield await task
