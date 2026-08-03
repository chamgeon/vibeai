"""Daily token-budget tracking against your OpenAI account's TPD limit.

Tracks actual usage (from response.usage), persisted per calendar day, so a
run stops making calls before it would blow through the daily cap - instead
of continuing to hammer the API and getting opaque 429s once the account
limit is hit.
"""

import json
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path

USAGE_PATH = Path(".cache/llm_usage/usage.json")
DEFAULT_DAILY_TOKEN_BUDGET = 3_000_000  # gpt-5 TPD limit as of this writing; override if yours differs


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class _Usage:
    date: str
    tokens_used: int


class TokenBudget:
    def __init__(self, daily_limit: int = DEFAULT_DAILY_TOKEN_BUDGET, path: Path = USAGE_PATH):
        self.daily_limit = daily_limit
        self._path = path
        self._lock = threading.Lock()
        self._usage = self._load()

    def _today(self) -> str:
        return date.today().isoformat()

    def _load(self) -> _Usage:
        if self._path.exists():
            data = json.loads(self._path.read_text())
            if data.get("date") == self._today():
                return _Usage(date=data["date"], tokens_used=data["tokens_used"])
        return _Usage(date=self._today(), tokens_used=0)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"date": self._usage.date, "tokens_used": self._usage.tokens_used})
        )

    def _roll_over_if_new_day(self) -> None:
        if self._usage.date != self._today():
            self._usage = _Usage(date=self._today(), tokens_used=0)

    def check(self) -> None:
        """Raise if today's usage has already reached the daily budget. Call before each API call."""
        with self._lock:
            self._roll_over_if_new_day()
            if self._usage.tokens_used >= self.daily_limit:
                raise BudgetExceededError(
                    f"Daily token budget exhausted: {self._usage.tokens_used}/{self.daily_limit} "
                    "tokens used today. Wait for the daily reset or raise TokenBudget's daily_limit."
                )

    def record(self, tokens: int) -> None:
        """Add actually-used tokens (from response.usage) to today's total."""
        with self._lock:
            self._roll_over_if_new_day()
            self._usage.tokens_used += tokens
            self._save()

    @property
    def used_today(self) -> int:
        return self._usage.tokens_used

    @property
    def remaining_today(self) -> int:
        return max(0, self.daily_limit - self._usage.tokens_used)


_default_budget: TokenBudget | None = None


def get_budget() -> TokenBudget:
    global _default_budget
    if _default_budget is None:
        _default_budget = TokenBudget()
    return _default_budget
