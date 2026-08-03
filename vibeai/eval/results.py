"""Lightweight run-level results logger, so scores are comparable across
prompt versions and test runs (not just visible as pytest pass/fail)."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results")


@dataclass
class ResultLog:
    records: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self, *, test_name: str, item: str, score: float | None, passed: bool, details: dict
    ) -> None:
        """score is None when the item never produced a result (e.g. it raised
        an exception) - distinguishes "ran and scored 0" from "never finished"."""
        self.records.append(
            {
                "test_name": test_name,
                "item": item,
                "score": score,
                "passed": passed,
                "details": details,
            }
        )

    def save(self, run_name: str | None = None) -> Path:
        RESULTS_DIR.mkdir(exist_ok=True)
        run_name = run_name or f"run_{int(time.time())}"
        path = RESULTS_DIR / f"{run_name}.json"
        path.write_text(json.dumps(self.records, indent=2))
        return path


result_log = ResultLog()
