"""Prompt-level aggregation: rolls up many per-image MetricResults (for one
prompt version) into a single trackable summary, so prompt iterations are
comparable the same way run-level ResultLog makes test runs comparable.

Output is split in two, per run:
  results/{metric_name}/{run_name}.json        - PromptEvalResult summary
  results/{metric_name}/{run_name}.per_image.jsonl - one ImageScore per line

The summary stays small (safe to diff between prompt versions) even as
per-image detail grows with dataset size.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import pstdev
from typing import Any

from vibeai.metrics.base import Metric, MetricResult

RESULTS_DIR = Path("results")


@dataclass
class ImageResult:
    """One image's outcome, to be folded into a prompt-level run.

    `details` is caller-supplied context (e.g. representation, atoms, the
    judge's full verdict) - the aggregator doesn't need to know its shape,
    it just carries it through to the per-image JSONL record.
    """

    image_path: Path
    result: MetricResult
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageScore:
    image_path: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageFailure:
    image_path: str
    score: float
    reason: str


@dataclass
class ImageError:
    """An image that never produced a MetricResult (e.g. the pipeline
    raised) - distinguishes "ran and scored low" from "never finished"."""

    image_path: str
    error_type: str
    error_message: str


@dataclass
class PromptEvalResult:
    metric_name: str
    representation_prompt_version: str
    decomposition_prompt_version: str
    model: str
    timestamp: str

    n: int  # images that produced a score
    n_errors: int  # images that raised instead of scoring
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    pass_rate: float
    threshold: float

    submetric_means: dict[str, float] = field(default_factory=dict)
    failures: list[ImageFailure] = field(default_factory=list)
    errors: list[ImageError] = field(default_factory=list)


def aggregate_prompt_results(
    metric: Metric,
    items: list[ImageResult],
    *,
    representation_prompt_version: str,
    decomposition_prompt_version: str,
    model: str,
    errors: list[ImageError] | None = None,
    run_name: str | None = None,
) -> tuple[PromptEvalResult, Path, Path]:
    """Aggregate one prompt version's ImageResults into a PromptEvalResult,
    and save it (summary JSON + per_image JSONL) under results/{metric.name}/.

    This is the single per-image artifact for a prompt-version run - it
    carries whatever debug detail (representation, atoms, judge verdict)
    the caller attaches via ImageResult.details. `errors` covers images
    that never produced a MetricResult at all (only `items` needs to be
    non-empty; an all-errors run with no successful items still saves,
    since the errors themselves are the signal worth tracking).

    Returns (result, summary_path, per_image_path).
    """
    errors = errors or []
    if not items and not errors:
        raise ValueError("aggregate_prompt_results requires at least one item or error")

    scores = [item.result.score for item in items]
    passed_flags = [metric.is_successful(item.result) for item in items]
    has_scores = bool(items)

    submetric_values: dict[str, list[float]] = {}
    for item in items:
        for key, value in metric.extract_submetrics(item.result).items():
            submetric_values.setdefault(key, []).append(value)
    submetric_means = {key: sum(vals) / len(vals) for key, vals in submetric_values.items()}

    failures = [
        ImageFailure(
            image_path=str(item.image_path), score=item.result.score, reason=item.result.reason
        )
        for item in items
        if not metric.is_successful(item.result)
    ]

    per_image = [
        ImageScore(
            image_path=str(item.image_path),
            score=item.result.score,
            passed=passed,
            details=item.details,
        )
        for item, passed in zip(items, passed_flags)
    ]

    prompt_result = PromptEvalResult(
        metric_name=metric.name,
        representation_prompt_version=representation_prompt_version,
        decomposition_prompt_version=decomposition_prompt_version,
        model=model,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        n=len(items),
        n_errors=len(errors),
        mean_score=sum(scores) / len(scores) if has_scores else 0.0,
        std_score=pstdev(scores) if has_scores else 0.0,
        min_score=min(scores) if has_scores else 0.0,
        max_score=max(scores) if has_scores else 0.0,
        pass_rate=sum(passed_flags) / len(passed_flags) if has_scores else 0.0,
        threshold=metric.threshold,
        submetric_means=submetric_means,
        failures=failures,
        errors=errors,
    )

    metric_dir = RESULTS_DIR / metric.name
    metric_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_name or (
        f"{representation_prompt_version}__{decomposition_prompt_version}_{int(time.time())}"
    )

    summary_path = metric_dir / f"{run_name}.json"
    summary_path.write_text(json.dumps(asdict(prompt_result), indent=2))

    per_image_path = metric_dir / f"{run_name}.per_image.jsonl"
    with per_image_path.open("w") as f:
        for image_score in per_image:
            f.write(json.dumps(asdict(image_score)) + "\n")

    return prompt_result, summary_path, per_image_path
