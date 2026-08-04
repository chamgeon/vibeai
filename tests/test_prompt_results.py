import json
from pathlib import Path

from vibeai.eval.prompt_results import ImageError, ImageResult, aggregate_prompt_results
from vibeai.metrics.base import Metric, MetricResult


class _StubMetric(Metric):
    name = "stub_metric"
    threshold = 0.7

    def measure(self, test_case) -> MetricResult:
        raise NotImplementedError

    def extract_submetrics(self, result: MetricResult) -> dict[str, float]:
        return {"sub_a": result.details["sub_a"] / 5}


def test_aggregate_prompt_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    items = [
        ImageResult(
            image_path=Path("img_1.jpg"),
            result=MetricResult(score=0.9, reason="good", details={"sub_a": 5}),
            details={"representation": "a photo of a dog"},
        ),
        ImageResult(
            image_path=Path("img_2.jpg"),
            result=MetricResult(score=0.4, reason="missed atoms", details={"sub_a": 2}),
            details={"representation": "a photo of a cat"},
        ),
    ]

    errors = [ImageError(image_path="img_3.jpg", error_type="TimeoutError", error_message="timed out")]

    result, summary_path, per_image_path = aggregate_prompt_results(
        _StubMetric(),
        items,
        representation_prompt_version="baseline",
        decomposition_prompt_version="v2",
        model="test-model",
        errors=errors,
        run_name="fixed_run",
    )

    assert result.n == 2
    assert result.n_errors == 1
    assert result.mean_score == 0.65
    assert result.pass_rate == 0.5
    assert result.submetric_means == {"sub_a": 0.7}
    assert len(result.failures) == 1
    assert result.failures[0].image_path == "img_2.jpg"
    assert len(result.errors) == 1
    assert result.errors[0].image_path == "img_3.jpg"
    assert result.errors[0].error_type == "TimeoutError"

    assert summary_path == Path("results/stub_metric/fixed_run.json")
    assert per_image_path == Path("results/stub_metric/fixed_run.per_image.jsonl")

    saved_summary = json.loads(summary_path.read_text())
    assert saved_summary["mean_score"] == 0.65

    per_image_lines = per_image_path.read_text().splitlines()
    assert len(per_image_lines) == 2
    assert json.loads(per_image_lines[0]) == {
        "image_path": "img_1.jpg",
        "score": 0.9,
        "passed": True,
        "details": {"representation": "a photo of a dog"},
    }


def test_aggregate_prompt_results_all_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    errors = [ImageError(image_path="img_1.jpg", error_type="TimeoutError", error_message="timed out")]

    result, _, per_image_path = aggregate_prompt_results(
        _StubMetric(),
        [],
        representation_prompt_version="baseline",
        decomposition_prompt_version="v2",
        model="test-model",
        errors=errors,
        run_name="all_errors_run",
    )

    assert result.n == 0
    assert result.n_errors == 1
    assert result.mean_score == 0.0
    assert result.pass_rate == 0.0
    assert per_image_path.read_text() == ""
