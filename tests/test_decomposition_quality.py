"""Evaluate the baseline representation + baseline decomposition prompts
using the decomposition-quality judge (Completeness / Claim Independence /
Atom Quality), across many images concurrently."""

from vibeai.eval.concurrency import gather_bounded
from vibeai.eval.dataset import load_image_paths
from vibeai.eval.results import result_log
from vibeai.metrics.decomposition_quality import DecompositionQualityMetric
from vibeai.pipeline.evaluate import evaluate_image

REPRESENTATION_PROMPT_VERSION = "baseline"
DECOMPOSITION_PROMPT_VERSION = "baseline"
CONCURRENCY = 30

IMAGES = load_image_paths(n=5, seed=0)


async def test_decomposition_quality_batch():
    metric = DecompositionQualityMetric()

    coros = [
        evaluate_image(
            image_path,
            metric,
            representation_prompt_version=REPRESENTATION_PROMPT_VERSION,
            decomposition_prompt_version=DECOMPOSITION_PROMPT_VERSION,
        )
        for image_path in IMAGES
    ]
    outcomes = await gather_bounded(coros, limit=CONCURRENCY, return_exceptions=True)

    failures = []
    for image_path, outcome in zip(IMAGES, outcomes):
        if isinstance(outcome, BaseException):
            failures.append(f"{image_path.name}: {type(outcome).__name__}: {outcome}")
            result_log.add(
                test_name="decomposition_quality",
                item=image_path.name,
                score=None,
                passed=False,
                details={
                    "error_type": type(outcome).__name__,
                    "error_message": str(outcome),
                },
            )
            continue

        test_case, result = outcome
        passed = metric.is_successful(result)
        result_log.add(
            test_name="decomposition_quality",
            item=test_case.image_path.name,
            score=result.score,
            passed=passed,
            details={
                "representation": test_case.representation,
                "atoms": test_case.atoms,
                **result.details,
            },
        )
        print(
            f"\n[{test_case.image_path.stem}] score={result.score:.2f} "
            f"atoms={len(test_case.atoms)}\n  {result.reason}"
        )
        if not passed:
            failures.append(f"{test_case.image_path.name}: score={result.score:.2f}")

    assert not failures, "Decomposition quality below threshold for:\n" + "\n".join(failures)
