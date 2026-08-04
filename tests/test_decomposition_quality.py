"""Evaluate the baseline representation + baseline decomposition prompts
using the decomposition-quality judge (Completeness / Claim Independence /
Atom Quality), across many images concurrently."""

from vibeai.eval.concurrency import gather_bounded
from vibeai.eval.dataset import load_image_paths
from vibeai.eval.prompt_results import ImageError, ImageResult, aggregate_prompt_results
from vibeai.metrics.decomposition_quality import DecompositionQualityMetric
from vibeai.pipeline.evaluate import evaluate_image

REPRESENTATION_PROMPT_VERSION = "baseline"
DECOMPOSITION_PROMPT_VERSION = "baseline"
CONCURRENCY = 30


async def test_decomposition_quality_batch(n_images):
    IMAGES = load_image_paths(n=n_images, seed=0)
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
    image_results = []
    image_errors = []
    for image_path, outcome in zip(IMAGES, outcomes):
        if isinstance(outcome, BaseException):
            failures.append(f"{image_path.name}: {type(outcome).__name__}: {outcome}")
            image_errors.append(
                ImageError(
                    image_path=image_path.name,
                    error_type=type(outcome).__name__,
                    error_message=str(outcome),
                )
            )
            continue

        test_case, result = outcome
        passed = metric.is_successful(result)
        image_results.append(
            ImageResult(
                image_path=test_case.image_path,
                result=result,
                details={
                    "representation": test_case.representation,
                    "atoms": test_case.atoms,
                    **result.details,
                },
            )
        )
        print(
            f"\n[{test_case.image_path.stem}] score={result.score:.2f} "
            f"atoms={len(test_case.atoms)}\n  {result.reason}"
        )
        if not passed:
            failures.append(f"{test_case.image_path.name}: score={result.score:.2f}")

    if image_results or image_errors:
        _, summary_path, per_image_path = aggregate_prompt_results(
            metric,
            image_results,
            representation_prompt_version=REPRESENTATION_PROMPT_VERSION,
            decomposition_prompt_version=DECOMPOSITION_PROMPT_VERSION,
            model=metric.model,
            errors=image_errors,
        )
        print(
            f"\nSaved prompt-level summary to {summary_path}\n"
            f"Saved per-image detail to {per_image_path}"
        )

    assert not failures, "Decomposition quality below threshold for:\n" + "\n".join(failures)
