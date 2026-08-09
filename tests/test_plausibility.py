"""Evaluate the baseline representation + baseline decomposition prompts
using the plausibility judge (per-atom evidence/vibe-inference checks),
across many images concurrently."""

from vibeai.eval.concurrency import gather_bounded_as_completed
from vibeai.eval.dataset import load_image_paths
from vibeai.eval.prompt_results import ImageError, ImageResult, aggregate_prompt_results
from vibeai.metrics.plausibility import PlausibilityMetric
from vibeai.pipeline.evaluate import evaluate_image

async def test_plausibility_batch(
    n_images, image_dir, representation_prompt_version, decomposition_prompt_version, concurrency
):
    IMAGES = load_image_paths(n=n_images, seed=0, data_dir=image_dir)
    metric = PlausibilityMetric()

    coros = [
        evaluate_image(
            image_path,
            metric,
            representation_prompt_version=representation_prompt_version,
            decomposition_prompt_version=decomposition_prompt_version,
        )
        for image_path in IMAGES
    ]
    failures = []
    image_results = []
    image_errors = []
    async for index, outcome in gather_bounded_as_completed(coros, limit=concurrency):
        image_path = IMAGES[index]
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
        print(f"{test_case.image_path.name}: {result.score:.2f}")
        if not passed:
            failures.append(f"{test_case.image_path.name}: score={result.score:.2f}")

    if image_results or image_errors:
        _, summary_path, per_image_path = aggregate_prompt_results(
            metric,
            image_results,
            representation_prompt_version=representation_prompt_version,
            decomposition_prompt_version=decomposition_prompt_version,
            model=metric.model,
            errors=image_errors,
        )
        print(
            f"\nSaved prompt-level summary to {summary_path}\n"
            f"Saved per-image detail to {per_image_path}"
        )

    assert not failures, "Plausibility below threshold for:\n" + "\n".join(failures)
