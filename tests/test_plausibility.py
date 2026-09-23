"""Evaluate the baseline representation + baseline decomposition prompts
using the plausibility judge (per-atom evidence/vibe-inference checks),
across many images concurrently."""

from vibeai.eval.concurrency import gather_bounded_as_completed
from vibeai.eval.dataset import load_image_paths
from vibeai.eval.prompt_results import ImageError, ImageResult, aggregate_prompt_results
from vibeai.metrics.plausibility import PlausibilityMetric
from vibeai.pipeline.evaluate import effective_models, evaluate_image
from vibeai.pipeline.workflow_batch import resolve_representations

async def test_plausibility_batch(
    n_images, image_dir, representation_prompt_version, decomposition_prompt_version, concurrency,
    eval_model, representation_model, decomposition_model,
):
    IMAGES = load_image_paths(n=n_images, seed=0, data_dir=image_dir)
    IMAGES, representations = await resolve_representations(
        IMAGES, representation_prompt_version, concurrency=concurrency
    )
    # What the run record should say produced the representation and the
    # atoms - not just which model judged them.
    run_representation_model, run_decomposition_model = effective_models(
        decomposition_prompt_version=decomposition_prompt_version,
        representation_model=representation_model,
        decomposition_model=decomposition_model,
        representation_supplied=representations is not None,
    )
    metric = PlausibilityMetric() if eval_model is None else PlausibilityMetric(model=eval_model)

    coros = [
        evaluate_image(
            image_path,
            metric,
            representation_prompt_version=representation_prompt_version,
            decomposition_prompt_version=decomposition_prompt_version,
            representation_model=representation_model,
            decomposition_model=decomposition_model,
            representation=representations.get(str(image_path)) if representations else None,
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
                    # Only present for structured decomposition prompts; kept
                    # out of the record entirely otherwise, so v1/baseline
                    # runs keep their existing shape.
                    **({"atom_details": test_case.atom_details} if test_case.atom_details else {}),
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
            representation_model=run_representation_model,
            decomposition_model=run_decomposition_model,
            errors=image_errors,
        )
        print(
            f"\nSaved prompt-level summary to {summary_path}\n"
            f"Saved per-image detail to {per_image_path}"
        )

    assert not failures, "Plausibility below threshold for:\n" + "\n".join(failures)
