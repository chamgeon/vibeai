"""End-to-end async pipeline: image -> representation -> decomposition -> judged score.

Exists so tests (and anything else that needs to score many images) share one
implementation of the represent -> decompose -> judge chain instead of each
re-assembling it.
"""

from pathlib import Path

from vibeai.eval.test_cases import DecompositionTestCase
from vibeai.metrics.base import Metric, MetricResult
from vibeai.pipeline.decompose import decompose_async
from vibeai.pipeline.represent import generate_representation_async


async def evaluate_image(
    image_path: Path,
    metric: Metric,
    representation_prompt_version: str = "baseline",
    decomposition_prompt_version: str = "baseline",
) -> tuple[DecompositionTestCase, MetricResult]:
    representation = await generate_representation_async(
        image_path, prompt_version=representation_prompt_version
    )
    atoms = await decompose_async(representation, prompt_version=decomposition_prompt_version)

    test_case = DecompositionTestCase(
        image_path=image_path,
        representation=representation,
        atoms=atoms,
        representation_prompt_version=representation_prompt_version,
        decomposition_prompt_version=decomposition_prompt_version,
    )
    result = await metric.measure_async(test_case)
    return test_case, result
