"""End-to-end async pipeline: image -> representation -> decomposition -> judged score.

Exists so tests (and anything else that needs to score many images) share one
implementation of the represent -> decompose -> judge chain instead of each
re-assembling it.
"""

from pathlib import Path

from vibeai.eval.test_cases import DecompositionTestCase
from vibeai.metrics.base import Metric, MetricResult
from vibeai.pipeline.decompose import decompose_async, decompose_direct
from vibeai.pipeline.represent import generate_representation_async

# Sentinel decomposition_prompt_version for representation prompts (e.g. "v2")
# that already emit a decomposed vibe representation - skips the separate
# decompose LLM call and extracts atoms straight from the representation.
DIRECT_DECOMPOSITION = "direct"


async def evaluate_image(
    image_path: Path,
    metric: Metric,
    representation_prompt_version: str = "baseline",
    decomposition_prompt_version: str = "baseline",
) -> tuple[DecompositionTestCase, MetricResult]:
    representation = await generate_representation_async(
        image_path, prompt_version=representation_prompt_version
    )
    if decomposition_prompt_version == DIRECT_DECOMPOSITION:
        atoms = decompose_direct(representation)
    else:
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
