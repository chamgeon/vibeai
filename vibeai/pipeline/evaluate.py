"""End-to-end async pipeline: image -> representation -> decomposition -> judged score.

Exists so tests (and anything else that needs to score many images) share one
implementation of the represent -> decompose -> judge chain instead of each
re-assembling it.
"""

from pathlib import Path

from vibeai.eval.test_cases import DecompositionTestCase
from vibeai.llm.client import DEFAULT_MODEL
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
    representation_model: str | None = None,
    decomposition_model: str | None = None,
) -> tuple[DecompositionTestCase, MetricResult]:
    """``representation_model``/``decomposition_model`` default to the
    pipeline's DEFAULT_MODEL; the judge's model is set on ``metric`` itself."""
    representation = await generate_representation_async(
        image_path,
        prompt_version=representation_prompt_version,
        model=representation_model or DEFAULT_MODEL,
    )
    atom_details = None
    if decomposition_prompt_version == DIRECT_DECOMPOSITION:
        atoms = decompose_direct(representation)
    else:
        atoms, atom_details = await decompose_async(
            representation,
            prompt_version=decomposition_prompt_version,
            model=decomposition_model or DEFAULT_MODEL,
        )

    test_case = DecompositionTestCase(
        image_path=image_path,
        representation=representation,
        atoms=atoms,
        representation_prompt_version=representation_prompt_version,
        decomposition_prompt_version=decomposition_prompt_version,
        atom_details=atom_details,
    )
    result = await metric.measure_async(test_case)
    return test_case, result
