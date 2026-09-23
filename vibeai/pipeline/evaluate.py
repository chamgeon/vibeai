"""End-to-end async pipeline: image -> representation -> decomposition -> judged score.

Exists so tests (and anything else that needs to score many images) share one
implementation of the represent -> decompose -> judge chain instead of each
re-assembling it.
"""

from pathlib import Path

from vibeai.eval.test_cases import DecompositionTestCase
from vibeai.llm.client import DEFAULT_DECOMPOSITION_MODEL, DEFAULT_MODEL
from vibeai.metrics.base import Metric, MetricResult
from vibeai.pipeline.decompose import decompose_async, decompose_direct
from vibeai.pipeline.represent import generate_representation_async

# Sentinel decomposition_prompt_version for representation prompts (e.g. "v2")
# that already emit a decomposed vibe representation - skips the separate
# decompose LLM call and extracts atoms straight from the representation.
DIRECT_DECOMPOSITION = "direct"


def effective_models(
    *,
    decomposition_prompt_version: str,
    representation_model: str | None,
    decomposition_model: str | None,
    representation_supplied: bool = False,
) -> tuple[str | None, str | None]:
    """The models an ``evaluate_image`` call will actually use, for the run
    record: the caller's overrides resolved against the pipeline defaults,
    and None for a step that makes no LLM call at all - a representation
    that was supplied rather than generated, or DIRECT_DECOMPOSITION, which
    reads atoms straight out of the representation.
    """
    return (
        None if representation_supplied else (representation_model or DEFAULT_MODEL),
        None
        if decomposition_prompt_version == DIRECT_DECOMPOSITION
        else (decomposition_model or DEFAULT_DECOMPOSITION_MODEL),
    )


async def evaluate_image(
    image_path: Path,
    metric: Metric,
    representation_prompt_version: str = "baseline",
    decomposition_prompt_version: str = "baseline",
    representation_model: str | None = None,
    decomposition_model: str | None = None,
    representation: str | None = None,
) -> tuple[DecompositionTestCase, MetricResult]:
    """``representation_model`` defaults to DEFAULT_MODEL and
    ``decomposition_model`` to DEFAULT_DECOMPOSITION_MODEL; the judge's model is
    set on ``metric`` itself.

    ``representation``, if given, is scored as-is and the generation call is
    skipped - for representations that did not come from a single
    ``prompts/representation.py`` call, such as the generator-evaluator
    workflow's output. ``representation_prompt_version`` is then only a label
    for the run, and ``representation_model`` is unused.
    """
    if representation is None:
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
            model=decomposition_model or DEFAULT_DECOMPOSITION_MODEL,
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
