"""Evaluate the baseline representation + baseline decomposition prompts
using the decomposition-quality judge (Completeness / Claim Independence /
Atom Quality)."""

import pytest

from vibeai.eval.dataset import load_image_paths
from vibeai.eval.results import result_log
from vibeai.eval.test_cases import DecompositionTestCase
from vibeai.metrics.decomposition_quality import DecompositionQualityMetric
from vibeai.pipeline.decompose import decompose
from vibeai.pipeline.represent import generate_representation

REPRESENTATION_PROMPT_VERSION = "baseline"
DECOMPOSITION_PROMPT_VERSION = "baseline"

IMAGES = load_image_paths(n=5, seed=0)


@pytest.fixture(scope="module")
def metric():
    return DecompositionQualityMetric()


@pytest.mark.parametrize("image_path", IMAGES, ids=lambda p: p.stem)
def test_decomposition_quality(image_path, metric):
    representation = generate_representation(
        image_path, prompt_version=REPRESENTATION_PROMPT_VERSION
    )
    atoms = decompose(representation, prompt_version=DECOMPOSITION_PROMPT_VERSION)

    test_case = DecompositionTestCase(
        image_path=image_path,
        representation=representation,
        atoms=atoms,
        representation_prompt_version=REPRESENTATION_PROMPT_VERSION,
        decomposition_prompt_version=DECOMPOSITION_PROMPT_VERSION,
    )
    result = metric.measure(test_case)
    passed = metric.is_successful(result)

    result_log.add(
        test_name="decomposition_quality",
        item=image_path.name,
        score=result.score,
        passed=passed,
        details={
            "representation": representation,
            "atoms": atoms,
            **result.details,
        },
    )

    print(f"\n[{image_path.stem}] score={result.score:.2f} atoms={len(atoms)}\n  {result.reason}")
    assert passed, (
        f"Decomposition quality below threshold for {image_path.name}: "
        f"score={result.score:.2f}\n{result.reason}"
    )
