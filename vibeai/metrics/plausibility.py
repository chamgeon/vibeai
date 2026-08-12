"""Plausibility metric: scores whether each vibe atom is a plausible claim
about the image (checking stated evidence and vibe-inference separately),
via LLM-as-a-judge."""

from vibeai.eval.parsing import extract_json
from vibeai.eval.test_cases import DecompositionTestCase
from vibeai.llm.client import DEFAULT_MODEL, call_with_image, call_with_image_async
from vibeai.metrics.base import Metric, MetricResult
from vibeai.pipeline.represent import MIME_TYPES
from vibeai.prompts.plausibility_eval import PLAUSIBILITY_EVAL_PROMPT

# A representation passes when the proportion of plausible atoms
# (final_verdict true) clears this threshold - see the base Metric's
# is_successful. The evidence-backed atom share is tracked as a submetric
# only, and doesn't gate pass/fail.
PLAUSIBLE_RATE_THRESHOLD = 0.7


def _build_prompt(test_case: DecompositionTestCase) -> str:
    atom_list = "\n".join(f"{i + 1}. {atom}" for i, atom in enumerate(test_case.atoms))
    return PLAUSIBILITY_EVAL_PROMPT.format(atom_list=atom_list)


def _load_image(test_case: DecompositionTestCase) -> tuple[bytes, str]:
    mime_type = MIME_TYPES.get(test_case.image_path.suffix.lower(), "image/jpeg")
    return test_case.image_path.read_bytes(), mime_type


_REQUIRED_ATOM_KEYS = {
    "atom",
    "type",
    "stated_evidence",
    "stated_vibe",
    "evidence_presence_check",
    "direct_check",
    "mapping_check",
    "final_verdict",
}


def _validate_atom(atom: dict, index: int) -> None:
    missing = _REQUIRED_ATOM_KEYS - atom.keys()
    if missing:
        raise ValueError(f"Judge atom {index} missing key(s) {sorted(missing)}: {atom!r}")
    if not isinstance(atom["stated_vibe"], str) or not atom["stated_vibe"].strip():
        raise ValueError(f"Judge atom {index} has empty/invalid stated_vibe: {atom!r}")


def _extract_and_validate_atoms(raw: str) -> list[dict]:
    """Parse + validate the judge's JSON. Raises ValueError on any failure
    (malformed JSON, missing required key, empty stated_vibe, ...) - used
    both as call_with_image's retry-triggering ``validate`` callback and to
    build the final MetricResult once a call has passed validation."""
    atoms = extract_json(raw)
    if not atoms:
        raise ValueError(f"Judge returned no atom verdicts: {raw!r}")
    for i, atom in enumerate(atoms):
        _validate_atom(atom, i)
    return atoms


def _parse_result(raw: str) -> MetricResult:
    atoms = _extract_and_validate_atoms(raw)
    plausible = sum(1 for atom in atoms if atom["final_verdict"])
    evidence_backed = sum(1 for atom in atoms if atom["type"] == "evidence_backed")
    total = len(atoms)

    reason = (
        f"{plausible}/{total} atoms plausible, "
        f"{evidence_backed}/{total} atoms evidence-backed"
    )
    return MetricResult(score=plausible / total, reason=reason, details={"atoms": atoms})


class PlausibilityMetric(Metric):
    name = "plausibility"
    threshold = PLAUSIBLE_RATE_THRESHOLD

    def __init__(self, model: str = DEFAULT_MODEL, threshold: float | None = None):
        self.model = model
        if threshold is not None:
            self.threshold = threshold

    def measure(self, test_case: DecompositionTestCase) -> MetricResult:
        image_bytes, mime_type = _load_image(test_case)
        raw = call_with_image(
            _build_prompt(test_case),
            image_bytes,
            mime_type=mime_type,
            model=self.model,
            call_type="judge",
            validate=_extract_and_validate_atoms,
        )
        return _parse_result(raw)

    async def measure_async(self, test_case: DecompositionTestCase) -> MetricResult:
        image_bytes, mime_type = _load_image(test_case)
        raw = await call_with_image_async(
            _build_prompt(test_case),
            image_bytes,
            mime_type=mime_type,
            model=self.model,
            call_type="judge",
            validate=_extract_and_validate_atoms,
        )
        return _parse_result(raw)

    def extract_submetrics(self, result: MetricResult) -> dict[str, float]:
        atoms = result.details["atoms"]
        submetrics: dict[str, float] = {}
        for atom_type in ("vibe_only", "evidence_backed"):
            typed = [a for a in atoms if a["type"] == atom_type]
            if typed:
                submetrics[f"{atom_type}_rate"] = sum(
                    1 for a in typed if a["final_verdict"]
                ) / len(typed)
        submetrics["evidence_backed_share"] = sum(
            1 for a in atoms if a["type"] == "evidence_backed"
        ) / len(atoms)
        return submetrics
