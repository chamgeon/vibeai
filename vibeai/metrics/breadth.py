"""Breadth metric: of the vibe aspects this image supports, how many did the
representation touch at all.

The pool-free counterpart to ``metrics/richness.py``. Richness measures
coverage of a reference vibe pool; breadth measures coverage of the ten vibe
aspects the judge finds in the image itself (see ``prompts/breadth_eval.py``
for why the workflow's richness gate cannot use the pool).

    score = applicable aspects covered / applicable aspects

Aspects the judge rules ``not_applicable`` leave the denominator, so an image
with no people is not penalised for saying nothing about the relation between
figures.

The metric doubles as the workflow's richness gate: ``gaps`` in the result
details is the feedback handed back to the generator, and the gate rejects on
a non-empty list, which is stricter than the 0.7 threshold used for batch
scoring.
"""

from vibeai.eval.parsing import extract_json
from vibeai.eval.test_cases import DecompositionTestCase
from vibeai.llm.client import DEFAULT_EVAL_MODEL, call_with_image, call_with_image_async
from vibeai.metrics.base import Metric, MetricResult
from vibeai.pipeline.represent import MIME_TYPES
from vibeai.prompts.breadth_eval import BREADTH_EVAL_PROMPT

# The ten aspects of prompts/breadth_eval.py, in the order the prompt lists
# them. The judge is required to return one verdict per aspect in this order,
# so the list is also the validation schema.
DIMENSIONS = (
    "subject",
    "interpersonal",
    "object",
    "environment",
    "composition",
    "colour and light",
    "texture and material",
    "style",
    "activity",
    "time",
)

# A representation passes when it touches this share of the aspects the image
# supports. Only used when breadth is scored as a batch metric - the workflow
# gate rejects on any gap at all.
BREADTH_THRESHOLD = 0.7

_VERDICTS = {"covered", "gap", "not_applicable"}


def _build_prompt(test_case: DecompositionTestCase) -> str:
    atom_list = "\n".join(f"{i + 1}. {atom}" for i, atom in enumerate(test_case.atoms))
    return BREADTH_EVAL_PROMPT.format(atom_list=atom_list)


def _load_image(test_case: DecompositionTestCase) -> tuple[bytes, str]:
    mime_type = MIME_TYPES.get(test_case.image_path.suffix.lower(), "image/jpeg")
    return test_case.image_path.read_bytes(), mime_type


def _validate_dimensions(raw: str) -> list[dict]:
    """Parse + validate the judge's JSON. Raises ValueError on any failure -
    used both as the retry-triggering ``validate`` callback and to build the
    MetricResult once a call has passed.

    The aspects are checked one-for-one and in order: the score is a fraction
    over the applicable aspects, so a judge that skipped or reordered one
    would quietly change what the denominator means.
    """
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got: {raw!r}")

    dimensions = parsed.get("dimensions")
    if not isinstance(dimensions, list):
        raise ValueError(f"'dimensions' missing or not a list: {raw!r}")
    if len(dimensions) != len(DIMENSIONS):
        raise ValueError(
            f"Expected {len(DIMENSIONS)} dimension(s), got {len(dimensions)}: {raw!r}"
        )

    for i, (entry, expected) in enumerate(zip(dimensions, DIMENSIONS)):
        if not isinstance(entry, dict):
            raise ValueError(f"dimensions[{i}] is not an object: {entry!r}")
        name = str(entry.get("dimension", "")).strip().lower()
        if name != expected:
            raise ValueError(
                f"dimensions[{i}] judges {name!r}, but aspect {i} is {expected!r} "
                "- one verdict per aspect, in the order the prompt lists them"
            )
        if entry.get("verdict") not in _VERDICTS:
            raise ValueError(
                f"dimensions[{i}] verdict must be one of {sorted(_VERDICTS)}: {entry!r}"
            )
        # A gap the judge won't name is not actionable feedback, and the
        # generator would be told to widen without being told where.
        if entry["verdict"] == "gap":
            missing = entry.get("missing")
            if not isinstance(missing, str) or not missing.strip():
                raise ValueError(f"dimensions[{i}] is a gap but names no missing vibe: {entry!r}")
    return dimensions


def _parse_result(raw: str) -> MetricResult:
    dimensions = _validate_dimensions(raw)

    covered = [d for d in dimensions if d["verdict"] == "covered"]
    gaps = [d for d in dimensions if d["verdict"] == "gap"]
    applicable = len(covered) + len(gaps)

    # An image the judge found no applicable aspect in gives nothing to
    # cover, so nothing was missed: 1.0 rather than a division by zero.
    score = len(covered) / applicable if applicable else 1.0
    reason = (
        f"{len(covered)}/{applicable} applicable aspect(s) covered; "
        f"gaps: {', '.join(d['dimension'] for d in gaps) if gaps else 'none'}"
    )
    return MetricResult(
        score=score,
        reason=reason,
        details={
            "dimensions": dimensions,
            "n_applicable": applicable,
            "n_covered": len(covered),
            "gaps": [
                {"dimension": d["dimension"], "missing": d["missing"]} for d in gaps
            ],
        },
    )


class BreadthMetric(Metric):
    name = "breadth"
    threshold = BREADTH_THRESHOLD

    def __init__(self, model: str = DEFAULT_EVAL_MODEL, threshold: float | None = None):
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
            validate=_validate_dimensions,
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
            validate=_validate_dimensions,
        )
        return _parse_result(raw)

    def extract_submetrics(self, result: MetricResult) -> dict[str, float]:
        return {"applicable_aspects": float(result.details["n_applicable"])}
