"""Richness metric: how much of an image's vibe pool a representation captured.

The pool (see ``pipeline/pool_construction.py``) is the reference set of vibes
several independent representation branches found in the image and the
plausibility judge verified against it. Richness asks the complementary
question to plausibility: not "is what the representation said true", but "how
much of what is there did it say".

Scoring runs the witness test in ``prompts/richness_eval.py`` over every pool
vibe against the representation's own vibe set:

    score = pool vibes judged *redundant* with the target set / pool vibes

A pool vibe judged redundant is one the target set already conveys - covered.
A pool vibe judged distinct is one the representation missed. The judge never
sees the image: both sides were already verified against it, so the only open
question is semantic coverage, and showing the image would invite the judge to
re-litigate whether a pool vibe is really there.

The target set is bare vibes, not atom sentences, because the pool is bare
vibes - comparing "warm afternoon light through the blinds, giving it a
languid vibe" against "languid" would make the witness test partly a judgement
about evidence phrasing. Structured (v2) decompositions name the vibe in a
dedicated field, so it is used directly; for prompts that emit plain sentences
the sentence is the best available target wording.
"""

import json
from pathlib import Path

from vibeai.eval.parsing import extract_json
from vibeai.eval.test_cases import DecompositionTestCase
from vibeai.llm.client import DEFAULT_EVAL_MODEL, call_text, call_text_async
from vibeai.metrics.base import Metric, MetricResult
from vibeai.pipeline.pool_construction import normalize_vibe
from vibeai.prompts.richness_eval import RICHNESS_EVAL_PROMPT

DEFAULT_POOL_PATH = Path("annotations/richness_holdout.vibes.json")

# A representation passes when it covers this share of its image's pool. The
# pool unions four branches, so no single representation is expected to cover
# all of it - this is a starting line to move, not a claim about what is
# achievable.
COVERAGE_THRESHOLD = 0.7

_VERDICTS = {"distinct", "redundant"}


def load_pool(path: Path = DEFAULT_POOL_PATH) -> dict[str, dict]:
    """Load a pool file written by ``pipeline.pool_construction``, keyed by
    image path string."""
    if not path.exists():
        raise FileNotFoundError(
            f"No vibe pool at {path}. Build one with "
            "`uv run -m vibeai.pipeline.pool_construction --image-dir <dir>`."
        )
    return json.loads(path.read_text())


def pool_image_paths(pool: dict[str, dict]) -> list[Path]:
    """The images a pool covers, in sorted order (the order a pool file is
    written in is the order its images were passed in)."""
    return sorted(Path(key) for key in pool)


def target_vibes(test_case: DecompositionTestCase) -> list[str]:
    """The representation's vibe set, deduplicated on the same surface-level
    key the pool uses, preserving order."""
    if test_case.atom_details:
        raw = [str(detail.get("vibe", "")).strip() for detail in test_case.atom_details]
    else:
        raw = [atom.strip() for atom in test_case.atoms]

    seen = set()
    vibes = []
    for vibe in raw:
        key = normalize_vibe(vibe)
        if vibe and key not in seen:
            seen.add(key)
            vibes.append(vibe)
    return vibes


def _format_list(items: list[str]) -> str:
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))


def _build_prompt(target: list[str], pool_vibes: list[str]) -> str:
    return RICHNESS_EVAL_PROMPT.format(
        target=_format_list(target) if target else "(empty)",
        pool=_format_list(pool_vibes),
    )


def _validate_judgements(raw: str, pool_vibes: list[str]) -> list[dict]:
    """Parse + validate the judge's JSON. Raises ValueError on any failure -
    used both as the retry-triggering ``validate`` callback and to build the
    MetricResult once a call has passed.

    The candidates are checked against the pool one-for-one and in order: the
    score is a fraction over the pool, so a judge that skipped, reordered or
    invented an entry would quietly change the denominator's meaning.
    """
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got: {raw!r}")

    judgements = parsed.get("judgements")
    if not isinstance(judgements, list):
        raise ValueError(f"'judgements' missing or not a list: {raw!r}")
    if len(judgements) != len(pool_vibes):
        raise ValueError(
            f"Expected {len(pool_vibes)} judgement(s), got {len(judgements)}: {raw!r}"
        )

    for i, (judgement, pool_vibe) in enumerate(zip(judgements, pool_vibes)):
        if not isinstance(judgement, dict):
            raise ValueError(f"judgements[{i}] is not an object: {judgement!r}")
        candidate = str(judgement.get("candidate", "")).strip()
        if normalize_vibe(candidate) != normalize_vibe(pool_vibe):
            raise ValueError(
                f"judgements[{i}] judges {candidate!r}, but pool[{i}] is {pool_vibe!r} "
                "- judgements must be one per pool vibe, in pool order"
            )
        if judgement.get("verdict") not in _VERDICTS:
            raise ValueError(
                f"judgements[{i}] verdict must be one of {sorted(_VERDICTS)}: {judgement!r}"
            )
    return judgements


def _weighted(entries: list[dict], covered: list[bool]) -> float:
    """Coverage with each pool vibe weighted by how many branches found it, so
    missing a vibe all four branches saw counts for more than missing one a
    single branch mentioned."""
    total = sum(entry.get("support", 1) for entry in entries)
    if not total:
        return 0.0
    return sum(
        entry.get("support", 1) for entry, is_covered in zip(entries, covered) if is_covered
    ) / total


def _parse_result(raw: str, entries: list[dict], target: list[str]) -> MetricResult:
    pool_vibes = [entry["vibe"] for entry in entries]
    judgements = _validate_judgements(raw, pool_vibes)

    covered = [j["verdict"] == "redundant" for j in judgements]
    n_covered = sum(covered)
    total = len(covered)

    missed = [vibe for vibe, is_covered in zip(pool_vibes, covered) if not is_covered]
    reason = (
        f"{n_covered}/{total} pool vibes covered by {len(target)} target vibe(s); "
        f"missed: {', '.join(missed) if missed else 'none'}"
    )
    return MetricResult(
        score=n_covered / total,
        reason=reason,
        details={
            "target": target,
            "pool_size": total,
            "n_covered": n_covered,
            "missed": missed,
            "weighted_coverage": _weighted(entries, covered),
            "judgements": judgements,
        },
    )


class RichnessMetric(Metric):
    """Needs a pool covering every image it is asked to measure; measuring an
    image the pool doesn't have is a KeyError rather than a zero, since an
    absent pool is a missing reference, not an uncovered image."""

    name = "richness"
    threshold = COVERAGE_THRESHOLD

    def __init__(
        self,
        pool: dict[str, dict] | None = None,
        pool_path: Path = DEFAULT_POOL_PATH,
        model: str = DEFAULT_EVAL_MODEL,
        threshold: float | None = None,
    ):
        self.pool = load_pool(pool_path) if pool is None else pool
        self.model = model
        if threshold is not None:
            self.threshold = threshold

    def _entries(self, test_case: DecompositionTestCase) -> list[dict]:
        key = str(test_case.image_path)
        record = self.pool.get(key)
        if record is None:
            raise KeyError(f"No vibe pool for {key}; pool covers {sorted(self.pool)}")
        entries = record["vibes"]
        if not entries:
            raise ValueError(f"Vibe pool for {key} is empty")
        return entries

    def measure(self, test_case: DecompositionTestCase) -> MetricResult:
        entries = self._entries(test_case)
        target = target_vibes(test_case)
        pool_vibes = [entry["vibe"] for entry in entries]
        raw = call_text(
            _build_prompt(target, pool_vibes),
            model=self.model,
            call_type="judge",
            validate=lambda text: _validate_judgements(text, pool_vibes),
        )
        return _parse_result(raw, entries, target)

    async def measure_async(self, test_case: DecompositionTestCase) -> MetricResult:
        entries = self._entries(test_case)
        target = target_vibes(test_case)
        pool_vibes = [entry["vibe"] for entry in entries]
        raw = await call_text_async(
            _build_prompt(target, pool_vibes),
            model=self.model,
            call_type="judge",
            validate=lambda text: _validate_judgements(text, pool_vibes),
        )
        return _parse_result(raw, entries, target)

    def extract_submetrics(self, result: MetricResult) -> dict[str, float]:
        return {"weighted_coverage": result.details["weighted_coverage"]}
