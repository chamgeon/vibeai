"""Decomposition-quality metric: scores Completeness and Atom Quality for a
(representation, atoms) pair via LLM-as-a-judge."""

from vibeai.eval.parsing import extract_json
from vibeai.eval.test_cases import DecompositionTestCase
from vibeai.llm.client import DEFAULT_EVAL_MODEL, call_text, call_text_async
from vibeai.metrics.base import Metric, MetricResult
from vibeai.prompts.decomposition_eval import DECOMPOSITION_EVAL_PROMPT


def _build_prompt(test_case: DecompositionTestCase) -> str:
    atoms_block = "\n".join(f"{i + 1}. {atom}" for i, atom in enumerate(test_case.atoms))
    return DECOMPOSITION_EVAL_PROMPT.format(
        representation=test_case.representation,
        atoms=atoms_block,
    )


_REQUIRED_JUDGEMENT_ENTRY_KEYS = {"atom", "reason", "evaluation", "verdict"}
_REQUIRED_EVAL_KEYS = {"affectiveness", "atomicity", "fidelity", "evidence_preservation"}


def _is_number(x, lo: float, hi: float) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and lo <= x <= hi


def _validate_judgement_entry(entry: dict, index: int) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"atomic_judgement[{index}] is not an object: {entry!r}")
    missing = _REQUIRED_JUDGEMENT_ENTRY_KEYS - entry.keys()
    if missing:
        raise ValueError(f"atomic_judgement[{index}] missing key(s) {sorted(missing)}: {entry!r}")
    if not isinstance(entry["atom"], str) or not entry["atom"].strip():
        raise ValueError(f"atomic_judgement[{index}] has empty/invalid atom: {entry!r}")
    if not isinstance(entry["reason"], str) or not entry["reason"].strip():
        raise ValueError(f"atomic_judgement[{index}] has empty/invalid reason: {entry!r}")

    evaluation = entry["evaluation"]
    if not isinstance(evaluation, dict) or evaluation.keys() != _REQUIRED_EVAL_KEYS:
        raise ValueError(f"atomic_judgement[{index}] evaluation key mismatch: {entry!r}")
    if not all(isinstance(v, bool) for v in evaluation.values()):
        raise ValueError(f"atomic_judgement[{index}] evaluation has non-bool value(s): {entry!r}")

    if entry["verdict"] not in ("Good", "Bad"):
        raise ValueError(f"atomic_judgement[{index}] verdict must be 'Good'/'Bad': {entry!r}")
    expected_verdict = "Good" if all(evaluation.values()) else "Bad"
    if entry["verdict"] != expected_verdict:
        # The judge's own rule: an atom is Good iff all 4 criteria are true.
        # A mismatch here is a judge reasoning error, not just a formatting
        # slip, so it's worth retrying too.
        raise ValueError(
            f"atomic_judgement[{index}] verdict {entry['verdict']!r} inconsistent with "
            f"evaluation {evaluation} (expected {expected_verdict!r}): {entry!r}"
        )


def _validate_judgement(raw: str, expected_atom_count: int) -> dict:
    """Parse + validate the judge's JSON. Raises ValueError on any failure -
    used both as call_text's retry-triggering ``validate`` callback and to
    build the final MetricResult once a call has passed validation."""
    judgement = extract_json(raw)
    if not isinstance(judgement, dict):
        raise ValueError(f"Expected a JSON object, got: {raw!r}")

    atomic_judgement = judgement.get("atomic_judgement")
    if not isinstance(atomic_judgement, list) or not atomic_judgement:
        raise ValueError(f"'atomic_judgement' missing or not a non-empty list: {raw!r}")
    if len(atomic_judgement) != expected_atom_count:
        raise ValueError(
            f"'atomic_judgement' has {len(atomic_judgement)} entries, expected "
            f"{expected_atom_count} (one per input atom): {raw!r}"
        )
    for i, entry in enumerate(atomic_judgement):
        _validate_judgement_entry(entry, i)

    fv = judgement.get("final_verdict")
    if not isinstance(fv, dict):
        raise ValueError(f"'final_verdict' missing or not an object: {raw!r}")

    completeness = fv.get("completeness")
    if (
        not isinstance(completeness, dict)
        or not _is_number(completeness.get("verdict"), 1, 5)
        or not isinstance(completeness.get("reason"), str)
        or not completeness["reason"].strip()
    ):
        raise ValueError(f"'final_verdict.completeness' malformed: {raw!r}")

    atom_quality = fv.get("atom_quality")
    if (
        not isinstance(atom_quality, dict)
        or not _is_number(atom_quality.get("verdict"), 0, 5)
        or not isinstance(atom_quality.get("reason"), str)
        or not atom_quality["reason"].strip()
    ):
        raise ValueError(f"'final_verdict.atom_quality' malformed: {raw!r}")

    return judgement


def _parse_result(raw: str, expected_atom_count: int) -> MetricResult:
    judgement = _validate_judgement(raw, expected_atom_count)

    fv = judgement["final_verdict"]
    completeness = fv["completeness"]["verdict"]
    atom_quality = fv["atom_quality"]["verdict"]

    overall_0_5 = (completeness + atom_quality) / 2
    reason = (
        f"completeness={completeness} ({fv['completeness']['reason']}); "
        f"atom_quality={atom_quality} ({fv['atom_quality']['reason']})"
    )
    return MetricResult(score=overall_0_5 / 5, reason=reason, details=judgement)


class DecompositionQualityMetric(Metric):
    name = "decomposition_quality"
    threshold = 0.7  # normalized; i.e. avg raw score >= 3.5 / 5

    def __init__(self, model: str = DEFAULT_EVAL_MODEL, threshold: float | None = None):
        self.model = model
        if threshold is not None:
            self.threshold = threshold

    def measure(self, test_case: DecompositionTestCase) -> MetricResult:
        expected_atom_count = len(test_case.atoms)
        raw = call_text(
            _build_prompt(test_case),
            model=self.model,
            call_type="judge",
            validate=lambda text: _validate_judgement(text, expected_atom_count),
        )
        return _parse_result(raw, expected_atom_count)

    async def measure_async(self, test_case: DecompositionTestCase) -> MetricResult:
        expected_atom_count = len(test_case.atoms)
        raw = await call_text_async(
            _build_prompt(test_case),
            model=self.model,
            call_type="judge",
            validate=lambda text: _validate_judgement(text, expected_atom_count),
        )
        return _parse_result(raw, expected_atom_count)

    def extract_submetrics(self, result: MetricResult) -> dict[str, float]:
        fv = result.details["final_verdict"]
        return {
            "completeness": fv["completeness"]["verdict"] / 5,
            "atom_quality": fv["atom_quality"]["verdict"] / 5,
        }
