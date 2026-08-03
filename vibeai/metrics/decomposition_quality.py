"""Decomposition-quality metric: scores Completeness, Claim Independence,
and Atom Quality for a (representation, atoms) pair via LLM-as-a-judge."""

from vibeai.eval.parsing import extract_json
from vibeai.eval.test_cases import DecompositionTestCase
from vibeai.llm.client import DEFAULT_MODEL, call_text
from vibeai.metrics.base import Metric, MetricResult
from vibeai.prompts.decomposition_eval import DECOMPOSITION_EVAL_PROMPT


class DecompositionQualityMetric(Metric):
    name = "decomposition_quality"
    threshold = 0.7  # normalized; i.e. avg raw score >= 3.5 / 5

    def __init__(self, model: str = DEFAULT_MODEL, threshold: float | None = None):
        self.model = model
        if threshold is not None:
            self.threshold = threshold

    def measure(self, test_case: DecompositionTestCase) -> MetricResult:
        atoms_block = "\n".join(f"{i + 1}. {atom}" for i, atom in enumerate(test_case.atoms))
        prompt = DECOMPOSITION_EVAL_PROMPT.format(
            representation=test_case.representation,
            atoms=atoms_block,
        )
        raw = call_text(prompt, model=self.model)
        judgement = extract_json(raw)

        fv = judgement["final_verdict"]
        completeness = fv["completeness"]["verdict"]
        claim_independence = fv["claim_independence"]["verdict"]
        atom_quality = fv["atom_quality"]["verdict"]

        overall_0_5 = (completeness + claim_independence + atom_quality) / 3
        reason = (
            f"completeness={completeness} ({fv['completeness']['reason']}); "
            f"claim_independence={claim_independence} ({fv['claim_independence']['reason']}); "
            f"atom_quality={atom_quality} ({fv['atom_quality']['reason']})"
        )
        return MetricResult(score=overall_0_5 / 5, reason=reason, details=judgement)
