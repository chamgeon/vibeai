"""Generator-evaluator workflow: image -> a representation both judges accept.

The batch evals score one representation per image and stop. This runs the
same pipeline as a loop, with the judges wired back into the generator:

    generate (v1) -> decompose (v2) -> evaluate -> accept?
        ^                                            |
        +------------- feedback (refine) ------------+

Two gates, both applied every round:

*   **Plausibility** is inherent to the representation, so the workflow's gate
    is the plausibility judge itself (``metrics/plausibility.py``), at a
    tighter bar than the batch metric's: ACCEPT_PLAUSIBLE_RATE of the atoms
    must be plausible. Rejection hands back the failed atoms and what failed
    about each.
*   **Richness** is defined against a vibe pool the generator must not see -
    the pool is the answer key the richness metric scores against, and a loop
    that fed pool misses back would be optimising against its own benchmark.
    The gate is ``metrics/breadth.py`` instead: the same breadth question,
    asked of the image over the ten vibe aspects rather than of a reference
    set. Rejection hands back the vibes it found unreached, and any gap at all
    is a rejection.

The two judges are independent, so they run concurrently and their feedback is
merged into one revision: a round that fails both should fix both at once
rather than spend two rounds alternating.

The loop runs at most MAX_ROUNDS times and stops early the first time both
gates pass. Terminating unaccepted is a normal outcome, not an error - the
last round's representation is still returned, with ``accepted`` False. Every
round is kept in ``WorkflowResult.rounds``, since what the loop did on the way
is the point of running it.

Every call - generation, decomposition, both judges - uses one model
(DEFAULT_MODEL), unlike the batch evals, which set the three independently.
"""

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from vibeai.eval.test_cases import DecompositionTestCase
from vibeai.llm.client import DEFAULT_MODEL, call_with_image_async
from vibeai.metrics.base import MetricResult
from vibeai.metrics.breadth import BreadthMetric
from vibeai.metrics.plausibility import PlausibilityMetric
from vibeai.pipeline.decompose import decompose_async, decompose_direct
from vibeai.pipeline.evaluate import DIRECT_DECOMPOSITION
from vibeai.pipeline.represent import MIME_TYPES, generate_representation_async
from vibeai.prompts.refine import REFINE_PROMPT

# Share of atoms that must be plausible for the plausibility gate to pass.
# Above the batch metric's 0.7: the loop exists to fix what a single pass got
# wrong, so it can afford to ask for more than a single pass is scored on.
ACCEPT_PLAUSIBLE_RATE = 0.9

# How many times the generate -> decompose -> evaluate loop body may run,
# counting the first (unrefined) candidate. At 3 the generator gets two
# chances to act on feedback.
MAX_ROUNDS = 3

DEFAULT_REPRESENTATION_PROMPT = "v1"
DEFAULT_DECOMPOSITION_PROMPT = "v2"


@dataclass
class Round:
    """One pass of the loop body, kept whether it was accepted or not."""

    index: int
    representation: str
    atoms: list[str]
    plausibility: MetricResult
    breadth: MetricResult
    plausibility_passed: bool
    breadth_passed: bool
    feedback: str = ""
    atom_details: list[dict] | None = None

    @property
    def accepted(self) -> bool:
        return self.plausibility_passed and self.breadth_passed


@dataclass
class WorkflowResult:
    image_path: Path
    rounds: list[Round] = field(default_factory=list)

    @property
    def final(self) -> Round:
        return self.rounds[-1]

    @property
    def accepted(self) -> bool:
        """Whether the loop stopped because both gates passed, rather than
        because it ran out of rounds."""
        return bool(self.rounds) and self.final.accepted

    @property
    def representation(self) -> str:
        return self.final.representation


def _load_image(image_path: Path) -> tuple[bytes, str]:
    mime_type = MIME_TYPES.get(image_path.suffix.lower(), "image/jpeg")
    return image_path.read_bytes(), mime_type


def _failure_reason(atom: dict) -> str:
    """Why the plausibility judge rejected this atom, in the generator's
    terms. The judge stops at the first check that fails, so the first false
    verdict is the reason."""
    presence = atom.get("evidence_presence_check")
    if presence and not presence.get("verdict"):
        return f"the visual evidence you named is not in the image - {presence.get('reasoning', '')}"

    direct = atom.get("direct_check")
    if direct and not direct.get("verdict"):
        if direct.get("contradicting_evidence"):
            contradicting = "; ".join(str(cue) for cue in direct["contradicting_evidence"])
            return f"the image works against this vibe: {contradicting}"
        return f"nothing in the image supports this vibe - {direct.get('reasoning', '')}"

    # A mapping failure is the one rejection that is not a drop: the judge
    # has already confirmed the evidence is in the image and the vibe is
    # inferable from it, so only the pairing is wrong. Said as flatly as the
    # other two, it reads as "delete this", and the generator loses an aspect
    # it had covered - so this reason spells out what survived and asks for a
    # re-pairing.
    mapping = atom.get("mapping_check")
    if mapping and not mapping.get("verdict"):
        return (
            "this vibe IS in the image and the evidence you named IS in the image - "
            "they are just not connected. The cue you named is only a minor or incidental "
            "contributor to this vibe, not what produces it. Keep the vibe and name the cue "
            f"that actually drives it; do not drop the claim - {mapping.get('reasoning', '')}"
        )

    # Shouldn't happen: final_verdict is false only when a check failed. Kept
    # as a reason rather than an assertion so one odd judge response degrades
    # the feedback instead of killing the run.
    return "judged implausible"


def plausibility_failures(result: MetricResult) -> list[dict]:
    """The rejected atoms, each with the reason it was rejected."""
    return [
        {"atom": atom["atom"], "reason": _failure_reason(atom)}
        for atom in result.details["atoms"]
        if not atom["final_verdict"]
    ]


def format_feedback(failures: list[dict], gaps: list[dict]) -> str:
    """The two judges' findings as one instruction block for the generator.

    Breadth gaps are passed as bare vibe phrases: the aspect name each came
    from is kept in the round record for analysis, but handing the generator
    the evaluator's ten axes would have it widen along the rubric rather than
    look harder at the image.
    """
    sections = []
    if failures:
        lines = "\n".join(f'- "{f["atom"]}" — {f["reason"]}' for f in failures)
        sections.append(
            "### Claims that did not hold up against the image\n"
            "Each of these was checked against the image and rejected.\n" + lines
        )
    if gaps:
        lines = "\n".join(f"- {gap['missing']}" for gap in gaps)
        sections.append(
            "### Vibe in the image the representation has not reached\n"
            "The image carries these feelings, and your representation spends no words on them.\n"
            + lines
        )
    return "\n\n".join(sections)


async def _generate(
    image_path: Path,
    previous: Round | None,
    representation_prompt_version: str,
    model: str,
) -> str:
    """The first candidate, or a revision of the previous one."""
    if previous is None:
        return await generate_representation_async(
            image_path, prompt_version=representation_prompt_version, model=model
        )

    image_bytes, mime_type = _load_image(image_path)
    prompt = REFINE_PROMPT.format(
        representation=previous.representation, feedback=previous.feedback
    )
    return await call_with_image_async(
        prompt, image_bytes, mime_type=mime_type, model=model, call_type="represent"
    )


async def _decompose(
    representation: str, decomposition_prompt_version: str, model: str
) -> tuple[list[str], list[dict] | None]:
    if decomposition_prompt_version == DIRECT_DECOMPOSITION:
        return decompose_direct(representation), None
    return await decompose_async(
        representation, prompt_version=decomposition_prompt_version, model=model
    )


async def run_workflow(
    image_path: Path,
    representation_prompt_version: str = DEFAULT_REPRESENTATION_PROMPT,
    decomposition_prompt_version: str = DEFAULT_DECOMPOSITION_PROMPT,
    model: str = DEFAULT_MODEL,
    max_rounds: int = MAX_ROUNDS,
    plausible_rate: float = ACCEPT_PLAUSIBLE_RATE,
) -> WorkflowResult:
    image_path = Path(image_path)
    plausibility = PlausibilityMetric(model=model, threshold=plausible_rate)
    breadth = BreadthMetric(model=model)

    result = WorkflowResult(image_path=image_path)
    previous: Round | None = None

    for index in range(max_rounds):
        representation = await _generate(
            image_path, previous, representation_prompt_version, model
        )
        atoms, atom_details = await _decompose(
            representation, decomposition_prompt_version, model
        )
        test_case = DecompositionTestCase(
            image_path=image_path,
            representation=representation,
            atoms=atoms,
            representation_prompt_version=representation_prompt_version,
            decomposition_prompt_version=decomposition_prompt_version,
            atom_details=atom_details,
        )

        plausibility_result, breadth_result = await asyncio.gather(
            plausibility.measure_async(test_case), breadth.measure_async(test_case)
        )
        gaps = breadth_result.details["gaps"]

        current = Round(
            index=index,
            representation=representation,
            atoms=atoms,
            atom_details=atom_details,
            plausibility=plausibility_result,
            breadth=breadth_result,
            plausibility_passed=plausibility.is_successful(plausibility_result),
            # Stricter than BreadthMetric's threshold: the gate is "is there
            # anything left to say", so one unreached vibe is a rejection.
            breadth_passed=not gaps,
        )
        current.feedback = (
            ""
            if current.accepted
            else format_feedback(plausibility_failures(plausibility_result), gaps)
        )
        result.rounds.append(current)

        if current.accepted:
            break
        previous = current

    return result


def round_summary(round_: Round) -> str:
    verdict = "accepted" if round_.accepted else "rejected"
    return (
        f"round {round_.index + 1}: {verdict} "
        f"(plausibility {round_.plausibility.score:.2f} "
        f"{'pass' if round_.plausibility_passed else 'fail'}, "
        f"breadth {round_.breadth.score:.2f} "
        f"{'pass' if round_.breadth_passed else 'fail'}, "
        f"{len(round_.atoms)} atoms)"
    )


def as_record(result: WorkflowResult) -> dict:
    """The whole trace as JSON, for saving or inspection."""
    return {
        "image_path": str(result.image_path),
        "accepted": result.accepted,
        "n_rounds": len(result.rounds),
        "representation": result.representation,
        "rounds": [
            {
                "index": round_.index,
                "representation": round_.representation,
                "atoms": round_.atoms,
                "atom_details": round_.atom_details,
                "plausibility_score": round_.plausibility.score,
                "plausibility_passed": round_.plausibility_passed,
                "plausibility_failures": plausibility_failures(round_.plausibility),
                "breadth_score": round_.breadth.score,
                "breadth_passed": round_.breadth_passed,
                "breadth_dimensions": round_.breadth.details["dimensions"],
                "feedback": round_.feedback,
            }
            for round_ in result.rounds
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the generator-evaluator workflow on one image."
    )
    parser.add_argument("image", type=Path, help="Image to represent")
    parser.add_argument(
        "--representation-prompt-version",
        default=DEFAULT_REPRESENTATION_PROMPT,
        help=f"Prompt for the first candidate (default: {DEFAULT_REPRESENTATION_PROMPT}); "
        "revisions always use the refine prompt",
    )
    parser.add_argument(
        "--decomposition-prompt-version",
        default=DEFAULT_DECOMPOSITION_PROMPT,
        help=f"Decomposition prompt version (default: {DEFAULT_DECOMPOSITION_PROMPT})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model for every call in the loop (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=MAX_ROUNDS, help=f"Default: {MAX_ROUNDS}"
    )
    parser.add_argument(
        "--plausible-rate",
        type=float,
        default=ACCEPT_PLAUSIBLE_RATE,
        help=f"Share of atoms that must be plausible (default: {ACCEPT_PLAUSIBLE_RATE})",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write the full trace as JSON")
    args = parser.parse_args()

    result = asyncio.run(
        run_workflow(
            args.image,
            representation_prompt_version=args.representation_prompt_version,
            decomposition_prompt_version=args.decomposition_prompt_version,
            model=args.model,
            max_rounds=args.max_rounds,
            plausible_rate=args.plausible_rate,
        )
    )

    for round_ in result.rounds:
        print(round_summary(round_))
        if round_.feedback:
            print(round_.feedback)
        print()
    print("accepted" if result.accepted else f"not accepted after {len(result.rounds)} round(s)")
    print()
    print(result.representation)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(as_record(result), indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
