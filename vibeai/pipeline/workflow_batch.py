"""Batch runner for the generator-evaluator workflow.

Runs ``pipeline.workflow.run_workflow`` over an image set and rolls the traces
up into one summary. It deliberately does not go through
``eval/prompt_results.py``: that aggregator is built around one MetricResult
per image, and a workflow run has two scores, an accept/reject, and a variable
number of rounds. The file layout is kept the same anyway -

    results/workflow/{run_name}.json                  - summary
    results/workflow/{run_name}.per_image.jsonl       - one full trace per line
    results/workflow/{run_name}.representations.json  - image -> final representation

- so runs sit beside the metric runs and diff the same way.

What the summary is for is the first-vs-final comparison. Both judges score
every round, so each image carries its own before/after, and the numbers that
say whether the loop is worth its cost are:

*   ``accept_rate`` - how often it converged inside the round budget.
*   ``first_*`` vs ``final_*`` - what the loop bought, per judge.
*   ``n_regressed_*`` - images whose final round scored *worse* than the best
    round they passed through. This is the keep-list question, measured: if
    it stays near zero across a real image set, returning the last round is
    fine and best-of-k selection is solving a problem the loop doesn't have.

Richness is not scored here. It needs the vibe pool, and the whole point of
the breadth gate is that the loop never sees it - so pool scoring is a
separate pass over the representations this run writes out (see README).

That pass is what the ``.representations.json`` file is for, and why it holds
nothing but image -> final representation. The workflow's own atoms and
verdicts are a by-product of the loop: they come from the loop's decomposer
and its judges, which run on DEFAULT_MODEL because the loop makes several
calls per round. Scoring against baseline and v1 has to re-decompose and
re-judge the final representation under the same prompts and the same (bigger)
eval model those runs used, or the comparison measures the workflow's cheaper
judges as much as its output.
"""

import argparse
import asyncio
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev

from vibeai.eval.concurrency import gather_bounded_as_completed
from vibeai.eval.dataset import load_image_paths
from vibeai.eval.prompt_results import RESULTS_DIR, ImageError
from vibeai.llm.client import DEFAULT_MODEL
from vibeai.metrics.richness import DEFAULT_POOL_PATH, load_pool, pool_image_paths
from vibeai.pipeline.workflow import (
    ACCEPT_PLAUSIBLE_RATE,
    DEFAULT_DECOMPOSITION_PROMPT,
    DEFAULT_REPRESENTATION_PROMPT,
    MAX_ROUNDS,
    WorkflowResult,
    as_record,
    plausibility_failures,
    round_summary,
    run_workflow,
)

WORKFLOW_RESULTS_DIR = RESULTS_DIR / "workflow"

# Sentinel representation_prompt_version for the batch evals: score the
# generator-evaluator workflow's output rather than a single representation
# prompt. It is not a key in prompts/representation.py - there is no one
# prompt that produces it - so ``resolve_representations`` supplies the text
# and ``evaluate_image`` skips its generation call. Runs land under
# results/<metric>/workflow__<decomp>_<time>.json, beside the prompt runs.
WORKFLOW_REPRESENTATION_VERSION = "workflow"


@dataclass
class WorkflowBatchSummary:
    representation_prompt_version: str
    decomposition_prompt_version: str
    model: str
    timestamp: str
    max_rounds: int
    plausible_rate: float

    n: int  # images that completed
    n_errors: int  # images that raised instead of completing

    accept_rate: float
    mean_rounds: float
    rounds_histogram: dict[str, int]

    # Round 1 is the unrefined candidate, so first -> final is what the
    # feedback loop actually changed.
    first_plausibility: float
    final_plausibility: float
    first_breadth: float
    final_breadth: float
    first_gaps: float
    final_gaps: float
    first_implausible_atoms: float
    final_implausible_atoms: float
    first_atoms: float
    final_atoms: float

    # Images whose final round scored below the best round they passed
    # through - the loop talked itself out of something it already had.
    n_regressed_plausibility: int
    n_regressed_breadth: int

    final_plausibility_std: float
    final_breadth_std: float

    not_accepted: list[str] = field(default_factory=list)
    errors: list[ImageError] = field(default_factory=list)


def _first_final(results: list[WorkflowResult], get) -> tuple[float, float]:
    return (
        mean(get(r.rounds[0]) for r in results),
        mean(get(r.final) for r in results),
    )


def summarize(
    results: list[WorkflowResult],
    errors: list[ImageError],
    *,
    representation_prompt_version: str,
    decomposition_prompt_version: str,
    model: str,
    max_rounds: int,
    plausible_rate: float,
) -> WorkflowBatchSummary:
    first_plaus, final_plaus = _first_final(results, lambda r: r.plausibility.score)
    first_breadth, final_breadth = _first_final(results, lambda r: r.breadth.score)
    first_gaps, final_gaps = _first_final(results, lambda r: len(r.breadth.details["gaps"]))
    first_bad, final_bad = _first_final(
        results, lambda r: len(plausibility_failures(r.plausibility))
    )
    first_atoms, final_atoms = _first_final(results, lambda r: len(r.atoms))

    final_plaus_scores = [r.final.plausibility.score for r in results]
    final_breadth_scores = [r.final.breadth.score for r in results]

    return WorkflowBatchSummary(
        representation_prompt_version=representation_prompt_version,
        decomposition_prompt_version=decomposition_prompt_version,
        model=model,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        max_rounds=max_rounds,
        plausible_rate=plausible_rate,
        n=len(results),
        n_errors=len(errors),
        accept_rate=mean(float(r.accepted) for r in results),
        mean_rounds=mean(len(r.rounds) for r in results),
        rounds_histogram={
            str(k): v for k, v in sorted(Counter(len(r.rounds) for r in results).items())
        },
        first_plausibility=first_plaus,
        final_plausibility=final_plaus,
        first_breadth=first_breadth,
        final_breadth=final_breadth,
        first_gaps=first_gaps,
        final_gaps=final_gaps,
        first_implausible_atoms=first_bad,
        final_implausible_atoms=final_bad,
        first_atoms=first_atoms,
        final_atoms=final_atoms,
        n_regressed_plausibility=sum(
            1
            for r in results
            if r.final.plausibility.score < max(x.plausibility.score for x in r.rounds)
        ),
        n_regressed_breadth=sum(
            1 for r in results if r.final.breadth.score < max(x.breadth.score for x in r.rounds)
        ),
        final_plausibility_std=pstdev(final_plaus_scores),
        final_breadth_std=pstdev(final_breadth_scores),
        not_accepted=[str(r.image_path) for r in results if not r.accepted],
        errors=errors,
    )


def representations(results: list[WorkflowResult]) -> dict[str, str]:
    """The run's actual product: image path -> final representation.

    Keyed by image path string, the same way a vibe pool file is, so a
    scoring pass can line the two up by key.
    """
    return {str(result.image_path): result.representation for result in results}


def save(
    summary: WorkflowBatchSummary, results: list[WorkflowResult], run_name: str
) -> tuple[Path, Path, Path]:
    WORKFLOW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = WORKFLOW_RESULTS_DIR / f"{run_name}.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2))

    per_image_path = WORKFLOW_RESULTS_DIR / f"{run_name}.per_image.jsonl"
    with per_image_path.open("w") as f:
        for result in results:
            f.write(json.dumps(as_record(result)) + "\n")

    representations_path = WORKFLOW_RESULTS_DIR / f"{run_name}.representations.json"
    representations_path.write_text(json.dumps(representations(results), indent=2))

    return summary_path, per_image_path, representations_path


def format_summary(summary: WorkflowBatchSummary) -> str:
    def delta(first: float, final: float) -> str:
        return f"{first:.2f} -> {final:.2f} ({final - first:+.2f})"

    lines = [
        f"{summary.n} image(s), {summary.n_errors} error(s), "
        f"{summary.model}, {summary.representation_prompt_version}"
        f"__{summary.decomposition_prompt_version}, max {summary.max_rounds} round(s)",
        f"accepted            {summary.accept_rate:.0%}",
        f"rounds              {summary.mean_rounds:.2f} mean {summary.rounds_histogram}",
        f"plausibility        {delta(summary.first_plausibility, summary.final_plausibility)}",
        f"breadth             {delta(summary.first_breadth, summary.final_breadth)}",
        f"gaps / image        {delta(summary.first_gaps, summary.final_gaps)}",
        f"implausible atoms   {delta(summary.first_implausible_atoms, summary.final_implausible_atoms)}",
        f"atoms / image       {delta(summary.first_atoms, summary.final_atoms)}",
        f"regressed vs best   plausibility {summary.n_regressed_plausibility}, "
        f"breadth {summary.n_regressed_breadth} (of {summary.n})",
    ]
    if summary.not_accepted:
        lines.append("not accepted        " + ", ".join(Path(p).name for p in summary.not_accepted))
    return "\n".join(lines)


async def run_batch(
    image_paths: list[Path],
    *,
    representation_prompt_version: str = DEFAULT_REPRESENTATION_PROMPT,
    decomposition_prompt_version: str = DEFAULT_DECOMPOSITION_PROMPT,
    model: str = DEFAULT_MODEL,
    max_rounds: int = MAX_ROUNDS,
    plausible_rate: float = ACCEPT_PLAUSIBLE_RATE,
    concurrency: int = 10,
) -> tuple[list[WorkflowResult], list[ImageError]]:
    """One image failing doesn't abort the batch: the loop is several LLM
    calls deep per image, so a late failure on one image shouldn't throw away
    the traces already paid for."""
    coros = [
        run_workflow(
            image_path,
            representation_prompt_version=representation_prompt_version,
            decomposition_prompt_version=decomposition_prompt_version,
            model=model,
            max_rounds=max_rounds,
            plausible_rate=plausible_rate,
        )
        for image_path in image_paths
    ]

    results: list[WorkflowResult] = []
    errors: list[ImageError] = []
    async for index, outcome in gather_bounded_as_completed(coros, limit=concurrency):
        image_path = image_paths[index]
        if isinstance(outcome, BaseException):
            errors.append(
                ImageError(
                    image_path=str(image_path),
                    error_type=type(outcome).__name__,
                    error_message=str(outcome),
                )
            )
            print(f"{image_path.name}: {type(outcome).__name__}: {outcome}")
            continue

        results.append(outcome)
        verdict = "accepted" if outcome.accepted else "NOT accepted"
        print(
            f"{image_path.name}: {verdict} after {len(outcome.rounds)} round(s) - "
            + "; ".join(
                f"r{r.index + 1} {r.plausibility.score:.2f}/{r.breadth.score:.2f}"
                for r in outcome.rounds
            )
        )

    # as_completed yields in finish order; sort so runs over the same image
    # set line up line-for-line in the per-image JSONL.
    results.sort(key=lambda r: str(r.image_path))
    return results, errors


def saved_representations(results_dir: Path | None = None) -> dict[str, str]:
    """Every workflow representation written so far, image path -> text.

    Merged over all ``*.representations.json`` in the results dir, oldest
    first, so a later run's representation for an image wins over an earlier
    one's. Nothing distinguishes runs made at different settings - an image
    is looked up by path alone - so forcing a regeneration means deleting or
    moving the file that holds it.
    """
    results_dir = results_dir or WORKFLOW_RESULTS_DIR
    merged: dict[str, str] = {}
    if not results_dir.exists():
        return merged
    for path in sorted(
        results_dir.glob("*.representations.json"), key=lambda p: p.stat().st_mtime
    ):
        merged.update(json.loads(path.read_text()))
    return merged


async def resolve_representations(
    image_paths: list[Path],
    representation_prompt_version: str,
    *,
    decomposition_prompt_version: str = DEFAULT_DECOMPOSITION_PROMPT,
    model: str = DEFAULT_MODEL,
    max_rounds: int = MAX_ROUNDS,
    plausible_rate: float = ACCEPT_PLAUSIBLE_RATE,
    concurrency: int = 10,
    results_dir: Path | None = None,
) -> tuple[list[Path], dict[str, str] | None]:
    """Resolve a batch eval's image set against the workflow sentinel.

    For any ordinary prompt version this is a no-op: returns the images
    unchanged and None, and the caller generates representations as before.

    For WORKFLOW_REPRESENTATION_VERSION it returns (images, mapping). Images
    already covered by a saved ``.representations.json`` are reused as-is;
    the rest are run through the workflow now and saved, so the eval can be
    pointed at the sentinel without a separate batch run first, and a rerun
    costs nothing. Images the workflow could not produce a representation for
    are dropped from the returned list rather than failing the whole eval -
    the batch prints each one's error as it happens.

    The keyword arguments configure the workflow only for images that still
    have to be run; they do not re-run or validate anything already saved.
    """
    if representation_prompt_version != WORKFLOW_REPRESENTATION_VERSION:
        return image_paths, None

    known = saved_representations(results_dir)
    missing = [path for path in image_paths if str(path) not in known]

    if missing:
        print(
            f"{len(image_paths) - len(missing)}/{len(image_paths)} representation(s) already "
            f"saved; running the workflow for {len(missing)}"
        )
        results, errors = await run_batch(
            missing,
            decomposition_prompt_version=decomposition_prompt_version,
            model=model,
            max_rounds=max_rounds,
            plausible_rate=plausible_rate,
            concurrency=concurrency,
        )
        if results:
            summary = summarize(
                results,
                errors,
                representation_prompt_version=DEFAULT_REPRESENTATION_PROMPT,
                decomposition_prompt_version=decomposition_prompt_version,
                model=model,
                max_rounds=max_rounds,
                plausible_rate=plausible_rate,
            )
            summary_path, _, representations_path = save(
                summary, results, f"autorun_{int(time.time())}"
            )
            print(f"\n{format_summary(summary)}")
            print(f"wrote {summary_path}\nwrote {representations_path}\n")
            known.update(representations(results))
    else:
        print(f"all {len(image_paths)} representation(s) already saved")

    resolved = {
        str(path): known[str(path)] for path in image_paths if str(path) in known
    }
    dropped = [path for path in image_paths if str(path) not in resolved]
    if dropped:
        print(
            f"skipping {len(dropped)} image(s) with no workflow representation: "
            + ", ".join(path.name for path in dropped)
        )
    return [path for path in image_paths if str(path) in resolved], resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the generator-evaluator workflow over an image set."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory of images (default: the dataset default, data/main_processed)",
    )
    parser.add_argument(
        "--pool",
        type=Path,
        nargs="?",
        const=DEFAULT_POOL_PATH,
        default=None,
        help="Take the image set from a vibe pool file instead of a directory, so the run "
        f"covers exactly the images richness can score (bare flag: {DEFAULT_POOL_PATH})",
    )
    parser.add_argument(
        "--n-images", type=int, default=None, help="Run only the first N images (default: all)"
    )
    parser.add_argument(
        "--representation-prompt-version",
        default=DEFAULT_REPRESENTATION_PROMPT,
        help=f"Prompt for each image's first candidate (default: {DEFAULT_REPRESENTATION_PROMPT}); "
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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max images in flight (default: 10). Each one runs up to 4 calls per round.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output file stem (default: <rep>__<decomp>_<unix time>)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every round's scores and feedback as the batch finishes",
    )
    args = parser.parse_args()

    if args.pool is not None:
        image_paths = pool_image_paths(load_pool(args.pool))
    else:
        image_paths = load_image_paths(data_dir=args.image_dir)
    if args.n_images is not None:
        image_paths = image_paths[: args.n_images]
    if not image_paths:
        raise SystemExit(f"No images found in {args.pool or args.image_dir}")

    results, errors = asyncio.run(
        run_batch(
            image_paths,
            representation_prompt_version=args.representation_prompt_version,
            decomposition_prompt_version=args.decomposition_prompt_version,
            model=args.model,
            max_rounds=args.max_rounds,
            plausible_rate=args.plausible_rate,
            concurrency=args.concurrency,
        )
    )
    if not results:
        raise SystemExit(f"No image completed ({len(errors)} error(s))")

    if args.verbose:
        for result in results:
            print(f"\n=== {result.image_path.name}")
            for round_ in result.rounds:
                print(round_summary(round_))
                if round_.feedback:
                    print(round_.feedback)

    summary = summarize(
        results,
        errors,
        representation_prompt_version=args.representation_prompt_version,
        decomposition_prompt_version=args.decomposition_prompt_version,
        model=args.model,
        max_rounds=args.max_rounds,
        plausible_rate=args.plausible_rate,
    )
    run_name = args.run_name or (
        f"{args.representation_prompt_version}__{args.decomposition_prompt_version}"
        f"_{int(time.time())}"
    )
    summary_path, per_image_path, representations_path = save(summary, results, run_name)

    print()
    print(format_summary(summary))
    print(f"\nwrote {summary_path}\nwrote {per_image_path}\nwrote {representations_path}")


if __name__ == "__main__":
    main()
