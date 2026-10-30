"""Paired permutation test between metric runs, so a gap between two prompt
versions can be called real or not.

The batch evals report one mean per run. Two means that differ say nothing on
their own: plausibility scores vary a lot image to image (std ~0.12 against a
0.04 gap), so a difference smaller than that spread can easily be which images
happened to land where. This answers the only question worth asking of two
runs - could chance alone have produced a gap this big?

    score = mean over images of (score_a - score_b)
    p     = share of random sign-flips whose mean gap is at least as extreme

**Paired.** Every run scores the same image set, so the test works on per-image
differences rather than on the two means. That cancels image difficulty: an
image every prompt version finds hard contributes a difference near zero
instead of adding its variance to both sides. Unpaired, the same comparison is
noisy enough to miss effects this size.

**Permutation, not a t-test.** A score is a fraction over ~11 atoms, so the
values are discrete, bounded in [0, 1] and piled up near 1.0 - nothing a
t-test's normality assumption fits. A permutation test assumes only that under
the null the two arms are interchangeable on each image, which is exactly what
is being tested: if the prompt version did nothing, each per-image difference
was as likely to come out with the opposite sign. So the null distribution is
built by flipping those signs at random and re-taking the mean.

``p = (count + 1) / (n_resamples + 1)`` - the +1s are the standard correction.
They keep p away from exactly 0, which 20k resamples cannot justify claiming;
the smallest p this reports is 1/20001.

What p is: the probability of a gap this big *if the two arms were
equivalent*. What it is not: the probability that they are equivalent, and not
a statement about effect size either. ``observed`` is the effect - read both.

Usage:
    python -m vibeai.eval.significance plausibility v1__v2_1790132953 workflow__v2_1790134094
    python -m vibeai.eval.significance plausibility baseline__x v1__y workflow__z
    python -m vibeai.eval.significance richness baseline__x v1__y --resamples 100000
"""

import argparse
import itertools
import json
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
SIGNIFICANCE_DIR = RESULTS_ROOT / "significance"

# Run-summary fields copied into a saved comparison, so the record says what
# was compared without opening each run's summary.
_RUN_FIELDS = (
    "representation_prompt_version",
    "decomposition_prompt_version",
    "representation_model",
    "decomposition_model",
    "model",
    "pool",
    "timestamp",
)

DEFAULT_RESAMPLES = 20_000
DEFAULT_SEED = 0


def load_scores(metric: str, run: str) -> dict[str, float]:
    """image_path -> score, from a batch eval's per-image file.

    Images the run errored on are absent from that file rather than present
    with a null score, so they drop out of the comparison by not being in the
    intersection - which is what should happen to them.
    """
    src = RESULTS_ROOT / metric / f"{run}.per_image.jsonl"
    if not src.exists():
        raise FileNotFoundError(src)
    scores = {}
    with src.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("score") is None:
                continue
            scores[record["image_path"]] = float(record["score"])
    return scores


@dataclass
class PermutationResult:
    """One pairwise comparison. ``observed`` is a - b, so a positive number
    means the first run scored higher."""

    observed: float
    p_value: float
    n: int
    n_resamples: int
    null_sd: float
    wins: int
    losses: int
    ties: int


def permutation_test(
    a: list[float],
    b: list[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> PermutationResult:
    """Two-sided paired sign-flip test over already-aligned score lists.

    ``a[i]`` and ``b[i]`` must be the two runs' scores for the same image. The
    seed is fixed so a reported p reproduces exactly; changing it moves p by a
    few ten-thousandths, which is the resampling noise this many resamples
    leaves and not something to tune.
    """
    if len(a) != len(b):
        raise ValueError(f"unaligned score lists: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("no images in common")

    differences = [x - y for x, y in zip(a, b)]
    observed = mean(differences)

    rng = random.Random(seed)
    null = [
        mean(d * rng.choice((1, -1)) for d in differences) for _ in range(n_resamples)
    ]
    # Two-sided: a gap is "as extreme" whichever direction it points, so the
    # comparison is on absolute value. Testing one-sided after seeing which
    # run won would halve p for free.
    at_least_as_extreme = sum(abs(x) >= abs(observed) for x in null)

    return PermutationResult(
        observed=observed,
        p_value=(at_least_as_extreme + 1) / (n_resamples + 1),
        n=len(differences),
        n_resamples=n_resamples,
        null_sd=pstdev(null),
        wins=sum(d > 0 for d in differences),
        losses=sum(d < 0 for d in differences),
        ties=sum(d == 0 for d in differences),
    )


def format_p(p: float) -> str:
    """``p = 0.0123``, or ``p < 0.0001`` once four decimals would round it to
    zero. The floor is 1/(n_resamples + 1), never 0, so printing 0.0000 would
    claim more than the test can."""
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"


def _run_summary(metric: str, run: str) -> dict:
    src = RESULTS_ROOT / metric / f"{run}.json"
    if not src.exists():
        return {}
    summary = json.loads(src.read_text())
    return {field: summary.get(field) for field in _RUN_FIELDS}


def save_comparison(record: dict, out_dir: Path = SIGNIFICANCE_DIR) -> Path:
    """Write one comparison to ``<metric>__<run>+<run>....json``. Named by its
    inputs, so re-running the same comparison overwrites it with the same
    numbers (the seed is fixed) instead of piling up copies."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{record['metric']}__{'+'.join(record['runs'])}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def compare(
    metric: str,
    runs: list[str],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Every pairwise comparison among ``runs``, all on one image set. Prints
    the comparison and returns it as a record for ``save_comparison``.

    The image set is the intersection of *all* the runs, not of each pair. A
    per-pair intersection would let each comparison run on a different set of
    images, and the means in the first table would then not be the means the
    tests were computed from.
    """
    loaded = {run: load_scores(metric, run) for run in runs}
    images = sorted(set.intersection(*(set(s) for s in loaded.values())))
    if not images:
        raise SystemExit("no images in common across the given runs")

    print(f"metric: {metric}")
    print(f"images compared: {len(images)}")
    for run, scores in loaded.items():
        dropped = len(scores) - len(images)
        note = f"  ({dropped} image(s) not shared by every run)" if dropped else ""
        print(f"  {run}: scored {len(scores)}{note}")

    print()
    run_records = []
    for run in runs:
        values = [loaded[run][image] for image in images]
        run_records.append({
            "run": run,
            **_run_summary(metric, run),
            "n_scored": len(loaded[run]),
            "mean": mean(values),
            "pass_rate_0.7": mean(v >= 0.7 for v in values),
        })
        print(f"  {run:<32} mean {mean(values):.4f}  pass@0.7 {mean(v >= 0.7 for v in values):.3f}")

    print()
    pairs = list(itertools.combinations(runs, 2))
    bonferroni = 0.05 / len(pairs)
    comparisons = []
    for first, second in pairs:
        result = permutation_test(
            [loaded[first][image] for image in images],
            [loaded[second][image] for image in images],
            n_resamples=n_resamples,
            seed=seed,
        )
        print(
            f"  {first} vs {second}\n"
            f"    diff {result.observed:+.4f}   {format_p(result.p_value)}   "
            f"(null sd {result.null_sd:.4f}, {result.n_resamples} resamples)\n"
            f"    {first} higher on {result.wins}, lower on {result.losses}, tied on {result.ties}"
        )
        comparisons.append({
            "a": first,
            "b": second,
            **asdict(result),
            "significant_bonferroni": result.p_value < bonferroni,
        })

    # Several tests on one dataset inflate the chance that one of them clears
    # 0.05 by luck. Bonferroni is the blunt correction, and blunt is fine when
    # the point is only to keep a borderline p from being read as a result.
    if len(pairs) > 1:
        print(
            f"\n  {len(pairs)} comparisons on one dataset - Bonferroni threshold "
            f"for 0.05 overall is p < {bonferroni:.4f}"
        )

    return {
        "metric": metric,
        "created_at": datetime.now(UTC).isoformat(),
        "test": "two-sided paired sign-flip Monte Carlo permutation test",
        "n_resamples": n_resamples,
        "seed": seed,
        "runs": runs,
        "n_images": len(images),
        "images": images,
        "run_stats": run_records,
        "comparisons": comparisons,
        "bonferroni_threshold": bonferroni,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metric", help="results/<metric>/ directory, e.g. plausibility or richness"
    )
    parser.add_argument(
        "runs", nargs="+", help="two or more run names to compare pairwise"
    )
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if len(args.runs) < 2:
        parser.error("need at least two runs to compare")
    record = compare(args.metric, args.runs, n_resamples=args.resamples, seed=args.seed)
    print(f"\nwrote {save_comparison(record)}")


if __name__ == "__main__":
    main()
