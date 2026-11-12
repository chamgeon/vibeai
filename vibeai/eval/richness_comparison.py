"""Pairwise richness comparison between representation methods.

Takes two or more batch-eval runs that decomposed the same images with a
structured (v2) decomposition - e.g. the plausibility runs for baseline, v1
and the workflow - and, for every pair of them and every shared image, asks
the richness judge (``metrics/richness.py``) which method's vibe set is
richer. Nothing is regenerated: the vibes are read straight out of the runs'
per-image records.

Each comparison is judged in both orders and the two verdicts are combined
(``metrics.richness.combine_verdicts``), so one comparison costs two calls.

    win rate = images where the system's set was judged richer / decisive images

Ties are counted and reported, but a tie says nothing about which method is
richer, so win rates and p-values are over the decisive (non-tie) images.

**Same image set for every pair.** As in ``eval/significance.py``, the images
are the intersection of *all* the runs, so each pair is judged on the same
images and the win rates are comparable across pairs.

**Sign test.** Under the null that two methods are equally rich, each decisive
image is a fair coin, so a pair's p is the exact two-sided binomial p of its
win count.

**Judge diagnostics.** ``consistency`` is the share of comparisons whose two
orders reached the same verdict - low consistency means many verdicts are
position-driven and get collapsed to ties. ``position_bias`` pools every
single-order call and asks how often Set A won among the decisive ones; both
orders are always judged, so this cannot favour a method, but it says how
lopsided the judge is.

Writes ``results/richness/<run>.json`` (summary) and ``.per_image.jsonl`` (one
comparison per line: both vibe sets, both orders' verdicts and full reasoning,
and the combined winner).

Usage:
    python -m vibeai.eval.richness_comparison plausibility \\
        baseline__v2_1790134860 v1__v2_1790132953 workflow__v2_1790134094
    python -m vibeai.eval.richness_comparison plausibility <run> <run> --labels v1 workflow
    python -m vibeai.eval.richness_comparison plausibility <run> <run> --n-images 10
"""

import argparse
import asyncio
import itertools
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from math import comb
from pathlib import Path

from vibeai.eval.concurrency import gather_bounded_as_completed
from vibeai.eval.prompt_results import RESULTS_DIR
from vibeai.metrics.richness import (
    DEFAULT_RICHNESS_MODEL,
    RichnessComparison,
    RichnessJudgement,
    vibe_set,
)

RICHNESS_RESULTS_DIR = RESULTS_DIR / "richness"

# Run-summary fields copied into the comparison record, so it says what each
# system was without opening each run's summary.
_RUN_FIELDS = (
    "representation_prompt_version",
    "decomposition_prompt_version",
    "representation_model",
    "decomposition_model",
)


def load_vibe_sets(metric: str, run: str) -> dict[str, list[str]]:
    """image_path -> vibe set, from a batch eval's per-image file."""
    src = RESULTS_DIR / metric / f"{run}.per_image.jsonl"
    if not src.exists():
        raise FileNotFoundError(src)
    sets = {}
    with src.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                sets[record["image_path"]] = vibe_set(record.get("details", {}))
            except ValueError as exc:
                raise ValueError(f"{src}: {record['image_path']}: {exc}") from exc
    return sets


def _run_summary(metric: str, run: str) -> dict:
    src = RESULTS_DIR / metric / f"{run}.json"
    if not src.exists():
        return {}
    summary = json.loads(src.read_text())
    return {field: summary.get(field) for field in _RUN_FIELDS}


def binomial_p(k: int, n: int) -> float:
    """Exact two-sided binomial p for k successes in n fair coin flips."""
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def format_p(p: float) -> str:
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"


def summarize(
    labels: list[str], judgements: list[RichnessJudgement]
) -> tuple[list[dict], dict[str, dict], dict]:
    """(per-pair stats, per-system totals, position-bias check)."""
    pairs = list(itertools.combinations(labels, 2))
    pair_records = []
    for first, second in pairs:
        in_pair = [j for j in judgements if {j.system_x, j.system_y} == {first, second}]
        wins_first = sum(j.winner == first for j in in_pair)
        wins_second = sum(j.winner == second for j in in_pair)
        n = wins_first + wins_second
        p = binomial_p(wins_first, n)
        pair_records.append({
            "a": first,
            "b": second,
            "n": len(in_pair),
            "n_decisive": n,
            "wins_a": wins_first,
            "wins_b": wins_second,
            "ties": len(in_pair) - n,
            "win_rate_a": wins_first / n if n else None,
            "p_value": p,
            "significant_bonferroni": p < 0.05 / len(pairs),
            "consistency": (
                sum(j.consistent for j in in_pair) / len(in_pair) if in_pair else None
            ),
        })

    systems = {}
    for label in labels:
        involved = [j for j in judgements if label in (j.system_x, j.system_y)]
        decisive = [j for j in involved if j.winner is not None]
        wins = sum(j.winner == label for j in decisive)
        systems[label] = {
            "wins": wins,
            "losses": len(decisive) - wins,
            "ties": len(involved) - len(decisive),
            "comparisons": len(involved),
            "win_rate": wins / len(decisive) if decisive else None,
        }

    calls = [order for j in judgements for order in j.orders]
    decisive_calls = [order for order in calls if order.winner is not None]
    wins_a = sum(order.winner_position == "A" for order in decisive_calls)
    position = {
        "n_calls": len(calls),
        "n_decisive": len(decisive_calls),
        "wins_a": wins_a,
        "win_rate_a": wins_a / len(decisive_calls) if decisive_calls else None,
        "p_value": binomial_p(wins_a, len(decisive_calls)),
        "consistency": sum(j.consistent for j in judgements) / len(judgements) if judgements else None,
    }
    return pair_records, systems, position


def format_summary(record: dict) -> str:
    lines = [
        f"judge: {record['model']}   images: {record['n_images']}   "
        f"comparisons: {record['n_comparisons']} (x2 orders)   errors: {len(record['errors'])}",
        "",
    ]
    for pair in record["pairs"]:
        if not pair["n"]:
            lines.append(f"  {pair['a']} vs {pair['b']}: no comparisons")
            continue
        line = (
            f"  {pair['a']} vs {pair['b']}: {pair['a']} {pair['wins_a']} - "
            f"{pair['wins_b']} {pair['b']}, {pair['ties']} tie(s)   "
            f"order-consistent {pair['consistency']:.0%}"
        )
        if pair["n_decisive"]:
            line += (
                f"\n    {pair['a']} win rate {pair['win_rate_a']:.1%} of decisive   "
                f"{format_p(pair['p_value'])}" + ("  *" if pair["significant_bonferroni"] else "")
            )
        lines.append(line)
    if len(record["pairs"]) > 1:
        lines.append(
            f"  (* significant at Bonferroni p < {record['bonferroni_threshold']:.4f})"
        )
    lines.append("")
    for label, stats in record["systems"].items():
        if stats["win_rate"] is not None:
            lines.append(
                f"  {label:<16} {stats['wins']}W {stats['losses']}L {stats['ties']}T "
                f"({stats['win_rate']:.1%} of decisive)"
            )
    position = record["position_bias"]
    if position["n_decisive"]:
        lines.append(
            f"\n  single calls: Set A won {position['wins_a']}/{position['n_decisive']} decisive "
            f"({position['win_rate_a']:.1%})   {format_p(position['p_value'])}"
        )
    return "\n".join(lines)


async def compare_runs(
    metric: str,
    runs: list[str],
    labels: list[str],
    *,
    model: str = DEFAULT_RICHNESS_MODEL,
    n_images: int | None = None,
    concurrency: int = 10,
) -> tuple[dict, list[RichnessJudgement]]:
    """``concurrency`` bounds comparisons in flight; each runs its two orders
    at once, so up to twice that many calls are open."""
    loaded = {label: load_vibe_sets(metric, run) for label, run in zip(labels, runs)}
    images = sorted(set.intersection(*(set(s) for s in loaded.values())))[:n_images]
    if not images:
        raise SystemExit("no images in common across the given runs")

    print(f"images compared: {len(images)}")
    for label, run in zip(labels, runs):
        dropped = len(loaded[label]) - len(images)
        note = f"  ({dropped} image(s) not compared)" if dropped else ""
        print(f"  {label:<16} {run}: {len(loaded[label])} image(s){note}")

    judge = RichnessComparison(model=model)
    tasks = [
        (image, first, second)
        for first, second in itertools.combinations(labels, 2)
        for image in images
    ]
    coros = [
        judge.compare_async(
            Path(image), first, loaded[first][image], second, loaded[second][image]
        )
        for image, first, second in tasks
    ]

    judgements = []
    errors = []
    async for index, outcome in gather_bounded_as_completed(coros, limit=concurrency):
        image, first, second = tasks[index]
        if isinstance(outcome, BaseException):
            print(
                f"  ! {Path(image).name} ({first} vs {second}): "
                f"{type(outcome).__name__}: {outcome}"
            )
            errors.append({
                "image_path": image,
                "systems": [first, second],
                "error_type": type(outcome).__name__,
                "error_message": str(outcome),
            })
            continue
        judgements.append(outcome)

    pairs, systems, position = summarize(labels, judgements)
    record = {
        "metric": "richness",
        "created_at": datetime.now(UTC).isoformat(),
        "model": model,
        "source_metric": metric,
        "systems_compared": [
            {"label": label, "run": run, **_run_summary(metric, run)}
            for label, run in zip(labels, runs)
        ],
        "orders": "both; agreement or win+tie is a win, conflict or two ties is a tie",
        "test": "exact two-sided binomial sign test over decisive comparisons",
        "n_images": len(images),
        "n_comparisons": len(judgements),
        "bonferroni_threshold": 0.05 / len(pairs),
        "pairs": pairs,
        "systems": systems,
        "position_bias": position,
        "errors": errors,
    }
    return record, judgements


def save(record: dict, judgements: list[RichnessJudgement], run_name: str) -> tuple[Path, Path]:
    RICHNESS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RICHNESS_RESULTS_DIR / f"{run_name}.json"
    summary_path.write_text(json.dumps(record, indent=2))

    per_image_path = RICHNESS_RESULTS_DIR / f"{run_name}.per_image.jsonl"
    ordered = sorted(judgements, key=lambda j: (j.system_x, j.system_y, j.image_path))
    with per_image_path.open("w") as f:
        for judgement in ordered:
            f.write(json.dumps({
                **asdict(judgement),
                "winner": judgement.winner,
                "consistent": judgement.consistent,
            }) + "\n")

    return summary_path, per_image_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "metric", help="results/<metric>/ directory the runs live in, e.g. plausibility"
    )
    parser.add_argument("runs", nargs="+", help="two or more run names to compare pairwise")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="system name for each run, in order (default: each run's representation prompt version)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_RICHNESS_MODEL,
        help=f"Judge model (default: {DEFAULT_RICHNESS_MODEL})",
    )
    parser.add_argument(
        "--n-images", type=int, default=None, help="Compare only the first N shared images"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max comparisons in flight (default: 10); each makes two calls at once",
    )
    parser.add_argument(
        "--run-name", default=None, help="Output file stem (default: <labels>_<unix time>)"
    )
    args = parser.parse_args()

    if len(args.runs) < 2:
        parser.error("need at least two runs to compare")
    labels = args.labels or [
        _run_summary(args.metric, run).get("representation_prompt_version") or run
        for run in args.runs
    ]
    if len(labels) != len(args.runs):
        parser.error(f"--labels has {len(labels)} name(s) for {len(args.runs)} run(s)")
    if len(set(labels)) != len(labels):
        parser.error(f"system labels must be unique, got {labels}; pass --labels")

    record, judgements = asyncio.run(
        compare_runs(
            args.metric,
            args.runs,
            labels,
            model=args.model,
            n_images=args.n_images,
            concurrency=args.concurrency,
        )
    )
    run_name = args.run_name or f"{'+'.join(labels)}_{int(time.time())}"
    summary_path, per_image_path = save(record, judgements, run_name)

    print()
    print(format_summary(record))
    print(f"\nwrote {summary_path}\nwrote {per_image_path}")


if __name__ == "__main__":
    main()
