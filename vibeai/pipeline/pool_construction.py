"""Vibe-pool construction: build a per-image pool of vibes for a holdout set.

The pool is the reference target a richness metric scores against - "how much
of what is really there did the representation capture" - so it has to be
broader than any single representation. Breadth comes from running several
independent representation branches over the same image and unioning what
survives:

    for each image:
        for each (representation model x representation prompt) branch:
            represent -> decompose (v2) -> plausibility judge
            keep the vibes whose direct_check passed
        pool = union of the kept vibes

Two things make this a pool rather than just a merged decomposition:

*   **Only ``direct_check`` gates survival**, not the judge's ``final_verdict``.
    ``direct_check`` asks whether the vibe itself is inferable from the image,
    independent of the evidence the atom happened to cite. An evidence_backed
    atom that fails ``mapping_check`` (its evidence was real and the vibe was
    inferable, but the evidence wasn't the main driver) has a bad *pairing* -
    the vibe is still a true thing about the image, so it belongs in the pool
    even though the atom as a whole was judged implausible. Atoms whose
    ``evidence_presence_check`` failed are excluded by the same rule without a
    special case: the judge stops early there, leaving ``direct_check`` null,
    so the vibe was never actually verified against the image.
*   **Evidence is dropped.** Pool entries are bare vibes. Evidence is a
    property of one representation's phrasing, not of the image, and a
    richness metric asks which vibes were captured - not whether they were
    justified the same way.

The number of branches a vibe was found by is kept as ``support``: a vibe all
four branches independently arrived at is a different kind of pool member than
one a single branch mentioned once.

A final merge pass collapses *lexical* variants of the same vibe word
("monumental" / "monumental feel" / "monumental scale" / "quietly monumental").
This is not a redundancy judgement, and deliberately so. Semantic redundancy is
decided at scoring time by the witness test in ``prompts/richness_eval.py``,
against the target set - which the pool cannot see. What lexical duplicates
actually break is *weighting*: four spellings of one vibe all receive the same
witness-test verdict, so that vibe counts four times in the coverage fraction.
Merging them fixes the weighting without pre-empting any judgement about
whether "grandeur" and "monumental" are the same vibe. They stay separate here;
the witness test decides.
"""

import argparse
import asyncio
import json
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from vibeai.eval.concurrency import gather_bounded, gather_bounded_as_completed
from vibeai.eval.dataset import load_image_paths
from vibeai.eval.parsing import extract_json
from vibeai.llm.client import (
    ANTHROPIC_EFFORT,
    ANTHROPIC_EFFORT_LEVELS,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_DECOMPOSITION_MODEL,
    DEFAULT_EVAL_MODEL,
    DEFAULT_MODEL,
    call_text_async,
    set_anthropic_effort,
)
from vibeai.metrics.plausibility import PlausibilityMetric
from vibeai.pipeline.evaluate import evaluate_image
from vibeai.prompts.pool_merge import POOL_MERGE_PROMPT

DEFAULT_IMAGE_DIR = Path("data/richness_holdout")
# Pools are hand-checkable reference data, not run output - they live outside
# the gitignored data/ and results/ trees so they're tracked.
DEFAULT_OUTPUT_DIR = Path("annotations")

# The branch axes: one representation prompt x model combination each. Varying
# the provider guards against one model's idiosyncratic read of an image
# becoming the ground truth.
DEFAULT_MODELS = (DEFAULT_MODEL, DEFAULT_ANTHROPIC_MODEL)
DEFAULT_REPRESENTATION_PROMPTS = ("baseline", "v1")

# Decomposition and judging are held fixed across branches so the only thing
# that varies is how the vibe was *seen*, not how it was cut up or scored.
POOL_DECOMPOSITION_PROMPT = "v2"


@dataclass(frozen=True)
class PoolBranch:
    """One (representation model, representation prompt) combination."""

    model: str
    representation_prompt: str

    @property
    def name(self) -> str:
        return f"{self.model}__{self.representation_prompt}"


def build_branches(
    models: list[str], representation_prompts: list[str]
) -> list[PoolBranch]:
    return [
        PoolBranch(model=model, representation_prompt=prompt)
        for model in models
        for prompt in representation_prompts
    ]


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_vibe(vibe: str) -> str:
    """Dedup key for a vibe. Only collapses surface variation (case, spacing,
    trailing punctuation) - "cozy" and "intimate" stay separate entries,
    because deciding those are the same vibe is a judgement call that belongs
    to whoever reviews the pool, not to a string comparison."""
    return _WHITESPACE_RE.sub(" ", vibe.strip().lower()).rstrip(".!,;:")


def _pool_vibe(judged_atom: dict, index: int, atom_details: list[dict] | None) -> str:
    """The wording a surviving atom contributes to the pool.

    Two components independently name the vibe inside an atom: the v2
    decomposer, which emits it as a dedicated ``vibe`` field, and the
    plausibility judge, which re-derives it as ``stated_vibe`` on the way to
    judging. The decomposer's is preferred - naming the vibe is its actual
    task, while the judge's is a by-product - and in practice the judge's
    drifts toward place nouns ("an atrium") and predicate clauses ("soften the
    otherwise stark, industrial materials") that make poor pool entries.

    The two are matched by atom text, falling back to position. Note this does
    loosen one thing: ``direct_check`` was run against ``stated_vibe``, so the
    text kept here is not verbatim what the judge ruled on. They describe the
    same atom, but a verdict is being carried across a rewording.
    """
    stated = str(judged_atom.get("stated_vibe", "")).strip()
    if not atom_details:
        return stated

    atom_text = str(judged_atom.get("atom", "")).strip()
    detail = next(
        (d for d in atom_details if str(d.get("atom", "")).strip() == atom_text), None
    )
    if detail is None and index < len(atom_details):
        detail = atom_details[index]
    if detail is None:
        return stated

    return str(detail.get("vibe", "")).strip() or stated


def surviving_vibes(
    judged_atoms: list[dict], atom_details: list[dict] | None = None
) -> list[str]:
    """The vibes of atoms whose ``direct_check`` passed.

    ``direct_check`` is null when the judge stopped early (evidence_backed
    atoms that failed ``evidence_presence_check``), which excludes them - the
    vibe was never checked against the image on its own terms.

    ``atom_details`` is the v2 decomposer's atom objects; without it (e.g. a
    decomposition prompt that emits plain sentences) the judge's
    ``stated_vibe`` is used instead.
    """
    vibes = []
    for index, atom in enumerate(judged_atoms):
        direct_check = atom.get("direct_check")
        if not isinstance(direct_check, dict) or direct_check.get("verdict") is not True:
            continue
        vibe = _pool_vibe(atom, index, atom_details)
        if vibe:
            vibes.append(vibe)
    return vibes


async def run_branch(
    image_path: Path,
    branch: PoolBranch,
    metric: PlausibilityMetric,
    decomposition_model: str,
) -> list[str]:
    """represent -> decompose -> judge for one image/branch, returning the
    vibes that survived."""
    test_case, result = await evaluate_image(
        image_path,
        metric,
        representation_prompt_version=branch.representation_prompt,
        decomposition_prompt_version=POOL_DECOMPOSITION_PROMPT,
        representation_model=branch.model,
        decomposition_model=decomposition_model,
    )
    return surviving_vibes(result.details["atoms"], test_case.atom_details)


def _new_record(image_path: Path, branches: list[PoolBranch]) -> dict:
    return {
        "image_path": str(image_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "branches": [branch.name for branch in branches],
        "vibes": [],
        "errors": [],
    }


def _finalize(record: dict, pooled: OrderedDict[str, dict]) -> dict:
    """Settle each entry's display form and ordering, then order the pool by
    how many branches found each vibe (most agreed-on first) and alphabetically.

    Everything decided here is deliberately independent of the order branches
    happened to finish in: a pool is a tracked, hand-reviewed file, and a rerun
    that shuffles ``found_by`` or swaps "Cozy." for "cozy" would show up as a
    diff that means nothing.
    """
    vibes = []
    for entry in pooled.values():
        forms = Counter(entry.pop("forms"))
        # Most-used surface form wins; ties broken alphabetically.
        entry["vibe"] = min(forms.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        entry["found_by"].sort()
        vibes.append(entry)

    record["vibes"] = sorted(vibes, key=lambda v: (-v["support"], v["vibe"].lower()))
    return record


def _build_merge_prompt(vibes: list[str]) -> str:
    return POOL_MERGE_PROMPT.format(
        pool="\n".join(f"{i + 1}. {vibe}" for i, vibe in enumerate(vibes))
    )


def _validate_merge(raw: str, expected: list[str]) -> list[dict]:
    """Parse + validate the merger's JSON. Raises ValueError on any failure -
    used both as call_text_async's retry-triggering ``validate`` callback and
    to build the merged pool once a call has passed.

    The strict partition check is the point: the merger is only allowed to
    *group* entries, so anything dropped, invented, reworded, or listed twice
    means it did something other than the task, and the pool would silently
    lose or duplicate a vibe.
    """
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got: {raw!r}")

    groups = parsed.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"'groups' missing or not a non-empty list: {raw!r}")

    seen: list[str] = []
    for i, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"groups[{i}] is not an object: {group!r}")
        vibe_word = group.get("vibe_word")
        if not isinstance(vibe_word, str) or not vibe_word.strip():
            raise ValueError(f"groups[{i}] has empty/invalid vibe_word: {group!r}")
        members = group.get("members")
        if not isinstance(members, list) or not members:
            raise ValueError(f"groups[{i}] members is not a non-empty list: {group!r}")
        for member in members:
            if not isinstance(member, str):
                raise ValueError(f"groups[{i}] member is not a string: {member!r}")
            seen.append(member)

    if sorted(seen) != sorted(expected):
        missing = sorted(set(expected) - set(seen))
        extra = sorted(set(seen) - set(expected))
        duplicated = sorted({v for v in seen if seen.count(v) > 1})
        raise ValueError(
            "Groups are not a partition of the input pool "
            f"(missing={missing}, not-in-pool={extra}, duplicated={duplicated}): {raw!r}"
        )
    return groups


async def merge_vibe_groups_async(vibes: list[str], model: str) -> list[dict]:
    """Group pool entries that are lexical variants of the same vibe word."""
    expected = list(vibes)
    raw = await call_text_async(
        _build_merge_prompt(vibes),
        model=model,
        call_type="pool_merge",
        validate=lambda text: _validate_merge(text, expected),
    )
    return _validate_merge(raw, expected)


async def merge_record(record: dict, model: str) -> dict:
    """Rewrite one image's ``vibes`` in place with the merged groups. Support is
    recomputed over the union of each group's members, so a vibe two branches
    phrased differently is finally counted as two branches agreeing."""
    entries = {entry["vibe"]: entry for entry in record["vibes"]}
    if len(entries) < 2:
        return record

    groups = await merge_vibe_groups_async(list(entries), model)

    merged = []
    for group in groups:
        members = sorted(group["members"], key=str.lower)
        branches = sorted({b for m in members for b in entries[m]["found_by"]})
        # The surviving wording is always one the representations actually used
        # - the shortest member, which is the one carrying the least incidental
        # scaffolding. Letting the merger write this field instead turned
        # "a faint reminder of an outside world kept gently at bay" into
        # "reminder" and "cool tones" into "cool": text no representation ever
        # produced, which a partition check cannot catch.
        entry = {
            "vibe": min(members, key=lambda m: (len(m), m.lower())),
            "vibe_word": group["vibe_word"].strip(),
            "support": len(branches),
            "found_by": branches,
        }
        # Keep the raw wording whenever the entry isn't just itself - it's what
        # makes a merge reviewable after the fact.
        if members != [entry["vibe"]]:
            entry["merged_from"] = members
        merged.append(entry)

    record["vibes"] = sorted(merged, key=lambda v: (-v["support"], v["vibe"].lower()))
    return record


async def merge_pool(
    pool: dict[str, dict], model: str, concurrency: int = 10, verbose: bool = True
) -> dict[str, dict]:
    """Merge every image's pool. One failure leaves that image unmerged (and
    says so on its ``errors``) rather than losing the raw pool."""
    keys = list(pool)
    outcomes = await gather_bounded(
        [merge_record(pool[key], model) for key in keys],
        limit=concurrency,
        return_exceptions=True,
    )
    for key, outcome in zip(keys, outcomes):
        if isinstance(outcome, BaseException):
            pool[key]["errors"].append({"branch": "merge", "error": repr(outcome)})
            if verbose:
                print(f"merge {key}: FAILED {outcome!r} (left unmerged)")
    return pool


async def build_pool(
    image_paths: list[Path],
    branches: list[PoolBranch],
    eval_model: str = DEFAULT_EVAL_MODEL,
    decomposition_model: str = DEFAULT_DECOMPOSITION_MODEL,
    merge_model: str | None = DEFAULT_EVAL_MODEL,
    concurrency: int = 10,
    verbose: bool = True,
) -> dict[str, dict]:
    """Build the pool for every image. All image x branch runs share one
    concurrency budget, so a slow branch doesn't stall the others; a branch
    that fails is recorded on the image's ``errors`` and the rest of its pool
    is still written. ``merge_model=None`` skips the lexical merge and returns
    the raw pool."""
    metric = PlausibilityMetric(model=eval_model)

    jobs = [(image_path, branch) for image_path in image_paths for branch in branches]
    coros = [
        run_branch(image_path, branch, metric, decomposition_model)
        for image_path, branch in jobs
    ]

    pool = {str(p): _new_record(p, branches) for p in image_paths}
    pooled: dict[str, OrderedDict[str, dict]] = {str(p): OrderedDict() for p in image_paths}

    done = 0
    async for index, outcome in gather_bounded_as_completed(coros, limit=concurrency):
        image_path, branch = jobs[index]
        key = str(image_path)
        done += 1

        if isinstance(outcome, BaseException):
            pool[key]["errors"].append({"branch": branch.name, "error": repr(outcome)})
            if verbose:
                print(f"[{done}/{len(jobs)}] {key} {branch.name}: FAILED {outcome!r}")
            continue

        for vibe in outcome:
            entry = pooled[key].setdefault(
                normalize_vibe(vibe), {"vibe": vibe, "support": 0, "found_by": [], "forms": []}
            )
            entry["forms"].append(vibe.strip())
            # One branch can state the same vibe in several atoms; support
            # counts distinct branches, not mentions.
            if branch.name not in entry["found_by"]:
                entry["found_by"].append(branch.name)
                entry["support"] += 1
        if verbose:
            print(f"[{done}/{len(jobs)}] {key} {branch.name}: {len(outcome)} vibe(s)")

    pool = {key: _finalize(record, pooled[key]) for key, record in pool.items()}
    if merge_model is not None:
        pool = await merge_pool(pool, merge_model, concurrency=concurrency, verbose=verbose)
    return pool


def save_pool(pool: dict[str, dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool, indent=2, ensure_ascii=False))
    return path


def summarize(pool: dict[str, dict], branches: list[PoolBranch]) -> str:
    sizes = [len(record["vibes"]) for record in pool.values()]
    n_errors = sum(len(record["errors"]) for record in pool.values())
    unanimous = sum(
        1
        for record in pool.values()
        for vibe in record["vibes"]
        if vibe["support"] == len(branches)
    )
    total = sum(sizes)
    mean = total / len(sizes) if sizes else 0.0
    n_merged = sum(
        1 for record in pool.values() for vibe in record["vibes"] if "merged_from" in vibe
    )
    return (
        f"{len(pool)} image(s), {len(branches)} branch(es)\n"
        f"entries built from >1 raw wording: {n_merged}\n"
        f"pooled vibes: {total} total, {mean:.1f} per image "
        f"(min {min(sizes, default=0)}, max {max(sizes, default=0)})\n"
        f"found by all {len(branches)} branches: {unanimous}\n"
        f"failed branch runs: {n_errors}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a per-image vibe pool for a holdout image set."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Directory of images to pool (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: annotations/<image-dir name>.vibes.json)",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=None,
        help="Pool only the first N images (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help=f"Representation models, one branch each (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--representation-prompts",
        nargs="+",
        default=list(DEFAULT_REPRESENTATION_PROMPTS),
        help=f"Representation prompt versions (default: {' '.join(DEFAULT_REPRESENTATION_PROMPTS)})",
    )
    parser.add_argument(
        "--decomposition-model",
        default=DEFAULT_DECOMPOSITION_MODEL,
        help="Model for the v2 decomposition step, held fixed across branches "
        f"(default: {DEFAULT_DECOMPOSITION_MODEL}). NOTE: annotations/richness_holdout.vibes.json "
        f"was built with {DEFAULT_MODEL}; rebuilding with a different decomposer changes the pool.",
    )
    parser.add_argument(
        "--eval-model",
        default=DEFAULT_EVAL_MODEL,
        help=f"Plausibility judge model (default: {DEFAULT_EVAL_MODEL})",
    )
    parser.add_argument(
        "--merge-model",
        default=DEFAULT_EVAL_MODEL,
        help=f"Model for the lexical merge pass (default: {DEFAULT_EVAL_MODEL})",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip the lexical merge and write the raw pooled vibes",
    )
    parser.add_argument(
        "--anthropic-effort",
        choices=ANTHROPIC_EFFORT_LEVELS,
        # Inherits the client's default rather than hardcoding one, so omitting
        # the flag leaves the setting alone instead of resetting it to None.
        default=ANTHROPIC_EFFORT,
        help=f"How hard Claude thinks (default: {ANTHROPIC_EFFORT}). "
        "Lower values also shorten the representation, so this changes what lands in the pool.",
    )
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    # Set before any call: effort is part of the Anthropic cache key, so this
    # has to be in place before the first lookup or the run reads entries
    # cached at a different effort.
    set_anthropic_effort(args.anthropic_effort)

    image_paths = load_image_paths(n=args.n_images, data_dir=args.image_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {args.image_dir}")

    branches = build_branches(args.models, args.representation_prompts)
    out_path = args.out or DEFAULT_OUTPUT_DIR / f"{args.image_dir.name}.vibes.json"

    pool = asyncio.run(
        build_pool(
            image_paths,
            branches,
            eval_model=args.eval_model,
            decomposition_model=args.decomposition_model,
            merge_model=None if args.no_merge else args.merge_model,
            concurrency=args.concurrency,
        )
    )

    saved = save_pool(pool, out_path)
    print()
    print(summarize(pool, branches))
    print(f"wrote {saved}")


if __name__ == "__main__":
    main()
