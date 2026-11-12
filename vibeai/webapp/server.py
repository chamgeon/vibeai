"""FastAPI backend for the human annotation / results-viewer webapps.

Serves the same (representation, atoms) pairs that were fed to an
LLM-as-a-judge metric, lets a human rate them with the same rubric (minus
free-text reasons for decomposition quality), and stores results keyed by
``image_path`` + run so they can be paired 1:1 with
``results/<metric>/<run>.per_image.jsonl`` for Cohen's kappa alignment
analysis. The human never sees the LLM's verdicts by default, to avoid
anchoring bias.

Also serves read-only results viewers (static/results.html + results.js,
static/plausibility_results.html + plausibility_results.js) for browsing a run's
images, representations, decompositions, and the LLM judge's
verdicts/reasoning directly — no annotation involved.

Every metric (decomposition_quality, plausibility, and any added
later — interpretability, ...) shares the same run-discovery, dataset,
LLM-judgement, human-annotation-store, and progress plumbing below,
parameterized by ``metric``. Adding a new metric only requires:
  1. a ``BLIND_ATOM_FNS[metric]`` entry, if its atoms need to be stripped of
     judge-only fields before being shown to an annotator (skip if atoms are
     already blind, e.g. plain strings)
  2. an ``EXTRA_DATASET_FNS[metric]`` entry, if the thing being rated isn't
     the atom list itself (skip if the atoms are what gets rated)
  3. a Pydantic annotation-input model + POST ``/api/<metric>/annotations``
     handler encoding that metric's rubric/scoring — this part is
     irreducibly metric-specific, since every metric's rubric differs.

Richness (static/richness.html + richness.js) is the exception: its unit is a
pairwise comparison (image x pair of systems), not an image, so it has its
own routes, keyed by a comparison id instead of ``image_path``.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
# Any image under data/ is servable, not just the main pool, so runs over a
# separate image set (e.g. data/richness_holdout) can be shown too. The
# containment check in /api/image is what keeps this from serving the rest of
# the repo.
DATA_ROOT = REPO_ROOT / "data"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Vibe Eval — Human Annotation")


# --- per-metric dataset plumbing (shared by every metric) ----------------


def _results_dir(metric: str) -> Path:
    return RESULTS_ROOT / metric


def _human_dir(metric: str) -> Path:
    return _results_dir(metric) / "human"


def _list_runs(metric: str) -> list[str]:
    d = _results_dir(metric)
    if not d.exists():
        return []
    return sorted(p.name.removesuffix(".per_image.jsonl") for p in d.glob("*.per_image.jsonl"))


def _load_per_image_records(metric: str, run: str) -> list[dict]:
    src = _results_dir(metric) / f"{run}.per_image.jsonl"
    if not src.exists():
        raise HTTPException(404, f"unknown run: {run}")
    records = []
    with src.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_llm_details(metric: str, run: str) -> dict[str, dict]:
    return {
        rec["image_path"]: {
            **rec.get("details", {}),
            "score": rec.get("score"),
            "passed": rec.get("passed"),
        }
        for rec in _load_per_image_records(metric, run)
    }


# Per-atom view shown to annotators before they've rated an image, so they
# aren't anchored by the LLM judge's own verdicts/scores/reasoning. Metrics
# whose atoms are already judge-free (e.g. decomposition_quality's plain
# atom strings) don't need an entry — identity is the default.
BLIND_ATOM_FNS: dict[str, Callable[[Any], Any]] = {
    "plausibility": lambda atom: {
        "atom": atom["atom"],
        "type": atom["type"],
        "stated_vibe": atom.get("stated_vibe"),
        "stated_evidence": atom.get("stated_evidence"),
    },
}


# Extra per-image fields a metric's annotator needs beyond (representation,
# atoms) - stripped of the judge's own verdicts, for the same anti-anchoring
# reason as BLIND_ATOM_FNS. No current metric needs one.
EXTRA_DATASET_FNS: dict[str, Callable[[dict], dict]] = {}


def _load_dataset(metric: str, run: str) -> list[dict]:
    blind = BLIND_ATOM_FNS.get(metric, lambda atom: atom)
    items = []
    for rec in _load_per_image_records(metric, run):
        details = rec.get("details", {})
        items.append(
            {
                "image_path": rec["image_path"],
                "representation": details.get("representation", ""),
                "atoms": [blind(a) for a in details.get("atoms", [])],
                **EXTRA_DATASET_FNS.get(metric, lambda _details: {})(details),
            }
        )
    return items


# --- per-metric human annotation store (shared by every metric) ----------


def _human_path(metric: str, run: str, annotator: str) -> Path:
    hd = _human_dir(metric)
    hd.mkdir(parents=True, exist_ok=True)
    safe_annotator = "".join(c if c.isalnum() or c in "-_." else "_" for c in annotator)
    if not safe_annotator:
        raise HTTPException(400, "invalid annotator id")
    return hd / f"{run}__{safe_annotator}.json"


def _load_human(metric: str, run: str, annotator: str) -> dict[str, dict]:
    path = _human_path(metric, run, annotator)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_human(metric: str, run: str, annotator: str, data: dict[str, dict]) -> None:
    path = _human_path(metric, run, annotator)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _dataset_image_paths(metric: str, run: str) -> set[str]:
    return {item["image_path"] for item in _load_dataset(metric, run)}


# --- richness: pairwise comparison annotation -----------------------------
#
# Registered before the generic /api/{metric}/... routes below would be, but
# "richness" is not in KNOWN_METRICS anyway, so they never collide.
#
# One item is one comparison from results/richness/<run>.per_image.jsonl. The
# annotator sees the two vibe sets as "Set 1" and "Set 2" with no system names;
# which system is Set 1 is a stable hash of the comparison id, so it is random
# across items but the same every time an item is opened.


def _comparison_id(record: dict) -> str:
    return f"{record['system_x']}|{record['system_y']}|{record['image_path']}"


def _set_1_system(comparison_id: str, system_x: str, system_y: str) -> str:
    digest = hashlib.sha256(comparison_id.encode()).digest()
    return system_x if digest[0] % 2 == 0 else system_y


def _richness_records(run: str) -> dict[str, dict]:
    return {_comparison_id(rec): rec for rec in _load_per_image_records("richness", run)}


@app.get("/api/richness/runs")
def list_richness_runs():
    return {"runs": _list_runs("richness")}


@app.get("/api/richness/dataset")
def get_richness_dataset(run: str):
    items = []
    for comparison_id, rec in _richness_records(run).items():
        set_1 = _set_1_system(comparison_id, rec["system_x"], rec["system_y"])
        x_is_1 = set_1 == rec["system_x"]
        items.append({
            "comparison_id": comparison_id,
            "image_path": rec["image_path"],
            # The pair, but not which side is which - fine to show, since it
            # doesn't say which set came from which system.
            "pair": sorted([rec["system_x"], rec["system_y"]]),
            "set_1": rec["vibes_x"] if x_is_1 else rec["vibes_y"],
            "set_2": rec["vibes_y"] if x_is_1 else rec["vibes_x"],
        })
    return {"items": items}


def _set_label(system: str | None, set_1: str) -> str:
    if system is None:
        return "tie"
    return "1" if system == set_1 else "2"


@app.get("/api/richness/llm_judgement")
def get_richness_llm_judgement(run: str, comparison_id: str):
    """Both orders' verdicts and reasoning, translated into the annotator's
    Set 1 / Set 2 frame, plus the system names behind the sets."""
    rec = _richness_records(run).get(comparison_id)
    if rec is None:
        raise HTTPException(404, "no LLM judgement for this comparison in this run")
    set_1 = _set_1_system(comparison_id, rec["system_x"], rec["system_y"])

    def order(o: dict) -> dict:
        shown_as_a = _set_label(o["system_a"], set_1)
        pos = o["winner_position"]
        winner = None if pos == "tie" else o["system_a"] if pos == "A" else o["system_b"]
        j = o.get("judgement", {})
        return {
            "set_shown_as_a": shown_as_a,
            "winner": _set_label(winner, set_1),
            "image_facets": j.get("image_facets", []),
            # Keyed by the annotator's set numbers, not the judge's A/B.
            "analysis": {
                shown_as_a: j.get("set_a_analysis", ""),
                ("2" if shown_as_a == "1" else "1"): j.get("set_b_analysis", ""),
            },
            "comparison": j.get("comparison", ""),
        }

    return {
        "winner": _set_label(rec["winner"], set_1),
        "consistent": rec["consistent"],
        "systems": {"1": set_1, "2": rec["system_y"] if set_1 == rec["system_x"] else rec["system_x"]},
        "orders": [order(rec["x_first"]), order(rec["y_first"])],
    }


@app.get("/api/richness/annotations")
def get_richness_annotations(run: str, annotator: str):
    return {"annotations": _load_human("richness", run, annotator)}


class RichnessAnnotationIn(BaseModel):
    run: str
    annotator: str
    comparison_id: str
    choice: Literal["1", "2", "tie"]
    reason: str | None = None


@app.post("/api/richness/annotations")
def save_richness_annotation(body: RichnessAnnotationIn):
    rec = _richness_records(body.run).get(body.comparison_id)
    if rec is None:
        raise HTTPException(400, "comparison not part of this run")
    set_1 = _set_1_system(body.comparison_id, rec["system_x"], rec["system_y"])
    set_2 = rec["system_y"] if set_1 == rec["system_x"] else rec["system_x"]
    winner = None if body.choice == "tie" else set_1 if body.choice == "1" else set_2

    record = {
        "comparison_id": body.comparison_id,
        "image_path": rec["image_path"],
        "annotator": body.annotator,
        "run": body.run,
        "updated_at": datetime.now(UTC).isoformat(),
        "set_1": set_1,
        "set_2": set_2,
        "choice": body.choice,
        # In system terms, the same shape as the LLM record's "winner", so
        # human/LLM agreement is a direct comparison.
        "winner": winner,
        "reason": body.reason,
    }
    data = _load_human("richness", body.run, body.annotator)
    data[body.comparison_id] = record
    _save_human("richness", body.run, body.annotator, data)
    return {"saved": record}


# --- shared GET routes, parameterized by metric ---------------------------
#
# These cover every metric automatically. Only the POST .../annotations
# handler (rubric-specific scoring) needs a new endpoint per metric.

KNOWN_METRICS = {"decomposition_quality", "plausibility"}


def _check_metric(metric: str) -> None:
    if metric not in KNOWN_METRICS:
        raise HTTPException(404, f"unknown metric: {metric}")


@app.get("/api/{metric}/runs")
def list_runs(metric: str):
    _check_metric(metric)
    return {"runs": _list_runs(metric)}


@app.get("/api/{metric}/dataset")
def get_dataset(metric: str, run: str):
    _check_metric(metric)
    return {"items": _load_dataset(metric, run)}


@app.get("/api/{metric}/llm_judgement")
def get_llm_judgement(metric: str, run: str, image_path: str):
    """Opt-in lookup of the LLM judge's own verdicts + reasons for one image,
    for post-hoc human/LLM comparison. Not included in .../dataset so the
    default annotation flow stays blind to the LLM's decision."""
    _check_metric(metric)
    details = _load_llm_details(metric, run)
    if image_path not in details:
        raise HTTPException(404, "no LLM judgement for this image in this run")
    return details[image_path]


@app.get("/api/{metric}/annotations")
def get_annotations(metric: str, run: str, annotator: str):
    _check_metric(metric)
    return {"annotations": _load_human(metric, run, annotator)}


@app.get("/api/{metric}/progress")
def get_progress(metric: str, run: str, annotator: str):
    _check_metric(metric)
    total = len(_load_dataset(metric, run))
    done = len(_load_human(metric, run, annotator))
    return {"total": total, "done": done}


@app.get("/api/image")
def get_image(path: str):
    candidate = (REPO_ROOT / path).resolve()
    if not candidate.is_relative_to(DATA_ROOT.resolve()) or not candidate.is_file():
        raise HTTPException(404, "image not found")
    return FileResponse(candidate)


# --- run-vs-run comparison (read-only) -----------------------------------


def _compare_row(image_path: str, a: dict | None, b: dict | None) -> dict:
    """One image's side-by-side. ``a``/``b`` are the two runs' per-image
    detail dicts; either can be missing when the runs cover different image
    sets, in which case the row carries what it has and no delta."""

    def leg(rec: dict | None) -> dict:
        if rec is None:
            return {"present": False}
        atoms = rec.get("atoms") or []
        representation = rec.get("representation") or ""
        return {
            "present": True,
            "score": rec.get("score"),
            "passed": rec.get("passed"),
            "n_atoms": len(atoms),
            # Judge-verdict shape (plausibility); metrics whose atoms are
            # plain strings simply report no failures.
            "n_failed": sum(
                1 for atom in atoms if isinstance(atom, dict) and not atom.get("final_verdict")
            ),
            "words": len(representation.split()),
        }

    left, right = leg(a), leg(b)
    delta = None
    if left["present"] and right["present"]:
        delta = right["score"] - left["score"]
    return {
        "image_path": image_path,
        "a": left,
        "b": right,
        "delta": delta,
        # "regressed" is b scoring below a - the run under test losing ground
        # against the reference. Tie and missing are distinguished so the page
        # never shows a green or red light for "we don't know".
        "status": (
            "missing"
            if delta is None
            else "regressed"
            if delta < 0
            else "improved"
            if delta > 0
            else "tied"
        ),
    }


@app.get("/api/compare")
def compare_runs(metric: str, run_a: str, run_b: str):
    """Two runs of one metric, joined per image. run_a is the reference and
    run_b the one under test, so a negative delta is a regression."""
    _check_metric(metric)
    a = _load_llm_details(metric, run_a)
    b = _load_llm_details(metric, run_b)

    rows = [_compare_row(key, a.get(key), b.get(key)) for key in sorted(a.keys() | b.keys())]
    scored = [row for row in rows if row["delta"] is not None]
    deltas = [row["delta"] for row in scored]

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "metric": metric,
        "run_a": run_a,
        "run_b": run_b,
        "rows": rows,
        "summary": {
            "n": len(rows),
            "n_paired": len(scored),
            "n_only_a": sum(1 for row in rows if not row["b"]["present"]),
            "n_only_b": sum(1 for row in rows if not row["a"]["present"]),
            "mean_a": mean([row["a"]["score"] for row in scored]),
            "mean_b": mean([row["b"]["score"] for row in scored]),
            "mean_delta": mean(deltas),
            "n_regressed": sum(1 for row in rows if row["status"] == "regressed"),
            "n_improved": sum(1 for row in rows if row["status"] == "improved"),
            "n_tied": sum(1 for row in rows if row["status"] == "tied"),
        },
    }


# --- decomposition_quality: annotation POST (rubric-specific) -------------


class AtomEvaluation(BaseModel):
    affectiveness: bool
    atomicity: bool
    fidelity: bool
    evidence_preservation: bool


class AtomJudgement(BaseModel):
    atom: str
    evaluation: AtomEvaluation


class DecompAnnotationIn(BaseModel):
    run: str
    annotator: str
    image_path: str
    atomic_judgement: list[AtomJudgement]
    completeness: Literal[1, 2, 3, 4, 5]


@app.post("/api/decomposition_quality/annotations")
def save_decomposition_annotation(body: DecompAnnotationIn):
    if body.image_path not in _dataset_image_paths("decomposition_quality", body.run):
        raise HTTPException(400, "image_path not part of this run's dataset")

    def is_good(aj: AtomJudgement) -> bool:
        return all(
            [
                aj.evaluation.affectiveness,
                aj.evaluation.atomicity,
                aj.evaluation.fidelity,
                aj.evaluation.evidence_preservation,
            ]
        )

    good_count = sum(1 for aj in body.atomic_judgement if is_good(aj))
    total = len(body.atomic_judgement)
    atom_quality = (good_count / total * 5) if total else 0.0

    record = {
        "image_path": body.image_path,
        "annotator": body.annotator,
        "run": body.run,
        "updated_at": datetime.now(UTC).isoformat(),
        "atomic_judgement": [
            {
                "atom": aj.atom,
                "evaluation": aj.evaluation.model_dump(),
                "verdict": "Good" if is_good(aj) else "Bad",
            }
            for aj in body.atomic_judgement
        ],
        "final_verdict": {
            "completeness": {"verdict": body.completeness},
            "atom_quality": {
                "good_atom_count": good_count,
                "total_atom_count": total,
                "verdict": round(atom_quality, 2),
            },
        },
        "score": round((body.completeness + atom_quality) / 2 / 5, 4),
    }

    data = _load_human("decomposition_quality", body.run, body.annotator)
    data[body.image_path] = record
    _save_human("decomposition_quality", body.run, body.annotator, data)
    return {"saved": record}


# --- plausibility: annotation POST (rubric-specific) -----------------------


class PlausAtomJudgement(BaseModel):
    atom: str
    type: Literal["vibe_only", "evidence_backed"]
    plausible: bool
    reason: str | None = None


class PlausAnnotationIn(BaseModel):
    run: str
    annotator: str
    image_path: str
    atoms: list[PlausAtomJudgement]


@app.post("/api/plausibility/annotations")
def save_plausibility_annotation(body: PlausAnnotationIn):
    if body.image_path not in _dataset_image_paths("plausibility", body.run):
        raise HTTPException(400, "image_path not part of this run's dataset")

    plausible_count = sum(1 for a in body.atoms if a.plausible)
    total = len(body.atoms)

    record = {
        "image_path": body.image_path,
        "annotator": body.annotator,
        "run": body.run,
        "updated_at": datetime.now(UTC).isoformat(),
        "atoms": [
            {
                "atom": a.atom,
                "type": a.type,
                "plausible": a.plausible,
                "reason": a.reason,
            }
            for a in body.atoms
        ],
        "score": round(plausible_count / total, 4) if total else 0.0,
    }

    data = _load_human("plausibility", body.run, body.annotator)
    data[body.image_path] = record
    _save_human("plausibility", body.run, body.annotator, data)
    return {"saved": record}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
