"""FastAPI backend for the human annotation / results-viewer webapps.

Serves the same (representation, atoms) pairs that were fed to an
LLM-as-a-judge metric, lets a human rate them with the same rubric (minus
free-text reasons for decomposition quality), and stores results keyed by
``image_path`` + run so they can be paired 1:1 with
``results/<metric>/<run>.per_image.jsonl`` for Cohen's kappa alignment
analysis. The human never sees the LLM's verdicts by default, to avoid
anchoring bias.

Also serves read-only results viewers (static/results.html + results.js,
static/plausibility_results.html + plausibility_results.js) for browsing a
run's images, representations, decompositions, and the LLM judge's
verdicts/reasoning directly — no annotation involved.

Every metric (decomposition_quality, plausibility, and any added later —
interpretability, richness, ...) shares the same run-discovery, dataset,
LLM-judgement, human-annotation-store, and progress plumbing below,
parameterized by ``metric``. Adding a new metric only requires:
  1. a ``BLIND_ATOM_FNS[metric]`` entry, if its atoms need to be stripped of
     judge-only fields before being shown to an annotator (skip if atoms are
     already blind, e.g. plain strings)
  2. a Pydantic annotation-input model + POST ``/api/<metric>/annotations``
     handler encoding that metric's rubric/scoring — this part is
     irreducibly metric-specific, since every metric's rubric differs.
"""

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
DATA_DIR = REPO_ROOT / "data" / "main_processed"
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
    if not candidate.is_relative_to(DATA_DIR.resolve()) or not candidate.is_file():
        raise HTTPException(404, "image not found")
    return FileResponse(candidate)


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
