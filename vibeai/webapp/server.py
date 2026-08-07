"""FastAPI backend for the decomposition-quality human annotation app.

Serves the same (representation, atoms) pairs that were fed to the
LLM-as-a-judge decomposition-quality metric, lets a human rate them with the
same rubric (completeness, per-atom criteria) minus free-text reasons, and
stores results keyed by ``image_path`` + run so they can
be paired 1:1 with ``results/decomposition_quality/<run>.per_image.jsonl``
for Cohen's kappa alignment analysis. The human never sees the LLM's
verdicts, to avoid anchoring bias.

Also serves a separate read-only results viewer (static/results.html +
results.js) for browsing a run's images, representations, decompositions,
and the LLM judge's verdicts/reasoning directly — no annotation involved.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "decomposition_quality"
HUMAN_DIR = RESULTS_DIR / "human"
DATA_DIR = REPO_ROOT / "data" / "main_processed"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Vibe Decomposition Quality — Human Annotation")


# --- dataset -----------------------------------------------------------


def _list_runs() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        p.name.removesuffix(".per_image.jsonl")
        for p in RESULTS_DIR.glob("*.per_image.jsonl")
    )


def _load_dataset(run: str) -> list[dict]:
    src = RESULTS_DIR / f"{run}.per_image.jsonl"
    if not src.exists():
        raise HTTPException(404, f"unknown run: {run}")
    items = []
    with src.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            details = rec.get("details", {})
            items.append(
                {
                    "image_path": rec["image_path"],
                    "representation": details.get("representation", ""),
                    "atoms": details.get("atoms", []),
                }
            )
    return items


def _load_llm_details(run: str) -> dict[str, dict]:
    src = RESULTS_DIR / f"{run}.per_image.jsonl"
    if not src.exists():
        raise HTTPException(404, f"unknown run: {run}")
    out = {}
    with src.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["image_path"]] = {
                **rec.get("details", {}),
                "score": rec.get("score"),
                "passed": rec.get("passed"),
            }
    return out


# --- human annotation store --------------------------------------------


def _human_path(run: str, annotator: str) -> Path:
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    safe_annotator = "".join(c if c.isalnum() or c in "-_." else "_" for c in annotator)
    if not safe_annotator:
        raise HTTPException(400, "invalid annotator id")
    return HUMAN_DIR / f"{run}__{safe_annotator}.json"


def _load_human(run: str, annotator: str) -> dict[str, dict]:
    path = _human_path(run, annotator)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_human(run: str, annotator: str, data: dict[str, dict]) -> None:
    path = _human_path(run, annotator)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


# --- API models ----------------------------------------------------------


class AtomEvaluation(BaseModel):
    affectiveness: bool
    atomicity: bool
    fidelity: bool
    evidence_preservation: bool


class AtomJudgement(BaseModel):
    atom: str
    evaluation: AtomEvaluation


class AnnotationIn(BaseModel):
    run: str
    annotator: str
    image_path: str
    atomic_judgement: list[AtomJudgement]
    completeness: Literal[1, 2, 3, 4, 5]


# --- API routes ----------------------------------------------------------


@app.get("/api/runs")
def list_runs():
    return {"runs": _list_runs()}


@app.get("/api/dataset")
def get_dataset(run: str):
    return {"items": _load_dataset(run)}


@app.get("/api/llm_judgement")
def get_llm_judgement(run: str, image_path: str):
    """Opt-in lookup of the LLM judge's own verdicts + reasons for one image,
    for post-hoc human/LLM comparison. Not included in /api/dataset so the
    default annotation flow stays blind to the LLM's decision."""
    details = _load_llm_details(run)
    if image_path not in details:
        raise HTTPException(404, "no LLM judgement for this image in this run")
    return details[image_path]


@app.get("/api/annotations")
def get_annotations(run: str, annotator: str):
    return {"annotations": _load_human(run, annotator)}


@app.post("/api/annotations")
def save_annotation(body: AnnotationIn):
    dataset_image_paths = {item["image_path"] for item in _load_dataset(body.run)}
    if body.image_path not in dataset_image_paths:
        raise HTTPException(400, "image_path not part of this run's dataset")

    good_count = sum(
        1
        for aj in body.atomic_judgement
        if all(
            [
                aj.evaluation.affectiveness,
                aj.evaluation.atomicity,
                aj.evaluation.fidelity,
                aj.evaluation.evidence_preservation,
            ]
        )
    )
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
                "verdict": "Good"
                if all(
                    [
                        aj.evaluation.affectiveness,
                        aj.evaluation.atomicity,
                        aj.evaluation.fidelity,
                        aj.evaluation.evidence_preservation,
                    ]
                )
                else "Bad",
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

    data = _load_human(body.run, body.annotator)
    data[body.image_path] = record
    _save_human(body.run, body.annotator, data)
    return {"saved": record}


@app.get("/api/progress")
def get_progress(run: str, annotator: str):
    total = len(_load_dataset(run))
    done = len(_load_human(run, annotator))
    return {"total": total, "done": done}


@app.get("/api/image")
def get_image(path: str):
    candidate = (REPO_ROOT / path).resolve()
    if not candidate.is_relative_to(DATA_DIR.resolve()) or not candidate.is_file():
        raise HTTPException(404, "image not found")
    return FileResponse(candidate)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
