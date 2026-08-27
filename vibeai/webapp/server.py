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

Every metric (decomposition_quality, plausibility, richness, and any added
later — interpretability, ...) shares the same run-discovery, dataset,
LLM-judgement, human-annotation-store, and progress plumbing below,
parameterized by ``metric``. Adding a new metric only requires:
  1. a ``KNOWN_METRICS`` entry, plus a ``SOURCE_RUN_METRICS`` entry if it is
     annotated over *another* metric's run output rather than its own
  2. a ``BLIND_ATOM_FNS[metric]`` entry, if its atoms need to be stripped of
     judge-only fields before being shown to an annotator (skip if atoms are
     already blind, e.g. plain strings)
  3. a Pydantic annotation-input model + POST ``/api/<metric>/annotations``
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

KNOWN_METRICS = {"decomposition_quality", "plausibility", "richness"}

# Metrics that are annotated over *another* metric's run output instead of
# their own: richness re-reads the (representation, atoms) pairs any run
# already produced and asks which vibe axes each atom touches, so it needs
# no LLM judge of its own to be annotatable. Their run ids are qualified as
# "<source_metric>/<run>" (e.g. "plausibility/v2__direct_1786613807") so a
# single annotation store can span every source run.
SOURCE_RUN_METRICS = {"richness"}


def _results_dir(metric: str) -> Path:
    return RESULTS_ROOT / metric


def _human_dir(metric: str) -> Path:
    return _results_dir(metric) / "human"


def _split_run(metric: str, run: str) -> tuple[str, str]:
    """Resolve a run id to the (metric dir, run name) its records live under.
    Only SOURCE_RUN_METRICS use the qualified "<source_metric>/<run>" form."""
    if metric in SOURCE_RUN_METRICS and "/" in run:
        source_metric, _, name = run.partition("/")
        if source_metric not in KNOWN_METRICS:
            raise HTTPException(404, f"unknown source metric: {source_metric}")
        return source_metric, name
    return metric, run


def _list_runs(metric: str) -> list[str]:
    """Run ids offered for a metric. A SOURCE_RUN_METRICS metric has no runs
    of its own to annotate, so it lists every metric's runs, qualified."""
    if metric in SOURCE_RUN_METRICS:
        return sorted(
            f"{m}/{p.name.removesuffix('.per_image.jsonl')}"
            for m in KNOWN_METRICS
            for p in _results_dir(m).glob("*.per_image.jsonl")
        )
    d = _results_dir(metric)
    if not d.exists():
        return []
    return sorted(p.name.removesuffix(".per_image.jsonl") for p in d.glob("*.per_image.jsonl"))


def _load_per_image_records(metric: str, run: str) -> list[dict]:
    source_metric, run_name = _split_run(metric, run)
    src = _results_dir(source_metric) / f"{run_name}.per_image.jsonl"
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
    source_metric, _ = _split_run(metric, run)
    blind = BLIND_ATOM_FNS.get(source_metric, lambda atom: atom)
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


def _safe_id(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def _human_path(metric: str, run: str, annotator: str) -> Path:
    hd = _human_dir(metric)
    hd.mkdir(parents=True, exist_ok=True)
    safe_annotator = _safe_id(annotator)
    if not safe_annotator:
        raise HTTPException(400, "invalid annotator id")
    # A qualified run id ("plausibility/v2__...") flattens to
    # "plausibility__v2__..."; unqualified ids keep their existing filenames.
    safe_run = _safe_id(run.replace("/", "__"))
    return hd / f"{safe_run}__{safe_annotator}.json"


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


# --- richness: annotation POST (rubric-specific) --------------------------
#
# Richness asks which of a fixed set of vibe axes each atom touches, so an
# image's representation can be scored on how much of the vibe space it
# covers rather than on whether its claims hold up.
#
# The denominator is per-image: an axis the image has nothing to say about
# (interpersonal dynamics in a solo portrait, subject in an empty landscape)
# is marked not-applicable by the annotator and dropped from both sides of
# the ratio, so a representation is never penalised for failing to cover a
# dimension that isn't there.

VIBE_AXES = (
    "subject",
    "interpersonal-group",
    "object",
    "environment",
    "composition",
    "color",
    "texture-material",
    "style",
    "activity",
    "time",
    "affect",
)

# Applicability is a property of photographs rather than of any one image for
# these four, so they can't be switched off and the denominator never falls
# below 4. The rest default to applicable and the annotator opts them out.
ALWAYS_APPLICABLE_AXES = ("environment", "color", "style", "affect")

# Served to the UI so the axis vocabulary, its wording and its applicability
# rule all live in one place. Mirrors the "Vibe Axis" section of the
# Richness experiment doc.
AXIS_INFO: dict[str, dict[str, str]] = {
    "subject": {
        "label": "Subject",
        "description": "Vibe evoked by the central subject(s) — primarily people, sometimes animals.",
        "applicability": "Applicable when a subject is present.",
    },
    "interpersonal-group": {
        "label": "Interpersonal & group",
        "description": "Vibe evoked by the relationship between subjects in the image.",
        "applicability": "Applicable when multiple subjects are visible.",
    },
    "object": {
        "label": "Object",
        "description": "Vibe evoked by objects, props and physical items in the image.",
        "applicability": "Applicable when an object is present.",
    },
    "environment": {
        "label": "Environment",
        "description": "Vibe evoked by the environment, setting, surroundings, architecture, weather.",
        "applicability": "Always applicable.",
    },
    "composition": {
        "label": "Composition",
        "description": "Vibe evoked by the composition of the image — framing, angle, arrangement, depth.",
        "applicability": "Applicable when composition is prominent.",
    },
    "color": {
        "label": "Color",
        "description": "Vibe evoked by the colour and lighting of the image.",
        "applicability": "Always applicable.",
    },
    "texture-material": {
        "label": "Texture & material",
        "description": "Vibe evoked by texture and material — surfaces, substances, their condition.",
        "applicability": "Applicable almost always.",
    },
    "style": {
        "label": "Style",
        "description": "Vibe evoked by the style/genre — minimal, futuristic, cyberpunk, candid, selfie, editorial, portrait, scenery, ...",
        "applicability": "Always applicable.",
    },
    "activity": {
        "label": "Activity",
        "description": "Vibe evoked by the activity expressed or implied in the image.",
        "applicability": "Applicable when the image is associated with an activity.",
    },
    "time": {
        "label": "Time",
        "description": "Vibe evoked by the time of day or the season.",
        "applicability": "Applicable when time can be inferred from the image.",
    },
    "affect": {
        "label": "Affect",
        "description": "Raw emotion, not attributed to any one image element.",
        "applicability": "Always applicable.",
    },
    "other": {
        "label": "other",
        "description": "A vibe dimension none of the eleven capture — say which in the reason box.",
        "applicability": "Recorded but never scored, so scores stay comparable across images.",
    },
}

# "other" is annotatable (with a free-text reason saying what it was) but is
# deliberately kept out of the coverage denominator, so scores stay
# comparable across images.
VibeAxis = Literal[
    "subject",
    "interpersonal-group",
    "object",
    "environment",
    "composition",
    "color",
    "texture-material",
    "style",
    "activity",
    "time",
    "affect",
    "other",
]

# Applicability is only ever recorded for the scored axes — "other" has no
# denominator to belong to.
ScoredVibeAxis = Literal[
    "subject",
    "interpersonal-group",
    "object",
    "environment",
    "composition",
    "color",
    "texture-material",
    "style",
    "activity",
    "time",
    "affect",
]


@app.get("/api/richness/axes")
def get_richness_axes():
    """The axis vocabulary, served so the annotation UI can't drift out of
    sync with what the POST handler accepts."""
    return {
        "axes": list(VIBE_AXES),
        "extra_axes": ["other"],
        "always_applicable": list(ALWAYS_APPLICABLE_AXES),
        "info": AXIS_INFO,
    }


class RichnessAtomJudgement(BaseModel):
    atom: str
    axes: list[VibeAxis]
    reason: str | None = None


class RichnessAnnotationIn(BaseModel):
    run: str
    annotator: str
    image_path: str
    atoms: list[RichnessAtomJudgement]
    # Which axes this image can be covered on at all. Absent (older clients)
    # means "all of them", i.e. the previous fixed denominator.
    applicable_axes: list[ScoredVibeAxis] | None = None


@app.post("/api/richness/annotations")
def save_richness_annotation(body: RichnessAnnotationIn):
    if body.image_path not in _dataset_image_paths("richness", body.run):
        raise HTTPException(400, "image_path not part of this run's dataset")

    order = {axis: i for i, axis in enumerate(VIBE_AXES + ("other",))}

    def normalize(axes) -> list[str]:
        """De-duplicate and put axes in canonical order, so annotations are
        comparable byte-for-byte regardless of the order they were clicked."""
        return sorted(set(axes), key=order.__getitem__)

    requested = set(VIBE_AXES if body.applicable_axes is None else body.applicable_axes)
    applicable_axes = normalize(requested | set(ALWAYS_APPLICABLE_AXES))
    na_axes = [axis for axis in VIBE_AXES if axis not in applicable_axes]

    atoms = [
        {
            "atom": a.atom,
            "axes": normalize(a.axes),
            "reason": a.reason,
        }
        for a in body.atoms
    ]

    axis_atom_counts = {
        axis: sum(1 for a in atoms if axis in a["axes"]) for axis in VIBE_AXES + ("other",)
    }
    covered_axes = [axis for axis in applicable_axes if axis_atom_counts[axis]]
    # Atoms tagged on an axis the annotator called N/A: not scored either
    # way, but worth surfacing since one of the two judgements is wrong.
    conflicting_axes = [axis for axis in na_axes if axis_atom_counts[axis]]
    unlabeled_atom_count = sum(1 for a in atoms if not a["axes"])

    record = {
        "image_path": body.image_path,
        "annotator": body.annotator,
        "run": body.run,
        "updated_at": datetime.now(UTC).isoformat(),
        "atoms": atoms,
        "axis_atom_counts": axis_atom_counts,
        "applicable_axes": applicable_axes,
        "na_axes": na_axes,
        "covered_axes": covered_axes,
        "conflicting_axes": conflicting_axes,
        "unlabeled_atom_count": unlabeled_atom_count,
        "score": round(len(covered_axes) / len(applicable_axes), 4),
    }

    data = _load_human("richness", body.run, body.annotator)
    data[body.image_path] = record
    _save_human("richness", body.run, body.annotator, data)
    return {"saved": record}


class NoCacheStaticFiles(StaticFiles):
    """Serve the annotation UI with ``Cache-Control: no-cache`` so the browser
    revalidates (ETag -> 304) instead of guessing.

    Without it Starlette sends no Cache-Control at all, so Chrome applies
    *heuristic* freshness derived from the file's old Last-Modified date and
    can sit on a cached style.css/app.js for days — silently showing an
    annotator a stale UI after an edit."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", NoCacheStaticFiles(directory=STATIC_DIR, html=True), name="static")
