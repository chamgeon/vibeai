"""Build the report and README figures from the saved result files.

    uv run --with matplotlib python docs/make_figures.py

matplotlib is not a project dependency; ``--with`` pulls it in for this run
only. Every number comes from results/ (no LLM calls), so rerunning after a new
eval run only needs the run names below updated.

Writes docs/figures/{plausibility,plausibility_by_type,richness_pairs,richness_control}.{svg,png}.
"""

import json
import random
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vibeai.eval.richness_control import design, fit_logit, load_decisive  # noqa: E402
from vibeai.eval.significance import load_scores  # noqa: E402

OUT_DIR = REPO_ROOT / "docs" / "figures"

PLAUSIBILITY_RUNS = {
    "gpt-5 judge": {
        "baseline": "baseline__v2_1790129592",
        "v1": "v1__v2_1790128719",
        "workflow": "workflow__v2_1790128650",
    },
    "Claude judge (claude-sonnet-5)": {
        "baseline": "baseline__v2_1790134860",
        "v1": "v1__v2_1790132953",
        "workflow": "workflow__v2_1790134094",
    },
}
RICHNESS_RUN = "baseline+v1+workflow_1794469247"
SYSTEMS = ["baseline", "v1", "workflow"]

# Reference palette, categorical slots 1-3 (validated all-pairs in light mode).
# Colour follows the system in every figure.
COLOR = {"baseline": "#2a78d6", "v1": "#eb6834", "workflow": "#1baf7a"}
TIE = "#d6d5cf"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": INK_2,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "svg.fonttype": "none",
})


def _style(ax, grid_axis="x"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def bootstrap_ci(values, n=10_000, seed=0):
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def plausibility_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.1), sharex=True)
    for ax, (judge, runs) in zip(axes, PLAUSIBILITY_RUNS.items()):
        scores = {s: load_scores("plausibility", run) for s, run in runs.items()}
        images = sorted(set.intersection(*(set(v) for v in scores.values())))
        for row, system in enumerate(SYSTEMS):
            values = [scores[system][i] for i in images]
            m = mean(values)
            lo, hi = bootstrap_ci(values)
            y = len(SYSTEMS) - 1 - row
            ax.plot([lo, hi], [y, y], color=COLOR[system], linewidth=2, solid_capstyle="round")
            ax.scatter([m], [y], s=46, color=COLOR[system], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.text(hi + 0.004, y, f"{m:.3f}", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(range(len(SYSTEMS)), list(reversed(SYSTEMS)))
        ax.set_ylim(-0.6, len(SYSTEMS) - 0.4)
        ax.set_title(f"{judge} · n = {len(images)}", fontsize=9, loc="left", color=INK)
        _style(ax)
    axes[0].set_xlim(0.78, 0.98)
    for ax in axes:
        ax.set_xlabel("mean plausible-atom rate, 95% bootstrap CI")
    fig.tight_layout(w_pad=2.5)
    _save(fig, "plausibility")


ATOM_TYPES = {"vibe_only": "vibe-only atoms", "evidence_backed": "evidence-backed atoms"}


def type_rates(run: str) -> dict[str, list[float]]:
    """Per-image plausible rate within each atom type (images with none of a
    type are skipped for that type), read from a plausibility run."""
    rates = {t: [] for t in ATOM_TYPES}
    path = REPO_ROOT / "results" / "plausibility" / f"{run}.per_image.jsonl"
    for line in path.open():
        atoms = json.loads(line)["details"]["atoms"]
        for atom_type in ATOM_TYPES:
            typed = [a["final_verdict"] for a in atoms if a["type"] == atom_type]
            if typed:
                rates[atom_type].append(sum(typed) / len(typed))
    return rates


def plausibility_by_type_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharex=True)
    offsets = {"baseline": 0.28, "v1": 0.0, "workflow": -0.28}
    for ax, (judge, runs) in zip(axes, PLAUSIBILITY_RUNS.items()):
        rates = {s: type_rates(run) for s, run in runs.items()}
        for group, atom_type in enumerate(ATOM_TYPES):
            center = 1 - group
            for system in SYSTEMS:
                values = rates[system][atom_type]
                m = mean(values)
                lo, hi = bootstrap_ci(values)
                y = center + offsets[system]
                ax.plot([lo, hi], [y, y], color=COLOR[system], linewidth=2, solid_capstyle="round")
                ax.scatter([m], [y], s=40, color=COLOR[system], edgecolor=SURFACE, linewidth=1.5, zorder=3)
                ax.text(hi + 0.004, y, f"{system} {m:.2f}", va="center", fontsize=7.5, color=INK_2)
        ax.set_yticks([1, 0], list(ATOM_TYPES.values()))
        ax.set_ylim(-0.5, 1.5)
        ax.axhline(0.5, color=GRID, linewidth=0.6)
        ax.set_title(judge, fontsize=9, loc="left")
        ax.set_xlabel("plausible rate within atom type, 95% bootstrap CI")
        _style(ax)
    axes[0].set_xlim(0.72, 1.04)
    fig.tight_layout(w_pad=2.5)
    _save(fig, "plausibility_by_type")


def richness_pairs_figure():
    summary = json.loads(
        (REPO_ROOT / "results" / "richness" / f"{RICHNESS_RUN}.json").read_text()
    )
    pairs = summary["pairs"]
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    for row, pair in enumerate(pairs):
        y = len(pairs) - 1 - row
        n = pair["n"]
        segments = [
            (pair["wins_a"], COLOR[pair["a"]], "#ffffff"),
            (pair["ties"], TIE, INK),
            (pair["wins_b"], COLOR[pair["b"]], "#ffffff"),
        ]
        left = 0.0
        for count, color, text_color in segments:
            width = count / n
            ax.barh(y, width, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
            if width >= 0.06:
                ax.text(left + width / 2, y, str(count), ha="center", va="center",
                        fontsize=8.5, color=text_color, fontweight="bold")
            left += width
        ax.text(-0.015, y, pair["a"], ha="right", va="center", color=INK_2)
        ax.text(1.015, y, pair["b"], ha="left", va="center", color=INK_2)
        p = pair["p_value"]
        ax.text(1.16, y, "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}", ha="left",
                va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(pairs) - 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.axvline(0.5, color=AXIS, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("share of the 97 images: left system richer · tie (grey) · right system richer")
    _save(fig, "richness_pairs")


def richness_control_figure():
    labels, rows = load_decisive(RICHNESS_RUN)
    outcomes = [row["x_won"] for row in rows]
    fits = {}
    for title, with_size in (("raw", False), ("controlled for set size", True)):
        features, names = design(labels, rows, with_size=with_size)
        fits[title] = fit_logit(features, outcomes, names)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.9), sharex=True, sharey=True)
    for ax, (title, fit) in zip(axes, fits.items()):
        for row, system in enumerate(labels[1:]):
            coef, se = fit.coef[row], fit.se(row)
            y = 1 - row
            ax.plot([coef - 1.96 * se, coef + 1.96 * se], [y, y], color=COLOR[system],
                    linewidth=2, solid_capstyle="round")
            ax.scatter([coef], [y], s=46, color=COLOR[system], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.text(coef + 1.96 * se + 0.12, y, f"{coef:+.2f}", va="center", fontsize=8.5)
        ax.axvline(0, color=COLOR["baseline"], linewidth=1.2, linestyle=(0, (3, 2)))
        ax.text(0.08, -0.55, "baseline = 0", color=INK_2, fontsize=8)
        ax.set_yticks([1, 0], labels[1:])
        ax.set_ylim(-0.7, 1.5)
        diff, se = fit.contrast(1, 0)
        ax.set_title(f"{title}: workflow − v1 = {diff:+.2f} ± {1.96 * se:.2f}",
                     fontsize=9, loc="left")
        ax.set_xlabel("strength vs baseline (log-odds), 95% CI")
        _style(ax)
    axes[0].set_xlim(-0.5, 5.2)
    fig.tight_layout(w_pad=2.5)
    _save(fig, "richness_control")


if __name__ == "__main__":
    plausibility_figure()
    plausibility_by_type_figure()
    richness_pairs_figure()
    richness_control_figure()
    print(f"wrote figures to {OUT_DIR}")
