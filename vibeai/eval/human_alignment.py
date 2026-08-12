"""Compute Cohen's kappa between LLM-as-a-judge verdicts and one or more
human annotation files produced by the webapp in ``vibeai/webapp``, for
either the ``decomposition_quality`` or ``plausibility`` metric.

Pairs records by ``image_path`` (the same key used in
``results/<metric>/<run>.per_image.jsonl`` and in
``results/<metric>/human/<run>__<annotator>.json``), so only images a given
human actually annotated are scored.

Usage:
    python -m vibeai.eval.human_alignment baseline__baseline_1785814420 --annotator mingeon
    python -m vibeai.eval.human_alignment <run> --annotator alice --annotator bob
    python -m vibeai.eval.human_alignment baseline__v1_1786409728 --metric plausibility --annotator mingeon
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"


def _confusion(y1: list, y2: list, labels: list) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for a, b in zip(y1, y2):
        matrix[index[a]][index[b]] += 1
    return matrix


def cohen_kappa(y1: list, y2: list, labels: list, weights: str | None = None) -> float:
    """Cohen's kappa; weights is None (unweighted), 'linear', or 'quadratic'."""
    n = len(labels)
    confusion = _confusion(y1, y2, labels)
    total = sum(sum(row) for row in confusion)
    if total == 0:
        return float("nan")

    row_sums = [sum(row) for row in confusion]
    col_sums = [sum(confusion[i][j] for i in range(n)) for j in range(n)]
    expected = [[row_sums[i] * col_sums[j] / total for j in range(n)] for i in range(n)]

    if weights is None:
        w = [[0.0 if i == j else 1.0 for j in range(n)] for i in range(n)]
    elif weights == "linear":
        w = [[abs(i - j) / (n - 1) for j in range(n)] for i in range(n)]
    elif weights == "quadratic":
        w = [[((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)]
    else:
        raise ValueError(f"unknown weights: {weights}")

    observed = sum(w[i][j] * confusion[i][j] for i in range(n) for j in range(n))
    expected_val = sum(w[i][j] * expected[i][j] for i in range(n) for j in range(n))
    if expected_val == 0:
        return float("nan")
    return 1 - observed / expected_val


def load_llm(metric: str, run: str) -> dict[str, dict]:
    src = RESULTS_ROOT / metric / f"{run}.per_image.jsonl"
    out = {}
    with src.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["image_path"]] = rec["details"]
    return out


def load_human(metric: str, run: str, annotator: str) -> dict[str, dict]:
    src = RESULTS_ROOT / metric / "human" / f"{run}__{annotator}.json"
    if not src.exists():
        raise FileNotFoundError(src)
    return json.loads(src.read_text())


def align_and_report_plausibility(run: str, annotators: list[str]) -> None:
    """Atom-level plausible/implausible agreement. The LLM judge's per-atom
    ``final_verdict`` is directly comparable to the human's boolean
    ``plausible`` field."""
    llm = load_llm("plausibility", run)

    for annotator in annotators:
        human = load_human("plausibility", run, annotator)
        shared = sorted(set(llm) & set(human))
        print(f"\n=== annotator: {annotator} — {len(shared)}/{len(human)} annotated images matched to LLM run '{run}' ===")
        if not shared:
            continue

        verdict_llm, verdict_human = [], []
        by_type_llm = {"vibe_only": [], "evidence_backed": []}
        by_type_human = {"vibe_only": [], "evidence_backed": []}

        for image_path in shared:
            l_atoms = llm[image_path]["atoms"]
            h_atoms = human[image_path]["atoms"]
            if len(l_atoms) != len(h_atoms):
                print(f"  ! atom count mismatch for {image_path} (llm={len(l_atoms)}, human={len(h_atoms)}); skipping")
                continue
            for la, ha in zip(l_atoms, h_atoms):
                l_plausible = la["final_verdict"]
                verdict_llm.append(l_plausible)
                verdict_human.append(ha["plausible"])
                by_type_llm[ha["type"]].append(l_plausible)
                by_type_human[ha["type"]].append(ha["plausible"])

        if verdict_llm:
            n_atoms = len(verdict_llm)
            agreement = sum(a == b for a, b in zip(verdict_llm, verdict_human)) / n_atoms
            print(f"  Atom plausibility (unweighted kappa, n={n_atoms} atoms): "
                  f"{cohen_kappa(verdict_llm, verdict_human, [True, False]):.3f} "
                  f"(raw agreement: {agreement:.3f})")
            for atom_type in by_type_llm:
                if by_type_llm[atom_type]:
                    kappa = cohen_kappa(by_type_llm[atom_type], by_type_human[atom_type], [True, False])
                    print(f"    - {atom_type:<16} (n={len(by_type_llm[atom_type])}) kappa: {kappa:.3f}")


def align_and_report(run: str, annotators: list[str]) -> None:
    llm = load_llm("decomposition_quality", run)

    for annotator in annotators:
        human = load_human("decomposition_quality", run, annotator)
        shared = sorted(set(llm) & set(human))
        print(f"\n=== annotator: {annotator} — {len(shared)}/{len(human)} annotated images matched to LLM run '{run}' ===")
        if not shared:
            continue

        completeness_llm, completeness_human = [], []
        atom_verdict_llm, atom_verdict_human = [], []
        criteria_llm = {k: [] for k in ["affectiveness", "atomicity", "fidelity", "evidence_preservation"]}
        criteria_human = {k: [] for k in criteria_llm}
        atom_quality_diffs = []

        for image_path in shared:
            l, h = llm[image_path], human[image_path]
            completeness_llm.append(l["final_verdict"]["completeness"]["verdict"])
            completeness_human.append(h["final_verdict"]["completeness"]["verdict"])
            atom_quality_diffs.append(
                abs(
                    l["final_verdict"]["atom_quality"]["verdict"]
                    - h["final_verdict"]["atom_quality"]["verdict"]
                )
            )

            l_atoms = l["atomic_judgement"]
            h_atoms = h["atomic_judgement"]
            if len(l_atoms) != len(h_atoms):
                print(f"  ! atom count mismatch for {image_path} (llm={len(l_atoms)}, human={len(h_atoms)}); skipping atom-level pairing for this image")
                continue
            for la, ha in zip(l_atoms, h_atoms):
                atom_verdict_llm.append(la["verdict"])
                atom_verdict_human.append(ha["verdict"])
                for k in criteria_llm:
                    criteria_llm[k].append(la["evaluation"][k])
                    criteria_human[k].append(ha["evaluation"][k])

        labels_1_5 = [1, 2, 3, 4, 5]
        print(f"  Completeness        (linear-weighted kappa, n={len(shared)}): "
              f"{cohen_kappa(completeness_llm, completeness_human, labels_1_5, 'linear'):.3f}")
        print(f"  Atom quality        (mean |Δ| on 0-5 scale, n={len(shared)}): "
              f"{sum(atom_quality_diffs) / len(atom_quality_diffs):.3f}")

        if atom_verdict_llm:
            n_atoms = len(atom_verdict_llm)
            print(f"  Atom verdict Good/Bad (unweighted kappa, n={n_atoms} atoms): "
                  f"{cohen_kappa(atom_verdict_llm, atom_verdict_human, ['Good', 'Bad']):.3f}")
            for k in criteria_llm:
                kappa = cohen_kappa(criteria_llm[k], criteria_human[k], [True, False])
                print(f"    - {k:<22} kappa: {kappa:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="run name, e.g. baseline__baseline_1785814420")
    parser.add_argument(
        "--metric", choices=["decomposition_quality", "plausibility"],
        default="decomposition_quality",
    )
    parser.add_argument(
        "--annotator", action="append", required=True,
        help="human annotator id (repeatable for multiple annotators)",
    )
    args = parser.parse_args()
    if args.metric == "plausibility":
        align_and_report_plausibility(args.run, args.annotator)
    else:
        align_and_report(args.run, args.annotator)


if __name__ == "__main__":
    main()
