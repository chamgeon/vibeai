"""Set-size control for a richness comparison run: is a system's win rate more
than "its vibe sets are longer"?

The richness judge is told to ignore how many vibes a set lists, but the
system ranking can still track set size. This separates the two with a
Bradley-Terry model - logistic regression on pairwise outcomes - with set
size as a covariate (the logistic analogue of ANCOVA):

    logit P(x judged richer than y) = (theta_x - theta_y) + beta * (size_x - size_y)

``theta`` is each system's richness strength at equal set size (the first
system is the reference, fixed at 0); ``beta`` is what one extra vibe is
worth. A ``theta`` gap that survives with ``beta`` in the model is a win the
size difference does not explain. The size-free model is fitted too, so the
shrinkage is visible.

Caveat on reading it: size is not a pure nuisance. If longer sets really
cover more facets, part of a system's true richness advantage flows through
size, and controlling for it removes that part as well. A gap that vanishes
under control means "not distinguishable from length", not "not richer".

Fitted on decisive comparisons only (ties dropped), as in the sign test.
Plain Newton-Raphson; standard errors from the inverse Hessian, Wald p-values.

Usage:
    python -m vibeai.eval.richness_control baseline+v1+workflow_1794469247
"""

import argparse
import json
import math
from dataclasses import dataclass

from vibeai.eval.richness_comparison import RICHNESS_RESULTS_DIR, format_p


def load_decisive(run: str) -> tuple[list[str], list[dict]]:
    """(system labels in run order, decisive comparisons as
    {x, y, size_x, size_y, x_won})."""
    summary = json.loads((RICHNESS_RESULTS_DIR / f"{run}.json").read_text())
    labels = [system["label"] for system in summary["systems_compared"]]
    rows = []
    with (RICHNESS_RESULTS_DIR / f"{run}.per_image.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["winner"] is None:
                continue
            rows.append({
                "x": record["system_x"],
                "y": record["system_y"],
                "size_x": len(record["vibes_x"]),
                "size_y": len(record["vibes_y"]),
                "x_won": record["winner"] == record["system_x"],
            })
    return labels, rows


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting; matrix is small and SPD."""
    n = len(vector)
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for r in range(n):
            if r != col:
                factor = aug[r][col] / aug[col][col]
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col])]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    columns = [_solve(matrix, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[columns[j][i] for j in range(n)] for i in range(n)]


@dataclass
class LogitFit:
    names: list[str]
    coef: list[float]
    cov: list[list[float]]
    log_likelihood: float
    n: int

    def se(self, i: int) -> float:
        return math.sqrt(self.cov[i][i])

    def contrast(self, i: int, j: int) -> tuple[float, float]:
        """(coef_i - coef_j, its standard error)."""
        diff = self.coef[i] - self.coef[j]
        var = self.cov[i][i] + self.cov[j][j] - 2 * self.cov[i][j]
        return diff, math.sqrt(var)


def fit_logit(features: list[list[float]], outcomes: list[bool], names: list[str]) -> LogitFit:
    """Maximum-likelihood logistic regression without an intercept."""
    k = len(names)
    coef = [0.0] * k
    for _ in range(100):
        grad = [0.0] * k
        hess = [[0.0] * k for _ in range(k)]
        for row, y in zip(features, outcomes):
            p = 1 / (1 + math.exp(-sum(c * v for c, v in zip(coef, row))))
            for i in range(k):
                grad[i] += (y - p) * row[i]
                for j in range(k):
                    hess[i][j] += p * (1 - p) * row[i] * row[j]
        step = _solve(hess, grad)
        coef = [c + s for c, s in zip(coef, step)]
        if max(abs(s) for s in step) < 1e-10:
            break
    log_likelihood = 0.0
    for row, y in zip(features, outcomes):
        z = sum(c * v for c, v in zip(coef, row))
        log_likelihood += -math.log1p(math.exp(-z)) if y else -math.log1p(math.exp(z))
    return LogitFit(names, coef, _inverse(hess), log_likelihood, len(outcomes))


def wald_p(estimate: float, se: float) -> float:
    return math.erfc(abs(estimate / se) / math.sqrt(2))


def design(labels: list[str], rows: list[dict], with_size: bool) -> tuple[list[list[float]], list[str]]:
    others = labels[1:]  # labels[0] is the reference, theta fixed at 0
    names = [f"theta_{label}" for label in others] + (["beta_size"] if with_size else [])
    features = []
    for row in rows:
        vector = [float(label == row["x"]) - float(label == row["y"]) for label in others]
        if with_size:
            vector.append(float(row["size_x"] - row["size_y"]))
        features.append(vector)
    return features, names


def report(labels: list[str], fit: LogitFit) -> list[str]:
    lines = [f"  (reference: theta_{labels[0]} = 0)"]
    for i, name in enumerate(fit.names):
        lines.append(
            f"  {name:<18} {fit.coef[i]:+.3f}  (se {fit.se(i):.3f}, "
            f"odds x{math.exp(fit.coef[i]):.2f})  {format_p(wald_p(fit.coef[i], fit.se(i)))}"
        )
    # Every pair of non-reference systems, as a contrast.
    others = labels[1:]
    for a in range(len(others)):
        for b in range(a + 1, len(others)):
            diff, se = fit.contrast(b, a)
            lines.append(
                f"  theta_{others[b]} - theta_{others[a]:<6} {diff:+.3f}  (se {se:.3f}, "
                f"odds x{math.exp(diff):.2f})  {format_p(wald_p(diff, se))}"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", help="results/richness/ run name")
    args = parser.parse_args()

    labels, rows = load_decisive(args.run)
    outcomes = [row["x_won"] for row in rows]
    print(f"{len(rows)} decisive comparisons, systems {labels}\n")

    features, names = design(labels, rows, with_size=False)
    plain = fit_logit(features, outcomes, names)
    features, names = design(labels, rows, with_size=True)
    controlled = fit_logit(features, outcomes, names)

    print("without size:")
    print("\n".join(report(labels, plain)))
    print("\nwith size covariate:")
    print("\n".join(report(labels, controlled)))

    lr = 2 * (controlled.log_likelihood - plain.log_likelihood)
    print(f"\n  likelihood-ratio test for size: chi2(1) = {lr:.2f}, "
          f"{format_p(math.erfc(math.sqrt(lr / 2)))}")


if __name__ == "__main__":
    main()
