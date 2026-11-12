"""Richness metric: which of two systems' vibe sets captures more of an image's
vibe, decided head to head by an LLM judge.

Richness has no absolute reference - there is no answer key listing every vibe
an image carries - so it is measured relatively. For one image and two
representation methods (systems), the judge sees the image and both systems'
vibe sets and picks the richer one, or calls a tie. Aggregating those verdicts
over an image set says which method is richer; ``eval/richness_comparison.py``
runs that.

Design choices:

*   **Bare vibes, evidence omitted.** The sets are the ``vibe`` fields of the
    structured (v2) decomposition, not the atom sentences, so the judge
    compares what each method felt rather than how it argued for it.
    Evidence phrasing would otherwise read as richness.
*   **Both orders.** Every comparison is judged twice, once with each system
    as Set A, and the two verdicts are combined (``combine_verdicts``). That
    cancels positional bias per image rather than only on average, and the
    share of comparisons whose verdict flips with position is a direct read on
    how far the judge can be trusted.
*   **Cross-family judge.** The representations come from gpt models, so the
    judge defaults to Claude, to keep self-preference out of the verdict.
*   **Richness only.** The prompt tells the judge to treat every vibe as true
    (plausibility is its own metric) and not to count vibes or words.

This does not subclass ``Metric``: a verdict belongs to a pair of test cases,
not to one, and there is no per-image score to threshold.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vibeai.eval.parsing import extract_json
from vibeai.llm.client import DEFAULT_ANTHROPIC_MODEL, call_with_image, call_with_image_async
from vibeai.pipeline.represent import MIME_TYPES
from vibeai.prompts.richness_eval import RICHNESS_COMPARISON_PROMPT

DEFAULT_RICHNESS_MODEL = DEFAULT_ANTHROPIC_MODEL

_VERDICTS = ("A", "B", "tie")
_REQUIRED_KEYS = {"image_facets", "set_a_analysis", "set_b_analysis", "comparison", "winner"}


def vibe_set(details: dict) -> list[str]:
    """A system's vibe set for one image, from a batch eval's per-image
    ``details``: the ``vibe`` field of each structured atom, deduplicated
    case- and whitespace-insensitively, in order.

    Raises ValueError for runs without ``atom_details`` - their atoms are
    sentences with the evidence baked in, which is exactly what the
    comparison has to leave out.
    """
    atom_details = details.get("atom_details")
    if not atom_details:
        raise ValueError(
            "record has no atom_details; richness needs a structured (v2) decomposition"
        )
    seen = set()
    vibes = []
    for detail in atom_details:
        vibe = str(detail.get("vibe", "")).strip()
        key = " ".join(vibe.lower().split())
        if vibe and key not in seen:
            seen.add(key)
            vibes.append(vibe)
    return vibes


@dataclass
class OrderedJudgement:
    """One judge call, with one system shown as Set A and the other as Set B."""

    system_a: str
    system_b: str
    winner_position: str  # "A" | "B" | "tie"
    judgement: dict[str, Any] = field(default_factory=dict)

    @property
    def winner(self) -> str | None:
        """The system judged richer, or None for a tie."""
        if self.winner_position == "tie":
            return None
        return self.system_a if self.winner_position == "A" else self.system_b


def combine_verdicts(first: str | None, second: str | None) -> str | None:
    """One verdict from the two orders' winners (None = tie).

    Agreement, or a win in one order and a tie in the other, is a win. Two
    ties are a tie, and so are two different winners: a verdict that follows
    position rather than content says the sets are too close for this judge.
    """
    if first == second:
        return first
    if first is None:
        return second
    if second is None:
        return first
    return None


@dataclass
class RichnessJudgement:
    """Both orders of one comparison, and the verdict they combine to."""

    image_path: str
    system_x: str
    system_y: str
    vibes_x: list[str]
    vibes_y: list[str]
    x_first: OrderedJudgement  # system_x shown as Set A
    y_first: OrderedJudgement  # system_y shown as Set A

    @property
    def winner(self) -> str | None:
        return combine_verdicts(self.x_first.winner, self.y_first.winner)

    @property
    def consistent(self) -> bool:
        """Both orders reached the same verdict (including both ties)."""
        return self.x_first.winner == self.y_first.winner

    @property
    def orders(self) -> tuple[OrderedJudgement, OrderedJudgement]:
        return self.x_first, self.y_first


def _format_set(vibes: list[str]) -> str:
    return "\n".join(f"- {vibe}" for vibe in vibes) if vibes else "(empty)"


def _build_prompt(vibes_a: list[str], vibes_b: list[str]) -> str:
    return RICHNESS_COMPARISON_PROMPT.format(
        vibes_a=_format_set(vibes_a), vibes_b=_format_set(vibes_b)
    )


def _load_image(image_path: Path) -> tuple[bytes, str]:
    mime_type = MIME_TYPES.get(image_path.suffix.lower(), "image/jpeg")
    return image_path.read_bytes(), mime_type


def _validate_judgement(raw: str) -> dict:
    """Parse + validate the judge's JSON. Raises ValueError on any failure -
    used both as the retry-triggering ``validate`` callback and to read the
    verdict once a call has passed."""
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got: {raw!r}")
    missing = _REQUIRED_KEYS - parsed.keys()
    if missing:
        raise ValueError(f"Judgement missing key(s) {sorted(missing)}: {raw!r}")
    if parsed["winner"] not in _VERDICTS:
        raise ValueError(f"winner must be one of {_VERDICTS}: {parsed['winner']!r}")
    return parsed


class RichnessComparison:
    name = "richness"

    def __init__(self, model: str = DEFAULT_RICHNESS_MODEL):
        self.model = model

    def _judge(
        self, image: tuple[bytes, str], system_a: str, vibes_a: list[str],
        system_b: str, vibes_b: list[str],
    ) -> OrderedJudgement:
        image_bytes, mime_type = image
        raw = call_with_image(
            _build_prompt(vibes_a, vibes_b),
            image_bytes,
            mime_type=mime_type,
            model=self.model,
            call_type="judge",
            validate=_validate_judgement,
        )
        parsed = _validate_judgement(raw)
        return OrderedJudgement(system_a, system_b, parsed["winner"], parsed)

    async def _judge_async(
        self, image: tuple[bytes, str], system_a: str, vibes_a: list[str],
        system_b: str, vibes_b: list[str],
    ) -> OrderedJudgement:
        image_bytes, mime_type = image
        raw = await call_with_image_async(
            _build_prompt(vibes_a, vibes_b),
            image_bytes,
            mime_type=mime_type,
            model=self.model,
            call_type="judge",
            validate=_validate_judgement,
        )
        parsed = _validate_judgement(raw)
        return OrderedJudgement(system_a, system_b, parsed["winner"], parsed)

    @staticmethod
    def _check(system_x: str, system_y: str) -> None:
        if system_x == system_y:
            raise ValueError(f"cannot compare system {system_x!r} with itself")

    def compare(
        self, image_path: Path, system_x: str, vibes_x: list[str], system_y: str, vibes_y: list[str]
    ) -> RichnessJudgement:
        self._check(system_x, system_y)
        image = _load_image(image_path)
        return RichnessJudgement(
            str(image_path), system_x, system_y, vibes_x, vibes_y,
            self._judge(image, system_x, vibes_x, system_y, vibes_y),
            self._judge(image, system_y, vibes_y, system_x, vibes_x),
        )

    async def compare_async(
        self, image_path: Path, system_x: str, vibes_x: list[str], system_y: str, vibes_y: list[str]
    ) -> RichnessJudgement:
        self._check(system_x, system_y)
        image = _load_image(image_path)
        x_first, y_first = await asyncio.gather(
            self._judge_async(image, system_x, vibes_x, system_y, vibes_y),
            self._judge_async(image, system_y, vibes_y, system_x, vibes_x),
        )
        return RichnessJudgement(
            str(image_path), system_x, system_y, vibes_x, vibes_y, x_first, y_first
        )
