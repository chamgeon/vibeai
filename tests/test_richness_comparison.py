"""Offline checks for the pairwise richness metric - no LLM calls."""

import pytest

from vibeai.eval.richness_comparison import binomial_p, summarize
from vibeai.metrics.richness import (
    OrderedJudgement,
    RichnessJudgement,
    _build_prompt,
    _validate_judgement,
    combine_verdicts,
    vibe_set,
)


def test_vibe_set_drops_evidence_and_duplicates():
    details = {
        "atom_details": [
            {"atom": "Rain on glass gives it a hush.", "vibe": "hush", "evidence": ["rain"]},
            {"atom": "The vibe is lonely.", "vibe": "lonely", "evidence": None},
            {"atom": "Quiet hush.", "vibe": "  Hush ", "evidence": None},
        ]
    }
    assert vibe_set(details) == ["hush", "lonely"]


def test_vibe_set_requires_structured_decomposition():
    with pytest.raises(ValueError):
        vibe_set({"atoms": ["The vibe is calm."]})


def test_prompt_contains_both_sets_without_system_names():
    prompt = _build_prompt(["calm", "nostalgic"], ["lonely"])
    assert "- calm\n- nostalgic" in prompt
    assert "- lonely" in prompt
    assert "baseline" not in prompt and "workflow" not in prompt


def test_validation():
    base = (
        '{"image_facets": [], "set_a_analysis": "", "set_b_analysis": "", '
        '"comparison": "", "winner": %s}'
    )
    assert _validate_judgement(base % '"B"')["winner"] == "B"
    assert _validate_judgement(base % '"tie"')["winner"] == "tie"
    with pytest.raises(ValueError):
        _validate_judgement(base % '"draw"')
    with pytest.raises(ValueError):
        _validate_judgement('{"winner": "A"}')


@pytest.mark.parametrize(
    "first, second, expected",
    [
        ("v1", "v1", "v1"),  # agreement
        ("v1", None, "v1"),  # win + tie
        (None, "v1", "v1"),
        (None, None, None),  # two ties
        ("v1", "workflow", None),  # position-driven conflict
    ],
)
def test_combine_verdicts(first, second, expected):
    assert combine_verdicts(first, second) == expected


def test_binomial_p():
    assert binomial_p(5, 10) == 1.0
    assert binomial_p(0, 10) == pytest.approx(2 / 1024)
    assert binomial_p(10, 10) == binomial_p(0, 10)


def _paired(image: str, x_first_pos: str, y_first_pos: str) -> RichnessJudgement:
    """v1 vs workflow, with each order's raw positional verdict."""
    return RichnessJudgement(
        image, "v1", "workflow", [], [],
        OrderedJudgement("v1", "workflow", x_first_pos),
        OrderedJudgement("workflow", "v1", y_first_pos),
    )


def test_ordered_judgement_maps_position_to_system():
    assert OrderedJudgement("workflow", "v1", "A").winner == "workflow"
    assert OrderedJudgement("workflow", "v1", "B").winner == "v1"
    assert OrderedJudgement("workflow", "v1", "tie").winner is None


def test_summarize():
    judgements = [
        _paired("a.jpg", "B", "A"),  # workflow in both orders: consistent win
        _paired("b.jpg", "tie", "A"),  # tie + workflow: workflow win, inconsistent
        _paired("c.jpg", "A", "A"),  # Set A both times: conflict -> tie
        _paired("d.jpg", "A", "B"),  # v1 in both orders
    ]
    pairs, systems, position = summarize(["v1", "workflow"], judgements)
    pair = pairs[0]
    assert (pair["wins_a"], pair["wins_b"], pair["ties"]) == (1, 2, 1)
    assert pair["n_decisive"] == 3 and pair["consistency"] == pytest.approx(0.5)
    assert systems["workflow"] == {
        "wins": 2, "losses": 1, "ties": 1, "comparisons": 4, "win_rate": pytest.approx(2 / 3)
    }
    # 8 calls, 1 tie: Set A won in a.jpg(y), b.jpg(y), c.jpg(x,y), d.jpg(x) = 5 of 7
    assert position["n_calls"] == 8 and position["n_decisive"] == 7
    assert position["wins_a"] == 5
