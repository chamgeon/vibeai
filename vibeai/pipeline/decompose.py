"""Decomposition: vibe representation -> list of atomic vibe claims."""

from vibeai.eval.parsing import extract_json
from vibeai.llm.client import DEFAULT_MODEL, call_text
from vibeai.prompts.decomposition import PROMPTS


def decompose(
    representation: str,
    prompt_version: str = "baseline",
    model: str = DEFAULT_MODEL,
) -> list[str]:
    prompt = PROMPTS[prompt_version].format(representation=representation)
    raw = call_text(prompt, model=model)
    atoms = extract_json(raw)
    if not isinstance(atoms, list):
        raise ValueError(f"Expected a JSON array of atoms, got: {raw!r}")
    return [str(atom) for atom in atoms]
