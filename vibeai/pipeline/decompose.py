"""Decomposition: vibe representation -> list of atomic vibe claims."""

from vibeai.eval.parsing import extract_json
from vibeai.llm.client import DEFAULT_MODEL, call_text, call_text_async
from vibeai.prompts.decomposition import PROMPTS


def decompose(
    representation: str,
    prompt_version: str = "baseline",
    model: str = DEFAULT_MODEL,
) -> list[str]:
    prompt = PROMPTS[prompt_version].format(representation=representation)
    raw = call_text(prompt, model=model, call_type="decompose")
    atoms = extract_json(raw)
    if not isinstance(atoms, list):
        raise ValueError(f"Expected a JSON array of atoms, got: {raw!r}")
    return [str(atom) for atom in atoms]


async def decompose_async(
    representation: str,
    prompt_version: str = "baseline",
    model: str = DEFAULT_MODEL,
) -> list[str]:
    prompt = PROMPTS[prompt_version].format(representation=representation)
    raw = await call_text_async(prompt, model=model, call_type="decompose")
    atoms = extract_json(raw)
    if not isinstance(atoms, list):
        raise ValueError(f"Expected a JSON array of atoms, got: {raw!r}")
    return [str(atom) for atom in atoms]
