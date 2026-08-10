"""Decomposition: vibe representation -> list of atomic vibe claims."""

from vibeai.eval.parsing import extract_json
from vibeai.llm.client import DEFAULT_MODEL, call_text, call_text_async
from vibeai.prompts.decomposition import PROMPTS


def _extract_and_validate_atoms(raw: str) -> list[str]:
    """Parse + validate the decomposer's JSON array of atoms. Raises
    ValueError on any failure (not a JSON array, empty, non-string or
    empty/whitespace-only entries) - used both as call_text's
    retry-triggering ``validate`` callback and to build the return value
    once a call has passed validation."""
    atoms = extract_json(raw)
    if not isinstance(atoms, list):
        raise ValueError(f"Expected a JSON array of atoms, got: {raw!r}")
    if not atoms:
        raise ValueError(f"Decomposer returned an empty atom list: {raw!r}")
    for i, atom in enumerate(atoms):
        if not isinstance(atom, str) or not atom.strip():
            raise ValueError(f"Atom {i} is not a non-empty string: {atom!r}")
    return atoms


def decompose(
    representation: str,
    prompt_version: str = "baseline",
    model: str = DEFAULT_MODEL,
) -> list[str]:
    prompt = PROMPTS[prompt_version].format(representation=representation)
    raw = call_text(
        prompt, model=model, call_type="decompose", validate=_extract_and_validate_atoms
    )
    return _extract_and_validate_atoms(raw)


async def decompose_async(
    representation: str,
    prompt_version: str = "baseline",
    model: str = DEFAULT_MODEL,
) -> list[str]:
    prompt = PROMPTS[prompt_version].format(representation=representation)
    raw = await call_text_async(
        prompt, model=model, call_type="decompose", validate=_extract_and_validate_atoms
    )
    return _extract_and_validate_atoms(raw)
