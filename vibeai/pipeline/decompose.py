"""Decomposition: vibe representation -> list of atomic vibe claims."""

from vibeai.eval.parsing import extract_json
from vibeai.llm.client import DEFAULT_MODEL, call_text, call_text_async
from vibeai.prompts.decomposition import PROMPTS

_REQUIRED_REPRESENTATION_KEYS = {
    "vibe_description",
    "vibe_decomposition",
    "contradiction_scan",
    "removal_test",
    "final_representation",
}


def decompose_direct(representation: str) -> list[str]:
    """Extract atoms directly from a representation prompt (e.g. v2) that
    already decomposes the vibe itself, instead of running a separate
    decomposition LLM call. ``representation`` is the raw JSON text produced
    by such a prompt; its ``final_representation`` list of
    ``{"vibe": ..., "evidence": ...}`` pairs is turned into atom sentences in
    the same style ``decompose()`` produces, so downstream metrics (which
    infer atom type from the sentence itself) see the same shape of input.
    """
    parsed = extract_json(representation)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object with a final_representation, got: {representation!r}")
    missing = _REQUIRED_REPRESENTATION_KEYS - parsed.keys()
    if missing:
        raise ValueError(f"Representation missing key(s) {sorted(missing)}: {representation!r}")

    final = parsed["final_representation"]
    if not isinstance(final, list) or not final:
        raise ValueError(f"final_representation must be a non-empty list, got: {final!r}")

    atoms = []
    for i, item in enumerate(final):
        if not isinstance(item, dict) or "vibe" not in item or "evidence" not in item:
            raise ValueError(f"final_representation[{i}] missing vibe/evidence: {item!r}")
        vibe = str(item["vibe"]).strip()
        evidence = str(item["evidence"]).strip()
        if not vibe:
            raise ValueError(f"final_representation[{i}] has empty vibe: {item!r}")
        if evidence and evidence.lower() != "none":
            atoms.append(f"{evidence}, giving it a {vibe} vibe.")
        else:
            atoms.append(f"The vibe is {vibe}.")
    return atoms


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
