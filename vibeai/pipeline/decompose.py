"""Decomposition: vibe representation -> list of atomic vibe claims."""

from vibeai.eval.parsing import extract_json
from vibeai.llm.client import DEFAULT_DECOMPOSITION_MODEL, call_text, call_text_async
from vibeai.prompts.decomposition import PROMPTS

# Decomposition prompt versions whose output is an array of atom *objects*
# ({atom, type, evidence, vibe}) rather than a flat array of atom sentences.
# The extra structure is carried alongside the sentences (see
# DecompositionTestCase.atom_details) so downstream metrics, which all consume
# ``atoms`` as plain strings, keep working unchanged.
STRUCTURED_PROMPT_VERSIONS = {"v2"}

_ATOM_TYPES = {"vibe_only", "evidence_backed"}
_REQUIRED_ATOM_KEYS = {"atom", "type", "evidence", "vibe"}

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


def _validate_atom_object(obj, index: int) -> None:
    """Check one atom object against the v2 output schema. Raises ValueError
    describing the first violation found."""
    if not isinstance(obj, dict):
        raise ValueError(f"Atom {index} is not a JSON object: {obj!r}")
    missing = _REQUIRED_ATOM_KEYS - obj.keys()
    if missing:
        raise ValueError(f"Atom {index} missing key(s) {sorted(missing)}: {obj!r}")

    for key in ("atom", "vibe"):
        if not isinstance(obj[key], str) or not obj[key].strip():
            raise ValueError(f"Atom {index} has empty/invalid {key}: {obj!r}")

    if obj["type"] not in _ATOM_TYPES:
        raise ValueError(
            f"Atom {index} type must be one of {sorted(_ATOM_TYPES)}: {obj!r}"
        )

    # evidence is null for vibe_only and a non-empty list of non-empty strings
    # for evidence_backed - the prompt states this as a hard rule, so a
    # mismatch means the decomposer misclassified the atom.
    evidence = obj["evidence"]
    if obj["type"] == "vibe_only":
        if evidence is not None:
            raise ValueError(f"Atom {index} is vibe_only but has evidence: {obj!r}")
    else:
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(
                f"Atom {index} is evidence_backed but evidence is not a non-empty list: {obj!r}"
            )
        for j, cue in enumerate(evidence):
            if not isinstance(cue, str) or not cue.strip():
                raise ValueError(
                    f"Atom {index} evidence[{j}] is not a non-empty string: {obj!r}"
                )


def _extract_and_validate_atom_objects(raw: str) -> list[dict]:
    """Parse + validate a structured decomposer's JSON array of atom objects.
    Raises ValueError on any failure - used both as call_text's
    retry-triggering ``validate`` callback and to build the return value once
    a call has passed validation."""
    objects = extract_json(raw)
    if not isinstance(objects, list):
        raise ValueError(f"Expected a JSON array of atom objects, got: {raw!r}")
    if not objects:
        raise ValueError(f"Decomposer returned an empty atom list: {raw!r}")
    for i, obj in enumerate(objects):
        _validate_atom_object(obj, i)
    return objects


def _parse_decomposition(raw: str, prompt_version: str) -> tuple[list[str], list[dict] | None]:
    """Turn a decomposer's raw output into (atom sentences, atom details).
    ``atom details`` is None for prompt versions that emit plain strings."""
    if prompt_version in STRUCTURED_PROMPT_VERSIONS:
        objects = _extract_and_validate_atom_objects(raw)
        return [obj["atom"] for obj in objects], objects
    return _extract_and_validate_atoms(raw), None


def _validator_for(prompt_version: str):
    if prompt_version in STRUCTURED_PROMPT_VERSIONS:
        return _extract_and_validate_atom_objects
    return _extract_and_validate_atoms


def decompose(
    representation: str,
    prompt_version: str = "baseline",
    model: str = DEFAULT_DECOMPOSITION_MODEL,
) -> tuple[list[str], list[dict] | None]:
    prompt = PROMPTS[prompt_version].format(representation=representation)
    raw = call_text(
        prompt, model=model, call_type="decompose", validate=_validator_for(prompt_version)
    )
    return _parse_decomposition(raw, prompt_version)


async def decompose_async(
    representation: str,
    prompt_version: str = "baseline",
    model: str = DEFAULT_DECOMPOSITION_MODEL,
) -> tuple[list[str], list[dict] | None]:
    prompt = PROMPTS[prompt_version].format(representation=representation)
    raw = await call_text_async(
        prompt, model=model, call_type="decompose", validate=_validator_for(prompt_version)
    )
    return _parse_decomposition(raw, prompt_version)
