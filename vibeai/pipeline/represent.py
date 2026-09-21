"""Vibe-representation generation: image -> natural-language vibe description."""

from pathlib import Path

from vibeai.eval.parsing import extract_json
from vibeai.llm.client import DEFAULT_MODEL, call_with_image, call_with_image_async
from vibeai.prompts.representation import PROMPTS

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

# Prompt versions that wrap the representation in their own reasoning - "dr"
# emits its draft and its two self-checks alongside the final paragraph. Only
# the final paragraph is the representation, so it's unwrapped here: everything
# downstream (decomposition, the judges, the annotation webapp) consumes this
# return value as the representation itself, and would otherwise read the
# draft and the critique as part of it.
#
# v2 is deliberately not in this set. Its JSON *is* what the pipeline wants,
# because DIRECT_DECOMPOSITION reads final_representation's vibe/evidence
# pairs straight out of it (see pipeline.decompose.decompose_direct).
WRAPPED_PROMPT_VERSIONS = {"dr"}


def _unwrap_representation(raw: str) -> str:
    """Pull the final paragraph out of a wrapped prompt's JSON. Raises
    ValueError on any failure - used both as call_with_image's retry-triggering
    ``validate`` callback and to build the return value once a call has passed
    validation."""
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got: {raw!r}")
    final = parsed.get("final_representation")
    if not isinstance(final, str) or not final.strip():
        raise ValueError(f"final_representation missing or not a non-empty string: {raw!r}")
    return final.strip()


def generate_representation(
    image_path: Path,
    prompt_version: str = "baseline",
    model: str = DEFAULT_MODEL,
) -> str:
    image_path = Path(image_path)
    prompt = PROMPTS[prompt_version]
    mime_type = MIME_TYPES.get(image_path.suffix.lower(), "image/jpeg")
    image_bytes = image_path.read_bytes()
    wrapped = prompt_version in WRAPPED_PROMPT_VERSIONS
    raw = call_with_image(
        prompt,
        image_bytes,
        mime_type=mime_type,
        model=model,
        call_type="represent",
        validate=_unwrap_representation if wrapped else None,
    )
    return _unwrap_representation(raw) if wrapped else raw


async def generate_representation_async(
    image_path: Path,
    prompt_version: str = "baseline",
    model: str = DEFAULT_MODEL,
) -> str:
    image_path = Path(image_path)
    prompt = PROMPTS[prompt_version]
    mime_type = MIME_TYPES.get(image_path.suffix.lower(), "image/jpeg")
    image_bytes = image_path.read_bytes()
    wrapped = prompt_version in WRAPPED_PROMPT_VERSIONS
    raw = await call_with_image_async(
        prompt,
        image_bytes,
        mime_type=mime_type,
        model=model,
        call_type="represent",
        validate=_unwrap_representation if wrapped else None,
    )
    return _unwrap_representation(raw) if wrapped else raw
