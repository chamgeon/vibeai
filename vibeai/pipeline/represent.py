"""Vibe-representation generation: image -> natural-language vibe description."""

from pathlib import Path

from vibeai.llm.client import DEFAULT_MODEL, call_with_image, call_with_image_async
from vibeai.prompts.representation import PROMPTS

_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def generate_representation(
    image_path: Path,
    prompt_version: str = "baseline",
    model: str = DEFAULT_MODEL,
) -> str:
    image_path = Path(image_path)
    prompt = PROMPTS[prompt_version]
    mime_type = _MIME_TYPES.get(image_path.suffix.lower(), "image/jpeg")
    image_bytes = image_path.read_bytes()
    return call_with_image(prompt, image_bytes, mime_type=mime_type, model=model)


async def generate_representation_async(
    image_path: Path,
    prompt_version: str = "baseline",
    model: str = DEFAULT_MODEL,
) -> str:
    image_path = Path(image_path)
    prompt = PROMPTS[prompt_version]
    mime_type = _MIME_TYPES.get(image_path.suffix.lower(), "image/jpeg")
    image_bytes = image_path.read_bytes()
    return await call_with_image_async(prompt, image_bytes, mime_type=mime_type, model=model)
