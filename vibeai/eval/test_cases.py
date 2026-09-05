"""Test case containers passed between the pipeline and metrics."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DecompositionTestCase:
    image_path: Path
    representation: str
    atoms: list[str]
    representation_prompt_version: str = "baseline"
    decomposition_prompt_version: str = "baseline"
    # Per-atom {atom, type, evidence, vibe} objects, for decomposition prompts
    # that emit them (see decompose.STRUCTURED_PROMPT_VERSIONS); None
    # otherwise. Parallel to ``atoms``, which holds the same atom sentences.
    atom_details: list[dict] | None = None
