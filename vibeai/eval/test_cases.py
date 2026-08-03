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
