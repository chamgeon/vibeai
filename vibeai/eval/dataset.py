"""Golden dataset loader: currently just the preprocessed image pool."""

import random
from pathlib import Path

DATA_DIR = Path("data/main_processed")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_image_paths(n: int | None = None, seed: int = 0) -> list[Path]:
    paths = sorted(
        p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS
    )
    if n is not None and n < len(paths):
        paths = random.Random(seed).sample(paths, n)
    return paths
