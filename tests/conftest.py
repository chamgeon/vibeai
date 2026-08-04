import time
from pathlib import Path

import pytest

from vibeai.eval.results import result_log


def pytest_addoption(parser):
    parser.addoption(
        "--n-images",
        default="5",
        help="Number of images to evaluate in batch eval tests, or 'all' for the full dataset.",
    )
    parser.addoption(
        "--image-dir",
        default=None,
        help="Directory of images to evaluate in batch eval tests. Defaults to data/main_processed.",
    )
    parser.addoption(
        "--representation-prompt-version",
        default="baseline",
        help="Representation prompt version to evaluate in batch eval tests.",
    )
    parser.addoption(
        "--decomposition-prompt-version",
        default="baseline",
        help="Decomposition prompt version to evaluate in batch eval tests.",
    )
    parser.addoption(
        "--concurrency",
        default="30",
        help="Max concurrent evaluations in batch eval tests.",
    )


@pytest.fixture
def n_images(request) -> int | None:
    """--n-images parsed into load_image_paths' `n` kwarg (None means all)."""
    raw = request.config.getoption("--n-images")
    return None if raw == "all" else int(raw)


@pytest.fixture
def image_dir(request) -> Path | None:
    """--image-dir parsed into load_image_paths' `data_dir` kwarg (None means default)."""
    raw = request.config.getoption("--image-dir")
    return None if raw is None else Path(raw)


@pytest.fixture
def representation_prompt_version(request) -> str:
    return request.config.getoption("--representation-prompt-version")


@pytest.fixture
def decomposition_prompt_version(request) -> str:
    return request.config.getoption("--decomposition-prompt-version")


@pytest.fixture
def concurrency(request) -> int:
    return int(request.config.getoption("--concurrency"))


def pytest_sessionfinish(session, exitstatus):
    if result_log.records:
        path = result_log.save(f"run_{int(time.time())}")
        print(f"\nSaved {len(result_log.records)} result(s) to {path}")
