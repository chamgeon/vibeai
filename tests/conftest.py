import time

import pytest

from vibeai.eval.results import result_log


def pytest_addoption(parser):
    parser.addoption(
        "--n-images",
        default="5",
        help="Number of images to evaluate in batch eval tests, or 'all' for the full dataset.",
    )


@pytest.fixture
def n_images(request) -> int | None:
    """--n-images parsed into load_image_paths' `n` kwarg (None means all)."""
    raw = request.config.getoption("--n-images")
    return None if raw == "all" else int(raw)


def pytest_sessionfinish(session, exitstatus):
    if result_log.records:
        path = result_log.save(f"run_{int(time.time())}")
        print(f"\nSaved {len(result_log.records)} result(s) to {path}")
