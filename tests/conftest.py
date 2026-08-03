import time

from vibeai.eval.results import result_log


def pytest_sessionfinish(session, exitstatus):
    if result_log.records:
        path = result_log.save(f"run_{int(time.time())}")
        print(f"\nSaved {len(result_log.records)} result(s) to {path}")
