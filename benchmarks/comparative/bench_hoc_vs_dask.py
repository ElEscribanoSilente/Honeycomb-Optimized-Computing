"""Phase 7.9 — Dask comparator. Skips cleanly if Dask isn't installed.

Install Dask to run::

    pip install "dask[distributed]>=2024.0"
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.comparative._workloads import (
    DEFAULT_PI_NUM_WORKERS,
    DEFAULT_PI_SAMPLES_PER_WORKER,
    aggregate_pi,
    monte_carlo_pi_chunk,
    pi_estimate_is_reasonable,
)

_HAS_DASK = (
    importlib.util.find_spec("dask") is not None
    and importlib.util.find_spec("dask.distributed") is not None
)
pytestmark = pytest.mark.skipif(
    not _HAS_DASK,
    reason="Dask not installed; pip install 'dask[distributed]' to run",
)


def _run_dask_pi(n_workers: int, samples_per_worker: int) -> float:
    from dask.distributed import Client

    with Client(
        n_workers=n_workers, threads_per_worker=1, processes=True, dashboard_address=None
    ) as client:
        futures = [
            client.submit(monte_carlo_pi_chunk, samples_per_worker, seed)
            for seed in range(n_workers)
        ]
        counts = client.gather(futures)
    return aggregate_pi(list(counts), samples_per_worker)


def test_dask_monte_carlo_pi(benchmark):
    def run() -> float:
        return _run_dask_pi(DEFAULT_PI_NUM_WORKERS, DEFAULT_PI_SAMPLES_PER_WORKER)

    estimate = benchmark.pedantic(run, rounds=3, warmup_rounds=1)
    assert pi_estimate_is_reasonable(estimate)
