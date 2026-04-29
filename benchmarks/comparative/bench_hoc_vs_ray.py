"""Phase 7.9 — Ray comparator. Skips cleanly if Ray isn't installed.

Install Ray to run::

    pip install ray>=2.0
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

# Skip the whole module if Ray isn't installed. ``find_spec`` is the
# recommended idiom — it doesn't import Ray, just checks whether the
# package is on sys.path.
_HAS_RAY = importlib.util.find_spec("ray") is not None
pytestmark = pytest.mark.skipif(
    not _HAS_RAY, reason="Ray not installed; pip install ray to run"
)


def _run_ray_pi(n_workers: int, samples_per_worker: int) -> float:
    import ray

    if not ray.is_initialized():
        ray.init(num_cpus=n_workers, log_to_driver=False, ignore_reinit_error=True)

    @ray.remote
    def _chunk(s: int) -> int:
        return monte_carlo_pi_chunk(samples_per_worker, seed=s)

    futures = [_chunk.remote(seed) for seed in range(n_workers)]
    counts = ray.get(futures)
    return aggregate_pi(list(counts), samples_per_worker)


def test_ray_monte_carlo_pi(benchmark):
    def run() -> float:
        return _run_ray_pi(DEFAULT_PI_NUM_WORKERS, DEFAULT_PI_SAMPLES_PER_WORKER)

    estimate = benchmark.pedantic(run, rounds=3, warmup_rounds=1)
    assert pi_estimate_is_reasonable(estimate)
