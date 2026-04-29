"""Phase 7.9 — multiprocessing.Pool comparator (stdlib, always
available)."""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

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


def _run_mp_pi(n_workers: int, samples_per_worker: int) -> float:
    """Run the Pi workload on a multiprocessing.Pool. Returns the pi
    estimate."""
    # ``spawn`` for cross-platform parity with the sandbox path. The
    # workload module is top-level inside ``benchmarks.comparative``
    # so the spawn child can re-import it.
    ctx = mp.get_context("spawn")
    args = [(samples_per_worker, seed) for seed in range(n_workers)]
    with ctx.Pool(processes=n_workers) as pool:
        counts = pool.starmap(monte_carlo_pi_chunk, args)
    return aggregate_pi(list(counts), samples_per_worker)


def test_mp_monte_carlo_pi(benchmark):
    def run() -> float:
        return _run_mp_pi(DEFAULT_PI_NUM_WORKERS, DEFAULT_PI_SAMPLES_PER_WORKER)

    estimate = benchmark.pedantic(run, rounds=3, warmup_rounds=1)
    assert pi_estimate_is_reasonable(estimate)
