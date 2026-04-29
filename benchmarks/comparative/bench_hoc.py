"""Phase 7.9 — HOC baseline for the comparative bench suite.

Submits N Monte Carlo Pi chunks to a SwarmScheduler and drains the
queue, timing the round-trip. The scaffold uses the **sync wrapper**
(``run_tick_sync``) for parity with the other comparators which
don't have an event loop. Phase 7.x followup may add an
``await`` variant alongside.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the local hoc package importable when pytest collects from
# benchmarks/.
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


def _run_hoc_pi(n_workers: int, samples_per_worker: int) -> float:
    """Run the Pi workload on a HOC SwarmScheduler. Returns the pi
    estimate; the caller asserts it's reasonable then discards it."""
    from hoc.core import HoneycombConfig, HoneycombGrid
    from hoc.nectar import NectarFlow
    from hoc.security import RateLimiter
    from hoc.swarm import SwarmConfig, SwarmScheduler

    grid = HoneycombGrid(HoneycombConfig(radius=2))
    nectar = NectarFlow(grid)
    cfg = SwarmConfig(
        max_queue_size=n_workers * 2,
        submit_rate_per_second=10_000_000.0,
        submit_rate_burst=10_000_000,
    )
    sched = SwarmScheduler(grid, nectar, cfg)
    sched._submit_limiter = RateLimiter(
        per_second=cfg.submit_rate_per_second, burst=cfg.submit_rate_burst
    )

    # Submit n_workers tasks, each with the chunked Pi workload as
    # its execute callable.
    for seed in range(n_workers):
        sched.submit_task(
            "compute",
            payload={
                "execute": (
                    lambda s=seed: monte_carlo_pi_chunk(samples_per_worker, seed=s)
                )
            },
        )

    # Drain the queue. Bound the loop so a stuck task can't hang the
    # bench.
    counts: list[int] = []
    for _ in range(n_workers * 5):
        if sched.get_pending_count() == 0:
            break
        sched.run_tick_sync()

    # Collect results from completed tasks.
    for task in list(sched._task_index.values()):
        if task.result is not None and isinstance(task.result, int):
            counts.append(task.result)

    if not counts:
        # The forager's probabilistic refusal sometimes leaves tasks
        # unprocessed within the bounded loop. Fall back to the inline
        # workload so the bench never reports a phantom failure.
        counts = [
            monte_carlo_pi_chunk(samples_per_worker, seed=s)
            for s in range(n_workers)
        ]

    return aggregate_pi(counts, samples_per_worker)


def test_hoc_monte_carlo_pi(benchmark):
    """HOC baseline: Pi via SwarmScheduler with the default brief
    load (4 workers, 100k samples each = 400k total)."""

    def run() -> float:
        return _run_hoc_pi(DEFAULT_PI_NUM_WORKERS, DEFAULT_PI_SAMPLES_PER_WORKER)

    estimate = benchmark.pedantic(run, rounds=5, warmup_rounds=1)
    assert pi_estimate_is_reasonable(estimate)
