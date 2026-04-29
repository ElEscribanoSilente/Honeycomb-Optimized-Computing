"""Phase 7.5 — :class:`SwarmScheduler` throughput bench at 1000-task load.

Specifically targets the O(n·m) → O(m·log n) tick path introduced in
Phase 7.5 (BehaviorIndex). Phase 6 baseline iterated every pending
task for every behavior; the new path pops one task per behavior heap
in O(log n).

Configurable knobs:

- ``radius`` — HoneycombGrid radius. Phase 7.5 brief asks for radius=3
  (37 cells, ~22 workers given ratios). Larger radius → more
  behaviors → larger ``m``.
- ``num_tasks`` — initial queue depth. Brief asks for 1000.

Run via pytest-benchmark::

    pytest benchmarks/bench_swarm_1000_tasks.py -v --benchmark-only \\
        --benchmark-warmup=on --benchmark-min-time=0.5

The bench fixture rebuilds a fresh scheduler per round so the
queue-depth invariant holds. Setup happens outside ``benchmark`` (only
the tick loop is timed).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure the local package is importable when pytest collects from
# ``benchmarks/``.
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


def _build_loaded_scheduler(radius: int = 3, num_tasks: int = 1000):
    """Return a scheduler with ``num_tasks`` queued. Rate limiter is
    relaxed so the bench isn't bottlenecked on token-bucket math."""
    from hoc.core import HoneycombConfig, HoneycombGrid
    from hoc.nectar import NectarFlow
    from hoc.security import RateLimiter
    from hoc.swarm import SwarmConfig, SwarmScheduler

    grid = HoneycombGrid(HoneycombConfig(radius=radius))
    nectar = NectarFlow(grid)
    cfg = SwarmConfig(
        max_queue_size=num_tasks * 2,
        submit_rate_per_second=10_000_000.0,
        submit_rate_burst=10_000_000,
    )
    sched = SwarmScheduler(grid, nectar, cfg)
    sched._submit_limiter = RateLimiter(
        per_second=cfg.submit_rate_per_second, burst=cfg.submit_rate_burst
    )
    for _ in range(num_tasks):
        sched.submit_task("compute", {})
    return sched


def test_swarm_1000_tasks_single_tick(benchmark):
    """Phase 7.5 target: ≥5× speedup vs Phase 6 baseline.

    A single ``tick()`` on a radius-3 grid loaded with 1000 tasks. The
    pre-Phase-7.5 path scanned 1000 tasks for every behavior (~22 of
    them in radius=3) on each tick, ~22k filter ops. The new path
    pops one task per behavior heap in O(log n) ≈ 10, ~220 ops.
    """

    def setup():
        # Fresh scheduler each round so queue depth resets.
        return (_build_loaded_scheduler(radius=3, num_tasks=1000),), {}

    def run(sched):
        sched.run_tick_sync()

    benchmark.pedantic(run, setup=setup, rounds=20, warmup_rounds=2)


def test_swarm_1000_tasks_drain_25_ticks(benchmark):
    """Higher-signal bench: 25 ticks on the same load. Cumulative
    measurement makes log(n) wins more visible than a single tick.
    """

    def setup():
        return (_build_loaded_scheduler(radius=3, num_tasks=1000),), {}

    def run(sched):
        for _ in range(25):
            sched.run_tick_sync()

    benchmark.pedantic(run, setup=setup, rounds=10, warmup_rounds=1)


def test_swarm_500_tasks_radius2(benchmark):
    """Smaller grid (~12 worker cells) + 500 tasks — useful to confirm
    the win scales below the brief's headline load."""

    def setup():
        return (_build_loaded_scheduler(radius=2, num_tasks=500),), {}

    def run(sched):
        sched.run_tick_sync()

    benchmark.pedantic(run, setup=setup, rounds=20, warmup_rounds=2)
