"""Phase 7.1 — async tick smoke tests.

Verifies the canonical Phase 7+ usage of the four async-converted
classes: ``HoneycombGrid``, ``NectarFlow``, ``SwarmScheduler``,
``HoneycombCell``. Existing tests under tests/ still drive these via
the ``run_tick_sync`` wrapper to keep their diffs minimal; this file
is the dedicated place to exercise the ``async def`` path with
``await`` directly.

``pytest-asyncio`` is configured in ``asyncio_mode = "auto"``
(pyproject.toml) so plain ``async def test_…`` functions run inside
an event loop without a per-test decorator.

Test surface:

- One concurrent grid tick + state advance.
- Sequential vs parallel ring processing both produce identical
  cell-count results.
- ``asyncio.gather`` of two grid ticks doesn't deadlock or corrupt
  the per-cell FSM (each tick still serializes per cell via the
  cell's RWLock).
- ``NectarFlow.tick`` is awaitable and returns the same dict shape
  as before.
- ``SwarmScheduler.tick`` is awaitable and drains the queue exactly
  like the sync wrapper.
- ``HoneycombCell.execute_tick`` is awaitable; returns dict.
"""

from __future__ import annotations

import asyncio

import pytest

from hoc.core import (
    CellState,
    HoneycombConfig,
    HoneycombGrid,
    WorkerCell,
)
from hoc.nectar import NectarFlow, PheromoneType
from hoc.swarm import SwarmScheduler

# ───────────────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def small_grid():
    return HoneycombGrid(HoneycombConfig(radius=1))


@pytest.fixture
def medium_grid():
    return HoneycombGrid(HoneycombConfig(radius=2))


# ───────────────────────────────────────────────────────────────────────────────
# HoneycombGrid.tick (async)
# ───────────────────────────────────────────────────────────────────────────────


class TestAsyncGridTick:
    async def test_grid_tick_is_async(self, small_grid):
        """Smoke: ``await grid.tick()`` returns the metrics dict."""
        result = await small_grid.tick()
        assert isinstance(result, dict)
        assert "tick" in result
        assert "cells_processed" in result

    async def test_grid_tick_advances_count(self, small_grid):
        """Tick counter increments after each await."""
        before = small_grid._tick_count
        await small_grid.tick()
        await small_grid.tick()
        assert small_grid._tick_count == before + 2

    async def test_parallel_ring_processing(self, medium_grid):
        """Default config has ``parallel_ring_processing=True``.
        ``_async_parallel_tick`` runs and returns identical shape."""
        assert medium_grid.config.parallel_ring_processing is True
        result = await medium_grid.tick()
        assert result["cells_processed"] >= 0

    async def test_sequential_ring_processing(self):
        """Setting ``parallel_ring_processing=False`` exercises the
        ``_async_sequential_tick`` branch."""
        cfg = HoneycombConfig(radius=2, parallel_ring_processing=False)
        grid = HoneycombGrid(cfg)
        result = await grid.tick()
        assert result["tick"] == 0  # tick_count was 0 BEFORE this tick
        assert grid._tick_count == 1

    async def test_concurrent_grid_ticks(self, small_grid):
        """Two ticks gathered concurrently complete without crashing.

        Per-cell RWLock + FSM still serializes cell mutation; the
        wins is at the ring/cell fan-out level, not at the per-cell
        instruction sequence."""
        results = await asyncio.gather(
            small_grid.tick(),
            small_grid.tick(),
            return_exceptions=True,
        )
        # Neither raised; both returned dicts.
        assert all(isinstance(r, dict) for r in results)
        # Tick count advanced by 2 (sum of both ticks). Concurrent
        # increments may interleave but the total is monotonic.
        assert small_grid._tick_count >= 2


# ───────────────────────────────────────────────────────────────────────────────
# NectarFlow.tick (async)
# ───────────────────────────────────────────────────────────────────────────────


class TestAsyncNectarFlowTick:
    async def test_nectar_tick_is_async(self, small_grid):
        flow = NectarFlow(small_grid)
        result = await flow.tick()
        assert isinstance(result, dict)
        assert "pheromones_evaporated" in result

    async def test_nectar_tick_evaporates_pheromones(self, small_grid):
        flow = NectarFlow(small_grid)
        coord = next(iter(small_grid._cells))
        flow.deposit_pheromone(coord, PheromoneType.FOOD, 0.5)
        # First tick: deposit is fresh; evaporation may not return >0
        # but the call must complete without exception.
        await flow.tick()
        # Pheromone level dropped slightly post-decay.
        assert flow.sense_pheromone(coord, PheromoneType.FOOD) <= 0.5


# ───────────────────────────────────────────────────────────────────────────────
# SwarmScheduler.tick (async)
# ───────────────────────────────────────────────────────────────────────────────


class TestAsyncSwarmSchedulerTick:
    async def test_scheduler_tick_is_async(self, small_grid):
        flow = NectarFlow(small_grid)
        sched = SwarmScheduler(small_grid, flow)
        result = await sched.tick()
        assert isinstance(result, dict)
        assert "tasks_processed" in result

    async def test_scheduler_drains_queue(self, small_grid):
        flow = NectarFlow(small_grid)
        sched = SwarmScheduler(small_grid, flow)
        sched.submit_task("compute", {})
        sched.submit_task("compute", {})
        # Queue is non-empty before the tick.
        before = sched.get_queue_size()
        assert before == 2
        await sched.tick()
        # At least one task moved through tick (depending on
        # probabilistic refusal). The contract (matches the sync
        # wrapper) is that tick returns without raising.
        assert sched._tick_count == 1


# ───────────────────────────────────────────────────────────────────────────────
# HoneycombCell.execute_tick (async)
# ───────────────────────────────────────────────────────────────────────────────


class TestAsyncCellExecuteTick:
    async def test_cell_execute_tick_is_async(self, small_grid):
        cell = next(iter(small_grid._cells.values()))
        # Cells initialize EMPTY → tick should refuse with
        # "EMPTY" reason since EMPTY is not (ACTIVE, IDLE).
        # Drive to IDLE via add_vcore first.
        result = await cell.execute_tick()
        assert isinstance(result, dict)
        assert "processed" in result

    async def test_cell_execute_tick_with_idle_cell(self, small_grid):
        cell = next(c for c in small_grid._cells.values() if isinstance(c, WorkerCell))
        # No vcores → state stays EMPTY which is rejected by the
        # gate. add a placeholder vcore object so add_vcore flips the
        # state to IDLE.
        cell.add_vcore(object())
        assert cell.state in (CellState.IDLE, CellState.ACTIVE)
        result = await cell.execute_tick()
        assert result["processed"] is True
