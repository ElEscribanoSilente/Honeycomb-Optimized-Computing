"""Phase 7.2 — sync wrapper (run_tick_sync / run_execute_tick_sync) tests.

The wrappers exist as a v1 → v2 migration aid for callers that can't
adopt ``await`` immediately. Each wrapper:

1. Emits a single :class:`DeprecationWarning` per process on first call.
2. Refuses to run from inside a live event loop (``RuntimeError``).
3. Returns the same result dict as the async ``await tick()`` would.

This file exercises all four wrappers (HoneycombGrid, NectarFlow,
SwarmScheduler, HoneycombCell.run_execute_tick_sync) end-to-end.
"""

from __future__ import annotations

import asyncio
import warnings

import pytest

from hoc.core import (
    HoneycombCell,
    HoneycombConfig,
    HoneycombGrid,
    WorkerCell,
)
from hoc.nectar import NectarFlow
from hoc.swarm import SwarmScheduler

# ───────────────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def grid() -> HoneycombGrid:
    return HoneycombGrid(HoneycombConfig(radius=1))


# ───────────────────────────────────────────────────────────────────────────────
# Result equivalence: wrapper vs async
# ───────────────────────────────────────────────────────────────────────────────


class TestWrapperEquivalence:
    def test_grid_wrapper_returns_dict(self, grid):
        result = grid.run_tick_sync()
        assert isinstance(result, dict)
        assert "tick" in result

    def test_nectar_wrapper_returns_dict(self, grid):
        flow = NectarFlow(grid)
        result = flow.run_tick_sync()
        assert isinstance(result, dict)
        assert "pheromones_evaporated" in result

    def test_scheduler_wrapper_returns_dict(self, grid):
        flow = NectarFlow(grid)
        sched = SwarmScheduler(grid, flow)
        result = sched.run_tick_sync()
        assert isinstance(result, dict)
        assert "tasks_processed" in result

    def test_cell_wrapper_returns_dict(self, grid):
        cell = next(c for c in grid._cells.values() if isinstance(c, WorkerCell))
        cell.add_vcore(object())  # flip EMPTY → IDLE
        result = cell.run_execute_tick_sync()
        assert isinstance(result, dict)
        assert "processed" in result


# ───────────────────────────────────────────────────────────────────────────────
# DeprecationWarning emission (one-shot per class per process)
# ───────────────────────────────────────────────────────────────────────────────


class TestDeprecationWarningOneShot:
    """The wrappers warn once per process (module-level flag). Each
    test resets the relevant class flag explicitly so the warning is
    observable from a clean slate.
    """

    def test_grid_emits_deprecation_warning_once(self, grid):
        # Reset the class-level flag so this test sees the first call.
        HoneycombGrid._SYNC_DEPRECATION_EMITTED = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            grid.run_tick_sync()
            grid.run_tick_sync()
            grid.run_tick_sync()

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1
        assert "v1→v2 migration aid" in str(deprecation_warnings[0].message)

    def test_nectar_emits_deprecation_warning_once(self, grid):
        NectarFlow._SYNC_DEPRECATION_EMITTED = False
        flow = NectarFlow(grid)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            flow.run_tick_sync()
            flow.run_tick_sync()

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1

    def test_scheduler_emits_deprecation_warning_once(self, grid):
        SwarmScheduler._SYNC_DEPRECATION_EMITTED = False
        flow = NectarFlow(grid)
        sched = SwarmScheduler(grid, flow)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            sched.run_tick_sync()
            sched.run_tick_sync()

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1

    def test_cell_emits_deprecation_warning_once(self, grid):
        HoneycombCell._SYNC_DEPRECATION_EMITTED = False
        cell = next(c for c in grid._cells.values() if isinstance(c, WorkerCell))
        cell.add_vcore(object())

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            cell.run_execute_tick_sync()
            cell.run_execute_tick_sync()

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1


# ───────────────────────────────────────────────────────────────────────────────
# RuntimeError when called from inside a running event loop
# ───────────────────────────────────────────────────────────────────────────────


class TestWrapperInsideEventLoop:
    """Calling ``run_tick_sync`` from inside an event loop is a usage
    error — the wrapper would deadlock or raise ``asyncio.run()``'s
    own error. We guard explicitly so the failure mode is clear."""

    async def test_grid_wrapper_inside_loop_raises(self, grid):
        with pytest.raises(RuntimeError, match="running event loop"):
            grid.run_tick_sync()

    async def test_nectar_wrapper_inside_loop_raises(self, grid):
        flow = NectarFlow(grid)
        with pytest.raises(RuntimeError, match="running event loop"):
            flow.run_tick_sync()

    async def test_scheduler_wrapper_inside_loop_raises(self, grid):
        flow = NectarFlow(grid)
        sched = SwarmScheduler(grid, flow)
        with pytest.raises(RuntimeError, match="running event loop"):
            sched.run_tick_sync()

    async def test_cell_wrapper_inside_loop_raises(self, grid):
        cell = next(c for c in grid._cells.values() if isinstance(c, WorkerCell))
        cell.add_vcore(object())
        with pytest.raises(RuntimeError, match="running event loop"):
            cell.run_execute_tick_sync()


# ───────────────────────────────────────────────────────────────────────────────
# Wrapper produces the same observable result as await tick()
# ───────────────────────────────────────────────────────────────────────────────


class TestWrapperParitiesAsync:
    """Running the same logical tick via the sync wrapper vs the
    async path produces equivalent observable state. We don't compare
    nanosecond timestamps; we compare the structural shape (cell
    states, queue size, tick counter) post-tick."""

    def test_grid_wrapper_parity(self):
        cfg = HoneycombConfig(radius=1)
        sync_grid = HoneycombGrid(cfg)
        async_grid = HoneycombGrid(cfg)

        sync_grid.run_tick_sync()
        sync_grid.run_tick_sync()

        async def run_async():
            await async_grid.tick()
            await async_grid.tick()

        asyncio.run(run_async())

        assert sync_grid._tick_count == async_grid._tick_count == 2
        # Same set of cell coords post-tick (ticks don't add/remove
        # cells in the default grid).
        assert set(sync_grid._cells) == set(async_grid._cells)
        # State distribution matches.
        for coord in sync_grid._cells:
            assert (
                sync_grid._cells[coord].state == async_grid._cells[coord].state
            ), f"state mismatch at {coord}"

    def test_scheduler_wrapper_parity(self):
        cfg = HoneycombConfig(radius=1)
        sync_grid = HoneycombGrid(cfg)
        async_grid = HoneycombGrid(cfg)
        sync_sched = SwarmScheduler(sync_grid, NectarFlow(sync_grid))
        async_sched = SwarmScheduler(async_grid, NectarFlow(async_grid))

        sync_sched.submit_task("compute", {})
        async_sched.submit_task("compute", {})

        sync_sched.run_tick_sync()

        async def run_async():
            await async_sched.tick()

        asyncio.run(run_async())

        assert sync_sched._tick_count == async_sched._tick_count == 1
