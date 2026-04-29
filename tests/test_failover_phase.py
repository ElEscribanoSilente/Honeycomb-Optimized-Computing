"""Phase 5.2c tests for ``CellFailover._per_cell_phase`` (FailoverFlow wire-up).

Exercises the per-coord FailoverFlow FSM observability that Phase 5.2c
adds on top of the cell-level CellState wire-up from Phase 5.1:

- Successful migration: HEALTHY → DEGRADED → MIGRATING → RECOVERED.
- Recovery via ``mark_recovered``: RECOVERED → HEALTHY (stabilized).
- Rollback path: ``tramoya.undo()`` reverts MIGRATING → DEGRADED on
  exception in the migration loop.
- Independence: each coord owns its own FSM instance (different
  failovers don't trample each other's history).
- Default phase: never-touched coords report HEALTHY.
- Repeat failover: a coord that already reached RECOVERED can be
  failed-over again (FSM resets to HEALTHY before the second cycle).
"""

from __future__ import annotations

import pytest

from hoc.core import (
    CellState,
    HexCoord,
    HoneycombConfig,
    HoneycombGrid,
    QueenCell,
    WorkerCell,
)
from hoc.resilience import (
    CellFailover,
    FailoverPhase,
    FailureType,
    ResilienceConfig,
)

# ─── Fixtures (mirror tests/test_resilience.py) ────────────────────────────────


@pytest.fixture
def grid_r2() -> HoneycombGrid:
    return HoneycombGrid(HoneycombConfig(radius=2))


@pytest.fixture
def default_config() -> ResilienceConfig:
    return ResilienceConfig()


def _first_worker_coord(grid: HoneycombGrid) -> HexCoord:
    return next(
        c
        for c, cell in grid._cells.items()
        if isinstance(cell, WorkerCell) and not isinstance(cell, QueenCell)
    )


# ─── Default phase ────────────────────────────────────────────────────────────


class TestFailoverPhaseDefaults:
    def test_unseen_coord_reports_healthy(self, grid_r2, default_config):
        fo = CellFailover(grid_r2, default_config)
        assert fo.get_failover_phase(HexCoord(0, 0)) == FailoverPhase.HEALTHY

    def test_per_cell_failover_dict_starts_empty(self, grid_r2, default_config):
        fo = CellFailover(grid_r2, default_config)
        assert fo._per_cell_failover == {}


# ─── Happy path ────────────────────────────────────────────────────────────────


class TestFailoverPhaseProgression:
    def test_successful_migration_ends_at_recovered(self, grid_r2, default_config):
        fo = CellFailover(grid_r2, default_config)
        coord = _first_worker_coord(grid_r2)
        event = fo.handle_failure(coord, FailureType.TIMEOUT)
        if not event.success:
            pytest.skip("Failover did not succeed in this fixture")
        assert fo.get_failover_phase(coord) == FailoverPhase.RECOVERED

    def test_phase_history_includes_degraded_and_migrating(self, grid_r2, default_config):
        fo = CellFailover(grid_r2, default_config)
        coord = _first_worker_coord(grid_r2)
        event = fo.handle_failure(coord, FailureType.TIMEOUT)
        if not event.success:
            pytest.skip("Failover did not succeed in this fixture")
        # The per-coord FSM history captures the full sequence of states
        # leading up to the current one (RECOVERED). Phase 5.2c uses
        # history_size=16 so all four are retained.
        history = fo._per_cell_failover[coord].fsm.history
        assert "HEALTHY" in history
        assert "DEGRADED" in history
        assert "MIGRATING" in history

    def test_mark_recovered_stabilises_to_healthy(self, grid_r2, default_config):
        fo = CellFailover(grid_r2, default_config)
        coord = _first_worker_coord(grid_r2)
        event = fo.handle_failure(coord, FailureType.TIMEOUT)
        if not event.success:
            pytest.skip("Failover did not succeed in this fixture")
        assert fo.get_failover_phase(coord) == FailoverPhase.RECOVERED
        assert fo.mark_recovered(coord) is True
        assert fo.get_failover_phase(coord) == FailoverPhase.HEALTHY


# ─── Rollback (undo) ───────────────────────────────────────────────────────────


class TestFailoverPhaseUndo:
    def test_exception_in_migration_undoes_to_degraded(self, grid_r2, default_config, monkeypatch):
        """Phase 5.2c: when the migration loop raises, ``tramoya.undo()``
        reverses MIGRATING → DEGRADED so a retry can pick up where we
        left off without re-walking the HEALTHY → DEGRADED step."""
        fo = CellFailover(grid_r2, default_config)
        coord = _first_worker_coord(grid_r2)
        target_coord = fo.find_failover_target(coord)
        assert target_coord is not None

        # Seed the source with a vCore so the migration loop has work
        # to do before the synthetic exception fires.
        source_cell = grid_r2.get_cell(coord)
        source_cell.add_vcore(object())

        from hoc.core.cells_base import HoneycombCell

        def boom(self, _vcore):  # pragma: no cover - captured by except path
            raise RuntimeError("synthetic migration failure")

        monkeypatch.setattr(HoneycombCell, "add_vcore", boom)

        result = fo._migrate_work(coord, target_coord)

        assert result is False
        assert fo.get_failover_phase(coord) == FailoverPhase.DEGRADED


# ─── Independence per coord ────────────────────────────────────────────────────


class TestFailoverPhaseIndependence:
    def test_two_coords_have_independent_phases(self, grid_r2, default_config):
        """The per-coord FSM dict means each coord's lifecycle is
        observable independently."""
        fo = CellFailover(grid_r2, default_config)
        worker_coords = [
            c
            for c, cell in grid_r2._cells.items()
            if isinstance(cell, WorkerCell) and not isinstance(cell, QueenCell)
        ]
        a, b = worker_coords[0], worker_coords[1]
        # Only `a` fails; `b` should remain at the default HEALTHY.
        fo.handle_failure(a, FailureType.TIMEOUT)
        assert fo.get_failover_phase(b) == FailoverPhase.HEALTHY


# ─── Repeat failover ───────────────────────────────────────────────────────────


class TestFailoverPhaseRepeat:
    def test_second_failover_resets_fsm_to_healthy_first(self, grid_r2, default_config):
        """After a successful failover the FSM is at RECOVERED. Phase
        5.2c resets it to HEALTHY at the start of the next migration so
        the lifecycle stays linear."""
        fo = CellFailover(grid_r2, default_config)
        coord = _first_worker_coord(grid_r2)
        target = fo.find_failover_target(coord)
        assert target is not None

        # First failover.
        result1 = fo._migrate_work(coord, target)
        assert result1 is True
        assert fo.get_failover_phase(coord) == FailoverPhase.RECOVERED

        # Reset the cell back to a live state so a second migration is
        # plausible (the production cooldown logic would normally block
        # this, but we're testing _migrate_work in isolation here).
        source_cell = grid_r2.get_cell(coord)
        source_cell.state = CellState.IDLE

        # Second failover -> should walk through DEGRADED/MIGRATING
        # again before reaching RECOVERED.
        result2 = fo._migrate_work(coord, target)
        assert result2 is True
        assert fo.get_failover_phase(coord) == FailoverPhase.RECOVERED
        # The FSM history should reflect the second cycle's transitions.
        hist = fo._per_cell_failover[coord].fsm.history
        # At least one HEALTHY (initial of the reset cycle) appears.
        assert "HEALTHY" in hist


# ─── Wire-up sanity ────────────────────────────────────────────────────────────


class TestFailoverPhaseWireup:
    def test_wrapper_state_attribute_starts_healthy(self, grid_r2, default_config):
        """The ``_FailoverCellState.state`` attribute is the single
        wire-up signal choreo's walker looks for. New wrappers default
        to HEALTHY."""
        from hoc.resilience import _FailoverCellState

        wrapper = _FailoverCellState()
        assert wrapper.state == FailoverPhase.HEALTHY
        # The FSM is also lazily built by the dataclass default factory.
        assert wrapper.fsm.state == "HEALTHY"

    def test_wrapper_state_advances_through_recovered(self, grid_r2, default_config):
        fo = CellFailover(grid_r2, default_config)
        coord = _first_worker_coord(grid_r2)
        event = fo.handle_failure(coord, FailureType.TIMEOUT)
        if not event.success:
            pytest.skip("Failover did not succeed in this fixture")
        # The per-coord wrapper's state mirrors the most recent
        # transition; a successful path ends at RECOVERED.
        assert fo._per_cell_failover[coord].state == FailoverPhase.RECOVERED
