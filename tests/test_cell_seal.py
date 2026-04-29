"""Phase 5.1 tests for ``HoneycombCell.seal()`` (graceful shutdown).

Exercises the new ``seal()`` method introduced in Phase 5.1 and the
companion ``CellState.SEALED`` wire-up via the ``admin_seal`` FSM
trigger. The tests check the public observable contract:

- ``seal()`` succeeds from any non-FAILED state.
- The cell is drained of vCores.
- The state transitions to ``SEALED`` (FSM event published).
- A final-metrics log line is emitted.
- ``add_vcore`` refuses to add to a sealed cell.
- ``is_available`` returns ``False`` for sealed cells.
- ``seal()`` is idempotent (returns ``False`` if already sealed).
- ``seal()`` refuses to seal a ``FAILED`` cell (use ``recover()`` first).

Anti-regression: pre-Phase-5 ``seal()`` did not exist; nothing else
should have changed for the other lifecycle methods.
"""

from __future__ import annotations

import logging

import pytest

from hoc.core import CellRole, CellState, HexCoord, HoneycombCell
from hoc.core.events import EventType


@pytest.fixture
def empty_cell() -> HoneycombCell:
    return HoneycombCell(coord=HexCoord(0, 0), role=CellRole.WORKER)


@pytest.fixture
def idle_cell() -> HoneycombCell:
    cell = HoneycombCell(coord=HexCoord(0, 0), role=CellRole.WORKER)
    cell.add_vcore(object())
    assert cell.state == CellState.IDLE
    return cell


# ─── Happy paths ────────────────────────────────────────────────────────────────


class TestSealSucceeds:
    def test_seal_from_empty_returns_true(self, empty_cell: HoneycombCell) -> None:
        assert empty_cell.seal() is True
        assert empty_cell.state == CellState.SEALED

    def test_seal_from_idle_returns_true(self, idle_cell: HoneycombCell) -> None:
        assert idle_cell.seal() is True
        assert idle_cell.state == CellState.SEALED

    def test_seal_from_active_returns_true(self, idle_cell: HoneycombCell) -> None:
        # Force ACTIVE through admin (production code goes via execute_tick;
        # we use the wildcard admin path to avoid needing a real vcore.tick()).
        idle_cell.state = CellState.ACTIVE
        assert idle_cell.seal() is True
        assert idle_cell.state == CellState.SEALED

    def test_seal_drains_vcores(self, idle_cell: HoneycombCell) -> None:
        idle_cell.add_vcore(object())  # second vcore
        assert idle_cell.vcore_count == 2
        idle_cell.seal()
        assert idle_cell.vcore_count == 0

    def test_seal_publishes_state_changed_event(self, empty_cell: HoneycombCell) -> None:
        events: list[tuple[str, str]] = []

        def listener(event):  # type: ignore[no-untyped-def]
            if event.type == EventType.CELL_STATE_CHANGED:
                events.append((event.data["old"], event.data["new"]))

        empty_cell._event_bus.subscribe(EventType.CELL_STATE_CHANGED, listener)
        empty_cell.seal()
        # _set_state emits the event with the old/new state names.
        assert ("EMPTY", "SEALED") in events

    def test_seal_logs_final_metrics(
        self, idle_cell: HoneycombCell, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="hoc.core.cells_base"):
            idle_cell.seal(reason="rebalance")
        msg = " ".join(rec.getMessage() for rec in caplog.records)
        # The format is "Cell <coord> sealed: reason=<reason> metrics={...}"
        assert "sealed" in msg
        assert "rebalance" in msg
        # final_metrics dict mentions the four captured fields.
        for field in ("ticks_processed", "error_count", "vcores_drained", "age_seconds"):
            assert field in msg


# ─── Refusal paths ─────────────────────────────────────────────────────────────


class TestSealRefuses:
    def test_seal_failed_cell_returns_false(self, empty_cell: HoneycombCell) -> None:
        # Force the cell to FAILED via the wildcard admin path.
        empty_cell.state = CellState.FAILED
        assert empty_cell.seal() is False
        # The state must remain FAILED (no spurious mutation).
        assert empty_cell.state == CellState.FAILED

    def test_seal_idempotent(self, empty_cell: HoneycombCell) -> None:
        assert empty_cell.seal() is True
        # Second call is a no-op.
        assert empty_cell.seal() is False
        assert empty_cell.state == CellState.SEALED


# ─── Sealed-cell consequences ──────────────────────────────────────────────────


class TestSealedCellConsequences:
    def test_sealed_cell_refuses_add_vcore(self, empty_cell: HoneycombCell) -> None:
        empty_cell.seal()
        # add_vcore must reject after the seal — the operational promise of
        # graceful shutdown is "drained, refusing tasks".
        assert empty_cell.add_vcore(object()) is False
        assert empty_cell.vcore_count == 0

    def test_sealed_cell_is_not_available(self, empty_cell: HoneycombCell) -> None:
        empty_cell.seal()
        assert empty_cell.is_available is False

    def test_sealed_state_appears_in_to_dict(self, empty_cell: HoneycombCell) -> None:
        empty_cell.seal()
        d = empty_cell.to_dict()
        assert d["state"] == "SEALED"
        assert d["vcores"] == 0

    def test_sealed_cell_fsm_history_contains_sealed_target(
        self, empty_cell: HoneycombCell
    ) -> None:
        empty_cell.seal()
        # FSM moves from EMPTY -> SEALED. The cell's FSM state is now
        # SEALED; previous state EMPTY appears in history.
        assert empty_cell.fsm.state == "SEALED"
        assert "EMPTY" in empty_cell.fsm.history
