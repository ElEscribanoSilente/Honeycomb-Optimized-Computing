"""Phase 5.3 tests for ``hoc.observability`` (structlog event log).

Covers:

- ``configure_logging(json=True)`` produces JSON-renderable strings;
  parsed records carry the expected fields (event, timestamp, level).
- ``cell.state_changed`` events fire from ``HoneycombCell._set_state``
  (via the ``log_cell_state_transition`` helper).
- ``cell.sealed`` events fire from ``HoneycombCell.seal``.
- ``failover.migrate_started`` + ``failover.migrate_completed`` (with
  ``result``) fire from ``CellFailover._migrate_work``.
- ``election.started`` + ``election.completed`` fire from
  ``QueenSuccession.elect_new_queen``.
- ``configure_logging`` is idempotent.

The tests use ``caplog`` to capture the messages that structlog
renders into the underlying stdlib log records — that's where the
JSON output ends up after ``LoggerFactory`` hands the rendered string
to ``logging.getLogger(name)``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from hoc.core import (
    CellRole,
    HexCoord,
    HoneycombCell,
    HoneycombConfig,
    HoneycombGrid,
    QueenCell,
    WorkerCell,
)
from hoc.core.observability import (
    EVENT_LOGGER_NAME,
    configure_logging,
    get_event_logger,
    reset_for_tests,
)
from hoc.resilience import CellFailover, FailureType, QueenSuccession, ResilienceConfig


@pytest.fixture(autouse=True)
def _reset_observability_between_tests():
    """structlog state leaks between tests; reset before + after."""
    reset_for_tests()
    yield
    reset_for_tests()


@pytest.fixture
def json_log(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Configure structlog for JSON output and route caplog to capture
    the rendered records."""
    configure_logging(json=True, level=logging.INFO)
    caplog.set_level(logging.INFO)
    return caplog


def _parse_events(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    """Parse the structlog-rendered messages from caplog records."""
    parsed: list[dict[str, Any]] = []
    for rec in caplog.records:
        msg = rec.getMessage()
        if not msg:
            continue
        try:
            obj = json.loads(msg)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            parsed.append(obj)
    return parsed


# ─── Configuration smoke ───────────────────────────────────────────────────────


class TestConfigureLogging:
    def test_configure_default_is_idempotent(self):
        configure_logging()
        configure_logging()  # second call is a no-op

    def test_configure_json_then_default_is_idempotent(self):
        configure_logging(json=True)
        configure_logging(json=True)

    def test_get_event_logger_returns_a_logger(self):
        log = get_event_logger()
        assert hasattr(log, "info")
        assert hasattr(log, "warning")

    def test_event_logger_default_name_is_hoc_events(self):
        # The channel name stays stable so collectors can rely on it.
        # Documented in EVENT_LOGGER_NAME and ADR-011.
        assert EVENT_LOGGER_NAME == "hoc.events"


# ─── JSON output is parseable ──────────────────────────────────────────────────


class TestJsonOutput:
    def test_json_line_parses_cleanly(self, json_log: pytest.LogCaptureFixture):
        log = get_event_logger("hoc.events.smoke")
        log.info("smoke.test", coord="(0,0)", value=42)
        events = _parse_events(json_log)
        assert events, "Expected at least one JSON-renderable record"
        rec = next(e for e in events if e.get("event") == "smoke.test")
        assert rec["coord"] == "(0,0)"
        assert rec["value"] == 42
        assert rec["level"] == "info"
        assert "timestamp" in rec


# ─── Cell state transition events ──────────────────────────────────────────────


class TestCellStateTransitionEvents:
    def test_set_state_emits_cell_state_changed(self, json_log: pytest.LogCaptureFixture):
        cell = HoneycombCell(coord=HexCoord(0, 0), role=CellRole.WORKER)
        cell.add_vcore(object())  # EMPTY -> IDLE
        events = _parse_events(json_log)
        records = [e for e in events if e.get("event") == "cell.state_changed"]
        assert len(records) >= 1
        rec = records[0]
        assert rec["from_state"] == "EMPTY"
        assert rec["to_state"] == "IDLE"
        assert "coord" in rec

    def test_seal_emits_cell_sealed(self, json_log: pytest.LogCaptureFixture):
        cell = HoneycombCell(coord=HexCoord(0, 0), role=CellRole.WORKER)
        cell.seal(reason="rebalance")
        events = _parse_events(json_log)
        records = [e for e in events if e.get("event") == "cell.sealed"]
        assert records, "Expected a cell.sealed event"
        rec = records[0]
        assert rec["reason"] == "rebalance"
        assert "ticks_processed" in rec
        assert "vcores_drained" in rec


# ─── Failover events ───────────────────────────────────────────────────────────


class TestFailoverEvents:
    def test_migrate_emits_started_and_completed(self, json_log: pytest.LogCaptureFixture):
        grid = HoneycombGrid(HoneycombConfig(radius=2))
        fo = CellFailover(grid, ResilienceConfig())
        coord = next(
            c
            for c, cell in grid._cells.items()
            if isinstance(cell, WorkerCell) and not isinstance(cell, QueenCell)
        )
        fo.handle_failure(coord, FailureType.TIMEOUT)
        events = _parse_events(json_log)
        starts = [e for e in events if e.get("event") == "failover.migrate_started"]
        completes = [e for e in events if e.get("event") == "failover.migrate_completed"]
        assert starts, "Expected a failover.migrate_started event"
        assert completes, "Expected a failover.migrate_completed event"
        assert completes[0]["result"] in ("success", "failed")
        assert completes[0]["source"] == str(coord)


# ─── Election events ───────────────────────────────────────────────────────────


class TestElectionEvents:
    def test_election_emits_started_and_completed(
        self, json_log: pytest.LogCaptureFixture, monkeypatch
    ):
        grid = HoneycombGrid(HoneycombConfig(radius=2))
        succ = QueenSuccession(grid, ResilienceConfig(min_queen_candidates=1))
        winner_coord = next(iter(grid._cells.keys()))
        monkeypatch.setattr(succ, "_identify_candidates", lambda: [winner_coord])
        monkeypatch.setattr(succ, "_conduct_election", lambda _: winner_coord)
        monkeypatch.setattr(
            succ,
            "_promote_to_queen",
            lambda _: QueenCell(winner_coord, grid.config),
        )
        succ.elect_new_queen()
        events = _parse_events(json_log)
        starts = [e for e in events if e.get("event") == "election.started"]
        completes = [e for e in events if e.get("event") == "election.completed"]
        assert starts, "Expected an election.started event"
        assert completes, "Expected an election.completed event"
        assert completes[0]["result"] == "elected"
        assert completes[0]["winner"] == str(winner_coord)
