"""Phase 6.4 — auto-checkpoint inside ``HoneycombGrid.tick`` + crash
recovery from the persisted snapshot.

The brief asks for a kill-mid-tick scenario where a restore preserves
"task queue + cell state + FSMs". The Phase 6.3 ``checkpoint`` blob
covers cell state + role + FSM history + tick_count + config.
SwarmScheduler's task queue is *not* in the blob (the scheduler is a
sibling object, not a member of ``HoneycombGrid``); persisting tasks
across restarts is documented as a gap deferred to a later phase.
This file therefore exercises the parts that the grid checkpoint
*does* preserve and verifies the kill-mid-run / restore loop is
durable for them.

Covered:

- ``checkpoint_interval_ticks`` validation in ``HoneycombConfig``.
- Auto-checkpoint fires at the configured interval and only then.
- Atomic write semantics: a crash between two interval boundaries
  recovers from the last successful interval, not from a torn file.
- Restored grid resumes with the captured ``_tick_count`` and
  per-cell state names + history; subsequent ticks advance from
  there without errors.
- Auto-checkpoint failure (e.g. unwritable path) does not abort
  the tick — the grid keeps running.
"""

from __future__ import annotations

import pytest

from hoc.core import CellState, HoneycombConfig, HoneycombGrid

# ───────────────────────────────────────────────────────────────────────────────
# Config validation
# ───────────────────────────────────────────────────────────────────────────────


class TestCheckpointConfigValidation:
    def test_interval_without_path_rejected(self):
        with pytest.raises(ValueError, match="checkpoint_path"):
            HoneycombConfig(radius=1, checkpoint_interval_ticks=10)

    def test_zero_interval_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            HoneycombConfig(
                radius=1,
                checkpoint_interval_ticks=0,
                checkpoint_path="snap.bin",
            )

    def test_negative_interval_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            HoneycombConfig(
                radius=1,
                checkpoint_interval_ticks=-3,
                checkpoint_path="snap.bin",
            )

    def test_disabled_by_default(self):
        cfg = HoneycombConfig(radius=1)
        assert cfg.checkpoint_interval_ticks is None
        assert cfg.checkpoint_path is None
        assert cfg.checkpoint_compress is False


# ───────────────────────────────────────────────────────────────────────────────
# Auto-checkpoint fires at the right interval
# ───────────────────────────────────────────────────────────────────────────────


class TestAutoCheckpointTiming:
    def test_no_checkpoint_when_disabled(self, tmp_path):
        # checkpoint_interval_ticks=None → no file created.
        target = tmp_path / "snap.bin"
        cfg = HoneycombConfig(radius=1)
        grid = HoneycombGrid(cfg)
        for _ in range(5):
            grid.run_tick_sync()
        assert not target.exists()
        grid.shutdown()

    def test_checkpoint_at_interval_only(self, tmp_path):
        target = tmp_path / "snap.bin"
        cfg = HoneycombConfig(
            radius=1,
            checkpoint_interval_ticks=3,
            checkpoint_path=str(target),
        )
        grid = HoneycombGrid(cfg)

        # tick 1: no snapshot yet (1 % 3 != 0)
        grid.run_tick_sync()
        assert not target.exists()
        # tick 2: still nothing
        grid.run_tick_sync()
        assert not target.exists()
        # tick 3: 3 % 3 == 0 → snapshot written
        grid.run_tick_sync()
        assert target.exists()
        grid.shutdown()

    def test_atomic_write_no_leftover_tmp(self, tmp_path):
        target = tmp_path / "snap.bin"
        cfg = HoneycombConfig(
            radius=1,
            checkpoint_interval_ticks=2,
            checkpoint_path=str(target),
        )
        grid = HoneycombGrid(cfg)
        for _ in range(2):
            grid.run_tick_sync()
        # Snapshot is in place; no leftover .tmp suffix.
        assert target.exists()
        assert not (tmp_path / "snap.bin.tmp").exists()
        grid.shutdown()


# ───────────────────────────────────────────────────────────────────────────────
# Crash + restore loop
# ───────────────────────────────────────────────────────────────────────────────


def _drop_grid(grid: HoneycombGrid) -> None:
    """Best-effort 'kill': release the grid object so the executor /
    health monitor stop. Mirrors a process termination as far as the
    in-memory state is concerned — anything not on disk is gone."""
    try:
        grid.shutdown()
    except Exception:
        pass


class TestCrashRecovery:
    def test_restore_after_crash_at_interval(self, tmp_path):
        """Grid runs for 10 ticks with auto-checkpoint every 5. The
        last successful snapshot reflects tick 10. We drop the grid
        (simulated crash) and restore — tick_count comes back as 10."""
        target = tmp_path / "snap.bin"
        cfg = HoneycombConfig(
            radius=1,
            checkpoint_interval_ticks=5,
            checkpoint_path=str(target),
        )
        grid = HoneycombGrid(cfg)
        for _ in range(10):
            grid.run_tick_sync()
        _drop_grid(grid)
        del grid

        restored = HoneycombGrid.restore_from_checkpoint(target)
        assert restored._tick_count == 10
        assert len(restored._cells) > 0
        restored.shutdown()

    def test_restore_after_crash_between_intervals(self, tmp_path):
        """Grid runs for 12 ticks with auto-checkpoint every 5.
        Snapshots happened at tick 5 and tick 10 — the last one wins.
        Restoring jumps back to tick 10 (not 12); the brief calls this
        the recovery point objective (RPO)."""
        target = tmp_path / "snap.bin"
        cfg = HoneycombConfig(
            radius=1,
            checkpoint_interval_ticks=5,
            checkpoint_path=str(target),
        )
        grid = HoneycombGrid(cfg)
        for _ in range(12):
            grid.run_tick_sync()
        _drop_grid(grid)

        restored = HoneycombGrid.restore_from_checkpoint(target)
        # Last snapshot was at tick 10, so restore lands at 10 not 12.
        assert restored._tick_count == 10
        restored.shutdown()

    def test_restored_grid_can_resume_ticking(self, tmp_path):
        target = tmp_path / "snap.bin"
        cfg = HoneycombConfig(
            radius=1,
            checkpoint_interval_ticks=5,
            checkpoint_path=str(target),
        )
        grid = HoneycombGrid(cfg)
        for _ in range(5):
            grid.run_tick_sync()
        _drop_grid(grid)

        restored = HoneycombGrid.restore_from_checkpoint(target)
        # Resume with the captured config (same checkpoint setup) —
        # the restored grid must be able to tick onward without error.
        for _ in range(3):
            restored.run_tick_sync()
        assert restored._tick_count == 8  # 5 + 3
        restored.shutdown()

    def test_cell_states_preserved_across_crash(self, tmp_path):
        target = tmp_path / "snap.bin"
        cfg = HoneycombConfig(
            radius=1,
            checkpoint_interval_ticks=2,
            checkpoint_path=str(target),
        )
        grid = HoneycombGrid(cfg)

        # Drive a couple of cells through known transitions before the
        # interval fires so the snapshot captures them.
        coords_with_changes = list(grid._cells.keys())[:3]
        for coord in coords_with_changes:
            cell = grid._cells[coord]
            cell.state = CellState.IDLE
            cell.state = CellState.ACTIVE

        for _ in range(2):  # hits the interval
            grid.run_tick_sync()
        _drop_grid(grid)

        restored = HoneycombGrid.restore_from_checkpoint(target)
        # The state of those cells survived the kill.
        for coord in coords_with_changes:
            # tick() walks ACTIVE → IDLE on completion; both are
            # acceptable post-tick states. The point is the cell is
            # not stuck in EMPTY (i.e. the snapshot did capture
            # meaningful progress).
            assert restored._cells[coord].state in (CellState.IDLE, CellState.ACTIVE)
        restored.shutdown()


# ───────────────────────────────────────────────────────────────────────────────
# Auto-checkpoint failure does not crash the live grid
# ───────────────────────────────────────────────────────────────────────────────


class TestCheckpointFailureResilience:
    def test_unwritable_path_logged_not_raised(self, tmp_path, caplog):
        # Use a path inside a non-existent directory to force IOError
        # at write time. ``_auto_checkpoint`` swallows + logs; tick
        # returns normally.
        bad_path = tmp_path / "no_such_dir" / "snap.bin"
        cfg = HoneycombConfig(
            radius=1,
            checkpoint_interval_ticks=1,
            checkpoint_path=str(bad_path),
        )
        grid = HoneycombGrid(cfg)
        # Tick should not raise even though the snapshot write fails.
        result = grid.run_tick_sync()
        assert isinstance(result, dict)
        assert result["tick"] == 0  # tick that just ran
        # ``_tick_count`` advanced past the broken interval.
        assert grid._tick_count == 1
        grid.shutdown()

    def test_grid_keeps_running_after_failed_checkpoint(self, tmp_path):
        bad_path = tmp_path / "no_such_dir" / "snap.bin"
        cfg = HoneycombConfig(
            radius=1,
            checkpoint_interval_ticks=1,
            checkpoint_path=str(bad_path),
        )
        grid = HoneycombGrid(cfg)
        # Several ticks all hit the broken path; none should raise.
        for _ in range(5):
            grid.run_tick_sync()
        assert grid._tick_count == 5
        grid.shutdown()


# ───────────────────────────────────────────────────────────────────────────────
# Manual checkpoint mid-run (no interval) still works
# ───────────────────────────────────────────────────────────────────────────────


class TestManualCheckpointMidRun:
    def test_manual_checkpoint_inside_loop(self, tmp_path):
        target = tmp_path / "manual.bin"
        cfg = HoneycombConfig(radius=1)  # no auto-checkpoint
        grid = HoneycombGrid(cfg)
        for i in range(10):
            grid.run_tick_sync()
            if i == 4:  # after 5th tick (i counts from 0)
                grid.checkpoint(target)

        # File exists and reflects the state at the time of
        # checkpoint (tick_count=5, since the 5th tick incremented
        # _tick_count from 4 to 5 before we wrote).
        assert target.exists()
        restored = HoneycombGrid.restore_from_checkpoint(target)
        assert restored._tick_count == 5
        # The live grid kept going past the snapshot.
        assert grid._tick_count == 10
        grid.shutdown()
        restored.shutdown()
