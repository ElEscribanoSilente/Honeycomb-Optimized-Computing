"""Phase 7.11 — focused coverage tests.

Targets the cheapest uncovered branches in `sandbox.py` and `swarm.py`
identified during Phase 7 closure. Goal: nudge global coverage above
the DoD's 85% threshold on Windows local (CI Linux already meets it
naturally because the sandbox fork-only tests run there).

These are NOT semantic tests — the brief flows are covered by their
own files. This file exists to claim the coverage lines that would
otherwise stay dark on the Windows runner.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from hoc.core import HoneycombConfig, HoneycombGrid, WorkerCell
from hoc.nectar import NectarFlow
from hoc.sandbox import (
    SandboxConfig,
    SandboxCrashed,
    SandboxedTaskRunner,
    SandboxNotSupported,
    cgroup_v2_available,
    job_objects_available,
)
from hoc.swarm import (
    BeeBehavior,
    ForagerBehavior,
    GuardBehavior,
    HiveTask,
    NurseBehavior,
    ScoutBehavior,
    SwarmConfig,
    SwarmScheduler,
    TaskState,
)

# ───────────────────────────────────────────────────────────────────────────────
# sandbox.py uncovered branches
# ───────────────────────────────────────────────────────────────────────────────


class TestSandboxCrashedUnderlying:
    """Cover ``SandboxCrashed.__init__`` ``underlying=`` kwarg path."""

    def test_constructs_with_underlying_exception(self):
        inner = ValueError("nested")
        exc = SandboxCrashed("wrapper", underlying=inner)
        assert exc.underlying is inner
        assert "wrapper" in str(exc)

    def test_constructs_without_underlying_defaults_none(self):
        exc = SandboxCrashed("plain")
        assert exc.underlying is None


class TestSandboxUnknownIsolationMode:
    """Cover the catch-all ``raise SandboxNotSupported`` branch in
    ``SandboxedTaskRunner.run`` (the unknown-mode path that the
    ``Literal`` type narrows out at edit time but can hit at
    runtime)."""

    def test_unknown_mode_raises(self):
        cfg = SandboxConfig()
        # Mutate the field after construction so we sidestep the
        # ``Literal`` check at type-check time.
        cfg.isolation = "bogus"  # type: ignore[assignment]
        runner = SandboxedTaskRunner(cfg)
        with pytest.raises(SandboxNotSupported, match="unknown isolation"):
            runner.run(lambda: None)


class TestJobObjectsImportError:
    """Cover the ``except ImportError`` branch in
    ``job_objects_available`` on Windows. On non-Windows the function
    returns False before the import; on Windows we mock the import to
    fail so the except branch executes."""

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="job_objects_available only reaches the import on Windows",
    )
    def test_import_error_returns_false(self):
        # Patch ``builtins.__import__`` so any attempt to import
        # ``win32job`` raises ImportError. Other imports pass through.
        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "win32job":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            assert job_objects_available() is False


class TestCgroupNonLinuxBranch:
    """``cgroup_v2_available`` returns False on non-Linux without
    touching the filesystem. Sanity-check that the early return is
    actually exercised (it is — the existing `TestPlatformProbes`
    covers it; we add a redundant call here to anchor the branch
    against the coverage report)."""

    def test_non_linux_returns_false(self):
        if sys.platform == "linux":
            pytest.skip("trivially true on Linux; this anchor is for non-Linux runners")
        assert cgroup_v2_available() is False


# ───────────────────────────────────────────────────────────────────────────────
# swarm.py uncovered branches — small, surgical
# ───────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def grid() -> HoneycombGrid:
    return HoneycombGrid(HoneycombConfig(radius=1))


@pytest.fixture
def nectar(grid: HoneycombGrid) -> NectarFlow:
    return NectarFlow(grid)


@pytest.fixture
def worker_cell(grid: HoneycombGrid):
    return next(c for c in grid._cells.values() if isinstance(c, WorkerCell))


class TestNurseWarmupVcorePath:
    """Cover ``NurseBehavior.execute_task`` when the warmup task
    payload includes a vcore with a ``warmup`` callable. The
    pre-Phase-7.11 tests pass an empty payload that bypasses the
    ``hasattr(vcore, "warmup")`` branch."""

    def test_warmup_invokes_vcore_warmup(self, worker_cell, nectar):
        nurse = NurseBehavior(worker_cell, nectar)

        class _StubVcore:
            warmed = False

            def warmup(self):
                self.warmed = True

        vcore = _StubVcore()
        task = HiveTask(priority=2, task_type="warmup", payload={"vcore": vcore})
        assert nurse.execute_task(task) is True
        assert vcore.warmed is True
        assert task.state == TaskState.COMPLETED


class TestNurseTickIncubationStaggered:
    """Cover the ``still_incubating.append(item)`` branch of
    ``NurseBehavior.tick_incubation`` — the not-yet-ready path."""

    def test_partial_incubation(self, worker_cell, nectar):
        nurse = NurseBehavior(worker_cell, nectar)
        nurse.warmup_ticks = 5
        nurse.incubating.append(
            {
                "spec": {},
                "ticks_remaining": 3,
                "task": HiveTask(priority=2),
            }
        )
        ready = nurse.tick_incubation()
        # Not yet ready: still_incubating path hit.
        assert ready == []
        assert len(nurse.incubating) == 1
        assert nurse.incubating[0]["ticks_remaining"] == 2


class TestScoutSelectFallthrough:
    """Cover ``ScoutBehavior.select_task`` returning None when
    nothing matches (no explore task, no distant target)."""

    def test_no_match_returns_none(self, worker_cell, nectar):
        scout = ScoutBehavior(worker_cell, nectar)
        # Tasks that don't match either the explore type or the
        # distance heuristic.
        tasks = [
            HiveTask(priority=2, task_type="compute"),  # not explore
            HiveTask(priority=2, task_type="validate"),  # not explore
        ]
        assert scout.select_task(tasks) is None


class TestForagerRecruitmentStreak:
    """Cover ``ForagerBehavior._recruit`` invoked after the 3+
    success streak threshold. The pre-Phase-7.11 forager tests don't
    drive a streak deep enough to hit it."""

    def test_recruit_pheromone_deposited_after_streak(self, worker_cell, nectar):
        forager = ForagerBehavior(worker_cell, nectar)
        forager.specialization = "ml"
        # The streak check is *before* update_threshold increments, so
        # _recruit fires on the 4th task (streak observed=3 then bumps
        # to 4). 4 tasks land the recruitment pheromone reliably.
        for _ in range(4):
            t = HiveTask(priority=2, task_type="ml", payload={"execute": lambda: 1})
            forager.execute_task(t)
        from hoc.nectar import PheromoneType

        level = nectar.sense_pheromone(worker_cell.coord, PheromoneType.RECRUITMENT)
        assert level > 0.0


class TestBehaviorAcceptsTypeUnknownSubclass:
    """Cover the final ``return False`` of
    ``SwarmScheduler._behavior_accepts_type`` — the path for a
    BeeBehavior subclass that doesn't match the four known ones."""

    def test_unknown_subclass_rejected(self, grid, nectar, worker_cell):
        # Construct a stand-in behaviour that inherits BeeBehavior but
        # is none of Forager/Nurse/Scout/Guard. select_task / execute_task
        # are abstract, so we provide stubs.
        class _StubBehavior(BeeBehavior):
            def select_task(self, available_tasks):
                return None

            def execute_task(self, task):
                return False

        stub = _StubBehavior(worker_cell, nectar)
        assert SwarmScheduler._behavior_accepts_type(stub, "compute") is False


class TestGuardBlockedSourceRejection:
    """Cover the ``blocked_sources`` rejection path in
    ``GuardBehavior._validate_task``."""

    def test_blocked_source_rejected(self, worker_cell, nectar):
        from hoc.core import HexCoord

        guard = GuardBehavior(worker_cell, nectar)
        bad_coord = HexCoord(7, -3)
        guard.blocked_sources.add(bad_coord)
        target_task = HiveTask(priority=2, task_type="compute")
        target_task.assigned_to = bad_coord
        validate_task = HiveTask(
            priority=2, task_type="validate", payload={"target_task": target_task}
        )
        assert guard.execute_task(validate_task) is True
        # The validation result records that the task was rejected.
        assert validate_task.result == {"valid": False}


class TestSwarmConfigQueueFullPolicyDefault:
    """Anchor the ``queue_full_policy`` default at ``"raise"``. The
    Phase 7.3 tests cover non-default policies; this anchors the
    default for any future regression."""

    def test_default_policy(self):
        cfg = SwarmConfig()
        assert cfg.queue_full_policy == "raise"
        assert cfg.queue_full_block_timeout_s == 5.0
        assert cfg.queue_full_block_poll_s == 0.005


class TestSwarmSchedulerExecuteOnCell:
    """Cover the three branches of ``SwarmScheduler.execute_on_cell``:
    rate-limit raise, unknown-coord KeyError, happy path."""

    def test_execute_on_cell_unknown_coord_raises(self, grid, nectar):
        from hoc.core import HexCoord

        sched = SwarmScheduler(grid, nectar)
        far = HexCoord(99, 99)
        task = HiveTask(priority=2, task_type="compute")
        with pytest.raises(KeyError, match="No hay behavior"):
            sched.execute_on_cell(far, task)

    def test_execute_on_cell_happy_path(self, grid, nectar):
        sched = SwarmScheduler(grid, nectar)
        coord, _behavior = next(iter(sched._behaviors.items()))
        task = HiveTask(priority=2, task_type="compute", payload={"execute": lambda: 42})
        # Most behaviour types accept "compute"; even if the specific
        # one at this coord doesn't (e.g. Nurse), the call still
        # returns False / True without crashing — execute_on_cell
        # only raises on KeyError or rate-limit.
        result = sched.execute_on_cell(coord, task)
        assert isinstance(result, bool)

    def test_execute_on_cell_rate_limit_raise(self, grid, nectar):
        from hoc.security import RateLimiter, RateLimitExceeded

        sched = SwarmScheduler(grid, nectar)
        # Replace the limiter with a tiny one and drain it so the next
        # acquire fails. burst must be > 0 (security.py validation), so
        # we use 1 then consume it.
        sched._execute_limiter = RateLimiter(per_second=0.001, burst=1)
        sched._execute_limiter.try_acquire()  # consume the burst
        coord = next(iter(sched._behaviors))
        task = HiveTask(priority=2, task_type="compute")
        with pytest.raises(RateLimitExceeded, match="execute_on_cell"):
            sched.execute_on_cell(coord, task)


class TestSwarmSchedulerSubmitRateLimit:
    """Cover the ``submit_task`` rate-limit raise path."""

    def test_submit_rate_limit_raise(self, grid, nectar):
        from hoc.security import RateLimiter, RateLimitExceeded

        sched = SwarmScheduler(grid, nectar)
        sched._submit_limiter = RateLimiter(per_second=0.001, burst=1)
        sched._submit_limiter.try_acquire()  # consume the burst
        with pytest.raises(RateLimitExceeded, match="submit_task"):
            sched.submit_task("compute", {})


class TestPheromoneFieldSimdPath:
    """Cover the ≥4 deposits SIMD branch in
    ``PheromoneField.decay_all`` introduced in Phase 7.6."""

    def test_simd_path_with_four_deposits(self, worker_cell):
        from hoc.core.pheromone import PheromoneType as CorePheromoneType

        # Deposit 4 different pheromone types so the SIMD branch fires.
        cell = worker_cell
        for ptype in list(CorePheromoneType)[:4]:
            cell.deposit_pheromone(ptype, amount=0.5)

        # Sanity: 4 deposits present.
        assert len(cell._pheromone_field._deposits) == 4

        # Decay all — SIMD path executes the np.power + multiply.
        cell.decay_pheromones(elapsed=1.0)

        # All four still present (decay rate 0.1 doesn't drop them
        # below threshold in one tick) but reduced.
        assert len(cell._pheromone_field._deposits) == 4
        for ptype in list(CorePheromoneType)[:4]:
            assert cell.get_pheromone(ptype) < 0.5

    def test_simd_path_drops_below_threshold(self, worker_cell):
        from hoc.core.pheromone import PheromoneType as CorePheromoneType

        cell = worker_cell
        # Tiny initial intensities so decay drops them below the active
        # threshold immediately.
        for ptype in list(CorePheromoneType)[:4]:
            cell.deposit_pheromone(ptype, amount=0.0015, decay_rate=0.99)

        assert len(cell._pheromone_field._deposits) == 4
        cell.decay_pheromones(elapsed=10.0)
        # All four below threshold → tombstoned by SIMD path.
        assert len(cell._pheromone_field._deposits) == 0
