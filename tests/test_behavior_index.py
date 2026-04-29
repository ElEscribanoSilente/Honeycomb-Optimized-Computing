"""Phase 7.5 — :class:`BehaviorIndex` tests + scheduler-integration
sanity tests.

Covers:

- API surface: insert / pop_best / remove + lazy size_for / compact.
- Heap order: pop_best returns highest-importance (lowest priority
  value) task first; tie-broken FIFO by insertion sequence.
- Tombstone semantics: pop_best claims; remove tombstones; insert
  clears tombstone (the FAILED → PENDING retry path).
- Compact: drops dead heap entries and frees the tombstone set.
- Cross-behavior dedup: a task inserted into N heaps gets popped only
  once (first behavior wins; others' pop skips the tombstoned entry).
- Scheduler integration: SwarmScheduler.tick still routes tasks
  correctly via the index — the behaviorIndex isn't visible from
  outside but the queue size + completion counts must match the
  pre-Phase-7.5 contract.
- Type-routing: type-mismatched tasks land in zero heaps; pinned
  tasks (target_cell set) land in exactly one heap.
"""

from __future__ import annotations

import pytest

from hoc.core import HexCoord, HoneycombConfig, HoneycombGrid, WorkerCell
from hoc.nectar import NectarFlow
from hoc.swarm import (
    BehaviorIndex,
    ForagerBehavior,
    GuardBehavior,
    HiveTask,
    NurseBehavior,
    ScoutBehavior,
    SwarmConfig,
    SwarmScheduler,
    TaskPriority,
    TaskState,
)

# ───────────────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def grid() -> HoneycombGrid:
    return HoneycombGrid(HoneycombConfig(radius=2))


@pytest.fixture
def nectar(grid: HoneycombGrid) -> NectarFlow:
    return NectarFlow(grid)


@pytest.fixture
def worker_cell(grid: HoneycombGrid):
    return next(c for c in grid._cells.values() if isinstance(c, WorkerCell))


@pytest.fixture
def forager(worker_cell, nectar) -> ForagerBehavior:
    return ForagerBehavior(worker_cell, nectar)


@pytest.fixture
def nurse(worker_cell, nectar) -> NurseBehavior:
    return NurseBehavior(worker_cell, nectar)


# ───────────────────────────────────────────────────────────────────────────────
# Pure BehaviorIndex semantics
# ───────────────────────────────────────────────────────────────────────────────


class TestBehaviorIndexBasics:
    def test_pop_empty_returns_none(self, forager):
        idx = BehaviorIndex()
        idx.register_behavior(forager)
        assert idx.pop_best(forager) is None

    def test_unregistered_pop_returns_none(self, forager):
        idx = BehaviorIndex()
        # Not registered — heap missing entirely.
        assert idx.pop_best(forager) is None

    def test_insert_then_pop(self, forager):
        idx = BehaviorIndex()
        idx.register_behavior(forager)
        task = HiveTask(priority=2, task_type="compute")
        idx.insert(task, forager)
        popped = idx.pop_best(forager)
        assert popped is task
        # Heap drained.
        assert idx.pop_best(forager) is None

    def test_priority_order(self, forager):
        """Lower priority value pops first (matches min-heap +
        TaskPriority.CRITICAL=0 convention)."""
        idx = BehaviorIndex()
        idx.register_behavior(forager)
        low = HiveTask(priority=TaskPriority.LOW.value, task_type="c")
        high = HiveTask(priority=TaskPriority.HIGH.value, task_type="c")
        critical = HiveTask(priority=TaskPriority.CRITICAL.value, task_type="c")
        idx.insert(low, forager)
        idx.insert(critical, forager)
        idx.insert(high, forager)

        assert idx.pop_best(forager) is critical
        assert idx.pop_best(forager) is high
        assert idx.pop_best(forager) is low

    def test_fifo_tie_break_at_same_priority(self, forager):
        """Equal-priority tasks come out in insertion order — the
        sequence counter ensures deterministic behaviour."""
        idx = BehaviorIndex()
        idx.register_behavior(forager)
        first = HiveTask(priority=2, task_type="c", task_id="t1")
        second = HiveTask(priority=2, task_type="c", task_id="t2")
        third = HiveTask(priority=2, task_type="c", task_id="t3")
        idx.insert(first, forager)
        idx.insert(second, forager)
        idx.insert(third, forager)
        assert idx.pop_best(forager) is first
        assert idx.pop_best(forager) is second
        assert idx.pop_best(forager) is third

    def test_remove_tombstones(self, forager):
        idx = BehaviorIndex()
        idx.register_behavior(forager)
        task = HiveTask(priority=2, task_type="c", task_id="t1")
        idx.insert(task, forager)
        assert idx.remove("t1") is True
        # Removed → pop_best skips it.
        assert idx.pop_best(forager) is None

    def test_remove_idempotent_returns_false_second_time(self, forager):
        idx = BehaviorIndex()
        idx.register_behavior(forager)
        task = HiveTask(priority=2, task_type="c", task_id="t1")
        idx.insert(task, forager)
        assert idx.remove("t1") is True
        assert idx.remove("t1") is False

    def test_pop_best_auto_tombstones(self, forager, nurse):
        """A task in two heaps is popped once: the second heap's
        pop_best skips the tombstoned entry."""
        idx = BehaviorIndex()
        idx.register_behavior(forager)
        idx.register_behavior(nurse)
        task = HiveTask(priority=1, task_type="c", task_id="t1")
        idx.insert(task, forager)
        idx.insert(task, nurse)
        first = idx.pop_best(forager)
        second = idx.pop_best(nurse)
        assert first is task
        assert second is None

    def test_re_insert_clears_tombstone(self, forager):
        """Phase 7.5 retry path: FAILED task transitions back to
        PENDING and is re-inserted. The tombstone from the previous
        pop must be cleared so the new insertion is visible."""
        idx = BehaviorIndex()
        idx.register_behavior(forager)
        task = HiveTask(priority=1, task_type="c", task_id="t1")
        idx.insert(task, forager)
        assert idx.pop_best(forager) is task  # tombstoned

        # Re-insert — same task_id, tombstone clears.
        idx.insert(task, forager)
        assert idx.pop_best(forager) is task

    def test_size_for(self, forager):
        idx = BehaviorIndex()
        idx.register_behavior(forager)
        assert idx.size_for(forager) == 0
        for i in range(5):
            idx.insert(HiveTask(priority=2, task_type="c", task_id=f"t{i}"), forager)
        assert idx.size_for(forager) == 5

    def test_size_for_excludes_tombstoned(self, forager):
        idx = BehaviorIndex()
        idx.register_behavior(forager)
        for i in range(3):
            idx.insert(HiveTask(priority=2, task_type="c", task_id=f"t{i}"), forager)
        idx.remove("t1")
        assert idx.size_for(forager) == 2

    def test_size_for_unregistered(self, forager):
        idx = BehaviorIndex()
        # forager not registered.
        assert idx.size_for(forager) == 0


class TestBehaviorIndexCompact:
    def test_compact_returns_zero_when_no_tombstones(self, forager):
        idx = BehaviorIndex()
        idx.register_behavior(forager)
        idx.insert(HiveTask(priority=1, task_type="c"), forager)
        assert idx.compact() == 0

    def test_compact_drops_tombstoned_entries(self, forager):
        idx = BehaviorIndex()
        idx.register_behavior(forager)
        for i in range(10):
            idx.insert(HiveTask(priority=2, task_type="c", task_id=f"t{i}"), forager)

        # Tombstone half via remove.
        for i in range(0, 10, 2):
            idx.remove(f"t{i}")

        removed = idx.compact()
        assert removed == 5
        # After compact, surviving entries are still poppable in order.
        survivors = []
        while True:
            task = idx.pop_best(forager)
            if task is None:
                break
            survivors.append(task.task_id)
        assert survivors == [f"t{i}" for i in range(1, 10, 2)]

    def test_compact_clears_tombstone_set(self, forager):
        idx = BehaviorIndex()
        idx.register_behavior(forager)
        idx.insert(HiveTask(priority=1, task_type="c", task_id="t1"), forager)
        idx.remove("t1")
        idx.compact()
        # After compact, re-inserting an id that was previously
        # tombstoned should be visible — the tombstone set was cleared.
        idx.insert(HiveTask(priority=1, task_type="c", task_id="t1"), forager)
        assert idx.pop_best(forager) is not None


# ───────────────────────────────────────────────────────────────────────────────
# Type-routing inside SwarmScheduler
# ───────────────────────────────────────────────────────────────────────────────


class TestSchedulerTypeRouting:
    def test_compute_task_routed_to_foragers_only(self, grid, nectar):
        sched = SwarmScheduler(grid, nectar)
        task = HiveTask(priority=2, task_type="compute")
        targets = sched._route_task_to_behaviors(task)
        # Foragers are the catch-all for unspecialized types.
        assert all(isinstance(b, ForagerBehavior) for b in targets)
        # And the count matches the foragers-ratio of the grid.
        n_foragers = sum(1 for b in sched._behaviors.values() if isinstance(b, ForagerBehavior))
        assert len(targets) == n_foragers

    def test_spawn_task_routed_to_nurses_only(self, grid, nectar):
        sched = SwarmScheduler(grid, nectar)
        task = HiveTask(priority=2, task_type="spawn")
        targets = sched._route_task_to_behaviors(task)
        assert all(isinstance(b, NurseBehavior) for b in targets)

    def test_warmup_task_routed_to_nurses_only(self, grid, nectar):
        sched = SwarmScheduler(grid, nectar)
        task = HiveTask(priority=2, task_type="warmup")
        targets = sched._route_task_to_behaviors(task)
        assert all(isinstance(b, NurseBehavior) for b in targets)

    def test_explore_task_routed_to_scouts_only(self, grid, nectar):
        sched = SwarmScheduler(grid, nectar)
        task = HiveTask(priority=2, task_type="explore")
        targets = sched._route_task_to_behaviors(task)
        assert all(isinstance(b, ScoutBehavior) for b in targets)

    def test_validate_task_routed_to_guards_only(self, grid, nectar):
        sched = SwarmScheduler(grid, nectar)
        task = HiveTask(priority=2, task_type="validate")
        targets = sched._route_task_to_behaviors(task)
        assert all(isinstance(b, GuardBehavior) for b in targets)

    def test_pinned_task_routed_to_one_behavior(self, grid, nectar):
        sched = SwarmScheduler(grid, nectar)
        target_coord = next(iter(sched._behaviors))
        task = HiveTask(priority=2, task_type="compute", target_cell=target_coord)
        targets = sched._route_task_to_behaviors(task)
        # Only one behavior matches by coord, and only if it accepts
        # the type. It might be a Nurse — in that case the task gets
        # routed to nobody (which is correct: pinned compute task to
        # a Nurse cell never made sense).
        target_behavior = sched._behaviors[target_coord]
        if isinstance(target_behavior, ForagerBehavior):
            assert targets == [target_behavior]
        else:
            assert targets == []

    def test_pinned_task_to_nonexistent_coord_routed_to_nobody(self, grid, nectar):
        sched = SwarmScheduler(grid, nectar)
        task = HiveTask(priority=2, task_type="compute", target_cell=HexCoord(99, 99))
        assert sched._route_task_to_behaviors(task) == []


# ───────────────────────────────────────────────────────────────────────────────
# Scheduler integration: index keeps pre-Phase-7.5 throughput contract
# ───────────────────────────────────────────────────────────────────────────────


class TestSchedulerIntegration:
    def test_submitted_task_lands_in_index(self, grid, nectar):
        sched = SwarmScheduler(grid, nectar)
        sched.submit_task("compute", {})
        # Find a forager and check its heap is non-empty.
        forager_coord, forager_b = next(
            (c, b) for c, b in sched._behaviors.items() if isinstance(b, ForagerBehavior)
        )
        assert sched._behavior_index.size_for(forager_b) >= 1

    def test_tick_drains_index_pending_tasks(self, grid, nectar):
        sched = SwarmScheduler(grid, nectar)
        # Submit enough compute tasks that several foragers drain on a
        # single tick.
        for _ in range(5):
            sched.submit_task("compute", {})
        sched.run_tick_sync()
        # Some tasks must have been processed.
        assert sched._tasks_completed + sched._tasks_failed > 0

    def test_tick_processes_specialized_tasks(self, grid, nectar):
        sched = SwarmScheduler(grid, nectar)
        sched.submit_task("spawn", {"spec": {}})
        sched.run_tick_sync()
        # Nurses handle spawn tasks — at least one should have completed.
        assert sched._tasks_completed >= 1

    def test_cancel_removes_from_index(self, grid, nectar):
        sched = SwarmScheduler(grid, nectar)
        task = sched.submit_task("compute", {})
        # Confirm it's in the index.
        forager_b = next(b for b in sched._behaviors.values() if isinstance(b, ForagerBehavior))
        assert sched._behavior_index.size_for(forager_b) >= 1
        sched.cancel_task(task.task_id)
        # After cancel the next pop_best for that behavior should not
        # return this task.
        sched.run_tick_sync()
        # Task is now CANCELLED, not RUNNING.
        assert task.state is TaskState.CANCELLED

    def test_compact_runs_periodically(self, grid, nectar):
        """Verify the periodic compact() pass keeps tombstones bounded.

        ``BehaviorIndex`` uses ``__slots__`` so we can't monkeypatch
        ``compact``; instead we observe its effect: after enough ticks
        the tombstone set is empty (because compact cleared it on the
        Nth tick) even though we cancelled many tasks along the way.
        """
        sched = SwarmScheduler(grid, nectar)
        # Bury a few cancelled tasks to grow tombstones.
        for _ in range(20):
            t = sched.submit_task("compute", {})
            sched.cancel_task(t.task_id)
        # Tombstones are now non-empty.
        assert len(sched._behavior_index._tombstoned) > 0

        # Run exactly INDEX_COMPACT_INTERVAL_TICKS ticks.
        for _ in range(SwarmScheduler.INDEX_COMPACT_INTERVAL_TICKS):
            sched.run_tick_sync()

        # Compact ran on the final tick → tombstone set cleared.
        assert sched._behavior_index._tombstoned == set()


# ───────────────────────────────────────────────────────────────────────────────
# Brief perf bench (smoke test, not the bench-grade measurement)
# ───────────────────────────────────────────────────────────────────────────────


class TestSchedulerThroughputSmoke:
    def test_1000_tasks_radius3(self):
        """Smoke test: scheduler handles 1000 tasks on a radius-3 grid
        without exploding. The actual bench (≥5× target vs Phase 6) is
        in benchmarks/bench_swarm.py — this just verifies correctness
        at scale."""
        cfg = HoneycombConfig(radius=3)
        grid = HoneycombGrid(cfg)
        nf = NectarFlow(grid)
        sc = SwarmConfig(
            max_queue_size=2000,
            submit_rate_per_second=1_000_000.0,
            submit_rate_burst=1_000_000,
        )
        sched = SwarmScheduler(grid, nf, sc)
        # Re-seat the limiter to use the bench-friendly rates.
        from hoc.security import RateLimiter as _RL

        sched._submit_limiter = _RL(
            per_second=sc.submit_rate_per_second, burst=sc.submit_rate_burst
        )

        for _ in range(1000):
            sched.submit_task("compute", {})

        # Queue is full.
        assert sched.get_queue_size() == 1000

        # Run several ticks; verify processed > 0 and no exceptions.
        total_processed = 0
        for _ in range(20):
            r = sched.run_tick_sync()
            total_processed += r["tasks_processed"]
        assert total_processed > 0
