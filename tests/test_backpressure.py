"""Phase 7.3 — :class:`SwarmScheduler` queue_full_policy tests.

Covers the four backpressure policies (``raise`` / ``drop_oldest`` /
``drop_newest`` / ``block``) and the ``tasks_dropped`` counter exposed
via :meth:`SwarmScheduler.get_stats`.

Brief test target: 10K tasks submitted with queue_size=100 +
drop_oldest → observed dropped == 9900.
"""

from __future__ import annotations

import threading
import time

import pytest

from hoc.core import HoneycombConfig, HoneycombGrid
from hoc.nectar import NectarFlow
from hoc.security import RateLimiter
from hoc.swarm import SwarmConfig, SwarmScheduler, TaskPriority, TaskState


def _make_scheduler(
    *, max_queue_size: int = 100, policy: str = "raise", **kwargs
) -> SwarmScheduler:
    """Helper: scheduler with a relaxed rate limiter so the queue
    fills before the rate limiter throttles us. The bench-grade
    drop test (10K submissions) needs the limiter out of the way."""
    grid = HoneycombGrid(HoneycombConfig(radius=1))
    nf = NectarFlow(grid)
    cfg = SwarmConfig(
        max_queue_size=max_queue_size,
        queue_full_policy=policy,  # type: ignore[arg-type]
        submit_rate_per_second=10_000_000.0,
        submit_rate_burst=10_000_000,
        **kwargs,
    )
    sched = SwarmScheduler(grid, nf, cfg)
    sched._submit_limiter = RateLimiter(
        per_second=cfg.submit_rate_per_second, burst=cfg.submit_rate_burst
    )
    return sched


# ───────────────────────────────────────────────────────────────────────────────
# Policy: raise (default; pre-Phase-7.3 behaviour)
# ───────────────────────────────────────────────────────────────────────────────


class TestRaisePolicy:
    def test_raise_when_full(self):
        sched = _make_scheduler(max_queue_size=2, policy="raise")
        sched.submit_task("compute", {})
        sched.submit_task("compute", {})
        with pytest.raises(RuntimeError, match="full"):
            sched.submit_task("compute", {})

    def test_no_drops_under_raise_policy(self):
        sched = _make_scheduler(max_queue_size=2, policy="raise")
        sched.submit_task("compute", {})
        sched.submit_task("compute", {})
        with pytest.raises(RuntimeError):
            sched.submit_task("compute", {})
        assert sched.get_stats()["tasks_dropped"] == 0


# ───────────────────────────────────────────────────────────────────────────────
# Policy: drop_oldest (eviction by lowest priority)
# ───────────────────────────────────────────────────────────────────────────────


class TestDropOldestPolicy:
    def test_drop_oldest_evicts_lowest_priority(self):
        sched = _make_scheduler(max_queue_size=2, policy="drop_oldest")
        # First two tasks both BACKGROUND priority. They fill the queue.
        t1 = sched.submit_task("compute", {}, priority=TaskPriority.BACKGROUND)
        t2 = sched.submit_task("compute", {}, priority=TaskPriority.BACKGROUND)
        # Third with HIGH priority forces eviction of one of the
        # BACKGROUND tasks (the lowest-priority worst candidate).
        t3 = sched.submit_task("compute", {}, priority=TaskPriority.HIGH)

        # Queue still bounded.
        assert sched.get_queue_size() == 2
        # The high-priority task is enqueued.
        ids_in_queue = {t.task_id for t in sched._task_queue}
        assert t3.task_id in ids_in_queue
        # Exactly one of t1/t2 was evicted.
        evicted = [t for t in (t1, t2) if t.state == TaskState.CANCELLED]
        assert len(evicted) == 1
        # tasks_dropped incremented.
        assert sched.get_stats()["tasks_dropped"] == 1

    def test_drop_oldest_keeps_higher_priority(self):
        """Submitting a CRITICAL on a queue full of LOW must not drop
        the CRITICAL — it must drop a LOW. Sanity check that the
        eviction picks the WORST priority, not the OLDEST chrono."""
        sched = _make_scheduler(max_queue_size=2, policy="drop_oldest")
        critical = sched.submit_task("compute", {}, priority=TaskPriority.CRITICAL)
        low = sched.submit_task("compute", {}, priority=TaskPriority.LOW)
        # Queue full. Submit a HIGH — should evict the LOW, keep the
        # CRITICAL.
        sched.submit_task("compute", {}, priority=TaskPriority.HIGH)

        ids_in_queue = {t.task_id for t in sched._task_queue}
        assert critical.task_id in ids_in_queue
        assert low.task_id not in ids_in_queue
        assert low.state == TaskState.CANCELLED

    def test_brief_load_10k_drops_9900(self):
        """Phase 7.3 brief target: 10K tasks submitted with
        queue_size=100 + drop_oldest → observed dropped == 9900.

        Note that "9900" assumes every overflow drops exactly one task,
        which matches the drop_oldest contract: each submit beyond the
        100th evicts one task.
        """
        sched = _make_scheduler(max_queue_size=100, policy="drop_oldest")
        for _ in range(10_000):
            sched.submit_task("compute", {})

        # Queue stays at the cap.
        assert sched.get_queue_size() == 100
        # Drop count = 10000 - 100 = 9900.
        assert sched.get_stats()["tasks_dropped"] == 9_900


# ───────────────────────────────────────────────────────────────────────────────
# Policy: drop_newest
# ───────────────────────────────────────────────────────────────────────────────


class TestDropNewestPolicy:
    def test_drop_newest_rejects_late_arrivals(self):
        sched = _make_scheduler(max_queue_size=2, policy="drop_newest")
        a = sched.submit_task("compute", {})
        b = sched.submit_task("compute", {})
        # Third submission rejected.
        c = sched.submit_task("compute", {})

        # The new task is returned but marked CANCELLED.
        assert c.state == TaskState.CANCELLED
        # Original tasks still queued.
        ids_in_queue = {t.task_id for t in sched._task_queue}
        assert a.task_id in ids_in_queue
        assert b.task_id in ids_in_queue
        assert c.task_id not in ids_in_queue
        # Drop counter advanced.
        assert sched.get_stats()["tasks_dropped"] == 1

    def test_drop_newest_does_not_modify_queue_order(self):
        sched = _make_scheduler(max_queue_size=3, policy="drop_newest")
        first = sched.submit_task("compute", {})
        second = sched.submit_task("compute", {})
        third = sched.submit_task("compute", {})
        # Full. Try a fourth.
        sched.submit_task("compute", {})
        # Original three still there in original order (heap may reorder
        # by priority, which all are NORMAL so insertion order isn't
        # canonical — instead check membership).
        ids = {t.task_id for t in sched._task_queue}
        assert {first.task_id, second.task_id, third.task_id} == ids


# ───────────────────────────────────────────────────────────────────────────────
# Policy: block (poll + timeout)
# ───────────────────────────────────────────────────────────────────────────────


class TestBlockPolicy:
    def test_block_succeeds_when_room_freed(self):
        sched = _make_scheduler(
            max_queue_size=1,
            policy="block",
            queue_full_block_timeout_s=2.0,
            queue_full_block_poll_s=0.01,
        )
        sched.submit_task("compute", {})  # fill the slot

        # Background thread frees the slot after 100ms.
        def _free_slot():
            time.sleep(0.1)
            with sched._lock:
                # Drain whatever is in the queue.
                if sched._task_queue:
                    t = sched._task_queue.pop(0)
                    sched._task_index.pop(t.task_id, None)

        threading.Thread(target=_free_slot, daemon=True).start()

        # This blocks until the background thread frees the slot. Should
        # complete well within the 2s timeout.
        new_task = sched.submit_task("compute", {})
        assert new_task.state == TaskState.PENDING

    def test_block_times_out_when_queue_stays_full(self):
        sched = _make_scheduler(
            max_queue_size=1,
            policy="block",
            queue_full_block_timeout_s=0.05,
            queue_full_block_poll_s=0.01,
        )
        sched.submit_task("compute", {})  # fill the slot

        with pytest.raises(RuntimeError, match="block policy timed out"):
            sched.submit_task("compute", {})


# ───────────────────────────────────────────────────────────────────────────────
# Counter persistence
# ───────────────────────────────────────────────────────────────────────────────


class TestTasksDroppedCounter:
    def test_drops_persist_across_to_dict(self):
        """``tasks_dropped`` survives a checkpoint round-trip via
        :meth:`SwarmScheduler.to_dict` / :meth:`from_dict`."""
        sched = _make_scheduler(max_queue_size=1, policy="drop_oldest")
        for _ in range(5):
            sched.submit_task("compute", {})

        assert sched._tasks_dropped == 4

        d = sched.to_dict()
        new_grid = HoneycombGrid(HoneycombConfig(radius=1))
        new_nf = NectarFlow(new_grid)
        restored = SwarmScheduler.from_dict(d, new_grid, new_nf)
        assert restored._tasks_dropped == 4
        assert restored.get_stats()["tasks_dropped"] == 4

    def test_default_policy_is_raise(self):
        sched = _make_scheduler()  # no explicit policy
        assert sched.config.queue_full_policy == "raise"
