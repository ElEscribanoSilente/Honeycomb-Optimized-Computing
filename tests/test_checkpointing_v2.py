"""Phase 7.10 — checkpoint v2 + ``SwarmScheduler.to_dict`` / ``from_dict``
+ ``HoneycombGrid.checkpoint(scheduler=...)`` tests.

Covers the gap left open in Phase 6.4: the SwarmScheduler's task
queue is now serialisable, the checkpoint blob version is bumped from
``0x01`` to ``0x02``, and ``decode_blob`` accepts both. The wire
format itself is identical between the two versions (header layout
unchanged) — only the version byte and the optional ``"scheduler"``
key in the inner payload differ.

Test surface:

- Version byte: ``encode_blob`` always writes ``0x02``, ``decode_blob``
  accepts both ``0x01`` and ``0x02``, and unknown versions are still
  rejected with ``ValueError``.
- HiveTask round-trip: priorities, ids, target_cells, state, FSM
  history all survive a serialise → deserialise pass.
- Callback sentinel: pre-checkpoint callback is replaced with the
  module-level sentinel; restored tasks expose
  ``callback_needs_reattach=True``.
- Payload sentinel: lambdas / non-primitive values inside ``payload``
  become the unserializable marker dict.
- 100 pending + 50 in-flight: the load profile from the brief
  round-trips with correct counts and per-state distribution.
- ``HoneycombGrid.checkpoint(scheduler=...)``: end-to-end where the
  grid blob carries the scheduler snapshot, restore happens in two
  steps (grid first, then scheduler against the new grid).
- v1-shaped blob (no ``scheduler`` key): restore_from_checkpoint
  returns ``None`` instead of raising.
"""

from __future__ import annotations

import heapq

import pytest

from hoc.core import HexCoord, HoneycombConfig, HoneycombGrid
from hoc.nectar import NectarFlow
from hoc.security import sign_payload
from hoc.storage.checkpoint import (
    HEADER_LEN,
    HMAC_TAG_LEN,
    SUPPORTED_VERSIONS,
    VERSION_BYTE,
    decode_blob,
    encode_blob,
)
from hoc.swarm import (
    HiveTask,
    SwarmScheduler,
    TaskPriority,
    TaskState,
)

# ───────────────────────────────────────────────────────────────────────────────
# Version byte sanity
# ───────────────────────────────────────────────────────────────────────────────


class TestVersionByteUpgrade:
    def test_default_version_is_v2(self):
        """Phase 7.10 bumped VERSION_BYTE from 0x01 to 0x02."""
        assert VERSION_BYTE == 0x02
        assert 0x01 in SUPPORTED_VERSIONS
        assert 0x02 in SUPPORTED_VERSIONS

    def test_encode_blob_emits_v2(self):
        blob = encode_blob({"hello": "world"})
        assert blob[0] == 0x02

    def test_decode_blob_accepts_v1_legacy(self):
        """Phase 6 blobs (0x01) must still decode under Phase 7."""
        body = b"\x00" + _mscs_dumps({"legacy": True})  # flag=0, plain mscs
        tag = sign_payload(body)
        legacy_blob = bytes([0x01]) + tag + body
        assert decode_blob(legacy_blob) == {"legacy": True}

    def test_decode_blob_accepts_v2(self):
        blob = encode_blob({"k": "v"})
        assert decode_blob(blob) == {"k": "v"}

    def test_decode_blob_rejects_unknown_version(self):
        body = b"\x00" + _mscs_dumps({"x": 1})
        tag = sign_payload(body)
        bogus = bytes([0x99]) + tag + body
        with pytest.raises(ValueError, match="unsupported.*version"):
            decode_blob(bogus)


def _mscs_dumps(obj: object) -> bytes:
    """Helper used by version tests to forge legacy v1 blobs.

    Imported lazily so the test module stays importable even if mscs
    rejects the registry on older Pythons (the security layer wraps
    mscs internally for production paths)."""
    import mscs as _mscs

    return bytes(_mscs.dumps(obj))


# ───────────────────────────────────────────────────────────────────────────────
# HiveTask serialization
# ───────────────────────────────────────────────────────────────────────────────


class TestHiveTaskSerialization:
    def test_minimal_task_roundtrip(self):
        task = HiveTask(priority=2, task_type="compute")
        original_id = task.task_id
        d = task.to_dict()
        restored = HiveTask.from_dict(d)
        assert restored.task_id == original_id
        assert restored.task_type == "compute"
        assert restored.priority == 2
        assert restored.state == TaskState.PENDING

    def test_target_cell_preserved(self):
        task = HiveTask(priority=1, task_type="compute", target_cell=HexCoord(2, -3))
        restored = HiveTask.from_dict(task.to_dict())
        assert restored.target_cell == HexCoord(2, -3)

    def test_assigned_to_preserved(self):
        task = HiveTask(priority=1, task_type="compute", assigned_to=HexCoord(0, 1))
        restored = HiveTask.from_dict(task.to_dict())
        assert restored.assigned_to == HexCoord(0, 1)

    def test_running_state_preserved(self):
        task = HiveTask(priority=1, task_type="compute")
        task.state = TaskState.RUNNING
        restored = HiveTask.from_dict(task.to_dict())
        assert restored.state == TaskState.RUNNING

    def test_failed_state_preserved(self):
        task = HiveTask(priority=1, task_type="compute")
        task.state = TaskState.RUNNING
        task.state = TaskState.FAILED
        restored = HiveTask.from_dict(task.to_dict())
        assert restored.state == TaskState.FAILED

    def test_attempts_and_max_attempts(self):
        task = HiveTask(priority=1, task_type="compute", max_attempts=5)
        task.attempts = 2
        restored = HiveTask.from_dict(task.to_dict())
        assert restored.attempts == 2
        assert restored.max_attempts == 5

    def test_callback_replaced_with_sentinel(self):
        task = HiveTask(
            priority=1,
            task_type="compute",
            callback=lambda r: None,
        )
        d = task.to_dict()
        assert d["callback"] == HiveTask.SENTINEL_CALLBACK_REATTACH
        restored = HiveTask.from_dict(d)
        assert restored.callback is None
        assert restored.callback_needs_reattach is True

    def test_no_callback_no_reattach_flag(self):
        task = HiveTask(priority=1, task_type="compute")
        restored = HiveTask.from_dict(task.to_dict())
        assert restored.callback_needs_reattach is False

    def test_payload_with_lambda_sentinelized(self):
        task = HiveTask(
            priority=1,
            task_type="compute",
            payload={"execute": lambda: 42, "x": 10, "name": "hello"},
        )
        d = task.to_dict()
        assert d["payload"]["execute"] == {"__hoc_unserializable__": "function"}
        assert d["payload"]["x"] == 10
        assert d["payload"]["name"] == "hello"
        restored = HiveTask.from_dict(d)
        assert restored.payload["x"] == 10
        assert restored.payload["name"] == "hello"
        assert "__hoc_unserializable__" in restored.payload["execute"]

    def test_nested_primitive_payload_survives(self):
        task = HiveTask(
            priority=1,
            task_type="compute",
            payload={"items": [1, 2, 3], "meta": {"author": "raul"}},
        )
        restored = HiveTask.from_dict(task.to_dict())
        assert restored.payload == {"items": [1, 2, 3], "meta": {"author": "raul"}}

    def test_error_field_survives(self):
        task = HiveTask(priority=1, task_type="compute")
        task.state = TaskState.RUNNING
        task.state = TaskState.FAILED
        task.error = "intentional"
        restored = HiveTask.from_dict(task.to_dict())
        assert restored.error == "intentional"

    def test_priority_ordering_preserved_after_roundtrip(self):
        critical = HiveTask(priority=TaskPriority.CRITICAL.value, task_type="c")
        normal = HiveTask(priority=TaskPriority.NORMAL.value, task_type="c")
        restored_c = HiveTask.from_dict(critical.to_dict())
        restored_n = HiveTask.from_dict(normal.to_dict())
        h: list[HiveTask] = []
        heapq.heappush(h, restored_n)
        heapq.heappush(h, restored_c)
        first = heapq.heappop(h)
        # Critical sorts first because dataclass(order=True) compares
        # priority first, then created_at — both fields survived.
        assert first.priority == TaskPriority.CRITICAL.value


# ───────────────────────────────────────────────────────────────────────────────
# SwarmScheduler serialization
# ───────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def grid() -> HoneycombGrid:
    return HoneycombGrid(HoneycombConfig(radius=2))


@pytest.fixture
def nectar_flow(grid: HoneycombGrid) -> NectarFlow:
    return NectarFlow(grid)


@pytest.fixture
def scheduler(grid: HoneycombGrid, nectar_flow: NectarFlow) -> SwarmScheduler:
    return SwarmScheduler(grid, nectar_flow)


class TestSchedulerSerialization:
    def test_empty_scheduler_roundtrip(self, scheduler, grid, nectar_flow):
        d = scheduler.to_dict()
        restored = SwarmScheduler.from_dict(d, grid, nectar_flow)
        assert restored.get_queue_size() == 0
        assert restored._tick_count == 0

    def test_tick_count_preserved(self, scheduler, grid, nectar_flow):
        scheduler._tick_count = 42
        scheduler._tasks_completed = 10
        scheduler._tasks_failed = 3
        d = scheduler.to_dict()
        restored = SwarmScheduler.from_dict(d, grid, nectar_flow)
        assert restored._tick_count == 42
        assert restored._tasks_completed == 10
        assert restored._tasks_failed == 3

    def test_pending_tasks_preserved(self, scheduler, grid, nectar_flow):
        scheduler.submit_task("compute", {"x": 1})
        scheduler.submit_task("compute", {"x": 2}, priority=TaskPriority.CRITICAL)
        scheduler.submit_task("explore", {"area": "north"})
        d = scheduler.to_dict()
        restored = SwarmScheduler.from_dict(d, grid, nectar_flow)
        assert restored.get_queue_size() == 3
        types = {t.task_type for t in restored._task_queue}
        assert types == {"compute", "explore"}

    def test_in_flight_tasks_preserved(self, scheduler, grid, nectar_flow):
        running_task = scheduler.submit_task("compute", {})
        running_task.state = TaskState.RUNNING
        running_task.assigned_to = HexCoord(1, 0)
        d = scheduler.to_dict()
        restored = SwarmScheduler.from_dict(d, grid, nectar_flow)
        assert restored.get_queue_size() == 1
        rt = next(iter(restored._task_queue))
        assert rt.state == TaskState.RUNNING
        assert rt.assigned_to == HexCoord(1, 0)

    def test_index_rebuilt(self, scheduler, grid, nectar_flow):
        t1 = scheduler.submit_task("compute", {})
        t2 = scheduler.submit_task("compute", {})
        d = scheduler.to_dict()
        restored = SwarmScheduler.from_dict(d, grid, nectar_flow)
        assert restored.get_task(t1.task_id) is not None
        assert restored.get_task(t2.task_id) is not None

    def test_behaviors_rebuilt_from_grid(self, scheduler, grid, nectar_flow):
        scheduler.submit_task("compute", {})
        d = scheduler.to_dict()
        restored = SwarmScheduler.from_dict(d, grid, nectar_flow)
        # Phase 7.10: behaviors are rebuilt from grid+config, not from the
        # checkpoint. Count must match the grid's worker-cell count.
        from hoc.core import WorkerCell

        worker_count = sum(1 for c in grid._cells.values() if isinstance(c, WorkerCell))
        assert len(restored._behaviors) == worker_count

    def test_100_pending_50_in_flight_brief_load(self, scheduler, grid, nectar_flow):
        """Brief: round-trip of scheduler with 100 pending + 50 in-flight."""
        # Submit 100 pending. We patch out the rate limiter so the burst
        # cap doesn't trip — the brief's load is the test here, not rate
        # limiting (covered by test_swarm.py).
        scheduler.config.submit_rate_per_second = 1_000_000.0
        scheduler.config.submit_rate_burst = 1_000_000
        scheduler.config.max_queue_size = 1_000
        # Re-seat the limiter to pick up the new rate (it was instantiated
        # in __init__ with the old config values).
        from hoc.security import RateLimiter as _RL

        scheduler._submit_limiter = _RL(
            per_second=scheduler.config.submit_rate_per_second,
            burst=scheduler.config.submit_rate_burst,
        )

        for i in range(100):
            scheduler.submit_task("compute", {"i": i})

        # Mark 50 tasks as RUNNING (in-flight). The other 100 stay PENDING.
        in_flight = scheduler.submit_task("compute", {"flagship": True})
        in_flight.state = TaskState.RUNNING
        for i in range(49):
            t = scheduler.submit_task("compute", {"flagship_buddy": i})
            t.state = TaskState.RUNNING

        d = scheduler.to_dict()
        restored = SwarmScheduler.from_dict(d, grid, nectar_flow, scheduler.config)
        assert restored.get_queue_size() == 150

        pending_count = sum(1 for t in restored._task_queue if t.state == TaskState.PENDING)
        running_count = sum(1 for t in restored._task_queue if t.state == TaskState.RUNNING)
        assert pending_count == 100
        assert running_count == 50


# ───────────────────────────────────────────────────────────────────────────────
# Grid + scheduler bundled checkpoint
# ───────────────────────────────────────────────────────────────────────────────


class TestGridCheckpointWithScheduler:
    def test_grid_only_checkpoint_returns_none_for_scheduler_restore(
        self, grid, nectar_flow, tmp_path
    ):
        path = tmp_path / "grid_only.bin"
        grid.checkpoint(path)
        restored_grid = HoneycombGrid.restore_from_checkpoint(path)
        new_nectar = NectarFlow(restored_grid)
        sched = SwarmScheduler.restore_from_checkpoint(path, restored_grid, new_nectar)
        # No "scheduler" key in payload → cleanly None.
        assert sched is None
        # Grid still restored fully.
        assert len(restored_grid._cells) == len(grid._cells)

    def test_bundled_checkpoint_round_trip(self, grid, nectar_flow, scheduler, tmp_path):
        scheduler.submit_task("compute", {"a": 1})
        scheduler.submit_task("compute", {"a": 2})
        scheduler._tick_count = 7

        path = tmp_path / "bundled.bin"
        grid.checkpoint(path, scheduler=scheduler)

        restored_grid = HoneycombGrid.restore_from_checkpoint(path)
        new_nectar = NectarFlow(restored_grid)
        restored_sched = SwarmScheduler.restore_from_checkpoint(path, restored_grid, new_nectar)

        assert restored_sched is not None
        assert restored_sched._tick_count == 7
        assert restored_sched.get_queue_size() == 2
        assert len(restored_grid._cells) == len(grid._cells)

    def test_bundled_checkpoint_compressed(self, grid, nectar_flow, scheduler, tmp_path):
        for i in range(20):
            scheduler.submit_task("compute", {"i": i})

        plain_path = tmp_path / "plain.bin"
        zlib_path = tmp_path / "zlib.bin"
        grid.checkpoint(plain_path, scheduler=scheduler, compress=False)
        grid.checkpoint(zlib_path, scheduler=scheduler, compress=True)

        # On a bundled blob with 20 repetitive tasks, zlib should win.
        assert zlib_path.stat().st_size <= plain_path.stat().st_size
        # Both decode to a valid scheduler.
        for path in (plain_path, zlib_path):
            new_grid = HoneycombGrid.restore_from_checkpoint(path)
            new_nectar = NectarFlow(new_grid)
            restored = SwarmScheduler.restore_from_checkpoint(path, new_grid, new_nectar)
            assert restored is not None
            assert restored.get_queue_size() == 20

    def test_v1_shaped_payload_returns_none(self, tmp_path):
        """Phase 7.10: a v2 blob whose payload happens to lack the
        scheduler key (e.g. a non-bundled checkpoint) returns None
        from SwarmScheduler.restore_from_checkpoint, not an error."""
        # encode_blob writes 0x02 always now, but the payload is the
        # plain grid dict without the "scheduler" key.
        grid = HoneycombGrid(HoneycombConfig(radius=1))
        path = tmp_path / "no_sched.bin"
        grid.checkpoint(path)  # no scheduler kwarg
        new_grid = HoneycombGrid.restore_from_checkpoint(path)
        new_nectar = NectarFlow(new_grid)
        sched = SwarmScheduler.restore_from_checkpoint(path, new_grid, new_nectar)
        assert sched is None


# ───────────────────────────────────────────────────────────────────────────────
# Atomicity / interaction with existing v1 checkpoint tests
# ───────────────────────────────────────────────────────────────────────────────


class TestPhase6CompatBackcompat:
    def test_phase6_grid_only_checkpoint_still_works(self, tmp_path):
        """Phase 6.3 round-trip stays byte-identical post-version-bump
        in terms of behaviour — the only observable change is the version
        byte now reads 0x02 in fresh blobs. Hand-forged 0x01 blobs from
        the wild keep working."""
        cfg = HoneycombConfig(radius=2, vcores_per_cell=4)
        g = HoneycombGrid(cfg)
        g._tick_count = 99
        path = tmp_path / "grid.bin"
        g.checkpoint(path)
        # Verify the on-disk byte agrees with VERSION_BYTE.
        assert path.read_bytes()[0] == VERSION_BYTE
        # Forge an 0x01 wrapper around the same payload — it should
        # decode and rebuild identically.
        blob = path.read_bytes()
        body = blob[HEADER_LEN:]
        new_tag = sign_payload(body)
        v1_blob = bytes([0x01]) + new_tag + body
        path.write_bytes(v1_blob)
        restored = HoneycombGrid.restore_from_checkpoint(path)
        assert restored._tick_count == 99
        assert len(restored._cells) == len(g._cells)
        # HMAC tag length sanity — Phase 6 size unchanged.
        assert HMAC_TAG_LEN == 32
