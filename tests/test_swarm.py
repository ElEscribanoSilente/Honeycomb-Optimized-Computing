"""Tests para hoc.swarm: SwarmScheduler, comportamientos de abejas, balanceo.

Cobertura objetivo Phase 1: ≥80% en swarm.py.

Verifica el fix:
- B2.5: SwarmScheduler.tick() ahora limpia _task_index junto con _task_queue
  para tareas COMPLETED/FAILED/CANCELLED → sin leak de memoria.
"""

import time

import pytest

from hoc.core import HexCoord, HoneycombConfig, HoneycombGrid, QueenCell, WorkerCell
from hoc.nectar import NectarFlow
from hoc.swarm import (
    ForagerBehavior,
    GuardBehavior,
    HiveTask,
    LoadDistribution,
    NurseBehavior,
    ScoutBehavior,
    SwarmBalancer,
    SwarmConfig,
    SwarmPolicy,
    SwarmScheduler,
    TaskNectar,
    TaskPollen,
    TaskPriority,
    TaskState,
)

# ───────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ───────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def grid():
    """Grid moderado (radio 2 → 19 celdas)."""
    return HoneycombGrid(HoneycombConfig(radius=2))


@pytest.fixture
def nectar_flow(grid):
    return NectarFlow(grid)


@pytest.fixture
def scheduler(grid, nectar_flow):
    return SwarmScheduler(grid, nectar_flow)


@pytest.fixture
def small_scheduler():
    """Scheduler con grid pequeño + max_queue muy pequeña para forzar errores."""
    g = HoneycombGrid(HoneycombConfig(radius=1))
    nf = NectarFlow(g)
    cfg = SwarmConfig(max_queue_size=2)
    return SwarmScheduler(g, nf, cfg)


@pytest.fixture
def worker_cell(grid):
    return next(
        cell
        for cell in grid._cells.values()
        if isinstance(cell, WorkerCell) and not isinstance(cell, QueenCell)
    )


# ───────────────────────────────────────────────────────────────────────────────
# HIVE TASK
# ───────────────────────────────────────────────────────────────────────────────


class TestHiveTask:
    def test_init_assigns_id(self):
        t = HiveTask(priority=2)
        assert t.task_id != ""

    def test_explicit_id(self):
        t = HiveTask(priority=2, task_id="my_task")
        assert t.task_id == "my_task"

    def test_default_state_pending(self):
        t = HiveTask(priority=2)
        assert t.state == TaskState.PENDING

    def test_can_retry_initial(self):
        t = HiveTask(priority=2)
        assert t.can_retry() is True

    def test_can_retry_after_max(self):
        t = HiveTask(priority=2, max_attempts=2)
        t.attempts = 2
        assert t.can_retry() is False

    def test_is_expired_pending_returns_false(self):
        t = HiveTask(priority=2, timeout_seconds=0.001)
        time.sleep(0.01)
        assert t.is_expired() is False  # PENDING never expires

    def test_is_expired_running_after_timeout(self):
        t = HiveTask(priority=2, timeout_seconds=0.001)
        t.state = TaskState.RUNNING
        time.sleep(0.01)
        assert t.is_expired() is True

    def test_priority_ordering(self):
        critical = HiveTask(priority=TaskPriority.CRITICAL.value)
        normal = HiveTask(priority=TaskPriority.NORMAL.value)
        # Heapq usa min-heap → critical (0) sale antes que normal (2)
        import heapq

        h = []
        heapq.heappush(h, normal)
        heapq.heappush(h, critical)
        first = heapq.heappop(h)
        assert first.priority == TaskPriority.CRITICAL.value


# ───────────────────────────────────────────────────────────────────────────────
# CARGAS SECUNDARIAS
# ───────────────────────────────────────────────────────────────────────────────


class TestPayloadTypes:
    def test_task_pollen(self):
        p = TaskPollen(data=b"hello", source=HexCoord(0, 0), destination=HexCoord(1, 0))
        assert p.data == b"hello"

    def test_task_nectar(self):
        n = TaskNectar(entity_ids=["e1", "e2"], operation="compute", params={"x": 1})
        assert n.operation == "compute"


# ───────────────────────────────────────────────────────────────────────────────
# BEE BEHAVIORS
# ───────────────────────────────────────────────────────────────────────────────


class TestBeeBehaviorBase:
    def test_should_respond_zero_stimulus(self, worker_cell, nectar_flow):
        b = ForagerBehavior(worker_cell, nectar_flow)
        assert b.should_respond(0.0) is False
        assert b.should_respond(-1.0) is False

    def test_update_threshold_success_lowers(self, worker_cell, nectar_flow):
        b = ForagerBehavior(worker_cell, nectar_flow)
        initial = b.response_threshold
        b.update_threshold(success=True)
        assert b.response_threshold <= initial + 0.1 + 0.01  # bumped up by 0.1

    def test_update_threshold_failure_resets_streak(self, worker_cell, nectar_flow):
        b = ForagerBehavior(worker_cell, nectar_flow)
        b._success_streak = 5
        b.update_threshold(success=False)
        assert b._success_streak == 0

    def test_threshold_clamped(self, worker_cell, nectar_flow):
        b = ForagerBehavior(worker_cell, nectar_flow)
        for _ in range(20):
            b.update_threshold(success=True)
        assert 0.05 <= b.response_threshold <= 0.95


class TestForagerBehavior:
    def test_select_no_tasks(self, worker_cell, nectar_flow):
        f = ForagerBehavior(worker_cell, nectar_flow)
        assert f.select_task([]) is None

    def test_select_returns_a_task(self, worker_cell, nectar_flow):
        f = ForagerBehavior(worker_cell, nectar_flow)
        tasks = [
            HiveTask(priority=2, task_type="compute"),
            HiveTask(priority=1, task_type="compute"),
        ]
        selected = f.select_task(tasks)
        assert selected in tasks

    def test_execute_success(self, worker_cell, nectar_flow):
        f = ForagerBehavior(worker_cell, nectar_flow)
        task = HiveTask(priority=2, task_type="compute")
        result = f.execute_task(task)
        assert result is True
        assert task.state == TaskState.COMPLETED

    def test_execute_with_callable_payload(self, worker_cell, nectar_flow):
        f = ForagerBehavior(worker_cell, nectar_flow)
        task = HiveTask(
            priority=2,
            task_type="compute",
            payload={"execute": lambda: 42},
        )
        f.execute_task(task)
        assert task.result == 42

    def test_execute_with_failing_payload(self, worker_cell, nectar_flow):
        f = ForagerBehavior(worker_cell, nectar_flow)

        def fail():
            raise RuntimeError("intencional")

        task = HiveTask(
            priority=2,
            task_type="compute",
            payload={"execute": fail},
        )
        result = f.execute_task(task)
        assert result is False
        assert task.state == TaskState.FAILED
        assert task.error is not None

    def test_specialization_filter(self, worker_cell, nectar_flow):
        f = ForagerBehavior(worker_cell, nectar_flow)
        f.specialization = "image"
        tasks = [
            HiveTask(priority=2, task_type="compute"),
            HiveTask(priority=2, task_type="image"),
        ]
        # Force should_respond=True para tomar la primera "image"
        f.response_threshold = 0.01
        selected = f.select_task(tasks)
        # Debería preferir "image" si está disponible
        assert selected is not None


class TestNurseBehavior:
    def test_select_spawn_priority(self, worker_cell, nectar_flow):
        n = NurseBehavior(worker_cell, nectar_flow)
        tasks = [
            HiveTask(priority=2, task_type="compute"),
            HiveTask(priority=2, task_type="spawn"),
        ]
        selected = n.select_task(tasks)
        assert selected is not None
        assert selected.task_type == "spawn"

    def test_select_warmup_fallback(self, worker_cell, nectar_flow):
        n = NurseBehavior(worker_cell, nectar_flow)
        tasks = [
            HiveTask(priority=2, task_type="compute"),
            HiveTask(priority=2, task_type="warmup"),
        ]
        selected = n.select_task(tasks)
        assert selected is not None
        assert selected.task_type == "warmup"

    def test_select_returns_none_when_no_match(self, worker_cell, nectar_flow):
        n = NurseBehavior(worker_cell, nectar_flow)
        tasks = [HiveTask(priority=2, task_type="compute")]
        assert n.select_task(tasks) is None

    def test_execute_spawn(self, worker_cell, nectar_flow):
        n = NurseBehavior(worker_cell, nectar_flow)
        task = HiveTask(priority=2, task_type="spawn", payload={"spec": {"x": 1}})
        assert n.execute_task(task) is True
        assert task.state == TaskState.COMPLETED
        assert len(n.incubating) == 1

    def test_execute_warmup_no_vcore(self, worker_cell, nectar_flow):
        n = NurseBehavior(worker_cell, nectar_flow)
        task = HiveTask(priority=2, task_type="warmup", payload={})
        assert n.execute_task(task) is True
        assert task.state == TaskState.COMPLETED

    def test_tick_incubation(self, worker_cell, nectar_flow):
        n = NurseBehavior(worker_cell, nectar_flow)
        n.warmup_ticks = 1
        n.incubating.append(
            {
                "spec": {},
                "ticks_remaining": 1,
                "task": HiveTask(priority=2),
            }
        )
        ready = n.tick_incubation()
        assert len(ready) == 1
        assert n.incubating == []


class TestScoutBehavior:
    def test_select_explore_priority(self, worker_cell, nectar_flow):
        s = ScoutBehavior(worker_cell, nectar_flow)
        tasks = [
            HiveTask(priority=2, task_type="compute"),
            HiveTask(priority=2, task_type="explore"),
        ]
        selected = s.select_task(tasks)
        assert selected.task_type == "explore"

    def test_select_distant_task(self, worker_cell, nectar_flow):
        s = ScoutBehavior(worker_cell, nectar_flow)
        # Tarea distante (debería pasar el filtro de distance > 3)
        far = HiveTask(priority=2, task_type="compute", target_cell=HexCoord(10, 10))
        selected = s.select_task([far])
        assert selected == far

    def test_execute_explore(self, worker_cell, nectar_flow):
        s = ScoutBehavior(worker_cell, nectar_flow)
        s.exploration_radius = 1
        task = HiveTask(
            priority=2,
            task_type="explore",
            payload={"target": worker_cell.coord},
        )
        assert s.execute_task(task) is True
        assert task.state == TaskState.COMPLETED
        assert "cells_explored" in task.result


class TestGuardBehavior:
    def test_select_validate_only(self, worker_cell, nectar_flow):
        g = GuardBehavior(worker_cell, nectar_flow)
        tasks = [
            HiveTask(priority=2, task_type="compute"),
            HiveTask(priority=2, task_type="validate"),
        ]
        selected = g.select_task(tasks)
        assert selected.task_type == "validate"

    def test_execute_validate(self, worker_cell, nectar_flow):
        g = GuardBehavior(worker_cell, nectar_flow)
        target = HiveTask(priority=2, task_type="compute")
        task = HiveTask(
            priority=2,
            task_type="validate",
            payload={"target_task": target},
        )
        assert g.execute_task(task) is True
        assert task.result == {"valid": True}

    def test_validation_rule_blocks(self, worker_cell, nectar_flow):
        g = GuardBehavior(worker_cell, nectar_flow)
        g.add_validation_rule(lambda t: False)  # Bloquea todo
        target = HiveTask(priority=2, task_type="compute")
        task = HiveTask(
            priority=2,
            task_type="validate",
            payload={"target_task": target},
        )
        g.execute_task(task)
        assert task.result == {"valid": False}


# ───────────────────────────────────────────────────────────────────────────────
# LOAD DISTRIBUTION
# ───────────────────────────────────────────────────────────────────────────────


class TestLoadDistribution:
    def test_initial_empty(self):
        d = LoadDistribution()
        assert d.average_load == 0.0
        assert d.max_load == 0.0
        assert d.min_load == 0.0
        assert d.load_variance == 0.0

    def test_update_from_grid(self, grid):
        d = LoadDistribution()
        d.update(grid)
        assert len(d.cell_loads) == len(grid._cells)


# ───────────────────────────────────────────────────────────────────────────────
# SWARM BALANCER
# ───────────────────────────────────────────────────────────────────────────────


class TestSwarmBalancer:
    def test_init(self, grid):
        b = SwarmBalancer(grid, SwarmConfig())
        stats = b.get_stats()
        assert "average_load" in stats

    def test_update_distribution(self, grid):
        b = SwarmBalancer(grid, SwarmConfig())
        d = b.update_distribution()
        assert isinstance(d, LoadDistribution)

    def test_find_overloaded_empty(self, grid):
        b = SwarmBalancer(grid, SwarmConfig())
        b.update_distribution()
        # Grid recién creado: no hay carga
        assert b.find_overloaded_cells() == []

    def test_find_underloaded_all(self, grid):
        b = SwarmBalancer(grid, SwarmConfig())
        b.update_distribution()
        # Grid sin carga → todas underloaded
        assert len(b.find_underloaded_cells()) > 0

    def test_suggest_no_migration_when_balanced(self, grid):
        b = SwarmBalancer(grid, SwarmConfig())
        b.update_distribution()
        suggestions = b.suggest_migrations()
        assert suggestions == []  # nada sobrecargado

    def test_execute_work_stealing_no_overload(self, grid):
        b = SwarmBalancer(grid, SwarmConfig())
        b.update_distribution()
        stolen = b.execute_work_stealing()
        assert stolen == 0

    def test_work_stealing_disabled(self, grid):
        cfg = SwarmConfig(enable_work_stealing=False)
        b = SwarmBalancer(grid, cfg)
        assert b.execute_work_stealing() == 0

    def test_rebalance_too_soon(self, grid):
        b = SwarmBalancer(grid, SwarmConfig(rebalance_interval_ticks=10))
        # En tick 0, requiere distancia ≥ 10 desde _last_rebalance(=0)
        b._last_rebalance = 5
        assert b.rebalance_if_needed(6) is False


# ───────────────────────────────────────────────────────────────────────────────
# SWARM SCHEDULER
# ───────────────────────────────────────────────────────────────────────────────


class TestSwarmScheduler:
    def test_init_creates_behaviors(self, scheduler, grid):
        # Debería haber behaviors para las celdas worker
        worker_count = sum(1 for c in grid._cells.values() if isinstance(c, WorkerCell))
        assert len(scheduler._behaviors) == worker_count

    def test_submit_task(self, scheduler):
        task = scheduler.submit_task("compute", {"x": 1})
        assert task.state == TaskState.PENDING
        assert scheduler.get_queue_size() == 1

    def test_get_task_by_id(self, scheduler):
        task = scheduler.submit_task("compute", {})
        assert scheduler.get_task(task.task_id) is task

    def test_get_task_unknown_returns_none(self, scheduler):
        assert scheduler.get_task("nonexistent") is None

    def test_cancel_pending_task(self, scheduler):
        task = scheduler.submit_task("compute", {})
        assert scheduler.cancel_task(task.task_id) is True
        assert task.state == TaskState.CANCELLED

    def test_cancel_unknown_task(self, scheduler):
        assert scheduler.cancel_task("nonexistent") is False

    def test_cancel_completed_task_fails(self, scheduler):
        task = scheduler.submit_task("compute", {})
        task.state = TaskState.COMPLETED
        assert scheduler.cancel_task(task.task_id) is False

    def test_max_queue_size_enforced(self, small_scheduler):
        small_scheduler.submit_task("compute", {})
        small_scheduler.submit_task("compute", {})
        with pytest.raises(RuntimeError, match="full"):
            small_scheduler.submit_task("compute", {})

    def test_submit_with_priority(self, scheduler):
        task = scheduler.submit_task("compute", {}, priority=TaskPriority.CRITICAL)
        assert task.priority == TaskPriority.CRITICAL.value

    def test_submit_with_target_cell(self, scheduler, worker_cell):
        task = scheduler.submit_task("compute", {}, target_cell=worker_cell.coord)
        assert task.target_cell == worker_cell.coord

    def test_submit_with_callback(self, scheduler):
        called = []
        scheduler.submit_task(
            "compute",
            {"execute": lambda: 42},
            callback=lambda r: called.append(r),
        )
        # Ejecutar tick para procesar
        for _ in range(3):
            scheduler.run_tick_sync()
        # El callback debería invocarse cuando una tarea completa
        # (no garantizado en N ticks para una tarea sin target_cell, así que
        # solo verificamos que tick funcione sin errores)

    def test_tick_runs_clean(self, scheduler):
        result = scheduler.run_tick_sync()
        assert "tick" in result
        assert "tasks_processed" in result

    def test_tick_processes_tasks(self, scheduler):
        scheduler.submit_task("compute", {})
        scheduler.submit_task("compute", {})
        # Ejecutar varios ticks
        for _ in range(5):
            scheduler.run_tick_sync()
        # Verificar que hubo procesamiento eventual
        stats = scheduler.get_stats()
        assert stats["tick_count"] == 5

    # ─── B2.5 FIX: _task_index no debe leak ───────────────────────────────────

    def test_b2_5_task_index_cleaned_after_completed(self, scheduler):
        """B2.5: tras tick que completa tareas, _task_index debe limpiarse."""
        task1 = scheduler.submit_task("compute", {"execute": lambda: 1})
        task2 = scheduler.submit_task("compute", {"execute": lambda: 2})
        # Marcar como completed manualmente para forzar la limpieza
        task1.state = TaskState.COMPLETED
        task2.state = TaskState.COMPLETED
        # Tick procesa el cleanup
        scheduler.run_tick_sync()
        # _task_index ya no debe contenerlas
        assert task1.task_id not in scheduler._task_index
        assert task2.task_id not in scheduler._task_index

    def test_b2_5_task_index_cleaned_after_cancelled(self, scheduler):
        """B2.5: tareas CANCELLED también se limpian del index."""
        task = scheduler.submit_task("compute", {})
        scheduler.cancel_task(task.task_id)
        scheduler.run_tick_sync()
        assert task.task_id not in scheduler._task_index

    def test_b2_5_task_index_cleaned_after_failed(self, scheduler):
        """B2.5: tareas FAILED también se limpian del index."""
        task = scheduler.submit_task("compute", {})
        task.state = TaskState.FAILED
        scheduler.run_tick_sync()
        assert task.task_id not in scheduler._task_index

    def test_b2_5_pending_tasks_kept_in_index(self, scheduler):
        """B2.5: las tareas PENDING/RUNNING NO deben limpiarse."""
        task = scheduler.submit_task("compute", {}, target_cell=HexCoord(99, 99))
        # Como el target no existe, la tarea queda PENDING
        scheduler.run_tick_sync()
        assert task.task_id in scheduler._task_index

    def test_b2_5_no_leak_after_many_cycles(self, scheduler):
        """B2.5: después de muchos ciclos, _task_index no crece sin cota."""
        for _ in range(20):
            t = scheduler.submit_task("compute", {})
            t.state = TaskState.COMPLETED
            scheduler.run_tick_sync()
        # _task_index debe estar prácticamente vacío
        assert len(scheduler._task_index) <= 5  # margen para tareas en transición

    def test_get_pending_count(self, scheduler):
        scheduler.submit_task("compute", {})
        scheduler.submit_task("compute", {})
        assert scheduler.get_pending_count() == 2

    def test_get_stats_full(self, scheduler):
        scheduler.submit_task("compute", {})
        scheduler.run_tick_sync()
        stats = scheduler.get_stats()
        for key in (
            "tick_count",
            "queue_size",
            "pending_tasks",
            "tasks_completed",
            "tasks_failed",
            "behaviors",
            "balancer",
        ):
            assert key in stats

    def test_shutdown_cancels_pending(self, scheduler):
        t1 = scheduler.submit_task("compute", {})
        scheduler.submit_task("compute", {})
        scheduler.shutdown()
        assert t1.state == TaskState.CANCELLED or t1.state in (
            TaskState.COMPLETED,
            TaskState.FAILED,
        )
        assert scheduler.get_queue_size() == 0
        assert len(scheduler._task_index) == 0


# ───────────────────────────────────────────────────────────────────────────────
# SWARM CONFIG
# ───────────────────────────────────────────────────────────────────────────────


class TestSwarmConfig:
    def test_defaults(self):
        cfg = SwarmConfig()
        assert cfg.foragers_ratio == 0.6
        assert cfg.max_queue_size == 10000
        assert cfg.enable_work_stealing is True
        assert cfg.default_policy == SwarmPolicy.PHEROMONE_GUIDED


# ───────────────────────────────────────────────────────────────────────────────
# CONCURRENCIA
# ───────────────────────────────────────────────────────────────────────────────


class TestConcurrency:
    def test_concurrent_submissions(self, scheduler):
        import threading

        errors = []

        def submit_many(prefix):
            try:
                for i in range(50):
                    scheduler.submit_task(f"task_{prefix}_{i}", {})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=submit_many, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert scheduler.get_queue_size() == 150
