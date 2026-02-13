"""
Tests pesados: cargas de trabajo intensivas y estrés del SwarmScheduler.

Ejecutan tareas CPU-intensivas (render, matrices, simulación, hash, Monte Carlo)
y comprueban que el scheduler completa correctamente.
"""
from __future__ import annotations

import pytest
from hoc import HoneycombGrid, HoneycombConfig, NectarFlow, SwarmScheduler
from hoc.swarm import TaskPriority, TaskState


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def grid_radius_2():
    """Grid pequeño para tests pesados (menos celdas = menos tiempo)."""
    return HoneycombConfig(radius=2)


@pytest.fixture
def scheduler(grid_radius_2):
    """Scheduler listo para encolar tareas."""
    grid = HoneycombGrid(grid_radius_2)
    nectar = NectarFlow(grid)
    return SwarmScheduler(grid, nectar)


# ─── Tests por tipo de carga ─────────────────────────────────────────────────

class TestHeavyByType:
    """Tests pesados por tipo de carga (una tarea por tipo)."""

    def _run_until_done(self, scheduler, tasks, max_ticks=800):
        for _ in range(max_ticks):
            if scheduler.get_pending_count() == 0:
                break
            scheduler.tick()
        return [t for t in tasks if t.state == TaskState.COMPLETED]

    def test_heavy_render_only(self, scheduler):
        """Varias tareas de render 3D."""
        from benchmarks.workload_heavy import workload_render_3d
        tasks = []
        for i in range(3):
            t = scheduler.submit_task(
                task_type="compute",
                payload={"execute": lambda w=workload_render_3d: w(64, 64, 4, 2), "name": f"r{i}"},
                priority=TaskPriority.NORMAL,
                timeout=60.0,
            )
            tasks.append(t)
        completed = self._run_until_done(scheduler, tasks)
        assert len(completed) == 3

    def test_heavy_matrix_only(self, scheduler):
        """Varias tareas de multiplicación de matrices."""
        from benchmarks.workload_heavy import workload_matrix_mult
        tasks = []
        for i in range(4):
            t = scheduler.submit_task(
                task_type="compute",
                payload={"execute": lambda: workload_matrix_mult(256, 2), "name": f"m{i}"},
                priority=TaskPriority.NORMAL,
                timeout=60.0,
            )
            tasks.append(t)
        completed = self._run_until_done(scheduler, tasks)
        assert len(completed) == 4

    def test_heavy_simulation_only(self, scheduler):
        """Varias tareas de simulación."""
        from benchmarks.workload_heavy import workload_simulation_steps
        tasks = []
        for i in range(4):
            t = scheduler.submit_task(
                task_type="compute",
                payload={"execute": lambda: workload_simulation_steps(800, 64), "name": f"s{i}"},
                priority=TaskPriority.NORMAL,
                timeout=60.0,
            )
            tasks.append(t)
        completed = self._run_until_done(scheduler, tasks)
        assert len(completed) == 4

    def test_heavy_hash_only(self, scheduler):
        """Varias tareas de trabajo tipo hash."""
        from benchmarks.workload_heavy import workload_hash_like
        tasks = []
        for i in range(4):
            t = scheduler.submit_task(
                task_type="compute",
                payload={"execute": lambda: workload_hash_like(16 * 1024, 1500), "name": f"h{i}"},
                priority=TaskPriority.NORMAL,
                timeout=60.0,
            )
            tasks.append(t)
        completed = self._run_until_done(scheduler, tasks)
        assert len(completed) == 4

    def test_heavy_monte_carlo_only(self, scheduler):
        """Varias tareas Monte Carlo."""
        from benchmarks.workload_heavy import workload_monte_carlo
        tasks = []
        for i in range(3):
            t = scheduler.submit_task(
                task_type="compute",
                payload={"execute": lambda: workload_monte_carlo(100_000, 4), "name": f"c{i}"},
                priority=TaskPriority.NORMAL,
                timeout=60.0,
            )
            tasks.append(t)
        completed = self._run_until_done(scheduler, tasks)
        assert len(completed) == 3


# ─── Test mixto ─────────────────────────────────────────────────────────────

class TestHeavyMixed:
    """Tests con mezcla de todos los tipos de carga."""

    def test_heavy_mixed_five_types(self, scheduler):
        """Una tarea de cada tipo: render, matrix, simulation, hash, monte_carlo."""
        from benchmarks.workload_heavy import (
            workload_render_3d,
            workload_matrix_mult,
            workload_simulation_steps,
            workload_hash_like,
            workload_monte_carlo,
        )
        payloads = [
            {"execute": lambda: workload_render_3d(48, 48, 3, 2)},
            {"execute": lambda: workload_matrix_mult(128, 2)},
            {"execute": lambda: workload_simulation_steps(400, 32)},
            {"execute": lambda: workload_hash_like(8 * 1024, 800)},
            {"execute": lambda: workload_monte_carlo(50_000, 3)},
        ]
        tasks = []
        for i, pl in enumerate(payloads):
            t = scheduler.submit_task(
                task_type="compute",
                payload={**pl, "index": i},
                priority=TaskPriority.NORMAL,
                timeout=60.0,
            )
            tasks.append(t)
        for _ in range(600):
            if scheduler.get_pending_count() == 0:
                break
            scheduler.tick()
        completed = sum(1 for t in tasks if t.state == TaskState.COMPLETED)
        assert completed == 5, f"Completadas {completed}/5"

    def test_heavy_mixed_uses_benchmark_runner(self):
        """Usa el runner del benchmark mixto con pocas tareas."""
        from benchmarks.bench_heavy_mixed import run_heavy_mixed_benchmark
        result = run_heavy_mixed_benchmark(
            tasks_per_type={
                "render_3d": 1,
                "matrix_mult": 1,
                "simulation": 1,
                "hash_work": 1,
                "monte_carlo": 1,
            },
            grid_radius=2,
            max_ticks=800,
        )
        assert result["total_completed"] >= 4
        assert result["total_failed"] == 0


# ─── Tests de estrés ─────────────────────────────────────────────────────────

class TestHeavyStress:
    """Tests de estrés: muchas tareas o cargas más grandes."""

    def test_stress_many_small_tasks(self):
        """Muchas tareas ligeras (hash corto) para saturar el scheduler."""
        from benchmarks.workload_heavy import workload_hash_like
        config = HoneycombConfig(radius=2)
        grid = HoneycombGrid(config)
        nectar = NectarFlow(grid)
        scheduler = SwarmScheduler(grid, nectar)
        n = 24
        tasks = []
        for i in range(n):
            t = scheduler.submit_task(
                task_type="compute",
                payload={"execute": lambda: workload_hash_like(1024, 200), "i": i},
                priority=TaskPriority.NORMAL,
                timeout=30.0,
            )
            tasks.append(t)
        for _ in range(1500):
            if scheduler.get_pending_count() == 0:
                break
            scheduler.tick()
        completed = sum(1 for t in tasks if t.state == TaskState.COMPLETED)
        assert completed == n, f"Completadas {completed}/{n}"

    def test_stress_mixed_heavy_medium(self):
        """Varias tareas mixtas de carga media-alta."""
        from benchmarks.bench_heavy_mixed import run_heavy_mixed_benchmark
        from benchmarks.workload_heavy import WORKLOADS
        # 1 de cada tipo, grid pequeño
        result = run_heavy_mixed_benchmark(
            tasks_per_type={k: 1 for k in WORKLOADS},
            grid_radius=2,
            max_ticks=2000,
        )
        assert result["total_completed"] == result["total_tasks"]
        assert result["total_failed"] == 0


# ─── Ejecución directa (sin pytest) ─────────────────────────────────────────

def _run_heavy_tests_direct() -> bool:
    """Ejecuta los tests pesados sin pytest. Útil si pytest falla por plugins."""
    from benchmarks.bench_heavy_mixed import run_heavy_mixed_benchmark

    ok = True
    # Test mixto rápido: solo tipos ligeros (1 de cada uno, sin matrix_svd/render grande)
    print("Test: mixed light (render_3d, matrix_mult, simulation, hash_work, monte_carlo) 1 each...")
    result = run_heavy_mixed_benchmark(
        tasks_per_type={
            "render_3d": 1, "matrix_mult": 1, "simulation": 1,
            "hash_work": 1, "monte_carlo": 1,
        },
        grid_radius=2,
        max_ticks=1000,
    )
    if result["total_completed"] != result["total_tasks"] or result["total_failed"] != 0:
        print(f"  FAIL: completed={result['total_completed']} failed={result['total_failed']}")
        ok = False
    else:
        print(f"  OK ({result['total_completed']} tareas en {result['elapsed_seconds']:.1f} s)")

    # Test estrés: 8 tareas ligeras (hash_work por defecto en workload_heavy es pesado; usamos menos)
    print("Test: stress 8 small hash tasks...")
    result2 = run_heavy_mixed_benchmark(
        tasks_per_type={"hash_work": 8},
        grid_radius=2,
        max_ticks=500,
    )
    if result2["total_completed"] != 8:
        print(f"  FAIL: completed={result2['total_completed']}/8")
        ok = False
    else:
        print(f"  OK (8 tareas en {result2['elapsed_seconds']:.1f} s)")
    return ok


if __name__ == "__main__":
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sys.exit(0 if _run_heavy_tests_direct() else 1)
