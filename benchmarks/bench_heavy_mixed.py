"""
Benchmark unificado de tareas pesadas variadas sobre SwarmScheduler.

Ejecuta tareas de distintos tipos (render, matrix, simulation, hash, monte_carlo)
y recoge tiempos, throughput y estadísticas por tipo.

Uso:
  python -m benchmarks.bench_heavy_mixed
  pytest benchmarks/bench_heavy_mixed.py -v -s
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import defaultdict

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


def run_heavy_mixed_benchmark(
    tasks_per_type: dict[str, int] | None = None,
    grid_radius: int = 3,
    max_ticks: int = 2000,
    timeout_per_task: float = 120.0,
) -> dict:
    """
    Ejecuta un benchmark con tareas pesadas de varios tipos.

    tasks_per_type: ej. {"render_3d": 2, "matrix_mult": 3, "simulation": 2, ...}
                    Si None, usa 2 de cada tipo por defecto.
    """
    from hoc import HoneycombGrid, HoneycombConfig, NectarFlow, SwarmScheduler
    from hoc.swarm import TaskPriority, TaskState
    from benchmarks.workload_heavy import WORKLOADS

    if tasks_per_type is None:
        tasks_per_type = {k: 2 for k in WORKLOADS}

    config = HoneycombConfig(radius=grid_radius)
    grid = HoneycombGrid(config)
    nectar = NectarFlow(grid)
    scheduler = SwarmScheduler(grid, nectar)

    # Encolar una tarea por tipo por cada count
    task_ids_by_type: dict[str, list[str]] = defaultdict(list)
    all_tasks = []
    for wtype, count in tasks_per_type.items():
        if wtype not in WORKLOADS:
            continue
        fn = WORKLOADS[wtype]
        for i in range(count):
            task = scheduler.submit_task(
                task_type="compute",
                payload={"execute": fn, "workload_type": wtype, "index": i},
                priority=TaskPriority.NORMAL,
                timeout=timeout_per_task,
            )
            task_ids_by_type[wtype].append(task.task_id)
            all_tasks.append((wtype, task))

    # Ejecutar ticks
    start = time.perf_counter()
    tick_results = []
    while True:
        pending = scheduler.get_pending_count()
        if pending == 0:
            break
        if len(tick_results) >= max_ticks:
            break
        res = scheduler.tick()
        tick_results.append(res)
    elapsed = time.perf_counter() - start

    # Resultados por tipo
    completed_by_type: dict[str, int] = defaultdict(int)
    failed_by_type: dict[str, int] = defaultdict(int)
    for wtype, task in all_tasks:
        if task.state == TaskState.COMPLETED:
            completed_by_type[wtype] += 1
        elif task.state == TaskState.FAILED:
            failed_by_type[wtype] += 1

    total_completed = sum(completed_by_type.values())
    total_failed = sum(failed_by_type.values())

    return {
        "elapsed_seconds": elapsed,
        "ticks_run": len(tick_results),
        "tasks_per_type": dict(tasks_per_type),
        "completed_by_type": dict(completed_by_type),
        "failed_by_type": dict(failed_by_type),
        "total_completed": total_completed,
        "total_failed": total_failed,
        "total_tasks": len(all_tasks),
        "tasks_per_second": total_completed / elapsed if elapsed > 0 else 0,
        "grid_cells": grid.cell_count,
        "scheduler_stats": scheduler.get_stats(),
    }


def test_heavy_mixed_benchmark():
    """Test: benchmark mixto con 1 tarea de cada tipo (rápido)."""
    result = run_heavy_mixed_benchmark(
        tasks_per_type={k: 1 for k in ["render_3d", "matrix_mult", "simulation", "hash_work", "monte_carlo"]},
        grid_radius=2,
        max_ticks=500,
    )
    assert result["total_completed"] >= 4
    assert result["elapsed_seconds"] >= 0


def test_heavy_mixed_all_types():
    """Test: 2 tareas de cada tipo (más pesado)."""
    from benchmarks.workload_heavy import WORKLOADS
    result = run_heavy_mixed_benchmark(
        tasks_per_type={k: 2 for k in WORKLOADS},
        grid_radius=2,
        max_ticks=1500,
    )
    assert result["total_completed"] == result["total_tasks"], (
        f"completed={result['total_completed']} total={result['total_tasks']}"
    )


if __name__ == "__main__":
    from benchmarks.workload_heavy import WORKLOADS

    print("HOC – Benchmark de tareas pesadas mixtas\n")
    tasks_per_type = {k: 2 for k in WORKLOADS}
    result = run_heavy_mixed_benchmark(
        tasks_per_type=tasks_per_type,
        grid_radius=3,
        max_ticks=2000,
    )

    print(f"  Tiempo total:     {result['elapsed_seconds']:.3f} s")
    print(f"  Tareas totales:   {result['total_tasks']} ({result['total_completed']} completadas, {result['total_failed']} fallidas)")
    print(f"  Ticks:            {result['ticks_run']}")
    print(f"  Throughput:       {result['tasks_per_second']:.2f} tareas/s")
    print(f"  Celdas grid:      {result['grid_cells']}")
    print("\n  Por tipo (completadas):")
    for wtype in WORKLOADS:
        c = result["completed_by_type"].get(wtype, 0)
        f = result["failed_by_type"].get(wtype, 0)
        print(f"    {wtype}: {c} ok, {f} failed")
    print("\n  Scheduler:", result["scheduler_stats"])
    sys.exit(0 if result["total_failed"] == 0 and result["total_completed"] == result["total_tasks"] else 1)
