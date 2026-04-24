"""
HOC Core · Grid — facade + :class:`HoneycombGrid`.

Re-exporta geometría y config de los submódulos hermanos, y aloja la
clase principal :class:`HoneycombGrid` más los helpers ``create_grid``
y ``benchmark_grid``. Extraído de ``core.py`` en Fase 3.3.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import suppress
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .cells import (
    CellRole,
    CellState,
    DroneCell,
    GuardCell,
    HoneycombCell,
    NurseryCell,
    QueenCell,
    ScoutCell,
    StorageCell,
    WorkerCell,
)
from .events import Event, EventBus, EventType, get_event_bus
from .grid_config import GridTopology, HoneycombConfig
from .grid_geometry import (
    HexCoord,
    HexDirection,
    HexPathfinder,
    HexRegion,
    HexRing,
)
from .health import CircuitState, HealthMonitor
from .locking import RWLock

logger = logging.getLogger(__name__)

__all__ = [
    # Geometría (re-exportada)
    "HexDirection",
    "HexCoord",
    "HexRegion",
    "HexPathfinder",
    "HexRing",
    # Config (re-exportada)
    "HoneycombConfig",
    "GridTopology",
    # Grid principal
    "HoneycombGrid",
    # Utilidades
    "create_grid",
    "benchmark_grid",
]


# ─── Cell factory ──────────────────────────────────────────────────────────────
# Mapping de roles a clases de celdas
_CELL_TYPE_MAP: dict[CellRole, type] = {
    CellRole.QUEEN: QueenCell,
    CellRole.WORKER: WorkerCell,
    CellRole.DRONE: DroneCell,
    CellRole.NURSERY: NurseryCell,
    CellRole.STORAGE: StorageCell,
    CellRole.GUARD: GuardCell,
    CellRole.SCOUT: ScoutCell,
}

_CELL_NAME_MAP: dict[str, type] = {cls.__name__: cls for cls in _CELL_TYPE_MAP.values()}


def _create_cell_by_role(
    role: CellRole, coord: HexCoord, config: HoneycombConfig | None = None
) -> HoneycombCell:
    """Factory para crear celdas por rol."""
    cell_cls = _CELL_TYPE_MAP.get(role, WorkerCell)
    if role == CellRole.QUEEN:
        return cell_cls(coord, config)
    return cell_cls(coord, config)


# ─── Grid hexagonal principal ──────────────────────────────────────────────────


class HoneycombGrid:
    """
    Grid Hexagonal Principal v3.0.

    Mejoras v3.0:
    - Índices por rol/estado para O(1) lookups
    - HealthMonitor integrado
    - from_dict() completo con restauración de estado
    - Auto-recovery de celdas fallidas
    - Graceful shutdown
    """

    __slots__ = (
        "_cells",
        "_event_bus",
        "_executor",
        "_health_monitor",
        "_last_tick_time",
        "_metrics_collector",
        "_queen",
        "_role_index",
        "_running",
        "_rw_lock",
        "_tick_count",
        "_topology",
        "config",
    )

    def __init__(self, config: HoneycombConfig | None = None, event_bus: EventBus | None = None):
        from ..metrics.collection import MetricsCollector  # lazy: ciclo core↔metrics

        self.config = config or HoneycombConfig()
        self._cells: dict[HexCoord, HoneycombCell] = {}
        self._queen: QueenCell | None = None
        self._rw_lock = RWLock()
        self._topology = GridTopology[self.config.topology.upper()]
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_parallel_rings, thread_name_prefix="hoc_"
        )
        self._metrics_collector = MetricsCollector(
            max_history=self.config.metrics_history_size,
            sample_rate=self.config.metrics_sample_rate,
        )
        self._event_bus = event_bus or get_event_bus()
        self._tick_count = 0
        self._last_tick_time = time.time()
        self._running = False

        # v3.0: Índice por rol para lookups O(1)
        self._role_index: dict[CellRole, set[HexCoord]] = defaultdict(set)

        # Inicializar grid
        self._initialize_grid()

        # v3.0: Health monitor
        self._health_monitor = HealthMonitor(
            self,
            self._event_bus,
            check_interval=self.config.health_check_interval_s,
            alert_threshold=self.config.health_alert_load_threshold,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # INICIALIZACIÓN
    # ─────────────────────────────────────────────────────────────────────────

    def _initialize_grid(self) -> None:
        """Inicializa el grid con celdas básicas."""
        origin = HexCoord.origin()

        # Crear reina en el centro
        self._queen = QueenCell(origin, self.config)
        self._add_cell_to_index(origin, self._queen)

        drone_count = 0
        nursery_count = 0

        for ring_r in range(1, self.config.radius + 1):
            for coord in origin.ring(ring_r):
                if (
                    ring_r == self.config.radius
                    and drone_count < self.config.drones_per_ring * ring_r
                ):
                    cell = DroneCell(coord, self.config)
                    drone_count += 1
                elif nursery_count < self.config.nurseries_per_grid and ring_r % 3 == 0:
                    cell = NurseryCell(coord, self.config)
                    nursery_count += 1
                else:
                    cell = WorkerCell(coord, self.config)

                self._add_cell_to_index(coord, cell)

                if isinstance(cell, WorkerCell):
                    self._queen.register_worker(cell)

        self._connect_all_neighbors()

        logger.info(f"HoneycombGrid initialized with {len(self._cells)} cells")

    def _add_cell_to_index(self, coord: HexCoord, cell: HoneycombCell) -> None:
        """Añade celda al storage y al índice por rol."""
        self._cells[coord] = cell
        self._role_index[cell.role].add(coord)

    def _remove_cell_from_index(self, coord: HexCoord) -> HoneycombCell | None:
        """Remueve celda del storage y del índice por rol."""
        cell = self._cells.pop(coord, None)
        if cell:
            self._role_index[cell.role].discard(coord)
        return cell

    def _connect_all_neighbors(self) -> None:
        for coord, cell in self._cells.items():
            for direction in HexDirection:
                neighbor_coord = self._resolve_neighbor_coord(coord, direction)
                neighbor = self._cells.get(neighbor_coord)
                cell.set_neighbor(direction, neighbor)

    def _resolve_neighbor_coord(self, coord: HexCoord, direction: HexDirection) -> HexCoord:
        """Resuelve coordenada del vecino considerando topología."""
        neighbor = coord.neighbor(direction)

        if self._topology == GridTopology.FLAT:
            return neighbor

        elif self._topology == GridTopology.TORUS:
            q = neighbor.q
            r = neighbor.r
            max_coord = self.config.radius

            if abs(q) > max_coord:
                q = -max_coord if q > 0 else max_coord

            if abs(r) > max_coord:
                r = -max_coord if r > 0 else max_coord

            s = -q - r
            if abs(s) > max_coord:
                return neighbor

            return HexCoord(q, r)

        return neighbor

    # ─────────────────────────────────────────────────────────────────────────
    # PROPIEDADES
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def queen(self) -> QueenCell | None:
        return self._queen

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def health_monitor(self) -> HealthMonitor:
        return self._health_monitor

    # ─────────────────────────────────────────────────────────────────────────
    # ACCESO A CELDAS
    # ─────────────────────────────────────────────────────────────────────────

    def get_cell(self, coord: HexCoord) -> HoneycombCell | None:
        with self._rw_lock.read_lock():
            return self._cells.get(coord)

    def get_or_create_cell(self, coord: HexCoord, cell_type: type = WorkerCell) -> HoneycombCell:
        with self._rw_lock.write_lock():
            if coord not in self._cells:
                cell = cell_type(coord, self.config)
                self._add_cell_to_index(coord, cell)
                self._connect_cell_neighbors(cell)

                if isinstance(cell, WorkerCell) and self._queen:
                    self._queen.register_worker(cell)

                self._event_bus.publish(
                    Event(
                        type=EventType.GRID_CELL_ADDED, source=self, data={"coord": coord.to_dict()}
                    )
                )

            return self._cells[coord]

    def remove_cell(self, coord: HexCoord) -> bool:
        with self._rw_lock.write_lock():
            if coord not in self._cells:
                return False

            cell = self._cells[coord]

            if cell.role == CellRole.QUEEN:
                return False

            for direction in HexDirection:
                neighbor = cell.get_neighbor(direction)
                if neighbor:
                    neighbor.set_neighbor(direction.opposite(), None)

            if isinstance(cell, WorkerCell) and self._queen:
                self._queen.unregister_worker(coord)

            self._remove_cell_from_index(coord)

            self._event_bus.publish(
                Event(
                    type=EventType.GRID_CELL_REMOVED, source=self, data={"coord": coord.to_dict()}
                )
            )

            return True

    def _connect_cell_neighbors(self, cell: HoneycombCell) -> None:
        for direction in HexDirection:
            neighbor_coord = self._resolve_neighbor_coord(cell.coord, direction)
            neighbor = self._cells.get(neighbor_coord)
            if neighbor:
                cell.set_neighbor(direction, neighbor)
                neighbor.set_neighbor(direction.opposite(), cell)

    # ─────────────────────────────────────────────────────────────────────────
    # CONSULTAS (v3.0: O(1) para consultas por rol)
    # ─────────────────────────────────────────────────────────────────────────

    def get_ring(self, radius: int) -> list[HoneycombCell]:
        origin = HexCoord.origin()
        with self._rw_lock.read_lock():
            return [self._cells[coord] for coord in origin.ring(radius) if coord in self._cells]

    def get_area(self, center: HexCoord, radius: int) -> list[HoneycombCell]:
        with self._rw_lock.read_lock():
            return [self._cells[coord] for coord in center.spiral(radius) if coord in self._cells]

    def get_cells_by_role(self, role: CellRole) -> list[HoneycombCell]:
        """v3.0: O(k) donde k = celdas del rol, no O(n) total."""
        with self._rw_lock.read_lock():
            return [
                self._cells[coord]
                for coord in self._role_index.get(role, set())
                if coord in self._cells
            ]

    def get_cells_by_state(self, state: CellState) -> list[HoneycombCell]:
        with self._rw_lock.read_lock():
            return [c for c in self._cells.values() if c.state == state]

    def find_available_cells(
        self, count: int = 1, near: HexCoord | None = None
    ) -> list[HoneycombCell]:
        with self._rw_lock.read_lock():
            available = [
                cell
                for cell in self._cells.values()
                if cell.is_available and isinstance(cell, WorkerCell)
            ]

            if near:
                available.sort(key=lambda c: c.coord.distance_to(near))
            else:
                available.sort(key=lambda c: c.load)

            return available[:count]

    def find_path(
        self, start: HexCoord, goal: HexCoord, cost_fn: Callable[[HexCoord], float] | None = None
    ) -> list[HexCoord] | None:
        """v3.0: Soporta costos variables."""
        pathfinder = HexPathfinder(
            walkable_check=lambda c: c in self._cells and self._cells[c].is_available,
            cost_fn=cost_fn,
        )
        return pathfinder.find_path(start, goal)

    # ─────────────────────────────────────────────────────────────────────────
    # ASIGNACIÓN DE TRABAJO
    # ─────────────────────────────────────────────────────────────────────────

    def assign_vcore(
        self, vcore: Any, preferred_coord: HexCoord | None = None
    ) -> HoneycombCell | None:
        cells = self.find_available_cells(3, near=preferred_coord)

        for cell in cells:
            if cell.add_vcore(vcore):
                return cell

        return None

    def assign_vcores_batch(self, vcores: list[Any]) -> dict[HexCoord, list[Any]]:
        assignments: dict[HexCoord, list[Any]] = defaultdict(list)

        cells = self.find_available_cells(len(vcores) * 2)
        cell_idx = 0

        for vcore in vcores:
            assigned = False
            attempts = 0

            while not assigned and attempts < len(cells):
                cell = cells[cell_idx % len(cells)]
                if cell.add_vcore(vcore):
                    assignments[cell.coord].append(vcore)
                    assigned = True
                cell_idx += 1
                attempts += 1

        return dict(assignments)

    # ─────────────────────────────────────────────────────────────────────────
    # TICK Y PROCESAMIENTO
    # ─────────────────────────────────────────────────────────────────────────

    def tick(self) -> dict[str, Any]:
        """Ejecuta un tick global del grid."""
        tick_start = time.time()

        self._event_bus.publish(
            Event(type=EventType.GRID_TICK_START, source=self, data={"tick": self._tick_count})
        )

        results = {
            "tick": self._tick_count,
            "cells_processed": 0,
            "total_vcores": 0,
            "errors": 0,
            "work_steals": 0,
            "auto_recovered": 0,
        }

        # Procesar celdas
        if self.config.parallel_ring_processing:
            ring_results = self._parallel_tick()
        else:
            ring_results = self._sequential_tick()

        results["cells_processed"] = ring_results["processed"]
        results["errors"] = ring_results["errors"]

        # Work-stealing
        results["work_steals"] = self._perform_work_stealing()

        # v3.0: Auto-recovery de celdas con circuit breaker half-open
        results["auto_recovered"] = self._attempt_auto_recovery()

        # Feromonas
        self._update_pheromones()

        results["total_vcores"] = sum(c.vcore_count for c in self._cells.values())

        tick_duration = time.time() - tick_start
        tps = 1.0 / tick_duration if tick_duration > 0 else 0

        with self._rw_lock.read_lock():
            worker_loads = [c.load for c in self._cells.values() if isinstance(c, WorkerCell)]
        from ..metrics.collection import GridMetrics  # lazy: ciclo core↔metrics

        metrics = GridMetrics(
            timestamp=time.time(),
            total_cells=len(self._cells),
            active_cells=len(self.get_cells_by_state(CellState.ACTIVE)),
            idle_cells=len(self.get_cells_by_state(CellState.IDLE)),
            failed_cells=len(self.get_cells_by_state(CellState.FAILED)),
            total_vcores=results["total_vcores"],
            average_load=float(np.mean(worker_loads)) if worker_loads else 0,
            max_load=float(max(worker_loads)) if worker_loads else 0,
            min_load=float(min(worker_loads)) if worker_loads else 0,
            load_stddev=float(np.std(worker_loads)) if worker_loads else 0,
            total_pheromones=sum(c.pheromone_level for c in self._cells.values()),
            ticks_per_second=tps,
            errors_per_second=results["errors"] / tick_duration if tick_duration > 0 else 0,
            work_steals=results["work_steals"],
        )

        self._metrics_collector.record(metrics)

        self._tick_count += 1
        self._last_tick_time = time.time()

        # v3.0: Health check periódico
        if self._health_monitor.should_check():
            self._health_monitor.check_health()

        self._event_bus.publish(Event(type=EventType.GRID_TICK_END, source=self, data=results))

        return results

    def _parallel_tick(self) -> dict[str, int]:
        processed = 0
        errors = 0

        futures: list[Future] = []

        for ring_r in range(self.config.radius + 1):
            ring_cells = self.get_ring(ring_r)
            if ring_cells:
                future = self._executor.submit(self._process_cells_batch, ring_cells)
                futures.append(future)

        for future in as_completed(futures):
            try:
                result = future.result(timeout=30)
                processed += result["processed"]
                errors += result["errors"]
            except Exception as e:
                logger.error(f"Ring processing error: {e}")
                errors += 1

        return {"processed": processed, "errors": errors}

    def _sequential_tick(self) -> dict[str, int]:
        processed = 0
        errors = 0

        with self._rw_lock.read_lock():
            cells = list(self._cells.values())

        for cell in cells:
            try:
                cell.execute_tick()
                processed += 1
            except Exception as e:
                errors += 1
                logger.error(f"Cell {cell.coord} tick error: {e}")

        return {"processed": processed, "errors": errors}

    def _process_cells_batch(self, cells: list[HoneycombCell]) -> dict[str, int]:
        processed = 0
        errors = 0

        for cell in cells:
            try:
                cell.execute_tick()
                processed += 1
            except Exception:
                errors += 1

        return {"processed": processed, "errors": errors}

    def _perform_work_stealing(self) -> int:
        total_stolen = 0

        with self._rw_lock.read_lock():
            workers = [c for c in self._cells.values() if isinstance(c, WorkerCell)]

        for worker in workers:
            if worker.can_steal_work():
                stolen = worker.attempt_work_stealing()
                total_stolen += stolen

        if total_stolen > 0:
            self._metrics_collector.increment("work_steals", total_stolen)

        return total_stolen

    def _attempt_auto_recovery(self) -> int:
        """v3.0: Intenta recuperar celdas cuyo circuit breaker está half-open."""
        recovered = 0

        with self._rw_lock.read_lock():
            failed = [
                c
                for c in self._cells.values()
                if c.state == CellState.FAILED and c.circuit_breaker.state == CircuitState.HALF_OPEN
            ]

        for cell in failed:
            if cell.recover():
                recovered += 1
                logger.info(f"Auto-recovered cell {cell.coord}")

        return recovered

    def _update_pheromones(self) -> None:
        with self._rw_lock.read_lock():
            cells = list(self._cells.values())

        for cell in cells:
            cell.decay_pheromones()

        if self.config.pheromone_diffusion_rate > 0:
            for cell in cells:
                cell.diffuse_pheromones()

    # ─────────────────────────────────────────────────────────────────────────
    # ESTADÍSTICAS Y MÉTRICAS
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        with self._rw_lock.read_lock():
            workers = [c for c in self._cells.values() if isinstance(c, WorkerCell)]
            worker_loads = [w.load for w in workers]

            return {
                "total_cells": len(self._cells),
                "worker_cells": len(workers),
                "drone_cells": len(self._role_index.get(CellRole.DRONE, set())),
                "nursery_cells": len(self._role_index.get(CellRole.NURSERY, set())),
                "guard_cells": len(self._role_index.get(CellRole.GUARD, set())),
                "scout_cells": len(self._role_index.get(CellRole.SCOUT, set())),
                "total_vcores": sum(c.vcore_count for c in self._cells.values()),
                "average_load": float(np.mean(worker_loads)) if worker_loads else 0,
                "load_stddev": float(np.std(worker_loads)) if worker_loads else 0,
                "max_load": float(max(worker_loads)) if worker_loads else 0,
                "min_load": float(min(worker_loads)) if worker_loads else 0,
                "total_pheromones": sum(c.pheromone_level for c in self._cells.values()),
                "failed_cells": sum(1 for c in self._cells.values() if c.state == CellState.FAILED),
                "tick_count": self._tick_count,
                "queen_coord": self._queen.coord.to_dict() if self._queen else None,
                "topology": self._topology.name,
            }

    def get_metrics_history(self, limit: int = 100) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self._metrics_collector.get_history(limit)]

    def get_cell_metrics(self) -> list[dict[str, Any]]:
        with self._rw_lock.read_lock():
            return [c.get_metrics().to_dict() for c in self._cells.values()]

    # ─────────────────────────────────────────────────────────────────────────
    # VISUALIZACIÓN
    # ─────────────────────────────────────────────────────────────────────────

    def visualize_ascii(self, show_load: bool = True) -> str:
        if not self._cells:
            return "Empty grid"

        lines = []

        for r in range(-self.config.radius, self.config.radius + 1):
            indent = " " * abs(r)
            row = []

            for q in range(-self.config.radius - min(0, r), self.config.radius - max(0, r) + 1):
                coord = HexCoord(q, r)

                if coord in self._cells:
                    cell = self._cells[coord]

                    if cell.role == CellRole.QUEEN:
                        row.append("👑")
                    elif cell.role == CellRole.DRONE:
                        row.append("🐝")
                    elif cell.role == CellRole.NURSERY:
                        row.append("🥚")
                    elif cell.role == CellRole.STORAGE:
                        row.append("📦")
                    elif cell.role == CellRole.GUARD:
                        row.append("🛡️")
                    elif cell.role == CellRole.SCOUT:
                        row.append("🔭")
                    elif cell.state == CellState.FAILED:
                        row.append("💀")
                    elif show_load:
                        if cell.load > self.config.viz_load_high:
                            row.append("🔴")
                        elif cell.load > self.config.viz_load_medium:
                            row.append("🟠")
                        elif cell.load > self.config.viz_load_low:
                            row.append("🟡")
                        elif cell.load > 0:
                            row.append("🟢")
                        else:
                            row.append("⬡ ")
                    else:
                        row.append("⬡ ")
                else:
                    row.append("  ")

            lines.append(indent + " ".join(row))

        return "\n".join(lines)

    def visualize_heatmap(self) -> NDArray[np.float64]:
        size = self.config.radius * 2 + 1
        heatmap = np.zeros((size, size))

        with self._rw_lock.read_lock():
            for coord, cell in self._cells.items():
                x = coord.q + self.config.radius
                y = coord.r + self.config.radius
                if 0 <= x < size and 0 <= y < size:
                    heatmap[y, x] = cell.load

        return heatmap

    # ─────────────────────────────────────────────────────────────────────────
    # SERIALIZACIÓN (v3.0: from_dict completo)
    # ─────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        with self._rw_lock.read_lock():
            return {
                "version": "3.0",
                "config": self.config.to_dict(),
                "topology": self._topology.name,
                "tick_count": self._tick_count,
                "cells": {f"{c.q},{c.r}": cell.to_dict() for c, cell in self._cells.items()},
                "stats": self.get_stats(),
            }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HoneycombGrid:
        """
        v3.0: Deserialización completa con restauración de estado.
        """
        config = HoneycombConfig.from_dict(data["config"])
        grid = cls(config)

        # Restaurar tick count
        grid._tick_count = data.get("tick_count", 0)

        # Restaurar estados de celdas
        for coord_str, cell_data in data.get("cells", {}).items():
            q, r = map(int, coord_str.split(","))
            coord = HexCoord(q, r)
            cell = grid.get_cell(coord)

            if cell is not None:
                state_name = cell_data.get("state", "EMPTY")
                with suppress(KeyError):
                    cell._state = CellState[state_name]

                cell._error_count = cell_data.get("errors", 0)
                cell._ticks_processed = cell_data.get("ticks", 0)
                cell._last_activity = cell_data.get("last_activity", time.time())

        return grid

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        logger.info("HoneycombGrid started")

    def stop(self) -> None:
        self._running = False
        logger.info("HoneycombGrid stopped")

    def shutdown(self) -> None:
        """v3.0: Graceful shutdown con cleanup garantizado."""
        self.stop()
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            # Python < 3.9 no tiene cancel_futures
            self._executor.shutdown(wait=True)
        self._event_bus.shutdown()
        logger.info("HoneycombGrid shutdown complete")

    def __enter__(self) -> HoneycombGrid:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return (
            f"HoneycombGrid(cells={len(self._cells)}, "
            f"radius={self.config.radius}, "
            f"topology={self._topology.name})"
        )


# ─── Helpers ───────────────────────────────────────────────────────────────────


def create_grid(radius: int = 10, topology: str = "flat", **kwargs) -> HoneycombGrid:
    """Factory function para crear grids."""
    config = HoneycombConfig(radius=radius, topology=topology, **kwargs)
    return HoneycombGrid(config)


def benchmark_grid(grid: HoneycombGrid, ticks: int = 100) -> dict[str, float]:
    """Ejecuta benchmark del grid."""
    times = []

    for _ in range(ticks):
        start = time.perf_counter()
        grid.tick()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "total_time": sum(times),
        "avg_tick_time": float(np.mean(times)),
        "min_tick_time": min(times),
        "max_tick_time": max(times),
        "ticks_per_second": 1.0 / float(np.mean(times)) if np.mean(times) > 0 else 0,
        "stddev": float(np.std(times)),
        "p99_tick_time": float(np.percentile(times, 99)),
    }
