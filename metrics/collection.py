"""
HOC Metrics · Collection (primitives + aggregators + transitional internals)
============================================================================

Primitivas de observabilidad (``Counter``/``Gauge``/``Histogram``/``Summary``),
las tres clases públicas de agregación (``CellMetrics``/``SwarmMetrics``/
``HiveMetrics``) y los tipos de datos auxiliares
(``MetricType``/``MetricLabel``/``MetricSample``/``CellMetricSnapshot``).

Adicionalmente aloja las tres clases "internas" que antes vivían en
``hoc.core._metrics_internal`` (Phase 3.3 transición):

- ``_InternalCellMetrics`` — dataclass minimal usado por
  :meth:`hoc.core.HoneycombCell.get_metrics`. Se re-exporta desde
  ``hoc.core`` bajo el alias público ``CellMetrics`` para preservar
  ``from hoc.core import CellMetrics`` (distinta identidad que la
  ``CellMetrics`` pública de este módulo).
- ``GridMetrics`` — snapshot agregado del grid, generado por
  ``HoneycombGrid.tick``.
- ``MetricsCollector`` — colector thread-safe de ``GridMetrics`` con
  historial y agregaciones.

Las dos últimas se importan de forma perezosa desde ``hoc.core.grid`` para
romper el ciclo ``core ↔ metrics``.

Extraído de ``metrics.py`` en Fase 3.3 (continuación).
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import numpy as np

from ..core.cells_base import CellRole, CellState
from ..core.grid_geometry import HexCoord

if TYPE_CHECKING:
    from ..core.cells_base import HoneycombCell
    from ..core.grid import HoneycombGrid

__all__ = [
    # Tipos base
    "MetricType",
    "MetricLabel",
    "MetricSample",
    # Primitivas
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    # Cell metrics (público)
    "CellMetricSnapshot",
    "CellMetrics",
    # Swarm / Hive
    "SwarmMetrics",
    "HiveMetrics",
    # Transicionales (antes en core/_metrics_internal.py)
    "_InternalCellMetrics",
    "GridMetrics",
    "MetricsCollector",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# TIPOS DE MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════


class MetricType(Enum):
    """Tipos de métricas."""

    COUNTER = auto()  # Solo incrementa
    GAUGE = auto()  # Puede subir/bajar
    HISTOGRAM = auto()  # Distribución de valores
    SUMMARY = auto()  # Resumen estadístico


@dataclass
class MetricLabel:
    """Etiquetas para una métrica."""

    name: str
    value: str


@dataclass
class MetricSample:
    """Una muestra de métrica."""

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


class Counter:
    """Contador monotónico (solo incrementa)."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0) -> None:
        """Incrementa el contador."""
        if value < 0:
            raise ValueError("Counter can only be incremented")
        with self._lock:
            self._value += value

    def get(self) -> float:
        return self._value

    def reset(self) -> None:
        with self._lock:
            self._value = 0.0


class Gauge:
    """Valor que puede subir o bajar."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def inc(self, value: float = 1.0) -> None:
        with self._lock:
            self._value += value

    def dec(self, value: float = 1.0) -> None:
        with self._lock:
            self._value -= value

    def get(self) -> float:
        return self._value


class Histogram:
    """Distribución de valores con buckets."""

    DEFAULT_BUCKETS: ClassVar[list[float]] = [
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    ]

    def __init__(self, name: str, description: str = "", buckets: list[float] | None = None):
        self.name = name
        self.description = description
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)

        self._bucket_counts: dict[float, int] = {b: 0 for b in self.buckets}
        self._bucket_counts[float("inf")] = 0
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        """Registra un valor observado."""
        with self._lock:
            self._sum += value
            self._count += 1

            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[bucket] += 1
            self._bucket_counts[float("inf")] += 1

    def get_buckets(self) -> dict[float, int]:
        return self._bucket_counts.copy()

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count else 0.0


class Summary:
    """Resumen estadístico con cuantiles."""

    def __init__(self, name: str, description: str = "", max_samples: int = 1000):
        self.name = name
        self.description = description
        self._samples: deque[float] = deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._samples.append(value)

    def quantile(self, q: float) -> float:
        """Calcula el cuantil q (0.0 - 1.0)."""
        with self._lock:
            if not self._samples:
                return 0.0
            sorted_samples = sorted(self._samples)
            idx = int(len(sorted_samples) * q)
            return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def sum(self) -> float:
        return sum(self._samples)

    @property
    def mean(self) -> float:
        return statistics.mean(self._samples) if self._samples else 0.0

    @property
    def stddev(self) -> float:
        return statistics.stdev(self._samples) if len(self._samples) > 1 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# CELL METRICS (PÚBLICO)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CellMetricSnapshot:
    """Snapshot de métricas de una celda."""

    coord: HexCoord
    timestamp: float
    state: str
    role: str
    load: float
    vcore_count: int
    neighbor_count: int
    pheromone_level: float
    error_count: int
    last_activity: float


class CellMetrics:
    """
    Recolector de métricas por celda.

    Proporciona métricas detalladas de celdas individuales
    incluyendo carga, estado, actividad y errores.

    Uso:
        metrics = CellMetrics(cell)
        metrics.record_tick()
        snapshot = metrics.get_snapshot()
    """

    def __init__(self, cell: HoneycombCell):
        self.cell = cell

        # Contadores
        self.ticks_processed = Counter("cell_ticks_total", "Total ticks processed")
        self.errors = Counter("cell_errors_total", "Total errors")
        self.vcores_added = Counter("cell_vcores_added_total", "vCores added")
        self.vcores_removed = Counter("cell_vcores_removed_total", "vCores removed")

        # Gauges
        self.load = Gauge("cell_load", "Current load")
        self.vcore_count = Gauge("cell_vcores", "Current vCore count")
        self.pheromone_level = Gauge("cell_pheromone", "Pheromone level")

        # Histograms
        self.tick_duration = Histogram(
            "cell_tick_duration_seconds",
            "Tick processing duration",
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        )

        # Historial
        self._history: deque[CellMetricSnapshot] = deque(maxlen=100)

    def record_tick(self, duration: float = 0.0) -> None:
        """Registra un tick procesado."""
        self.ticks_processed.inc()
        self.tick_duration.observe(duration)
        self._update_gauges()

    def record_error(self) -> None:
        """Registra un error."""
        self.errors.inc()

    def record_vcore_change(self, added: bool) -> None:
        """Registra cambio de vCore."""
        if added:
            self.vcores_added.inc()
        else:
            self.vcores_removed.inc()
        self._update_gauges()

    def _update_gauges(self) -> None:
        """Actualiza métricas gauge."""
        self.load.set(self.cell.load)
        self.vcore_count.set(len(self.cell._vcores))
        self.pheromone_level.set(self.cell.pheromone_level)

    def get_snapshot(self) -> CellMetricSnapshot:
        """Obtiene snapshot actual."""
        snapshot = CellMetricSnapshot(
            coord=self.cell.coord,
            timestamp=time.time(),
            state=self.cell.state.name,
            role=self.cell.role.name,
            load=self.cell.load,
            vcore_count=len(self.cell._vcores),
            neighbor_count=self.cell.neighbor_count,
            pheromone_level=self.cell.pheromone_level,
            error_count=self.cell._error_count,
            last_activity=self.cell._last_activity,
        )
        self._history.append(snapshot)
        return snapshot

    def get_history(self) -> list[CellMetricSnapshot]:
        """Retorna historial de snapshots."""
        return list(self._history)

    def get_metrics_dict(self) -> dict[str, Any]:
        """Retorna métricas como diccionario."""
        return {
            "coord": {"q": self.cell.coord.q, "r": self.cell.coord.r},
            "ticks_processed": self.ticks_processed.get(),
            "errors": self.errors.get(),
            "load": self.load.get(),
            "vcore_count": self.vcore_count.get(),
            "tick_duration_mean": self.tick_duration.mean,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SWARM METRICS
# ═══════════════════════════════════════════════════════════════════════════════


class SwarmMetrics:
    """
    Métricas del scheduler de enjambre.

    Monitorea:
    - Tareas procesadas/fallidas/pendientes
    - Latencia de scheduling
    - Distribución de comportamientos
    - Work stealing

    Uso:
        metrics = SwarmMetrics()
        metrics.record_task_completed(task_type, duration)
        stats = metrics.get_stats()
    """

    def __init__(self):
        # Contadores de tareas
        self.tasks_submitted = Counter("swarm_tasks_submitted_total", "Tasks submitted")
        self.tasks_completed = Counter("swarm_tasks_completed_total", "Tasks completed")
        self.tasks_failed = Counter("swarm_tasks_failed_total", "Tasks failed")
        self.tasks_cancelled = Counter("swarm_tasks_cancelled_total", "Tasks cancelled")

        # Por tipo de tarea
        self._tasks_by_type: dict[str, Counter] = defaultdict(
            lambda: Counter("swarm_tasks_by_type", "Tasks by type")
        )

        # Gauges
        self.queue_size = Gauge("swarm_queue_size", "Current queue size")
        self.pending_tasks = Gauge("swarm_pending_tasks", "Pending tasks")
        self.active_workers = Gauge("swarm_active_workers", "Active workers")

        # Histograms
        self.task_duration = Histogram(
            "swarm_task_duration_seconds",
            "Task execution duration",
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
        )
        self.queue_wait_time = Histogram(
            "swarm_queue_wait_seconds",
            "Time task waits in queue",
            buckets=[0.01, 0.1, 1.0, 5.0, 30.0, 60.0, 300.0],
        )

        # Work stealing
        self.work_stolen = Counter("swarm_work_stolen_total", "Work stolen count")

        # Comportamientos
        self._behavior_counts: dict[str, int] = defaultdict(int)

    def record_task_submitted(self, task_type: str) -> None:
        """Registra tarea enviada."""
        self.tasks_submitted.inc()
        self._tasks_by_type[task_type].inc()

    def record_task_completed(self, task_type: str, duration: float) -> None:
        """Registra tarea completada."""
        self.tasks_completed.inc()
        self.task_duration.observe(duration)

    def record_task_failed(self, task_type: str) -> None:
        """Registra tarea fallida."""
        self.tasks_failed.inc()

    def record_queue_wait(self, wait_time: float) -> None:
        """Registra tiempo de espera en cola."""
        self.queue_wait_time.observe(wait_time)

    def record_work_stolen(self, count: int) -> None:
        """Registra trabajo robado."""
        self.work_stolen.inc(count)

    def update_queue_stats(self, queue_size: int, pending: int) -> None:
        """Actualiza estadísticas de cola."""
        self.queue_size.set(queue_size)
        self.pending_tasks.set(pending)

    def record_behavior_distribution(self, behaviors: dict[str, int]) -> None:
        """Registra distribución de comportamientos."""
        self._behavior_counts = behaviors.copy()

    def get_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas completas."""
        return {
            "tasks": {
                "submitted": self.tasks_submitted.get(),
                "completed": self.tasks_completed.get(),
                "failed": self.tasks_failed.get(),
                "success_rate": (self.tasks_completed.get() / max(1, self.tasks_submitted.get())),
            },
            "queue": {
                "size": self.queue_size.get(),
                "pending": self.pending_tasks.get(),
            },
            "duration": {
                "mean": self.task_duration.mean,
                "count": self.task_duration.count,
            },
            "work_stolen": self.work_stolen.get(),
            "behaviors": dict(self._behavior_counts),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HIVE METRICS (GLOBAL)
# ═══════════════════════════════════════════════════════════════════════════════


class HiveMetrics:
    """
    Métricas globales del panal.

    Agrega métricas de todas las celdas y subsistemas,
    proporcionando una vista unificada del estado del cluster.

    Uso:
        metrics = HiveMetrics(grid)
        metrics.collect()
        report = metrics.generate_report()
    """

    def __init__(self, grid: HoneycombGrid):
        self.grid = grid

        # Métricas de celdas
        self._cell_metrics: dict[HexCoord, CellMetrics] = {}

        # Inicializar métricas por celda
        for coord, cell in grid._cells.items():
            self._cell_metrics[coord] = CellMetrics(cell)

        # Métricas globales
        self.total_ticks = Counter("hive_ticks_total", "Total ticks")
        self.total_vcores = Gauge("hive_vcores_total", "Total vCores")
        self.total_pheromones = Gauge("hive_pheromones_total", "Total pheromones")
        self.average_load = Gauge("hive_average_load", "Average load")

        # Por estado
        self.cells_by_state = {
            state: Gauge(f"hive_cells_{state.name.lower()}", f"Cells in {state.name}")
            for state in CellState
        }

        # Por rol
        self.cells_by_role = {
            role: Gauge(f"hive_cells_{role.name.lower()}", f"Cells with role {role.name}")
            for role in CellRole
        }

        # Historial
        self._collection_history: deque[dict] = deque(maxlen=1000)
        self._lock = threading.Lock()

    def collect(self) -> dict[str, Any]:
        """
        Recolecta métricas de todo el panal.

        Returns:
            Diccionario con métricas recolectadas
        """
        self.total_ticks.inc()

        # Contar por estado y rol
        state_counts = {state: 0 for state in CellState}
        role_counts = {role: 0 for role in CellRole}

        total_load = 0.0
        total_vcores = 0
        total_pheromones = 0.0

        for coord, cell in self.grid._cells.items():
            state_counts[cell.state] += 1
            role_counts[cell.role] += 1
            total_load += cell.load
            total_vcores += len(cell._vcores)
            total_pheromones += cell.pheromone_level

            # Actualizar métricas de celda
            self._cell_metrics[coord].get_snapshot()

        # Actualizar gauges
        self.total_vcores.set(total_vcores)
        self.total_pheromones.set(total_pheromones)
        self.average_load.set(total_load / len(self.grid._cells) if self.grid._cells else 0)

        for state, count in state_counts.items():
            self.cells_by_state[state].set(count)

        for role, count in role_counts.items():
            self.cells_by_role[role].set(count)

        # Crear snapshot
        snapshot = {
            "timestamp": time.time(),
            "tick": self.total_ticks.get(),
            "cells": len(self.grid._cells),
            "total_vcores": total_vcores,
            "average_load": self.average_load.get(),
            "total_pheromones": total_pheromones,
            "by_state": {s.name: c for s, c in state_counts.items()},
            "by_role": {r.name: c for r, c in role_counts.items()},
        }

        with self._lock:
            self._collection_history.append(snapshot)

        return snapshot

    def get_cell_metrics(self, coord: HexCoord) -> CellMetrics | None:
        """Obtiene métricas de una celda específica."""
        return self._cell_metrics.get(coord)

    def get_ring_metrics(self, radius: int) -> dict[str, Any]:
        """Obtiene métricas agregadas de un anillo."""
        origin = HexCoord.origin()
        ring_coords = origin.ring(radius)

        loads = []
        vcores = 0
        errors = 0

        for coord in ring_coords:
            cell = self.grid.get_cell(coord)
            if cell:
                loads.append(cell.load)
                vcores += len(cell._vcores)
                errors += cell._error_count

        return {
            "radius": radius,
            "cells": len(ring_coords),
            "average_load": statistics.mean(loads) if loads else 0,
            "total_vcores": vcores,
            "total_errors": errors,
        }

    def generate_report(self) -> str:
        """Genera reporte de métricas en formato texto."""
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║               HOC HIVE METRICS REPORT                        ║",
            "╠══════════════════════════════════════════════════════════════╣",
        ]

        # Métricas globales
        lines.append(
            f"║  Total Ticks:      {self.total_ticks.get():>10.0f}                        ║"
        )
        lines.append(f"║  Total Cells:      {len(self.grid._cells):>10}                        ║")
        lines.append(
            f"║  Total vCores:     {self.total_vcores.get():>10.0f}                        ║"
        )
        lines.append(
            f"║  Average Load:     {self.average_load.get():>10.2%}                        ║"
        )
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        # Por estado
        lines.append("║  CELLS BY STATE:                                             ║")
        for state in CellState:
            count = self.cells_by_state[state].get()
            if count > 0:
                lines.append(f"║    {state.name:<15} {count:>10.0f}                           ║")

        lines.append("╠══════════════════════════════════════════════════════════════╣")

        # Por rol
        lines.append("║  CELLS BY ROLE:                                              ║")
        for role in CellRole:
            count = self.cells_by_role[role].get()
            if count > 0:
                lines.append(f"║    {role.name:<15} {count:>10.0f}                           ║")

        lines.append("╚══════════════════════════════════════════════════════════════╝")

        return "\n".join(lines)

    def get_history(self, limit: int = 100) -> list[dict]:
        """Retorna historial de colecciones."""
        with self._lock:
            return list(self._collection_history)[-limit:]

    def export_prometheus(self) -> str:
        """Exporta métricas en formato Prometheus."""
        lines = []

        # Formato: metric_name{labels} value timestamp
        lines.append(f"hive_ticks_total {self.total_ticks.get()}")
        lines.append(f"hive_vcores_total {self.total_vcores.get()}")
        lines.append(f"hive_average_load {self.average_load.get()}")
        lines.append(f"hive_pheromones_total {self.total_pheromones.get()}")

        for state in CellState:
            lines.append(
                f'hive_cells_by_state{{state="{state.name}"}} {self.cells_by_state[state].get()}'
            )

        for role in CellRole:
            lines.append(
                f'hive_cells_by_role{{role="{role.name}"}} {self.cells_by_role[role].get()}'
            )

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSICIONALES (antes en core/_metrics_internal.py)
#
# ``_InternalCellMetrics`` es distinto de la ``CellMetrics`` pública definida
# arriba: es un dataclass minimal que usa
# :meth:`hoc.core.HoneycombCell.get_metrics`. Se re-exporta desde
# ``hoc.core`` bajo el alias público ``CellMetrics`` para preservar
# ``from hoc.core import CellMetrics`` con la identidad original.
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class _InternalCellMetrics:
    """Métricas de una celda individual."""

    coord: HexCoord
    role: CellRole
    state: CellState
    load: float
    vcore_count: int
    error_count: int
    ticks_processed: int
    pheromone_total: float
    neighbor_count: int
    last_activity: float
    circuit_state: str = "CLOSED"  # v3.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "coord": self.coord.to_dict(),
            "role": self.role.name,
            "state": self.state.name,
            "load": self.load,
            "vcore_count": self.vcore_count,
            "error_count": self.error_count,
            "ticks_processed": self.ticks_processed,
            "pheromone_total": self.pheromone_total,
            "neighbor_count": self.neighbor_count,
            "last_activity": self.last_activity,
            "circuit_state": self.circuit_state,
        }


@dataclass(slots=True)
class GridMetrics:
    """Métricas agregadas del grid."""

    timestamp: float
    total_cells: int
    active_cells: int
    idle_cells: int
    failed_cells: int
    total_vcores: int
    average_load: float
    max_load: float
    min_load: float
    load_stddev: float
    total_pheromones: float
    ticks_per_second: float
    errors_per_second: float
    work_steals: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """Colector de métricas thread-safe con historial y agregaciones."""

    __slots__ = ("_counters", "_history", "_last_sample", "_lock", "_max_history", "_sample_rate")

    def __init__(self, max_history: int = 1000, sample_rate: float = 1.0):
        self._history: deque = deque(maxlen=max_history)
        self._max_history = max_history
        self._sample_rate = sample_rate
        self._last_sample = 0.0
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)

    def record(self, metrics: GridMetrics) -> None:
        current = time.time()
        if current - self._last_sample >= 1.0 / self._sample_rate:
            with self._lock:
                self._history.append(metrics)
                self._last_sample = current

    def increment(self, counter: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[counter] += amount

    def get_counter(self, counter: str) -> int:
        return self._counters.get(counter, 0)

    def get_history(self, limit: int | None = None) -> list[GridMetrics]:
        with self._lock:
            if limit:
                return list(self._history)[-limit:]
            return list(self._history)

    def get_latest(self) -> GridMetrics | None:
        with self._lock:
            return self._history[-1] if self._history else None

    def get_averages(self, window: int = 60) -> dict[str, float]:
        history = self.get_history(window)
        if not history:
            return {}

        return {
            "avg_load": float(np.mean([m.average_load for m in history])),
            "avg_vcores": float(np.mean([m.total_vcores for m in history])),
            "avg_tps": float(np.mean([m.ticks_per_second for m in history])),
            "avg_errors": float(np.mean([m.errors_per_second for m in history])),
        }
