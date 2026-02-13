"""
HOC Metrics - Sistema de Métricas y Observabilidad
===================================================

Proporciona monitoreo, métricas y visualización del panal:

MÉTRICAS:
- HiveMetrics: Métricas globales del panal
- CellMetrics: Métricas por celda individual
- SwarmMetrics: Métricas del scheduler

VISUALIZACIÓN:
- HoneycombVisualizer: Renderizado del grid hexagonal
- HeatmapRenderer: Mapas de calor de carga/actividad
- FlowVisualizer: Visualización de flujos de comunicación

Estructura de métricas:

    ┌────────────────────────────────────────────────────────────┐
    │                     MetricsCollector                        │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
    │  │  Counter    │  │   Gauge     │  │  Histogram  │        │
    │  │  (events)   │  │  (current)  │  │  (distrib)  │        │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
    │         │                │                │                │
    │         └────────────────┼────────────────┘                │
    │                          │                                 │
    │                    ┌─────▼─────┐                          │
    │                    │ Exporter  │                          │
    │                    │ (Prom/OT) │                          │
    │                    └───────────┘                          │
    └────────────────────────────────────────────────────────────┘

"""

from __future__ import annotations

import time
import math
import threading
import statistics
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Set, Tuple, Callable,
    Any, TypeVar, Deque, Iterator, TextIO
)
from collections import defaultdict, deque
from io import StringIO

from .core import (
    HexCoord, HexDirection, HoneycombGrid, HoneycombCell,
    QueenCell, WorkerCell, DroneCell, NurseryCell,
    CellRole, CellState
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ═══════════════════════════════════════════════════════════════════════════════
# TIPOS DE MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════

class MetricType(Enum):
    """Tipos de métricas."""
    COUNTER = auto()     # Solo incrementa
    GAUGE = auto()       # Puede subir/bajar
    HISTOGRAM = auto()   # Distribución de valores
    SUMMARY = auto()     # Resumen estadístico


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
    labels: Dict[str, str] = field(default_factory=dict)


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
    
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    
    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None
    ):
        self.name = name
        self.description = description
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        
        self._bucket_counts: Dict[float, int] = {b: 0 for b in self.buckets}
        self._bucket_counts[float('inf')] = 0
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
            self._bucket_counts[float('inf')] += 1
    
    def get_buckets(self) -> Dict[float, int]:
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
    
    def __init__(
        self,
        name: str,
        description: str = "",
        max_samples: int = 1000
    ):
        self.name = name
        self.description = description
        self._samples: Deque[float] = deque(maxlen=max_samples)
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
# CELL METRICS
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
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        )
        
        # Historial
        self._history: Deque[CellMetricSnapshot] = deque(maxlen=100)
    
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
        self.pheromone_level.set(self.cell._pheromone_level)
    
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
            pheromone_level=self.cell._pheromone_level,
            error_count=self.cell._error_count,
            last_activity=self.cell._last_activity,
        )
        self._history.append(snapshot)
        return snapshot
    
    def get_history(self) -> List[CellMetricSnapshot]:
        """Retorna historial de snapshots."""
        return list(self._history)
    
    def get_metrics_dict(self) -> Dict[str, Any]:
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
        self._tasks_by_type: Dict[str, Counter] = defaultdict(
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
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
        )
        self.queue_wait_time = Histogram(
            "swarm_queue_wait_seconds",
            "Time task waits in queue",
            buckets=[0.01, 0.1, 1.0, 5.0, 30.0, 60.0, 300.0]
        )
        
        # Work stealing
        self.work_stolen = Counter("swarm_work_stolen_total", "Work stolen count")
        
        # Comportamientos
        self._behavior_counts: Dict[str, int] = defaultdict(int)
    
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
    
    def record_behavior_distribution(self, behaviors: Dict[str, int]) -> None:
        """Registra distribución de comportamientos."""
        self._behavior_counts = behaviors.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas."""
        return {
            "tasks": {
                "submitted": self.tasks_submitted.get(),
                "completed": self.tasks_completed.get(),
                "failed": self.tasks_failed.get(),
                "success_rate": (
                    self.tasks_completed.get() /
                    max(1, self.tasks_submitted.get())
                ),
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
        self._cell_metrics: Dict[HexCoord, CellMetrics] = {}
        
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
        self._collection_history: Deque[Dict] = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def collect(self) -> Dict[str, Any]:
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
            total_pheromones += cell._pheromone_level
            
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
    
    def get_cell_metrics(self, coord: HexCoord) -> Optional[CellMetrics]:
        """Obtiene métricas de una celda específica."""
        return self._cell_metrics.get(coord)
    
    def get_ring_metrics(self, radius: int) -> Dict[str, Any]:
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
        lines.append(f"║  Total Ticks:      {self.total_ticks.get():>10.0f}                        ║")
        lines.append(f"║  Total Cells:      {len(self.grid._cells):>10}                        ║")
        lines.append(f"║  Total vCores:     {self.total_vcores.get():>10.0f}                        ║")
        lines.append(f"║  Average Load:     {self.average_load.get():>10.2%}                        ║")
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
    
    def get_history(self, limit: int = 100) -> List[Dict]:
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
            lines.append(f'hive_cells_by_state{{state="{state.name}"}} {self.cells_by_state[state].get()}')
        
        for role in CellRole:
            lines.append(f'hive_cells_by_role{{role="{role.name}"}} {self.cells_by_role[role].get()}')
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

class ColorScheme(Enum):
    """Esquemas de color para visualización."""
    LOAD = auto()        # Por carga (verde → rojo)
    STATE = auto()       # Por estado
    ROLE = auto()        # Por rol
    PHEROMONE = auto()   # Por nivel de feromona
    ACTIVITY = auto()    # Por actividad reciente


class HoneycombVisualizer:
    """
    Visualizador del grid hexagonal.
    
    Renderiza el panal en diferentes formatos:
    - ASCII art
    - SVG
    - HTML interactivo
    
    Uso:
        viz = HoneycombVisualizer(grid)
        print(viz.render_ascii())
        svg = viz.render_svg()
    """
    
    # Caracteres para renderizado ASCII
    ASCII_CHARS = {
        CellRole.QUEEN: "👑",
        CellRole.WORKER: "⬡",
        CellRole.DRONE: "🐝",
        CellRole.NURSERY: "🥚",
        CellRole.STORAGE: "📦",
        CellRole.GUARD: "🛡",
        CellRole.SCOUT: "🔍",
    }
    
    LOAD_CHARS = ["⬡", "🟢", "🟡", "🟠", "🔴"]
    
    STATE_CHARS = {
        CellState.EMPTY: "○",
        CellState.ACTIVE: "●",
        CellState.IDLE: "◐",
        CellState.SPAWNING: "◉",
        CellState.MIGRATING: "↔",
        CellState.FAILED: "✗",
        CellState.RECOVERING: "↻",
        CellState.SEALED: "▣",
    }
    
    def __init__(self, grid: HoneycombGrid):
        self.grid = grid
        self._color_scheme = ColorScheme.LOAD
    
    def set_color_scheme(self, scheme: ColorScheme) -> None:
        """Establece el esquema de color."""
        self._color_scheme = scheme
    
    def render_ascii(
        self,
        scheme: Optional[ColorScheme] = None,
        show_coords: bool = False
    ) -> str:
        """
        Renderiza el grid como ASCII art.
        
        Args:
            scheme: Esquema de color a usar
            show_coords: Mostrar coordenadas
            
        Returns:
            String con el grid renderizado
        """
        scheme = scheme or self._color_scheme
        lines = []
        
        radius = self.grid.config.radius
        
        for r in range(-radius, radius + 1):
            # Offset para alineación hexagonal
            indent = " " * abs(r)
            row = []
            
            for q in range(-radius, radius + 1):
                coord = HexCoord(q, r)
                
                if coord in self.grid._cells:
                    cell = self.grid._cells[coord]
                    char = self._get_cell_char(cell, scheme)
                    
                    if show_coords:
                        char = f"{char}({q},{r})"
                    
                    row.append(char)
                else:
                    row.append("  ")
            
            lines.append(indent + " ".join(row))
        
        return "\n".join(lines)
    
    def _get_cell_char(self, cell: HoneycombCell, scheme: ColorScheme) -> str:
        """Obtiene el carácter para una celda según el esquema."""
        if scheme == ColorScheme.ROLE:
            return self.ASCII_CHARS.get(cell.role, "⬡")
        
        elif scheme == ColorScheme.STATE:
            return self.STATE_CHARS.get(cell.state, "?")
        
        elif scheme == ColorScheme.LOAD:
            load_idx = min(int(cell.load * len(self.LOAD_CHARS)), len(self.LOAD_CHARS) - 1)
            return self.LOAD_CHARS[load_idx]
        
        elif scheme == ColorScheme.PHEROMONE:
            if cell._pheromone_level > 0.7:
                return "🔥"
            elif cell._pheromone_level > 0.3:
                return "🌡"
            else:
                return "❄"
        
        elif scheme == ColorScheme.ACTIVITY:
            if cell._last_activity > 0.5:
                return "⚡"
            else:
                return "💤"
        
        return "⬡"
    
    def render_svg(
        self,
        width: int = 800,
        height: int = 600,
        scheme: Optional[ColorScheme] = None
    ) -> str:
        """
        Renderiza el grid como SVG.
        
        Args:
            width: Ancho del SVG
            height: Alto del SVG
            scheme: Esquema de color
            
        Returns:
            String con SVG
        """
        scheme = scheme or self._color_scheme
        
        # Calcular escala
        radius = self.grid.config.radius
        hex_size = min(width, height) / (radius * 4 + 2)
        center_x = width / 2
        center_y = height / 2
        
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            f'<rect width="100%" height="100%" fill="#1a1a2e"/>',
        ]
        
        for coord, cell in self.grid._cells.items():
            # Convertir a coordenadas pixel
            x = center_x + hex_size * (3/2 * coord.q)
            y = center_y + hex_size * (math.sqrt(3)/2 * coord.q + math.sqrt(3) * coord.r)
            
            # Color según esquema
            color = self._get_cell_color(cell, scheme)
            
            # Generar hexágono
            points = []
            for i in range(6):
                angle = math.pi / 3 * i
                px = x + hex_size * 0.9 * math.cos(angle)
                py = y + hex_size * 0.9 * math.sin(angle)
                points.append(f"{px:.1f},{py:.1f}")
            
            svg_parts.append(
                f'<polygon points="{" ".join(points)}" '
                f'fill="{color}" stroke="#ffffff" stroke-width="1"/>'
            )
            
            # Etiqueta opcional
            if cell.role == CellRole.QUEEN:
                svg_parts.append(
                    f'<text x="{x}" y="{y}" text-anchor="middle" '
                    f'dominant-baseline="central" fill="white" font-size="12">👑</text>'
                )
        
        svg_parts.append('</svg>')
        return "\n".join(svg_parts)
    
    def _get_cell_color(self, cell: HoneycombCell, scheme: ColorScheme) -> str:
        """Obtiene el color para una celda según el esquema."""
        if scheme == ColorScheme.LOAD:
            # Verde a rojo según carga
            r = int(255 * cell.load)
            g = int(255 * (1 - cell.load))
            return f"rgb({r},{g},100)"
        
        elif scheme == ColorScheme.STATE:
            colors = {
                CellState.EMPTY: "#333333",
                CellState.ACTIVE: "#00ff00",
                CellState.IDLE: "#888888",
                CellState.SPAWNING: "#ffff00",
                CellState.MIGRATING: "#00ffff",
                CellState.FAILED: "#ff0000",
                CellState.RECOVERING: "#ff8800",
                CellState.SEALED: "#0000ff",
            }
            return colors.get(cell.state, "#ffffff")
        
        elif scheme == ColorScheme.ROLE:
            colors = {
                CellRole.QUEEN: "#ffd700",
                CellRole.WORKER: "#4a90d9",
                CellRole.DRONE: "#ff6600",
                CellRole.NURSERY: "#ff69b4",
                CellRole.STORAGE: "#808080",
                CellRole.GUARD: "#8b0000",
                CellRole.SCOUT: "#00ced1",
            }
            return colors.get(cell.role, "#ffffff")
        
        elif scheme == ColorScheme.PHEROMONE:
            intensity = min(cell._pheromone_level, 1.0)
            return f"rgb({int(255*intensity)},100,{int(255*(1-intensity))})"
        
        return "#ffffff"
    
    def render_html(self, scheme: Optional[ColorScheme] = None) -> str:
        """Renderiza como HTML interactivo."""
        svg = self.render_svg(scheme=scheme)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>HOC Honeycomb Visualizer</title>
    <style>
        body {{
            background: #1a1a2e;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }}
        svg polygon:hover {{
            stroke-width: 3;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    {svg}
</body>
</html>
"""
        return html


class HeatmapRenderer:
    """
    Renderiza mapas de calor del panal.
    
    Visualiza distribución de:
    - Carga
    - Feromonas
    - Errores
    - Actividad
    
    Uso:
        heatmap = HeatmapRenderer(grid)
        svg = heatmap.render("load")
    """
    
    def __init__(self, grid: HoneycombGrid):
        self.grid = grid
    
    def render(
        self,
        metric: str = "load",
        width: int = 600,
        height: int = 600
    ) -> str:
        """
        Renderiza mapa de calor.
        
        Args:
            metric: Métrica a visualizar (load, pheromone, errors)
            width: Ancho
            height: Alto
            
        Returns:
            SVG string
        """
        # Obtener valores
        values = {}
        for coord, cell in self.grid._cells.items():
            if metric == "load":
                values[coord] = cell.load
            elif metric == "pheromone":
                values[coord] = cell._pheromone_level
            elif metric == "errors":
                values[coord] = min(cell._error_count / 10, 1.0)
            else:
                values[coord] = 0.0
        
        # Normalizar
        max_val = max(values.values()) if values else 1.0
        if max_val > 0:
            values = {k: v / max_val for k, v in values.items()}
        
        # Crear visualizador temporal con colores personalizados
        viz = HoneycombVisualizer(self.grid)
        
        # Renderizar con color personalizado basado en valores
        return self._render_heatmap_svg(values, width, height)
    
    def _render_heatmap_svg(
        self,
        values: Dict[HexCoord, float],
        width: int,
        height: int
    ) -> str:
        """Renderiza SVG de mapa de calor."""
        radius = self.grid.config.radius
        hex_size = min(width, height) / (radius * 4 + 2)
        center_x = width / 2
        center_y = height / 2
        
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            f'<rect width="100%" height="100%" fill="#000000"/>',
        ]
        
        for coord, value in values.items():
            # Convertir a coordenadas pixel
            x = center_x + hex_size * (3/2 * coord.q)
            y = center_y + hex_size * (math.sqrt(3)/2 * coord.q + math.sqrt(3) * coord.r)
            
            # Color de mapa de calor (azul → rojo)
            r = int(255 * value)
            b = int(255 * (1 - value))
            color = f"rgb({r},0,{b})"
            
            # Generar hexágono
            points = []
            for i in range(6):
                angle = math.pi / 3 * i
                px = x + hex_size * 0.9 * math.cos(angle)
                py = y + hex_size * 0.9 * math.sin(angle)
                points.append(f"{px:.1f},{py:.1f}")
            
            svg_parts.append(
                f'<polygon points="{" ".join(points)}" '
                f'fill="{color}" stroke="#333333" stroke-width="0.5"/>'
            )
        
        # Leyenda
        svg_parts.append(self._render_legend(width, height))
        
        svg_parts.append('</svg>')
        return "\n".join(svg_parts)
    
    def _render_legend(self, width: int, height: int) -> str:
        """Renderiza leyenda del mapa de calor."""
        legend_width = 20
        legend_height = 100
        x = width - legend_width - 20
        y = (height - legend_height) / 2
        
        parts = []
        
        # Gradiente
        steps = 20
        for i in range(steps):
            value = i / steps
            r = int(255 * value)
            b = int(255 * (1 - value))
            step_y = y + legend_height * (1 - i / steps)
            step_height = legend_height / steps + 1
            parts.append(
                f'<rect x="{x}" y="{step_y}" width="{legend_width}" '
                f'height="{step_height}" fill="rgb({r},0,{b})"/>'
            )
        
        # Labels
        parts.append(
            f'<text x="{x + legend_width + 5}" y="{y + 10}" '
            f'fill="white" font-size="10">High</text>'
        )
        parts.append(
            f'<text x="{x + legend_width + 5}" y="{y + legend_height}" '
            f'fill="white" font-size="10">Low</text>'
        )
        
        return "\n".join(parts)


class FlowVisualizer:
    """
    Visualiza flujos de comunicación en el panal.
    
    Muestra:
    - Rastros de feromonas
    - Patrones de danza
    - Comandos reales
    
    Uso:
        flow = FlowVisualizer(grid, nectar_flow)
        svg = flow.render_pheromone_trails()
    """
    
    def __init__(self, grid: HoneycombGrid, nectar_flow: Optional[Any] = None):
        self.grid = grid
        self.nectar_flow = nectar_flow
    
    def render_pheromone_trails(
        self,
        width: int = 600,
        height: int = 600
    ) -> str:
        """Renderiza rastros de feromonas."""
        radius = self.grid.config.radius
        hex_size = min(width, height) / (radius * 4 + 2)
        center_x = width / 2
        center_y = height / 2
        
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            f'<rect width="100%" height="100%" fill="#1a1a2e"/>',
        ]
        
        # Dibujar celdas base
        for coord, cell in self.grid._cells.items():
            x = center_x + hex_size * (3/2 * coord.q)
            y = center_y + hex_size * (math.sqrt(3)/2 * coord.q + math.sqrt(3) * coord.r)
            
            svg_parts.append(
                f'<circle cx="{x}" cy="{y}" r="{hex_size*0.3}" '
                f'fill="#333366" opacity="0.5"/>'
            )
        
        # Dibujar conexiones de feromonas
        for coord, cell in self.grid._cells.items():
            if cell._pheromone_level < 0.1:
                continue
            
            x1 = center_x + hex_size * (3/2 * coord.q)
            y1 = center_y + hex_size * (math.sqrt(3)/2 * coord.q + math.sqrt(3) * coord.r)
            
            for neighbor in cell.get_all_neighbors():
                if neighbor and neighbor._pheromone_level > 0.1:
                    x2 = center_x + hex_size * (3/2 * neighbor.coord.q)
                    y2 = center_y + hex_size * (math.sqrt(3)/2 * neighbor.coord.q + math.sqrt(3) * neighbor.coord.r)
                    
                    intensity = (cell._pheromone_level + neighbor._pheromone_level) / 2
                    opacity = min(intensity, 0.8)
                    width = 1 + intensity * 3
                    
                    svg_parts.append(
                        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                        f'stroke="#ff6600" stroke-width="{width}" opacity="{opacity}"/>'
                    )
        
        svg_parts.append('</svg>')
        return "\n".join(svg_parts)
    
    def render_activity_flow(
        self,
        width: int = 600,
        height: int = 600
    ) -> str:
        """Renderiza flujo de actividad."""
        # Similar a pheromone trails pero basado en actividad
        radius = self.grid.config.radius
        hex_size = min(width, height) / (radius * 4 + 2)
        center_x = width / 2
        center_y = height / 2
        
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            f'<rect width="100%" height="100%" fill="#0a0a1a"/>',
        ]
        
        # Partículas de actividad
        for coord, cell in self.grid._cells.items():
            if cell._last_activity < 0.1:
                continue
            
            x = center_x + hex_size * (3/2 * coord.q)
            y = center_y + hex_size * (math.sqrt(3)/2 * coord.q + math.sqrt(3) * coord.r)
            
            # Círculo pulsante
            radius_val = hex_size * 0.3 * (1 + cell._last_activity * 0.5)
            opacity = 0.3 + cell._last_activity * 0.5
            
            svg_parts.append(
                f'<circle cx="{x}" cy="{y}" r="{radius_val}" '
                f'fill="#00ff00" opacity="{opacity}"/>'
            )
        
        svg_parts.append('</svg>')
        return "\n".join(svg_parts)
    
    def get_flow_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de flujo."""
        total_pheromone = 0.0
        active_cells = 0
        connections = 0
        
        for coord, cell in self.grid._cells.items():
            total_pheromone += cell._pheromone_level
            
            if cell._last_activity > 0.1:
                active_cells += 1
            
            for neighbor in cell.get_all_neighbors():
                if neighbor and cell._pheromone_level > 0.1 and neighbor._pheromone_level > 0.1:
                    connections += 1
        
        return {
            "total_pheromone": total_pheromone,
            "active_cells": active_cells,
            "pheromone_connections": connections // 2,  # Dividir por 2 (bidireccional)
            "average_pheromone": total_pheromone / len(self.grid._cells) if self.grid._cells else 0,
        }
