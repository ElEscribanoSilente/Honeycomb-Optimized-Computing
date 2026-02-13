"""
HOC Swarm Scheduler - Scheduling Bio-Inspirado
===============================================

Implementa scheduling de tareas usando metáforas de colmena:

ROLES DE ABEJAS:
- Forager (Recolectora): Busca y ejecuta trabajo disponible
- Nurse (Nodriza): Cuida procesos nuevos, warmup
- Scout (Exploradora): Busca nuevos recursos/trabajo
- Guard (Guardia): Validación y seguridad

COMPORTAMIENTOS:
- Reclutamiento: Abejas exitosas reclutan a otras
- División del trabajo: Roles dinámicos según necesidad
- Umbral de respuesta: Diferentes sensibilidades a tareas
- Aprendizaje: Mejora con experiencia

Flujo de scheduling:

    ┌─────────────────────────────────────────────────────────────┐
    │                     SwarmScheduler                          │
    │                                                             │
    │   ┌─────────┐     ┌─────────┐     ┌─────────┐              │
    │   │  Task   │────▶│  Queue  │────▶│ Worker  │              │
    │   │  Pool   │     │ (Nectar)│     │  Pool   │              │
    │   └─────────┘     └────┬────┘     └────┬────┘              │
    │                        │               │                    │
    │                        ▼               ▼                    │
    │   ┌──────────────────────────────────────────────────────┐ │
    │   │                  Bee Behaviors                       │ │
    │   │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │ │
    │   │  │Forager │ │ Nurse  │ │ Scout  │ │ Guard  │        │ │
    │   │  └────────┘ └────────┘ └────────┘ └────────┘        │ │
    │   └──────────────────────────────────────────────────────┘ │
    │                        │                                    │
    │                        ▼                                    │
    │   ┌──────────────────────────────────────────────────────┐ │
    │   │              Pheromone Feedback                      │ │
    │   │         (Success → Reinforce Trail)                  │ │
    │   └──────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────┘

"""

from __future__ import annotations

import time
import random
import threading
import heapq
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Set, Tuple, Callable,
    Any, Iterator, Deque, TypeVar, Protocol
)
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import numpy as np

from .core import HexCoord, HexDirection, HoneycombCell, HoneycombGrid, WorkerCell
from .nectar import PheromoneTrail, PheromoneType, NectarFlow

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ═══════════════════════════════════════════════════════════════════════════════
# TAREAS
# ═══════════════════════════════════════════════════════════════════════════════

class TaskState(Enum):
    """Estado de una tarea."""
    PENDING = auto()      # Esperando ser tomada
    ASSIGNED = auto()     # Asignada a un worker
    RUNNING = auto()      # En ejecución
    COMPLETED = auto()    # Completada exitosamente
    FAILED = auto()       # Fallida
    CANCELLED = auto()    # Cancelada


class TaskPriority(Enum):
    """Prioridad de tarea."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass(order=True)
class HiveTask:
    """
    Una tarea en el scheduler del enjambre.
    
    Las tareas pueden ser:
    - Trabajo de cómputo (ejecutar cerebros)
    - Spawning de entidades
    - Migración de datos
    - Mantenimiento del sistema
    """
    priority: int = field(compare=True)
    created_at: float = field(compare=True, default_factory=time.time)
    task_id: str = field(compare=False, default="")
    task_type: str = field(compare=False, default="compute")
    target_cell: Optional[HexCoord] = field(compare=False, default=None)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)
    state: TaskState = field(compare=False, default=TaskState.PENDING)
    assigned_to: Optional[HexCoord] = field(compare=False, default=None)
    attempts: int = field(compare=False, default=0)
    max_attempts: int = field(compare=False, default=3)
    timeout_seconds: float = field(compare=False, default=30.0)
    callback: Optional[Callable[[Any], None]] = field(compare=False, default=None)
    result: Any = field(compare=False, default=None)
    error: Optional[str] = field(compare=False, default=None)
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"task_{id(self)}_{time.time():.0f}"
    
    def is_expired(self) -> bool:
        """Verifica si la tarea expiró."""
        if self.state == TaskState.RUNNING:
            elapsed = time.time() - self.created_at
            return elapsed > self.timeout_seconds
        return False
    
    def can_retry(self) -> bool:
        """Verifica si se puede reintentar."""
        return self.attempts < self.max_attempts


# Alias para tipos comunes de carga
@dataclass
class TaskPollen:
    """Carga de datos pequeña (pollen = polen)."""
    data: bytes
    source: HexCoord
    destination: HexCoord


@dataclass
class TaskNectar:
    """Carga de datos procesable (nectar = néctar)."""
    entity_ids: List[str]
    operation: str
    params: Dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════════
# COMPORTAMIENTOS DE ABEJAS
# ═══════════════════════════════════════════════════════════════════════════════

class BeeBehavior(ABC):
    """
    Comportamiento base de una abeja trabajadora.
    
    Cada comportamiento define cómo una celda selecciona
    y ejecuta tareas, basándose en umbrales de respuesta
    y retroalimentación del ambiente (feromonas).
    """
    
    def __init__(self, cell: HoneycombCell, nectar_flow: NectarFlow):
        self.cell = cell
        self.nectar = nectar_flow
        self.experience: Dict[str, float] = defaultdict(float)
        self.response_threshold: float = 0.5
        self._last_task: Optional[HiveTask] = None
        self._success_streak: int = 0
    
    @abstractmethod
    def select_task(self, available_tasks: List[HiveTask]) -> Optional[HiveTask]:
        """Selecciona una tarea para ejecutar."""
        pass
    
    @abstractmethod
    def execute_task(self, task: HiveTask) -> bool:
        """Ejecuta una tarea. Retorna True si exitosa."""
        pass
    
    def update_threshold(self, success: bool) -> None:
        """
        Actualiza el umbral de respuesta basado en resultados.
        
        Exitoso → Baja umbral (más propenso a tomar tareas similares)
        Fallido → Sube umbral (menos propenso a tomar tareas similares)
        """
        delta = 0.1 if success else -0.1
        self.response_threshold = max(0.1, min(0.9, self.response_threshold + delta))
        
        if success:
            self._success_streak += 1
            # Bonus por racha
            if self._success_streak >= 3:
                self.response_threshold *= 0.9
        else:
            self._success_streak = 0
    
    def should_respond(self, stimulus: float) -> bool:
        """
        Modelo de umbral de respuesta.
        
        Probabilidad de responder = stimulus^2 / (stimulus^2 + threshold^2)
        """
        if stimulus <= 0:
            return False
        
        prob = (stimulus ** 2) / (stimulus ** 2 + self.response_threshold ** 2)
        return random.random() < prob
    
    def deposit_success_pheromone(self, task: HiveTask) -> None:
        """Deposita feromona de éxito después de completar tarea."""
        self.nectar.deposit_pheromone(
            self.cell.coord,
            PheromoneType.SUCCESS,
            0.5 + (self._success_streak * 0.1),
            metadata={"task_type": task.task_type}
        )
    
    def deposit_failure_pheromone(self, task: HiveTask) -> None:
        """Deposita feromona de fallo."""
        self.nectar.deposit_pheromone(
            self.cell.coord,
            PheromoneType.FAILURE,
            0.3,
            metadata={"task_type": task.task_type}
        )


class ForagerBehavior(BeeBehavior):
    """
    Comportamiento de Recolectora.
    
    - Busca activamente trabajo
    - Sigue rastros de feromonas de éxito
    - Prioriza tareas con alta calidad percibida
    - Deposita feromonas de reclutamiento cuando encuentra buen trabajo
    """
    
    def __init__(self, cell: HoneycombCell, nectar_flow: NectarFlow):
        super().__init__(cell, nectar_flow)
        self.specialization: Optional[str] = None
        self.recruitment_threshold: float = 0.7
    
    def select_task(self, available_tasks: List[HiveTask]) -> Optional[HiveTask]:
        if not available_tasks:
            return None
        
        # Filtrar por especialización si existe
        if self.specialization:
            specialized = [t for t in available_tasks if t.task_type == self.specialization]
            if specialized:
                available_tasks = specialized
        
        # Ordenar por estímulo (prioridad + feromonas)
        scored_tasks = []
        for task in available_tasks:
            # Estímulo base por prioridad
            priority_score = (5 - task.priority) / 5
            
            # Bonus por feromonas de éxito en la celda destino
            pheromone_score = 0
            if task.target_cell:
                pheromone_score = self.nectar.sense_pheromone(
                    task.target_cell, PheromoneType.SUCCESS
                ) * 0.5
            
            # Penalización por feromonas de fallo
            failure_penalty = self.nectar.sense_pheromone(
                self.cell.coord, PheromoneType.FAILURE
            ) * 0.3
            
            total_stimulus = priority_score + pheromone_score - failure_penalty
            scored_tasks.append((total_stimulus, task))
        
        # Seleccionar probabilísticamente (no siempre la mejor)
        scored_tasks.sort(reverse=True)
        
        for stimulus, task in scored_tasks:
            if self.should_respond(stimulus):
                return task
        
        # Si ninguna pasó el umbral, tomar la mejor de todas formas
        return scored_tasks[0][1] if scored_tasks else None
    
    def execute_task(self, task: HiveTask) -> bool:
        """Ejecuta la tarea de recolección/cómputo."""
        try:
            task.state = TaskState.RUNNING
            task.assigned_to = self.cell.coord
            
            # Simular ejecución (en implementación real, ejecutar vCore)
            # Aquí integraríamos con CAMV
            if task.payload.get("execute"):
                result = task.payload["execute"]()
            else:
                result = True
            
            task.state = TaskState.COMPLETED
            task.result = result
            
            # Actualizar experiencia
            self.experience[task.task_type] += 1
            
            # Depositar feromonas
            self.deposit_success_pheromone(task)
            
            # Reclutar si fue muy exitoso
            if self._success_streak >= 3:
                self._recruit()
            
            self.update_threshold(True)
            return True
            
        except Exception as e:
            task.state = TaskState.FAILED
            task.error = str(e)
            task.attempts += 1
            
            self.deposit_failure_pheromone(task)
            self.update_threshold(False)
            return False
    
    def _recruit(self) -> None:
        """Deposita feromonas de reclutamiento."""
        self.nectar.deposit_pheromone(
            self.cell.coord,
            PheromoneType.RECRUITMENT,
            1.0,
            metadata={"specialization": self.specialization}
        )


class NurseBehavior(BeeBehavior):
    """
    Comportamiento de Nodriza.
    
    - Cuida procesos nuevos (warmup de vCores)
    - Prepara entidades recién spawneadas
    - Transfiere entidades listas a workers
    """
    
    def __init__(self, cell: HoneycombCell, nectar_flow: NectarFlow):
        super().__init__(cell, nectar_flow)
        self.incubating: List[Any] = []
        self.warmup_ticks: int = 3
    
    def select_task(self, available_tasks: List[HiveTask]) -> Optional[HiveTask]:
        # Priorizar tareas de spawning/incubación
        spawn_tasks = [t for t in available_tasks if t.task_type == "spawn"]
        if spawn_tasks:
            return spawn_tasks[0]
        
        # También acepta tareas de warmup
        warmup_tasks = [t for t in available_tasks if t.task_type == "warmup"]
        if warmup_tasks:
            return warmup_tasks[0]
        
        return None
    
    def execute_task(self, task: HiveTask) -> bool:
        try:
            task.state = TaskState.RUNNING
            
            if task.task_type == "spawn":
                # Incubar nueva entidad
                entity_spec = task.payload.get("spec", {})
                self.incubating.append({
                    "spec": entity_spec,
                    "ticks_remaining": self.warmup_ticks,
                    "task": task,
                })
                task.state = TaskState.COMPLETED
                return True
                
            elif task.task_type == "warmup":
                # Warmup de vCore existente
                vcore = task.payload.get("vcore")
                if vcore and hasattr(vcore, "warmup"):
                    vcore.warmup()
                task.state = TaskState.COMPLETED
                return True
            
            return False
            
        except Exception as e:
            task.state = TaskState.FAILED
            task.error = str(e)
            return False
    
    def tick_incubation(self) -> List[Any]:
        """Avanza la incubación y retorna entidades listas."""
        ready = []
        still_incubating = []
        
        for item in self.incubating:
            item["ticks_remaining"] -= 1
            if item["ticks_remaining"] <= 0:
                ready.append(item)
            else:
                still_incubating.append(item)
        
        self.incubating = still_incubating
        return ready


class ScoutBehavior(BeeBehavior):
    """
    Comportamiento de Exploradora.
    
    - Explora celdas lejanas buscando recursos
    - Informa sobre carga en diferentes áreas
    - Útil para balanceo de carga proactivo
    """
    
    def __init__(self, cell: HoneycombCell, nectar_flow: NectarFlow):
        super().__init__(cell, nectar_flow)
        self.explored: Set[HexCoord] = set()
        self.exploration_radius: int = 5
    
    def select_task(self, available_tasks: List[HiveTask]) -> Optional[HiveTask]:
        # Scouts prefieren tareas de exploración
        explore_tasks = [t for t in available_tasks if t.task_type == "explore"]
        if explore_tasks:
            return explore_tasks[0]
        
        # También aceptan tareas en celdas lejanas
        distant_tasks = [
            t for t in available_tasks
            if t.target_cell and self.cell.coord.distance_to(t.target_cell) > 3
        ]
        if distant_tasks:
            return distant_tasks[0]
        
        return None
    
    def execute_task(self, task: HiveTask) -> bool:
        try:
            task.state = TaskState.RUNNING
            
            if task.task_type == "explore":
                # Explorar área
                target = task.payload.get("target", self.cell.coord)
                report = self._explore_area(target)
                task.result = report
                task.state = TaskState.COMPLETED
                
                # Depositar información encontrada
                if report.get("resources"):
                    self.nectar.deposit_pheromone(
                        target,
                        PheromoneType.FOOD,
                        report["resources"],
                    )
                
                return True
            
            return False
            
        except Exception as e:
            task.state = TaskState.FAILED
            task.error = str(e)
            return False
    
    def _explore_area(self, center: HexCoord) -> Dict[str, Any]:
        """Explora un área y reporta hallazgos."""
        report = {
            "center": center,
            "cells_explored": 0,
            "total_load": 0,
            "available_cells": 0,
            "resources": 0,
        }
        
        for coord in center.spiral(self.exploration_radius):
            self.explored.add(coord)
            report["cells_explored"] += 1
            
            # En implementación real, acceder a datos de grid
            # Por ahora, simular
            report["available_cells"] += 1
        
        return report


class GuardBehavior(BeeBehavior):
    """
    Comportamiento de Guardia.
    
    - Valida tareas entrantes
    - Detecta anomalías
    - Bloquea tareas maliciosas o mal formadas
    """
    
    def __init__(self, cell: HoneycombCell, nectar_flow: NectarFlow):
        super().__init__(cell, nectar_flow)
        self.blocked_sources: Set[HexCoord] = set()
        self.validation_rules: List[Callable[[HiveTask], bool]] = []
    
    def select_task(self, available_tasks: List[HiveTask]) -> Optional[HiveTask]:
        # Guards manejan tareas de validación
        validate_tasks = [t for t in available_tasks if t.task_type == "validate"]
        if validate_tasks:
            return validate_tasks[0]
        return None
    
    def execute_task(self, task: HiveTask) -> bool:
        try:
            task.state = TaskState.RUNNING
            
            if task.task_type == "validate":
                target_task = task.payload.get("target_task")
                if target_task:
                    is_valid = self._validate_task(target_task)
                    task.result = {"valid": is_valid}
                    task.state = TaskState.COMPLETED
                    
                    if not is_valid:
                        self.nectar.deposit_pheromone(
                            self.cell.coord,
                            PheromoneType.DANGER,
                            0.8,
                        )
                    
                    return True
            
            return False
            
        except Exception as e:
            task.state = TaskState.FAILED
            task.error = str(e)
            return False
    
    def _validate_task(self, task: HiveTask) -> bool:
        """Valida una tarea contra las reglas."""
        # Verificar fuente bloqueada
        if task.assigned_to in self.blocked_sources:
            return False
        
        # Aplicar reglas custom
        for rule in self.validation_rules:
            if not rule(task):
                return False
        
        return True
    
    def add_validation_rule(self, rule: Callable[[HiveTask], bool]) -> None:
        """Añade una regla de validación."""
        self.validation_rules.append(rule)


# ═══════════════════════════════════════════════════════════════════════════════
# POLÍTICAS DE SCHEDULING
# ═══════════════════════════════════════════════════════════════════════════════

class SwarmPolicy(Enum):
    """Políticas de scheduling del enjambre."""
    PRIORITY = auto()          # Siempre la de mayor prioridad
    ROUND_ROBIN = auto()       # Rotación equitativa
    PHEROMONE_GUIDED = auto()  # Guiado por feromonas
    RANDOM = auto()            # Aleatorio
    LEAST_LOADED = auto()      # Celda menos cargada
    LOCALITY = auto()          # Preferir celdas cercanas


@dataclass
class SwarmConfig:
    """Configuración del scheduler de enjambre."""
    
    # Políticas
    default_policy: SwarmPolicy = SwarmPolicy.PHEROMONE_GUIDED
    
    # Workers
    foragers_ratio: float = 0.6      # 60% recolectoras
    nurses_ratio: float = 0.15       # 15% nodrizas
    scouts_ratio: float = 0.15       # 15% exploradoras
    guards_ratio: float = 0.1        # 10% guardias
    
    # Tareas
    max_queue_size: int = 10000
    task_timeout_seconds: float = 30.0
    max_task_retries: int = 3
    
    # Balanceo
    rebalance_interval_ticks: int = 10
    load_threshold_high: float = 0.8
    load_threshold_low: float = 0.2
    
    # Feromonas
    success_pheromone_weight: float = 1.0
    failure_pheromone_weight: float = 0.5
    
    # Work stealing
    enable_work_stealing: bool = True
    steal_threshold: float = 0.3
    steal_batch_size: int = 5


# ═══════════════════════════════════════════════════════════════════════════════
# BALANCEADOR DE ENJAMBRE
# ═══════════════════════════════════════════════════════════════════════════════

class LoadDistribution:
    """Estadísticas de distribución de carga."""
    
    def __init__(self):
        self.cell_loads: Dict[HexCoord, float] = {}
        self.ring_loads: Dict[int, float] = {}
        self.total_load: float = 0.0
        self.timestamp: float = time.time()
    
    def update(self, grid: HoneycombGrid) -> None:
        """Actualiza estadísticas desde el grid."""
        self.cell_loads.clear()
        self.ring_loads.clear()
        self.total_load = 0.0
        self.timestamp = time.time()
        
        for coord, cell in grid._cells.items():
            load = cell.load
            self.cell_loads[coord] = load
            self.total_load += load
            
            # Calcular ring
            ring = coord.distance_to(HexCoord.origin())
            if ring not in self.ring_loads:
                self.ring_loads[ring] = 0.0
            self.ring_loads[ring] += load
        
        # Promediar por ring
        ring_counts = defaultdict(int)
        for coord in self.cell_loads:
            ring = coord.distance_to(HexCoord.origin())
            ring_counts[ring] += 1
        
        for ring in self.ring_loads:
            if ring_counts[ring] > 0:
                self.ring_loads[ring] /= ring_counts[ring]
    
    @property
    def average_load(self) -> float:
        if not self.cell_loads:
            return 0.0
        return self.total_load / len(self.cell_loads)
    
    @property
    def max_load(self) -> float:
        return max(self.cell_loads.values()) if self.cell_loads else 0.0
    
    @property
    def min_load(self) -> float:
        return min(self.cell_loads.values()) if self.cell_loads else 0.0
    
    @property
    def load_variance(self) -> float:
        if not self.cell_loads:
            return 0.0
        avg = self.average_load
        return sum((l - avg) ** 2 for l in self.cell_loads.values()) / len(self.cell_loads)


class SwarmBalancer:
    """
    Balanceador de carga del enjambre.
    
    Usa una combinación de:
    - Trabajo robado (work stealing)
    - Migración guiada por feromonas
    - Redistribución periódica
    """
    
    def __init__(self, grid: HoneycombGrid, config: SwarmConfig):
        self.grid = grid
        self.config = config
        self._distribution = LoadDistribution()
        self._last_rebalance = 0
        self._migrations_performed = 0
        self._lock = threading.Lock()
    
    def update_distribution(self) -> LoadDistribution:
        """Actualiza y retorna la distribución de carga."""
        with self._lock:
            self._distribution.update(self.grid)
            return self._distribution
    
    def find_overloaded_cells(self) -> List[HexCoord]:
        """Encuentra celdas sobrecargadas."""
        return [
            coord for coord, load in self._distribution.cell_loads.items()
            if load >= self.config.load_threshold_high
        ]
    
    def find_underloaded_cells(self) -> List[HexCoord]:
        """Encuentra celdas subcargadas."""
        return [
            coord for coord, load in self._distribution.cell_loads.items()
            if load <= self.config.load_threshold_low
        ]
    
    def suggest_migrations(self) -> List[Tuple[HexCoord, HexCoord, int]]:
        """
        Sugiere migraciones de trabajo.
        
        Returns:
            Lista de (origen, destino, cantidad) para migrar
        """
        suggestions = []
        
        overloaded = self.find_overloaded_cells()
        underloaded = self.find_underloaded_cells()
        
        if not overloaded or not underloaded:
            return suggestions
        
        for src in overloaded:
            src_cell = self.grid.get_cell(src)
            if not src_cell:
                continue
            
            # Encontrar el destino más cercano que esté subcargado
            underloaded.sort(key=lambda c: src.distance_to(c))
            
            for dst in underloaded:
                dst_cell = self.grid.get_cell(dst)
                if not dst_cell:
                    continue
                
                # Calcular cuántos migrar
                src_vcores = len(src_cell._vcores)
                dst_capacity = self.config.steal_batch_size
                migrate_count = min(
                    src_vcores // 3,  # No migrar más de 1/3
                    dst_capacity,
                    self.config.steal_batch_size
                )
                
                if migrate_count > 0:
                    suggestions.append((src, dst, migrate_count))
                    break
        
        return suggestions
    
    def execute_work_stealing(self) -> int:
        """
        Ejecuta work stealing.
        
        Returns:
            Número de tareas robadas
        """
        if not self.config.enable_work_stealing:
            return 0
        
        total_stolen = 0
        
        underloaded = self.find_underloaded_cells()
        
        for coord in underloaded:
            cell = self.grid.get_cell(coord)
            if not cell or not isinstance(cell, WorkerCell):
                continue
            
            if not cell.can_steal_work():
                continue
            
            # Buscar vecino con más carga
            best_neighbor = None
            best_load = 0
            
            for neighbor in cell.get_all_neighbors():
                if isinstance(neighbor, WorkerCell) and neighbor.load > self.config.load_threshold_high:
                    if neighbor.load > best_load:
                        best_load = neighbor.load
                        best_neighbor = neighbor
            
            if best_neighbor:
                stolen = cell.steal_from(best_neighbor, self.config.steal_batch_size)
                total_stolen += stolen
                self._migrations_performed += stolen
        
        return total_stolen
    
    def rebalance_if_needed(self, tick: int) -> bool:
        """
        Rebalancea si es necesario.
        
        Returns:
            True si se realizó rebalanceo
        """
        if tick - self._last_rebalance < self.config.rebalance_interval_ticks:
            return False
        
        self._last_rebalance = tick
        self.update_distribution()
        
        # Check if rebalance needed
        if self._distribution.load_variance < 0.1:
            return False  # Already well balanced
        
        # Execute work stealing
        stolen = self.execute_work_stealing()
        
        return stolen > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del balanceador."""
        return {
            "average_load": self._distribution.average_load,
            "max_load": self._distribution.max_load,
            "min_load": self._distribution.min_load,
            "load_variance": self._distribution.load_variance,
            "overloaded_cells": len(self.find_overloaded_cells()),
            "underloaded_cells": len(self.find_underloaded_cells()),
            "migrations_performed": self._migrations_performed,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULER PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

class SwarmScheduler:
    """
    Scheduler de Enjambre Principal.
    
    Coordina:
    - Cola de tareas por prioridad
    - Asignación a celdas basada en comportamientos
    - Balanceo de carga
    - Retroalimentación por feromonas
    """
    
    def __init__(
        self,
        grid: HoneycombGrid,
        nectar_flow: NectarFlow,
        config: Optional[SwarmConfig] = None
    ):
        self.grid = grid
        self.nectar = nectar_flow
        self.config = config or SwarmConfig()
        
        # Cola de tareas (heap por prioridad)
        self._task_queue: List[HiveTask] = []
        self._task_index: Dict[str, HiveTask] = {}
        
        # Comportamientos por celda
        self._behaviors: Dict[HexCoord, BeeBehavior] = {}
        
        # Balanceador
        self._balancer = SwarmBalancer(grid, self.config)
        
        # Estado
        self._tick_count = 0
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._lock = threading.RLock()
        
        # Inicializar comportamientos
        self._initialize_behaviors()
    
    def _initialize_behaviors(self) -> None:
        """Asigna comportamientos a las celdas según ratios configurados."""
        worker_cells = [
            cell for cell in self.grid._cells.values()
            if isinstance(cell, WorkerCell)
        ]
        
        n_total = len(worker_cells)
        n_foragers = int(n_total * self.config.foragers_ratio)
        n_nurses = int(n_total * self.config.nurses_ratio)
        n_scouts = int(n_total * self.config.scouts_ratio)
        # El resto son guardias
        
        random.shuffle(worker_cells)
        
        for i, cell in enumerate(worker_cells):
            if i < n_foragers:
                self._behaviors[cell.coord] = ForagerBehavior(cell, self.nectar)
            elif i < n_foragers + n_nurses:
                self._behaviors[cell.coord] = NurseBehavior(cell, self.nectar)
            elif i < n_foragers + n_nurses + n_scouts:
                self._behaviors[cell.coord] = ScoutBehavior(cell, self.nectar)
            else:
                self._behaviors[cell.coord] = GuardBehavior(cell, self.nectar)
    
    def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        target_cell: Optional[HexCoord] = None,
        timeout: float = 30.0,
        callback: Optional[Callable] = None
    ) -> HiveTask:
        """
        Envía una tarea al scheduler.
        
        Args:
            task_type: Tipo de tarea
            payload: Datos de la tarea
            priority: Prioridad
            target_cell: Celda destino específica
            timeout: Timeout en segundos
            callback: Función a llamar al completar
            
        Returns:
            La tarea creada
        """
        task = HiveTask(
            priority=priority.value,
            task_type=task_type,
            target_cell=target_cell,
            payload=payload,
            timeout_seconds=timeout,
            callback=callback,
        )
        
        with self._lock:
            if len(self._task_queue) >= self.config.max_queue_size:
                raise RuntimeError("Task queue full")
            
            heapq.heappush(self._task_queue, task)
            self._task_index[task.task_id] = task
        
        logger.debug(f"Task submitted: {task.task_id} ({task_type})")
        return task
    
    def get_task(self, task_id: str) -> Optional[HiveTask]:
        """Obtiene una tarea por ID."""
        return self._task_index.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancela una tarea."""
        task = self._task_index.get(task_id)
        if task and task.state == TaskState.PENDING:
            task.state = TaskState.CANCELLED
            return True
        return False
    
    def tick(self) -> Dict[str, Any]:
        """
        Ejecuta un tick del scheduler.
        
        Returns:
            Estadísticas del tick
        """
        self._tick_count += 1
        
        results = {
            "tick": self._tick_count,
            "tasks_processed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "work_stolen": 0,
        }
        
        with self._lock:
            # Obtener tareas pendientes
            pending_tasks = [
                t for t in self._task_queue
                if t.state == TaskState.PENDING
            ]
            # Tareas ya asignadas en este tick (cada tarea solo se ejecuta una vez por tick)
            claimed_this_tick: Set[str] = set()

            # Distribuir tareas a comportamientos
            for coord, behavior in self._behaviors.items():
                # Filtrar tareas relevantes y no asignadas aún en este tick
                available = [
                    t for t in pending_tasks
                    if (t.target_cell is None or t.target_cell == coord)
                    and t.task_id not in claimed_this_tick
                ]

                # Seleccionar tarea
                task = behavior.select_task(available)
                if not task:
                    continue

                # Reservar para que ningún otro behavior la tome en este tick
                claimed_this_tick.add(task.task_id)

                # Ejecutar
                results["tasks_processed"] += 1
                success = behavior.execute_task(task)
                
                if success:
                    results["tasks_completed"] += 1
                    self._tasks_completed += 1
                    
                    # Callback si existe
                    if task.callback:
                        try:
                            task.callback(task.result)
                        except Exception as e:
                            logger.error(f"Task callback error: {e}")
                else:
                    # Reintentar o marcar como fallida
                    if task.can_retry():
                        task.state = TaskState.PENDING
                    else:
                        results["tasks_failed"] += 1
                        self._tasks_failed += 1
            
            # Limpiar tareas completadas/fallidas
            self._task_queue = [
                t for t in self._task_queue
                if t.state in (TaskState.PENDING, TaskState.RUNNING)
            ]
            heapq.heapify(self._task_queue)
        
        # Balancear carga
        if self._balancer.rebalance_if_needed(self._tick_count):
            results["work_stolen"] = self._balancer._migrations_performed
        
        return results
    
    def get_queue_size(self) -> int:
        """Retorna el tamaño de la cola."""
        return len(self._task_queue)
    
    def get_pending_count(self) -> int:
        """Retorna el número de tareas pendientes."""
        return sum(1 for t in self._task_queue if t.state == TaskState.PENDING)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del scheduler."""
        behavior_counts = defaultdict(int)
        for behavior in self._behaviors.values():
            behavior_counts[type(behavior).__name__] += 1
        
        return {
            "tick_count": self._tick_count,
            "queue_size": len(self._task_queue),
            "pending_tasks": self.get_pending_count(),
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "behaviors": dict(behavior_counts),
            "balancer": self._balancer.get_stats(),
        }
    
    def shutdown(self) -> None:
        """Apaga el scheduler limpiamente."""
        with self._lock:
            # Cancelar tareas pendientes
            for task in self._task_queue:
                if task.state == TaskState.PENDING:
                    task.state = TaskState.CANCELLED
            
            self._task_queue.clear()
            self._task_index.clear()
        
        logger.info("SwarmScheduler shutdown complete")
