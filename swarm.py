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

import asyncio
import heapq
import logging
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    ClassVar,
    Literal,
    TypeVar,
)

# Absolute import: state_machines is top-level, same rationale as
# core/cells_base.py (dual import paths `HOC.*` vs top-level via sys.path).
from state_machines.base import HocStateMachine
from state_machines.reified import transition
from state_machines.task_fsm import build_task_fsm

from .core import HexCoord, HoneycombCell, HoneycombGrid, WorkerCell
from .nectar import NectarFlow, PheromoneType
from .security import (
    RateLimitExceeded,
    sanitize_error,
    secure_random,
    secure_shuffle,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# SERIALIZATION HELPERS (Phase 7.10)
# ═══════════════════════════════════════════════════════════════════════════════
# Used by HiveTask.to_dict / SwarmScheduler.to_dict to keep the checkpoint
# blob free of non-portable references (callables, closures, custom
# objects). Anything that is not a primitive / nested primitive container
# is replaced with a small dict carrying the rejected type name so a
# future reader can audit what was dropped.


_UNSERIALIZABLE_KEY: str = "__hoc_unserializable__"


def _is_safe_value(v: Any) -> bool:
    """Return True if ``v`` is a primitive or a nested container of
    primitives that survives a mscs round-trip without registry tweaks.
    """
    if v is None or isinstance(v, (bool, int, float, str, bytes)):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_safe_value(item) for item in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _is_safe_value(val) for k, val in v.items())
    return False


def _safe_serialize_value(v: Any) -> Any:
    """Best-effort: return ``v`` if it survives a checkpoint round-trip,
    else a sentinel describing the rejected type."""
    if _is_safe_value(v):
        return v
    return {_UNSERIALIZABLE_KEY: type(v).__name__}


def _safe_serialize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply :func:`_safe_serialize_value` per key. Non-serializable
    entries are replaced with the sentinel marker; the rest pass
    through unchanged."""
    return {k: _safe_serialize_value(v) for k, v in payload.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# TAREAS
# ═══════════════════════════════════════════════════════════════════════════════


class TaskState(Enum):
    """Estado de una tarea.

    Phase 4.3: ``ASSIGNED`` was removed (B12-bis cleanup). It was
    declared in Phase 1 to model a ``PENDING -> ASSIGNED -> RUNNING``
    two-step but no production path ever assigned it -- workers go
    straight ``PENDING -> RUNNING`` when they claim a task. If Phase 5+
    needs visibility into the "claimed but not yet executing" interval
    (e.g. for metrics), reintroduce with a real wire-up.
    """

    PENDING = auto()  # Esperando ser tomada
    RUNNING = auto()  # En ejecución
    COMPLETED = auto()  # Completada exitosamente
    FAILED = auto()  # Fallida
    CANCELLED = auto()  # Cancelada


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
    target_cell: HexCoord | None = field(compare=False, default=None)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)
    state: TaskState = field(compare=False, default=TaskState.PENDING)
    assigned_to: HexCoord | None = field(compare=False, default=None)
    attempts: int = field(compare=False, default=0)
    max_attempts: int = field(compare=False, default=3)
    timeout_seconds: float = field(compare=False, default=30.0)
    callback: Callable[[Any], None] | None = field(compare=False, default=None)
    result: Any = field(compare=False, default=None)
    error: str | None = field(compare=False, default=None)

    def __post_init__(self) -> None:
        if not self.task_id:
            self.task_id = f"task_{id(self)}_{time.time():.0f}"
        # Phase 4.1: wire TaskLifecycle FSM into state mutations. The
        # dataclass __init__ runs before __post_init__, so the initial
        # ``state`` field assignment has already landed via __setattr__
        # without the FSM being present (see __setattr__ guard).
        self._fsm: HocStateMachine = build_task_fsm()
        # If the caller passed a non-default state through __init__ (rare;
        # tests do this to skip the lifecycle), seed the FSM to match.
        # reset() bypasses guards — it's a configuration call, not a
        # transition.
        if self.state is not TaskState.PENDING:
            self._fsm.reset(self.state.name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Phase 4.1: validate ``state`` mutations against the FSM.
        #
        # Guard on ``_fsm`` being present in __dict__ rather than using
        # ``hasattr`` (which is truthy during getattr-descend) so that the
        # dataclass-generated __init__ passes through unchecked — at that
        # point __post_init__ has not yet attached the FSM.
        if name == "state" and "_fsm" in self.__dict__:
            current = self.__dict__.get("state")
            # Allow idempotent re-assignment (``task.state = task.state``)
            # without a transition. Identity check is sufficient because
            # TaskState is an Enum with singleton members.
            if current is not value:
                self._fsm.transition_to(value.name)
        super().__setattr__(name, value)

    def is_expired(self) -> bool:
        """Verifica si la tarea expiró."""
        if self.state == TaskState.RUNNING:
            elapsed = time.time() - self.created_at
            return elapsed > self.timeout_seconds
        return False

    def can_retry(self) -> bool:
        """Verifica si se puede reintentar."""
        return self.attempts < self.max_attempts

    # ── Phase 7.10: checkpoint serialization ──────────────────────────
    # Sentinel marking that a task's callback was attached pre-checkpoint
    # but cannot be carried across the snapshot boundary. Restored tasks
    # inherit ``callback=None``; consumers inspect
    # ``task.callback_needs_reattach`` to decide whether to rebind.
    SENTINEL_CALLBACK_REATTACH: str = "<callback-reattach-required>"

    def to_dict(self) -> dict[str, Any]:
        """Phase 7.10: serialize this task for the checkpoint blob.

        Callables (``callback``, lambdas inside ``payload``) cannot
        survive a checkpoint round-trip — they're replaced with a
        sentinel marker so a future operator can identify what needs
        rebinding. Primitive fields and HexCoord are preserved.
        """
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "priority": self.priority,
            "created_at": self.created_at,
            "target_cell": self.target_cell.to_dict() if self.target_cell else None,
            "payload": _safe_serialize_payload(self.payload),
            "state": self.state.name,
            "assigned_to": self.assigned_to.to_dict() if self.assigned_to else None,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "timeout_seconds": self.timeout_seconds,
            "callback": self.SENTINEL_CALLBACK_REATTACH if self.callback is not None else None,
            "result": _safe_serialize_value(self.result),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HiveTask:
        """Phase 7.10: rebuild a HiveTask from its checkpoint dict.

        ``callback`` is reset to ``None`` regardless of whether the
        original task had one — the sentinel only signals that the
        original task carried a callback. Consumers can detect this by
        reading ``task.callback_needs_reattach`` on the restored task.
        """
        target_cell = HexCoord.from_dict(data["target_cell"]) if data.get("target_cell") else None
        assigned_to = HexCoord.from_dict(data["assigned_to"]) if data.get("assigned_to") else None
        state = TaskState[data["state"]]
        task = cls(
            priority=data["priority"],
            created_at=data["created_at"],
            task_id=data["task_id"],
            task_type=data["task_type"],
            target_cell=target_cell,
            payload=data.get("payload", {}),
            state=state,
            assigned_to=assigned_to,
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
            timeout_seconds=data.get("timeout_seconds", 30.0),
            callback=None,
            result=data.get("result"),
            error=data.get("error"),
        )
        # ``object.__setattr__`` to bypass the dataclass's tightened
        # __setattr__ — the flag is metadata, not a real field.
        object.__setattr__(
            task,
            "callback_needs_reattach",
            data.get("callback") == cls.SENTINEL_CALLBACK_REATTACH,
        )
        return task

    # ── Phase 4.2: reified transitions (additive API) ─────────────────
    # These methods are functionally equivalent to ``self.state = X`` —
    # all call-sites in swarm.py continue to use direct mutation. The
    # reified API exists so callers can write ``task.claim(worker)``
    # instead of ``task.state = TaskState.RUNNING; task.assigned_to = ...``,
    # which reads more naturally and pins the lifecycle in HiveTask
    # itself (not scattered across SwarmScheduler methods).

    @transition(from_=TaskState.PENDING, to=TaskState.RUNNING)
    def claim(self, worker: HoneycombCell) -> None:
        """Worker takes ownership of this task and begins execution."""
        self.assigned_to = worker.coord

    @transition(from_=TaskState.RUNNING, to=TaskState.COMPLETED)
    def complete(self, result: Any = None) -> None:
        """Mark this task complete. ``result`` is stored on the task."""
        self.result = result

    @transition(from_=TaskState.RUNNING, to=TaskState.FAILED)
    def fail(self, error: str) -> None:
        """Mark this task failed. ``error`` is stored on the task."""
        self.error = error
        self.attempts += 1

    @transition(from_=TaskState.FAILED, to=TaskState.PENDING)
    def retry(self) -> None:
        """Return a failed task to the queue for another attempt.
        Caller is responsible for checking :meth:`can_retry` first."""
        self.assigned_to = None
        self.error = None


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

    entity_ids: list[str]
    operation: str
    params: dict[str, Any]


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
        self.experience: dict[str, float] = defaultdict(float)
        self.response_threshold: float = 0.5
        self._last_task: HiveTask | None = None
        self._success_streak: int = 0

    @abstractmethod
    def select_task(self, available_tasks: list[HiveTask]) -> HiveTask | None:
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

        Phase 2: usamos ``secrets.SystemRandom`` (CSPRNG) para que un atacante
        con conocimiento del seed global de ``random`` no pueda predecir qué
        tareas tomará una celda. Aunque la secuencia no necesita ser segura
        criptográficamente *por sí misma*, esta función decide qué trabajo
        acepta una abeja — manipular esa decisión es un vector de carga/
        denegación. CSPRNG es barato (μs por llamada) y cierra el vector.
        """
        if stimulus <= 0:
            return False

        prob = (stimulus**2) / (stimulus**2 + self.response_threshold**2)
        return secure_random() < prob

    def deposit_success_pheromone(self, task: HiveTask) -> None:
        """Deposita feromona de éxito después de completar tarea."""
        self.nectar.deposit_pheromone(
            self.cell.coord,
            PheromoneType.SUCCESS,
            0.5 + (self._success_streak * 0.1),
            metadata={"task_type": task.task_type},
        )

    def deposit_failure_pheromone(self, task: HiveTask) -> None:
        """Deposita feromona de fallo."""
        self.nectar.deposit_pheromone(
            self.cell.coord, PheromoneType.FAILURE, 0.3, metadata={"task_type": task.task_type}
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
        self.specialization: str | None = None
        self.recruitment_threshold: float = 0.7

    def select_task(self, available_tasks: list[HiveTask]) -> HiveTask | None:
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
            pheromone_score: float = 0.0
            if task.target_cell:
                pheromone_score = (
                    self.nectar.sense_pheromone(task.target_cell, PheromoneType.SUCCESS) * 0.5
                )

            # Penalización por feromonas de fallo
            failure_penalty = (
                self.nectar.sense_pheromone(self.cell.coord, PheromoneType.FAILURE) * 0.3
            )

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
            metadata={"specialization": self.specialization},
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
        self.incubating: list[Any] = []
        self.warmup_ticks: int = 3

    def select_task(self, available_tasks: list[HiveTask]) -> HiveTask | None:
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
                self.incubating.append(
                    {
                        "spec": entity_spec,
                        "ticks_remaining": self.warmup_ticks,
                        "task": task,
                    }
                )
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

    def tick_incubation(self) -> list[Any]:
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
        self.explored: set[HexCoord] = set()
        self.exploration_radius: int = 5

    def select_task(self, available_tasks: list[HiveTask]) -> HiveTask | None:
        # Scouts prefieren tareas de exploración
        explore_tasks = [t for t in available_tasks if t.task_type == "explore"]
        if explore_tasks:
            return explore_tasks[0]

        # También aceptan tareas en celdas lejanas
        distant_tasks = [
            t
            for t in available_tasks
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

    def _explore_area(self, center: HexCoord) -> dict[str, Any]:
        """Explora un área y reporta hallazgos."""
        report: dict[str, Any] = {
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
        self.blocked_sources: set[HexCoord] = set()
        self.validation_rules: list[Callable[[HiveTask], bool]] = []

    def select_task(self, available_tasks: list[HiveTask]) -> HiveTask | None:
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
        return all(rule(task) for rule in self.validation_rules)

    def add_validation_rule(self, rule: Callable[[HiveTask], bool]) -> None:
        """Añade una regla de validación."""
        self.validation_rules.append(rule)


# ═══════════════════════════════════════════════════════════════════════════════
# POLÍTICAS DE SCHEDULING
# ═══════════════════════════════════════════════════════════════════════════════


class SwarmPolicy(Enum):
    """Políticas de scheduling del enjambre."""

    PRIORITY = auto()  # Siempre la de mayor prioridad
    ROUND_ROBIN = auto()  # Rotación equitativa
    PHEROMONE_GUIDED = auto()  # Guiado por feromonas
    RANDOM = auto()  # Aleatorio
    LEAST_LOADED = auto()  # Celda menos cargada
    LOCALITY = auto()  # Preferir celdas cercanas


@dataclass
class SwarmConfig:
    """Configuración del scheduler de enjambre."""

    # Políticas
    default_policy: SwarmPolicy = SwarmPolicy.PHEROMONE_GUIDED

    # Workers
    foragers_ratio: float = 0.6  # 60% recolectoras
    nurses_ratio: float = 0.15  # 15% nodrizas
    scouts_ratio: float = 0.15  # 15% exploradoras
    guards_ratio: float = 0.1  # 10% guardias

    # Tareas
    max_queue_size: int = 10000
    task_timeout_seconds: float = 30.0
    max_task_retries: int = 3

    # Phase 7.3: backpressure policy when ``max_queue_size`` is hit.
    # ``raise`` is the pre-Phase-7 behaviour and remains the default for
    # safety: callers see a hard error and can decide how to react.
    # ``drop_oldest`` evicts the lowest-priority task to make room for the
    # newcomer (highest-priority new tasks displace stale background work).
    # ``drop_newest`` rejects the new task silently (caller gets a sentinel
    # ``HiveTask`` with ``state=CANCELLED``).
    # ``block`` polls every ``queue_full_block_poll_s`` seconds for up to
    # ``queue_full_block_timeout_s`` waiting for room; only meaningful for
    # the sync path (the async path will get a true asyncio.Queue in a
    # future Phase 7.x).
    queue_full_policy: Literal["raise", "drop_oldest", "drop_newest", "block"] = "raise"
    queue_full_block_timeout_s: float = 5.0
    queue_full_block_poll_s: float = 0.005

    # Phase 2: rate limiting para cerrar vectores de DoS contra el API público.
    # `submit_rate_per_second` permite ráfagas iniciales via `burst` — default
    # burst = 2× rate para no romper cargas normales.
    submit_rate_per_second: float = 1000.0
    submit_rate_burst: int = 2000
    execute_rate_per_second: float = 10000.0
    execute_rate_burst: int = 20000

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

    def __init__(self) -> None:
        self.cell_loads: dict[HexCoord, float] = {}
        self.ring_loads: dict[int, float] = {}
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
        ring_counts: defaultdict[int, int] = defaultdict(int)
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
        return sum((load - avg) ** 2 for load in self.cell_loads.values()) / len(self.cell_loads)


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

    def find_overloaded_cells(self) -> list[HexCoord]:
        """Encuentra celdas sobrecargadas."""
        return [
            coord
            for coord, load in self._distribution.cell_loads.items()
            if load >= self.config.load_threshold_high
        ]

    def find_underloaded_cells(self) -> list[HexCoord]:
        """Encuentra celdas subcargadas."""
        return [
            coord
            for coord, load in self._distribution.cell_loads.items()
            if load <= self.config.load_threshold_low
        ]

    def suggest_migrations(self) -> list[tuple[HexCoord, HexCoord, int]]:
        """
        Sugiere migraciones de trabajo.

        Returns:
            Lista de (origen, destino, cantidad) para migrar
        """
        suggestions: list[tuple[HexCoord, HexCoord, int]] = []

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
                    self.config.steal_batch_size,
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
            best_load: float = 0.0

            for neighbor in cell.get_all_neighbors():
                if (
                    isinstance(neighbor, WorkerCell)
                    and neighbor.load > self.config.load_threshold_high
                ) and neighbor.load > best_load:
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

    def get_stats(self) -> dict[str, Any]:
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
# BEHAVIOR INDEX (Phase 7.5 — O(n·m) → O(m·log n))
# ═══════════════════════════════════════════════════════════════════════════════


class BehaviorIndex:
    """Per-behavior priority-ordered task index.

    Replaces the O(n·m) filter loop in :meth:`SwarmScheduler.tick` with
    O(m·log n) heap operations: each registered behavior has its own
    min-heap, and submit_task inserts the task only into heaps of
    behaviors that would actually consider it (type-routing).

    The API is intentionally minimal — three operations cover the
    scheduler's needs:

    - :meth:`insert` adds a task to the heap of one behavior. The
      scheduler calls this once per matching behavior at submit time.
    - :meth:`pop_best` returns and *claims* the highest-priority active
      task for a behavior. Claim is enforced via a tombstone set so
      other behaviors' heaps lazily skip the entry on their next pop.
    - :meth:`remove` tombstones a task by id (used for cancel /
      complete / fail-no-retry).

    The tombstone set is bounded by :meth:`compact`, which the
    scheduler invokes every N ticks (configurable). Without compaction,
    tombstones accumulate across ticks but pop_best stays correct via
    lazy deletion — compaction is purely for memory pressure.

    Heap entry layout: ``(priority, sequence, task_id, task)``. The
    ``sequence`` counter breaks ties FIFO. Note that ``task.priority``
    follows the existing convention (``CRITICAL=0``, ``BACKGROUND=4``)
    so a min-heap returns the highest-importance task first without
    inversion.
    """

    __slots__ = ("_counter", "_heaps", "_tombstoned")

    def __init__(self) -> None:
        self._heaps: dict[BeeBehavior, list[tuple[int, int, str, HiveTask]]] = {}
        self._tombstoned: set[str] = set()
        self._counter: int = 0

    def register_behavior(self, behavior: BeeBehavior) -> None:
        """Allocate an empty heap for ``behavior``. Idempotent."""
        self._heaps.setdefault(behavior, [])

    def unregister_behavior(self, behavior: BeeBehavior) -> None:
        """Drop ``behavior``'s heap. Pending tasks still live in the
        scheduler's main queue; only the index reference is gone."""
        self._heaps.pop(behavior, None)

    def insert(self, task: HiveTask, behavior: BeeBehavior) -> None:
        """Add ``task`` to ``behavior``'s heap.

        Re-inserting a previously tombstoned task (the FAILED → PENDING
        retry path) clears the tombstone so the new entry is visible.
        Behaviors that haven't been registered are auto-registered with
        an empty heap; this lets the scheduler add behaviors lazily
        without an explicit register step.
        """
        if behavior not in self._heaps:
            self._heaps[behavior] = []
        self._tombstoned.discard(task.task_id)
        self._counter += 1
        heapq.heappush(
            self._heaps[behavior],
            (task.priority, self._counter, task.task_id, task),
        )

    def pop_best(self, behavior: BeeBehavior) -> HiveTask | None:
        """Return and claim the highest-priority active task for
        ``behavior``. Returns ``None`` if the heap is empty (or all
        entries are tombstoned). Lazy-cleans tombstoned entries on
        the way down."""
        heap = self._heaps.get(behavior)
        if heap is None:
            return None
        while heap:
            top_id = heap[0][2]
            if top_id in self._tombstoned:
                heapq.heappop(heap)
                continue
            entry = heapq.heappop(heap)
            self._tombstoned.add(top_id)
            return entry[3]
        return None

    def remove(self, task_id: str) -> bool:
        """Tombstone ``task_id`` so subsequent pop_best calls skip it.

        Returns ``True`` if the task was active, ``False`` if it was
        already tombstoned (lets callers detect double-removes
        cheaply)."""
        if task_id in self._tombstoned:
            return False
        self._tombstoned.add(task_id)
        return True

    def compact(self) -> int:
        """Filter tombstoned entries from all heaps and clear the
        tombstone set. Returns the count of pruned heap entries.

        Without compaction tombstones accumulate as dead heap entries
        that pop_best skips; compaction reclaims that space. Cheap
        relative to the savings of the new tick path — the scheduler
        runs it every 10 ticks (one compact per ~10× insert+pop
        rounds)."""
        if not self._tombstoned:
            return 0
        removed_total = 0
        for heap in self._heaps.values():
            before = len(heap)
            heap[:] = [e for e in heap if e[2] not in self._tombstoned]
            heapq.heapify(heap)
            removed_total += before - len(heap)
        self._tombstoned.clear()
        return removed_total

    def size_for(self, behavior: BeeBehavior) -> int:
        """Number of active (non-tombstoned) entries in ``behavior``'s
        heap. Counts each task once even if it appears multiple times
        — duplicates are an edge case (e.g. retry re-insert before
        compaction); the live count is what matters for callers."""
        heap = self._heaps.get(behavior)
        if heap is None:
            return 0
        seen: set[str] = set()
        for entry in heap:
            tid = entry[2]
            if tid in self._tombstoned or tid in seen:
                continue
            seen.add(tid)
        return len(seen)


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

    # Phase 7.5: how often to compact the BehaviorIndex's tombstone set.
    # Larger values trade memory for fewer compaction passes; 10 ticks
    # keeps compaction overhead well below the win from O(log n) pops.
    INDEX_COMPACT_INTERVAL_TICKS: int = 10

    def __init__(
        self, grid: HoneycombGrid, nectar_flow: NectarFlow, config: SwarmConfig | None = None
    ):
        self.grid = grid
        self.nectar = nectar_flow
        self.config = config or SwarmConfig()

        # Cola de tareas (heap por prioridad)
        self._task_queue: list[HiveTask] = []
        self._task_index: dict[str, HiveTask] = {}

        # Comportamientos por celda
        self._behaviors: dict[HexCoord, BeeBehavior] = {}

        # Phase 7.5: secondary index (per-behavior priority heap) that
        # the tick() loop consults instead of filtering pending_tasks
        # for every behavior.
        self._behavior_index: BehaviorIndex = BehaviorIndex()

        # Balanceador
        self._balancer = SwarmBalancer(grid, self.config)

        # Estado
        self._tick_count = 0
        self._tasks_completed = 0
        self._tasks_failed = 0
        # Phase 7.3: cumulative count of tasks dropped at submit time
        # because the queue was full and the policy decided to evict
        # rather than raise. Exposed via ``get_stats()`` so callers can
        # detect backpressure.
        self._tasks_dropped = 0
        self._lock = threading.RLock()

        # Phase 2: rate limiting de APIs públicas para mitigar DoS.
        # Instanciamos los limitadores aquí para leer config en runtime.
        from .security import RateLimiter as _RateLimiter

        self._submit_limiter = _RateLimiter(
            per_second=self.config.submit_rate_per_second,
            burst=self.config.submit_rate_burst,
        )
        self._execute_limiter = _RateLimiter(
            per_second=self.config.execute_rate_per_second,
            burst=self.config.execute_rate_burst,
        )

        # Inicializar comportamientos
        self._initialize_behaviors()

    def _initialize_behaviors(self) -> None:
        """Asigna comportamientos a las celdas según ratios configurados."""
        worker_cells = [cell for cell in self.grid._cells.values() if isinstance(cell, WorkerCell)]

        n_total = len(worker_cells)
        n_foragers = int(n_total * self.config.foragers_ratio)
        n_nurses = int(n_total * self.config.nurses_ratio)
        n_scouts = int(n_total * self.config.scouts_ratio)
        # El resto son guardias

        # Phase 2: CSPRNG shuffle para que la asignación inicial de roles
        # no sea predecible desde fuera.
        secure_shuffle(worker_cells)

        for i, cell in enumerate(worker_cells):
            if i < n_foragers:
                self._behaviors[cell.coord] = ForagerBehavior(cell, self.nectar)
            elif i < n_foragers + n_nurses:
                self._behaviors[cell.coord] = NurseBehavior(cell, self.nectar)
            elif i < n_foragers + n_nurses + n_scouts:
                self._behaviors[cell.coord] = ScoutBehavior(cell, self.nectar)
            else:
                self._behaviors[cell.coord] = GuardBehavior(cell, self.nectar)

        # Phase 7.5: register every behavior with the secondary index
        # so submit_task can route into per-behavior heaps.
        for behavior in self._behaviors.values():
            self._behavior_index.register_behavior(behavior)

    # ── Phase 7.5: type-routing helper for BehaviorIndex ──────────────
    # Decides which registered behaviors a freshly-submitted task should
    # be inserted under. Mirrors the implicit filter from each
    # behavior's :meth:`select_task` so the index never offers a task
    # the behavior would refuse anyway. Routing rules:
    # - ``target_cell`` is set: ONLY the behavior at that exact coord
    #   (if any), and only if that behavior accepts the task_type.
    # - ``target_cell`` is None: every registered behavior whose
    #   ``select_task`` would accept the task_type.

    _NURSE_TYPES: frozenset[str] = frozenset({"spawn", "warmup"})
    _SCOUT_TYPES: frozenset[str] = frozenset({"explore"})
    _GUARD_TYPES: frozenset[str] = frozenset({"validate"})

    @classmethod
    def _behavior_accepts_type(cls, behavior: BeeBehavior, task_type: str) -> bool:
        """Return True if ``behavior``'s select_task would consider a
        task of ``task_type`` (target_cell match handled separately)."""
        if isinstance(behavior, NurseBehavior):
            return task_type in cls._NURSE_TYPES
        if isinstance(behavior, ScoutBehavior):
            # Scouts also accept distant-target tasks but those carry a
            # target_cell and follow the pinned-routing branch above —
            # at this point we only see global tasks, so type alone.
            return task_type in cls._SCOUT_TYPES
        if isinstance(behavior, GuardBehavior):
            return task_type in cls._GUARD_TYPES
        # ForagerBehavior is the catch-all. It takes any task type
        # except those that another behavior class specializes in.
        if isinstance(behavior, ForagerBehavior):
            return task_type not in (cls._NURSE_TYPES | cls._SCOUT_TYPES | cls._GUARD_TYPES)
        return False

    def _route_task_to_behaviors(self, task: HiveTask) -> list[BeeBehavior]:
        """Phase 7.5: decide which behaviors should receive ``task`` in
        their BehaviorIndex heap."""
        if task.target_cell is not None:
            pinned = self._behaviors.get(task.target_cell)
            if pinned is None:
                return []
            return [pinned] if self._behavior_accepts_type(pinned, task.task_type) else []
        return [
            b for b in self._behaviors.values() if self._behavior_accepts_type(b, task.task_type)
        ]

    def submit_task(
        self,
        task_type: str,
        payload: dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        target_cell: HexCoord | None = None,
        timeout: float = 30.0,
        callback: Callable[[Any], None] | None = None,
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
        # Phase 2: rate limiting. Rechaza submits si el bucket está vacío
        # para impedir que un cliente agotó la cola por flooding.
        if not self._submit_limiter.try_acquire():
            raise RateLimitExceeded(
                f"submit_task rate limit exceeded "
                f"({self.config.submit_rate_per_second}/s, burst={self.config.submit_rate_burst})"
            )

        task = HiveTask(
            priority=priority.value,
            task_type=task_type,
            target_cell=target_cell,
            payload=payload,
            timeout_seconds=timeout,
            callback=callback,
        )

        # Phase 7.3: backpressure policy dispatch happens BEFORE the
        # heap push so the policy decides whether the new task lands.
        if not self._reserve_queue_slot(task):
            # Drop policy returned False → don't enqueue this task. The
            # caller still gets a HiveTask back (state=CANCELLED) so
            # signatures stay compatible with the pre-Phase-7.3 contract.
            task.state = TaskState.CANCELLED
            return task

        with self._lock:
            heapq.heappush(self._task_queue, task)
            self._task_index[task.task_id] = task

            # Phase 7.5: insert into the BehaviorIndex so tick() can
            # pop in O(log n) instead of filtering pending_tasks for
            # every behavior.
            for behavior in self._route_task_to_behaviors(task):
                self._behavior_index.insert(task, behavior)

        logger.debug(f"Task submitted: {task.task_id} ({task_type})")
        return task

    def _reserve_queue_slot(self, incoming: HiveTask) -> bool:
        """Phase 7.3: enforce ``queue_full_policy`` and return whether
        ``incoming`` should be enqueued. Increments
        ``_tasks_dropped`` when the policy evicts an existing task or
        rejects the new one.

        Policy semantics:

        - ``"raise"`` — preserves pre-Phase-7.3 behaviour: hard error
          when full.
        - ``"drop_oldest"`` — find the **lowest-priority** task (largest
          priority value, hence the worst by HOC's CRITICAL=0
          convention) and evict it. The new task is enqueued; the
          dropped task is marked ``CANCELLED``.
        - ``"drop_newest"`` — return ``False``; caller marks the new
          task ``CANCELLED`` and skips the heap push.
        - ``"block"`` — poll until room frees up or
          ``queue_full_block_timeout_s`` elapses; raises
          :class:`RuntimeError` on timeout.
        """
        max_size = self.config.max_queue_size
        policy = self.config.queue_full_policy

        with self._lock:
            if len(self._task_queue) < max_size:
                return True

            if policy == "raise":
                raise RuntimeError("Task queue full")

            if policy == "drop_oldest":
                # The heap orders by priority ascending (CRITICAL=0 is
                # smallest, BACKGROUND=4 is largest). Lowest-priority
                # task = largest priority value = worst candidate to
                # keep. Linear scan is O(n) but Phase 7.3 trades that
                # for the simpler invariant: heapq has no built-in
                # "remove worst" op.
                worst_idx = max(
                    range(len(self._task_queue)),
                    key=lambda i: (
                        self._task_queue[i].priority,
                        -self._task_queue[i].created_at,
                    ),
                )
                dropped = self._task_queue.pop(worst_idx)
                heapq.heapify(self._task_queue)
                dropped.state = TaskState.CANCELLED
                self._task_index.pop(dropped.task_id, None)
                self._behavior_index.remove(dropped.task_id)
                self._tasks_dropped += 1
                return True

            if policy == "drop_newest":
                self._tasks_dropped += 1
                return False

            # policy == "block"
            deadline = time.time() + self.config.queue_full_block_timeout_s
            poll = self.config.queue_full_block_poll_s

        # Lock dropped for the polling sleep. Re-acquire on each pass
        # to check whether room freed up.
        if policy == "block":
            while time.time() < deadline:
                time.sleep(poll)
                with self._lock:
                    if len(self._task_queue) < max_size:
                        return True
            raise RuntimeError(
                "Task queue full; block policy timed out after "
                f"{self.config.queue_full_block_timeout_s}s"
            )

        # Unknown policy — should be caught by Literal type at edit time.
        raise RuntimeError(f"unknown queue_full_policy: {policy!r}")

    def execute_on_cell(self, coord: HexCoord, task: HiveTask) -> bool:
        """
        Ejecuta una tarea directamente en una celda por su comportamiento
        asociado. Phase 2: rate-limited para cerrar el vector de "ejecución
        forzada" desde caller no-confiable.

        Returns:
            True si la ejecución fue exitosa.

        Raises:
            RateLimitExceeded: si se supera el ritmo permitido.
            KeyError: si no hay behavior registrado para la celda.
        """
        if not self._execute_limiter.try_acquire():
            raise RateLimitExceeded(
                f"execute_on_cell rate limit exceeded "
                f"({self.config.execute_rate_per_second}/s, burst={self.config.execute_rate_burst})"
            )
        behavior = self._behaviors.get(coord)
        if behavior is None:
            raise KeyError(f"No hay behavior registrado para la celda {coord}")
        return behavior.execute_task(task)

    def get_task(self, task_id: str) -> HiveTask | None:
        """Obtiene una tarea por ID."""
        return self._task_index.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancela una tarea."""
        task = self._task_index.get(task_id)
        if task and task.state == TaskState.PENDING:
            task.state = TaskState.CANCELLED
            # Phase 7.5: tombstone in BehaviorIndex too so a tick that
            # races cancel doesn't pop the cancelled task.
            self._behavior_index.remove(task_id)
            return True
        return False

    # Phase 7.2: one-shot guard so ``run_tick_sync`` emits its
    # DeprecationWarning only the first time it's called per process.
    _SYNC_DEPRECATION_EMITTED: ClassVar[bool] = False

    async def tick(self) -> dict[str, Any]:
        """Phase 7.1: async tick.

        Body dispatched to ``asyncio.to_thread`` so the existing
        :class:`threading.RLock` in :class:`SwarmScheduler` works
        without async re-plumbing. The hot path
        (:class:`BehaviorIndex.pop_best` + ``execute_task``) is
        CPU-light, so the to_thread hop is essentially free; the win
        from async lives at the grid level (concurrent
        :meth:`HoneycombCell.execute_tick` calls).
        """
        return await asyncio.to_thread(self._sync_tick)

    def _sync_tick(self) -> dict[str, Any]:
        """
        Ejecuta un tick del scheduler.

        Phase 7.5: replaces the O(n·m) filter loop (every behavior
        scans every pending task) with O(m·log n) pop_best calls
        against :class:`BehaviorIndex`. Behaviors still see
        ``select_task([candidate])`` for the probabilistic refusal
        path — refused tasks are re-inserted into the index so the
        next tick can offer them again.

        Phase 7.1: renamed from ``tick`` to ``_sync_tick``; the
        public :meth:`tick` is now an async wrapper.

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
            # Phase 7.5: per-behavior pop_best instead of filter+select.
            for _coord, behavior in self._behaviors.items():
                task = self._behavior_index.pop_best(behavior)
                if task is None:
                    continue

                # Pre-Phase-7.5 the tick() filter explicitly required
                # ``state == PENDING``. Test fixtures and external
                # mutators can flip a queued task to a terminal state
                # without going through the scheduler — pop_best is
                # unaware. Drop those silently here; they get pruned by
                # the cleanup loop below.
                if task.state is not TaskState.PENDING:
                    continue

                # Preserve the probabilistic-refusal contract from
                # :meth:`BeeBehavior.select_task`. If the behavior
                # declines the offered task, re-insert (which clears
                # the tombstone) so the next tick can retry.
                selected = behavior.select_task([task])
                if selected is None:
                    self._behavior_index.insert(task, behavior)
                    continue

                # Ejecutar
                results["tasks_processed"] += 1
                success = behavior.execute_task(selected)

                if success:
                    results["tasks_completed"] += 1
                    self._tasks_completed += 1

                    # Callback si existe
                    if selected.callback:
                        try:
                            selected.callback(selected.result)
                        except Exception as e:
                            logger.error(f"Task callback error: {sanitize_error(e)}")
                else:
                    # Reintentar o marcar como fallida
                    if selected.can_retry():
                        selected.state = TaskState.PENDING
                        # Re-insert into the index so the retry is
                        # visible on the next tick.
                        for b in self._route_task_to_behaviors(selected):
                            self._behavior_index.insert(selected, b)
                    else:
                        results["tasks_failed"] += 1
                        self._tasks_failed += 1

            # Phase 1 fix (B2.5): limpiar también ``_task_index`` además de
            # ``_task_queue``. Antes el índice crecía sin cota porque las tareas
            # COMPLETED/FAILED/CANCELLED nunca se removían — leak silencioso de
            # memoria proporcional al throughput total a lo largo de la vida del
            # scheduler.
            for t in self._task_queue:
                if t.state not in (TaskState.PENDING, TaskState.RUNNING):
                    self._task_index.pop(t.task_id, None)
                    # Phase 7.5: also tombstone in the BehaviorIndex so
                    # cancelled / completed / non-retryable failed
                    # tasks can't surface from a heap.
                    self._behavior_index.remove(t.task_id)

            self._task_queue = [
                t for t in self._task_queue if t.state in (TaskState.PENDING, TaskState.RUNNING)
            ]
            heapq.heapify(self._task_queue)

            # Phase 7.5: bound tombstone memory by compacting heaps
            # every INDEX_COMPACT_INTERVAL_TICKS. Cheap relative to the
            # tick savings; pop_best stays correct between compacts via
            # lazy deletion.
            if self._tick_count % self.INDEX_COMPACT_INTERVAL_TICKS == 0:
                self._behavior_index.compact()

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

    def get_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas completas del scheduler."""
        behavior_counts: defaultdict[str, int] = defaultdict(int)
        for behavior in self._behaviors.values():
            behavior_counts[type(behavior).__name__] += 1

        return {
            "tick_count": self._tick_count,
            "queue_size": len(self._task_queue),
            "pending_tasks": self.get_pending_count(),
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            # Phase 7.3: cumulative drops attributable to the
            # ``queue_full_policy``. 0 under the default ``"raise"``
            # policy (which simply raises).
            "tasks_dropped": self._tasks_dropped,
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

    def run_tick_sync(self) -> dict[str, Any]:
        """Phase 7.2: blocking wrapper for legacy sync callers.

        Equivalent to ``asyncio.run(self.tick())``. Emits
        :class:`DeprecationWarning` once per process. Raises
        ``RuntimeError`` if called from inside a running event loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "SwarmScheduler.run_tick_sync called from a running event "
                "loop; use 'await scheduler.tick()' instead."
            )
        if not SwarmScheduler._SYNC_DEPRECATION_EMITTED:
            warnings.warn(
                "SwarmScheduler.run_tick_sync is a v1→v2 migration aid; "
                "switch callers to 'await scheduler.tick()'. This wrapper "
                "will be removed in HOC v3.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            SwarmScheduler._SYNC_DEPRECATION_EMITTED = True
        return self._sync_tick()

    # ── Phase 7.10: checkpoint serialization ──────────────────────────

    SERIALIZATION_VERSION: str = "1.0"
    """Schema version of the SwarmScheduler.to_dict payload. Independent
    of the checkpoint blob version (storage/checkpoint.py) — the blob
    version describes the wire frame, this one describes the inner
    scheduler dict shape."""

    def to_dict(self) -> dict[str, Any]:
        """Phase 7.10: serialize the scheduler's task queue + counters.

        Behaviors are *not* serialized. They're rebuilt on restore from
        the grid + config (the role assignment is randomized, but the
        ratio reproduces the original distribution within ±1 cell).
        Same for the balancer state (load distribution is recomputed
        on the next ``rebalance_if_needed`` call).
        """
        with self._lock:
            return {
                "version": self.SERIALIZATION_VERSION,
                "tick_count": self._tick_count,
                "tasks_completed": self._tasks_completed,
                "tasks_failed": self._tasks_failed,
                # Phase 7.3: include drops counter in the snapshot so
                # restored schedulers preserve their backpressure
                # history.
                "tasks_dropped": self._tasks_dropped,
                "queue": [t.to_dict() for t in self._task_queue],
            }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        grid: HoneycombGrid,
        nectar_flow: NectarFlow,
        config: SwarmConfig | None = None,
    ) -> SwarmScheduler:
        """Phase 7.10: rebuild a scheduler from a previously serialized
        state. Behaviors and balancer are freshly initialized from the
        provided ``grid`` + ``config``; the queue + index + counters
        come from ``data``."""
        scheduler = cls(grid, nectar_flow, config)
        scheduler._tick_count = data.get("tick_count", 0)
        scheduler._tasks_completed = data.get("tasks_completed", 0)
        scheduler._tasks_failed = data.get("tasks_failed", 0)
        # Phase 7.3: restore drops counter (defaults to 0 for v1 blobs
        # written before the field existed).
        scheduler._tasks_dropped = data.get("tasks_dropped", 0)

        with scheduler._lock:
            scheduler._task_queue = []
            scheduler._task_index = {}
            for task_dict in data.get("queue", []):
                task = HiveTask.from_dict(task_dict)
                heapq.heappush(scheduler._task_queue, task)
                scheduler._task_index[task.task_id] = task
                # Phase 7.5: re-populate the BehaviorIndex so the
                # restored scheduler's tick() finds tasks in O(log n).
                # Only PENDING tasks are eligible for selection — RUNNING
                # tasks won't be picked by tick() until they finish, so
                # we still index them in case the worker re-queues them.
                for behavior in scheduler._route_task_to_behaviors(task):
                    scheduler._behavior_index.insert(task, behavior)

        return scheduler

    @classmethod
    def restore_from_checkpoint(
        cls,
        path: Any,
        grid: HoneycombGrid,
        nectar_flow: NectarFlow,
        config: SwarmConfig | None = None,
    ) -> SwarmScheduler | None:
        """Phase 7.10: load a SwarmScheduler from a checkpoint blob.

        Returns ``None`` if the blob has no ``"scheduler"`` key (e.g.
        a v1 blob from Phase 6 or a v2 blob written before scheduler
        registration). Decoder errors propagate verbatim — caller is
        expected to handle ``MSCSecurityError`` / ``ValueError`` the
        same way as :meth:`HoneycombGrid.restore_from_checkpoint`.
        """
        from pathlib import Path as _Path

        from .storage.checkpoint import decode_blob

        blob = _Path(path).read_bytes()
        payload = decode_blob(blob)
        if not isinstance(payload, dict):
            return None
        sched_data = payload.get("scheduler")
        if sched_data is None:
            return None
        return cls.from_dict(sched_data, grid, nectar_flow, config)
