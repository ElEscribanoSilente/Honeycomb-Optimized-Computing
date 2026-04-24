"""
HOC Core · Cells base (private)
===============================

Base class de celdas del panal: ``HoneycombCell`` y sus enums de soporte
``CellState`` y ``CellRole``.

Este módulo contiene la clase padre que implementa toda la lógica común:
gestión de estado con eventos, vCores, feromonas, circuit breaker, locking
read-write, callbacks, serialización y métricas.

Las subclases especializadas (``QueenCell``, ``WorkerCell``, etc.) viven
en :mod:`hoc.core.cells_specialized` y se re-exportan desde
:mod:`hoc.core.cells`.

Extraído de ``core.py`` en Fase 3.3.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from .events import Event, EventBus, EventType, get_event_bus
from .grid_config import HoneycombConfig
from .grid_geometry import HexCoord, HexDirection
from .health import CircuitBreaker
from .locking import RWLock
from .pheromone import PheromoneField, PheromoneType

if TYPE_CHECKING:
    from ..metrics.collection import _InternalCellMetrics as CellMetrics

logger = logging.getLogger(__name__)

__all__ = [
    "CellState",
    "CellRole",
    "HoneycombCell",
]


# ═══════════════════════════════════════════════════════════════════════════════
# ESTADOS Y ROLES DE CELDAS
# ═══════════════════════════════════════════════════════════════════════════════


class CellState(Enum):
    """Estado de una celda del panal."""

    EMPTY = auto()
    ACTIVE = auto()
    IDLE = auto()
    SPAWNING = auto()
    MIGRATING = auto()
    FAILED = auto()
    RECOVERING = auto()
    SEALED = auto()
    OVERLOADED = auto()


class CellRole(Enum):
    """Rol especializado de una celda."""

    QUEEN = auto()
    WORKER = auto()
    DRONE = auto()
    NURSERY = auto()
    STORAGE = auto()
    GUARD = auto()
    SCOUT = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA BASE DEL PANAL (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════


class HoneycombCell:
    """
    Celda individual del panal hexagonal (v3.0).

    Mejoras v3.0:
    - Circuit breaker integrado
    - Estado atómico (no adquiere lock para reads de tipos simples)
    - Cambios de estado internos usan _set_state() para consistencia
    """

    __slots__ = (
        "__weakref__",
        "_circuit_breaker",
        "_config",
        "_creation_time",
        "_error_count",
        "_event_bus",
        "_last_activity",
        "_load",
        "_metadata",
        "_neighbors",
        "_pheromone_field",
        "_rw_lock",
        "_state",
        "_state_callbacks",
        "_ticks_processed",
        "_vcores",
        "coord",
        "role",
    )

    def __init__(
        self,
        coord: HexCoord,
        role: CellRole = CellRole.WORKER,
        config: HoneycombConfig | None = None,
        event_bus: EventBus | None = None,
    ):
        if not isinstance(coord, HexCoord):
            raise TypeError(f"coord must be HexCoord, got {type(coord)}")

        self.coord = coord
        self.role = role
        self._state = CellState.EMPTY
        self._vcores: list[Any] = []
        self._load: float = 0.0
        self._neighbors: dict[HexDirection, HoneycombCell | None] = {d: None for d in HexDirection}
        self._rw_lock = RWLock()
        self._metadata: dict[str, Any] = {}
        self._pheromone_field = PheromoneField()
        self._last_activity: float = time.time()
        self._error_count: int = 0
        self._config = config or HoneycombConfig()
        self._event_bus = event_bus or get_event_bus()
        self._ticks_processed: int = 0
        self._state_callbacks: list[Callable[[CellState, CellState], None]] = []
        self._creation_time = time.time()
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=self._config.max_consecutive_errors,
            recovery_timeout=self._config.circuit_breaker_recovery_s,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # GESTIÓN DE ESTADO (v3.0: método centralizado)
    # ─────────────────────────────────────────────────────────────────────────

    def _set_state(self, new_state: CellState) -> None:
        """
        v3.0: Método interno para cambiar estado con emisión de eventos.
        DEBE llamarse dentro de un write_lock ya adquirido.
        """
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        logger.debug(f"Cell {self.coord}: {old_state.name} → {new_state.name}")

        for callback in self._state_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

        self._event_bus.publish(
            Event(
                type=EventType.CELL_STATE_CHANGED,
                source=self,
                data={"old": old_state.name, "new": new_state.name},
            )
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PROPIEDADES
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def state(self) -> CellState:
        """
        Lectura atómica del estado.
        Python GIL garantiza atomicidad de lecturas de referencia.
        """
        return self._state

    @state.setter
    def state(self, value: CellState) -> None:
        with self._rw_lock.write_lock():
            self._set_state(value)

    @property
    def load(self) -> float:
        """Carga actual (0.0 - 1.0). Lectura atómica (float es inmutable)."""
        return self._load

    @property
    def is_available(self) -> bool:
        return self._state in (CellState.EMPTY, CellState.IDLE)

    @property
    def is_overloaded(self) -> bool:
        return self._load > self._config.steal_threshold_high

    @property
    def neighbor_count(self) -> int:
        with self._rw_lock.read_lock():
            return sum(1 for n in self._neighbors.values() if n is not None)

    @property
    def vcore_count(self) -> int:
        return len(self._vcores)

    @property
    def pheromone_level(self) -> float:
        return self._pheromone_field.total_intensity

    @property
    def age(self) -> float:
        return time.time() - self._creation_time

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """v3.0: Acceso al circuit breaker."""
        return self._circuit_breaker

    # ─────────────────────────────────────────────────────────────────────────
    # GESTIÓN DE VECINOS
    # ─────────────────────────────────────────────────────────────────────────

    def get_neighbor(self, direction: HexDirection) -> HoneycombCell | None:
        with self._rw_lock.read_lock():
            return self._neighbors.get(direction)

    def set_neighbor(
        self, direction: HexDirection, cell: HoneycombCell | None, bidirectional: bool = False
    ) -> None:
        with self._rw_lock.write_lock():
            self._neighbors[direction] = cell

        if bidirectional and cell is not None:
            cell.set_neighbor(direction.opposite(), self, bidirectional=False)

    def get_all_neighbors(self) -> list[HoneycombCell]:
        with self._rw_lock.read_lock():
            return [n for n in self._neighbors.values() if n is not None]

    def get_neighbor_loads(self) -> dict[HexDirection, float]:
        with self._rw_lock.read_lock():
            return {d: n.load for d, n in self._neighbors.items() if n is not None}

    # ─────────────────────────────────────────────────────────────────────────
    # GESTIÓN DE vCORES
    # ─────────────────────────────────────────────────────────────────────────

    def add_vcore(self, vcore: Any) -> bool:
        """Añade un vCore a la celda."""
        with self._rw_lock.write_lock():
            if len(self._vcores) >= self._config.vcores_per_cell:
                return False

            self._vcores.append(vcore)
            self._update_load()
            self._last_activity = time.time()

            # v3.0 FIX: usar _set_state() para emitir evento
            if self._state == CellState.EMPTY:
                self._set_state(CellState.IDLE)

            self._event_bus.publish(
                Event(
                    type=EventType.VCORE_ASSIGNED,
                    source=self,
                    data={"vcore_count": len(self._vcores)},
                )
            )

            return True

    def remove_vcore(self, vcore: Any) -> bool:
        """Remueve un vCore de la celda."""
        with self._rw_lock.write_lock():
            try:
                self._vcores.remove(vcore)
                self._update_load()

                # v3.0 FIX: usar _set_state()
                if not self._vcores:
                    self._set_state(CellState.EMPTY)

                self._event_bus.publish(
                    Event(
                        type=EventType.VCORE_REMOVED,
                        source=self,
                        data={"vcore_count": len(self._vcores)},
                    )
                )

                return True
            except ValueError:
                return False

    def get_vcores(self) -> list[Any]:
        with self._rw_lock.read_lock():
            return list(self._vcores)

    def _update_load(self) -> None:
        """Actualiza la métrica de carga. DEBE llamarse dentro de write_lock."""
        old_load = self._load
        self._load = len(self._vcores) / max(1, self._config.vcores_per_cell)

        if abs(old_load - self._load) > self._config.load_change_event_threshold:
            self._event_bus.publish(
                Event(
                    type=EventType.CELL_LOAD_CHANGED,
                    source=self,
                    data={"old": old_load, "new": self._load},
                ),
                async_=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # SISTEMA DE FEROMONAS
    # ─────────────────────────────────────────────────────────────────────────

    def deposit_pheromone(
        self,
        ptype: PheromoneType,
        amount: float,
        source: HexCoord | None = None,
        decay_rate: float | None = None,
    ) -> None:
        self._pheromone_field.deposit(
            ptype, amount, source, decay_rate or self._config.pheromone_decay_rate
        )

        self._event_bus.publish(
            Event(
                type=EventType.PHEROMONE_DEPOSITED,
                source=self,
                data={"type": ptype.name, "amount": amount},
            ),
            async_=True,
        )

    def get_pheromone(self, ptype: PheromoneType) -> float:
        return self._pheromone_field.get_intensity(ptype)

    def decay_pheromones(self, elapsed: float = 1.0) -> None:
        self._pheromone_field.decay_all(elapsed)

    def diffuse_pheromones(self) -> None:
        """Difunde feromonas a celdas vecinas."""
        rate = self._config.pheromone_diffusion_rate

        for ptype in PheromoneType:
            intensity = self._pheromone_field.get_intensity(ptype)
            if intensity > self._config.pheromone_diffuse_threshold:
                diffuse_amount = intensity * rate / 6

                for neighbor in self.get_all_neighbors():
                    neighbor.deposit_pheromone(ptype, diffuse_amount, source=self.coord)

    def follow_pheromone_gradient(self, ptype: PheromoneType) -> HexDirection | None:
        """Encuentra la dirección con mayor gradiente de feromona."""
        max_intensity = 0.0
        best_direction = None

        with self._rw_lock.read_lock():
            for direction, neighbor in self._neighbors.items():
                if neighbor is not None:
                    intensity = neighbor.get_pheromone(ptype)
                    if intensity > max_intensity:
                        max_intensity = intensity
                        best_direction = direction

        return best_direction

    # ─────────────────────────────────────────────────────────────────────────
    # PROCESAMIENTO (v3.0: con circuit breaker)
    # ─────────────────────────────────────────────────────────────────────────

    def execute_tick(self) -> dict[str, Any]:
        """Ejecuta un tick de procesamiento con circuit breaker."""
        # v3.0: Verificar circuit breaker antes de ejecutar
        if not self._circuit_breaker.allow_request():
            return {"processed": False, "reason": "CIRCUIT_OPEN"}

        with self._rw_lock.write_lock():
            if self._state == CellState.FAILED:
                return {"processed": False, "reason": "FAILED"}

            if self._state not in (CellState.ACTIVE, CellState.IDLE):
                return {"processed": False, "reason": self._state.name}

            self._set_state(CellState.ACTIVE)
            results = []
            errors = []

            for vcore in self._vcores:
                try:
                    if hasattr(vcore, "tick"):
                        result = vcore.tick()
                        results.append(result)
                except Exception as e:
                    self._error_count += 1
                    errors.append(str(e))
                    self._circuit_breaker.record_failure()
                    logger.error(f"Cell {self.coord} vCore error: {e}")

                    if not self._circuit_breaker.allow_request():
                        self._set_state(CellState.FAILED)
                        self._event_bus.publish(
                            Event(
                                type=EventType.CELL_ERROR,
                                source=self,
                                data={"errors": self._error_count},
                            )
                        )
                        self._event_bus.publish(
                            Event(
                                type=EventType.CIRCUIT_BREAKER_OPENED,
                                source=self,
                                data={"coord": self.coord.to_dict()},
                            )
                        )
                        break

            self._last_activity = time.time()
            self._ticks_processed += 1

            if self._state == CellState.ACTIVE and not errors:
                self._error_count = 0
                self._circuit_breaker.record_success()
                self._set_state(CellState.IDLE)

            return {
                "processed": True,
                "results": results,
                "errors": errors,
                "tick": self._ticks_processed,
            }

    def recover(self) -> bool:
        """Intenta recuperar una celda fallida."""
        with self._rw_lock.write_lock():
            if self._state != CellState.FAILED:
                return False

            self._set_state(CellState.RECOVERING)
            self._error_count = 0
            self._circuit_breaker.reset()

            self._vcores = []
            self._update_load()

            self._set_state(CellState.EMPTY)

            self._event_bus.publish(Event(type=EventType.CELL_RECOVERED, source=self, data={}))

            return True

    # ─────────────────────────────────────────────────────────────────────────
    # CALLBACKS
    # ─────────────────────────────────────────────────────────────────────────

    def on_state_change(
        self, callback: Callable[[CellState, CellState], None]
    ) -> Callable[[], None]:
        self._state_callbacks.append(callback)

        def unregister():
            if callback in self._state_callbacks:
                self._state_callbacks.remove(callback)

        return unregister

    # ─────────────────────────────────────────────────────────────────────────
    # SERIALIZACIÓN
    # ─────────────────────────────────────────────────────────────────────────

    def get_metrics(self) -> CellMetrics:
        # Lazy import rompe el ciclo: la CellMetrics interna vive en
        # hoc.metrics.collection como ``_InternalCellMetrics`` y depende de
        # CellRole/CellState (este módulo). Esta celda la construye en runtime.
        from ..metrics.collection import _InternalCellMetrics as CellMetrics

        with self._rw_lock.read_lock():
            return CellMetrics(
                coord=self.coord,
                role=self.role,
                state=self._state,
                load=self._load,
                vcore_count=len(self._vcores),
                error_count=self._error_count,
                ticks_processed=self._ticks_processed,
                pheromone_total=self._pheromone_field.total_intensity,
                neighbor_count=sum(1 for n in self._neighbors.values() if n),
                last_activity=self._last_activity,
                circuit_state=self._circuit_breaker.state.name,
            )

    def to_dict(self) -> dict[str, Any]:
        with self._rw_lock.read_lock():
            return {
                "coord": self.coord.to_dict(),
                "role": self.role.name,
                "state": self._state.name,
                "load": self._load,
                "vcores": len(self._vcores),
                "neighbors": sum(1 for n in self._neighbors.values() if n),
                "pheromones": self._pheromone_field.to_dict(),
                "errors": self._error_count,
                "ticks": self._ticks_processed,
                "last_activity": self._last_activity,
                "metadata": dict(self._metadata),
                "circuit_breaker": self._circuit_breaker.to_dict(),
            }

    def __repr__(self) -> str:
        return f"Cell({self.coord.q},{self.coord.r}" f"|{self.role.name}|{self._state.name})"

    def __hash__(self) -> int:
        return hash(self.coord)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HoneycombCell):
            return self.coord == other.coord
        return False
