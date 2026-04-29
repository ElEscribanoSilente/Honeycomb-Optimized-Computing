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

Phase 7.1: :meth:`HoneycombCell.execute_tick` is now ``async``. The
sync body lives in :meth:`_sync_execute_tick`; the async wrapper
dispatches it to ``asyncio.to_thread`` so existing locking and vCore
contracts stay unchanged. Legacy callers can use
:meth:`run_execute_tick_sync` (a one-shot DeprecationWarning is
emitted on first call).
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from collections import deque
from collections.abc import Callable
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, ClassVar

# Absolute imports for state_machines: ``core`` is importable both as
# ``core`` (top-level, used by tests) and as ``hoc.core`` (declared package,
# used by external consumers). A relative ``from ..state_machines`` resolves
# only in the second case. Absolute imports work in both because
# ``state_machines/`` sits at the same level as ``core/`` on sys.path.
from state_machines.base import HocStateMachine, IllegalStateTransition
from state_machines.cell_fsm import build_cell_fsm

from .events import Event, EventBus, EventType, get_event_bus
from .grid_config import HoneycombConfig
from .grid_geometry import HexCoord, HexDirection
from .health import CircuitBreaker
from .locking import RWLock

# Phase 5.3: structured event log lives in ``core.observability`` so it
# avoids the dual-import dance that the top-level package layout
# (``package-dir = {hoc = "."}``) imposes on top-level modules. Relative
# import is fine here — both consumers (cell + cells_specialized) live
# in the same subpackage.
from .observability import log_cell_state_transition
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
    """Estado de una celda del panal.

    Phase 4.3: ``SPAWNING`` and ``OVERLOADED`` were removed (B12-ter
    cleanup). Both were aspirational states that no production path ever
    assigned. ``MIGRATING`` and ``SEALED`` are kept as **reserved** —
    Phase 5 will wire them up (MIGRATING during ``CellFailover.migrate_cell``
    for observability; SEALED for graceful shutdown). Until then
    ``choreo`` continues to flag them as ``dead_state`` warnings.
    """

    EMPTY = auto()
    ACTIVE = auto()
    IDLE = auto()
    MIGRATING = auto()  # reserved: Phase 5 wire-up in CellFailover.migrate_cell
    FAILED = auto()
    RECOVERING = auto()
    SEALED = auto()  # reserved: Phase 5 wire-up for graceful shutdown


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
# FSM VIEW (Phase 6.6)
# ═══════════════════════════════════════════════════════════════════════════════


class _CellFsmView:
    """Read-only proxy that exposes ``cell.fsm.state`` and
    ``cell.fsm.history`` over the per-cell state slot + history deque.

    Phase 6.6 fix: the spec object behind every cell's transition
    validation is now a single :class:`HocStateMachine` shared on the
    class (``HoneycombCell._CLASS_FSM``) instead of one tramoya
    machine allocated per cell. Cells track their own current state
    in ``_state`` (already part of the slot layout) and a small
    ``_state_history`` deque; this view stitches them back into the
    ``state`` / ``history`` API that the rest of the codebase
    (``test_cell_seal.py``, ``test_resilience.py``,
    ``test_state_machines.py``) inspects.

    Construction is cheap (``__slots__`` with a single weak-ish
    cell reference). Allocating one of these per ``cell.fsm`` access
    is intentional: the alternative — caching the view as a
    per-instance slot — would re-introduce one allocation per cell at
    construction time, defeating the perf fix.
    """

    __slots__ = ("_cell",)

    def __init__(self, cell: HoneycombCell) -> None:
        self._cell = cell

    @property
    def state(self) -> str:
        """Current state name (matches ``cell.state.name``)."""
        return self._cell._state.name

    @property
    def history(self) -> list[str]:
        """Previous state names, most recent last. Bounded by
        ``HoneycombCell._HISTORY_MAXLEN``."""
        return list(self._cell._state_history)

    def transition_to(self, target: str) -> str:
        """Drive a state transition by raw state name.

        Routes through the cell's typed setter so locking, event-bus
        publish, structured-log emission, and history-deque update all
        happen in the usual order. Returns the new state name on
        success.

        Raises :class:`IllegalStateTransition` with
        ``reason="unknown_state"`` if ``target`` is not a known
        ``CellState`` member, and with ``reason="no_edge"`` if the
        spec rejects the transition.

        Prefer ``cell.state = CellState.X`` for typed call-sites; this
        method exists for debugging / operator tools that arrive with a
        string name and for the atomicity-of-setter contract test.
        """
        cls_fsm = HoneycombCell._CLASS_FSM
        if target not in cls_fsm.states:
            raise IllegalStateTransition(
                cls_fsm.name,
                self._cell._state.name,
                target,
                reason="unknown_state",
            )
        # Lookup is total here because the spec FSM is built from
        # CellState enum names — every state in cls_fsm.states is also
        # a CellState member.
        self._cell.state = CellState[target]
        return self._cell._state.name


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
        # Phase 6.6: per-cell history deque (replaced the per-cell tramoya
        # machine). Bounded to ``_HISTORY_MAXLEN``.
        "_state_history",
        "_ticks_processed",
        "_vcores",
        "coord",
        "role",
    )

    # Phase 6.6: class-level shared FSM. One ``build_cell_fsm()`` per
    # process instead of one per cell. Used only for
    # :meth:`HocStateMachine.is_legal_transition` (pure spec-graph check),
    # so its internal ``_machine.state`` is never mutated and the share
    # is safe across thousands of concurrent cells. ``ClassVar`` keeps
    # mypy happy and excludes the attribute from ``__slots__``.
    _CLASS_FSM: ClassVar[HocStateMachine] = build_cell_fsm()
    # Cap the per-cell state history. Matches the ``history_size=8`` that
    # was set on the per-cell tramoya machine pre-Phase-6.6.
    _HISTORY_MAXLEN: ClassVar[int] = 8
    # Phase 7.2: one-shot guard so ``run_execute_tick_sync`` emits its
    # DeprecationWarning only the first time it's called per process.
    # Class-level mutable ClassVar; all instances share the flag.
    _SYNC_DEPRECATION_EMITTED: ClassVar[bool] = False

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
        # Phase 6.6: per-cell history deque. The FSM spec lives on the
        # class (``_CLASS_FSM``); each cell tracks its own current state
        # in ``_state`` (set above) and its own past states here.
        # ``_set_state`` below validates each transition against the
        # shared spec without ever touching the spec object's internal
        # state — so the share is safe.
        self._state_history: deque[str] = deque(maxlen=self._HISTORY_MAXLEN)

    # ─────────────────────────────────────────────────────────────────────────
    # GESTIÓN DE ESTADO (v3.0: método centralizado)
    # ─────────────────────────────────────────────────────────────────────────

    def _set_state(self, new_state: CellState) -> None:
        """
        v3.0: Método interno para cambiar estado con emisión de eventos.
        DEBE llamarse dentro de un write_lock ya adquirido.

        Phase 4: la transición se valida contra el FSM antes de
        comprometer ``_state``. Una transición no documentada lanza
        :class:`IllegalStateTransition`. Las transiciones idempotentes
        (``old == new``) se silencian sin tocar el FSM, manteniendo el
        contrato pre-Phase-4 de los callers.

        Phase 6.6: la validación ahora consulta el spec compartido
        ``_CLASS_FSM.is_legal_transition`` (pura, sin mutar). El estado
        per-cell vive en ``self._state`` y el historial en
        ``self._state_history`` — no se aloca un tramoya machine por
        cell. Reduce el costo de ``HoneycombCell()`` en ~40 % en grids
        de radio 2-3.
        """
        old_state = self._state
        if old_state == new_state:
            return

        # Phase 6.6: spec-only check against the shared FSM. If the edge
        # is missing, raise the same exception type as the legacy
        # per-instance ``transition_to`` did (``no_edge`` reason). _state
        # is not mutated, callbacks don't fire, no event is published.
        if not self._CLASS_FSM.is_legal_transition(old_state.name, new_state.name):
            raise IllegalStateTransition(
                self._CLASS_FSM.name,
                old_state.name,
                new_state.name,
                reason="no_edge",
            )

        # Append BEFORE mutating ``_state`` so a concurrent reader of
        # ``self.fsm.history`` never sees the new state without its
        # predecessor in the trail.
        self._state_history.append(old_state.name)

        self._state = new_state
        logger.debug(f"Cell {self.coord}: {old_state.name} → {new_state.name}")

        # Phase 5.3: emit a structured ``cell.state_changed`` event for
        # the observability log. The helper hides the field-name
        # convention so future readers add events with the same shape.
        log_cell_state_transition(self.coord, old_state.name, new_state.name)

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

    @property
    def fsm(self) -> _CellFsmView:
        """Phase 4 + 6.6: read-only proxy exposing this cell's current
        state name (``cell.fsm.state``) and bounded transition history
        (``cell.fsm.history``).

        Pre-Phase-6.6 this returned a per-cell ``HocStateMachine``
        instance. Phase 6.6 replaced that with a class-level shared FSM
        plus a per-cell history deque to drop the per-construction
        cost; the returned view stitches them back into the same two
        attributes the rest of the codebase reads. Mutation must go
        through :attr:`state` setter (the view is read-only)."""
        return _CellFsmView(self)

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
        """Añade un vCore a la celda.

        Phase 5.1: una celda SEALED rechaza nuevos vCores. ``seal()`` es
        el path de graceful shutdown — aceptar trabajo nuevo invalidaría
        la promesa operacional ("drained, refusing tasks").
        """
        with self._rw_lock.write_lock():
            if self._state == CellState.SEALED:
                return False

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

    async def execute_tick(self) -> dict[str, Any]:
        """Phase 7.1: async signature.

        The body is dispatched to a worker thread via
        :func:`asyncio.to_thread` so the existing :class:`RWLock` +
        synchronous vCore semantics stay correct. Cells can be ticked
        concurrently from :meth:`HoneycombGrid.tick` via
        ``asyncio.gather`` with no GIL contention beyond what the
        thread pool already enforces.

        Per-vCore async dispatch (``await asyncio.to_thread(vcore.tick)``
        for each vCore) is a future Phase 7.6+ optimisation; see
        ADR-016 for the trade-off discussion.
        """
        return await asyncio.to_thread(self._sync_execute_tick)

    def _sync_execute_tick(self) -> dict[str, Any]:
        """Phase 7.1: the pre-Phase-7.1 ``execute_tick`` body, kept
        synchronous so :class:`RWLock` and vCore APIs (most of which
        remain blocking) work without extra plumbing.

        Use :meth:`execute_tick` (async) from new code or
        :meth:`run_execute_tick_sync` (sync wrapper with deprecation
        warning) for legacy callers.
        """
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

    def run_execute_tick_sync(self) -> dict[str, Any]:
        """Phase 7.2: blocking wrapper for legacy callers.

        Equivalent to ``asyncio.run(self.execute_tick())`` for use
        outside an event loop. Emits :class:`DeprecationWarning` once
        per process — switch new call-sites to ``await
        cell.execute_tick()``. From inside a running loop this raises
        ``RuntimeError`` (use ``await`` instead).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "HoneycombCell.run_execute_tick_sync called from a running "
                "event loop; use 'await cell.execute_tick()' instead."
            )
        if not HoneycombCell._SYNC_DEPRECATION_EMITTED:
            warnings.warn(
                "HoneycombCell.run_execute_tick_sync is a v1→v2 migration "
                "aid; switch callers to 'await cell.execute_tick()'. This "
                "wrapper will be removed in HOC v3.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            HoneycombCell._SYNC_DEPRECATION_EMITTED = True
        # Phase 7.2: skip the to_thread hop; we already lack an event
        # loop, so ``_sync_execute_tick`` runs in this thread directly.
        return self._sync_execute_tick()

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

    def seal(self, reason: str = "graceful_shutdown") -> bool:
        """Phase 5.1: graceful shutdown of this cell.

        Drains all vCores, refuses new tasks, captures final metrics in the
        log, and transitions to ``SEALED``. SEALED is intended to be
        terminal — the production paths never revive a sealed cell — but
        the FSM keeps the wildcard admin transitions available for tests
        and operator overrides.

        Idempotent: returns ``False`` if the cell is already sealed.
        Refuses to seal a ``FAILED`` cell (use ``recover()`` first).
        Returns ``True`` on a successful seal.
        """
        with self._rw_lock.write_lock():
            if self._state == CellState.SEALED:
                return False
            if self._state == CellState.FAILED:
                return False

            # Drain vCores. The cells lose their work; we do not migrate
            # here — graceful shutdown is opt-in and the operator is
            # expected to have rebalanced ahead of the call.
            drained_vcores = len(self._vcores)
            self._vcores = []
            self._update_load()

            # Capture final metrics for the audit log. We log rather than
            # publish a dedicated event so the seal() reason stays paired
            # with the snapshot in the same log line.
            final_metrics = {
                "ticks_processed": self._ticks_processed,
                "error_count": self._error_count,
                "pheromone_total": self._pheromone_field.total_intensity,
                "vcores_drained": drained_vcores,
                "age_seconds": round(time.time() - self._creation_time, 3),
            }

            # Mutate via _set_state so the FSM validates the admin_seal
            # transition and the CELL_STATE_CHANGED event fires.
            self._set_state(CellState.SEALED)

            logger.info(
                "Cell %s sealed: reason=%s metrics=%s",
                self.coord,
                reason,
                final_metrics,
            )
            # Phase 5.3: structured ``cell.sealed`` event so log
            # collectors can count graceful shutdowns separately from
            # the underlying state-change event emitted by _set_state.
            from .observability import get_event_logger

            get_event_logger("hoc.events.cell").info(
                "cell.sealed",
                coord=str(self.coord),
                reason=reason,
                **final_metrics,
            )
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
                # Phase 6.3: ``state_history`` joins to_dict so checkpoint
                # blobs can preserve the FSM trail across restarts. The
                # legacy ``from_dict`` ignores unknown keys gracefully,
                # so older checkpoints (pre-Phase-6.3, no history) still
                # restore — the new attribute simply stays empty.
                "state_history": list(self._state_history),
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
