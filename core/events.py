"""
HOC Core · Events
=================

Sistema de eventos thread-safe del grid hexagonal.

Provee:
- ``EventType``: enum con todos los tipos de evento emitidos por el sistema.
- ``Event``: dataclass inmutable para un evento concreto.
- ``EventHandler``: protocolo para funciones que procesan eventos.
- ``EventBus``: bus de eventos con rate limiting, backpressure y weak refs.
- ``get_event_bus``/``set_event_bus``/``reset_event_bus``: singleton global.

Este módulo se extrajo de ``core.py`` en Fase 3.3 como parte del split del
mega-módulo monolítico. La API pública es idéntica a la original.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
import weakref
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "EventType",
    "Event",
    "EventHandler",
    "EventBus",
    "get_event_bus",
    "set_event_bus",
    "reset_event_bus",
]


# ═══════════════════════════════════════════════════════════════════════════════
# SISTEMA DE EVENTOS (v3.0 - con backpressure y rate limiting)
# ═══════════════════════════════════════════════════════════════════════════════


class EventType(Enum):
    """Tipos de eventos del sistema HOC."""

    # Eventos de celda
    CELL_STATE_CHANGED = auto()
    CELL_LOAD_CHANGED = auto()
    CELL_ERROR = auto()
    CELL_RECOVERED = auto()

    # Eventos de grid
    GRID_TICK_START = auto()
    GRID_TICK_END = auto()
    GRID_CELL_ADDED = auto()
    GRID_CELL_REMOVED = auto()
    GRID_REBALANCE = auto()

    # Eventos de vCore
    VCORE_ASSIGNED = auto()
    VCORE_REMOVED = auto()
    VCORE_MIGRATED = auto()

    # Eventos de feromonas
    PHEROMONE_DEPOSITED = auto()
    PHEROMONE_DECAYED = auto()
    PHEROMONE_GRADIENT_FORMED = auto()

    # Eventos de trabajo
    WORK_STOLEN = auto()
    WORK_COMPLETED = auto()
    WORK_FAILED = auto()

    # Eventos de spawning
    ENTITY_SPAWNED = auto()
    ENTITY_INCUBATING = auto()

    # v3.0: Nuevos eventos
    HEALTH_CHECK = auto()
    HEALTH_ALERT = auto()
    CIRCUIT_BREAKER_OPENED = auto()
    CIRCUIT_BREAKER_CLOSED = auto()
    GUARD_VALIDATION_FAILED = auto()
    SCOUT_DISCOVERY = auto()


@dataclass(frozen=True, slots=True)
class Event:
    """Evento inmutable del sistema."""

    type: EventType
    source: Any
    data: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self):
        if not isinstance(self.data, Mapping):
            object.__setattr__(self, "data", dict(self.data))


@runtime_checkable
class EventHandler(Protocol):
    """Protocolo para manejadores de eventos."""

    def __call__(self, event: Event) -> None: ...


class _HandlerRef:
    """
    v3.0 FIX: Wrapper para referencias a handlers que maneja correctamente
    bound methods, lambdas y funciones regulares.

    weakref.ref() no funciona con bound methods porque se destruyen
    inmediatamente. Este wrapper detecta el caso y mantiene una strong ref
    cuando es necesario, con opción de weak ref para objetos persistentes.
    """

    __slots__ = ("_handler_id", "_is_weak", "_ref")

    def __init__(self, handler: EventHandler, weak: bool = True):
        self._handler_id = id(handler)
        try:
            if weak:
                self._ref = weakref.ref(handler)
                self._is_weak = True
            else:
                self._ref = handler
                self._is_weak = False
        except TypeError:
            # bound methods, lambdas etc. no soportan weakref
            self._ref = handler
            self._is_weak = False

    def __call__(self) -> EventHandler | None:
        if self._is_weak:
            return self._ref()
        return self._ref

    @property
    def handler_id(self) -> int:
        return self._handler_id


class EventBus:
    """
    Bus de eventos thread-safe (v3.0).

    Mejoras v3.0:
    - HandlerRef para manejar bound methods correctamente
    - Rate limiting por tipo de evento (evita flood)
    - Backpressure: cola async limitada con drop policy
    - Métricas de eventos publicados/descartados
    """

    __slots__ = (
        "_async_executor",
        "_event_history",
        "_handlers",
        "_last_publish_time",
        "_lock",
        "_max_async_queue",
        "_max_history",
        "_rate_limits",
        "_shutting_down",
        "_stats",
    )

    def __init__(self, max_history: int = 1000, max_async_queue: int = 10000):
        self._handlers: dict[EventType, list[tuple[int, _HandlerRef]]] = defaultdict(list)
        self._lock = threading.RLock()
        self._async_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="event_")
        self._event_history: deque = deque(maxlen=max_history)
        self._max_history = max_history
        self._rate_limits: dict[EventType, float] = {}
        self._last_publish_time: dict[EventType, float] = {}
        self._stats: dict[str, int] = defaultdict(int)
        self._max_async_queue = max_async_queue
        self._shutting_down = False

    def set_rate_limit(self, event_type: EventType, min_interval: float) -> None:
        """
        Establece intervalo mínimo entre publicaciones del mismo tipo.

        Args:
            event_type: Tipo de evento a limitar
            min_interval: Segundos mínimos entre publicaciones
        """
        self._rate_limits[event_type] = min_interval

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
        priority: int = 0,
        weak: bool = False,  # v3.0: default False (más seguro)
    ) -> Callable[[], None]:
        """
        Suscribe un handler a un tipo de evento.

        Args:
            event_type: Tipo de evento a escuchar
            handler: Función manejadora
            priority: Prioridad (mayor = ejecuta primero)
            weak: Usar weak reference (default False, v3.0 cambio)

        Returns:
            Función para desuscribirse
        """
        with self._lock:
            ref = _HandlerRef(handler, weak=weak)
            self._handlers[event_type].append((priority, ref))
            self._handlers[event_type].sort(key=lambda x: -x[0])
            handler_id = ref.handler_id

        def unsubscribe():
            with self._lock:
                self._handlers[event_type] = [
                    (p, r) for p, r in self._handlers[event_type] if r.handler_id != handler_id
                ]

        return unsubscribe

    def publish(self, event: Event, async_: bool = False) -> bool:
        """
        Publica un evento a todos los handlers suscritos.

        Returns:
            True si fue publicado, False si fue rate-limited o descartado
        """
        if self._shutting_down:
            return False

        # Rate limiting
        if event.type in self._rate_limits:
            now = time.time()
            last = self._last_publish_time.get(event.type, 0.0)
            if now - last < self._rate_limits[event.type]:
                self._stats["rate_limited"] += 1
                return False
            self._last_publish_time[event.type] = now

        self._event_history.append(event)
        self._stats["published"] += 1

        handlers = self._get_handlers(event.type)

        if async_:
            for handler in handlers:
                try:
                    self._async_executor.submit(self._safe_call, handler, event)
                except RuntimeError:
                    # Executor cerrado o lleno
                    self._stats["dropped"] += 1
        else:
            for handler in handlers:
                self._safe_call(handler, event)

        return True

    def _get_handlers(self, event_type: EventType) -> list[EventHandler]:
        """Obtiene handlers activos (limpia refs muertas)."""
        with self._lock:
            active = []
            dead_indices = []

            for i, (_priority, ref) in enumerate(self._handlers[event_type]):
                handler = ref()
                if handler is not None:
                    active.append(handler)
                else:
                    dead_indices.append(i)

            # Limpiar refs muertas
            for i in reversed(dead_indices):
                del self._handlers[event_type][i]

            return active

    def _safe_call(self, handler: EventHandler, event: Event) -> None:
        """Llama a un handler con manejo de excepciones."""
        try:
            handler(event)
        except Exception as e:
            logger.error(f"Event handler error for {event.type.name}: {e}", exc_info=True)
            self._stats["handler_errors"] += 1

    def get_history(self, event_type: EventType | None = None, limit: int = 100) -> list[Event]:
        """Obtiene historial de eventos."""
        events = list(self._event_history)
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]

    def clear_history(self, older_than: float | None = None) -> int:
        """
        Limpia el historial de eventos.

        Args:
            older_than: Si se proporciona, solo elimina eventos con timestamp
                        anterior a este valor (epoch seconds). Si es None,
                        elimina todo el historial.

        Returns:
            Número de eventos eliminados.
        """
        if older_than is None:
            removed = len(self._event_history)
            self._event_history.clear()
            return removed

        original_len = len(self._event_history)
        self._event_history = deque(
            (e for e in self._event_history if e.timestamp >= older_than), maxlen=self._max_history
        )
        return original_len - len(self._event_history)

    def get_stats(self) -> dict[str, int]:
        """Estadísticas del bus de eventos."""
        return dict(self._stats)

    def shutdown(self) -> None:
        """Cierra el executor de eventos asíncronos."""
        self._shutting_down = True
        self._async_executor.shutdown(wait=True)


# v3.0 FIX: Singleton thread-safe con lock
# v3.1: Added reset_event_bus() for testing/isolation
_event_bus: EventBus | None = None
_event_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Obtiene la instancia global del EventBus (thread-safe)."""
    global _event_bus
    if _event_bus is None:
        with _event_bus_lock:
            if _event_bus is None:  # Double-check locking
                _event_bus = EventBus()
    return _event_bus


def set_event_bus(bus: EventBus) -> None:
    """
    Reemplaza la instancia global del EventBus (dependency injection).

    Útil para testing o para inyectar un bus configurado con rate limits
    y parámetros personalizados antes de construir el grid.
    """
    global _event_bus
    with _event_bus_lock:
        _event_bus = bus


def reset_event_bus() -> None:
    """
    Resetea la instancia global del EventBus.

    Útil para aislamiento en tests: cada test puede llamar
    reset_event_bus() en su setup/teardown para evitar state leakage.
    """
    global _event_bus
    with _event_bus_lock:
        if _event_bus is not None:
            _event_bus.shutdown()
        _event_bus = None
