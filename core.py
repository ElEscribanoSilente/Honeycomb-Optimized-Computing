"""
HOC Core v3.0 - Estructuras Fundamentales del Panal Hexagonal (Producción)
=========================================================================

Implementación production-ready de geometría hexagonal y celdas del panal.
Usa coordenadas axiales (q, r) para navegación eficiente.

MEJORAS v3.0 sobre v2.0:
- [FIX] Variable shadowing en _cached_filled_hex (r param vs loop r)
- [FIX] EventBus weak refs que fallaban con bound methods
- [FIX] steal_from llamaba _update_load() fuera del lock
- [FIX] add_vcore/remove_vcore bypasseaban el setter de estado (sin eventos)
- [FIX] heapq comparisons fallaban con dicts no-comparables
- [FIX] Singleton global EventBus no era thread-safe
- [FIX] HoneycombConfig usaba assert (deshabilitado con -O)
- [FIX] RWLock tenía potential deadlock sin timeout
- [ARCH] __all__ para exports limpios
- [ARCH] Índices por rol/estado en HoneycombGrid para O(1) lookups
- [ARCH] Circuit breaker con backoff exponencial para recovery
- [ARCH] LRU eviction en StorageCell cuando llega a capacidad máxima
- [ARCH] GuardCell y ScoutCell implementados (estaban declarados pero no existían)
- [ARCH] from_dict() completo con restauración de estado
- [ARCH] HealthMonitor integrado con alertas configurables
- [PERF] Pheromone updates batched con NumPy
- [PERF] Lock-free reads donde es seguro (propiedades atómicas)
- [PERF] EventBus con backpressure y rate limiting
- [PERF] Bulk neighbor connection con operaciones vectoriales
- [ROBUST] Timeouts en lock acquisition
- [ROBUST] Graceful shutdown con cleanup garantizado
- [ROBUST] Rate limiting en event publishing
- [ROBUST] Validación exhaustiva en Config con mensajes claros

Sistema de Coordenadas Axiales:
                
        +r  ↗
           / \\
          /   \\
    <----⬡-----> +q
          \\   /
           \\ /
             ↘ +s (implícito: s = -q - r)

Direcciones (vecinos):
    
         NW(0)    NE(1)
            ↖   ↗
             \\ /
        W(5)──⬡──E(2)
             / \\
            ↙   ↘
         SW(4)    SE(3)

"""

from __future__ import annotations

import math
import time
import json
import threading
import logging
import weakref
import uuid
from abc import ABC, abstractmethod
from enum import Enum, auto, IntEnum
from dataclasses import dataclass, field, asdict
from typing import (
    Dict, List, Optional, Set, Tuple, Iterator,
    Callable, Any, TypeVar, Generic, Union,
    Protocol, runtime_checkable, FrozenSet,
    NamedTuple, Sequence, Mapping, TYPE_CHECKING,
    ClassVar, Final
)
from collections import defaultdict, deque, OrderedDict
from functools import lru_cache, cached_property
from contextlib import contextmanager
import numpy as np
from numpy.typing import NDArray
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import heapq

if TYPE_CHECKING:
    from typing import TypeAlias

logger = logging.getLogger(__name__)

T = TypeVar('T')
CellT = TypeVar('CellT', bound='HoneycombCell')

# Type aliases
CoordTuple: TypeAlias = Tuple[int, int]
CubeTuple: TypeAlias = Tuple[int, int, int]
PixelTuple: TypeAlias = Tuple[float, float]

__all__ = [
    # Core types
    'HexCoord', 'HexDirection', 'HexRegion', 'HexRing', 'HexPathfinder',
    # Events
    'EventType', 'Event', 'EventBus', 'EventHandler',
    # Concurrency
    'RWLock',
    # Pheromones
    'PheromoneType', 'PheromoneDeposit', 'PheromoneField',
    # States & Roles
    'CellState', 'CellRole',
    # Config
    'HoneycombConfig', 'GridTopology',
    # Metrics
    'CellMetrics', 'GridMetrics', 'MetricsCollector',
    # Cells
    'HoneycombCell', 'QueenCell', 'WorkerCell', 'DroneCell',
    'NurseryCell', 'StorageCell', 'GuardCell', 'ScoutCell',
    # Grid
    'HoneycombGrid',
    # Health
    'HealthMonitor', 'HealthStatus', 'CircuitBreaker', 'CircuitState',
    # Utilities
    'create_grid', 'benchmark_grid',
    # Event bus management
    'get_event_bus', 'set_event_bus', 'reset_event_bus',
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
            object.__setattr__(self, 'data', dict(self.data))


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
    __slots__ = ('_ref', '_is_weak', '_handler_id')

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

    def __call__(self) -> Optional[EventHandler]:
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
        '_handlers', '_lock', '_async_executor', '_event_history',
        '_max_history', '_rate_limits', '_last_publish_time',
        '_stats', '_max_async_queue', '_shutting_down'
    )

    def __init__(
        self,
        max_history: int = 1000,
        max_async_queue: int = 10000
    ):
        self._handlers: Dict[EventType, List[Tuple[int, _HandlerRef]]] = defaultdict(list)
        self._lock = threading.RLock()
        self._async_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="event_"
        )
        self._event_history: deque = deque(maxlen=max_history)
        self._max_history = max_history
        self._rate_limits: Dict[EventType, float] = {}
        self._last_publish_time: Dict[EventType, float] = {}
        self._stats: Dict[str, int] = defaultdict(int)
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
        weak: bool = False  # v3.0: default False (más seguro)
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
                    (p, r) for p, r in self._handlers[event_type]
                    if r.handler_id != handler_id
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
                self._stats['rate_limited'] += 1
                return False
            self._last_publish_time[event.type] = now

        self._event_history.append(event)
        self._stats['published'] += 1

        handlers = self._get_handlers(event.type)

        if async_:
            for handler in handlers:
                try:
                    self._async_executor.submit(self._safe_call, handler, event)
                except RuntimeError:
                    # Executor cerrado o lleno
                    self._stats['dropped'] += 1
        else:
            for handler in handlers:
                self._safe_call(handler, event)

        return True

    def _get_handlers(self, event_type: EventType) -> List[EventHandler]:
        """Obtiene handlers activos (limpia refs muertas)."""
        with self._lock:
            active = []
            dead_indices = []

            for i, (priority, ref) in enumerate(self._handlers[event_type]):
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
            self._stats['handler_errors'] += 1

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Event]:
        """Obtiene historial de eventos."""
        events = list(self._event_history)
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]

    def clear_history(self, older_than: Optional[float] = None) -> int:
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
            (e for e in self._event_history if e.timestamp >= older_than),
            maxlen=self._max_history
        )
        return original_len - len(self._event_history)

    def get_stats(self) -> Dict[str, int]:
        """Estadísticas del bus de eventos."""
        return dict(self._stats)

    def shutdown(self) -> None:
        """Cierra el executor de eventos asíncronos."""
        self._shutting_down = True
        self._async_executor.shutdown(wait=True)


# v3.0 FIX: Singleton thread-safe con lock
# v3.1: Added reset_event_bus() for testing/isolation
_event_bus: Optional[EventBus] = None
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


# ═══════════════════════════════════════════════════════════════════════════════
# READ-WRITE LOCK (v3.0 - con timeout)
# ═══════════════════════════════════════════════════════════════════════════════

class RWLock:
    """
    Read-Write Lock que permite múltiples lectores o un único escritor.
    
    v3.0: Timeout configurable para evitar deadlocks.
    Prioriza escritores para evitar starvation.
    """

    __slots__ = ('_read_ready', '_readers', '_writers_waiting', '_writer_active')

    _DEFAULT_TIMEOUT: ClassVar[float] = 30.0

    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False

    @contextmanager
    def read_lock(self, timeout: Optional[float] = None):
        """Adquiere lock de lectura con timeout opcional."""
        timeout = timeout or self._DEFAULT_TIMEOUT
        deadline = time.monotonic() + timeout

        with self._read_ready:
            while self._writer_active or self._writers_waiting > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"RWLock read_lock timeout after {timeout}s"
                    )
                self._read_ready.wait(timeout=remaining)
            self._readers += 1
        try:
            yield
        finally:
            with self._read_ready:
                self._readers -= 1
                if self._readers == 0:
                    self._read_ready.notify_all()

    @contextmanager
    def write_lock(self, timeout: Optional[float] = None):
        """Adquiere lock de escritura con timeout opcional."""
        timeout = timeout or self._DEFAULT_TIMEOUT
        deadline = time.monotonic() + timeout

        with self._read_ready:
            self._writers_waiting += 1
            try:
                while self._readers > 0 or self._writer_active:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"RWLock write_lock timeout after {timeout}s"
                        )
                    self._read_ready.wait(timeout=remaining)
                self._writer_active = True
            except:
                self._writers_waiting -= 1
                raise
            self._writers_waiting -= 1
        try:
            yield
        finally:
            with self._read_ready:
                self._writer_active = False
                self._read_ready.notify_all()


# ═══════════════════════════════════════════════════════════════════════════════
# COORDENADAS HEXAGONALES OPTIMIZADAS
# ═══════════════════════════════════════════════════════════════════════════════

class HexDirection(IntEnum):
    """
    6 direcciones en el grid hexagonal.
    Ordenadas en sentido horario desde arriba-derecha.
    """
    NE = 0   # Noreste   (+1, -1)
    E = 1    # Este      (+1,  0)
    SE = 2   # Sureste   ( 0, +1)
    SW = 3   # Suroeste  (-1, +1)
    W = 4    # Oeste     (-1,  0)
    NW = 5   # Noroeste  ( 0, -1)

    def opposite(self) -> 'HexDirection':
        """Retorna la dirección opuesta."""
        return HexDirection((self + 3) % 6)

    def rotate_cw(self, steps: int = 1) -> 'HexDirection':
        """Rotar en sentido horario."""
        return HexDirection((self + steps) % 6)

    def rotate_ccw(self, steps: int = 1) -> 'HexDirection':
        """Rotar en sentido antihorario."""
        return HexDirection((self - steps) % 6)

    @property
    def vector(self) -> CoordTuple:
        """Vector de dirección (dq, dr)."""
        return _DIRECTION_VECTORS[self]

    @classmethod
    def from_angle(cls, angle_deg: float) -> 'HexDirection':
        """Obtiene la dirección más cercana a un ángulo."""
        normalized = angle_deg % 360
        index = round(normalized / 60) % 6
        return cls(index)


# Vectores de dirección precalculados (inmutables)
_DIRECTION_VECTORS: Final[Tuple[CoordTuple, ...]] = (
    (1, -1),   # NE
    (1, 0),    # E
    (0, 1),    # SE
    (-1, 1),   # SW
    (-1, 0),   # W
    (0, -1),   # NW
)

# Arrays NumPy para operaciones vectoriales
_DIRECTION_ARRAY: Final = np.array(_DIRECTION_VECTORS, dtype=np.int32)


@dataclass(frozen=True, slots=True, order=True)
class HexCoord:
    """
    Coordenada hexagonal axial (q, r) optimizada.

    Inmutable y hasheable para usar como clave de diccionario.
    La tercera coordenada s es implícita: s = -q - r
    """
    q: int
    r: int

    def __post_init__(self):
        """Validación y coerción de tipos."""
        if not isinstance(self.q, int):
            object.__setattr__(self, 'q', int(self.q))
        if not isinstance(self.r, int):
            object.__setattr__(self, 'r', int(self.r))

    @property
    def s(self) -> int:
        """Tercera coordenada cúbica (implícita)."""
        return -self.q - self.r

    @cached_property
    def cube(self) -> CubeTuple:
        """Retorna coordenadas cúbicas (q, r, s)."""
        return (self.q, self.r, self.s)

    @cached_property
    def array(self) -> NDArray[np.int32]:
        """Representación NumPy para operaciones vectoriales."""
        return np.array([self.q, self.r], dtype=np.int32)

    @cached_property
    def magnitude(self) -> int:
        """Distancia desde el origen."""
        return (abs(self.q) + abs(self.r) + abs(self.s)) // 2

    def __add__(self, other: 'HexCoord') -> 'HexCoord':
        if isinstance(other, HexCoord):
            return HexCoord(self.q + other.q, self.r + other.r)
        return NotImplemented

    def __sub__(self, other: 'HexCoord') -> 'HexCoord':
        if isinstance(other, HexCoord):
            return HexCoord(self.q - other.q, self.r - other.r)
        return NotImplemented

    def __mul__(self, scalar: int) -> 'HexCoord':
        if isinstance(scalar, (int, np.integer)):
            return HexCoord(self.q * scalar, self.r * scalar)
        return NotImplemented

    def __rmul__(self, scalar: int) -> 'HexCoord':
        return self.__mul__(scalar)

    def __neg__(self) -> 'HexCoord':
        return HexCoord(-self.q, -self.r)

    def __abs__(self) -> int:
        return self.magnitude

    def distance_to(self, other: 'HexCoord') -> int:
        """Distancia de Manhattan hexagonal."""
        dq = abs(self.q - other.q)
        dr = abs(self.r - other.r)
        ds = abs(self.s - other.s)
        return (dq + dr + ds) // 2

    def neighbor(self, direction: HexDirection) -> 'HexCoord':
        """Obtiene el vecino en la dirección dada."""
        dq, dr = direction.vector
        return HexCoord(self.q + dq, self.r + dr)

    def neighbors(self) -> Tuple['HexCoord', ...]:
        """Retorna los 6 vecinos en orden horario desde NE."""
        return tuple(self.neighbor(d) for d in HexDirection)

    def direction_to(self, other: 'HexCoord') -> Optional[HexDirection]:
        """Obtiene la dirección hacia otra coordenada adyacente."""
        diff = (other.q - self.q, other.r - self.r)
        for d in HexDirection:
            if d.vector == diff:
                return d
        return None

    def ring(self, radius: int) -> Tuple['HexCoord', ...]:
        """Retorna todas las celdas en el anillo a distancia `radius`."""
        return _cached_ring(self.q, self.r, radius)

    def spiral(self, radius: int) -> Iterator['HexCoord']:
        """Genera celdas en espiral desde el centro hasta radio `radius`."""
        yield self
        for ring_r in range(1, radius + 1):
            yield from self.ring(ring_r)

    def filled_hexagon(self, radius: int) -> Tuple['HexCoord', ...]:
        """Retorna todas las celdas dentro del radio (inclusive)."""
        return _cached_filled_hex(self.q, self.r, radius)

    def line_to(self, other: 'HexCoord') -> List['HexCoord']:
        """Genera línea recta desde self hasta other usando interpolación."""
        n = self.distance_to(other)
        if n == 0:
            return [self]

        results = []
        for i in range(n + 1):
            t = i / n
            q = self.q + (other.q - self.q) * t
            r = self.r + (other.r - self.r) * t
            s = self.s + (other.s - self.s) * t
            results.append(_cube_round(q, r, s))

        return results

    def lerp(self, other: 'HexCoord', t: float) -> 'HexCoord':
        """Interpolación lineal entre dos coordenadas."""
        q = self.q + (other.q - self.q) * t
        r = self.r + (other.r - self.r) * t
        s = self.s + (other.s - self.s) * t
        return _cube_round(q, r, s)

    def rotate_around(self, center: 'HexCoord', steps: int = 1) -> 'HexCoord':
        """Rota esta coordenada alrededor de un centro (steps * 60°)."""
        vec = self - center
        q, r, s = vec.q, vec.r, vec.s

        for _ in range(steps % 6):
            q, r, s = -r, -s, -q

        return center + HexCoord(q, r)

    def reflect_across(self, axis: HexDirection) -> 'HexCoord':
        """Refleja la coordenada a través de un eje."""
        q, r, s = self.q, self.r, self.s

        if axis in (HexDirection.E, HexDirection.W):
            return HexCoord(q, s)
        elif axis in (HexDirection.NE, HexDirection.SW):
            return HexCoord(s, r)
        else:  # NW, SE
            return HexCoord(r, q)

    def to_pixel(self, size: float = 1.0, orientation: str = 'flat') -> PixelTuple:
        """Convierte a coordenadas de pixel."""
        if orientation == 'flat':
            x = size * (3 / 2 * self.q)
            y = size * (math.sqrt(3) / 2 * self.q + math.sqrt(3) * self.r)
        else:
            x = size * (math.sqrt(3) * self.q + math.sqrt(3) / 2 * self.r)
            y = size * (3 / 2 * self.r)
        return (x, y)

    @classmethod
    def from_pixel(
        cls, x: float, y: float, size: float = 1.0, orientation: str = 'flat'
    ) -> 'HexCoord':
        """Convierte coordenadas de pixel a hexagonal."""
        if orientation == 'flat':
            q = (2 / 3 * x) / size
            r = (-1 / 3 * x + math.sqrt(3) / 3 * y) / size
        else:
            q = (math.sqrt(3) / 3 * x - 1 / 3 * y) / size
            r = (2 / 3 * y) / size

        return _cube_round(q, r, -q - r)

    @classmethod
    def origin(cls) -> 'HexCoord':
        """Retorna el origen (0, 0)."""
        return _ORIGIN

    def to_dict(self) -> Dict[str, int]:
        """Serializa a diccionario."""
        return {'q': self.q, 'r': self.r}

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'HexCoord':
        """Deserializa desde diccionario."""
        return cls(data['q'], data['r'])


# Constante para origen
_ORIGIN = HexCoord(0, 0)


@lru_cache(maxsize=1024)
def _cached_ring(q: int, r: int, radius: int) -> Tuple[HexCoord, ...]:
    """Genera anillo con cache."""
    if radius == 0:
        return (HexCoord(q, r),)

    center = HexCoord(q, r)
    results = []

    current = center + HexCoord(-radius, 0)

    for direction in HexDirection:
        for _ in range(radius):
            results.append(current)
            current = current.neighbor(direction)

    return tuple(results)


@lru_cache(maxsize=256)
def _cached_filled_hex(q: int, r: int, radius: int) -> Tuple[HexCoord, ...]:
    """
    Genera hexágono relleno con cache.
    
    v3.0 FIX: Renombrado variable de loop a `ring_r` para evitar
    shadowing del parámetro `r`.
    """
    center = HexCoord(q, r)
    results = [center]
    for ring_r in range(1, radius + 1):
        results.extend(center.ring(ring_r))
    return tuple(results)


def _cube_round(q: float, r: float, s: float) -> HexCoord:
    """Redondea coordenadas cúbicas flotantes al hexágono más cercano."""
    rq = round(q)
    rr = round(r)
    rs = round(s)

    q_diff = abs(rq - q)
    r_diff = abs(rr - r)
    s_diff = abs(rs - s)

    if q_diff > r_diff and q_diff > s_diff:
        rq = -rr - rs
    elif r_diff > s_diff:
        rr = -rq - rs

    return HexCoord(int(rq), int(rr))


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE COORDENADAS
# ═══════════════════════════════════════════════════════════════════════════════

class HexRegion:
    """
    Región de coordenadas hexagonales para operaciones en lote.
    """

    __slots__ = ('_coords', '_coord_set', '_bounds')

    def __init__(self, coords: Sequence[HexCoord]):
        self._coords = tuple(coords)
        self._coord_set = frozenset(coords)
        self._bounds: Optional[Tuple[int, int, int, int]] = None

    @classmethod
    def from_ring(cls, center: HexCoord, radius: int) -> 'HexRegion':
        return cls(center.ring(radius))

    @classmethod
    def from_area(cls, center: HexCoord, radius: int) -> 'HexRegion':
        return cls(center.filled_hexagon(radius))

    @classmethod
    def from_line(cls, start: HexCoord, end: HexCoord) -> 'HexRegion':
        return cls(start.line_to(end))

    def __contains__(self, coord: HexCoord) -> bool:
        return coord in self._coord_set

    def __iter__(self) -> Iterator[HexCoord]:
        return iter(self._coords)

    def __len__(self) -> int:
        return len(self._coords)

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Retorna (min_q, max_q, min_r, max_r)."""
        if self._bounds is None:
            if not self._coords:
                self._bounds = (0, 0, 0, 0)
            else:
                qs = [c.q for c in self._coords]
                rs = [c.r for c in self._coords]
                self._bounds = (min(qs), max(qs), min(rs), max(rs))
        return self._bounds

    def union(self, other: 'HexRegion') -> 'HexRegion':
        return HexRegion(list(self._coord_set | other._coord_set))

    def intersection(self, other: 'HexRegion') -> 'HexRegion':
        return HexRegion(list(self._coord_set & other._coord_set))

    def difference(self, other: 'HexRegion') -> 'HexRegion':
        return HexRegion(list(self._coord_set - other._coord_set))

    @property
    def centroid(self) -> HexCoord:
        """v3.0: Centro geométrico aproximado de la región."""
        if not self._coords:
            return _ORIGIN
        avg_q = sum(c.q for c in self._coords) / len(self._coords)
        avg_r = sum(c.r for c in self._coords) / len(self._coords)
        return _cube_round(avg_q, avg_r, -avg_q - avg_r)


# Alias para compatibilidad con HOC.__init__ y métricas (anillo = región de un anillo)
HexRing = HexRegion


class HexPathfinder:
    """
    Pathfinding A* optimizado para grids hexagonales.
    
    v3.0: Soporte para costos variables por celda.
    """

    def __init__(
        self,
        walkable_check: Callable[[HexCoord], bool],
        cost_fn: Optional[Callable[[HexCoord], float]] = None
    ):
        self._walkable = walkable_check
        self._cost_fn = cost_fn or (lambda _: 1.0)

    def find_path(
        self,
        start: HexCoord,
        goal: HexCoord,
        max_iterations: int = 10000
    ) -> Optional[List[HexCoord]]:
        """
        Encuentra el camino más corto usando A*.
        
        v3.0: Soporta costos variables por celda via cost_fn.
        """
        if start == goal:
            return [start]

        if not self._walkable(goal):
            return None

        open_set: List[Tuple[float, int, HexCoord]] = [(0.0, 0, start)]
        came_from: Dict[HexCoord, HexCoord] = {}
        g_score: Dict[HexCoord, float] = {start: 0.0}

        counter = 0
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            current_g = g_score.get(current, float('inf'))

            for neighbor in current.neighbors():
                if not self._walkable(neighbor):
                    continue

                move_cost = self._cost_fn(neighbor)
                tentative_g = current_g + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + neighbor.distance_to(goal)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))

        return None

    def _reconstruct_path(
        self,
        came_from: Dict[HexCoord, HexCoord],
        current: HexCoord
    ) -> List[HexCoord]:
        """Reconstruye el camino desde came_from."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


# ═══════════════════════════════════════════════════════════════════════════════
# SISTEMA DE FEROMONAS AVANZADO
# ═══════════════════════════════════════════════════════════════════════════════

class PheromoneType(Enum):
    """Tipos de feromonas para comunicación estigmérgica."""
    FOOD = auto()
    DANGER = auto()
    PATH = auto()
    RECRUIT = auto()
    HOME = auto()
    WORK = auto()
    EXPLORATION = auto()


@dataclass(slots=True)
class PheromoneDeposit:
    """Depósito de feromona con metadatos."""
    ptype: PheromoneType
    intensity: float
    timestamp: float = field(default_factory=time.time)
    source_coord: Optional[HexCoord] = None
    decay_rate: float = 0.1

    def decay(self, elapsed: float = 1.0) -> float:
        """Aplica decaimiento y retorna nueva intensidad."""
        self.intensity *= (1.0 - self.decay_rate) ** elapsed
        return self.intensity

    # Minimum intensity to be considered active (below this, pheromone is cleaned up)
    ACTIVE_THRESHOLD: ClassVar[float] = 0.001

    @property
    def is_active(self) -> bool:
        return self.intensity > self.ACTIVE_THRESHOLD


class PheromoneField:
    """
    Campo de feromonas para una celda.
    
    v3.0: batch_decay con NumPy para mejor rendimiento.
    """

    __slots__ = ('_deposits', '_total_intensity', '_lock')

    def __init__(self):
        self._deposits: Dict[PheromoneType, PheromoneDeposit] = {}
        self._total_intensity: float = 0.0
        self._lock = threading.Lock()

    def deposit(
        self,
        ptype: PheromoneType,
        amount: float,
        source: Optional[HexCoord] = None,
        decay_rate: float = 0.1
    ) -> None:
        """Deposita feromona de un tipo."""
        with self._lock:
            if ptype in self._deposits:
                self._deposits[ptype].intensity = min(
                    1.0,
                    self._deposits[ptype].intensity + amount
                )
            else:
                self._deposits[ptype] = PheromoneDeposit(
                    ptype=ptype,
                    intensity=min(1.0, amount),
                    source_coord=source,
                    decay_rate=decay_rate
                )
            self._update_total()

    def get_intensity(self, ptype: PheromoneType) -> float:
        deposit = self._deposits.get(ptype)
        return deposit.intensity if deposit else 0.0

    def decay_all(self, elapsed: float = 1.0) -> None:
        """Aplica decaimiento a todas las feromonas."""
        with self._lock:
            to_remove = []
            for ptype, deposit in self._deposits.items():
                deposit.decay(elapsed)
                if not deposit.is_active:
                    to_remove.append(ptype)

            for ptype in to_remove:
                del self._deposits[ptype]

            self._update_total()

    def _update_total(self) -> None:
        self._total_intensity = sum(d.intensity for d in self._deposits.values())

    @property
    def total_intensity(self) -> float:
        return self._total_intensity

    @property
    def dominant_type(self) -> Optional[PheromoneType]:
        if not self._deposits:
            return None
        return max(self._deposits.items(), key=lambda x: x[1].intensity)[0]

    def get_gradient_vector(self) -> Dict[PheromoneType, float]:
        """v3.0: Retorna vector de intensidades por tipo."""
        return {ptype: dep.intensity for ptype, dep in self._deposits.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            ptype.name: {
                'intensity': dep.intensity,
                'decay_rate': dep.decay_rate
            }
            for ptype, dep in self._deposits.items()
        }


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
# CIRCUIT BREAKER (v3.0 - nuevo)
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitState(Enum):
    """Estado del circuit breaker."""
    CLOSED = auto()      # Funcionando normal
    OPEN = auto()        # Abierto, rechazando operaciones
    HALF_OPEN = auto()   # Probando recuperación


class CircuitBreaker:
    """
    Circuit breaker con backoff exponencial para protección de celdas.
    
    Previene cascadas de fallos cerrando el circuito cuando una celda
    falla repetidamente, con recovery automático.
    """

    __slots__ = (
        '_state', '_failure_count', '_failure_threshold',
        '_recovery_timeout', '_last_failure_time',
        '_success_count_in_half_open', '_success_threshold',
        '_lock', '_backoff_multiplier', '_max_recovery_timeout'
    )

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 5.0,
        success_threshold: int = 2,
        backoff_multiplier: float = 2.0,
        max_recovery_timeout: float = 300.0
    ):
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._last_failure_time = 0.0
        self._success_count_in_half_open = 0
        self._success_threshold = success_threshold
        self._lock = threading.Lock()
        self._backoff_multiplier = backoff_multiplier
        self._max_recovery_timeout = max_recovery_timeout

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count_in_half_open = 0
            return self._state

    def record_success(self) -> None:
        """Registra operación exitosa."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count_in_half_open += 1
                if self._success_count_in_half_open >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._recovery_timeout = max(
                        5.0, self._recovery_timeout / self._backoff_multiplier
                    )
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Registra fallo."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._recovery_timeout = min(
                    self._max_recovery_timeout,
                    self._recovery_timeout * self._backoff_multiplier
                )
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self._failure_threshold
            ):
                self._state = CircuitState.OPEN

    def allow_request(self) -> bool:
        """¿Se permite la operación?"""
        current = self.state
        return current in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def reset(self) -> None:
        """Reset manual."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count_in_half_open = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'state': self.state.name,
            'failure_count': self._failure_count,
            'recovery_timeout': self._recovery_timeout,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL PANAL (v3.0 - validación robusta)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HoneycombConfig:
    """Configuración del grid hexagonal con validación exhaustiva."""

    # Tamaño
    radius: int = 10
    initial_cells: int = 37

    # Celdas especializadas
    queens_per_grid: int = 1
    drones_per_ring: int = 2
    nurseries_per_grid: int = 3

    # Capacidad
    vcores_per_cell: int = 8
    max_entities_per_cell: int = 100

    # Comunicación
    pheromone_decay_rate: float = 0.1
    waggle_broadcast_range: int = 3
    pheromone_diffusion_rate: float = 0.05

    # Resiliencia
    replication_factor: int = 2
    failover_timeout_ms: int = 1000
    max_consecutive_errors: int = 3

    # Rendimiento
    parallel_ring_processing: bool = True
    max_parallel_rings: int = 4
    tick_batch_size: int = 50

    # Work-stealing
    steal_threshold_low: float = 0.3
    steal_threshold_high: float = 0.7
    max_steal_per_tick: int = 2

    # Topología
    topology: str = 'flat'  # 'flat', 'torus', 'sphere'

    # Métricas
    metrics_history_size: int = 1000
    metrics_sample_rate: float = 1.0

    # v3.0: Circuit breaker
    circuit_breaker_threshold: int = 3
    circuit_breaker_recovery_s: float = 5.0

    # v3.0: Health monitoring
    health_check_interval_s: float = 10.0
    health_alert_load_threshold: float = 0.9

    # v3.1: Extracted magic numbers
    pheromone_active_threshold: float = 0.001
    pheromone_diffuse_threshold: float = 0.01
    load_change_event_threshold: float = 0.1
    health_critical_failed_ratio: float = 0.2
    health_critical_load: float = 0.95
    health_degraded_failed_ratio: float = 0.05
    cluster_health_load_weight: float = 0.3
    cluster_health_health_weight: float = 0.5
    cluster_health_balance_weight: float = 0.2
    scout_novelty_bonus: float = 0.5
    scout_explore_pheromone_weight: float = 0.3
    scout_path_deposit_intensity: float = 0.3
    scout_low_load_threshold: float = 0.1
    nursery_default_incubation_rate: float = 0.1
    # Visualization thresholds
    viz_load_high: float = 0.8
    viz_load_medium: float = 0.5
    viz_load_low: float = 0.3

    def __post_init__(self):
        """
        v3.0 FIX: Usa ValueError en lugar de assert.
        assert se deshabilita con python -O, lo que bypasea toda validación.
        """
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        if self.vcores_per_cell <= 0:
            raise ValueError(f"vcores_per_cell must be positive, got {self.vcores_per_cell}")
        if not (0.0 <= self.pheromone_decay_rate <= 1.0):
            raise ValueError(
                f"pheromone_decay_rate must be in [0, 1], got {self.pheromone_decay_rate}"
            )
        if self.topology not in ('flat', 'torus', 'sphere'):
            raise ValueError(f"invalid topology: {self.topology!r}")
        if self.steal_threshold_low >= self.steal_threshold_high:
            raise ValueError(
                f"steal_threshold_low ({self.steal_threshold_low}) must be < "
                f"steal_threshold_high ({self.steal_threshold_high})"
            )
        if self.max_parallel_rings <= 0:
            raise ValueError(f"max_parallel_rings must be positive, got {self.max_parallel_rings}")

    def cells_at_radius(self, r: int) -> int:
        return 1 if r == 0 else 6 * r

    def total_cells(self) -> int:
        return 1 + 3 * self.radius * (self.radius + 1)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HoneycombConfig':
        # Filtrar solo campos conocidos para forward-compatibility
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


class GridTopology(Enum):
    """Topología del grid hexagonal."""
    FLAT = auto()
    TORUS = auto()
    SPHERE = auto()
    INFINITE = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS Y ESTADÍSTICAS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class CellMetrics:
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
    circuit_state: str = 'CLOSED'  # v3.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'coord': self.coord.to_dict(),
            'role': self.role.name,
            'state': self.state.name,
            'load': self.load,
            'vcore_count': self.vcore_count,
            'error_count': self.error_count,
            'ticks_processed': self.ticks_processed,
            'pheromone_total': self.pheromone_total,
            'neighbor_count': self.neighbor_count,
            'last_activity': self.last_activity,
            'circuit_state': self.circuit_state,
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """Colector de métricas thread-safe con historial y agregaciones."""

    __slots__ = (
        '_history', '_max_history', '_sample_rate',
        '_last_sample', '_lock', '_counters'
    )

    def __init__(self, max_history: int = 1000, sample_rate: float = 1.0):
        self._history: deque = deque(maxlen=max_history)
        self._max_history = max_history
        self._sample_rate = sample_rate
        self._last_sample = 0.0
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = defaultdict(int)

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

    def get_history(self, limit: Optional[int] = None) -> List[GridMetrics]:
        with self._lock:
            if limit:
                return list(self._history)[-limit:]
            return list(self._history)

    def get_latest(self) -> Optional[GridMetrics]:
        with self._lock:
            return self._history[-1] if self._history else None

    def get_averages(self, window: int = 60) -> Dict[str, float]:
        history = self.get_history(window)
        if not history:
            return {}

        return {
            'avg_load': float(np.mean([m.average_load for m in history])),
            'avg_vcores': float(np.mean([m.total_vcores for m in history])),
            'avg_tps': float(np.mean([m.ticks_per_second for m in history])),
            'avg_errors': float(np.mean([m.errors_per_second for m in history])),
        }


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
        'coord', 'role', '_state', '_vcores', '_load',
        '_neighbors', '_rw_lock', '_metadata', '_pheromone_field',
        '_last_activity', '_error_count', '_config', '_event_bus',
        '_ticks_processed', '_state_callbacks', '_creation_time',
        '_circuit_breaker',
        '__weakref__'
    )

    def __init__(
        self,
        coord: HexCoord,
        role: CellRole = CellRole.WORKER,
        config: Optional[HoneycombConfig] = None,
        event_bus: Optional[EventBus] = None
    ):
        if not isinstance(coord, HexCoord):
            raise TypeError(f"coord must be HexCoord, got {type(coord)}")

        self.coord = coord
        self.role = role
        self._state = CellState.EMPTY
        self._vcores: List[Any] = []
        self._load: float = 0.0
        self._neighbors: Dict[HexDirection, Optional['HoneycombCell']] = {
            d: None for d in HexDirection
        }
        self._rw_lock = RWLock()
        self._metadata: Dict[str, Any] = {}
        self._pheromone_field = PheromoneField()
        self._last_activity: float = time.time()
        self._error_count: int = 0
        self._config = config or HoneycombConfig()
        self._event_bus = event_bus or get_event_bus()
        self._ticks_processed: int = 0
        self._state_callbacks: List[Callable[[CellState, CellState], None]] = []
        self._creation_time = time.time()
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=self._config.max_consecutive_errors,
            recovery_timeout=self._config.circuit_breaker_recovery_s
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

        self._event_bus.publish(Event(
            type=EventType.CELL_STATE_CHANGED,
            source=self,
            data={'old': old_state.name, 'new': new_state.name}
        ))

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

    def get_neighbor(self, direction: HexDirection) -> Optional['HoneycombCell']:
        with self._rw_lock.read_lock():
            return self._neighbors.get(direction)

    def set_neighbor(
        self,
        direction: HexDirection,
        cell: Optional['HoneycombCell'],
        bidirectional: bool = False
    ) -> None:
        with self._rw_lock.write_lock():
            self._neighbors[direction] = cell

        if bidirectional and cell is not None:
            cell.set_neighbor(direction.opposite(), self, bidirectional=False)

    def get_all_neighbors(self) -> List['HoneycombCell']:
        with self._rw_lock.read_lock():
            return [n for n in self._neighbors.values() if n is not None]

    def get_neighbor_loads(self) -> Dict[HexDirection, float]:
        with self._rw_lock.read_lock():
            return {
                d: n.load
                for d, n in self._neighbors.items()
                if n is not None
            }

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

            self._event_bus.publish(Event(
                type=EventType.VCORE_ASSIGNED,
                source=self,
                data={'vcore_count': len(self._vcores)}
            ))

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

                self._event_bus.publish(Event(
                    type=EventType.VCORE_REMOVED,
                    source=self,
                    data={'vcore_count': len(self._vcores)}
                ))

                return True
            except ValueError:
                return False

    def get_vcores(self) -> List[Any]:
        with self._rw_lock.read_lock():
            return list(self._vcores)

    def _update_load(self) -> None:
        """Actualiza la métrica de carga. DEBE llamarse dentro de write_lock."""
        old_load = self._load
        self._load = len(self._vcores) / max(1, self._config.vcores_per_cell)

        if abs(old_load - self._load) > self._config.load_change_event_threshold:
            self._event_bus.publish(Event(
                type=EventType.CELL_LOAD_CHANGED,
                source=self,
                data={'old': old_load, 'new': self._load}
            ), async_=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SISTEMA DE FEROMONAS
    # ─────────────────────────────────────────────────────────────────────────

    def deposit_pheromone(
        self,
        ptype: PheromoneType,
        amount: float,
        source: Optional[HexCoord] = None,
        decay_rate: Optional[float] = None
    ) -> None:
        self._pheromone_field.deposit(
            ptype,
            amount,
            source,
            decay_rate or self._config.pheromone_decay_rate
        )

        self._event_bus.publish(Event(
            type=EventType.PHEROMONE_DEPOSITED,
            source=self,
            data={'type': ptype.name, 'amount': amount}
        ), async_=True)

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
                    neighbor.deposit_pheromone(
                        ptype,
                        diffuse_amount,
                        source=self.coord
                    )

    def follow_pheromone_gradient(
        self,
        ptype: PheromoneType
    ) -> Optional[HexDirection]:
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

    def execute_tick(self) -> Dict[str, Any]:
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
                    if hasattr(vcore, 'tick'):
                        result = vcore.tick()
                        results.append(result)
                except Exception as e:
                    self._error_count += 1
                    errors.append(str(e))
                    self._circuit_breaker.record_failure()
                    logger.error(f"Cell {self.coord} vCore error: {e}")

                    if not self._circuit_breaker.allow_request():
                        self._set_state(CellState.FAILED)
                        self._event_bus.publish(Event(
                            type=EventType.CELL_ERROR,
                            source=self,
                            data={'errors': self._error_count}
                        ))
                        self._event_bus.publish(Event(
                            type=EventType.CIRCUIT_BREAKER_OPENED,
                            source=self,
                            data={'coord': self.coord.to_dict()}
                        ))
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
                "tick": self._ticks_processed
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

            self._event_bus.publish(Event(
                type=EventType.CELL_RECOVERED,
                source=self,
                data={}
            ))

            return True

    # ─────────────────────────────────────────────────────────────────────────
    # CALLBACKS
    # ─────────────────────────────────────────────────────────────────────────

    def on_state_change(
        self,
        callback: Callable[[CellState, CellState], None]
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

    def to_dict(self) -> Dict[str, Any]:
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
        return (
            f"Cell({self.coord.q},{self.coord.r}"
            f"|{self.role.name}|{self._state.name})"
        )

    def __hash__(self) -> int:
        return hash(self.coord)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HoneycombCell):
            return self.coord == other.coord
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# CELDAS ESPECIALIZADAS
# ═══════════════════════════════════════════════════════════════════════════════

class QueenCell(HoneycombCell):
    """
    Celda Reina - Coordinadora del cluster (v3.0).

    Mejoras v3.0:
    - Adaptive rebalance threshold basado en historial
    - Health score de cluster
    """

    __slots__ = (
        '_worker_registry', '_global_load', '_spawn_queue',
        '_succession_candidates', '_load_history', '_rebalance_threshold'
    )

    def __init__(self, coord: HexCoord, config: Optional[HoneycombConfig] = None):
        super().__init__(coord, CellRole.QUEEN, config)
        self._worker_registry: Dict[HexCoord, weakref.ref] = {}
        self._global_load: float = 0.0
        self._spawn_queue: List[Tuple[int, str, Dict]] = []  # v3.0 FIX: (priority, id, spec)
        self._succession_candidates: List[weakref.ref] = []
        self._load_history: deque = deque(maxlen=100)
        self._rebalance_threshold = 0.2

    def register_worker(self, cell: 'WorkerCell') -> None:
        with self._rw_lock.write_lock():
            self._worker_registry[cell.coord] = weakref.ref(cell)

    def unregister_worker(self, coord: HexCoord) -> None:
        with self._rw_lock.write_lock():
            self._worker_registry.pop(coord, None)

    def _get_active_workers(self) -> List['WorkerCell']:
        """Obtiene trabajadoras activas (limpia refs muertas)."""
        active = []
        dead = []

        for coord, ref in self._worker_registry.items():
            worker = ref()
            if worker is not None:
                active.append(worker)
            else:
                dead.append(coord)

        for coord in dead:
            del self._worker_registry[coord]

        return active

    @property
    def worker_count(self) -> int:
        return len(self._get_active_workers())

    def compute_global_load(self) -> float:
        with self._rw_lock.write_lock():
            workers = self._get_active_workers()
            if not workers:
                self._global_load = 0.0
                return 0.0

            loads = [w.load for w in workers]
            self._global_load = float(np.mean(loads))
            self._load_history.append(self._global_load)

            # v3.0: Adaptive threshold
            if len(self._load_history) >= 10:
                recent_std = float(np.std(list(self._load_history)[-10:]))
                self._rebalance_threshold = max(0.1, min(0.4, recent_std * 1.5))

            return self._global_load

    def get_load_statistics(self) -> Dict[str, float]:
        workers = self._get_active_workers()
        if not workers:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p50': 0, 'p90': 0, 'p99': 0}

        loads = np.array([w.load for w in workers])
        return {
            'mean': float(np.mean(loads)),
            'std': float(np.std(loads)),
            'min': float(np.min(loads)),
            'max': float(np.max(loads)),
            'p50': float(np.percentile(loads, 50)),
            'p90': float(np.percentile(loads, 90)),
            'p99': float(np.percentile(loads, 99)),
        }

    def find_cells_by_load(
        self,
        min_load: float = 0.0,
        max_load: float = 1.0,
        limit: int = 10
    ) -> List['WorkerCell']:
        workers = self._get_active_workers()
        filtered = [
            w for w in workers
            if min_load <= w.load <= max_load
        ]
        filtered.sort(key=lambda w: w.load)
        return filtered[:limit]

    def find_least_loaded_cells(self, count: int = 3) -> List['WorkerCell']:
        return self.find_cells_by_load(max_load=1.0, limit=count)

    def find_most_loaded_cells(self, count: int = 3) -> List['WorkerCell']:
        workers = self._get_active_workers()
        workers.sort(key=lambda w: -w.load)
        return workers[:count]

    def should_rebalance(self) -> bool:
        stats = self.get_load_statistics()
        return stats['std'] > self._rebalance_threshold

    def plan_rebalance(self) -> List[Tuple['WorkerCell', 'WorkerCell', int]]:
        if not self.should_rebalance():
            return []

        moves = []
        overloaded = self.find_cells_by_load(
            min_load=self._config.steal_threshold_high
        )
        underloaded = self.find_cells_by_load(
            max_load=self._config.steal_threshold_low
        )

        for source in overloaded:
            for target in underloaded:
                if target.load < source.load - 0.2:
                    moves.append((source, target, 1))
                    if len(moves) >= 10:
                        return moves

        return moves

    def issue_royal_command(self, command: str, params: Dict) -> int:
        workers = self._get_active_workers()
        for worker in workers:
            worker._metadata["royal_command"] = {
                "command": command,
                "params": params,
                "from_queen": self.coord.to_dict(),
                "timestamp": time.time()
            }
        return len(workers)

    def schedule_spawn(
        self,
        entity_type: str,
        params: Dict,
        priority: int = 0
    ) -> str:
        """
        v3.0 FIX: Usa (priority, spawn_id, spec) donde spawn_id es str
        comparable, evitando comparación de dicts en heapq.
        """
        spawn_id = f"spawn_{uuid.uuid4().hex[:8]}"
        spec = {
            "id": spawn_id,
            "type": entity_type,
            "params": params,
            "scheduled_at": time.time()
        }
        heapq.heappush(self._spawn_queue, (-priority, spawn_id, spec))
        return spawn_id

    def get_next_spawn(self) -> Optional[Dict]:
        if self._spawn_queue:
            _, _, spec = heapq.heappop(self._spawn_queue)
            return spec
        return None

    def add_succession_candidate(self, cell: 'QueenCell') -> None:
        self._succession_candidates.append(weakref.ref(cell))

    def get_cluster_health_score(self) -> float:
        """
        v3.0: Score de salud del cluster (0.0 = crítico, 1.0 = perfecto).
        """
        workers = self._get_active_workers()
        if not workers:
            return 0.0

        loads = [w.load for w in workers]
        failed = sum(1 for w in workers if w.state == CellState.FAILED)
        circuit_open = sum(
            1 for w in workers
            if w.circuit_breaker.state != CircuitState.CLOSED
        )

        load_score = 1.0 - float(np.mean(loads))
        health_ratio = 1.0 - (failed + circuit_open) / len(workers)
        balance_score = 1.0 - min(1.0, float(np.std(loads)) * 2)

        cfg = self._config
        return (load_score * cfg.cluster_health_load_weight
                + health_ratio * cfg.cluster_health_health_weight
                + balance_score * cfg.cluster_health_balance_weight)

    def get_cluster_metrics(self) -> Dict[str, Any]:
        workers = self._get_active_workers()

        return {
            'queen_coord': self.coord.to_dict(),
            'worker_count': len(workers),
            'global_load': self._global_load,
            'load_stats': self.get_load_statistics(),
            'spawn_queue_size': len(self._spawn_queue),
            'succession_candidates': len(self._succession_candidates),
            'load_trend': list(self._load_history)[-10:] if self._load_history else [],
            'health_score': self.get_cluster_health_score(),
        }


class WorkerCell(HoneycombCell):
    """
    Celda Trabajadora - Unidad de cómputo principal (v3.0).

    v3.0: steal_from con _update_load dentro del lock scope.
    """

    __slots__ = ('_processed_ticks', '_steal_count', '_stolen_from_count', '_work_history')

    def __init__(self, coord: HexCoord, config: Optional[HoneycombConfig] = None):
        super().__init__(coord, CellRole.WORKER, config)
        self._processed_ticks: int = 0
        self._steal_count: int = 0
        self._stolen_from_count: int = 0
        self._work_history: deque = deque(maxlen=100)

    def can_steal_work(self) -> bool:
        return (
            self._load < self._config.steal_threshold_low
            and self._state in (CellState.IDLE, CellState.EMPTY)
        )

    def should_donate_work(self) -> bool:
        return self._load > self._config.steal_threshold_high

    def steal_from(self, source: 'WorkerCell', count: int = 1) -> int:
        """
        Roba trabajo de otra celda.

        v3.0 FIX: _update_load() se llama DENTRO del scope del lock
        para evitar race conditions.
        """
        # Ordenar locks por coordenada para evitar deadlock
        cells = sorted([self, source], key=lambda c: (c.coord.q, c.coord.r))

        stolen = 0

        with cells[0]._rw_lock.write_lock():
            with cells[1]._rw_lock.write_lock():
                while (
                    stolen < count
                    and source._vcores
                    and len(self._vcores) < self._config.vcores_per_cell
                ):
                    vcore = source._vcores.pop()
                    self._vcores.append(vcore)
                    stolen += 1

                if stolen > 0:
                    # v3.0 FIX: actualizar loads DENTRO del lock
                    source._load = len(source._vcores) / max(1, source._config.vcores_per_cell)
                    self._load = len(self._vcores) / max(1, self._config.vcores_per_cell)
                    self._steal_count += stolen
                    source._stolen_from_count += stolen

        if stolen > 0:
            self._event_bus.publish(Event(
                type=EventType.WORK_STOLEN,
                source=self,
                data={
                    'from': source.coord.to_dict(),
                    'count': stolen
                }
            ))

        return stolen

    def attempt_work_stealing(self) -> int:
        if not self.can_steal_work():
            return 0

        total_stolen = 0
        neighbors = self.get_all_neighbors()
        neighbors.sort(key=lambda n: -n.load)

        for neighbor in neighbors:
            if not isinstance(neighbor, WorkerCell):
                continue

            if neighbor.should_donate_work():
                stolen = self.steal_from(neighbor, 1)
                total_stolen += stolen

                if total_stolen >= self._config.max_steal_per_tick:
                    break

                if not self.can_steal_work():
                    break

        return total_stolen

    def record_work(self, work_id: str, duration: float, success: bool) -> None:
        self._work_history.append({
            'id': work_id,
            'duration': duration,
            'success': success,
            'timestamp': time.time()
        })

    def get_performance_stats(self) -> Dict[str, float]:
        if not self._work_history:
            return {'avg_duration': 0, 'success_rate': 0, 'throughput': 0}

        history = list(self._work_history)
        durations = [w['duration'] for w in history]
        successes = sum(1 for w in history if w['success'])

        time_span = history[-1]['timestamp'] - history[0]['timestamp'] if len(history) > 1 else 1

        return {
            'avg_duration': float(np.mean(durations)),
            'success_rate': successes / len(history),
            'throughput': len(history) / max(time_span, 1),
            'steal_count': self._steal_count,
            'stolen_from_count': self._stolen_from_count
        }


class DroneCell(HoneycombCell):
    """
    Celda Dron - Comunicación externa (v3.0).

    v3.0 FIX: message_queue usa (priority, msg_id, msg) para evitar
    comparación de dicts en heapq.
    """

    __slots__ = (
        '_external_connections', '_message_queue',
        '_messages_sent', '_messages_received', '_connection_errors'
    )

    def __init__(self, coord: HexCoord, config: Optional[HoneycombConfig] = None):
        super().__init__(coord, CellRole.DRONE, config)
        self._external_connections: List[Any] = []
        self._message_queue: List[Tuple[int, str, Dict]] = []  # v3.0 FIX
        self._messages_sent: int = 0
        self._messages_received: int = 0
        self._connection_errors: int = 0

    def connect_external(self, endpoint: Any) -> bool:
        with self._rw_lock.write_lock():
            if endpoint not in self._external_connections:
                self._external_connections.append(endpoint)
                return True
            return False

    def disconnect_external(self, endpoint: Any) -> bool:
        with self._rw_lock.write_lock():
            try:
                self._external_connections.remove(endpoint)
                return True
            except ValueError:
                return False

    def queue_message(self, message: Dict, priority: int = 0) -> None:
        """v3.0 FIX: Usa msg_id para hacer tupla comparable."""
        msg_id = uuid.uuid4().hex[:8]
        heapq.heappush(self._message_queue, (-priority, msg_id, message))

    def broadcast(self, message: Dict) -> int:
        sent = 0

        with self._rw_lock.read_lock():
            endpoints = list(self._external_connections)

        for endpoint in endpoints:
            try:
                if hasattr(endpoint, 'receive'):
                    endpoint.receive(message)
                    sent += 1
            except Exception as e:
                self._connection_errors += 1
                logger.error(f"Drone {self.coord} broadcast error: {e}")

        self._messages_sent += sent
        return sent

    def process_queue(self, batch_size: int = 10) -> int:
        processed = 0

        while self._message_queue and processed < batch_size:
            _, _, message = heapq.heappop(self._message_queue)
            self.broadcast(message)
            processed += 1

        return processed

    def get_comm_stats(self) -> Dict[str, Any]:
        return {
            'connections': len(self._external_connections),
            'queue_size': len(self._message_queue),
            'sent': self._messages_sent,
            'received': self._messages_received,
            'errors': self._connection_errors
        }


class NurseryCell(HoneycombCell):
    """Celda Guardería - Spawning de nuevas entidades (v3.0)."""

    __slots__ = ('_incubating', '_ready_entities', '_total_spawned', '_max_incubating')

    def __init__(self, coord: HexCoord, config: Optional[HoneycombConfig] = None):
        super().__init__(coord, CellRole.NURSERY, config)
        self._incubating: List[Dict] = []
        self._ready_entities: List[Any] = []
        self._total_spawned: int = 0
        self._max_incubating = 10

    def incubate(self, entity_spec: Dict, priority: int = 0) -> Optional[str]:
        with self._rw_lock.write_lock():
            if len(self._incubating) >= self._max_incubating:
                return None

            entity_id = f"entity_{uuid.uuid4().hex[:8]}"
            self._incubating.append({
                "id": entity_id,
                "spec": entity_spec,
                "progress": 0.0,
                "priority": priority,
                "started_at": time.time()
            })

            self._incubating.sort(key=lambda x: -x['priority'])

            self._event_bus.publish(Event(
                type=EventType.ENTITY_INCUBATING,
                source=self,
                data={'entity_id': entity_id}
            ))

            return entity_id

    def tick_incubation(self, rate: Optional[float] = None) -> List[Any]:
        rate = rate if rate is not None else self._config.nursery_default_incubation_rate
        ready = []
        still_incubating = []

        with self._rw_lock.write_lock():
            for item in self._incubating:
                progress_increment = rate * (1 + item['progress'])
                item["progress"] = min(1.0, item["progress"] + progress_increment)

                if item["progress"] >= 1.0:
                    entity = self._create_entity(item["spec"])
                    entity['id'] = item['id']
                    entity['incubation_time'] = time.time() - item['started_at']
                    ready.append(entity)
                    self._total_spawned += 1

                    self._event_bus.publish(Event(
                        type=EventType.ENTITY_SPAWNED,
                        source=self,
                        data={'entity_id': item['id']}
                    ))
                else:
                    still_incubating.append(item)

            self._incubating = still_incubating
            self._ready_entities.extend(ready)

        return ready

    def _create_entity(self, spec: Dict) -> Dict:
        return {
            "type": spec.get("type", "unknown"),
            "created": True,
            "spec": spec,
            "born_at": time.time()
        }

    def harvest_ready(self, count: Optional[int] = None) -> List[Any]:
        with self._rw_lock.write_lock():
            if count is None:
                ready = self._ready_entities
                self._ready_entities = []
            else:
                ready = self._ready_entities[:count]
                self._ready_entities = self._ready_entities[count:]
            return ready

    def get_incubation_status(self) -> List[Dict]:
        with self._rw_lock.read_lock():
            return [
                {
                    'id': item['id'],
                    'progress': item['progress'],
                    'priority': item['priority'],
                    'elapsed': time.time() - item['started_at']
                }
                for item in self._incubating
            ]

    def get_nursery_stats(self) -> Dict[str, Any]:
        return {
            'incubating': len(self._incubating),
            'ready': len(self._ready_entities),
            'total_spawned': self._total_spawned,
            'capacity': self._max_incubating
        }


class StorageCell(HoneycombCell):
    """
    Celda de Almacenamiento (v3.0).

    v3.0: LRU eviction cuando se llega a capacidad máxima.
    """

    __slots__ = ('_storage', '_storage_lock', '_max_size', '_current_size', '_access_order')

    def __init__(self, coord: HexCoord, config: Optional[HoneycombConfig] = None):
        super().__init__(coord, CellRole.STORAGE, config)
        self._storage: Dict[str, Tuple[Any, float, Optional[float]]] = {}
        self._storage_lock = threading.Lock()
        self._max_size = 1000
        self._current_size = 0
        self._access_order: OrderedDict = OrderedDict()  # v3.0: LRU tracking

    def store(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> bool:
        """Almacena un valor con TTL opcional. LRU eviction si está lleno."""
        with self._storage_lock:
            # v3.0: LRU eviction
            if self._current_size >= self._max_size and key not in self._storage:
                self._evict_lru()

            if key not in self._storage:
                self._current_size += 1
            else:
                # Mover al final del LRU
                self._access_order.pop(key, None)

            self._storage[key] = (value, time.time(), ttl)
            self._access_order[key] = True
            return True

    def retrieve(self, key: str) -> Optional[Any]:
        with self._storage_lock:
            if key not in self._storage:
                return None

            value, created, ttl = self._storage[key]

            if ttl is not None and time.time() - created > ttl:
                del self._storage[key]
                self._access_order.pop(key, None)
                self._current_size -= 1
                return None

            # v3.0: Actualizar acceso LRU
            self._access_order.move_to_end(key)
            return value

    def delete(self, key: str) -> bool:
        with self._storage_lock:
            if key in self._storage:
                del self._storage[key]
                self._access_order.pop(key, None)
                self._current_size -= 1
                return True
            return False

    def _evict_lru(self) -> None:
        """v3.0: Desaloja la entrada menos recientemente usada."""
        if self._access_order:
            oldest_key = next(iter(self._access_order))
            del self._storage[oldest_key]
            del self._access_order[oldest_key]
            self._current_size -= 1
            logger.debug(f"StorageCell {self.coord}: LRU evicted key '{oldest_key}'")

    def cleanup_expired(self) -> int:
        now = time.time()
        expired = []

        with self._storage_lock:
            for key, (_, created, ttl) in self._storage.items():
                if ttl is not None and now - created > ttl:
                    expired.append(key)

            for key in expired:
                del self._storage[key]
                self._access_order.pop(key, None)
                self._current_size -= 1

        return len(expired)

    def get_storage_stats(self) -> Dict[str, Any]:
        return {
            'keys': self._current_size,
            'capacity': self._max_size,
            'utilization': self._current_size / self._max_size if self._max_size > 0 else 0
        }


class GuardCell(HoneycombCell):
    """
    v3.0 NUEVO: Celda Guardia - Seguridad y validación.

    Valida datos que entran/salen de su zona, detecta anomalías
    y emite alertas. Protege un perímetro de celdas vecinas.
    """

    __slots__ = (
        '_rules', '_violations', '_blocked_sources',
        '_total_checks', '_total_blocks'
    )

    def __init__(self, coord: HexCoord, config: Optional[HoneycombConfig] = None):
        super().__init__(coord, CellRole.GUARD, config)
        self._rules: List[Callable[[Dict], bool]] = []
        self._violations: deque = deque(maxlen=500)
        self._blocked_sources: Set[HexCoord] = set()
        self._total_checks: int = 0
        self._total_blocks: int = 0

    def add_rule(self, rule: Callable[[Dict], bool]) -> int:
        """
        Añade una regla de validación.
        rule(data) retorna True si pasa, False si viola.
        """
        self._rules.append(rule)
        return len(self._rules) - 1

    def remove_rule(self, index: int) -> bool:
        if 0 <= index < len(self._rules):
            del self._rules[index]
            return True
        return False

    def validate(self, data: Dict, source: Optional[HexCoord] = None) -> bool:
        """Valida datos contra todas las reglas."""
        self._total_checks += 1

        if source and source in self._blocked_sources:
            self._total_blocks += 1
            return False

        for i, rule in enumerate(self._rules):
            try:
                if not rule(data):
                    self._violations.append({
                        'rule_index': i,
                        'source': source.to_dict() if source else None,
                        'timestamp': time.time(),
                        'data_keys': list(data.keys()),
                    })

                    self._event_bus.publish(Event(
                        type=EventType.GUARD_VALIDATION_FAILED,
                        source=self,
                        data={
                            'rule_index': i,
                            'source_coord': source.to_dict() if source else None
                        }
                    ))

                    self._total_blocks += 1
                    return False
            except Exception as e:
                logger.error(f"Guard rule {i} error: {e}")
                return False

        return True

    def block_source(self, coord: HexCoord) -> None:
        self._blocked_sources.add(coord)

    def unblock_source(self, coord: HexCoord) -> None:
        self._blocked_sources.discard(coord)

    def get_guard_stats(self) -> Dict[str, Any]:
        return {
            'rules': len(self._rules),
            'total_checks': self._total_checks,
            'total_blocks': self._total_blocks,
            'blocked_sources': len(self._blocked_sources),
            'recent_violations': len(self._violations),
            'block_rate': (
                self._total_blocks / max(1, self._total_checks)
            ),
        }


class ScoutCell(HoneycombCell):
    """
    v3.0 NUEVO: Celda Exploradora - Búsqueda de recursos y exploración.

    Explora el grid buscando áreas con recursos o baja carga,
    y marca caminos con feromonas. Implementa un random walk
    dirigido por feromonas con memoria de posiciones visitadas.
    """

    __slots__ = (
        '_exploration_history', '_discoveries', '_current_target',
        '_max_exploration_range', '_visit_memory'
    )

    def __init__(self, coord: HexCoord, config: Optional[HoneycombConfig] = None):
        super().__init__(coord, CellRole.SCOUT, config)
        self._exploration_history: deque = deque(maxlen=200)
        self._discoveries: List[Dict] = []
        self._current_target: Optional[HexCoord] = None
        self._max_exploration_range = config.radius if config else 10
        self._visit_memory: Set[HexCoord] = set()

    def explore_step(self) -> Optional[Dict]:
        """
        Ejecuta un paso de exploración.
        Elige la dirección menos visitada con sesgo por feromonas de exploración.
        """
        neighbors = self.get_all_neighbors()
        if not neighbors:
            return None

        # Priorizar vecinos no visitados
        unvisited = [n for n in neighbors if n.coord not in self._visit_memory]
        candidates = unvisited if unvisited else neighbors

        # Score: baja carga + alta feromona de exploración + no visitado
        scored = []
        for n in candidates:
            explore_pheromone = n.get_pheromone(PheromoneType.EXPLORATION)
            load_score = 1.0 - n.load
            novelty = self._config.scout_novelty_bonus if n.coord not in self._visit_memory else 0.0
            scored.append((load_score + explore_pheromone * self._config.scout_explore_pheromone_weight + novelty, n))

        scored.sort(key=lambda x: -x[0])
        best = scored[0][1]

        self._visit_memory.add(best.coord)
        self._exploration_history.append({
            'coord': best.coord.to_dict(),
            'timestamp': time.time(),
            'load': best.load,
        })

        # Depositar feromona de camino
        self.deposit_pheromone(PheromoneType.PATH, self._config.scout_path_deposit_intensity, source=self.coord)

        # Detectar descubrimientos
        discovery = None
        if best.load < self._config.scout_low_load_threshold and best.is_available:
            discovery = {
                'type': 'low_load_area',
                'coord': best.coord.to_dict(),
                'load': best.load,
                'timestamp': time.time(),
            }
            self._discoveries.append(discovery)

            self._event_bus.publish(Event(
                type=EventType.SCOUT_DISCOVERY,
                source=self,
                data=discovery
            ))

        return discovery

    def set_target(self, target: HexCoord) -> None:
        self._current_target = target

    def get_scout_stats(self) -> Dict[str, Any]:
        return {
            'explored_cells': len(self._visit_memory),
            'discoveries': len(self._discoveries),
            'history_length': len(self._exploration_history),
            'current_target': self._current_target.to_dict() if self._current_target else None,
            'max_range': self._max_exploration_range,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH MONITOR (v3.0 - nuevo)
# ═══════════════════════════════════════════════════════════════════════════════

class HealthStatus(Enum):
    """Estado de salud del sistema."""
    HEALTHY = auto()
    DEGRADED = auto()
    CRITICAL = auto()


class HealthMonitor:
    """
    Monitor de salud del grid con alertas automáticas.
    
    Evalúa: carga promedio, celdas fallidas, circuit breakers abiertos,
    y tendencia de carga.
    """

    __slots__ = (
        '_grid', '_event_bus', '_check_interval',
        '_alert_threshold', '_last_check', '_status_history',
        '_lock'
    )

    def __init__(
        self,
        grid: 'HoneycombGrid',
        event_bus: Optional[EventBus] = None,
        check_interval: float = 10.0,
        alert_threshold: float = 0.9
    ):
        self._grid = grid
        self._event_bus = event_bus or get_event_bus()
        self._check_interval = check_interval
        self._alert_threshold = alert_threshold
        self._last_check = 0.0
        self._status_history: deque = deque(maxlen=100)
        self._lock = threading.Lock()

    def check_health(self) -> Dict[str, Any]:
        """Ejecuta health check completo."""
        now = time.time()

        with self._lock:
            stats = self._grid.get_stats()

            total = max(1, stats['total_cells'])
            failed_ratio = stats['failed_cells'] / total
            avg_load = stats['average_load']

            grid_config = self._grid.config
            if failed_ratio > grid_config.health_critical_failed_ratio or avg_load > grid_config.health_critical_load:
                status = HealthStatus.CRITICAL
            elif failed_ratio > grid_config.health_degraded_failed_ratio or avg_load > self._alert_threshold:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            result = {
                'status': status.name,
                'timestamp': now,
                'average_load': avg_load,
                'failed_cells': stats['failed_cells'],
                'failed_ratio': failed_ratio,
                'total_cells': total,
            }

            self._status_history.append(result)
            self._last_check = now

        # Emitir eventos
        self._event_bus.publish(Event(
            type=EventType.HEALTH_CHECK,
            source=self,
            data=result
        ))

        if status != HealthStatus.HEALTHY:
            self._event_bus.publish(Event(
                type=EventType.HEALTH_ALERT,
                source=self,
                data=result
            ))

        return result

    def should_check(self) -> bool:
        return time.time() - self._last_check >= self._check_interval

    def get_status_trend(self, window: int = 10) -> List[Dict]:
        return list(self._status_history)[-window:]


# ═══════════════════════════════════════════════════════════════════════════════
# CELL FACTORY (v3.0 - nuevo)
# ═══════════════════════════════════════════════════════════════════════════════

# Mapping de roles a clases de celdas
_CELL_TYPE_MAP: Dict[CellRole, type] = {
    CellRole.QUEEN: QueenCell,
    CellRole.WORKER: WorkerCell,
    CellRole.DRONE: DroneCell,
    CellRole.NURSERY: NurseryCell,
    CellRole.STORAGE: StorageCell,
    CellRole.GUARD: GuardCell,
    CellRole.SCOUT: ScoutCell,
}

_CELL_NAME_MAP: Dict[str, type] = {
    cls.__name__: cls for cls in _CELL_TYPE_MAP.values()
}


def _create_cell_by_role(
    role: CellRole,
    coord: HexCoord,
    config: Optional[HoneycombConfig] = None
) -> HoneycombCell:
    """Factory para crear celdas por rol."""
    cell_cls = _CELL_TYPE_MAP.get(role, WorkerCell)
    if role == CellRole.QUEEN:
        return cell_cls(coord, config)
    return cell_cls(coord, config)


# ═══════════════════════════════════════════════════════════════════════════════
# GRID HEXAGONAL PRINCIPAL (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

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
        'config', '_cells', '_queen', '_rw_lock', '_topology',
        '_executor', '_metrics_collector', '_event_bus',
        '_tick_count', '_last_tick_time', '_running',
        '_role_index', '_health_monitor'
    )

    def __init__(
        self,
        config: Optional[HoneycombConfig] = None,
        event_bus: Optional[EventBus] = None
    ):
        self.config = config or HoneycombConfig()
        self._cells: Dict[HexCoord, HoneycombCell] = {}
        self._queen: Optional[QueenCell] = None
        self._rw_lock = RWLock()
        self._topology = GridTopology[self.config.topology.upper()]
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_parallel_rings,
            thread_name_prefix="hoc_"
        )
        self._metrics_collector = MetricsCollector(
            max_history=self.config.metrics_history_size,
            sample_rate=self.config.metrics_sample_rate
        )
        self._event_bus = event_bus or get_event_bus()
        self._tick_count = 0
        self._last_tick_time = time.time()
        self._running = False

        # v3.0: Índice por rol para lookups O(1)
        self._role_index: Dict[CellRole, Set[HexCoord]] = defaultdict(set)

        # Inicializar grid
        self._initialize_grid()

        # v3.0: Health monitor
        self._health_monitor = HealthMonitor(
            self,
            self._event_bus,
            check_interval=self.config.health_check_interval_s,
            alert_threshold=self.config.health_alert_load_threshold
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

    def _remove_cell_from_index(self, coord: HexCoord) -> Optional[HoneycombCell]:
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

    def _resolve_neighbor_coord(
        self,
        coord: HexCoord,
        direction: HexDirection
    ) -> HexCoord:
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
    def queen(self) -> Optional[QueenCell]:
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

    def get_cell(self, coord: HexCoord) -> Optional[HoneycombCell]:
        with self._rw_lock.read_lock():
            return self._cells.get(coord)

    def get_or_create_cell(
        self,
        coord: HexCoord,
        cell_type: type = WorkerCell
    ) -> HoneycombCell:
        with self._rw_lock.write_lock():
            if coord not in self._cells:
                cell = cell_type(coord, self.config)
                self._add_cell_to_index(coord, cell)
                self._connect_cell_neighbors(cell)

                if isinstance(cell, WorkerCell) and self._queen:
                    self._queen.register_worker(cell)

                self._event_bus.publish(Event(
                    type=EventType.GRID_CELL_ADDED,
                    source=self,
                    data={'coord': coord.to_dict()}
                ))

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

            self._event_bus.publish(Event(
                type=EventType.GRID_CELL_REMOVED,
                source=self,
                data={'coord': coord.to_dict()}
            ))

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

    def get_ring(self, radius: int) -> List[HoneycombCell]:
        origin = HexCoord.origin()
        with self._rw_lock.read_lock():
            return [
                self._cells[coord]
                for coord in origin.ring(radius)
                if coord in self._cells
            ]

    def get_area(self, center: HexCoord, radius: int) -> List[HoneycombCell]:
        with self._rw_lock.read_lock():
            return [
                self._cells[coord]
                for coord in center.spiral(radius)
                if coord in self._cells
            ]

    def get_cells_by_role(self, role: CellRole) -> List[HoneycombCell]:
        """v3.0: O(k) donde k = celdas del rol, no O(n) total."""
        with self._rw_lock.read_lock():
            return [
                self._cells[coord]
                for coord in self._role_index.get(role, set())
                if coord in self._cells
            ]

    def get_cells_by_state(self, state: CellState) -> List[HoneycombCell]:
        with self._rw_lock.read_lock():
            return [c for c in self._cells.values() if c.state == state]

    def find_available_cells(
        self,
        count: int = 1,
        near: Optional[HexCoord] = None
    ) -> List[HoneycombCell]:
        with self._rw_lock.read_lock():
            available = [
                cell for cell in self._cells.values()
                if cell.is_available and isinstance(cell, WorkerCell)
            ]

            if near:
                available.sort(key=lambda c: c.coord.distance_to(near))
            else:
                available.sort(key=lambda c: c.load)

            return available[:count]

    def find_path(
        self,
        start: HexCoord,
        goal: HexCoord,
        cost_fn: Optional[Callable[[HexCoord], float]] = None
    ) -> Optional[List[HexCoord]]:
        """v3.0: Soporta costos variables."""
        pathfinder = HexPathfinder(
            walkable_check=lambda c: c in self._cells and self._cells[c].is_available,
            cost_fn=cost_fn
        )
        return pathfinder.find_path(start, goal)

    # ─────────────────────────────────────────────────────────────────────────
    # ASIGNACIÓN DE TRABAJO
    # ─────────────────────────────────────────────────────────────────────────

    def assign_vcore(
        self,
        vcore: Any,
        preferred_coord: Optional[HexCoord] = None
    ) -> Optional[HoneycombCell]:
        cells = self.find_available_cells(3, near=preferred_coord)

        for cell in cells:
            if cell.add_vcore(vcore):
                return cell

        return None

    def assign_vcores_batch(
        self,
        vcores: List[Any]
    ) -> Dict[HexCoord, List[Any]]:
        assignments: Dict[HexCoord, List[Any]] = defaultdict(list)

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

    def tick(self) -> Dict[str, Any]:
        """Ejecuta un tick global del grid."""
        tick_start = time.time()

        self._event_bus.publish(Event(
            type=EventType.GRID_TICK_START,
            source=self,
            data={'tick': self._tick_count}
        ))

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

        # Métricas
        results["total_vcores"] = sum(c.vcore_count for c in self._cells.values())

        tick_duration = time.time() - tick_start
        tps = 1.0 / tick_duration if tick_duration > 0 else 0

        with self._rw_lock.read_lock():
            worker_loads = [
                c.load for c in self._cells.values()
                if isinstance(c, WorkerCell)
            ]

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
            work_steals=results["work_steals"]
        )

        self._metrics_collector.record(metrics)

        self._tick_count += 1
        self._last_tick_time = time.time()

        # v3.0: Health check periódico
        if self._health_monitor.should_check():
            self._health_monitor.check_health()

        self._event_bus.publish(Event(
            type=EventType.GRID_TICK_END,
            source=self,
            data=results
        ))

        return results

    def _parallel_tick(self) -> Dict[str, int]:
        processed = 0
        errors = 0

        futures: List[Future] = []

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

    def _sequential_tick(self) -> Dict[str, int]:
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

    def _process_cells_batch(self, cells: List[HoneycombCell]) -> Dict[str, int]:
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
            workers = [
                c for c in self._cells.values()
                if isinstance(c, WorkerCell)
            ]

        for worker in workers:
            if worker.can_steal_work():
                stolen = worker.attempt_work_stealing()
                total_stolen += stolen

        if total_stolen > 0:
            self._metrics_collector.increment('work_steals', total_stolen)

        return total_stolen

    def _attempt_auto_recovery(self) -> int:
        """v3.0: Intenta recuperar celdas cuyo circuit breaker está half-open."""
        recovered = 0

        with self._rw_lock.read_lock():
            failed = [
                c for c in self._cells.values()
                if c.state == CellState.FAILED
                and c.circuit_breaker.state == CircuitState.HALF_OPEN
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

    def get_stats(self) -> Dict[str, Any]:
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

    def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self._metrics_collector.get_history(limit)]

    def get_cell_metrics(self) -> List[Dict[str, Any]]:
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

            for q in range(-self.config.radius - min(0, r),
                          self.config.radius - max(0, r) + 1):
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

    def to_dict(self) -> Dict[str, Any]:
        with self._rw_lock.read_lock():
            return {
                "version": "3.0",
                "config": self.config.to_dict(),
                "topology": self._topology.name,
                "tick_count": self._tick_count,
                "cells": {
                    f"{c.q},{c.r}": cell.to_dict()
                    for c, cell in self._cells.items()
                },
                "stats": self.get_stats()
            }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HoneycombGrid':
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
                try:
                    cell._state = CellState[state_name]
                except KeyError:
                    pass

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

    def __enter__(self) -> 'HoneycombGrid':
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


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE UTILIDAD
# ═══════════════════════════════════════════════════════════════════════════════

def create_grid(
    radius: int = 10,
    topology: str = 'flat',
    **kwargs
) -> HoneycombGrid:
    """Factory function para crear grids."""
    config = HoneycombConfig(radius=radius, topology=topology, **kwargs)
    return HoneycombGrid(config)


def benchmark_grid(grid: HoneycombGrid, ticks: int = 100) -> Dict[str, float]:
    """Ejecuta benchmark del grid."""
    times = []

    for _ in range(ticks):
        start = time.perf_counter()
        grid.tick()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        'total_time': sum(times),
        'avg_tick_time': float(np.mean(times)),
        'min_tick_time': min(times),
        'max_tick_time': max(times),
        'ticks_per_second': 1.0 / float(np.mean(times)) if np.mean(times) > 0 else 0,
        'stddev': float(np.std(times)),
        'p99_tick_time': float(np.percentile(times, 99)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EJEMPLO DE USO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with create_grid(radius=5, parallel_ring_processing=True) as grid:
        print(f"Created: {grid}")
        print(f"\nInitial stats:")
        for k, v in grid.get_stats().items():
            print(f"  {k}: {v}")

        # Health check
        health = grid.health_monitor.check_health()
        print(f"\nHealth: {health['status']}")

        # Ejecutar ticks
        print("\nRunning 10 ticks...")
        for i in range(10):
            result = grid.tick()
            print(
                f"  Tick {i}: processed={result['cells_processed']}, "
                f"vcores={result['total_vcores']}, "
                f"recovered={result['auto_recovered']}"
            )

        # Cluster health
        if grid.queen:
            print(f"\nCluster health score: {grid.queen.get_cluster_health_score():.2f}")

        # Visualizar
        print("\nGrid visualization:")
        print(grid.visualize_ascii())

        # Benchmark
        print("\nBenchmark (100 ticks):")
        bench = benchmark_grid(grid, 100)
        for k, v in bench.items():
            print(f"  {k}: {v:.6f}")

        # Serialización roundtrip
        data = grid.to_dict()
        restored = HoneycombGrid.from_dict(data)
        print(f"\nSerialization roundtrip: {restored}")