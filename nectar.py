"""
HOC Nectar Flow - Sistema de Comunicación Bio-Inspirado
========================================================

Implementa comunicación entre celdas usando metáforas de colmena:

1. FEROMONAS (Stigmergy):
   - Rastros químicos virtuales
   - Decaen con el tiempo
   - Guían el comportamiento emergente

2. WAGGLE DANCE:
   - Protocolo de broadcast direccional
   - Codifica distancia y dirección a recursos
   - Inspirado en la danza de las abejas

3. ROYAL JELLY:
   - Canal de alta prioridad
   - Comunicación reina → colmena
   - Comandos críticos del sistema

Flujo de datos:

    ┌─────────────────────────────────────────────────────────┐
    │                     NectarFlow                          │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
    │  │ Pheromone   │  │   Waggle    │  │   Royal     │     │
    │  │   Trails    │  │   Dance     │  │   Jelly     │     │
    │  │  (Passive)  │  │  (Active)   │  │ (Priority)  │     │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
    │         │                │                │             │
    │         └────────────────┼────────────────┘             │
    │                          │                              │
    │                    ┌─────▼─────┐                        │
    │                    │  Channel  │                        │
    │                    │  Router   │                        │
    │                    └─────┬─────┘                        │
    │                          │                              │
    │         ┌────────────────┼────────────────┐             │
    │         ▼                ▼                ▼             │
    │    [Cell A]         [Cell B]         [Cell C]           │
    └─────────────────────────────────────────────────────────┘

"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
import warnings
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    ClassVar,
    TypeVar,
    cast,
)

import mscs as _mscs
import numpy as np

from .core import HexCoord, HexDirection, HoneycombGrid
from .security import (
    sanitize_error,
    sign_payload as _sign_payload,
    verify_signature as _verify_signature,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# TIPOS DE FEROMONA
# ═══════════════════════════════════════════════════════════════════════════════


class PheromoneType(Enum):
    """Tipos de feromonas con diferentes propósitos."""

    # Rastros de trabajo
    TRAIL = auto()  # Camino general
    FOOD = auto()  # Recurso encontrado
    DANGER = auto()  # Peligro/error detectado

    # Señales de estado
    BUSY = auto()  # Celda ocupada
    AVAILABLE = auto()  # Celda disponible

    # Coordinación
    RECRUITMENT = auto()  # Reclutamiento de ayuda
    ALARM = auto()  # Alerta general

    # Optimización
    SUCCESS = auto()  # Tarea completada exitosamente
    FAILURE = auto()  # Tarea fallida

    def decay_rate(self) -> float:
        """Tasa de decaimiento por tipo."""
        rates = {
            PheromoneType.TRAIL: 0.05,
            PheromoneType.FOOD: 0.03,
            PheromoneType.DANGER: 0.15,
            PheromoneType.BUSY: 0.2,
            PheromoneType.AVAILABLE: 0.1,
            PheromoneType.RECRUITMENT: 0.08,
            PheromoneType.ALARM: 0.25,
            PheromoneType.SUCCESS: 0.02,
            PheromoneType.FAILURE: 0.1,
        }
        return rates.get(self, 0.1)


class PheromonePhase(Enum):
    """Phase 5.2a: lifecycle phase of a :class:`PheromoneDeposit`.

    Mirrors the states of the PheromoneDeposit FSM in
    :mod:`hoc.state_machines.pheromone_fsm`. Per the perf budget
    documented in ADR-007 (and re-stated in Phase 5's brief), this is a
    **static-only** wire-up: ``PheromoneDeposit`` carries a ``state``
    field that ``PheromoneTrail.evaporate`` and ``diffuse_to_neighbors``
    update inside their existing loops, but no per-instance FSM is
    allocated and no runtime guard validation happens. The FSM in
    ``state_machines/`` remains the documentation source of truth and
    the property-test target.

    Values are the FSM state strings so a future runtime wire-up can use
    ``transition_to(phase.value)`` if performance ever tolerates it.
    """

    FRESH = "FRESH"
    DECAYING = "DECAYING"
    DIFFUSING = "DIFFUSING"
    EVAPORATED = "EVAPORATED"


# Phase 5.2a: age boundary between FRESH and DECAYING. Must match the
# default in ``hoc.state_machines.pheromone_fsm.DEFAULT_FRESHNESS_WINDOW``
# (kept duplicated to avoid importing from state_machines in this hot
# module — the constant is small and unlikely to drift).
PHEROMONE_FRESHNESS_WINDOW: float = 5.0


@dataclass
class PheromoneDeposit:
    """
    Un depósito individual de feromona.

    Phase 2: soporta firma HMAC-SHA256 sobre los campos de identidad
    inmutables (``ptype``, ``source``, timestamp original). ``intensity``
    NO forma parte del HMAC porque varía con deposits, decay y diffusion.

    Phase 5.2a: el campo ``state`` mantiene la fase del lifecycle
    (FRESH/DECAYING/DIFFUSING/EVAPORATED). Set por
    ``PheromoneTrail.evaporate`` y ``diffuse_to_neighbors``; no entra
    por la FSM en runtime — es un mirror del estado real para
    observabilidad + detección estática vía choreo.
    """

    ptype: PheromoneType
    intensity: float
    timestamp: float
    source: HexCoord | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    signature: bytes | None = None  # HMAC-SHA256 sobre identity fields
    # Phase 5.2a: lifecycle phase mirror. Default FRESH; transitions
    # happen in PheromoneTrail's evaporate/diffuse loops without
    # per-instance FSM allocation (perf-critical path).
    state: PheromonePhase = PheromonePhase.FRESH

    def decay(self, elapsed: float) -> float:
        """Aplica decaimiento basado en tiempo transcurrido."""
        decay_factor = math.exp(-self.ptype.decay_rate() * elapsed)
        self.intensity *= decay_factor
        return self.intensity

    def _canonical_payload(self) -> bytes:
        """
        Bytes estables para HMAC. Solo incluye campos de identidad inmutables
        (``ptype`` + ``source``). ``intensity``, ``timestamp`` y ``metadata``
        evolucionan con re-depósitos, decay y difusión — firmarlos obligaría
        a re-firmar en cada mutación, inútil cuando todos los nodos comparten
        la misma clave HMAC. El propósito de la firma aquí es atestiguar
        origen ("este depósito proviene de un nodo con la clave HMAC"),
        no inmutabilidad de valor.
        """
        src = (self.source.q, self.source.r) if self.source is not None else None
        return cast(
            bytes,
            _mscs.dumps(
                {
                    "kind": "pheromone",
                    "ptype": self.ptype.value,
                    "source": src,
                }
            ),
        )

    def sign(self, key: bytes | None = None) -> PheromoneDeposit:
        """Firma este depósito con HMAC-SHA256. Retorna ``self``."""
        self.signature = _sign_payload(self._canonical_payload(), key=key)
        return self

    def verify(self, key: bytes | None = None) -> bool:
        """Verifica la firma. Retorna False si falta o no coincide."""
        if self.signature is None:
            return False
        return _verify_signature(self._canonical_payload(), self.signature, key=key)


class PheromoneDecay(Enum):
    """Estrategias de decaimiento."""

    EXPONENTIAL = auto()  # Decae exponencialmente
    LINEAR = auto()  # Decae linealmente
    STEP = auto()  # Decae en escalones
    NONE = auto()  # No decae


# ═══════════════════════════════════════════════════════════════════════════════
# RASTRO DE FEROMONAS
# ═══════════════════════════════════════════════════════════════════════════════


class PheromoneTrail:
    """
    Sistema de rastros de feromonas para comunicación indirecta.

    Implementa stigmergy: coordinación a través del ambiente,
    sin comunicación directa entre agentes.

    Uso:
        trail = PheromoneTrail()
        trail.deposit(coord, PheromoneType.FOOD, 1.0)
        level = trail.sense(coord, PheromoneType.FOOD)
        gradient = trail.follow_gradient(coord, PheromoneType.FOOD)
    """

    # Minimum intensity below which a deposit is cleaned up
    CLEANUP_THRESHOLD: float = 0.001
    # Default diffusion rate (fraction of intensity spread to 6 neighbors per tick)
    DEFAULT_DIFFUSION_RATE: float = 0.05
    # Default minimum intensity to participate in diffusion
    DEFAULT_DIFFUSE_THRESHOLD: float = 0.01
    # Default minimum intensity for hotspot detection
    DEFAULT_HOTSPOT_THRESHOLD: float = 0.5
    # Phase 2: bounded growth para mitigar DoS.
    DEFAULT_MAX_COORDS: int = 10_000
    DEFAULT_MAX_METADATA_KEYS: int = 100

    def __init__(
        self,
        decay_strategy: PheromoneDecay = PheromoneDecay.EXPONENTIAL,
        max_intensity: float = 10.0,
        evaporation_interval: float = 1.0,
        max_coords: int | None = None,
        max_metadata_keys: int | None = None,
    ):
        # Phase 1 fix (B3): validar parámetros en construcción para prevenir NaN/inf
        # downstream. Antes los valores inválidos se silenciaban en uso, generando
        # comportamiento impredecible.
        if not isinstance(max_intensity, (int, float)) or max_intensity <= 0:
            raise ValueError(f"max_intensity debe ser float > 0, recibido: {max_intensity!r}")
        if not isinstance(evaporation_interval, (int, float)) or evaporation_interval < 0:
            raise ValueError(
                f"evaporation_interval debe ser float >= 0, recibido: {evaporation_interval!r}"
            )
        if not isinstance(decay_strategy, PheromoneDecay):
            raise TypeError(
                f"decay_strategy debe ser PheromoneDecay, recibido: {type(decay_strategy).__name__}"
            )

        # Phase 2: OrderedDict para tracking LRU y bound total de coordenadas.
        # El attacker model: inundar feromonas en 10K+ coordenadas únicas hasta
        # agotar memoria. El cap + LRU cierra ese vector. Per-coord ya está
        # acotado por el número finito de PheromoneType (~9 entradas max).
        self._deposits: OrderedDict[HexCoord, dict[PheromoneType, PheromoneDeposit]] = OrderedDict()
        self._decay_strategy = decay_strategy
        self._max_intensity = float(max_intensity)
        self._evaporation_interval = float(evaporation_interval)
        self._last_evaporation = time.time()
        self._max_coords = int(max_coords) if max_coords is not None else self.DEFAULT_MAX_COORDS
        self._max_metadata_keys = (
            int(max_metadata_keys)
            if max_metadata_keys is not None
            else self.DEFAULT_MAX_METADATA_KEYS
        )
        if self._max_coords <= 0:
            raise ValueError(f"max_coords debe ser > 0, recibido: {max_coords!r}")
        if self._max_metadata_keys <= 0:
            raise ValueError(f"max_metadata_keys debe ser > 0, recibido: {max_metadata_keys!r}")
        self._lock = threading.RLock()

    def _enforce_per_coord_bound(self, coord: HexCoord) -> None:
        """
        Phase 2: aplica cap LRU sobre el número total de coordenadas con
        depósitos. Llamado desde ``deposit`` al crear una entrada nueva.
        Evicta la coordenada más vieja (OrderedDict.popitem(last=False))
        hasta cumplir con ``self._max_coords``.

        El caller ya sostiene ``self._lock``.
        """
        # Marcar ``coord`` como la más reciente (move_to_end).
        self._deposits.move_to_end(coord, last=True)
        while len(self._deposits) > self._max_coords:
            self._deposits.popitem(last=False)

    def deposit(
        self,
        coord: HexCoord,
        ptype: PheromoneType,
        intensity: float,
        source: HexCoord | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """
        Deposita feromona en una coordenada.

        Args:
            coord: Ubicación del depósito
            ptype: Tipo de feromona
            intensity: Cantidad a depositar
            source: Origen del depósito (opcional)
            metadata: Datos adicionales

        Returns:
            Nivel total después del depósito
        """
        with self._lock:
            coord_entries = self._deposits.get(coord)
            if coord_entries is not None and ptype in coord_entries:
                deposit = coord_entries[ptype]
                deposit.intensity = min(self._max_intensity, deposit.intensity + intensity)
                deposit.timestamp = time.time()
                if metadata:
                    # Phase 2: cap de metadata_keys para cerrar un vector
                    # secundario de DoS (metadata unbounded por deposit).
                    self._merge_metadata_bounded(deposit, metadata)
                # Phase 2: marcamos coord como la más reciente (LRU)
                self._deposits.move_to_end(coord, last=True)
                # Phase 2: firma no cambia porque el payload canónico sólo
                # cubre identidad (ptype+source), ambos inmutables.
            else:
                new_deposit = PheromoneDeposit(
                    ptype=ptype,
                    intensity=min(self._max_intensity, intensity),
                    timestamp=time.time(),
                    source=source,
                    metadata={},
                )
                # Phase 5.2a: explicit FRESH assignment (the dataclass
                # default is FRESH already, but choreo's walker only
                # detects ``obj.state = ENUM.MEMBER`` Assign nodes and
                # not AnnAssign defaults — this statement makes FRESH a
                # targeted state in the static analysis).
                new_deposit.state = PheromonePhase.FRESH
                if metadata:
                    self._merge_metadata_bounded(new_deposit, metadata)
                # Phase 2: firmar en creación. Verificaciones downstream
                # (ej. replicación entre procesos) descartan depósitos
                # sin firma válida.
                new_deposit.sign()
                if coord_entries is None:
                    coord_entries = {}
                    self._deposits[coord] = coord_entries
                coord_entries[ptype] = new_deposit
                # Phase 2: aplicar bound global de coordenadas (LRU).
                self._enforce_per_coord_bound(coord)

            return self._deposits[coord][ptype].intensity

    def _merge_metadata_bounded(
        self,
        deposit: PheromoneDeposit,
        new_metadata: dict[str, Any],
    ) -> None:
        """Fusiona ``new_metadata`` en ``deposit.metadata`` respetando el cap."""
        for k, v in new_metadata.items():
            if k in deposit.metadata or len(deposit.metadata) < self._max_metadata_keys:
                deposit.metadata[k] = v
            # else: descartar silenciosamente para no romper caller legítimo.

    def sense(self, coord: HexCoord, ptype: PheromoneType | None = None) -> float:
        """
        Detecta el nivel de feromona en una coordenada.

        Args:
            coord: Ubicación a sensar
            ptype: Tipo específico (None = total de todos)

        Returns:
            Nivel de feromona (0.0 si no hay)
        """
        with self._lock:
            if coord not in self._deposits:
                return 0.0

            if ptype is not None:
                deposit = self._deposits[coord].get(ptype)
                return deposit.intensity if deposit else 0.0

            return sum(d.intensity for d in self._deposits[coord].values())

    def sense_area(
        self, center: HexCoord, radius: int, ptype: PheromoneType | None = None
    ) -> dict[HexCoord, float]:
        """Detecta feromonas en un área."""
        result = {}
        for coord in center.spiral(radius):
            level = self.sense(coord, ptype)
            if level > 0:
                result[coord] = level
        return result

    def follow_gradient(
        self, coord: HexCoord, ptype: PheromoneType, prefer_unexplored: bool = True
    ) -> HexDirection | None:
        """
        Determina la mejor dirección siguiendo el gradiente de feromona.

        Args:
            coord: Posición actual
            ptype: Tipo de feromona a seguir
            prefer_unexplored: Preferir direcciones sin explorar

        Returns:
            Mejor dirección o None si no hay gradiente
        """
        with self._lock:
            best_direction = None
            best_score = 0.0

            for direction in HexDirection:
                neighbor = coord.neighbor(direction)
                level = self.sense(neighbor, ptype)

                # Añadir algo de ruido para evitar loops
                noise = np.random.random() * 0.1
                score = level + noise

                # Bonus para celdas no visitadas
                if prefer_unexplored and neighbor not in self._deposits:
                    score += 0.5

                if score > best_score:
                    best_score = score
                    best_direction = direction

            return best_direction

    def evaporate(self, force: bool = False) -> int:
        """
        Aplica evaporación a todas las feromonas.

        Returns:
            Número de depósitos eliminados
        """
        now = time.time()

        if not force and (now - self._last_evaporation) < self._evaporation_interval:
            return 0

        removed = 0

        with self._lock:
            self._last_evaporation = now

            to_remove = []

            for coord, deposits in self._deposits.items():
                dead_types = []
                for ptype, deposit in deposits.items():
                    elapsed = now - deposit.timestamp

                    if self._decay_strategy == PheromoneDecay.EXPONENTIAL:
                        deposit.decay(elapsed)
                    elif self._decay_strategy == PheromoneDecay.LINEAR:
                        deposit.intensity -= ptype.decay_rate() * elapsed
                    elif self._decay_strategy == PheromoneDecay.STEP and elapsed > (
                        1.0 / ptype.decay_rate()
                    ):
                        deposit.intensity *= 0.5

                    # Phase 5.2a: lifecycle phase mirror. Below the cleanup
                    # threshold the deposit is queued for removal — mark
                    # EVAPORATED (terminal). Once age crosses the freshness
                    # window we leave FRESH for DECAYING. The two attribute
                    # writes per deposit are the entire perf budget for
                    # this wire-up; no FSM is consulted.
                    if deposit.intensity < self.CLEANUP_THRESHOLD:
                        deposit.state = PheromonePhase.EVAPORATED
                        dead_types.append(ptype)
                    elif elapsed > PHEROMONE_FRESHNESS_WINDOW:
                        deposit.state = PheromonePhase.DECAYING

                for ptype in dead_types:
                    del deposits[ptype]
                    removed += 1

                if not deposits:
                    to_remove.append(coord)

            for coord in to_remove:
                del self._deposits[coord]

        return removed

    def diffuse_to_neighbors(
        self,
        diffusion_rate: float | None = None,
        valid_coords: set[HexCoord] | None = None,
        threshold: float | None = None,
    ) -> int:
        """
        Difunde feromonas a los 6 vecinos hexagonales de cada celda con depósitos.
        Respeta la topología hexagonal: cada celda tiene exactamente 6 direcciones.
        Debe llamarse después del decaimiento en cada tick.
        """
        diffusion_rate = (
            diffusion_rate if diffusion_rate is not None else self.DEFAULT_DIFFUSION_RATE
        )
        threshold = threshold if threshold is not None else self.DEFAULT_DIFFUSE_THRESHOLD
        # Phase 1 fix (B3): rangos inválidos ahora elevan ValueError en lugar de
        # retornar 0 silenciosamente. Permitir 0 explícito (no-op) sin error.
        if not 0.0 <= diffusion_rate < 1.0:
            raise ValueError(f"diffusion_rate debe estar en [0, 1), recibido: {diffusion_rate!r}")
        if threshold < 0.0:
            raise ValueError(f"threshold debe ser >= 0, recibido: {threshold!r}")
        if diffusion_rate == 0.0:
            return 0
        spread_per_neighbor = diffusion_rate / 6.0
        with self._lock:
            new_deposits: list[
                tuple[HexCoord, PheromoneType, float, HexCoord | None, dict[str, Any] | None]
            ] = []
            for coord, deposits in self._deposits.items():
                for ptype, deposit in list(deposits.items()):
                    if deposit.intensity < threshold:
                        continue
                    # Phase 5.2a: transient DIFFUSING during the spread,
                    # then back to DECAYING once the deposit's intensity
                    # has been fanned out. The original deposit's intensity
                    # is unchanged here (the spread happens via deposit()
                    # below on neighbours), so DECAYING is the right
                    # post-spread phase.
                    deposit.state = PheromonePhase.DIFFUSING
                    amount = deposit.intensity * spread_per_neighbor
                    for direction in HexDirection:
                        neighbor_coord = coord.neighbor(direction)
                        if valid_coords is not None and neighbor_coord not in valid_coords:
                            continue
                        new_deposits.append(
                            (
                                neighbor_coord,
                                ptype,
                                amount,
                                coord,
                                None,
                            )
                        )
                    deposit.state = PheromonePhase.DECAYING
            for coord, ptype, intensity, source, meta in new_deposits:
                self.deposit(coord, ptype, intensity, source=source, metadata=meta)
        return len(new_deposits)

    def get_hotspots(
        self, ptype: PheromoneType, threshold: float | None = None, limit: int = 10
    ) -> list[tuple[HexCoord, float]]:
        """Obtiene las ubicaciones con mayor concentración."""
        threshold = threshold if threshold is not None else self.DEFAULT_HOTSPOT_THRESHOLD
        with self._lock:
            hotspots = []
            for coord, deposits in self._deposits.items():
                if ptype in deposits and deposits[ptype].intensity >= threshold:
                    hotspots.append((coord, deposits[ptype].intensity))

            hotspots.sort(key=lambda x: x[1], reverse=True)
            return hotspots[:limit]

    def clear(self, coord: HexCoord | None = None, ptype: PheromoneType | None = None) -> None:
        """Limpia feromonas."""
        with self._lock:
            if coord is None:
                if ptype is None:
                    self._deposits.clear()
                else:
                    for deposits in self._deposits.values():
                        deposits.pop(ptype, None)
            else:
                if coord in self._deposits:
                    if ptype is None:
                        del self._deposits[coord]
                    else:
                        self._deposits[coord].pop(ptype, None)

    def get_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas del sistema de feromonas."""
        with self._lock:
            total_deposits = sum(len(d) for d in self._deposits.values())
            total_intensity = sum(
                d.intensity for deposits in self._deposits.values() for d in deposits.values()
            )

            by_type: defaultdict[str, float] = defaultdict(float)
            for deposits in self._deposits.values():
                for ptype, deposit in deposits.items():
                    by_type[ptype.name] += deposit.intensity

            return {
                "locations": len(self._deposits),
                "total_deposits": total_deposits,
                "total_intensity": total_intensity,
                "by_type": dict(by_type),
            }


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOLO WAGGLE DANCE
# ═══════════════════════════════════════════════════════════════════════════════


class DanceDirection(Enum):
    """Direcciones codificadas en la danza."""

    UP = 0  # Norte (referencia solar)
    UP_RIGHT = 60
    RIGHT = 90
    DOWN_RIGHT = 120
    DOWN = 180
    DOWN_LEFT = 240
    LEFT = 270
    UP_LEFT = 300

    @classmethod
    def from_angle(cls, angle: float) -> DanceDirection:
        """Convierte un ángulo a dirección de danza."""
        normalized = angle % 360
        for direction in cls:
            if abs(normalized - direction.value) < 30:
                return direction
        return cls.UP

    def to_hex_direction(self) -> HexDirection:
        """Convierte a dirección hexagonal."""
        mapping = {
            DanceDirection.UP: HexDirection.NW,
            DanceDirection.UP_RIGHT: HexDirection.NE,
            DanceDirection.RIGHT: HexDirection.E,
            DanceDirection.DOWN_RIGHT: HexDirection.SE,
            DanceDirection.DOWN: HexDirection.SW,
            DanceDirection.DOWN_LEFT: HexDirection.SW,
            DanceDirection.LEFT: HexDirection.W,
            DanceDirection.UP_LEFT: HexDirection.NW,
        }
        return mapping.get(self, HexDirection.E)


@dataclass
class DanceMessage:
    """
    Mensaje codificado en una danza waggle.

    La danza de las abejas codifica:
    - Dirección al recurso (relativa al sol)
    - Distancia al recurso (duración de la danza)
    - Calidad del recurso (vigor de la danza)

    Phase 2: soporta firma HMAC-SHA256 sobre los campos de identidad
    inmutables (``source``, ``direction``, ``distance``, ``resource_type``,
    ``timestamp``). ``quality`` y ``ttl`` evolucionan durante la propagación,
    así que no forman parte del HMAC — la firma atestigua únicamente que
    el mensaje proviene de un nodo con la clave compartida.
    """

    source: HexCoord  # Quien baila
    direction: DanceDirection  # Hacia dónde
    distance: int  # Qué tan lejos (en celdas)
    quality: float  # Qué tan bueno (0.0 - 1.0)
    resource_type: str  # Tipo de recurso
    timestamp: float = field(default_factory=time.time)
    ttl: int = 10  # Time to live (broadcasts restantes)
    metadata: dict[str, Any] = field(default_factory=dict)
    signature: bytes | None = None  # Phase 2: HMAC-SHA256 sobre identity

    def _canonical_payload(self) -> bytes:
        """Bytes estables para HMAC. Excluye ``quality`` y ``ttl`` mutables."""
        return cast(
            bytes,
            _mscs.dumps(
                {
                    "kind": "dance",
                    "source": (self.source.q, self.source.r),
                    "direction": self.direction.value,
                    "distance": self.distance,
                    "resource_type": self.resource_type,
                    "timestamp": round(self.timestamp, 6),
                }
            ),
        )

    def sign(self, key: bytes | None = None) -> DanceMessage:
        """Firma este mensaje con HMAC-SHA256. Retorna ``self`` para chaining."""
        self.signature = _sign_payload(self._canonical_payload(), key=key)
        return self

    def verify(self, key: bytes | None = None) -> bool:
        """Verifica la firma. False si falta o no coincide."""
        if self.signature is None:
            return False
        return _verify_signature(self._canonical_payload(), self.signature, key=key)

    def encode(self) -> bytes:
        """Codifica el mensaje para transmisión."""
        # Formato compacto: tipo|dir|dist|quality
        data = f"{self.resource_type}|{self.direction.value}|{self.distance}|{self.quality:.2f}"
        return data.encode()

    @classmethod
    def decode(cls, data: bytes, source: HexCoord) -> DanceMessage:
        """Decodifica un mensaje recibido."""
        parts = data.decode().split("|")
        return cls(
            source=source,
            direction=DanceDirection.from_angle(float(parts[1])),
            distance=int(parts[2]),
            quality=float(parts[3]),
            resource_type=parts[0],
        )

    def target_coord(self) -> HexCoord:
        """Calcula la coordenada objetivo aproximada."""
        direction = self.direction.to_hex_direction()
        dq, dr = {
            HexDirection.NE: (1, -1),
            HexDirection.E: (1, 0),
            HexDirection.SE: (0, 1),
            HexDirection.SW: (-1, 1),
            HexDirection.W: (-1, 0),
            HexDirection.NW: (0, -1),
        }[direction]

        return HexCoord(self.source.q + dq * self.distance, self.source.r + dr * self.distance)


class WaggleDance:
    """
    Protocolo de comunicación Waggle Dance.

    Permite a las celdas "bailar" para comunicar ubicaciones
    de recursos o trabajo disponible a sus vecinos.

    Características:
    - Broadcast direccional (se propaga más fuerte en una dirección)
    - Atenuación por distancia
    - Competencia entre mensajes (el más fuerte gana atención)
    """

    def __init__(
        self, broadcast_range: int = 5, attenuation: float = 0.8, competition_threshold: float = 0.3
    ):
        self._active_dances: dict[HexCoord, list[DanceMessage]] = defaultdict(list)
        self._broadcast_range = broadcast_range
        self._attenuation = attenuation
        self._competition_threshold = competition_threshold
        self._lock = threading.RLock()
        self._observers: list[Callable[[DanceMessage], None]] = []

    def start_dance(
        self,
        dancer: HexCoord,
        direction: DanceDirection,
        distance: int,
        quality: float,
        resource_type: str = "generic",
        metadata: dict[str, Any] | None = None,
    ) -> DanceMessage:
        """
        Inicia una danza en la coordenada especificada.

        Args:
            dancer: Posición del bailarín
            direction: Dirección al recurso
            distance: Distancia al recurso
            quality: Calidad del recurso
            resource_type: Tipo de recurso
            metadata: Información adicional

        Returns:
            El mensaje de danza creado
        """
        message = DanceMessage(
            source=dancer,
            direction=direction,
            distance=distance,
            quality=quality,
            resource_type=resource_type,
            ttl=self._broadcast_range * 2,
            metadata=metadata or {},
        )
        # Phase 2: firmar al origen. La firma sobrevive a la propagación
        # porque _canonical_payload excluye quality y ttl (los mutables).
        message.sign()

        with self._lock:
            self._active_dances[dancer].append(message)

            # Notificar observadores
            for observer in self._observers:
                try:
                    observer(message)
                except Exception as e:
                    logger.error(f"Dance observer error: {sanitize_error(e)}")

        return message

    def propagate(self, grid: HoneycombGrid) -> int:
        """
        Propaga las danzas activas a través del grid.

        Returns:
            Número de mensajes propagados
        """
        propagated = 0

        with self._lock:
            new_dances = defaultdict(list)

            for source, dances in self._active_dances.items():
                for dance in dances:
                    if dance.ttl <= 0:
                        continue

                    # Obtener celda fuente
                    source_cell = grid.get_cell(source)
                    if not source_cell:
                        continue

                    # Propagar a vecinos con sesgo direccional
                    preferred_direction = dance.direction.to_hex_direction()

                    for direction in HexDirection:
                        neighbor = source_cell.get_neighbor(direction)
                        if not neighbor:
                            continue

                        # Calcular atenuación
                        attenuation = self._attenuation

                        # Bonus si es la dirección preferida
                        if direction == preferred_direction:
                            attenuation = min(1.0, attenuation * 1.5)
                        # Penalización si es dirección opuesta
                        elif direction == preferred_direction.opposite():
                            attenuation *= 0.5

                        # Crear mensaje atenuado. Phase 2: preservamos la
                        # firma original — la firma solo cubre identity
                        # fields (source/direction/distance/resource_type/
                        # timestamp), que no cambian en propagación. Atenuar
                        # quality/ttl no invalida la firma.
                        propagated_dance = DanceMessage(
                            source=dance.source,
                            direction=dance.direction,
                            distance=dance.distance,
                            quality=dance.quality * attenuation,
                            resource_type=dance.resource_type,
                            timestamp=dance.timestamp,
                            ttl=dance.ttl - 1,
                            metadata=dance.metadata.copy(),
                            signature=dance.signature,
                        )

                        # Solo propagar si supera umbral
                        if propagated_dance.quality >= self._competition_threshold:
                            new_dances[neighbor.coord].append(propagated_dance)
                            propagated += 1

            # Fusionar danzas competidoras (quedarse con la mejor)
            for coord, dances in new_dances.items():
                if len(dances) > 3:
                    # Ordenar por calidad y quedarse con las 3 mejores
                    dances.sort(key=lambda d: d.quality, reverse=True)
                    new_dances[coord] = dances[:3]

            # Actualizar danzas activas
            self._active_dances.clear()
            self._active_dances.update(new_dances)

        return propagated

    def observe_dances(self, observer: HexCoord, radius: int = 1) -> list[DanceMessage]:
        """
        Observa las danzas cercanas a una posición.

        Args:
            observer: Posición del observador
            radius: Radio de observación

        Returns:
            Lista de danzas observadas
        """
        with self._lock:
            observed = []
            for coord in observer.spiral(radius):
                if coord in self._active_dances:
                    observed.extend(self._active_dances[coord])

            # Ordenar por calidad
            observed.sort(key=lambda d: d.quality, reverse=True)
            return observed

    def add_observer(self, callback: Callable[[DanceMessage], None]) -> None:
        """Añade un observador de danzas."""
        self._observers.append(callback)

    def clear_old_dances(self, max_age: float = 60.0) -> int:
        """Limpia danzas antiguas."""
        now = time.time()
        removed = 0

        with self._lock:
            for coord in list(self._active_dances.keys()):
                original_count = len(self._active_dances[coord])
                self._active_dances[coord] = [
                    d for d in self._active_dances[coord] if (now - d.timestamp) < max_age
                ]
                removed += original_count - len(self._active_dances[coord])

                if not self._active_dances[coord]:
                    del self._active_dances[coord]

        return removed

    def get_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas del sistema de danza."""
        with self._lock:
            total_dances = sum(len(d) for d in self._active_dances.values())
            by_type: defaultdict[str, int] = defaultdict(int)

            for dances in self._active_dances.values():
                for dance in dances:
                    by_type[dance.resource_type] += 1

            return {
                "active_locations": len(self._active_dances),
                "total_dances": total_dances,
                "by_resource_type": dict(by_type),
            }


# ═══════════════════════════════════════════════════════════════════════════════
# ROYAL JELLY - CANAL DE ALTA PRIORIDAD
# ═══════════════════════════════════════════════════════════════════════════════


class RoyalCommand(Enum):
    """Tipos de comandos reales."""

    SWARM = auto()  # Iniciar enjambre (migración masiva)
    HIBERNATE = auto()  # Entrar en hibernación
    WAKE = auto()  # Despertar
    EVACUATE = auto()  # Evacuar área
    REINFORCE = auto()  # Reforzar área
    BALANCE = auto()  # Balancear carga
    SPAWN = auto()  # Crear nuevas entidades
    CULL = auto()  # Reducir población
    EMERGENCY = auto()  # Emergencia general


@dataclass
class RoyalMessage:
    """
    Mensaje de la reina.

    Phase 2: incluye ``issuer`` (HexCoord de la celda que emite el comando)
    y ``signature`` HMAC-SHA256. ``acknowledged`` es mutable y no forma
    parte del HMAC.

    Reglas de emisión:
    - Cualquier celda puede emitir con priority < 8.
    - Solo la QueenCell actual puede emitir priority >= 8 (ver
      :meth:`RoyalJelly.issue_command`).
    """

    command: RoyalCommand
    priority: int  # 0-10 (10 = máxima)
    target: HexCoord | None  # Destino específico o None para broadcast
    params: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    acknowledged: set[HexCoord] = field(default_factory=set)
    issuer: HexCoord | None = None  # Phase 2: quién emitió el comando
    signature: bytes | None = None  # Phase 2: HMAC-SHA256

    def _canonical_payload(self) -> bytes:
        """Bytes estables para HMAC. Excluye ``acknowledged`` mutable."""
        issuer = (self.issuer.q, self.issuer.r) if self.issuer is not None else None
        target = (self.target.q, self.target.r) if self.target is not None else None
        # params se incluye serializado canónicamente vía mscs para detectar
        # manipulación de argumentos (p.e. atacante cambia ``target_radius``
        # en un EVACUATE para ampliar el área evacuada).
        return cast(
            bytes,
            _mscs.dumps(
                {
                    "kind": "royal",
                    "command": self.command.value,
                    "priority": self.priority,
                    "target": target,
                    "issuer": issuer,
                    "timestamp": round(self.timestamp, 6),
                    "params": self.params,
                }
            ),
        )

    def sign(self, key: bytes | None = None) -> RoyalMessage:
        """Firma este comando con HMAC-SHA256. Retorna ``self``."""
        self.signature = _sign_payload(self._canonical_payload(), key=key)
        return self

    def verify(self, key: bytes | None = None) -> bool:
        """Verifica la firma. False si falta o no coincide."""
        if self.signature is None:
            return False
        return _verify_signature(self._canonical_payload(), self.signature, key=key)


class RoyalJelly:
    """
    Canal de comunicación de alta prioridad Reina → Colmena.

    Características:
    - Entrega garantizada a todas las celdas
    - Prioridad sobre otros tipos de comunicación
    - Acknowledgement de recepción
    - Cola de comandos pendientes

    Phase 2 — políticas de autorización:
    - Solo la :class:`QueenCell` actual puede emitir comandos con
      ``priority >= HIGH_PRIORITY_THRESHOLD``. Intentos de otras celdas
      son rechazados con :class:`PermissionError`.
    - Todos los comandos se firman con HMAC-SHA256 (``RoyalMessage.sign``).
    - Cuando ocurre sucesión de reina, el ``QueenSuccession`` llama a
      :meth:`update_queen_coord` para que los checks siguientes se evalúen
      contra la reina nueva.
    """

    # Phase 2: prioridad mínima que requiere ser emitida por la Queen.
    HIGH_PRIORITY_THRESHOLD: int = 8

    def __init__(self, queen_coord: HexCoord):
        self._queen_coord = queen_coord
        self._pending_commands: list[RoyalMessage] = []
        self._command_history: deque[RoyalMessage] = deque(maxlen=100)
        self._lock = threading.RLock()
        self._subscribers: set[HexCoord] = set()

    def update_queen_coord(self, new_queen: HexCoord) -> None:
        """
        Actualiza la coordenada de la reina. Llamar tras una sucesión
        exitosa. Thread-safe.
        """
        with self._lock:
            self._queen_coord = new_queen

    @property
    def queen_coord(self) -> HexCoord:
        """Retorna la coord de la reina actual (read-only property)."""
        return self._queen_coord

    def issue_command(
        self,
        command: RoyalCommand,
        priority: int = 5,
        target: HexCoord | None = None,
        params: dict[str, Any] | None = None,
        *,
        issuer: HexCoord | None = None,
    ) -> RoyalMessage:
        """
        Emite un comando real.

        Args:
            command: Tipo de comando
            priority: Prioridad (0-10)
            target: Destino específico o None para broadcast
            params: Parámetros adicionales
            issuer: Coord de la celda que emite. Si priority >=
                ``HIGH_PRIORITY_THRESHOLD`` debe ser la reina actual.
                Si None y priority < threshold, se asume la reina (compat).

        Returns:
            El mensaje creado, firmado con HMAC-SHA256.

        Raises:
            PermissionError: si ``priority >= HIGH_PRIORITY_THRESHOLD``
                             y el issuer no es la reina actual.
        """
        clamped_priority = min(10, max(0, priority))

        # Phase 2: Queen-only enforcement para alta prioridad. Esto cierra
        # el vector "DroneCell forja EMERGENCY priority=10" — aun con la
        # clave HMAC compartida, solo la QueenCell puede invocar este
        # path sin recibir PermissionError.
        effective_issuer = issuer if issuer is not None else self._queen_coord
        if clamped_priority >= self.HIGH_PRIORITY_THRESHOLD:
            if issuer is None:
                raise PermissionError(
                    f"issuer es obligatorio para priority>={self.HIGH_PRIORITY_THRESHOLD}"
                )
            if issuer != self._queen_coord:
                raise PermissionError(
                    f"Solo la reina ({self._queen_coord}) puede emitir "
                    f"comandos con priority={clamped_priority}; issuer={issuer}"
                )

        message = RoyalMessage(
            command=command,
            priority=clamped_priority,
            target=target,
            params=params or {},
            issuer=effective_issuer,
        )
        # Phase 2: firmar tras validar autorización.
        message.sign()

        with self._lock:
            # Insertar ordenado por prioridad (mayor primero)
            inserted = False
            for i, pending in enumerate(self._pending_commands):
                if message.priority > pending.priority:
                    self._pending_commands.insert(i, message)
                    inserted = True
                    break

            if not inserted:
                self._pending_commands.append(message)

        logger.info(
            "Royal command issued: %s (priority=%d, issuer=%s)",
            command.name,
            clamped_priority,
            effective_issuer,
        )
        return message

    def subscribe(self, coord: HexCoord) -> None:
        """Suscribe una celda al canal real."""
        with self._lock:
            self._subscribers.add(coord)

    def unsubscribe(self, coord: HexCoord) -> None:
        """Desuscribe una celda."""
        with self._lock:
            self._subscribers.discard(coord)

    def get_commands(self, cell_coord: HexCoord, limit: int = 10) -> list[RoyalMessage]:
        """
        Obtiene comandos pendientes para una celda.

        Args:
            cell_coord: Coordenada de la celda
            limit: Máximo de comandos a retornar

        Returns:
            Lista de comandos aplicables
        """
        with self._lock:
            applicable: list[RoyalMessage] = []
            for cmd in self._pending_commands:
                if len(applicable) >= limit:
                    break

                # Comando es aplicable si:
                # 1. Es broadcast (target=None)
                # 2. Es para esta celda específica
                # 3. No ha sido ya reconocido por esta celda
                if (
                    cmd.target is None or cmd.target == cell_coord
                ) and cell_coord not in cmd.acknowledged:
                    applicable.append(cmd)

            return applicable

    def acknowledge(self, command: RoyalMessage, cell_coord: HexCoord) -> None:
        """
        Reconoce la recepción de un comando.

        Cuando todas las celdas suscritas reconocen un comando,
        se mueve al historial.
        """
        with self._lock:
            command.acknowledged.add(cell_coord)

            # Si es comando específico y fue reconocido, mover a historial
            if (
                command.target is not None
                and cell_coord == command.target
                and command in self._pending_commands
            ) or (
                command.target is None
                and self._subscribers.issubset(command.acknowledged)
                and command in self._pending_commands
            ):
                self._pending_commands.remove(command)
                self._command_history.append(command)

    def emergency_broadcast(
        self,
        message: str,
        params: dict[str, Any] | None = None,
        *,
        issuer: HexCoord | None = None,
    ) -> None:
        """
        Emite una alerta de emergencia con máxima prioridad.

        Phase 2: ``issuer`` debe ser la reina actual (o None, en cuyo caso
        asumimos a la reina). Si otra celda lo invoca, se lanza
        :class:`PermissionError`.
        """
        self.issue_command(
            RoyalCommand.EMERGENCY,
            priority=10,
            target=None,
            params={"message": message, **(params or {})},
            issuer=issuer if issuer is not None else self._queen_coord,
        )

    def get_pending_count(self) -> int:
        """Retorna el número de comandos pendientes."""
        return len(self._pending_commands)

    def get_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas del canal."""
        with self._lock:
            return {
                "pending_commands": len(self._pending_commands),
                "subscribers": len(self._subscribers),
                "history_size": len(self._command_history),
                # B12 fix (Phase 4): ``cmd`` here iterates RoyalCommand enum
                # members directly. The previous code referenced ``cmd.command``
                # (an attribute the enum does not have) which would AttributeError
                # at runtime; mypy strict on this file caught it. Use ``cmd``
                # itself for both the dict key (its name) and the equality check.
                "commands_by_type": {
                    cmd.name: sum(1 for c in self._pending_commands if c.command == cmd)
                    for cmd in RoyalCommand
                },
            }


# ═══════════════════════════════════════════════════════════════════════════════
# NECTAR FLOW - SISTEMA UNIFICADO
# ═══════════════════════════════════════════════════════════════════════════════


class NectarPriority(Enum):
    """Prioridad de canales de comunicación."""

    LOW = 1  # Feromonas pasivas
    MEDIUM = 5  # Danzas normales
    HIGH = 8  # Comandos importantes
    CRITICAL = 10  # Royal Jelly


@dataclass
class NectarChannel:
    """Un canal de comunicación en el sistema."""

    name: str
    priority: NectarPriority
    buffer_size: int = 1000
    _queue: deque[Any] = field(default_factory=lambda: deque(maxlen=1000))


class NectarFlow:
    """
    Sistema Unificado de Comunicación del Panal.

    Integra todos los subsistemas de comunicación:
    - Feromonas (comunicación pasiva/ambiental)
    - Waggle Dance (comunicación activa/direccional)
    - Royal Jelly (canal de alta prioridad)

    Actualización por tick (topología hexagonal):
    1. Evaporación (decaimiento) de feromonas
    2. Difusión a los 6 vecinos por celda (si diffusion_rate > 0)
    3. Propagación de danzas
    """

    def __init__(
        self,
        grid: HoneycombGrid,
        pheromone_diffusion_rate: float = 0.05,
    ):
        self.grid = grid
        self._pheromone_diffusion_rate = max(0.0, min(1.0, pheromone_diffusion_rate))
        self._pheromones = PheromoneTrail()
        self._dance = WaggleDance()
        self._royal = RoyalJelly(grid.queen.coord if grid.queen else HexCoord.origin())
        self._lock = threading.RLock()

        # Registrar celdas como suscriptores del canal real
        for coord in grid._cells:
            self._royal.subscribe(coord)

    # ─────────────────────────────────────────────────────────────────────────
    # FEROMONAS
    # ─────────────────────────────────────────────────────────────────────────

    def deposit_pheromone(
        self, coord: HexCoord, ptype: PheromoneType, intensity: float, **kwargs: Any
    ) -> float:
        """Deposita feromona en una coordenada."""
        return self._pheromones.deposit(coord, ptype, intensity, **kwargs)

    def sense_pheromone(self, coord: HexCoord, ptype: PheromoneType | None = None) -> float:
        """Detecta feromona en una coordenada."""
        return self._pheromones.sense(coord, ptype)

    def follow_pheromone(self, coord: HexCoord, ptype: PheromoneType) -> HexDirection | None:
        """Sigue el gradiente de feromona."""
        return self._pheromones.follow_gradient(coord, ptype)

    # ─────────────────────────────────────────────────────────────────────────
    # WAGGLE DANCE
    # ─────────────────────────────────────────────────────────────────────────

    def start_dance(
        self,
        dancer: HexCoord,
        direction: DanceDirection,
        distance: int,
        quality: float,
        resource_type: str = "generic",
        **kwargs: Any,
    ) -> DanceMessage:
        """Inicia una danza."""
        return self._dance.start_dance(
            dancer, direction, distance, quality, resource_type, **kwargs
        )

    def observe_dances(self, observer: HexCoord, radius: int = 1) -> list[DanceMessage]:
        """Observa danzas cercanas."""
        return self._dance.observe_dances(observer, radius)

    # ─────────────────────────────────────────────────────────────────────────
    # ROYAL JELLY
    # ─────────────────────────────────────────────────────────────────────────

    def royal_command(
        self,
        command: RoyalCommand,
        priority: int = 5,
        target: HexCoord | None = None,
        params: dict[str, Any] | None = None,
        *,
        issuer: HexCoord | None = None,
    ) -> RoyalMessage:
        """
        Emite un comando real.

        Phase 2: para ``priority >= 8`` se requiere ``issuer`` y debe ser
        la QueenCell actual (enforced en ``RoyalJelly.issue_command``).
        """
        return self._royal.issue_command(command, priority, target, params, issuer=issuer)

    def get_royal_commands(self, cell_coord: HexCoord, limit: int = 10) -> list[RoyalMessage]:
        """Obtiene comandos reales pendientes para una celda."""
        return self._royal.get_commands(cell_coord, limit)

    def acknowledge_command(self, command: RoyalMessage, cell_coord: HexCoord) -> None:
        """Reconoce un comando real."""
        self._royal.acknowledge(command, cell_coord)

    # ─────────────────────────────────────────────────────────────────────────
    # SISTEMA GLOBAL
    # ─────────────────────────────────────────────────────────────────────────

    # Phase 7.2: one-shot guard so ``run_tick_sync`` emits its
    # DeprecationWarning only the first time it's called per process.
    _SYNC_DEPRECATION_EMITTED: ClassVar[bool] = False

    async def tick(self) -> dict[str, Any]:
        """Phase 7.1: async tick.

        Pheromone decay / diffusion + waggle dance propagation are
        CPU-light but holds the lock for the entire pass; we dispatch
        the body to ``asyncio.to_thread`` so the event loop stays
        responsive when many flows tick concurrently (e.g. a
        multi-grid simulation in Phase 8).

        Order preserved: decay → diffusion → dance propagation → old
        dance cleanup → command count.
        """
        return await asyncio.to_thread(self._sync_tick)

    def _sync_tick(self) -> dict[str, Any]:
        """Phase 7.1: the pre-Phase-7.1 ``tick`` body, kept synchronous
        so :class:`PheromoneTrail` and :class:`WaggleDance` locking
        works without async re-plumbing."""
        results = {
            "pheromones_evaporated": 0,
            "pheromones_diffused": 0,
            "dances_propagated": 0,
            "commands_pending": 0,
        }

        # 1. Decaimiento (evaporación)
        results["pheromones_evaporated"] = self._pheromones.evaporate()

        # 2. Difusión en topología hexagonal (solo a celdas existentes en el grid)
        if self._pheromone_diffusion_rate > 0:
            valid = set(self.grid._cells.keys())
            results["pheromones_diffused"] = self._pheromones.diffuse_to_neighbors(
                diffusion_rate=self._pheromone_diffusion_rate,
                valid_coords=valid,
            )

        # 3. Propagación de danzas
        results["dances_propagated"] = self._dance.propagate(self.grid)

        # Limpiar danzas viejas
        self._dance.clear_old_dances()

        # Comandos pendientes
        results["commands_pending"] = self._royal.get_pending_count()

        return results

    def run_tick_sync(self) -> dict[str, Any]:
        """Phase 7.2: blocking wrapper for legacy sync callers.

        Equivalent to ``asyncio.run(self.tick())``. Emits
        :class:`DeprecationWarning` once per process. Raises
        ``RuntimeError`` if called from inside a running loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "NectarFlow.run_tick_sync called from a running event loop; "
                "use 'await flow.tick()' instead."
            )
        if not NectarFlow._SYNC_DEPRECATION_EMITTED:
            warnings.warn(
                "NectarFlow.run_tick_sync is a v1→v2 migration aid; switch "
                "callers to 'await flow.tick()'. This wrapper will be "
                "removed in HOC v3.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            NectarFlow._SYNC_DEPRECATION_EMITTED = True
        return self._sync_tick()

    def get_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas consolidadas."""
        return {
            "pheromones": self._pheromones.get_stats(),
            "dance": self._dance.get_stats(),
            "royal": self._royal.get_stats(),
        }

    @property
    def pheromones(self) -> PheromoneTrail:
        return self._pheromones

    @property
    def dance(self) -> WaggleDance:
        return self._dance

    @property
    def royal(self) -> RoyalJelly:
        return self._royal
