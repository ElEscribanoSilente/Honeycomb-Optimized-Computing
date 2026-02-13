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

import math
import time
import heapq
import threading
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Set, Tuple, Callable,
    Any, Iterator, Deque, TypeVar,
)
from collections import defaultdict, deque
import numpy as np

from .core import HexCoord, HexDirection, HoneycombCell, HoneycombGrid

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ═══════════════════════════════════════════════════════════════════════════════
# TIPOS DE FEROMONA
# ═══════════════════════════════════════════════════════════════════════════════

class PheromoneType(Enum):
    """Tipos de feromonas con diferentes propósitos."""
    
    # Rastros de trabajo
    TRAIL = auto()           # Camino general
    FOOD = auto()            # Recurso encontrado
    DANGER = auto()          # Peligro/error detectado
    
    # Señales de estado
    BUSY = auto()            # Celda ocupada
    AVAILABLE = auto()       # Celda disponible
    
    # Coordinación
    RECRUITMENT = auto()     # Reclutamiento de ayuda
    ALARM = auto()           # Alerta general
    
    # Optimización
    SUCCESS = auto()         # Tarea completada exitosamente
    FAILURE = auto()         # Tarea fallida
    
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


@dataclass
class PheromoneDeposit:
    """Un depósito individual de feromona."""
    ptype: PheromoneType
    intensity: float
    timestamp: float
    source: Optional[HexCoord] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def decay(self, elapsed: float) -> float:
        """Aplica decaimiento basado en tiempo transcurrido."""
        decay_factor = math.exp(-self.ptype.decay_rate() * elapsed)
        self.intensity *= decay_factor
        return self.intensity


class PheromoneDecay(Enum):
    """Estrategias de decaimiento."""
    EXPONENTIAL = auto()   # Decae exponencialmente
    LINEAR = auto()        # Decae linealmente
    STEP = auto()          # Decae en escalones
    NONE = auto()          # No decae


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

    def __init__(
        self,
        decay_strategy: PheromoneDecay = PheromoneDecay.EXPONENTIAL,
        max_intensity: float = 10.0,
        evaporation_interval: float = 1.0
    ):
        self._deposits: Dict[HexCoord, Dict[PheromoneType, PheromoneDeposit]] = defaultdict(dict)
        self._decay_strategy = decay_strategy
        self._max_intensity = max_intensity
        self._evaporation_interval = evaporation_interval
        self._last_evaporation = time.time()
        self._lock = threading.RLock()
    
    def deposit(
        self,
        coord: HexCoord,
        ptype: PheromoneType,
        intensity: float,
        source: Optional[HexCoord] = None,
        metadata: Optional[Dict] = None
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
            if ptype in self._deposits[coord]:
                deposit = self._deposits[coord][ptype]
                deposit.intensity = min(
                    self._max_intensity,
                    deposit.intensity + intensity
                )
                deposit.timestamp = time.time()
                if metadata:
                    deposit.metadata.update(metadata)
            else:
                self._deposits[coord][ptype] = PheromoneDeposit(
                    ptype=ptype,
                    intensity=min(self._max_intensity, intensity),
                    timestamp=time.time(),
                    source=source,
                    metadata=metadata or {}
                )
            
            return self._deposits[coord][ptype].intensity
    
    def sense(self, coord: HexCoord, ptype: Optional[PheromoneType] = None) -> float:
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
        self,
        center: HexCoord,
        radius: int,
        ptype: Optional[PheromoneType] = None
    ) -> Dict[HexCoord, float]:
        """Detecta feromonas en un área."""
        result = {}
        for coord in center.spiral(radius):
            level = self.sense(coord, ptype)
            if level > 0:
                result[coord] = level
        return result
    
    def follow_gradient(
        self,
        coord: HexCoord,
        ptype: PheromoneType,
        prefer_unexplored: bool = True
    ) -> Optional[HexDirection]:
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
                    elif self._decay_strategy == PheromoneDecay.STEP:
                        if elapsed > (1.0 / ptype.decay_rate()):
                            deposit.intensity *= 0.5
                    
                    # Marcar para eliminación si es muy bajo
                    if deposit.intensity < self.CLEANUP_THRESHOLD:
                        dead_types.append(ptype)
                
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
        diffusion_rate: Optional[float] = None,
        valid_coords: Optional[Set[HexCoord]] = None,
        threshold: Optional[float] = None
    ) -> int:
        """
        Difunde feromonas a los 6 vecinos hexagonales de cada celda con depósitos.
        Respeta la topología hexagonal: cada celda tiene exactamente 6 direcciones.
        Debe llamarse después del decaimiento en cada tick.
        """
        diffusion_rate = diffusion_rate if diffusion_rate is not None else self.DEFAULT_DIFFUSION_RATE
        threshold = threshold if threshold is not None else self.DEFAULT_DIFFUSE_THRESHOLD
        if diffusion_rate <= 0 or diffusion_rate >= 1:
            return 0
        spread_per_neighbor = diffusion_rate / 6.0
        with self._lock:
            new_deposits: List[Tuple[HexCoord, PheromoneType, float, Optional[HexCoord], Optional[Dict]]] = []
            for coord, deposits in self._deposits.items():
                for ptype, deposit in list(deposits.items()):
                    if deposit.intensity < threshold:
                        continue
                    amount = deposit.intensity * spread_per_neighbor
                    for direction in HexDirection:
                        neighbor_coord = coord.neighbor(direction)
                        if valid_coords is not None and neighbor_coord not in valid_coords:
                            continue
                        new_deposits.append((
                            neighbor_coord,
                            ptype,
                            amount,
                            coord,
                            None,
                        ))
            for coord, ptype, intensity, source, meta in new_deposits:
                self.deposit(coord, ptype, intensity, source=source, metadata=meta)
        return len(new_deposits)
    
    def get_hotspots(
        self,
        ptype: PheromoneType,
        threshold: Optional[float] = None,
        limit: int = 10
    ) -> List[Tuple[HexCoord, float]]:
        """Obtiene las ubicaciones con mayor concentración."""
        threshold = threshold if threshold is not None else self.DEFAULT_HOTSPOT_THRESHOLD
        with self._lock:
            hotspots = []
            for coord, deposits in self._deposits.items():
                if ptype in deposits and deposits[ptype].intensity >= threshold:
                    hotspots.append((coord, deposits[ptype].intensity))
            
            hotspots.sort(key=lambda x: x[1], reverse=True)
            return hotspots[:limit]
    
    def clear(self, coord: Optional[HexCoord] = None, ptype: Optional[PheromoneType] = None) -> None:
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de feromonas."""
        with self._lock:
            total_deposits = sum(len(d) for d in self._deposits.values())
            total_intensity = sum(
                d.intensity
                for deposits in self._deposits.values()
                for d in deposits.values()
            )
            
            by_type = defaultdict(float)
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
    UP = 0          # Norte (referencia solar)
    UP_RIGHT = 60
    RIGHT = 90
    DOWN_RIGHT = 120
    DOWN = 180
    DOWN_LEFT = 240
    LEFT = 270
    UP_LEFT = 300
    
    @classmethod
    def from_angle(cls, angle: float) -> 'DanceDirection':
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
    """
    source: HexCoord              # Quien baila
    direction: DanceDirection     # Hacia dónde
    distance: int                 # Qué tan lejos (en celdas)
    quality: float                # Qué tan bueno (0.0 - 1.0)
    resource_type: str            # Tipo de recurso
    timestamp: float = field(default_factory=time.time)
    ttl: int = 10                 # Time to live (broadcasts restantes)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def encode(self) -> bytes:
        """Codifica el mensaje para transmisión."""
        # Formato compacto: tipo|dir|dist|quality
        data = f"{self.resource_type}|{self.direction.value}|{self.distance}|{self.quality:.2f}"
        return data.encode()
    
    @classmethod
    def decode(cls, data: bytes, source: HexCoord) -> 'DanceMessage':
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
        
        return HexCoord(
            self.source.q + dq * self.distance,
            self.source.r + dr * self.distance
        )


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
        self,
        broadcast_range: int = 5,
        attenuation: float = 0.8,
        competition_threshold: float = 0.3
    ):
        self._active_dances: Dict[HexCoord, List[DanceMessage]] = defaultdict(list)
        self._broadcast_range = broadcast_range
        self._attenuation = attenuation
        self._competition_threshold = competition_threshold
        self._lock = threading.RLock()
        self._observers: List[Callable[[DanceMessage], None]] = []
    
    def start_dance(
        self,
        dancer: HexCoord,
        direction: DanceDirection,
        distance: int,
        quality: float,
        resource_type: str = "generic",
        metadata: Optional[Dict] = None
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
            metadata=metadata or {}
        )
        
        with self._lock:
            self._active_dances[dancer].append(message)
            
            # Notificar observadores
            for observer in self._observers:
                try:
                    observer(message)
                except Exception as e:
                    logger.error(f"Dance observer error: {e}")
        
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
                        
                        # Crear mensaje atenuado
                        propagated_dance = DanceMessage(
                            source=dance.source,
                            direction=dance.direction,
                            distance=dance.distance,
                            quality=dance.quality * attenuation,
                            resource_type=dance.resource_type,
                            timestamp=dance.timestamp,
                            ttl=dance.ttl - 1,
                            metadata=dance.metadata.copy()
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
    
    def observe_dances(
        self,
        observer: HexCoord,
        radius: int = 1
    ) -> List[DanceMessage]:
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
                    d for d in self._active_dances[coord]
                    if (now - d.timestamp) < max_age
                ]
                removed += original_count - len(self._active_dances[coord])
                
                if not self._active_dances[coord]:
                    del self._active_dances[coord]
        
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de danza."""
        with self._lock:
            total_dances = sum(len(d) for d in self._active_dances.values())
            by_type = defaultdict(int)
            
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
    SWARM = auto()           # Iniciar enjambre (migración masiva)
    HIBERNATE = auto()       # Entrar en hibernación
    WAKE = auto()            # Despertar
    EVACUATE = auto()        # Evacuar área
    REINFORCE = auto()       # Reforzar área
    BALANCE = auto()         # Balancear carga
    SPAWN = auto()           # Crear nuevas entidades
    CULL = auto()            # Reducir población
    EMERGENCY = auto()       # Emergencia general


@dataclass
class RoyalMessage:
    """Mensaje de la reina."""
    command: RoyalCommand
    priority: int                   # 0-10 (10 = máxima)
    target: Optional[HexCoord]      # Destino específico o None para broadcast
    params: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    acknowledged: Set[HexCoord] = field(default_factory=set)


class RoyalJelly:
    """
    Canal de comunicación de alta prioridad Reina → Colmena.
    
    Características:
    - Entrega garantizada a todas las celdas
    - Prioridad sobre otros tipos de comunicación
    - Acknowledgement de recepción
    - Cola de comandos pendientes
    """
    
    def __init__(self, queen_coord: HexCoord):
        self._queen_coord = queen_coord
        self._pending_commands: List[RoyalMessage] = []
        self._command_history: Deque[RoyalMessage] = deque(maxlen=100)
        self._lock = threading.RLock()
        self._subscribers: Set[HexCoord] = set()
    
    def issue_command(
        self,
        command: RoyalCommand,
        priority: int = 5,
        target: Optional[HexCoord] = None,
        params: Optional[Dict] = None
    ) -> RoyalMessage:
        """
        Emite un comando real.
        
        Args:
            command: Tipo de comando
            priority: Prioridad (0-10)
            target: Destino específico o None para broadcast
            params: Parámetros adicionales
            
        Returns:
            El mensaje creado
        """
        message = RoyalMessage(
            command=command,
            priority=min(10, max(0, priority)),
            target=target,
            params=params or {},
        )
        
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
        
        logger.info(f"Royal command issued: {command.name} (priority={priority})")
        return message
    
    def subscribe(self, coord: HexCoord) -> None:
        """Suscribe una celda al canal real."""
        with self._lock:
            self._subscribers.add(coord)
    
    def unsubscribe(self, coord: HexCoord) -> None:
        """Desuscribe una celda."""
        with self._lock:
            self._subscribers.discard(coord)
    
    def get_commands(
        self,
        cell_coord: HexCoord,
        limit: int = 10
    ) -> List[RoyalMessage]:
        """
        Obtiene comandos pendientes para una celda.
        
        Args:
            cell_coord: Coordenada de la celda
            limit: Máximo de comandos a retornar
            
        Returns:
            Lista de comandos aplicables
        """
        with self._lock:
            applicable = []
            for cmd in self._pending_commands:
                if len(applicable) >= limit:
                    break
                
                # Comando es aplicable si:
                # 1. Es broadcast (target=None)
                # 2. Es para esta celda específica
                # 3. No ha sido ya reconocido por esta celda
                if cmd.target is None or cmd.target == cell_coord:
                    if cell_coord not in cmd.acknowledged:
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
            if command.target is not None and cell_coord == command.target:
                if command in self._pending_commands:
                    self._pending_commands.remove(command)
                    self._command_history.append(command)
            
            # Si es broadcast y todos reconocieron, mover a historial
            elif command.target is None:
                if self._subscribers.issubset(command.acknowledged):
                    if command in self._pending_commands:
                        self._pending_commands.remove(command)
                        self._command_history.append(command)
    
    def emergency_broadcast(self, message: str, params: Optional[Dict] = None) -> None:
        """Emite una alerta de emergencia con máxima prioridad."""
        self.issue_command(
            RoyalCommand.EMERGENCY,
            priority=10,
            target=None,
            params={"message": message, **(params or {})}
        )
    
    def get_pending_count(self) -> int:
        """Retorna el número de comandos pendientes."""
        return len(self._pending_commands)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del canal."""
        with self._lock:
            return {
                "pending_commands": len(self._pending_commands),
                "subscribers": len(self._subscribers),
                "history_size": len(self._command_history),
                "commands_by_type": {
                    cmd.command.name: sum(
                        1 for c in self._pending_commands
                        if c.command == cmd.command
                    )
                    for cmd in RoyalCommand
                },
            }


# ═══════════════════════════════════════════════════════════════════════════════
# NECTAR FLOW - SISTEMA UNIFICADO
# ═══════════════════════════════════════════════════════════════════════════════

class NectarPriority(Enum):
    """Prioridad de canales de comunicación."""
    LOW = 1        # Feromonas pasivas
    MEDIUM = 5     # Danzas normales
    HIGH = 8       # Comandos importantes
    CRITICAL = 10  # Royal Jelly


@dataclass
class NectarChannel:
    """Un canal de comunicación en el sistema."""
    name: str
    priority: NectarPriority
    buffer_size: int = 1000
    _queue: Deque = field(default_factory=lambda: deque(maxlen=1000))


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
        for coord in grid._cells.keys():
            self._royal.subscribe(coord)
    
    # ─────────────────────────────────────────────────────────────────────────
    # FEROMONAS
    # ─────────────────────────────────────────────────────────────────────────
    
    def deposit_pheromone(
        self,
        coord: HexCoord,
        ptype: PheromoneType,
        intensity: float,
        **kwargs
    ) -> float:
        """Deposita feromona en una coordenada."""
        return self._pheromones.deposit(coord, ptype, intensity, **kwargs)
    
    def sense_pheromone(
        self,
        coord: HexCoord,
        ptype: Optional[PheromoneType] = None
    ) -> float:
        """Detecta feromona en una coordenada."""
        return self._pheromones.sense(coord, ptype)
    
    def follow_pheromone(
        self,
        coord: HexCoord,
        ptype: PheromoneType
    ) -> Optional[HexDirection]:
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
        **kwargs
    ) -> DanceMessage:
        """Inicia una danza."""
        return self._dance.start_dance(
            dancer, direction, distance, quality, resource_type, **kwargs
        )
    
    def observe_dances(
        self,
        observer: HexCoord,
        radius: int = 1
    ) -> List[DanceMessage]:
        """Observa danzas cercanas."""
        return self._dance.observe_dances(observer, radius)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ROYAL JELLY
    # ─────────────────────────────────────────────────────────────────────────
    
    def royal_command(
        self,
        command: RoyalCommand,
        priority: int = 5,
        target: Optional[HexCoord] = None,
        params: Optional[Dict] = None
    ) -> RoyalMessage:
        """Emite un comando real."""
        return self._royal.issue_command(command, priority, target, params)
    
    def get_royal_commands(
        self,
        cell_coord: HexCoord,
        limit: int = 10
    ) -> List[RoyalMessage]:
        """Obtiene comandos reales pendientes para una celda."""
        return self._royal.get_commands(cell_coord, limit)
    
    def acknowledge_command(self, command: RoyalMessage, cell_coord: HexCoord) -> None:
        """Reconoce un comando real."""
        self._royal.acknowledge(command, cell_coord)
    
    # ─────────────────────────────────────────────────────────────────────────
    # SISTEMA GLOBAL
    # ─────────────────────────────────────────────────────────────────────────
    
    def tick(self) -> Dict[str, Any]:
        """
        Ejecuta un tick del sistema de comunicación (orden: decaimiento → difusión → danzas).
        
        - Evapora feromonas (decaimiento)
        - Difunde feromonas a los 6 vecinos hexagonales (si diffusion_rate > 0)
        - Propaga danzas waggle
        - Procesa cola de comandos reales
        """
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
    
    def get_stats(self) -> Dict[str, Any]:
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
