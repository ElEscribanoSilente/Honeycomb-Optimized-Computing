"""
HOC Resilience - Sistema de Resiliencia del Panal
===================================================

Implementa tolerancia a fallos y recuperación usando
metáforas de colmena:

FAILOVER:
- Detección de celdas fallidas
- Migración automática de trabajo
- Redistribución de carga

REPLICACIÓN:
- Espejo hexagonal en vecinos
- Redundancia por anillos
- Quorum para consistencia

SUCESIÓN DE REINA:
- Detección de pérdida de reina
- Elección de nueva reina
- Transferencia de coordinación

RECUPERACIÓN:
- Reparación de celdas dañadas
- Reconstrucción de datos
- Re-balanceo post-fallo

Diagrama de resiliencia:

    ┌────────────────────────────────────────────────────────────┐
    │                    HiveResilience                          │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
    │  │ HealthCheck │  │  Failover   │  │  Recovery   │        │
    │  │   Monitor   │  │   Handler   │  │   Manager   │        │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
    │         │                │                │                │
    │         └────────────────┼────────────────┘                │
    │                          │                                 │
    │                    ┌─────▼─────┐                          │
    │                    │  Cell     │                          │
    │                    │  Manager  │                          │
    │                    └───────────┘                          │
    │                                                            │
    │  ⬡ ⬡ ⬡ ⬡ ⬡  →  ⬡ ✗ ⬡ ⬡ ⬡  →  ⬡ ⬡ ⬡ ⬡ ⬡                 │
    │   (healthy)      (detected)      (recovered)              │
    └────────────────────────────────────────────────────────────┘

"""

from __future__ import annotations

import time
import threading
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Set, Tuple, Callable,
    Any, TypeVar, Deque
)
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import mscs as _mscs

from .core import (
    HexCoord, HexDirection, HoneycombGrid, HoneycombCell,
    QueenCell, WorkerCell, DroneCell, NurseryCell,
    CellRole, CellState, HoneycombConfig
)
from .security import (
    sign_payload as _sign_payload,
    verify_signature as _verify_signature,
    sanitize_error,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ═══════════════════════════════════════════════════════════════════════════════
# ESTADOS Y EVENTOS DE RESILIENCIA
# ═══════════════════════════════════════════════════════════════════════════════

class HealthStatus(Enum):
    """Estado de salud de una celda."""
    HEALTHY = auto()      # Funcionando correctamente
    DEGRADED = auto()     # Funcionando con limitaciones
    UNHEALTHY = auto()    # Fallando intermitentemente
    FAILED = auto()       # Completamente fallido
    RECOVERING = auto()   # En proceso de recuperación
    UNKNOWN = auto()      # Estado desconocido


class FailureType(Enum):
    """Tipos de fallo detectables."""
    TIMEOUT = auto()           # No responde
    ERROR_THRESHOLD = auto()   # Demasiados errores
    MEMORY_EXHAUSTED = auto()  # Sin memoria
    OVERLOAD = auto()          # Sobrecargado
    CORRUPTION = auto()        # Datos corruptos
    NETWORK = auto()           # Fallo de comunicación
    HARDWARE = auto()          # Fallo de hardware subyacente


class RecoveryAction(Enum):
    """Acciones de recuperación."""
    RESTART = auto()       # Reiniciar celda
    MIGRATE = auto()       # Migrar trabajo
    REPLICATE = auto()     # Replicar desde espejo
    REBUILD = auto()       # Reconstruir desde cero
    QUARANTINE = auto()    # Aislar celda
    ESCALATE = auto()      # Escalar a intervención manual


@dataclass
class HealthReport:
    """Reporte de salud de una celda."""
    coord: HexCoord
    status: HealthStatus
    failure_type: Optional[FailureType] = None
    error_count: int = 0
    last_check: float = field(default_factory=time.time)
    response_time_ms: float = 0.0
    load: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
    
    @property
    def needs_attention(self) -> bool:
        return self.status in (HealthStatus.UNHEALTHY, HealthStatus.FAILED)


@dataclass
class FailoverEvent:
    """Evento de failover registrado."""
    source_coord: HexCoord
    target_coord: Optional[HexCoord]
    failure_type: FailureType
    recovery_action: RecoveryAction
    timestamp: float = field(default_factory=time.time)
    success: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResilienceConfig:
    """Configuración del sistema de resiliencia."""
    
    # Health checking
    health_check_interval_ticks: int = 5
    timeout_threshold_ms: float = 1000.0
    error_threshold: int = 3
    degraded_load_threshold: float = 0.8
    
    # Failover
    failover_timeout_ms: float = 5000.0
    max_failover_attempts: int = 3
    failover_cooldown_ticks: int = 10
    
    # Replicación
    replication_factor: int = 2
    mirror_on_neighbors: bool = True
    quorum_size: int = 2
    
    # Sucesión de reina
    queen_heartbeat_interval: int = 3
    queen_election_timeout_ticks: int = 10
    min_queen_candidates: int = 3
    
    # Recuperación
    auto_recovery: bool = True
    recovery_batch_size: int = 5
    max_concurrent_recoveries: int = 3


# ═══════════════════════════════════════════════════════════════════════════════
# CELL FAILOVER
# ═══════════════════════════════════════════════════════════════════════════════

class CellFailover:
    """
    Maneja el failover de celdas individuales.
    
    Detecta fallos, migra trabajo a celdas saludables,
    y coordina la recuperación.
    
    Uso:
        failover = CellFailover(grid, config)
        failover.handle_failure(coord, FailureType.TIMEOUT)
        target = failover.find_failover_target(coord)
    """
    
    def __init__(self, grid: HoneycombGrid, config: ResilienceConfig):
        self.grid = grid
        self.config = config
        
        # Estado
        self._failed_cells: Set[HexCoord] = set()
        self._failover_history: Deque[FailoverEvent] = deque(maxlen=100)
        self._cooldowns: Dict[HexCoord, int] = {}
        self._lock = threading.RLock()
    
    def handle_failure(
        self,
        coord: HexCoord,
        failure_type: FailureType
    ) -> FailoverEvent:
        """
        Maneja un fallo de celda.
        
        Args:
            coord: Coordenada de la celda fallida
            failure_type: Tipo de fallo
            
        Returns:
            Evento de failover con resultado
        """
        event = FailoverEvent(
            source_coord=coord,
            target_coord=None,
            failure_type=failure_type,
            recovery_action=RecoveryAction.MIGRATE,
        )
        
        with self._lock:
            # Verificar cooldown
            if coord in self._cooldowns and self._cooldowns[coord] > 0:
                logger.warning(f"Cell {coord} in failover cooldown")
                event.details["skipped"] = "cooldown"
                return event
            
            # Marcar como fallida
            self._failed_cells.add(coord)
            
            # Encontrar destino
            target = self.find_failover_target(coord)
            event.target_coord = target
            
            if not target:
                event.recovery_action = RecoveryAction.QUARANTINE
                event.details["error"] = "no_target_available"
                self._failover_history.append(event)
                return event
            
            # Ejecutar migración
            success = self._migrate_work(coord, target)
            event.success = success
            
            if success:
                self._cooldowns[coord] = self.config.failover_cooldown_ticks
                logger.info(f"Failover {coord} → {target} successful")
            else:
                logger.error(f"Failover {coord} → {target} failed")
            
            self._failover_history.append(event)
            return event
    
    def find_failover_target(self, failed_coord: HexCoord) -> Optional[HexCoord]:
        """
        Encuentra una celda destino para failover.
        
        Prioriza:
        1. Vecinos directos saludables
        2. Celdas en el mismo anillo
        3. Cualquier celda disponible
        """
        failed_cell = self.grid.get_cell(failed_coord)
        if not failed_cell:
            return None
        
        # 1. Buscar en vecinos
        for neighbor in failed_cell.get_all_neighbors():
            if (neighbor and 
                isinstance(neighbor, WorkerCell) and
                neighbor.coord not in self._failed_cells and
                neighbor.is_available and
                neighbor.load < self.config.degraded_load_threshold):
                return neighbor.coord
        
        # 2. Buscar en el mismo anillo
        radius = failed_coord.distance_to(HexCoord.origin())
        for coord in failed_coord.ring(radius):
            cell = self.grid.get_cell(coord)
            if (cell and 
                isinstance(cell, WorkerCell) and
                coord not in self._failed_cells and
                cell.is_available and
                cell.load < self.config.degraded_load_threshold):
                return coord
        
        # 3. Buscar cualquier celda disponible
        available = self.grid.find_available_cells(1)
        for cell in available:
            if cell.coord not in self._failed_cells:
                return cell.coord
        
        return None
    
    def _migrate_work(self, source: HexCoord, target: HexCoord) -> bool:
        """Migra el trabajo de una celda a otra."""
        source_cell = self.grid.get_cell(source)
        target_cell = self.grid.get_cell(target)
        
        if not source_cell or not target_cell:
            return False
        
        try:
            # Migrar vCores
            for vcore in list(source_cell._vcores):
                if target_cell.add_vcore(vcore):
                    source_cell.remove_vcore(vcore)
            
            # Actualizar estado
            source_cell.state = CellState.FAILED
            
            return True
        except Exception as e:
            logger.error(f"Migration error: {sanitize_error(e)}")
            return False
    
    def mark_recovered(self, coord: HexCoord) -> bool:
        """Marca una celda como recuperada."""
        with self._lock:
            if coord in self._failed_cells:
                self._failed_cells.remove(coord)
                cell = self.grid.get_cell(coord)
                if cell:
                    cell.state = CellState.IDLE
                return True
            return False
    
    def tick(self) -> None:
        """Procesa un tick (actualiza cooldowns)."""
        with self._lock:
            for coord in list(self._cooldowns.keys()):
                self._cooldowns[coord] -= 1
                if self._cooldowns[coord] <= 0:
                    del self._cooldowns[coord]
    
    def get_failed_cells(self) -> Set[HexCoord]:
        """Retorna las celdas actualmente fallidas."""
        return self._failed_cells.copy()
    
    def get_failover_history(self) -> List[FailoverEvent]:
        """Retorna el historial de failovers."""
        return list(self._failover_history)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de failover."""
        total_events = len(self._failover_history)
        successful = sum(1 for e in self._failover_history if e.success)
        
        return {
            "failed_cells": len(self._failed_cells),
            "total_failovers": total_events,
            "successful_failovers": successful,
            "success_rate": successful / total_events if total_events else 1.0,
            "cells_in_cooldown": len(self._cooldowns),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# QUEEN SUCCESSION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Vote:
    """
    Un voto firmado en una elección de reina.

    Phase 2 — Raft-like:
    - ``term`` monótono y acotado al término de la elección en curso.
    - ``signature`` HMAC-SHA256 cubre (voter, candidate, term, timestamp).
    - Un voter no puede aparecer en más de un voto por term (enforced al tallyar).
    """
    voter: HexCoord
    candidate: HexCoord
    term: int
    timestamp: float = field(default_factory=time.time)
    signature: Optional[bytes] = None

    def _canonical_payload(self) -> bytes:
        return _mscs.dumps({
            "kind": "vote",
            "voter": (self.voter.q, self.voter.r),
            "candidate": (self.candidate.q, self.candidate.r),
            "term": self.term,
            "timestamp": round(self.timestamp, 6),
        })

    def sign(self, key: Optional[bytes] = None) -> "Vote":
        """Firma el voto con HMAC-SHA256. Retorna ``self``."""
        self.signature = _sign_payload(self._canonical_payload(), key=key)
        return self

    def verify(self, key: Optional[bytes] = None) -> bool:
        """True si la firma existe y es válida."""
        if self.signature is None:
            return False
        return _verify_signature(self._canonical_payload(), self.signature, key=key)


class QueenSuccession:
    """
    Gestiona la sucesión de la reina en caso de fallo.

    Implementa un protocolo de elección similar a Raft, adaptado a la
    topología hexagonal (Phase 2 hardening):

    - ``term_number`` monotónico por elección. Votos de términos antiguos
      son descartados.
    - Votos firmados con HMAC-SHA256 (``Vote.sign/verify``).
    - Rechazo de votos duplicados (cada voter cuenta una sola vez por term).
    - Quórum numérico MAYORITARIO (``>50%``) — el fix B4 de Fase 1 exigía
      mayoría; Fase 2 añade la autenticación criptográfica necesaria para
      que la mayoría sea vinculante frente a votos forjados.

    Proceso de sucesión:
    1. Detectar pérdida de heartbeat de la reina
    2. Iniciar período de elección (incrementa term)
    3. Celdas candidatas proponen liderazgo
    4. Votación firmada ponderada por distancia/carga
    5. Tally con rechazo de duplicados/firmas inválidas
    6. Nueva reina asume coordinación si hay mayoría

    Uso:
        succession = QueenSuccession(grid, config)
        if succession.check_queen_health():
            print("Queen healthy")
        else:
            new_queen = succession.elect_new_queen()
    """

    def __init__(self, grid: HoneycombGrid, config: ResilienceConfig):
        self.grid = grid
        self.config = config

        # Estado
        self._last_queen_heartbeat = time.time()
        self._election_in_progress = False
        self._candidates: List[HexCoord] = []
        self._votes: Dict[HexCoord, int] = {}
        # Phase 2: term number monotónico por elección (Raft-like).
        self._term_number: int = 0
        self._lock = threading.RLock()

    @property
    def current_term(self) -> int:
        """Retorna el último ``term`` incrementado por ``_conduct_election``."""
        return self._term_number
    
    def check_queen_health(self) -> bool:
        """
        Verifica la salud de la reina actual.
        
        Returns:
            True si la reina está saludable
        """
        queen = self.grid.queen
        
        if not queen:
            return False
        
        # Verificar estado
        if queen.state == CellState.FAILED:
            return False
        
        # Verificar heartbeat (en implementación real)
        elapsed = time.time() - self._last_queen_heartbeat
        timeout = self.config.queen_heartbeat_interval * 2
        
        return elapsed < timeout
    
    def register_heartbeat(self) -> None:
        """Registra un heartbeat de la reina."""
        self._last_queen_heartbeat = time.time()
    
    def elect_new_queen(self) -> Optional[QueenCell]:
        """
        Inicia una elección para nueva reina.
        
        Returns:
            Nueva QueenCell o None si la elección falló
        """
        with self._lock:
            if self._election_in_progress:
                logger.warning("Election already in progress")
                return None
            
            self._election_in_progress = True
            self._candidates.clear()
            self._votes.clear()
        
        try:
            # 1. Identificar candidatos
            candidates = self._identify_candidates()
            
            if len(candidates) < self.config.min_queen_candidates:
                logger.error("Not enough candidates for election")
                return None
            
            # 2. Votación
            winner = self._conduct_election(candidates)
            
            if not winner:
                logger.error("Election failed - no winner")
                return None
            
            # 3. Promover ganador
            new_queen = self._promote_to_queen(winner)
            
            if new_queen:
                logger.info(f"New queen elected at {winner}")
            
            return new_queen
            
        finally:
            with self._lock:
                self._election_in_progress = False
    
    def _identify_candidates(self) -> List[HexCoord]:
        """Identifica celdas candidatas a reina."""
        candidates = []
        
        for coord, cell in self.grid._cells.items():
            if not isinstance(cell, WorkerCell):
                continue
            
            # Criterios de elegibilidad
            if cell.state == CellState.FAILED:
                continue
            
            if cell.load > self.config.degraded_load_threshold:
                continue
            
            # Preferir celdas cercanas al centro
            distance = coord.distance_to(HexCoord.origin())
            if distance <= self.grid.config.radius // 2:
                candidates.append(coord)
        
        return candidates
    
    def _pick_best_candidate(
        self,
        voter: HexCoord,
        candidates: List[HexCoord],
    ) -> Optional[HexCoord]:
        """
        Elige el mejor candidato desde la perspectiva de ``voter``.
        Score = -distancia - (load × 5). Mayor es mejor.
        """
        best_candidate: Optional[HexCoord] = None
        best_score = float("-inf")
        for candidate in candidates:
            candidate_cell = self.grid.get_cell(candidate)
            if not candidate_cell:
                continue
            distance = voter.distance_to(candidate)
            score = -distance - (candidate_cell.load * 5)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate

    def _tally_votes(
        self,
        votes: List[Vote],
        candidates: Set[HexCoord],
        expected_term: int,
    ) -> Optional[HexCoord]:
        """
        Cuenta votos firmados. Rechaza:

        - Votos con firma HMAC inválida o ausente.
        - Votos con ``term`` distinto del esperado (anti-replay de elecciones).
        - Votos para candidatos no registrados.
        - Votos duplicados del mismo ``voter`` (solo el primero cuenta).

        Retorna el candidato ganador si supera mayoría estricta (>50%)
        de los votos válidos. Retorna ``None`` en otro caso.
        """
        tallies: Dict[HexCoord, int] = defaultdict(int)
        voters_counted: Set[HexCoord] = set()
        rejected = {
            "bad_signature": 0,
            "wrong_term": 0,
            "unknown_candidate": 0,
            "duplicate_voter": 0,
        }

        for vote in votes:
            if not vote.verify():
                rejected["bad_signature"] += 1
                logger.warning(
                    "Vote from %s rejected: invalid/missing signature", vote.voter
                )
                continue
            if vote.term != expected_term:
                rejected["wrong_term"] += 1
                logger.warning(
                    "Vote from %s rejected: term %d != expected %d",
                    vote.voter, vote.term, expected_term,
                )
                continue
            if vote.candidate not in candidates:
                rejected["unknown_candidate"] += 1
                logger.warning(
                    "Vote from %s rejected: candidate %s not in candidate set",
                    vote.voter, vote.candidate,
                )
                continue
            if vote.voter in voters_counted:
                rejected["duplicate_voter"] += 1
                logger.warning(
                    "Vote from %s rejected: duplicate in term %d",
                    vote.voter, vote.term,
                )
                continue
            voters_counted.add(vote.voter)
            tallies[vote.candidate] += 1

        total = sum(tallies.values())
        if total == 0:
            logger.error(
                "Election term %d failed: no valid votes. Rejected: %s",
                expected_term, rejected,
            )
            return None

        winner = max(tallies, key=lambda c: tallies[c])
        majority_threshold = total // 2 + 1
        if tallies[winner] < majority_threshold:
            logger.error(
                "Election term %d failed: no quorum "
                "(winner %s had %d/%d, needed %d). Rejected: %s",
                expected_term, winner, tallies[winner], total,
                majority_threshold, rejected,
            )
            return None
        return winner

    def _conduct_election(self, candidates: List[HexCoord]) -> Optional[HexCoord]:
        """
        Conduce la votación entre candidatos. Phase 2: votos firmados con
        HMAC-SHA256 y ``term`` monotónico.
        """
        with self._lock:
            self._term_number += 1
            term = self._term_number

        # Cada celda viva construye y firma un voto.
        votes: List[Vote] = []
        for coord, cell in self.grid._cells.items():
            if cell.state == CellState.FAILED:
                continue
            best = self._pick_best_candidate(coord, candidates)
            if best is None:
                continue
            votes.append(Vote(voter=coord, candidate=best, term=term).sign())

        # Phase 1 fix (B4) reforzado por Phase 2: el tally ahora rechaza
        # votos forjados sin firma válida además de exigir mayoría.
        return self._tally_votes(votes, set(candidates), expected_term=term)
    
    def _promote_to_queen(self, coord: HexCoord) -> Optional[QueenCell]:
        """Promueve una celda a reina."""
        old_cell = self.grid.get_cell(coord)
        
        if not old_cell:
            return None
        
        # Crear nueva reina
        new_queen = QueenCell(coord, self.grid.config)
        
        # Transferir vCores
        for vcore in old_cell._vcores:
            new_queen.add_vcore(vcore)
        
        # Actualizar grid
        self.grid._cells[coord] = new_queen
        
        # Registrar workers
        for c, cell in self.grid._cells.items():
            if isinstance(cell, WorkerCell):
                new_queen.register_worker(cell)
        
        # Actualizar referencia
        old_queen = self.grid._queen
        if old_queen:
            old_queen.state = CellState.FAILED
        
        self.grid._queen = new_queen
        self._last_queen_heartbeat = time.time()
        
        return new_queen
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de sucesión."""
        return {
            "queen_healthy": self.check_queen_health(),
            "election_in_progress": self._election_in_progress,
            "last_heartbeat_ago": time.time() - self._last_queen_heartbeat,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# REDUNDANCIA HEXAGONAL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MirrorState:
    """Estado de un espejo de datos."""
    source: HexCoord
    mirror: HexCoord
    data_hash: str
    last_sync: float = field(default_factory=time.time)
    is_stale: bool = False


class MirrorCell:
    """
    Representa una celda espejo para replicación.
    
    Mantiene una copia de los datos de otra celda
    para tolerancia a fallos.
    """
    
    def __init__(self, source: HexCoord, mirror: HexCoord):
        self.source = source
        self.mirror = mirror
        self._data: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}
        self._last_sync = time.time()
        self._lock = threading.Lock()
    
    def sync_from_source(self, data: Dict[str, Any]) -> bool:
        """Sincroniza datos desde la fuente."""
        with self._lock:
            self._data = data.copy()
            self._last_sync = time.time()
            return True
    
    def get_data(self) -> Dict[str, Any]:
        """Obtiene los datos espejados."""
        with self._lock:
            return self._data.copy()
    
    def is_stale(self, max_age: float = 60.0) -> bool:
        """Verifica si los datos están obsoletos."""
        return (time.time() - self._last_sync) > max_age


class HexRedundancy:
    """
    Gestiona la redundancia hexagonal del panal.
    
    Implementa replicación basada en la topología hexagonal,
    aprovechando los 6 vecinos de cada celda.
    
    Estrategias:
    - MIRROR: Espejo en 1-2 vecinos más cercanos
    - RING: Fragmentos distribuidos en el anillo
    - QUORUM: Escritura a N réplicas, lectura de M
    
    Uso:
        redundancy = HexRedundancy(grid, config)
        redundancy.replicate_cell(coord)
        data = redundancy.read_with_fallback(coord)
    """
    
    class Strategy(Enum):
        MIRROR = auto()
        RING = auto()
        QUORUM = auto()
    
    def __init__(
        self,
        grid: HoneycombGrid,
        config: ResilienceConfig,
        strategy: 'HexRedundancy.Strategy' = None
    ):
        self.grid = grid
        self.config = config
        self.strategy = strategy or self.Strategy.MIRROR
        
        # Mapeo de espejos
        self._mirrors: Dict[HexCoord, List[MirrorCell]] = {}
        self._lock = threading.RLock()
    
    def setup_replication(self, coord: HexCoord) -> List[HexCoord]:
        """
        Configura replicación para una celda.
        
        Returns:
            Lista de coordenadas de réplicas
        """
        cell = self.grid.get_cell(coord)
        if not cell:
            return []
        
        replicas = []
        
        if self.strategy == self.Strategy.MIRROR:
            replicas = self._setup_mirror_replication(coord, cell)
        elif self.strategy == self.Strategy.RING:
            replicas = self._setup_ring_replication(coord, cell)
        elif self.strategy == self.Strategy.QUORUM:
            replicas = self._setup_quorum_replication(coord, cell)
        
        return replicas
    
    def _setup_mirror_replication(
        self,
        coord: HexCoord,
        cell: HoneycombCell
    ) -> List[HexCoord]:
        """Configura replicación en espejo (vecinos cercanos)."""
        mirrors = []
        neighbors = cell.get_all_neighbors()
        
        for neighbor in neighbors[:self.config.replication_factor]:
            if neighbor and isinstance(neighbor, WorkerCell):
                mirror = MirrorCell(coord, neighbor.coord)
                
                with self._lock:
                    if coord not in self._mirrors:
                        self._mirrors[coord] = []
                    self._mirrors[coord].append(mirror)
                
                mirrors.append(neighbor.coord)
        
        return mirrors
    
    def _setup_ring_replication(
        self,
        coord: HexCoord,
        cell: HoneycombCell
    ) -> List[HexCoord]:
        """Configura replicación en anillo."""
        mirrors = []
        ring_coords = coord.ring(1)
        
        # Distribuir en posiciones opuestas del anillo
        step = len(ring_coords) // self.config.replication_factor
        
        for i in range(self.config.replication_factor):
            idx = (i * step) % len(ring_coords)
            mirror_coord = ring_coords[idx]
            mirror_cell = self.grid.get_cell(mirror_coord)
            
            if mirror_cell and isinstance(mirror_cell, WorkerCell):
                mirror = MirrorCell(coord, mirror_coord)
                
                with self._lock:
                    if coord not in self._mirrors:
                        self._mirrors[coord] = []
                    self._mirrors[coord].append(mirror)
                
                mirrors.append(mirror_coord)
        
        return mirrors
    
    def _setup_quorum_replication(
        self,
        coord: HexCoord,
        cell: HoneycombCell
    ) -> List[HexCoord]:
        """Configura replicación con quorum."""
        # Similar a mirror pero con más réplicas
        return self._setup_mirror_replication(coord, cell)
    
    def replicate_data(self, coord: HexCoord, data: Dict[str, Any]) -> int:
        """
        Replica datos a todas las réplicas de una celda.
        
        Returns:
            Número de réplicas actualizadas
        """
        with self._lock:
            mirrors = self._mirrors.get(coord, [])
        
        replicated = 0
        for mirror in mirrors:
            if mirror.sync_from_source(data):
                replicated += 1
        
        return replicated
    
    def read_with_fallback(self, coord: HexCoord) -> Optional[Dict[str, Any]]:
        """
        Lee datos con fallback a réplicas.
        
        Intenta leer de la celda primaria, si falla
        intenta las réplicas.
        """
        cell = self.grid.get_cell(coord)
        
        # Intentar primaria
        if cell and cell.state != CellState.FAILED:
            return {"primary": True, "coord": coord}
        
        # Fallback a réplicas
        with self._lock:
            mirrors = self._mirrors.get(coord, [])
        
        for mirror in mirrors:
            if not mirror.is_stale():
                return mirror.get_data()
        
        return None
    
    def get_replicas(self, coord: HexCoord) -> List[HexCoord]:
        """Obtiene las réplicas de una celda."""
        with self._lock:
            mirrors = self._mirrors.get(coord, [])
            return [m.mirror for m in mirrors]
    
    def verify_consistency(self, coord: HexCoord) -> bool:
        """Verifica consistencia entre primaria y réplicas."""
        with self._lock:
            mirrors = self._mirrors.get(coord, [])
        
        if not mirrors:
            return True
        
        # Comparar hashes de datos
        # En implementación real, calcular hash de datos
        return all(not m.is_stale() for m in mirrors)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de redundancia."""
        with self._lock:
            total_mirrors = sum(len(m) for m in self._mirrors.values())
            stale_mirrors = sum(
                1 for mirrors in self._mirrors.values()
                for m in mirrors if m.is_stale()
            )
        
        return {
            "strategy": self.strategy.name,
            "cells_with_mirrors": len(self._mirrors),
            "total_mirrors": total_mirrors,
            "stale_mirrors": stale_mirrors,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SWARM RECOVERY
# ═══════════════════════════════════════════════════════════════════════════════

class SwarmRecovery:
    """
    Gestiona la recuperación a nivel de enjambre.
    
    Coordina recuperaciones masivas cuando múltiples
    celdas fallan simultáneamente.
    
    Uso:
        recovery = SwarmRecovery(grid, config)
        recovery.assess_damage()
        recovery.execute_recovery_plan()
    """
    
    def __init__(self, grid: HoneycombGrid, config: ResilienceConfig):
        self.grid = grid
        self.config = config
        
        # Estado
        self._damaged_cells: Set[HexCoord] = set()
        self._recovery_queue: Deque[HexCoord] = deque()
        self._in_progress: Set[HexCoord] = set()
        self._lock = threading.RLock()
    
    def assess_damage(self) -> Dict[str, Any]:
        """
        Evalúa el daño actual del panal.
        
        Returns:
            Reporte de evaluación
        """
        report = {
            "total_cells": len(self.grid._cells),
            "failed_cells": 0,
            "degraded_cells": 0,
            "affected_rings": set(),
            "queen_affected": False,
        }
        
        with self._lock:
            self._damaged_cells.clear()
            
            for coord, cell in self.grid._cells.items():
                if cell.state == CellState.FAILED:
                    self._damaged_cells.add(coord)
                    report["failed_cells"] += 1
                    report["affected_rings"].add(
                        coord.distance_to(HexCoord.origin())
                    )
                elif cell.state == CellState.RECOVERING:
                    report["degraded_cells"] += 1
            
            if self.grid.queen and self.grid.queen.state == CellState.FAILED:
                report["queen_affected"] = True
        
        report["affected_rings"] = list(report["affected_rings"])
        report["damage_percentage"] = (
            report["failed_cells"] / report["total_cells"] * 100
        )
        
        return report
    
    def create_recovery_plan(self) -> List[Tuple[HexCoord, RecoveryAction]]:
        """
        Crea un plan de recuperación priorizado.
        
        Returns:
            Lista de (coordenada, acción) ordenada por prioridad
        """
        plan = []
        
        with self._lock:
            # Prioridad 1: Reina
            if self.grid.queen and self.grid.queen.state == CellState.FAILED:
                plan.append((self.grid.queen.coord, RecoveryAction.REBUILD))
            
            # Prioridad 2: Celdas cercanas al centro
            damaged_by_distance = sorted(
                self._damaged_cells,
                key=lambda c: c.distance_to(HexCoord.origin())
            )
            
            for coord in damaged_by_distance:
                if coord == self.grid.queen.coord if self.grid.queen else False:
                    continue
                
                # Determinar acción
                cell = self.grid.get_cell(coord)
                if cell and cell._error_count > self.config.error_threshold * 2:
                    action = RecoveryAction.REBUILD
                else:
                    action = RecoveryAction.RESTART
                
                plan.append((coord, action))
        
        return plan
    
    def execute_recovery_plan(
        self,
        plan: Optional[List[Tuple[HexCoord, RecoveryAction]]] = None
    ) -> Dict[str, int]:
        """
        Ejecuta el plan de recuperación.
        
        Returns:
            Estadísticas de ejecución
        """
        if plan is None:
            plan = self.create_recovery_plan()
        
        stats = {
            "attempted": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
        }
        
        for coord, action in plan:
            # Verificar límite de concurrencia
            if len(self._in_progress) >= self.config.max_concurrent_recoveries:
                stats["skipped"] += 1
                continue
            
            stats["attempted"] += 1
            
            try:
                success = self._execute_single_recovery(coord, action)
                if success:
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                logger.error(f"Recovery error for {coord}: {sanitize_error(e)}")
                stats["failed"] += 1
        
        return stats
    
    def _execute_single_recovery(
        self,
        coord: HexCoord,
        action: RecoveryAction
    ) -> bool:
        """Ejecuta recuperación de una celda."""
        cell = self.grid.get_cell(coord)
        if not cell:
            return False
        
        with self._lock:
            self._in_progress.add(coord)
        
        try:
            if action == RecoveryAction.RESTART:
                return self._restart_cell(coord, cell)
            elif action == RecoveryAction.REBUILD:
                return self._rebuild_cell(coord, cell)
            elif action == RecoveryAction.REPLICATE:
                return self._replicate_from_mirror(coord)
            else:
                return False
        finally:
            with self._lock:
                self._in_progress.discard(coord)
                self._damaged_cells.discard(coord)
    
    def _restart_cell(self, coord: HexCoord, cell: HoneycombCell) -> bool:
        """Reinicia una celda."""
        cell.state = CellState.RECOVERING
        cell._error_count = 0
        
        # Simular reinicio
        time.sleep(0.01)  # En implementación real, proceso async
        
        cell.state = CellState.IDLE
        logger.info(f"Cell {coord} restarted")
        return True
    
    def _rebuild_cell(self, coord: HexCoord, cell: HoneycombCell) -> bool:
        """Reconstruye una celda desde cero."""
        cell.state = CellState.RECOVERING
        
        # Limpiar estado
        cell._vcores.clear()
        cell._error_count = 0
        cell._pheromone_level = 0.0
        cell._metadata.clear()
        
        # Re-inicializar
        cell.state = CellState.EMPTY
        logger.info(f"Cell {coord} rebuilt")
        return True
    
    def _replicate_from_mirror(self, coord: HexCoord) -> bool:
        """Replica datos desde espejo."""
        # En implementación real, obtener datos de HexRedundancy
        cell = self.grid.get_cell(coord)
        if cell:
            cell.state = CellState.IDLE
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de recuperación."""
        return {
            "damaged_cells": len(self._damaged_cells),
            "in_progress": len(self._in_progress),
            "queue_size": len(self._recovery_queue),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# COMB REPAIR
# ═══════════════════════════════════════════════════════════════════════════════

class CombRepair:
    """
    Repara el "panal" (estructura de datos) dañada.
    
    Verifica y repara:
    - Conexiones entre vecinos
    - Consistencia de datos
    - Integridad de referencias
    
    Uso:
        repair = CombRepair(grid)
        issues = repair.scan_for_issues()
        repair.repair_all(issues)
    """
    
    @dataclass
    class RepairIssue:
        """Un problema detectado."""
        coord: HexCoord
        issue_type: str
        severity: int  # 1-10
        details: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, grid: HoneycombGrid):
        self.grid = grid
        self._repairs_performed: List[Dict] = []
    
    def scan_for_issues(self) -> List['CombRepair.RepairIssue']:
        """
        Escanea el grid buscando problemas.
        
        Returns:
            Lista de problemas encontrados
        """
        issues = []
        
        for coord, cell in self.grid._cells.items():
            # Verificar conexiones de vecinos
            neighbor_issues = self._check_neighbor_connections(coord, cell)
            issues.extend(neighbor_issues)
            
            # Verificar estado consistente
            state_issues = self._check_state_consistency(coord, cell)
            issues.extend(state_issues)
            
            # Verificar integridad de datos
            data_issues = self._check_data_integrity(coord, cell)
            issues.extend(data_issues)
        
        # Ordenar por severidad
        issues.sort(key=lambda i: i.severity, reverse=True)
        
        return issues
    
    def _check_neighbor_connections(
        self,
        coord: HexCoord,
        cell: HoneycombCell
    ) -> List['CombRepair.RepairIssue']:
        """Verifica que las conexiones de vecinos sean bidireccionales."""
        issues = []
        
        for direction in HexDirection:
            neighbor = cell.get_neighbor(direction)
            if not neighbor:
                continue
            
            # Verificar conexión bidireccional
            reverse_neighbor = neighbor.get_neighbor(direction.opposite())
            if reverse_neighbor != cell:
                issues.append(self.RepairIssue(
                    coord=coord,
                    issue_type="broken_neighbor_link",
                    severity=7,
                    details={
                        "direction": direction.name,
                        "neighbor": neighbor.coord,
                    }
                ))
        
        return issues
    
    def _check_state_consistency(
        self,
        coord: HexCoord,
        cell: HoneycombCell
    ) -> List['CombRepair.RepairIssue']:
        """Verifica consistencia de estado."""
        issues = []
        
        # Celda activa sin vCores
        if cell.state == CellState.ACTIVE and not cell._vcores:
            issues.append(self.RepairIssue(
                coord=coord,
                issue_type="active_without_vcores",
                severity=5,
                details={"state": cell.state.name}
            ))
        
        # Load inconsistente
        expected_load = len(cell._vcores) / max(1, cell._config.vcores_per_cell)
        if abs(cell._load - expected_load) > 0.1:
            issues.append(self.RepairIssue(
                coord=coord,
                issue_type="inconsistent_load",
                severity=3,
                details={
                    "actual_load": cell._load,
                    "expected_load": expected_load,
                }
            ))
        
        return issues
    
    def _check_data_integrity(
        self,
        coord: HexCoord,
        cell: HoneycombCell
    ) -> List['CombRepair.RepairIssue']:
        """Verifica integridad de datos."""
        issues = []

        # Phase 2: verificamos integridad serializando con ``mscs`` en lugar
        # de ``pickle``. Beneficios:
        # 1. ``mscs`` rechaza tipos imposibles de serializar (lo que queremos
        #    detectar como "corrupto") pero no permite payload RCE si un
        #    atacante lograra plantar un callable en ``_metadata`` —al menos
        #    la verificación no ejecuta código durante la serialización.
        # 2. Se mantiene el contrato original: error → issue de severidad 8.
        from . import security as _security
        try:
            _security.serialize(cell._metadata, sign=False)
        except Exception:
            issues.append(self.RepairIssue(
                coord=coord,
                issue_type="corrupt_metadata",
                severity=8,
                details={}
            ))

        return issues
    
    def repair_issue(self, issue: 'CombRepair.RepairIssue') -> bool:
        """
        Repara un problema específico.
        
        Returns:
            True si la reparación fue exitosa
        """
        cell = self.grid.get_cell(issue.coord)
        if not cell:
            return False
        
        success = False
        
        if issue.issue_type == "broken_neighbor_link":
            success = self._repair_neighbor_link(issue)
        elif issue.issue_type == "active_without_vcores":
            success = self._repair_state_mismatch(issue)
        elif issue.issue_type == "inconsistent_load":
            success = self._repair_load_calculation(issue)
        elif issue.issue_type == "corrupt_metadata":
            success = self._repair_metadata(issue)
        
        if success:
            self._repairs_performed.append({
                "coord": issue.coord,
                "issue_type": issue.issue_type,
                "timestamp": time.time(),
            })
        
        return success
    
    def _repair_neighbor_link(self, issue: 'CombRepair.RepairIssue') -> bool:
        """Repara conexión de vecino rota.

        Phase 1 fix (B8): antes ``HexDirection[issue.details["direction"]]`` y
        ``issue.details["neighbor"]`` propagaban KeyError sin manejo si los
        detalles estaban malformados, crasheando ``repair_issue`` para todos los
        tipos. Ahora un detalle inválido falla limpiamente (returna False con
        log) sin afectar otras reparaciones.
        """
        cell = self.grid.get_cell(issue.coord)

        try:
            direction_name = issue.details["direction"]
            direction = HexDirection[direction_name]
            neighbor_coord = issue.details["neighbor"]
        except (KeyError, TypeError) as e:
            logger.error(
                f"Repair neighbor_link at {issue.coord}: invalid issue details "
                f"({type(e).__name__}: {e})"
            )
            return False

        neighbor = self.grid.get_cell(neighbor_coord)

        if cell and neighbor:
            cell.set_neighbor(direction, neighbor)
            neighbor.set_neighbor(direction.opposite(), cell)
            return True

        return False
    
    def _repair_state_mismatch(self, issue: 'CombRepair.RepairIssue') -> bool:
        """Repara estado inconsistente."""
        cell = self.grid.get_cell(issue.coord)
        if cell:
            cell.state = CellState.IDLE
            return True
        return False
    
    def _repair_load_calculation(self, issue: 'CombRepair.RepairIssue') -> bool:
        """Recalcula carga."""
        cell = self.grid.get_cell(issue.coord)
        if cell:
            cell._update_load()
            return True
        return False
    
    def _repair_metadata(self, issue: 'CombRepair.RepairIssue') -> bool:
        """Repara metadata corrupta."""
        cell = self.grid.get_cell(issue.coord)
        if cell:
            cell._metadata = {}
            return True
        return False
    
    def repair_all(
        self,
        issues: Optional[List['CombRepair.RepairIssue']] = None
    ) -> Dict[str, int]:
        """
        Repara todos los problemas.
        
        Returns:
            Estadísticas de reparación
        """
        if issues is None:
            issues = self.scan_for_issues()
        
        stats = {"attempted": 0, "successful": 0, "failed": 0}
        
        for issue in issues:
            stats["attempted"] += 1
            if self.repair_issue(issue):
                stats["successful"] += 1
            else:
                stats["failed"] += 1
        
        return stats
    
    def get_repair_history(self) -> List[Dict]:
        """Retorna historial de reparaciones."""
        return self._repairs_performed.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# SISTEMA UNIFICADO
# ═══════════════════════════════════════════════════════════════════════════════

class HiveResilience:
    """
    Sistema de Resiliencia Unificado del Panal.
    
    Coordina todos los subsistemas de tolerancia a fallos:
    - Failover de celdas
    - Sucesión de reina
    - Redundancia hexagonal
    - Recuperación de enjambre
    - Reparación de estructura
    
    Uso:
        resilience = HiveResilience(grid)
        
        # Monitoreo automático
        resilience.tick()
        
        # Manejo de fallo
        resilience.handle_cell_failure(coord)
        
        # Recuperación masiva
        resilience.initiate_swarm_recovery()
    """
    
    def __init__(
        self,
        grid: HoneycombGrid,
        config: Optional[ResilienceConfig] = None
    ):
        self.grid = grid
        self.config = config or ResilienceConfig()
        
        # Subsistemas
        self._failover = CellFailover(grid, self.config)
        self._succession = QueenSuccession(grid, self.config)
        self._redundancy = HexRedundancy(grid, self.config)
        self._recovery = SwarmRecovery(grid, self.config)
        self._repair = CombRepair(grid)
        
        # Estado
        self._tick_count = 0
        self._health_reports: Dict[HexCoord, HealthReport] = {}
        self._lock = threading.RLock()
    
    def tick(self) -> Dict[str, Any]:
        """
        Ejecuta un tick del sistema de resiliencia.
        
        Returns:
            Resumen del tick
        """
        self._tick_count += 1
        results = {"tick": self._tick_count}
        
        # Health check periódico
        if self._tick_count % self.config.health_check_interval_ticks == 0:
            results["health_check"] = self._perform_health_check()
        
        # Verificar reina
        if self._tick_count % self.config.queen_heartbeat_interval == 0:
            if not self._succession.check_queen_health():
                logger.warning("Queen health check failed")
                results["queen_issue"] = True
        
        # Actualizar cooldowns
        self._failover.tick()
        
        return results
    
    def _perform_health_check(self) -> Dict[str, Any]:
        """Ejecuta health check en todas las celdas."""
        check_results = {
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "failed": 0,
        }
        
        for coord, cell in self.grid._cells.items():
            report = self._check_cell_health(coord, cell)
            self._health_reports[coord] = report
            
            if report.status == HealthStatus.HEALTHY:
                check_results["healthy"] += 1
            elif report.status == HealthStatus.DEGRADED:
                check_results["degraded"] += 1
            elif report.status == HealthStatus.UNHEALTHY:
                check_results["unhealthy"] += 1
                
                # Auto-recuperación
                if self.config.auto_recovery:
                    self.handle_cell_failure(
                        coord,
                        report.failure_type or FailureType.ERROR_THRESHOLD
                    )
            elif report.status == HealthStatus.FAILED:
                check_results["failed"] += 1
        
        return check_results
    
    def _check_cell_health(
        self,
        coord: HexCoord,
        cell: HoneycombCell
    ) -> HealthReport:
        """Verifica la salud de una celda."""
        report = HealthReport(
            coord=coord,
            status=HealthStatus.HEALTHY,
            load=cell.load,
        )
        
        # Verificar estado
        if cell.state == CellState.FAILED:
            report.status = HealthStatus.FAILED
            report.failure_type = FailureType.ERROR_THRESHOLD
            return report
        
        # Verificar errores
        if cell._error_count >= self.config.error_threshold:
            report.status = HealthStatus.UNHEALTHY
            report.failure_type = FailureType.ERROR_THRESHOLD
            report.error_count = cell._error_count
            return report
        
        # Verificar carga
        if cell.load >= self.config.degraded_load_threshold:
            report.status = HealthStatus.DEGRADED
            report.failure_type = FailureType.OVERLOAD
        
        return report
    
    def handle_cell_failure(
        self,
        coord: HexCoord,
        failure_type: FailureType
    ) -> FailoverEvent:
        """Maneja un fallo de celda."""
        return self._failover.handle_failure(coord, failure_type)
    
    def initiate_queen_succession(self) -> Optional[QueenCell]:
        """Inicia sucesión de reina."""
        return self._succession.elect_new_queen()
    
    def setup_replication(self, coord: HexCoord) -> List[HexCoord]:
        """Configura replicación para una celda."""
        return self._redundancy.setup_replication(coord)
    
    def initiate_swarm_recovery(self) -> Dict[str, Any]:
        """Inicia recuperación a nivel de enjambre."""
        assessment = self._recovery.assess_damage()
        
        if assessment["damage_percentage"] > 50:
            logger.critical("Severe damage detected, initiating full recovery")
        
        plan = self._recovery.create_recovery_plan()
        stats = self._recovery.execute_recovery_plan(plan)
        
        return {
            "assessment": assessment,
            "recovery_stats": stats,
        }
    
    def repair_structure(self) -> Dict[str, int]:
        """Repara la estructura del panal."""
        return self._repair.repair_all()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de salud del sistema."""
        status_counts = {s: 0 for s in HealthStatus}
        
        for report in self._health_reports.values():
            status_counts[report.status] += 1
        
        return {
            "total_cells": len(self.grid._cells),
            "by_status": {s.name: c for s, c in status_counts.items()},
            "queen_healthy": self._succession.check_queen_health(),
            "failed_cells": list(self._failover.get_failed_cells()),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del sistema."""
        return {
            "tick_count": self._tick_count,
            "failover": self._failover.get_stats(),
            "succession": self._succession.get_stats(),
            "redundancy": self._redundancy.get_stats(),
            "recovery": self._recovery.get_stats(),
            "health_summary": self.get_health_summary(),
        }
    
    @property
    def failover(self) -> CellFailover:
        return self._failover
    
    @property
    def succession(self) -> QueenSuccession:
        return self._succession
    
    @property
    def redundancy(self) -> HexRedundancy:
        return self._redundancy
    
    @property
    def recovery(self) -> SwarmRecovery:
        return self._recovery
    
    @property
    def repair(self) -> CombRepair:
        return self._repair
