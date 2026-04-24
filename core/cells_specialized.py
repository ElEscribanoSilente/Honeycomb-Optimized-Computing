"""
HOC Core · Specialized cells (private)
======================================

Celdas especializadas por rol: ``QueenCell``, ``WorkerCell``, ``DroneCell``,
``NurseryCell``, ``StorageCell``, ``GuardCell`` y ``ScoutCell``.

Cada una implementa la semántica específica de su rol:
- Queen: coordinación del cluster, spawn queue, health score.
- Worker: trabajo computacional y work-stealing entre vecinas.
- Drone: comunicación externa y broadcasting a endpoints.
- Nursery: incubación de nuevas entidades.
- Storage: almacenamiento LRU con TTL.
- Guard: validación de datos y reglas de seguridad.
- Scout: exploración del grid y descubrimiento de zonas.

Extraído de ``core.py`` en Fase 3.3.
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import Any

import numpy as np

from ._queen import QueenCell
from .cells_base import CellRole, CellState, HoneycombCell
from .events import Event, EventType
from .grid_config import HoneycombConfig
from .grid_geometry import HexCoord
from .pheromone import PheromoneType

logger = logging.getLogger(__name__)

__all__ = [
    "QueenCell",
    "WorkerCell",
    "DroneCell",
    "NurseryCell",
    "StorageCell",
    "GuardCell",
    "ScoutCell",
]


# ═══════════════════════════════════════════════════════════════════════════════
# CELDAS ESPECIALIZADAS
# ═══════════════════════════════════════════════════════════════════════════════

# NOTE: ``QueenCell`` vive en :mod:`hoc.core._queen` pero se re-exporta desde
# este módulo. Se separó durante Fase 3.3 para que cells_specialized.py cupiera
# bajo el límite DoD de 800 LOC por archivo.


class WorkerCell(HoneycombCell):
    """
    Celda Trabajadora - Unidad de cómputo principal (v3.0).

    v3.0: steal_from con _update_load dentro del lock scope.
    """

    __slots__ = ("_processed_ticks", "_steal_count", "_stolen_from_count", "_work_history")

    def __init__(self, coord: HexCoord, config: HoneycombConfig | None = None):
        super().__init__(coord, CellRole.WORKER, config)
        self._processed_ticks: int = 0
        self._steal_count: int = 0
        self._stolen_from_count: int = 0
        self._work_history: deque = deque(maxlen=100)

    def can_steal_work(self) -> bool:
        return self._load < self._config.steal_threshold_low and self._state in (
            CellState.IDLE,
            CellState.EMPTY,
        )

    def should_donate_work(self) -> bool:
        return self._load > self._config.steal_threshold_high

    def steal_from(self, source: WorkerCell, count: int = 1) -> int:
        """
        Roba trabajo de otra celda.

        v3.0 FIX: _update_load() se llama DENTRO del scope del lock
        para evitar race conditions.
        """
        # Ordenar locks por coordenada para evitar deadlock
        cells = sorted([self, source], key=lambda c: (c.coord.q, c.coord.r))

        stolen = 0

        with cells[0]._rw_lock.write_lock(), cells[1]._rw_lock.write_lock():
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
            self._event_bus.publish(
                Event(
                    type=EventType.WORK_STOLEN,
                    source=self,
                    data={"from": source.coord.to_dict(), "count": stolen},
                )
            )

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
        self._work_history.append(
            {"id": work_id, "duration": duration, "success": success, "timestamp": time.time()}
        )

    def get_performance_stats(self) -> dict[str, float]:
        if not self._work_history:
            return {"avg_duration": 0, "success_rate": 0, "throughput": 0}

        history = list(self._work_history)
        durations = [w["duration"] for w in history]
        successes = sum(1 for w in history if w["success"])

        time_span = history[-1]["timestamp"] - history[0]["timestamp"] if len(history) > 1 else 1

        return {
            "avg_duration": float(np.mean(durations)),
            "success_rate": successes / len(history),
            "throughput": len(history) / max(time_span, 1),
            "steal_count": self._steal_count,
            "stolen_from_count": self._stolen_from_count,
        }


class DroneCell(HoneycombCell):
    """
    Celda Dron - Comunicación externa (v3.0).

    v3.0 FIX: message_queue usa (priority, msg_id, msg) para evitar
    comparación de dicts en heapq.
    """

    __slots__ = (
        "_connection_errors",
        "_external_connections",
        "_message_queue",
        "_messages_received",
        "_messages_sent",
    )

    def __init__(self, coord: HexCoord, config: HoneycombConfig | None = None):
        super().__init__(coord, CellRole.DRONE, config)
        self._external_connections: list[Any] = []
        self._message_queue: list[tuple[int, str, dict]] = []  # v3.0 FIX
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

    def queue_message(self, message: dict, priority: int = 0) -> None:
        """v3.0 FIX: Usa msg_id para hacer tupla comparable."""
        msg_id = uuid.uuid4().hex[:8]
        heapq.heappush(self._message_queue, (-priority, msg_id, message))

    def broadcast(self, message: dict) -> int:
        sent = 0

        with self._rw_lock.read_lock():
            endpoints = list(self._external_connections)

        for endpoint in endpoints:
            try:
                if hasattr(endpoint, "receive"):
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

    def get_comm_stats(self) -> dict[str, Any]:
        return {
            "connections": len(self._external_connections),
            "queue_size": len(self._message_queue),
            "sent": self._messages_sent,
            "received": self._messages_received,
            "errors": self._connection_errors,
        }


class NurseryCell(HoneycombCell):
    """Celda Guardería - Spawning de nuevas entidades (v3.0)."""

    __slots__ = ("_incubating", "_max_incubating", "_ready_entities", "_total_spawned")

    def __init__(self, coord: HexCoord, config: HoneycombConfig | None = None):
        super().__init__(coord, CellRole.NURSERY, config)
        self._incubating: list[dict] = []
        self._ready_entities: list[Any] = []
        self._total_spawned: int = 0
        self._max_incubating = 10

    def incubate(self, entity_spec: dict, priority: int = 0) -> str | None:
        with self._rw_lock.write_lock():
            if len(self._incubating) >= self._max_incubating:
                return None

            entity_id = f"entity_{uuid.uuid4().hex[:8]}"
            self._incubating.append(
                {
                    "id": entity_id,
                    "spec": entity_spec,
                    "progress": 0.0,
                    "priority": priority,
                    "started_at": time.time(),
                }
            )

            self._incubating.sort(key=lambda x: -x["priority"])

            self._event_bus.publish(
                Event(type=EventType.ENTITY_INCUBATING, source=self, data={"entity_id": entity_id})
            )

            return entity_id

    def tick_incubation(self, rate: float | None = None) -> list[Any]:
        rate = rate if rate is not None else self._config.nursery_default_incubation_rate
        ready = []
        still_incubating = []

        with self._rw_lock.write_lock():
            for item in self._incubating:
                progress_increment = rate * (1 + item["progress"])
                item["progress"] = min(1.0, item["progress"] + progress_increment)

                if item["progress"] >= 1.0:
                    entity = self._create_entity(item["spec"])
                    entity["id"] = item["id"]
                    entity["incubation_time"] = time.time() - item["started_at"]
                    ready.append(entity)
                    self._total_spawned += 1

                    self._event_bus.publish(
                        Event(
                            type=EventType.ENTITY_SPAWNED,
                            source=self,
                            data={"entity_id": item["id"]},
                        )
                    )
                else:
                    still_incubating.append(item)

            self._incubating = still_incubating
            self._ready_entities.extend(ready)

        return ready

    def _create_entity(self, spec: dict) -> dict:
        return {
            "type": spec.get("type", "unknown"),
            "created": True,
            "spec": spec,
            "born_at": time.time(),
        }

    def harvest_ready(self, count: int | None = None) -> list[Any]:
        with self._rw_lock.write_lock():
            if count is None:
                ready = self._ready_entities
                self._ready_entities = []
            else:
                ready = self._ready_entities[:count]
                self._ready_entities = self._ready_entities[count:]
            return ready

    def get_incubation_status(self) -> list[dict]:
        with self._rw_lock.read_lock():
            return [
                {
                    "id": item["id"],
                    "progress": item["progress"],
                    "priority": item["priority"],
                    "elapsed": time.time() - item["started_at"],
                }
                for item in self._incubating
            ]

    def get_nursery_stats(self) -> dict[str, Any]:
        return {
            "incubating": len(self._incubating),
            "ready": len(self._ready_entities),
            "total_spawned": self._total_spawned,
            "capacity": self._max_incubating,
        }


class StorageCell(HoneycombCell):
    """
    Celda de Almacenamiento (v3.0).

    v3.0: LRU eviction cuando se llega a capacidad máxima.
    """

    __slots__ = ("_access_order", "_current_size", "_max_size", "_storage", "_storage_lock")

    def __init__(self, coord: HexCoord, config: HoneycombConfig | None = None):
        super().__init__(coord, CellRole.STORAGE, config)
        self._storage: dict[str, tuple[Any, float, float | None]] = {}
        self._storage_lock = threading.Lock()
        self._max_size = 1000
        self._current_size = 0
        self._access_order: OrderedDict = OrderedDict()  # v3.0: LRU tracking

    def store(self, key: str, value: Any, ttl: float | None = None) -> bool:
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

    def retrieve(self, key: str) -> Any | None:
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

    def get_storage_stats(self) -> dict[str, Any]:
        return {
            "keys": self._current_size,
            "capacity": self._max_size,
            "utilization": self._current_size / self._max_size if self._max_size > 0 else 0,
        }


class GuardCell(HoneycombCell):
    """
    v3.0 NUEVO: Celda Guardia - Seguridad y validación.

    Valida datos que entran/salen de su zona, detecta anomalías
    y emite alertas. Protege un perímetro de celdas vecinas.
    """

    __slots__ = ("_blocked_sources", "_rules", "_total_blocks", "_total_checks", "_violations")

    def __init__(self, coord: HexCoord, config: HoneycombConfig | None = None):
        super().__init__(coord, CellRole.GUARD, config)
        self._rules: list[Callable[[dict], bool]] = []
        self._violations: deque = deque(maxlen=500)
        self._blocked_sources: set[HexCoord] = set()
        self._total_checks: int = 0
        self._total_blocks: int = 0

    def add_rule(self, rule: Callable[[dict], bool]) -> int:
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

    def validate(self, data: dict, source: HexCoord | None = None) -> bool:
        """Valida datos contra todas las reglas."""
        self._total_checks += 1

        if source and source in self._blocked_sources:
            self._total_blocks += 1
            return False

        for i, rule in enumerate(self._rules):
            try:
                if not rule(data):
                    self._violations.append(
                        {
                            "rule_index": i,
                            "source": source.to_dict() if source else None,
                            "timestamp": time.time(),
                            "data_keys": list(data.keys()),
                        }
                    )

                    self._event_bus.publish(
                        Event(
                            type=EventType.GUARD_VALIDATION_FAILED,
                            source=self,
                            data={
                                "rule_index": i,
                                "source_coord": source.to_dict() if source else None,
                            },
                        )
                    )

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

    def get_guard_stats(self) -> dict[str, Any]:
        return {
            "rules": len(self._rules),
            "total_checks": self._total_checks,
            "total_blocks": self._total_blocks,
            "blocked_sources": len(self._blocked_sources),
            "recent_violations": len(self._violations),
            "block_rate": (self._total_blocks / max(1, self._total_checks)),
        }


class ScoutCell(HoneycombCell):
    """
    v3.0 NUEVO: Celda Exploradora - Búsqueda de recursos y exploración.

    Explora el grid buscando áreas con recursos o baja carga,
    y marca caminos con feromonas. Implementa un random walk
    dirigido por feromonas con memoria de posiciones visitadas.
    """

    __slots__ = (
        "_current_target",
        "_discoveries",
        "_exploration_history",
        "_max_exploration_range",
        "_visit_memory",
    )

    def __init__(self, coord: HexCoord, config: HoneycombConfig | None = None):
        super().__init__(coord, CellRole.SCOUT, config)
        self._exploration_history: deque = deque(maxlen=200)
        self._discoveries: list[dict] = []
        self._current_target: HexCoord | None = None
        self._max_exploration_range = config.radius if config else 10
        self._visit_memory: set[HexCoord] = set()

    def explore_step(self) -> dict | None:
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
            scored.append(
                (
                    load_score
                    + explore_pheromone * self._config.scout_explore_pheromone_weight
                    + novelty,
                    n,
                )
            )

        scored.sort(key=lambda x: -x[0])
        best = scored[0][1]

        self._visit_memory.add(best.coord)
        self._exploration_history.append(
            {
                "coord": best.coord.to_dict(),
                "timestamp": time.time(),
                "load": best.load,
            }
        )

        # Depositar feromona de camino
        self.deposit_pheromone(
            PheromoneType.PATH, self._config.scout_path_deposit_intensity, source=self.coord
        )

        # Detectar descubrimientos
        discovery = None
        if best.load < self._config.scout_low_load_threshold and best.is_available:
            discovery = {
                "type": "low_load_area",
                "coord": best.coord.to_dict(),
                "load": best.load,
                "timestamp": time.time(),
            }
            self._discoveries.append(discovery)

            self._event_bus.publish(
                Event(type=EventType.SCOUT_DISCOVERY, source=self, data=discovery)
            )

        return discovery

    def set_target(self, target: HexCoord) -> None:
        self._current_target = target

    def get_scout_stats(self) -> dict[str, Any]:
        return {
            "explored_cells": len(self._visit_memory),
            "discoveries": len(self._discoveries),
            "history_length": len(self._exploration_history),
            "current_target": self._current_target.to_dict() if self._current_target else None,
            "max_range": self._max_exploration_range,
        }
