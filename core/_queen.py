"""
HOC Core · QueenCell (private)
==============================

``QueenCell`` — celda coordinadora del cluster. Extraída de
:mod:`hoc.core.cells_specialized` durante Fase 3.3 para respetar el
límite DoD de 800 LOC por archivo.

La ``QueenCell`` mantiene un registro debilmente referenciado de
``WorkerCell`` activas, gestiona una cola de spawn con prioridades,
propone rebalances de carga y reporta un health score agregado del
cluster.

No exportado directamente desde ``hoc.core``; se re-exporta desde
:mod:`hoc.core.cells`.
"""

from __future__ import annotations

import heapq
import logging
import time
import uuid
import weakref
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from .cells_base import CellRole, CellState, HoneycombCell
from .grid_config import HoneycombConfig
from .grid_geometry import HexCoord
from .health import CircuitState

if TYPE_CHECKING:
    from .cells_specialized import WorkerCell

logger = logging.getLogger(__name__)

__all__ = ["QueenCell"]


class QueenCell(HoneycombCell):
    """
    Celda Reina - Coordinadora del cluster (v3.0).

    Mejoras v3.0:
    - Adaptive rebalance threshold basado en historial
    - Health score de cluster
    """

    __slots__ = (
        "_global_load",
        "_load_history",
        "_rebalance_threshold",
        "_spawn_queue",
        "_succession_candidates",
        "_worker_registry",
    )

    def __init__(self, coord: HexCoord, config: HoneycombConfig | None = None):
        super().__init__(coord, CellRole.QUEEN, config)
        self._worker_registry: dict[HexCoord, weakref.ref] = {}
        self._global_load: float = 0.0
        self._spawn_queue: list[tuple[int, str, dict]] = []  # v3.0 FIX: (priority, id, spec)
        self._succession_candidates: list[weakref.ref] = []
        self._load_history: deque = deque(maxlen=100)
        self._rebalance_threshold = 0.2

    def register_worker(self, cell: WorkerCell) -> None:
        with self._rw_lock.write_lock():
            self._worker_registry[cell.coord] = weakref.ref(cell)

    def unregister_worker(self, coord: HexCoord) -> None:
        with self._rw_lock.write_lock():
            self._worker_registry.pop(coord, None)

    def _get_active_workers(self) -> list[WorkerCell]:
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

    def get_load_statistics(self) -> dict[str, float]:
        workers = self._get_active_workers()
        if not workers:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "p50": 0, "p90": 0, "p99": 0}

        loads = np.array([w.load for w in workers])
        return {
            "mean": float(np.mean(loads)),
            "std": float(np.std(loads)),
            "min": float(np.min(loads)),
            "max": float(np.max(loads)),
            "p50": float(np.percentile(loads, 50)),
            "p90": float(np.percentile(loads, 90)),
            "p99": float(np.percentile(loads, 99)),
        }

    def find_cells_by_load(
        self, min_load: float = 0.0, max_load: float = 1.0, limit: int = 10
    ) -> list[WorkerCell]:
        workers = self._get_active_workers()
        filtered = [w for w in workers if min_load <= w.load <= max_load]
        filtered.sort(key=lambda w: w.load)
        return filtered[:limit]

    def find_least_loaded_cells(self, count: int = 3) -> list[WorkerCell]:
        return self.find_cells_by_load(max_load=1.0, limit=count)

    def find_most_loaded_cells(self, count: int = 3) -> list[WorkerCell]:
        workers = self._get_active_workers()
        workers.sort(key=lambda w: -w.load)
        return workers[:count]

    def should_rebalance(self) -> bool:
        stats = self.get_load_statistics()
        return stats["std"] > self._rebalance_threshold

    def plan_rebalance(self) -> list[tuple[WorkerCell, WorkerCell, int]]:
        if not self.should_rebalance():
            return []

        moves = []
        overloaded = self.find_cells_by_load(min_load=self._config.steal_threshold_high)
        underloaded = self.find_cells_by_load(max_load=self._config.steal_threshold_low)

        for source in overloaded:
            for target in underloaded:
                if target.load < source.load - 0.2:
                    moves.append((source, target, 1))
                    if len(moves) >= 10:
                        return moves

        return moves

    def issue_royal_command(self, command: str, params: dict) -> int:
        workers = self._get_active_workers()
        for worker in workers:
            worker._metadata["royal_command"] = {
                "command": command,
                "params": params,
                "from_queen": self.coord.to_dict(),
                "timestamp": time.time(),
            }
        return len(workers)

    def schedule_spawn(self, entity_type: str, params: dict, priority: int = 0) -> str:
        """
        v3.0 FIX: Usa (priority, spawn_id, spec) donde spawn_id es str
        comparable, evitando comparación de dicts en heapq.
        """
        spawn_id = f"spawn_{uuid.uuid4().hex[:8]}"
        spec = {"id": spawn_id, "type": entity_type, "params": params, "scheduled_at": time.time()}
        heapq.heappush(self._spawn_queue, (-priority, spawn_id, spec))
        return spawn_id

    def get_next_spawn(self) -> dict | None:
        if self._spawn_queue:
            _, _, spec = heapq.heappop(self._spawn_queue)
            return spec
        return None

    def add_succession_candidate(self, cell: QueenCell) -> None:
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
        circuit_open = sum(1 for w in workers if w.circuit_breaker.state != CircuitState.CLOSED)

        load_score = 1.0 - float(np.mean(loads))
        health_ratio = 1.0 - (failed + circuit_open) / len(workers)
        balance_score = 1.0 - min(1.0, float(np.std(loads)) * 2)

        cfg = self._config
        return (
            load_score * cfg.cluster_health_load_weight
            + health_ratio * cfg.cluster_health_health_weight
            + balance_score * cfg.cluster_health_balance_weight
        )

    def get_cluster_metrics(self) -> dict[str, Any]:
        workers = self._get_active_workers()

        return {
            "queen_coord": self.coord.to_dict(),
            "worker_count": len(workers),
            "global_load": self._global_load,
            "load_stats": self.get_load_statistics(),
            "spawn_queue_size": len(self._spawn_queue),
            "succession_candidates": len(self._succession_candidates),
            "load_trend": list(self._load_history)[-10:] if self._load_history else [],
            "health_score": self.get_cluster_health_score(),
        }
