"""
HOC Bridge — high-level CAMV / Vent adapters built on top of mappers.

Phase 6 split from the legacy ``bridge.py`` monolith. Hosts the
``CAMVHoneycombBridge`` (full integration façade) and the
``VentHoneycombAdapter`` (entity placement on hex topology).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from ..core import HexCoord, HoneycombGrid, WorkerCell
from ..security import sanitize_error
from .converters import CartesianToHex, HexToCartesian
from .mappers import GridToHypervisorMapper, HypervisorProtocol

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# BRIDGE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BridgeConfig:
    """Configuración del bridge HOC-CAMV."""

    # Mapeo
    auto_initialize: bool = True
    vcores_per_worker: int = 8

    # Coordinación
    hex_size: float = 1.0
    layout: HexToCartesian.Layout = field(default_factory=lambda: HexToCartesian.Layout.FLAT_TOP)

    # Sincronización
    sync_interval_ticks: int = 10
    health_check_interval: int = 5


class CAMVHoneycombBridge:
    """
    Bridge principal entre HOC y CAMV.

    Integra todos los componentes de mapeo y conversión,
    proporcionando una interfaz unificada para la comunicación
    entre el sistema de panal y el hipervisor.

    Uso:
        bridge = CAMVHoneycombBridge(grid, hypervisor)
        bridge.initialize()

        # Ejecutar trabajo
        result = bridge.execute_on_cell(coord, payload)

        # Migrar carga
        bridge.migrate_vcores(source, target, count=3)
    """

    def __init__(
        self,
        grid: HoneycombGrid,
        hypervisor: HypervisorProtocol | None = None,
        config: BridgeConfig | None = None,
    ):
        self.grid = grid
        self.hypervisor: HypervisorProtocol | None = hypervisor
        self.config = config or BridgeConfig()

        # Componentes
        self._grid_mapper = GridToHypervisorMapper(grid, hypervisor)
        self._hex_to_cart = HexToCartesian(size=self.config.hex_size, layout=self.config.layout)
        self._cart_to_hex = CartesianToHex(size=self.config.hex_size, layout=self.config.layout)

        # Estado
        self._lock = threading.RLock()
        self._tick_count = 0
        self._initialized = False

        # Estadísticas
        self._executions = 0
        self._migrations = 0
        self._errors = 0

    def initialize(self) -> bool:
        """
        Inicializa el bridge.

        Returns:
            True si la inicialización fue exitosa
        """
        with self._lock:
            if self._initialized:
                return True

            success = self._grid_mapper.initialize_mapping()
            self._initialized = success

            if success:
                logger.info("CAMVHoneycombBridge initialized successfully")
            else:
                logger.error("CAMVHoneycombBridge initialization failed")

            return success

    def execute_on_cell(
        self, coord: HexCoord, payload: dict[str, Any], vcore_index: int = 0
    ) -> Any | None:
        """
        Ejecuta un payload en un vCore de una celda.

        Args:
            coord: Coordenada de la celda
            payload: Datos a ejecutar
            vcore_index: Índice del vCore (0 = primero disponible)

        Returns:
            Resultado de la ejecución o None si falló
        """
        if not self._initialized:
            logger.error("Bridge not initialized")
            return None

        vcores = self._grid_mapper.get_vcores_for_cell(coord)

        if not vcores or vcore_index >= len(vcores):
            logger.error(f"No vCore at index {vcore_index} for cell {coord}")
            self._errors += 1
            return None

        try:
            vcore = vcores[vcore_index]
            result = vcore.execute(payload)
            self._executions += 1
            return result
        except Exception as e:
            logger.error(f"Execution error on cell {coord}: {sanitize_error(e)}")
            self._errors += 1
            return None

    def broadcast_to_ring(
        self, center: HexCoord, radius: int, payload: dict[str, Any]
    ) -> dict[HexCoord, Any]:
        """
        Ejecuta payload en todas las celdas de un anillo.

        Returns:
            Diccionario de resultados por coordenada
        """
        results = {}

        for coord in center.ring(radius):
            cell = self.grid.get_cell(coord)
            if cell and isinstance(cell, WorkerCell):
                result = self.execute_on_cell(coord, payload)
                results[coord] = result

        return results

    def migrate_vcores(self, source: HexCoord, target: HexCoord, count: int = 1) -> int:
        """
        Migra vCores de una celda a otra.

        Args:
            source: Celda origen
            target: Celda destino
            count: Número de vCores a migrar

        Returns:
            Número de vCores migrados
        """
        vcore_ids = self._grid_mapper._cell_mapper.get_vcore_ids(source)
        migrated = 0

        for _i, vcore_id in enumerate(vcore_ids[:count]):
            if self._grid_mapper.migrate_vcore(vcore_id, target):
                migrated += 1
                self._migrations += 1

        return migrated

    def hex_to_cartesian(self, coord: HexCoord) -> tuple[float, float]:
        """Convierte coordenada hexagonal a cartesiana."""
        return self._hex_to_cart.convert(coord)

    def cartesian_to_hex(self, x: float, y: float) -> HexCoord:
        """Convierte coordenada cartesiana a hexagonal."""
        return self._cart_to_hex.convert(x, y)

    def tick(self) -> dict[str, Any]:
        """
        Ejecuta un tick del bridge.

        Returns:
            Estadísticas del tick
        """
        self._tick_count += 1

        results = {
            "tick": self._tick_count,
            "synced": False,
            "health_checked": False,
        }

        # Sincronización periódica
        if self._tick_count % self.config.sync_interval_ticks == 0:
            self._sync_with_hypervisor()
            results["synced"] = True

        # Health check
        if self._tick_count % self.config.health_check_interval == 0:
            self._health_check()
            results["health_checked"] = True

        return results

    def _sync_with_hypervisor(self) -> None:
        """Sincroniza estado con el hypervisor."""
        if self.hypervisor is None:
            return

        # En implementación real, sincronizar métricas y estado
        logger.debug("Bridge sync with hypervisor")

    def _health_check(self) -> None:
        """Verifica salud de los vCores."""
        # En implementación real, verificar estado de cada vCore
        logger.debug("Bridge health check")

    def get_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas del bridge."""
        return {
            "initialized": self._initialized,
            "tick_count": self._tick_count,
            "executions": self._executions,
            "migrations": self._migrations,
            "errors": self._errors,
            "grid_mapper": self._grid_mapper.get_mapping_stats(),
        }


class VentHoneycombAdapter:
    """
    Adaptador para integrar entidades de Vent con HOC.

    Traduce operaciones de Vent (brains, entities) a
    operaciones del panal hexagonal.

    Uso:
        adapter = VentHoneycombAdapter(grid, bridge)
        cell = adapter.assign_entity(entity)
        adapter.execute_brain(entity_id, brain_state)
    """

    def __init__(self, grid: HoneycombGrid, bridge: CAMVHoneycombBridge):
        self.grid = grid
        self.bridge = bridge

        # Mapeo entity → cell
        self._entity_cells: dict[str, HexCoord] = {}
        self._lock = threading.RLock()

    def assign_entity(
        self, entity_id: str, preferred_coord: HexCoord | None = None
    ) -> HexCoord | None:
        """
        Asigna una entidad a una celda del panal.

        Args:
            entity_id: ID de la entidad
            preferred_coord: Coordenada preferida (opcional)

        Returns:
            Coordenada asignada o None si falló
        """
        with self._lock:
            if entity_id in self._entity_cells:
                return self._entity_cells[entity_id]

            # Encontrar celda disponible
            if preferred_coord and self.grid.get_cell(preferred_coord):
                coord = preferred_coord
            else:
                available = self.grid.find_available_cells(1)
                if not available:
                    return None
                coord = available[0].coord

            self._entity_cells[entity_id] = coord
            return coord

    def get_entity_cell(self, entity_id: str) -> HexCoord | None:
        """Obtiene la celda de una entidad."""
        return self._entity_cells.get(entity_id)

    def execute_brain(self, entity_id: str, brain_state: dict[str, Any]) -> Any | None:
        """
        Ejecuta el brain de una entidad en su celda asignada.

        Returns:
            Resultado de la ejecución
        """
        coord = self._entity_cells.get(entity_id)
        if not coord:
            logger.error(f"Entity {entity_id} not assigned to any cell")
            return None

        payload = {
            "entity_id": entity_id,
            "brain_state": brain_state,
            "operation": "execute_brain",
        }

        return self.bridge.execute_on_cell(coord, payload)

    def migrate_entity(self, entity_id: str, target_coord: HexCoord) -> bool:
        """Migra una entidad a una nueva celda."""
        with self._lock:
            if entity_id not in self._entity_cells:
                return False

            old_coord = self._entity_cells[entity_id]

            # Migrar vCore asociado
            vcore_ids = self.bridge._grid_mapper._cell_mapper.get_vcore_ids(old_coord)
            if vcore_ids and not self.bridge._grid_mapper.migrate_vcore(vcore_ids[0], target_coord):
                return False

            self._entity_cells[entity_id] = target_coord
            return True

    def remove_entity(self, entity_id: str) -> bool:
        """Elimina una entidad del panal."""
        with self._lock:
            return self._entity_cells.pop(entity_id, None) is not None

    def get_entities_in_cell(self, coord: HexCoord) -> list[str]:
        """Obtiene todas las entidades en una celda."""
        return [eid for eid, c in self._entity_cells.items() if c == coord]

    def get_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas del adaptador."""
        entities_per_cell = {}
        for _entity_id, coord in self._entity_cells.items():
            key = f"{coord.q},{coord.r}"
            entities_per_cell[key] = entities_per_cell.get(key, 0) + 1

        return {
            "total_entities": len(self._entity_cells),
            "cells_used": len(set(self._entity_cells.values())),
            "entities_per_cell": entities_per_cell,
        }


__all__ = [
    "BridgeConfig",
    "CAMVHoneycombBridge",
    "VentHoneycombAdapter",
]
