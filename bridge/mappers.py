"""
HOC Bridge — protocols + cell-to-vCore mappers.

Phase 6 split from the legacy ``bridge.py`` monolith. Defines the CAMV
integration surface (Protocols) and the bidirectional mappers between
hex coordinates and vCore IDs. Public API preserved through
``hoc.bridge.__init__``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core import HexCoord, HoneycombGrid, WorkerCell
from ..security import sanitize_error

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOLOS DE CAMV (Interfaces esperadas)
# ═══════════════════════════════════════════════════════════════════════════════


class VCoreProtocol(Protocol):
    """Protocolo que debe implementar un vCore de CAMV."""

    @property
    def vcore_id(self) -> str: ...

    @property
    def state(self) -> str: ...

    def execute(self, payload: dict[str, Any]) -> Any: ...

    def warmup(self) -> None: ...

    def shutdown(self) -> None: ...

    def get_metrics(self) -> dict[str, Any]: ...


class HypervisorProtocol(Protocol):
    """Protocolo que debe implementar un Hypervisor de CAMV."""

    def allocate_vcore(self, config: dict[str, Any]) -> VCoreProtocol: ...

    def deallocate_vcore(self, vcore_id: str) -> bool: ...

    def get_vcores(self) -> list[VCoreProtocol]: ...

    def get_stats(self) -> dict[str, Any]: ...


class NeuralFabricProtocol(Protocol):
    """Protocolo para el NeuralFabric de CAMV."""

    def send(self, source: str, target: str, message: Any) -> bool: ...

    def broadcast(self, source: str, message: Any) -> int: ...

    def subscribe(self, topic: str, callback: Callable) -> str: ...


# ═══════════════════════════════════════════════════════════════════════════════
# MAPPERS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class VCoreMappingEntry:
    """Entrada del mapeo celda → vCore."""

    cell_coord: HexCoord
    vcore_id: str
    vcore_ref: VCoreProtocol | None = None
    created_at: float = field(default_factory=lambda: __import__("time").time())
    metadata: dict[str, Any] = field(default_factory=dict)


class CellToVCoreMapper:
    """
    Mapea celdas del panal a vCores de CAMV.

    Mantiene un registro bidireccional entre HexCoord y vCore IDs,
    permitiendo lookups eficientes en ambas direcciones.

    Uso:
        mapper = CellToVCoreMapper()
        mapper.map_cell(coord, vcore)
        vcore = mapper.get_vcore(coord)
        coord = mapper.get_cell(vcore_id)
    """

    def __init__(self, max_vcores_per_cell: int = 8):
        self.max_vcores_per_cell = max_vcores_per_cell
        self._coord_to_vcores: dict[HexCoord, list[VCoreMappingEntry]] = {}
        self._vcore_to_coord: dict[str, HexCoord] = {}
        self._lock = threading.RLock()

    def map_cell(
        self, coord: HexCoord, vcore: VCoreProtocol, metadata: dict[str, Any] | None = None
    ) -> bool:
        """
        Mapea un vCore a una celda.

        Args:
            coord: Coordenada de la celda
            vcore: Referencia al vCore
            metadata: Metadatos opcionales

        Returns:
            True si el mapeo fue exitoso
        """
        vcore_id = getattr(vcore, "vcore_id", str(id(vcore)))

        with self._lock:
            # Verificar capacidad
            if coord in self._coord_to_vcores:
                if len(self._coord_to_vcores[coord]) >= self.max_vcores_per_cell:
                    logger.warning(f"Cell {coord} at vCore capacity")
                    return False
            else:
                self._coord_to_vcores[coord] = []

            # Verificar duplicado
            if vcore_id in self._vcore_to_coord:
                logger.warning(f"vCore {vcore_id} already mapped")
                return False

            # Crear entrada
            entry = VCoreMappingEntry(
                cell_coord=coord, vcore_id=vcore_id, vcore_ref=vcore, metadata=metadata or {}
            )

            self._coord_to_vcores[coord].append(entry)
            self._vcore_to_coord[vcore_id] = coord

            return True

    def unmap_vcore(self, vcore_id: str) -> bool:
        """
        Elimina el mapeo de un vCore.

        Returns:
            True si se eliminó correctamente
        """
        with self._lock:
            if vcore_id not in self._vcore_to_coord:
                return False

            coord = self._vcore_to_coord.pop(vcore_id)

            if coord in self._coord_to_vcores:
                self._coord_to_vcores[coord] = [
                    e for e in self._coord_to_vcores[coord] if e.vcore_id != vcore_id
                ]
                if not self._coord_to_vcores[coord]:
                    del self._coord_to_vcores[coord]

            return True

    def get_vcores(self, coord: HexCoord) -> list[VCoreProtocol]:
        """Obtiene todos los vCores de una celda."""
        with self._lock:
            entries = self._coord_to_vcores.get(coord, [])
            return [e.vcore_ref for e in entries if e.vcore_ref is not None]

    def get_vcore_ids(self, coord: HexCoord) -> list[str]:
        """Obtiene todos los IDs de vCores de una celda."""
        with self._lock:
            entries = self._coord_to_vcores.get(coord, [])
            return [e.vcore_id for e in entries]

    def get_cell(self, vcore_id: str) -> HexCoord | None:
        """Obtiene la celda de un vCore."""
        return self._vcore_to_coord.get(vcore_id)

    def migrate_vcore(self, vcore_id: str, new_coord: HexCoord) -> bool:
        """
        Migra un vCore a una nueva celda.

        Returns:
            True si la migración fue exitosa
        """
        with self._lock:
            if vcore_id not in self._vcore_to_coord:
                return False

            old_coord = self._vcore_to_coord[vcore_id]

            # Encontrar entrada
            entry = None
            for e in self._coord_to_vcores.get(old_coord, []):
                if e.vcore_id == vcore_id:
                    entry = e
                    break

            if not entry:
                return False

            # Verificar capacidad destino
            if new_coord in self._coord_to_vcores:
                if len(self._coord_to_vcores[new_coord]) >= self.max_vcores_per_cell:
                    return False
            else:
                self._coord_to_vcores[new_coord] = []

            # Migrar
            self._coord_to_vcores[old_coord].remove(entry)
            if not self._coord_to_vcores[old_coord]:
                del self._coord_to_vcores[old_coord]

            entry.cell_coord = new_coord
            self._coord_to_vcores[new_coord].append(entry)
            self._vcore_to_coord[vcore_id] = new_coord

            return True

    def get_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas del mapper."""
        with self._lock:
            total_vcores = len(self._vcore_to_coord)
            cells_with_vcores = len(self._coord_to_vcores)

            vcores_per_cell = [len(entries) for entries in self._coord_to_vcores.values()]

            return {
                "total_vcores": total_vcores,
                "cells_with_vcores": cells_with_vcores,
                "avg_vcores_per_cell": (
                    sum(vcores_per_cell) / len(vcores_per_cell) if vcores_per_cell else 0
                ),
                "max_vcores_in_cell": max(vcores_per_cell) if vcores_per_cell else 0,
            }


class GridToHypervisorMapper:
    """
    Mapea el grid hexagonal completo a un hypervisor de CAMV.

    Gestiona la relación entre la topología del panal y la
    infraestructura de virtualización subyacente.

    Uso:
        mapper = GridToHypervisorMapper(grid, hypervisor)
        mapper.initialize_mapping()
        stats = mapper.get_mapping_stats()
    """

    def __init__(self, grid: HoneycombGrid, hypervisor: HypervisorProtocol | None = None):
        self.grid = grid
        self.hypervisor: HypervisorProtocol | None = hypervisor
        self._cell_mapper = CellToVCoreMapper(grid.config.vcores_per_cell)
        self._lock = threading.RLock()
        self._initialized = False

    def initialize_mapping(self) -> bool:
        """
        Inicializa el mapeo entre grid y hypervisor.

        Crea vCores para cada celda trabajadora.
        """
        if self.hypervisor is None:
            logger.warning("No hypervisor configured, using stub mapping")
            return self._initialize_stub_mapping()

        with self._lock:
            for coord, cell in self.grid._cells.items():
                if isinstance(cell, WorkerCell):
                    # Solicitar vCores del hypervisor
                    for i in range(self.grid.config.vcores_per_cell):
                        try:
                            vcore = self.hypervisor.allocate_vcore(
                                {
                                    "cell_coord": {"q": coord.q, "r": coord.r},
                                    "index": i,
                                }
                            )
                            if vcore is not None:
                                self._cell_mapper.map_cell(coord, vcore)
                            # Si no hay vCores libres (hypervisor devuelve None), se omite sin error
                        except Exception as e:
                            logger.error(
                                f"Failed to allocate vCore for {coord}: {sanitize_error(e)}"
                            )

            self._initialized = True
            return True

    def _initialize_stub_mapping(self) -> bool:
        """Inicializa mapeo stub sin hypervisor real."""

        @dataclass
        class StubVCore:
            vcore_id: str
            state: str = "ready"

            def execute(self, payload):
                return None

            def warmup(self):
                pass

            def shutdown(self):
                pass

            def get_metrics(self):
                return {}

        with self._lock:
            vcore_counter = 0
            for coord, cell in self.grid._cells.items():
                if isinstance(cell, WorkerCell):
                    for _i in range(self.grid.config.vcores_per_cell):
                        vcore = StubVCore(vcore_id=f"stub_vcore_{vcore_counter}")
                        self._cell_mapper.map_cell(coord, vcore)
                        vcore_counter += 1

            self._initialized = True
            return True

    def get_vcores_for_cell(self, coord: HexCoord) -> list[VCoreProtocol]:
        """Obtiene los vCores asignados a una celda."""
        return self._cell_mapper.get_vcores(coord)

    def get_cell_for_vcore(self, vcore_id: str) -> HexCoord | None:
        """Obtiene la celda de un vCore."""
        return self._cell_mapper.get_cell(vcore_id)

    def migrate_vcore(self, vcore_id: str, target_coord: HexCoord) -> bool:
        """Migra un vCore a una nueva celda."""
        return self._cell_mapper.migrate_vcore(vcore_id, target_coord)

    def get_mapping_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas del mapeo."""
        return {
            "initialized": self._initialized,
            "hypervisor_present": self.hypervisor is not None,
            "cell_mapper": self._cell_mapper.get_stats(),
        }


__all__ = [
    "VCoreProtocol",
    "HypervisorProtocol",
    "NeuralFabricProtocol",
    "VCoreMappingEntry",
    "CellToVCoreMapper",
    "GridToHypervisorMapper",
]
