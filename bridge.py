"""
HOC Bridge - Integración con CAMV (Cognitive Architecture for Multi-Virtualization)
=====================================================================================

Proporciona adaptadores y mapeos entre HOC y el sistema CAMV,
permitiendo que la topología hexagonal del panal se integre
transparentemente con el hipervisor de núcleos virtuales.

Mapeos principales:

    HOC                              CAMV
    ═══                              ════
    HoneycombGrid          ←→        CAMVHypervisor
    HoneycombCell          ←→        vCore
    QueenCell              ←→        CAMVRuntime
    NectarFlow             ←→        NeuralFabric
    SwarmScheduler         ←→        BrainScheduler
    HiveMemory             ←→        HTMC

Conversiones de coordenadas:

    Hexagonal (q, r)  ←→  Cartesiano (x, y)  ←→  vCore ID
    
         ⬡       
        ⬡ ⬡     ←→    [x,y grid]    ←→    vCore_0, vCore_1, ...
       ⬡ ⬡ ⬡

"""

from __future__ import annotations

import math
import hashlib
import threading
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Set, Tuple, Callable,
    Any, TypeVar, Generic, Iterator, Protocol, Union
)
from abc import ABC, abstractmethod

from .core import (
    HexCoord, HexDirection, HoneycombGrid, HoneycombCell,
    QueenCell, WorkerCell, DroneCell, NurseryCell,
    CellRole, CellState, HoneycombConfig
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOLOS DE CAMV (Interfaces esperadas)
# ═══════════════════════════════════════════════════════════════════════════════

class VCoreProtocol(Protocol):
    """Protocolo que debe implementar un vCore de CAMV."""
    
    @property
    def vcore_id(self) -> str: ...
    
    @property
    def state(self) -> str: ...
    
    def execute(self, payload: Dict[str, Any]) -> Any: ...
    
    def warmup(self) -> None: ...
    
    def shutdown(self) -> None: ...
    
    def get_metrics(self) -> Dict[str, Any]: ...


class HypervisorProtocol(Protocol):
    """Protocolo que debe implementar un Hypervisor de CAMV."""
    
    def allocate_vcore(self, config: Dict[str, Any]) -> VCoreProtocol: ...
    
    def deallocate_vcore(self, vcore_id: str) -> bool: ...
    
    def get_vcores(self) -> List[VCoreProtocol]: ...
    
    def get_stats(self) -> Dict[str, Any]: ...


class NeuralFabricProtocol(Protocol):
    """Protocolo para el NeuralFabric de CAMV."""
    
    def send(self, source: str, target: str, message: Any) -> bool: ...
    
    def broadcast(self, source: str, message: Any) -> int: ...
    
    def subscribe(self, topic: str, callback: Callable) -> str: ...


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSORES DE COORDENADAS
# ═══════════════════════════════════════════════════════════════════════════════

class HexToCartesian:
    """
    Conversor de coordenadas hexagonales a cartesianas.
    
    Soporta dos layouts:
    - FLAT_TOP: Hexágonos con lado plano arriba
    - POINTY_TOP: Hexágonos con vértice arriba
    
    Uso:
        converter = HexToCartesian(size=1.0, layout=HexLayout.FLAT_TOP)
        x, y = converter.convert(HexCoord(2, 3))
        center = converter.center(HexCoord(0, 0))
    """
    
    class Layout(Enum):
        FLAT_TOP = auto()
        POINTY_TOP = auto()
    
    def __init__(
        self,
        size: float = 1.0,
        layout: 'HexToCartesian.Layout' = None,
        origin: Tuple[float, float] = (0.0, 0.0)
    ):
        self.size = size
        self.layout = layout or self.Layout.FLAT_TOP
        self.origin = origin
    
    def convert(self, coord: HexCoord) -> Tuple[float, float]:
        """
        Convierte coordenada hexagonal a cartesiana.
        
        Args:
            coord: Coordenada hexagonal (q, r)
            
        Returns:
            Tupla (x, y) en coordenadas cartesianas
        """
        if self.layout == self.Layout.FLAT_TOP:
            x = self.size * (3/2 * coord.q)
            y = self.size * (math.sqrt(3)/2 * coord.q + math.sqrt(3) * coord.r)
        else:  # POINTY_TOP
            x = self.size * (math.sqrt(3) * coord.q + math.sqrt(3)/2 * coord.r)
            y = self.size * (3/2 * coord.r)
        
        return (x + self.origin[0], y + self.origin[1])
    
    def center(self, coord: HexCoord) -> Tuple[float, float]:
        """Obtiene el centro de un hexágono en coordenadas cartesianas."""
        return self.convert(coord)
    
    def corners(self, coord: HexCoord) -> List[Tuple[float, float]]:
        """
        Obtiene las 6 esquinas de un hexágono.
        
        Returns:
            Lista de 6 tuplas (x, y) en orden horario
        """
        cx, cy = self.center(coord)
        corners = []
        
        for i in range(6):
            if self.layout == self.Layout.FLAT_TOP:
                angle = math.pi / 3 * i
            else:
                angle = math.pi / 3 * i + math.pi / 6
            
            x = cx + self.size * math.cos(angle)
            y = cy + self.size * math.sin(angle)
            corners.append((x, y))
        
        return corners
    
    def bounding_box(self, coord: HexCoord) -> Tuple[float, float, float, float]:
        """
        Obtiene el bounding box de un hexágono.
        
        Returns:
            Tupla (min_x, min_y, max_x, max_y)
        """
        corners = self.corners(coord)
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return (min(xs), min(ys), max(xs), max(ys))


class CartesianToHex:
    """
    Conversor de coordenadas cartesianas a hexagonales.
    
    Uso:
        converter = CartesianToHex(size=1.0)
        coord = converter.convert(3.5, 2.1)
        nearest = converter.nearest(3.5, 2.1)
    """
    
    def __init__(
        self,
        size: float = 1.0,
        layout: HexToCartesian.Layout = None,
        origin: Tuple[float, float] = (0.0, 0.0)
    ):
        self.size = size
        self.layout = layout or HexToCartesian.Layout.FLAT_TOP
        self.origin = origin
    
    def convert(self, x: float, y: float) -> HexCoord:
        """
        Convierte coordenadas cartesianas a hexagonales.
        
        Args:
            x: Coordenada X
            y: Coordenada Y
            
        Returns:
            Coordenada hexagonal más cercana
        """
        # Ajustar por origen
        x = x - self.origin[0]
        y = y - self.origin[1]
        
        if self.layout == HexToCartesian.Layout.FLAT_TOP:
            q = (2/3 * x) / self.size
            r = (-1/3 * x + math.sqrt(3)/3 * y) / self.size
        else:  # POINTY_TOP
            q = (math.sqrt(3)/3 * x - 1/3 * y) / self.size
            r = (2/3 * y) / self.size
        
        return self._axial_round(q, r)
    
    def nearest(self, x: float, y: float) -> HexCoord:
        """Alias para convert()."""
        return self.convert(x, y)
    
    def _axial_round(self, q: float, r: float) -> HexCoord:
        """Redondea coordenadas axiales flotantes."""
        s = -q - r
        
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
    
    def in_hexagon(self, x: float, y: float, coord: HexCoord) -> bool:
        """
        Verifica si un punto está dentro de un hexágono.
        
        Args:
            x, y: Punto a verificar
            coord: Hexágono a verificar
            
        Returns:
            True si el punto está dentro del hexágono
        """
        nearest = self.convert(x, y)
        return nearest == coord


# ═══════════════════════════════════════════════════════════════════════════════
# MAPPERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VCoreMappingEntry:
    """Entrada del mapeo celda → vCore."""
    cell_coord: HexCoord
    vcore_id: str
    vcore_ref: Optional[VCoreProtocol] = None
    created_at: float = field(default_factory=lambda: __import__('time').time())
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        self._coord_to_vcores: Dict[HexCoord, List[VCoreMappingEntry]] = {}
        self._vcore_to_coord: Dict[str, HexCoord] = {}
        self._lock = threading.RLock()
    
    def map_cell(
        self,
        coord: HexCoord,
        vcore: VCoreProtocol,
        metadata: Optional[Dict[str, Any]] = None
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
        vcore_id = getattr(vcore, 'vcore_id', str(id(vcore)))
        
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
                cell_coord=coord,
                vcore_id=vcore_id,
                vcore_ref=vcore,
                metadata=metadata or {}
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
                    e for e in self._coord_to_vcores[coord]
                    if e.vcore_id != vcore_id
                ]
                if not self._coord_to_vcores[coord]:
                    del self._coord_to_vcores[coord]
            
            return True
    
    def get_vcores(self, coord: HexCoord) -> List[VCoreProtocol]:
        """Obtiene todos los vCores de una celda."""
        with self._lock:
            entries = self._coord_to_vcores.get(coord, [])
            return [e.vcore_ref for e in entries if e.vcore_ref is not None]
    
    def get_vcore_ids(self, coord: HexCoord) -> List[str]:
        """Obtiene todos los IDs de vCores de una celda."""
        with self._lock:
            entries = self._coord_to_vcores.get(coord, [])
            return [e.vcore_id for e in entries]
    
    def get_cell(self, vcore_id: str) -> Optional[HexCoord]:
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del mapper."""
        with self._lock:
            total_vcores = len(self._vcore_to_coord)
            cells_with_vcores = len(self._coord_to_vcores)
            
            vcores_per_cell = [
                len(entries) for entries in self._coord_to_vcores.values()
            ]
            
            return {
                "total_vcores": total_vcores,
                "cells_with_vcores": cells_with_vcores,
                "avg_vcores_per_cell": sum(vcores_per_cell) / len(vcores_per_cell) if vcores_per_cell else 0,
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
    
    def __init__(
        self,
        grid: HoneycombGrid,
        hypervisor: Optional[HypervisorProtocol] = None
    ):
        self.grid = grid
        self.hypervisor: Optional[HypervisorProtocol] = hypervisor
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
                            vcore = self.hypervisor.allocate_vcore({
                                "cell_coord": {"q": coord.q, "r": coord.r},
                                "index": i,
                            })
                            if vcore is not None:
                                self._cell_mapper.map_cell(coord, vcore)
                            # Si no hay vCores libres (hypervisor devuelve None), se omite sin error
                        except Exception as e:
                            logger.error(f"Failed to allocate vCore for {coord}: {e}")
            
            self._initialized = True
            return True
    
    def _initialize_stub_mapping(self) -> bool:
        """Inicializa mapeo stub sin hypervisor real."""
        
        @dataclass
        class StubVCore:
            vcore_id: str
            state: str = "ready"
            
            def execute(self, payload): return None
            def warmup(self): pass
            def shutdown(self): pass
            def get_metrics(self): return {}
        
        with self._lock:
            vcore_counter = 0
            for coord, cell in self.grid._cells.items():
                if isinstance(cell, WorkerCell):
                    for i in range(self.grid.config.vcores_per_cell):
                        vcore = StubVCore(vcore_id=f"stub_vcore_{vcore_counter}")
                        self._cell_mapper.map_cell(coord, vcore)
                        vcore_counter += 1
            
            self._initialized = True
            return True
    
    def get_vcores_for_cell(self, coord: HexCoord) -> List[VCoreProtocol]:
        """Obtiene los vCores asignados a una celda."""
        return self._cell_mapper.get_vcores(coord)
    
    def get_cell_for_vcore(self, vcore_id: str) -> Optional[HexCoord]:
        """Obtiene la celda de un vCore."""
        return self._cell_mapper.get_cell(vcore_id)
    
    def migrate_vcore(self, vcore_id: str, target_coord: HexCoord) -> bool:
        """Migra un vCore a una nueva celda."""
        return self._cell_mapper.migrate_vcore(vcore_id, target_coord)
    
    def get_mapping_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del mapeo."""
        return {
            "initialized": self._initialized,
            "hypervisor_present": self.hypervisor is not None,
            "cell_mapper": self._cell_mapper.get_stats(),
        }


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
        hypervisor: Optional[HypervisorProtocol] = None,
        config: Optional[BridgeConfig] = None
    ):
        self.grid = grid
        self.hypervisor: Optional[HypervisorProtocol] = hypervisor
        self.config = config or BridgeConfig()
        
        # Componentes
        self._grid_mapper = GridToHypervisorMapper(grid, hypervisor)
        self._hex_to_cart = HexToCartesian(
            size=self.config.hex_size,
            layout=self.config.layout
        )
        self._cart_to_hex = CartesianToHex(
            size=self.config.hex_size,
            layout=self.config.layout
        )
        
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
        self,
        coord: HexCoord,
        payload: Dict[str, Any],
        vcore_index: int = 0
    ) -> Optional[Any]:
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
            logger.error(f"Execution error on cell {coord}: {e}")
            self._errors += 1
            return None
    
    def broadcast_to_ring(
        self,
        center: HexCoord,
        radius: int,
        payload: Dict[str, Any]
    ) -> Dict[HexCoord, Any]:
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
    
    def migrate_vcores(
        self,
        source: HexCoord,
        target: HexCoord,
        count: int = 1
    ) -> int:
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
        
        for i, vcore_id in enumerate(vcore_ids[:count]):
            if self._grid_mapper.migrate_vcore(vcore_id, target):
                migrated += 1
                self._migrations += 1
        
        return migrated
    
    def hex_to_cartesian(self, coord: HexCoord) -> Tuple[float, float]:
        """Convierte coordenada hexagonal a cartesiana."""
        return self._hex_to_cart.convert(coord)
    
    def cartesian_to_hex(self, x: float, y: float) -> HexCoord:
        """Convierte coordenada cartesiana a hexagonal."""
        return self._cart_to_hex.convert(x, y)
    
    def tick(self) -> Dict[str, Any]:
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
    
    def get_stats(self) -> Dict[str, Any]:
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
    
    def __init__(
        self,
        grid: HoneycombGrid,
        bridge: CAMVHoneycombBridge
    ):
        self.grid = grid
        self.bridge = bridge
        
        # Mapeo entity → cell
        self._entity_cells: Dict[str, HexCoord] = {}
        self._lock = threading.RLock()
    
    def assign_entity(
        self,
        entity_id: str,
        preferred_coord: Optional[HexCoord] = None
    ) -> Optional[HexCoord]:
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
    
    def get_entity_cell(self, entity_id: str) -> Optional[HexCoord]:
        """Obtiene la celda de una entidad."""
        return self._entity_cells.get(entity_id)
    
    def execute_brain(
        self,
        entity_id: str,
        brain_state: Dict[str, Any]
    ) -> Optional[Any]:
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
    
    def migrate_entity(
        self,
        entity_id: str,
        target_coord: HexCoord
    ) -> bool:
        """Migra una entidad a una nueva celda."""
        with self._lock:
            if entity_id not in self._entity_cells:
                return False
            
            old_coord = self._entity_cells[entity_id]
            
            # Migrar vCore asociado
            vcore_ids = self.bridge._grid_mapper._cell_mapper.get_vcore_ids(old_coord)
            if vcore_ids:
                if not self.bridge._grid_mapper.migrate_vcore(vcore_ids[0], target_coord):
                    return False
            
            self._entity_cells[entity_id] = target_coord
            return True
    
    def remove_entity(self, entity_id: str) -> bool:
        """Elimina una entidad del panal."""
        with self._lock:
            return self._entity_cells.pop(entity_id, None) is not None
    
    def get_entities_in_cell(self, coord: HexCoord) -> List[str]:
        """Obtiene todas las entidades en una celda."""
        return [
            eid for eid, c in self._entity_cells.items()
            if c == coord
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del adaptador."""
        entities_per_cell = {}
        for entity_id, coord in self._entity_cells.items():
            key = f"{coord.q},{coord.r}"
            entities_per_cell[key] = entities_per_cell.get(key, 0) + 1
        
        return {
            "total_entities": len(self._entity_cells),
            "cells_used": len(set(self._entity_cells.values())),
            "entities_per_cell": entities_per_cell,
        }
