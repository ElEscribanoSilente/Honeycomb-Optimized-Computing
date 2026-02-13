"""
HOC Hive Memory - Sistema de Memoria Distribuida
=================================================

Implementa almacenamiento distribuido con metáfora de colmena:

CAPAS:
- PollenCache: Cache L1 local ultra-rápido (volátil)
- CombStorage: Almacenamiento en celdas del panal (distribuido)
- HoneyArchive: Archivo comprimido de largo plazo (persistente)

Flujo de datos:

    READ:  PollenCache → CombStorage → HoneyArchive
    WRITE: PollenCache ← CombStorage ← HoneyArchive

    ┌───────────────────────────────────────────────────────────┐
    │                      HiveMemory                           │
    │  ┌─────────────┐                                          │
    │  │ PollenCache │  ← Hot data (ns access)                  │
    │  │   (L1)      │                                          │
    │  └──────┬──────┘                                          │
    │         │                                                 │
    │         ▼                                                 │
    │  ┌─────────────┐                                          │
    │  │ CombStorage │  ← Distributed across cells              │
    │  │   (L2)      │     ⬡ ⬡ ⬡ ⬡ ⬡                           │
    │  └──────┬──────┘                                          │
    │         │                                                 │
    │         ▼                                                 │
    │  ┌─────────────┐                                          │
    │  │HoneyArchive │  ← Compressed persistent storage         │
    │  │   (L3)      │                                          │
    │  └─────────────┘                                          │
    └───────────────────────────────────────────────────────────┘

"""

from __future__ import annotations

import time
import hashlib
import threading
import zlib
import pickle
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Set, Tuple, Callable,
    Any, TypeVar, Generic, Iterator
)
from collections import OrderedDict
import numpy as np

from .core import HexCoord, HoneycombCell, HoneycombGrid

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ═══════════════════════════════════════════════════════════════════════════════
# POLÍTICAS
# ═══════════════════════════════════════════════════════════════════════════════

class EvictionPolicy(Enum):
    """Políticas de evicción de cache."""
    LRU = auto()           # Least Recently Used
    LFU = auto()           # Least Frequently Used
    FIFO = auto()          # First In First Out
    RANDOM = auto()        # Random eviction
    SIZE_BASED = auto()    # Evict largest first


class ReplicationPolicy(Enum):
    """Políticas de replicación."""
    NONE = auto()          # Sin replicación
    MIRROR = auto()        # Espejo exacto en celda vecina
    RING = auto()          # Replicar en anillo
    QUORUM = auto()        # Requiere quorum para escritura


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryConfig:
    """Configuración del sistema de memoria."""
    
    # PollenCache (L1)
    pollen_max_items: int = 10000
    pollen_max_size_bytes: int = 100 * 1024 * 1024  # 100MB
    pollen_eviction: EvictionPolicy = EvictionPolicy.LRU
    pollen_ttl_seconds: float = 60.0
    
    # CombStorage (L2)
    comb_replication: ReplicationPolicy = ReplicationPolicy.MIRROR
    comb_max_items_per_cell: int = 1000
    comb_compression_enabled: bool = True
    comb_compression_level: int = 6
    
    # HoneyArchive (L3)
    honey_compression_enabled: bool = True
    honey_compression_level: int = 9
    honey_checkpoint_interval: int = 100  # ticks
    
    # General
    write_through: bool = True  # Write to all layers immediately
    read_through: bool = True   # Populate cache on read misses


# ═══════════════════════════════════════════════════════════════════════════════
# POLLEN CACHE (L1)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CacheEntry:
    """Entrada individual de cache."""
    key: str
    value: Any
    size_bytes: int
    created_at: float
    accessed_at: float
    access_count: int = 0
    
    def touch(self) -> None:
        """Actualiza tiempo de acceso."""
        self.accessed_at = time.time()
        self.access_count += 1
    
    def is_expired(self, ttl: float) -> bool:
        """Verifica si la entrada expiró."""
        return (time.time() - self.created_at) > ttl


class PollenCache:
    """
    Cache L1 local ultra-rápido.
    
    Características:
    - Acceso O(1)
    - TTL configurable
    - Múltiples políticas de evicción
    - Estadísticas de hit/miss
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._total_size: int = 0
        self._lock = threading.RLock()
        
        # Estadísticas
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene un valor del cache.
        
        Returns:
            Valor o None si no existe/expirado
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Verificar TTL
            if entry.is_expired(self.config.pollen_ttl_seconds):
                self._evict_key(key)
                self._misses += 1
                return None
            
            # Actualizar acceso
            entry.touch()
            
            # Mover al final para LRU
            if self.config.pollen_eviction == EvictionPolicy.LRU:
                self._cache.move_to_end(key)
            
            self._hits += 1
            return entry.value
    
    def put(self, key: str, value: Any) -> bool:
        """
        Almacena un valor en el cache.
        
        Returns:
            True si se almacenó correctamente
        """
        # Estimar tamaño
        try:
            serialized = pickle.dumps(value)
            size = len(serialized)
        except Exception:
            size = 1024  # Fallback
        
        with self._lock:
            # Evict si es necesario
            while (
                len(self._cache) >= self.config.pollen_max_items or
                self._total_size + size > self.config.pollen_max_size_bytes
            ):
                if not self._evict_one():
                    return False
            
            # Crear entrada
            now = time.time()
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size,
                created_at=now,
                accessed_at=now,
            )
            
            # Reemplazar si existe
            if key in self._cache:
                old_entry = self._cache[key]
                self._total_size -= old_entry.size_bytes
            
            self._cache[key] = entry
            self._total_size += size
            
            return True
    
    def delete(self, key: str) -> bool:
        """Elimina una entrada."""
        with self._lock:
            return self._evict_key(key)
    
    def _evict_key(self, key: str) -> bool:
        """Evicta una key específica."""
        if key not in self._cache:
            return False
        
        entry = self._cache.pop(key)
        self._total_size -= entry.size_bytes
        self._evictions += 1
        return True
    
    def _evict_one(self) -> bool:
        """Evicta una entrada según la política."""
        if not self._cache:
            return False
        
        if self.config.pollen_eviction == EvictionPolicy.LRU:
            # Evict oldest (first in OrderedDict)
            key = next(iter(self._cache))
        
        elif self.config.pollen_eviction == EvictionPolicy.LFU:
            # Evict least frequently used
            key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
        
        elif self.config.pollen_eviction == EvictionPolicy.FIFO:
            key = next(iter(self._cache))
        
        elif self.config.pollen_eviction == EvictionPolicy.SIZE_BASED:
            key = max(self._cache.keys(), key=lambda k: self._cache[k].size_bytes)
        
        else:  # RANDOM
            import random
            key = random.choice(list(self._cache.keys()))
        
        return self._evict_key(key)
    
    def clear(self) -> None:
        """Limpia todo el cache."""
        with self._lock:
            self._cache.clear()
            self._total_size = 0
    
    def cleanup_expired(self) -> int:
        """Limpia entradas expiradas."""
        removed = 0
        with self._lock:
            expired = [
                key for key, entry in self._cache.items()
                if entry.is_expired(self.config.pollen_ttl_seconds)
            ]
            for key in expired:
                self._evict_key(key)
                removed += 1
        return removed
    
    @property
    def size(self) -> int:
        return len(self._cache)
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "items": len(self._cache),
            "size_bytes": self._total_size,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": self.hit_rate,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# COMB STORAGE (L2)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CombCell:
    """Una celda de almacenamiento en el panal."""
    coord: HexCoord
    data: Dict[str, bytes] = field(default_factory=dict)
    metadata: Dict[str, Dict] = field(default_factory=dict)
    
    @property
    def item_count(self) -> int:
        return len(self.data)
    
    @property
    def total_size(self) -> int:
        return sum(len(v) for v in self.data.values())


class CombStorage:
    """
    Almacenamiento distribuido en celdas del panal.
    
    Los datos se distribuyen basándose en hash de la key,
    mapeando a coordenadas hexagonales.
    
    Características:
    - Distribución consistente por hash
    - Replicación configurable
    - Compresión opcional
    """
    
    def __init__(self, grid: HoneycombGrid, config: MemoryConfig):
        self.grid = grid
        self.config = config
        self._cells: Dict[HexCoord, CombCell] = {}
        self._lock = threading.RLock()
        
        # Inicializar celdas de almacenamiento
        for coord in grid._cells:
            self._cells[coord] = CombCell(coord=coord)
    
    def _hash_to_coord(self, key: str) -> HexCoord:
        """Mapea una key a una coordenada hexagonal."""
        h = hashlib.sha256(key.encode()).digest()
        
        # Usar primeros bytes para q y r
        q = int.from_bytes(h[0:4], 'big', signed=True) % (self.grid.config.radius * 2 + 1) - self.grid.config.radius
        r = int.from_bytes(h[4:8], 'big', signed=True) % (self.grid.config.radius * 2 + 1) - self.grid.config.radius
        
        coord = HexCoord(q, r)
        
        # Si la coordenada no existe, buscar la más cercana
        if coord not in self._cells:
            min_dist = float('inf')
            closest = HexCoord.origin()
            for c in self._cells:
                d = coord.distance_to(c)
                if d < min_dist:
                    min_dist = d
                    closest = c
            coord = closest
        
        return coord
    
    def _get_replicas(self, primary: HexCoord) -> List[HexCoord]:
        """Obtiene las coordenadas de réplicas."""
        if self.config.comb_replication == ReplicationPolicy.NONE:
            return []
        
        elif self.config.comb_replication == ReplicationPolicy.MIRROR:
            # Una réplica en el vecino más cercano disponible
            grid_cell = self.grid.get_cell(primary)
            if grid_cell:
                neighbors = grid_cell.get_all_neighbors()
                if neighbors:
                    return [neighbors[0].coord]
            return []
        
        elif self.config.comb_replication == ReplicationPolicy.RING:
            # Réplicas en el anillo
            return [
                c for c in primary.ring(1)
                if c in self._cells
            ][:2]  # Máximo 2 réplicas
        
        return []
    
    def _compress(self, data: bytes) -> bytes:
        """Comprime datos si está habilitado."""
        if self.config.comb_compression_enabled:
            return zlib.compress(data, self.config.comb_compression_level)
        return data
    
    def _decompress(self, data: bytes) -> bytes:
        """Descomprime datos si es necesario."""
        if self.config.comb_compression_enabled:
            try:
                return zlib.decompress(data)
            except zlib.error:
                return data  # Datos no comprimidos
        return data
    
    def put(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Almacena un valor.
        
        Args:
            key: Clave única
            value: Valor a almacenar
            metadata: Metadatos opcionales
            
        Returns:
            True si se almacenó correctamente
        """
        try:
            # Serializar y comprimir
            serialized = pickle.dumps(value)
            compressed = self._compress(serialized)
            
            # Determinar ubicación
            primary = self._hash_to_coord(key)
            replicas = self._get_replicas(primary)
            
            with self._lock:
                # Verificar capacidad
                primary_cell = self._cells[primary]
                if primary_cell.item_count >= self.config.comb_max_items_per_cell:
                    logger.warning(f"CombStorage cell {primary} at capacity")
                    return False
                
                # Almacenar en primaria
                primary_cell.data[key] = compressed
                primary_cell.metadata[key] = {
                    "created_at": time.time(),
                    "size_original": len(serialized),
                    "size_compressed": len(compressed),
                    **(metadata or {})
                }
                
                # Replicar
                for replica_coord in replicas:
                    replica_cell = self._cells.get(replica_coord)
                    if replica_cell:
                        replica_cell.data[key] = compressed
                        replica_cell.metadata[key] = primary_cell.metadata[key]
            
            return True
            
        except Exception as e:
            logger.error(f"CombStorage put error: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene un valor.
        
        Returns:
            Valor o None si no existe
        """
        primary = self._hash_to_coord(key)
        
        with self._lock:
            cell = self._cells.get(primary)
            if not cell or key not in cell.data:
                # Intentar réplicas
                for replica_coord in self._get_replicas(primary):
                    replica = self._cells.get(replica_coord)
                    if replica and key in replica.data:
                        cell = replica
                        break
                else:
                    return None
            
            try:
                compressed = cell.data[key]
                decompressed = self._decompress(compressed)
                return pickle.loads(decompressed)
            except Exception as e:
                logger.error(f"CombStorage get error: {e}")
                return None
    
    def delete(self, key: str) -> bool:
        """Elimina un valor."""
        primary = self._hash_to_coord(key)
        replicas = self._get_replicas(primary)
        
        with self._lock:
            deleted = False
            
            # Eliminar de primaria
            cell = self._cells.get(primary)
            if cell and key in cell.data:
                del cell.data[key]
                cell.metadata.pop(key, None)
                deleted = True
            
            # Eliminar de réplicas
            for replica_coord in replicas:
                replica = self._cells.get(replica_coord)
                if replica and key in replica.data:
                    del replica.data[key]
                    replica.metadata.pop(key, None)
            
            return deleted
    
    def exists(self, key: str) -> bool:
        """Verifica si una key existe."""
        primary = self._hash_to_coord(key)
        cell = self._cells.get(primary)
        return cell is not None and key in cell.data
    
    def get_cell_stats(self, coord: HexCoord) -> Optional[Dict]:
        """Obtiene estadísticas de una celda."""
        cell = self._cells.get(coord)
        if not cell:
            return None
        
        return {
            "coord": {"q": coord.q, "r": coord.r},
            "items": cell.item_count,
            "total_size": cell.total_size,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas globales."""
        total_items = sum(c.item_count for c in self._cells.values())
        total_size = sum(c.total_size for c in self._cells.values())
        
        return {
            "cells": len(self._cells),
            "total_items": total_items,
            "total_size": total_size,
            "avg_items_per_cell": total_items / len(self._cells) if self._cells else 0,
            "replication": self.config.comb_replication.name,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HONEY ARCHIVE (L3)
# ═══════════════════════════════════════════════════════════════════════════════

class HoneyArchive:
    """
    Archivo persistente comprimido de largo plazo.
    
    Características:
    - Alta compresión
    - Acceso menos frecuente
    - Checkpoint periódico
    - Recuperación de fallos
    """
    
    def __init__(self, config: MemoryConfig, base_path: str = "/tmp/honey"):
        self.config = config
        self.base_path = base_path
        self._archive: Dict[str, bytes] = {}
        self._metadata: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self._tick_count = 0
    
    def _compress(self, data: bytes) -> bytes:
        """Compresión de alto nivel."""
        if self.config.honey_compression_enabled:
            return zlib.compress(data, self.config.honey_compression_level)
        return data
    
    def _decompress(self, data: bytes) -> bytes:
        """Descompresión."""
        if self.config.honey_compression_enabled:
            try:
                return zlib.decompress(data)
            except zlib.error:
                return data
        return data
    
    def archive(self, key: str, value: Any, metadata: Optional[Dict] = None) -> bool:
        """
        Archiva un valor.
        
        Args:
            key: Clave única
            value: Valor a archivar
            metadata: Metadatos opcionales
            
        Returns:
            True si se archivó correctamente
        """
        try:
            serialized = pickle.dumps(value)
            compressed = self._compress(serialized)
            
            with self._lock:
                self._archive[key] = compressed
                self._metadata[key] = {
                    "archived_at": time.time(),
                    "size_original": len(serialized),
                    "size_compressed": len(compressed),
                    "compression_ratio": len(serialized) / len(compressed) if compressed else 1,
                    **(metadata or {})
                }
            
            return True
            
        except Exception as e:
            logger.error(f"HoneyArchive archive error: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Recupera un valor archivado.
        
        Returns:
            Valor o None si no existe
        """
        with self._lock:
            if key not in self._archive:
                return None
            
            try:
                compressed = self._archive[key]
                decompressed = self._decompress(compressed)
                return pickle.loads(decompressed)
            except Exception as e:
                logger.error(f"HoneyArchive retrieve error: {e}")
                return None
    
    def delete(self, key: str) -> bool:
        """Elimina un valor archivado."""
        with self._lock:
            if key in self._archive:
                del self._archive[key]
                self._metadata.pop(key, None)
                return True
            return False
    
    def tick(self) -> None:
        """Procesa un tick (para checkpoints)."""
        self._tick_count += 1
        
        if self._tick_count % self.config.honey_checkpoint_interval == 0:
            self._checkpoint()
    
    def _checkpoint(self) -> None:
        """Guarda checkpoint a disco."""
        # En implementación real, serializar a disco
        logger.debug(f"HoneyArchive checkpoint at tick {self._tick_count}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas."""
        total_original = sum(m.get("size_original", 0) for m in self._metadata.values())
        total_compressed = sum(m.get("size_compressed", 0) for m in self._metadata.values())
        
        return {
            "items": len(self._archive),
            "total_size_original": total_original,
            "total_size_compressed": total_compressed,
            "overall_compression_ratio": total_original / total_compressed if total_compressed else 1,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HIVE MEMORY - SISTEMA UNIFICADO
# ═══════════════════════════════════════════════════════════════════════════════

class HiveMemory:
    """
    Sistema de Memoria Unificado del Panal.
    
    Coordina las tres capas de almacenamiento:
    - PollenCache (L1): Cache local rápido
    - CombStorage (L2): Almacenamiento distribuido
    - HoneyArchive (L3): Archivo persistente
    
    Uso:
        memory = HiveMemory(grid)
        
        # Escribir
        memory.put("entity_123", entity_data)
        
        # Leer (busca en todas las capas)
        data = memory.get("entity_123")
        
        # Archivar explícitamente
        memory.archive("old_entity", old_data)
    """
    
    def __init__(
        self,
        grid: HoneycombGrid,
        config: Optional[MemoryConfig] = None
    ):
        self.config = config or MemoryConfig()
        
        # Capas de memoria
        self._pollen = PollenCache(self.config)
        self._comb = CombStorage(grid, self.config)
        self._honey = HoneyArchive(self.config)
        
        self._lock = threading.RLock()
    
    def put(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict] = None,
        skip_cache: bool = False,
        archive: bool = False
    ) -> bool:
        """
        Almacena un valor.
        
        Args:
            key: Clave única
            value: Valor a almacenar
            metadata: Metadatos opcionales
            skip_cache: No almacenar en cache L1
            archive: También archivar en L3
            
        Returns:
            True si se almacenó correctamente
        """
        success = True
        
        # L2: CombStorage (siempre)
        if not self._comb.put(key, value, metadata):
            success = False
        
        # L1: PollenCache (opcional)
        if not skip_cache and self.config.write_through:
            self._pollen.put(key, value)
        
        # L3: HoneyArchive (opcional)
        if archive:
            self._honey.archive(key, value, metadata)
        
        return success
    
    def get(
        self,
        key: str,
        include_archive: bool = False
    ) -> Optional[Any]:
        """
        Obtiene un valor, buscando en todas las capas.
        
        Args:
            key: Clave a buscar
            include_archive: También buscar en L3
            
        Returns:
            Valor o None si no existe
        """
        # L1: PollenCache
        value = self._pollen.get(key)
        if value is not None:
            return value
        
        # L2: CombStorage
        value = self._comb.get(key)
        if value is not None:
            # Populate cache on read miss
            if self.config.read_through:
                self._pollen.put(key, value)
            return value
        
        # L3: HoneyArchive (opcional)
        if include_archive:
            value = self._honey.retrieve(key)
            if value is not None:
                # Promote to upper layers
                if self.config.read_through:
                    self._pollen.put(key, value)
                    self._comb.put(key, value)
                return value
        
        return None
    
    def delete(self, key: str, all_layers: bool = True) -> bool:
        """
        Elimina un valor.
        
        Args:
            key: Clave a eliminar
            all_layers: Eliminar de todas las capas
            
        Returns:
            True si se eliminó de al menos una capa
        """
        deleted = False
        
        deleted |= self._pollen.delete(key)
        deleted |= self._comb.delete(key)
        
        if all_layers:
            deleted |= self._honey.delete(key)
        
        return deleted
    
    def archive(self, key: str, value: Optional[Any] = None) -> bool:
        """
        Archiva un valor en L3.
        
        Si no se proporciona valor, archiva desde L2.
        """
        if value is None:
            value = self._comb.get(key)
            if value is None:
                return False
        
        return self._honey.archive(key, value)
    
    def exists(self, key: str) -> bool:
        """Verifica si una key existe en alguna capa."""
        if self._pollen.get(key) is not None:
            return True
        return self._comb.exists(key)
    
    def tick(self) -> None:
        """Procesa un tick del sistema de memoria."""
        # Limpiar entradas expiradas del cache
        self._pollen.cleanup_expired()
        
        # Tick del archive
        self._honey.tick()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de todas las capas."""
        return {
            "pollen_cache": self._pollen.get_stats(),
            "comb_storage": self._comb.get_stats(),
            "honey_archive": self._honey.get_stats(),
        }
    
    @property
    def pollen(self) -> PollenCache:
        return self._pollen
    
    @property
    def comb(self) -> CombStorage:
        return self._comb
    
    @property
    def honey(self) -> HoneyArchive:
        return self._honey
