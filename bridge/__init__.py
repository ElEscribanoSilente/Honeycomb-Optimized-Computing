"""
HOC Bridge subpackage — integración con CAMV (Cognitive Architecture for
Multi-Virtualization).

Phase 6.5 split del monolito histórico ``bridge.py`` (886 LOC) en tres
módulos cohesivos:

- :mod:`hoc.bridge.converters` — geometría hex ↔ cartesiana.
- :mod:`hoc.bridge.mappers` — protocolos CAMV + bidirectional mapping
  celda ↔ vCore.
- :mod:`hoc.bridge.adapters` — el bridge principal (CAMV) y el
  adaptador para entidades de Vent.

Mismo patrón que Phase 3 con ``core.py`` → ``core/`` y ``metrics.py``
→ ``metrics/``: el facade público en ``__init__.py`` re-exporta los
nombres que el resto del repo (y los usuarios externos) consumen vía
``from hoc.bridge import …`` o ``from hoc import …``.

Mapeos principales:

    HOC                              CAMV
    ═══                              ════
    HoneycombGrid          ←→        CAMVHypervisor
    HoneycombCell          ←→        vCore
    QueenCell              ←→        CAMVRuntime
    NectarFlow             ←→        NeuralFabric
    SwarmScheduler         ←→        BrainScheduler
    HiveMemory             ←→        HTMC
"""

from __future__ import annotations

from .adapters import (
    BridgeConfig,
    CAMVHoneycombBridge,
    VentHoneycombAdapter,
)
from .converters import (
    CartesianToHex,
    HexToCartesian,
)
from .mappers import (
    CellToVCoreMapper,
    GridToHypervisorMapper,
    HypervisorProtocol,
    NeuralFabricProtocol,
    VCoreMappingEntry,
    VCoreProtocol,
)

__all__ = [
    # Adapters
    "BridgeConfig",
    "CAMVHoneycombBridge",
    "VentHoneycombAdapter",
    # Converters
    "HexToCartesian",
    "CartesianToHex",
    # Mappers
    "CellToVCoreMapper",
    "GridToHypervisorMapper",
    "VCoreMappingEntry",
    # Protocols
    "VCoreProtocol",
    "HypervisorProtocol",
    "NeuralFabricProtocol",
]
