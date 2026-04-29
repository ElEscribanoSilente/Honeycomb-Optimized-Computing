"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    HOC - Honeycomb Optimized Computing                       ║
║           Computación Bio-Inspirada con Topología Hexagonal                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║                              🐝 ARQUITECTURA 🐝                              ║
║                                                                              ║
║       La estructura hexagonal (panal) ofrece propiedades únicas:             ║
║       • Máxima eficiencia de empaquetado (ratio área/perímetro)              ║
║       • 6 vecinos directos (vs 4 en grids cuadrados)                         ║
║       • Distribución uniforme de carga                                       ║
║       • Rutas de comunicación más cortas                                     ║
║       • Auto-organización emergente                                          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║                    ⬡ ⬡ ⬡     GRID HEXAGONAL     ⬡ ⬡ ⬡                       ║
║                                                                              ║
║                         ⬡       ⬡       ⬡                                    ║
║                       ⬡   ⬡   ⬡   ⬡   ⬡   ⬡                                 ║
║                         ⬡   👑   ⬡   ⬡   ⬡                                   ║
║                       ⬡   ⬡   ⬡   ⬡   ⬡   ⬡                                 ║
║                         ⬡       ⬡       ⬡                                    ║
║                                                                              ║
║                 Cada celda ⬡ puede contener múltiples vCores                 ║
║                 La reina 👑 coordina el cluster (Queen Cell)                  ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                      INTEGRACIÓN CON CAMV                               │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                         │ ║
║  │  HOC                              CAMV                                  │ ║
║  │  ═══                              ════                                  │ ║
║  │  HoneycombGrid          ←→        CAMVHypervisor                        │ ║
║  │  HoneycombCell          ←→        vCore                                 │ ║
║  │  QueenCell              ←→        CAMVRuntime                           │ ║
║  │  NectarFlow             ←→        NeuralFabric                          │ ║
║  │  SwarmScheduler         ←→        BrainScheduler                        │ ║
║  │  HiveMemory             ←→        HTMC                                  │ ║
║  │                                                                         │ ║
║  │  HOC extiende CAMV con:                                                 │ ║
║  │  • Topología hexagonal optimizada                                       │ ║
║  │  • Scheduling basado en feromonas (stigmergy)                           │ ║
║  │  • Comunicación por danza (Waggle Dance Protocol)                       │ ║
║  │  • Auto-balanceo tipo colmena                                           │ ║
║  │  • Resiliencia con redundancia hexagonal                                │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  JERARQUÍA DE COMPONENTES:                                                   ║
║                                                                              ║
║  HoneycombGrid (Grid hexagonal principal)                                    ║
║    ├── QueenCell (Celda reina - coordinación)                               ║
║    │     └── QueenCore (Cerebro de coordinación)                            ║
║    ├── WorkerCell[] (Celdas trabajadoras - cómputo)                         ║
║    │     └── vCore[] (Virtual cores de CAMV)                                ║
║    ├── DroneCell[] (Celdas dron - comunicación externa)                     ║
║    │     └── ExternalBridge (Puente a otros grids)                          ║
║    └── NurseryCell[] (Celdas guardería - spawning)                          ║
║          └── EntityIncubator (Incubadora de entidades)                      ║
║                                                                              ║
║  NectarFlow (Sistema de comunicación)                                        ║
║    ├── PheromoneTrail (Rastros de feromonas)                                ║
║    ├── WaggleDance (Protocolo de danza)                                     ║
║    └── RoyalJelly (Canal de alta prioridad)                                 ║
║                                                                              ║
║  SwarmScheduler (Scheduler bio-inspirado)                                    ║
║    ├── ForagerBehavior (Búsqueda de trabajo)                                ║
║    ├── NurseBehavior (Cuidado de nuevos procesos)                           ║
║    └── ScoutBehavior (Exploración de recursos)                              ║
║                                                                              ║
║  HiveMemory (Sistema de memoria distribuida)                                 ║
║    ├── CombStorage (Almacenamiento en celdas)                               ║
║    ├── PollenCache (Cache de datos frecuentes)                              ║
║    └── HoneyArchive (Archivo persistente comprimido)                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Versión: 1.0.0
Autor: Vent Framework
Licencia: MIT
"""

__version__ = "1.0.0"
__author__ = "Vent Framework"
__license__ = "MIT"

# ═══════════════════════════════════════════════════════════════════════════════
# CORE - Estructuras fundamentales del panal
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# CAMV BRIDGE - Integración con CAMV
# ═══════════════════════════════════════════════════════════════════════════════
from .bridge import (
    # Adaptadores
    CAMVHoneycombBridge,
    CartesianToHex,
    # Mapeos
    CellToVCoreMapper,
    GridToHypervisorMapper,
    # Conversores
    HexToCartesian,
    VentHoneycombAdapter,
)
from .core import (
    CellRole,
    CellState,
    DroneCell,
    # Event bus management (v3.1)
    EventBus,
    GridTopology,
    # Coordenadas hexagonales
    HexCoord,
    HexDirection,
    HexRing,
    # Tipos de celdas
    HoneycombCell,
    HoneycombConfig,
    # Grid principal
    HoneycombGrid,
    NurseryCell,
    QueenCell,
    WorkerCell,
    get_event_bus,
    reset_event_bus,
    set_event_bus,
)

# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVABILITY - Logging estructurado (Phase 5.3)
# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5.3: structured event logging lives in ``hoc.core.observability``
# (a sibling of cells_base/grid/etc. inside the core/ subpackage). It
# avoids the dual-import dance that the package-dir trick imposes on
# top-level modules. Re-export the public API here so callers can do
# ``from hoc import configure_logging, get_event_logger``.
from .core.observability import (
    EVENT_LOGGER_NAME,
    configure_logging,
    get_event_logger,
)

# ═══════════════════════════════════════════════════════════════════════════════
# HIVE MEMORY - Sistema de memoria distribuida
# ═══════════════════════════════════════════════════════════════════════════════
from .memory import (
    CombCell,
    # Capas de almacenamiento
    CombStorage,
    # Políticas
    EvictionPolicy,
    # Memoria principal
    HiveMemory,
    HoneyArchive,
    MemoryConfig,
    PollenCache,
    ReplicationPolicy,
)

# ═══════════════════════════════════════════════════════════════════════════════
# METRICS - Observabilidad
# ═══════════════════════════════════════════════════════════════════════════════
from .metrics import (
    CellMetrics,
    FlowVisualizer,
    HeatmapRenderer,
    # Métricas
    HiveMetrics,
    # Visualización
    HoneycombVisualizer,
    SwarmMetrics,
)

# ═══════════════════════════════════════════════════════════════════════════════
# NECTAR FLOW - Sistema de comunicación
# ═══════════════════════════════════════════════════════════════════════════════
from .nectar import (
    DanceDirection,
    DanceMessage,
    NectarChannel,
    # Flujo principal
    NectarFlow,
    NectarPriority,
    PheromoneDecay,
    # Feromonas
    PheromoneTrail,
    PheromoneType,
    RoyalCommand,
    # Canal de alta prioridad
    RoyalJelly,
    # Protocolos de comunicación
    WaggleDance,
)

# ═══════════════════════════════════════════════════════════════════════════════
# RESILIENCE - Sistema de resiliencia
# ═══════════════════════════════════════════════════════════════════════════════
from .resilience import (
    CellFailover,
    CombRepair,
    # Replicación
    HexRedundancy,
    # Tolerancia a fallos
    HiveResilience,
    MirrorCell,
    QueenSuccession,
    # Recuperación
    SwarmRecovery,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SANDBOX - Aislamiento de procesos (Phase 7.4)
# ═══════════════════════════════════════════════════════════════════════════════
from .sandbox import (
    IsolationMode,
    SandboxConfig,
    SandboxCrashed,
    SandboxedTaskRunner,
    SandboxError,
    SandboxNotSupported,
    SandboxTimeout,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SWARM SCHEDULER - Scheduling bio-inspirado
# ═══════════════════════════════════════════════════════════════════════════════
from .swarm import (
    # Comportamientos
    BeeBehavior,
    ForagerBehavior,
    GuardBehavior,
    # Tareas
    HiveTask,
    LoadDistribution,
    NurseBehavior,
    ScoutBehavior,
    # Balanceo
    SwarmBalancer,
    SwarmConfig,
    SwarmPolicy,
    # Scheduler principal
    SwarmScheduler,
    TaskNectar,
    TaskPollen,
)

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__license__",
    # Core
    "HoneycombGrid",
    "HoneycombConfig",
    "GridTopology",
    "HoneycombCell",
    "CellState",
    "CellRole",
    "QueenCell",
    "WorkerCell",
    "DroneCell",
    "NurseryCell",
    "HexCoord",
    "HexDirection",
    "HexRing",
    "EventBus",
    "get_event_bus",
    "set_event_bus",
    "reset_event_bus",
    # Nectar Flow
    "NectarFlow",
    "NectarChannel",
    "NectarPriority",
    "WaggleDance",
    "DanceMessage",
    "DanceDirection",
    "PheromoneTrail",
    "PheromoneType",
    "PheromoneDecay",
    "RoyalJelly",
    "RoyalCommand",
    # Swarm Scheduler
    "SwarmScheduler",
    "SwarmConfig",
    "SwarmPolicy",
    "BeeBehavior",
    "ForagerBehavior",
    "NurseBehavior",
    "ScoutBehavior",
    "GuardBehavior",
    "HiveTask",
    "TaskPollen",
    "TaskNectar",
    "SwarmBalancer",
    "LoadDistribution",
    # Hive Memory
    "HiveMemory",
    "MemoryConfig",
    "CombStorage",
    "CombCell",
    "PollenCache",
    "HoneyArchive",
    "EvictionPolicy",
    "ReplicationPolicy",
    # CAMV Bridge
    "CAMVHoneycombBridge",
    "VentHoneycombAdapter",
    "CellToVCoreMapper",
    "GridToHypervisorMapper",
    "HexToCartesian",
    "CartesianToHex",
    # Resilience
    "HiveResilience",
    "CellFailover",
    "QueenSuccession",
    "HexRedundancy",
    "MirrorCell",
    "SwarmRecovery",
    "CombRepair",
    # Sandbox (Phase 7.4)
    "IsolationMode",
    "SandboxConfig",
    "SandboxedTaskRunner",
    "SandboxError",
    "SandboxTimeout",
    "SandboxCrashed",
    "SandboxNotSupported",
    # Observability (Phase 5.3) — re-exported from hoc.core.observability
    "configure_logging",
    "get_event_logger",
    "EVENT_LOGGER_NAME",
    # Metrics
    "HiveMetrics",
    "CellMetrics",
    "SwarmMetrics",
    "HoneycombVisualizer",
    "HeatmapRenderer",
    "FlowVisualizer",
]
