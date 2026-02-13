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

from .core import (
    # Grid principal
    HoneycombGrid,
    HoneycombConfig,
    GridTopology,

    # Tipos de celdas
    HoneycombCell,
    CellState,
    CellRole,
    QueenCell,
    WorkerCell,
    DroneCell,
    NurseryCell,

    # Coordenadas hexagonales
    HexCoord,
    HexDirection,
    HexRing,

    # Event bus management (v3.1)
    EventBus,
    get_event_bus,
    set_event_bus,
    reset_event_bus,
)

# ═══════════════════════════════════════════════════════════════════════════════
# NECTAR FLOW - Sistema de comunicación
# ═══════════════════════════════════════════════════════════════════════════════

from .nectar import (
    # Flujo principal
    NectarFlow,
    NectarChannel,
    NectarPriority,
    
    # Protocolos de comunicación
    WaggleDance,
    DanceMessage,
    DanceDirection,
    
    # Feromonas
    PheromoneTrail,
    PheromoneType,
    PheromoneDecay,
    
    # Canal de alta prioridad
    RoyalJelly,
    RoyalCommand,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SWARM SCHEDULER - Scheduling bio-inspirado
# ═══════════════════════════════════════════════════════════════════════════════

from .swarm import (
    # Scheduler principal
    SwarmScheduler,
    SwarmConfig,
    SwarmPolicy,
    
    # Comportamientos
    BeeBehavior,
    ForagerBehavior,
    NurseBehavior,
    ScoutBehavior,
    GuardBehavior,
    
    # Tareas
    HiveTask,
    TaskPollen,
    TaskNectar,
    
    # Balanceo
    SwarmBalancer,
    LoadDistribution,
)

# ═══════════════════════════════════════════════════════════════════════════════
# HIVE MEMORY - Sistema de memoria distribuida
# ═══════════════════════════════════════════════════════════════════════════════

from .memory import (
    # Memoria principal
    HiveMemory,
    MemoryConfig,
    
    # Capas de almacenamiento
    CombStorage,
    CombCell,
    PollenCache,
    HoneyArchive,
    
    # Políticas
    EvictionPolicy,
    ReplicationPolicy,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CAMV BRIDGE - Integración con CAMV
# ═══════════════════════════════════════════════════════════════════════════════

from .bridge import (
    # Adaptadores
    CAMVHoneycombBridge,
    VentHoneycombAdapter,
    
    # Mapeos
    CellToVCoreMapper,
    GridToHypervisorMapper,
    
    # Conversores
    HexToCartesian,
    CartesianToHex,
)

# ═══════════════════════════════════════════════════════════════════════════════
# RESILIENCE - Sistema de resiliencia
# ═══════════════════════════════════════════════════════════════════════════════

from .resilience import (
    # Tolerancia a fallos
    HiveResilience,
    CellFailover,
    QueenSuccession,
    
    # Replicación
    HexRedundancy,
    MirrorCell,
    
    # Recuperación
    SwarmRecovery,
    CombRepair,
)

# ═══════════════════════════════════════════════════════════════════════════════
# METRICS - Observabilidad
# ═══════════════════════════════════════════════════════════════════════════════

from .metrics import (
    # Métricas
    HiveMetrics,
    CellMetrics,
    SwarmMetrics,
    
    # Visualización
    HoneycombVisualizer,
    HeatmapRenderer,
    FlowVisualizer,
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
    
    # Metrics
    "HiveMetrics",
    "CellMetrics",
    "SwarmMetrics",
    "HoneycombVisualizer",
    "HeatmapRenderer",
    "FlowVisualizer",
]
