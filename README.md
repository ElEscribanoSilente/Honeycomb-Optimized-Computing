# HOC - Honeycomb Optimized Computing

**Computación Bio-Inspirada con Topología Hexagonal**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](CHANGELOG.md)

> ⚠️ **v2.0.0 BREAKING — async tick API.** Las cuatro tick methods top-level
> (`HoneycombGrid.tick`, `NectarFlow.tick`, `SwarmScheduler.tick`,
> `HoneycombCell.execute_tick`) son ahora `async def`. Callers v1.x deben
> migrar a `await` o usar los wrappers `run_tick_sync` (one-shot
> `DeprecationWarning`, removidos en v3.0). Ver [Migración v1 → v2](#migración-v1--v2)
> abajo y el [CHANGELOG](CHANGELOG.md) para detalle completo.

```
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
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Instalación

HOC es un paquete Python independiente. Instálalo con:

```bash
pip install -e .
```

O desde el directorio del proyecto con dependencias de desarrollo:

```bash
pip install -e ".[dev]"
```

### Dependencias

- **Producción**: `numpy>=1.21.0`, `mscs` (zero-deps secure
  serialization), `tramoya>=1.5.0` (FSM engine), `structlog>=25.0.0`
  (structured logging). `asyncio`, `sqlite3`, `zlib` son stdlib.
- **Desarrollo**: `pytest`, `pytest-asyncio`, `pytest-benchmark`,
  `pytest-cov`, `pytest-mock`, `pytest-timeout`, `hypothesis`, `ruff`,
  `black`, `mypy`, `bandit`, `pip-audit`, `radon`. Todas pinneadas
  en `requirements-dev.txt`.

### Extras opcionales

```bash
# JIT acceleration (Phase 7.6+ scaffold; full bridge deferred a Phase 9)
pip install -e ".[jit]"        # añade numba>=0.59

# Windows sandbox dependency (paired con Phase 7.4 stub)
pip install -e ".[sandbox-windows]"  # añade pywin32>=306
```

## Uso rápido (v2.0.0 async)

```python
import asyncio
from hoc import (
    HoneycombGrid, HexCoord, NectarFlow,
    SwarmScheduler, HiveMemory, HiveResilience,
    HiveMetrics, HoneycombVisualizer,
)

async def main():
    # Crear grid hexagonal
    grid = HoneycombGrid()
    print(f"Grid creado con {grid.cell_count} celdas")

    # Sistema de comunicación
    nectar = NectarFlow(grid)

    # Scheduler bio-inspirado (Phase 7.5 BehaviorIndex incluido)
    scheduler = SwarmScheduler(grid, nectar)

    # Memoria distribuida (Phase 6 storage backends)
    memory = HiveMemory(grid)

    # Resiliencia
    resilience = HiveResilience(grid)

    # Métricas y visualización
    metrics = HiveMetrics(grid)
    viz = HoneycombVisualizer(grid)

    # Tick del sistema — todas las clases user-facing son async desde v2.0.0
    await grid.tick()
    await nectar.tick()
    await scheduler.tick()

    # HiveResilience y HiveMemory.tick siguen sync (no son user-facing tick paths)
    resilience.tick()
    memory.tick()
    metrics.collect()

    # Múltiples grids en paralelo — el patrón canónico Phase 7+
    grid_a, grid_b = HoneycombGrid(), HoneycombGrid()
    await asyncio.gather(grid_a.tick(), grid_b.tick())

asyncio.run(main())
```

### Sandbox opcional (Phase 7.4)

```python
from hoc import SandboxedTaskRunner, SandboxConfig, SandboxCrashed, SandboxTimeout

# POSIX-only en v1; Windows raises SandboxNotSupported (deferred a Phase 7.x)
runner = SandboxedTaskRunner(SandboxConfig(timeout_s=5.0, isolation="process"))

try:
    result = runner.run(my_payload, arg1, kw=val)
except SandboxTimeout:
    ...  # tarea excedió el timeout; el panal sigue corriendo
except SandboxCrashed as exc:
    ...  # crash (SIGSEGV, OOM, exit non-zero); detalle en str(exc)
```

### Migración v1 → v2

```python
# v1.x (Phase 6 y anterior)
grid.tick()
nectar.tick()
scheduler.tick()
cell.execute_tick()

# v2.0+ canónico
await grid.tick()
await nectar.tick()
await scheduler.tick()
await cell.execute_tick()

# v2.0+ legacy bridge (DeprecationWarning una sola vez; removido en v3.0)
grid.run_tick_sync()
nectar.run_tick_sync()
scheduler.run_tick_sync()
cell.run_execute_tick_sync()
```

Los wrappers `run_tick_sync` rechazan ser llamados desde un event loop
activo (`RuntimeError`); úsalos sólo para callers sync legacy. El path
canónico Phase 7+ es siempre `await`.

## Tests

Ejecuta la suite de tests:

```bash
pytest tests/ -v
```

Con cobertura:

```bash
pytest tests/ -v --cov=hoc --cov-report=html
```

## Benchmarks

Ejecuta los benchmarks de rendimiento (requiere `pytest-benchmark`):

```bash
pytest benchmarks/ -v --benchmark-only
```

**Trabajo pesado (mini render 3D):** prueba el SwarmScheduler con una carga CPU intensiva (raycasting NumPy):

```bash
python -m benchmarks.bench_swarm_render
```

**Benchmark mixto de tareas pesadas:** varios tipos de carga (render, matrices, simulación, hash, Monte Carlo, tareas matemáticas complejas: autovalores, FFT, integración, sistemas lineales, raíces de polinomios):

```bash
python -m benchmarks.bench_heavy_mixed
```

**Tests pesados** (tareas por tipo, mixtos, estrés):

```bash
pytest tests/test_heavy.py -v
# o sin pytest (si fallan plugins):
python -m tests.test_heavy
```

Los análisis de resultados están en `benchmarks/ANALISIS_RENDER.md` y `benchmarks/ANALISIS_BENCHMARK_PESADOS.md`.

Resultados típicos post-Phase-7 (ejemplo, Windows + Python 3.13):

| Operación | Tiempo medio | Δ vs Phase 6 |
|-----------|--------------|--------------|
| HexCoord creación | ~480 ns | = |
| Vecino hexagonal | ~546 ns | = |
| Distancia hex | ~267 ns | = |
| Depósito feromona | ~1.2 µs | = |
| Grid creation (r=2) | ~600 µs | = |
| **Swarm 1000 tasks single tick (r=3)** | **~1.7 ms** | **≈ 6× speedup** ✅ |
| Swarm 1000 tasks drain 25 ticks (r=3) | ~32 ms | new |
| NectarFlow tick | ~12 µs | event-loop overhead |
| Grid tick (r=2) | ~1.1 ms | event-loop overhead |

> **Nota sobre los grid_tick / nectar tick deltas vs Phase 6**: el
> overhead añadido viene de `asyncio.gather` + `asyncio.to_thread`.
> En workloads pequeños (radius 1-2) eso suma vs el `ThreadPoolExecutor`
> directo. La ganancia paga al subir cell count o concurrencia entre
> grids — el bench `swarm_1000_tasks` (radius=3 + 1000 tasks) hits el
> target ≥ 5× del brief al combinarse con `BehaviorIndex` (Phase 7.5).
> Ver [`docs/perf/baseline_v2.md`](docs/perf/baseline_v2.md) para la
> narrativa completa.

### Profiling

```bash
# Pretty flame graph via py-spy (script imprime el comando exacto)
python scripts/profile_grid.py --radius 3 --ticks 200

# Self-contained: cProfile sin py-spy
python scripts/profile_grid.py --inproc --radius 3 --ticks 200
```

Guía completa: [`docs/perf/profiling.md`](docs/perf/profiling.md).

## Estructura del paquete

```
HOC/                                   # repo root = paquete `hoc`
├── __init__.py                        # Exports principales
├── nectar.py                          # Comunicación (feromonas, WaggleDance, RoyalJelly)
├── swarm.py                           # Scheduler bio-inspirado + BehaviorIndex
├── sandbox.py                         # Phase 7.4: process isolation opt-in
├── memory.py                          # Memoria distribuida
├── resilience.py                      # Tolerancia a fallos
├── security.py                        # Boundary única para mscs (HMAC + serialización)
├── core/                              # Phase 3.3: split de core.py (3,615 LOC) en 14 submódulos
│   ├── grid.py                        #   HoneycombGrid (async tick desde Phase 7)
│   ├── grid_geometry.py               #   HexCoord, HexDirection, HexPathfinder
│   ├── grid_config.py                 #   HoneycombConfig (validation + checkpoint config)
│   ├── cells_base.py                  #   HoneycombCell (async execute_tick)
│   ├── cells_specialized.py           #   Worker/Drone/Nursery/Storage/Guard/Scout
│   ├── _queen.py                      #   QueenCell
│   ├── events.py                      #   EventBus
│   ├── health.py                      #   CircuitBreaker + HealthMonitor
│   ├── locking.py                     #   RWLock
│   ├── observability.py               #   Phase 5.3: structlog boundary
│   └── pheromone.py                   #   PheromoneField (numpy SIMD desde Phase 7.6)
├── metrics/                           # Phase 3.3: split de metrics.py
├── bridge/                            # Phase 6.5: split de bridge.py (886 LOC)
│   ├── converters.py                  #   HexToCartesian / CartesianToHex
│   ├── mappers.py                     #   CellToVCoreMapper / GridToHypervisorMapper
│   └── adapters.py                    #   CAMVHoneycombBridge / VentHoneycombAdapter
├── storage/                           # Phase 6: persistencia pluggable
│   ├── base.py                        #   StorageBackend Protocol + MemoryBackend
│   ├── sqlite.py                      #   SQLiteBackend (WAL + schema versioning)
│   └── checkpoint.py                  #   encode_blob/decode_blob (HMAC + zlib + mscs)
├── state_machines/                    # Phase 4: tramoya FSMs
│   ├── base.py                        #   HocStateMachine + IllegalStateTransition
│   ├── cell_fsm.py                    #   CellState
│   ├── task_fsm.py                    #   TaskLifecycle
│   ├── pheromone_fsm.py
│   ├── failover_fsm.py
│   ├── succession_fsm.py
│   └── reified.py                     #   @transition decorator (Phase 4.2)
├── choreo/                            # Phase 4.1: static FSM checker
├── tests/                             # 1062 tests (8 skipped en Windows)
├── benchmarks/                        # pytest-benchmark suites
├── docs/                              # ADRs (18) + perf docs + state-machines.md
├── snapshot/                          # Auditorías por phase + bench baselines
├── scripts/                           # Tooling (profile_grid.py, compare_bench.py, ...)
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Módulos principales

### Core (`hoc.core`)
- **HoneycombGrid**: Grid principal. `async def tick()` desde v2.0.0;
  `run_tick_sync()` wrapper para callers sync legacy.
- **HoneycombCell**: Celda base. `async def execute_tick()` desde
  v2.0.0; class-level shared FSM (`_CLASS_FSM` ClassVar) introducido
  en Phase 6.6.
- **HexCoord**, **HexDirection**, **HexRing**, **HexPathfinder**:
  Coordenadas axiales (q, r) y geometría hexagonal.
- **EventBus**: Pub/sub para eventos del grid.
- **HealthMonitor**, **CircuitBreaker**: Auto-recovery (Phase 1+).
- **PheromoneField**: Feromonas internas por celda (numpy-vectorisado
  cuando hay 4+ deposits, Phase 7.6).

### NectarFlow (`hoc.nectar`)
- **PheromoneTrail**: Feromonas digitales con decaimiento y difusión.
- **WaggleDance**: Protocolo de danza (dirección, distancia, calidad).
- **RoyalJelly**: Canal de alta prioridad reina → colmena (HMAC-firmado
  desde Phase 2; Queen-only enforcement en priority ≥ 8).
- `async def tick()` desde v2.0.0; `run_tick_sync()` wrapper.

### SwarmScheduler (`hoc.swarm`)
- **HiveTask**: Dataclass de tarea con FSM wired (Phase 4.1) +
  `to_dict`/`from_dict` (Phase 7.10).
- **ForagerBehavior** / **NurseBehavior** / **ScoutBehavior** /
  **GuardBehavior**: 4 comportamientos de abeja.
- **BehaviorIndex** (Phase 7.5): per-behaviour priority heap;
  `pop_best` reduce el filter loop a O(m·log n).
- **SwarmBalancer**: Balanceo de carga con work-stealing.
- **SwarmConfig**: incluye `queue_full_policy` (raise/drop_oldest/
  drop_newest/block) desde Phase 7.3.
- `async def tick()` desde v2.0.0; `run_tick_sync()` wrapper.

### Sandbox (`hoc.sandbox`) ← Phase 7.4
- **SandboxedTaskRunner**: process isolation opt-in con timeout duro.
- **SandboxConfig**: `isolation: "none" | "process" | "cgroup" | "job_object"`.
  POSIX-only en v1 para `"process"` (fork-based); Windows raises
  `SandboxNotSupported`. cgroup / job_object stubbed.
- **SandboxError** hierarchy: `SandboxTimeout`, `SandboxCrashed`,
  `SandboxNotSupported`.

### Storage (`hoc.storage`) ← Phase 6
- **StorageBackend** Protocol: 5 métodos (put/get/delete/keys/__contains__).
- **MemoryBackend** (default): thread-safe dict wrapper.
- **SQLiteBackend**: WAL + schema versioning + connection-per-thread,
  stdlib only.
- **encode_blob** / **decode_blob**: checkpoint format con HMAC +
  zlib + mscs strict. Wire format `[version (1B) | hmac (32B) |
  flag (1B) | payload]`. v1 (Phase 6) y v2 (Phase 7.10) coexisten.

### HiveMemory (`hoc.memory`)
- **PollenCache** (L1, in-memory hot cache).
- **CombStorage** (L2, distributed across cells).
- **HoneyArchive** (L3, persistent — usa `StorageBackend` desde Phase 6).

### Bridge (`hoc.bridge`)
- **HexToCartesian**, **CartesianToHex**: Conversión de coordenadas.
- **CellToVCoreMapper**, **GridToHypervisorMapper**: Mapeos CAMV.
- **CAMVHoneycombBridge**: Bridge HOC ↔ CAMV.
- **VentHoneycombAdapter**: Adaptador para entidades Vent.

### Resilience (`hoc.resilience`)
- **HiveResilience**: Failover, sucesión de reina, recuperación.
- **CellFailover** + **QueenSuccession** (Raft-like, votos firmados,
  desde Phase 2): FSMs wired desde Phase 5.
- **HexRedundancy**, **MirrorCell**, **SwarmRecovery**, **CombRepair**.

### Metrics (`hoc.metrics`)
- **HoneycombVisualizer**: Renderizado ASCII/SVG.
- **HeatmapRenderer**, **FlowVisualizer**.
- **HiveMetrics**, **CellMetrics**, **SwarmMetrics**.

## Decisiones de arquitectura

Las decisiones de diseño están documentadas en
[`docs/adr/`](docs/adr/) (18 ADRs al cierre de Phase 7):

- **ADR-001 a 005**: Phase 1-2 (security, mscs, HMAC, quorum).
- **ADR-006**: Strict mypy graduación schedule.
- **ADR-007 a 010**: Phase 4 (tramoya, choreo, reified transitions,
  dead enum cleanup).
- **ADR-011 a 012**: Phase 5 (observability stack, choreo --strict).
- **ADR-013 a 015**: Phase 6 (storage backend, checkpoint format,
  class-level FSM).
- **ADR-016**: Async tick loop (Phase 7.1+7.2).
- **ADR-017**: Sandboxing model (Phase 7.4).
- **ADR-018**: BehaviorIndex perf optimisation (Phase 7.5).

## Especificación NectarFlow

Ver **`NECTAR_SPEC.md`** para la especificación detallada de feromona digital, protocolo Waggle Dance y difusión hexagonal.

## Características clave

| Característica | Descripción |
|---------------|-------------|
| **Topología Hexagonal** | 6 vecinos por celda, empaquetado óptimo |
| **Bio-Inspirado** | Feromonas, danzas, comportamientos de abejas |
| **Async desde v2.0.0** | `await grid.tick()` canónico; gather composable cross-grids |
| **Persistencia** | Storage backends pluggables (Memory/SQLite); checkpoints HMAC-signed |
| **Sandboxing opt-in** | Process isolation con timeout duro (POSIX); crashes contenidos |
| **Distribuido** | Memoria en 3 capas, replicación hexagonal |
| **Resiliente** | Failover automático, sucesión Raft-like, FSMs formales |
| **Observable** | structlog estructurado, métricas, visualización ASCII/SVG |
| **Seguro** | mscs en lugar de pickle, HMAC-SHA256 en mensajes, bandit 0/0/0 |
| **Integrable** | Bridge CAMV, adaptador Vent |

## Roadmap

10-phase stabilization roadmap (v1.0.0 → v3.0.0). Estado al
2026-04-28:

- Phases 1-7: **CERRADAS** (v1.1.0 → v2.0.0).
- Phase 8: Multi-nodo distribuido (próximo).
- Phase 9: GPU + Rust extensions vía PyO3 + Cython.
- Phase 10: AI/ML + research output (v3.0.0).

Detalle completo + dependencias entre phases: [`ROADMAP.md`](ROADMAP.md).

## Licencia

MIT License
