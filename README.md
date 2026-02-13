# HOC - Honeycomb Optimized Computing

**Computación Bio-Inspirada con Topología Hexagonal**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

- **Producción**: `numpy>=1.21.0`
- **Desarrollo**: `pytest`, `pytest-benchmark`, `pytest-cov`

## Uso rápido

```python
from hoc import (
    HoneycombGrid, HexCoord, NectarFlow, 
    SwarmScheduler, HiveMemory, HiveResilience,
    HiveMetrics, HoneycombVisualizer
)

# Crear grid hexagonal
grid = HoneycombGrid()
print(f"Grid creado con {grid.cell_count} celdas")

# Sistema de comunicación
nectar = NectarFlow(grid)

# Scheduler bio-inspirado
scheduler = SwarmScheduler(grid, nectar)

# Memoria distribuida
memory = HiveMemory(grid)

# Resiliencia
resilience = HiveResilience(grid)

# Métricas y visualización
metrics = HiveMetrics(grid)
viz = HoneycombVisualizer(grid)

# Tick del sistema
grid.tick()
nectar.tick()
scheduler.tick()
resilience.tick()
metrics.collect()
```

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

Resultados típicos (ejemplo):

| Operación | Tiempo medio |
|-----------|--------------|
| HexCoord creación | ~480 ns |
| Vecino hexagonal | ~546 ns |
| Distancia hex | ~267 ns |
| Depósito feromona | ~1.2 µs |
| NectarFlow tick | ~5.4 µs |
| Grid tick (r=2) | ~430 µs |

## Estructura del paquete

```
HOC/
├── __init__.py      # Exports principales
├── core.py          # Grid hexagonal (HoneycombGrid, HexCoord)
├── nectar.py        # Comunicación (feromonas, WaggleDance)
├── swarm.py         # Scheduler bio-inspirado
├── memory.py        # Memoria distribuida
├── bridge.py        # Integración CAMV (conversores, mapeos)
├── resilience.py    # Tolerancia a fallos
├── metrics.py       # Observabilidad y visualización
├── tests/           # Tests unitarios
├── benchmarks/      # Benchmarks de rendimiento
├── pyproject.toml   # Configuración del paquete
├── requirements.txt # Dependencias
└── README.md
```

## Módulos principales

### Core (`core.py`)
- **HexCoord**: Coordenadas axiales (q, r) y geometría hexagonal
- **HoneycombGrid**: Grid principal con QueenCell, WorkerCell, etc.
- **HexDirection**, **HexRing**: Navegación en topología hexagonal

### NectarFlow (`nectar.py`)
- **PheromoneTrail**: Feromonas digitales con decaimiento y difusión
- **WaggleDance**: Protocolo de danza (dirección, distancia, calidad)
- **RoyalJelly**: Canal de alta prioridad reina → colmena

### SwarmScheduler (`swarm.py`)
- **ForagerBehavior**, **NurseBehavior**, **ScoutBehavior**, **GuardBehavior**
- **SwarmBalancer**: Balanceo de carga con work-stealing

### HiveMemory (`memory.py`)
- **PollenCache** (L1), **CombStorage** (L2), **HoneyArchive** (L3)

### Bridge (`bridge.py`)
- **HexToCartesian**, **CartesianToHex**: Conversión de coordenadas
- **CAMVHoneycombBridge**: Bridge HOC ↔ CAMV
- **VentHoneycombAdapter**: Adaptador para entidades Vent

### Resilience (`resilience.py`)
- **HiveResilience**: Failover, sucesión de reina, recuperación

### Metrics (`metrics.py`)
- **HoneycombVisualizer**: Renderizado ASCII/SVG
- **HeatmapRenderer**, **FlowVisualizer**

## Especificación NectarFlow

Ver **`NECTAR_SPEC.md`** para la especificación detallada de feromona digital, protocolo Waggle Dance y difusión hexagonal.

## Características clave

| Característica | Descripción |
|---------------|-------------|
| **Topología Hexagonal** | 6 vecinos por celda, empaquetado óptimo |
| **Bio-Inspirado** | Feromonas, danzas, comportamientos de abejas |
| **Distribuido** | Memoria en 3 capas, replicación hexagonal |
| **Resiliente** | Failover automático, sucesión de reina |
| **Observable** | Métricas, visualización ASCII/SVG |
| **Integrable** | Bridge CAMV, adaptador Vent |

## Licencia

MIT License
