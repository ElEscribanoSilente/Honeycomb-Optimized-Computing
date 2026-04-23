# HOC Roadmap — 10 Fases

**Versión inicial**: v1.0.0 (tag `v1.0.0-baseline`, ver [snapshot/SNAPSHOT.md](snapshot/SNAPSHOT.md))
**Versión objetivo**: v3.0.0
**Duración estimada**: ~12-18 meses (asumiendo 1-2 fases/mes)

---

## Principios del roadmap

1. **Nada se rompe**: cada fase preserva backwards-compat o documenta breaking changes.
2. **Tests primero**: ningún cambio entra a `main` sin tests + auditorías que pasen.
3. **Cada fase entrega valor**: aunque las siguientes no se completen, lo entregado mejora el proyecto.
4. **Auditorías por fase**: seguridad + calidad general + cobertura de tests.
5. **Rollback siempre posible**: tag por fase + branch baseline preservada.

---

## Pipeline estándar por fase

Toda fase incluye **al cierre**:
- ✅ **Auditoría de seguridad** (mismo formato que la baseline; comparar deltas)
- ✅ **Auditoría general** (arquitectura, calidad, performance, deuda técnica)
- ✅ **Tests**: unit (≥80% módulos tocados) + integration + property-based (Hypothesis donde aplique)
- ✅ **Benchmarks**: regresión vs fase anterior (no degradar throughput >5%)
- ✅ **CHANGELOG.md** actualizado (a partir de Fase 3)
- ✅ **Tag git**: `v{X.Y.Z}-phase{NN}`
- ✅ **Documentación**: README + docs específicos de la fase

---

# 📋 FASES

---

## FASE 1 — Estabilización crítica ✅ CERRADA (2026-04-22)
**Objetivo**: Cero bugs críticos, cobertura mínima sólida.
**Duración real**: 1 sesión
**Tag**: `v1.1.0-phase01`
**Cierre**: ver [snapshot/PHASE_01_CLOSURE.md](snapshot/PHASE_01_CLOSURE.md)

**Resultado**: 378 tests pasando (340 nuevos), cobertura 83–95% en los 4
módulos previamente sin tests, 0 bandit HIGH, 0 vulns de dependencias. Se
corrigieron los 7 bugs B1–B5/B7/B8 más 3 bugs adicionales descubiertos
durante el testing (B2.5 leak, B9 attribute error, B10 cached_property+slots).
B6 (TOCTOU load) diferido como no bloqueante.

### Bugs a corregir
| ID | Severidad | Ubicación | Fix |
|----|-----------|-----------|-----|
| B1 | 🔴 Crítica | [core.py:518-521](core.py) | Race en `RWLock`: `try/finally` correcto, eliminar `bare except` |
| B2 | 🔴 Crítica | [swarm.py:943-1020](swarm.py) | TOCTOU en `SwarmScheduler.tick()`: extender lock |
| B3 | 🔴 Crítica | [nectar.py:314-352](nectar.py) | Validar `decay_rate` y `diffusion_rate` en `__init__` |
| B4 | 🟠 Alta | [resilience.py:530](resilience.py) | Quorum vinculante: `raise ValueError` si no hay mayoría |
| B5 | 🟠 Alta | [memory.py:223-229](memory.py) | Resta de bytes ANTES del check de capacidad |
| B6 | 🟠 Alta | [swarm.py:732-738](swarm.py), [resilience.py:279](resilience.py) | TOCTOU en load: copiar bajo lock |
| B7 | 🟡 Media | [metrics.py:169](metrics.py) | Histogram bounds + `sum == count` invariant |
| B8 | 🟡 Media | [resilience.py:1203](resilience.py) | `try/except KeyError` en `HexDirection[...]` |

### Tests faltantes
- [ ] `tests/test_memory.py` — cobertura ≥80% de `memory.py`
- [ ] `tests/test_resilience.py` — cobertura ≥80% de `resilience.py`
- [ ] `tests/test_swarm.py` — task queue, behavior selection, work stealing
- [ ] `tests/test_metrics.py` — collectors + edge cases en histogramas
- [ ] **Hypothesis**: invariantes de `HexCoord` (rotación, distancia simétrica)
- [ ] **Hypothesis**: invariantes de `PheromoneTrail` (decaimiento monótono)

### Auditorías al cierre
- 🔒 Security audit: confirmar 0 críticos, regresión vs baseline
- 🏛️ General audit: cobertura objetivo ≥75% global, ≥80% módulos críticos

### Definition of Done
- Todos los bugs críticos/altos cerrados o documentados como "won't fix" con razón
- Cobertura ≥75% global (medida con `pytest-cov`)
- CI básico corriendo (placeholder hasta Fase 3)

---

## FASE 2 — Seguridad & Hardening ✅ CERRADA (2026-04-23)
**Objetivo**: Eliminar vectores críticos de seguridad. **Aquí integramos `mscs`** para reemplazar pickle.
**Duración real**: 1 sesión
**Tag**: `v1.2.0-phase02`
**Cierre**: ver [snapshot/PHASE_02_CLOSURE.md](snapshot/PHASE_02_CLOSURE.md)

**Resultado**: 421 tests pasando (43 nuevos de seguridad), cobertura mejorada
en `nectar.py` (62% → 72%) y agregado `security.py` (83%). 0 usos de
`pickle` en producción, 0 usos de `random.random()` sensibles. Bandit 100%
limpio (0 HIGH, 0 MEDIUM, 0 LOW vs 3 MEDIUM + 4 LOW previos). pip-audit
limpio. HMAC-SHA256 sobre `DanceMessage`/`RoyalMessage`/`PheromoneDeposit`;
Queen-only enforcement en `RoyalCommand` priority ≥ 8; protocolo Raft-like
con `term_number` monotónico y votos firmados en `QueenSuccession`. Rate
limiting en `submit_task`/`execute_on_cell`. Path validation en
`HoneyArchive` (incluye fix `/tmp/honey` → `tempfile.gettempdir()`).
Bounded growth en `PheromoneTrail._deposits` (OrderedDict + LRU). Overhead
end-to-end medido: +3.5% sobre baseline (target <5%).

### Cambios estructurales

#### 2.1 Reemplazo de pickle con [mscs](https://pypi.org/project/mscs/) (autor: @Esraderey)
**Por qué mscs**: librería propia, MIT, zero deps, registry de clases (no ejecuta código arbitrario), HMAC-SHA256 nativo, soporte NumPy/PyTorch.

| Ubicación actual | Cambio |
|------------------|--------|
| [memory.py:495,640](memory.py) `pickle.loads()` | `mscs.deserialize()` con registry whitelist |
| [memory.py:199,435,607](memory.py) `pickle.dumps()` | `mscs.serialize()` con HMAC-SHA256 |
| [resilience.py:1159](resilience.py) | Validación vía `mscs` schema |

```python
# Ejemplo de uso planificado
import mscs
registry = mscs.Registry()
registry.register(HoneycombCell)
registry.register(PheromoneDeposit)

serializer = mscs.Serializer(registry, hmac_key=config.hmac_secret)
blob = serializer.dumps(cell_data)  # firmado
cell = serializer.loads(blob)  # solo reconstruye clases registradas
```

#### 2.2 Autenticación en NectarFlow / RoyalJelly
- HMAC-SHA256 en cada `DanceMessage`, `RoyalCommand`, `PheromoneDeposit`
- Solo `QueenCell` puede emitir `RoyalCommand` con `priority >= 8`
- Verificación criptográfica de origen vía `cell_id` firmado
- Reutiliza el HMAC de mscs (consistencia)

#### 2.3 Quorum criptográficamente vinculante en QueenSuccession
- Protocolo tipo Raft con `term_number` monotónico
- Votos firmados; verificación de identidad antes de contar
- Rechazo si no se alcanza mayoría real

#### 2.4 Otros hardenings
- [ ] `random.random()` → `secrets.SystemRandom()` en decisiones sensibles ([swarm.py:212](swarm.py), [memory.py:269](memory.py))
- [ ] Path validation con `pathlib.Path.resolve()` en `HoneyArchive` ([memory.py:571](memory.py))
- [ ] Rate limiting en `submit_task()` y `execute_on_cell()` (decorator `@rate_limit(per_second=N)`)
- [ ] Sanitize logs (no exception details en producción; debug-only)
- [ ] Bounded growth en `PheromoneTrail._deposits` (max_per_coord + LRU)

### Tests específicos de seguridad
- [ ] Test de payload pickle malicioso → mscs lo rechaza
- [ ] Test de RoyalCommand forjado por DroneCell → rechazado por HMAC
- [ ] Test de QueenSuccession con votos duplicados → rechazado
- [ ] Test de DoS por feromonas: 10K deposits → memoria bounded
- [ ] Test de path traversal en HoneyArchive

### Auditorías al cierre
- 🔒 Security: 0 críticos, 0 altos. Pen-test interno básico.
- 🏛️ General: revisar overhead de HMAC (<5% throughput).

---

## FASE 3 — Tooling, CI/CD & Code Quality
**Objetivo**: Higiene de proyecto OSS-grade.
**Duración**: 2 semanas
**Tag al cerrar**: `v1.3.0-phase03`

### Tooling
- [ ] `ruff` (replace flake8/isort/pyupgrade) — config en `pyproject.toml`
- [ ] `black` — line length 100
- [ ] `mypy --strict` con `[tool.mypy]` en `pyproject.toml`
- [ ] `pre-commit` con: ruff, black, mypy, trailing-whitespace, check-yaml
- [ ] `pytest-cov` con threshold ≥80%
- [ ] Pin versions en `requirements-dev.txt`

### CI/CD
- [ ] `.github/workflows/test.yml`: matriz Python 3.10/3.11/3.12 × Linux/macOS/Windows
- [ ] `.github/workflows/lint.yml`: ruff + mypy + black --check
- [ ] `.github/workflows/security.yml`: `bandit`, `pip-audit`, `safety`
- [ ] `.github/workflows/docs.yml`: build sphinx en cada PR
- [ ] `.github/workflows/release.yml`: auto-publish a PyPI en tag

### Refactor de código
- [ ] Dividir `core.py` (3.624 LOC):
  - `core/grid.py` — HexCoord, HoneycombGrid
  - `core/cells.py` — HoneycombCell + subtipos
  - `core/events.py` — EventBus
  - `core/health.py` — HealthMonitor, CircuitBreaker
  - `core/locking.py` — RWLock
- [ ] Mover `MetricsCollector` de `core.py` → `metrics.py`
- [ ] Dividir `metrics.py` (1.176 LOC):
  - `metrics/collection.py`
  - `metrics/visualization.py`
  - `metrics/rendering.py` (ASCII/SVG/HTML)
- [ ] Magic numbers → constantes nombradas en `core/constants.py`

### Documentación de proyecto
- [ ] `CHANGELOG.md` con historial v0.1 → v1.3
- [ ] `CONTRIBUTING.md`
- [ ] `CODE_OF_CONDUCT.md`
- [ ] `SECURITY.md` (vulnerability reporting)
- [ ] `docs/adr/` con ADRs retroactivos (decisión de topología hex, mscs vs pickle, etc.)

### Auditorías al cierre
- 🔒 Security: pip-audit + bandit limpios
- 🏛️ General: complejidad ciclomática <10 por función, sin módulos >800 LOC

---

## FASE 4 — Configuración & Developer Experience
**Objetivo**: HOC usable sin necesidad de leer código fuente. **Integración profunda de [tramoya](https://pypi.org/project/tramoya/)** para state machines formales.
**Duración**: 3-4 semanas
**Tag al cerrar**: `v1.4.0-phase04`

### Integración de tramoya
**Por qué tramoya**: librería propia, MIT, zero deps, FSM con guards/hooks, undo, viz Mermaid/Graphviz.

State machines a formalizar:

| Dominio | Estados | Por qué tramoya |
|---------|---------|-----------------|
| **CellState** ([core.py](core.py)) | `INITIALIZING → IDLE → BUSY → DEGRADED → FAILED → RECOVERING → IDLE` | Guards previenen transiciones ilegales (e.g., FAILED→BUSY) |
| **TaskLifecycle** ([swarm.py](swarm.py)) | `PENDING → CLAIMED → RUNNING → COMPLETED \| FAILED \| CANCELLED` | Hooks para metrics; undo para retry |
| **QueenSuccession** ([resilience.py](resilience.py)) | `STABLE → DETECTING → NOMINATING → VOTING → ELECTED \| FAILED` | Guards = quorum criptográfico (refuerza fix de Fase 1) |
| **PheromoneDeposit** ([nectar.py](nectar.py)) | `FRESH → DECAYING → DIFFUSING → EVAPORATED` | Hooks para evicción automática |
| **FailoverFlow** ([resilience.py](resilience.py)) | `HEALTHY → DEGRADED → MIGRATING → RECOVERED \| LOST` | Undo para rollback de migración fallida |

Bonus: **exportar TODOS los state machines a Mermaid** para `docs/state-machines.md` (auto-generado).

```python
# Ejemplo
from tramoya import StateMachine

cell_fsm = StateMachine(
    states=["initializing", "idle", "busy", "degraded", "failed", "recovering"],
    transitions=[
        ("initializing", "idle", lambda ctx: ctx["health"] > 0.9),
        ("idle", "busy", lambda ctx: ctx["task_count"] > 0),
        ("busy", "degraded", lambda ctx: ctx["health"] < 0.5),
        ("degraded", "failed", lambda ctx: ctx["health"] < 0.1),
        ("failed", "recovering", lambda ctx: ctx["queen_signal"] == "recover"),
        ("recovering", "idle", lambda ctx: ctx["health"] > 0.8),
    ],
    on_enter={"failed": notify_queen, "recovering": start_recovery_protocol},
)
```

### Configuración externa
- [ ] `HoneycombConfig.from_env()` — `HOC_RADIUS=3 HOC_VCORES=8 ...`
- [ ] `HoneycombConfig.from_yaml(path)` — config files
- [ ] `HoneycombConfig.from_toml(path)`
- [ ] Validación con `pydantic` o equivalente (mscs schemas si soporta)

### CLI tool
```bash
hoc init my-hive          # crea proyecto template
hoc run workload.py       # ejecuta con HOC
hoc bench                 # benchmarks suite
hoc inspect <cell-id>     # introspección de célula
hoc doctor                # diagnóstico de salud
hoc viz                   # exporta Mermaid de state machines
hoc dash                  # arranca dashboard (preview de Fase 5)
```
- Implementación con `click` o `typer`

### Plugin system
- Custom behaviors vía `entry_points`:
```toml
# en proyecto usuario
[project.entry-points."hoc.behaviors"]
architect = "my_pkg.behaviors:ArchitectBehavior"
```

### Documentación
- [ ] Sphinx + ReadTheDocs config
- [ ] Tutoriales Jupyter en `examples/`:
  - `01_quickstart.ipynb`
  - `02_custom_behaviors.ipynb`
  - `03_distributed_workload.ipynb`
- [ ] Type stubs (`.pyi`) publicados

### Auditorías al cierre
- 🔒 Security: validar que CLI no permite injection
- 🏛️ General: docs coverage ≥90% public API

---

## FASE 5 — Observabilidad
**Objetivo**: Visibilidad operacional production-grade. **Integración de [trame](https://kitware.github.io/trame/)** (asumiendo que "tramoya" del usuario era trame para dashboards) — opcional, ver nota.
**Duración**: 3 semanas
**Tag al cerrar**: `v1.5.0-phase05`

> **Nota**: tramoya (state machines) ya se usa en Fase 4. Para el dashboard interactivo proponemos **trame** (Kitware). Si prefieres otro stack (Streamlit/Plotly Dash/FastAPI+HTMX), ajustar aquí.

### OpenTelemetry
- [ ] Traces: cada `HiveTask` = root span; cada behavior = child span
- [ ] Metrics: throughput, latencia p50/p95/p99, queue depth, pheromone density
- [ ] Logs: correlation con trace_id
- [ ] Exporters: OTLP, Jaeger, Tempo

### Logs estructurados
- [ ] `structlog` con JSON output
- [ ] Context-binding por cell_id, task_id

### Dashboard web (con trame)
- [ ] Visualización 2D del panal en tiempo real (heatmap de feromonas)
- [ ] Panel de estado por celda (state machine viz, carga, tareas)
- [ ] Gráfico de danzas activas (vector arrows)
- [ ] Timeline de eventos (failover, succession)
- [ ] Controles: pause/resume tick, inject task, kill cell (chaos)

### Prometheus + Grafana
- [ ] `/metrics` endpoint Prometheus-compatible
- [ ] Grafana dashboards templates (JSON exports en `docs/grafana/`)

### Auditorías al cierre
- 🔒 Security: dashboard tras auth (mTLS opcional)
- 🏛️ General: overhead observabilidad <3% throughput

---

## FASE 6 — Persistencia & Storage
**Objetivo**: HoneyArchive con backends reales; recovery completo tras crash.
**Duración**: 3-4 semanas
**Tag al cerrar**: `v1.6.0-phase06`

### Backends pluggables
| Backend | Caso de uso | Dependencia |
|---------|-------------|-------------|
| `MemoryBackend` (actual) | Tests, dev rápido | — |
| `SQLiteBackend` | Single-node prod, dev portable | stdlib |
| `LMDBBackend` | High-throughput local | `lmdb` |
| `RocksDBBackend` | TB-scale local | `python-rocksdb` |
| `S3Backend` | Cloud-native | `boto3` |
| `RedisBackend` | Distributed cache | `redis-py` |

### Serialización (consolidación)
- Toda serialización vía **mscs** (resultado de Fase 2)
- Schemas versionados; migration utilities

### Checkpointing
- [ ] Snapshot periódico de estado completo (config: `checkpoint_interval`)
- [ ] Snapshot incremental (delta vs último checkpoint)
- [ ] Compresión opcional (`zstd`)

### Recovery
- [ ] `HoneycombGrid.restore_from_checkpoint(path)` reconstruye estado
- [ ] Validación de integridad (HMAC + schema)
- [ ] Replay de eventos desde el último checkpoint

### Tests específicos
- [ ] Crash simulado durante `tick()` → recovery preserva tareas en queue
- [ ] Corrupción de checkpoint → fallback al previo
- [ ] Migration entre versiones de schema

### Auditorías al cierre
- 🔒 Security: backends no exponen datos en transit (TLS donde aplique)
- 🏛️ General: backup/restore tested end-to-end

---

## FASE 7 — Async/Await & Performance
**Objetivo**: Throughput 5-10× mejor; sandboxing real de tareas.
**Duración**: 4-5 semanas
**Tag al cerrar**: `v2.0.0-phase07` ⚠️ **breaking change** (API async)

### Async migration
- [ ] `SwarmScheduler.tick()` → `async def tick()`
- [ ] `NectarFlow.tick()` → async
- [ ] `HoneycombGrid.tick()` → async
- [ ] Backwards-compat: wrapper sync `run_tick_sync()` para usuarios legacy
- [ ] Tests con `pytest-asyncio`

### Sandboxing de tareas
- [ ] Aislamiento opcional vía `multiprocessing.Pool` con timeouts duros
- [ ] Linux: `cgroups` para CPU/memory limits
- [ ] Windows: Job Objects equivalentes
- [ ] Crash en tarea NO mata el panal

### Performance hotspots
- [ ] Optimizar `SwarmScheduler.tick()`: O(n·m) → O(n log n) con índices por behavior
- [ ] Vectorización SIMD adicional (NumPy + posible `numba` JIT)
- [ ] Lock-free data structures donde aplique (`atomic` types)
- [ ] Backpressure: bounded queues con drop policy
- [ ] Profile con `py-spy` y `cProfile`; reportes en `docs/perf/`

### Cython extensions (opcional)
- [ ] Hot path: `HexCoord.distance()`, `PheromoneField.decay_all()` en Cython

### Benchmarks comparativos
- [ ] HOC vs Ray vs Dask vs `multiprocessing.Pool`
- [ ] Identificar workloads donde HOC gana (sweet spot)
- [ ] Reporte público en `docs/benchmarks-comparison.md`

### Auditorías al cierre
- 🔒 Security: async no introduce race conditions nuevas
- 🏛️ General: throughput ≥5× vs baseline

---

## FASE 8 — Distribución Multi-nodo
**Objetivo**: HOC realmente distribuido (no solo nombre).
**Duración**: 6-8 semanas
**Tag al cerrar**: `v2.1.0-phase08`

### Transporte
- [ ] gRPC (recomendado) o ZeroMQ entre instancias
- [ ] Schemas de mensajes con **mscs** (consistencia con Fases 2/6)
- [ ] mTLS entre nodos (certificados auto-firmados o PKI)

### Topología federada
- [ ] Cada máquina = "subpanal" con N celdas locales
- [ ] Bordes intercambian feromonas con vecinos remotos
- [ ] Reina global o reinas regionales (configurable)
- [ ] State machines de federación (con tramoya)

### Service discovery
- [ ] mDNS/Bonjour (zero-config local)
- [ ] Consul/etcd (cloud)
- [ ] DNS-based (Kubernetes-friendly)

### Network resilience
- [ ] Retries con backoff exponencial
- [ ] Circuit breakers a nivel RPC
- [ ] Tolerancia a particiones de red (CAP: AP — favoreceríamos disponibilidad)

### Chaos engineering
- [ ] `chaos-mesh` integration o tooling propio
- [ ] Tests: kill node, slow network, packet loss, partition

### Auditorías al cierre
- 🔒 Security: pen-test de canal RPC; auth obligatoria
- 🏛️ General: latencia inter-nodo aceptable; CAP comportamiento documentado

---

## FASE 9 — GPU & Aceleración
**Objetivo**: Aprovechar hardware moderno (GPUs, SIMD avanzado, Rust).
**Duración**: 4-6 semanas
**Tag al cerrar**: `v2.5.0-phase09`

### GPU support
- [ ] `GPUCell` con backend pluggable (CuPy, PyTorch, JAX)
- [ ] Auto-detección de GPUs disponibles
- [ ] Workloads pesados ([benchmarks/workload_heavy.py](benchmarks/workload_heavy.py)) offload automático:
  - SVD, FFT, Monte Carlo, matrix mult → GPU si disponible
  - Fallback graceful a CPU si no hay GPU
- [ ] Multi-GPU coordination (1 GPU por subset de celdas)

### Rust extensions vía PyO3
- [ ] Hot paths: hexagonal math, RWLock, atomic counters
- [ ] Build via `maturin`; wheel multi-platform

### WebAssembly export (experimental)
- [ ] Subset de HOC compilable a WASM
- [ ] Use case: edge computing, browser-based simulations

### Auditorías al cierre
- 🔒 Security: GPU memory isolation; no leaks entre tareas
- 🏛️ General: speedup 10-100× en workloads numéricos vectorizables

---

## FASE 10 — AI/ML & Research
**Objetivo**: Diferenciadores únicos; output de investigación.
**Duración**: 6-8 semanas
**Tag al cerrar**: `v3.0.0-phase10` 🎉 (release mayor)

### Auto-tuning con Reinforcement Learning
- [ ] Agente RL (PPO con `stable-baselines3`) que ajusta:
  - Ratios de behaviors (foragers/nurses/scouts/guards)
  - `pheromone_decay_rate`, `diffusion_rate`
  - Quorum thresholds
- [ ] Reward = throughput × latencia⁻¹ × success_rate
- [ ] Trained policies publicadas en HuggingFace

### Genetic evolution
- [ ] `DEAP` para evolucionar configuraciones
- [ ] Poblaciones de panales compitiendo en workloads
- [ ] Cross-over entre configs ganadoras

### Multi-colony / Federación
- [ ] Múltiples colonias coordinando vía "swarming" (división celular)
- [ ] Multi-tenancy: aislamiento entre colonias

### DAG support para tareas con dependencias
- [ ] `TaskGraph` estilo Airflow/Prefect pero bio-inspirado
- [ ] State machine con tramoya por nodo del DAG
- [ ] Visualización del DAG en dashboard (Fase 5)

### Imitation learning entre celdas
- [ ] Celdas exitosas comparten "memoria" (epigenética simulada)
- [ ] Vecinas heredan estrategias

### Research output
- [ ] Paper técnico: "HOC: Stigmergic Computing on Hexagonal Topologies"
- [ ] Comparison vs Ray/Dask con números reales
- [ ] Open dataset con resultados reproducibles
- [ ] Submit a workshop/conference (e.g., ICDCS, EuroSys workshops)

### Auditorías al cierre
- 🔒 Security: modelos RL no leak datos de entrenamiento
- 🏛️ General: v3.0.0 production-ready, documented, benchmarked

---

# Resumen visual

```
v1.0.0 baseline
   │
   ├─ Fase 1: Estabilización ────────────► v1.1.0 (cero bugs críticos)
   ├─ Fase 2: Seguridad + mscs ─────────► v1.2.0 (0 críticos sec)
   ├─ Fase 3: Tooling + CI ──────────────► v1.3.0 (proyecto OSS-grade)
   ├─ Fase 4: DX + tramoya ──────────────► v1.4.0 (state machines formales)
   ├─ Fase 5: Observabilidad + trame ────► v1.5.0 (dashboard live)
   ├─ Fase 6: Persistencia ──────────────► v1.6.0 (recovery real)
   │
   ├─ Fase 7: Async + Performance ───────► v2.0.0 ⚠️ breaking
   ├─ Fase 8: Multi-nodo ────────────────► v2.1.0 (true distributed)
   ├─ Fase 9: GPU + Rust ────────────────► v2.5.0 (HW moderno)
   │
   └─ Fase 10: AI/ML + Research ─────────► v3.0.0 🎉
```

# Dependencias entre fases

```
Fase 1 ─────┬─► Fase 2 ─┬─► Fase 6 (mscs serialization)
            │           └─► Fase 8 (mscs RPC schemas)
            │
            ├─► Fase 3 ─┬─► Fase 4 (tooling base para refactors)
            │           └─► Fase 5
            │
            ├─► Fase 4 ─┬─► Fase 5 (state machines visualizables)
            │           ├─► Fase 8 (FSM federación)
            │           └─► Fase 10 (FSM por DAG node)
            │
            └─► Fase 7 ─┬─► Fase 8 (async multi-nodo)
                        ├─► Fase 9 (async GPU offload)
                        └─► Fase 10 (async RL agents)
```

# Métricas de éxito por fase

| Fase | Métrica clave | Baseline | Target |
|------|---------------|----------|--------|
| 1 | Bugs críticos | 3 | 0 |
| 1 | Cobertura tests | ~30% | ≥75% |
| 2 | Vulnerabilidades críticas | 1 | 0 |
| 2 | Overhead HMAC | — | <5% |
| 3 | LOC en `core.py` | 3.624 | <800 (split) |
| 3 | mypy errors | N/A | 0 (--strict) |
| 4 | State machines formales | 0 | 5+ |
| 5 | Observability overhead | — | <3% |
| 6 | Recovery time tras crash | N/A | <10s |
| 7 | Throughput vs baseline | 0.47 t/s | ≥2.5 t/s (5×) |
| 8 | Latencia inter-nodo | N/A | <50ms p99 |
| 9 | Speedup en SVD/FFT | 1× | 10-100× con GPU |
| 10 | Auto-tuning improvement | manual | ≥30% throughput |

# Notas finales

- **Roadmap es vivo**: ajustar según aprendizajes de cada fase.
- **Releases públicos** después de Fase 3 (CI/CD listo para PyPI auto-publish).
- **Cada fase abre branch propia**: `phase/01-stabilization`, `phase/02-security`, etc.
- **Política de merge a main**: PR review + tests + ambas auditorías deben pasar.
- **CHANGELOG.md** se inicia en Fase 3 y se mantiene desde ahí.
