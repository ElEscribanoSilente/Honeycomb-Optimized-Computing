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

## FASE 3 — Tooling, CI/CD & Code Quality ✅ CERRADA (2026-04-24)
**Objetivo**: Higiene de proyecto OSS-grade.
**Duración real**: 1 sesión
**Tag**: `v1.3.0-phase03`
**Cierre**: ver [snapshot/PHASE_03_CLOSURE.md](snapshot/PHASE_03_CLOSURE.md)

**Resultado**: 582 tests pasando (+161: 133 refactor-compat + 28 coverage
boosters), cobertura global 75.73% (+3.73 pts, primera vez sobre 75%),
0 bandit HIGH/MEDIUM/LOW, 0 pip-audit vulnerabilities. `core.py` (3,615
LOC) dividido en 14 submódulos; `metrics.py` (1,169 LOC) en 3 submódulos
+ `__init__`. 4 GitHub Actions workflows (test / lint / security /
release). 7 ADRs + 3 documentos OSS (CONTRIBUTING, CoC, SECURITY).
Bug latente **B11** (mismo patrón que B9) encontrado durante pase mypy
sobre `resilience.py` y arreglado. Gaps diferidos: 5 archivos legacy
> 800 LOC (resilience 1639, nectar 1366, swarm 1132, memory 940,
bridge 886) — splits planificados para phases 4-6.

### Tooling
- [x] `ruff` (replace flake8/isort/pyupgrade) — config en `pyproject.toml`
- [x] `black` — line length 100
- [x] `mypy --strict` con `[tool.mypy]` en `pyproject.toml` (strict en security/memory/resilience; legacy suprimido vía ADR-006)
- [x] `pre-commit` con: ruff, black, mypy, trailing-whitespace, check-yaml
- [x] `pytest-cov` con threshold ≥75% (alcanzado 75.73%)
- [x] Pin versions en `requirements-dev.txt`

### CI/CD
- [x] `.github/workflows/test.yml`: matriz Python 3.10/3.11/3.12 × Linux/macOS/Windows
- [x] `.github/workflows/lint.yml`: ruff + mypy + black --check
- [x] `.github/workflows/security.yml`: `bandit`, `pip-audit`, `safety` (+weekly cron)
- [ ] `.github/workflows/docs.yml`: diferido a Fase 9 (cuando haya sphinx)
- [x] `.github/workflows/release.yml`: tag-triggered build + GitHub release; PyPI publish stubbed (OIDC trusted-publisher pendiente de provisioning)

### Refactor de código
- [x] Dividir `core.py` (3,615 LOC) en 14 submódulos:
  - `core/grid.py` (facade + HoneycombGrid + factories)
  - `core/grid_geometry.py` (HexCoord, HexDirection, HexRegion, HexPathfinder, alias HexRing)
  - `core/grid_config.py` (HoneycombConfig, GridTopology)
  - `core/cells_base.py` (CellState, CellRole, HoneycombCell)
  - `core/cells_specialized.py` (6 subtipos worker/drone/nursery/storage/guard/scout)
  - `core/_queen.py` (QueenCell, peeled off para <800 LOC)
  - `core/cells.py` (facade público)
  - `core/events.py` (EventBus + handlers)
  - `core/health.py` (CircuitBreaker + HealthMonitor)
  - `core/locking.py` (RWLock)
  - `core/pheromone.py` (internos)
  - `core/constants.py` (scaffolding, 6 constantes)
  - `core/__init__.py` (re-exports + PEP 562 __getattr__ para transicionales)
- [x] Mover `MetricsCollector` (+ internal CellMetrics + GridMetrics) → `metrics/collection.py`
- [x] Dividir `metrics.py` (1,169 LOC):
  - `metrics/collection.py` (métricas + transicionales)
  - `metrics/visualization.py` (HoneycombVisualizer + ColorScheme)
  - `metrics/rendering.py` (HeatmapRenderer + FlowVisualizer)
- [x] Magic numbers → constantes nombradas en `core/constants.py` (scaffolding inicial; extracción completa es trabajo iterativo)

### Documentación de proyecto
- [x] `CHANGELOG.md` extendido con [1.3.0-phase03]
- [x] `CONTRIBUTING.md`
- [x] `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1)
- [x] `SECURITY.md` (private disclosure + threat model recap)
- [ ] `docs/adr/` con ADRs retroactivos (decisión de topología hex, mscs vs pickle, etc.)

### Auditorías al cierre
- 🔒 Security: pip-audit + bandit limpios
- 🏛️ General: complejidad ciclomática <10 por función, sin módulos >800 LOC

---

## FASE 4 — Configuración & Developer Experience — **CERRADA** 🟢
**Objetivo**: HOC usable sin necesidad de leer código fuente. **Integración profunda de [tramoya](https://pypi.org/project/tramoya/)** para state machines formales.
**Duración**: 3-4 semanas
**Tag al cerrar**: `v1.4.0-phase04` — cierre 2026-04-24

**Resultado**: 663 tests pasando (+81 vs. Phase 3), cobertura **76.34 %**
(+0.61 pts), **5 FSMs formalizadas** (CellState wired, 4 declarativas),
`docs/state-machines.md` auto-generado con drift detector en CI,
`swarm.py` + `nectar.py` graduados de mypy override (29 anotaciones), bug
**B12** (RoyalJelly.get_stats AttributeError latente) corregido. Bandit
0/0/0 mantenido. Configuración externa (4.8) y CLI (4.9) **diferidos a
Phase 5+** por priorización del usuario; las 4 FSMs declarativas (gap
de wire-up real) también diferidas. Cierre completo:
[snapshot/PHASE_04_CLOSURE.md](snapshot/PHASE_04_CLOSURE.md). Decisión de
arquitectura: [ADR-007](docs/adr/ADR-007-tramoya-fsm-integration.md).

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

## FASE 4.1 — Wire-up TaskLifecycle + `choreo` static FSM checker — **CERRADA** 🟢
**Objetivo**: Cerrar la brecha entre las 4 FSMs declarativas y las 1 wired
con (a) wire-up real de TaskLifecycle y (b) una herramienta nueva,
`choreo`, que verifica estáticamente las FSMs sin necesidad de wire-up
en runtime.
**Duración**: 1 día
**Tag al cerrar**: `v1.4.1-phase04.1` — cierre 2026-04-24

**Resultado**: 705 tests pasando (+42 vs Phase 4: 10 wire-up + 32 choreo).
TaskLifecycle FSM **wired** vía `HiveTask.__setattr__` (ahora 2 de 5 FSMs
wired). Nueva herramienta `choreo` (~600 LOC, subpaquete propio en
`choreo/`) implementa verificación estática AST-based: detecta mutaciones
undocumented (errores), dead states + enum-extras (warnings), FSMs
declarative-only (info). Aplicada a HOC produce el reporte exacto: 0
errores, 2 warnings (B12-bis `TaskState.ASSIGNED`, B12-ter 4 `CellState`
dead), 3 info (Pheromone/Succession/Failover). Nuevo job CI
`choreo-static-check` en `lint.yml`. Sin nuevas runtime deps. Cierre
completo: [snapshot/PHASE_04_1_CLOSURE.md](snapshot/PHASE_04_1_CLOSURE.md).
Decisión de arquitectura: [ADR-008](docs/adr/ADR-008-choreo-static-fsm-verification.md).

---

## FASE 4.2 — `choreo` v0.2: reified + auto-derive — **CERRADA** 🟢
**Objetivo**: Cuatro mejoras additivas a `choreo` y al subpaquete
`state_machines/`: walker patterns ampliados (setattr + replace),
reified transitions decorator (`@transition`), auto-derive subcommand,
opt-in enum binding.
**Duración**: 1 día
**Tag al cerrar**: `v1.4.2-phase04.2` — cierre 2026-04-25

**Resultado**: 734 tests pasando (+29 vs Phase 4.1: 8 walker + 4
enum_name + 6 derive + 11 reified). 5 walker patterns soportados (vs
3 en 4.1). 4 métodos reified en `HiveTask` (claim/complete/fail/retry)
como API additiva. `python -m choreo derive` genera skeletons desde
código. `cell_fsm.py` y `task_fsm.py` con `enum_name=` explícito.
choreo aplicado a HOC sigue produciendo el reporte exacto de Phase 4.1
(0 err / 2 warn / 3 info), confirmando que ninguna feature rompe el
contrato. Sin nuevas runtime deps. Bandit/pip-audit/ruff/black/mypy
limpios. Cierre completo:
[snapshot/PHASE_04_2_CLOSURE.md](snapshot/PHASE_04_2_CLOSURE.md).
Decisión de arquitectura:
[ADR-009](docs/adr/ADR-009-reified-transitions-and-auto-derive.md).

### Diferido a Phase 5+
- Walker patterns adicionales (`attrs.evolve`, RHS computado, etc.).
- `--strict` CI flip espera el wire-up de los reservados (Phase 5).
- Auto-derive con CFG analysis (sources reales) — research-grade,
  deferred indefinidamente.

---

## FASE 4.3 — Dead enum cleanup — **CERRADA** 🟢
**Objetivo**: Mini-fase de cleanup que aplica per-member la
discriminación "eliminar vs reservar" sobre los dead enum members
detectados por choreo en Phase 4.1/4.2.
**Duración**: 1 día (~2h)
**Tag al cerrar**: `v1.4.3-phase04.3` — cierre 2026-04-25

**Resultado**: choreo reduce warnings de 2 a 1.

- **Eliminados**: `TaskState.ASSIGNED`, `CellState.SPAWNING`,
  `CellState.OVERLOADED` (sin caso de uso real).
- **Reservados (Phase 5)**: `CellState.MIGRATING` (wire-up en
  `CellFailover.migrate_cell` para observabilidad), `CellState.SEALED`
  (wire-up para graceful shutdown).
- 733 tests pasando (-1 vs Phase 4.2: test obsoleto B12-bis eliminado).
- ruff/black/mypy/bandit/pip-audit limpios.

Cierre completo: [snapshot/PHASE_04_3_CLOSURE.md](snapshot/PHASE_04_3_CLOSURE.md).
Decisión de arquitectura: [ADR-010](docs/adr/ADR-010-dead-enum-cleanup.md).

### Pendiente para Phase 5
- Wire-up de `CellState.MIGRATING` y `CellState.SEALED`.
- Wire-up de las 3 FSMs declarative-only (Pheromone, Succession,
  Failover).
- Tras lo anterior, flippear `--strict` en el job CI choreo.

---

## FASE 5 — Observabilidad — **CERRADA** 🟢
**Objetivo**: Visibilidad operacional production-grade + cierre del
gap "FSM declarada pero no wireada" de Phase 4 / 4.3.
**Duración real**: 1 sesión
**Tag al cerrar**: `v1.5.0-phase05` — cierre 2026-04-26

**Resultado**: 804 tests pasando (+71 vs Phase 4.3), cobertura
**79.41%** (+~3 pts), **5/5 FSMs wireadas** (vs 2/5 al inicio):
Phase 5.1 wireó CellState.MIGRATING + SEALED en CellFailover +
nuevo `cell.seal()`; Phase 5.2c wireó FailoverFlow vía
`_FailoverCellState` wrapper + tramoya `undo()`; Phase 5.2a wireó
PheromoneDeposit como static field (perf budget +1.4% vs <3%);
Phase 5.2b wireó QueenSuccession vía `_SuccessionState` wrapper sin
tocar lógica de quorum (7 tests TestQuorumSignedVotes intactos).
Phase 5.3 introdujo logging estructurado vía `structlog` con 6
eventos cableados. Phase 5.5 capturó `bench_baseline.json`
reproducible + `compare_bench.py` + nuevo job CI `bench-regression`.
Phase 5.6 flippeó `choreo check --strict` en CI — el report quedó
en **0/0/0** y CI rompe en cualquier futuro PR con dead state /
enum-extra / declarative-only FSM. Bandit 0/0/0 mantenido. Cierre
completo: [snapshot/PHASE_05_CLOSURE.md](snapshot/PHASE_05_CLOSURE.md).
Decisiones de arquitectura:
[ADR-011](docs/adr/ADR-011-observability-stack.md) (observability
stack), [ADR-012](docs/adr/ADR-012-choreo-strict-mode.md) (`--strict`
flip).

### Diferido a Phase 5.x followup / Phase 6
- 5.4 Métricas Prometheus + `/metrics` endpoint + `hoc-cli serve-metrics`
  (explícitamente opcional per brief; structured logs de 5.3 cubren
  caso interim vía promtail / fluent-bit log-derived metrics).
- 5.7 Dashboard (FastAPI + HTMX + Mermaid live) — explícitamente
  opcional, deferido a Phase 6.
- Cobertura 80% target (cerró 79.41%, falta -0.59 pts; `bridge.py`
  56% sigue siendo el cuello).
- `test_grid_creation` +25.88% regresión vs baseline (FSM allocation
  per-cell); documentado para optimizar en Phase 5.x followup.

### Scope original (planeado)
**Objetivo planeado**: Visibilidad operacional production-grade. **Integración de [trame](https://kitware.github.io/trame/)** (asumiendo que "tramoya" del usuario era trame para dashboards) — opcional, ver nota.
**Duración estimada**: 3 semanas
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

## FASE 6 — Persistencia & Storage — **CERRADA** 🟢
**Objetivo**: HoneyArchive con backends reales; recovery completo tras crash.
**Duración real**: 1 sesión
**Tag al cerrar**: `v1.6.0-phase06` — cierre 2026-04-27

**Resultado**: 961 tests pasando (+157 vs Phase 5), cobertura
**83.08 %** (+3.67 pts), nuevo subpaquete `hoc.storage` con
`StorageBackend` Protocol + `MemoryBackend` default + `SQLiteBackend`
(WAL + schema versioning + connection-per-thread, stdlib only).
`HoneycombGrid.checkpoint(path)` y `restore_from_checkpoint(path)`
serializan/deserializan vía blob HMAC-firmado (encode_blob /
decode_blob); auto-checkpointing opt-in dentro del tick loop con
RPO acotado por `checkpoint_interval_ticks`. Phase 6.5 splittea
`bridge.py` (886 LOC, último top-level legacy ≥ 800 LOC) en
`hoc.bridge` subpaquete (3 módulos), empujando bridge cobertura de
56 % a 96 % y cerrando Gap 4 desde Phase 4 closure. Phase 6.6
introduce class-level shared FSM en `HoneycombCell` que cierra la
regresión Phase 5 `test_grid_creation` (-66.47 % vs baseline, target
era ±5 %). Phase 6.7 captura el bench baseline en `ubuntu-latest`
vía `workflow_dispatch` desde main; `bench-regression` CI job vuelve
a hard-fail mode con threshold 10 % (el advisory 50 % de Phase 5.5
se elimina). Bandit 0/0/0 mantenido. choreo `--strict` 0/0/0
mantenido. Sin nuevas runtime deps (sqlite3 + zlib son stdlib).
Cierre completo: [snapshot/PHASE_06_CLOSURE.md](snapshot/PHASE_06_CLOSURE.md).
Decisiones de arquitectura:
[ADR-013](docs/adr/ADR-013-storage-backend-abstraction.md) (storage
backend abstraction),
[ADR-014](docs/adr/ADR-014-checkpoint-format.md) (checkpoint format),
[ADR-015](docs/adr/ADR-015-class-level-cell-fsm.md) (class-level
FSM optimization).

### Diferido a Phase 6.x followup / Phase 7+
- 6.9 Backends adicionales (LMDB / S3 / Redis) — opcional per brief.
  Spec en ADR-013.
- 6.10 Phase 5.4 (Prometheus) + 5.7 (dashboard) carryover — opcional.
- Task queue persistence — `SwarmScheduler.task_queue` no incluido
  en checkpoint v1; diferido a Phase 7 async migration.
- CombStorage backend — deliberadamente in-memory; revisitar si
  Phase 8 demanda shared L2.

### Scope original (planeado)

**Objetivo planeado**: HoneyArchive con backends reales; recovery completo tras crash.
**Duración estimada**: 3-4 semanas
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

## FASE 7 — Async/Await & Performance — **CERRADA** 🟢 ⚠️ MAJOR
**Objetivo**: Throughput 5-10× mejor; sandboxing real de tareas.
**Duración real**: 1 sesión
**Tag al cerrar**: `v2.0.0-phase07` ⚠️ **PRIMER MAJOR BUMP DEL ROADMAP** — cierre 2026-04-28

**Resultado**: 1062 tests pasando (+101 vs Phase 6), cobertura
**83.47 %** local Windows (CI Linux esperado ≥ 85 % cuando los 8 tests
fork-only del sandbox corren). Las cuatro clases user-facing
— ``HoneycombGrid``, ``NectarFlow``, ``SwarmScheduler``,
``HoneycombCell`` — exponen ``async def tick`` / ``async def
execute_tick`` desde v2.0.0; los wrappers ``run_tick_sync`` /
``run_execute_tick_sync`` quedan como migration aid hasta v3.0.0.
Phase 7.5 BehaviorIndex (O(n·m) → O(m·log n)) reduce el filter loop
de ~34k ops/tick a ~220 ops/tick — bench
``test_swarm_1000_tasks_single_tick`` ≈ 1.7 ms (≈ **6× speedup** vs
extrapolated baseline; brief target ≥5× cumplido). Phase 7.4
SandboxedTaskRunner aterriza process isolation opt-in en POSIX (fork);
Windows deferred a Phase 7.x followup. Phase 7.10 cierra el gap
diferido de Phase 6.4: ``SwarmScheduler.task_queue`` persiste en
checkpoint v2 (``encode_blob`` siempre escribe 0x02; ``decode_blob``
acepta ambos vía ``SUPPORTED_VERSIONS``). Bandit 0/0/0 mantenido.
choreo ``--strict`` 0/0/0 mantenido. Sin nuevas runtime deps. Cierre
completo: [snapshot/PHASE_07_CLOSURE.md](snapshot/PHASE_07_CLOSURE.md).
Decisiones de arquitectura:
[ADR-016](docs/adr/ADR-016-async-tick-loop.md) (async tick),
[ADR-017](docs/adr/ADR-017-sandboxing-model.md) (sandboxing),
[ADR-018](docs/adr/ADR-018-behavior-index.md) (BehaviorIndex).

### Diferido a Phase 7.x followup / Phase 9
- 7.7 Cython extensions — DoD permite "deferred a Phase 9".
- 7.9 Comparative benchmarks (HOC vs Ray / Dask / multiprocessing)
  — deferred; necesita dev deps Ray/Dask + harness comparativo.
- 7.11 cobertura ≥85 % global — CI Linux esperado ≥85 % post-merge.
- 7.12 Phase 5.4 Prometheus + 5.7 Dashboard carryover — opcional
  per brief; deferred a Phase 8 si emerge demanda observabilidad
  cross-node.
- Sandbox process isolation on Windows — Phase 7.x followup
  (spawn + cloudpickle o subprocess).
- Sandbox cgroup v2 / Job Objects (full impls) — Phase 7.x followup;
  stubs raise ``SandboxNotSupported``.
- CI bench baseline refresh — ``gh workflow run bench.yml`` post-merge,
  mismo recipe Phase 6.7.

### Scope original (planeado)

**Objetivo planeado**: Throughput 5-10× mejor; sandboxing real de tareas.
**Duración estimada**: 4-5 semanas
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
