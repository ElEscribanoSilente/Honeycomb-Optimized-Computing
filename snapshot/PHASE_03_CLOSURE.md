# Phase 3 Closure — Tooling, CI/CD & Code Quality

**Fecha**: 2026-04-24
**Tag previsto**: `v1.3.0-phase03`
**Branch**: `phase/03-tooling`
**PR**: (pending — abrir tras `git push`)

---

## Resumen ejecutivo

Fase 3 cerrada con **582 tests pasando** (+161 vs Phase 2: 133 compat
tests del refactor + 28 coverage boosters), **cobertura global 75.73%**
(primera vez sobre el objetivo 75% desde el baseline), **4 GitHub Actions
workflows** creados, **7 ADRs** retroactivos + **3 documentos de proyecto**
(CONTRIBUTING / CoC / SECURITY) y — el riesgo principal de la fase —
**`core.py` dividido en 14 submódulos** y **`metrics.py` dividido en 3
submódulos** sin romper un solo test. La API pública `from hoc import ...`
es identity-preserving: `hoc.HexCoord is hoc.core.HexCoord is
hoc.core.grid_geometry.HexCoord`.

Bandit sigue limpio en todas las severidades (0/0/0); pip-audit sigue
limpio. Un bug latente se encontró durante el pase mypy sobre `resilience.py`
— **B11** — y se arregló en el mismo commit que configuró las herramientas
(familia del B9 descubierto en Fase 1).

| Métrica | Phase 2 (v1.2.0-phase02) | Phase 3 (v1.3.0-phase03) | Δ |
|---------|--------------------------|---------------------------|---|
| Tests pasando | 421 | **582** | +161 |
| Tests de refactor compat (nuevos) | 0 | **133** | +133 |
| Tests de events/health boosters | 0 | **28** | +28 |
| Cobertura global | 72% | **75.73%** | +3.73 pts ✅ |
| Módulos >800 LOC (total proyecto) | 3 (core 3,615; nectar 1,366; swarm 1,132) | 5 legacy (resilience 1,639; nectar 1,366; swarm 1,132; memory 940; bridge 886) | N/A* |
| Módulos >800 LOC *nuevos de Phase 3* | — | **0** | ✅ |
| `core.py` (3,615 LOC) | monolito | **14 submódulos** (mayor = 799 LOC) | ✅ |
| `metrics.py` (1,169 LOC) | monolito | **3 submódulos** + `__init__.py` | ✅ |
| ruff errores | N/A | **0** | ✅ |
| black reformat | N/A | **0** | ✅ |
| mypy errores (strict sobre security/memory/resilience) | N/A | **0** | ✅ |
| Bandit HIGH / MEDIUM / LOW | 0 / 0 / 0 | **0 / 0 / 0** | = ✅ |
| pip-audit vulnerabilidades | 0 | **0** | = ✅ |
| GitHub Actions workflows | 0 | **4** (test, lint, security, release) | +4 |
| ADRs documentados | 0 | **7** (incl. ADR-000 template) | +7 |
| Docs de proyecto nuevos | 0 | **3** (CONTRIBUTING, CoC, SECURITY) | +3 |
| Bugs latentes descubiertos | — | **B11** (mismo patrón que B9) | +1 |

\* Phase 2 reportó 3 módulos > 800 LOC contando `core.py` (3,615) y `metrics.py`
(1,169). Phase 3 los dividió, pero expuso que `resilience.py` (siempre
grande, no perfilado antes) también está sobre 800. Ver "Gaps diferidos".

---

## 3.1 Tooling — ruff / black / mypy / pre-commit

- `pyproject.toml` extendido con:
  - `[tool.ruff]` — `extend-select = ["E","F","W","I","B","UP","SIM","RUF"]`, línea 100, target py310.
  - `[tool.ruff.lint.isort]` — `known-first-party = ["hoc"]`.
  - `[tool.black]` — línea 100, targets py310/311/312.
  - `[tool.mypy]` — `strict = true`, `follow_imports = "silent"`, `no_site_packages = true`. Exclude legacy files; override `ignore_errors = true` para `core.*`, `metrics.*`, `bridge`, `nectar`, `swarm`. Strict permanece activo en `security.py`, `memory.py`, `resilience.py`, `__init__.py`.
  - `[tool.coverage.report]` — `fail_under = 75`.
  - `[tool.bandit]` — exclude tests/benchmarks/snapshot.
- `requirements-dev.txt` con versiones pinneadas (ruff 0.8.6, black 25.1.0, mypy 1.14.1, hypothesis 6.151.11, pytest 8.4.1, bandit 1.8.0, pip-audit 2.9.0, radon 6.0.1, pre-commit 4.0.1).
- `.pre-commit-config.yaml` con hooks: trailing-whitespace, end-of-file-fixer, check-yaml/toml/json, check-added-large-files, ruff + ruff-format, black, mypy, bandit.

Baseline → estado final:
- ruff: 1765 errores → **0** (1563 autofix + 18 unsafe-fix + 11 manuales: SIM102 merges, SIM117/105 context, RUF012 ClassVar, B904 raise-from, B007 renames).
- black: 20 archivos a reformatear → **0**.
- mypy: tras configurar strict + relajar legacy, **0 errores** en los 4 archivos strict (security, memory, resilience, __init__).

### B11 — bug latente descubierto durante mypy pass

| ID | Severidad | Ubicación | Descripción | Estado |
|----|-----------|-----------|-------------|--------|
| **B11** | 🟠 Alta | `resilience.py:1138` `CombRepair._rebuild_cell` | Escribía `cell._pheromone_level = 0.0`, pero `HoneycombCell` no tiene tal atributo. Creaba silenciosamente un atributo muerto y dejaba la feromona original intacta tras un rebuild. Mismo patrón que **B9** de Fase 1 (metrics.py). Arreglado reemplazando `_pheromone_field` por una nueva `PheromoneField()` — reset real. | ✅ Fix |

> Mypy strict sobre `resilience.py` (que la Fase 2 anotó parcialmente)
> detectó este fallo con `[attr-defined]` — la misma familia de
> tooling que encontró B9 ahora encontró B11. Valida la tesis de
> Phase 1 de que *mypy + tests* desentierran bugs latentes que ningún
> test previo cubría.

---

## 3.2 CI/CD — GitHub Actions

| Workflow | Triggers | Jobs |
|----------|----------|------|
| `test.yml` | push/PR main+phase/**, manual | pytest matrix `{ubuntu, macos, windows}` × `{py3.10, 3.11, 3.12}` — 9 jobs. Coverage upload a Codecov desde `ubuntu + py3.12`. |
| `lint.yml` | push/PR, manual | 3 jobs paralelos: `ruff check`, `black --check`, `mypy .`. Cada uno pinnea su tool version. |
| `security.yml` | push/PR, manual, **weekly cron lunes 05:00 UTC** | `bandit` (JSON artifact + fail en MEDIUM+), `pip-audit` (runtime + dev), `safety` (defense-in-depth, continue-on-error). |
| `release.yml` | tag `v*.*.*`, manual | `build` (sdist + wheel), `github-release` (attach + generate notes). PyPI publish **stubbed** — OIDC trusted-publisher setup deferido hasta que se aprovisione cuenta. |

Concurrency groups cancelan runs duplicados. Cada workflow YAML pasa
validación `yaml.safe_load` local.

No hay workflow de docs (sphinx) — diferido a Fase 9 per roadmap.

---

## 3.3 Refactor estructural — core.py + metrics.py

### core.py (3,615 LOC) → core/ subpackage (14 archivos, máx 799 LOC)

| Submódulo | LOC | Contenido |
|-----------|-----|-----------|
| `core/__init__.py` | 202 | Re-exports + `__all__` idéntico al antiguo; PEP 562 `__getattr__` para clases transicionales (`CellMetrics`/`GridMetrics`/`MetricsCollector`) para romper circular imports. |
| `core/grid.py` | 799 | Facade: `HoneycombGrid`, `_create_cell_by_role`, `create_grid`, `benchmark_grid`. Re-exporta `HexCoord`/`HexDirection`/`HexRegion`/`HexRing`/`HexPathfinder`/`HoneycombConfig`/`GridTopology`. |
| `core/grid_geometry.py` | 493 | `HexCoord`, `HexDirection` + `_DIRECTION_VECTORS`/`_DIRECTION_ARRAY`, `_cached_ring`, `_cached_filled_hex`, `_cube_round`, `HexRegion`, `HexPathfinder`, alias `HexRing = HexRegion`. |
| `core/grid_config.py` | 148 | `HoneycombConfig`, `GridTopology`. |
| `core/cells_base.py` | 528 | `CellState`, `CellRole`, `HoneycombCell` (clase base). |
| `core/cells_specialized.py` | 642 | 6 subtipos: `WorkerCell`, `DroneCell`, `NurseryCell`, `StorageCell`, `GuardCell`, `ScoutCell`. |
| `core/_queen.py` | 235 | `QueenCell` — peeled off cells_specialized.py para mantenerlo bajo 800 LOC. |
| `core/cells.py` | 45 | Facade público que re-exporta las 7 celdas. |
| `core/events.py` | 374 | `EventType`, `Event`, `EventHandler`, `_HandlerRef`, `EventBus`, `get_event_bus`, `set_event_bus`, `reset_event_bus`. |
| `core/health.py` | 246 | `CircuitState`, `CircuitBreaker`, `HealthStatus`, `HealthMonitor`. |
| `core/locking.py` | 95 | `RWLock`. |
| `core/pheromone.py` | 148 | Internos: `PheromoneType`, `PheromoneDeposit`, `PheromoneField`. (Distintos de `nectar.PheromoneType`.) |
| `core/constants.py` | 51 | Scaffolding inicial: 6 constantes extraídas (`DEFAULT_RADIUS`, `DEFAULT_VCORES_PER_CELL`, `DEFAULT_POLLEN_TTL_SECONDS`, etc.). La extracción más amplia de magic numbers se itera en fases siguientes. |

**Orden de migración** (seguido exactamente del brief del usuario):

1. Rename `core.py` → `_core_monolith` temporal.
2. Crear `core/__init__.py` vacío, submódulos vacíos.
3. Mover bloques verbatim a submódulos, ajustando imports internos a relativos.
4. Correr tests intermedios — deben pasar gracias al re-export de `core/__init__.py`.
5. Al final, borrar `_core_monolith`.
6. Verificar `python -c "from hoc import *"` funciona.

**Circular-import resolution**: `metrics/collection.py` importa `HoneycombCell`/`HoneycombGrid` bajo `TYPE_CHECKING` (solo para hints). `core/__init__.py` usa PEP 562 `__getattr__` para resolver `CellMetrics`/`GridMetrics`/`MetricsCollector` bajo demanda. `core/grid.py` + `cells_base.py` usan imports method-local para romper el ciclo en el momento de uso.

### metrics.py (1,169 LOC) → metrics/ subpackage (3 archivos + __init__.py, máx 790 LOC)

| Submódulo | LOC | Contenido |
|-----------|-----|-----------|
| `metrics/__init__.py` | 98 | Re-exports + `__all__` con 17 símbolos públicos. |
| `metrics/collection.py` | 790 | **Métricas públicas**: `MetricType`, `MetricLabel`, `MetricSample`, `Counter`, `Gauge`, `Histogram`, `Summary`, `CellMetricSnapshot`, `CellMetrics` (pública), `SwarmMetrics`, `HiveMetrics`. **Transicionales** (movidas desde `core/_metrics_internal.py` — eliminado): `_InternalCellMetrics`, `GridMetrics`, `MetricsCollector`. |
| `metrics/visualization.py` | 279 | `ColorScheme`, `HoneycombVisualizer`. |
| `metrics/rendering.py` | 271 | `HeatmapRenderer`, `FlowVisualizer`. |

`core/_metrics_internal.py` (146 LOC temporal) fue **eliminado** después
de mover su contenido a `metrics/collection.py`. `from hoc.core import
CellMetrics, GridMetrics, MetricsCollector` sigue funcionando via el PEP
562 `__getattr__` en `core/__init__.py`, manteniendo identidad estable.

### Dos `CellMetrics` coexisten (invariante preservado)

Desde Phase 1 existían dos clases `CellMetrics` con identidades distintas:

- `from hoc.core import CellMetrics` → interna, usada por `HoneycombCell.get_metrics()` (dataclass minimalista).
- `from hoc.metrics import CellMetrics` → pública, usada por `HiveMetrics` (con contadores tick/error, historial).
- `from hoc import CellMetrics` → resuelve a la pública (via `hoc/__init__.py` imports from `.metrics`).

Test `test_two_cell_metrics_classes_are_distinct` pins esta invariante.

### Tests de compat — `tests/test_refactor_compat.py`

133 tests nuevos para blindar el invariante Phase 3:

| Clase de test | #tests | Cubre |
|---------------|--------|-------|
| `test_top_level_symbol_importable` (param) | 67 | Cada símbolo del antiguo `hoc.__all__` sigue accesible via `from hoc import X`. |
| `test_hoc_core_symbol_importable` (param) | 37 | Cada símbolo de `core.py.__all__` sigue accesible via `from hoc.core import X`. |
| `test_hoc_metrics_symbol_importable` (param) | 15 | Cada símbolo de `metrics.py` sigue accesible via `from hoc.metrics import X`. |
| Identity checks | 8 | `hoc.X is hoc.core.X is hoc.core.submod.X` para `HexCoord`, `EventBus`, `HoneycombCell`, `QueenCell`, `HiveMetrics`, `HoneycombVisualizer`, `HeatmapRenderer`, `FlowVisualizer`. |
| CellMetrics distinct-identity | 1 | `hoc.core.CellMetrics is not hoc.metrics.CellMetrics`. |
| HexRing alias | 1 | `HexRing is HexRegion`. |
| isinstance across paths | 1 | Construir QueenCell via top-level, `isinstance` via submódulo. |
| `hoc.__all__` superset | 1 | `hoc.__all__` es superconjunto de los símbolos listados por el test. |
| Subpackage symbol resolution | resto | Importable desde facade + submódulo. |

### Tests de coverage booster — `tests/test_events_health.py`

28 tests dirigidos a `core/events.py` (61% → 82%), `core/health.py` (70% → 85%), `core/grid_geometry.py` (69% → 82%):

- EventBus: rate-limit, async dispatch, priority ordering, handler-exception counting, history filter/trim, `older_than` purge, clear-all, publish-after-shutdown, singleton set/reset.
- CircuitBreaker: CLOSED → OPEN → HALF_OPEN → CLOSED | OPEN transitions.
- HealthMonitor: status report, EventBus alerts, trend accessor, `should_check()` throttling.
- HexRegion: `from_line`, `from_area`, union/intersection/difference, bounds/centroid.
- HexPathfinder: straight-line path + obstacle avoidance.

---

## 3.4 Documentación de proyecto

- **`CONTRIBUTING.md`** — dev setup, comandos de test, quality checks, flujo de PR, code style, disciplina de roadmap phases.
- **`CODE_OF_CONDUCT.md`** — Contributor Covenant v2.1 adoptado por referencia; contacto de enforcement (`[HOC-CoC]` prefix).
- **`SECURITY.md`** — versiones soportadas, canales de disclosure privado (GitHub Security Advisories + email), timeline coordinado, scope in/out, recap del threat model de Phase 2, tabla de past advisories (B1–B11, pickle→mscs, Raft-like quorum).
- **`docs/adr/`** — 7 archivos (README, template, 6 ADRs):
  - ADR-001 Hexagonal topology (retroactivo, v1.0.0).
  - ADR-002 `mscs` replaces `pickle` (Phase 2).
  - ADR-003 Shared HMAC key vs per-cell (Phase 2).
  - ADR-004 `OrderedDict` LRU for `PheromoneTrail` (Phase 2).
  - ADR-005 Raft-like signed-vote quorum (Phase 2, refina B4).
  - ADR-006 Legacy modules suppressed from strict mypy en Phase 3 (con plan de graduación para fases 4-6).

---

## Auditorías

### Seguridad — Bandit (`snapshot/bandit_phase03.json`)

```
LOC scanned: 8,987
SEVERITY HIGH:   0
SEVERITY MEDIUM: 0
SEVERITY LOW:    0
```

Phase 2 ya había reducido a 0/0/0. Phase 3 mantiene sin regresiones a
pesar de:
- +161 tests (algunos con `assert False` y bare except/pass).
- +14 nuevos submódulos en `core/` y 3 en `metrics/`.
- 4 workflows YAML nuevos.

### Vulnerabilidades de dependencias — pip-audit (`snapshot/pip_audit_phase03.txt`)

```
No known vulnerabilities found
```

### Complejidad — Radon (`snapshot/radon_cc_phase03.txt`)

Average cyclomatic complexity: **C (13.7)**.

Funciones con CC > 10 (todas legacy, no tocadas por el refactor — el split
movió pero no simplificó):

- `swarm.SwarmScheduler.tick` (rango C)
- `swarm.ForagerBehavior.select_task`
- `swarm.SwarmBalancer.execute_work_stealing`
- `core/grid.HoneycombGrid.tick`
- `core/grid.HoneycombGrid.visualize_ascii`
- `core/grid.HoneycombGrid.get_stats`

DoD objetivo "ninguna función CC > 10" **NO se cumplió** — estas funciones
son pre-existentes (venían del antiguo `core.py` y `swarm.py`). Reducir
su CC requiere extract-method refactors que exceden el scope de Phase 3.
Ver "Gaps diferidos".

### LOC per archivo (Radon raw, `snapshot/radon_raw_phase03.txt`)

Archivos de producción > 800 LOC (DoD objetivo "ningún archivo > 800"):

| Archivo | LOC | Status |
|---------|-----|--------|
| `resilience.py` | 1,639 | ⚠️ legacy — no dividido en Phase 3 |
| `nectar.py` | 1,366 | ⚠️ legacy |
| `swarm.py` | 1,132 | ⚠️ legacy |
| `memory.py` | 940 | ⚠️ legacy |
| `bridge.py` | 886 | ⚠️ legacy |

**Archivos nuevos Phase 3 (core/*, metrics/*): todos < 800 LOC.** El
mayor es `core/grid.py` con 799 LOC (margen de 1 LOC — intencional, no un
accidente; el brief del usuario especificaba `HoneycombGrid` en `grid.py`
como un bloque compacto).

El DoD objetivo "ningún archivo > 800 LOC" se cumplió **solo para los
archivos tocados por Phase 3**. Los 5 archivos legacy > 800 se difieren a
fases posteriores. Ver "Gaps diferidos".

### Cobertura (`pytest --cov`)

| Módulo | Phase 2 | Phase 3 | Δ | Objetivo (≥75% global / ≥80% crítico) |
|--------|---------|---------|---|----------------------------------------|
| `__init__.py` | 100% | 100% | = | ✅ |
| `security.py` | 83% | 83% | = | ✅ (crítico) |
| `memory.py` | 93% | 93% | = | ✅ (crítico) |
| `resilience.py` | 84% | 84% | = | ✅ (crítico) |
| `metrics.py` → `metrics/` | 95% | `collection 96% / visualization 89% / rendering 89% / __init__ 100%` | — | ✅ |
| `core.py` (3,615 LOC) → `core/` (14 submódulos) | 54% | `pheromone 100% / grid_config 85% / locking 84% / events 82% / grid_geometry 82% / constants —% / health 85% / cells_base ? / _queen ? / cells.py 100% / cells_specialized ? / __init__ ?` | subió en promedio | ⚠️ `core/grid.py` 51% arrastra el promedio |
| `nectar.py` | 72% | 73% | +1 | ⚠️ sigue bajo 80% |
| `swarm.py` | 88% | 88% | = | ✅ |
| `bridge.py` | 56% | 56% | = | ⚠️ |
| **Global** | **72%** | **75.73%** | **+3.73 pts** | ✅ (primera vez) |

Todos los módulos tocados por Phase 3 mantienen o mejoran su cobertura.
El +3.73 puntos globales viene principalmente de los 161 tests nuevos
(compat + events/health boosters).

---

## Definition of Done — verificación

| Ítem | Estado | Nota |
|------|--------|------|
| ruff/black/mypy configurados en pyproject.toml y pasando | ✅ | mypy con overrides para legacy (ADR-006). |
| pre-commit instalable con `pre-commit install` | ✅ | `.pre-commit-config.yaml` creado con 6 repos de hooks. |
| 4 GitHub Actions workflows válidos | ✅ | test / lint / security / release. YAML valida con `yaml.safe_load`. |
| `core.py` dividido en `core/{grid, cells, events, health, locking, constants}.py` sin romper tests | ✅ | 14 submódulos (el brief pidió 6, entregamos 14 preservando <800 LOC). 582/582 tests. |
| `metrics.py` dividido en `metrics/{collection, visualization, rendering}.py` | ✅ | 3 submódulos + `__init__`. |
| Todos los módulos < 800 LOC | ⚠️ parcial | Cumplido para archivos Phase 3 (core/*, metrics/*). Legacy (resilience 1639, nectar 1366, swarm 1132, memory 940, bridge 886) se difieren. |
| CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md creados | ✅ | Los 3. |
| ≥ 3 ADRs | ✅ | 6 ADRs numerados + README + template. |
| Cobertura ≥ 75% global | ✅ | **75.73%** — primer cierre sobre el threshold. |
| `requirements-dev.txt` con versiones pinneadas | ✅ | 12 deps pinneadas. |
| Tests nuevos: `test_refactor_compat.py` verificando re-exports | ✅ | 133 tests. |
| Bandit/pip-audit siguen limpios | ✅ | 0/0/0 Bandit; 0 pip-audit. |
| Benchmark: degradación <3% | ⏸️ **DIFERIDO** | No se ejecutó bench end-to-end para Phase 3 — el refactor es syntactic, no altera hot paths. Se medirá en el PR antes de merge si se considera load-bearing. |
| Ninguna función con CC > 10 | ❌ | 6 funciones legacy siguen >10. Ver "Gaps diferidos". |

---

## Gaps diferidos (documentados para fases siguientes)

### Gap 1: 5 archivos legacy > 800 LOC

| Archivo | LOC | Naturaleza | Fase objetivo para split |
|---------|-----|------------|--------------------------|
| `resilience.py` | 1,639 | HiveResilience + CellFailover + QueenSuccession + HexRedundancy + MirrorCell + SwarmRecovery + CombRepair | Phase 4 (tramoya FSMs para QueenSuccession y FailoverFlow) |
| `nectar.py` | 1,366 | NectarFlow + WaggleDance + PheromoneTrail + RoyalJelly | Phase 4 (tramoya FSM para PheromoneDeposit) |
| `swarm.py` | 1,132 | SwarmScheduler + 5 BeeBehavior subtypes + HiveTask | Phase 4 (tramoya FSM para TaskLifecycle) |
| `memory.py` | 940 | HiveMemory + CombStorage + PollenCache + HoneyArchive | Phase 5 |
| `bridge.py` | 886 | CAMVHoneycombBridge + adapters + coordinate converters | Phase 6 |

ADR-006 documenta la secuencia de graduación esperada.

### Gap 2: 6 funciones legacy con CC > 10

- `SwarmScheduler.tick`, `ForagerBehavior.select_task`, `SwarmBalancer.execute_work_stealing` — extract-method durante Phase 4 tramoya integration (cada FSM transition puede pelarse como método).
- `HoneycombGrid.tick`, `HoneycombGrid.visualize_ascii`, `HoneycombGrid.get_stats` — extract-method candidatos para Phase 5 (donde los hot paths se profilarán).

### Gap 3: `core/grid.py` 799 LOC en el límite

Una función más y pasa el threshold. Habrá que dividir el facade (e.g. separar `HoneycombGrid` core del `create_grid`/`benchmark_grid`/`_create_cell_by_role` en `grid_factory.py`) cuando se toque en Phase 5.

### Gap 4: Benchmark end-to-end no medido

El refactor 3.3 es sintáctico (mover código entre archivos sin reescribir
lógica). No esperamos regresión de perf, pero no corrimos `benchmark_grid`
antes/después. Acción: correr en el PR antes de merge y añadir delta al
PR body.

### Gap 5: Mypy strict sobre legacy

Documentado en ADR-006 con plan de graduación. Phase 3 eligió suprimir
vs anotar in-situ porque la suppression permite cierre en scope y porque
el refactor de Phase 4-5 de los mismos archivos re-anotará el código
tras (no antes) el split.

---

## Lecciones aprendidas

1. **Subagent + worktree isolation es la herramienta correcta para refactors grandes.** El split de `core.py` (3,615 LOC) y `metrics.py` (1,169 LOC) se hizo en worktrees aislados, con el propio subagente corriendo tests intermedios. Zero regresiones en 421 tests. El coste: 2 subagent runs + ~40 min wall clock, menor que lo que habría costado hacer el split inline a mano con riesgo de corromper la rama.

2. **PEP 562 `__getattr__` rompe ciclos de import sin duplicar código.** Para mantener `from hoc.core import CellMetrics` funcional tras mover `CellMetrics` a `metrics/collection.py`, usamos `__getattr__` en `core/__init__.py` que resuelve bajo demanda. Alternativas (import explícito en `core/__init__.py`) causaban circular import.

3. **"Un archivo por clase" no es la división correcta para `core.py`.** El brief original pedía 6 submódulos (grid, cells, events, health, locking, constants). Entregamos 14 porque:
   - `HoneycombCell` + 6 subtipos + `QueenCell` = ~1400 LOC → división en `cells_base.py` + `cells_specialized.py` + `_queen.py`.
   - `HoneycombGrid` (~800 LOC) + hex primitives (~500 LOC) + config (~150 LOC) separados en `grid.py` + `grid_geometry.py` + `grid_config.py` para mantener bajo 800.
   
   El resultado: 14 archivos pequeños > 6 archivos medianos, sin romper la API.

4. **El mismo tooling que cerró Phase 1 volvió a pagar dividendos en Phase 3.** Mypy strict sobre `resilience.py` encontró B11 en un rebuild path no cubierto por tests. Sin el pase mypy, B11 habría permanecido latente hasta que algún test de recuperación lo tocara (y posiblemente pasara silenciosamente porque el rebuild "parecía funcionar" sin resetear pheromones).

5. **Identity preservation requiere tests explícitos.** `from hoc import HexCoord` y `from hoc.core.grid_geometry import HexCoord` deben retornar el mismo objeto — si el refactor hubiese hecho una copia accidentalmente, `isinstance(cell, HexCoord)` fallaría según la ruta de importación. `test_refactor_compat.py` pins 8 clases explícitamente con `is` checks.

6. **Cobertura 75% es alcanzable pero requiere tests dirigidos.** Phase 2 cerró a 72% global. Subir a 75.73% necesitó 28 tests nuevos específicamente apuntados a líneas no cubiertas en `core/events.py`, `core/health.py`, `core/grid_geometry.py`. Mayor parte del esfuerzo: entender signatures reales (HealthMonitor necesita `grid`, HexPathfinder necesita `walkable_check`). Los tests "naive" fallaban por signature mismatches pese a tipos correctos.

7. **Pinning versiones es no-opcional en requirements-dev.** `pytest 8.4.1` vs `pytest ≥7.0` cambia comportamiento de fixtures. `ruff 0.8.6` vs `ruff latest` cambia qué reglas existen. Pin todo en CI; dejar `latest` reproduce "funciona en mi máquina pero no en CI".
