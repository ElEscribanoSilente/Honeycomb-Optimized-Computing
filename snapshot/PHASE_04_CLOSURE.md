# Phase 4 Closure — Configuración & Developer Experience (FSM integration)

**Fecha**: 2026-04-24
**Tag previsto**: `v1.4.0-phase04`
**Branch**: `phase/04-config-dx`
**PR**: (pending — abrir tras `git push`)

---

## Resumen ejecutivo

Fase 4 cerrada con la integración de [tramoya](https://pypi.org/project/tramoya/)
como motor de state machines formales para HOC. El subpaquete
`state_machines/` documenta cinco FSMs (CellState, PheromoneDeposit,
TaskLifecycle, QueenSuccession, FailoverFlow) y exporta diagramas Mermaid
deterministas a `docs/state-machines.md`. La FSM de **CellState está
wire-uped al setter de `HoneycombCell.state`** (validación de transiciones
en runtime); las otras cuatro son **declarativas-only** por trade-offs
documentados (performance / refactor-cost / behaviour-change risk). Se
añadieron **81 tests nuevos** (582 → 663) y la fase **graduó `swarm.py`
y `nectar.py` del override mypy de Phase 3 (ADR-006)** anotando 29
errores inline. Bandit y pip-audit siguen limpios. Cobertura global subió
a **76.34 %** (+0.61 pts).

Tres bugs latentes surgieron durante el wire-up de la FSM y la graduación
mypy:

- **B12** (real, runtime-impacting): `RoyalJelly.get_stats` referenciaba
  un atributo inexistente (`cmd.command` sobre un enum). Mismo patrón que
  B9 (Phase 1) y B11 (Phase 3): mypy strict captura el lookup, runtime
  habría arrojado `AttributeError`. **Arreglado.**
- **B12-bis** (deferred): `TaskState.ASSIGNED` declarado en el enum pero
  nunca asignado en producción. Decisión (eliminar vs. wire-up two-step)
  diferida a Phase 5+.
- **B12-ter** (deferred): `CellState.{SPAWNING, MIGRATING, SEALED,
  OVERLOADED}` declarados pero nunca asignados. Misma resolución pendiente.

| Métrica | Phase 3 (v1.3.0-phase03) | Phase 4 (v1.4.0-phase04) | Δ |
|---------|--------------------------|---------------------------|---|
| Tests pasando | 582 | **663** | +81 |
| Tests de FSM (nuevos) | 0 | **81** (57 unit + 16 hypothesis + 8 mermaid) | +81 |
| Cobertura global | 75.73 % | **76.34 %** | +0.61 pts ✅ |
| FSMs formalizadas | 0 | **5** (1 wired, 4 declarative) | +5 |
| docs/state-machines.md | n/a | **auto-generado** (deterministic) | new |
| ruff errores | 0 | **0** | = ✅ |
| black reformat | 0 | **0** | = ✅ |
| mypy errores (strict + state_machines + swarm/nectar graduated) | 0 | **0** | = ✅ |
| Bandit HIGH / MEDIUM / LOW | 0 / 0 / 0 | **0 / 0 / 0** | = ✅ |
| pip-audit vulnerabilidades | 0 | **0** | = ✅ |
| GitHub Actions workflows | 4 | **4** (`lint.yml` + 1 nuevo job `state-machines-doc`) | = (+1 job) |
| Módulos graduados de mypy override | — | **swarm.py, nectar.py** (vs. ADR-006 schedule) | per plan |
| Bugs latentes descubiertos | B11 | **B12** (real, fixed) + B12-bis/ter (deferred) | +1 fixed |
| Runtime dependencies pinneadas | numpy | numpy + **tramoya==1.4.0** | +1 |

---

## 4.1 Tooling — tramoya runtime dep

- `requirements.txt`: pinneado `tramoya==1.4.0`. Runtime dep, no dev,
  porque las FSMs se construyen en producción
  (`HoneycombCell.__init__` instancia un FSM por celda).
- `pyproject.toml`:
  - `[project].dependencies` += `"tramoya>=1.4.0"`.
  - `[tool.setuptools].packages` += `"hoc.state_machines"`.
  - `[[tool.mypy.overrides]]` external block: `tramoya, tramoya.*` con
    `ignore_missing_imports` (no PEP 561 marker upstream).
  - `[tool.pytest.ini_options].norecursedirs` += `state_machines`.
- Nuevo override mypy (`HOC.state_machines.*` con `ignore_errors`)
  documentado en `pyproject.toml` con razón (cwd-name inference vs.
  sys.path search). Estricto preservado vía CLI con `mypy
  --explicit-package-bases state_machines/*.py` + nuevo job CI.

---

## 4.2 `state_machines/` subpaquete + wrapper sobre tramoya

- **`state_machines/base.py`**:
  - `HocStateMachine`: adapter tipado sobre `tramoya.Machine`. Expone
    una API destination-driven (`transition_to(target)`) además de la
    trigger-driven (`trigger(name)`), porque el código pre-Phase-4 hace
    `obj.state = NEW_STATE` directamente. El wrapper resuelve qué
    trigger disparar buscando en un index `dest -> [(source, trigger)]`.
  - `HocTransition` dataclass: `source / dest / trigger? / guard? /
    action?`. Auto-genera nombres `<source>__to__<dest>` cuando no se
    especifica.
  - `IllegalStateTransition` única excepción con `fsm_name, source,
    target, reason ("no_edge" | "guard_rejected" | "unknown_state" |
    "empty_history")`. Wraps `tramoya.InvalidTransition` y
    `tramoya.GuardRejected` — los callers no necesitan importar
    tramoya.
  - Pass-throughs: `to_mermaid()`, `to_dot()`, `subscribe()`/
    `unsubscribe()`, `is_final`, `is_stuck()`, `undo()`, `reset()`,
    `available_triggers`, `history`.
- **`state_machines/__init__.py`**: re-exports de los símbolos
  públicos.
- **Misma tesis que Phase 2 con `mscs`** (`hoc.security`): tramoya se
  importa **solo** desde `state_machines/base.py`. Si se cambia la
  librería downstream en una fase futura, se toca un único archivo.

### Trade-off de tipo: `^state_machines/` excluido del `mypy .` global

`pyproject.toml [tool.mypy].exclude` añade `^state_machines/`. Razón
(documentada en pyproject + ADR-007): mypy infiere un nombre de paquete
del cwd directory (`HOC.state_machines.*` cuando el cwd es `D:\HOC`)
**y** lo descubre via sys.path search (`state_machines.*`). El error
"Source file found twice under different module names" bloquea todo el
run. Strict mypy se preserva listando los archivos explícitamente con
shell glob (`mypy --explicit-package-bases state_machines/*.py`); el
exclude solo aplica al directorio scan, no a args explícitos. Nuevo
step en `lint.yml` `mypy` job ejerce la invocación.

---

## 4.3 CellState FSM — wired into `HoneycombCell`

| Aspecto | Estado |
|---------|--------|
| Estados | 9 (mismos que `CellState` enum) |
| Transitions | 14 (9 lifecycle + 5 admin/wildcard) |
| Wired | ✅ `HoneycombCell.state.setter` → `_fsm.transition_to(new.name)` |
| Initial state | `EMPTY` |

### Decisión cardinal: estados reales, no aspiracionales

El brief original proponía `INITIALIZING / IDLE / BUSY / DEGRADED /
FAILED / RECOVERING`. La realidad de `core/cells_base.py:51-62`:

```python
class CellState(Enum):
    EMPTY = auto()
    ACTIVE = auto()
    IDLE = auto()
    SPAWNING = auto()
    MIGRATING = auto()
    FAILED = auto()
    RECOVERING = auto()
    SEALED = auto()
    OVERLOADED = auto()
```

Mapping real ↔ brief:

| Brief | Realidad | Notas |
|-------|----------|-------|
| INITIALIZING | EMPTY | sin vCores |
| IDLE | IDLE | ✓ |
| BUSY | ACTIVE | rename, semántica equivalente |
| DEGRADED | *no existe* | `ACTIVE → FAILED` directo cuando circuit-breaker abre |
| FAILED | FAILED | ✓ |
| RECOVERING | RECOVERING | ✓ |
| *no en brief* | SPAWNING, MIGRATING, SEALED, OVERLOADED | declarados pero **nunca asignados** (B12-ter) |

Per el principio cardinal de Phase 4 ("FSMs **observan** transiciones
existentes; no las **cambian**"), la FSM modela los estados/transitiones
reales:

#### Lifecycle (source explícito)

| Source | Dest | Trigger | Call-site |
|--------|------|---------|-----------|
| EMPTY | IDLE | `vcore_added` | cells_base.py:267 (add_vcore) |
| IDLE | EMPTY | `vcore_drained_idle` | cells_base.py:288 (remove_vcore) |
| ACTIVE | EMPTY | `vcore_drained_active` | cells_base.py:288 (remove_vcore from ACTIVE) |
| IDLE | ACTIVE | `tick_started` | cells_base.py:395 (execute_tick) |
| ACTIVE | IDLE | `tick_completed` | cells_base.py:434 (execute_tick success) |
| ACTIVE | FAILED | `tick_failed` | cells_base.py:411 (circuit open) |
| FAILED | RECOVERING | `recovery_started` | cells_base.py:449 (recover) |
| RECOVERING | EMPTY | `recovery_completed` | cells_base.py:456 (recover end) |
| RECOVERING | IDLE | `recovery_restored` | resilience.py:1134 (CombRepair) |

#### Admin / failover (source = wildcard)

| Source | Dest | Trigger | Call-site |
|--------|------|---------|-----------|
| `*` | FAILED | `admin_mark_failed` | resilience.py:332, 722 + tests |
| `*` | IDLE | `admin_set_idle` | resilience.py:346, 1134, 1163, 1389 |
| `*` | RECOVERING | `admin_recover` | resilience.py:1128, 1140 |
| `*` | EMPTY | `admin_reset` | resilience.py:1154 (CombRepair) |
| `*` | ACTIVE | `admin_force_active` | tests/test_resilience.py:657, 704 |

#### B12-ter — dead states

`SPAWNING`, `MIGRATING`, `SEALED`, `OVERLOADED` no son destino de ninguna
transición en la FSM (porque ningún call-site los asigna). Si un PR
futuro escribiera `cell.state = CellState.SPAWNING`, la FSM levantaría
`IllegalStateTransition(reason="no_edge")` — exactamente la canary que
queremos. Decisión (eliminar del enum vs. wire-up de la lógica que
debería usarlos) **diferida** a Phase 5+.

---

## 4.4 / 4.5 / 4.6 — Cuatro FSMs declarativas

Por trade-offs específicos (documentados en cada módulo y en ADR-007),
las cuatro FSMs restantes ship como **declarativas-only**: la spec se
captura para documentación + property tests + Mermaid export, pero el
código de producción **no** atraviesa `transition_to`.

| FSM | Estados | Transiciones | Razón declarativa-only |
|-----|---------|--------------|------------------------|
| **PheromoneDeposit** (4.5) | 4 (FRESH/DECAYING/DIFFUSING/EVAPORATED) | 5 con guards | ~90k objects/trail; per-instance FSM o validador con lock excede el budget de overhead |
| **TaskLifecycle** (4.4a) | 5 (PENDING/RUNNING/COMPLETED/FAILED/CANCELLED) | 6 | `HiveTask.state` mutado desde ~15 call-sites en swarm.py + tests; wire-up via `__setattr__` requiere reescribir tests o relajar la FSM con wildcards (drena valor de validación) |
| **QueenSuccession** (4.4b) | 6 (STABLE/DETECTING/NOMINATING/VOTING/ELECTED/FAILED) | 9 | resilience.py mantiene un único bool `_election_in_progress`; las 6 fases son posiciones dentro de `_conduct_election`. Wire-up requiere split del método + re-validar los 7 tests `TestQuorumSignedVotes` (security-critical) |
| **FailoverFlow** (4.6) | 5 (HEALTHY/DEGRADED/MIGRATING/RECOVERED/LOST) | 6, undo en MIGRATING | `CellFailover` no mantiene fase per-cell; wire-up de `undo()` requiere secuenciar rollback dentro del try/except existente. Mejor alineado con el split de resilience.py per ADR-006 (Phase 5+) |

### B12 (TaskLifecycle) — ASSIGNED dead state

`TaskState.ASSIGNED` declarado en swarm.py:85 entre PENDING y RUNNING.
**Ningún call-site escribe `task.state = TaskState.ASSIGNED`** — workers
hacen `PENDING → RUNNING` directamente cuando reclaman tareas
(swarm.py:308, 382, 461, 531). El brief proponía
`PENDING → ASSIGNED → RUNNING`; la FSM modela el flujo real
(`PENDING → RUNNING`).

### Valor de las FSMs declarativas

A pesar de no ser load-bearing en runtime:

- `docs/state-machines.md` documenta cada lifecycle para nuevos
  contribuyentes.
- Property tests con Hypothesis validan invariantes del grafo
  (estados terminales, retry path, etc.).
- Trigger names mapean a call-sites — el wire-up futuro es mecánico.
- `tests/test_state_machines.py` ejerce explícitamente:
  - **TaskLifecycle**: COMPLETED y CANCELLED son terminales; FAILED no
    (retry).
  - **QueenSuccession**: VOTING → ELECTED bloqueada sin
    `quorum_reached + signatures_valid + term_matches` (refleja el
    quorum-criptográfico de Phase 2).
  - **FailoverFlow**: undo en MIGRATING devuelve a DEGRADED.
  - **PheromoneDeposit**: EVAPORATED es terminal; intensity threshold
    fuerza la transición.

---

## 4.7 Mermaid export

- **`scripts/generate_state_machines_md.py`**: itera el registro de
  FSMs (5 builders), exporta `to_mermaid()` cada uno, escribe
  `docs/state-machines.md` con índice + sección por FSM. Modos: write,
  `--stdout`, `--check` (drift detector usado por CI).
- **Output determinista** byte-a-byte para una spec dada.
  `tests/test_mermaid_export.py::TestDeterminism::test_back_to_back_runs_match`
  pins el contrato.
- Lazy-import de cada builder: el script corre durante el wire-up
  incremental (cada FSM aparece tan pronto como su módulo existe).
- **Drift detector**: nuevo job `state-machines-doc` en `lint.yml` corre
  `--check`, falla si `docs/state-machines.md` no coincide con la spec
  actual. Mismo contrato que el test local.

---

## 4.10 Graduación ADR-006 — `swarm.py` + `nectar.py` mypy strict

Per ADR-006 schedule, Phase 4 graduó ambos módulos del
`[[tool.mypy.overrides]]` `ignore_errors=true`.

| Módulo | Errores expuestos | Anotaciones |
|--------|-------------------|-------------|
| swarm.py | 11 | `__post_init__` returns, dict generics, defaultdict generics, `Callable[[Any], None]`, float-vs-int initializers |
| nectar.py | 18 | `cast(bytes, _mscs.dumps(...))` (3×), `dict[str, Any]`, defaultdict generics, `**kwargs: Any`, `_queue: deque[Any]`, **B12 fix** |

### B12 (real, runtime-impacting)

Mypy strict sobre `nectar.py:1174` detectó:

```python
"commands_by_type": {
    cmd.command.name: sum(
        1 for c in self._pending_commands if c.command == cmd.command
    )
    for cmd in RoyalCommand  # ← cmd is a RoyalCommand enum value
},
```

`cmd` itera los miembros del enum — no tiene atributo `.command`.
`RoyalJelly.get_stats()` habría arrojado `AttributeError` la primera vez
que se llamara. No fue capturado en tests porque ningún test ejerce esa
ruta. Mismo patrón que **B9** (Phase 1 metrics.py) y **B11** (Phase 3
resilience.py): mypy strict expone fallos de attribute lookup que tests
no cubren.

**Fix**: `cmd.command.name` → `cmd.name`, y `c.command == cmd.command` →
`c.command == cmd`. Comportamiento original (agrupar pending commands
por su valor RoyalCommand) preservado.

---

## Auditorías

### Seguridad — Bandit (`snapshot/bandit_phase04.json`)

```
LOC scanned:     9,896
Files scanned:   31
SEVERITY HIGH:   0
SEVERITY MEDIUM: 0
SEVERITY LOW:    0
```

Phase 2 redujo a 0/0/0; Phase 3 mantuvo; Phase 4 mantiene.

### Vulnerabilidades de dependencias — pip-audit (`snapshot/pip_audit_phase04.txt`)

```
No known vulnerabilities found  (runtime: numpy + tramoya)
No known vulnerabilities found  (dev: pytest, ruff, black, mypy, bandit, pip-audit, ...)
```

`tramoya==1.4.0` (nueva runtime dep) limpia.

### Complejidad — Radon (`snapshot/radon_cc_phase04.txt`)

Funciones con CC > 10 (todas legacy, **no nuevas en Phase 4**):

| Archivo | Función | Razón legacy |
|---------|---------|--------------|
| nectar.py | `PheromoneTrail.evaporate` | hot path original |
| nectar.py | `PheromoneTrail.diffuse_to_neighbors` | hot path |
| nectar.py | `WaggleDance.propagate` | hot path |
| resilience.py | `CellFailover.find_failover_target` | n-way candidate selection |
| swarm.py | `SwarmScheduler.tick` | scheduler hot loop |
| swarm.py | `ForagerBehavior.select_task` | scoring + tie-break |
| swarm.py | `SwarmBalancer.execute_work_stealing` | n-neighbor scan |
| core/grid.py | `HoneycombGrid.tick` | per-cell loop |
| core/grid.py | `HoneycombGrid.visualize_ascii` | rendering |
| core/grid.py | `HoneycombGrid.get_stats` | aggregation |

Average CC: **C (13.6)** — mismo nivel que Phase 3 (radon agora ve
nectar/swarm que antes estaban en mypy override; los hot paths no
fueron tocados por la integración FSM).

### LOC per archivo (Radon raw, `snapshot/radon_raw_phase04.txt`)

Archivos > 800 LOC sin cambios significativos vs Phase 3:

| Archivo | LOC | Phase 3 | Phase 4 |
|---------|-----|---------|---------|
| resilience.py | 1,639 | 1,639 | 1,639 |
| nectar.py | ~1,378 | 1,366 | +12 (annotations + B12 fix) |
| swarm.py | ~1,138 | 1,132 | +6 (annotations) |
| memory.py | 940 | 940 | 940 |
| bridge.py | 886 | 886 | 886 |

Ningún archivo nuevo de Phase 4 supera 800 LOC. El mayor nuevo es
`state_machines/base.py` (~395 LOC) y `tests/test_state_machines.py`
(~430 LOC).

### Cobertura (`pytest --cov`)

| Módulo | Phase 3 | Phase 4 | Δ | Notas |
|--------|---------|---------|---|-------|
| `state_machines/__init__.py` | — | 100% | new | |
| `state_machines/base.py` | — | 93% | new | wrapper API |
| `state_machines/cell_fsm.py` | — | 100% | new | |
| `state_machines/{pheromone, task, succession, failover}_fsm.py` | — | 100% c/u | new | |
| `core/cells_base.py` | (parte de core/ 51%) | (mismo) | = | wire-up de FSM cubierto |
| `nectar.py` | 73% | 73% | = | annotations no añaden líneas |
| `swarm.py` | 88% | 88% | = | |
| **Global** | **75.73%** | **76.34%** | **+0.61 pts** | sobre threshold 75% |

### Benchmark — `snapshot/bench_phase04.json`

`pytest benchmarks/ --benchmark-only` corrió 11 benchmarks (3 skipped),
captura archivada en `snapshot/bench_phase04.json`. Means clave:

| Benchmark | Mean (ms) |
|-----------|-----------|
| `test_grid_creation` | 1.85 ms |
| `test_grid_tick` | 0.46 ms |
| `test_nectar_flow_tick` | 0.01 ms |

**No hay snapshot/bench_phase03.json** — Phase 3 cerró con benchmark
diferido (Gap 4 de PHASE_03_CLOSURE.md). Por lo tanto **no se puede
calcular un % de overhead vs. Phase 3 baseline**. Interpretación cualitativa:
Phase 4 toca paths hot mínimamente (solo `_set_state` en cells_base.py
añade un `transition_to` por cambio de estado de celda; las otras 4
FSMs son declarativas-only). El path crítico de
`HoneycombGrid.tick → HoneycombCell.execute_tick → _set_state` es ahora
2 transitions FSM por celda activa, lo cual con tramoya
(~300 LOC, dispatch O(transitions con ese trigger)) es del orden de
microsegundos.

**Acción para Phase 5**: Phase 5 captura un baseline reproducible para
poder medir overhead de futuras integraciones de FSMs declarativas
cuando se wire-upen.

---

## Definition of Done — verificación

| Ítem | Estado | Nota |
|------|--------|------|
| tramoya pinneado en requirements.txt + pyproject.toml + mypy override | ✅ | `tramoya==1.4.0`, ADR-007 documenta integración |
| `state_machines/` subpaquete con 5 FSMs + base.py | ✅ | 1 wired + 4 declarativas |
| CellState FSM wired en HoneycombCell, transiciones ilegales rechazadas | ✅ | `cell.state = CellState.SPAWNING` levanta `IllegalStateTransition` |
| TaskLifecycle FSM modelada | ✅ declarativa | wire-up gap (B12 ASSIGNED) |
| QueenSuccession FSM modelada, 7 tests quorum siguen pasando | ✅ declarativa | `TestQuorumSignedVotes` 7/7 verde |
| PheromoneDeposit FSM modelada | ✅ declarativa | gap por performance budget |
| FailoverFlow FSM modelada con edges undo | ✅ declarativa | gap; undo en MIGRATING testeado |
| docs/state-machines.md auto-generado y en git | ✅ | 5 FSMs, determinista |
| Script generate_state_machines_md.py + drift CI | ✅ | nuevo job `state-machines-doc` en lint.yml |
| `hoc-cli` instalado | ⚠️ **DEFERRED** | 4.9 deprioritized; ningún path crítico lo necesita en Phase 4 |
| Config system con load order docs/yaml/env/programmatic | ⚠️ **DEFERRED** | 4.8 deprioritized; HOC_HMAC_KEY existente sigue funcionando vía os.environ |
| swarm.py + nectar.py graduados de mypy override | ✅ | 29 anotaciones inline, **B12 real fixed** |
| Tests: state_machines (57) + property (16) + mermaid (8) | ✅ | 81 nuevos = 663 total |
| 582 tests Phase 3 siguen pasando | ✅ | sin cambios en tests pre-existentes |
| Cobertura ≥ 78% global | ⚠️ **76.34%** | +0.61 pts; objetivo 78% no alcanzado (FSMs nuevas son código nuevo + bien testeado, pero el resto del repo sigue dominando el promedio) |
| Bandit/pip-audit siguen limpios | ✅ | 0/0/0 + 0 |
| Overhead bench <5% | ⚠️ **inmedible** | Phase 3 no capturó baseline (Gap 4 de Phase 3) |
| 30+ CI jobs siguen verdes | ⚠️ **post-push** | local audits ✅; CI verde se valida en el PR |

### Items deferred

- **4.8 (Config system)** y **4.9 (CLI)**: el usuario explícitamente
  autorizó posponer si el budget de tiempo aprieta ("son
  independientes de las FSMs y pueden esperar"). Phase 5 (observability)
  o Phase 6 son hosts naturales para ambos.
- **4.11 (Split swarm/nectar)**: deferred. La graduación mypy ya hizo
  el trabajo más difícil; el split puro queda más natural cuando
  resilience.py también entra en Phase 5.

---

## Gaps diferidos (documentados para fases siguientes)

### Gap 1: Wire-up de las 4 FSMs declarativas

| FSM | Phase objetivo | Razón |
|-----|----------------|-------|
| PheromoneDeposit | profilable / Phase 6 | Necesita pasar por profiling de evaporate hot path antes de añadir overhead |
| TaskLifecycle | Phase 5 | Junto con observability (cada transition de tarea debería ser un span) |
| QueenSuccession | Phase 5 | Junto con split de resilience.py |
| FailoverFlow | Phase 5 | Junto con split de resilience.py |

### Gap 2: B12-bis y B12-ter (dead states)

| Bug | Resolución pendiente |
|-----|----------------------|
| B12-bis: `TaskState.ASSIGNED` nunca asignado | Phase 5: o eliminar del enum, o wire-up de `PENDING → ASSIGNED → RUNNING` two-step en SwarmScheduler |
| B12-ter: `CellState.{SPAWNING, MIGRATING, SEALED, OVERLOADED}` nunca asignados | Phase 5+: per estado, decidir si eliminar o wire-up del callsite faltante (e.g., MIGRATING podría usarse durante CellFailover.migrate_cell) |

### Gap 3: Benchmark baseline reproducible

Phase 3 no capturó `snapshot/bench_phase03.json`; Phase 4 captura
`snapshot/bench_phase04.json` pero no puede calcular overhead vs.
Phase 3. **Acción Phase 5**: capturar baseline + diff workflow.

### Gap 4: Coverage 78% objective no alcanzado

Phase 4 cierra a 76.34%. Para llegar a 78% sin reducir threshold se
requiere cobertura adicional sobre `nectar.py` (73%) y `bridge.py` (56%).
Diferido a Phase 5 (test boost + posible split).

### Gap 5: 6 funciones legacy con CC > 10 (heredado de Phase 3)

Sin cambios. Phase 5 (observability) o split de Phase 5 (resilience)
son los hosts naturales para refactor extract-method.

---

## Lecciones aprendidas

1. **El principio cardinal "FSMs observan, no proponen" se confirmó
   inmediatamente.** El brief proponía estados aspiracionales para
   `CellState` (`INITIALIZING / IDLE / BUSY / DEGRADED`) pero el código
   real usa `EMPTY / IDLE / ACTIVE / ...` sin DEGRADED. Auditar
   call-sites antes de escribir la FSM, en lugar de seguir la spec del
   brief, evitó romper 582 tests existentes.

2. **El boundary pattern de Phase 2 (`mscs` aislado en `hoc.security`)
   funcionó igualmente bien para tramoya.** Un solo módulo
   (`state_machines/base.py`) toca tramoya; futuros cambios de upstream
   no se filtran al resto del repo.

3. **mypy strict sigue encontrando bugs latentes que tests no cubren.**
   B9 (Phase 1, metrics.py), B11 (Phase 3, resilience.py), y ahora
   **B12** (Phase 4, nectar.py:1174) — todos detectados durante
   graduación o anotación, todos del mismo patrón "attribute lookup
   contra un campo que no existe en el tipo". Confirma la tesis de
   ADR-006 de graduar gradualmente.

4. **"Wire-up real" no siempre es la elección correcta.** Para 4 de las
   5 FSMs, hacerlas declarativas-only es honesto: el código de
   producción no tiene un campo de estado donde la FSM pueda enchufarse
   sin invadir, y forzarlo conflictaría con tests pre-Phase-3 o
   degradaría hot paths. La FSM declarativa entrega aún:
   documentación, property tests, Mermaid diagrams, y prepara el
   wire-up futuro.

5. **El layout dual del package (`core` top-level + `hoc.core`) sigue
   siendo una fuente de fricción.** mypy infiere paquete del cwd
   directory; cuando un módulo se importa desde dos rutas, el "Source
   file found twice" bloquea. Solución (`exclude` + `--explicit-package-
   bases` + lista explícita de archivos via shell glob) funciona pero
   es delicada — documentar fuertemente en pyproject.toml + ADR-007
   reduce fricción para futuros contribuyentes.

6. **Property tests con Hypothesis tienen alto valor para FSMs.** Los
   16 tests de `test_state_machines_property.py` cubren combinatorias
   que tests unitarios no podrían en tiempo razonable: random walks de
   transiciones, terminal-state invariantes, no-orphan-states. Por el
   precio de 16 tests cortos, Hypothesis exhibe ~80 ejemplos por
   ejecución; es como tener ~1280 tests aleatorios cada vez.

7. **Capturar el baseline de bench la primera vez es no-opcional.**
   Phase 3 difirió el bench y Phase 4 paga el costo: no puedo decir si
   la integración FSM cumple `<5%` overhead. Phase 5 debe capturar
   `bench_phase05.json` aún si no implementa nada que afecte
   performance — solo para tener el baseline reproducible.
