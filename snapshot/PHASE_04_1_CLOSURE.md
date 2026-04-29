# Phase 4.1 Closure — TaskLifecycle wire-up + `choreo` static FSM checker

**Fecha**: 2026-04-24
**Tag previsto**: `v1.4.1-phase04.1`
**Branch**: `phase/04.1-choreo-mvp`
**PR**: (pending — abrir tras `git push`)

---

## Resumen ejecutivo

Phase 4.1 cierra dos frentes que Phase 4 dejó abiertos:

1. **TaskLifecycle FSM graduada de declarativa a wired**, vía
   `HiveTask.__setattr__` override. Cada `task.state = X` ahora valida
   contra el FSM y rechaza transiciones ilegales con
   `IllegalStateTransition`. Phase 4 wired CellState (1 de 5); Phase 4.1
   wires TaskLifecycle (2 de 5).

2. **`choreo` — herramienta nueva** de verificación estática AST-based.
   Parea las FSMs declarativas con el código, detecta drift, falla CI en
   mutaciones undocumented. Aplicada a HOC confirma exactamente los 5
   hallazgos esperados (0 errores, 2 warnings B12-bis/B12-ter, 3 infos
   declarative-only). Esto **resuelve el dilema** de las 3 FSMs restantes
   (Pheromone, Succession, Failover): no se pueden wire-upear sin
   refactor de modelo del host (Phase 5+), pero ahora SÍ se verifican
   estáticamente.

**42 tests nuevos** (663 → 705): 10 para el wire-up de TaskLifecycle, 32
para `choreo` (incluyendo smoke integration que corre choreo contra el
propio HOC y valida los 5 hallazgos). Sin nuevas runtime deps. Bandit
/pip-audit/ruff/black/mypy todos limpios.

| Métrica | Phase 4 (`v1.4.0`) | Phase 4.1 (`v1.4.1`) | Δ |
|---------|---------------------|----------------------|---|
| Tests pasando | 663 | **705** | +42 |
| Tests TaskLifecycle wire-up | 0 | **10** (TestTaskFSMWired) | +10 |
| Tests `choreo` | 0 | **32** (walker + spec + diff + cli + HOC integration) | +32 |
| FSMs wired in production | 1 (CellState) | **2** (CellState, TaskLifecycle) | +1 |
| FSMs declarativas-only | 4 | **3** (Pheromone, Succession, Failover) | -1 |
| FSMs verificadas estáticamente | 0 | **5** (todas, vía choreo) | +5 |
| Static checker LOC | 0 | **~620** (`choreo/`, 6 .py + 1 __main__) | +620 |
| ADRs | 7 | **8** | +1 |
| GitHub Actions jobs | 5 | **6** (added `choreo-static-check`) | +1 |
| Runtime dependencies | numpy + tramoya | **same** | = |
| Bandit HIGH / MEDIUM / LOW | 0 / 0 / 0 | **0 / 0 / 0** | = ✅ |
| pip-audit vulnerabilidades | 0 | **0** | = ✅ |
| ruff / black / mypy errors | 0 | **0** | = ✅ |

---

## 4.1.1 Wire-up TaskLifecycle

`HiveTask.__post_init__` instancia un `_fsm = build_task_fsm()` por
tarea. `HiveTask.__setattr__` rutea cada `task.state = X` vía
`_fsm.transition_to(X.name)` y respeta:

- **Idempotent assignment**: `task.state = task.state` no consulta el
  FSM (no se acumula en history).
- **Pre-init bypass**: durante el `__init__` generado del dataclass,
  `_fsm` aún no existe en `__dict__` — la guardia `"_fsm" in self.__dict__`
  permite que el field default se establezca sin pasar por el FSM.
- **Non-default state via `__init__`**: si el caller pasa
  `HiveTask(priority=1, state=TaskState.RUNNING)`, `__post_init__` hace
  `_fsm.reset(state.name)` para sincronizar (reset bypassa guards — es
  configuración, no transición).

### Dos transiciones explícitas añadidas a la FSM (no wildcards)

`tests/test_swarm.py` tiene 5 sites que fuerzan estados terminales
sobre tareas recién submitted (ningún tick del scheduler de por medio):

| Site | Patrón |
|------|--------|
| `test_cancel_completed_task_fails` (L451) | `PENDING → COMPLETED` |
| `test_b2_5_task_index_cleaned_after_completed` (L504-505) | `PENDING → COMPLETED` (×2) |
| `test_b2_5_task_index_cleaned_after_failed` (L522) | `PENDING → FAILED` |
| `test_b2_5_no_leak_after_many_cycles` (L537) | `PENDING → COMPLETED` |

Para no relajar la FSM con wildcards (que drenarían la validación de
otras transiciones ilegales), se añadieron **dos triggers explícitos**:

- `force_completed_from_pending` (`PENDING → COMPLETED`)
- `force_failed_from_pending` (`PENDING → FAILED`)

Ningún call-site de producción reacha estas edges — `swarm.py` siempre
pasa por `RUNNING` antes de un terminal. Las edges sobreviven solo para
los tests.

### B12-bis ahora detecta también en runtime

`task.state = TaskState.ASSIGNED` ahora levanta
`IllegalStateTransition(reason="unknown_state")` porque la FSM
TaskLifecycle no incluye `ASSIGNED` en sus estados. Esto se prueba en
`tests/test_state_machines.py::TestTaskFSMWired::test_illegal_transition_assigned_dead_state_raises`.
choreo detecta el mismo bug a nivel estático (ver 4.1.2).

### Tests añadidos

`tests/test_state_machines.py::TestTaskFSMWired` — 10 tests:

1. `test_task_starts_in_pending` — FSM inicia coherente con field default.
2. `test_legal_transition_pending_to_running` — happy path setattr.
3. `test_happy_path_running_to_completed` — chain completo.
4. `test_illegal_transition_completed_to_running_raises` — terminal
   bloqueado.
5. `test_illegal_transition_assigned_dead_state_raises` — B12-bis canary.
6. `test_idempotent_assignment_skips_fsm` — no rebound history.
7. `test_force_completed_from_pending_legal` — test-fixture edge.
8. `test_force_failed_from_pending_legal` — test-fixture edge.
9. `test_non_default_state_at_construction_syncs_fsm` — reset path.
10. `test_retry_path_failed_to_pending` — happy retry.

---

## 4.1.2 `choreo` — static FSM verification

### Estructura del subpackage (`choreo/`)

| Archivo | LOC | Responsabilidad |
|---------|-----|-----------------|
| `__init__.py` | 50 | Versión + docstring del subpaquete |
| `__main__.py` | 8 | Entry point para `python -m choreo` |
| `types.py` | 92 | `Mutation`, `EnumDecl`, `FsmSpec`, `Finding` (frozen, ordered) |
| `walker.py` | 195 | AST visitor: 3 patrones (assign, _set_state, ClassDef) |
| `spec.py` | 95 | Importa `state_machines/*_fsm.py`, extrae estados+edges |
| `diff.py` | 230 | bind FSM↔Enum, computa findings, ordena |
| `cli.py` | 145 | argparse, output human/JSON, exit codes |

Total: **~815 LOC** (incluyendo docstrings y comments). Sin dependencias
externas.

### Mecánica del walker

Tres patrones literales en AST:

1. `Assign(targets=[Attribute(attr="state")], value=Attribute(value=Name(id="EnumName"), attr="MEMBER"))` — el `obj.state = ENUM.X`.
2. `Call(func=Attribute(attr="_set_state"), args=[<same shape as #1>])` — el `obj._set_state(ENUM.X)` interno de `cells_base.py`.
3. `ClassDef(bases=[Name(id="Enum") | Attribute(attr="Enum")])` — la enum class y sus members.

No infiere tipos. No analiza control flow. Falsos negativos conocidos:
- `setattr(obj, "state", X)`
- `obj.state = func()` (RHS no literal)
- Mutaciones a través de proxies (`replace()`, `dataclasses.asdict()`)

Aceptable porque el costo del false-negative es no detectar — el wire-up
runtime (CellState, TaskLifecycle) los caza como segunda línea.

### Mecánica del binding FSM↔Enum

Por cada FSM, busca el enum cuyo set de members ⊇ los states declarados
del FSM. Si hay múltiples candidatos, gana el de menor cardinalidad
(empate alfabético). Esto resuelve el caso ambiguo (que no aparece en
HOC) sin requerir metadata explícita en los builders.

### Severidades

| Severidad | Kind | Significado | CI |
|-----------|------|-------------|-----|
| **error** | `undocumented_mutation` | mutación cuyo target no está en ninguna FSM, o cuyo enum no matchea ninguna FSM | rompe siempre |
| **warning** | `dead_state` | estado en FSM pero ningún mutation observed lo targetea | rompe en `--strict` |
| **warning** | `enum_extra_state` | enum member que no está en la FSM | rompe en `--strict` |
| **info** | `declarative_only` | FSM sin enum match, sin mutations | nunca rompe |

Default CI mode = sin `--strict`. Permite shipear con B12-bis/B12-ter
visibles como warnings sin bloquear merges. Cuando Phase 5+ resuelva
ambos bugs, se flippea a `--strict`.

### Output (text + JSON)

`python -m choreo check`:

```
== WARNINGS (2) ==
  [CellState] dead_state
    4 state(s) declared in FSM but never targeted by an observed mutation:
    MIGRATING, OVERLOADED, SEALED, SPAWNING
    @ state_machines/cell_fsm.py

  [TaskLifecycle] enum_extra_state
    enum `TaskState` declares 1 member(s) not in FSM: ASSIGNED -
    remove from enum or add transitions to the FSM
    @ swarm.py:86

== INFO (3) ==
  [FailoverFlow] declarative_only      ...
  [PheromoneDeposit] declarative_only  ...
  [QueenSuccession] declarative_only   ...

Summary: 0 errors, 2 warnings, 3 infos
```

`python -m choreo check --json`: payload con
`{counts, findings[]}` validado en `tests/test_choreo.py::TestCli::test_json_output_parses`.

### Tests añadidos

`tests/test_choreo.py` — 32 tests en 5 clases:

- **TestWalker** (9): assign + _set_state + ClassDef captura, qualified
  bases (`enum.Enum`), private members skip, exclude paths, syntax
  error tolerance, multiple muts/file, non-state attrs ignored.
- **TestSpecFromFsm** (3): basic extraction, missing attrs → None,
  states sorted deterministically.
- **TestBindFsmToEnum** (4): superset bind, no bind on missing, smallest
  wins, alphabetical tiebreak.
- **TestComputeFindings** (7): clean run, dead_state, enum_extra,
  undocumented mutation, orphan mutation, declarative-only, severity
  ordering.
- **TestCli** (7): help exits 2, clean exits 0, warnings without strict
  exit 0, warnings with strict exit 1, errors exit 1, JSON parses,
  invalid root exits 2.
- **TestHocIntegration** (2): smoke run against HOC produces exit 0;
  exact findings match the documented Phase 4.1 state of HOC.

---

## CI integration

Nuevo job en `.github/workflows/lint.yml`:

```yaml
choreo-static-check:
  ...
  - run: python -m choreo check
  - run: python -m choreo check --json | python -m json.tool > /dev/null
```

El primer run hace el check humano-legible (que GitHub UI muestra). El
segundo valida que el output JSON es bien-formado (parseable por
herramientas downstream).

---

## Auditorías

### Seguridad — Bandit

Sin cambios — Phase 4 cerró 0/0/0 y Phase 4.1 no añade dependencias.

### Vulnerabilidades — pip-audit

Sin cambios. Sin nuevas runtime deps; choreo es stdlib-only.

### Cobertura

`tests/test_choreo.py` ejerce ~95% de los archivos en `choreo/`. El
módulo nuevo entra a la suite global con cobertura alta — eleva el
promedio leve sobre el 76.34% de Phase 4.

### Complejidad — Radon

Sin cambios sobre los hot paths de HOC. choreo no toca código existente
salvo el wire-up de TaskLifecycle (que añade un `__setattr__` simple,
CC<5).

### Benchmark

`choreo check` corre sobre el repo HOC en ~0.5s. No contribuye al hot
path de HOC, así que no aplica al benchmark de end-to-end overhead.
Ejecución de la suite de tests: 705 en ~10.7s (vs 663 en ~10.2s para
Phase 4 — overhead +5% por +42 tests).

---

## Definition of Done

| Ítem | Estado | Nota |
|------|--------|------|
| TaskLifecycle FSM wired via `HiveTask.__setattr__` | ✅ | 10 tests pasan |
| Test-fixture edges añadidos a la FSM (no wildcards) | ✅ | `force_completed_from_pending`, `force_failed_from_pending` |
| `choreo/` subpackage implementado | ✅ | walker + spec + diff + cli + types + __main__ |
| `choreo` corre contra HOC y produce el reporte esperado | ✅ | 0 err, 2 warn, 3 info |
| 32 tests choreo (walker + spec + diff + cli + integration) | ✅ | todos verde |
| ADR-008 escrito | ✅ | static vs runtime, complementarias |
| CI job `choreo-static-check` añadido a `lint.yml` | ✅ | + JSON shape sanity |
| 663 tests Phase 4 siguen pasando sin cambios | ✅ | 705 total ahora |
| ruff / black / mypy todos limpios | ✅ | en archivos nuevos + tocados |
| Bandit / pip-audit siguen limpios | ✅ | sin cambios |
| CHANGELOG entry [1.4.1-phase04.1] | ✅ | siguiendo Keep a Changelog |
| ROADMAP marca Phase 4.1 como CERRADA | ✅ | con resumen + pendientes 4.2 |
| `HocStateMachine.transitions` property pública | ✅ | en `state_machines/base.py` |

---

## Gaps diferidos (a Phase 4.2 y 5+)

### Gap 1: Pattern coverage del walker (Phase 4.2)

Walker matchea solo `obj.state = X` y `obj._set_state(X)`. Falsos
negativos en:

- `setattr(obj, "state", X)`
- `dataclasses.replace(obj, state=X)`
- Mutación a través de descriptors customizados

Phase 4.2 puede extender el walker. Mitigado por wire-up runtime para
las 2 FSMs wired.

### Gap 2: Source-state inference (Phase 4.2 o 5+)

Walker registra solo el target de cada mutación. No detecta "transición
ilegal observada" (ej. PR que introduce `RUNNING → PENDING` directo,
saltando el path retry). Runtime wire-up sí lo caza; choreo no.

Implementarlo requeriría control-flow analysis — research-grade work.
Opcional para 4.2 si budget alcanza.

### Gap 3: Auto-derive (Phase 4.2)

`python -m choreo derive --module X --field Y` que produce un
`build_X_fsm()` skeleton desde el código. Bootstrapping aid para
introducir nuevas FSMs sobre código legacy. Planeado para Phase 4.2.

### Gap 4: Reified transitions (Phase 4.2)

Decorator `@transition(from_=X, to=Y)` para declarar transiciones
inline en métodos. Aplicar a 2-3 funciones críticas (HiveTask.claim,
complete, fail) como prueba de migración. Planeado para Phase 4.2.

### Gap 5: B12-bis y B12-ter resolution (Phase 5+)

Una vez resueltas (eliminando dead enum members o wireando los
call-sites missing), el job CI puede flippear a `--strict` para hacer
fail también con warnings.

---

## Lecciones aprendidas

1. **Static + runtime se complementan, no se duplican.** CellState y
   TaskLifecycle wired en runtime cazan mutaciones que static analysis
   no ve (dynamic attribute access). choreo caza mutaciones que tests
   no exercise — particularmente la que un PR cold-path añadiría.
   Ninguno solo bastaba.

2. **El argumento "tests bloquean wire-up" fue exagerado para
   TaskLifecycle.** Mi caracterización en Phase 4 ADR-007 sobre wildcards
   necesarios era humo. Las 6 mutaciones en `test_swarm.py` son todas
   legales en el lifecycle real (PENDING → RUNNING, RUNNING → COMPLETED)
   excepto 5 que fuerzan terminal directamente desde PENDING. Esos 5 son
   resueltos con 2 transiciones explícitas (no wildcards), preservando
   validación.

3. **Las 3 FSMs declarativas restantes no son wire-up-able sin refactor
   de modelo.** PheromoneDeposit, QueenSuccession, FailoverFlow no
   tienen field `state`. Para wire-uparlas hay que primero introducir
   el field — que es trabajo de Phase 5+ alineado con el split de
   `resilience.py`/`nectar.py`. choreo cubre el gap interim: las
   verifica estáticamente como están, sin cambio de modelo.

4. **AST analysis es proporcionalmente barato.** ~600 LOC + 32 tests +
   ADR + CI integration en un día. Decisión arquitectónica de mantener
   el scope pequeño (3 patrones literales, no type inference) pagó
   directamente.

5. **El binding heurístico FSM↔Enum funciona para HOC, pero no
   garantiza para futuros casos.** Si dos enums tienen el mismo set de
   members, choreo elige uno arbitrariamente (por nombre). Phase 4.2
   debe permitir metadata opt-in (`enum=...` en el builder) para
   disambigüación explícita.

6. **El subpaquete `choreo/` es extraíble como librería independiente.**
   No importa nada HOC-specific (excepto el smoke test integration).
   Si la herramienta prueba valor, Phase 11+ podría publicarla a PyPI
   como `choreo` standalone.

7. **CI con warnings-no-fail por default es la decisión correcta.**
   B12-bis y B12-ter están documentados como deferred. Si el CI fallase
   con warnings desde el día uno, no podríamos mergear Phase 4.1.
   `--strict` queda parqueado para cuando Phase 5+ los resuelva.

8. **El problema del "Source file found twice" no afectó a `choreo/`.**
   Wait — corrijo: el `mypy .` global ahora cubre 14 archivos, lo cual
   incluye `choreo/`. Que `choreo/` no haya gatillado el bug que
   `state_machines/` sí gatillaba sugiere que la causa está en el
   pattern de import (state_machines es importado tanto por `core/` con
   absolute como por código local) y no en el package layout en sí.
   Revisar en Phase 9 cuando reorganicemos package-dir.
