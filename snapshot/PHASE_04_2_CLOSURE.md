# Phase 4.2 Closure — Reified transitions + auto-derive (`choreo` v0.2)

**Fecha**: 2026-04-25
**Tag previsto**: `v1.4.2-phase04.2`
**Branch**: `phase/04.2-choreo-reified`
**PR**: (pending — abrir tras `git push`)

---

## Resumen ejecutivo

Phase 4.2 entrega `choreo` v0.2 con cuatro mejoras additivas sobre Phase
4.1, sin romper ningún contrato de la fase anterior:

1. **Walker patterns ampliados** — captura `setattr(obj, "state", X)` y
   `dataclasses.replace(obj, state=X)`, cerrando rutas que un refactor
   futuro podría usar para bypassear choreo silently.

2. **Reified transitions (`@transition`)** — decorator nuevo en
   `state_machines/reified.py`. Marca un método como una transición de
   estado; valida pre-condición, ejecuta el método, y muta `self.state`
   solo si el método retorna sin excepción. Aplicado a 4 métodos de
   `HiveTask` (`claim`, `complete`, `fail`, `retry`) como API additiva
   — los call-sites legacy en `swarm.py` siguen usando direct mutation
   sin cambios.

3. **Auto-derive (`choreo derive`)** — nuevo subcomando que toma un
   `.py` y emite un skeleton de `build_X_fsm()` matching las mutations
   observadas. Bootstrapping aid; usa `WILDCARD` para sources, el
   contribuyente edita.

4. **`enum_name=` opt-in en `HocStateMachine`** — cuando se setea,
   choreo usa esa binding explícita en lugar de la heurística de
   member-subset. Previene ambigüedad futura. `cell_fsm.py` y
   `task_fsm.py` lo usan; las 3 declarative-only no.

**29 tests nuevos** (705 → 734): 8 walker + 4 enum_name + 6 derive +
11 reified. Sin nuevas runtime deps. Bandit/pip-audit/ruff/black/mypy
todos limpios. choreo aplicado a HOC sigue produciendo el reporte
exacto: 0 errors / 2 warnings (B12-bis, B12-ter) / 3 info.

| Métrica | Phase 4.1 (`v1.4.1`) | Phase 4.2 (`v1.4.2`) | Δ |
|---------|----------------------|----------------------|---|
| Tests pasando | 705 | **734** | +29 |
| Walker patterns soportados | 3 | **5** (assign, _set_state, ClassDef, **setattr**, **dataclasses.replace**) | +2 |
| `@transition` decorator | n/a | **disponible en `state_machines.transition`** | new |
| Reified methods en HiveTask | 0 | **4** (claim, complete, fail, retry) | +4 |
| `choreo derive` subcommand | n/a | **disponible** | new |
| `HocStateMachine.enum_name` | n/a | **opt-in para 2/5 FSMs** | new |
| ADRs | 8 | **9** | +1 |
| GitHub Actions jobs | 6 | **6** (sin nuevos) | = |
| Runtime dependencies | numpy + tramoya | **same** | = |
| Bandit HIGH / MEDIUM / LOW | 0 / 0 / 0 | **0 / 0 / 0** | = ✅ |
| pip-audit vulnerabilidades | 0 | **0** | = ✅ |
| ruff / black / mypy errors | 0 | **0** | = ✅ |
| `choreo check` findings (HOC) | 0 err / 2 warn / 3 info | **0 / 2 / 3** (idéntico) | = |

---

## 4.2.1 Walker patterns

`choreo/walker.py` ahora reconoce 5 patrones (vs 3 en Phase 4.1):

| Patrón | Pattern field | Ejemplo |
|--------|---------------|---------|
| `obj.state = ENUM.X` | `"assign"` | `cell.state = CellState.IDLE` |
| `obj._set_state(ENUM.X)` | `"_set_state"` | `self._set_state(CellState.ACTIVE)` |
| `setattr(obj, "state", ENUM.X)` | `"setattr"` (NEW) | `setattr(task, "state", TaskState.RUNNING)` |
| `dataclasses.replace(obj, state=ENUM.X)` | `"dataclasses.replace"` (NEW) | `replace(task, state=TaskState.COMPLETED)` |
| `class X(Enum): ...` | (enum decl) | `class TaskState(Enum): ...` |

`setattr` matchea solo cuando el atributo es un literal `"state"` —
nombres dinámicos siguen siendo falsos negativos (consistente con
trade-off de Phase 4.1).

`dataclasses.replace` matchea ambos `dataclasses.replace(...)` y
`replace(...)` (con `from dataclasses import replace`). El kwarg
`state=` es la señal — sin ese kwarg, el walker no produce una
mutación.

### Tests añadidos

`tests/test_choreo.py::TestWalker` — 6 tests nuevos:
- `test_captures_setattr_pattern`
- `test_setattr_non_state_ignored`
- `test_setattr_dynamic_attr_name_ignored`
- `test_captures_dataclasses_replace_pattern`
- `test_captures_bare_replace_pattern`
- `test_replace_without_state_kwarg_ignored`

---

## 4.2.2 Reified transitions

### Decorator API

`state_machines/reified.py` exporta `@transition`:

```python
@transition(from_=TaskState.PENDING, to=TaskState.RUNNING)
def claim(self, worker: WorkerCell) -> None:
    self.assigned_to = worker.coord
```

Comportamiento:
1. Valida `self.state is from_`. Levanta `IllegalStateTransition(reason="reified_from_mismatch")` si no coincide.
2. Ejecuta el método (puede mutar otros attributes; no debe mutar `self.state`).
3. Si retorna OK → `self.state = to` (pasa por `__setattr__` → wired FSM revalida).
4. Si excepción → no muta state.

`from_=None` skipa la pre-condición.

Decorator atributo `__choreo_transition__` = `(from_, to)` para
introspección. Phase 4.3+ puede usarlo para construir FSMs desde
decoradores.

### Aplicación en HiveTask

| Método | from_ | to | Estado en swarm.py |
|--------|-------|-----|--------------------|
| `claim(worker)` | PENDING | RUNNING | API additiva — call-sites siguen con direct mutation |
| `complete(result=None)` | RUNNING | COMPLETED | idem |
| `fail(error)` | RUNNING | FAILED | idem |
| `retry()` | FAILED | PENDING | idem |

Los 16 sites en `swarm.py` que usan `task.state = X` directo **no se
modificaron**. Las dos APIs coexisten:
- Direct: para test fixtures, internal state-injection, paths
  performance-críticos.
- Reified: para nuevo código que beneficia de self-documentation.

### Tests añadidos

`tests/test_state_machines.py::TestReifiedDecoratorIsolated` (5 tests):
- `test_transition_runs_method_and_mutates_state`
- `test_from_mismatch_raises`
- `test_method_exception_does_not_mutate`
- `test_from_none_skips_precondition`
- `test_metadata_attached`

`tests/test_state_machines.py::TestReifiedHiveTask` (6 tests):
- `test_claim_running`
- `test_complete_stores_result`
- `test_fail_increments_attempts`
- `test_retry_loops_back_to_pending`
- `test_claim_from_running_raises`
- `test_method_exception_leaves_state_unchanged`

---

## 4.2.3 Auto-derive

### CLI usage

```bash
python -m choreo derive swarm.py                   # FSM skeleton to stdout
python -m choreo derive swarm.py -o task_fsm.py    # to file
python -m choreo derive m.py --fsm-name MyFSM      # override FSM name
python -m choreo derive m.py --enum-name X         # restrict to enum X
python -m choreo derive m.py --initial PENDING     # set initial state
```

### Output structure

El skeleton incluye:
- Docstring de auto-gen + warning ("Review carefully before committing").
- Imports (`HocStateMachine`, `HocTransition`, `WILDCARD`).
- Constantes de estados (`TASKSTATE_RUNNING = 'RUNNING'`, etc.).
- `ALL_STATES` tuple.
- Función `build_<stem>_fsm()` con transitions WILDCARD-source y
  comments `# observed at <file>:<line>` por cada edge.

El usuario edita los WILDCARDs con sources reales antes de commitear.

### Naming heuristics

- `enum_name` ending in `State` → fsm_name = stem + `Lifecycle`
  (e.g. `TaskState` → `TaskLifecycle`).
- Otherwise → fsm_name = enum_name + `FSM` (e.g. `Color` → `ColorFSM`).
- `builder_name` derived from CamelCase → snake_case + `_fsm` prefix
  (e.g. `TaskLifecycle` → `build_task_fsm`).

### Tests añadidos

`tests/test_choreo.py::TestDerive` (6 tests) + 3 tests CLI
(`test_derive_subcommand_to_stdout`, `test_derive_subcommand_to_file`,
`test_derive_invalid_module_exits_two`).

---

## 4.2.4 Explicit `enum_name=`

`HocStateMachine.__init__` acepta `enum_name: str | None = None`.
Cuando se pasa, choreo prefiere esa binding sobre la heurística.

Lógica de bind en `choreo/diff.py::bind_fsm_to_enum`:
1. Si `fsm.enum_name` set y un `EnumDecl` con ese nombre exists:
   - Si members ⊇ states → bind.
   - Si no → return None (signals inconsistency, no falls back).
2. Si named enum no exists → falls back to heuristic (typo
   resilience).
3. Sin `enum_name` → heurística (Phase 4.1 behavior).

### Builders actualizados

| FSM | `enum_name=` |
|-----|--------------|
| `cell_fsm.py` | `"CellState"` |
| `task_fsm.py` | `"TaskState"` |
| `pheromone_fsm.py` | (omitted — no host enum) |
| `succession_fsm.py` | (omitted) |
| `failover_fsm.py` | (omitted) |

Strings, no `type[Enum]`, para evitar circular imports
(`task_fsm.py` ↔ `swarm.py`, `cell_fsm.py` ↔ `core/cells_base.py`).

### Tests añadidos

`tests/test_choreo.py::TestBindFsmToEnum` (4 tests nuevos):
- `test_explicit_enum_name_overrides_heuristic`
- `test_explicit_enum_name_inconsistent_returns_none`
- `test_explicit_enum_name_unknown_falls_back_to_heuristic`
- (preserved 4 originales)

---

## Auditorías

### Seguridad — Bandit

```
HIGH: 0 / MEDIUM: 0 / LOW: 0
```

Sin cambios — Phase 4.1 cerró 0/0/0 y Phase 4.2 mantiene.

### Vulnerabilidades — pip-audit

```
runtime: No known vulnerabilities found
dev:     No known vulnerabilities found
```

### Lint / format / mypy

```
ruff:  All checks passed!
black: 59 files left unchanged
mypy:  Success (15 source files in `mypy .`)
mypy:  Success (8 source files in --explicit-package-bases state_machines/*.py)
```

### Cobertura

`tests/test_choreo.py` y los tests reified suman cobertura over `choreo/`
y `state_machines/reified.py`. Cobertura global se mantiene cerca del
77 % de Phase 4.1 (los 29 tests nuevos cubren código nuevo de bajo LOC).

### Complejidad — Radon

`radon cc . -a -nc`: avg CC = **C (13.27)** — sin regression vs Phase
4.1 (13.31). Sin nuevos hot paths con CC > 10. Los archivos nuevos de
Phase 4.2 (`choreo/derive.py` ~210 LOC, `state_machines/reified.py`
~95 LOC) tienen funciones todas con CC < 5.

### Benchmark — `snapshot/bench_phase04_2.txt`

11 benchmarks pass + 3 skipped. Means consistentes con Phase 4.1
(test_grid_creation ~1.3ms, test_grid_tick ~460μs). Sin overhead nuevo
en hot paths — `@transition` no se aplica en hot loops.

### Mermaid drift check

```
OK: docs/state-machines.md matches FSM specs (5 FSMs)
```

`enum_name=` no afecta el output Mermaid — invariante preservada.

### choreo check (against HOC)

```
== WARNINGS (2) ==
  [CellState] dead_state: MIGRATING, OVERLOADED, SEALED, SPAWNING
  [TaskLifecycle] enum_extra_state: ASSIGNED

== INFO (3) ==
  [FailoverFlow] declarative_only
  [PheromoneDeposit] declarative_only
  [QueenSuccession] declarative_only

Summary: 0 errors, 2 warnings, 3 infos
```

Idéntico a Phase 4.1. La binding explícita de CellState y TaskLifecycle
funciona — choreo encuentra los mismos hallazgos sin la heurística.

---

## Definition of Done

| Ítem | Estado | Nota |
|------|--------|------|
| Walker pattern: `setattr(obj, "state", X)` | ✅ | + 3 tests |
| Walker pattern: `dataclasses.replace(obj, state=X)` | ✅ | + 3 tests |
| `@transition` decorator implementado | ✅ | `state_machines/reified.py` |
| HiveTask reified API (claim/complete/fail/retry) | ✅ | + 6 tests, additive |
| Auto-derive (`choreo derive`) implementado + CLI | ✅ | + 9 tests |
| Opt-in `enum_name=` en HocStateMachine | ✅ | + 4 tests, builders actualizados |
| 705 tests Phase 4.1 siguen pasando | ✅ | 734 total |
| ruff / black / mypy / bandit / pip-audit clean | ✅ | sin regressions |
| `choreo check` against HOC produces same report | ✅ | 0 err / 2 warn / 3 info |
| Mermaid drift check pass | ✅ | sin cambios |
| Bench within Phase 4.1 baseline | ✅ | sin regression measurable |
| ADR-009 escrito | ✅ | reified + auto-derive + walker patterns + enum_name |
| CHANGELOG entry [1.4.2-phase04.2] | ✅ | (this commit) |
| ROADMAP marca Phase 4.2 como CERRADA | ✅ | (this commit) |

---

## Gaps diferidos (a Phase 5+)

### Gap 1: Auto-derive con control-flow analysis

Walker actual emite `WILDCARD` para `source` de cada edge. Implementar
inferencia de source state requiere control-flow analysis — research
work, deferred indefinidamente o hasta Phase 11+.

### Gap 2: Walker patterns adicionales

Patrones todavía missed: `attrs.evolve(obj, state=X)`,
`obj.state = func()` (RHS computed), descriptor-driven mutations.
Phase 5+ puede extender el walker si surgen casos reales.

### Gap 3: B12-bis y B12-ter resolution

Aún pendientes desde Phase 4. Phase 5+ debe decidir:
- Eliminar dead enum members (`TaskState.ASSIGNED`, 4 `CellState`).
- O wirear los call-sites missing.

### Gap 4: `--strict` flip en CI

Phase 4.1 dejó CI en modo no-strict. Phase 4.2 lo mantiene. Una vez
resuelto Gap 3, flippear a `--strict` para que warnings fallen CI.

### Gap 5: Reified pattern en otras FSMs

CellState wired tiene 8+ call-sites en `resilience.py`; aplicar
reified (`cell.fail()`, `cell.recover()`, etc.) no es trivial porque
los call-sites son admin/failover, no lifecycle natural. Phase 5+
puede evaluarlo cuando split de `resilience.py`.

### Gap 6: Decorator-based FSM derivation

`@transition` ya guarda metadata. Una herramienta que lee los
decoradores en una clase y construye un `HocStateMachine`
automáticamente sería natural extension. Phase 5+ si valor
demostrado.

---

## Lecciones aprendidas

1. **Walker pattern coverage paid off mecánicamente.** Agregar
   `setattr` y `dataclasses.replace` requirió ~30 LOC en walker.py +
   ~50 LOC de tests. Bajo esfuerzo, alta resilience contra refactors
   futuros.

2. **`@transition` es valor self-documenting más que correctness
   value.** Phase 4.1 wired `__setattr__` ya garantiza que `task.state
   = X` está validado. `@transition` añade legibilidad — `task.claim()`
   reads better than `task.state = TaskState.RUNNING` — pero no
   mejora la safety de runtime.

3. **Auto-derive con WILDCARD-source es honest sobre lo que el tool
   sabe.** No pretender saber el control flow → output que claramente
   dice "edit me before committing". Mejor que falsear sources que
   podrían estar mal.

4. **Strings sobre `type[Enum]` para `enum_name=` evitó circular
   imports.** Trade-off: pierdo type safety; gano simplicidad de
   import. En HOC actual el package layout fuerza la decisión.

5. **Tests inline para reified-decorator subclasses requieren cuidado
   con scope.** Un primer intento falló porque importé `transition`
   dentro de un test pero usé el decorator antes del import. Lección:
   imports siempre primero, luego subclase, luego instanciar.

6. **El Phase 4.1 contract es la prueba de invarianza.** choreo
   produce idéntico output al de Phase 4.1 (0/2/3) — confirmando que
   las 4 features de Phase 4.2 son additive: ninguna cambia el
   comportamiento del checker actual.

7. **Auto-derive es genuino bootstrapping aid, no una herramienta
   "auto-mágica".** El skeleton output es ~50% del trabajo; el
   contributor edita los WILDCARDs (que es el 50% intelectual). El
   tool ahorra typing, no thinking.
