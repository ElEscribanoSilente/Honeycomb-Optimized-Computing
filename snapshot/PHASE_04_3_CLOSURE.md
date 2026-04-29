# Phase 4.3 Closure — Dead enum cleanup (B12-bis + B12-ter)

**Fecha**: 2026-04-25
**Tag previsto**: `v1.4.3-phase04.3`
**Branch**: `phase/04.3-cleanup`
**PR**: (pending — abrir tras `git push`)

---

## Resumen ejecutivo

Phase 4.3 es una mini-fase de cleanup que aplica per-member la
discriminación "eliminar vs reservar" sobre los enum members dead
detectados por choreo en Phase 4.1/4.2. Resultado: **el reporte de
`choreo check` baja de 2 warnings a 1**, sin tocar tooling, sin nuevas
features, sin nuevas dependencies.

Aplicado:

- **`TaskState.ASSIGNED` eliminado** (B12-bis resuelto).
- **`CellState.SPAWNING` y `CellState.OVERLOADED` eliminados**.
- **`CellState.MIGRATING` y `CellState.SEALED` reservados** para
  wire-up en Phase 5 (observability).

| Métrica | Phase 4.2 (`v1.4.2`) | Phase 4.3 (`v1.4.3`) | Δ |
|---------|----------------------|----------------------|---|
| Tests pasando | 734 | **733** | -1 (eliminado test obsoleto B12-bis) |
| `CellState` members | 9 | **7** (-2: SPAWNING, OVERLOADED) | -2 |
| `TaskState` members | 6 | **5** (-1: ASSIGNED) | -1 |
| `choreo check` warnings | 2 | **1** | -1 ✅ |
| `choreo check` errors | 0 | **0** | = |
| `choreo check` info | 3 | **3** | = (declarative-only, deferred to Phase 5) |
| ADRs | 9 | **10** | +1 |
| Bandit HIGH/MEDIUM/LOW | 0/0/0 | **0/0/0** | = ✅ |
| pip-audit | 0 vulns | **0** | = ✅ |
| ruff/black/mypy | clean | **clean** | = ✅ |

---

## 4.3.1 Decisión per-member

ADR-010 documenta el reasoning. Resumen:

| Enum member | Decisión | Justificación corta |
|---|---|---|
| `TaskState.ASSIGNED` | **Eliminar** | Workers van `PENDING → RUNNING` atómico al claim. El "claimed-not-yet-running" interval no es observable externamente. |
| `CellState.SPAWNING` | **Eliminar** | Cells construyen en `EMPTY` y van a `IDLE` al primer `add_vcore`. No hay constructor pause donde SPAWNING viva. |
| `CellState.OVERLOADED` | **Eliminar** | Circuit breaker tiene 2 estados (cerrado=ACTIVE, abierto=FAILED). Sin threshold intermedio. |
| `CellState.MIGRATING` | **Reservar (Phase 5)** | Observabilidad de migraciones in-flight tiene valor real. `CellFailover.migrate_cell` debe setear source state durante el progreso. |
| `CellState.SEALED` | **Reservar (Phase 5)** | Graceful shutdown is a real ops feature. Hoy todo va via FAILED — disruptivo y noisy. |

---

## 4.3.2 Cambios concretos

### `core/cells_base.py`

CellState reducido de 9 a 7 members:

```python
class CellState(Enum):
    EMPTY = auto()
    ACTIVE = auto()
    IDLE = auto()
    MIGRATING = auto()  # reserved: Phase 5 wire-up in CellFailover.migrate_cell
    FAILED = auto()
    RECOVERING = auto()
    SEALED = auto()  # reserved: Phase 5 wire-up for graceful shutdown
```

Docstring actualizado documentando el cleanup.

### `swarm.py`

TaskState reducido de 6 a 5 members:

```python
class TaskState(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
```

Docstring documenta que ASSIGNED fue removido.

### `state_machines/cell_fsm.py`

- `CELL_STATE_SPAWNING` y `CELL_STATE_OVERLOADED` constants removidos.
- `ALL_CELL_STATES` tuple reducida de 9 a 7 elementos.
- Comment block agregado marcando MIGRATING y SEALED como reserved.

### `metrics/visualization.py`

- `STATE_CHARS[CellState.SPAWNING]` removido.
- `colors[CellState.SPAWNING]` removido.
- `MIGRATING` y `SEALED` se mantienen en ambas tablas (ya estaban).

### Tests actualizados

| Test | Cambio |
|------|--------|
| `test_state_count` (CellState) | `== 9` → `== 7` |
| `test_dead_state_unreachable_via_lifecycle` | Usa SEALED en lugar de SPAWNING |
| `test_illegal_transition_raises_and_does_not_mutate` | Usa SEALED en lugar de SPAWNING |
| `test_illegal_transition_assigned_dead_state_raises` | **Eliminado** (B12-bis ya no aplica) |
| `test_render_includes_state_count_and_initial` | `(9)` → `(7)` |
| `test_hoc_findings_exact` | Asserts `dead_state` solo MIGRATING+SEALED; `enum_extra_state` no debe aparecer |

### Auto-regeneración

`docs/state-machines.md` regenerado — el CellState diagram tiene 7 nodos
en lugar de 9.

---

## 4.3.3 choreo check — antes / después

**Antes (Phase 4.2):**

```
== WARNINGS (2) ==
  [CellState] dead_state
    4 state(s) ... not targeted by ... mutation: MIGRATING, OVERLOADED, SEALED, SPAWNING
  [TaskLifecycle] enum_extra_state
    enum `TaskState` declares 1 member(s) not in FSM: ASSIGNED ...

== INFO (3) ==
  [FailoverFlow / PheromoneDeposit / QueenSuccession] declarative_only

Summary: 0 errors, 2 warnings, 3 infos
```

**Después (Phase 4.3):**

```
== WARNINGS (1) ==
  [CellState] dead_state
    2 state(s) ... not targeted by ... mutation: MIGRATING, SEALED

== INFO (3) ==
  [FailoverFlow / PheromoneDeposit / QueenSuccession] declarative_only

Summary: 0 errors, 1 warning, 3 infos
```

El warning restante representa estados *intencionalmente reservados* —
documentado en ADR-010 — que Phase 5 wireará.

---

## Auditorías

```
ruff:    All checks passed!
black:   59 files left unchanged
mypy:    Success (15 source files in `mypy .`)
mypy:    Success (8 source files in --explicit-package-bases state_machines/*.py)
bandit:  HIGH: 0 / MEDIUM: 0 / LOW: 0
pip-audit: No known vulnerabilities (runtime + dev)
radon CC:  C(13.27) -- sin regression
mermaid drift: OK (regenerado tras eliminación de 2 estados)
benchmarks: 11 passed, 3 skipped (sin regression)
```

---

## Definition of Done

| Ítem | Estado | Nota |
|------|--------|------|
| `TaskState.ASSIGNED` eliminado | ✅ | swarm.py + tests |
| `CellState.SPAWNING` eliminado | ✅ | cells_base.py + cell_fsm.py + visualization.py + tests |
| `CellState.OVERLOADED` eliminado | ✅ | cells_base.py + cell_fsm.py |
| `CellState.MIGRATING` reservado | ✅ | docstring inline marca Phase 5 wire-up |
| `CellState.SEALED` reservado | ✅ | docstring inline marca Phase 5 wire-up |
| `choreo check` reduce warnings 2 → 1 | ✅ | confirmado en `test_hoc_findings_exact` |
| 734 tests Phase 4.2 siguen pasando (modulo el eliminado) | ✅ | 733 total |
| ruff / black / mypy / bandit / pip-audit clean | ✅ | sin regressions |
| `docs/state-machines.md` regenerado | ✅ | 7 nodes en CellState diagram |
| ADR-010 escrito | ✅ | per-member rationale |
| CHANGELOG entry [1.4.3-phase04.3] | ✅ | (this commit) |
| ROADMAP marca Phase 4.3 como CERRADA | ✅ | (this commit) |

---

## Gaps diferidos a Phase 5

### Gap 1: Wire-up de `CellState.MIGRATING`

`CellFailover.migrate_cell` debe setear `source.state = MIGRATING`
al inicio, transicionar a `FAILED` on success o regresar a `ACTIVE`
en rollback. ~1 hora trabajo + tests. Aporta observabilidad de
migraciones in-flight.

### Gap 2: Wire-up de `CellState.SEALED`

Nuevo método `cell.seal()` para graceful shutdown — drains vCores,
refuses new work, persists final metrics. ~2 horas trabajo + tests.
Reemplaza el "todo va a FAILED" actual.

### Gap 3: Wire-up de las 3 FSMs declarative-only

Pheromone, Succession, Failover siguen sin host enum. Phase 5 introduce
state fields y los wira (~12-15 horas total). Después de eso,
`choreo check` reporta 0 warnings, 0 info.

### Gap 4: `--strict` CI flip

Una vez Gap 1-3 resueltos, choreo report queda en 0/0/0 — flippear
el job a `--strict` para que warnings nuevos rompan CI también.

---

## Lecciones aprendidas

1. **Per-member discrimination es la decisión correcta.** Tratar
   "eliminar todo" o "wirear todo" en bulk hubiera mezclado micro-
   decisiones independientes. La pregunta "¿este nombre tiene caso
   de uso real?" se respondió per name, dando claridad para el
   futuro.

2. **Reservar names sin wire-up es legítimo solo con commitment de
   uso.** MIGRATING y SEALED quedan como warnings hasta Phase 5.
   ADR-010 documenta el commitment. Si Phase 5 los descarta, deben
   eliminarse. Reserved sin uso indefinido = technical debt
   re-disfrazado.

3. **`choreo check` evolucionando reporte es buena UX.** El usuario
   ve "2 → 1 warning" como progreso visible en CI. Confirma que la
   herramienta valora — no solo dice "todo bien" sino que pinta
   cambio.

4. **Eliminar tests obsoletos es honest cleanup.** El test
   `test_illegal_transition_assigned_dead_state_raises` validaba un
   bug que ya no existe. Mantenerlo "por si acaso" hubiera sido
   ruido — confirmar un comportamiento que no puede ocurrir.

5. **Phase 5 ya tiene scope concreto pre-asignado.** ADR-010 + closure
   marcan MIGRATING + SEALED como to-do. Phase 5 no tiene que
   debatirlo — el commitment ya está.
