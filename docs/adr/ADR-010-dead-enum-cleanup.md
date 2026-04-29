# ADR-010: Dead enum-member cleanup — eliminate vs reserve

- **Status**: Accepted
- **Date**: 2026-04-25
- **Phase**: Phase 4.3

## Context

ADR-007 / Phase 4 closure documented two latent bugs that `choreo`
later confirmed (ADR-008):

- **B12-bis**: `TaskState.ASSIGNED` declared in the enum at
  `swarm.py:90` but never assigned in production. Phase 4.1's wire-up
  caused `task.state = TaskState.ASSIGNED` to raise
  `IllegalStateTransition(reason="unknown_state")` at runtime.
- **B12-ter**: 4 `CellState` enum members
  (`SPAWNING`, `MIGRATING`, `SEALED`, `OVERLOADED`) similarly declared
  but never assigned. Same runtime behaviour via the CellState wire-up.

Both bugs surfaced as `dead_state` / `enum_extra_state` warnings in
every `choreo check` run. Phase 4.1/4.2 deferred resolution to keep
scope focused on tooling. By Phase 4.3 the choice between *eliminate*
and *wire up* needed to be made.

The two paths:

1. **Eliminate** — remove the enum members. Trivial, low-risk, but
   forecloses any future use of the names. Must justify per member
   that the concept genuinely has no value.
2. **Wire up** — implement the call-sites that should produce the
   state. Larger scope; some require model refactor (e.g. `MIGRATING`
   needs `CellFailover.migrate_cell` to set source-cell state during
   migration). Adds operational visibility but is behaviour change.

A third path applies per-member: **eliminate the truly aspirational
ones, reserve the operationally useful ones for a later wire-up
phase**.

## Decision

Apply the per-member discrimination — Phase 4.3 eliminates the names
that have **no defensible use case**, keeps the names that have
**clear operational value** as reserved, and tags the wire-up of the
reserved ones for Phase 5 (observability).

### Eliminated (Phase 4.3)

| Name | Original intent | Why eliminate |
|---|---|---|
| `TaskState.ASSIGNED` | Two-step `PENDING → ASSIGNED → RUNNING` lifecycle | Workers go `PENDING → RUNNING` atomically when they claim. The "claimed but not yet executing" interval is a thread-of-execution detail invisible to outside observers. No metrics, no logs, no caller depends on it. |
| `CellState.SPAWNING` | Cell-being-created state | Cells construct in `EMPTY` (no vCores) and transition directly to `IDLE` on first `add_vcore`. No instrumented constructor pause exists where `SPAWNING` would live. |
| `CellState.OVERLOADED` | Soft-warning state before circuit-breaker trips | The circuit breaker either is closed (cell `ACTIVE`) or open (cell `FAILED`). There is no implemented intermediate threshold. The state was aspirational; current code never set it. |

### Reserved for Phase 5 wire-up

| Name | Phase 5 wire-up target | Why keep |
|---|---|---|
| `CellState.MIGRATING` | Set `source.state = MIGRATING` at the start of `CellFailover.migrate_cell`, transition to `FAILED` on success, back to `ACTIVE` on rollback | Observability of in-flight migrations. Currently `migrate_cell` sets `source=FAILED` and `target=IDLE` atomically — operators cannot tell whether a migration is in progress. |
| `CellState.SEALED` | New `cell.seal()` for graceful shutdown — drains vCores, refuses new work, persists final metrics | A real ops feature. Today shutdown is "everything goes to FAILED via the failover handler" — disruptive and noisy. |

Both reserved names continue to appear as `dead_state` warnings in
`choreo check` until Phase 5 wires them. The Phase 4.3 closure
documents this; the warnings are no longer "unknown bugs" but
"reserved-for-future-use", visible in CI.

## Alternatives considered

### Wire all four `CellState` survivors and `TaskState.ASSIGNED`

Implements every state. ~12-15 hours of refactoring across
`resilience.py`, `swarm.py`, and tests. Mixes a cleanup task with two
behavior changes (`MIGRATING` and `SEALED` are real lifecycle changes;
`ASSIGNED` and `SPAWNING` are micro-events that nobody asked for).
**Rejected** — split the work: Phase 4.3 cleanup-only, Phase 5
behavior changes within the observability scope.

### Eliminate everything (including `MIGRATING` and `SEALED`)

Simple, ~10 minutes. But forecloses two operational features that
have clear value (Phase 5's observability roadmap explicitly calls
for migration visibility and graceful shutdown). Re-introducing
later means re-justifying the names. **Rejected** — keep the
reserved names.

### Wire only `MIGRATING` and `SEALED` now (no eliminations)

Resolves B12-ter completely but leaves B12-bis. Inconsistent
treatment. Also, wire-up of `MIGRATING` requires re-validating the
4 CellFailover tests in `tests/test_resilience.py`; doing that
within Phase 4.x would push the closure into Phase 5 territory.
**Rejected** — Phase 5 has the proper scope.

### Use `# noqa: choreo:dead_state` markers instead of eliminating

Suppress the warning per-member without changing the enum. Keeps
the names "for future use" without commitment. **Rejected** —
deferred without justification is technical debt with no expiry.
The discrimination above forces the question "is this name worth
keeping?" — eliminate when the answer is no.

## Consequences

### Easier

- **`choreo check` warning count drops from 2 to 1.** A cleaner
  baseline; future regressions stand out more.
- **`TaskState` and `CellState` enums match production reality.**
  Readers don't have to wonder whether `ASSIGNED` or `SPAWNING` are
  used somewhere they missed.
- **Phase 5's wire-up scope is clearer.** Two reserved names with
  documented intent — wire them or eliminate them, no third option.

### Harder

- **`metrics/visualization.py` lost two glyphs/colors.** `SPAWNING`
  was rendered as `◉` / `#ffff00`. The render path is `colors.get(...,
  "#ffffff")` so unknown states fall back gracefully. No external
  caller used the SPAWNING glyph.
- **One test (`test_illegal_transition_assigned_dead_state_raises`)
  was deleted.** It validated B12-bis runtime detection — no longer
  applicable since `ASSIGNED` is gone. The general
  `test_unknown_target` in `TestTransitionTo` covers "FSM rejects
  unknown state names".

### Risk / follow-up

- **If Phase 5 decides not to wire `MIGRATING` or `SEALED`, eliminate
  them.** Reserved names that stay reserved indefinitely become the
  same kind of dead code this ADR cleaned up.
- **The `--strict` CI flip is closer.** Phase 4.3 reduces warnings
  from 2 to 1. After Phase 5 wires `MIGRATING` and `SEALED`, the
  `dead_state` warning disappears. Then the 3 `declarative_only`
  infos remain but those don't fail strict mode (info severity).
- **No changes to choreo itself.** Phase 4.3 is content-cleanup, not
  tooling. ADR-008 / ADR-009 stand unchanged.

## References

- `core/cells_base.py:CellState` — 4 members reduced to 7.
- `swarm.py:TaskState` — 6 members reduced to 5.
- `state_machines/cell_fsm.py` — `CELL_STATE_SPAWNING` and
  `CELL_STATE_OVERLOADED` constants removed.
- `metrics/visualization.py` — `SPAWNING` entries dropped from
  `STATE_CHARS` and `colors`.
- `tests/test_state_machines.py` — `test_state_count` updated to 7;
  `test_dead_state_unreachable_via_lifecycle` uses `SEALED` instead
  of `SPAWNING`; `test_illegal_transition_raises_and_does_not_mutate`
  uses `SEALED`; `test_illegal_transition_assigned_dead_state_raises`
  deleted.
- `tests/test_choreo.py::TestHocIntegration` — assertions updated to
  expect 1 dead_state finding (MIGRATING + SEALED) and zero
  enum_extra_state findings.
- `tests/test_mermaid_export.py::test_render_includes_state_count_and_initial`
  — `(9)` → `(7)`.
- ADR-007 / ADR-008 — Phase 4 / 4.1 history of B12-bis and B12-ter.
- Phase 5 plan — wire-up of `MIGRATING` + `SEALED` is part of the
  observability scope.
