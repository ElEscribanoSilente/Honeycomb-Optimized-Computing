# ADR-009: Reified transitions + auto-derive — `choreo` v0.2

- **Status**: Accepted
- **Date**: 2026-04-25
- **Phase**: Phase 4.2

## Context

Phase 4.1 (ADR-008) shipped `choreo` v0.1 — a static AST checker
matching `obj.state = X` and `obj._set_state(X)`. It produced the
expected 5 findings against HOC (B12-bis ASSIGNED, B12-ter 4 dead
states, 3 declarative-only FSMs). Two follow-ups were tagged for the
next sub-phase:

1. **Walker pattern coverage**: refactors that switch from direct
   assignment to `setattr(obj, "state", X)` or
   `dataclasses.replace(obj, state=X)` would silently bypass choreo.
2. **Reified transitions**: `task.state = TaskState.RUNNING` is
   functionally correct after the wire-up, but reads like a primitive.
   The *meaning* of the transition (a worker has claimed the task)
   lives elsewhere — typically in an unrelated method in
   `SwarmScheduler`. A self-documenting alternative would let callers
   write `task.claim(worker)`.
3. **Auto-derive**: introducing a new FSM by hand — listing every
   `obj.state = X` site, drafting the spec, hand-writing the file
   in `state_machines/` — has friction. Bootstrapping should be
   mechanical.
4. **Heuristic enum binding**: choreo binds an FSM to an enum by
   member-subset match. For HOC this is unambiguous, but a future
   contributor adding a second matching enum would hit a non-obvious
   tiebreak rule. Explicit metadata is preferable when known.

## Decision

Phase 4.2 ships **`choreo` v0.2** with four additive changes. Nothing
about Phase 4.1's contract is broken — every hook the v0.1 CI job
relied on still works.

### 1. Walker patterns

`choreo/walker.py` matches three additional patterns:

- `setattr(obj, "state", EnumName.MEMBER)` — only when the attribute
  name is a string literal (dynamic names are accepted as a false
  negative, matching the existing trade-off).
- `dataclasses.replace(obj, state=EnumName.MEMBER)` — both the
  qualified form and the bare `replace(...)` after
  `from dataclasses import replace`.

Each produces a `Mutation` with a distinct `pattern` field
(`"setattr"`, `"dataclasses.replace"`) so callers can filter or
report by shape.

### 2. Reified transitions — `state_machines/reified.py`

A new decorator factory:

```python
@transition(from_=TaskState.PENDING, to=TaskState.RUNNING)
def claim(self, worker: WorkerCell) -> None:
    self.assigned_to = worker.coord
```

The decorator:

1. **Pre-condition**: checks `self.state is from_` (or skips if
   `from_=None`). Raises `IllegalStateTransition(reason=
   "reified_from_mismatch")` on mismatch.
2. **Body**: executes the wrapped method.
3. **Post-condition**: on clean return, sets `self.state = to`. The
   mutation passes through `__setattr__` and (if the host has a
   wired FSM) is validated again — the second check is redundant
   under normal use but catches a subtle case: if the user decorates
   a method whose `(from_, to)` pair is *not* an edge in the
   underlying FSM, the post-condition wire-up rejects it.
4. **Exception**: if the method raises, state is **not** mutated.

The decorator stores `(from_, to)` on the wrapped method as
`__choreo_transition__` for introspection. Future tooling (e.g.
choreo's walker extended to read decorators) can derive the FSM
graph from these annotations.

Reified transitions are **additive**, not replacement. `task.state =
TaskState.RUNNING` continues to work. The decorator is opt-in per
method.

### 3. Auto-derive — `python -m choreo derive`

A new CLI subcommand and a new module `choreo/derive.py`. Given a
single `.py` file, the tool walks the AST, collects every
`obj.state = X`-shape mutation, picks the most-frequent enum, and
emits a Python source skeleton:

```python
# python -m choreo derive swarm.py > state_machines/task_fsm.py
"""
Auto-generated FSM skeleton -- derived from swarm.py.
...
"""
from .base import HocStateMachine, HocTransition, WILDCARD

TASKSTATE_PENDING = 'PENDING'
TASKSTATE_RUNNING = 'RUNNING'
...

def build_task_fsm() -> HocStateMachine:
    transitions: list[HocTransition] = [
        # observed at swarm.py:340 (assign)
        HocTransition(WILDCARD, TASKSTATE_RUNNING),
        ...
    ]
    return HocStateMachine(
        name="TaskLifecycle",
        states=list(ALL_STATES),
        transitions=transitions,
        initial=TASKSTATE_PENDING,
        enum_name="TaskState",
    )
```

The skeleton uses `WILDCARD` for every `source` because choreo does
not perform control-flow analysis (deferred — see "Risk / follow-up").
The user **must** review and edit before committing.

Auto-derive is **a bootstrapping aid**, not a replacement for
hand-written FSMs. The expected workflow is: derive, edit (specify
real source states), wire up via `__setattr__` or `@transition`,
then run `choreo check` to confirm coverage.

### 4. Explicit `enum_name=` in `HocStateMachine`

`HocStateMachine.__init__` accepts an optional `enum_name: str | None`
keyword. When set, it overrides choreo's member-subset heuristic in
the diff layer:

- If the named enum exists *and* its members are a superset of the
  FSM's states → bind explicitly.
- If the named enum exists *but* its members do not contain all
  FSM states → return None (signals inconsistency; better than
  silently falling through to the heuristic, which would mask the
  bug).
- If the named enum does not exist anywhere → fall back to the
  heuristic. This handles the case where the user typoed the name
  or renamed the enum.

`cell_fsm.py` and `task_fsm.py` were updated to pass
`enum_name="CellState"` and `enum_name="TaskState"` respectively.
`pheromone_fsm.py`, `succession_fsm.py`, `failover_fsm.py` continue
to omit it (they don't model an enum host).

The enum is referenced as a **string**, not a `type[Enum]`, to keep
the FSM modules importable without pulling in `core/cells_base.py` or
`swarm.py`. That would create circular imports (those modules already
import `state_machines`).

## Alternatives considered

### Make reified transitions REPLACE direct mutation

Force every `task.state = X` callsite to migrate to a method. This
would break Phase 4.1's contract that direct mutation continues to
work and would require touching ~16 call-sites in `swarm.py`. The
wire-up via `__setattr__` already centralizes validation; reification
is a *style* improvement, not a correctness one. **Rejected** — kept
both APIs.

### Auto-derive with control-flow analysis (real source states)

A more sophisticated version of `derive` would walk the function
bodies, build a CFG, and infer the source state of each mutation
from the predecessor blocks. The output would have real source-
state values rather than `WILDCARD`. This is research-grade work
(comparable to a flow-sensitive type checker) and out of scope for
Phase 4.x. **Rejected** — wildcard + manual edit is acceptable for
bootstrapping.

### `enum=type[Enum]` instead of `enum_name=str`

Type-safe and refactor-friendly. Would let mypy verify the binding.
But pulling the enum class into FSM modules creates circular imports
in HOC's current layout. **Rejected** for Phase 4.2; revisit when
Phase 5+ refactors the package layout (and the enum location).

### Use a separate FSM library for reified transitions (`transitions`, `python-statemachine`)

Both libraries have decorator-based APIs. But adopting one would
introduce a second FSM engine alongside `tramoya`, fragmenting the
runtime + static + reified surface. Our `@transition` decorator is
~30 LOC and reuses `HocStateMachine` for validation. **Rejected**
— internal decorator wins on simplicity.

## Consequences

### Easier

- **Refactors don't bypass choreo silently.** A PR that switches
  `cell.state = CellState.IDLE` to
  `dataclasses.replace(cell, state=CellState.IDLE)` is still caught.
- **New FSMs bootstrap quickly.** `choreo derive --module X.py`
  produces a starting point in seconds. The user edits the
  WILDCARDs, commits, runs `choreo check`.
- **Reified transitions improve readability.** `task.claim(worker)`
  reads better than the equivalent two-line direct-mutation idiom.
  Self-documenting at the method level.
- **Explicit enum binding.** Future contributors don't depend on
  the heuristic being unambiguous — the binding is named in the
  builder.

### Harder

- **Two APIs for state mutation.** Direct (`task.state = X`) and
  reified (`task.claim(worker)`) coexist. New contributors need
  to know both exist. CONTRIBUTING.md should document the choice
  (use reified for new lifecycle methods; direct for ad-hoc internal
  state-injection in tests).
- **Auto-derive output looks "done" but isn't.** Inexperienced
  contributors might commit the WILDCARD-only skeleton without
  editing. Mitigation: the generated docstring says "Review
  carefully before committing" prominently, and choreo will emit
  a warning if every edge has source `WILDCARD` (Phase 4.3+
  enhancement).

### Risk / follow-up

- **`@transition` + dataclass interaction.** The decorator works on
  any class but interacts with dataclass-generated `__init__` only
  if the host overrides `__setattr__` (which `HiveTask` does). For
  hosts without a wired FSM, the decorator's post-condition
  `self.state = to` is unchecked — that's by design (the decorator
  is the validation layer). Documented in the module docstring.
- **Walker pattern coverage.** Three more patterns matched, but
  patterns like `obj.state = func()` (computed RHS) and
  `attrs.evolve(obj, state=X)` are still missed. Phase 5+ may
  extend.
- **Auto-derive false positives.** If a module mutates the same
  enum field on two unrelated classes (e.g. both `HiveTask.state`
  and `Logger.state`), `derive` collapses them into one FSM. The
  `--enum-name` flag lets the user disambiguate; better partition
  in the walker would require type resolution (out of scope).
- **`enum_name="X"` typo silently degrades to heuristic.** If the
  user types `enum_name="TaskStateee"`, the named enum doesn't
  exist; choreo falls back to the heuristic. This is intentional
  (avoid hard failures on minor typos), but it does mean a typo
  could mask a binding bug. Phase 4.3+ may add a warning in the
  diff layer when the named enum is missing.

## References

- `choreo/walker.py` — extended with patterns 3 and 4.
- `choreo/derive.py` — auto-derive entry point.
- `choreo/cli.py` — new `derive` subcommand.
- `state_machines/reified.py` — `@transition` decorator.
- `state_machines/cell_fsm.py`, `state_machines/task_fsm.py` —
  builders updated with `enum_name=`.
- `swarm.py:HiveTask.claim/complete/fail/retry` — first reified
  application.
- `tests/test_choreo.py::TestDerive` — 6 derive tests.
- `tests/test_state_machines.py::TestReifiedDecoratorIsolated` — 5
  decorator unit tests.
- `tests/test_state_machines.py::TestReifiedHiveTask` — 6 HiveTask
  reified-method tests.
- ADR-008 — Phase 4.1 static checker that this builds on.
