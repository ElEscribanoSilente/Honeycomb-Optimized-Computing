# ADR-015: Class-level shared FSM in HoneycombCell

- **Status**: Accepted
- **Date**: 2026-04-27
- **Phase**: Phase 6 (6.6)

## Context

Phase 4 wired ``HoneycombCell.state`` to a per-instance
:class:`HocStateMachine` (one ``build_cell_fsm()`` per cell, stored
in the cell's ``_fsm`` slot). Each cell paid the cost of one
``tramoya.Machine`` allocation in ``__init__``.

PHASE_05_CLOSURE.md flagged ``test_grid_creation`` as the only real
performance regression versus the v1.0.0 baseline — +25.88 %
(1731 µs → 2179 µs). Profiling pinned the cause to the per-cell
FSM allocation: a radius-2 grid is 19 cells, radius-3 is 37, and
each one pays ~50 µs for the FSM construction.

The Phase 5 plan recommended "class-level shared FSM" as the fix
but flagged a subtle correctness issue: the underlying tramoya
machine has its own ``_machine.state``, and a shared instance would
contaminate across cells if every transition mutated it.

## Decision

Replace the per-cell FSM with a class-level spec object plus a
per-cell history slot, validated through a *pure* spec check that
never mutates the spec object's internal state.

### Mechanism

1. **``HocStateMachine.is_legal_transition(source, target)``** —
   new method (``state_machines/base.py``). Pure structural lookup
   against ``_dest_index``: returns ``True`` if an edge exists
   from ``source`` to ``target`` (wildcard sources match), else
   ``False``. Reads neither ``_machine.state`` nor any other
   instance state. Evaluates no guards. Returns ``False`` for
   unknown ``source`` / ``target`` (does not raise).

2. **``HoneycombCell._CLASS_FSM: ClassVar[HocStateMachine]``** —
   built once at class-definition time via ``build_cell_fsm()``.
   The spec object is consulted on every ``_set_state`` call but
   *never* driven through ``transition_to`` / ``trigger`` — its
   ``_machine.state`` stays at the initial value forever and the
   share is concurrency-safe.

3. **Per-cell state** lives where it always did
   (``self._state: CellState``) plus a new
   ``self._state_history: deque[str]`` (bounded by
   ``_HISTORY_MAXLEN=8``, matching the pre-fix per-instance
   ``history_size``).

4. **``_set_state(new_state)``** calls
   ``_CLASS_FSM.is_legal_transition``, raises
   :class:`IllegalStateTransition` with ``reason="no_edge"`` if
   the spec rejects, then appends to history before mutating
   ``_state``. Atomicity contract preserved (see Phase 5
   ``test_setter_atomicity_on_fsm_failure``).

5. **``cell.fsm``** returns a tiny ``_CellFsmView``: 1-slot proxy
   exposing ``state`` (= ``cell._state.name``), ``history`` (= a
   list copy of ``_state_history``), and ``transition_to(name)``
   (drives the cell setter; raises
   ``IllegalStateTransition(reason="unknown_state")`` for bogus
   names). The three call-sites that introspect the FSM
   (``test_cell_seal``, ``test_resilience``, ``test_state_machines``)
   only read ``state`` and ``history`` — both attributes are
   exactly the API the proxy presents.

### Why a proxy rather than direct attribute access

``cell.state.name`` returns the same string as ``cell.fsm.state``
in this design, so a reader might wonder why the proxy exists.
Reasons:

- **API stability.** Phase 4 documented ``cell.fsm`` as the
  introspection point. Removing it would break callers that
  read ``cell.fsm.state`` (the test suite has six of them).
- **History.** ``cell.fsm.history`` is the only public path to
  the bounded transition trail. Inlining it as
  ``cell.state_history`` would expose the deque type as part
  of the public API.
- **Cost.** Building a ``_CellFsmView`` is ~80 ns; reads of
  ``cell.fsm.state`` go via Python's attribute lookup
  machinery either way. The benchmarks show no observable
  regression (see below).

### Why not cache the proxy as a per-cell slot

The first instinct was ``__slots__ += ("_fsm_view",)`` and lazy
init. **Rejected** — that re-introduces one allocation per cell
in ``__init__`` (the slot itself), which is the exact cost the
fix exists to eliminate. Building the proxy on access (when most
cells are *not* introspected) is cheaper than allocating storage
for every cell up front.

## Bench impact

Measured with ``--benchmark-warmup=on --benchmark-min-time=0.5``
versus ``snapshot/bench_baseline.json`` (pre-Phase-5 baseline).
Phase 5 = the column the ADR is improving on; Phase 6.6 is the
post-fix number.

| benchmark              | Phase 5 (µs) | Phase 6.6 (µs) | Δ vs baseline |
|------------------------|--------------|----------------|---------------|
| test_grid_creation     | 2179         | 580            | **-66.47 %**  |
| test_grid_tick         | 477          | 463            | +3.91 %       |
| test_nectar_flow_tick  | 5.24         | 5.18           | -2.64 %       |
| test_dance_start       | 20.1         | 19.8           | -14.97 %      |

The Phase 5 ``test_grid_creation`` regression is gone — Phase 6.6
overshoots the original baseline by ~3× faster (the budget target
was ±5 %; the fix lands at -66 %). Other benches drift inside the
noise floor; nothing regresses past 10 % (the CI threshold
restored in Phase 6.7).

## Consequences

- **Memory.** ``HoneycombCell.__slots__`` net change: -1
  (``_fsm`` removed, ``_state_history`` added). Per-cell memory
  drops by ``sizeof(HocStateMachine) + sizeof(tramoya.Machine)``,
  ~hundreds of bytes; the bounded ``deque(maxlen=8)`` is
  dwarfed by it.
- **Concurrency.** Every cell's transition consults the same
  spec object. Reads of ``_dest_index`` are dict lookups, which
  the GIL serializes for free; the spec is otherwise immutable
  after construction. No new lock.
- **FSM history per cell** is now bounded at 8 entries
  (``_HISTORY_MAXLEN``). Phase 4 used the same value; the
  observable history shape is unchanged.
- **Test atomicity contract** preserved: a transition that
  fails the ``is_legal_transition`` check raises
  ``IllegalStateTransition`` *before* ``_state`` is mutated,
  same as pre-Phase-6.6.

## Alternatives considered

### Reset the shared FSM between transitions

Have every ``_set_state`` call ``_CLASS_FSM.reset(self._state.name)``
before ``transition_to``. **Rejected**: ``reset`` is a side
effect, and racing two threads on the same cell — let alone two
threads on two different cells — would be a footgun. The pure
``is_legal_transition`` check has no side effects and is safe by
construction.

### Cache the FSM in a module-level dict by class

``_FSM_BY_CLASS: dict[type, HocStateMachine] = {}``, lazy-init.
**Rejected**: marginal complexity for no benefit. ``ClassVar``
on ``HoneycombCell`` is the canonical Python way to express
"there's one of these per class".

### Move FSM validation to a free function

``state_machines.is_legal_cell_transition(source, target)``
short-circuiting the spec object entirely. **Rejected**: the
function would either re-build the spec on each call (defeats
the point) or cache it module-level (same as the current fix
but without the ``HocStateMachine`` API surface). Putting the
method on the wrapper class is the natural place; future FSMs
that adopt the same pattern (Phase 7+) get the helper for free.

## Status

Cardinal Phase 5 invariant — ``cell.state`` setter is FSM-
validated, illegal transitions raise — is preserved. The 858 tests
that pass at the close of Phase 6.5 also pass at the close of
Phase 6.6 with no edits.

choreo ``check --strict`` continues to report 0/0/0 — the FSM
spec graph is unchanged; only the storage of the runtime state
moved.
