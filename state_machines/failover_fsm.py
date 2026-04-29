"""
FailoverFlow FSM (Phase 4.6 + Phase 5.2c)
=========================================

Conceptual flow of a cell failover in
:class:`hoc.resilience.CellFailover`:

::

    HEALTHY ─circuit_open─► DEGRADED ─strategy_picked─► MIGRATING
                                                            │
                                              ┌─────────────┴──────────┐
                                              ▼                        ▼
                                          RECOVERED               LOST
                                              │                        │
                                              └──stabilized──► HEALTHY ◀┘
                                                                   ▲
                                                                   │
                                                              new_cells_provisioned
                                                                   │
                                                                   └── (from LOST)

Phase 5.2c wire-up
------------------

Wired in :meth:`hoc.resilience.CellFailover._migrate_work`:

- ``CellFailover._per_cell_phase: dict[HexCoord, FailoverPhase]`` tracks
  per-coord lifecycle.
- ``CellFailover._per_cell_fsm: dict[HexCoord, HocStateMachine]`` keeps
  one FSM instance per coord so ``tramoya.undo()`` works on the right
  history.
- ``CellFailover._last_failover_phase: FailoverPhase`` mirrors the most
  recent transition; this is the attribute the static checker
  ``choreo`` walks for the ``obj.attr = ENUM.MEMBER`` pattern (it does
  not yet handle dict subscript assignments).
- The lifecycle on a successful migration is
  ``HEALTHY → DEGRADED → MIGRATING → RECOVERED``. On exception the FSM
  ``undo()`` reverses the last transition (``MIGRATING → DEGRADED``).
- ``CellFailover.get_failover_phase(coord)`` exposes the per-cell phase
  to operators / observability tooling.

Undo
----

Tramoya's :meth:`tramoya.Machine.undo` is the natural fit for the
``MIGRATING → LOST`` failure path: if the workload migration stalls or
all mirror cells reject the workload, ``undo()`` returns the FSM to
``DEGRADED`` and the caller compensates by reattaching the workload to
the source (when the source is still responsive).

The caller of :meth:`hoc.resilience.CellFailover._migrate_work` is
still responsible for restoring vCores to the source cell when the FSM
rolls back; ``undo()`` only reverts FSM state, not external side
effects. Phase 5.2c restores ``CellState`` (the cell-level marker) and
the failover phase, but the vCore-level rollback is intentionally left
out (matches the pre-Phase-5 behaviour).
"""

from __future__ import annotations

from .base import HocStateMachine, HocTransition

FAILOVER_HEALTHY = "HEALTHY"
FAILOVER_DEGRADED = "DEGRADED"
FAILOVER_MIGRATING = "MIGRATING"
FAILOVER_RECOVERED = "RECOVERED"
FAILOVER_LOST = "LOST"

ALL_FAILOVER_STATES: tuple[str, ...] = (
    FAILOVER_HEALTHY,
    FAILOVER_DEGRADED,
    FAILOVER_MIGRATING,
    FAILOVER_RECOVERED,
    FAILOVER_LOST,
)


def build_failover_fsm() -> HocStateMachine:
    """Build the FailoverFlow FSM. Declarative-only — see module docstring."""
    transitions: list[HocTransition] = [
        # Circuit breaker on the source cell opens: we leave HEALTHY for
        # DEGRADED. cells_base.py:411 (CIRCUIT_BREAKER_OPENED event), and
        # CellFailover picks up the failed-cells set.
        HocTransition(
            FAILOVER_HEALTHY,
            FAILOVER_DEGRADED,
            trigger="circuit_opened",
            guard=lambda ctx: bool(ctx.get("circuit_open", False)),
        ),
        # Failover strategy selected and a target chosen — start migration.
        # CellFailover.migrate_cell is invoked. resilience.py:317-337.
        HocTransition(
            FAILOVER_DEGRADED,
            FAILOVER_MIGRATING,
            trigger="strategy_picked",
            guard=lambda ctx: bool(ctx.get("target_cell") is not None),
        ),
        # Migration succeeded: target accepted vCores. resilience.py:332
        # (the source is marked FAILED, target inherits load).
        HocTransition(
            FAILOVER_MIGRATING,
            FAILOVER_RECOVERED,
            trigger="migration_succeeded",
            guard=lambda ctx: bool(ctx.get("vcores_migrated", 0) > 0),
        ),
        # Migration timeout or all mirrors rejected the workload — task
        # is lost. resilience.py:335-337 catches Exception.
        HocTransition(FAILOVER_MIGRATING, FAILOVER_LOST, trigger="migration_failed"),
        # New cells provisioned to take over the failed coordinate.
        # SwarmRecovery / HexRedundancy fill the slot.
        HocTransition(
            FAILOVER_LOST,
            FAILOVER_HEALTHY,
            trigger="cells_provisioned",
            guard=lambda ctx: bool(ctx.get("replacement_count", 0) > 0),
        ),
        # Stabilization window: RECOVERED settles back to HEALTHY after
        # the new owner stays operational across N ticks.
        HocTransition(
            FAILOVER_RECOVERED,
            FAILOVER_HEALTHY,
            trigger="stabilized",
            guard=lambda ctx: bool(
                ctx.get("ticks_stable", 0) >= ctx.get("stabilization_window", 1)
            ),
        ),
    ]

    return HocStateMachine(
        name="FailoverFlow",
        states=list(ALL_FAILOVER_STATES),
        transitions=transitions,
        initial=FAILOVER_HEALTHY,
        # Larger history because undo on the migration path is part of
        # the design — keep enough states to support multi-step rollbacks.
        history_size=16,
        # Phase 5.2c: explicit binding to the host enum so choreo skips
        # member-subset heuristics. ``FailoverPhase`` lives in
        # ``resilience.py`` (next to the ``CellFailover`` class that
        # owns the per-cell state). String to avoid a circular import.
        enum_name="FailoverPhase",
    )
