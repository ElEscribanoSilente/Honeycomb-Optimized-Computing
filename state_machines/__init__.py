"""
HOC State Machines (Phase 4)
============================

Formal state machines for HOC's lifecycle-bearing types, built on top of
the ``tramoya`` library. This subpackage is the **only** place that imports
``tramoya`` directly — same isolation tactic Phase 2 used for ``mscs`` in
:mod:`hoc.security`.

Modules in this package:

- :mod:`hoc.state_machines.base` — :class:`HocStateMachine` adapter and
  :class:`IllegalStateTransition` exception. All other FSMs use this.
- :mod:`hoc.state_machines.cell_fsm` — :class:`CellState` lifecycle
  (Phase 4.3).
- :mod:`hoc.state_machines.task_fsm` — :class:`TaskState` lifecycle
  (Phase 4.4a).
- :mod:`hoc.state_machines.succession_fsm` — :class:`QueenSuccession`
  election lifecycle (Phase 4.4b).
- :mod:`hoc.state_machines.pheromone_fsm` — :class:`PheromoneDeposit`
  decay lifecycle (Phase 4.5).
- :mod:`hoc.state_machines.failover_fsm` — :class:`CellFailover` migration
  flow with undo (Phase 4.6).

Design principle (cardinal invariant of Phase 4):
*FSMs **observe** the existing transitions of HOC; they do not change them.*
Each FSM is descriptive — it documents and validates the transitions the
production code already executes — not aspirational.
"""

from __future__ import annotations

from .base import (
    WILDCARD,
    HocStateMachine,
    HocTransition,
    IllegalStateTransition,
)
from .reified import transition

__all__ = [
    "HocStateMachine",
    "HocTransition",
    "IllegalStateTransition",
    "WILDCARD",
    "transition",
]
