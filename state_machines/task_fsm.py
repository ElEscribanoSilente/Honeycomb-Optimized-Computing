"""
TaskLifecycle FSM (Phase 4.4a, wired in Phase 4.1)
==================================================

Lifecycle of a :class:`hoc.swarm.HiveTask`:

::

    PENDING ─claim──► RUNNING ─success──► COMPLETED  (terminal)
       │                │
       │                ├─exception──────► FAILED ──retry──┐
       │                │                                  │
       └─shutdown───────┴──► CANCELLED  (terminal)         │
                                                           │
       ◄───────────────────────────────────────────────────┘

Wired via ``HiveTask.__setattr__`` (Phase 4.1)
----------------------------------------------

Phase 4 shipped this FSM as declarative-only. Phase 4.1 wires it into
``HiveTask.__setattr__`` so every ``task.state = X`` mutation routes
through :meth:`HocStateMachine.transition_to`. Illegal transitions (e.g.
``COMPLETED → RUNNING``, ``FAILED → COMPLETED`` without going through the
retry path) raise :class:`IllegalStateTransition`.

Two explicit test-fixture edges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tests/test_swarm.py`` uses five ``task.state = TaskState.{COMPLETED,
FAILED}`` assignments on a freshly-submitted task (still ``PENDING``) as
fault-injection shortcuts: instead of running the scheduler's
``tick()`` loop to drive a task to terminal state, the test forces the
state directly. To accommodate these without relaxing the FSM with
wildcards, two explicit trigger edges are declared:

- ``force_completed_from_pending`` — used by
  ``test_cancel_completed_task_fails`` and the three B2.5 index-leak
  tests (lines 451, 504, 505, 537 of ``tests/test_swarm.py``).
- ``force_failed_from_pending`` — used by
  ``test_b2_5_task_index_cleaned_after_failed`` (line 522).

These edges are **not** reachable from production call-sites in
``swarm.py`` — the scheduler always goes through ``PENDING → RUNNING``
before any terminal transition. They are documented here so a future
reader understands why the FSM permits them.

ASSIGNED is dead state (B12)
----------------------------

The :class:`hoc.swarm.TaskState` enum declares ``ASSIGNED`` (value 2)
between ``PENDING`` and ``RUNNING``. The brief proposes
``PENDING → ASSIGNED → RUNNING`` as the happy path. **No production
call-site or test ever writes ``task.state = TaskState.ASSIGNED``.**
Workers go straight ``PENDING → RUNNING`` when they claim a task in
``SwarmScheduler.execute_task``, ``SwarmScheduler.execute_with_failover``,
``SwarmScheduler.execute_with_circuit_breaker``, etc. (swarm.py:308, 382,
461, 531).

This FSM models the production flow: ``PENDING → RUNNING`` directly. The
``ASSIGNED`` state is **not** declared in this FSM. The Phase 4 closure
files this as **B12** with two options for resolution:

1. Remove ``ASSIGNED`` from the ``TaskState`` enum (the simpler fix —
   nothing depends on it).
2. Wire up the ``PENDING → ASSIGNED → RUNNING`` two-step in
   :class:`SwarmScheduler` (more accurately models the lock-acquire ←→
   begin-work boundary, but is a behavior change, out of scope for
   Phase 4).

Decision deferred to closure review with Raul.

Lifecycle reference
-------------------

- ``PENDING``: created and queued. ``HiveTask.__init__`` default.
- ``RUNNING``: worker has the task and is executing
  (``SwarmScheduler.execute_task`` etc.).
- ``COMPLETED``: terminal. Successful execution. ``callback`` (if any)
  fires.
- ``FAILED``: terminal *unless retry path triggers*. Exception caught
  during execution, or task expired (``HiveTask.is_expired``). May
  transition back to ``PENDING`` if ``can_retry()`` is true (swarm.py:1072).
- ``CANCELLED``: terminal. Set on shutdown for tasks still ``PENDING``
  (swarm.py:1010, 1127).
"""

from __future__ import annotations

from .base import HocStateMachine, HocTransition

TASK_PENDING = "PENDING"
TASK_RUNNING = "RUNNING"
TASK_COMPLETED = "COMPLETED"
TASK_FAILED = "FAILED"
TASK_CANCELLED = "CANCELLED"

# ASSIGNED is intentionally absent — see "ASSIGNED is dead state" in the
# module docstring. ``TaskState.ASSIGNED`` exists in the enum but no
# production code transitions to it. B12 in Phase 4 closure.

ALL_TASK_STATES: tuple[str, ...] = (
    TASK_PENDING,
    TASK_RUNNING,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_CANCELLED,
)

TERMINAL_TASK_STATES: frozenset[str] = frozenset({TASK_COMPLETED, TASK_CANCELLED})
"""States with no outgoing transitions in the happy lifecycle. ``FAILED``
is **not** terminal because the retry path can transition back to
``PENDING`` (swarm.py:1072)."""


def build_task_fsm() -> HocStateMachine:
    """
    Build the TaskLifecycle FSM. Wired into ``HiveTask.__setattr__`` in
    Phase 4.1 — every ``task.state = X`` mutation routes through
    :meth:`HocStateMachine.transition_to`.

    Used at construction time by ``HiveTask.__post_init__`` (one FSM per
    task). Also used by :mod:`scripts.generate_state_machines_md` and the
    property tests.
    """
    transitions: list[HocTransition] = [
        # ── Lifecycle ──────────────────────────────────────────────────
        # SwarmScheduler.execute_task / execute_with_failover /
        # execute_with_circuit_breaker / execute_with_retry — worker
        # claims a PENDING task and immediately runs it. swarm.py:308,
        # 382, 461, 531.
        HocTransition(TASK_PENDING, TASK_RUNNING, trigger="claimed"),
        # Successful tick. swarm.py:318, 394, 402, 468, 538.
        HocTransition(TASK_RUNNING, TASK_COMPLETED, trigger="completed"),
        # Exception during execution OR is_expired() returned True.
        # swarm.py:335, 408, 483, 552.
        HocTransition(TASK_RUNNING, TASK_FAILED, trigger="failed"),
        # Retry path. SwarmScheduler.execute_with_retry returns the task
        # to the queue if can_retry(). swarm.py:1072.
        HocTransition(TASK_FAILED, TASK_PENDING, trigger="retry"),
        # Shutdown / explicit cancel before pickup. swarm.py:1009-1010,
        # 1126-1127.
        HocTransition(TASK_PENDING, TASK_CANCELLED, trigger="cancelled_pending"),
        # Test fixtures and force-completion paths can mark a RUNNING
        # task CANCELLED if shutdown beat the worker. test_swarm.py:444,
        # 451 (force-completed path), 566 (CANCELLED race).
        HocTransition(TASK_RUNNING, TASK_CANCELLED, trigger="cancelled_running"),
        # ── Test-fixture edges (see module docstring) ──────────────────
        # Five test-sites assign terminal states to freshly-submitted
        # PENDING tasks to shortcut the scheduler loop. No production
        # path exercises these.
        HocTransition(TASK_PENDING, TASK_COMPLETED, trigger="force_completed_from_pending"),
        HocTransition(TASK_PENDING, TASK_FAILED, trigger="force_failed_from_pending"),
    ]

    return HocStateMachine(
        name="TaskLifecycle",
        states=list(ALL_TASK_STATES),
        transitions=transitions,
        initial=TASK_PENDING,
        history_size=4,
        # Phase 4.2: explicit binding to the host enum. choreo uses this
        # to detect B12-bis (TaskState.ASSIGNED in enum but not in FSM)
        # without relying on member-subset heuristics. String to avoid
        # circular import (swarm.py imports build_task_fsm from this
        # module).
        enum_name="TaskState",
    )
