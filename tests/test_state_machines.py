"""
Phase 4 unit tests for the state machines subpackage.

Covers:

- :class:`HocStateMachine` wrapper API (transition_to, can_transition_to,
  trigger, undo, reset, observers, mermaid/dot export).
- :class:`IllegalStateTransition` exception surface (attributes for
  ``fsm_name``, ``source``, ``target``, ``reason``).
- Each of the five FSMs (Cell, Pheromone, Task, Succession, Failover):
  state set, initial state, sample legal transitions, sample illegal
  transitions raise.
- The CellState FSM wired into ``HoneycombCell``: cell.state.setter
  rejects illegal transitions with :class:`IllegalStateTransition`.
"""

from __future__ import annotations

import pytest

from core.cells_base import CellRole, CellState, HoneycombCell
from core.grid_geometry import HexCoord
from hoc.swarm import HiveTask, TaskState
from state_machines.base import (
    WILDCARD,
    HocStateMachine,
    HocTransition,
    IllegalStateTransition,
)
from state_machines.cell_fsm import (
    ALL_CELL_STATES,
    CELL_STATE_EMPTY,
    CELL_STATE_FAILED,
    CELL_STATE_IDLE,
    CELL_STATE_MIGRATING,
    CELL_STATE_RECOVERING,
    CELL_STATE_SEALED,
    build_cell_fsm,
)
from state_machines.failover_fsm import (
    ALL_FAILOVER_STATES,
    FAILOVER_DEGRADED,
    FAILOVER_HEALTHY,
    FAILOVER_LOST,
    FAILOVER_MIGRATING,
    FAILOVER_RECOVERED,
    build_failover_fsm,
)
from state_machines.pheromone_fsm import (
    ALL_PHEROMONE_STATES,
    PHEROMONE_DECAYING,
    PHEROMONE_DIFFUSING,
    PHEROMONE_EVAPORATED,
    PHEROMONE_FRESH,
    build_pheromone_fsm,
)
from state_machines.succession_fsm import (
    ALL_SUCCESSION_STATES,
    SUCCESSION_DETECTING,
    SUCCESSION_ELECTED,
    SUCCESSION_FAILED,
    SUCCESSION_NOMINATING,
    SUCCESSION_STABLE,
    SUCCESSION_VOTING,
    build_succession_fsm,
)
from state_machines.task_fsm import (
    ALL_TASK_STATES,
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_PENDING,
    TASK_RUNNING,
    TERMINAL_TASK_STATES,
    build_task_fsm,
)

# ─── HocStateMachine wrapper ──────────────────────────────────────────────────


class TestHocStateMachineConstruction:
    def test_minimal_machine(self):
        fsm = HocStateMachine(
            name="t",
            states=["a", "b"],
            transitions=[HocTransition("a", "b")],
            initial="a",
        )
        assert fsm.name == "t"
        assert fsm.state == "a"
        assert fsm.initial == "a"
        assert fsm.states == {"a", "b"}

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="non-empty name"):
            HocStateMachine(
                name="",
                states=["a"],
                transitions=[],
                initial="a",
            )

    def test_auto_trigger_naming(self):
        fsm = HocStateMachine(
            name="t",
            states=["a", "b", "c"],
            transitions=[
                HocTransition("a", "b"),
                HocTransition("b", "c"),
            ],
            initial="a",
        )
        # Auto-generated names follow "<src>__to__<dst>"
        assert "a__to__b" in fsm.available_triggers

    def test_explicit_trigger_kept(self):
        fsm = HocStateMachine(
            name="t",
            states=["a", "b"],
            transitions=[HocTransition("a", "b", trigger="go")],
            initial="a",
        )
        assert "go" in fsm.available_triggers


class TestTransitionTo:
    def setup_method(self):
        self.fsm = HocStateMachine(
            name="t",
            states=["a", "b", "c"],
            transitions=[
                HocTransition("a", "b"),
                HocTransition("b", "c", guard=lambda ctx: bool(ctx.get("ok", False))),
                HocTransition(WILDCARD, "a", trigger="reset"),
            ],
            initial="a",
        )

    def test_legal_transition(self):
        new_state = self.fsm.transition_to("b")
        assert new_state == "b"
        assert self.fsm.state == "b"

    def test_illegal_no_edge(self):
        with pytest.raises(IllegalStateTransition) as excinfo:
            self.fsm.transition_to("c")  # a -> c: no edge
        assert excinfo.value.reason == "no_edge"
        assert excinfo.value.source == "a"
        assert excinfo.value.target == "c"
        assert excinfo.value.fsm_name == "t"

    def test_unknown_target(self):
        with pytest.raises(IllegalStateTransition) as excinfo:
            self.fsm.transition_to("z")
        assert excinfo.value.reason == "unknown_state"

    def test_guard_rejection(self):
        self.fsm.transition_to("b")
        with pytest.raises(IllegalStateTransition) as excinfo:
            self.fsm.transition_to("c", ok=False)
        assert excinfo.value.reason == "guard_rejected"

    def test_guard_acceptance(self):
        self.fsm.transition_to("b")
        new_state = self.fsm.transition_to("c", ok=True)
        assert new_state == "c"

    def test_wildcard_path_used_when_explicit_missing(self):
        # From a, transition_to("a") via wildcard "reset" trigger.
        self.fsm.transition_to("b")
        new_state = self.fsm.transition_to("a")
        assert new_state == "a"

    def test_explicit_takes_priority_over_wildcard(self):
        # Construct an FSM where a wildcard and an explicit transition both
        # land on the same dest. Explicit should win.
        order: list[str] = []
        fsm = HocStateMachine(
            name="t",
            states=["a", "b"],
            transitions=[
                HocTransition(
                    "a", "b", trigger="explicit", action=lambda c: order.append("explicit")
                ),
                HocTransition(
                    WILDCARD, "b", trigger="wildcard", action=lambda c: order.append("wildcard")
                ),
            ],
            initial="a",
        )
        fsm.transition_to("b")
        assert order == ["explicit"]


class TestCanTransitionTo:
    def test_returns_true_for_legal(self):
        fsm = HocStateMachine(
            name="t",
            states=["a", "b"],
            transitions=[HocTransition("a", "b")],
            initial="a",
        )
        assert fsm.can_transition_to("b") is True

    def test_returns_false_for_unknown_target(self):
        fsm = HocStateMachine(
            name="t",
            states=["a"],
            transitions=[],
            initial="a",
        )
        assert fsm.can_transition_to("z") is False

    def test_returns_false_when_guard_blocks(self):
        fsm = HocStateMachine(
            name="t",
            states=["a", "b"],
            transitions=[HocTransition("a", "b", guard=lambda ctx: False)],
            initial="a",
        )
        assert fsm.can_transition_to("b") is False

    def test_guard_evaluation_does_not_mutate_ctx(self):
        # Guards receive a frozen view in tramoya. Smoke check that we
        # don't accidentally mutate via the wrapper.
        fsm = HocStateMachine(
            name="t",
            states=["a", "b"],
            transitions=[HocTransition("a", "b", guard=lambda ctx: ctx.get("k", False))],
            initial="a",
            ctx={"original": True},
        )
        fsm.can_transition_to("b", k=True)
        # Original ctx should not have been polluted by kwargs.
        assert "k" not in fsm.ctx
        assert fsm.ctx["original"] is True


class TestUndoAndReset:
    def test_undo_returns_to_previous_state(self):
        fsm = HocStateMachine(
            name="t",
            states=["a", "b", "c"],
            transitions=[
                HocTransition("a", "b"),
                HocTransition("b", "c"),
            ],
            initial="a",
        )
        fsm.transition_to("b")
        fsm.transition_to("c")
        fsm.undo()
        assert fsm.state == "b"
        fsm.undo()
        assert fsm.state == "a"

    def test_undo_empty_history_raises(self):
        fsm = HocStateMachine(
            name="t",
            states=["a", "b"],
            transitions=[HocTransition("a", "b")],
            initial="a",
        )
        with pytest.raises(IllegalStateTransition) as excinfo:
            fsm.undo()
        assert excinfo.value.reason == "empty_history"

    def test_reset_to_initial(self):
        fsm = HocStateMachine(
            name="t",
            states=["a", "b"],
            transitions=[HocTransition("a", "b")],
            initial="a",
        )
        fsm.transition_to("b")
        fsm.reset()
        assert fsm.state == "a"

    def test_reset_to_explicit_state(self):
        fsm = HocStateMachine(
            name="t",
            states=["a", "b", "c"],
            transitions=[HocTransition("a", "b")],
            initial="a",
        )
        fsm.reset("c")
        assert fsm.state == "c"

    def test_reset_to_unknown_raises(self):
        fsm = HocStateMachine(
            name="t",
            states=["a"],
            transitions=[],
            initial="a",
        )
        with pytest.raises(IllegalStateTransition) as excinfo:
            fsm.reset("z")
        assert excinfo.value.reason == "unknown_state"


class TestObservers:
    def test_observer_fires_on_transition(self):
        fsm = HocStateMachine(
            name="t",
            states=["a", "b"],
            transitions=[HocTransition("a", "b")],
            initial="a",
        )
        events: list[tuple[str, str, str]] = []

        def obs(trigger, src, dst, ctx):
            events.append((trigger, src, dst))

        fsm.subscribe(obs)
        fsm.transition_to("b")
        assert events == [("a__to__b", "a", "b")]

    def test_unsubscribe_stops_events(self):
        fsm = HocStateMachine(
            name="t",
            states=["a", "b", "c"],
            transitions=[HocTransition("a", "b"), HocTransition("b", "c")],
            initial="a",
        )
        events: list[tuple[str, str, str]] = []

        def obs(trigger, src, dst, ctx):
            events.append((trigger, src, dst))

        fsm.subscribe(obs)
        fsm.transition_to("b")
        fsm.unsubscribe(obs)
        fsm.transition_to("c")
        assert len(events) == 1


class TestVisualization:
    def test_mermaid_contains_transitions(self):
        fsm = HocStateMachine(
            name="t",
            states=["a", "b"],
            transitions=[HocTransition("a", "b", trigger="go")],
            initial="a",
        )
        out = fsm.to_mermaid()
        assert out.startswith("stateDiagram-v2")
        assert "a --> b : go" in out

    def test_dot_contains_transitions(self):
        fsm = HocStateMachine(
            name="t",
            states=["a", "b"],
            transitions=[HocTransition("a", "b", trigger="go")],
            initial="a",
        )
        out = fsm.to_dot()
        assert out.startswith("digraph")
        assert "go" in out


# ─── CellState FSM (wired) ────────────────────────────────────────────────────


class TestCellStateFSMStandalone:
    def test_state_count(self):
        fsm = build_cell_fsm()
        # Phase 4.3: SPAWNING and OVERLOADED removed; 7 states remain.
        assert len(fsm.states) == 7
        assert fsm.states == set(ALL_CELL_STATES)

    def test_initial_is_empty(self):
        fsm = build_cell_fsm()
        assert fsm.state == CELL_STATE_EMPTY

    def test_lifecycle_empty_to_idle(self):
        fsm = build_cell_fsm()
        fsm.transition_to(CELL_STATE_IDLE)
        assert fsm.state == CELL_STATE_IDLE

    def test_unknown_state_name_raises(self):
        # Phase 5.1 wired MIGRATING (admin_start_migration) and SEALED
        # (admin_seal), so no enum CellState member is dead anymore.
        # The FSM's "unknown_state" rejection path is exercised here with
        # a non-enum name to keep coverage on the IllegalStateTransition
        # error surface.
        fsm = build_cell_fsm()
        with pytest.raises(IllegalStateTransition) as excinfo:
            fsm.transition_to("NOT_A_REAL_STATE")
        assert excinfo.value.reason == "unknown_state"

    def test_admin_start_migration_from_active_succeeds(self):
        # Phase 5.1: CellFailover._migrate_work uses admin_start_migration
        # to mark the source cell MIGRATING from any live state.
        fsm = build_cell_fsm()
        fsm.transition_to(CELL_STATE_IDLE)  # vcore_added
        fsm.transition_to("ACTIVE")  # tick_started
        fsm.transition_to(CELL_STATE_MIGRATING)
        assert fsm.state == CELL_STATE_MIGRATING

    def test_admin_seal_from_idle_succeeds(self):
        # Phase 5.1: HoneycombCell.seal() targets SEALED via admin_seal.
        fsm = build_cell_fsm()
        fsm.transition_to(CELL_STATE_IDLE)
        fsm.transition_to(CELL_STATE_SEALED)
        assert fsm.state == CELL_STATE_SEALED

    def test_admin_seal_from_empty_succeeds(self):
        # Sealing a freshly-constructed cell (still EMPTY) is valid.
        fsm = build_cell_fsm()
        fsm.transition_to(CELL_STATE_SEALED)
        assert fsm.state == CELL_STATE_SEALED

    def test_admin_start_migration_from_empty_succeeds(self):
        # An EMPTY cell can also be migrated (no-op vcore loop) — wildcard
        # source admits it.
        fsm = build_cell_fsm()
        fsm.transition_to(CELL_STATE_MIGRATING)
        assert fsm.state == CELL_STATE_MIGRATING

    def test_admin_path_failed_from_anywhere(self):
        fsm = build_cell_fsm()
        # From EMPTY, admin_mark_failed (wildcard) should work.
        fsm.transition_to(CELL_STATE_FAILED)
        assert fsm.state == CELL_STATE_FAILED

    def test_admin_recover_path(self):
        fsm = build_cell_fsm()
        fsm.transition_to(CELL_STATE_FAILED)  # admin
        fsm.transition_to(CELL_STATE_RECOVERING)  # via FAILED -> RECOVERING (recovery_started)
        fsm.transition_to(CELL_STATE_EMPTY)  # via RECOVERING -> EMPTY (recovery_completed)
        assert fsm.state == CELL_STATE_EMPTY


class TestCellStateFSMWired:
    """Phase 4: HoneycombCell.state.setter routes through the FSM."""

    def _make_cell(self):
        return HoneycombCell(coord=HexCoord(0, 0), role=CellRole.WORKER)

    def test_cell_starts_in_empty(self):
        cell = self._make_cell()
        assert cell.state == CellState.EMPTY
        assert cell.fsm.state == CELL_STATE_EMPTY

    def test_legal_transition_via_setter(self):
        cell = self._make_cell()
        cell.state = CellState.IDLE  # explicit lifecycle: EMPTY -> IDLE
        assert cell.state == CellState.IDLE
        assert cell.fsm.state == CELL_STATE_IDLE

    def test_setter_atomicity_on_fsm_failure(self):
        # Phase 5.1 wired all CellState members, so no enum value triggers
        # an FSM rejection through the typed setter. The setter's
        # atomicity contract ("if FSM raises, _state is not mutated") is
        # still load-bearing — exercise it via cell.fsm directly with a
        # bogus state name that the FSM rejects with reason="unknown_state".
        cell = self._make_cell()
        initial = cell.state
        with pytest.raises(IllegalStateTransition) as excinfo:
            cell.fsm.transition_to("NOT_A_REAL_STATE")
        assert excinfo.value.reason == "unknown_state"
        # The cell's _state attribute is unchanged regardless.
        assert cell.state == initial

    def test_idempotent_assignment_skips_fsm(self):
        # Old contract: setting state to current value is a no-op. Phase 4
        # preserves this — the FSM is not consulted, history is not pushed.
        cell = self._make_cell()
        cell.state = CellState.EMPTY  # already EMPTY
        assert cell.fsm.history == []  # history empty: no transition fired

    def test_admin_failover_path_via_setter(self):
        # Wildcard `admin_mark_failed` lets any state move to FAILED,
        # mirroring CellFailover.migrate_cell behaviour at resilience.py:332.
        cell = self._make_cell()
        cell.state = CellState.FAILED
        assert cell.state == CellState.FAILED


# ─── PheromoneDeposit FSM ─────────────────────────────────────────────────────


class TestPheromoneFSM:
    def test_state_count_and_initial(self):
        fsm = build_pheromone_fsm()
        assert fsm.states == set(ALL_PHEROMONE_STATES)
        assert fsm.state == PHEROMONE_FRESH

    def test_aged_out_guarded(self):
        fsm = build_pheromone_fsm()
        # Without age beyond freshness window, FRESH -> DECAYING is blocked.
        with pytest.raises(IllegalStateTransition) as excinfo:
            fsm.transition_to(PHEROMONE_DECAYING, age=0.0, freshness_window=5.0)
        assert excinfo.value.reason == "guard_rejected"

        fsm.transition_to(PHEROMONE_DECAYING, age=10.0, freshness_window=5.0)
        assert fsm.state == PHEROMONE_DECAYING

    def test_evaporated_is_terminal(self):
        fsm = build_pheromone_fsm()
        fsm.transition_to(PHEROMONE_DECAYING, age=10.0, freshness_window=5.0)
        fsm.transition_to(
            PHEROMONE_EVAPORATED,
            intensity=0.0,
            cleanup_threshold=0.001,
        )
        # No outgoing transitions from EVAPORATED.
        assert fsm.is_final

    def test_diffuse_round_trip(self):
        fsm = build_pheromone_fsm()
        fsm.transition_to(PHEROMONE_DECAYING, age=10.0, freshness_window=5.0)
        fsm.transition_to(
            PHEROMONE_DIFFUSING,
            intensity=1.0,
            diffuse_threshold=0.01,
            neighbors_reachable=3,
        )
        fsm.transition_to(PHEROMONE_DECAYING)
        assert fsm.state == PHEROMONE_DECAYING


# ─── TaskLifecycle FSM ────────────────────────────────────────────────────────


class TestTaskFSM:
    def test_state_count_and_initial(self):
        fsm = build_task_fsm()
        assert fsm.states == set(ALL_TASK_STATES)
        assert fsm.state == TASK_PENDING

    def test_assigned_is_dead(self):
        # ASSIGNED is intentionally absent from this FSM (B12 candidate).
        # The TaskState enum has it but the FSM does not.
        fsm = build_task_fsm()
        assert "ASSIGNED" not in fsm.states

    def test_happy_path(self):
        fsm = build_task_fsm()
        fsm.transition_to(TASK_RUNNING)
        fsm.transition_to(TASK_COMPLETED)
        assert fsm.state == TASK_COMPLETED

    def test_retry_from_failed(self):
        fsm = build_task_fsm()
        fsm.transition_to(TASK_RUNNING)
        fsm.transition_to(TASK_FAILED)
        # Retry: FAILED -> PENDING.
        fsm.transition_to(TASK_PENDING)
        assert fsm.state == TASK_PENDING

    def test_failed_is_not_terminal_due_to_retry(self):
        # FAILED has outgoing edge (retry).
        assert TASK_FAILED not in TERMINAL_TASK_STATES

    def test_completed_and_cancelled_terminal(self):
        # COMPLETED and CANCELLED are documented as terminal.
        assert TASK_COMPLETED in TERMINAL_TASK_STATES
        assert TASK_CANCELLED in TERMINAL_TASK_STATES

    def test_cancel_running_path(self):
        fsm = build_task_fsm()
        fsm.transition_to(TASK_RUNNING)
        fsm.transition_to(TASK_CANCELLED)
        assert fsm.state == TASK_CANCELLED

    def test_no_path_from_completed(self):
        fsm = build_task_fsm()
        fsm.transition_to(TASK_RUNNING)
        fsm.transition_to(TASK_COMPLETED)
        with pytest.raises(IllegalStateTransition):
            fsm.transition_to(TASK_PENDING)


class TestTaskFSMWired:
    """Phase 4.1: HiveTask.__setattr__ routes ``state`` mutations through
    the FSM. Illegal transitions raise :class:`IllegalStateTransition`
    and the state field is not mutated."""

    def test_task_starts_in_pending(self):
        task = HiveTask(priority=1)
        assert task.state == TaskState.PENDING
        assert task._fsm.state == TASK_PENDING

    def test_legal_transition_pending_to_running(self):
        task = HiveTask(priority=1)
        task.state = TaskState.RUNNING
        assert task.state == TaskState.RUNNING
        assert task._fsm.state == TASK_RUNNING

    def test_happy_path_running_to_completed(self):
        task = HiveTask(priority=1)
        task.state = TaskState.RUNNING
        task.state = TaskState.COMPLETED
        assert task._fsm.state == TASK_COMPLETED

    def test_illegal_transition_completed_to_running_raises(self):
        # COMPLETED is terminal (no outgoing edge back to RUNNING).
        task = HiveTask(priority=1)
        task.state = TaskState.RUNNING
        task.state = TaskState.COMPLETED
        with pytest.raises(IllegalStateTransition) as excinfo:
            task.state = TaskState.RUNNING
        assert excinfo.value.reason == "no_edge"
        # State did not mutate.
        assert task.state == TaskState.COMPLETED

    def test_idempotent_assignment_skips_fsm(self):
        # Assigning the same state is a no-op — FSM is not consulted,
        # history is not pushed.
        task = HiveTask(priority=1)
        task.state = TaskState.RUNNING
        assert len(task._fsm.history) == 1  # PENDING -> RUNNING recorded
        task.state = TaskState.RUNNING  # idempotent
        assert len(task._fsm.history) == 1  # still 1

    def test_force_completed_from_pending_legal(self):
        # Test-fixture edge (see task_fsm module docstring). The five
        # test_swarm.py sites that force terminal states on PENDING tasks
        # must keep working.
        task = HiveTask(priority=1)
        task.state = TaskState.COMPLETED  # PENDING -> COMPLETED direct
        assert task._fsm.state == TASK_COMPLETED

    def test_force_failed_from_pending_legal(self):
        task = HiveTask(priority=1)
        task.state = TaskState.FAILED  # PENDING -> FAILED direct
        assert task._fsm.state == TASK_FAILED

    def test_non_default_state_at_construction_syncs_fsm(self):
        # Caller passes state=RUNNING via __init__. The FSM must seed
        # to RUNNING, not stay at PENDING (which would make subsequent
        # transitions validate from the wrong source).
        task = HiveTask(priority=1, state=TaskState.RUNNING)
        assert task._fsm.state == TASK_RUNNING
        # And from there, a legal transition proceeds.
        task.state = TaskState.COMPLETED
        assert task._fsm.state == TASK_COMPLETED

    def test_retry_path_failed_to_pending(self):
        # FAILED -> PENDING is the retry edge. Used by swarm.py:1072.
        task = HiveTask(priority=1)
        task.state = TaskState.RUNNING
        task.state = TaskState.FAILED
        task.state = TaskState.PENDING  # retry
        assert task._fsm.state == TASK_PENDING


# ─── Reified transitions (Phase 4.2) ──────────────────────────────────────────


class TestReifiedDecoratorIsolated:
    """Phase 4.2: ``@transition`` decorator unit tests on a synthetic
    host class (no HiveTask, no FSM wire-up) to verify the decorator's
    behavior in isolation."""

    def _make_host(self):
        from state_machines.reified import transition

        class Host:
            def __init__(self):
                self.state = "A"
                self.work_done = False

            @transition(from_="A", to="B")
            def go_b(self):
                self.work_done = True

            @transition(from_=None, to="C")
            def force_c(self):
                # from_=None means "any source"
                pass

            @transition(from_="A", to="B")
            def boom(self):
                raise RuntimeError("kaboom")

        return Host

    def test_transition_runs_method_and_mutates_state(self):
        Host = self._make_host()
        h = Host()
        h.go_b()
        assert h.state == "B"
        assert h.work_done is True

    def test_from_mismatch_raises(self):
        Host = self._make_host()
        h = Host()
        h.state = "X"  # Bypass: we test the decorator's pre-check.
        with pytest.raises(IllegalStateTransition) as excinfo:
            h.go_b()
        assert excinfo.value.reason == "reified_from_mismatch"
        assert h.state == "X"  # No mutation.
        assert getattr(h, "work_done", False) is False

    def test_method_exception_does_not_mutate(self):
        Host = self._make_host()
        h = Host()
        with pytest.raises(RuntimeError, match="kaboom"):
            h.boom()
        assert h.state == "A"  # Decorator did NOT advance to "B".

    def test_from_none_skips_precondition(self):
        Host = self._make_host()
        h = Host()
        h.state = "anything-goes"
        h.force_c()
        assert h.state == "C"

    def test_metadata_attached(self):
        Host = self._make_host()
        assert Host.go_b.__choreo_transition__ == ("A", "B")
        assert Host.force_c.__choreo_transition__ == (None, "C")


class TestReifiedHiveTask:
    """Phase 4.2: HiveTask.claim/complete/fail/retry are reified
    transitions over the existing TaskLifecycle wire-up."""

    def test_claim_running(self):
        task = HiveTask(priority=1)
        # We need a HoneycombCell-like worker; use the resilience tests'
        # helper or just construct one directly.
        worker = HoneycombCell(coord=HexCoord(1, 1), role=CellRole.WORKER)
        task.claim(worker)
        assert task.state == TaskState.RUNNING
        assert task.assigned_to == worker.coord

    def test_complete_stores_result(self):
        task = HiveTask(priority=1)
        worker = HoneycombCell(coord=HexCoord(1, 1), role=CellRole.WORKER)
        task.claim(worker)
        task.complete(result="ok")
        assert task.state == TaskState.COMPLETED
        assert task.result == "ok"

    def test_fail_increments_attempts(self):
        task = HiveTask(priority=1)
        worker = HoneycombCell(coord=HexCoord(1, 1), role=CellRole.WORKER)
        task.claim(worker)
        task.fail("oh no")
        assert task.state == TaskState.FAILED
        assert task.error == "oh no"
        assert task.attempts == 1

    def test_retry_loops_back_to_pending(self):
        task = HiveTask(priority=1)
        worker = HoneycombCell(coord=HexCoord(1, 1), role=CellRole.WORKER)
        task.claim(worker)
        task.fail("oh no")
        task.retry()
        assert task.state == TaskState.PENDING
        assert task.error is None
        assert task.assigned_to is None

    def test_claim_from_running_raises(self):
        # Claim is from PENDING; cannot claim a RUNNING task.
        task = HiveTask(priority=1)
        worker = HoneycombCell(coord=HexCoord(1, 1), role=CellRole.WORKER)
        task.claim(worker)
        with pytest.raises(IllegalStateTransition):
            task.claim(worker)

    def test_method_exception_leaves_state_unchanged(self):
        # A method body that raises should leave state unchanged.
        from state_machines.reified import transition

        class BoomTask(HiveTask):
            @transition(from_=TaskState.RUNNING, to=TaskState.COMPLETED)
            def boom_complete(self) -> None:
                raise RuntimeError("explode")

        bt = BoomTask(priority=1)
        bt.state = TaskState.RUNNING
        with pytest.raises(RuntimeError, match="explode"):
            bt.boom_complete()
        assert bt.state == TaskState.RUNNING  # not COMPLETED


# ─── QueenSuccession FSM ──────────────────────────────────────────────────────


class TestSuccessionFSM:
    def test_state_count_and_initial(self):
        fsm = build_succession_fsm()
        assert fsm.states == set(ALL_SUCCESSION_STATES)
        assert fsm.state == SUCCESSION_STABLE

    def test_full_election_path(self):
        fsm = build_succession_fsm()
        fsm.transition_to(
            SUCCESSION_DETECTING,
            elapsed_since_heartbeat=10.0,
            timeout_threshold=5.0,
        )
        fsm.transition_to(
            SUCCESSION_NOMINATING,
            missed_heartbeats=3,
            confirm_ticks=2,
        )
        fsm.transition_to(SUCCESSION_VOTING, candidate_count=5)
        fsm.transition_to(
            SUCCESSION_ELECTED,
            quorum_reached=True,
            signatures_valid=True,
            term_matches=True,
        )
        # ELECTED -> STABLE after promote.
        fsm.transition_to(SUCCESSION_STABLE)
        assert fsm.state == SUCCESSION_STABLE

    def test_voting_to_elected_blocked_without_quorum(self):
        fsm = build_succession_fsm()
        fsm.reset(SUCCESSION_VOTING)
        with pytest.raises(IllegalStateTransition) as excinfo:
            fsm.transition_to(
                SUCCESSION_ELECTED,
                quorum_reached=False,
                signatures_valid=True,
                term_matches=True,
            )
        assert excinfo.value.reason == "guard_rejected"

    def test_voting_to_elected_blocked_without_signatures(self):
        fsm = build_succession_fsm()
        fsm.reset(SUCCESSION_VOTING)
        with pytest.raises(IllegalStateTransition):
            fsm.transition_to(
                SUCCESSION_ELECTED,
                quorum_reached=True,
                signatures_valid=False,
                term_matches=True,
            )

    def test_failed_can_cooldown_to_stable(self):
        fsm = build_succession_fsm()
        fsm.reset(SUCCESSION_FAILED)
        fsm.transition_to(SUCCESSION_STABLE)
        assert fsm.state == SUCCESSION_STABLE


# ─── FailoverFlow FSM ─────────────────────────────────────────────────────────


class TestFailoverFSM:
    def test_state_count_and_initial(self):
        fsm = build_failover_fsm()
        assert fsm.states == set(ALL_FAILOVER_STATES)
        assert fsm.state == FAILOVER_HEALTHY

    def test_full_migration_path(self):
        fsm = build_failover_fsm()
        fsm.transition_to(FAILOVER_DEGRADED, circuit_open=True)
        fsm.transition_to(FAILOVER_MIGRATING, target_cell="some-coord")
        fsm.transition_to(FAILOVER_RECOVERED, vcores_migrated=4)
        fsm.transition_to(
            FAILOVER_HEALTHY,
            ticks_stable=5,
            stabilization_window=3,
        )
        assert fsm.state == FAILOVER_HEALTHY

    def test_migration_undo_path(self):
        fsm = build_failover_fsm()
        fsm.transition_to(FAILOVER_DEGRADED, circuit_open=True)
        fsm.transition_to(FAILOVER_MIGRATING, target_cell="some-coord")
        # Undo MIGRATING — back to DEGRADED.
        fsm.undo()
        assert fsm.state == FAILOVER_DEGRADED

    def test_migration_failure_to_lost(self):
        fsm = build_failover_fsm()
        fsm.transition_to(FAILOVER_DEGRADED, circuit_open=True)
        fsm.transition_to(FAILOVER_MIGRATING, target_cell="some-coord")
        fsm.transition_to(FAILOVER_LOST)
        assert fsm.state == FAILOVER_LOST

    def test_lost_recovery_via_provisioning(self):
        fsm = build_failover_fsm()
        fsm.reset(FAILOVER_LOST)
        fsm.transition_to(FAILOVER_HEALTHY, replacement_count=2)
        assert fsm.state == FAILOVER_HEALTHY
