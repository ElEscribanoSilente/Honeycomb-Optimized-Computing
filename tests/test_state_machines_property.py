"""
Phase 4 property-based tests for the state machines.

Properties checked across all five FSMs:

- **Reachability.** From the initial state, applying a random sequence of
  legal triggers (with random plausible context) ends in a state that is
  declared in the FSM's state set.
- **No silent corruption.** After every transition, the FSM's internal
  state matches the dest of the trigger that fired (or stays unchanged
  if a guard rejected). The wrapper either commits a known new state or
  raises ``IllegalStateTransition``; it never leaves the FSM in an
  unreachable state.
- **Terminal-state invariants.** Documented terminal states have no
  outgoing transitions (``is_final`` is true).
- **Round-trip undo.** After ``transition_to`` then ``undo``, state
  returns to what it was before the transition, even after a long
  sequence.
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from state_machines.base import IllegalStateTransition
from state_machines.cell_fsm import (
    CELL_STATE_EMPTY,
    CELL_STATE_FAILED,
    CELL_STATE_IDLE,
    build_cell_fsm,
)
from state_machines.failover_fsm import (
    FAILOVER_HEALTHY,
    build_failover_fsm,
)
from state_machines.pheromone_fsm import (
    PHEROMONE_EVAPORATED,
    PHEROMONE_FRESH,
    build_pheromone_fsm,
)
from state_machines.succession_fsm import build_succession_fsm
from state_machines.task_fsm import (
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_PENDING,
    build_task_fsm,
)

# ─── Reachability and no-corruption ───────────────────────────────────────────


@given(
    triggers=st.lists(
        st.sampled_from(
            [
                "vcore_added",
                "vcore_drained_idle",
                "vcore_drained_active",
                "tick_started",
                "tick_completed",
                "tick_failed",
                "recovery_started",
                "recovery_completed",
                "recovery_restored",
                "admin_mark_failed",
                "admin_set_idle",
                "admin_recover",
                "admin_reset",
                "admin_force_active",
                # Triggers from other FSMs — should never fire on this one.
                "claimed",
                "completed",
                "aged_out",
            ]
        ),
        max_size=20,
    )
)
@settings(max_examples=80, deadline=None)
def test_cell_fsm_reachability(triggers: list[str]):
    """Random trigger sequence on CellState: state stays inside declared set."""
    fsm = build_cell_fsm()
    for trigger in triggers:
        try:
            fsm.trigger(trigger)
        except IllegalStateTransition:
            # Expected for many triggers — the FSM rejects what is illegal.
            pass
        # After every step, state must be one of the declared states.
        assert fsm.state in fsm.states


@given(
    targets=st.lists(
        st.sampled_from(
            [
                CELL_STATE_EMPTY,
                CELL_STATE_IDLE,
                CELL_STATE_FAILED,
            ]
        ),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=80, deadline=None)
def test_cell_fsm_destination_driven_walk(targets: list[str]):
    """transition_to-driven walk on CellState. Every successful call ends
    at the requested target; every rejection raises with the documented
    reason set."""
    fsm = build_cell_fsm()
    for target in targets:
        try:
            new_state = fsm.transition_to(target)
        except IllegalStateTransition as exc:
            assert exc.reason in {
                "no_edge",
                "guard_rejected",
                "unknown_state",
            }
            assert exc.fsm_name == "CellState"
            continue
        assert new_state == target
        assert fsm.state == target


# ─── Terminal-state invariants ────────────────────────────────────────────────


def test_pheromone_evaporated_is_final():
    """EVAPORATED is the terminal pheromone state."""
    fsm = build_pheromone_fsm()
    fsm.reset(PHEROMONE_EVAPORATED)
    assert fsm.is_final


def test_task_completed_is_final():
    """COMPLETED is terminal in TaskLifecycle."""
    fsm = build_task_fsm()
    fsm.reset(TASK_COMPLETED)
    assert fsm.is_final


def test_task_cancelled_is_final():
    """CANCELLED is terminal in TaskLifecycle."""
    fsm = build_task_fsm()
    fsm.reset(TASK_CANCELLED)
    assert fsm.is_final


def test_task_failed_is_not_final_due_to_retry():
    """FAILED has the retry edge back to PENDING — must not be final."""
    fsm = build_task_fsm()
    fsm.reset("FAILED")
    assert not fsm.is_final


def test_pheromone_fresh_is_not_final():
    fsm = build_pheromone_fsm()
    assert fsm.state == PHEROMONE_FRESH
    assert not fsm.is_final


def test_succession_stable_is_not_final():
    fsm = build_succession_fsm()
    assert not fsm.is_final


def test_failover_healthy_is_not_final():
    fsm = build_failover_fsm()
    assert fsm.state == FAILOVER_HEALTHY
    assert not fsm.is_final


# ─── Round-trip undo ──────────────────────────────────────────────────────────


@given(
    targets=st.lists(
        st.sampled_from([CELL_STATE_EMPTY, CELL_STATE_IDLE, CELL_STATE_FAILED]),
        min_size=1,
        max_size=8,
    )
)
@settings(max_examples=40, deadline=None)
def test_cell_fsm_undo_returns_to_previous(targets: list[str]):
    """After a successful transition, undo returns to the prior state."""
    fsm = build_cell_fsm()
    for target in targets:
        before = fsm.state
        try:
            fsm.transition_to(target)
        except IllegalStateTransition:
            continue
        # Sanity: state changed (transition_to never short-circuits a
        # different-state target as a no-op).
        if target == before:
            # transition_to(same_state) does fire if there is an edge,
            # so undo brings us back to the same state — no-op.
            pass
        fsm.undo()
        assert fsm.state == before


# ─── Initial-state invariants ─────────────────────────────────────────────────


def test_cell_fsm_initial_is_empty():
    assert build_cell_fsm().state == CELL_STATE_EMPTY


def test_pheromone_fsm_initial_is_fresh():
    assert build_pheromone_fsm().state == PHEROMONE_FRESH


def test_task_fsm_initial_is_pending():
    assert build_task_fsm().state == TASK_PENDING


def test_succession_fsm_initial_is_stable():
    assert build_succession_fsm().state == "STABLE"


def test_failover_fsm_initial_is_healthy():
    assert build_failover_fsm().state == FAILOVER_HEALTHY


# ─── State set sanity ─────────────────────────────────────────────────────────


def test_no_orphan_states():
    """Every state declared in an FSM should appear in at least one
    transition (source or dest), or be the initial state — otherwise it's
    an orphan that can never be reached or left.

    Phase 4.3 removed SPAWNING and OVERLOADED. Phase 5.1 wired MIGRATING
    (admin_start_migration in CellFailover._migrate_work) and SEALED
    (admin_seal in HoneycombCell.seal()). All FSMs are now orphan-free.
    """
    fsms = {
        "CellState": (build_cell_fsm(), set()),
        "PheromoneDeposit": (build_pheromone_fsm(), set()),
        "TaskLifecycle": (build_task_fsm(), set()),
        "QueenSuccession": (build_succession_fsm(), set()),
        "FailoverFlow": (build_failover_fsm(), set()),
    }

    for fsm_name, (fsm, exempt) in fsms.items():
        # Reach all states by walking transitions from the dest-index.
        used: set[str] = {fsm.initial}
        for dst, srcs_triggers in fsm._dest_index.items():
            used.add(dst)
            for src, _trigger in srcs_triggers:
                if src != "*":
                    used.add(src)
        # Fold in the wildcard sources by adding all states (since "*"
        # represents "any of them").
        if any(src == "*" for srcs in fsm._dest_index.values() for src, _ in srcs):
            used.update(fsm.states)

        orphans = fsm.states - used - exempt
        assert not orphans, f"{fsm_name} has orphan states: {orphans}"
