"""
PheromoneDeposit FSM (Phase 4.5 + Phase 5.2a)
=============================================

Conceptual lifecycle of a :class:`hoc.nectar.PheromoneDeposit`:

::

    FRESH ──┐
            │
            ▼
        DECAYING ◀──┐
            │       │
        ┌───┴──┐    │
        ▼      ▼    │
    DIFFUSING ─┘    │
        │           │
        └────► EVAPORATED  (terminal)

Phase 5.2a wire-up — *static-only*
----------------------------------

Phase 5.2a added a ``state: PheromonePhase`` field on
:class:`hoc.nectar.PheromoneDeposit` and the corresponding mutations in
``PheromoneTrail.evaporate`` / ``diffuse_to_neighbors``. This is a
**static-only** wire-up: the field mirrors the phase the deposit is
conceptually in, but no per-instance ``HocStateMachine`` is allocated
and no runtime guard validation happens. The reason is the perf budget
(< 3 % overhead on ``test_nectar_flow_tick``) and the deposit
population (~90k objects per trail at default caps). A per-instance FSM
or a global validator with a lock would exceed the budget by orders
of magnitude.

The mirror gives us the things we wanted from a runtime wire-up
without the cost:

- **Observability**: operators can read ``deposit.state`` and tell
  FRESH apart from DECAYING / DIFFUSING / EVAPORATED.
- **Static checking**: ``choreo`` walks the explicit
  ``deposit.state = PheromonePhase.X`` assignments in ``nectar.py``
  and treats the FSM as wired (no longer ``declarative_only``).
- **Property tests**: still target the FSM graph via
  ``state_machines/`` so the spec stays the source of truth for
  reachability invariants.

If a future phase decides explicit FSM transitions are worth the cost,
the spec is already here; wire-up becomes a per-call swap of the
attribute mutation for ``transition_to(phase.value)``.

Lifecycle phases (definitions)
------------------------------

- **FRESH**: deposit was created (or refreshed) within
  ``DEFAULT_FRESHNESS_WINDOW`` seconds. Intensity is at or near maximum.
- **DECAYING**: age exceeds the freshness window; intensity is dropping
  per the configured ``PheromoneDecay`` strategy.
- **DIFFUSING**: a special transient phase entered when
  ``HoneycombCell.diffuse_pheromones`` spreads a fraction of the deposit
  to neighbours. The original deposit is still in DECAYING after the
  spread completes — DIFFUSING is observed only during the operation.
- **EVAPORATED**: terminal. Intensity dropped below
  :data:`hoc.nectar.PheromoneTrail.CLEANUP_THRESHOLD` and the deposit is
  scheduled for removal from the trail's ``OrderedDict``.
"""

from __future__ import annotations

from .base import HocStateMachine, HocTransition

PHEROMONE_FRESH = "FRESH"
PHEROMONE_DECAYING = "DECAYING"
PHEROMONE_DIFFUSING = "DIFFUSING"
PHEROMONE_EVAPORATED = "EVAPORATED"

ALL_PHEROMONE_STATES: tuple[str, ...] = (
    PHEROMONE_FRESH,
    PHEROMONE_DECAYING,
    PHEROMONE_DIFFUSING,
    PHEROMONE_EVAPORATED,
)

# Freshness window in seconds. Matches the conceptual "fresh deposit"
# definition the brief uses; not enforced by current code (no state
# field on PheromoneDeposit), but documented here as the boundary
# between FRESH and DECAYING for property tests.
DEFAULT_FRESHNESS_WINDOW: float = 5.0


def build_pheromone_fsm() -> HocStateMachine:
    """
    Build the conceptual PheromoneDeposit lifecycle FSM. Used by
    :mod:`scripts.generate_state_machines_md` and the property tests.
    Production code does not call ``transition_to`` on this FSM — see the
    module docstring for why.
    """
    transitions: list[HocTransition] = [
        # FRESH age out -> DECAYING. The only legal exit from FRESH; the
        # phase boundary is `age > DEFAULT_FRESHNESS_WINDOW`.
        HocTransition(
            PHEROMONE_FRESH,
            PHEROMONE_DECAYING,
            trigger="aged_out",
            guard=lambda ctx: bool(
                ctx.get("age", 0.0) > ctx.get("freshness_window", DEFAULT_FRESHNESS_WINDOW)
            ),
        ),
        # DECAYING -> DIFFUSING when the cell spreads to neighbours
        # (HoneycombCell.diffuse_pheromones). Two guards: intensity above
        # the diffuse threshold, and at least one reachable neighbour.
        HocTransition(
            PHEROMONE_DECAYING,
            PHEROMONE_DIFFUSING,
            trigger="diffusion_started",
            guard=lambda ctx: bool(
                ctx.get("intensity", 0.0) > ctx.get("diffuse_threshold", 0.01)
                and ctx.get("neighbors_reachable", 0) > 0
            ),
        ),
        # DIFFUSING -> DECAYING after the diffusion call returns. Intensity
        # has been split with neighbours; the original goes back to plain
        # decay.
        HocTransition(
            PHEROMONE_DIFFUSING,
            PHEROMONE_DECAYING,
            trigger="diffusion_completed",
        ),
        # DECAYING -> EVAPORATED when intensity falls below the cleanup
        # threshold. Terminal state.
        HocTransition(
            PHEROMONE_DECAYING,
            PHEROMONE_EVAPORATED,
            trigger="intensity_below_cleanup",
            guard=lambda ctx: bool(ctx.get("intensity", 1.0) < ctx.get("cleanup_threshold", 0.001)),
        ),
        # DIFFUSING -> EVAPORATED if the diffusion itself drains the
        # deposit below the cleanup threshold (a rare but allowed shortcut).
        HocTransition(
            PHEROMONE_DIFFUSING,
            PHEROMONE_EVAPORATED,
            trigger="intensity_below_cleanup_during_diffuse",
            guard=lambda ctx: bool(ctx.get("intensity", 1.0) < ctx.get("cleanup_threshold", 0.001)),
        ),
    ]

    return HocStateMachine(
        name="PheromoneDeposit",
        states=list(ALL_PHEROMONE_STATES),
        transitions=transitions,
        initial=PHEROMONE_FRESH,
        # Phase 5.2a: no per-instance history — the wire-up is
        # static-only (a ``state`` attribute mirror on the deposit
        # dataclass, no runtime FSM allocation).
        history_size=0,
        # Phase 5.2a: explicit binding to the host enum so choreo skips
        # the member-subset heuristic. The enum lives in ``nectar.py``
        # next to the ``PheromoneDeposit`` dataclass that owns the
        # ``state`` field.
        enum_name="PheromonePhase",
    )
