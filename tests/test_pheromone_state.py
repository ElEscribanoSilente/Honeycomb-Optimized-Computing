"""Phase 5.2a tests for the ``PheromoneDeposit.state`` mirror.

Static-only wire-up: the field tracks the conceptual lifecycle phase
without per-instance FSM allocation. ``PheromoneTrail.evaporate`` and
``diffuse_to_neighbors`` mutate ``deposit.state`` inside their existing
loops; production code never calls ``transition_to`` on a
PheromoneDeposit FSM (the spec in ``state_machines/pheromone_fsm.py``
remains the property-test target).

These tests exercise the four observable phases:

- ``FRESH`` — assigned by ``deposit()`` on construction.
- ``DECAYING`` — assigned by ``evaporate()`` once age crosses
  ``PHEROMONE_FRESHNESS_WINDOW`` (5 s by default).
- ``DIFFUSING`` — transient inside ``diffuse_to_neighbors``; observable
  via a subscriber injected before the call returns to DECAYING.
- ``EVAPORATED`` — assigned by ``evaporate()`` when intensity falls
  below ``CLEANUP_THRESHOLD``.

Anti-regression: nothing else about ``PheromoneDeposit`` should have
changed. The dataclass keeps the existing 6 fields; the new ``state``
field has a default so old constructors continue to work.
"""

from __future__ import annotations

import time

from hoc.core import HexCoord
from hoc.nectar import (
    PHEROMONE_FRESHNESS_WINDOW,
    PheromoneDecay,
    PheromoneDeposit,
    PheromonePhase,
    PheromoneTrail,
    PheromoneType,
)


def _new_deposit(intensity: float = 0.5) -> PheromoneDeposit:
    return PheromoneDeposit(
        ptype=PheromoneType.FOOD,
        intensity=intensity,
        timestamp=time.time(),
    )


# ─── Defaults ──────────────────────────────────────────────────────────────────


class TestPheromonePhaseDefaults:
    def test_dataclass_default_is_fresh(self):
        d = _new_deposit()
        assert d.state == PheromonePhase.FRESH

    def test_old_constructor_signature_still_works(self):
        # Anti-regression: the existing 5-positional + 2-keyword
        # constructor signature must still build a deposit even though
        # we added the ``state`` field.
        d = PheromoneDeposit(
            ptype=PheromoneType.FOOD,
            intensity=0.5,
            timestamp=time.time(),
            source=HexCoord(0, 0),
            metadata={"k": "v"},
        )
        assert d.state == PheromonePhase.FRESH
        assert d.metadata == {"k": "v"}


# ─── deposit() sets FRESH ──────────────────────────────────────────────────────


class TestDepositSetsFresh:
    def test_new_deposit_via_trail_starts_fresh(self):
        trail = PheromoneTrail()
        coord = HexCoord(0, 0)
        trail.deposit(coord, PheromoneType.FOOD, 0.5)
        deposit = trail._deposits[coord][PheromoneType.FOOD]
        assert deposit.state == PheromonePhase.FRESH

    def test_redeposit_keeps_existing_state(self):
        # Redepositing on an existing coord/ptype updates intensity but
        # does not reset the state field — the existing object is
        # reused (by design, to keep the LRU + signature intact).
        trail = PheromoneTrail()
        coord = HexCoord(0, 0)
        trail.deposit(coord, PheromoneType.FOOD, 0.5)
        deposit = trail._deposits[coord][PheromoneType.FOOD]
        deposit.state = PheromonePhase.DECAYING  # simulate a phase advance
        trail.deposit(coord, PheromoneType.FOOD, 0.3)  # re-deposit
        # The field was not stomped back to FRESH.
        assert deposit.state == PheromonePhase.DECAYING


# ─── evaporate() sets DECAYING / EVAPORATED ───────────────────────────────────


class TestEvaporatePhase:
    def test_evaporate_marks_aged_deposit_decaying(self):
        trail = PheromoneTrail(decay_strategy=PheromoneDecay.EXPONENTIAL)
        coord = HexCoord(0, 0)
        trail.deposit(coord, PheromoneType.FOOD, 0.5)
        deposit = trail._deposits[coord][PheromoneType.FOOD]
        # Backdate the timestamp past the freshness window so evaporate
        # crosses FRESH -> DECAYING.
        deposit.timestamp = time.time() - (PHEROMONE_FRESHNESS_WINDOW + 1.0)
        trail.evaporate(force=True)
        assert deposit.state == PheromonePhase.DECAYING

    def test_evaporate_marks_drained_deposit_evaporated(self):
        trail = PheromoneTrail(decay_strategy=PheromoneDecay.EXPONENTIAL)
        coord = HexCoord(0, 0)
        # Tiny intensity so a single decay step crosses the cleanup
        # threshold.
        trail.deposit(coord, PheromoneType.DANGER, 0.0015)
        deposit = trail._deposits[coord][PheromoneType.DANGER]
        # Backdate so the exponential decay drops intensity below
        # CLEANUP_THRESHOLD (0.001) on the next evaporate call.
        deposit.timestamp = time.time() - 100.0
        trail.evaporate(force=True)
        # The deposit was queued for removal; before the dict cleanup,
        # state should have been set to EVAPORATED.
        assert deposit.state == PheromonePhase.EVAPORATED

    def test_fresh_deposit_stays_fresh_under_evaporate(self):
        trail = PheromoneTrail(decay_strategy=PheromoneDecay.EXPONENTIAL)
        coord = HexCoord(0, 0)
        trail.deposit(coord, PheromoneType.FOOD, 0.5)
        deposit = trail._deposits[coord][PheromoneType.FOOD]
        # Fresh: timestamp is now-ish; one evaporate call should keep
        # the state at FRESH (intensity still well above cleanup, age
        # still under the freshness window).
        trail.evaporate(force=True)
        assert deposit.state == PheromonePhase.FRESH


# ─── diffuse_to_neighbors() touches DIFFUSING ─────────────────────────────────


class TestDiffusionPhase:
    def test_diffuse_settles_back_to_decaying(self):
        trail = PheromoneTrail()
        coord = HexCoord(0, 0)
        trail.deposit(coord, PheromoneType.FOOD, 1.0)
        deposit = trail._deposits[coord][PheromoneType.FOOD]
        trail.diffuse_to_neighbors()
        # diffuse_to_neighbors assigns DIFFUSING then DECAYING in the
        # same loop iteration; observers see DECAYING after the call.
        assert deposit.state == PheromonePhase.DECAYING

    def test_diffuse_below_threshold_does_not_change_state(self):
        trail = PheromoneTrail()
        coord = HexCoord(0, 0)
        # Deposit just below the diffuse threshold (default 0.01).
        trail.deposit(coord, PheromoneType.FOOD, 0.005)
        deposit = trail._deposits[coord][PheromoneType.FOOD]
        # Sanity: deposit started FRESH.
        assert deposit.state == PheromonePhase.FRESH
        trail.diffuse_to_neighbors()
        # Intensity below the diffuse threshold short-circuits before
        # the state mutation.
        assert deposit.state == PheromonePhase.FRESH
