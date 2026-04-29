"""Phase 5.2b tests for the ``QueenSuccession._succession_state`` wire-up.

Static-only mirror: ``QueenSuccession`` keeps a ``_SuccessionState``
wrapper whose ``state`` attribute is mutated inline at each lifecycle
step in :meth:`elect_new_queen` and :meth:`_conduct_election`. The
quorum / signing / term-counter logic from Phase 2 is **not** changed
— the 7 ``TestQuorumSignedVotes`` tests are the canary for any
regression there.

These tests cover phase observability without depending on the random
outcome of an actual vote: ``_identify_candidates`` and
``_conduct_election`` are monkeypatched to deterministic returns so
the test can assert exactly which phases the wire-up walks.
"""

from __future__ import annotations

import pytest

from hoc.core import HoneycombConfig, HoneycombGrid, QueenCell
from hoc.resilience import (
    QueenSuccession,
    ResilienceConfig,
    SuccessionPhase,
)


@pytest.fixture
def grid_r2() -> HoneycombGrid:
    return HoneycombGrid(HoneycombConfig(radius=2))


@pytest.fixture
def default_config() -> ResilienceConfig:
    return ResilienceConfig()


@pytest.fixture
def lenient_config() -> ResilienceConfig:
    """``min_queen_candidates=1`` so a single mocked candidate is enough
    to take the NOMINATING / VOTING / ELECTED path."""
    return ResilienceConfig(min_queen_candidates=1)


# ─── Defaults ──────────────────────────────────────────────────────────────────


class TestSuccessionPhaseDefault:
    def test_default_phase_is_stable(self, grid_r2, default_config):
        succ = QueenSuccession(grid_r2, default_config)
        assert succ.phase == SuccessionPhase.STABLE

    def test_default_phase_history_is_empty(self, grid_r2, default_config):
        succ = QueenSuccession(grid_r2, default_config)
        assert succ.phase_history == []


# ─── Successful election walks the lifecycle ───────────────────────────────────


class TestSuccessfulElectionPhases:
    def test_full_lifecycle_progression(self, grid_r2, lenient_config, monkeypatch):
        """Mock the candidate set + tally so a deterministic winner
        produces the full STABLE→DETECTING→NOMINATING→VOTING→ELECTED→
        STABLE walk."""
        succ = QueenSuccession(grid_r2, lenient_config)
        winner_coord = next(iter(grid_r2._cells.keys()))

        # Force candidate identification + tally to a known winner.
        monkeypatch.setattr(succ, "_identify_candidates", lambda: [winner_coord])
        monkeypatch.setattr(succ, "_conduct_election", lambda _: winner_coord)
        monkeypatch.setattr(
            succ,
            "_promote_to_queen",
            lambda _: QueenCell(winner_coord, grid_r2.config),
        )

        new_queen = succ.elect_new_queen()
        assert new_queen is not None
        history = succ.phase_history
        # The success path enters DETECTING then NOMINATING (after
        # candidates identified). VOTING + ELECTED happen inside the
        # mocked ``_conduct_election`` -> not appended here. The path
        # finishes at STABLE.
        assert SuccessionPhase.DETECTING in history
        assert SuccessionPhase.NOMINATING in history
        assert SuccessionPhase.STABLE in history
        # Final phase is STABLE so the instance is reusable.
        assert succ.phase == SuccessionPhase.STABLE

    def test_history_order_matches_lifecycle(self, grid_r2, lenient_config, monkeypatch):
        succ = QueenSuccession(grid_r2, lenient_config)
        winner_coord = next(iter(grid_r2._cells.keys()))
        monkeypatch.setattr(succ, "_identify_candidates", lambda: [winner_coord])
        monkeypatch.setattr(succ, "_conduct_election", lambda _: winner_coord)
        monkeypatch.setattr(
            succ,
            "_promote_to_queen",
            lambda _: QueenCell(winner_coord, grid_r2.config),
        )
        succ.elect_new_queen()
        history = succ.phase_history
        idx_detecting = history.index(SuccessionPhase.DETECTING)
        idx_nominating = history.index(SuccessionPhase.NOMINATING)
        idx_stable = history.index(SuccessionPhase.STABLE)
        assert idx_detecting < idx_nominating < idx_stable


# ─── Failure paths ─────────────────────────────────────────────────────────────


class TestFailedElectionPhases:
    def test_too_few_candidates_lands_at_failed(self, grid_r2):
        # min_queen_candidates higher than the grid can supply forces
        # the early-exit "Not enough candidates" path.
        config = ResilienceConfig(min_queen_candidates=999)
        succ = QueenSuccession(grid_r2, config)
        result = succ.elect_new_queen()
        assert result is None
        assert SuccessionPhase.DETECTING in succ.phase_history
        assert SuccessionPhase.FAILED in succ.phase_history
        assert succ.phase == SuccessionPhase.FAILED

    def test_no_winner_lands_at_failed(self, grid_r2, lenient_config, monkeypatch):
        succ = QueenSuccession(grid_r2, lenient_config)
        winner_coord = next(iter(grid_r2._cells.keys()))
        monkeypatch.setattr(succ, "_identify_candidates", lambda: [winner_coord])
        # Tally rejects: no winner.
        monkeypatch.setattr(succ, "_conduct_election", lambda _: None)
        result = succ.elect_new_queen()
        assert result is None
        # Path: DETECTING -> NOMINATING -> FAILED (the explicit FAILED
        # in elect_new_queen after _conduct_election returned None).
        history = succ.phase_history
        assert SuccessionPhase.NOMINATING in history
        assert SuccessionPhase.FAILED in history

    def test_promote_failure_lands_at_failed(self, grid_r2, lenient_config, monkeypatch):
        """Even when the tally produced a winner, a failed
        ``_promote_to_queen`` flips the final phase to FAILED."""
        succ = QueenSuccession(grid_r2, lenient_config)
        winner_coord = next(iter(grid_r2._cells.keys()))
        monkeypatch.setattr(succ, "_identify_candidates", lambda: [winner_coord])
        monkeypatch.setattr(succ, "_conduct_election", lambda _: winner_coord)
        # Promote returns None -> the elect_new_queen failure branch.
        monkeypatch.setattr(succ, "_promote_to_queen", lambda _: None)
        result = succ.elect_new_queen()
        assert result is None
        assert succ.phase == SuccessionPhase.FAILED


# ─── _set_phase exercises every member ─────────────────────────────────────────


class TestSetPhaseExhaustive:
    @pytest.mark.parametrize("phase", list(SuccessionPhase))
    def test_set_phase_accepts_every_member(self, grid_r2, default_config, phase):
        succ = QueenSuccession(grid_r2, default_config)
        succ._set_phase(phase)
        assert succ.phase == phase
        assert succ.phase_history[-1] == phase


# ─── _conduct_election walks VOTING + ELECTED|FAILED ──────────────────────────


class TestConductElectionPhases:
    def test_conduct_election_with_winner_walks_voting_to_elected(
        self, grid_r2, default_config, monkeypatch
    ):
        succ = QueenSuccession(grid_r2, default_config)
        winner = next(iter(grid_r2._cells.keys()))
        monkeypatch.setattr(succ, "_tally_votes", lambda *_args, **_kw: winner)
        result = succ._conduct_election([winner])
        assert result == winner
        history = succ.phase_history
        assert SuccessionPhase.VOTING in history
        assert SuccessionPhase.ELECTED in history

    def test_conduct_election_without_winner_walks_voting_to_failed(
        self, grid_r2, default_config, monkeypatch
    ):
        succ = QueenSuccession(grid_r2, default_config)
        winner = next(iter(grid_r2._cells.keys()))
        monkeypatch.setattr(succ, "_tally_votes", lambda *_args, **_kw: None)
        result = succ._conduct_election([winner])
        assert result is None
        history = succ.phase_history
        assert SuccessionPhase.VOTING in history
        assert SuccessionPhase.FAILED in history
