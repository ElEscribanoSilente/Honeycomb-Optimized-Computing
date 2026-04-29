"""
QueenSuccession FSM (Phase 4.4b + Phase 5.2b)
=============================================

Conceptual phases of a queen-succession election in
:class:`hoc.resilience.QueenSuccession`:

::

    STABLE ─heartbeat_lost─► DETECTING ─confirmed─► NOMINATING
                                  ▲                     │
                                  │                     ▼
                                  └─stabilized──   VOTING
                                          │            │
                                          │   ┌────────┴────────┐
                                          │   │                 │
                                          │   ▼                 ▼
                                       FAILED            ELECTED
                                          │   ◄─cooldown─       │
                                          └─────────────────────┘
                                                                │
                                                                ▼
                                                            STABLE  (after promote)

Phase 5.2b wire-up — *static-only*
----------------------------------

Phase 5.2b added a ``_succession_state: _SuccessionState`` wrapper to
:class:`hoc.resilience.QueenSuccession` and inline ``_set_phase``
mutations in :meth:`elect_new_queen` and :meth:`_conduct_election`.
The phases progress STABLE → DETECTING → NOMINATING → VOTING →
ELECTED|FAILED → STABLE during an election.

The wire-up is **static-only**: no per-instance ``HocStateMachine`` is
allocated and no runtime guard validation happens. The reason is the
security-critical nature of ``_tally_votes`` (Phase 2 / B4 hardening) —
splitting the existing call into multiple FSM-driven methods risks
regressing the 7 ``TestQuorumSignedVotes`` tests. Static mirroring
gives us:

- **Observability**: ``QueenSuccession.phase`` returns the current
  ``SuccessionPhase``; ``phase_history`` returns the ordered list of
  phases this instance has occupied (replayable lifecycle).
- **Static checking**: ``choreo`` walks the explicit
  ``_succession_state.state = SuccessionPhase.X`` assignments and
  treats the FSM as wired (no longer ``declarative_only``).
- **Anti-regression**: the original tally / signing / term logic is
  untouched; the 7 quorum tests pass without modification.

The spec FSM here remains the source of truth for the lifecycle graph
and the property-test target. If future work decides explicit FSM
transitions are worth the security re-validation cost, the spec is
already aligned with what the wire-up observes.

Election security guards (mapped from existing tally code)
----------------------------------------------------------

The brief specified several guards on the ``VOTING → ELECTED`` edge.
These all map to the existing ``_tally_votes`` rejections (resilience.py
~590-672), which Phase 1 (B4) and Phase 2 (signed votes + term)
hardened. The FSM's guard re-states them so the diagram is honest:

- ``quorum_reached``: a strict majority of *valid* votes name a single
  candidate (``tallies[winner] >= total // 2 + 1``).
- ``signatures_valid``: every counted vote passed
  :meth:`hoc.resilience.Vote.verify` (HMAC-SHA256 over canonical bytes).
- ``term_matches``: ``vote.term == expected_term`` for every counted
  vote — anti-replay across elections.
"""

from __future__ import annotations

from .base import HocStateMachine, HocTransition

SUCCESSION_STABLE = "STABLE"
SUCCESSION_DETECTING = "DETECTING"
SUCCESSION_NOMINATING = "NOMINATING"
SUCCESSION_VOTING = "VOTING"
SUCCESSION_ELECTED = "ELECTED"
SUCCESSION_FAILED = "FAILED"

ALL_SUCCESSION_STATES: tuple[str, ...] = (
    SUCCESSION_STABLE,
    SUCCESSION_DETECTING,
    SUCCESSION_NOMINATING,
    SUCCESSION_VOTING,
    SUCCESSION_ELECTED,
    SUCCESSION_FAILED,
)


def build_succession_fsm() -> HocStateMachine:
    """Build the QueenSuccession FSM. Declarative-only — see module docstring."""
    transitions: list[HocTransition] = [
        # Heartbeat timeout opens the detection window. The HiveResilience
        # heartbeat loop calls check_queen_health() and sees `elapsed >
        # queen_heartbeat_interval * 2`. resilience.py ~485-497.
        HocTransition(
            SUCCESSION_STABLE,
            SUCCESSION_DETECTING,
            trigger="heartbeat_lost",
            guard=lambda ctx: bool(
                ctx.get("elapsed_since_heartbeat", 0.0) > ctx.get("timeout_threshold", 0.0)
            ),
        ),
        # Confirmation: missed heartbeats persisted past the confirm
        # window — start nominating. Currently the code commits to the
        # election after a single timeout; the FSM allows for the
        # multi-tick confirm window the brief describes.
        HocTransition(
            SUCCESSION_DETECTING,
            SUCCESSION_NOMINATING,
            trigger="failure_confirmed",
            guard=lambda ctx: bool(ctx.get("missed_heartbeats", 0) >= ctx.get("confirm_ticks", 1)),
        ),
        # False alarm: queen recovered before confirm window.
        HocTransition(SUCCESSION_DETECTING, SUCCESSION_STABLE, trigger="false_alarm"),
        # _identify_candidates returned a non-empty set — proceed to vote.
        HocTransition(
            SUCCESSION_NOMINATING,
            SUCCESSION_VOTING,
            trigger="candidates_identified",
            guard=lambda ctx: bool(ctx.get("candidate_count", 0) > 0),
        ),
        # No candidates — election aborts; HiveResilience may retry after
        # cooldown.
        HocTransition(SUCCESSION_NOMINATING, SUCCESSION_FAILED, trigger="no_candidates"),
        # Successful tally: quorum + valid signatures + term match.
        # _tally_votes returns a winner (resilience.py:672).
        HocTransition(
            SUCCESSION_VOTING,
            SUCCESSION_ELECTED,
            trigger="quorum_reached",
            guard=lambda ctx: bool(
                ctx.get("quorum_reached", False)
                and ctx.get("signatures_valid", False)
                and ctx.get("term_matches", False)
            ),
        ),
        # Tally rejected: insufficient votes, bad signatures, term
        # mismatch (resilience.py:651-671).
        HocTransition(SUCCESSION_VOTING, SUCCESSION_FAILED, trigger="tally_rejected"),
        # Promote completes: new queen takes over. resilience.py:697-727.
        HocTransition(SUCCESSION_ELECTED, SUCCESSION_STABLE, trigger="queen_promoted"),
        # Cooldown: failed election can be retried after a wait period.
        HocTransition(SUCCESSION_FAILED, SUCCESSION_STABLE, trigger="cooldown_expired"),
    ]

    return HocStateMachine(
        name="QueenSuccession",
        states=list(ALL_SUCCESSION_STATES),
        transitions=transitions,
        initial=SUCCESSION_STABLE,
        history_size=16,
        # Phase 5.2b: explicit binding to the host enum so choreo skips
        # the member-subset heuristic. The enum lives in ``resilience.py``
        # next to ``QueenSuccession``.
        enum_name="SuccessionPhase",
    )
