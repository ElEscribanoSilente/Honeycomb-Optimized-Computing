# ADR-005: Raft-like signed-vote quorum for `QueenSuccession`, not full Paxos

- **Status**: Accepted
- **Date**: 2026-04-23 (retroactive — decision made during Phase 2)
- **Phase**: Phase 2 (refines Phase 1 fix B4)

## Context

`QueenSuccession` elects a new queen cell when the current queen fails.
Phase 1 fix **B4** made the election quorum-binding (majority required;
`None` returned if no majority reached) but did not authenticate votes.
Phase 2's threat model includes a compromised or colluding peer that can
forge vote messages to swing an election — in particular:

- **Stuffing**: injecting extra votes for a preferred candidate.
- **Replay**: submitting votes from a previous election term.
- **Forgery**: signing a vote as a different peer.
- **Unknown candidate**: voting for a peer not in the official candidate
  set (undetected → arbitrary peer becomes queen).

## Decision

Refactor `_conduct_election` into a **Raft-like protocol** with these
properties:

1. Monotonically increasing `current_term` — incremented at the start of
   every election. Stored in `QueenSuccession` instance state.
2. Votes are a `Vote` dataclass with fields
   `(voter, candidate, term, timestamp, signature)`. `signature` is
   HMAC-SHA256 over the canonical serialization of the other fields.
3. `_tally_votes(votes, candidates, expected_term)` rejects votes with:
   - Missing or invalid HMAC signature (`rejected["bad_signature"]`).
   - Term ≠ `expected_term` (`rejected["wrong_term"]`, anti-replay).
   - Candidate not in the official set (`rejected["unknown_candidate"]`).
   - Duplicate voter — only first vote counts
     (`rejected["duplicate_voter"]`).
4. Quorum is strictly > 50% of the electorate (Phase 1 B4 fix reinforced
   by this authentication layer).

## Alternatives considered

### Full Raft / Paxos

Both provide stronger guarantees (safety under asynchrony, liveness under
partial synchrony) but require:

- A replicated log.
- A protocol for log replication (AppendEntries in Raft).
- Heartbeat machinery with leader leases.
- Clarity on which RPC layer carries the protocol messages.

HOC's election is a *one-shot* decision triggered by queen failure
detection. We don't need a replicated log (cluster config is static or
managed out-of-band), we already have a transport (`NectarFlow`), and
HoC is not trying to be a consensus library. Full Raft would be ~1500
LOC of new code vs the current ~200. Rejected as disproportionate.

### Byzantine consensus (PBFT, HotStuff)

Addresses peers that actively lie. HOC's threat model is "compromised
peer with shared HMAC key" (see [ADR-003](ADR-003-hmac-shared-key.md)),
not arbitrary Byzantine. HMAC covers most impersonation; Byzantine
protocols would add ~2000 LOC for a marginal real-world improvement.
Rejected.

### Vector clocks for term ordering

Would detect concurrent elections but doesn't replace the anti-replay
check — and vector clocks are per-peer, adding per-peer state that the
shared-HMAC model does not need. Rejected.

### Keep B4 fix and accept forge risk

The B4 fix ensures numerical majority but not authenticity of votes. In
Phase 2's threat model (in-cluster compromised peer can see everything
on the message bus), a compromised peer could trivially stuff the
tally. Rejected as insufficient.

## Consequences

**Easier**:

- Anti-replay: an old election's votes can't swing a new one. Term
  incrementing makes this automatic.
- Forge detection: unsigned or badly-signed votes are counted but
  rejected. Tests pin the rejection counters
  (`TestQuorumSignedVotes::test_*`).
- Auditability: `_tally_votes` returns a dict of rejection reasons,
  making a post-mortem of a disputed election trivially observable.
- The *Raft-like* (not Raft) labeling keeps the implementation honest —
  we borrow the term concept and majority quorum, not the full protocol.

**Harder**:

- Vote signing adds HMAC cost per vote. Not measured separately but
  included in the +3.5% Phase 2 end-to-end overhead.
- Contributors unfamiliar with distributed consensus need to learn the
  term / majority concept. Mitigated by ADRs and the docstring on
  `QueenSuccession`.

**Risk / follow-up**:

- **Liveness under partial quorum**: if the cluster is split and neither
  side has > 50%, no queen is elected. This matches B4's intent
  (correctness over liveness) but means HOC is *not* available during
  severe partitions. Acceptable for v1.x.
- **Graduation to full Raft**: if HOC grows a replicated-log need (e.g.
  to sync task state across a partition), this ADR's implementation is
  a stepping stone toward Raft but not Raft itself. Revisit then.

## References

- `resilience.QueenSuccession._conduct_election`,
  `resilience.QueenSuccession._tally_votes`,
  `resilience.Vote`.
- Phase 2 closure: `snapshot/PHASE_02_CLOSURE.md` §2.3.
- Phase 1 B4 fix context: `snapshot/PHASE_01_CLOSURE.md` "Bugs
  corregidos" B4.
- Test coverage: `tests/test_security.py::TestQuorumSignedVotes` (7
  tests: duplicate voter, unsigned, tampered signature, wrong term,
  unknown candidate, majority, monotonic term).
