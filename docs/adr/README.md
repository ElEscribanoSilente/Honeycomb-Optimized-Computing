# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for HOC, following the
[format proposed by Michael Nygard](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).

## Why ADRs

Design decisions that took months of discussion should not evaporate into
Slack threads and PR comments. An ADR captures the *context*, *decision*,
and *consequences* of a non-trivial architectural choice so that a future
contributor (including future you) can reconstruct the reasoning without
excavating commit history.

## When to write one

Write an ADR when the team makes a decision that:

- Affects more than one module.
- Rules out alternatives that a reader would reasonably reach for.
- Has non-obvious tradeoffs or is likely to be revisited.
- Introduces or retires a dependency.
- Changes the shape of the public API.

You do NOT need an ADR for local refactors, small bug fixes, or decisions
fully explained by the code itself.

## Index

| ADR | Title | Status | Phase |
|-----|-------|--------|-------|
| [ADR-001](ADR-001-hexagonal-topology.md) | Hexagonal topology for cell grid | Accepted | v1.0.0 |
| [ADR-002](ADR-002-mscs-over-pickle.md) | `mscs` replaces `pickle` for all production serialization | Accepted | Phase 2 |
| [ADR-003](ADR-003-hmac-shared-key.md) | Shared HMAC key across peers (not per-cell keys) | Accepted | Phase 2 |
| [ADR-004](ADR-004-ordereddict-lru-pheromones.md) | `OrderedDict` LRU for `PheromoneTrail._deposits` | Accepted | Phase 2 |
| [ADR-005](ADR-005-raft-like-quorum.md) | Raft-like signed-vote quorum in `QueenSuccession` (not full Paxos) | Accepted | Phase 2 |
| [ADR-006](ADR-006-mypy-legacy-suppression.md) | Legacy modules suppressed from strict mypy in Phase 3 | Accepted | Phase 3 |

## Template

Use [ADR-000-template.md](ADR-000-template.md) when writing a new ADR.
