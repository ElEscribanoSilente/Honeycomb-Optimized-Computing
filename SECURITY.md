# Security Policy

HOC (Honeycomb Optimized Computing) takes security seriously. This document
describes how to report vulnerabilities and what the project considers
in-scope.

## Supported versions

HOC is currently in stabilization (Phase 3 of the 10-phase roadmap). Only
the latest tagged release receives security fixes:

| Version             | Status           | Security fixes |
|---------------------|------------------|----------------|
| `v1.3.0-phase03`    | active (Phase 3) | yes            |
| `v1.2.0-phase02`    | previous         | critical only  |
| `v1.1.0-phase01`    | older            | no             |
| `v1.0.0-baseline`   | historical       | no             |

Once HOC reaches `v2.0.0` (roadmap Phase 10), a longer-term support policy
will be documented here.

## Reporting a vulnerability

**Please do not file a public GitHub issue for security vulnerabilities.**

Use one of these private channels instead:

1. **GitHub Security Advisories** (preferred): open a private advisory at
   <https://github.com/esraderey/Honeycomb-Optimized-Computing/security/advisories/new>.
   This keeps the report private until a fix is released.

2. **Email**: `csrrtr@gmail.com` with subject prefix `[HOC-SEC]`. Please
   include reproduction steps, affected versions, and your expected impact
   assessment. PGP is not currently required.

We aim to acknowledge reports within **72 hours** and to provide a first
status update within **7 days**. Critical issues (RCE, signature bypass,
cryptographic weakness) are prioritized over lower-severity findings.

## Disclosure timeline

- **Day 0**: report received, acknowledgement sent.
- **Day 1–14**: triage, reproduction, severity assessment, fix development.
- **Day 14–30**: fix released in a new tag, reporter credited in the release
  notes unless anonymity was requested.
- **Day 30+**: public advisory published with CVE (if applicable).

Extension of the timeline by mutual agreement is normal for complex issues.
We request coordinated disclosure — please do not publish details before
the fix ships.

## Scope

In scope:

- Any `hoc.*` module in the `main` branch (code path reachable from public
  APIs listed in `__init__.py`).
- Cryptographic primitives in `security.py` (HMAC-SHA256 construction, path
  validation, rate limiting, sanitize\_error).
- Serialization boundary (`mscs` integration — Phase 2). Payload forgery,
  registry bypass, strict-mode escape attempts.
- Authentication / authorization logic in `RoyalJelly.issue_command` and
  `QueenSuccession._tally_votes` (Queen-only enforcement, quorum forgery,
  term-replay).
- Resource exhaustion in `PheromoneTrail` (bounded growth caps, LRU
  eviction).

Out of scope:

- Issues requiring physical / local access to a peer's machine.
- Issues in third-party dependencies that are already tracked by their
  respective projects (file them upstream; we will update the dependency
  once a fix is released).
- Denial-of-service from legitimate heavy workloads (HOC is a distributed
  compute framework — expect resource consumption).
- Issues requiring admin/root on a peer where the attacker could just modify
  the binary.
- Anything predicated on running a development / debug build
  (`HOC_DEBUG=1`) in production.

## Threat model (Phase 2 baseline)

HOC assumes:

1. **Shared HMAC key** across all nodes in a cluster (`HOC_HMAC_KEY` env
   var). All peers that possess the key can sign messages. Therefore:
   - Authenticity of a signed message proves *a peer with the key signed it*,
     not *who* signed it.
   - Role-based enforcement (e.g. Queen-only for priority ≥ 8) is enforced
     *in addition to* HMAC via the `issuer` parameter.
2. **Serialized blobs** stored in `CombStorage` / `HoneyArchive` are
   HMAC-signed; tampering between `put` and `get` is detected.
3. **Paths** passed to `HoneyArchive` keys are untrusted; `security.safe_join`
   and `_validate_key` reject traversal (`../`), absolute paths, and null
   bytes.
4. **CSPRNG** (`secrets.SystemRandom`) is used for any decision an attacker
   could manipulate by predicting `random.random()` seeds (eviction policy,
   role shuffle, work-acceptance threshold).

If an attacker obtains the HMAC key, they can impersonate any peer. Key
rotation is manual in Phase 2 and will be addressed in a later phase.

## Past advisories

| Advisory | Phase | Summary |
|----------|-------|---------|
| B1–B8    | Phase 1 | Latent bugs in RWLock, scheduler TOCTOU, election quorum, PollenCache eviction, etc. See `snapshot/PHASE_01_CLOSURE.md`. |
| pickle → mscs | Phase 2 | Replaced `pickle` with `mscs` + HMAC across 5 call sites. See `snapshot/PHASE_02_CLOSURE.md`. |
| QueenSuccession Raft-like | Phase 2 | Signed votes with monotonic term, rejection of duplicate / bad-term / unsigned votes. |
| B11      | Phase 3 | `resilience._rebuild_cell` wrote to a non-existent attribute, leaving pheromone state intact after rebuild. Not a remote vulnerability (reachable only from trusted recovery path) but silently incorrect. See `snapshot/PHASE_03_CLOSURE.md`. |

## Credits

We credit reporters in the release notes for the fix and optionally in this
document. Let us know in your report whether you would like to be credited
and how.
