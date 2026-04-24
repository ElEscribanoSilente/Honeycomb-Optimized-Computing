# ADR-003: Shared HMAC key across peers, not per-cell asymmetric keys

- **Status**: Accepted
- **Date**: 2026-04-23 (retroactive — decision made during Phase 2)
- **Phase**: Phase 2

## Context

Phase 2 introduced message authentication across `NectarFlow` /
`RoyalJelly` / `QueenSuccession`. The design choice was between:

1. **Shared symmetric key** — one HMAC key possessed by all peers in a
   cluster. Signing = computing an HMAC; verifying = comparing HMACs.
2. **Per-cell asymmetric keypair** — each cell has a private signing key
   and a published public verify key (Ed25519 or similar).

HOC's deployment model is: a cluster of peers that fully trust each other
at the topology level (they share the same executable, the same config,
and run inside the same administrative boundary — typically a single
cluster on a single tenant). External adversaries cannot join the cluster
without being added explicitly.

## Decision

Use a single **shared HMAC-SHA256 key** provisioned via the
`HOC_HMAC_KEY` environment variable. Every peer can sign, every peer can
verify. Role-based authorization (e.g. Queen-only for `priority >= 8`) is
enforced *in addition to* HMAC via the `issuer` parameter on
`RoyalJelly.issue_command`.

## Alternatives considered

### Per-cell Ed25519 keypairs with a published verify key

**Pros**: binds a signature to a specific cell — any peer can prove which
cell signed a message without needing the private key. Enables key
rotation per cell without a cluster-wide restart.

**Cons**: much more complex. Needs:

- A trust anchor for the verify keys (PKI) — either hard-coded roots or a
  distributed ledger. HOC doesn't have either.
- Key distribution at cluster bootstrap — who tells new cells the current
  verify keys? Adds a cold-start problem.
- Per-sign / per-verify cost ~100× HMAC. Phase 2 measured +3.5% overhead
  with HMAC; Ed25519 would push that to ~30–50% in hot paths like
  `WaggleDance.start_dance`.
- Does not solve the actual threat model — see below.

**Rejected** as out of proportion to the threat.

### No message authentication (keep v1.0.0)

Rejected by the Phase 2 threat model. An attacker on the network (or a
compromised peer) can forge arbitrary `RoyalCommand` messages and take
over the cluster.

### Hybrid — shared HMAC + per-cluster TLS on the wire

Considered for deployments where peers are on different hosts. Orthogonal
to this ADR — the message layer is still HMAC'd so that in-memory
manipulation (e.g. via a co-tenant on the same host) is also caught.
`HOC` does not ship a transport yet; when it does, TLS will supplement
HMAC, not replace it.

## Consequences

**Easier**:

- Zero-PKI bootstrap: set one env var, all cells speak the same language.
- HMAC is CPU-cheap (~1μs per sign/verify in CPython). Phase 2 total
  overhead measured at +3.5% end-to-end.
- The `mscs` library already accepts `hmac_key=` as a parameter — no
  extra glue.
- One key to rotate means rotation is a single coordinated action, not a
  per-cell ceremony.

**Harder / limitations**:

- A signed message proves *a peer with the key signed it*, not *which
  peer*. Any in-cluster attacker who obtains the key can impersonate any
  role.
- Role enforcement is *not* achieved by HMAC. Queen-only, drone-only, etc.
  need the `issuer` check in addition. Easy to forget — this risk is
  mitigated by centralizing enforcement in `RoyalJelly.issue_command`
  rather than having each subsystem re-implement the check.
- Key rotation requires all peers to adopt the new key near-simultaneously.
  A graceful key-rollover protocol (old + new key accepted for a window)
  is deferred.

**Risk / follow-up**:

- **Key leakage**: if the HMAC key leaks (e.g. via logs, debugger, memory
  dump), the entire cluster's authentication collapses. Mitigations:
  `HOC_HMAC_KEY` read once at import, no logging of the value, no API to
  print it, and production environments should source the key from a
  secret manager.
- **Bus factor on the enforcement check**: the Queen-only check lives in
  exactly one call site. A test
  (`TestRoyalCommandQueenOnly::test_drone_blocked`) pins it in place. Do
  not remove without updating that test.
- **Graduation to asymmetric keys**: if HOC ever operates across trust
  boundaries (different tenants on the same cluster), this ADR will be
  revisited. Not in scope for the 10-phase roadmap.

## References

- `security.py`: `get_hmac_key`, `sign_payload`, `verify_signature`.
- `RoyalJelly.issue_command` (`nectar.py`) — enforces
  `issuer == queen_coord` for `priority >= 8`.
- Phase 2 closure: `snapshot/PHASE_02_CLOSURE.md` §2.2 and §"Lecciones
  aprendidas" #3.
- Test coverage: `tests/test_security.py::TestRoyalCommandQueenOnly` (6
  tests), `TestHmacPrimitives` (5 tests).
