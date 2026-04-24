# ADR-002: `mscs` replaces `pickle` for all production serialization

- **Status**: Accepted
- **Date**: 2026-04-23 (retroactive — decision made during Phase 2)
- **Phase**: Phase 2

## Context

v1.0 used Python's built-in `pickle` for three persistence paths:

1. `memory.PollenCache.put` — L1 cache size estimation.
2. `memory.CombStorage.put/get` — L2 distributed cell storage.
3. `memory.HoneyArchive.archive/retrieve` — L3 compressed long-term store.

`pickle` has a well-known security property: deserializing an attacker-
controlled blob is equivalent to running arbitrary Python
(`__reduce__` returns a `(callable, args)` tuple that the loader invokes).
Bandit flags all five call sites as MEDIUM severity (B301 / B403).

In Phase 2, HOC added a new attack surface: `NectarFlow` messages
(`DanceMessage`, `RoyalMessage`, `PheromoneDeposit`) need to be signed and
verified. The chosen primitive is HMAC-SHA256 over a *canonical* byte
serialization of identity fields. Using `pickle` here would have been
doubly unsafe (insecure serialization AND signature over non-canonical
bytes).

## Decision

Replace `pickle` with **`mscs` 2.4.0** across all production code paths. A
centralized module `security.py` wraps the `mscs` API (`serialize`,
`deserialize`, `sign_payload`, `verify_signature`) so the rest of the
project depends on a stable HOC-owned interface, not directly on `mscs`.

Rules:

- All production serialization goes through `security.serialize` /
  `security.deserialize`.
- Sensitive deserialization sites (`CombStorage.get`, `HoneyArchive.retrieve`)
  pass `verify=True, strict=True`.
- `pickle` is permitted only in test fixtures that *deliberately* exercise
  legacy / hostile payloads (e.g. `TestMscsRejectsMalicious`).

## Alternatives considered

### Keep `pickle` + HMAC the blob

HMAC would authenticate the bytes but would not prevent a malicious signed
blob from constructing arbitrary classes via `__reduce__` on deserialize
— because in HOC's model the attacker can be a peer with the shared HMAC
key (see [ADR-003](ADR-003-hmac-shared-key.md)). Rejected.

### `json` + a custom type registry

Would work for simple objects but HOC serializes `HexCoord`,
`HoneycombCell` state snapshots, and other non-JSON-native types.
Implementing the registry, reviver, and bytes-for-non-JSON-types (e.g.
`bytes` itself) is exactly what `mscs` does — no reason to reimplement.

### `msgpack` or `cbor2` + explicit schema

Both are fast and compact but neither ships a class registry with
strict-mode enforcement. Could be built on top but again: reinventing
`mscs`.

### Protocol Buffers / Cap'n Proto

Heavy: requires a `.proto` file per serialized type, and adds a build
step. Overkill for HOC's internal use and introduces a non-Python
dependency. Rejected for v1.x; may reconsider if cross-language peers
become a requirement.

## Consequences

**Easier**:

- Bandit MEDIUM findings go from 3 → 0.
- Deserialization is restricted to a pre-registered set of classes — the
  classic `pickle` RCE-via-`__reduce__` vector is closed.
- `mscs` exposes `hmac_key=` at the serialize/deserialize boundary, so
  authentication and serialization share one code path.
- Phase 2 added 43 security tests (`tests/test_security.py`) that would
  have been impossible to write meaningfully against `pickle`.

**Harder**:

- Every new serializable class must be registered with `mscs.register(cls)`
  at import time. Missing a registration produces a clear error at
  serialize time but is an extra step.
- `mscs` is a project dependency. Its maintainer is the same as HOC's,
  which is operationally fine but worth noting (bus factor 1 on that
  dependency).
- End-to-end overhead measured at +3.5% (submit 1000 tasks + 50 ticks),
  concentrated in HMAC computation. Acceptable per the Phase 2 <5%
  target.

**Risk / follow-up**:

- If `mscs` ever ships a breaking API change, `security.py` is the single
  isolation point. Keep that module thin and under the control of HOC
  maintainers.
- Key rotation for HMAC is manual. Addressed in a later phase.

## References

- `security.py`: `serialize`, `deserialize`, `sign_payload`,
  `verify_signature`.
- Phase 2 closure: `snapshot/PHASE_02_CLOSURE.md` §2.1.
- Bandit comparison: `snapshot/bandit_phase01.json` (3 MEDIUM pickle)
  vs `snapshot/bandit_phase02.json` (0).
- Test coverage: `tests/test_security.py::TestMscsRejectsMalicious` (5
  tests).
- `mscs` source: <https://github.com/esraderey/mscs> (2.4.0, MIT, zero
  deps).
