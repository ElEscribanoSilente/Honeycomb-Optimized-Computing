# ADR-013: StorageBackend protocol — pluggable persistence for HoneyArchive

- **Status**: Accepted
- **Date**: 2026-04-27
- **Phase**: Phase 6 (6.1, 6.2)

## Context

Phase 6's headline is "HoneyArchive goes from in-memory only to
pluggable backends + checkpointing + crash recovery". Phase 1 left
``HoneyArchive`` as a single ``dict[str, bytes]`` plus an
``OrderedDict`` metadata sidecar. Phase 2 wrapped it with
HMAC + mscs (path validation, signed serialization). Phase 3 / 4 / 5
did not touch its persistence layer.

For Phase 6 the operational ask is real: the archive must survive a
process restart. The brief listed candidate backends (SQLite, LMDB,
S3, Redis) and called out that "el backend solo ve bytes opacos"
— i.e. HMAC + mscs framing must stay above the persistence
boundary so swapping the medium does not weaken the security
envelope.

The decision to make is the *shape* of that boundary: a typed
``Protocol`` versus an ``ABC`` versus a hand-rolled mixin, what the
methods are, what concurrency contract they make, and where the
default in-memory implementation lives.

## Decision

### Protocol, not ABC

``hoc.storage.StorageBackend`` is a ``typing.Protocol`` decorated
``@runtime_checkable``. Five methods + one dunder:

```python
class StorageBackend(Protocol):
    def put(self, key: str, value: bytes) -> None: ...
    def get(self, key: str) -> bytes | None: ...
    def delete(self, key: str) -> bool: ...
    def keys(self, prefix: str = "") -> Iterator[str]: ...
    def __contains__(self, key: object) -> bool: ...
```

Why a Protocol:

- **Structural typing.** The two concrete backends shipped in
  Phase 6 (``MemoryBackend``, ``SQLiteBackend``) do not share an
  inheritance tree — they have different concurrency primitives
  (``RLock`` vs SQLite's WAL). An ABC forces a shared parent;
  Protocol does not.
- **Existing conventions.** Phase 4 used a Protocol shape for the
  CAMV bridge (``VCoreProtocol``, ``HypervisorProtocol``,
  ``NeuralFabricProtocol`` in ``hoc.bridge.mappers``). Same pattern
  here is the path of least surprise.
- **Cheap extension.** Future LMDB / S3 / Redis backends can wrap
  third-party clients without inheriting from a HOC base. They just
  need to expose the five methods.
- **``runtime_checkable``** so tests can assert
  ``isinstance(backend, StorageBackend)`` without committing to a
  concrete class.

### Five methods, no more

The brief listed exactly the five above. We considered adding:

- ``__len__`` — useful on the Memory side (O(1) ``len(backend)``).
  Excluded from the protocol because SQL backends would have to
  emit ``SELECT COUNT(*)`` (O(N)) or maintain a sidecar counter
  (extra write path). ``MemoryBackend`` exposes ``__len__`` as a
  bonus; SQLiteBackend does not.
- ``clear()`` — convenient for tests but a sharp edge in production.
  Not required; tests build a fresh backend per case.
- ``batch_put`` / ``transaction`` — relevant only when a real
  driver supports them. Defer until a backend wants to expose
  better-than-loop semantics; the protocol stays minimal until
  then.
- ``close()`` — useful for SQLite (releases the connection +
  flushes WAL). It exists on ``SQLiteBackend`` but is intentionally
  not in the protocol. The default ``MemoryBackend`` has nothing
  to close; forcing every backend to expose ``close`` would be
  busywork.

### Thread-safety as a contract

Every conformant backend must serialize concurrent calls.
``MemoryBackend`` does it with an internal ``threading.RLock``;
``SQLiteBackend`` relies on connection-per-thread + SQLite's WAL
(reader / single-writer concurrency). HOC is multithreaded by
default (the grid uses a ``ThreadPoolExecutor``); a non-thread-safe
backend would surface as flaky tests, not as a clean error.

The protocol documentation states this explicitly. ``keys()`` is
allowed to return a *snapshot* iterator rather than a live view —
i.e. concurrent mutations may or may not appear in an in-flight
iteration; callers that need a stable view should ``list(...)`` the
iterator before iterating.

### Default backend on the constructor

``HoneyArchive.__init__(config, base_path=None, backend=None)``.
``backend=None`` constructs a fresh ``MemoryBackend`` — same
observable behaviour as pre-Phase-6. Passing
``SQLiteBackend(path)`` (or any other future backend) swaps the
medium without touching the archive's HMAC + mscs envelope.

Migration path for existing call-sites: zero-impact. Pre-Phase-6
``HoneyArchive(MemoryConfig())`` still works. Tests that poked at
``archive._archive: dict[str, bytes]`` directly have not been
ported; that attribute was internal and only ``test_memory.py``
touched ``_metadata`` (which is unchanged).

### CombStorage stays in-memory

The Phase 6 brief mentioned refactoring both ``CombStorage`` (L2)
and ``HoneyArchive`` (L3). Only L3 actually got the backend.
Rationale (also captured in the Phase 6.1 commit message):

- ``CombStorage``'s distribution model — hash a key to a HexCoord,
  replicate to ring neighbours — is part of the *cache* semantics,
  not the persistence medium. A flat key-value backend would lose
  the hex topology or require a compound-key layout
  (``"{q},{r}:{key}"``) that hides the physical placement.
- The use case for L3 persistence is real (durable archive
  surviving restarts). The use case for L2 persistence is not —
  Comb is a distributed cache; if a process restarts, the cache
  warms up from L3.
- Adding a backend to CombStorage now would be speculative; the
  Phase 6.1 scope deliberately stops at L3.

A future phase can revisit if a real durable-L2 requirement appears
(e.g. multi-node Phase 8 with a shared cache).

### SQLite as the first real backend (Phase 6.2)

The brief named SQLite as the default real backend. We keep that
choice. Differentiators:

- **stdlib.** No new runtime dep; ``sqlite3`` ships with CPython.
  Phase 2's ``mscs`` choice and Phase 4's ``tramoya`` choice both
  optimised for "small, sharp deps"; reaching for stdlib first is
  the same posture.
- **WAL.** ``PRAGMA journal_mode=WAL`` gives concurrent readers
  + one writer with crash-safe durability. Skipped for
  ``:memory:`` databases (which have no journal file). Tested.
- **Connection-per-thread.** SQLite connections are not
  thread-safe; ``threading.local`` gives each thread its own.
  ``check_same_thread=True`` enforces the contract.
- **Schema versioning** in a sibling ``_schema_version`` table.
  Migrations run in a single transaction at startup.
  ``_run_migrations_to_current`` is the entrypoint future schema
  bumps extend.

Alternatives considered and deferred:

- **LMDB.** Higher write throughput than SQLite, but adds a
  ~500 KB external dep + lacks SQL ergonomics. Phase 6.9
  (optional) per the brief.
- **S3.** Cloud-native; the right answer for distributed
  Phase 8+. Adds boto3 (~10 MB transitive) and is meaningless
  on a single-node setup. Phase 6.9 (optional).
- **Redis.** Distributed cache, not a persistent store. Same
  Phase 6.9 bucket, same rationale as S3.

## Consequences

- **Storage subpackage** lives at ``hoc.storage`` — strict mypy
  from day one (not in the legacy override list). The protocol +
  ``MemoryBackend`` total ~120 LOC; ``SQLiteBackend`` is ~250.
- **Test suite** parametrizes one ``backend`` fixture across
  ``MemoryBackend`` + ``SQLiteBackend(tmp_path / "...")``. Future
  backends append a new ``param`` string and the contract suite
  runs against them automatically.
- **HoneyArchive's public API** is unchanged for existing
  callers; the optional ``backend=`` kwarg is purely additive.
- **mscs registry** is untouched — this layer ships *bytes*, not
  custom classes. Phase 6.3's checkpoint blob does sit on the
  same registry and is signed by the same HMAC key, but that's
  scoped to ``HoneycombGrid.checkpoint`` (see ADR-014).

## Status of follow-ups

- Phase 6.9 (optional, deferred): LMDB / S3 / Redis backends.
- Phase 7 (async migration): the protocol may need an
  ``AsyncStorageBackend`` sibling. Decide then.
- ``CombStorage`` may grow its own backend abstraction if
  Phase 8 demands shared L2 across nodes.
