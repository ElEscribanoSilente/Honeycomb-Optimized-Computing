# ADR-014: Checkpoint blob format — HMAC + optional zlib + mscs strict

- **Status**: Accepted
- **Date**: 2026-04-27
- **Phase**: Phase 6 (6.3, 6.4)

## Context

Phase 6.3 introduces ``HoneycombGrid.checkpoint(path)`` /
``HoneycombGrid.restore_from_checkpoint(path)``. The brief asked for
three security / robustness invariants on the on-disk blob:

1. **Authentic.** A bit-flip in the file is rejected; tampered
   checkpoints never round-trip.
2. **mscs strict mode.** An unregistered class in the inner
   payload is rejected at load.
3. **Optional compression** for large grids.

The decision is the byte-level wire format and the order of checks
on the read path.

## Decision

### Wire format

```
+-------------+--------------+----------------------+----------------------+
| byte 0      | bytes 1..32  | byte 33              | bytes 34..end        |
+=============+==============+======================+======================+
| version     | hmac_sha256  | compression_flag     | payload (mscs;       |
|  (=0x01)    | (32 bytes)   | (0=none,1=zlib)      | optionally zlib)     |
+-------------+--------------+----------------------+----------------------+
```

- **Version byte (1 B)** — currently ``0x01``. A future format
  bump (Phase 7+ adds fields, encryption, etc.) gets a new
  version byte. Decode rejects any unknown version with
  ``ValueError``, distinguishable from the security-failure path
  below.
- **HMAC tag (32 B)** — HMAC-SHA256 over
  ``compression_flag || payload`` (the 33 bytes starting at byte
  33). Computed via ``hoc.security.sign_payload``. Verified via
  ``hoc.security.verify_signature`` (constant-time comparison).
- **Compression flag (1 B)** — ``0x00`` for raw, ``0x01`` for
  zlib. Any other value is rejected with ``ValueError``. The
  flag is *inside* the HMAC envelope so an attacker cannot
  re-flag a blob without invalidating the tag.
- **Payload** — ``mscs.dumps(grid_dict)`` output, optionally
  zlib-compressed at level 6. The grid passes a ``dict[str, Any]``
  of primitives (``HoneycombGrid.to_dict()``); this avoids
  registering ``HoneycombCell``/``CellState``/etc. with the mscs
  registry — the dict serializes through mscs's primitive support
  alone.

### Why HMAC outside the mscs envelope

mscs already provides HMAC-SHA256 inside its own ``dumps`` /
``loads`` API (with the ``hmac_key`` argument). The Phase 6.3
brief asked for "HMAC sobre el blob (reusa
security.sign_payload)" — i.e. cover the *whole* on-disk artefact,
including the compression byte and any zlib container.

Calling ``mscs.dumps(payload, hmac_key=…)`` would only sign the
plaintext, not the post-compression bytes. A bit-flip inside the
compressed block would slip past mscs's check (the inner HMAC
verifies the post-decompress plaintext, by which time we've
already paid the decompression cost on attacker-controlled bytes).

So we use mscs *without* its hmac key here, then sign the
compression-flag-prepended body with our outer HMAC. The Phase 2
``hoc.security.get_hmac_key`` invariant carries through —
``HOC_HMAC_KEY`` env var produces a deterministic key that
multiple processes can share.

### Order of checks on the read path

The decode pipeline is:

1. **Length sanity.** Below ``HEADER_LEN + 1`` bytes → ``ValueError``.
2. **Version.** Mismatch → ``ValueError`` (not ``MSCSecurityError``;
   a forward-compat bump is not a security failure).
3. **HMAC tag.** Constant-time compare over
   ``compression_flag || payload`` → ``MSCSecurityError`` if it
   fails. **This runs before any decompression.** That's the
   defence against zlib-bomb inputs: malicious payload that
   expands 1000× would fault us out of memory if we decompressed
   first; running HMAC first means an unsigned bomb is rejected
   in ~30 µs without ever touching ``zlib.decompress``.
4. **Compression flag.** Unknown value → ``ValueError``.
5. **zlib decompress** (if flag=1) — wrap zlib errors as
   ``ValueError``.
6. **mscs strict-mode load** — rejects unregistered classes in
   the payload.

Each step's exception type is *distinguishable*. Tests assert
specific types so a regression can't downgrade a security failure
to a generic ``ValueError`` without tripping the suite.

### Atomic write

``HoneycombGrid.checkpoint`` writes to ``path.tmp`` and calls
``Path.replace(path)``. ``replace`` is atomic on Windows (since
Python 3.3) and on POSIX. A crash between bytes leaves either the
prior snapshot or no snapshot at ``path`` — never a torn file.

Tested in ``tests/test_checkpointing.py::TestAtomicCheckpointWrite``.

### Auto-checkpoint inside ``tick()`` (Phase 6.4)

``HoneycombConfig.checkpoint_interval_ticks`` + ``checkpoint_path``
opt the live grid into self-snapshotting. The hook fires *after*
``_tick_count`` advances, so a restored grid resumes on the *next*
tick rather than re-running the one that just snapshotted.

Failures inside ``_auto_checkpoint`` are caught + logged via
``security.sanitize_error`` and swallowed. A bad path or a full
disk must not abort the live tick — the grid has to keep running;
the next interval gets another shot. Tested in
``TestCheckpointFailureResilience``.

## Recovery semantics

- **RPO (recovery point objective)**: at most
  ``checkpoint_interval_ticks`` ticks of work since the last
  successful snapshot. With ``interval=100`` and ~1 tick / second,
  a kill loses at most 100 s of grid evolution.
- **RTO (recovery time objective)**: dominated by ``mscs.loads`` +
  cell reconstruction. For a radius-2 grid (~19 cells), restore
  takes < 5 ms on commodity hardware (measured in
  ``test_restored_grid_can_resume_ticking``). Larger grids scale
  linearly with cell count.
- **Anti-tamper**: any single bit flip rejects the blob with a
  distinguishable ``MSCSecurityError`` before any payload is
  parsed.

## Consequences

- **No new runtime dep.** ``zlib`` and ``sqlite3`` are stdlib;
  ``mscs`` is already pinned since Phase 2.
- **Storage subpackage** owns the codec: ``hoc.storage.checkpoint``
  is pure encode / decode; ``HoneycombGrid`` only wraps it with
  ``to_dict`` / ``from_dict`` and the atomic write.
- **Forward compat** — version byte makes any future format bump
  diagnosable (``ValueError`` with the exact byte). The current
  format (v1) covers config + cell state + role + history +
  tick_count.
- **Documented gap**: ``SwarmScheduler.task_queue`` is *not* in
  v1. The scheduler is a sibling object, not a member of the
  grid; persisting tasks across restarts is deferred to a future
  phase. ``test_crash_recovery.py`` calls this out so a reader
  doesn't expect "task queue preserved" out of v1.

## Alternatives considered

### sign-inside-mscs only

Cover the inner plaintext only. **Rejected**: a bit-flip in
compressed bytes wouldn't fail until ``mscs.loads`` (after we'd
already decompressed an attacker-controlled blob). Outer HMAC is
defence in depth.

### zstandard instead of zlib

The Phase 6 brief mentioned ``zstandard`` (BSD, ~200 KB) as the
compression backend. We picked zlib because it's stdlib (no new
dep), and the marginal compression gain (zstd ~10–30 % smaller
than zlib level 6) does not pay for the dep here. If a future
phase shows a real I/O bottleneck, the format can be extended
with a new compression flag (``0x02 = zstd``) without breaking
v1 readers.

### JSON / TOML payload

Human-readable, but loses binary fidelity for the future Phase
6.x extensions (vCore handles, byte buffers in pheromones).
mscs is already the canonical HOC framing; reaching for a
separate format here would split the security review surface.

## Validation

- ``tests/test_checkpointing.py`` — 22 tests covering encode /
  decode round-trip (raw + zlib), all six wire-format error
  paths, every ``HoneycombGrid.to_dict`` field round-trips,
  atomic write semantics, tamper resistance.
- ``tests/test_crash_recovery.py`` — 14 tests covering the
  auto-checkpoint integration: interval validation, timing,
  crash + restore at and between intervals, restored grid
  resumes ticking, failure resilience.
