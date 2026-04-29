# ADR-017: Sandboxing model (Phase 7.4)

**Status**: Accepted (2026-04-28)
**Context**: Phase 7 — opt-in process isolation for HiveTask payloads
**Decision-makers**: Esraderey

## Context

The pre-Phase-7 SwarmScheduler runs every task in the panal's own
process. A runaway loop hangs the tick; an exception falls through
the scheduler's catch; a SIGSEGV / OOM kills the process. For
contained execution we need an isolation boundary.

The brief enumerated four candidate isolation modes, with platform
trade-offs:

- ``"none"`` — no isolation; pass-through.
- ``"process"`` — child process; OS-level crash containment.
- ``"cgroup"`` — Linux cgroups v2 for memory/CPU limits.
- ``"job_object"`` — Windows Job Objects for the same.

## Considered alternatives

### Subprocess vs multiprocessing

- **subprocess** — write a small Python program to a temp file, run
  it. Pickling-free but verbose; arg passing is a hand-rolled
  serialisation.
- **multiprocessing** — Python's stdlib abstraction. Two modes:
  - ``fork`` (POSIX only) — child inherits parent's memory; no
    pickling.
  - ``spawn`` (cross-platform) — child re-executes Python; payload
    must be picklable + importable.

We chose **multiprocessing fork** for the v1 implementation. Spawn
caused a ``ModuleNotFoundError: 'HOC'`` in the child during pytest
collection — the test-module qualname under
``--import-mode=importlib`` doesn't survive pickle on Windows.
Subprocess is the natural fallback for Windows; deferred to Phase
7.x followup.

### When to enforce timeout

- **At the API boundary** — wrap every payload call.
- **Inside the worker** — signal.alarm on POSIX, threading.Timer on
  Windows.

We chose the API boundary: ``Process.join(timeout)`` then
``Process.kill()``. The kill path is OS-native (``SIGKILL`` on POSIX,
``TerminateProcess`` on Windows when we land Windows support).

### cgroup integration depth

- **Spawn under systemd-run --user** — root-less, requires systemd.
- **Direct cgroups v2 manipulation** — write to
  ``/sys/fs/cgroup/...``; needs either root or a delegated cgroup
  tree.

Both options are real Phase 7.x followup work; v1 ships a stub that
raises ``SandboxNotSupported`` to keep the API surface honest. The
brief flagged Linux + macOS as the primary deployment targets;
delaying full cgroup integration until Phase 8 (multi-node) lets us
combine the limits with per-node isolation in one pass.

## Decision

Phase 7.4 v1 ships:

- ``hoc.sandbox`` module with ``SandboxConfig`` + ``SandboxedTaskRunner``.
- ``"none"`` — pass-through.
- ``"process"`` (POSIX only) — fork-based; timeout via ``Process.join``;
  kill on overrun; crashes (non-zero exit / OS kill / SIGSEGV) surface
  as ``SandboxCrashed`` with the exit code.
- ``"cgroup"`` — stub; raises ``SandboxNotSupported``.
- ``"job_object"`` — stub; raises ``SandboxNotSupported``.
- ``"process"`` on Windows — explicitly raises ``SandboxNotSupported``
  with a message pointing to the Phase 7.x followup. Better than a
  confusing pickle traceback.
- Probe helpers ``cgroup_v2_available()`` / ``job_objects_available()``
  return ``bool`` answers so future Phase 7.x followups can branch on
  them at runtime.

Default isolation is ``"none"``. Users who want crash containment
must explicitly construct ``SandboxConfig(isolation="process", ...)``.

The runner is **not** automatically wired into ``SwarmScheduler``.
Phase 7.4 deliberately keeps the integration manual — callers who
need it can wrap their ``execute_task`` calls. A future Phase 7.x
might add ``SwarmConfig.sandbox`` for declarative wiring; for now,
the explicit construction makes the perf cost (subprocess fork) and
behavioural change (no shared memory) opt-in by design.

## Consequences

### Positive

- The brief's headline guarantee — "crash en task NO mata el
  panal" — is verified by tests: SIGSEGV in a fork'd child surfaces
  as ``SandboxCrashed`` in the parent without taking the test
  process down.
- API surface is honest: callers asking for unsupported modes get a
  typed error, not a hang or pickle backtrace.
- Future cgroup / job_object work has a clean attachment point
  (the existing ``SandboxConfig.isolation`` literal).

### Negative

- Windows users can't use ``"process"`` isolation in v1. They get a
  clear error and have to wait for Phase 7.x's subprocess-based
  fallback or the Job Objects implementation.
- Memory/CPU limits don't work in v1 even on Linux (cgroup is
  stubbed). ``timeout_s`` is the only enforced limit.
- ``fork``-based isolation has a known caveat: the child inherits
  open file descriptors. Phase 7.x followup may switch to ``spawn``
  + cloudpickle for full hygiene at the cost of pickling overhead.

### Neutral

- ``pywin32`` is registered as ``extras_require={"sandbox-windows":
  ["pywin32"]}``. Installing the extra is a no-op until Phase 7.x
  lands the actual Job Objects path; the slot is reserved so
  deployments can declare intent ahead of time.

## References

- Phase 7 brief in ``ROADMAP.md`` § FASE 7 § 7.4.
- ``hoc/sandbox.py`` — implementation.
- ``tests/test_sandbox.py`` — verification (16 tests; 8 skipped on
  Windows for the fork-only paths).
- Future Phase 7.x followup: full Linux cgroup + Windows Job Objects.
