# ADR-016: Async tick loop (Phase 7.1+7.2)

**Status**: Accepted (2026-04-28)
**Context**: Phase 7 — async/await + performance migration
**Decision-makers**: Esraderey (project owner)

## Context

HOC's pre-Phase-7 tick loop fanned out cell processing through a
``ThreadPoolExecutor`` (``HoneycombGrid._executor``). The model:

- ``HoneycombGrid.tick()`` calls ``_parallel_tick()``.
- ``_parallel_tick`` submits one Future per ring batch.
- ``as_completed`` drains the futures.

Two pressures forced a rethink at Phase 7:

1. **Cooperative scheduling**: Phase 8 (multi-node) will introduce
   network round-trips. Threads block; coroutines yield. A blocking
   call inside a worker thread holds a slot the scheduler could
   otherwise dispatch.
2. **Composability**: callers wanting to drive multiple grids
   concurrently (simulation harnesses, multi-tenant deployments) had
   to manage their own threads on top of HOC's threads. ``await
   asyncio.gather(grid_a.tick(), grid_b.tick())`` is the more
   ergonomic primitive.

## Considered alternatives

### A — keep ThreadPoolExecutor; expose `tick_async()` wrapper

- Add ``async def tick_async()`` that does ``await
  asyncio.to_thread(self.tick)``.
- Pro: zero behaviour change; async API for free.
- Con: doesn't actually compose. Two ``tick_async()`` calls run on
  the same thread pool serialised by GIL; gather adds overhead, no
  win.

### B — TaskGroup-based fan-out (3.11+)

- Use ``async with asyncio.TaskGroup() as tg: tg.create_task(...)``.
- Pro: cleanest cancellation + exception propagation.
- Con: 3.11+ only. HOC's CI matrix is 3.10/3.11/3.12. Bumping to 3.11
  is a separate decision. Phase 7 brief explicitly authorised either
  TaskGroup or ``asyncio.gather``; we picked the wider-compat path.

### C — `asyncio.gather` + `Semaphore` for ring fan-out (chosen)

- Replace ``ThreadPoolExecutor`` with ``asyncio.gather`` over a list
  of ring-batch coroutines, bounded by
  ``asyncio.Semaphore(max_parallel_rings)``.
- Per-cell ``execute_tick`` dispatches its body to
  ``asyncio.to_thread`` so existing locking + sync vCore APIs stay
  unchanged.
- Pro: works on 3.10+, avoids the TaskGroup-only constraint, plays
  cleanly with ``await grid.tick()`` from any caller, and the
  ``Semaphore`` preserves the pre-Phase-7 ``max_parallel_rings``
  knob's meaning (now bounding gather concurrency instead of
  pool max_workers).
- Con: gather doesn't cancel siblings on error by default; Phase 7.x
  followup may revisit.

## Decision

**C**. The four user-facing tick methods become async:

- ``HoneycombGrid.tick()``
- ``NectarFlow.tick()``
- ``SwarmScheduler.tick()``
- ``HoneycombCell.execute_tick()``

Each has a paired ``run_tick_sync()`` (or ``run_execute_tick_sync``)
wrapper that emits one ``DeprecationWarning`` per process and
otherwise runs ``asyncio.run(self.tick())``. The wrapper exists to
let legacy sync callers stay sync until the v3.0 removal.

Wrapper rules:

- Refuses to run from inside a live event loop (``RuntimeError``).
- Emits ``DeprecationWarning`` once per class per process via a
  ``ClassVar[bool]`` flag.
- The brief required only ``HoneycombGrid.run_tick_sync``; we
  extended the same wrapper to NectarFlow / SwarmScheduler /
  HoneycombCell to keep the test-migration diff small. Documented
  here as a deliberate widening of the brief.

Per-vCore ``await asyncio.to_thread(vcore.tick)`` (the Phase 7 brief's
ideal) is **not** what we landed. Rationale: holding the cell's
``RWLock`` across an await is unsafe in async because other cell
mutations on the same event loop thread can re-enter the lock. The
chosen pattern — entire ``_sync_execute_tick`` body in a
``to_thread`` — keeps the lock semantics intact at the cost of
finer-grained vCore parallelism. Phase 7.6+ may revisit if profiling
shows a real ceiling.

## Consequences

### Positive

- ``await grid.tick()`` is the canonical Phase 7+ API.
- Multiple grids compose under ``asyncio.gather`` with no extra
  threading.
- Bench ``test_swarm_1000_tasks_single_tick`` hits ≈1.7ms (≈6× the
  pre-7.5 baseline) because the new path combines async fan-out with
  Phase 7.5's BehaviorIndex.
- Phase 8 (multi-node) gets a clean ``await
  remote_grid.proxy_tick()`` integration point with no thread-pool
  re-architecture.

### Negative

- ``test_grid_tick`` at radius 1 / 2 reads as **slower** under
  Phase 7 (≈1.1ms vs ≈463μs) because event-loop overhead exceeds
  the work at very small cell counts. The CI bench baseline
  (``snapshot/bench_baseline_ci.json``) will be refreshed during
  Phase 7 close-out via ``gh workflow run bench.yml``.
- Mixing ``run_tick_sync`` and ``await tick()`` in the same call
  graph is illegal — the wrapper raises if called from an active
  loop. Documented; tests cover.
- The 50+ existing test call-sites that previously did
  ``grid.tick()`` had to migrate to ``grid.run_tick_sync()``. One-
  line diff per call but visible in the commit.

### Neutral

- Threading is **not** removed: ``asyncio.to_thread`` still uses
  Python's default thread pool. The win is at the dispatch layer,
  not at the GIL layer. Phase 9 (Cython / numba) is the right place
  to attack the GIL.

## Migration path (v1 → v2)

```python
# v1.x (Phase 6 and earlier)
grid.tick()

# v2.0+ canonical
await grid.tick()

# v2.0+ legacy (one-shot DeprecationWarning, removed in v3.0)
grid.run_tick_sync()
```

The wrappers are documented as removed in v3.0 (Phase 10's release).
Plenty of runway for downstream callers to migrate.

## References

- Phase 7 brief in `ROADMAP.md` § FASE 7.
- ``snapshot/PHASE_07_CLOSURE.md`` — closure narrative.
- ``tests/test_async_tick.py`` — canonical async usage.
- ``tests/test_sync_compat.py`` — wrapper contract.
