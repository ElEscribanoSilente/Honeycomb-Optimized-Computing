# ADR-018: BehaviorIndex perf optimisation (Phase 7.5)

**Status**: Accepted (2026-04-28)
**Context**: Phase 7 — perf attack on the SwarmScheduler hot path
**Decision-makers**: Esraderey

## Context

The Phase 5 closure flagged ``SwarmScheduler.tick`` as O(n·m): the
filter loop iterated every pending task for every behaviour. For the
Phase 7 brief's reference load (n=1000 tasks, m≈22 worker behaviours
on a radius-3 grid), that's ~22k filter ops per tick plus the
internal scoring loops in ``ForagerBehavior.select_task`` (~12k more
ops). The perf budget for Phase 7 — ≥5× throughput vs the v1.6.0
baseline — couldn't be met without attacking this directly.

## Considered alternatives

### A — Single global priority queue, dispatched at pop time

- One ``heapq`` of all pending tasks.
- ``tick()`` pops the top, decides which behaviour to dispatch to.
- Pro: trivially simple data structure.
- Con: dispatch decision is still O(m) per pop; for n tasks you do
  O(n) pops × O(m) dispatch = back to O(n·m) just shifted to a
  later phase.

### B — Per-behaviour-class queue (4 heaps total)

- One heap per behaviour CLASS (Forager, Nurse, Scout, Guard).
- ``submit_task`` routes by ``task_type`` → class.
- ``tick()`` iterates each class once.
- Pro: m=4 is tiny; the loop is cheap.
- Con: behaviour instances at different coords can't be distinguished
  per pop — need a secondary lookup. Pinned tasks
  (``target_cell != None``) need special-case routing anyway.

### C — Per-behaviour-instance queue (chosen)

- One heap per registered behaviour INSTANCE.
- ``submit_task`` routes by ``_route_task_to_behaviors``: pinned
  tasks go only to the behaviour at that coord; global tasks fan
  out across all behaviours of the matching class.
- ``tick()`` calls ``pop_best(behaviour)`` per behaviour.
- Pro: matches the brief's API spec exactly. Per-pop is O(log n).
  Dispatch happens at submit time, not tick time.
- Con: a single global task lands in ~12 forager heaps (one entry
  per forager). To prevent re-execution we tombstone on pop.

## Decision

**C** with lazy tombstoning. Concretely:

- New ``BehaviorIndex`` class (`swarm.py`):
  - ``insert(task, behaviour)`` — heappush into the behaviour's heap.
    Sequence counter for FIFO tie-break at equal priority.
  - ``pop_best(behaviour)`` — heappop the highest-priority active
    task; auto-tombstone its id. Lazy-skip tombstoned entries on the
    way down.
  - ``remove(task_id)`` — tombstone for cancel / complete /
    fail-no-retry.
  - ``compact()`` — periodically (every
    ``INDEX_COMPACT_INTERVAL_TICKS=10`` ticks) prunes tombstoned
    entries from heaps and clears the tombstone set. Called from
    ``SwarmScheduler.tick``.

- ``SwarmScheduler`` integration:
  - ``_initialize_behaviors`` registers each behaviour with the
    index.
  - ``submit_task`` calls ``_route_task_to_behaviors(task)`` to find
    matching behaviours, then ``index.insert(task, b)`` for each.
  - ``tick()`` replaces the O(n·m) filter loop with one
    ``pop_best(b)`` per behaviour. Probabilistic refusal preserved:
    if ``behaviour.select_task([candidate])`` returns ``None``, the
    candidate is re-inserted (tombstone clears on re-insert).
  - ``cancel_task`` + cleanup loop also call ``index.remove(...)``.

- Routing (``_route_task_to_behaviors``):
  - ``target_cell`` set: pinned. Goes to the single behaviour at
    that coord, only if the type is acceptable.
  - ``target_cell`` None: global. Fans out to every behaviour whose
    class matches the task type. Foragers are the catch-all.

## Consequences

### Positive

- **Throughput**: ``test_swarm_1000_tasks_single_tick`` ≈1.7ms,
  ≈6× faster than the extrapolated pre-Phase-7.5 baseline. The
  ≥5× target lands.
- **Per-tick cost**: m·log(n) ≈ 220 ops vs ~34k pre-Phase-7.5.
- **Memory bounded**: ``compact()`` runs every 10 ticks; tombstone
  set never grows unbounded.
- **Submission cost**: O(log n) per matching behaviour. For a
  pinned task, that's one insert. For a global compute task on a
  radius-3 grid, ~12 inserts (one per forager). Still vastly
  cheaper than the pre-Phase-7.5 loop.

### Negative

- **Behaviour change**: probabilistic refusal in ``ForagerBehavior``
  (``should_respond``) now re-inserts the task instead of leaving
  it for the next behaviour to try. Practically equivalent for
  Phase 1 / 2 / ... tests (they don't probe the refusal mechanism
  at the scheduler level), but a strict-equivalence purist could
  argue it's not byte-identical.
- **Test fixtures that flip ``task.state`` directly**: a couple of
  tests (B2.5 fixtures in `test_swarm.py`) marked tasks as
  COMPLETED via direct attribute assignment, then ran ``tick()``.
  Pre-Phase-7.5 the filter excluded them; now ``pop_best`` returns
  them. The fix: ``tick`` re-checks ``task.state is PENDING``
  after pop and skips otherwise. Documented in the tick body.
- **Routing maintenance burden**: adding a new behaviour subclass
  requires updating ``_behavior_accepts_type`` to declare its
  acceptable types. Trade-off vs the previous "every behaviour
  filters in select_task" pattern.

### Neutral

- ``BehaviorIndex.compact`` is O(total heap entries) per call.
  At every-10-tick cadence with ~12 foragers and 1000 tasks, that's
  ~12k ops every 10 ticks = 1.2k ops amortised per tick. Still well
  below the pre-Phase-7.5 22k.
- The ``size_for(behaviour)`` helper does a linear scan to count
  non-tombstoned entries. Used only by tests; no perf-path consumer.

## References

- Phase 7 brief in ``ROADMAP.md`` § FASE 7 § 7.5.
- ``swarm.py`` — ``BehaviorIndex`` + integration.
- ``tests/test_behavior_index.py`` — 28 tests covering API +
  scheduler integration.
- ``benchmarks/bench_swarm_1000_tasks.py`` — perf measurement.
