# ADR-004: `OrderedDict` LRU for `PheromoneTrail._deposits`

- **Status**: Accepted
- **Date**: 2026-04-23 (retroactive — decision made during Phase 2)
- **Phase**: Phase 2

## Context

`PheromoneTrail._deposits` maps each hex coord to a per-type dict of
`PheromoneDeposit`. In v1.0.0 and v1.1.0 this was a
`defaultdict(dict)` with no upper bound on the number of coordinates. An
attacker (or a buggy producer) could deposit pheromones at unbounded
distinct coords and exhaust host memory.

The cap needs to:

1. Enforce a maximum number of *distinct coordinates* (the actual DoS
   vector — the per-coord enum `PheromoneType` already caps intensity per
   coord at a small constant, see lesson #4 in
   `snapshot/PHASE_02_CLOSURE.md`).
2. Evict the "oldest" coord when the cap is reached, where "oldest" means
   least-recently touched (deposited or queried). Strict insertion order
   would evict long-lived-but-active coords; a random-eviction policy
   would drop hot paths.
3. Keep `deposit` / `sense` / `follow_gradient` on their current O(1)
   path.

## Decision

Back `_deposits` with a `collections.OrderedDict` keyed by `HexCoord`.
On every `deposit` or update, call `move_to_end(coord)` to mark the coord
as most-recently-used. When `len(_deposits) >= max_coords`,
`popitem(last=False)` drops the LRU entry.

Secondary: cap `metadata` per deposit at `max_metadata_keys` entries
(default 100) via the same OrderedDict-LRU pattern to prevent unbounded
per-deposit metadata growth.

Defaults: `max_coords = 10_000`, `max_metadata_keys = 100`. Configurable
in `PheromoneTrail.__init__`.

## Alternatives considered

### `functools.lru_cache`

Wrong abstraction — `lru_cache` caches return values of a function call,
not the contents of a container. Rejected.

### `cachetools.LRUCache`

External dependency. `cachetools` is well-maintained but adds a runtime
dep for a single use case we can satisfy with stdlib. Rejected to keep
zero runtime dependencies beyond `numpy` and `mscs`.

### Random eviction

Simpler to implement (O(1) amortized via `random.choice` on keys) but
drops hot paths with the same probability as cold ones. Workloads where
a handful of coords are pheromone hubs would suffer. Rejected.

### Per-type LRU (one `OrderedDict` per `PheromoneType`)

Reduces contention between ptype "lanes" but multiplies memory overhead
by the number of types (9 in Phase 2). The benefit is marginal given
that `PheromoneTrail._lock` already serializes all access. Rejected for
simplicity.

### Bounded `deque` of `(coord, deposit)`

Preserves insertion order but scanning for "coord X's current deposits"
becomes O(n). We read coords more often than we write. Rejected.

## Consequences

**Easier**:

- Bounded memory: worst-case `len(_deposits) == max_coords`. At default
  `max_coords = 10_000`, a coord entry is ~200 bytes, total ≤ 2 MB.
- DoS vector closed: a flooder that deposits at 1M distinct coords ends
  up with the first 990_000 already evicted by the time attack traffic
  peaks. Test:
  `TestPheromoneBoundedDoS::test_flood_distinct_coords_stays_bounded`.
- Stdlib-only; no dependency delta.
- Hot paths (a few coords, many deposits) are preserved because
  `move_to_end` keeps them at the MRU end.

**Harder**:

- Evictions are no longer deterministic from the caller's perspective:
  "a deposit I made 30 minutes ago may be gone". Contract documented in
  `PheromoneTrail.deposit` docstring. Callers that need durability should
  use `memory.CombStorage`, not `PheromoneTrail`.
- `OrderedDict.popitem(last=False)` is O(1) but requires holding the
  `_lock`. Increases hold time marginally — measurable in micro but
  amortized in end-to-end tests.

**Risk / follow-up**:

- If `max_coords` is set too low for a legitimate workload, useful
  pheromone information gets evicted. Ops should size based on grid
  radius (a `radius=N` grid has `1 + 3N(N+1)` coords — e.g. radius 20 is
  1261 coords, well below the default).

## References

- `nectar.PheromoneTrail._deposits` (implementation).
- `PheromoneTrail.DEFAULT_MAX_COORDS`,
  `PheromoneTrail.DEFAULT_MAX_METADATA_KEYS` (class constants).
- Phase 2 closure: `snapshot/PHASE_02_CLOSURE.md` §2.4 "Bounded growth"
  and "Lecciones aprendidas" #4.
- Test coverage: `tests/test_security.py::TestPheromoneBoundedDoS` (4
  tests covering same-coord flood, distinct-coord flood, metadata flood,
  auto-sign).
