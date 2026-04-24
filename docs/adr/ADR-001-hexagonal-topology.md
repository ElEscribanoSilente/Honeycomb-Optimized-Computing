# ADR-001: Hexagonal topology for the cell grid

- **Status**: Accepted
- **Date**: retroactive — decision made at project inception (v1.0.0)
- **Phase**: pre-roadmap

## Context

HOC is a distributed-compute framework inspired by beehive organization
(queen, workers, drones, nurses, foragers, pheromone trails, waggle dance).
At the topological level, cells need neighbors: a worker cell looks at its
neighbors to pass nectar (tasks), decay pheromones, and mirror state during
failover. The topology shape determines:

- Average graph distance (affects latency of cluster-wide gossip).
- Redundancy model (how many neighbors back up a given cell).
- Visual / debugging intuition.
- Packing efficiency (area per cell relative to perimeter).

## Decision

HOC uses a **flat-top regular hexagonal grid** with axial coordinates
`(q, r)` and the cube-coordinate invariant `q + r + s = 0`. Each cell has
exactly **6 neighbors** via `HexDirection` (EAST, SOUTHEAST, SOUTHWEST,
WEST, NORTHWEST, NORTHEAST). Distances are Manhattan-like on the cube
representation: `max(|Δq|, |Δr|, |Δs|)`.

## Alternatives considered

### Square grid (4-connected)

Simpler math but only 4 neighbors → less redundancy, higher diameter, and
diagonal moves either inflate distance or require 8-connectivity (losing
the simplicity). Rejected.

### Square grid (8-connected)

8 neighbors but *anisotropic*: diagonal steps are `√2` further in
Euclidean space than axial steps. Distance metrics become awkward and the
pheromone gradient visualization loses intuition. Rejected.

### Triangular grid

Only 3 neighbors — minimum redundancy. Visualization is awkward because
alternating triangles point up/down. Rejected.

### Graph with arbitrary edges (e.g. small-world, Watts-Strogatz)

More flexibility in wiring and theoretically lower diameter, but loses the
geometric intuition that makes bio-inspired metaphors legible (there is no
"direction to nearest food source" without embedding). Considered and
rejected for v1 — may return in a later phase as a topology strategy.

## Consequences

**Easier**:

- 6 neighbors provide natural 2-of-3 mirror redundancy (per side).
- Cube coords make rotations, reflections, and ring enumeration trivial.
- ASCII / SVG rendering renders aesthetically (see `HoneycombVisualizer`).
- Graph diameter in a hex with radius `r` is `2r` vs `~sqrt(2)·r` for a
  square grid of the same area — comparable.
- Packing ratio (area / perimeter) maximal among regular tilings of the
  plane.

**Harder**:

- New contributors must learn axial / cube / offset coordinate conversions.
  `HexCoord` encapsulates this, but initial ramp-up is steeper.
- Existing libraries often assume square grids; integration with
  off-the-shelf visualizers requires adapters (`bridge.HexToCartesian`).
- Distance is not L1 or L∞ but the cube metric. Programmers used to
  Manhattan distance occasionally miscount.

**Risk / follow-up**:

- If a future workload needs anisotropic distances (e.g. a north-south
  "fast lane"), the hex topology will be the wrong primitive. Revisit
  then.

## References

- Red Blob Games' hex grid reference:
  <https://www.redblobgames.com/grids/hexagons/>
- HOC internal: `core.HexCoord`, `core.HexDirection`, `core.HoneycombGrid`.
- Hypothesis property tests in `tests/test_property.py` verify algebraic
  invariants (cube identity, ring size, rotation).
