# ADR-019: Cython / numba acceleration — deferred to Phase 9

**Status**: Accepted (2026-04-28)
**Context**: Phase 7 — closure decisions on optional perf items
**Decision-makers**: Esraderey

## Context

The Phase 7 brief listed two optional acceleration paths:

- **7.6** — SIMD vectorisation (numpy where applicable) + numba JIT
  for hot paths (`HexCoord.distance` batch calls, etc.). Marked
  "opcional pero recomendado". DoD: "SIMD + numba opcional ...
  funcional o deferred a Phase 9".
- **7.7** — Cython extensions for `HexCoord.distance`,
  `PheromoneField.decay_all`, `RWLock`. DoD: "Si Cython agrega
  complejidad de build > value, defer a Phase 9 junto con Rust
  extensions".

Phase 9 (per `ROADMAP.md`) is GPU + Rust extensions via PyO3 +
optional WebAssembly export. It's the natural home for compiled-language
integration because the build infrastructure (cibuildwheel, multi-
platform wheels, Rust toolchain) lands together.

## Considered alternatives

### A — Land everything in Phase 7

- Implement numba JIT bridge on `HexCoord._axial_distance(q1, r1, q2,
  r2)` with `@njit(cache=True)`.
- Implement Cython extensions for the three hot paths.
- Pro: full perf payback in v2.0.0.
- Con: doubles Phase 7 budget. Cython build infrastructure
  (cibuildwheel, manylinux + macOS + Windows wheels) is non-trivial
  and benefits from being tackled together with Rust extensions in
  Phase 9. numba has a 30 MB install footprint that's overkill for
  a single hot path.

### B — Land the SIMD vectorisation only; defer JIT + Cython

- Phase 7.6 ships the numpy SIMD path on `PheromoneField.decay_all`
  (numpy is already a runtime dep). 4-deposit threshold below which
  the Python loop wins on overhead.
- `extras_require={"jit": ["numba"]}` slot reserved so deployments
  can pre-install numba ahead of the Phase 9 wire-up.
- `extras_require={"sandbox-windows": ["pywin32"]}` slot reserved
  for Phase 7.x followup Windows sandbox.
- Phase 7.7 Cython explicitly deferred to Phase 9.

### C — Defer everything; only document

- Skip Phase 7.6 entirely; both SIMD and JIT/Cython are Phase 9.
- Pro: cleanest commit boundary.
- Con: leaves an obvious win unrealised — the numpy SIMD path is
  ~40 LOC and uses an existing dep.

## Decision

**B**. Phase 7 ships:

- `core/pheromone.py::PheromoneField.decay_all` numpy-vectorised
  when n ≥ 4 deposits (`np.power` + multiply + tombstone scan).
  Below n=3, the per-deposit Python loop is faster — numpy setup
  overhead dominates. Threshold validated experimentally; revisit
  if profiling shows real-world cell pheromone counts skew higher.
- `pyproject.toml` registers `[project.optional-dependencies]`:
  - `jit = ["numba>=0.59"]`
  - `sandbox-windows = ["pywin32>=306; sys_platform == 'win32'"]`
- No actual numba JIT wrapper lands. The slot is reserved.

Phase 7.7 (Cython) is **deferred to Phase 9**. Phase 9 will land
both Cython and Rust (PyO3) extensions in a single
cibuildwheel-driven wheel build; combining them amortises the
cross-platform CI work.

The full numba JIT bridge (`@njit(cache=True)` on
`_axial_distance(q1, r1, q2, r2)` with pure-Python fallback) is
**deferred to a Phase 7.x followup or Phase 9**. Whichever
ships first picks up the reserved `[jit]` extras slot.

## Consequences

### Positive

- v2.0.0 ships with the cheapest perf win (numpy SIMD on
  `decay_all`) at zero new dependency cost (numpy is already
  required).
- The `[jit]` and `[sandbox-windows]` extras slots are public
  contracts: deployments can write `pip install hoc[jit]` ahead of
  time without the install failing on a missing key.
- Phase 9's Cython + Rust + GPU work lands as one cohesive
  acceleration phase — the build pipeline (cibuildwheel, manylinux
  tags, macOS universal2, Windows MSVC) is set up once for all
  three.

### Negative

- Hot-path users who'd benefit from JIT (`HexPathfinder` batch
  calls) get only the Python implementation in v2.0.0. The
  workaround is to install numba locally and write their own
  `@njit` decorator on `HexCoord.distance_to`; the public API
  doesn't expose a hook for that yet (Phase 9 will).
- Cython users get nothing in v2.0.0. Acceptable per the brief's
  explicit "deferred a Phase 9" allowance.

### Neutral

- The `[jit]` extras key being reserved-but-empty is mildly
  unusual. Documented in `README.md` § Extras opcionales: "Phase
  7.6+ scaffold; full bridge deferred a Phase 9".
- Phase 9's roadmap entry already mentions Rust extensions via
  PyO3 + cibuildwheel. The Cython carryover is a small line item
  in that phase, not a separate sub-phase.

## When to revisit

Phase 9. Specific triggers that'd accelerate the revisit:

1. Profiling on production workloads shows `HexPathfinder` batch
   calls dominating tick time (currently we see `BehaviorIndex.pop_best`
   and `_async_parallel_tick` as the two largest blocks).
2. A Phase 8 multi-node deployment hits cross-node round-trip
   ceilings that compiled-language hot paths could relieve.
3. A user PR lands the `@njit` bridge as a stand-alone contribution
   — the API hook is small, ~30 LOC including the fallback.

## References

- `ROADMAP.md` § FASE 7 § 7.6 + 7.7 brief.
- `ROADMAP.md` § FASE 9 — GPU & Aceleración.
- `core/pheromone.py::PheromoneField.decay_all` — Phase 7.6 SIMD
  path.
- `pyproject.toml` § `[project.optional-dependencies]` — `jit` +
  `sandbox-windows` extras slots.
- `snapshot/PHASE_07_CLOSURE.md` § "Items deferred a Phase 7.x
  followup / Phase 9".
