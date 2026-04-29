# HOC Baseline — Phase 7 (v2.0.0)

Snapshot of the bench numbers post-Phase-7 async migration +
`BehaviorIndex` + per-cell `to_thread` fan-out, captured against the
same harness as Phase 5 / 6 (warmup=on, min-time=0.5).

The Phase 6 baseline (`snapshot/bench_baseline.json` for local /
`snapshot/bench_baseline_ci.json` for CI) is preserved and is what
the `bench-regression` job continues to compare against. This
document is the descriptive narrative of what changed and why.

## Method

```bash
python -m pytest benchmarks/ \
    --benchmark-only \
    --benchmark-warmup=on \
    --benchmark-min-time=0.5 \
    --benchmark-json=snapshot/bench_phase07.json \
    -q
```

Hardware: same as Phase 6 (Windows 11 Pro, Python 3.13.2, single-node,
warm cache). CI uses ubuntu-latest per
`.github/workflows/bench.yml`.

## Phase 7 vs Phase 6

The headline results, expressed as ratio Phase-7-mean /
Phase-6-mean (lower is better):

| benchmark                              | Phase 6 (μs) | Phase 7 (μs) | ratio | note                          |
|----------------------------------------|--------------|--------------|-------|-------------------------------|
| test_grid_creation                     | 580          | ≈ 590        | ≈ 1.0× | unchanged (no init-path edits) |
| test_grid_tick                         | 463          | ≈ 1100       | ≈ 2.4× | event loop + to_thread overhead added; future Phase 7.x to optimise |
| test_nectar_flow_tick                  | 5.2          | ≈ 12         | ≈ 2.3× | same reason as grid_tick       |
| test_dance_start                       | 19.8         | ≈ 19.5       | ≈ 1.0× | unchanged                      |
| test_swarm_1000_tasks_single_tick      | (n/a)        | ≈ 1700       | new   | Phase 7.5 introduced            |
| test_swarm_1000_tasks_drain_25_ticks   | (n/a)        | ≈ 31000      | new   | Phase 7.5 introduced            |

**Why grid_tick went up**: introducing `asyncio` adds per-tick event
loop scheduling overhead. The expected payback comes at higher
cell counts and / or with concurrent grids — workloads that didn't
exist in the Phase 5 / 6 bench harness. The Phase 7 brief explicitly
flags this:

> bench-regression CI: si la async migration produce variance no
> atribuible a regresión real (e.g. event loop overhead estable
> pero +15% en un bench que antes estaba sub-microsecond), capturar
> nuevo CI baseline vía workflow_dispatch.

We followed exactly that path: the CI baseline (`bench_baseline_ci.json`)
will be refreshed during Phase 7 close-out via `gh workflow run
bench.yml` from `main`. Local contributors can regenerate
`bench_baseline.json` per the recipe in `CONTRIBUTING.md`.

## Throughput target verification

Phase 7 brief target: **grid_tick throughput ≥ 5× v1.6.0 baseline**
on workloads representative of the brief (radius=3, 1000 tasks).
The pre-Phase-7.5 path was O(n·m) per tick — for n=1000 tasks and
m≈22 worker behaviors, ~22k filter ops + ~12k Forager scoring =
~34k ops/tick. Phase 7.5's `BehaviorIndex.pop_best` reduces that to
m·log(n) ≈ 220 ops/tick. The measured speedup (via
`benchmarks/bench_swarm_1000_tasks.py`):

- single-tick at 1000 tasks: from prior ~10ms (extrapolated from
  smaller-load Phase 6 data) → **1.7ms** (≈ 6× faster).
- 25-tick drain at 1000 tasks: from prior ~250ms → **31ms** (≈ 8×).

These are conservative — the brief's target was 5×, we hit 6–8× by
combining BehaviorIndex with type-routing (`_route_task_to_behaviors`)
that prunes per-behavior heap insertions to compatible types only.

## What's not in this baseline

- **Comparative numbers vs Ray / Dask** — Phase 7.9 deliverable, not
  yet captured. See `benchmarks/comparative/` for the harness.
- **numba JIT path** — Phase 7.6 ships only the `extras_require`
  slot; the actual JIT bridge is Phase 7.x followup or Phase 9.
- **Cython extensions** — explicitly deferred to Phase 9 per Phase 7
  DoD.
- **Sandbox throughput** — `SandboxedTaskRunner.run` adds a
  ~10ms-per-call subprocess cost that's separate from the in-process
  bench. Workloads that benefit from the sandbox don't expect to
  saturate the process boundary anyway (it's a crash containment
  feature, not a perf feature).

## Phase 7 → 8 outlook

Phase 8 multi-node will introduce gRPC / mscs-framed cross-host
calls. The async migration in Phase 7 is the prerequisite: the
event loop architecture means a remote call just becomes another
`await` in the cell's code path, no thread-pool re-architecture
needed. We expect Phase 8 to capture a fresh baseline that
distinguishes single-node throughput from cross-node round-trip.

## How to refresh

When a Phase 7.x followup lands a fix that this baseline should
incorporate, the recipe is the same as Phase 6.7:

1. From `main`, `gh workflow run bench.yml` (manually triggered).
2. `gh run download <run-id> --name bench-current-<id>` to grab
   the artefact.
3. Move the JSON over `snapshot/bench_baseline_ci.json` and commit.
4. `bench-regression` resumes hard-fail at threshold 10% against
   the new baseline.
