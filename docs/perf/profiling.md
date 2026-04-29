# HOC Profiling Guide (Phase 7.8)

How to profile a HOC workload with `py-spy`, capture a flame graph, and
read it. Aimed at contributors who want to confirm that a perf
proposal targets real hot paths.

## Setup

`py-spy` is a sampling profiler that doesn't require recompiling
Python. Install it once:

```bash
pip install py-spy
```

It runs as a separate process and attaches to the target. On Linux you
may need elevated privileges (or `setcap cap_sys_ptrace=eip
$(which py-spy)`); on macOS/Windows the default install is enough.

## One-liner: profile a benchmark

The simplest invocation — record 30 seconds of a long-running workload
and produce an SVG flame graph:

```bash
py-spy record \
    --output flame-grid-tick.svg \
    --duration 30 \
    --rate 250 \
    -- python -m benchmarks.bench_swarm_1000_tasks
```

`--rate 250` is the sampling rate (Hz). 100 is the default; bumping it
to 250 catches more sub-microsecond frames at the cost of disk volume.

## With the helper script

`scripts/profile_grid.py` wraps the common case (small grid, N ticks,
fixed seed) so you don't have to remember the right `python -c` block:

```bash
python scripts/profile_grid.py --radius 3 --ticks 200 \
    --output snapshot/flame-radius3-200ticks.svg
```

The script internally uses `await grid.tick()` (the canonical Phase 7+
path), so the flame graph reflects the async tick loop's actual cost.

## Reading a flame graph

Each row is a stack frame; width = total CPU time spent in that frame.
What to look for in HOC traces:

- `_async_parallel_tick` should dominate when
  `parallel_ring_processing=True` (the default for radius ≥ 2). If a
  ring batch saturates a single thread, you'll see a very wide
  `_async_process_cells_batch` frame.
- `_sync_execute_tick` (per-cell) should be the deepest CPU sink.
  Anything wider than vCore work points to lock contention.
- `BehaviorIndex.pop_best` should be tiny (O(log n)). If it's big,
  the index isn't getting compacted and tombstoned entries are
  lingering — check that `INDEX_COMPACT_INTERVAL_TICKS` is firing
  every N ticks.
- `PheromoneField.decay_all` should split between the SIMD path
  (`np.power`) and the per-deposit Python loop. The Python loop
  appears for cells with ≤3 deposits; SIMD for ≥4.
- `asyncio.to_thread` overhead — wide rows here mean the thread
  pool is saturated; consider increasing `max_parallel_rings` or
  `default_executor` size.

## Comparing two profiles

Take a baseline before your change, apply the change, take another
profile, and diff:

```bash
py-spy record --output baseline.svg ...
# apply patch
py-spy record --output patched.svg ...

# Open both in a browser and compare visually, OR use
# https://github.com/jlfwong/speedscope to overlay both.
```

For non-visual comparison, `pytest-benchmark` is more rigorous (it's
what the CI bench job uses). py-spy is for finding *where* the time
goes; pytest-benchmark is for confirming a change *moves the needle*.

## CI integration

`bench-regression` job (`.github/workflows/bench.yml`) runs
`pytest-benchmark` against `snapshot/bench_baseline_ci.json` with a
10% threshold. If the threshold trips:

1. Run `python scripts/profile_grid.py` locally with the same radius
   and ticks the bench uses.
2. Compare the flame graph against the previous baseline (committed
   under `snapshot/flame-*.svg` per phase).
3. Either fix the regression or — if the regression is intentional
   (e.g., correctness fix that costs CPU) — refresh the baseline via
   `gh workflow run bench.yml` from main and commit the new
   `bench_baseline_ci.json`.

## Phase 7-specific gotchas

- **`async def tick`**: py-spy's flame graph shows the coroutine call
  chain, not the conceptual "tick stack". The body of `tick` looks
  shallow because `await asyncio.to_thread(...)` returns control to
  the event loop; the actual CPU work is in a worker thread. Use
  `--threads` to profile across all threads:

  ```bash
  py-spy record --threads --duration 30 -- python ...
  ```

- **`run_tick_sync` deprecation warnings**: Phase 7.2's wrapper emits
  `DeprecationWarning` exactly once per process. The warning's
  `stacklevel=2` points to the caller, not the wrapper, which is
  what you usually want when migrating call sites to `await`.

- **Sandbox process forks**: `SandboxedTaskRunner` with
  `isolation="process"` spawns subprocesses. py-spy can attach to
  any of them via `--pid`, but doesn't follow children automatically.

## Further reading

- `docs/perf/baseline_v2.md` — Phase 7 vs Phase 6 baseline numbers.
- `snapshot/bench_baseline_ci.json` — the CI baseline (refreshed
  per major phase).
- `CONTRIBUTING.md` — bench harness conventions and the
  `compare_bench.py` helper.
