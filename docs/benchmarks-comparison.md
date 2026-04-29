# HOC vs Ray / Dask / multiprocessing — comparative benchmarks

**Status**: Phase 7.9 scaffold landed in v2.0.0; full results
(numbers + analysis) deferred to a Phase 7.x followup. This document
describes the methodology and the harness layout so anyone can run
the comparison locally.

## TL;DR

The Phase 7.9 brief asked for HOC measured against the mainstream
Python distributed-compute stacks under three workloads:

- Monte Carlo Pi, 1M samples (embarrassingly parallel).
- SVD 1000×1000 (numpy-bound).
- FFT 2D 4096×4096 (numpy-bound).

Phase 7 closure shipped the **scaffolding** — a workload module +
four bench files (HOC baseline, multiprocessing, Ray-skip-if-missing,
Dask-skip-if-missing) — with Monte Carlo Pi wired end-to-end. SVD
and FFT are stubbed in `_workloads.py` waiting for the followup to
land them in the bench harness.

The "sweet spot" hypothesis from the Phase 7 brief is that HOC wins
on **many-small-tasks-with-locality**: scenarios where the hex
topology lets the scheduler co-locate related work onto the same
neighbourhood without an explicit DAG. The followup's job is to
back that up with numbers.

## Layout

```
benchmarks/comparative/
├── __init__.py
├── _workloads.py          -- Shared Monte Carlo Pi + SVD/FFT stubs
├── bench_hoc.py           -- HOC baseline (always runnable)
├── bench_hoc_vs_mp.py     -- multiprocessing.Pool (stdlib)
├── bench_hoc_vs_ray.py    -- Ray (skip if not installed)
└── bench_hoc_vs_dask.py   -- Dask (skip if not installed)
```

Each comparator uses `importlib.util.find_spec` to detect its
dependency. Skipped benches show up in pytest output as "skipped"
rather than going silent — you'll see at a glance which backends
were measured vs which were absent.

## How to run

```bash
# Install the optional comparators (not in the [dev] extras to keep
# the default install lean; Ray + Dask combined add ~40 MB).
pip install ray "dask[distributed]"

# Run the full comparative suite.
pytest benchmarks/comparative/ -v --benchmark-only \
    --benchmark-warmup=on --benchmark-min-time=0.5

# Run only HOC vs multiprocessing (no extra deps).
pytest benchmarks/comparative/bench_hoc.py \
       benchmarks/comparative/bench_hoc_vs_mp.py \
       -v --benchmark-only
```

## Methodology

**Hardware parity**: all four backends run on the same machine, same
core count, same Python version. Ray and Dask are configured for
process-based workers (no thread mode) so the comparison is
apples-to-apples against multiprocessing.Pool. HOC's tick loop fans
out via the async path (`asyncio.to_thread` per cell) — that's the
canonical Phase 7 path and the closest analogue to the others'
`Pool.map` / `client.gather`.

**Workload sanity**: every backend's pi estimate is checked against
`math.pi` with a ±0.02 tolerance (≈ 7-sigma at 400k samples). A
failed estimate fails the bench, so a backend can't accidentally
publish numbers from a partial run.

**Default sizes** (set in `_workloads.py`):

- 4 workers
- 100k samples per worker = 400k total samples per round

The followup will likely raise these to the brief's headline 1M
samples + bigger task counts; the defaults are sized so the
scaffolding can run in CI without hogging time.

## Phase 7.x followup checklist

- [ ] Wire SVD 1000×1000 + FFT 2D 4096×4096 workloads through
  `bench_hoc.py` / `bench_hoc_vs_mp.py` / `bench_hoc_vs_ray.py` /
  `bench_hoc_vs_dask.py`.
- [ ] Capture results on a reference machine (Linux + ubuntu-latest
  CI runner). Commit JSON via `gh workflow run` artifact pattern,
  same recipe Phase 6.7 introduced for `bench_baseline_ci.json`.
- [ ] Update this document with a results table + flame-graph
  analysis of HOC's sweet-spot scenarios.
- [ ] Decide whether Ray / Dask deserve a permanent slot in
  `[project.optional-dependencies]` (e.g., a `compare` extras key)
  or stay install-on-demand.

## Why this isn't in v2.0.0

The Phase 7 closure listed comparative benchmarks as "deferred to
Phase 7.x followup". Three reasons:

1. **Dependency footprint**: Ray (~30 MB) and Dask (~10 MB) would
   bloat the dev install for a benchmark that runs once per
   release cycle. Keeping them install-on-demand matches how
   downstream users will actually exercise the comparison.
2. **Methodology rigour**: comparative perf benchmarks need
   careful tuning (worker counts, warmup rounds, statistical
   significance gating) to produce numbers worth publishing.
   Phase 7's primary deliverable was the async migration; the
   comparative bench deserved its own focused phase.
3. **The brief explicitly authorised it**: "deferred a Phase 7.x
   followup" was named in the closure DoD as an acceptable
   outcome.

The scaffold landing in v2.0.0 means a follow-up PR can add
results additively without any structural churn.

## References

- Phase 7 brief in `ROADMAP.md` § FASE 7 § 7.9.
- `snapshot/PHASE_07_CLOSURE.md` § "Items deferred a Phase 7.x
  followup / Phase 9".
- `benchmarks/comparative/_workloads.py` — shared workload
  definitions.
