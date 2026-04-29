"""Phase 7.9 — Comparative benchmarks (HOC vs Ray / Dask / multiprocessing).

The Phase 7 brief specified comparing HOC throughput against the
mainstream Python distributed-compute stacks. Phase 7 closure
deferred the **full** implementation to a Phase 7.x followup
because Ray (~30 MB) and Dask (~10 MB) are heavyweight dev deps
that aren't justified for the current scope. This package ships
the scaffolding so a follow-up can land the comparators additively
without touching the Phase 7 closure.

Layout::

    comparative/
    ├── _workloads.py            -- Shared workloads (Monte Carlo Pi, ...)
    ├── bench_hoc.py             -- HOC baseline (always runnable)
    ├── bench_hoc_vs_mp.py       -- multiprocessing.Pool (stdlib)
    ├── bench_hoc_vs_ray.py      -- Ray (skip if not installed)
    └── bench_hoc_vs_dask.py     -- Dask (skip if not installed)

Each comparator uses ``importlib.util.find_spec`` to detect its
dependency and skips the bench cleanly when missing. The skipped
benches are still collected by pytest so they show up in the
output as "skipped" rather than going silent.

Brief's target workloads (Monte Carlo Pi is implemented in v1;
SVD and FFT 2D are stubbed for the followup):

- Monte Carlo Pi, 1M samples — embarrassingly parallel, sub-second
  per worker, exercises submit+drain overhead.
- SVD 1000x1000 — numpy-bound; useful for testing how each backend
  handles GIL-bound work.
- FFT 2D 4096x4096 — numpy-bound; same.

Run via pytest::

    pytest benchmarks/comparative/ -v --benchmark-only
"""

from __future__ import annotations
