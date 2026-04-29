"""Phase 7.9 — Shared workloads for the comparative benchmark suite.

These need to be top-level callables (not closures or lambdas) so
``multiprocessing.Pool`` and ``Ray`` can pickle them. The workload
shapes are tuned to be interesting under each backend:

- ``monte_carlo_pi_chunk(n)`` — embarrassingly parallel, no I/O,
  shouldn't saturate at 4-8 workers; ideal for measuring
  submit+drain overhead.
- ``svd_1000_chunk(seed)`` — numpy SVD on a 1000x1000 matrix; GIL-
  bound; differentiates backends that can release the GIL during
  numpy work (Ray with ProcessPoolExecutor) from those that can't.
- ``fft_2d_4096_chunk(seed)`` — numpy FFT on 4096x4096; same as SVD
  but more memory pressure.
"""

from __future__ import annotations

import math
import random


def monte_carlo_pi_chunk(n_samples: int, seed: int = 0) -> int:
    """Estimate count of points falling inside the unit circle in
    [0,1]x[0,1]. Returns the count (caller divides by n_samples and
    multiplies by 4 to get pi). Pure Python so it parallelises
    cleanly across processes."""
    rng = random.Random(seed)
    inside = 0
    for _ in range(n_samples):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside


def aggregate_pi(per_worker_counts: list[int], samples_per_worker: int) -> float:
    """Combine per-worker counts into a final pi estimate."""
    total_inside = sum(per_worker_counts)
    total_samples = samples_per_worker * len(per_worker_counts)
    return 4.0 * total_inside / total_samples


def svd_1000_chunk(seed: int) -> float:
    """Phase 7.x followup: numpy SVD on a 1000x1000 matrix. Stubbed
    for v1 — the scaffolding works but the workload is not yet
    wired into the comparative bench harness. Returns the largest
    singular value as a sanity-check scalar.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((1000, 1000))
    s = np.linalg.svd(matrix, compute_uv=False)
    return float(s[0])


def fft_2d_4096_chunk(seed: int) -> float:
    """Phase 7.x followup: numpy 2D FFT on a 4096x4096 array.
    Stubbed for v1 — see ``svd_1000_chunk``. Returns the magnitude
    of the DC bin as a sanity-check scalar.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((4096, 4096))
    spectrum = np.fft.fft2(arr)
    return float(abs(spectrum[0, 0]))


__all__ = [
    "monte_carlo_pi_chunk",
    "aggregate_pi",
    "svd_1000_chunk",
    "fft_2d_4096_chunk",
]


# ────────────────────────────────────────────────────────────────────
# Default sizes for the Monte Carlo workload — kept small enough that
# every comparator finishes in seconds, big enough that submit-drain
# overhead doesn't dominate.
# ────────────────────────────────────────────────────────────────────

DEFAULT_PI_SAMPLES_PER_WORKER: int = 100_000
DEFAULT_PI_NUM_WORKERS: int = 4
DEFAULT_PI_TARGET_TOTAL: int = (
    DEFAULT_PI_SAMPLES_PER_WORKER * DEFAULT_PI_NUM_WORKERS
)


def pi_estimate_is_reasonable(estimate: float, tol: float = 0.02) -> bool:
    """Sanity check: pi estimate within ``tol`` of math.pi.

    With 400k samples, the std dev of the estimate is ~0.0026, so
    ±0.02 is a comfortable 7-sigma tolerance. Used by every
    comparator to assert correctness before publishing timings."""
    return abs(estimate - math.pi) < tol
