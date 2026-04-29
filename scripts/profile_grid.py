"""Phase 7.8 — wrapper for ``py-spy record`` on a HOC grid workload.

Used as the canonical "profile a tick loop" recipe by ``docs/perf/
profiling.md``. Spawns a small synthetic workload (configurable
radius + tick count) so the resulting flame graph is reproducible
across machines and reflects the Phase 7+ async tick path.

Usage::

    python scripts/profile_grid.py
    python scripts/profile_grid.py --radius 3 --ticks 200
    python scripts/profile_grid.py --output /tmp/flame.svg

The script doesn't run py-spy itself — instead it prints the exact
``py-spy record`` command to invoke. This avoids the privilege
elevation dance (py-spy needs ptrace on Linux; the wrapper would
need to be re-invoked with sudo). The user copies the printed
command and runs it.

For a self-contained profile-and-render flow without py-spy, pass
``--inproc``: the workload runs directly in-process and prints
``cProfile`` stats. Less detailed than a flame graph but doesn't
require external tooling.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import pstats
import sys
import time
from pathlib import Path

# Make the local hoc package importable.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


async def _run_workload(radius: int, ticks: int) -> dict[str, float]:
    """Synthetic load: build a grid, fire ``ticks`` async ticks
    sequentially. Returns timing summary."""
    from hoc.core import HoneycombConfig, HoneycombGrid

    cfg = HoneycombConfig(radius=radius)
    grid = HoneycombGrid(cfg)

    # Warm one tick so the FSM / metrics caches are primed before we
    # measure.
    await grid.tick()

    start = time.perf_counter()
    for _ in range(ticks):
        await grid.tick()
    elapsed = time.perf_counter() - start

    return {
        "radius": float(radius),
        "ticks": float(ticks),
        "elapsed_s": elapsed,
        "ticks_per_second": ticks / elapsed if elapsed > 0 else 0.0,
        "us_per_tick": elapsed * 1_000_000 / ticks if ticks > 0 else 0.0,
    }


def _print_pyspy_command(radius: int, ticks: int, output: Path) -> None:
    """Print the ``py-spy record`` command for the given workload.
    The user copies and runs it themselves."""
    cmd = [
        "py-spy",
        "record",
        "--output",
        str(output),
        "--rate",
        "250",
        "--threads",
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        "--inproc",
        "--radius",
        str(radius),
        "--ticks",
        str(ticks),
    ]
    print("# Phase 7.8 — py-spy command for the configured workload:")
    print("# (copy + paste, then open the resulting SVG in a browser)")
    print()
    print(" ".join(cmd))
    print()
    print("# Notes:")
    print("# - On Linux you may need ``setcap cap_sys_ptrace=eip $(which py-spy)``")
    print("#   or run as root.")
    print("# - The workload itself is the same one that runs under --inproc;")
    print("#   the only difference is py-spy attaches as a sampling profiler.")


def _run_inproc(radius: int, ticks: int) -> None:
    """Run the workload directly + dump cProfile stats to stdout.
    Used both by ``--inproc`` and as the workload that py-spy
    attaches to (it's sys.executable scripts/profile_grid.py
    --inproc with the same args)."""
    pr = cProfile.Profile()
    pr.enable()
    summary = asyncio.run(_run_workload(radius, ticks))
    pr.disable()

    print("=" * 60)
    print("Workload summary")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:20s} = {v:.3f}")
    print()
    print("=" * 60)
    print("cProfile top-30 by cumulative time")
    print("=" * 60)
    stats = pstats.Stats(pr)
    stats.sort_stats("cumulative").print_stats(30)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 7.8 — profile a HOC grid tick loop.")
    parser.add_argument(
        "--radius",
        type=int,
        default=3,
        help="Grid radius. Default 3 (matches Phase 7.5 brief).",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=200,
        help="Number of ticks to drive. Default 200.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("snapshot/flame.svg"),
        help="py-spy output path. Default snapshot/flame.svg.",
    )
    parser.add_argument(
        "--inproc",
        action="store_true",
        help=(
            "Run the workload in-process and dump cProfile stats. "
            "Used internally by py-spy's invocation."
        ),
    )
    args = parser.parse_args()

    if args.inproc:
        _run_inproc(args.radius, args.ticks)
        return 0

    _print_pyspy_command(args.radius, args.ticks, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
