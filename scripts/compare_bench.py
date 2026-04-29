#!/usr/bin/env python3
"""Compare two pytest-benchmark JSON snapshots and report % diff per benchmark.

Usage:
    python scripts/compare_bench.py snapshot/bench_baseline.json snapshot/bench_phase05.json

Exit code 0 if no benchmark regressed beyond ``--threshold`` (default 10%);
exit code 1 if any did. Designed for the ``bench-regression`` CI job in
Phase 5.

The JSON shape expected is the **condensed** one we ship in
``snapshot/bench_baseline.json``: ``{"benchmarks": [{"name", "stats":
{"mean", "min", "max", "stddev", "median", "rounds"}}]}``. The raw
pytest-benchmark output (with full per-round samples, ~35 MB) is
condensed to summary stats before commit; the script in
``scripts/condense_bench.py`` does the conversion if a fresh raw JSON
needs to be archived.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, dict[str, Any]]:
    """Return ``{benchmark_name: stats_dict}`` from a condensed snapshot."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {b["name"]: b["stats"] for b in data.get("benchmarks", [])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline snapshot JSON")
    parser.add_argument("current", type=Path, help="Current snapshot JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Regression threshold (percent). Default: 10.0",
    )
    parser.add_argument(
        "--metric",
        choices=("mean", "median", "min"),
        default="mean",
        help="Which stat to compare. Default: mean",
    )
    args = parser.parse_args(argv)

    baseline = _load(args.baseline)
    current = _load(args.current)

    common = sorted(set(baseline) & set(current))
    only_baseline = sorted(set(baseline) - set(current))
    only_current = sorted(set(current) - set(baseline))

    print(f"{'Benchmark':<40} {'Baseline (ms)':>14} {'Current (ms)':>14} {'% diff':>10}")
    print("-" * 80)

    regressions: list[tuple[str, float]] = []
    max_regression = float("-inf")
    for name in common:
        b_val = baseline[name][args.metric] * 1000
        c_val = current[name][args.metric] * 1000
        diff_pct = (c_val - b_val) / b_val * 100 if b_val else 0.0
        marker = " (REGRESSION)" if diff_pct > args.threshold else ""
        print(f"{name:<40} {b_val:>14.4f} {c_val:>14.4f} {diff_pct:>+9.2f}%{marker}")
        if diff_pct > max_regression:
            max_regression = diff_pct
        if diff_pct > args.threshold:
            regressions.append((name, diff_pct))

    if only_baseline:
        print(f"\nOnly in baseline (not in current): {only_baseline}")
    if only_current:
        print(f"Only in current (new since baseline): {only_current}")

    if regressions:
        print(f"\nFAIL: {len(regressions)} benchmark(s) regressed > {args.threshold:.1f}%:")
        for name, diff in regressions:
            print(f"  {name}: +{diff:.2f}%")
        return 1

    print(f"\nOK: max regression {max_regression:+.2f}% (threshold {args.threshold:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
