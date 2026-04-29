#!/usr/bin/env python3
"""
scripts/generate_state_machines_md.py
=====================================

Auto-generates ``docs/state-machines.md`` by exporting Mermaid diagrams
from each registered HOC state machine. Used in CI to detect drift
between FSM definitions and the doc.

Usage::

    python scripts/generate_state_machines_md.py            # writes file
    python scripts/generate_state_machines_md.py --stdout   # prints to stdout
    python scripts/generate_state_machines_md.py --check    # exit 1 if drift

The output is **deterministic** for a given FSM spec. ``test_mermaid_export.py``
asserts byte-equality across runs.

Adding a new FSM
----------------

1. Implement a builder function in ``state_machines/<your>_fsm.py`` that
   returns a fresh :class:`HocStateMachine` (see ``state_machines/cell_fsm.py``
   for the pattern).
2. Register the builder in ``_fsm_registry`` below with the desired
   display order.
3. Run ``python scripts/generate_state_machines_md.py`` to regenerate
   ``docs/state-machines.md``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

# Make the repo root importable when this script is invoked from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from state_machines.base import HocStateMachine  # noqa: E402

FSMBuilder = Callable[[], HocStateMachine]


def _fsm_registry() -> list[tuple[str, FSMBuilder]]:
    """
    Ordered list of (display_name, builder) for all FSMs.

    The display order matches Phase 4 implementation order so the doc
    reads as a natural lifecycle from the simplest (CellState) to the
    most complex (FailoverFlow).

    Builders are imported lazily and skipped if the module does not exist
    yet — this lets the script run during incremental Phase 4 wire-up,
    each new FSM appearing as soon as its module is added.
    """
    registry: list[tuple[str, FSMBuilder]] = []

    def _register(display: str, module: str, attr: str) -> None:
        try:
            mod = __import__(f"state_machines.{module}", fromlist=[attr])
            builder: FSMBuilder = getattr(mod, attr)
            registry.append((display, builder))
        except ImportError:
            # FSM module not yet wired up — skip silently.
            pass

    _register("CellState", "cell_fsm", "build_cell_fsm")
    _register("PheromoneDeposit", "pheromone_fsm", "build_pheromone_fsm")
    _register("TaskLifecycle", "task_fsm", "build_task_fsm")
    _register("QueenSuccession", "succession_fsm", "build_succession_fsm")
    _register("FailoverFlow", "failover_fsm", "build_failover_fsm")
    return registry


def render(fsms: list[tuple[str, HocStateMachine]]) -> str:
    """Render the full Markdown document. Trailing newline included."""
    lines: list[str] = []
    lines.append("# HOC State Machines")
    lines.append("")
    lines.append(
        "> **Auto-generated** by `scripts/generate_state_machines_md.py`. " "Do not edit by hand."
    )
    lines.append(
        "> Regenerate after touching any FSM in `state_machines/`. CI fails "
        "if this file drifts from the FSM specs."
    )
    lines.append("")
    lines.append(
        "Each diagram below is exported from a `HocStateMachine` instance "
        "via `to_mermaid()`. Triggers shown in the labels are the synthetic "
        "names auto-generated from `<source>__to__<dest>`; explicit triggers "
        "appear when an FSM module declares them."
    )
    lines.append("")

    if not fsms:
        lines.append("_No state machines registered yet._")
        lines.append("")
        return "\n".join(lines) + "\n"

    # Table of contents
    lines.append("## Index")
    lines.append("")
    for name, _ in fsms:
        anchor = name.lower().replace(" ", "-")
        lines.append(f"- [{name}](#{anchor})")
    lines.append("")

    for name, fsm in fsms:
        lines.append(f"## {name}")
        lines.append("")
        lines.append(
            f"- **States** ({len(fsm.states)}): " + ", ".join(f"`{s}`" for s in sorted(fsm.states))
        )
        lines.append(f"- **Initial state**: `{fsm.initial}`")
        lines.append("")
        lines.append("```mermaid")
        lines.append(fsm.to_mermaid())
        lines.append("```")
        lines.append("")

    return "\n".join(lines) + "\n"


def _build_all(registry: list[tuple[str, FSMBuilder]]) -> list[tuple[str, HocStateMachine]]:
    """Instantiate every registered FSM."""
    return [(name, builder()) for name, builder in registry]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write to stdout instead of docs/state-machines.md",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare regenerated content against on-disk docs/state-machines.md; "
        "exit 1 on drift. Used by CI.",
    )
    args = parser.parse_args(argv)

    fsms = _build_all(_fsm_registry())
    out = render(fsms)
    target = _REPO_ROOT / "docs" / "state-machines.md"

    if args.stdout:
        sys.stdout.write(out)
        return 0

    if args.check:
        if not target.exists():
            print(f"FAIL: {target} does not exist", file=sys.stderr)
            return 1
        on_disk = target.read_text(encoding="utf-8")
        if on_disk != out:
            print(
                f"FAIL: {target} is stale. "
                "Regenerate with: python scripts/generate_state_machines_md.py",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {target} matches FSM specs ({len(fsms)} FSMs)")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(out, encoding="utf-8")
    print(f"Wrote {target} ({len(fsms)} FSMs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
