"""
FSM spec loader.

Imports each ``state_machines/*_fsm.py`` module, finds the
``build_<name>_fsm()`` function, calls it, and extracts the resulting
:class:`HocStateMachine` into a :class:`FsmSpec` (states + transitions).

The loader assumes the project layout used by HOC: ``state_machines`` is
top-level (``D:\\HOC\\state_machines``) and contains one module per FSM
(``cell_fsm.py``, ``task_fsm.py``, etc.), each exporting a function
named ``build_<stem>`` (e.g. ``build_cell_fsm``).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from .types import FsmSpec


def load_specs(root: Path, *, subdir: str = "state_machines") -> list[FsmSpec]:
    """Load every ``<root>/<subdir>/*_fsm.py`` and return a list of
    :class:`FsmSpec`.

    Side effect: ``str(root)`` is prepended to ``sys.path`` so that
    relative imports inside the FSM modules resolve. The path is *not*
    removed afterwards — choreo runs as a one-shot CLI process where
    sys.path pollution is acceptable.
    """
    sm_dir = root / subdir
    if not sm_dir.is_dir():
        return []

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    specs: list[FsmSpec] = []
    for path in sorted(sm_dir.glob("*_fsm.py")):
        stem = path.stem  # e.g. "cell_fsm"
        module_name = f"{subdir}.{stem}"
        builder_name = f"build_{stem}"  # e.g. "build_cell_fsm"

        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError, SyntaxError):
            # Treat unimportable modules as absent. Most likely cause is
            # a circular import, a missing optional dep, or a syntax
            # error in the spec file itself. Other exception types (e.g.
            # NameError from broken module-level code) propagate so the
            # user sees the bug.
            continue

        builder = getattr(module, builder_name, None)
        if builder is None or not callable(builder):
            continue

        try:
            fsm = builder()
        except (TypeError, ValueError, AttributeError):
            # A builder that returns the wrong type or trips on its own
            # arguments is "skip-able"; deeper errors (RuntimeError, etc.)
            # propagate so they are not silently swallowed.
            continue

        spec = _spec_from_fsm(fsm, source_file=str(path.relative_to(root)).replace("\\", "/"))
        if spec is not None:
            specs.append(spec)

    return specs


def _spec_from_fsm(fsm: Any, *, source_file: str) -> FsmSpec | None:
    """Extract a :class:`FsmSpec` from a ``HocStateMachine`` instance.

    Tolerates duck-typed FSM objects (anything with ``name``, ``states``,
    ``transitions``); used both for production HocStateMachine and for
    test fakes.
    """
    try:
        name = str(fsm.name)
        states_set = fsm.states
        transitions = fsm.transitions
    except AttributeError:
        return None

    # Phase 4.2: optional explicit enum binding. Tolerated as missing on
    # FSM objects that predate this attribute (i.e. test fakes).
    enum_name_raw = getattr(fsm, "enum_name", None)
    enum_name = str(enum_name_raw) if enum_name_raw else None

    # Sort states for determinism. ``set`` iteration order is not
    # guaranteed across runs.
    states = tuple(sorted(states_set))

    edges = tuple(sorted((str(src), str(dst), str(trig)) for src, dst, trig in transitions))

    return FsmSpec(
        name=name,
        source_file=source_file,
        states=states,
        transitions=edges,
        enum_name=enum_name,
    )
