"""
Auto-derive an FSM skeleton from observed mutations in a single module.

Given a Python source file and an attribute name (default: ``state``),
walks the file's AST to collect every ``obj.<attr> = EnumName.MEMBER``
(plus ``_set_state`` / ``setattr`` / ``dataclasses.replace``) mutation,
then emits Python source for a ``build_<name>_fsm()`` function that
matches the observed targets.

Limitations
-----------

The walker captures only the *target* state of each mutation. Sources
are wildcarded (``WILDCARD``) in the generated skeleton — the user
must edit the file to specify legal predecessors per edge.

This is a **bootstrapping aid**, not a complete FSM generator. The
expected workflow:

1. ``python -m choreo derive --module swarm.py --field state``
2. Pipe to a new file under ``state_machines/`` (e.g. ``something_fsm.py``).
3. Hand-edit transitions to replace WILDCARD sources with real ones.
4. Add a wire-up site (``__setattr__`` override or runtime check).
5. Run ``python -m choreo check`` to confirm coverage.
"""

from __future__ import annotations

import textwrap
from collections import Counter
from pathlib import Path

from .types import Mutation
from .walker import walk_file


def derive(
    module_path: Path,
    *,
    fsm_name: str | None = None,
    enum_name: str | None = None,
    initial_state: str | None = None,
) -> str:
    """Return Python source code for an FSM skeleton matching the
    observed mutations in ``module_path``.

    ``fsm_name`` defaults to a CamelCase form derived from the module
    stem (e.g. ``cell_fsm`` → ``CellState``-ish; we just title-case the
    stem). ``enum_name`` defaults to the most-frequent enum across
    captured mutations. ``initial_state`` defaults to the first member
    of the most-frequent enum's targets, in source order.
    """
    mutations, _ = walk_file(module_path)

    if not mutations:
        return _empty_template(module_path, fsm_name=fsm_name)

    # Pick the most-mentioned enum if not specified explicitly.
    if enum_name is None:
        counter = Counter(m.enum_name for m in mutations)
        enum_name = counter.most_common(1)[0][0]

    # Restrict to mutations on that enum.
    relevant: list[Mutation] = [m for m in mutations if m.enum_name == enum_name]

    # Ordered, deduplicated list of target states.
    seen: set[str] = set()
    states_in_order: list[str] = []
    for m in relevant:
        if m.member_name not in seen:
            seen.add(m.member_name)
            states_in_order.append(m.member_name)

    # FSM name + builder function name.
    if fsm_name is None:
        fsm_name = _default_fsm_name(enum_name)
    builder_name = _builder_name_from_fsm(fsm_name)

    # Initial state: first observed target unless overridden.
    if initial_state is None:
        initial_state = states_in_order[0]
    elif initial_state not in seen:
        # User provided an initial that no mutation targets — keep it
        # but include it in states. (E.g. PENDING is the dataclass
        # default but no mutation in this module sets it.)
        states_in_order.insert(0, initial_state)

    return _render_template(
        module_path=module_path,
        fsm_name=fsm_name,
        builder_name=builder_name,
        enum_name=enum_name,
        states=states_in_order,
        initial=initial_state,
        mutations=relevant,
    )


def _default_fsm_name(enum_name: str) -> str:
    """``TaskState`` → ``TaskLifecycle``. Conservative: append
    ``Lifecycle`` if the enum name ends with ``State``, else use
    ``<enum>FSM``. Either way the user can rename."""
    if enum_name.endswith("State"):
        return enum_name[: -len("State")] + "Lifecycle"
    return f"{enum_name}FSM"


def _builder_name_from_fsm(fsm_name: str) -> str:
    """``TaskLifecycle`` → ``build_task_fsm`` (matches the
    ``state_machines/<stem>_fsm.py`` + ``build_<stem>_fsm`` convention).

    Splits CamelCase into snake_case, appends ``_fsm``, prefixes with
    ``build_``."""
    snake = _camel_to_snake(fsm_name)
    if snake.endswith("_lifecycle"):
        snake = snake[: -len("_lifecycle")]
    return f"build_{snake}_fsm"


def _camel_to_snake(name: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _render_template(
    *,
    module_path: Path,
    fsm_name: str,
    builder_name: str,
    enum_name: str,
    states: list[str],
    initial: str,
    mutations: list[Mutation],
) -> str:
    """Render the skeleton."""
    prefix = enum_name.upper()  # e.g. "TASKSTATE"

    state_consts = "\n".join(f"{prefix}_{s} = {s!r}" for s in states)
    states_tuple = "ALL_STATES = (\n    " + ",\n    ".join(f"{prefix}_{s}" for s in states) + ",\n)"

    edges = "\n".join(
        f"        # observed at {m.file}:{m.line} ({m.pattern})\n"
        f"        HocTransition(WILDCARD, {prefix}_{m.member_name}),"
        for m in mutations
    )

    initial_const = f"{prefix}_{initial}"

    return textwrap.dedent('''\
        """
        Auto-generated FSM skeleton -- derived from {module_path}.

        Review carefully before committing. The auto-derive walker
        captured only the *target* state of each mutation; sources are
        wildcarded (``WILDCARD``). Replace each ``WILDCARD`` with the
        actual legal source state, or remove the edge if it is not a
        real transition.

        Run ``python -m choreo check`` after editing to confirm the FSM
        covers every observed mutation.
        """

        from __future__ import annotations

        from .base import HocStateMachine, HocTransition, WILDCARD

        # --- States ----------------------------------------------------
        {state_consts}

        {states_tuple}


        def {builder_name}() -> HocStateMachine:
            """TODO: write a docstring."""
            transitions: list[HocTransition] = [
        {edges}
            ]

            return HocStateMachine(
                name={fsm_name!r},
                states=list(ALL_STATES),
                transitions=transitions,
                initial={initial_const},
                enum_name={enum_name!r},
            )
        ''').format(
        module_path=module_path.name,
        state_consts=state_consts,
        states_tuple=states_tuple,
        edges=edges,
        builder_name=builder_name,
        fsm_name=fsm_name,
        enum_name=enum_name,
        initial_const=initial_const,
    )


def _empty_template(module_path: Path, *, fsm_name: str | None) -> str:
    """When no mutations were found, emit a minimal placeholder rather
    than failing. The user gets a clear comment explaining what to do."""
    name = fsm_name or "MyFSM"
    return textwrap.dedent(f'''\
        """
        Auto-generated FSM skeleton -- derived from {module_path.name}.

        No `obj.state = X` (or _set_state / setattr / dataclasses.replace)
        mutations were observed in this module. Either:

        1. The host class does not yet have a ``state`` field -- add one
           and re-run choreo derive.
        2. The mutations use a pattern choreo does not match (e.g.
           computed attribute names). File a bug or extend walker.py.
        3. The FSM is genuinely declarative-only -- write the spec by
           hand.

        Edit the placeholder below to define the FSM by hand.
        """

        from __future__ import annotations

        from .base import HocStateMachine, HocTransition

        # TODO: declare states, transitions, initial.


        def build_{_camel_to_snake(name)}_fsm() -> HocStateMachine:
            return HocStateMachine(
                name={name!r},
                states=[],
                transitions=[],
                initial="",
            )
        ''')
