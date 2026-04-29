"""
Frozen dataclasses shared across the choreo MVP.

Every type is hashable and ordered for deterministic test output.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Mutation:
    """A single ``obj.state = EnumName.MEMBER`` (or ``obj._set_state(...)``)
    assignment found in code.

    The walker captures the *target* state (RHS) only. The current state
    of ``obj`` at the call-site is not inferred — that would require
    control-flow analysis, out of MVP scope.
    """

    file: str
    line: int
    enum_name: str
    member_name: str
    pattern: str = "assign"  # "assign" | "_set_state"


@dataclass(frozen=True, order=True)
class EnumDecl:
    """A class statement that subclasses ``Enum`` (direct or via attribute
    access — ``enum.Enum``). Used to discover the universe of states
    declared by enums and bind them to FSMs by member-subset matching."""

    file: str
    line: int
    name: str
    members: tuple[str, ...]


@dataclass(frozen=True, order=True)
class FsmSpec:
    """An FSM declared in ``state_machines/`` after import + introspection.

    ``transitions`` is the full edge set, including wildcards (source
    may be the literal ``"*"`` for wildcard sources).

    ``enum_name`` is set when the FSM was constructed with
    ``HocStateMachine(..., enum_name="X")``. When present, choreo binds
    to the matching :class:`EnumDecl` directly (overriding the
    member-subset heuristic).
    """

    name: str
    source_file: str
    states: tuple[str, ...]
    transitions: tuple[tuple[str, str, str], ...]  # (source, dest, trigger)
    enum_name: str | None = None


@dataclass(frozen=True, order=True)
class Finding:
    """A single drift report item."""

    severity: str  # "error" | "warning" | "info"
    fsm: str
    kind: str  # See KIND_* below.
    message: str
    file: str = ""
    line: int = 0


# ─── Finding kinds ─────────────────────────────────────────────────────────

KIND_UNDOCUMENTED_MUTATION = "undocumented_mutation"
"""Mutation observed whose target is not in any FSM's state set, or whose
enum is not modeled by any FSM. Always an error."""

KIND_DEAD_STATE = "dead_state"
"""A state declared in an FSM (and bound enum, if any) that no observed
mutation targets. Warning."""

KIND_ENUM_EXTRA_STATE = "enum_extra_state"
"""An enum member that does not appear in the FSM's declared states —
e.g. ``TaskState.ASSIGNED`` while the FSM only has 5 states. Warning."""

KIND_DECLARATIVE_ONLY = "declarative_only"
"""An FSM with no bound enum and no mutations whose enum matches its
state set. Info — typical for FSMs whose host object has no state
field (Pheromone, Succession, Failover in HOC). Never fails CI."""

KIND_ENUM_UNBOUND = "enum_unbound"
"""An FSM with no enum that matches its state set by subset. Info if
the FSM is otherwise unused (declarative-only); warning if mutations
exist whose enum is unrelated."""
