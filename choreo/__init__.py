"""
choreo — Static FSM verification for HOC (Phase 4.1)
====================================================

Reads the spec FSMs in ``state_machines/*.py`` and walks the codebase
under ``D:\\HOC`` (or the configured root) looking for ``obj.state = X``
mutations. Reports drift:

- Errors:   undocumented mutations (target state not declared in any FSM)
- Warnings: dead states (declared but no observed target),
            enum extras (enum member not in FSM)
- Info:     declarative-only FSMs (no observed mutations)

Why static instead of runtime
-----------------------------

Phase 4 wired one of five FSMs (CellState) into runtime via
``HoneycombCell.state.setter``; the other four were declarative-only
because their host objects do not have a ``state`` field that can be
intercepted (see ADR-007). Phase 4.1 wires TaskLifecycle similarly
via ``HiveTask.__setattr__`` — but the remaining three (Pheromone,
Succession, Failover) cannot be wired without first introducing
state fields in their host objects, a refactor deferred to Phase 5+.

choreo bridges the gap: the four declarative FSMs become statically
verifiable. The tool runs in CI and fails the build on undocumented
mutations, even though the FSMs do not enforce at runtime.

Entry point
-----------

::

    python -m choreo check                    # walk and report
    python -m choreo check --json             # machine-readable
    python -m choreo check --strict           # warnings → errors
    python -m choreo check --root /other/repo # alt root

See ADR-008 for the architectural decision and ADR-007 for the runtime
counterpart.
"""

from __future__ import annotations

__version__ = "0.1.0"
