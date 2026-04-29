"""
Drift detector — compares observed mutations against declared FSM specs.

For each FSM, attempts to bind it to an enum class found by the walker
(by member-name subset match). Then computes:

- **Undocumented mutations**: a mutation whose enum matches an FSM but
  whose target state is not declared in that FSM. **Error**.
- **Mutation against unbound enum**: a mutation whose enum does not
  match any FSM. **Error** (suggests the enum is missing a spec).
- **Dead states**: states declared in an FSM but no observed mutation
  targets them. **Warning** (the canary that catches B12-ter style
  dead-code in the enum).
- **Enum extras**: enum members not declared in the FSM (the canary
  that catches B12-bis: TaskState.ASSIGNED). **Warning**.
- **Declarative-only**: an FSM with no bound enum and no observed
  mutations. **Info** — fine for HOC's PheromoneDeposit / Succession
  / Failover, where the host object has no state field.

The bind heuristic: an FSM binds to an enum E iff every state in the
FSM is a member of E. If multiple enums satisfy this, the smallest
(by member count) wins; ties broken alphabetically by enum name.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .types import (
    KIND_DEAD_STATE,
    KIND_DECLARATIVE_ONLY,
    KIND_ENUM_EXTRA_STATE,
    KIND_ENUM_UNBOUND,
    KIND_UNDOCUMENTED_MUTATION,
    EnumDecl,
    Finding,
    FsmSpec,
    Mutation,
)


def bind_fsm_to_enum(fsm: FsmSpec, enums: list[EnumDecl]) -> EnumDecl | None:
    """Return the enum bound to ``fsm``, or None.

    Phase 4.2: explicit binding takes priority. If ``fsm.enum_name`` is
    set, look up the enum with that exact name. If found and its members
    are a superset of ``fsm.states``, return it; if found but inconsistent
    return None (the caller will surface this as a diagnostic). If
    ``fsm.enum_name`` is unset, fall back to the member-subset heuristic
    from Phase 4.1.
    """
    if fsm.enum_name is not None:
        # Explicit binding requested. Match by name, validate consistency.
        for enum in enums:
            if enum.name == fsm.enum_name:
                if set(fsm.states).issubset(set(enum.members)):
                    return enum
                # Name match but member mismatch — return None so the
                # caller flags an inconsistency (declarative_only fallback
                # would mask the bug).
                return None
        # Named enum not found anywhere. Fall back to heuristic — the
        # user may have renamed and forgotten; heuristic catches that.
        # Falls through to the heuristic block below.

    fsm_states = set(fsm.states)
    candidates: list[EnumDecl] = []
    for enum in enums:
        if fsm_states.issubset(set(enum.members)):
            candidates.append(enum)

    if not candidates:
        return None

    # Smallest member count wins; ties → alphabetical by name.
    candidates.sort(key=lambda e: (len(e.members), e.name))
    return candidates[0]


def compute_findings(
    specs: Iterable[FsmSpec],
    mutations: Iterable[Mutation],
    enums: Iterable[EnumDecl],
) -> list[Finding]:
    """Produce the full list of findings for a single check run.

    Output is sorted: errors first, then warnings, then info; within
    each severity, alphabetical by FSM name + kind.
    """
    specs_list = list(specs)
    mutations_list = list(mutations)
    enums_list = list(enums)

    # Map enum name → mutations targeting it.
    by_enum: dict[str, list[Mutation]] = defaultdict(list)
    for m in mutations_list:
        by_enum[m.enum_name].append(m)

    findings: list[Finding] = []

    # Track which enums got bound (to detect orphan mutations later).
    bound_enums: set[str] = set()

    for spec in specs_list:
        bound = bind_fsm_to_enum(spec, enums_list)
        if bound is None:
            findings.extend(_findings_for_unbound_spec(spec, by_enum, mutations_list))
            continue

        bound_enums.add(bound.name)
        findings.extend(_findings_for_bound_spec(spec, bound, by_enum.get(bound.name, [])))

    # Mutations against an enum that no spec declares state-set-subset of.
    for enum_name, muts in by_enum.items():
        if enum_name in bound_enums:
            continue
        # The walker is permissive — many enums in the codebase are not
        # state machines (TaskPriority, RoyalCommand, etc.). Only flag
        # an enum as "undocumented" if there is at least one FSM whose
        # name suggests it but didn't bind. For MVP: flag every unbound
        # enum that has mutations as an error.
        for m in muts:
            findings.append(
                Finding(
                    severity="error",
                    fsm="<none>",
                    kind=KIND_UNDOCUMENTED_MUTATION,
                    message=(
                        f"mutation `{m.enum_name}.{m.member_name}` does not match "
                        f"any FSM spec - declare an FSM in state_machines/ or "
                        f"remove the mutation"
                    ),
                    file=m.file,
                    line=m.line,
                )
            )

    return _sorted_findings(findings)


def _findings_for_bound_spec(
    spec: FsmSpec,
    enum: EnumDecl,
    muts: list[Mutation],
) -> list[Finding]:
    findings: list[Finding] = []
    fsm_states = set(spec.states)
    enum_members = set(enum.members)
    observed_targets = {m.member_name for m in muts}

    # 1. Undocumented mutations: target not in fsm.states.
    for m in muts:
        if m.member_name not in fsm_states:
            findings.append(
                Finding(
                    severity="error",
                    fsm=spec.name,
                    kind=KIND_UNDOCUMENTED_MUTATION,
                    message=(
                        f"target `{m.enum_name}.{m.member_name}` not declared "
                        f"in FSM `{spec.name}` - add the state/transition or "
                        f"remove the mutation"
                    ),
                    file=m.file,
                    line=m.line,
                )
            )

    # 2. Dead states: in fsm.states but no observed target.
    dead = sorted(fsm_states - observed_targets)
    if dead:
        findings.append(
            Finding(
                severity="warning",
                fsm=spec.name,
                kind=KIND_DEAD_STATE,
                message=(
                    f"{len(dead)} state(s) declared in FSM but never targeted "
                    f"by an observed mutation: {', '.join(dead)}"
                ),
                file=spec.source_file,
                line=0,
            )
        )

    # 3. Enum extras: in enum.members but not in fsm.states.
    extras = sorted(enum_members - fsm_states)
    if extras:
        findings.append(
            Finding(
                severity="warning",
                fsm=spec.name,
                kind=KIND_ENUM_EXTRA_STATE,
                message=(
                    f"enum `{enum.name}` declares {len(extras)} member(s) "
                    f"not in FSM: {', '.join(extras)} - remove from enum or "
                    f"add transitions to the FSM"
                ),
                file=enum.file,
                line=enum.line,
            )
        )

    return findings


def _findings_for_unbound_spec(
    spec: FsmSpec,
    by_enum: dict[str, list[Mutation]],
    all_mutations: list[Mutation],
) -> list[Finding]:
    """An FSM with no enum whose members are a superset of fsm.states.
    Either the FSM is declarative-only (no host enum) or there's a
    naming mismatch."""
    findings: list[Finding] = []

    # Heuristic: if any mutation's enum_name matches the FSM name (case-
    # insensitive substring), this is likely a misalignment, not
    # declarative-only.
    name_lower = spec.name.lower()
    suspected: list[Mutation] = []
    for muts in by_enum.values():
        for m in muts:
            if m.enum_name.lower() in name_lower or name_lower in m.enum_name.lower():
                suspected.append(m)

    if suspected:
        for m in suspected[:3]:  # cap noise
            findings.append(
                Finding(
                    severity="warning",
                    fsm=spec.name,
                    kind=KIND_ENUM_UNBOUND,
                    message=(
                        f"FSM `{spec.name}` could not bind to any enum, but "
                        f"mutation `{m.enum_name}.{m.member_name}` looks "
                        f"related - declare matching state names in the FSM"
                    ),
                    file=m.file,
                    line=m.line,
                )
            )
        return findings

    # No matching mutations — declarative-only.
    findings.append(
        Finding(
            severity="info",
            fsm=spec.name,
            kind=KIND_DECLARATIVE_ONLY,
            message=(
                f"FSM `{spec.name}` is declarative-only: no enum class "
                f"declares its state set as members, and no observed "
                f"mutation targets any of its states. The FSM serves as "
                f"documentation + property-test target only"
            ),
            file=spec.source_file,
            line=0,
        )
    )
    return findings


def _sorted_findings(findings: list[Finding]) -> list[Finding]:
    """Stable sort: severity (error/warning/info) → fsm → kind → file → line."""
    severity_rank = {"error": 0, "warning": 1, "info": 2}
    return sorted(
        findings,
        key=lambda f: (
            severity_rank.get(f.severity, 3),
            f.fsm,
            f.kind,
            f.file,
            f.line,
        ),
    )
