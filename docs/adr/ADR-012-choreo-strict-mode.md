# ADR-012: Flip ``choreo check`` to ``--strict`` in CI

- **Status**: Accepted
- **Date**: 2026-04-26
- **Phase**: Phase 5.6

## Context

``choreo`` (introduced in Phase 4.1, see ADR-008) is a static AST
checker that compares observed ``obj.state = ENUM.MEMBER`` mutations
against declared FSM specs in ``state_machines/``. Its severities:

| Severity | Findings produced when … |
|----------|--------------------------|
| ``error`` | Mutation targets a state not declared in the spec |
| ``warning`` | Spec declares a state with no observed mutation (``dead_state``) or an enum member not in the spec (``enum_extra_state``) |
| ``info`` | FSM is declarative-only (no enum binding, no mutations) |

Phase 4.1 wired the CI job in non-strict mode: errors break the build,
warnings + info are reported but allowed. The intent — documented in
ADR-008 — was "flip ``--strict`` once the known dead-state +
enum-extras backlog is closed".

That backlog cleared over Phases 4.3 (``B12-bis`` + ``B12-ter``
removed/reserved) and 5.1–5.2 (every reserved member wired,
PheromoneDeposit / FailoverFlow / QueenSuccession upgraded from
declarative to wired). The current ``choreo check`` against HOC main
produces:

```
choreo: no drift detected.
Summary: 0 errors, 0 warnings, 0 infos
```

This ADR records the decision to commit the flip.

## Decision

**Flip the CI invocation in ``.github/workflows/lint.yml`` from
``python -m choreo check`` to ``python -m choreo check --strict``.**
The local invocation in ``CONTRIBUTING.md`` follows.

``--strict`` raises every warning and info to error severity. Any
future PR that introduces:

- a dead state (declares an FSM state with no targeting mutation),
- an enum-extra (declares an enum member that no FSM models),
- a declarative-only FSM (declares a spec with no host enum + no
  observed mutations),

will break CI on the offending PR. The author has to either wire the
state, remove the dead member, or document a deliberate gap (and
loosen the spec) before merging.

## Alternatives considered

### Stay non-strict, surface warnings as PR comments

A pre-merge comment instead of a hard fail. Rejected because:

- The commitment in ADR-008 + ADR-010 was explicit: ``--strict`` once
  the known backlog clears. Phase 5 closes that backlog. Soft mode
  here would be moving the goalpost.
- Comments add reviewer friction but no consequence — easy to ignore.
  CI enforcement keeps the FSM-spec contract honest.

### Strict only on ``main``, soft on PR branches

PR feedback would still be advisory; the ``main`` gate would reject
already-merged regressions. Rejected because the regression is
already in main at that point — too late to be useful.

### Add a per-PR opt-out for "I know what I'm doing" cases

A label or magic string in the commit message that bypasses
``--strict``. Rejected because the existing alternative is cleaner:
update the FSM spec in the same PR. ``--strict`` only fires on
findings ``choreo`` thinks are wrong; the author can always change
``state_machines/*_fsm.py`` to declare the new pattern explicitly.

## Consequences

### Easier

- **Future regressions surface on the PR that introduces them.** An
  added enum member without a wire-up plan, a new FSM that ships
  declarative-only because the author "didn't get to wiring yet",
  a refactor that drops the last call-site for an existing state —
  all caught at PR time.
- **The Phase 5 closure can claim "0/0/0 enforced" without weasel
  words.** Strict mode is a stronger guarantee than "currently 0/0/0
  but anyone could add to the backlog tomorrow".
- **ADR-008 + ADR-010 follow-throughs land.** Both ADRs left the
  ``--strict`` flip as future work; this closes the loop.

### Harder

- **PRs that intentionally add a declarative-only FSM cannot merge
  without updating the spec.** That is the explicit intent — the
  rule is the rule. The author has two paths forward:
  1. Wire the FSM at the same time (preferred).
  2. Delete or downgrade the FSM declaration if it turned out to be
     premature.
- **A future ``choreo`` upgrade that tightens the rules can break
  builds.** Mitigated by pinning ``choreo`` is a subpackage of HOC
  itself (no external pin); changes ship with HOC and are reviewable
  in the same PR. Nothing to lock down externally.
- **Code reviewers must understand the failure mode.** A failing
  ``choreo-static-check`` job needs an action: wire, delete, or
  loosen. The error messages name the kind (``dead_state``,
  ``enum_extra_state``, ``declarative_only``); contributors who
  haven't seen the tool yet can read ADR-008 + the docstring of the
  failing spec.

### Risk / follow-up

- **If the rule generates excessive friction in practice** — e.g. a
  refactor naturally introduces transient declarative-only FSMs that
  authors plan to wire in a follow-up PR — we revert this commit,
  re-document the condition, and keep ``choreo`` in non-strict mode
  with a tracked backlog. The current pure-static-only approach has
  no escape hatch by design; if that proves wrong, the next ADR can
  add one (a ``# choreo: declarative-only-by-design`` comment, for
  instance).
- **Walker pattern coverage limitation.** ``choreo``'s walker only
  recognises the literal attribute name ``state`` (and the
  ``_set_state`` method, ``setattr(obj, "state", ...)``,
  ``dataclasses.replace(obj, state=...)`` patterns). Other
  conventions (e.g. ``self._phase = X``, dict subscript assignment)
  are invisible. Phase 5.2c worked around this by introducing a
  ``_FailoverCellState`` wrapper with a ``state`` attribute (Phase
  5.2b did the same with ``_SuccessionState``). If a future FSM
  cannot adapt to one of these patterns, extending the walker (the
  natural place is ``choreo/walker.py:_Visitor.visit_Assign``) is
  the path forward — keep the strict-mode contract intact by
  expanding the recognised patterns rather than carving out
  exceptions.

## References

- ``.github/workflows/lint.yml`` — ``choreo-static-check`` job, now
  ``python -m choreo check --strict``.
- ``CONTRIBUTING.md`` — local invocation updated to match.
- ADR-008 — ``choreo`` introduction; this ADR closes its
  ``--strict`` follow-up.
- ADR-009 — ``choreo`` v0.2 (reified transitions + auto-derive); the
  enum_name binding it added powers the explicit FSM↔enum bindings
  Phase 5.2 used.
- ADR-010 — Dead enum-member cleanup; the per-member discrimination
  there is what Phase 5.1 + 5.2 wired.
- ADR-011 — Observability stack; complementary scope (logs vs static
  contract).
- Phase 5 closure — ``snapshot/PHASE_05_CLOSURE.md``.
