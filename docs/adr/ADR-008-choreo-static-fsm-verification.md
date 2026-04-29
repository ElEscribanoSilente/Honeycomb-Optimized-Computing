# ADR-008: choreo — static FSM verification, complementary to runtime wire-up

- **Status**: Accepted
- **Date**: 2026-04-24
- **Phase**: Phase 4.1

## Context

Phase 4 (ADR-007) shipped 5 FSMs in `state_machines/`. Of these, 1 was
**wired** at runtime (CellState, via `HoneycombCell.state.setter`) and
4 were **declarative-only** (TaskLifecycle, PheromoneDeposit,
QueenSuccession, FailoverFlow). Phase 4.1 wired TaskLifecycle additionally
via `HiveTask.__setattr__`, leaving 3 still declarative.

Two structural reasons block runtime wire-up of the remaining three:

1. **No host state field.** PheromoneDeposit's lifecycle is derived from
   `intensity + age`, not stored as a `state` attribute. QueenSuccession's
   six "phases" are positions inside `_conduct_election`, not values
   between calls. CellFailover tracks failed cells in a `set`, never a
   per-cell phase. To wire any of these, we would first need to introduce
   a state field — a refactor of the host class, deferred to Phase 5+.

2. **Hot-path performance.** PheromoneDeposit allocates ~90k objects per
   trail at default caps. Per-instance FSM allocation or a global
   validator under a lock would breach the `<5 %` overhead budget by
   orders of magnitude.

The Phase 4 closure documented two real but unresolved bugs that the
declarative FSMs surface but cannot enforce:

- **B12-bis**: `TaskState.ASSIGNED` declared in the enum at swarm.py:90
  but never assigned by any production call-site. Phase 4.1's wire-up
  catches this at runtime via `IllegalStateTransition`, but only on
  exercise of the offending path.
- **B12-ter**: `CellState.{SPAWNING, MIGRATING, SEALED, OVERLOADED}`
  declared in the enum at core/cells_base.py but never assigned. The
  Phase 4 wire-up catches them at runtime, but the dead enum members
  remain visible to readers and to tooling.

Both bugs are detectable without runtime execution: walk the source,
find every `obj.state = X` assignment, compare against the FSM specs
and the enum members. We did not have such a tool.

## Decision

Build `choreo`, a small static analysis tool that lives in the HOC
repository at `choreo/` (parallel to `state_machines/`). It implements
the AST-walk + spec-load + drift-detect approach, runs from the CLI
(`python -m choreo check`), and is enforced by a new CI job
`choreo-static-check` in `lint.yml`.

### Output model

choreo produces three severities:

- **errors**: undocumented mutations (a `obj.state = X` whose target X
  is not declared in the matching FSM, or whose enum is not modeled by
  any FSM). Errors fail the CI build.
- **warnings**: dead states (declared in FSM but never targeted) and
  enum extras (declared in enum but not in FSM). Warnings do not fail
  CI by default; `--strict` mode promotes them to errors.
- **info**: declarative-only FSMs (no bound enum, no observed
  mutations). Never fail CI.

This split lets Phase 4.1 ship with B12-bis/B12-ter as known warnings
without breaking CI, while still catching genuinely new issues
(undocumented mutations introduced by future PRs).

### Architecture

choreo is intentionally simple — five files, ~600 LOC total:

- `choreo/walker.py` — `ast.NodeVisitor` matching two literal patterns
  (`obj.state = ENUM.MEMBER`, `obj._set_state(ENUM.MEMBER)`) and one
  class shape (`class X(Enum)` or `class X(enum.Enum)`).
- `choreo/spec.py` — imports each `state_machines/*_fsm.py`, calls
  `build_<stem>()`, extracts `(states, transitions)` from the resulting
  `HocStateMachine`. Tolerates duck-typed FSM objects.
- `choreo/diff.py` — binds each FSM to an enum by member-subset match
  (smallest enum wins), computes the four kinds of findings.
- `choreo/cli.py` — argparse + human/JSON output + exit codes.
- `choreo/types.py` — frozen dataclasses for findings/mutations/specs.

The walker performs **no type resolution**: it matches purely on
syntactic shape. False negatives are accepted — `setattr(obj, "state",
X)` and `obj.state = func()` are not detected — in exchange for keeping
the implementation small, fast (~0.5s on the HOC repo), and free of
external dependencies.

### Why static + runtime, not either alone

CellState and TaskLifecycle are wired at runtime. choreo runs at lint
time. Neither subsumes the other:

- **Runtime** catches mutations that static analysis misses (dynamic
  attribute access, late-bound calls, computed enum members).
- **Static** catches mutations whose runtime path is not exercised by
  tests (e.g. a cold-path PR that adds `task.state = TaskState.ASSIGNED`
  in a branch that no test covers).

For the 3 FSMs that cannot be wired today (Pheromone, Succession,
Failover), choreo is the only enforcement layer. Once Phase 5+ introduces
state fields and wires them, choreo continues to provide compile-time
detection — the layers compose.

## Alternatives considered

### Use a third-party static analyzer (mypy plugin, semgrep, ruff plugin)

mypy has no FSM-aware checks; building one as a plugin requires deep
integration with the type system that does not pay back for our scale
(5 FSMs, ~30 call-sites). semgrep would work for pattern matching but
adds an external CI dependency and requires writing rules in semgrep's
DSL — opaque and one-off. ruff is a linter for style/safety, not for
domain-specific cross-file consistency. **Rejected** in favour of an
in-repo tool we control.

### Use the runtime FSMs to derive observed transitions (test-time tracing)

Instrument the wire-up sites to log every `transition_to`, run the
test suite, accumulate the observed transitions, compare to the FSM
spec. This works for the 2 wired FSMs but cannot help the 3 unwired
ones (no instrumentation point). It also requires the test suite to
exercise every transition — which it does not, by design (force-
completion tests skip lifecycle states). **Rejected** as not
generalizable.

### Auto-derive the FSM from the code (no spec at all)

The walker could simply emit `class X(Enum)` + `obj.state = X.M` data
and call that the FSM. But the value of an FSM is the **declared**
graph being more constrained than the observed code — the FSM rejects
illegal transitions the code might otherwise be tempted to write. An
auto-derived FSM has no constraints to enforce. Phase 4.2 plans to
add `choreo derive --module X --field Y` as a *bootstrapping* aid, not
a replacement for the spec. **Rejected** for Phase 4.1.

### Wait for full wire-up of all 5 FSMs (Phase 5+) and skip choreo

Possible, but loses the catch-net for B12-bis/B12-ter and any future
dead-state bugs introduced before the wire-up lands. Static analysis is
proportionally cheap to build (~600 LOC, 1 day's work) and starts paying
dividends immediately. Postponing means more drift accumulates between
the spec and the code. **Rejected** as too passive.

## Consequences

### Easier

- **B12-bis and B12-ter are visible at lint time.** A future contributor
  scanning CI logs sees the warnings every run; they cannot creep
  further into the codebase without notice.
- **CI catches new undocumented mutations.** A PR that introduces
  `task.state = TaskState.ASSIGNED` (or any new enum member that the
  FSM has not declared) fails the choreo job before merge. Matches the
  same protection runtime wire-up gives, but at the build level.
- **Composes with declarative-only FSMs.** Pheromone/Succession/Failover
  are documented and tested but not enforced at runtime. choreo doesn't
  need a host state field to verify them — once a future PR introduces
  `pheromone.state = X` (e.g. when Phase 5 adds the field), choreo
  starts checking it without any code change.
- **Independent of tramoya.** choreo only needs `HocStateMachine.name`,
  `.states`, and `.transitions`. If we ever swap the FSM library
  (ADR-007's boundary pattern), choreo continues to work as long as the
  wrapper preserves those three properties.

### Harder

- **Two more files to touch when adding a new FSM.**
  `state_machines/X_fsm.py` AND optionally an entry in the matching
  enum class. choreo will reject the FSM at lint time if its states
  don't subset any enum's members. CONTRIBUTING.md needs an updated
  checklist (Phase 4.1 closure adds it).
- **Heuristic enum binding.** If two enums have member sets that
  superset the same FSM, the smallest-by-member-count tiebreaks
  alphabetically. For HOC this is unambiguous (CellState ↔ cell_fsm,
  TaskState ↔ task_fsm). For a future FSM with two compatible enums,
  the contributor must rename one. Phase 4.2 plans to allow opt-in
  metadata in `build_X_fsm()` (`enum=...`) to disambiguate explicitly.
- **No source-state inference.** choreo records target states only; it
  cannot detect "this PR introduced a `RUNNING → PENDING` direct, which
  the FSM does not allow". Runtime wire-up catches this; choreo does
  not. Adding control-flow analysis is research-grade work, deferred
  beyond Phase 4.x.

### Risk / follow-up

- **Pattern coverage.** choreo's two literal patterns cover every
  current HOC call-site, but a refactor that introduces
  `setattr(obj, "state", X)` or `dataclasses.replace(obj, state=X)`
  would silently bypass choreo. Mitigated by the runtime wire-up
  (CellState, TaskLifecycle) for the wired FSMs, but a blind spot for
  the others until Phase 4.2 extends the walker.
- **`--strict` mode parking.** The CI job runs without `--strict`,
  which means warnings do not fail. After Phase 5+ resolves
  B12-bis/B12-ter (either by removing the dead enum members or by
  wiring up the missing call-sites), the CI step should be flipped
  to `--strict` to lock the gain.
- **Independent extraction.** choreo is small enough to extract to a
  standalone PyPI package later. The package layout (`choreo/`
  parallel to `state_machines/`, no HOC-specific imports inside) was
  chosen with that path in mind. Phase 4.2 may publish if the tool
  proves itself.

## References

- `choreo/` — implementation.
- `tests/test_choreo.py` — 32 tests including HOC integration smoke.
- `.github/workflows/lint.yml` — `choreo-static-check` job.
- ADR-007 — runtime tramoya wire-up; choreo is the static counterpart.
- Phase 4.1 closure — application to HOC produces exactly the
  expected report (0 errors, 2 warnings for B12-bis/ter, 3 info for
  the still-declarative FSMs).
- B12-bis: `swarm.py:90` — `TaskState.ASSIGNED` dead enum member.
- B12-ter: `core/cells_base.py:51` — 4 dead `CellState` members.
