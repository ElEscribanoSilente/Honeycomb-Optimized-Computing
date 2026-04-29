# ADR-007: Tramoya as the FSM engine, with one wired + four declarative state machines

- **Status**: Accepted
- **Date**: 2026-04-24
- **Phase**: Phase 4

## Context

Phase 4's brief proposed five state machines for HOC's lifecycle-bearing
types: `CellState`, `TaskLifecycle`, `QueenSuccession`, `PheromoneDeposit`,
and `FailoverFlow`. The brief specified `tramoya` as the FSM library
(MIT, ~300 LOC, zero deps, guards/hooks/undo + Mermaid export — and
authored by Raul, the HOC project owner).

Three sub-decisions had to be resolved:

1. **How tightly to couple HOC to the tramoya API.** The brief itself
   noted that the same situation arose with `mscs` in Phase 2 (real API
   diverged from what the brief described).
2. **Whether to *wire* every FSM into production code or to ship some as
   *declarative-only* documentation.** Wiring requires per-instance FSMs
   and changes to mutation paths; for hot-path code, that may exceed the
   Phase 4 `<5 %` overhead budget.
3. **How to keep `state_machines/` under strict mypy** when it is
   reachable from two import paths (`state_machines.*` via `sys.path`
   and `HOC.state_machines.*` via cwd-based package inference) and
   `mypy .` errors with "Source file found twice".

## Decision

### 1. Boundary pattern (same as Phase 2 with `mscs`)

`tramoya` is imported **only** in `state_machines/base.py`. Downstream
modules (the four wired/declarative FSM files, the wire-up site in
`core/cells_base.py`, the Mermaid export script, the tests) import
`HocStateMachine` / `HocTransition` / `IllegalStateTransition` /
`WILDCARD` from `state_machines.base`. Future swaps of the FSM engine
touch one file.

The wrapper exposes:

- A **destination-driven** API (`transition_to(target)`) that preserves
  the pre-Phase-4 contract `obj.state = X`. Tramoya is event-driven —
  callers fire named triggers and the engine resolves the destination —
  but production HOC code historically did `cell.state = NEW`. The
  wrapper builds a `dest -> [(source, trigger)]` index at construction
  time and `transition_to` walks it to find the right trigger.
- A **trigger-driven** passthrough (`trigger(name)`) for callers that
  prefer events (observability hooks, tests, future REPL inspection).
- A single **`IllegalStateTransition`** exception that wraps tramoya's
  `InvalidTransition` and `GuardRejected`. `reason` is a small
  string-enum (`"no_edge" | "guard_rejected" | "unknown_state" |
  "empty_history"`) so callers can branch without parsing messages.

### 2. One wired, four declarative

The five FSMs are not equal in the cost/value of wiring:

| FSM | Decision | Rationale |
|-----|----------|-----------|
| **CellState** | **Wired** | One mutation point in `_set_state`; existing `state.setter` already centralised. Marginal cost is one lookup per transition; gains a runtime detector for un-documented states (B12-ter dead states are caught immediately). |
| **PheromoneDeposit** | Declarative | ~90k objects per trail at default caps. Per-instance FSM allocation or a global validator with a lock would exceed the `<5 %` overhead budget by orders of magnitude. No `state` field exists — phases derive from intensity + age. |
| **TaskLifecycle** | Declarative | `HiveTask.state` mutated from ~15 call-sites in `swarm.py` plus many test fixtures. Wiring through `__setattr__` would force tests that inject task state for fault simulation through the FSM — either we relax the FSM with wildcards (drains validation value) or rewrite tests (out of scope for Phase 4). |
| **QueenSuccession** | Declarative | `resilience.py` uses a single boolean flag plus monotonic term counter; the six phases are positions inside `_conduct_election`, not states stored between calls. Wiring requires splitting `elect_new_queen` into 4-5 phase methods AND re-running the 7 `TestQuorumSignedVotes` tests — conflates an FSM-modeling phase with a security-critical refactor. |
| **FailoverFlow** | Declarative | `CellFailover` tracks failed cells in a set; no per-cell phase state. Undo on `MIGRATING → LOST` is the natural use of `tramoya.undo()` but requires careful sequencing inside the existing try/except. Better aligned with the Phase 5+ split of `resilience.py` per ADR-006. |

The four declarative FSMs are **non-trivially valuable on their own**:

- Mermaid export documents lifecycles for new contributors
  (`docs/state-machines.md`).
- Property tests with Hypothesis validate graph structure (terminal
  states, retry path, etc.) — `tests/test_state_machines_property.py`.
- Trigger names map cleanly to call-sites — when a future phase wires
  one of them, the work is mechanical.

### 3. The mypy "Source file found twice" workaround

`pyproject.toml [tool.setuptools].package-dir = {hoc = "."}` makes the
repo root act as the `hoc` package root. Combined with `pythonpath =
["."]`, tests can import `from core import ...` (top-level) and
`from hoc.core import ...` (declared package) — both resolve to the
same files.

For `state_machines/`, mypy hits a corner case: when an absolutely-
imported module (`from state_machines.base import ...` in
`core/cells_base.py`) is followed during a directory scan, mypy infers
**two module names** for `state_machines/base.py`:

- `state_machines.base` (sys.path search from `cwd/state_machines/...`)
- `HOC.state_machines.base` (cwd is `D:\HOC`, treated as package)

`mypy .` then errors with **"Source file found twice under different
module names"**, blocking the entire run before any per-file checking.

Three workarounds were tried:

1. **`namespace_packages = true` + `explicit_package_bases = true`** —
   stops the cwd-name inference, but breaks every `from .core import …`
   relative import in legacy top-level files (resilience.py, swarm.py,
   nectar.py, memory.py, bridge.py, `__init__.py`) because the cwd is
   no longer treated as a package, so relative imports have no parent.
   Rejected.

2. **`[[tool.mypy.overrides]] module = ["HOC.state_machines.*"]
   ignore_errors = true`** — silently swallows errors, but the
   "found twice" message fires *before* per-module configuration is
   evaluated. Has no effect. Rejected.

3. **Exclude `^state_machines/` from `mypy .` and run a second
   invocation that lists files explicitly** — this works. The
   `[tool.mypy].exclude` blocks the directory scan only; passing
   `state_machines/*.py` as args via shell glob skips the exclude check.

We picked option 3. Strict mypy is preserved by:

- `pyproject.toml [tool.mypy].exclude` += `^state_machines/` (with a
  long comment block pointing at this ADR).
- CI lint workflow appends to the `mypy` job:
  `python -m mypy --explicit-package-bases state_machines/*.py`.
- Local development uses the same invocation when manually checking;
  documented in `CONTRIBUTING.md`.

## Alternatives considered

### Use a different FSM library (`transitions`, `python-statemachine`)

Both are larger (1000-2000+ LOC), require class inheritance, and have
different feature sets (no `undo`, no Mermaid export, no decorator-only
API). The brief specifically targeted `tramoya`. Sticking with tramoya
also lets us absorb upstream changes via the wrapper without dependency
churn. Rejected.

### Wire all five FSMs

For two of them (PheromoneDeposit, FailoverFlow) it provably exceeds
performance/refactor budgets in the limited Phase 4 window. For the
other two (TaskLifecycle, QueenSuccession) the *correct* wire-up
requires structural changes that would conflate phases. Going partial
on wiring would create a confusing two-tier system inside Phase 4.
Going full would push the closure into Phase 5+ territory. Rejected;
we ship 1 wired + 4 declarative, with the declarative FSMs already
specced for mechanical wire-up in Phase 5.

### Move `state_machines/` inside `core/` to avoid the mypy issue

Keeps the module import-able under both paths (`core.state_machines.*`
and `hoc.core.state_machines.*`), but those still hit the same
"found twice" error because mypy infers `HOC.core.state_machines.*`
when the cwd is `D:\HOC`. The fix would only differ in moving the
exclude line. Plus, `state_machines/` is conceptually orthogonal to
`core/` (it bridges multiple subsystems: cells, tasks, succession,
pheromones, failover) — nesting it under `core/` would be an
imports-driven organisation choice. Rejected.

### Use a `mypy.ini` inside `state_machines/` with a different config

mypy's discovery picks the nearest config file from cwd, so `cd state_
machines && mypy .` would use a sub-config. But: (a) the sub-config
must duplicate every flag from `pyproject.toml`, (b) it splits the
truth of "what mypy enforces on this repo" into two files, (c) CI
needs to know to `cd` before invoking. Less robust than the explicit-
file-list approach. Rejected.

## Consequences

### Easier

- **Future swaps of the FSM engine touch one file.** Same Phase 2
  pattern that absorbed any `mscs` API drift in `hoc.security`.
- **Strict mypy on every new state machine.** Bugs in the FSM
  definitions are caught in `mypy --explicit-package-bases
  state_machines/*.py` exactly the same way they would be in any other
  strict-checked module.
- **Mermaid + property tests pay for themselves on the four declarative
  FSMs.** Even without runtime wiring, contributors get diagrams and
  Hypothesis-driven invariants.

### Harder

- **The mypy invocation is non-obvious.** `mypy .` no longer covers the
  whole repo; you must remember the second command. Mitigated by the
  comment block in `pyproject.toml [tool.mypy].exclude`, the explicit
  CI step in `lint.yml`, and this ADR.
- **Per-FSM developers must remember the difference between wired and
  declarative.** Each FSM module's docstring opens with that distinction.
  When a developer wires a previously-declarative FSM in a future phase,
  they need to update the docstring + add tests for the new mutation
  path.
- **`HoneycombCell` carries a per-instance `HocStateMachine`.** ~9
  states, ~14 transitions × overhead is small per cell, but at scale
  (10k+ cells in a large grid) it is non-zero. If profiling in Phase 5
  shows it matters, the cell could share a class-level FSM and keep the
  per-instance string state alone — this ADR's wrapper supports it
  (`reset(state)` + `transition_to`).

### Risk / follow-up

- **Wire-up of the four declarative FSMs is a real Phase 5 commitment**,
  not optional polish. The Phase 4 closure flags it explicitly.
- **`state_machines/` accumulating new FSMs requires updating both
  `_fsm_registry()` in the Mermaid script and the tests in
  `test_state_machines.py`/`test_state_machines_property.py`.** A short
  checklist in `CONTRIBUTING.md` would help future contributors.
- **The "Source file found twice" hack is surface-area for future
  refactors.** If Phase 9 or 10 reorganises the package layout, this
  workaround should be re-evaluated alongside `package-dir = {hoc =
  "."}` itself.

## References

- `state_machines/base.py` — wrapper implementation.
- `state_machines/{cell,pheromone,task,succession,failover}_fsm.py` —
  five FSMs with rationale-rich docstrings.
- `scripts/generate_state_machines_md.py` — Mermaid export with
  `--check` mode for CI drift detection.
- `docs/state-machines.md` — auto-generated visual reference.
- `pyproject.toml [tool.mypy].exclude` — long comment pointing here.
- `.github/workflows/lint.yml` — `mypy` job + new `state-machines-doc`
  job.
- `tests/test_state_machines.py`, `tests/test_state_machines_property.py`,
  `tests/test_mermaid_export.py` — 81 Phase 4 tests.
- Phase 4 closure: `snapshot/PHASE_04_CLOSURE.md` "4.2 wrapper" and
  "4.3 CellState wired" sections.
- B12 discovery: `nectar.py:~1174` `RoyalJelly.get_stats` —
  `cmd.command` on a `RoyalCommand` enum value, would have raised
  `AttributeError` at runtime. Fix arrived during the swarm/nectar
  graduation (ADR-006), but the failure pattern is the same family
  ADR-007 prepares for: typed boundaries surface latent bugs.
