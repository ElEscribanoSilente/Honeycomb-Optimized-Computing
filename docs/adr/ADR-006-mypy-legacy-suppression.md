# ADR-006: Legacy modules suppressed from strict mypy in Phase 3

- **Status**: Accepted
- **Date**: 2026-04-23
- **Phase**: Phase 3

## Context

Phase 3 introduces `mypy --strict` as the standard type-checker for HOC.
However, HOC has ~8,666 LOC spread across 8 modules that predate a strict
typing regime. An initial `mypy --strict .` on the whole codebase reports
~100 errors in `core.py`, `metrics.py`, `bridge.py`, `nectar.py`, and
`swarm.py`. Most are:

- Missing `dict[str, ...]` generic parameters.
- Implicit `Optional` (`def f(x: T = None)` without `| None`).
- Missing return annotations.
- Untyped `Callable` parameters.
- Mutable class attributes without `ClassVar`.

Fixing all 100 inline would add ~2 days of annotation work that
contributes nothing to Phase 3's actual goals (tooling + CI + split
`core.py`). Worse, the ongoing `core.py` / `metrics.py` split would
require *re-annotating* the same code twice (once before, once after the
split).

## Decision

In Phase 3, `mypy --strict` is enforced on **`security.py`, `memory.py`,
`resilience.py`** (and `__init__.py`, which is mostly re-exports). These
are the modules that Phase 2 and Phase 3 either created or substantially
touched.

Legacy modules `core.py`, `metrics.py`, `bridge.py`, `nectar.py`,
`swarm.py` are **excluded** from `mypy` scanning via
`[tool.mypy].exclude` in `pyproject.toml`. A parallel
`[[tool.mypy.overrides]]` block with `ignore_errors = true` covers the
module-path form, so imports into scanned code don't trigger strict
checking transitively.

Future phases graduate modules out of this suppression one at a time,
with the expected sequence:

| Phase | Module graduated | Rationale |
|-------|------------------|-----------|
| 4     | `swarm.py`       | Touched heavily for `tramoya` FSM integration. |
| 4     | `nectar.py`      | Touched for `PheromoneDeposit` FSM. |
| 5     | `core/` subpkg   | Refactored in Phase 3 → natural to type next. |
| 5     | `metrics/` subpkg | Same. |
| 6     | `bridge.py`      | Stable; no structural churn expected. |

## Alternatives considered

### Annotate everything in Phase 3

Blocks Phase 3 by ~2 days for annotation work that is mechanical, not
architectural. Re-annotation required after the `core.py` split anyway.
Rejected.

### Relax strictness globally

Setting `strict = false` and re-enabling individual flags would let all
files pass the cursory check but would hide *real* issues in
`security.py` (which Phase 2 added with types and Phase 3 now checks).
The strict/lax boundary is the whole point of mypy enforcement.
Rejected.

### Use `# type: ignore[error-code]` inline

Fine for a handful of errors; absurd for 100. Also creates 100 lines of
noise that must be reviewed and maintained. Rejected.

### `[[tool.mypy.overrides]]` with softened flags (no `ignore_errors`)

Initial attempt — set `disallow_untyped_defs = false`,
`check_untyped_defs = false`, `warn_return_any = false`. Didn't work:
mypy still reports `type-arg`, `attr-defined`, `assignment`, `operator`
errors because those aren't gated on the strict flags. Rejected; moved
to `ignore_errors = true`.

### Use `exclude` only (drop the module-path override)

Works for `python -m mypy .` (scans nothing in the excluded files) but
breaks when a scanned module imports from a legacy one — mypy follows
the import and re-scans. Solved by pairing `exclude` (for direct input)
with the `[[tool.mypy.overrides]]` block (for import-following), with
`follow_imports = "silent"` as the global default.

## Consequences

**Easier**:

- Phase 3 closes in scope. No annotation work piled onto the tooling
  phase.
- `security.py` / `memory.py` / `resilience.py` (the modules that matter
  most for security and stability) are strict-typed.
- Future phases gain a clear sequence: remove a module from the
  suppression list, fix the exposed errors inline, commit.

**Harder**:

- Contributors touching a legacy module do not get mypy feedback on
  their changes. CONTRIBUTING.md calls this out explicitly and
  recommends adding annotations when making non-trivial edits.
- Three states (strict, legacy-suppressed, excluded-tests) increase
  cognitive load for new contributors. Mitigated by the config comments
  in `pyproject.toml` pointing back to this ADR.
- The `exclude` regex and the `[[tool.mypy.overrides]].module` list
  duplicate information; both must be kept in sync.

**Risk / follow-up**:

- Forgetting to graduate a module when it is heavily touched. Mitigation:
  CI does not check this automatically, but code review should flag PRs
  that touch legacy modules without bringing them out of suppression.
- Real type bugs in legacy modules remain hidden. **B11** (found during
  Phase 3's mypy pass on `resilience.py`) is exactly this kind of bug —
  a silent attribute assignment on a non-existent field. Accept that
  more B-numbers will surface as modules graduate.

## References

- `pyproject.toml` `[tool.mypy]`, `[[tool.mypy.overrides]]`, `exclude`
  patterns.
- Phase 3 closure: `snapshot/PHASE_03_CLOSURE.md` "Mypy gap" section.
- Phase 4 closure: `snapshot/PHASE_04_CLOSURE.md` "4.10 Graduación
  ADR-006" section.
- `.github/workflows/lint.yml` (CI enforcement on `python -m mypy .`).
- CONTRIBUTING.md "Code style" section on types.
- B11 discovery: `resilience.py:1138` — wrote to
  `cell._pheromone_level` (non-existent); corrected to
  `cell._pheromone_field = PheromoneField()`.
- B12 discovery (Phase 4 graduation): `nectar.py:~1174`
  `RoyalJelly.get_stats` referenced `cmd.command` on a `RoyalCommand`
  enum value (no such attribute). Fix: `cmd.command.name` →
  `cmd.name`, `c.command == cmd.command` → `c.command == cmd`.

## Phase 4 graduation outcome (2026-04-24)

The graduation schedule above held: Phase 4 removed both `swarm` and
`nectar` from the `[[tool.mypy.overrides]]` `ignore_errors=true` list.
The 29 errors that surfaced (11 in swarm, 18 in nectar) were annotated
inline. The pattern of fixes was uniform — generic-parameter omissions,
implicit `Optional`s, missing return annotations, dict/defaultdict
generics, `Callable`s without type args, `cast(bytes, _mscs.dumps(...))`
to bridge the `Any` return — exactly what the ADR predicted.

The graduation also surfaced **B12** (described above) as a
runtime-impacting bug, validating the third bug in the B9/B11 family
that mypy strict catches: silent attribute lookups against types that
do not declare the field.

Remaining schedule (unchanged from the original ADR):

| Phase | Module graduated | Status |
|-------|------------------|--------|
| 4 | `swarm.py` | ✅ Graduated 2026-04-24 (Phase 4) |
| 4 | `nectar.py` | ✅ Graduated 2026-04-24 (Phase 4) |
| 5 | `core/` subpkg | pending |
| 5 | `metrics/` subpkg | pending |
| 6 | `bridge.py` | pending |
