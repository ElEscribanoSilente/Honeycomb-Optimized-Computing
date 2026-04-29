# Contributing to HOC

Thanks for taking the time to contribute. HOC (Honeycomb Optimized Computing)
is a bio-inspired distributed computing framework; contributions ranging from
bug fixes to new topology algorithms are welcome.

If you're here for the first time, read [ROADMAP.md](ROADMAP.md) and
[CHANGELOG.md](CHANGELOG.md) first — HOC is being stabilized in a documented
10-phase roadmap and active phases have specific constraints.

## Code of conduct

By participating, you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md) (Contributor Covenant v2.1).

## Reporting security issues

**Do not file a public GitHub issue for security vulnerabilities.** See
[SECURITY.md](SECURITY.md) for private disclosure channels.

---

## Development environment

```bash
# Clone
git clone https://github.com/esraderey/Honeycomb-Optimized-Computing.git
cd Honeycomb-Optimized-Computing

# Install editable + dev extras
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install mscs   # secure serialization, Phase 2+ requirement

# (Optional) Install pre-commit hooks so formatters run on every commit.
pre-commit install
```

Supported Python: **3.10, 3.11, 3.12** (tested on Ubuntu, macOS, Windows in
`.github/workflows/test.yml`).

## Running the test suite

```bash
# Full suite (~10s)
python -m pytest tests/ --ignore=tests/test_heavy.py

# With coverage, threshold enforced by pyproject.toml
python -m pytest tests/ --ignore=tests/test_heavy.py --cov --cov-report=term

# Including heavy benchmarks (multi-minute)
python -m pytest tests/

# Only the security suite (Phase 2)
python -m pytest tests/test_security.py

# Property-based tests via Hypothesis
python -m pytest tests/test_property.py
```

`tests/test_heavy.py` runs multi-minute workloads and is excluded from default
CI; run it locally before large refactors.

## Running quality checks

```bash
# Lint (must be 0 errors before merge)
python -m ruff check .

# Formatting (must be 0 diffs before merge)
python -m black --check .

# Types (strict on security/memory/resilience; legacy suppressed via
# [tool.mypy].exclude — see ADR-006).
# NOTE: your local checkout directory name must be a valid Python identifier
# (e.g. `hoc`, `HOC`, `my_hoc` — no hyphens). See pyproject.toml comment
# above the [tool.mypy] section for background.
python -m mypy .

# Security
python -m bandit -r . -x ./tests,./benchmarks,./snapshot
python -m pip_audit -r requirements.txt

# Complexity
python -m radon cc . -a -nc
python -m radon raw . -s

# Phase 5.5 / 6.7: bench regression check. The repo carries two baselines:
#
#   - ``snapshot/bench_baseline.json``     — captured on a developer machine
#                                            (Windows). Use this LOCALLY:
#                                            its noise floor matches your
#                                            hardware so a real regression
#                                            stands out.
#   - ``snapshot/bench_baseline_ci.json``  — captured on GitHub Actions
#                                            ``ubuntu-latest`` runner. CI
#                                            uses this one (Phase 6.7
#                                            removed the cross-runner
#                                            mismatch from the comparison).
#
# ``--benchmark-warmup=on`` + ``--benchmark-min-time=0.5`` reduce the noise
# floor on sub-microsecond benches.
python -m pytest benchmarks/ \
    --benchmark-only \
    --benchmark-json=snapshot/bench_current.json \
    --benchmark-warmup=on \
    --benchmark-min-time=0.5 \
    -q

# Compare against snapshot/bench_baseline.json (LOCAL). Fail if any
# benchmark regressed > 10% on the mean.
python scripts/compare_bench.py \
    snapshot/bench_baseline.json \
    snapshot/bench_current.json \
    --threshold 10.0
```

All of the above run automatically on every PR via GitHub Actions
(`.github/workflows/lint.yml`, `.github/workflows/security.yml`,
`.github/workflows/test.yml`, `.github/workflows/bench.yml`). CI compares
against `snapshot/bench_baseline_ci.json`; local comparison uses
`snapshot/bench_baseline.json` (see commands above).

### Regenerating the CI baseline (`snapshot/bench_baseline_ci.json`)

Cross-runner variance was the reason Phase 5.5 left the bench job in
advisory mode (`continue-on-error: true`, threshold raised to 50%).
Phase 6.7 fixed that by capturing a baseline on the same hardware CI
uses:

```bash
# 1. Trigger the bench workflow on the target ref (typically main).
gh workflow run bench.yml --ref main

# 2. Wait for it to finish and find the run id.
gh run list --workflow=bench.yml --limit 1

# 3. Download the artifact (a single file: bench_current.json).
gh run download <run-id> -n bench-current-<run-id> -D .tmp_bench/

# 4. Move it into place and commit alongside the bump that motivated it.
mv .tmp_bench/bench_current.json snapshot/bench_baseline_ci.json
git add snapshot/bench_baseline_ci.json
git commit -m "chore(bench): refresh CI baseline (<short-rationale>)"
```

Refresh the CI baseline only when there's a deliberate, documented
performance change that the CI threshold would otherwise flag (e.g. a
deliberate optimization that knocks 20% off a hot path, or a runner
image upgrade that uniformly shifts the noise floor). Routine refreshes
mask regressions — don't do it as part of "keeping the baseline
current".

## Making a change

1. **Pick an issue** or propose one. For non-trivial work, file a GitHub issue
   first so scope can be aligned with the active phase.
2. **Branch from `main`.** Use a descriptive slug: `feature/pheromone-gradient`,
   `fix/queen-succession-replay`, `docs/explain-stigmergy`. If you're closing a
   roadmap phase, use `phase/NN-slug` (e.g. `phase/04-config`).
3. **Write tests first when fixing bugs.** The failing test is part of the PR.
4. **Keep commits focused.** One logical change per commit. Commit messages
   follow the pattern used in Phase 1/2/3 closures — a short imperative
   subject with an em-dash headline, then a body explaining *why* with
   references to bug IDs / issue numbers.
5. **Run the full quality checks locally before opening the PR.** CI will run
   them too, but failing CI slows review.
6. **Open the PR against `main`.** If you're closing a phase, also update
   `ROADMAP.md`, `CHANGELOG.md`, and add a `snapshot/PHASE_NN_CLOSURE.md`.

## Code style

- **Formatter**: `black` (line length 100). Every file must pass
  `black --check`.
- **Linter**: `ruff` (see `[tool.ruff]` in `pyproject.toml` for the enabled
  rule set: `E`, `F`, `W`, `I`, `B`, `UP`, `SIM`, `RUF`).
- **Imports**: `ruff` handles isort-equivalent sort. Module groups
  (stdlib → third-party → first-party) separated by a blank line.
- **Types**: new or touched modules MUST pass `mypy --strict`. Legacy
  modules (`core.py`, `metrics.py`, `bridge.py`, `nectar.py`, `swarm.py`)
  are suppressed in Phase 3 but will be tightened module-by-module. If you
  touch a legacy module substantially, consider graduating it out of the
  suppression list.
- **Docstrings**: Spanish for module-level docstrings (existing convention);
  English OK for function docstrings. Google or NumPy style — be consistent
  within a module.

## What makes a good PR

- **Small and focused.** One bug fix or one feature at a time. Large refactors
  should land as a sequence of reviewable commits (see Phase 3 `core.py` split
  for an example).
- **Tests that would catch the regression.** Unit tests for logic; property
  tests (Hypothesis) for algebraic invariants; security tests for new attack
  surface.
- **No new `bandit` findings.** HOC maintains 0 HIGH / 0 MEDIUM / 0 LOW
  Bandit findings since Phase 2.
- **No new `pip-audit` findings.** If a new dependency has a known CVE,
  justify it in the PR description or don't add it.
- **No public API breaking changes** without an accompanying major version
  bump and explicit call-out. `from hoc import HoneycombGrid, HexCoord, ...`
  is a contract, not an implementation detail.

## Roadmap phases

Each phase has a Definition of Done that must be met before the phase is
tagged (`v1.N.0-phaseNN`) and merged to `main`. Phase details live in
[ROADMAP.md](ROADMAP.md). Contributions outside the active phase are still
welcome but may be held for review until the current phase is closed.

## Questions

Open a GitHub Discussion if you're unsure whether a change fits, or file an
issue with the "question" label.
