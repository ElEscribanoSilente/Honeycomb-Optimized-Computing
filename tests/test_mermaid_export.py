"""
Phase 4 tests for ``scripts/generate_state_machines_md.py``.

Two invariants:

1. **Determinism.** Running the generator twice yields byte-identical
   output. Property tests rely on this for re-runnable snapshots, and CI
   uses ``--check`` to detect drift between FSM specs and the doc.
2. **Coverage.** Every FSM module under ``state_machines/`` with a
   ``build_<x>_fsm`` factory is registered and rendered.
3. **Drift detector.** ``docs/state-machines.md`` on disk matches what
   the generator produces — this is the same check CI runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make ``scripts/`` importable as a top-level module.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import generate_state_machines_md as gen  # noqa: E402


def _render_fresh() -> str:
    """Build all registered FSMs and render the markdown."""
    fsms = gen._build_all(gen._fsm_registry())
    return gen.render(fsms)


class TestDeterminism:
    def test_back_to_back_runs_match(self):
        first = _render_fresh()
        second = _render_fresh()
        assert first == second, "generator output must be byte-identical across runs"


class TestCoverage:
    """Every FSM module must be registered and contribute a section."""

    def test_all_five_fsms_registered(self):
        registry = gen._fsm_registry()
        names = [name for name, _builder in registry]
        assert names == [
            "CellState",
            "PheromoneDeposit",
            "TaskLifecycle",
            "QueenSuccession",
            "FailoverFlow",
        ], f"FSM registry order/contents drifted: {names}"

    def test_each_fsm_renders_a_section(self):
        out = _render_fresh()
        for name in (
            "CellState",
            "PheromoneDeposit",
            "TaskLifecycle",
            "QueenSuccession",
            "FailoverFlow",
        ):
            assert f"## {name}" in out, f"section heading missing for {name}"

    def test_each_section_has_mermaid_block(self):
        out = _render_fresh()
        # Five FSMs → five fenced code blocks.
        assert out.count("```mermaid") == 5
        assert out.count("```\n") >= 5  # closing fences

    def test_index_links_to_each_fsm(self):
        out = _render_fresh()
        for slug in (
            "#cellstate",
            "#pheromonedeposit",
            "#tasklifecycle",
            "#queensuccession",
            "#failoverflow",
        ):
            assert slug in out


class TestDriftDetector:
    """``docs/state-machines.md`` on disk must equal generator output.
    Same check the CI lint workflow runs (``--check`` mode)."""

    def test_on_disk_matches_generator(self):
        target = _REPO_ROOT / "docs" / "state-machines.md"
        assert target.exists(), f"{target} missing — run generate_state_machines_md.py"
        on_disk = target.read_text(encoding="utf-8")
        fresh = _render_fresh()
        if on_disk != fresh:
            pytest.fail(
                "docs/state-machines.md is stale. "
                "Run: python scripts/generate_state_machines_md.py"
            )


class TestRenderEdgeCases:
    def test_empty_registry_produces_message(self):
        out = gen.render([])
        assert "_No state machines registered yet._" in out
        assert out.endswith("\n")

    def test_render_includes_state_count_and_initial(self):
        out = _render_fresh()
        # Picking CellState as the canary -- after Phase 4.3 cleanup it
        # has 7 states (SPAWNING and OVERLOADED removed).
        assert "**States** (7)" in out
        assert "**Initial state**: `EMPTY`" in out
