"""
Phase 4.1 tests for the static FSM checker (`choreo`).

Coverage:

- Walker: capture ``obj.state = X.Y``, ``obj._set_state(X.Y)``, and
  ``class X(Enum)`` declarations; respect exclude paths; tolerate
  malformed sources.
- Spec loader: extract :class:`FsmSpec` from duck-typed FSM objects
  (the actual import path is exercised by HOC's own ``state_machines/``
  in the smoke test below).
- Diff: bind FSMs to enums by member-subset; detect dead states,
  enum extras, undocumented mutations, declarative-only FSMs.
- CLI: exit codes for clean/warning/error/strict modes; JSON output
  validates as JSON.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from choreo import walker as choreo_walker
from choreo.cli import main as choreo_main
from choreo.derive import derive
from choreo.diff import bind_fsm_to_enum, compute_findings
from choreo.spec import _spec_from_fsm
from choreo.types import (
    KIND_DEAD_STATE,
    KIND_DECLARATIVE_ONLY,
    KIND_ENUM_EXTRA_STATE,
    KIND_UNDOCUMENTED_MUTATION,
    EnumDecl,
    Finding,
    FsmSpec,
    Mutation,
)

# ─── Walker ───────────────────────────────────────────────────────────────────


def _write(tmp_path: Path, name: str, content: str) -> Path:
    """Helper to write a Python source file with dedented content."""
    file = tmp_path / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(textwrap.dedent(content), encoding="utf-8")
    return file


class TestWalker:
    def test_captures_assign_pattern(self, tmp_path: Path):
        _write(
            tmp_path,
            "mod.py",
            """
            from enum import Enum

            class Color(Enum):
                RED = 1
                BLUE = 2

            def f(obj):
                obj.state = Color.RED
            """,
        )
        muts, enums = choreo_walker.walk(tmp_path)
        assert len(muts) == 1
        assert muts[0].enum_name == "Color"
        assert muts[0].member_name == "RED"
        assert muts[0].pattern == "assign"

    def test_captures_set_state_pattern(self, tmp_path: Path):
        _write(
            tmp_path,
            "mod.py",
            """
            from enum import Enum

            class Color(Enum):
                RED = 1

            def f(self):
                self._set_state(Color.RED)
            """,
        )
        muts, _ = choreo_walker.walk(tmp_path)
        assert len(muts) == 1
        assert muts[0].pattern == "_set_state"
        assert muts[0].member_name == "RED"

    def test_captures_enum_decl(self, tmp_path: Path):
        _write(
            tmp_path,
            "mod.py",
            """
            from enum import Enum

            class Color(Enum):
                RED = 1
                BLUE = 2
                GREEN = 3
            """,
        )
        _, enums = choreo_walker.walk(tmp_path)
        assert len(enums) == 1
        assert enums[0].name == "Color"
        assert set(enums[0].members) == {"RED", "BLUE", "GREEN"}

    def test_captures_qualified_enum_base(self, tmp_path: Path):
        # class X(enum.Enum) — Attribute base, not Name.
        _write(
            tmp_path,
            "mod.py",
            """
            import enum

            class Color(enum.Enum):
                RED = 1
            """,
        )
        _, enums = choreo_walker.walk(tmp_path)
        assert len(enums) == 1

    def test_skips_private_enum_members(self, tmp_path: Path):
        # Underscore-prefixed assignments inside an Enum body are typically
        # not real enum values (helpers, sentinel constants). Skip them.
        _write(
            tmp_path,
            "mod.py",
            """
            from enum import Enum

            class Color(Enum):
                RED = 1
                _internal = 99
            """,
        )
        _, enums = choreo_walker.walk(tmp_path)
        assert enums[0].members == ("RED",)

    def test_excludes_default_dirs(self, tmp_path: Path):
        # Files under tests/ should be skipped under default exclude.
        _write(
            tmp_path,
            "tests/mod.py",
            """
            from enum import Enum

            class Color(Enum):
                RED = 1

            def f(obj):
                obj.state = Color.RED
            """,
        )
        muts, _ = choreo_walker.walk(tmp_path)
        assert muts == []

    def test_silent_on_syntax_error(self, tmp_path: Path):
        (tmp_path / "broken.py").write_text("this is not valid python (((", encoding="utf-8")
        # Should not raise.
        muts, enums = choreo_walker.walk(tmp_path)
        assert muts == []
        assert enums == []

    def test_multiple_mutations_in_single_file(self, tmp_path: Path):
        _write(
            tmp_path,
            "mod.py",
            """
            from enum import Enum

            class Color(Enum):
                RED = 1
                BLUE = 2

            def f(obj):
                obj.state = Color.RED
                obj.state = Color.BLUE
                obj._set_state(Color.RED)
            """,
        )
        muts, _ = choreo_walker.walk(tmp_path)
        assert len(muts) == 3
        targets = {m.member_name for m in muts}
        assert targets == {"RED", "BLUE"}

    def test_ignores_non_state_attrs(self, tmp_path: Path):
        # Setting other attributes should not produce mutations.
        _write(
            tmp_path,
            "mod.py",
            """
            from enum import Enum

            class Color(Enum):
                RED = 1

            def f(obj):
                obj.role = Color.RED
                obj.kind = Color.RED
            """,
        )
        muts, _ = choreo_walker.walk(tmp_path)
        assert muts == []

    def test_captures_setattr_pattern(self, tmp_path: Path):
        _write(
            tmp_path,
            "mod.py",
            """
            from enum import Enum

            class Color(Enum):
                RED = 1

            def f(obj):
                setattr(obj, "state", Color.RED)
            """,
        )
        muts, _ = choreo_walker.walk(tmp_path)
        assert len(muts) == 1
        assert muts[0].pattern == "setattr"
        assert muts[0].enum_name == "Color"
        assert muts[0].member_name == "RED"

    def test_setattr_non_state_ignored(self, tmp_path: Path):
        _write(
            tmp_path,
            "mod.py",
            """
            from enum import Enum

            class Color(Enum):
                RED = 1

            def f(obj):
                setattr(obj, "role", Color.RED)
            """,
        )
        muts, _ = choreo_walker.walk(tmp_path)
        assert muts == []

    def test_setattr_dynamic_attr_name_ignored(self, tmp_path: Path):
        # When the attr name is a variable, not a literal, choreo cannot
        # statically know the target — accept the false negative.
        _write(
            tmp_path,
            "mod.py",
            """
            from enum import Enum

            class Color(Enum):
                RED = 1

            def f(obj, attr_name):
                setattr(obj, attr_name, Color.RED)
            """,
        )
        muts, _ = choreo_walker.walk(tmp_path)
        assert muts == []

    def test_captures_dataclasses_replace_pattern(self, tmp_path: Path):
        _write(
            tmp_path,
            "mod.py",
            """
            import dataclasses
            from enum import Enum

            class Color(Enum):
                RED = 1

            def f(obj):
                dataclasses.replace(obj, state=Color.RED)
            """,
        )
        muts, _ = choreo_walker.walk(tmp_path)
        assert len(muts) == 1
        assert muts[0].pattern == "dataclasses.replace"
        assert muts[0].member_name == "RED"

    def test_captures_bare_replace_pattern(self, tmp_path: Path):
        _write(
            tmp_path,
            "mod.py",
            """
            from dataclasses import replace
            from enum import Enum

            class Color(Enum):
                RED = 1

            def f(obj):
                replace(obj, state=Color.RED)
            """,
        )
        muts, _ = choreo_walker.walk(tmp_path)
        assert len(muts) == 1
        assert muts[0].pattern == "dataclasses.replace"

    def test_replace_without_state_kwarg_ignored(self, tmp_path: Path):
        _write(
            tmp_path,
            "mod.py",
            """
            import dataclasses
            from enum import Enum

            class Color(Enum):
                RED = 1

            def f(obj):
                dataclasses.replace(obj, role=Color.RED)
            """,
        )
        muts, _ = choreo_walker.walk(tmp_path)
        assert muts == []


# ─── Spec helper ──────────────────────────────────────────────────────────────


class _FakeFSM:
    """Duck-typed FSM with the attributes _spec_from_fsm reads."""

    def __init__(self, name: str, states: set[str], transitions: list[tuple[str, str, str]]):
        self.name = name
        self.states = states
        self.transitions = transitions


class TestSpecFromFsm:
    def test_extracts_basic_spec(self):
        fake = _FakeFSM(
            name="X",
            states={"A", "B"},
            transitions=[("A", "B", "go")],
        )
        spec = _spec_from_fsm(fake, source_file="x_fsm.py")
        assert spec is not None
        assert spec.name == "X"
        assert spec.states == ("A", "B")
        assert spec.transitions == (("A", "B", "go"),)

    def test_returns_none_for_missing_attrs(self):
        class Bad:
            name = "x"
            # missing states, transitions

        assert _spec_from_fsm(Bad(), source_file="x") is None

    def test_states_sorted_deterministically(self):
        fake = _FakeFSM(
            name="X",
            states={"Z", "A", "M"},
            transitions=[],
        )
        spec = _spec_from_fsm(fake, source_file="x")
        assert spec is not None
        assert spec.states == ("A", "M", "Z")


# ─── Diff: bind ───────────────────────────────────────────────────────────────


class TestBindFsmToEnum:
    def _fsm(self, name: str, states: tuple[str, ...]) -> FsmSpec:
        return FsmSpec(name=name, source_file="x", states=states, transitions=())

    def _enum(self, name: str, members: tuple[str, ...]) -> EnumDecl:
        return EnumDecl(file="x.py", line=1, name=name, members=members)

    def test_binds_when_members_superset(self):
        fsm = self._fsm("X", ("A", "B"))
        enum = self._enum("XEnum", ("A", "B", "C"))
        assert bind_fsm_to_enum(fsm, [enum]) is enum

    def test_no_bind_when_members_missing(self):
        fsm = self._fsm("X", ("A", "B", "C"))
        enum = self._enum("XEnum", ("A", "B"))  # missing C
        assert bind_fsm_to_enum(fsm, [enum]) is None

    def test_smallest_enum_wins(self):
        fsm = self._fsm("X", ("A",))
        small = self._enum("Small", ("A", "B"))
        large = self._enum("Large", ("A", "B", "C", "D"))
        assert bind_fsm_to_enum(fsm, [large, small]) is small

    def test_alphabetical_tiebreak(self):
        fsm = self._fsm("X", ("A",))
        zeta = self._enum("Zeta", ("A", "B"))
        alpha = self._enum("Alpha", ("A", "B"))
        assert bind_fsm_to_enum(fsm, [zeta, alpha]) is alpha

    def test_explicit_enum_name_overrides_heuristic(self):
        # Two enums both qualify for the heuristic, but the FSM names
        # the larger one explicitly. Phase 4.2: explicit wins.
        fsm = FsmSpec(
            name="X",
            source_file="x",
            states=("A", "B"),
            transitions=(),
            enum_name="LargeOne",
        )
        small = self._enum("SmallOne", ("A", "B"))
        large = self._enum("LargeOne", ("A", "B", "C", "D"))
        # Heuristic would have picked SmallOne; explicit enum_name picks
        # LargeOne.
        assert bind_fsm_to_enum(fsm, [small, large]) is large

    def test_explicit_enum_name_inconsistent_returns_none(self):
        # FSM names an enum whose members do NOT cover the FSM's states.
        # Returns None (signals inconsistency rather than silently
        # falling through to the heuristic).
        fsm = FsmSpec(
            name="X",
            source_file="x",
            states=("A", "B", "MISSING"),
            transitions=(),
            enum_name="WrongOne",
        )
        wrong = self._enum("WrongOne", ("A", "B"))  # missing MISSING
        assert bind_fsm_to_enum(fsm, [wrong]) is None

    def test_explicit_enum_name_unknown_falls_back_to_heuristic(self):
        # Named enum doesn't exist in the codebase. Fall back to
        # heuristic so the user still gets a binding (probably with a
        # warning that the named enum is missing — Phase 4.2 leaves
        # warning emission to the diff layer).
        fsm = FsmSpec(
            name="X",
            source_file="x",
            states=("A", "B"),
            transitions=(),
            enum_name="DoesNotExist",
        )
        match = self._enum("Other", ("A", "B"))
        assert bind_fsm_to_enum(fsm, [match]) is match


# ─── Diff: findings ───────────────────────────────────────────────────────────


def _spec(name: str, states: tuple[str, ...]) -> FsmSpec:
    return FsmSpec(name=name, source_file=f"{name}.py", states=states, transitions=())


def _enum_decl(name: str, members: tuple[str, ...]) -> EnumDecl:
    return EnumDecl(file=f"{name}.py", line=10, name=name, members=members)


def _mut(enum_name: str, member: str, line: int = 1) -> Mutation:
    return Mutation(file="m.py", line=line, enum_name=enum_name, member_name=member)


class TestComputeFindings:
    def test_clean_run_no_findings_for_full_coverage(self):
        # Every fsm state has an observed mutation; no enum extras.
        spec = _spec("X", ("A", "B"))
        enum = _enum_decl("X", ("A", "B"))
        muts = [_mut("X", "A"), _mut("X", "B")]
        findings = compute_findings([spec], muts, [enum])
        assert findings == []

    def test_dead_state_warning(self):
        spec = _spec("X", ("A", "B", "DEAD"))
        enum = _enum_decl("X", ("A", "B", "DEAD"))
        muts = [_mut("X", "A"), _mut("X", "B")]
        findings = compute_findings([spec], muts, [enum])
        assert len(findings) == 1
        assert findings[0].severity == "warning"
        assert findings[0].kind == KIND_DEAD_STATE
        assert "DEAD" in findings[0].message

    def test_enum_extra_state_warning(self):
        spec = _spec("X", ("A", "B"))
        enum = _enum_decl("X", ("A", "B", "EXTRA"))
        muts = [_mut("X", "A"), _mut("X", "B")]
        findings = compute_findings([spec], muts, [enum])
        assert len(findings) == 1
        assert findings[0].kind == KIND_ENUM_EXTRA_STATE
        assert "EXTRA" in findings[0].message

    def test_undocumented_mutation_error(self):
        spec = _spec("X", ("A", "B"))
        enum = _enum_decl("X", ("A", "B", "UNKNOWN"))
        muts = [_mut("X", "UNKNOWN")]
        findings = compute_findings([spec], muts, [enum])
        # The mutation targets a state not in the FSM. Even though the
        # enum has it, the FSM does not — undocumented mutation.
        kinds = [f.kind for f in findings]
        assert KIND_UNDOCUMENTED_MUTATION in kinds
        # And ENUM_EXTRA_STATE for UNKNOWN.
        assert KIND_ENUM_EXTRA_STATE in kinds

    def test_orphan_mutation_against_unmodeled_enum(self):
        # Mutation references an enum nobody declares as fsm.
        muts = [_mut("Mystery", "X")]
        findings = compute_findings([], muts, [])
        assert len(findings) == 1
        assert findings[0].severity == "error"
        assert findings[0].kind == KIND_UNDOCUMENTED_MUTATION

    def test_declarative_only_fsm(self):
        spec = _spec("LonelyFSM", ("FRESH", "DECAYING"))
        # No enum declares those names; no mutations.
        findings = compute_findings([spec], [], [])
        assert len(findings) == 1
        assert findings[0].severity == "info"
        assert findings[0].kind == KIND_DECLARATIVE_ONLY

    def test_findings_sorted_severity_first(self):
        spec_with_dead = _spec("X", ("A", "DEAD"))
        enum = _enum_decl("X", ("A", "DEAD"))
        spec_lonely = _spec("LonelyFSM", ("FOO",))

        muts = [_mut("X", "A"), _mut("Mystery", "Z")]
        findings = compute_findings([spec_with_dead, spec_lonely], muts, [enum])
        # First the error (orphan Mystery), then warning (dead_state),
        # then info (declarative_only).
        severities = [f.severity for f in findings]
        assert severities == ["error", "warning", "info"]


# ─── Derive ───────────────────────────────────────────────────────────────────


class TestDerive:
    """Phase 4.2: ``choreo derive`` auto-generates a FSM skeleton from
    observed mutations in a single module."""

    def test_emits_skeleton_for_simple_module(self, tmp_path: Path):
        mod = _write(
            tmp_path,
            "thing.py",
            """
            from enum import Enum

            class ThingState(Enum):
                IDLE = 1
                BUSY = 2

            def f(obj):
                obj.state = ThingState.IDLE
                obj.state = ThingState.BUSY
            """,
        )
        out = derive(mod)
        # Expected structure markers.
        assert "build_thing_fsm" in out
        assert "ThingLifecycle" in out
        assert "enum_name='ThingState'" in out
        assert "THINGSTATE_IDLE = 'IDLE'" in out
        assert "THINGSTATE_BUSY = 'BUSY'" in out
        assert "WILDCARD, THINGSTATE_IDLE" in out
        assert "WILDCARD, THINGSTATE_BUSY" in out

    def test_overrides_fsm_name(self, tmp_path: Path):
        mod = _write(
            tmp_path,
            "thing.py",
            """
            from enum import Enum

            class ThingState(Enum):
                A = 1

            def f(obj):
                obj.state = ThingState.A
            """,
        )
        out = derive(mod, fsm_name="MyCustomFSM")
        assert "MyCustomFSM" in out
        assert "build_my_custom_fsm" in out

    def test_overrides_enum_name(self, tmp_path: Path):
        # Module mentions two enums; user pins one.
        mod = _write(
            tmp_path,
            "thing.py",
            """
            from enum import Enum

            class A(Enum):
                X = 1

            class B(Enum):
                Y = 2

            def f(obj):
                obj.state = A.X
                obj.state = B.Y
            """,
        )
        out = derive(mod, enum_name="B")
        assert "enum_name='B'" in out
        assert "B_Y = 'Y'" in out
        # Default-most-frequent picks A (alphabetical tie); explicit B
        # should override.
        assert "A_X" not in out

    def test_overrides_initial(self, tmp_path: Path):
        mod = _write(
            tmp_path,
            "thing.py",
            """
            from enum import Enum

            class S(Enum):
                A = 1
                B = 2

            def f(obj):
                obj.state = S.A
                obj.state = S.B
            """,
        )
        out = derive(mod, initial_state="B")
        assert "initial=S_B" in out

    def test_initial_not_in_observed_targets_added(self, tmp_path: Path):
        # Caller specifies an initial state that no mutation observed.
        # Should still appear in the generated states block.
        mod = _write(
            tmp_path,
            "thing.py",
            """
            from enum import Enum

            class S(Enum):
                A = 1
                B = 2

            def f(obj):
                obj.state = S.A
            """,
        )
        out = derive(mod, initial_state="PRE_OBSERVED")
        assert "S_PRE_OBSERVED = 'PRE_OBSERVED'" in out
        assert "initial=S_PRE_OBSERVED" in out

    def test_empty_module_emits_placeholder(self, tmp_path: Path):
        mod = _write(
            tmp_path,
            "empty.py",
            """
            def f():
                pass
            """,
        )
        out = derive(mod)
        assert "No `obj.state = X`" in out
        assert "TODO" in out


# ─── CLI ──────────────────────────────────────────────────────────────────────


class TestCli:
    def test_help_returns_2_without_subcommand(self, capsys):
        # argparse with required=True will exit with SystemExit(2)
        # before main even returns; assert that behaviour.
        with pytest.raises(SystemExit) as exc:
            choreo_main([])
        assert exc.value.code == 2

    def test_clean_run_exits_zero(self, tmp_path: Path, monkeypatch, capsys):
        # Empty repo → no specs, no mutations → no findings.
        monkeypatch.chdir(tmp_path)
        rc = choreo_main(["check"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "no drift detected" in out.lower()

    def test_warnings_exit_zero_without_strict(self, tmp_path: Path, monkeypatch, capsys):
        # Patch the inner functions to inject one warning finding.
        warning = Finding(
            severity="warning",
            fsm="Test",
            kind="dead_state",
            message="injected",
        )
        monkeypatch.setattr("choreo.cli.load_specs", lambda root, subdir: [])
        monkeypatch.setattr("choreo.cli.walk", lambda root: ([], []))
        monkeypatch.setattr("choreo.cli.compute_findings", lambda s, m, e: [warning])
        rc = choreo_main(["check", "--root", str(tmp_path)])
        assert rc == 0  # warnings alone do not fail

    def test_warnings_exit_one_with_strict(self, tmp_path: Path, monkeypatch, capsys):
        warning = Finding(
            severity="warning",
            fsm="Test",
            kind="dead_state",
            message="injected",
        )
        monkeypatch.setattr("choreo.cli.load_specs", lambda root, subdir: [])
        monkeypatch.setattr("choreo.cli.walk", lambda root: ([], []))
        monkeypatch.setattr("choreo.cli.compute_findings", lambda s, m, e: [warning])
        rc = choreo_main(["check", "--root", str(tmp_path), "--strict"])
        assert rc == 1

    def test_errors_exit_one(self, tmp_path: Path, monkeypatch, capsys):
        error = Finding(
            severity="error",
            fsm="Test",
            kind="undocumented_mutation",
            message="injected",
        )
        monkeypatch.setattr("choreo.cli.load_specs", lambda root, subdir: [])
        monkeypatch.setattr("choreo.cli.walk", lambda root: ([], []))
        monkeypatch.setattr("choreo.cli.compute_findings", lambda s, m, e: [error])
        rc = choreo_main(["check", "--root", str(tmp_path)])
        assert rc == 1

    def test_json_output_parses(self, tmp_path: Path, monkeypatch, capsys):
        monkeypatch.setattr("choreo.cli.load_specs", lambda root, subdir: [])
        monkeypatch.setattr("choreo.cli.walk", lambda root: ([], []))
        monkeypatch.setattr("choreo.cli.compute_findings", lambda s, m, e: [])
        rc = choreo_main(["check", "--root", str(tmp_path), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["counts"]["error"] == 0
        assert payload["findings"] == []

    def test_invalid_root_exits_two(self, tmp_path: Path, capsys):
        rc = choreo_main(["check", "--root", str(tmp_path / "does-not-exist")])
        assert rc == 2

    def test_derive_subcommand_to_stdout(self, tmp_path: Path, capsys):
        mod = _write(
            tmp_path,
            "widget.py",
            """
            from enum import Enum

            class WidgetState(Enum):
                ON = 1

            def f(obj):
                obj.state = WidgetState.ON
            """,
        )
        rc = choreo_main(["derive", str(mod)])
        out = capsys.readouterr().out
        assert rc == 0
        # WidgetState -> WidgetLifecycle -> build_widget_fsm
        assert "build_widget_fsm" in out

    def test_derive_subcommand_to_file(self, tmp_path: Path, capsys):
        mod = _write(
            tmp_path,
            "widget.py",
            """
            from enum import Enum

            class WidgetState(Enum):
                ON = 1

            def f(obj):
                obj.state = WidgetState.ON
            """,
        )
        out_file = tmp_path / "out_fsm.py"
        rc = choreo_main(["derive", str(mod), "-o", str(out_file)])
        assert rc == 0
        assert out_file.exists()
        assert "build_widget_fsm" in out_file.read_text(encoding="utf-8")

    def test_derive_invalid_module_exits_two(self, tmp_path: Path):
        rc = choreo_main(["derive", str(tmp_path / "nope.py")])
        assert rc == 2


# ─── Smoke test against HOC itself ────────────────────────────────────────────


class TestHocIntegration:
    """End-to-end check that choreo, run against the HOC repo it lives in,
    produces exactly the findings documented in the current closure.

    Phase 5.2b cleanup status:

    - **0 errors** -- no undocumented mutations.
    - **0 warnings** -- MIGRATING / SEALED wired in Phase 5.1.
    - **0 info** -- all 5 FSMs are now wired (CellState 5.1,
      FailoverFlow 5.2c, PheromoneDeposit 5.2a, QueenSuccession 5.2b,
      TaskLifecycle stayed wired since Phase 4.1). Phase 5.6 can flip
      ``--strict`` in CI.
    """

    def test_hoc_smoke(self):
        from choreo.cli import main

        # Calling main without monkeypatching exercises the real walker
        # and spec loader against HOC. We need the HOC root, which is
        # the parent of this test file's repo.
        repo_root = Path(__file__).resolve().parents[1]
        rc = main(["check", "--root", str(repo_root), "--json"])
        assert rc == 0  # warnings allowed by default; no errors expected

    def test_hoc_findings_exact(self, capsys):
        # Run choreo programmatically and check the structured output.
        from choreo.spec import load_specs
        from choreo.walker import walk

        repo_root = Path(__file__).resolve().parents[1]
        specs = load_specs(repo_root)
        muts, enums = walk(repo_root)
        findings = compute_findings(specs, muts, enums)

        by_kind: dict[str, list[Finding]] = {}
        for f in findings:
            by_kind.setdefault(f.kind, []).append(f)

        # Expected structure (exact counts, validated against the Phase
        # 5.1 wire-up state of HOC).
        assert (
            "undocumented_mutation" not in by_kind
        ), f"Unexpected undocumented mutations: {by_kind.get('undocumented_mutation')}"

        # Phase 5.1: MIGRATING + SEALED wired -> no dead_state findings.
        assert "dead_state" not in by_kind, (
            f"Phase 5.1 wired MIGRATING and SEALED, no dead_state findings expected: "
            f"{by_kind.get('dead_state')}"
        )

        # Phase 4.3 removed TaskState.ASSIGNED entirely, so the
        # enum_extra_state finding from Phase 4.1/4.2 should be gone.
        assert "enum_extra_state" not in by_kind, (
            f"Phase 4.3 should have eliminated enum_extra_state findings: "
            f"{by_kind.get('enum_extra_state')}"
        )

        decl = by_kind.get("declarative_only", [])
        decl_names = {f.fsm for f in decl}
        # Phase 5.2b wired QueenSuccession via the ``_succession_state``
        # wrapper. All 5 FSMs are now wired (TaskLifecycle stayed
        # wired since Phase 4.1).
        assert decl_names == set()
