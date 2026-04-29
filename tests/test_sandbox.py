"""Phase 7.4 — :class:`SandboxedTaskRunner` tests.

Verifies that the new sandbox module isolates tasks from the panal:

- ``"none"`` mode is a pass-through that returns / raises in the
  current process (parity with no sandbox at all).
- ``"process"`` mode forks a child (POSIX-only in v1; Windows is
  deferred — see :mod:`hoc.sandbox`); a payload exception, OS-level
  crash (SIGSEGV via ctypes NULL-deref, or ``os._exit``), or timeout
  is observed as a :class:`SandboxTimeout` / :class:`SandboxCrashed`
  in the parent without taking the test process down.
- ``"cgroup"`` and ``"job_object"`` modes raise
  :class:`SandboxNotSupported` in v1 (deferred to Phase 7.x).
- The probe helpers (``cgroup_v2_available`` /
  ``job_objects_available``) return platform-honest answers without
  raising.

Process-isolation tests are skipped on Windows (no fork support); the
sandbox raises :class:`SandboxNotSupported` there. Linux + macOS run
the full suite.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

from hoc.sandbox import (
    SandboxConfig,
    SandboxCrashed,
    SandboxedTaskRunner,
    SandboxNotSupported,
    SandboxTimeout,
    cgroup_v2_available,
    job_objects_available,
)

# ───────────────────────────────────────────────────────────────────────────────
# Module-level payloads (top-level so multiprocessing.spawn can pickle them)
# ───────────────────────────────────────────────────────────────────────────────


def _ok_payload(x: int, y: int) -> int:
    return x + y


def _payload_with_kwargs(x: int, *, scale: int) -> int:
    return x * scale


def _slow_payload(secs: float) -> str:
    time.sleep(secs)
    return "done"


def _raising_payload() -> None:
    raise ValueError("intentional sandbox-test exception")


def _exit_nonzero_payload() -> None:
    """Hard-exit the worker process. The parent sees this as a
    crashed sandbox: no result was queued."""
    os._exit(13)


def _segfault_payload() -> None:
    """Trigger a SIGSEGV in the child via NULL-pointer deref. ctypes
    is the most reliable cross-platform way to do this in pure
    Python."""
    import ctypes

    ctypes.string_at(0)  # type: ignore[arg-type]


# ───────────────────────────────────────────────────────────────────────────────
# "none" isolation: pass-through
# ───────────────────────────────────────────────────────────────────────────────


class TestNoneIsolation:
    def test_pass_through_return(self):
        runner = SandboxedTaskRunner(SandboxConfig(isolation="none"))
        assert runner.run(_ok_payload, 2, 3) == 5

    def test_pass_through_raises(self):
        runner = SandboxedTaskRunner(SandboxConfig(isolation="none"))
        with pytest.raises(ValueError, match="intentional"):
            runner.run(_raising_payload)


# ───────────────────────────────────────────────────────────────────────────────
# "process" isolation: subprocess + timeout + crash containment
# ───────────────────────────────────────────────────────────────────────────────


_SKIP_WINDOWS_PROCESS = pytest.mark.skipif(
    sys.platform == "win32",
    reason="process isolation deferred to Phase 7.x on Windows (no fork)",
)


@_SKIP_WINDOWS_PROCESS
class TestProcessIsolationOk:
    def test_returns_result_from_child(self):
        runner = SandboxedTaskRunner(SandboxConfig(isolation="process", timeout_s=10.0))
        assert runner.run(_ok_payload, 7, 8) == 15

    def test_kwargs_propagate(self):
        runner = SandboxedTaskRunner(SandboxConfig(isolation="process", timeout_s=10.0))
        assert runner.run(_payload_with_kwargs, 3, scale=4) == 12

    def test_payload_exception_becomes_sandbox_crashed(self):
        runner = SandboxedTaskRunner(SandboxConfig(isolation="process", timeout_s=10.0))
        with pytest.raises(SandboxCrashed, match="ValueError"):
            runner.run(_raising_payload)


@_SKIP_WINDOWS_PROCESS
class TestProcessIsolationTimeout:
    def test_timeout_kills_child(self):
        runner = SandboxedTaskRunner(SandboxConfig(isolation="process", timeout_s=0.5))
        start = time.perf_counter()
        with pytest.raises(SandboxTimeout):
            runner.run(_slow_payload, 5.0)
        elapsed = time.perf_counter() - start
        # Should have aborted near the 0.5s timeout (with ~2s grace
        # for the kill+join). Definitely shouldn't take the full 5s.
        assert elapsed < 4.0, f"sandbox didn't kill quickly: elapsed={elapsed:.2f}s"

    def test_panal_survives_timeout(self):
        """After the sandboxed task is killed, the parent process
        keeps running — proven by being able to run a second task
        that succeeds."""
        runner = SandboxedTaskRunner(SandboxConfig(isolation="process", timeout_s=0.5))
        with pytest.raises(SandboxTimeout):
            runner.run(_slow_payload, 5.0)
        # Same runner, fresh task — must work.
        assert runner.run(_ok_payload, 1, 2) == 3


@_SKIP_WINDOWS_PROCESS
class TestProcessIsolationCrash:
    def test_os_exit_is_crash(self):
        """``os._exit(13)`` in the child = no result queued, non-zero
        exit code. Parent sees :class:`SandboxCrashed`."""
        runner = SandboxedTaskRunner(SandboxConfig(isolation="process", timeout_s=10.0))
        with pytest.raises(SandboxCrashed):
            runner.run(_exit_nonzero_payload)

    def test_segfault_is_crash(self):
        """SIGSEGV in the child surfaces as :class:`SandboxCrashed`,
        not a process-wide segfault. The brief's headline guarantee:
        crashes don't propagate to the panal."""
        runner = SandboxedTaskRunner(SandboxConfig(isolation="process", timeout_s=10.0))
        with pytest.raises(SandboxCrashed):
            runner.run(_segfault_payload)
        # Sanity: the test process is still alive (otherwise pytest
        # would have died).
        assert True

    def test_panal_survives_crash(self):
        runner = SandboxedTaskRunner(SandboxConfig(isolation="process", timeout_s=10.0))
        with pytest.raises(SandboxCrashed):
            runner.run(_segfault_payload)
        # Subsequent task works.
        assert runner.run(_ok_payload, 4, 5) == 9


# ───────────────────────────────────────────────────────────────────────────────
# Process isolation on Windows is deferred — verifies the failure mode is
# explicit (SandboxNotSupported) rather than a confusing pickle error.
# ───────────────────────────────────────────────────────────────────────────────


class TestProcessIsolationWindowsDeferred:
    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Phase 7.4 v1 raises SandboxNotSupported only on Windows",
    )
    def test_process_mode_on_windows_raises_not_supported(self):
        runner = SandboxedTaskRunner(SandboxConfig(isolation="process"))
        with pytest.raises(SandboxNotSupported, match="Phase 7.x"):
            runner.run(_ok_payload, 1, 2)


# ───────────────────────────────────────────────────────────────────────────────
# Stub modes: cgroup / job_object
# ───────────────────────────────────────────────────────────────────────────────


class TestStubbedIsolationModes:
    def test_cgroup_mode_raises_not_supported(self):
        runner = SandboxedTaskRunner(SandboxConfig(isolation="cgroup"))
        with pytest.raises(SandboxNotSupported, match="cgroup"):
            runner.run(_ok_payload, 1, 2)

    def test_job_object_mode_raises_not_supported(self):
        runner = SandboxedTaskRunner(SandboxConfig(isolation="job_object"))
        with pytest.raises(SandboxNotSupported, match="job_object"):
            runner.run(_ok_payload, 1, 2)


# ───────────────────────────────────────────────────────────────────────────────
# Platform probes
# ───────────────────────────────────────────────────────────────────────────────


class TestPlatformProbes:
    def test_cgroup_v2_available_returns_bool_no_raise(self):
        result = cgroup_v2_available()
        assert isinstance(result, bool)
        # On non-Linux it must be False.
        if sys.platform != "linux":
            assert result is False

    def test_job_objects_available_returns_bool_no_raise(self):
        result = job_objects_available()
        assert isinstance(result, bool)
        # On non-Windows it must be False.
        if sys.platform != "win32":
            assert result is False


# ───────────────────────────────────────────────────────────────────────────────
# Default config sanity
# ───────────────────────────────────────────────────────────────────────────────


class TestDefaultSandboxConfig:
    def test_defaults_are_safe(self):
        cfg = SandboxConfig()
        # Default = OFF. Per the brief: "Sandbox: por defecto OFF
        # (isolation='none'). Activar requiere config explícita."
        assert cfg.isolation == "none"
        assert cfg.timeout_s == 30.0
        assert cfg.memory_limit_mb is None
        assert cfg.cpu_limit_percent is None
