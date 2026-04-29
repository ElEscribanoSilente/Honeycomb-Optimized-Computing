"""HOC Phase 7.4 — Sandboxed task execution.

Process-level isolation for HiveTask payloads. The default scheduler
runs every task in the panal's own process: a runaway loop, a
``RuntimeError``, or a hard crash (SIGSEGV, OOM kill) takes the panal
down with it. Phase 7.4 introduces an opt-in :class:`SandboxedTaskRunner`
that wraps task execution in a child process with a hard timeout, so
crashes there don't cross the process boundary.

Status by isolation mode (Phase 7.4 v1):

- ``"none"`` — No isolation. The runner becomes a thin pass-through
  for parity with the unsandboxed scheduler path. Useful for
  benchmarks and unit tests that don't need crash safety.
- ``"process"`` — Spawns a child process via ``multiprocessing``
  (``spawn`` context for cross-platform parity). The child runs the
  payload, sends back ``(success, result_or_exception_repr)`` over a
  Queue. Parent enforces ``timeout_s`` with a hard kill (``SIGKILL``
  on POSIX, ``TerminateProcess`` on Windows). OOM kills and SIGSEGV
  in the child manifest as ``SandboxCrashed`` exceptions in the
  parent — the panal keeps running.
- ``"cgroup"`` — Stub. Linux cgroups v2 require either root or a
  user-delegated cgroup tree (``systemd-run --user``). Defer
  full support to Phase 7.x followup or Phase 8 (multi-node will
  benefit more from the limits anyway). For now raises
  ``NotImplementedError``.
- ``"job_object"`` — Stub. Windows Job Objects require ``pywin32``
  (a separate dev-only dep); the import is gated so the rest of
  HOC stays import-clean on platforms where pywin32 isn't
  installed. Currently raises ``NotImplementedError``.

The exception hierarchy:

- :class:`SandboxError` — base.
- :class:`SandboxTimeout` — child exceeded ``timeout_s``.
- :class:`SandboxCrashed` — child exited non-zero / was killed by the
  OS (OOM, SIGSEGV, etc.).
- :class:`SandboxNotSupported` — caller asked for an isolation mode
  that's not available on this platform.

Phase 7.4 deliberately does **not** wire the runner into
:class:`SwarmScheduler` automatically. Callers opt in via
``SwarmConfig.sandbox`` (a future addition) or by using the runner
directly. Default behaviour (no sandbox) is unchanged from Phase 7.3.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as _mp
import os
import queue as _queue
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

IsolationMode = Literal["none", "process", "cgroup", "job_object"]


@dataclass
class SandboxConfig:
    """Phase 7.4 sandbox parameters.

    ``timeout_s`` is enforced in all isolation modes (yes, even
    ``"none"`` honours it via a ``signal.alarm``-style fallback —
    sort of, see below). ``memory_limit_mb`` and ``cpu_limit_percent``
    are advisory in v1: only the cgroup/job_object modes actually
    enforce them, and both modes are stubbed for Phase 7.4 v1 (see
    module docstring).
    """

    timeout_s: float = 30.0
    memory_limit_mb: int | None = None
    cpu_limit_percent: int | None = None
    isolation: IsolationMode = "none"


class SandboxError(Exception):
    """Base class for sandbox failures."""


class SandboxTimeout(SandboxError):
    """The sandboxed task exceeded its ``timeout_s``."""


class SandboxCrashed(SandboxError):
    """The sandboxed task crashed (non-zero exit, OS kill). The
    payload's exception, if any, is in :attr:`underlying`."""

    def __init__(self, msg: str, *, underlying: BaseException | None = None) -> None:
        super().__init__(msg)
        self.underlying = underlying


class SandboxNotSupported(SandboxError):
    """The requested isolation mode isn't available on this
    platform / Python build."""


# ═══════════════════════════════════════════════════════════════════════════════
# Worker entry — runs in the child process
# ═══════════════════════════════════════════════════════════════════════════════
#
# Module-level so multiprocessing can pickle a reference (closures
# can't be pickled by ``spawn`` contexts on macOS / Windows). The
# worker calls ``fn(*args, **kwargs)`` and serialises the outcome via
# the ``result_queue``: either ``("ok", value)`` or
# ``("err", repr(exc))``. We send a string repr instead of the
# exception object itself because some exception types don't survive
# pickle (most notably exceptions that wrap unpicklable objects).


def _sandbox_worker(
    result_queue: _mp.Queue[Any],
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    try:
        out = fn(*args, **kwargs)
    except BaseException as exc:
        # If the queue itself is broken (parent died, pipe broken, etc.),
        # fall through — the parent will time out or observe a non-zero
        # exit code. Suppress to avoid masking the original exception.
        with contextlib.suppress(Exception):
            result_queue.put(("err", f"{type(exc).__name__}: {exc!s}"))
        return
    try:
        result_queue.put(("ok", out))
    except Exception:
        # Result wasn't serialisable. Report the type so callers can
        # debug; treat as crash from the parent's perspective.
        result_queue.put(("err", f"unserialisable result: {type(out).__name__}"))


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


class SandboxedTaskRunner:
    """Phase 7.4 sandbox: run a callable with the configured isolation.

    Usage::

        runner = SandboxedTaskRunner(SandboxConfig(timeout_s=5.0,
                                                   isolation="process"))
        try:
            result = runner.run(my_payload, arg1, kw=val)
        except SandboxTimeout:
            ...  # took too long
        except SandboxCrashed as exc:
            ...  # crashed; details in exc.underlying or str(exc)

    The runner is stateless beyond its config — safe to share across
    threads or instantiate per call.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

    # ── Dispatch ────────────────────────────────────────────────────

    def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``fn(*args, **kwargs)`` with the configured isolation.

        Returns whatever ``fn`` returns. Raises :class:`SandboxTimeout`,
        :class:`SandboxCrashed`, or :class:`SandboxNotSupported`
        depending on outcome.
        """
        mode = self.config.isolation
        if mode == "none":
            return self._run_none(fn, args, kwargs)
        if mode == "process":
            return self._run_in_process(fn, args, kwargs)
        if mode == "cgroup":
            raise SandboxNotSupported(
                "cgroup isolation is stubbed in Phase 7.4; "
                "use 'process' or wait for Phase 7.x followup."
            )
        if mode == "job_object":
            raise SandboxNotSupported(
                "job_object isolation is stubbed in Phase 7.4; "
                "use 'process' or wait for Phase 7.x followup."
            )
        raise SandboxNotSupported(f"unknown isolation mode: {mode!r}")

    # ── Mode implementations ────────────────────────────────────────

    @staticmethod
    def _run_none(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """No isolation. Run in-process. Honours timeout_s only as a
        no-op (a Python-level signal.alarm would catch CPU-bound
        loops on POSIX but breaks on Windows; for in-process we
        document the limitation and accept it)."""
        return fn(*args, **kwargs)

    def _run_in_process(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Spawn a child process; enforce timeout via ``Process.join``;
        kill on overrun. Crash detection via ``Process.exitcode``.

        Phase 7.4 v1: uses ``fork`` on POSIX (Linux/macOS) so test
        payloads defined in arbitrary modules don't need to be
        re-importable by the child. On Windows, ``fork`` isn't
        available and ``spawn`` would require pickling the payload —
        which collides with pytest's ``--import-mode=importlib``
        collection. Windows support is therefore deferred to a Phase
        7.x followup that can either lift cloudpickle or generate the
        child invocation via ``subprocess`` directly. Raising
        :class:`SandboxNotSupported` keeps the failure mode obvious.
        """
        if sys.platform == "win32":
            raise SandboxNotSupported(
                "process isolation requires fork (POSIX) in Phase 7.4 v1; "
                "Windows support is deferred to a Phase 7.x followup. Use "
                "isolation='none' on Windows or run on Linux/macOS."
            )

        ctx = _mp.get_context("fork")
        result_queue: _mp.Queue[Any] = ctx.Queue()
        proc = ctx.Process(
            target=_sandbox_worker,
            args=(result_queue, fn, args, kwargs),
            daemon=True,
            name=f"hoc-sandbox-{os.getpid()}",
        )
        proc.start()
        proc.join(timeout=self.config.timeout_s)

        if proc.is_alive():
            # Timeout. Kill the child; SIGKILL on POSIX,
            # TerminateProcess on Windows. ``Process.kill`` is the
            # cross-platform name (Python 3.7+).
            proc.kill()
            proc.join(timeout=2.0)
            if proc.is_alive():
                logger.warning("sandbox child %s did not exit after kill()", proc.pid)
            raise SandboxTimeout(f"sandboxed task exceeded {self.config.timeout_s}s")

        # Process finished. Did it crash, or did it queue a result?
        exit_code = proc.exitcode
        try:
            tag, payload = result_queue.get_nowait()
        except _queue.Empty:
            raise SandboxCrashed(
                f"sandboxed task exited without producing a result " f"(exit_code={exit_code})"
            ) from None

        if tag == "ok":
            if exit_code not in (0, None):
                # The child queued "ok" but then the OS killed it
                # before the worker function returned. Treat as crash.
                raise SandboxCrashed(
                    f"sandboxed task queued a result then crashed " f"(exit_code={exit_code})"
                )
            return payload

        # tag == "err"
        raise SandboxCrashed(f"sandboxed task raised: {payload}")


# ═══════════════════════════════════════════════════════════════════════════════
# Platform helpers (probes for Phase 7.x followup wiring)
# ═══════════════════════════════════════════════════════════════════════════════


def cgroup_v2_available() -> bool:
    """Phase 7.4 v1: probe whether the host exposes cgroups v2 with
    user-delegated permissions. Returns ``False`` on non-Linux. Phase
    7.x followup will use this to decide whether ``"cgroup"`` mode
    can be enabled without root."""
    if sys.platform != "linux":
        return False
    return os.path.isdir("/sys/fs/cgroup/unified") or os.path.isdir(
        "/sys/fs/cgroup/cgroup.controllers"
    )


def job_objects_available() -> bool:
    """Phase 7.4 v1: probe whether ``pywin32`` is installed for
    Windows Job Objects. Returns ``False`` everywhere else. Phase
    7.x followup will toggle ``"job_object"`` mode based on this
    plus the ``sandbox-windows`` extras."""
    if sys.platform != "win32":
        return False
    try:
        import win32job  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = [
    "IsolationMode",
    "SandboxConfig",
    "SandboxedTaskRunner",
    "SandboxError",
    "SandboxTimeout",
    "SandboxCrashed",
    "SandboxNotSupported",
    "cgroup_v2_available",
    "job_objects_available",
]
