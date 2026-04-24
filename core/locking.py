"""
HOC Core · Locking
==================

Primitiva de sincronización read-write con timeout.

Provee ``RWLock``, un lock que admite múltiples lectores simultáneos o un
único escritor, con priorización de escritores para evitar starvation.

Extraído de ``core.py`` en Fase 3.3.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import ClassVar

__all__ = ["RWLock"]


# ═══════════════════════════════════════════════════════════════════════════════
# READ-WRITE LOCK (v3.0 - con timeout)
# ═══════════════════════════════════════════════════════════════════════════════


class RWLock:
    """
    Read-Write Lock que permite múltiples lectores o un único escritor.

    v3.0: Timeout configurable para evitar deadlocks.
    Prioriza escritores para evitar starvation.
    """

    __slots__ = ("_read_ready", "_readers", "_writer_active", "_writers_waiting")

    _DEFAULT_TIMEOUT: ClassVar[float] = 30.0

    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False

    @contextmanager
    def read_lock(self, timeout: float | None = None):
        """Adquiere lock de lectura con timeout opcional."""
        timeout = timeout or self._DEFAULT_TIMEOUT
        deadline = time.monotonic() + timeout

        with self._read_ready:
            while self._writer_active or self._writers_waiting > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"RWLock read_lock timeout after {timeout}s")
                self._read_ready.wait(timeout=remaining)
            self._readers += 1
        try:
            yield
        finally:
            with self._read_ready:
                self._readers -= 1
                if self._readers == 0:
                    self._read_ready.notify_all()

    @contextmanager
    def write_lock(self, timeout: float | None = None):
        """Adquiere lock de escritura con timeout opcional.

        Phase 1 fix (B1): refactor a try/finally — el bare ``except:`` original
        capturaba ``BaseException`` (KeyboardInterrupt, SystemExit) silenciando
        interrupciones legítimas. ``try/finally`` garantiza ``_writers_waiting -= 1``
        en cualquier camino sin tocar el flujo de excepciones.
        """
        timeout = timeout or self._DEFAULT_TIMEOUT
        deadline = time.monotonic() + timeout

        with self._read_ready:
            self._writers_waiting += 1
            try:
                while self._readers > 0 or self._writer_active:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"RWLock write_lock timeout after {timeout}s")
                    self._read_ready.wait(timeout=remaining)
                self._writer_active = True
            finally:
                self._writers_waiting -= 1
        try:
            yield
        finally:
            with self._read_ready:
                self._writer_active = False
                self._read_ready.notify_all()
