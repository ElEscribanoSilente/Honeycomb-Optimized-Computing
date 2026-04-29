"""
HOC Storage — the :class:`StorageBackend` protocol and the default
:class:`MemoryBackend` implementation.

A backend stores raw ``bytes`` keyed by ``str``. It does not know
about HMAC envelopes, mscs framing, compression, or path validation —
all of those live one layer up in :mod:`hoc.memory.HoneyArchive`. This
keeps the security envelope (Phase 2 invariants) decoupled from the
storage medium and lets future Phase 6.x backends focus on
durability / throughput / concurrency without re-implementing the
crypto.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Minimal key-value persistence interface.

    Implementations must be **thread-safe**: concurrent
    ``put``/``get``/``delete``/``keys``/``__contains__`` from many
    threads is the expected operating mode (HOC is multi-threaded by
    default and a future async migration will only add load).

    ``keys()`` may return either a live iterator or a snapshot — the
    protocol does not require the iterator to reflect concurrent
    mutations. Callers that need a stable view should ``list(...)``
    the iterator before iterating.

    Methods:

    - ``put(key, value)`` — store / overwrite. No return value.
    - ``get(key)`` — return the stored bytes or ``None`` if absent.
    - ``delete(key)`` — return ``True`` if the key existed and was
      removed, ``False`` otherwise.
    - ``keys(prefix="")`` — yield all keys whose name starts with
      ``prefix`` (default: every key).
    - ``__contains__(key)`` — membership check; ``key in backend``.
    """

    def put(self, key: str, value: bytes) -> None: ...

    def get(self, key: str) -> bytes | None: ...

    def delete(self, key: str) -> bool: ...

    def keys(self, prefix: str = "") -> Iterator[str]: ...

    def __contains__(self, key: object) -> bool: ...


class MemoryBackend:
    """In-memory ``dict``-backed :class:`StorageBackend`.

    Phase 6.1 default. Preserves the pre-Phase-6 behaviour of
    ``HoneyArchive`` — every ``archive(...)`` / ``retrieve(...)``
    pair lives only as long as the process — while exposing the
    same protocol the SQLite / LMDB / S3 / Redis backends will use
    in later phases.

    Thread-safe via an internal :class:`threading.RLock`. The lock
    is reentrant so ``__contains__`` / ``keys`` callers can hold it
    transitively without deadlocking against ``put``/``get``/``delete``
    that may be called from a callback under the same lock.
    """

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self._lock = threading.RLock()

    def put(self, key: str, value: bytes) -> None:
        if not isinstance(key, str):
            raise TypeError(f"key must be str, got {type(key).__name__}")
        if not isinstance(value, (bytes, bytearray)):
            raise TypeError(f"value must be bytes, got {type(value).__name__}")
        with self._lock:
            self._data[key] = bytes(value)

    def get(self, key: str) -> bytes | None:
        with self._lock:
            return self._data.get(key)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def keys(self, prefix: str = "") -> Iterator[str]:
        with self._lock:
            # Snapshot to a list so callers can iterate without holding
            # the lock and without seeing mutations mid-iteration.
            snapshot = [k for k in self._data if k.startswith(prefix)]
        return iter(snapshot)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


__all__ = ["StorageBackend", "MemoryBackend"]
