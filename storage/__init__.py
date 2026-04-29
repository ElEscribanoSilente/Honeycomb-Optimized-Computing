"""
HOC Storage subpackage — pluggable persistence backends (Phase 6.1+).

Phase 6 introduces a tiny key-value protocol that the persistence
layer (``HoneyArchive``) sits on top of. The default backend
(``MemoryBackend``) is a thread-safe ``dict`` wrapper — it preserves
the pre-Phase-6 in-memory behaviour byte-for-byte. Future phases swap
in real persistence (SQLite, LMDB, S3, Redis) without touching the
HMAC + mscs framing layer that lives one floor up.

Public surface:

- :class:`StorageBackend` — the protocol every backend implements.
- :class:`MemoryBackend` — default in-memory implementation.

The boundary mirrors the conventions already used in the codebase:

- ``hoc.security`` is the single import site for ``mscs``.
- ``hoc.state_machines.base`` is the single import site for
  ``tramoya``.
- ``hoc.core.observability`` is the single import site for
  ``structlog``.

``hoc.storage.*`` follows suit: the protocol is here, concrete
implementations may import their drivers (``sqlite3``, ``lmdb``,
``boto3``…) but the rest of HOC only sees ``StorageBackend``.
"""

from __future__ import annotations

from .base import MemoryBackend, StorageBackend
from .checkpoint import decode_blob, encode_blob
from .sqlite import SQLiteBackend

__all__ = [
    "StorageBackend",
    "MemoryBackend",
    "SQLiteBackend",
    "encode_blob",
    "decode_blob",
]
