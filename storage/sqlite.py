"""
HOC Storage — :class:`SQLiteBackend` (Phase 6.2).

Disk-backed :class:`StorageBackend` using stdlib ``sqlite3`` (no new
runtime deps). Designed for single-node production and dev portability:

- **WAL mode** for concurrent readers + one writer with crash-safe
  durability (the WAL file persists writes even if the process is
  killed mid-transaction).
- **Connection-per-thread** via :class:`threading.local`. SQLite
  connections are not thread-safe, but each thread gets its own
  connection (auto-created on first use) and the underlying database
  file serializes the writes.
- **Schema versioning** via a ``_schema_version`` table — future
  Phase 6.x migrations can bump the schema without breaking existing
  databases. Migrations run in a single transaction at startup.

The HMAC + mscs envelope stays on the layer above
(:class:`hoc.memory.HoneyArchive`); the backend stores the opaque
bytes verbatim under the supplied string key.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path

# Phase 6.2 schema version. Bump + add a migration in
# ``_run_migrations_to_current`` whenever the on-disk shape changes.
SCHEMA_VERSION: int = 1


class SQLiteBackend:
    """Disk-backed :class:`hoc.storage.StorageBackend`.

    Parameters
    ----------
    db_path:
        Filesystem path to the SQLite database file. Use
        ``":memory:"`` for an in-process throwaway database (single
        thread only — SQLite's in-memory mode does not share state
        across threads, so this is for unit tests only).

    The schema is initialized on construction. If the database
    already exists, its current schema version is read and any
    pending migrations are applied. Concurrent ``SQLiteBackend``
    instances pointing at the same file are safe (WAL serializes
    writes), but each instance keeps its own per-thread connection
    pool.

    Notes
    -----
    The connection is configured with ``isolation_level=None`` (Python
    side: autocommit) so that explicit ``BEGIN``/``COMMIT`` are not
    needed for our single-statement updates. SQLite still wraps each
    write in its own atomic transaction; WAL provides the crash-safety.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._tls = threading.local()
        self._init_lock = threading.Lock()
        self._initialized = False
        # Initialize schema eagerly so a fresh backend is ready to
        # accept put/get without a separate setup call.
        self._ensure_initialized()

    # ── Connection pool ───────────────────────────────────────────────────

    def _connection(self) -> sqlite3.Connection:
        """Per-thread SQLite connection. Auto-created on first use."""
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                self._db_path,
                isolation_level=None,  # autocommit — explicit txns not needed
                check_same_thread=True,  # one conn per thread is the contract
            )
            # WAL mode: concurrent readers + one writer, durable writes.
            # Skipped for ``:memory:`` databases (WAL is journal-based and
            # in-memory has no journal file).
            if self._db_path != ":memory:":
                conn.execute("PRAGMA journal_mode=WAL")
                # NORMAL keeps fsync only at WAL checkpoint time, not per
                # transaction. Safe under WAL: at most the last few
                # commits since the last checkpoint can be lost on a
                # power failure (process crash is fully recovered).
                conn.execute("PRAGMA synchronous=NORMAL")
            # Foreign keys are off in this schema, but turn on the pragma
            # so future migrations that add FKs work without surprises.
            conn.execute("PRAGMA foreign_keys=ON")
            self._tls.conn = conn
        return conn

    # ── Schema bootstrap + migrations ─────────────────────────────────────

    def _ensure_initialized(self) -> None:
        """Run pending migrations exactly once per backend instance."""
        with self._init_lock:
            if self._initialized:
                return
            conn = self._connection()
            current = self._read_schema_version(conn)
            if current < SCHEMA_VERSION:
                self._run_migrations_to_current(conn, current)
            self._initialized = True

    @staticmethod
    def _read_schema_version(conn: sqlite3.Connection) -> int:
        """Return the schema version stored in the DB, or 0 if the
        ``_schema_version`` table does not exist (fresh database)."""
        cur = conn.execute(
            "SELECT name FROM sqlite_master " "WHERE type='table' AND name='_schema_version'"
        )
        if cur.fetchone() is None:
            return 0
        cur = conn.execute("SELECT version FROM _schema_version ORDER BY version DESC LIMIT 1")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def _run_migrations_to_current(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Apply migrations sequentially from ``from_version + 1`` up to
        :data:`SCHEMA_VERSION`. Each migration is wrapped in a
        ``BEGIN``/``COMMIT`` so a partial migration cannot leave the
        DB half-applied — the next start retries from the same point."""
        if from_version < 1 <= SCHEMA_VERSION:
            self._migrate_to_v1(conn)
        # Future migrations:
        # if from_version < 2 <= SCHEMA_VERSION:
        #     self._migrate_to_v2(conn)
        # ...

    @staticmethod
    def _migrate_to_v1(conn: sqlite3.Connection) -> None:
        """v0 → v1: create ``honey_archive`` + ``_schema_version`` tables."""
        conn.execute("BEGIN")
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS honey_archive (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_honey_archive_created_at "
                "ON honey_archive(created_at)"
            )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at REAL NOT NULL
                )
                """)
            conn.execute(
                "INSERT INTO _schema_version (version, applied_at) VALUES (?, ?)",
                (1, time.time()),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    # ── StorageBackend protocol ───────────────────────────────────────────

    def put(self, key: str, value: bytes) -> None:
        if not isinstance(key, str):
            raise TypeError(f"key must be str, got {type(key).__name__}")
        if not isinstance(value, (bytes, bytearray)):
            raise TypeError(f"value must be bytes, got {type(value).__name__}")
        now = time.time()
        conn = self._connection()
        # ``ON CONFLICT … DO UPDATE`` is SQLite ≥3.24, available since
        # Python 3.7+. We leave ``created_at`` unchanged on conflict so
        # the originally inserted timestamp is preserved across updates.
        conn.execute(
            "INSERT INTO honey_archive (key, value, created_at, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "value=excluded.value, updated_at=excluded.updated_at",
            (key, bytes(value), now, now),
        )

    def get(self, key: str) -> bytes | None:
        conn = self._connection()
        cur = conn.execute("SELECT value FROM honey_archive WHERE key = ?", (key,))
        row = cur.fetchone()
        return bytes(row[0]) if row is not None else None

    def delete(self, key: str) -> bool:
        conn = self._connection()
        cur = conn.execute("DELETE FROM honey_archive WHERE key = ?", (key,))
        return cur.rowcount > 0

    def keys(self, prefix: str = "") -> Iterator[str]:
        conn = self._connection()
        if prefix:
            # Escape SQL LIKE wildcards in the user-supplied prefix so
            # ``"50%"`` matches the literal string "50%" rather than
            # "anything starting with 50". Backslash is the ESCAPE char.
            escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            cur = conn.execute(
                "SELECT key FROM honey_archive " "WHERE key LIKE ? ESCAPE '\\' " "ORDER BY key",
                (escaped + "%",),
            )
        else:
            cur = conn.execute("SELECT key FROM honey_archive ORDER BY key")
        # ``fetchall`` then iter so the cursor closes immediately and
        # the caller can iterate without holding the connection.
        rows = cur.fetchall()
        return iter(row[0] for row in rows)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        conn = self._connection()
        cur = conn.execute("SELECT 1 FROM honey_archive WHERE key = ? LIMIT 1", (key,))
        return cur.fetchone() is not None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close this thread's connection. Other threads keep theirs.

        Tests use this to flush WAL and release the file handle so
        the temporary database can be cleaned up. Production callers
        usually do not need to close — the connection is closed when
        the thread exits.
        """
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            conn.close()
            self._tls.conn = None

    def __repr__(self) -> str:
        return f"SQLiteBackend(db_path={self._db_path!r})"


__all__ = ["SQLiteBackend", "SCHEMA_VERSION"]
