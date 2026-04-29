"""Phase 6.2 — :class:`SQLiteBackend` specific tests.

The cross-backend protocol contract is exercised by
``tests/test_storage_backend.py`` (the SQLite backend is appended to
``_BACKEND_FACTORIES`` there in Phase 6.2). This file covers the
SQLite-specific concerns:

- WAL mode is actually enabled and persistent.
- Schema versioning bootstraps a fresh database, and a second open
  on an existing database does not re-run migrations.
- Crash semantics: closing the backend mid-test and reopening
  recovers all committed data (the WAL replay path).
- Concurrent writers across multiple threads do not lose updates.
- Integration with :class:`hoc.memory.HoneyArchive`.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from hoc.memory import HoneyArchive, MemoryConfig
from hoc.storage import SQLiteBackend, StorageBackend
from hoc.storage.sqlite import SCHEMA_VERSION

# ───────────────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "honey.db"


@pytest.fixture
def backend(db_path: Path) -> SQLiteBackend:
    return SQLiteBackend(db_path)


# ───────────────────────────────────────────────────────────────────────────────
# Basic roundtrips
# ───────────────────────────────────────────────────────────────────────────────


class TestSQLiteBackendBasics:
    def test_satisfies_storage_backend_protocol(self, backend):
        assert isinstance(backend, StorageBackend)

    def test_in_memory_database(self):
        # ``:memory:`` works for single-thread quick tests; WAL mode
        # is correctly skipped (in-memory DBs have no journal file).
        backend = SQLiteBackend(":memory:")
        backend.put("k", b"v")
        assert backend.get("k") == b"v"

    def test_put_get_roundtrip(self, backend):
        backend.put("k1", b"hello")
        assert backend.get("k1") == b"hello"

    def test_put_overwrite_preserves_created_at(self, backend, db_path):
        # Insert, fetch created_at, update, verify created_at unchanged
        # (only updated_at moves on conflict — ON CONFLICT … DO UPDATE
        # leaves created_at alone by design).
        backend.put("k", b"v1")
        # Direct DB read — bypass the protocol to inspect the schema.
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute(
            "SELECT created_at, updated_at FROM honey_archive WHERE key = ?",
            ("k",),
        )
        created_v1, updated_v1 = cur.fetchone()
        conn.close()
        # Re-put: created_at stays, updated_at advances.
        import time

        time.sleep(0.01)
        backend.put("k", b"v2")
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute(
            "SELECT created_at, updated_at FROM honey_archive WHERE key = ?",
            ("k",),
        )
        created_v2, updated_v2 = cur.fetchone()
        conn.close()
        assert created_v2 == pytest.approx(created_v1)
        assert updated_v2 > updated_v1

    def test_delete_existing(self, backend):
        backend.put("k", b"v")
        assert backend.delete("k") is True
        assert backend.get("k") is None

    def test_delete_missing(self, backend):
        assert backend.delete("ghost") is False

    def test_contains(self, backend):
        backend.put("k", b"v")
        assert "k" in backend
        assert "ghost" not in backend
        # Non-str key returns False (does not raise) — graceful.
        assert 123 not in backend  # type: ignore[operator]

    def test_keys_with_prefix(self, backend):
        backend.put("user:1", b"a")
        backend.put("user:2", b"b")
        backend.put("session:abc", b"s")
        users = list(backend.keys("user:"))
        assert sorted(users) == ["user:1", "user:2"]

    def test_keys_prefix_escapes_sql_wildcards(self, backend):
        # Phase 6.2: SQL ``%`` and ``_`` are LIKE wildcards. The
        # backend escapes them in the user-supplied prefix so a
        # literal "50%" prefix matches only keys that actually start
        # with "50%", not "any 2-char start with 5".
        backend.put("50%off", b"discount")
        backend.put("5xoff", b"different")
        results = list(backend.keys("50%"))
        assert results == ["50%off"]

    def test_put_rejects_non_str_key(self, backend):
        with pytest.raises(TypeError):
            backend.put(123, b"v")  # type: ignore[arg-type]

    def test_put_rejects_non_bytes_value(self, backend):
        with pytest.raises(TypeError):
            backend.put("k", "string")  # type: ignore[arg-type]

    def test_binary_value_preserved(self, backend):
        payload = bytes(range(256))
        backend.put("bin", payload)
        assert backend.get("bin") == payload


# ───────────────────────────────────────────────────────────────────────────────
# Schema versioning
# ───────────────────────────────────────────────────────────────────────────────


class TestSchemaVersioning:
    def test_fresh_db_creates_schema(self, db_path):
        SQLiteBackend(db_path)
        # Inspect the schema directly.
        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        conn.close()
        assert "honey_archive" in tables
        assert "_schema_version" in tables

    def test_schema_version_recorded(self, db_path):
        SQLiteBackend(db_path)
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute("SELECT version FROM _schema_version")
        row = cur.fetchone()
        conn.close()
        assert row[0] == SCHEMA_VERSION

    def test_index_exists(self, db_path):
        SQLiteBackend(db_path)
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute(
            "SELECT name FROM sqlite_master " "WHERE type='index' AND tbl_name='honey_archive'"
        )
        index_names = {row[0] for row in cur.fetchall()}
        conn.close()
        assert "idx_honey_archive_created_at" in index_names

    def test_reopen_does_not_remigrate(self, db_path):
        # First open: schema created. Insert one row.
        b1 = SQLiteBackend(db_path)
        b1.put("k", b"v")
        b1.close()
        # Second open: schema version is read, no migration runs.
        # If the v1 migration were re-run, the data would be
        # preserved (CREATE TABLE IF NOT EXISTS) but the version
        # row would be inserted twice. We guard against that by
        # asserting the version table has exactly one row.
        b2 = SQLiteBackend(db_path)
        assert b2.get("k") == b"v"
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute("SELECT COUNT(*) FROM _schema_version")
        count = cur.fetchone()[0]
        conn.close()
        b2.close()
        assert count == 1

    def test_v0_db_migrates_to_v1(self, db_path):
        """A pre-Phase-6 database (no _schema_version table) should be
        treated as version 0 and migrated to v1 on first open."""
        # Create a db file with no schema at all.
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE unrelated (x INTEGER)")
        conn.commit()
        conn.close()

        # Open with SQLiteBackend — should run migrate_to_v1.
        backend = SQLiteBackend(db_path)
        assert backend._read_schema_version(backend._connection()) == 1

        # Honey archive table now exists alongside the unrelated one.
        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        conn.close()
        assert "honey_archive" in tables
        assert "unrelated" in tables  # untouched
        backend.close()


# ───────────────────────────────────────────────────────────────────────────────
# WAL + crash recovery
# ───────────────────────────────────────────────────────────────────────────────


class TestWALAndCrashRecovery:
    def test_wal_mode_enabled_for_file_db(self, db_path):
        backend = SQLiteBackend(db_path)
        conn = backend._connection()
        cur = conn.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0].lower()
        backend.close()
        assert mode == "wal"

    def test_wal_mode_skipped_for_memory_db(self):
        backend = SQLiteBackend(":memory:")
        conn = backend._connection()
        cur = conn.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0].lower()
        # In-memory databases use ``memory`` journal mode by default.
        assert mode == "memory"

    def test_committed_data_survives_close_and_reopen(self, db_path):
        # Simulate "process crash": close all connections without
        # explicit fsync. WAL durability says committed transactions
        # survive a crash up to the last successful commit.
        b1 = SQLiteBackend(db_path)
        for i in range(20):
            b1.put(f"k{i}", f"v{i}".encode())
        b1.close()
        # Reopen: every value comes back.
        b2 = SQLiteBackend(db_path)
        for i in range(20):
            assert b2.get(f"k{i}") == f"v{i}".encode()
        b2.close()

    def test_concurrent_readers_during_write(self, db_path):
        """WAL gives readers a consistent snapshot while a writer is
        active. Spawn one writer + a few readers and verify that no
        reader sees a corrupted partial write."""
        backend = SQLiteBackend(db_path)
        # Pre-populate so readers have something to find.
        for i in range(10):
            backend.put(f"k{i}", f"initial-{i}".encode())

        errors: list[BaseException] = []
        stop = threading.Event()

        def writer():
            try:
                # Each iteration overwrites with a new value. No reader
                # should ever see a bytes value that isn't either
                # "initial-N" or "updated-N".
                for cycle in range(20):
                    for i in range(10):
                        backend.put(f"k{i}", f"updated-{cycle}-{i}".encode())
                stop.set()
            except Exception as e:
                errors.append(e)
                stop.set()

        def reader():
            try:
                while not stop.is_set():
                    for i in range(10):
                        v = backend.get(f"k{i}")
                        # Either the initial seed or a writer-cycle value;
                        # never None (we pre-populated) and never garbage.
                        assert v is not None
                        decoded = v.decode()
                        assert decoded.startswith(("initial-", "updated-"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        backend.close()
        assert errors == []


# ───────────────────────────────────────────────────────────────────────────────
# Concurrent writes
# ───────────────────────────────────────────────────────────────────────────────


class TestConcurrentWrites:
    def test_no_lost_updates_under_load(self, backend):
        n_threads = 6
        m_keys = 30
        errors: list[BaseException] = []

        def worker(tid: int) -> None:
            try:
                for j in range(m_keys):
                    backend.put(f"t{tid}:k{j}", f"{tid}-{j}".encode())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # Verify every key is readable.
        for tid in range(n_threads):
            for j in range(m_keys):
                assert backend.get(f"t{tid}:k{j}") == f"{tid}-{j}".encode()


# ───────────────────────────────────────────────────────────────────────────────
# HoneyArchive integration
# ───────────────────────────────────────────────────────────────────────────────


class TestHoneyArchiveOnSQLite:
    def test_round_trip_through_archive(self, db_path):
        backend = SQLiteBackend(db_path)
        archive = HoneyArchive(MemoryConfig(), backend=backend)
        archive.archive("entity:1", {"name": "alice", "score": 42})
        # Reading goes through HMAC verification + mscs deserialization.
        assert archive.retrieve("entity:1") == {"name": "alice", "score": 42}
        # Bytes physically live in the SQLite DB.
        assert "entity:1" in backend
        backend.close()

    def test_persistence_across_archive_instances(self, db_path):
        """Two HoneyArchive instances pointing at the same SQLite file
        must read each other's archived values — the whole point of
        persistence."""
        b1 = SQLiteBackend(db_path)
        a1 = HoneyArchive(MemoryConfig(), backend=b1)
        a1.archive("k", {"v": 1})
        b1.close()

        b2 = SQLiteBackend(db_path)
        a2 = HoneyArchive(MemoryConfig(), backend=b2)
        # Note: ``_metadata`` is per-instance (in-memory dict on the
        # archive, not on the backend), so cross-instance metadata is
        # NOT preserved. The archived value itself is.
        assert a2.retrieve("k") == {"v": 1}
        b2.close()

    def test_delete_removes_from_backend(self, db_path):
        backend = SQLiteBackend(db_path)
        archive = HoneyArchive(MemoryConfig(), backend=backend)
        archive.archive("k", "v")
        archive.delete("k")
        assert "k" not in backend
        backend.close()
