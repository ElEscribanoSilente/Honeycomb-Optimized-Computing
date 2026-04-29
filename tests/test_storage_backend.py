"""Phase 6.1 — protocol compliance tests for ``hoc.storage`` backends.

The tests are parametrized on a ``backend`` fixture so each new
:class:`StorageBackend` implementation (Phase 6.2 SQLiteBackend, future
LMDB / S3 / Redis) can opt into the same contract by extending the
``params`` list. The compliance suite covers the five protocol methods
plus the thread-safety / iterator-snapshot guarantees the protocol
documentation makes.

``MemoryBackend`` also gets a small set of dedicated tests for its
specific affordances (``__len__``, type validation on ``put``).
"""

from __future__ import annotations

import threading

import pytest

from hoc.storage import MemoryBackend, SQLiteBackend, StorageBackend

# ───────────────────────────────────────────────────────────────────────────────
# Parametrized backend fixture
# ───────────────────────────────────────────────────────────────────────────────


# Phase 6.2: SQLiteBackend joins MemoryBackend in the contract suite.
# Future LMDB / S3 / Redis backends append their own param string here.
@pytest.fixture(params=["memory", "sqlite"])
def backend(request, tmp_path) -> StorageBackend:
    """Yields a fresh :class:`StorageBackend` instance per test.

    ``tmp_path`` is an unused pytest builtin for the memory backend
    but disk-backed backends (SQLite, LMDB, …) need a writable
    location. Receiving it here keeps the fixture signature uniform
    across params.
    """
    kind = request.param
    if kind == "memory":
        return MemoryBackend()
    if kind == "sqlite":
        return SQLiteBackend(tmp_path / "compliance.db")
    raise ValueError(f"Unknown backend kind: {kind!r}")


# ───────────────────────────────────────────────────────────────────────────────
# Protocol membership
# ───────────────────────────────────────────────────────────────────────────────


class TestProtocolMembership:
    def test_memory_backend_satisfies_protocol(self):
        # ``StorageBackend`` is ``runtime_checkable`` — isinstance works.
        assert isinstance(MemoryBackend(), StorageBackend)


# ───────────────────────────────────────────────────────────────────────────────
# Compliance: every backend must satisfy this contract
# ───────────────────────────────────────────────────────────────────────────────


class TestStorageBackendCompliance:
    def test_put_get_roundtrip(self, backend):
        backend.put("k1", b"hello")
        assert backend.get("k1") == b"hello"

    def test_get_missing_returns_none(self, backend):
        assert backend.get("missing") is None

    def test_put_overwrite(self, backend):
        backend.put("k", b"first")
        backend.put("k", b"second")
        assert backend.get("k") == b"second"

    def test_delete_existing_returns_true(self, backend):
        backend.put("k", b"v")
        assert backend.delete("k") is True
        assert backend.get("k") is None

    def test_delete_missing_returns_false(self, backend):
        assert backend.delete("ghost") is False

    def test_contains_existing(self, backend):
        backend.put("k", b"v")
        assert "k" in backend

    def test_contains_missing(self, backend):
        assert "ghost" not in backend

    def test_keys_empty_on_fresh_backend(self, backend):
        assert list(backend.keys()) == []

    def test_keys_returns_all(self, backend):
        for i in range(5):
            backend.put(f"k{i}", f"v{i}".encode())
        all_keys = list(backend.keys())
        assert sorted(all_keys) == [f"k{i}" for i in range(5)]

    def test_keys_with_prefix(self, backend):
        backend.put("user:1", b"alice")
        backend.put("user:2", b"bob")
        backend.put("session:abc", b"session_data")
        users = sorted(backend.keys("user:"))
        assert users == ["user:1", "user:2"]
        sessions = sorted(backend.keys("session:"))
        assert sessions == ["session:abc"]

    def test_keys_iterator_is_snapshot(self, backend):
        """The protocol allows backends to snapshot keys() output, so we
        only assert that pre-snapshot keys are visible — concurrent
        mutations are not required to appear in the iterator."""
        backend.put("a", b"1")
        backend.put("b", b"2")
        keys_iter = backend.keys()
        # If we add a key now, the snapshot may or may not include it —
        # the protocol leaves this implementation-defined.
        backend.put("c", b"3")
        materialized = list(keys_iter)
        # ``a`` and ``b`` were committed before the iterator was taken
        # so they MUST be visible.
        assert "a" in materialized
        assert "b" in materialized

    def test_delete_then_put_roundtrip(self, backend):
        backend.put("k", b"first")
        backend.delete("k")
        backend.put("k", b"second")
        assert backend.get("k") == b"second"

    def test_empty_value_is_valid(self, backend):
        backend.put("empty", b"")
        assert backend.get("empty") == b""
        assert "empty" in backend

    def test_binary_value_preserved(self, backend):
        # Non-utf8 bytes — must round-trip without coercion.
        payload = bytes(range(256))
        backend.put("bin", payload)
        assert backend.get("bin") == payload


# ───────────────────────────────────────────────────────────────────────────────
# Thread-safety: every backend must serialize concurrent put/get
# ───────────────────────────────────────────────────────────────────────────────


class TestStorageBackendThreadSafety:
    def test_concurrent_writers_no_corruption(self, backend):
        """N threads each writing M keys finish without exception, and
        all N*M values are readable afterwards."""
        n_threads = 8
        m_keys = 50
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

        # Every key must be readable.
        for tid in range(n_threads):
            for j in range(m_keys):
                assert backend.get(f"t{tid}:k{j}") == f"{tid}-{j}".encode()


# ───────────────────────────────────────────────────────────────────────────────
# MemoryBackend-specific tests
# ───────────────────────────────────────────────────────────────────────────────


class TestMemoryBackendSpecific:
    def test_len_empty(self):
        b = MemoryBackend()
        assert len(b) == 0

    def test_len_after_puts(self):
        b = MemoryBackend()
        b.put("a", b"1")
        b.put("b", b"2")
        assert len(b) == 2

    def test_len_after_overwrite(self):
        b = MemoryBackend()
        b.put("a", b"1")
        b.put("a", b"2")  # overwrite, not new entry
        assert len(b) == 1

    def test_put_rejects_non_str_key(self):
        b = MemoryBackend()
        with pytest.raises(TypeError):
            b.put(123, b"v")  # type: ignore[arg-type]

    def test_put_rejects_non_bytes_value(self):
        b = MemoryBackend()
        with pytest.raises(TypeError):
            b.put("k", "string")  # type: ignore[arg-type]

    def test_put_accepts_bytearray_and_stores_as_bytes(self):
        b = MemoryBackend()
        b.put("k", bytearray(b"abc"))
        got = b.get("k")
        assert isinstance(got, bytes)
        assert got == b"abc"


# ───────────────────────────────────────────────────────────────────────────────
# Integration: HoneyArchive can be constructed with an alternate backend
# ───────────────────────────────────────────────────────────────────────────────


class TestHoneyArchiveBackendIntegration:
    def test_default_backend_is_memory(self):
        from hoc.memory import HoneyArchive, MemoryConfig

        archive = HoneyArchive(MemoryConfig())
        assert isinstance(archive._backend, MemoryBackend)

    def test_custom_backend_is_used(self):
        from hoc.memory import HoneyArchive, MemoryConfig

        custom = MemoryBackend()
        archive = HoneyArchive(MemoryConfig(), backend=custom)
        assert archive._backend is custom

        # Round-trip: archive → retrieve via the custom backend.
        archive.archive("k", {"value": 42})
        assert archive.retrieve("k") == {"value": 42}
        # The bytes ended up in the custom backend (not a fresh dict).
        assert "k" in custom

    def test_swap_backend_does_not_break_metadata(self):
        from hoc.memory import HoneyArchive, MemoryConfig

        b = MemoryBackend()
        archive = HoneyArchive(MemoryConfig(), backend=b)
        archive.archive("k", "x" * 1000, metadata={"source": "test"})
        # _metadata is independent of the backend (it stores Python
        # objects, not bytes) but archive() still populates it.
        assert archive._metadata["k"]["source"] == "test"
        assert "size_original" in archive._metadata["k"]
        assert "size_compressed" in archive._metadata["k"]

    def test_get_stats_reads_from_backend(self):
        from hoc.memory import HoneyArchive, MemoryConfig

        archive = HoneyArchive(MemoryConfig())
        archive.archive("a", "x" * 100)
        archive.archive("b", "y" * 200)
        stats = archive.get_stats()
        assert stats["items"] == 2

    def test_delete_keeps_metadata_in_sync(self):
        from hoc.memory import HoneyArchive, MemoryConfig

        archive = HoneyArchive(MemoryConfig())
        archive.archive("k", "v")
        archive.delete("k")
        assert archive.retrieve("k") is None
        assert "k" not in archive._metadata
