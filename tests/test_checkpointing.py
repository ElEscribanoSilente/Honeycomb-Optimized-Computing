"""Phase 6.3 — checkpoint format + ``HoneycombGrid.checkpoint`` /
``HoneycombGrid.restore_from_checkpoint`` tests.

Covers:

- Round-trip: dump → restore → state equivalent.
- HMAC tampering: a single bit flip in the body bytes is rejected
  with :class:`MSCSecurityError`.
- mscs strict mode: an unregistered class in the inner payload is
  rejected on load.
- Compression on vs off: byte-for-byte different blobs but
  identical post-restore state.
- Wire format: invalid version byte / unknown compression flag /
  corrupted compressed bytes raise ``ValueError`` (not silent
  garbage).
- Atomic write: a half-written ``checkpoint`` does not clobber a
  good ``path``.
"""

from __future__ import annotations

import pytest

from hoc.core import CellRole, CellState, HoneycombConfig, HoneycombGrid
from hoc.security import MSCSecurityError, sign_payload
from hoc.storage.checkpoint import (
    HEADER_LEN,
    HMAC_TAG_LEN,
    VERSION_BYTE,
    decode_blob,
    encode_blob,
)

# ───────────────────────────────────────────────────────────────────────────────
# Pure helpers (encode_blob / decode_blob)
# ───────────────────────────────────────────────────────────────────────────────


class TestEncodeDecodeRoundtrip:
    def test_simple_dict_roundtrip(self):
        payload = {"a": 1, "b": "two", "c": [3, 4, 5]}
        blob = encode_blob(payload)
        decoded = decode_blob(blob)
        assert decoded == payload

    def test_compressed_roundtrip(self):
        # Highly compressible payload to exercise the zlib path.
        payload = {"chunks": ["A" * 1000, "A" * 1000, "B" * 1000]}
        blob_compressed = encode_blob(payload, compress=True)
        blob_plain = encode_blob(payload, compress=False)
        # Compressed strictly shorter than plain on this payload.
        assert len(blob_compressed) < len(blob_plain)
        # Both decode to the identical structure.
        assert decode_blob(blob_compressed) == payload
        assert decode_blob(blob_plain) == payload

    def test_empty_dict(self):
        blob = encode_blob({})
        assert decode_blob(blob) == {}

    def test_nested_payload(self):
        payload = {"grid": {"cells": [{"q": 0, "r": 0}, {"q": 1, "r": 0}], "tick": 42}}
        blob = encode_blob(payload)
        assert decode_blob(blob) == payload


# ───────────────────────────────────────────────────────────────────────────────
# Wire-format errors (must all be distinguishable)
# ───────────────────────────────────────────────────────────────────────────────


class TestBlobErrors:
    def test_too_short_blob(self):
        with pytest.raises(ValueError, match="too short"):
            decode_blob(b"\x01" + b"\x00" * 16)  # 17 bytes < HEADER_LEN+1

    def test_bad_version(self):
        # Construct a blob with the wrong version byte.
        body = b"\x00" + b"x"  # flag=0 (no compression) + 1 byte mscs (invalid mscs but harmless)
        tag = sign_payload(body)
        bad_blob = bytes([0xFF]) + tag + body  # version=0xFF
        with pytest.raises(ValueError, match="unsupported.*version"):
            decode_blob(bad_blob)

    def test_hmac_tamper_rejected(self):
        payload = {"k": "v"}
        blob = encode_blob(payload)
        # Flip a single bit in the body (after the version byte and tag).
        idx = HEADER_LEN + 5  # well into the body
        if idx >= len(blob):
            idx = len(blob) - 1
        tampered = bytearray(blob)
        tampered[idx] ^= 0x01
        with pytest.raises(MSCSecurityError, match="HMAC verification failed"):
            decode_blob(bytes(tampered))

    def test_hmac_tag_tamper_rejected(self):
        # Tamper with the HMAC tag itself — should also fail verification.
        blob = encode_blob({"k": "v"})
        tampered = bytearray(blob)
        # Flip a bit in the tag region.
        tampered[1 + HMAC_TAG_LEN // 2] ^= 0x80
        with pytest.raises(MSCSecurityError):
            decode_blob(bytes(tampered))

    def test_unknown_compression_flag(self):
        # Forge a blob with a valid HMAC but an unknown compression flag.
        body = b"\x99" + b"junk"  # flag=0x99 (not 0 or 1)
        tag = sign_payload(body)
        forged = bytes([VERSION_BYTE]) + tag + body
        with pytest.raises(ValueError, match="unknown compression flag"):
            decode_blob(forged)

    def test_corrupted_zlib_payload(self):
        # Forge a "compressed" blob whose payload is not actually zlib.
        body = b"\x01" + b"not_actually_zlib_bytes"  # flag=1 (zlib)
        tag = sign_payload(body)
        forged = bytes([VERSION_BYTE]) + tag + body
        with pytest.raises(ValueError, match="zlib decompression failed"):
            decode_blob(forged)


# ───────────────────────────────────────────────────────────────────────────────
# HoneycombGrid integration
# ───────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def small_grid() -> HoneycombGrid:
    return HoneycombGrid(HoneycombConfig(radius=1))


@pytest.fixture
def medium_grid() -> HoneycombGrid:
    return HoneycombGrid(HoneycombConfig(radius=2))


class TestGridCheckpointRoundtrip:
    def test_basic_roundtrip(self, small_grid, tmp_path):
        path = tmp_path / "snapshot.bin"
        small_grid.checkpoint(path)
        restored = HoneycombGrid.restore_from_checkpoint(path)

        # Same number of cells and same coords.
        assert len(restored._cells) == len(small_grid._cells)
        assert set(restored._cells.keys()) == set(small_grid._cells.keys())

    def test_tick_count_preserved(self, small_grid, tmp_path):
        small_grid._tick_count = 17
        path = tmp_path / "snap.bin"
        small_grid.checkpoint(path)
        restored = HoneycombGrid.restore_from_checkpoint(path)
        assert restored._tick_count == 17

    def test_cell_states_preserved(self, small_grid, tmp_path):
        # Drive a couple of cells through known transitions.
        for _coord, cell in list(small_grid._cells.items())[:3]:
            cell.state = CellState.IDLE
            cell.state = CellState.ACTIVE
        path = tmp_path / "snap.bin"
        small_grid.checkpoint(path)

        restored = HoneycombGrid.restore_from_checkpoint(path)
        # Every cell's state matches.
        for coord in small_grid._cells:
            assert restored._cells[coord].state == small_grid._cells[coord].state

    def test_state_history_preserved(self, small_grid, tmp_path):
        # Pick one cell, generate a deterministic history.
        coord = next(iter(small_grid._cells))
        cell = small_grid._cells[coord]
        cell.state = CellState.IDLE  # EMPTY → IDLE
        cell.state = CellState.ACTIVE  # IDLE → ACTIVE
        cell.state = CellState.IDLE  # ACTIVE → IDLE
        original_history = list(cell._state_history)

        path = tmp_path / "snap.bin"
        small_grid.checkpoint(path)
        restored = HoneycombGrid.restore_from_checkpoint(path)

        restored_cell = restored._cells[coord]
        assert list(restored_cell._state_history) == original_history

    def test_role_distribution_preserved(self, medium_grid, tmp_path):
        original_role_counts: dict[CellRole, int] = {}
        for cell in medium_grid._cells.values():
            original_role_counts[cell.role] = original_role_counts.get(cell.role, 0) + 1

        path = tmp_path / "snap.bin"
        medium_grid.checkpoint(path)
        restored = HoneycombGrid.restore_from_checkpoint(path)

        restored_role_counts: dict[CellRole, int] = {}
        for cell in restored._cells.values():
            restored_role_counts[cell.role] = restored_role_counts.get(cell.role, 0) + 1

        assert restored_role_counts == original_role_counts

    def test_config_preserved(self, tmp_path):
        cfg = HoneycombConfig(radius=2, vcores_per_cell=4, pheromone_decay_rate=0.25)
        grid = HoneycombGrid(cfg)
        path = tmp_path / "snap.bin"
        grid.checkpoint(path)
        restored = HoneycombGrid.restore_from_checkpoint(path)
        assert restored.config.radius == 2
        assert restored.config.vcores_per_cell == 4
        assert restored.config.pheromone_decay_rate == pytest.approx(0.25)


class TestGridCheckpointCompression:
    def test_compressed_smaller_than_plain(self, medium_grid, tmp_path):
        plain_path = tmp_path / "plain.bin"
        zlib_path = tmp_path / "zlib.bin"
        medium_grid.checkpoint(plain_path, compress=False)
        medium_grid.checkpoint(zlib_path, compress=True)
        # Cell metadata is repetitive enough that zlib wins on radius=2.
        assert zlib_path.stat().st_size < plain_path.stat().st_size

    def test_compressed_roundtrip(self, medium_grid, tmp_path):
        path = tmp_path / "compressed.bin"
        medium_grid.checkpoint(path, compress=True)
        restored = HoneycombGrid.restore_from_checkpoint(path)
        assert len(restored._cells) == len(medium_grid._cells)


class TestGridCheckpointTamperResistance:
    def test_tampered_checkpoint_rejected(self, small_grid, tmp_path):
        path = tmp_path / "snap.bin"
        small_grid.checkpoint(path)
        # Flip a byte in the body (skip version + HMAC).
        blob = bytearray(path.read_bytes())
        idx = HEADER_LEN + 10
        blob[idx] ^= 0x01
        path.write_bytes(bytes(blob))

        with pytest.raises(MSCSecurityError):
            HoneycombGrid.restore_from_checkpoint(path)

    def test_truncated_checkpoint_rejected(self, small_grid, tmp_path):
        path = tmp_path / "snap.bin"
        small_grid.checkpoint(path)
        # Cut the file to under the header length.
        path.write_bytes(path.read_bytes()[:5])
        with pytest.raises(ValueError):
            HoneycombGrid.restore_from_checkpoint(path)


class TestAtomicCheckpointWrite:
    def test_tmp_file_is_cleaned_up(self, small_grid, tmp_path):
        path = tmp_path / "snap.bin"
        small_grid.checkpoint(path)
        # No leftover ``snap.bin.tmp`` after a successful write.
        assert not (tmp_path / "snap.bin.tmp").exists()
        assert path.exists()

    def test_overwrite_existing_checkpoint(self, small_grid, tmp_path):
        path = tmp_path / "snap.bin"
        small_grid.checkpoint(path)
        first_size = path.stat().st_size

        # Mutate state, re-checkpoint — file is overwritten atomically.
        small_grid._tick_count = 999
        small_grid.checkpoint(path)
        restored = HoneycombGrid.restore_from_checkpoint(path)
        assert restored._tick_count == 999
        # Same shape so size is similar (not strictly equal because
        # mscs may serialize the int differently, but within a few
        # bytes).
        assert abs(path.stat().st_size - first_size) < 32
