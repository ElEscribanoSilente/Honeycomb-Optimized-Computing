"""
HOC Storage — checkpoint blob format (Phase 6.3, v2 in Phase 7.10).

Pure encode / decode helpers for HMAC-signed, optionally-compressed,
mscs-framed blobs. Knows nothing about :class:`HoneycombGrid` — the
caller passes any mscs-serializable Python object (typically a dict
of primitives) and gets back an opaque byte blob safe to commit to
disk, ship over the wire, or restore later.

Wire format (all fields are big-endian where applicable)::

    +---------+--------------+----------------+--------------------+
    | byte 0  | bytes 1..32  | byte 33        | bytes 34..end      |
    +=========+==============+================+====================+
    | version | hmac_tag     | compression_fl | payload (mscs;     |
    | (0x01,  | (32 bytes,   | (0=none,1=zlib)| optionally zlib-   |
    |  0x02)  | sha256-mac)  |                | compressed)        |
    +---------+--------------+----------------+--------------------+

The HMAC covers ``[compression_flag] || payload`` (the 33 bytes
starting at byte 33 in the blob). The version byte is intentionally
NOT covered — a future version bump will produce a different blob
shape; verifying the version-byte separately keeps the failure mode
distinguishable (version mismatch → ``ValueError``; tampered body →
:class:`MSCSecurityError`).

Versioning history
------------------

- ``0x01`` (Phase 6.3): grid-only payload (config + per-cell state).
- ``0x02`` (Phase 7.10): same wire shape, but the payload may carry
  a top-level ``"scheduler"`` key with the SwarmScheduler task queue
  state. Old (v1) blobs decode unchanged; new blobs always emit
  v2 even when the scheduler key is absent so downstream consumers
  can rely on a single advertised version.

Why HMAC outside the mscs envelope?
-----------------------------------

mscs already provides HMAC-SHA256 inside its own dump/load API. The
Phase 6.3 brief asks for "HMAC sobre el blob (reusa
security.sign_payload)" — i.e. cover the *whole* on-disk artefact,
including the compression byte and any zlib container, so a bit-flip
in the compressed bytes can't slip through to mscs's plaintext check.

We therefore call ``mscs.dumps`` *without* its hmac key (no inner
HMAC; the resulting bytes are still framed by mscs but unsigned),
optionally compress, prepend the compression flag, and HMAC-sign
the result via :func:`hoc.security.sign_payload`. ``decode_blob``
verifies the outer HMAC first; only then does it decompress and
``mscs.loads`` in strict mode.
"""

from __future__ import annotations

import zlib
from typing import Any, cast

import mscs as _mscs

from ..security import MSCSecurityError, sign_payload, verify_signature

VERSION_BYTE: int = 0x02  # Phase 7.10: bumped from 0x01
SUPPORTED_VERSIONS: frozenset[int] = frozenset({0x01, 0x02})
HMAC_TAG_LEN: int = 32  # HMAC-SHA256
HEADER_LEN: int = 1 + HMAC_TAG_LEN  # version (1) + tag (32) = 33

COMPRESSION_NONE: int = 0x00
COMPRESSION_ZLIB: int = 0x01


def encode_blob(payload: Any, *, compress: bool = False) -> bytes:
    """Encode ``payload`` into a HMAC-signed checkpoint blob.

    ``payload`` must be mscs-serializable. For Phase 6.3 the grid
    passes a ``dict[str, Any]`` of primitives; future phases may
    register extra classes.

    Parameters
    ----------
    compress:
        If ``True``, zlib-compress the mscs serialization before
        signing. Useful for large grids; for small ones the framing
        overhead exceeds the savings.
    """
    serialized = cast(bytes, _mscs.dumps(payload))
    if compress:
        serialized = zlib.compress(serialized, level=6)

    flag = COMPRESSION_ZLIB if compress else COMPRESSION_NONE
    body = bytes([flag]) + serialized
    tag = sign_payload(body)

    return bytes([VERSION_BYTE]) + tag + body


def decode_blob(blob: bytes) -> Any:
    """Verify + decode a checkpoint blob produced by :func:`encode_blob`.

    Order of checks (each failure raises a distinguishable exception):

    1. Length sanity — too short to contain the header → ``ValueError``.
    2. Version byte mismatch → ``ValueError``.
    3. HMAC tag mismatch → :class:`MSCSecurityError`. (Tampering check
       runs *before* decompression so a malicious blob can't OOM us
       via a zlib bomb.)
    4. Unknown compression flag → ``ValueError``.
    5. zlib decompression failure → ``ValueError``.
    6. mscs strict-mode rejection (unregistered class in payload) →
       re-raised verbatim.
    """
    if len(blob) < HEADER_LEN + 1:  # need at least version + tag + flag
        raise ValueError(f"checkpoint blob too short: {len(blob)} bytes")

    version = blob[0]
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported checkpoint version: 0x{version:02x}")

    tag = blob[1 : 1 + HMAC_TAG_LEN]
    body = blob[1 + HMAC_TAG_LEN :]

    if not verify_signature(body, tag):
        raise MSCSecurityError("checkpoint HMAC verification failed")

    flag = body[0]
    serialized = body[1:]

    if flag == COMPRESSION_NONE:
        pass
    elif flag == COMPRESSION_ZLIB:
        try:
            serialized = zlib.decompress(serialized)
        except zlib.error as exc:
            raise ValueError(f"zlib decompression failed: {exc}") from exc
    else:
        raise ValueError(f"unknown compression flag: 0x{flag:02x}")

    # mscs strict=True rejects unregistered classes — defensive guard
    # against a payload constructed from a different mscs registry.
    return _mscs.loads(serialized, strict=True)


__all__ = [
    "encode_blob",
    "decode_blob",
    "VERSION_BYTE",
    "SUPPORTED_VERSIONS",
    "HMAC_TAG_LEN",
    "HEADER_LEN",
    "COMPRESSION_NONE",
    "COMPRESSION_ZLIB",
]
