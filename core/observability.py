"""HOC observability — structured event logging (Phase 5.3).

This module wraps :mod:`structlog` to give the rest of HOC a
configurable structured event log. The convention is:

- ``log.info("event_name", **structured_fields)`` — every key event in
  the lifecycle of a cell, task, election, or migration goes through
  one of these calls. The event name is a dotted string
  (``cell.state_changed``, ``failover.migrate_started``, ...) so log
  consumers can filter by event family without parsing the message.
- Production callers pin JSON output via ``configure_logging(json=True)``
  (one log line == one JSON object, ``stdout``-friendly for ELK /
  Loki / etc.).
- Dev callers leave the default colored ``ConsoleRenderer`` for
  human-readable output during local runs.

The module also exposes ``get_event_logger(name)`` for callers that
want the structlog-typed logger directly. The standard library
``logging.getLogger`` calls peppered through HOC continue to work — we
do **not** reconfigure the stdlib root logger here so existing
``caplog``-based tests keep passing.

See ADR-011 for the full rationale (why structlog over loguru, why JSON
over OTLP for now, how this composes with future Prometheus + dashboard
in 5.4 / 5.7).
"""

from __future__ import annotations

import logging
from typing import Any

import structlog

__all__ = [
    "configure_logging",
    "get_event_logger",
    "EVENT_LOGGER_NAME",
]

# All HOC structured events go through this logger name. Filtering by
# this prefix is the easy way to get only the structured-event stream
# in a downstream collector.
EVENT_LOGGER_NAME = "hoc.events"

_configured: bool = False


def configure_logging(*, json: bool = False, level: int = logging.INFO) -> None:
    """Configure structlog for HOC's event channel.

    Idempotent: calling twice is a no-op.

    Args:
        json: ``True`` for line-delimited JSON output (production).
            ``False`` for the colored ``ConsoleRenderer`` (dev).
        level: Standard-library logging level; structlog respects it
            via :func:`structlog.stdlib.filter_by_level`.
    """
    global _configured
    if _configured:
        return

    # Configure stdlib so the structlog ``ProcessorFormatter`` flow can
    # forward records (and so any pre-existing ``logging.getLogger``
    # caller continues to function during the transition).
    logging.basicConfig(
        format="%(message)s",
        level=level,
    )

    # Processor chain. ``filter_by_level`` runs first so we drop low-
    # severity records before paying for the full processor pipeline.
    shared_processors: list[structlog.types.Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer(sort_keys=True)
    else:
        # ConsoleRenderer with colors — pleasant for local dev. Set
        # ``colors=False`` for plain text in CI logs.
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Caching the BoundLoggerLazyProxy snapshots the processor
        # chain at first call, which makes test-time reconfiguration
        # (configure_logging(json=True) after a default call) silently
        # ineffective. Disable so reset_for_tests + reconfigure works.
        cache_logger_on_first_use=False,
    )

    _configured = True


def get_event_logger(name: str = EVENT_LOGGER_NAME) -> structlog.stdlib.BoundLogger:
    """Return a structlog ``BoundLogger`` for the given channel.

    The logger name defaults to :data:`EVENT_LOGGER_NAME` so callers can
    filter by ``hoc.events`` in their log aggregator. Pass a sub-name
    (e.g. ``"hoc.events.failover"``) to scope further.

    Does **not** auto-call :func:`configure_logging`. If structlog is
    unconfigured the caller gets the library's defaults (ConsoleRenderer
    with bold + cyan); set the JSON mode explicitly at process startup
    via ``configure_logging(json=True)`` to opt in to production
    formatting. Auto-configuring here would inadvertently override an
    already-set JSON config when this module is imported via two paths
    (the ``from hoc.observability import …`` vs ``from observability
    import …`` ambiguity that the dual-layout package permits).
    """
    return structlog.get_logger(name)


def reset_for_tests() -> None:
    """Reset the module-level configured flag — for tests that want to
    re-configure structlog with different settings (e.g. ``json=True``
    after a previous default config).

    Not part of the public API. Tests in ``tests/test_logging.py`` use it.
    """
    global _configured
    structlog.reset_defaults()
    _configured = False


def log_cell_state_transition(
    coord: Any,
    from_state: str,
    to_state: str,
    *,
    cause: str | None = None,
    **extra: Any,
) -> None:
    """Helper: emit a ``cell.state_changed`` event with the canonical
    field shape used elsewhere in HOC.

    Called from :meth:`hoc.core.cells_base.HoneycombCell._set_state`.
    Extracted so the call site stays one line and the field names stay
    consistent across the codebase (operators querying the log can
    rely on the schema).
    """
    log = get_event_logger(f"{EVENT_LOGGER_NAME}.cell")
    log.info(
        "cell.state_changed",
        coord=str(coord),
        from_state=from_state,
        to_state=to_state,
        cause=cause,
        **extra,
    )
