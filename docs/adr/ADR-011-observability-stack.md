# ADR-011: Observability stack — structlog now, Prometheus + dashboard later

- **Status**: Accepted
- **Date**: 2026-04-26
- **Phase**: Phase 5

## Context

Phase 5's brief positioned observability as the "phase that makes HOC
operable without reading source code". Three layers were on the table:

1. **Structured logs** — every meaningful state transition emits a
   typed event with a stable schema, so log aggregators (Loki / ELK /
   Datadog) can filter by event family and chart by field.
2. **Metrics** — counters / gauges / histograms scraped by a TSDB
   (Prometheus / Grafana) for dashboards + alerting.
3. **Traces** — distributed tracing via OTLP for end-to-end latency
   attribution.

The brief specified the first two as Phase 5 deliverables and named
candidate libraries (structlog, prometheus_client). Tracing was
implicitly deferred — HOC is single-process today and the cost of
adding OpenTelemetry SDK + collector to a library that does not yet
do RPC is hard to justify.

This ADR captures the per-layer choices and their rationale.

## Decision

### Structured logs — ``structlog``

Phase 5.3 ships ``hoc/core/observability.py``, a thin wrapper over
[structlog](https://www.structlog.org/). The wrapper exposes:

- ``configure_logging(json: bool = False, level: int = INFO)`` — call
  once at startup. ``json=True`` pins line-delimited JSON output for
  production; ``False`` (default) gives the ``ConsoleRenderer`` for
  human-readable dev output.
- ``get_event_logger(name="hoc.events")`` — returns a structlog
  ``BoundLogger``. Sub-channels (``hoc.events.cell``,
  ``hoc.events.failover``, ``hoc.events.election``) scope further.
- ``log_cell_state_transition(coord, from_state, to_state)`` — helper
  used by ``HoneycombCell._set_state`` so the event-name + field-name
  schema stays stable across the codebase.

Wired event sites (Phase 5.3):

| Event              | Source                                 | Fields                                    |
|--------------------|----------------------------------------|-------------------------------------------|
| ``cell.state_changed`` | ``HoneycombCell._set_state``        | coord, from_state, to_state, cause        |
| ``cell.sealed``    | ``HoneycombCell.seal``                 | coord, reason, ticks_processed, error_count, vcores_drained, age_seconds |
| ``failover.migrate_started`` | ``CellFailover._migrate_work`` | source, target, original_state            |
| ``failover.migrate_completed`` | idem                          | source, target, vcores_migrated, result, [error] |
| ``election.started`` | ``QueenSuccession.elect_new_queen``  | term                                      |
| ``election.completed`` | idem                                | term, candidate_count, [winner], result   |

### Why structlog (and not loguru / OpenTelemetry / stdlib only)

- **structlog** provides typed-kwargs logging with pluggable
  processors, JSON and Console renderers, and stdlib integration.
  ~70 KB, MIT, zero transitive deps. Fits HOC's "small, sharp deps"
  posture (same family as ``mscs`` and ``tramoya``).
- **loguru** has a friendlier ergonomics surface but requires
  re-routing the stdlib root logger (intrusive across a codebase that
  already uses ``logging.getLogger(__name__)`` widely) and ships
  ``win32-setctime`` as a transitive dep on Windows. Rejected.
- **stdlib ``logging.LoggerAdapter`` + JSON formatter** would work,
  but the typed-kwargs ergonomics + the processor-chain customisation
  are what we want — re-implementing them on top of stdlib costs more
  code than depending on structlog. Rejected.
- **OpenTelemetry SDK** — over-scoped for Phase 5. The SDK + collector
  + exporter trio adds ~3 MB of deps and an out-of-process collector
  for value HOC cannot consume yet (single-process, no RPC). Phase 8
  (multi-node) will revisit; if traces are needed before then, the
  structlog calls already carry a ``timestamp`` field that downstream
  tools can correlate.

### Why ``hoc/core/observability.py`` (not ``hoc/observability.py``)

A top-level ``observability.py`` would inherit the dual-import issue
ADR-007 documents for ``state_machines/``: the
``package-dir = {hoc = "."}`` setting makes the cwd both a Python
package (``HOC``) and a sys.path entry, so ``observability.py`` is
discoverable as both ``observability`` and ``HOC.observability`` and
mypy bails with "Source file found twice".

For ``state_machines/`` the workaround is the explicit-file mypy
invocation and a directory exclude. Trying the same pattern for a
single top-level file failed (the Windows path matcher does not match
the cwd-relative regex on a leaf file the same way it does on a
directory tree). Moving the module into the existing ``core/``
subpackage avoids the issue structurally — relative imports from
``cells_base.py`` and ``resilience.py`` (``from .observability``,
``from .core.observability``) resolve to one module name only.

The trade-off: callers say ``from hoc.core.observability import …``
or use the ``hoc`` re-export (``from hoc import configure_logging``).
The slightly longer import path is acceptable given the alternative
is a maintenance hazard documented across two ADRs.

### Cache: ``cache_logger_on_first_use=False``

structlog's default is ``True``, which snapshots the processor chain
on first call to a given logger name. That breaks test-time
reconfiguration: a test that calls ``configure_logging(json=True)``
after a previous test left the default config silently keeps the
default renderer because the proxy was cached. We set ``False`` so
each ``log.info`` rebuilds from the current global config. The
performance cost is a per-call lookup of the global config dict —
negligible at HOC's logging volume.

### No auto-configure inside ``get_event_logger``

The function does **not** call ``configure_logging`` if the module
has not been configured. Previous version did, which silently
re-configured structlog to dev-mode when the module was imported via
two paths (``observability`` and ``hoc.observability``) — each path
has its own ``_configured`` flag, so the path that lost the race
re-configured. The dual-import dance was eliminated structurally by
moving the module into ``core/``, but the auto-configure removal
stays as defensive programming: production callers should call
``configure_logging(json=True)`` explicitly at startup, exactly once,
in the application entrypoint. Tests do the same.

### Metrics — ``prometheus_client`` deferred

Phase 5.4 specified Prometheus counters/gauges + a ``/metrics``
endpoint. **Deferred to Phase 5.4-followup or Phase 6** for budget
reasons — the structured-logs layer + the FSM wire-ups consumed the
session's effective wire-up budget. The deferred work is concrete:

- ``prometheus_client`` runtime dep (~50 KB, MIT, single dep).
- 5 collectors: ``hoc_cell_state_total{state, role}`` gauge,
  ``hoc_task_state_total{state}`` counter, ``hoc_migrations_total{result}``
  counter, ``hoc_election_duration_seconds`` histogram,
  ``hoc_pheromone_deposits_active`` gauge.
- ``start_metrics_server(port=9090)`` wrapping
  ``prometheus_client.start_http_server``.
- ``hoc-cli serve-metrics --port 9090`` entry point (depends on the
  Phase 4.9 CLI scaffold also deferred).

The structlog event names + fields are designed so a future Prometheus
collector can consume them via promtail / fluent-bit log-derived
metrics in the meantime, with no code change in HOC.

### Dashboard — deferred to Phase 6

Phase 5.7 was already marked optional in the brief. The planned stack
(FastAPI + HTMX + Mermaid live-updated) is moderate scope and not on
the critical path for operability. Deferred to Phase 6 alongside the
persistence work; the Mermaid export from Phase 4 already gives
contributors a static lifecycle reference.

## Alternatives considered

### structlog vs loguru

See above — structlog wins on dependency posture and stdlib
integration. loguru is the friendlier API but the trade-offs do not
fit HOC's existing ``logging.getLogger`` callsite distribution.

### Dispatch every event through the FSM observers (``HocStateMachine.subscribe``)

The wrapper API has ``subscribe(observer)``. We could attach a
structlog handler to every FSM instance. Considered and rejected for
Phase 5.3 because:

- Per-cell FSM instances mean N observers attached at scale (~10k
  cells) — each subscription holds a closure reference, and the
  observer interface is per-transition (one call per state mutation).
  The cost is small but non-zero.
- The structured-event helper at the wire-up site is more explicit:
  the schema is defined right next to the code that emits it. No
  observer indirection makes the contract easier to read.
- Future phases that add a Prometheus collector (5.4-followup) can
  use observers to count transitions without modifying call sites —
  that's the natural use of subscribe(). Phase 5.3 leaves the door
  open without committing.

### Bind contextvars (request_id-style threading)

structlog has ``contextvars.merge_contextvars`` for thread-local
context binding (e.g. attaching a ``correlation_id`` to every log
emitted within a tick). HOC is single-process today and the SwarmScheduler
tick boundaries are not formally request-scoped, so we did not enable
contextvars in the initial config. Phase 8 (multi-node) is the
natural time to revisit.

## Consequences

### Easier

- **Operators can answer "what is HOC doing right now?"** without
  reading source. ``hoc.events.*`` events name the state machines
  they observe; field names match the FSM state strings; JSON output
  flows into any log aggregator.
- **Future Prometheus collector is mechanical.** The event sites
  already exist; a 5.4 follow-up subscribes a counter to each one
  and the 5 collectors are configured.
- **No reconfiguration of the stdlib root logger.** Existing
  ``logging.getLogger(__name__)`` calls + ``caplog`` tests keep
  working. structlog adds the structured channel alongside.

### Harder

- **Two import paths to know about.** Callers do
  ``from hoc.core.observability import configure_logging`` (or the
  ``hoc`` re-export). Slightly longer than ``hoc.observability``
  would have been; the trade-off is documented in this ADR.
- **Production callers must call ``configure_logging`` at startup.**
  The auto-configure path was removed (see above). Forgetting the
  call leaves structlog in its library-default mode (ConsoleRenderer
  with bold + cyan), which produces non-JSON output even with a JSON
  log aggregator pointed at stdout. The fix is one line in the
  application entrypoint.
- **No Prometheus / dashboard yet.** Phase 5 closes with logs only;
  metric-based alerting + the dashboard ship in Phase 5.4-followup
  or Phase 6.

### Risk / follow-up

- **structlog upstream changes.** The processor-chain API is the
  stable public surface; we pin ``structlog>=25.0.0`` (current at
  time of writing). If a future release breaks the JSON renderer
  invocation we contain the blast radius to ``observability.py``.
- **Cache trade-off.** ``cache_logger_on_first_use=False`` adds a
  per-call dict lookup. If a profile shows it matters at HOC's
  logging volume, we can flip back to ``True`` and add a
  ``reconfigure()`` helper for tests that explicitly invalidates
  the cache. Today it does not matter.
- **Phase 5.4 closure of the deferred Prometheus work** is the
  natural follow-up. The event names + field shapes here are the
  contract a Prometheus collector consumes; if 5.4 is slid further
  out, the Phase 5 closure will document the gap.

## References

- ``hoc/core/observability.py`` — wrapper implementation.
- ``hoc/__init__.py`` — re-exports ``configure_logging``,
  ``get_event_logger``, ``EVENT_LOGGER_NAME``.
- ``hoc/core/cells_base.py:_set_state`` — first hookup site.
- ``hoc/core/cells_base.py:seal`` — second hookup site.
- ``hoc/resilience.py:_migrate_work`` — failover hookup.
- ``hoc/resilience.py:elect_new_queen`` — election hookup.
- ``tests/test_logging.py`` — 9 tests covering JSON output + per-event
  field shapes + idempotence.
- ``snapshot/PHASE_05_CLOSURE.md`` — Phase 5 closure.
- ADR-007 — dual-import workaround that informed the
  ``hoc/core/observability.py`` placement.
- ADR-012 — ``choreo --strict`` flip, complementary to this ADR.
