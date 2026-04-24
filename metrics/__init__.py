"""
HOC Metrics - Sistema de Métricas y Observabilidad
===================================================

Proporciona monitoreo, métricas y visualización del panal:

MÉTRICAS:
- HiveMetrics: Métricas globales del panal
- CellMetrics: Métricas por celda individual
- SwarmMetrics: Métricas del scheduler

VISUALIZACIÓN:
- HoneycombVisualizer: Renderizado del grid hexagonal
- HeatmapRenderer: Mapas de calor de carga/actividad
- FlowVisualizer: Visualización de flujos de comunicación

Estructura de métricas::

    ┌────────────────────────────────────────────────────────────┐
    │                     MetricsCollector                        │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
    │  │  Counter    │  │   Gauge     │  │  Histogram  │        │
    │  │  (events)   │  │  (current)  │  │  (distrib)  │        │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
    │         │                │                │                │
    │         └────────────────┼────────────────┘                │
    │                          │                                 │
    │                    ┌─────▼─────┐                          │
    │                    │ Exporter  │                          │
    │                    │ (Prom/OT) │                          │
    │                    └───────────┘                          │
    └────────────────────────────────────────────────────────────┘

Fase 3.3 (continuación) — estructura del subpaquete
---------------------------------------------------
``metrics.py`` se partió en submódulos internos. Todos los símbolos de la
API pública antigua se re-exportan desde aquí con identidad preservada
(``hoc.metrics.HiveMetrics is hoc.metrics.collection.HiveMetrics``):

- :mod:`.collection` — primitivas ``Counter``/``Gauge``/``Histogram``/
  ``Summary`` + dataclasses + ``CellMetrics``/``SwarmMetrics``/
  ``HiveMetrics``. También hospeda las tres clases transicionales
  (``_InternalCellMetrics``/``GridMetrics``/``MetricsCollector``) que
  antes vivían en ``hoc.core._metrics_internal``.
- :mod:`.visualization` — ``ColorScheme`` + ``HoneycombVisualizer``.
- :mod:`.rendering` — ``HeatmapRenderer`` + ``FlowVisualizer``.
"""

from __future__ import annotations

from .collection import (
    CellMetrics,
    CellMetricSnapshot,
    Counter,
    Gauge,
    GridMetrics,
    Histogram,
    HiveMetrics,
    MetricLabel,
    MetricSample,
    MetricsCollector,
    MetricType,
    Summary,
    SwarmMetrics,
    _InternalCellMetrics,
)
from .rendering import FlowVisualizer, HeatmapRenderer
from .visualization import ColorScheme, HoneycombVisualizer

__all__ = [
    # Tipos base
    "MetricType",
    "MetricLabel",
    "MetricSample",
    # Primitivas
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    # Cell metrics (público)
    "CellMetricSnapshot",
    "CellMetrics",
    # Swarm / Hive
    "SwarmMetrics",
    "HiveMetrics",
    # Visualización
    "ColorScheme",
    "HoneycombVisualizer",
    "HeatmapRenderer",
    "FlowVisualizer",
    # Transicionales (antes en core/_metrics_internal.py); expuestos para
    # consumidores internos y tests que pudiesen cazar tipos exactos. La
    # API pública top-level (`hoc.CellMetrics`) sigue resolviendo al
    # ``CellMetrics`` público de esta capa.
    "_InternalCellMetrics",
    "GridMetrics",
    "MetricsCollector",
]
