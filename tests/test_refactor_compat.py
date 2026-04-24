"""
Phase 3 refactor compatibility tests.

Ensures the Phase 3.3 split of `core.py` → `core/` subpackage and
`metrics.py` → `metrics/` subpackage preserves every contract that external
users rely on:

1. Every public symbol previously importable at top-level (``from hoc import X``)
   remains importable with the same name.
2. Every public symbol previously importable from the old monolith
   (``from hoc.core import X``, ``from hoc.metrics import X``) remains
   importable from the same qualified path.
3. **Identity**: `hoc.core.HexCoord is hoc.core.grid_geometry.HexCoord` — the
   subpackage `__init__.py` re-exports must return the *same* class object
   from every access path, not a shallow copy.
4. `isinstance` checks continue to work across import paths.
5. The two ``CellMetrics`` classes that existed since Phase 1 (public in
   ``metrics`` + internal in ``core``) still have distinct identities after
   the refactor.

If one of these tests fails, a downstream user's code will break with an
``ImportError`` or a silent identity mismatch — the worst class of
regression because it may not surface until runtime ``isinstance`` fails.
"""

from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# Top-level re-export parity
# ---------------------------------------------------------------------------

# Matches the __all__ of the pre-refactor hoc.__init__. This list is
# intentionally hard-coded rather than computed from the current __all__ so
# that an accidental removal from __all__ is caught by THIS test, not by a
# silent downstream break.
TOP_LEVEL_SYMBOLS = [
    # Core — grid
    "HoneycombGrid",
    "HoneycombConfig",
    "GridTopology",
    # Core — cells
    "HoneycombCell",
    "CellState",
    "CellRole",
    "QueenCell",
    "WorkerCell",
    "DroneCell",
    "NurseryCell",
    # Core — coords
    "HexCoord",
    "HexDirection",
    "HexRing",
    # Core — events
    "EventBus",
    "get_event_bus",
    "set_event_bus",
    "reset_event_bus",
    # Nectar
    "NectarFlow",
    "NectarChannel",
    "NectarPriority",
    "WaggleDance",
    "DanceMessage",
    "DanceDirection",
    "PheromoneTrail",
    "PheromoneType",
    "PheromoneDecay",
    "RoyalJelly",
    "RoyalCommand",
    # Swarm
    "SwarmScheduler",
    "SwarmConfig",
    "SwarmPolicy",
    "BeeBehavior",
    "ForagerBehavior",
    "NurseBehavior",
    "ScoutBehavior",
    "GuardBehavior",
    "HiveTask",
    "TaskPollen",
    "TaskNectar",
    "SwarmBalancer",
    "LoadDistribution",
    # Memory
    "HiveMemory",
    "MemoryConfig",
    "CombStorage",
    "CombCell",
    "PollenCache",
    "HoneyArchive",
    "EvictionPolicy",
    "ReplicationPolicy",
    # Bridge
    "CAMVHoneycombBridge",
    "VentHoneycombAdapter",
    "CellToVCoreMapper",
    "GridToHypervisorMapper",
    "HexToCartesian",
    "CartesianToHex",
    # Resilience
    "HiveResilience",
    "CellFailover",
    "QueenSuccession",
    "HexRedundancy",
    "MirrorCell",
    "SwarmRecovery",
    "CombRepair",
    # Metrics
    "HiveMetrics",
    "CellMetrics",
    "SwarmMetrics",
    "HoneycombVisualizer",
    "HeatmapRenderer",
    "FlowVisualizer",
]


@pytest.mark.parametrize("symbol", TOP_LEVEL_SYMBOLS)
def test_top_level_symbol_importable(symbol: str) -> None:
    """Every pre-refactor top-level symbol still imports from `hoc`."""
    hoc = importlib.import_module("hoc")
    assert hasattr(hoc, symbol), f"hoc.{symbol} disappeared after Phase 3 refactor"


def test_top_level_all_is_superset() -> None:
    """`hoc.__all__` lists every symbol TOP_LEVEL_SYMBOLS asserts."""
    import hoc

    missing = set(TOP_LEVEL_SYMBOLS) - set(hoc.__all__)
    assert (
        not missing
    ), f"Symbols asserted by test_refactor_compat but missing from hoc.__all__: {missing}"


# ---------------------------------------------------------------------------
# Subpackage re-export parity
# ---------------------------------------------------------------------------

CORE_SYMBOLS = [
    "HexCoord",
    "HexDirection",
    "HexRegion",
    "HexRing",
    "HexPathfinder",
    "EventType",
    "Event",
    "EventBus",
    "EventHandler",
    "RWLock",
    "PheromoneType",
    "PheromoneDeposit",
    "PheromoneField",
    "CellState",
    "CellRole",
    "HoneycombConfig",
    "GridTopology",
    "CellMetrics",
    "GridMetrics",
    "MetricsCollector",
    "HoneycombCell",
    "QueenCell",
    "WorkerCell",
    "DroneCell",
    "NurseryCell",
    "StorageCell",
    "GuardCell",
    "ScoutCell",
    "HoneycombGrid",
    "HealthMonitor",
    "HealthStatus",
    "CircuitBreaker",
    "CircuitState",
    "create_grid",
    "benchmark_grid",
    "get_event_bus",
    "set_event_bus",
    "reset_event_bus",
]


@pytest.mark.parametrize("symbol", CORE_SYMBOLS)
def test_hoc_core_symbol_importable(symbol: str) -> None:
    """Pre-refactor ``from hoc.core import X`` paths still resolve."""
    core = importlib.import_module("hoc.core")
    assert hasattr(core, symbol), f"hoc.core.{symbol} disappeared after Phase 3.3 split"


METRICS_SYMBOLS = [
    "MetricType",
    "MetricLabel",
    "MetricSample",
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    "CellMetricSnapshot",
    "CellMetrics",
    "SwarmMetrics",
    "HiveMetrics",
    "ColorScheme",
    "HoneycombVisualizer",
    "HeatmapRenderer",
    "FlowVisualizer",
]


@pytest.mark.parametrize("symbol", METRICS_SYMBOLS)
def test_hoc_metrics_symbol_importable(symbol: str) -> None:
    """Pre-refactor ``from hoc.metrics import X`` paths still resolve."""
    metrics = importlib.import_module("hoc.metrics")
    assert hasattr(metrics, symbol), f"hoc.metrics.{symbol} disappeared after Phase 3.3 split"


# ---------------------------------------------------------------------------
# Identity checks — the critical invariant
# ---------------------------------------------------------------------------


def test_core_hex_coord_identity() -> None:
    """HexCoord resolved from the subpackage facade is the SAME class as
    from the submodule that defines it."""
    from hoc import HexCoord as hoc_HexCoord
    from hoc.core import HexCoord as core_HexCoord

    assert hoc_HexCoord is core_HexCoord
    # At least one submodule declares HexCoord; find and verify identity.
    for mod_name in ("hoc.core.grid", "hoc.core.grid_geometry"):
        try:
            submod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            continue
        if hasattr(submod, "HexCoord"):
            assert (
                submod.HexCoord is hoc_HexCoord
            ), f"{mod_name}.HexCoord has a different identity than hoc.HexCoord"
            break
    else:
        pytest.fail("HexCoord not found in any expected core submodule")


def test_core_event_bus_identity() -> None:
    """EventBus must be the same class across all import paths."""
    from hoc import EventBus as hoc_EventBus
    from hoc.core import EventBus as core_EventBus
    from hoc.core.events import EventBus as events_EventBus

    assert hoc_EventBus is core_EventBus is events_EventBus


def test_core_honeycomb_cell_identity() -> None:
    """HoneycombCell identity across all paths."""
    from hoc import HoneycombCell as hoc_HoneycombCell
    from hoc.core import HoneycombCell as core_HoneycombCell

    assert hoc_HoneycombCell is core_HoneycombCell

    # Find the submodule that defines HoneycombCell.
    for mod_name in ("hoc.core.cells", "hoc.core.cells_base"):
        try:
            submod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            continue
        if hasattr(submod, "HoneycombCell"):
            assert submod.HoneycombCell is hoc_HoneycombCell
            break
    else:
        pytest.fail("HoneycombCell not found in any expected core submodule")


def test_core_queen_cell_identity() -> None:
    """Specialized cell subclasses preserve identity across facade + submodule."""
    from hoc import QueenCell as hoc_QueenCell
    from hoc.core import QueenCell as core_QueenCell

    assert hoc_QueenCell is core_QueenCell
    # QueenCell should be a subclass of HoneycombCell in every import path.
    from hoc.core import HoneycombCell

    assert issubclass(hoc_QueenCell, HoneycombCell)


def test_metrics_hive_metrics_identity() -> None:
    """HiveMetrics identity across facade + submodule."""
    from hoc import HiveMetrics as hoc_HiveMetrics
    from hoc.metrics import HiveMetrics as pkg_HiveMetrics
    from hoc.metrics.collection import HiveMetrics as sub_HiveMetrics

    assert hoc_HiveMetrics is pkg_HiveMetrics is sub_HiveMetrics


def test_metrics_visualizer_identity() -> None:
    """Visualizer classes live in metrics/visualization.py but re-export from
    the subpackage facade AND the top-level `hoc`."""
    from hoc import HoneycombVisualizer as hoc_V
    from hoc.metrics import HoneycombVisualizer as pkg_V
    from hoc.metrics.visualization import HoneycombVisualizer as sub_V

    assert hoc_V is pkg_V is sub_V


def test_metrics_rendering_identity() -> None:
    """HeatmapRenderer / FlowVisualizer live in metrics/rendering.py but
    are re-exported from the facade."""
    from hoc import FlowVisualizer as hoc_F, HeatmapRenderer as hoc_H
    from hoc.metrics import FlowVisualizer as pkg_F, HeatmapRenderer as pkg_H
    from hoc.metrics.rendering import (
        FlowVisualizer as sub_F,
        HeatmapRenderer as sub_H,
    )

    assert hoc_H is pkg_H is sub_H
    assert hoc_F is pkg_F is sub_F


def test_two_cell_metrics_classes_are_distinct() -> None:
    """Pre-refactor, `hoc.CellMetrics` (from metrics.py) and
    `hoc.core.CellMetrics` (internal, from the old core.py monolith) were
    distinct classes. Phase 3.3 merged the internal one into
    metrics/collection.py as a private alias, but the public contract —
    two distinct CellMetrics classes accessible via different paths — must
    be preserved for any code that relied on the distinction.
    """
    from hoc.core import CellMetrics as internal_CellMetrics
    from hoc.metrics import CellMetrics as public_CellMetrics

    assert internal_CellMetrics is not public_CellMetrics, (
        "The internal (core) and public (metrics) CellMetrics classes collapsed "
        "into one — downstream code that constructs them differently will break."
    )


def test_top_level_cell_metrics_resolves_to_public() -> None:
    """``from hoc import CellMetrics`` resolved to the public one before
    the refactor (because __init__.py imports from `.metrics`). It still does."""
    from hoc import CellMetrics as hoc_CellMetrics
    from hoc.metrics import CellMetrics as metrics_CellMetrics

    assert hoc_CellMetrics is metrics_CellMetrics


# ---------------------------------------------------------------------------
# isinstance contract
# ---------------------------------------------------------------------------


def test_isinstance_honeycomb_cell_across_paths() -> None:
    """Constructing a QueenCell via one path, isinstance-checking via another,
    must succeed. This would fail silently if the subpackage re-export
    were a shallow copy instead of an identity re-export."""
    from hoc import HexCoord, HoneycombCell, QueenCell

    queen = QueenCell(coord=HexCoord(0, 0))
    # isinstance against the base class reached via the top-level alias.
    assert isinstance(queen, HoneycombCell)
    # And against the QueenCell alias itself.
    assert isinstance(queen, QueenCell)
    # And against the submodule version.
    from hoc.core import QueenCell as core_QueenCell

    assert isinstance(queen, core_QueenCell)


# ---------------------------------------------------------------------------
# HexRing alias contract
# ---------------------------------------------------------------------------


def test_hex_ring_is_alias_of_hex_region() -> None:
    """Pre-refactor: ``HexRing = HexRegion`` in core.py. Post-refactor the
    alias is preserved at `hoc.core` level."""
    from hoc.core import HexRegion, HexRing

    assert HexRing is HexRegion
