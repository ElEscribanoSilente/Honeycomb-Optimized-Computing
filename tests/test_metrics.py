"""Tests para hoc.metrics: Counter, Gauge, Histogram, Summary, HiveMetrics, visualizadores.

Cobertura objetivo Phase 1: ≥75% en metrics.py.
"""

import pytest

from hoc.core import (
    HexCoord,
    HoneycombConfig,
    HoneycombGrid,
)
from hoc.metrics import (
    CellMetrics,
    CellMetricSnapshot,
    ColorScheme,
    Counter,
    FlowVisualizer,
    Gauge,
    HeatmapRenderer,
    Histogram,
    HiveMetrics,
    HoneycombVisualizer,
    MetricLabel,
    MetricSample,
    MetricType,
    Summary,
    SwarmMetrics,
)

# ───────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ───────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def grid():
    return HoneycombGrid(HoneycombConfig(radius=2))


@pytest.fixture
def small_grid():
    return HoneycombGrid(HoneycombConfig(radius=1))


# ───────────────────────────────────────────────────────────────────────────────
# COUNTER
# ───────────────────────────────────────────────────────────────────────────────


class TestCounter:
    def test_initial_zero(self):
        c = Counter("test")
        assert c.get() == 0.0

    def test_inc_default(self):
        c = Counter("test")
        c.inc()
        assert c.get() == 1.0

    def test_inc_with_value(self):
        c = Counter("test")
        c.inc(5.0)
        assert c.get() == 5.0

    def test_inc_accumulates(self):
        c = Counter("test")
        c.inc(2)
        c.inc(3)
        assert c.get() == 5.0

    def test_inc_negative_raises(self):
        c = Counter("test")
        with pytest.raises(ValueError):
            c.inc(-1)

    def test_reset(self):
        c = Counter("test")
        c.inc(10)
        c.reset()
        assert c.get() == 0.0


# ───────────────────────────────────────────────────────────────────────────────
# GAUGE
# ───────────────────────────────────────────────────────────────────────────────


class TestGauge:
    def test_initial_zero(self):
        g = Gauge("test")
        assert g.get() == 0.0

    def test_set(self):
        g = Gauge("test")
        g.set(42.0)
        assert g.get() == 42.0

    def test_inc(self):
        g = Gauge("test")
        g.set(5)
        g.inc(2)
        assert g.get() == 7.0

    def test_dec(self):
        g = Gauge("test")
        g.set(10)
        g.dec(3)
        assert g.get() == 7.0

    def test_dec_below_zero_allowed(self):
        g = Gauge("test")
        g.dec(5)
        assert g.get() == -5.0


# ───────────────────────────────────────────────────────────────────────────────
# HISTOGRAM
# ───────────────────────────────────────────────────────────────────────────────


class TestHistogram:
    def test_default_buckets(self):
        h = Histogram("test")
        assert h.buckets == sorted(Histogram.DEFAULT_BUCKETS)

    def test_custom_buckets(self):
        h = Histogram("test", buckets=[1.0, 5.0, 2.0])
        assert h.buckets == [1.0, 2.0, 5.0]

    def test_observe_increments_count(self):
        h = Histogram("test")
        h.observe(0.5)
        assert h.count == 1
        assert h.sum == 0.5

    def test_observe_multiple(self):
        h = Histogram("test")
        for v in (1.0, 2.0, 3.0):
            h.observe(v)
        assert h.count == 3
        assert h.sum == 6.0
        assert h.mean == 2.0

    def test_buckets_cumulative_prometheus_style(self):
        """Los buckets son cumulativos (todas las observaciones <= bucket)."""
        h = Histogram("test", buckets=[1.0, 5.0, 10.0])
        h.observe(0.5)
        h.observe(2.0)
        h.observe(7.0)
        h.observe(15.0)
        buckets = h.get_buckets()
        # ≤ 1.0: solo 0.5 → 1
        assert buckets[1.0] == 1
        # ≤ 5.0: 0.5, 2.0 → 2
        assert buckets[5.0] == 2
        # ≤ 10.0: 0.5, 2.0, 7.0 → 3
        assert buckets[10.0] == 3
        # +Inf: todos los 4
        assert buckets[float("inf")] == 4

    def test_mean_empty(self):
        h = Histogram("test")
        assert h.mean == 0.0


# ───────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ───────────────────────────────────────────────────────────────────────────────


class TestSummary:
    def test_observe(self):
        s = Summary("test")
        s.observe(1.0)
        s.observe(2.0)
        assert s.count == 2
        assert s.sum == 3.0

    def test_max_samples_limit(self):
        s = Summary("test", max_samples=3)
        for i in range(10):
            s.observe(float(i))
        assert s.count == 3
        # Ultimas tres muestras
        assert sorted(list(s._samples)) == [7.0, 8.0, 9.0]

    def test_quantile_empty(self):
        s = Summary("test")
        assert s.quantile(0.5) == 0.0

    def test_quantile_median(self):
        s = Summary("test")
        for v in (1.0, 2.0, 3.0, 4.0, 5.0):
            s.observe(v)
        # quantile(0.5) en [1,2,3,4,5] → idx 2 → 3.0
        assert s.quantile(0.5) == 3.0

    def test_quantile_p99(self):
        s = Summary("test")
        for v in (1.0, 2.0, 3.0, 4.0, 5.0):
            s.observe(v)
        # idx = int(5 * 0.99) = 4 → 5.0
        assert s.quantile(0.99) == 5.0

    def test_mean(self):
        s = Summary("test")
        for v in (1, 2, 3):
            s.observe(v)
        assert s.mean == 2.0

    def test_stddev_single_sample(self):
        s = Summary("test")
        s.observe(5.0)
        assert s.stddev == 0.0

    def test_stddev_multiple(self):
        s = Summary("test")
        for v in (2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0):
            s.observe(v)
        assert s.stddev > 0


# ───────────────────────────────────────────────────────────────────────────────
# METRIC TYPES (auxiliares)
# ───────────────────────────────────────────────────────────────────────────────


class TestMetricTypes:
    def test_metric_label(self):
        lbl = MetricLabel(name="env", value="prod")
        assert lbl.name == "env"
        assert lbl.value == "prod"

    def test_metric_sample_default_timestamp(self):
        sample = MetricSample(name="x", value=1.0)
        assert sample.timestamp > 0
        assert sample.labels == {}

    def test_metric_type_enum(self):
        assert MetricType.COUNTER != MetricType.GAUGE


# ───────────────────────────────────────────────────────────────────────────────
# CELL METRICS
# ───────────────────────────────────────────────────────────────────────────────


class TestCellMetrics:
    def test_init_creates_metrics(self, grid):
        cell = next(iter(grid._cells.values()))
        cm = CellMetrics(cell)
        assert cm.ticks_processed.get() == 0
        assert cm.errors.get() == 0

    def test_record_tick(self, grid):
        cell = next(iter(grid._cells.values()))
        cm = CellMetrics(cell)
        cm.record_tick(duration=0.001)
        assert cm.ticks_processed.get() == 1
        assert cm.tick_duration.count == 1

    def test_record_error(self, grid):
        cell = next(iter(grid._cells.values()))
        cm = CellMetrics(cell)
        cm.record_error()
        assert cm.errors.get() == 1

    def test_record_vcore_change(self, grid):
        cell = next(iter(grid._cells.values()))
        cm = CellMetrics(cell)
        cm.record_vcore_change(added=True)
        cm.record_vcore_change(added=False)
        assert cm.vcores_added.get() == 1
        assert cm.vcores_removed.get() == 1

    def test_get_snapshot(self, grid):
        cell = next(iter(grid._cells.values()))
        cm = CellMetrics(cell)
        snap = cm.get_snapshot()
        assert isinstance(snap, CellMetricSnapshot)
        assert snap.coord == cell.coord
        assert snap.role == cell.role.name

    def test_get_history_grows(self, grid):
        cell = next(iter(grid._cells.values()))
        cm = CellMetrics(cell)
        for _ in range(3):
            cm.get_snapshot()
        assert len(cm.get_history()) == 3

    def test_get_metrics_dict(self, grid):
        cell = next(iter(grid._cells.values()))
        cm = CellMetrics(cell)
        cm.record_tick(0.001)
        d = cm.get_metrics_dict()
        for key in (
            "coord",
            "ticks_processed",
            "errors",
            "load",
            "vcore_count",
            "tick_duration_mean",
        ):
            assert key in d


# ───────────────────────────────────────────────────────────────────────────────
# SWARM METRICS
# ───────────────────────────────────────────────────────────────────────────────


class TestSwarmMetrics:
    def test_record_task_submitted(self):
        m = SwarmMetrics()
        m.record_task_submitted("compute")
        assert m.tasks_submitted.get() == 1

    def test_record_task_completed(self):
        m = SwarmMetrics()
        m.record_task_completed("compute", duration=0.5)
        assert m.tasks_completed.get() == 1
        assert m.task_duration.count == 1

    def test_record_task_failed(self):
        m = SwarmMetrics()
        m.record_task_failed("compute")
        assert m.tasks_failed.get() == 1

    def test_record_queue_wait(self):
        m = SwarmMetrics()
        m.record_queue_wait(0.1)
        assert m.queue_wait_time.count == 1

    def test_record_work_stolen(self):
        m = SwarmMetrics()
        m.record_work_stolen(5)
        assert m.work_stolen.get() == 5

    def test_update_queue_stats(self):
        m = SwarmMetrics()
        m.update_queue_stats(queue_size=10, pending=3)
        assert m.queue_size.get() == 10
        assert m.pending_tasks.get() == 3

    def test_record_behavior_distribution(self):
        m = SwarmMetrics()
        m.record_behavior_distribution({"Forager": 5, "Nurse": 2})
        stats = m.get_stats()
        assert stats["behaviors"] == {"Forager": 5, "Nurse": 2}

    def test_get_stats_structure(self):
        m = SwarmMetrics()
        m.record_task_submitted("compute")
        m.record_task_completed("compute", 0.1)
        stats = m.get_stats()
        assert "tasks" in stats
        assert "queue" in stats
        assert "duration" in stats
        assert stats["tasks"]["success_rate"] == 1.0


# ───────────────────────────────────────────────────────────────────────────────
# HIVE METRICS
# ───────────────────────────────────────────────────────────────────────────────


class TestHiveMetrics:
    def test_init_creates_cell_metrics(self, grid):
        hm = HiveMetrics(grid)
        assert len(hm._cell_metrics) == len(grid._cells)

    def test_collect_returns_snapshot(self, grid):
        hm = HiveMetrics(grid)
        snap = hm.collect()
        assert "tick" in snap
        assert "cells" in snap
        assert snap["cells"] == len(grid._cells)
        assert "by_state" in snap
        assert "by_role" in snap

    def test_collect_increments_total_ticks(self, grid):
        hm = HiveMetrics(grid)
        hm.collect()
        hm.collect()
        assert hm.total_ticks.get() == 2

    def test_get_cell_metrics(self, grid):
        hm = HiveMetrics(grid)
        coord = next(iter(grid._cells.keys()))
        cm = hm.get_cell_metrics(coord)
        assert cm is not None

    def test_get_cell_metrics_unknown(self, grid):
        hm = HiveMetrics(grid)
        assert hm.get_cell_metrics(HexCoord(999, 999)) is None

    def test_get_ring_metrics(self, grid):
        hm = HiveMetrics(grid)
        ring = hm.get_ring_metrics(radius=1)
        assert "cells" in ring
        assert ring["cells"] == 6  # ring(1) tiene 6 celdas
        assert "average_load" in ring
        assert "total_vcores" in ring

    def test_get_ring_metrics_outside_grid(self, grid):
        hm = HiveMetrics(grid)
        ring = hm.get_ring_metrics(radius=99)
        # Ring de radio fuera del grid → loads vacío → average 0
        assert ring["average_load"] == 0

    def test_generate_report(self, grid):
        hm = HiveMetrics(grid)
        hm.collect()
        report = hm.generate_report()
        assert "HOC HIVE METRICS REPORT" in report
        assert "Total Ticks" in report

    def test_get_history(self, grid):
        hm = HiveMetrics(grid)
        hm.collect()
        hm.collect()
        history = hm.get_history()
        assert len(history) == 2

    def test_get_history_limit(self, grid):
        hm = HiveMetrics(grid)
        for _ in range(10):
            hm.collect()
        assert len(hm.get_history(limit=3)) == 3

    def test_export_prometheus(self, grid):
        hm = HiveMetrics(grid)
        hm.collect()
        prom = hm.export_prometheus()
        assert "hive_ticks_total" in prom
        assert "hive_cells_by_state" in prom
        assert "hive_cells_by_role" in prom


# ───────────────────────────────────────────────────────────────────────────────
# HONEYCOMB VISUALIZER
# ───────────────────────────────────────────────────────────────────────────────


class TestHoneycombVisualizer:
    def test_init(self, grid):
        viz = HoneycombVisualizer(grid)
        assert viz.grid is grid

    def test_set_color_scheme(self, grid):
        viz = HoneycombVisualizer(grid)
        viz.set_color_scheme(ColorScheme.STATE)
        assert viz._color_scheme == ColorScheme.STATE

    def test_render_ascii_load(self, grid):
        viz = HoneycombVisualizer(grid)
        out = viz.render_ascii(scheme=ColorScheme.LOAD)
        assert isinstance(out, str)
        assert len(out) > 0

    def test_render_ascii_state(self, grid):
        viz = HoneycombVisualizer(grid)
        out = viz.render_ascii(scheme=ColorScheme.STATE)
        assert isinstance(out, str)

    def test_render_ascii_role(self, grid):
        viz = HoneycombVisualizer(grid)
        out = viz.render_ascii(scheme=ColorScheme.ROLE)
        # Debería contener emoji de queen
        assert isinstance(out, str)

    def test_render_ascii_pheromone(self, grid):
        viz = HoneycombVisualizer(grid)
        out = viz.render_ascii(scheme=ColorScheme.PHEROMONE)
        assert isinstance(out, str)

    def test_render_ascii_activity(self, grid):
        viz = HoneycombVisualizer(grid)
        out = viz.render_ascii(scheme=ColorScheme.ACTIVITY)
        assert isinstance(out, str)

    def test_render_ascii_with_coords(self, grid):
        viz = HoneycombVisualizer(grid)
        out = viz.render_ascii(show_coords=True)
        # Las coords deberían aparecer
        assert "(" in out

    def test_render_svg(self, grid):
        viz = HoneycombVisualizer(grid)
        svg = viz.render_svg()
        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "polygon" in svg

    def test_render_svg_with_scheme(self, grid):
        viz = HoneycombVisualizer(grid)
        svg = viz.render_svg(scheme=ColorScheme.STATE)
        assert "<svg" in svg

    def test_render_html(self, grid):
        viz = HoneycombVisualizer(grid)
        html = viz.render_html()
        assert "<!DOCTYPE html>" in html
        assert "<svg" in html


# ───────────────────────────────────────────────────────────────────────────────
# HEATMAP RENDERER
# ───────────────────────────────────────────────────────────────────────────────


class TestHeatmapRenderer:
    def test_render_load(self, grid):
        hm = HeatmapRenderer(grid)
        svg = hm.render(metric="load")
        assert svg.startswith("<svg")

    def test_render_pheromone(self, grid):
        hm = HeatmapRenderer(grid)
        svg = hm.render(metric="pheromone")
        assert "<svg" in svg

    def test_render_errors(self, grid):
        hm = HeatmapRenderer(grid)
        svg = hm.render(metric="errors")
        assert "<svg" in svg

    def test_render_unknown_metric(self, grid):
        hm = HeatmapRenderer(grid)
        svg = hm.render(metric="nonexistent")
        # No debería crashear
        assert "<svg" in svg


# ───────────────────────────────────────────────────────────────────────────────
# FLOW VISUALIZER
# ───────────────────────────────────────────────────────────────────────────────


class TestFlowVisualizer:
    def test_init(self, grid):
        flow = FlowVisualizer(grid)
        assert flow.grid is grid

    def test_render_pheromone_trails(self, grid):
        flow = FlowVisualizer(grid)
        svg = flow.render_pheromone_trails()
        assert svg.startswith("<svg")
        assert "</svg>" in svg

    def test_render_activity_flow(self, grid):
        flow = FlowVisualizer(grid)
        svg = flow.render_activity_flow()
        assert "<svg" in svg

    def test_get_flow_stats(self, grid):
        flow = FlowVisualizer(grid)
        stats = flow.get_flow_stats()
        for key in (
            "total_pheromone",
            "active_cells",
            "pheromone_connections",
            "average_pheromone",
        ):
            assert key in stats


# ───────────────────────────────────────────────────────────────────────────────
# CONCURRENCIA BÁSICA
# ───────────────────────────────────────────────────────────────────────────────


class TestConcurrency:
    def test_counter_concurrent_inc(self):
        import threading

        c = Counter("test")

        def worker():
            for _ in range(1000):
                c.inc()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert c.get() == 4000

    def test_gauge_concurrent_inc_dec(self):
        import threading

        g = Gauge("test")

        def worker():
            for _ in range(500):
                g.inc()
                g.dec()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert g.get() == 0  # net zero

    def test_histogram_concurrent_observe(self):
        import threading

        h = Histogram("test")

        def worker():
            for i in range(100):
                h.observe(0.001 * i)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert h.count == 300
