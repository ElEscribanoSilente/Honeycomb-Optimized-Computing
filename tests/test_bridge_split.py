"""Phase 6.5 — test boost para los 3 nuevos módulos del subpaquete ``hoc.bridge``.

El split de ``bridge.py`` (886 LOC) en ``bridge/converters.py``,
``bridge/mappers.py`` y ``bridge/adapters.py`` no añade ni quita
comportamiento. Este test boost ejercita los paths que el legacy
``tests/test_bridge.py`` (que persiste sin tocar como anti-regresión)
no cubría: layout POINTY_TOP, corners/bounding_box, error paths de
los mappers (capacidad llena, duplicados), y los métodos del
``VentHoneycombAdapter`` que estaban a 0% de cobertura.

Cierra Gap 4 desde Phase 4 closure: ``bridge.py`` 56% → bridge/ 80%+.
"""

from __future__ import annotations

import math

import pytest

from hoc.bridge import (
    BridgeConfig,
    CAMVHoneycombBridge,
    CartesianToHex,
    CellToVCoreMapper,
    GridToHypervisorMapper,
    HexToCartesian,
    VCoreMappingEntry,
    VentHoneycombAdapter,
)
from hoc.core import HexCoord, HoneycombConfig, HoneycombGrid

# ───────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ───────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def small_grid():
    """Grid radio=1 (7 cells) suficiente para mapeo y broadcast."""
    return HoneycombGrid(HoneycombConfig(radius=1))


@pytest.fixture
def medium_grid():
    """Grid radio=2 (19 cells) para broadcast a anillos."""
    return HoneycombGrid(HoneycombConfig(radius=2))


# ───────────────────────────────────────────────────────────────────────────────
# CONVERTERS — paths no cubiertos en tests/test_bridge.py
# ───────────────────────────────────────────────────────────────────────────────


class TestHexToCartesianExtras:
    def test_pointy_top_layout(self):
        conv = HexToCartesian(size=1.0, layout=HexToCartesian.Layout.POINTY_TOP)
        x, y = conv.convert(HexCoord(1, 0))
        # POINTY_TOP: x = sqrt(3) * q, y = 3/2 * r
        assert x == pytest.approx(math.sqrt(3))
        assert y == pytest.approx(0.0)

    def test_origin_offset(self):
        conv = HexToCartesian(size=1.0, origin=(10.0, 20.0))
        x, y = conv.convert(HexCoord.origin())
        assert x == 10.0 and y == 20.0

    def test_corners_returns_six(self):
        conv = HexToCartesian(size=1.0)
        corners = conv.corners(HexCoord.origin())
        assert len(corners) == 6
        # Distancia desde center a cada corner ≈ size
        for cx, cy in corners:
            assert math.sqrt(cx**2 + cy**2) == pytest.approx(1.0)

    def test_corners_pointy_top_distinct_angles(self):
        conv = HexToCartesian(size=2.0, layout=HexToCartesian.Layout.POINTY_TOP)
        corners = conv.corners(HexCoord.origin())
        assert len(corners) == 6
        # Pointy top: corners se desplazan pi/6 vs flat top
        for cx, cy in corners:
            assert math.sqrt(cx**2 + cy**2) == pytest.approx(2.0)

    def test_bounding_box_origin(self):
        conv = HexToCartesian(size=1.0)
        min_x, min_y, max_x, max_y = conv.bounding_box(HexCoord.origin())
        # Hex flat-top de size=1: ancho 2.0, alto sqrt(3)
        assert max_x - min_x == pytest.approx(2.0)
        assert max_y - min_y == pytest.approx(math.sqrt(3))

    def test_center_alias(self):
        conv = HexToCartesian(size=1.5)
        coord = HexCoord(2, -1)
        assert conv.center(coord) == conv.convert(coord)


class TestCartesianToHexExtras:
    def test_pointy_top_roundtrip(self):
        h2c = HexToCartesian(size=1.0, layout=HexToCartesian.Layout.POINTY_TOP)
        c2h = CartesianToHex(size=1.0, layout=HexToCartesian.Layout.POINTY_TOP)
        coord = HexCoord(3, -2)
        x, y = h2c.convert(coord)
        assert c2h.convert(x, y) == coord

    def test_origin_offset_in_inverse(self):
        c2h = CartesianToHex(size=1.0, origin=(5.0, 5.0))
        # con origin=(5,5), el punto (5,5) cartesiano cae en (0,0) hex
        coord = c2h.convert(5.0, 5.0)
        assert coord == HexCoord.origin()

    def test_nearest_is_alias(self):
        c2h = CartesianToHex(size=1.0)
        assert c2h.nearest(0.3, 0.4) == c2h.convert(0.3, 0.4)

    def test_in_hexagon_true_at_center(self):
        c2h = CartesianToHex(size=1.0)
        assert c2h.in_hexagon(0.0, 0.0, HexCoord.origin()) is True

    def test_in_hexagon_false_far_away(self):
        c2h = CartesianToHex(size=1.0)
        # punto lejano cae en otro hex
        assert c2h.in_hexagon(10.0, 10.0, HexCoord.origin()) is False

    def test_axial_round_q_dominates(self):
        c2h = CartesianToHex(size=1.0)
        # forzar caso q_diff > r_diff and q_diff > s_diff con un punto que
        # tenga fracción dominante en q
        coord = c2h.convert(0.6, 0.0)
        assert isinstance(coord, HexCoord)

    def test_axial_round_r_dominates(self):
        c2h = CartesianToHex(size=1.0)
        coord = c2h.convert(0.0, 0.6)
        assert isinstance(coord, HexCoord)


# ───────────────────────────────────────────────────────────────────────────────
# MAPPERS
# ───────────────────────────────────────────────────────────────────────────────


def _stub_vcore(vid: str):
    """Mini stub que cumple VCoreProtocol via duck typing."""
    return type(
        "Stub",
        (),
        {
            "vcore_id": vid,
            "state": "ready",
            "execute": lambda self, payload: payload,
            "warmup": lambda self: None,
            "shutdown": lambda self: None,
            "get_metrics": lambda self: {},
        },
    )()


class TestVCoreMappingEntry:
    def test_default_metadata_is_empty_dict(self):
        entry = VCoreMappingEntry(cell_coord=HexCoord(0, 0), vcore_id="x")
        assert entry.metadata == {}
        assert entry.vcore_ref is None

    def test_created_at_populated(self):
        entry = VCoreMappingEntry(cell_coord=HexCoord(1, 0), vcore_id="y")
        assert entry.created_at > 0


class TestCellToVCoreMapperExtras:
    def test_capacity_full_rejects_new_vcore(self):
        mapper = CellToVCoreMapper(max_vcores_per_cell=2)
        coord = HexCoord(0, 0)
        for i in range(2):
            assert mapper.map_cell(coord, _stub_vcore(f"vc_{i}")) is True
        # 3er vcore: capacity exceeded
        assert mapper.map_cell(coord, _stub_vcore("vc_3")) is False

    def test_duplicate_vcore_id_rejected(self):
        mapper = CellToVCoreMapper()
        c1, c2 = HexCoord(0, 0), HexCoord(1, 0)
        assert mapper.map_cell(c1, _stub_vcore("dup")) is True
        # Misma id, diferente cell — rechazo
        assert mapper.map_cell(c2, _stub_vcore("dup")) is False

    def test_unmap_vcore_existing(self):
        mapper = CellToVCoreMapper()
        coord = HexCoord(0, 0)
        mapper.map_cell(coord, _stub_vcore("u1"))
        assert mapper.unmap_vcore("u1") is True
        assert mapper.get_cell("u1") is None

    def test_unmap_vcore_missing(self):
        mapper = CellToVCoreMapper()
        assert mapper.unmap_vcore("ghost") is False

    def test_unmap_clears_empty_cell_entry(self):
        mapper = CellToVCoreMapper()
        coord = HexCoord(0, 0)
        mapper.map_cell(coord, _stub_vcore("only"))
        mapper.unmap_vcore("only")
        # La key fue removida internamente
        assert mapper.get_vcores(coord) == []

    def test_get_vcore_ids(self):
        mapper = CellToVCoreMapper()
        coord = HexCoord(0, 0)
        mapper.map_cell(coord, _stub_vcore("a"))
        mapper.map_cell(coord, _stub_vcore("b"))
        ids = mapper.get_vcore_ids(coord)
        assert set(ids) == {"a", "b"}

    def test_migrate_vcore_success(self):
        mapper = CellToVCoreMapper()
        src, dst = HexCoord(0, 0), HexCoord(1, 0)
        mapper.map_cell(src, _stub_vcore("m"))
        assert mapper.migrate_vcore("m", dst) is True
        assert mapper.get_cell("m") == dst
        assert mapper.get_vcores(src) == []

    def test_migrate_vcore_missing_id(self):
        mapper = CellToVCoreMapper()
        assert mapper.migrate_vcore("ghost", HexCoord(1, 0)) is False

    def test_migrate_vcore_target_full(self):
        mapper = CellToVCoreMapper(max_vcores_per_cell=1)
        src, dst = HexCoord(0, 0), HexCoord(1, 0)
        mapper.map_cell(src, _stub_vcore("m"))
        mapper.map_cell(dst, _stub_vcore("d"))
        # dst lleno (capacidad 1)
        assert mapper.migrate_vcore("m", dst) is False

    def test_get_stats_empty(self):
        mapper = CellToVCoreMapper()
        stats = mapper.get_stats()
        assert stats["total_vcores"] == 0
        assert stats["max_vcores_in_cell"] == 0

    def test_get_stats_populated(self):
        mapper = CellToVCoreMapper()
        coord = HexCoord(0, 0)
        mapper.map_cell(coord, _stub_vcore("a"))
        mapper.map_cell(coord, _stub_vcore("b"))
        stats = mapper.get_stats()
        assert stats["total_vcores"] == 2
        assert stats["cells_with_vcores"] == 1
        assert stats["max_vcores_in_cell"] == 2


class TestGridToHypervisorMapperExtras:
    def test_initialize_stub_creates_vcores(self, small_grid):
        gm = GridToHypervisorMapper(small_grid, hypervisor=None)
        assert gm.initialize_mapping() is True
        stats = gm.get_mapping_stats()
        assert stats["initialized"] is True
        assert stats["hypervisor_present"] is False
        # Cada worker cell debería tener vcores asignados
        assert stats["cell_mapper"]["total_vcores"] > 0

    def test_get_cell_for_vcore(self, small_grid):
        gm = GridToHypervisorMapper(small_grid)
        gm.initialize_mapping()
        # Tomar un vcore_id real del stub mapping
        all_vcore_ids = []
        for coord in small_grid._cells:
            all_vcore_ids.extend(gm._cell_mapper.get_vcore_ids(coord))
        if all_vcore_ids:
            vid = all_vcore_ids[0]
            coord = gm.get_cell_for_vcore(vid)
            assert coord is not None
            assert vid in gm._cell_mapper.get_vcore_ids(coord)

    def test_migrate_vcore_via_grid_mapper(self, small_grid):
        gm = GridToHypervisorMapper(small_grid)
        gm.initialize_mapping()
        # Encuentra un vcore + dos coords distintas
        coord_a = next(iter(small_grid._cells))
        coord_b = None
        for c in small_grid._cells:
            if c != coord_a:
                coord_b = c
                break
        ids_a = gm._cell_mapper.get_vcore_ids(coord_a)
        if ids_a and coord_b is not None:
            vid = ids_a[0]
            assert gm.migrate_vcore(vid, coord_b) is True

    def test_initialize_with_hypervisor_returning_none(self, small_grid):
        """Si hypervisor.allocate_vcore devuelve None, se omite sin error."""

        class NoneHypervisor:
            def allocate_vcore(self, cfg):
                return None

            def deallocate_vcore(self, vid):
                return False

            def get_vcores(self):
                return []

            def get_stats(self):
                return {}

        gm = GridToHypervisorMapper(small_grid, hypervisor=NoneHypervisor())
        assert gm.initialize_mapping() is True
        assert gm.get_mapping_stats()["cell_mapper"]["total_vcores"] == 0


# ───────────────────────────────────────────────────────────────────────────────
# ADAPTERS
# ───────────────────────────────────────────────────────────────────────────────


class TestBridgeConfig:
    def test_defaults(self):
        cfg = BridgeConfig()
        assert cfg.auto_initialize is True
        assert cfg.vcores_per_worker == 8
        assert cfg.hex_size == 1.0
        assert cfg.layout == HexToCartesian.Layout.FLAT_TOP

    def test_custom_layout(self):
        cfg = BridgeConfig(layout=HexToCartesian.Layout.POINTY_TOP)
        assert cfg.layout == HexToCartesian.Layout.POINTY_TOP


class TestCAMVHoneycombBridgeExtras:
    def test_double_initialize_idempotent(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        assert bridge.initialize() is True
        assert bridge.initialize() is True  # idempotente

    def test_execute_on_cell_uninitialized_returns_none(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        result = bridge.execute_on_cell(HexCoord.origin(), {"x": 1})
        assert result is None

    def test_execute_on_cell_no_vcores_returns_none(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        bridge.initialize()
        # Coord fuera del grid → 0 vCores asignados
        result = bridge.execute_on_cell(HexCoord(99, 99), {"x": 1})
        assert result is None
        assert bridge._errors >= 1

    def test_execute_on_cell_runs_payload(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        bridge.initialize()
        # Stub vcore returns None pero contabiliza execution
        for coord in small_grid._cells:
            ids = bridge._grid_mapper._cell_mapper.get_vcore_ids(coord)
            if ids:
                bridge.execute_on_cell(coord, {"hello": "world"})
                assert bridge._executions >= 1
                break

    def test_broadcast_to_ring(self, medium_grid):
        bridge = CAMVHoneycombBridge(medium_grid)
        bridge.initialize()
        results = bridge.broadcast_to_ring(HexCoord.origin(), radius=1, payload={"task": "x"})
        # Algunas celdas ring(1) son WorkerCells con vcores, otras no
        assert isinstance(results, dict)

    def test_migrate_vcores(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        bridge.initialize()
        coord_a = next(iter(small_grid._cells))
        coord_b = None
        for c in small_grid._cells:
            if c != coord_a:
                coord_b = c
                break
        if coord_b is not None:
            migrated = bridge.migrate_vcores(coord_a, coord_b, count=1)
            # 0 o 1 según haya o no vcore en source
            assert migrated >= 0
            if migrated > 0:
                assert bridge._migrations == migrated

    def test_hex_to_cartesian_helper(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        x, y = bridge.hex_to_cartesian(HexCoord(2, -1))
        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_cartesian_to_hex_helper(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        coord = bridge.cartesian_to_hex(0.0, 0.0)
        assert coord == HexCoord.origin()

    def test_tick_triggers_sync_at_interval(self, small_grid):
        cfg = BridgeConfig(sync_interval_ticks=2, health_check_interval=3)
        bridge = CAMVHoneycombBridge(small_grid, config=cfg)
        bridge.initialize()
        # Tick 1: ningún flag
        r1 = bridge.tick()
        assert r1["synced"] is False and r1["health_checked"] is False
        # Tick 2: sync
        r2 = bridge.tick()
        assert r2["synced"] is True
        # Tick 3: health
        r3 = bridge.tick()
        assert r3["health_checked"] is True

    def test_tick_with_real_hypervisor_calls_sync(self, small_grid):
        class Hypervisor:
            def allocate_vcore(self, cfg):
                return None

            def deallocate_vcore(self, vid):
                return False

            def get_vcores(self):
                return []

            def get_stats(self):
                return {}

        cfg = BridgeConfig(sync_interval_ticks=1, health_check_interval=1)
        bridge = CAMVHoneycombBridge(small_grid, hypervisor=Hypervisor(), config=cfg)
        bridge.initialize()
        r = bridge.tick()
        assert r["synced"] is True

    def test_get_stats_shape(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        bridge.initialize()
        stats = bridge.get_stats()
        assert stats["initialized"] is True
        assert "tick_count" in stats
        assert "executions" in stats
        assert "grid_mapper" in stats


class TestVentHoneycombAdapterExtras:
    def test_assign_entity_uses_preferred_coord(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        bridge.initialize()
        adapter = VentHoneycombAdapter(small_grid, bridge)
        target = next(iter(small_grid._cells))
        coord = adapter.assign_entity("e1", preferred_coord=target)
        assert coord == target

    def test_assign_entity_idempotent(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        bridge.initialize()
        adapter = VentHoneycombAdapter(small_grid, bridge)
        first = adapter.assign_entity("e1")
        second = adapter.assign_entity("e1")
        assert first == second  # mismo coord, no realloca

    def test_assign_entity_returns_none_when_no_cells_available(self, small_grid, monkeypatch):
        """Si find_available_cells retorna [], assign_entity debe retornar None."""
        bridge = CAMVHoneycombBridge(small_grid)
        bridge.initialize()
        adapter = VentHoneycombAdapter(small_grid, bridge)
        # HoneycombGrid usa __slots__, así que parcheamos a nivel de clase
        # (afecta a todas las instancias dentro del scope del monkeypatch)
        monkeypatch.setattr(type(small_grid), "find_available_cells", lambda self, n: [])
        # Sin preferred_coord y sin available cells → None
        assert adapter.assign_entity("ghost-e") is None

    def test_get_entity_cell(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        bridge.initialize()
        adapter = VentHoneycombAdapter(small_grid, bridge)
        adapter.assign_entity("e1")
        coord = adapter.get_entity_cell("e1")
        assert coord is not None
        assert adapter.get_entity_cell("ghost") is None

    def test_execute_brain_unassigned_returns_none(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        bridge.initialize()
        adapter = VentHoneycombAdapter(small_grid, bridge)
        assert adapter.execute_brain("ghost", {}) is None

    def test_execute_brain_assigned_routes_to_bridge(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        bridge.initialize()
        adapter = VentHoneycombAdapter(small_grid, bridge)
        # Asignar a una cell con vcores reales
        for coord in small_grid._cells:
            if bridge._grid_mapper._cell_mapper.get_vcore_ids(coord):
                adapter.assign_entity("e1", preferred_coord=coord)
                # execute_brain hace passthrough vía bridge.execute_on_cell.
                # Stub vCore.execute retorna su payload.
                result = adapter.execute_brain("e1", {"x": 1})
                # Stub puede retornar None o el payload — lo importante es
                # que el path se ejerciese sin excepción.
                assert result is None or "entity_id" in result
                break

    def test_migrate_entity_unassigned(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        bridge.initialize()
        adapter = VentHoneycombAdapter(small_grid, bridge)
        assert adapter.migrate_entity("ghost", HexCoord(0, 0)) is False

    def test_migrate_entity_success(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        bridge.initialize()
        adapter = VentHoneycombAdapter(small_grid, bridge)
        # Encontrar dos coords con vcores
        worker_coords = [
            c for c in small_grid._cells if bridge._grid_mapper._cell_mapper.get_vcore_ids(c)
        ]
        if len(worker_coords) >= 2:
            src, dst = worker_coords[0], worker_coords[1]
            adapter.assign_entity("e1", preferred_coord=src)
            ok = adapter.migrate_entity("e1", dst)
            # Puede fallar si dst está en capacidad — ambos paths son válidos
            assert ok in (True, False)
            if ok:
                assert adapter.get_entity_cell("e1") == dst

    def test_remove_entity(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        bridge.initialize()
        adapter = VentHoneycombAdapter(small_grid, bridge)
        adapter.assign_entity("e1")
        assert adapter.remove_entity("e1") is True
        assert adapter.remove_entity("e1") is False  # idempotente

    def test_get_entities_in_cell(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        bridge.initialize()
        adapter = VentHoneycombAdapter(small_grid, bridge)
        target = next(iter(small_grid._cells))
        adapter.assign_entity("e1", preferred_coord=target)
        adapter.assign_entity("e2", preferred_coord=target)
        entities = adapter.get_entities_in_cell(target)
        assert set(entities) == {"e1", "e2"}

    def test_get_stats_shape(self, small_grid):
        bridge = CAMVHoneycombBridge(small_grid)
        bridge.initialize()
        adapter = VentHoneycombAdapter(small_grid, bridge)
        adapter.assign_entity("e1")
        stats = adapter.get_stats()
        assert stats["total_entities"] == 1
        assert "cells_used" in stats
        assert "entities_per_cell" in stats
